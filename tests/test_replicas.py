"""Replica pool: URL parsing, least-in-flight choice, thread safety, and failover in the verify clients."""
import threading
import time

import httpx
import pytest

from pipeline import clients, replicas

RULES = [{"id": "cash_unattended", "verify_prompt": "Does the cashier leave cash unattended?"}]
A, B, C = "http://a/verify", "http://b/verify", "http://c/verify"


@pytest.fixture(autouse=True)
def fresh_pool(monkeypatch):
    for k in ("IV3_URL", "IV3_URLS"):
        monkeypatch.delenv(k, raising=False)
    clients._reset_pool()
    yield
    clients._reset_pool()


def test_parse_urls_dedup_and_default():
    assert replicas.parse_urls(" a, b ,,a,c ", "d") == ["a", "b", "c"]
    assert replicas.parse_urls("", "d") == ["d"]
    assert replicas.parse_urls(None, "d") == ["d"]
    assert replicas.parse_urls(" , ", "d") == ["d"]


def test_pool_picks_least_in_flight_and_rotates_ties():
    pool = replicas.Pool([A, B, C])
    assert sorted(pool.pick() for _ in range(3)) == [A, B, C]
    pool = replicas.Pool([A, B, C])
    picked = [pool.pick() for _ in range(3)]
    assert picked == [A, B, C]
    pool.release(picked[1])
    assert pool.pick() == B


def test_pool_fallbacks_fewest_in_flight_first():
    pool = replicas.Pool([A, B, C])
    pool.take(B)
    pool.take(B)
    pool.take(C)
    assert pool.fallbacks(A) == [C, B]


def test_pool_use_releases_on_error():
    pool = replicas.Pool([A, B])
    with pytest.raises(RuntimeError):
        with pool.use():
            raise RuntimeError
    assert sorted([pool.pick(), pool.pick()]) == [A, B]  # nothing is left counted as in flight


def test_pool_thread_safe_spread():
    urls = [A, B, C]
    pool = replicas.Pool(urls)
    lock = threading.Lock()
    inflight = {u: 0 for u in urls}
    peak = {u: 0 for u in urls}
    start = threading.Barrier(30)

    def work():
        start.wait()
        with pool.use() as url:
            with lock:
                inflight[url] += 1
                peak[url] = max(peak[url], inflight[url])
            time.sleep(0.02)
            with lock:
                inflight[url] -= 1

    threads = [threading.Thread(target=work) for _ in range(30)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert max(peak.values()) <= 10  # ceil(30 / 3)
    assert all(v == 0 for v in inflight.values())


def ok(p=0.9):
    return httpx.Response(200, json={"p_yes": p, "type_id": "none", "observation": "ok"})


def fake_post(monkeypatch, handler):
    seen = []

    def post(url, json, timeout, trust_env):
        seen.append(url)
        return handler(url)
    monkeypatch.setattr(clients.httpx, "post", post)
    return seen


def test_verify_fails_over_on_connect_error(monkeypatch):
    monkeypatch.setenv("IV3_URLS", f"{A},{B}")

    def handler(url):
        if url == A:
            raise httpx.ConnectError("down")
        return ok(0.7)
    seen = fake_post(monkeypatch, handler)
    assert clients.verify("w.mp4", RULES, 10) == (0.7, "none", "ok")
    assert seen == [A, B]


def test_verify_multi_fails_over_on_connect_error(monkeypatch):
    monkeypatch.setenv("IV3_URLS", f"{A},{B}")

    def handler(url):
        if url == A:
            raise httpx.ConnectError("down")
        return httpx.Response(200, json={"p_yes_list": [0.4], "observation": "o"})
    seen = fake_post(monkeypatch, handler)
    assert clients.verify_multi("w.mp4", RULES, 10) == ([0.4], "o")
    assert seen == [A, B]


def test_verify_all_replicas_down_returns_none(monkeypatch):
    monkeypatch.setenv("IV3_URLS", f"{A},{B}")

    def handler(url):
        raise httpx.ConnectError("down")
    seen = fake_post(monkeypatch, handler)
    assert clients.verify("w.mp4", RULES, 10) == (None, "", "")
    assert clients.verify_multi("w.mp4", RULES, 10) == ([None], "")
    assert len(seen) == 4  # one retry per call, no more


def test_timeout_does_not_fail_over(monkeypatch):
    monkeypatch.setenv("IV3_URLS", f"{A},{B}")

    def handler(url):
        raise httpx.ReadTimeout("slow")
    seen = fake_post(monkeypatch, handler)
    assert clients.verify("w.mp4", RULES, 10) == (None, "", "")
    assert seen == [A]


def test_single_url_keeps_old_behaviour(monkeypatch):
    monkeypatch.setenv("IV3_URL", "http://x/verify")
    assert clients.iv3_urls() == ["http://x/verify"]
    seen = fake_post(monkeypatch, lambda url: ok())
    assert clients.verify("w.mp4", RULES, 10)[0] == 0.9
    assert seen == ["http://x/verify"]


def test_iv3_urls_default_and_urls_env(monkeypatch):
    assert clients.iv3_urls() == [clients.IV3_URL]
    monkeypatch.setenv("IV3_URL", "http://x/verify")
    monkeypatch.setenv("IV3_URLS", f"{A}, {B}")
    assert clients.iv3_urls() == [A, B]


def test_iv3_pool_is_one_object_under_a_burst(monkeypatch):
    monkeypatch.setenv("IV3_URLS", f"{A},{B}")
    n = 16
    start = threading.Barrier(n)
    got = []

    def grab():
        start.wait()
        got.append(clients._iv3_pool())

    threads = [threading.Thread(target=grab) for _ in range(n)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(got) == n and all(p is got[0] for p in got)
