"""Spreading requests over several identical InternVideo3 servers on one GPU. Each server handles one request at a
time, so a caller picks the replica with the fewest requests in flight. In-process and thread-safe; two processes
(the API and the RAG server) each keep their own counts."""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import TypeVar

import httpx

T = TypeVar("T")


def parse_urls(value: str | None, default: str) -> list[str]:
    """Comma-separated URLs, stripped, empties dropped, duplicates removed in order. [default] when none are left."""
    urls = list(dict.fromkeys(u for u in (p.strip() for p in (value or "").split(",")) if u))
    return urls or [default]


class Pool:
    """Least-in-flight choice over urls; ties go round-robin."""

    def __init__(self, urls: list[str]):
        if not urls:
            raise ValueError("a pool needs at least one URL")
        self.urls = list(urls)
        self._busy = {u: 0 for u in self.urls}
        self._next = 0
        self._lock = threading.Lock()

    def pick(self) -> str:
        """The URL with the fewest requests in flight (counted from now); release it when done."""
        with self._lock:
            n = len(self.urls)
            fewest = min(self._busy.values())
            for k in range(n):
                i = (self._next + k) % n
                if self._busy[self.urls[i]] == fewest:
                    self._next = (i + 1) % n
                    self._busy[self.urls[i]] += 1
                    return self.urls[i]
            raise AssertionError("unreachable")

    def take(self, url: str) -> str:
        """Count a request on a given URL (the failover's choice)."""
        with self._lock:
            self._busy[url] += 1
        return url

    def release(self, url: str) -> None:
        with self._lock:
            if self._busy.get(url, 0) > 0:
                self._busy[url] -= 1

    @contextmanager
    def use(self):
        url = self.pick()
        try:
            yield url
        finally:
            self.release(url)

    def fallbacks(self, url: str) -> list[str]:
        """The other URLs, fewest in flight first (ties in list order)."""
        with self._lock:
            return sorted((u for u in self.urls if u != url), key=lambda u: self._busy[u])


def call_with_failover(pool: Pool, call: Callable[[str, float], T], timeout: float) -> T:
    """call(url, seconds) on a picked replica. On httpx.ConnectError (the server is down) it is retried once on the
    least busy other replica with what is left of the timeout. Timeouts and every other error are raised as they are:
    a slow replica must not double the load."""
    t0 = time.monotonic()
    with pool.use() as url:
        try:
            return call(url, timeout)
        except httpx.ConnectError:
            others = pool.fallbacks(url)
            left = timeout - (time.monotonic() - t0)
            if not others or left <= 0:
                raise
            first = others[0]
    pool.take(first)
    try:
        return call(first, left)
    finally:
        pool.release(first)
