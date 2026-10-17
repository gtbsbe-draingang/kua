"""API additions for the result UI: richer violations, video_url, the video route, /static."""
import base64
import hashlib
import hmac
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api
import jobs
from pipeline.types import Event, Frame, PipelineResult
from rules import load_rules

KEY = {"X-API-Key": "k"}
ROOT = Path(__file__).resolve().parent.parent
BRIEF_KEYS = ["type", "timecode_sec", "description", "evidence_frames"]
OLD_KEYS = BRIEF_KEYS + ["type_id", "rule", "confidence"]
NEW_KEYS = ["observation", "justification", "type_source", "t_start", "t_end"]


@pytest.fixture(scope="module")
def rules():
    return load_rules(ROOT / "rules.yaml", strict=False)


def expected_token(job_id, key="k"):
    return hmac.new(key.encode(), job_id.encode(), hashlib.sha256).hexdigest()[:32]


def stub_analyze(path, deadline, on_stage):
    raise AssertionError("no job should run in these tests")


def make_client(tmp_path, rules, analyze=stub_analyze):
    """No `with`: the lifespan does not run, so no worker picks up the jobs the tests plant."""
    return TestClient(api.create_app(tmp_path / "jobs", "k", analyze, rules))


def plant_job(tmp_path, **state):
    jid = jobs.new_job(tmp_path / "jobs")
    d = tmp_path / "jobs" / jid
    jobs.write_state(d, **state)
    return jid, d


def make_event(t, **kw):
    return Event(rule_id="cash_unattended", t_start=t - 1.04, t_end=t + 2.06, timecode=t, ground_score=0.8,
                 type_probs={}, verify_score=0.6, kept=True, reason="verified",
                 evidence=[Frame(t, b"\xff\xd8z")], **kw)


# --- violations[]: new keys -------------------------------------------------------------------

def test_violation_has_new_keys_after_the_existing_ones(rules):
    ev = make_event(3.04, observation="  Кассир отходит.  ", justification="\nТекст.\t")
    v = api.to_api("abc", PipelineResult("violation", [ev], {}, {}), rules)["violations"][0]
    assert list(v) == OLD_KEYS + NEW_KEYS
    assert v["observation"] == "Кассир отходит."
    assert v["justification"] == "Текст."
    assert v["type_source"] == "fallback"
    assert v["t_start"] == 2.0 and v["t_end"] == 5.1
    assert type(v["t_start"]) is float and type(v["t_end"]) is float
    assert type(v["observation"]) is str and type(v["justification"]) is str


def test_violation_observation_none_becomes_empty_and_type_source_follows_event(rules):
    other = next(r for r in rules if r != "cash_unattended")
    ev = make_event(3.0, observation=None, justification="Текст.", judge_rule_id=other, type_source="judge")
    v = api.to_api("abc", PipelineResult("violation", [ev], {}, {}), rules)["violations"][0]
    assert v["observation"] == "" and v["type_source"] == "judge"
    # description is still built exactly as before: no observation, so it starts with the justification
    assert v["description"].startswith("Текст. Основание: ")


def test_violation_window_is_rounded_to_one_decimal_as_float(rules):
    ev = make_event(10.0)
    ev.t_start, ev.t_end = 3, 8  # ints must still come out as floats
    v = api.to_api("abc", PipelineResult("violation", [ev], {}, {}), rules)["violations"][0]
    assert v["t_start"] == 3.0 and v["t_end"] == 8.0
    assert type(v["t_start"]) is float and type(v["t_end"]) is float


def test_jury_fields_unchanged(rules):
    """The brief's fields and the old extras keep their names, types, values and order."""
    ev = make_event(3.04, observation="Кассир отходит.", justification="Текст.")
    body = api.to_api("abc", PipelineResult("violation", [ev], {}, {}), rules)
    assert list(body) == ["job_id", "status", "verdict", "violations"]
    assert body["job_id"] == "abc" and body["status"] == "done" and body["verdict"] == "violation"
    v = body["violations"][0]
    cat = rules["cash_unattended"]["citations"][0]
    from rules import citation_text
    assert {k: v[k] for k in OLD_KEYS} == {
        "type": rules["cash_unattended"]["name"],
        "timecode_sec": 3.0,
        "description": f"Кассир отходит. Текст. Основание: {citation_text(rules['cash_unattended'])}.",
        "evidence_frames": [{"timestamp_sec": 3.0, "image_base64": base64.b64encode(b"\xff\xd8z").decode()}],
        "type_id": "cash_unattended",
        "rule": {k: cat.get(k, "") for k in ("document", "clause", "quote", "url")},
        "confidence": 0.6,
    }
    assert list(v["evidence_frames"][0]) == ["timestamp_sec", "image_base64"]
    assert json.loads(json.dumps(body)) == body  # still plain JSON


def test_clean_result_has_no_violations_and_no_extra_keys(rules):
    body = api.to_api("abc", PipelineResult("no_violation", [], {}, {}), rules)
    assert body == {"job_id": "abc", "status": "done", "verdict": "no_violation", "violations": []}


# --- GET /jobs/{id}: video_url, elapsed_sec -----------------------------------------------------

def test_processing_response_has_video_url_and_elapsed(tmp_path, rules):
    jid, _ = plant_job(tmp_path, status="processing", stage="grounding", created_at=time.time() - 12.3, error=None)
    r = make_client(tmp_path, rules).get(f"/api/v1/jobs/{jid}", headers=KEY)
    body = r.json()
    assert list(body) == ["job_id", "status", "stage", "video_url", "elapsed_sec"]
    assert body["job_id"] == jid and body["status"] == "processing" and body["stage"] == "grounding"
    assert body["video_url"] == f"/api/v1/jobs/{jid}/video?token={expected_token(jid)}"
    assert type(body["elapsed_sec"]) is float and 12.3 <= body["elapsed_sec"] <= 14.3
    assert body["elapsed_sec"] == round(body["elapsed_sec"], 1)


def test_processing_response_elapsed_is_null_without_created_at(tmp_path, rules):
    jid, d = plant_job(tmp_path, status="processing", stage="queued")
    state = jobs.read_state(d)
    assert "created_at" not in state
    body = make_client(tmp_path, rules).get(f"/api/v1/jobs/{jid}", headers=KEY).json()
    assert body["elapsed_sec"] is None
    assert body["video_url"] == f"/api/v1/jobs/{jid}/video?token={expected_token(jid)}"


def test_failed_response_has_video_url(tmp_path, rules):
    jid, _ = plant_job(tmp_path, status="failed", stage="decoding", created_at=1.0, error="ValueError: boom")
    body = make_client(tmp_path, rules).get(f"/api/v1/jobs/{jid}", headers=KEY).json()
    assert list(body) == ["job_id", "status", "error", "video_url"]
    assert body["status"] == "failed" and body["error"] == "ValueError: boom"
    assert body["video_url"] == f"/api/v1/jobs/{jid}/video?token={expected_token(jid)}"


def test_unreadable_result_reports_failed_with_video_url(tmp_path, rules):
    jid, _ = plant_job(tmp_path, status="done", stage="writing", created_at=1.0, error=None)
    body = make_client(tmp_path, rules).get(f"/api/v1/jobs/{jid}", headers=KEY).json()
    assert body["status"] == "failed" and body["error"].startswith("result unavailable")
    assert body["video_url"] == f"/api/v1/jobs/{jid}/video?token={expected_token(jid)}"


def test_done_response_has_video_url_after_the_result_keys(tmp_path, rules):
    jid, d = plant_job(tmp_path, status="done", stage="writing", created_at=1.0, error=None)
    res = PipelineResult("violation", [make_event(3.0, observation="Кассир.", justification="Текст.")], {}, {})
    jobs.write_json_atomic(d / "result.json", api.to_api(jid, res, rules))
    jobs.write_json_atomic(d / "debug.json", {"verdict": "violation"})
    c = make_client(tmp_path, rules)
    body = c.get(f"/api/v1/jobs/{jid}", headers=KEY).json()
    assert list(body) == ["job_id", "status", "verdict", "violations", "video_url"]
    assert body["video_url"] == f"/api/v1/jobs/{jid}/video?token={expected_token(jid)}"
    assert list(body["violations"][0]) == OLD_KEYS + NEW_KEYS
    assert "elapsed_sec" not in body and "debug" not in body
    dbg = c.get(f"/api/v1/jobs/{jid}?debug=true", headers=KEY).json()
    assert dbg["debug"] == {"verdict": "violation"} and dbg["video_url"] == body["video_url"]


def test_video_url_token_differs_per_job_and_per_key(tmp_path, rules):
    a, _ = plant_job(tmp_path, status="failed", error="x")
    b, _ = plant_job(tmp_path, status="failed", error="x")
    c = make_client(tmp_path, rules)
    ua = c.get(f"/api/v1/jobs/{a}", headers=KEY).json()["video_url"]
    ub = c.get(f"/api/v1/jobs/{b}", headers=KEY).json()["video_url"]
    assert ua.split("token=")[1] != ub.split("token=")[1]
    assert len(ua.split("token=")[1]) == 32
    assert expected_token(a, "k") != expected_token(a, "other")


# --- GET /jobs/{id}/video -----------------------------------------------------------------------

PAYLOAD = bytes(range(256)) * 8  # 2048 bytes


def video_job(tmp_path, payload=PAYLOAD):
    jid, d = plant_job(tmp_path, status="done", stage="writing", created_at=1.0, error=None)
    (d / "input.mov").write_bytes(payload)
    return jid


def test_video_with_token_needs_no_header(tmp_path, rules):
    jid = video_job(tmp_path)
    r = make_client(tmp_path, rules).get(f"/api/v1/jobs/{jid}/video?token={expected_token(jid)}")
    assert r.status_code == 200 and r.content == PAYLOAD
    assert r.headers["content-type"] == "video/mp4"
    assert r.headers["accept-ranges"] == "bytes"


def test_video_url_from_the_job_response_works_as_is(tmp_path, rules):
    jid = video_job(tmp_path)
    c = make_client(tmp_path, rules)
    url = c.get(f"/api/v1/jobs/{jid}", headers=KEY).json()["video_url"]
    r = c.get(url)  # no header: exactly what <video src> does
    assert r.status_code == 200 and r.content == PAYLOAD


def test_video_with_api_key_header_and_no_token(tmp_path, rules):
    jid = video_job(tmp_path)
    r = make_client(tmp_path, rules).get(f"/api/v1/jobs/{jid}/video", headers=KEY)
    assert r.status_code == 200 and r.content == PAYLOAD


def test_video_valid_header_beats_a_wrong_token(tmp_path, rules):
    jid = video_job(tmp_path)
    r = make_client(tmp_path, rules).get(f"/api/v1/jobs/{jid}/video?token=wrong", headers=KEY)
    assert r.status_code == 200


def test_video_wrong_header_with_valid_token_is_ok(tmp_path, rules):
    jid = video_job(tmp_path)
    r = make_client(tmp_path, rules).get(f"/api/v1/jobs/{jid}/video?token={expected_token(jid)}",
                                         headers={"X-API-Key": "bad"})
    assert r.status_code == 200


@pytest.mark.parametrize("query", ["", "?token=", "?token=wrong", "?token=%D0%BA%D0%BB%D1%8E%D1%87", "?key=k"])
def test_video_without_valid_auth_401(tmp_path, rules, query):
    jid = video_job(tmp_path)
    c = make_client(tmp_path, rules)
    r = c.get(f"/api/v1/jobs/{jid}/video{query}")
    assert r.status_code == 401 and r.json() == {"detail": "invalid or missing token"}
    r = c.get(f"/api/v1/jobs/{jid}/video{query}", headers={"X-API-Key": "bad"})
    assert r.status_code == 401 and r.json() == {"detail": "invalid or missing token"}


def test_video_token_of_another_job_rejected(tmp_path, rules):
    a, b = video_job(tmp_path), video_job(tmp_path)
    c = make_client(tmp_path, rules)
    assert c.get(f"/api/v1/jobs/{a}/video?token={expected_token(b)}").status_code == 401
    assert c.get(f"/api/v1/jobs/{b}/video?token={expected_token(b)}").status_code == 200


def test_video_token_made_with_another_key_rejected(tmp_path, rules):
    jid = video_job(tmp_path)
    r = make_client(tmp_path, rules).get(f"/api/v1/jobs/{jid}/video?token={expected_token(jid, 'other')}")
    assert r.status_code == 401


def test_video_range_request_gives_206(tmp_path, rules):
    jid = video_job(tmp_path)
    r = make_client(tmp_path, rules).get(f"/api/v1/jobs/{jid}/video?token={expected_token(jid)}",
                                         headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100 and r.content == PAYLOAD[:100]
    assert r.headers["content-range"] == f"bytes 0-99/{len(PAYLOAD)}"
    assert r.headers["content-type"] == "video/mp4"
    # seeking into the middle and an open-ended range work too
    r = make_client(tmp_path, rules).get(f"/api/v1/jobs/{jid}/video?token={expected_token(jid)}",
                                         headers={"Range": "bytes=1000-"})
    assert r.status_code == 206 and r.content == PAYLOAD[1000:]


def test_video_unknown_bad_or_missing_input_is_404(tmp_path, rules):
    c = make_client(tmp_path, rules)
    unknown = "aaaaaaaaaaaa"
    assert c.get(f"/api/v1/jobs/{unknown}/video?token={expected_token(unknown)}").status_code == 404
    for bad in ("NOTHEX123456", "abc", "0123456789abc"):
        assert c.get(f"/api/v1/jobs/{bad}/video?token={expected_token(bad)}").status_code == 404
        assert c.get(f"/api/v1/jobs/{bad}/video", headers=KEY).status_code == 404
    jid, _ = plant_job(tmp_path, status="failed", error="x")  # job folder without input.mov
    assert c.get(f"/api/v1/jobs/{jid}/video?token={expected_token(jid)}").status_code == 404
    assert c.get(f"/api/v1/jobs/{jid}/video", headers=KEY).status_code == 404


def test_video_unauthorised_request_cannot_probe_for_jobs(tmp_path, rules):
    """An unknown job and a known one answer the same 401 when the caller has no valid credential."""
    jid = video_job(tmp_path)
    c = make_client(tmp_path, rules)
    assert c.get(f"/api/v1/jobs/{jid}/video").status_code == 401
    assert c.get("/api/v1/jobs/aaaaaaaaaaaa/video").status_code == 401


def test_video_end_to_end_with_a_real_clip(tmp_path, rules, portrait_clip):
    from pipeline import stub

    with TestClient(api.create_app(tmp_path / "jobs", "k", stub.analyze, rules)) as c:
        with open(portrait_clip, "rb") as f:
            jid = c.post("/api/v1/analyze", files={"file": ("c.mov", f, "video/quicktime")},
                         headers=KEY).json()["job_id"]
        end = time.time() + 20
        while time.time() < end:
            body = c.get(f"/api/v1/jobs/{jid}", headers=KEY).json()
            if body["status"] != "processing":
                break
            time.sleep(0.1)
        assert body["status"] == "done"
        v = body["violations"][0]
        assert v["type_source"] == "fallback" and v["justification"] and v["observation"]
        assert 0 <= v["t_start"] <= v["timecode_sec"] <= v["t_end"]
        r = c.get(body["video_url"])
        assert r.status_code == 200 and r.content == portrait_clip.read_bytes()


# --- /static and / ------------------------------------------------------------------------------

@pytest.fixture
def static_dir(tmp_path, monkeypatch):
    d = tmp_path / "static"
    (d / "fonts").mkdir(parents=True)
    (d / "index.html").write_text("<!doctype html><title>Kuá</title>", encoding="utf-8")
    (d / "ui.css").write_text("body{margin:0}", encoding="utf-8")
    (d / "ui.js").write_text("console.log(1)", encoding="utf-8")
    (d / "fonts" / "f.woff2").write_bytes(b"wOF2\x00\x01")
    (tmp_path / "secret.txt").write_text("nope", encoding="utf-8")
    monkeypatch.setattr(api, "STATIC_DIR", d)
    return d


def test_static_files_are_served_without_a_key(tmp_path, rules, static_dir):
    c = make_client(tmp_path, rules)
    r = c.get("/static/ui.css")
    assert r.status_code == 200 and r.text == "body{margin:0}" and r.headers["content-type"].startswith("text/css")
    r = c.get("/static/ui.js")
    assert r.status_code == 200 and "javascript" in r.headers["content-type"]
    r = c.get("/static/fonts/f.woff2")
    assert r.status_code == 200 and r.content == b"wOF2\x00\x01" and r.headers["content-type"] == "font/woff2"


def test_static_missing_file_and_traversal_are_404(tmp_path, rules, static_dir):
    c = make_client(tmp_path, rules)
    assert c.get("/static/nope.css").status_code == 404
    assert c.get("/static/fonts/").status_code == 404
    assert c.get("/static/../secret.txt").status_code == 404
    assert c.get("/static/%2e%2e/secret.txt").status_code == 404


def test_index_still_served_at_root(tmp_path, rules, static_dir):
    r = make_client(tmp_path, rules).get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html") and "Kuá" in r.text


def test_index_404_when_missing(tmp_path, rules, static_dir):
    (static_dir / "index.html").unlink()
    c = make_client(tmp_path, rules)
    assert c.get("/").status_code == 404
    assert c.get("/static/ui.css").status_code == 200


def test_missing_static_dir_is_404_not_a_crash(tmp_path, rules, monkeypatch):
    monkeypatch.setattr(api, "STATIC_DIR", tmp_path / "does-not-exist")
    c = make_client(tmp_path, rules)
    assert c.get("/").status_code == 404
    assert c.get("/static/ui.css").status_code == 404


def test_real_static_dir_is_mounted(tmp_path, rules):
    assert make_client(tmp_path, rules).get("/static/index.html").status_code == 200
