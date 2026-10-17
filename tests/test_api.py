import base64
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api
import jobs
from pipeline import stub
from pipeline.types import PipelineResult
from rules import load_rules

KEY = {"X-API-Key": "k"}
ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def rules():
    return load_rules(ROOT / "rules.yaml", strict=False)


def make_client(tmp_path, rules, analyze=stub.analyze, **kw):
    return TestClient(api.create_app(tmp_path / "jobs", "k", analyze, rules, **kw))


def upload(c, path, name="clip.mov", headers=KEY):
    with open(path, "rb") as f:
        return c.post("/api/v1/analyze", files={"file": (name, f, "video/quicktime")}, headers=headers)


def wait(c, jid, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        body = c.get(f"/api/v1/jobs/{jid}", headers=KEY).json()
        if body["status"] != "processing":
            return body
        time.sleep(0.1)
    raise AssertionError("timeout")


def test_post_without_key_401(tmp_path, rules, portrait_clip):
    with make_client(tmp_path, rules) as c:
        assert upload(c, portrait_clip, headers={}).status_code == 401
        assert upload(c, portrait_clip, headers={"X-API-Key": "bad"}).status_code == 401


def test_get_without_key_401(tmp_path, rules):
    with make_client(tmp_path, rules) as c:
        assert c.get("/api/v1/jobs/aaaaaaaaaaaa").status_code == 401


def test_healthz_and_index_need_no_key(tmp_path, rules):
    with make_client(tmp_path, rules) as c:
        assert c.get("/healthz").json() == {"ok": True}
        assert c.get("/").status_code in (200, 404)


def test_post_returns_job_id(tmp_path, rules, portrait_clip):
    with make_client(tmp_path, rules) as c:
        r = upload(c, portrait_clip)
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "processing" and len(body["job_id"]) == 12


def test_job_reaches_done_with_schema(tmp_path, rules, portrait_clip):
    with make_client(tmp_path, rules) as c:
        jid = upload(c, portrait_clip).json()["job_id"]
        body = wait(c, jid)
        assert body["status"] == "done" and body["job_id"] == jid
        assert body["verdict"] == "violation"
        v = body["violations"][0]
        assert v["timecode_sec"] >= 0
        assert 1 <= len(v["evidence_frames"]) <= 3
        for fr in v["evidence_frames"]:
            assert base64.b64decode(fr["image_base64"]).startswith(b"\xff\xd8")
        assert v["evidence_frames"][0]["timestamp_sec"] == v["timecode_sec"]
        assert v["type"] == rules["cash_unattended"]["name"]
        assert v["type_id"] == "cash_unattended"
        assert "Основание:" in v["description"]
        assert set(v["rule"]) == {"document", "clause", "quote", "url"}


def test_landscape_clip_runs(tmp_path, rules, landscape_clip):
    with make_client(tmp_path, rules) as c:
        jid = upload(c, landscape_clip).json()["job_id"]
        assert wait(c, jid)["status"] == "done"


def test_broken_file_fails_job(tmp_path, rules):
    empty = tmp_path / "empty.mov"
    empty.write_bytes(b"")
    text = tmp_path / "text.mov"
    text.write_text("this is not a video")
    with make_client(tmp_path, rules) as c:
        for p in (empty, text):
            jid = upload(c, p).json()["job_id"]
            body = wait(c, jid)
            assert body["status"] == "failed" and body["error"]


def test_upload_over_limit_413(tmp_path, rules, portrait_clip, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "1024")
    big = tmp_path / "big.mov"
    big.write_bytes(b"x" * (2 * 1024 * 1024 + 10))
    with make_client(tmp_path, rules) as c:
        assert upload(c, big).status_code == 413
        # no job folder is left behind
        assert not [d for d in (tmp_path / "jobs").iterdir()]


def test_upload_slightly_over_limit_without_content_length_check(tmp_path, rules, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "1024")
    f = tmp_path / "f.mov"
    f.write_bytes(b"x" * 5000)  # under limit+1MiB slack: caught by the streaming count
    with make_client(tmp_path, rules) as c:
        assert upload(c, f).status_code == 413
        assert not list((tmp_path / "jobs").iterdir())


def multipart_body(payload: bytes, boundary="BNDRY", field="file"):
    return (f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; filename="a.mov"\r\n'
            f'Content-Type: video/quicktime\r\n\r\n').encode() + payload + f"\r\n--{boundary}--\r\n".encode()


def test_wrong_key_with_big_body_401_no_folder(tmp_path, rules):
    with make_client(tmp_path, rules) as c:
        r = c.post("/api/v1/analyze", content=multipart_body(b"x" * 3_000_000),
                   headers={"X-API-Key": "bad", "Content-Type": "multipart/form-data; boundary=BNDRY"})
        assert r.status_code == 401
        assert not list((tmp_path / "jobs").iterdir())


def test_chunked_body_over_limit_413_no_folder(tmp_path, rules, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "1024")
    body = multipart_body(b"x" * 100_000)

    def gen():  # generator body: sent chunked, no Content-Length
        for i in range(0, len(body), 8192):
            yield body[i:i + 8192]

    with make_client(tmp_path, rules) as c:
        r = c.post("/api/v1/analyze", content=gen(),
                   headers={**KEY, "Content-Type": "multipart/form-data; boundary=BNDRY"})
        assert r.status_code == 413
        assert not list((tmp_path / "jobs").iterdir())


def test_chunked_body_under_limit_ok(tmp_path, rules, portrait_clip):
    body = multipart_body(portrait_clip.read_bytes())

    def gen():
        for i in range(0, len(body), 65536):
            yield body[i:i + 65536]

    with make_client(tmp_path, rules) as c:
        r = c.post("/api/v1/analyze", content=gen(),
                   headers={**KEY, "Content-Type": "multipart/form-data; boundary=BNDRY"})
        assert r.status_code == 200
        assert wait(c, r.json()["job_id"])["status"] == "done"


def test_missing_file_part_rejected_no_folder(tmp_path, rules):
    with make_client(tmp_path, rules) as c:
        r = c.post("/api/v1/analyze", content=multipart_body(b"abc", field="other"),
                   headers={**KEY, "Content-Type": "multipart/form-data; boundary=BNDRY"})
        assert r.status_code == 422 and "file" in r.json()["detail"]
        r = c.post("/api/v1/analyze", content=b"{}", headers={**KEY, "Content-Type": "application/json"})
        assert r.status_code == 400
        assert not list((tmp_path / "jobs").iterdir())


def test_resume_resets_stage_to_queued(tmp_path, rules):
    import threading

    jobs_dir = tmp_path / "jobs"
    jid = jobs.new_job(jobs_dir)
    d = jobs_dir / jid
    (d / "input.mov").write_bytes(b"x")
    jobs.write_state(d, status="processing", stage="verifying", created_at=time.time(), error=None)
    gate = threading.Event()

    def blocked(path, deadline, on_stage):
        gate.wait(5)
        return PipelineResult("no_violation", [], {}, {})

    with make_client(tmp_path, rules, analyze=blocked, workers=1) as c:
        body = c.get(f"/api/v1/jobs/{jid}", headers=KEY).json()
        assert body == {"job_id": jid, "status": "processing", "stage": "queued"}
        gate.set()
        assert wait(c, jid)["status"] == "done"


def test_done_without_result_json_reports_failed(tmp_path, rules):
    jobs_dir = tmp_path / "jobs"
    jid = jobs.new_job(jobs_dir)
    jobs.write_state(jobs_dir / jid, status="done", stage="writing", created_at=1.0, error=None)
    with make_client(tmp_path, rules) as c:
        body = c.get(f"/api/v1/jobs/{jid}?debug=1", headers=KEY)
        assert body.status_code == 200 and body.json()["status"] == "failed" and body.json()["error"]


def test_unknown_status_treated_as_processing_and_missing_debug_omitted(tmp_path, rules, portrait_clip):
    jobs_dir = tmp_path / "jobs"
    jid = jobs.new_job(jobs_dir)
    jobs.write_state(jobs_dir / jid, created_at=1.0)
    with make_client(tmp_path, rules) as c:
        assert c.get(f"/api/v1/jobs/{jid}", headers=KEY).json()["status"] == "processing"
        jid2 = upload(c, portrait_clip).json()["job_id"]
        wait(c, jid2)
        (jobs_dir / jid2 / "debug.json").unlink()
        assert "debug" not in c.get(f"/api/v1/jobs/{jid2}?debug=1", headers=KEY).json()


def test_debug_true_accepted(tmp_path, rules, portrait_clip):
    with make_client(tmp_path, rules) as c:
        jid = upload(c, portrait_clip).json()["job_id"]
        wait(c, jid)
        assert "debug" in c.get(f"/api/v1/jobs/{jid}?debug=true", headers=KEY).json()


def test_rotated_clip_runs(tmp_path, rules, rotated_clip):
    with make_client(tmp_path, rules) as c:
        assert wait(c, upload(c, rotated_clip).json()["job_id"])["status"] == "done"


def test_unknown_job_404(tmp_path, rules):
    with make_client(tmp_path, rules) as c:
        assert c.get("/api/v1/jobs/aaaaaaaaaaaa", headers=KEY).status_code == 404


def test_bad_job_id_404(tmp_path, rules):
    with make_client(tmp_path, rules) as c:
        assert c.get("/api/v1/jobs/..%2F..%2Fetc", headers=KEY).status_code == 404
        assert c.get("/api/v1/jobs/NOTHEX123456", headers=KEY).status_code == 404


def test_pending_job_resumes_on_start(tmp_path, rules, portrait_clip):
    jobs_dir = tmp_path / "jobs"
    jid = jobs.new_job(jobs_dir)
    d = jobs_dir / jid
    (d / "input.mov").write_bytes(portrait_clip.read_bytes())
    jobs.write_state(d, status="processing", stage="queued", created_at=time.time(), error=None)
    with make_client(tmp_path, rules) as c:
        assert wait(c, jid)["status"] == "done"


def test_pending_job_without_input_fails(tmp_path, rules):
    jobs_dir = tmp_path / "jobs"
    jid = jobs.new_job(jobs_dir)
    jobs.write_state(jobs_dir / jid, status="processing", stage="queued", created_at=time.time(), error=None)
    with make_client(tmp_path, rules) as c:
        assert wait(c, jid)["status"] == "failed"


def test_cyrillic_filename_upload(tmp_path, rules, portrait_clip):
    with make_client(tmp_path, rules) as c:
        r = upload(c, portrait_clip, name="Х-4.mov")
        assert r.status_code == 200
        assert wait(c, r.json()["job_id"])["status"] == "done"


def test_clean_result_has_empty_list(tmp_path, rules, portrait_clip):
    def clean(path, deadline, on_stage):
        return PipelineResult("no_violation", [], {}, {})

    with make_client(tmp_path, rules, analyze=clean) as c:
        body = wait(c, upload(c, portrait_clip).json()["job_id"])
        assert body["verdict"] == "no_violation" and body["violations"] == []


def test_debug_param_returns_events(tmp_path, rules, portrait_clip):
    with make_client(tmp_path, rules) as c:
        jid = upload(c, portrait_clip).json()["job_id"]
        wait(c, jid)
        plain = c.get(f"/api/v1/jobs/{jid}", headers=KEY).json()
        assert "debug" not in plain
        dbg = c.get(f"/api/v1/jobs/{jid}?debug=1", headers=KEY).json()
        assert dbg["debug"]["events"]
        assert "image_base64" not in str(dbg["debug"])
        assert "jpeg" not in str(dbg["debug"])


def test_to_api_uses_judge_rule_and_sorts(rules):
    from pipeline.types import Event, Frame

    def ev(t, **kw):
        return Event(rule_id="cash_unattended", t_start=t, t_end=t, timecode=t, ground_score=0.8,
                     type_probs={}, verify_score=None, kept=True, reason="x",
                     evidence=[Frame(t, b"\xff\xd8z")], **kw)

    other = next(r for r in rules if r != "cash_unattended")
    res = PipelineResult("violation", [ev(9.04), ev(3.0, judge_rule_id=other, type_source="judge"),
                                       ev(5.0, ) ], {}, {})
    res.events[2].kept = False
    body = api.to_api("abc", res, rules)
    assert [v["timecode_sec"] for v in body["violations"]] == [3.0, 9.0]
    assert body["violations"][0]["type_id"] == other
    assert body["violations"][0]["confidence"] == 0.8
