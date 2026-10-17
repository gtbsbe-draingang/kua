import json
import re

import jobs


def test_new_job_is_12_hex(tmp_path):
    jid = jobs.new_job(tmp_path)
    assert re.fullmatch(r"[0-9a-f]{12}", jid)
    assert (tmp_path / jid).is_dir()
    assert jobs.new_job(tmp_path) != jid


def test_write_json_atomic_leaves_no_temp(tmp_path):
    p = tmp_path / "x.json"
    jobs.write_json_atomic(p, {"a": "Х"})
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": "Х"}
    assert [f.name for f in tmp_path.iterdir()] == ["x.json"]


def test_write_state_merges(tmp_path):
    d = tmp_path / "j"
    d.mkdir()
    jobs.write_state(d, status="processing", stage="queued", created_at=1.0, error=None)
    jobs.write_state(d, stage="decoding")
    s = jobs.read_state(d)
    assert s == {"status": "processing", "stage": "decoding", "created_at": 1.0, "error": None}


def test_pending_jobs_oldest_first(tmp_path):
    for name, status, ts in [("aaaaaaaaaaaa", "processing", 5.0), ("bbbbbbbbbbbb", "processing", 2.0),
                             ("cccccccccccc", "done", 1.0)]:
        d = tmp_path / name
        d.mkdir()
        jobs.write_state(d, status=status, stage="queued", created_at=ts, error=None)
    (tmp_path / "dddddddddddd").mkdir()  # no state.json: ignored
    assert jobs.pending_jobs(tmp_path) == ["bbbbbbbbbbbb", "aaaaaaaaaaaa"]
