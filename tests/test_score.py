import base64
import csv
import json
from pathlib import Path

import pytest

from eval import eval as ev
from eval.score import match_events, summarize, check_schema, make_record

JPEG = base64.b64encode(b"\xff\xd8\xff\xe0abc").decode()


def test_match_within_tol():
    m = match_events([{"timecode_sec": 23.5}], [{"t_label": 22}])
    assert len(m) == 1 and m[0][:2] == (0, 0) and m[0][2] == pytest.approx(1.5)


def test_no_match_outside_tol():
    assert match_events([{"timecode_sec": 25}], [{"t_label": 22}]) == []


def test_two_events_one_clip():
    m = match_events([{"timecode_sec": 57.4}, {"timecode_sec": 67.0}],
                     [{"t_label": 57}, {"t_label": 68}])
    assert [(p, t) for p, t, _ in m] == [(0, 0), (1, 1)]


def _rec(clip, viol, truth, pred, verdict, status="done", latency=10.0):
    return make_record(clip, viol, truth, pred, verdict, status, latency, [])


def test_summarize_recall_and_fp():
    recs = [
        _rec("X-1", 1, [{"t_label": 22, "type_id": ""}], [{"timecode_sec": 22.4}], "violation"),
        _rec("X-2", 1, [{"t_label": 10, "type_id": ""}], [], "no_violation"),
        _rec("X-3", 0, [], [{"timecode_sec": 5}], "violation"),
    ]
    s = summarize(recs)
    assert s["recall_clips"] == 0.5 and s["fp_clips"] == 1
    assert s["n_clips"] == 3 and s["max_latency_s"] == 10.0
    assert s["tc_within_1s"] == 0.5 and s["tc_max"] == pytest.approx(0.4)


def test_found_requires_all_events():
    r = _rec("X-4", 1, [{"t_label": 57, "type_id": ""}, {"t_label": 68, "type_id": ""}],
             [{"timecode_sec": 57.2}], "violation")
    assert r["found"] is False
    assert summarize([r])["recall_clips"] == 0.0


def test_type_acc_ignores_missing_truth_type():
    recs = [
        _rec("X-1", 1, [{"t_label": 22, "type_id": ""}], [{"timecode_sec": 22, "type_id": "a"}], "violation"),
        _rec("X-2", 1, [{"t_label": 10, "type_id": "a"}], [{"timecode_sec": 10, "type_id": "b"}], "violation"),
    ]
    assert summarize(recs)["type_acc"] == 0.0
    assert summarize(recs[:1])["type_acc"] is None


def _good_body(type_id="other"):
    return {"status": "done", "verdict": "violation", "violations": [{
        "type": "t", "type_id": type_id, "timecode_sec": 1.0, "description": "d",
        "evidence_frames": [{"timestamp_sec": 1.0, "image_base64": JPEG}]}]}


def test_schema_check():
    rules = {"other": {}}
    assert check_schema(_good_body(), rules) == []
    bad = _good_body()
    bad["violations"][0]["evidence_frames"][0]["image_base64"] = base64.b64encode(b"nope").decode()
    assert check_schema(bad, rules)
    assert check_schema(_good_body("zzz"), rules)
    b2 = _good_body(); b2["verdict"] = "maybe"
    assert check_schema(b2, rules)


def test_resolve_clips_cyrillic(tmp_path):
    (tmp_path / "X-1.mov").write_bytes(b"")
    (tmp_path / "Х-3.MOV").write_bytes(b"")
    m = ev.resolve_clips(tmp_path)
    assert m["X-1"].name == "X-1.mov" and m["X-3"].name == "Х-3.MOV"
    assert "X-9" not in m


def test_load_labels_groups_events(tmp_path):
    p = tmp_path / "l.csv"
    p.write_text("clip,has_violation,t_label,t_start,t_end,type_id,angle,source,note\n"
                 "X-4,1,57,,,,main,given,\nX-4,1,68,,,,main,given,\nX-3,0,,,,,main,given,\n")
    lab = ev.load_labels(p)
    assert lab["X-4"]["truth"] == [{"t_label": 57.0, "type_id": ""}, {"t_label": 68.0, "type_id": ""}]
    assert lab["X-3"]["has_violation"] == 0 and lab["X-3"]["truth"] == []


def _dbg_event(t, ground, verify):
    return {"rule_id": "other", "t_start": t, "t_end": t, "timecode": t, "ground_score": ground,
            "type_probs": {}, "verify_score": verify, "kept": False, "reason": "x", "evidence": []}


def test_sweep_selection(tmp_path):
    logic = pytest.importorskip("pipeline.logic")
    run = tmp_path / "run"; run.mkdir()
    labels = tmp_path / "l.csv"
    labels.write_text("clip,has_violation,t_label,t_start,t_end,type_id,angle,source,note\n"
                      "X-1,1,10,,,,main,given,\nX-2,0,,,,,main,given,\n")
    (run / "X-1.json").write_text(json.dumps({"status": "done", "verdict": "violation", "violations": [],
        "debug": {"events": [_dbg_event(10, 0.6, 0.7)]}}))
    (run / "X-2.json").write_text(json.dumps({"status": "done", "verdict": "no_violation", "violations": [],
        "debug": {"events": [_dbg_event(5, 0.3, 0.2)]}}))
    out = tmp_path / "t.json"
    out.write_text(json.dumps({"cand": 0.25, "verify": 0.5, "strong": 0.8, "fallback": 0.5, "top_k": 6}))
    assert ev.main(["sweep", "--run", str(run), "--labels", str(labels), "--out", str(out)]) == 0
    th = json.loads(out.read_text())
    assert th["top_k"] == 6
    # the clean clip's event (ground 0.3) must stay out: cand >0.3 before loosening -> chosen cand 0.35, loosened to 0.30
    assert th["cand"] <= 0.6 and th["verify"] <= 0.7
