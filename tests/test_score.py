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


def ref_keep(ground, verify, vlm_ok, th):
    if vlm_ok and verify is not None and verify >= th["verify"]:
        return True, "verified"
    if th["strong"] is not None and ground >= th["strong"]:
        return True, "strong_ground"
    if not vlm_ok and ground >= th["fallback"]:
        return True, "fallback_no_vlm"
    return False, "rejected"


HDR = "clip,has_violation,t_label,t_start,t_end,type_id,angle,source,note\n"


def _write_run(tmp_path, clips, extra_rows=""):
    run = tmp_path / "run"; run.mkdir()
    labels = tmp_path / "l.csv"
    labels.write_text(HDR + "X-1,1,10,,,,main,given,\nX-2,0,,,,,main,given,\n" + extra_rows)
    for name, body in clips.items():
        (run / f"{name}.json").write_text(json.dumps(body))
    return run, labels


def _done(*events):
    return {"status": "done", "verdict": "x", "violations": [], "debug": {"events": list(events)}}


def _sweep(run, labels, out, keep=ref_keep):
    return ev.main(["sweep", "--run", str(run), "--labels", str(labels), "--out", str(out)], keep=keep)


def _rows(text, k, v):
    """The table rows (as tokens) for top_k k and verify v."""
    return [ln.split() for ln in text.splitlines() if ln.split()[:2] == [str(k), f"{v:.2f}"]]


# Legacy (segments) runs: no ranks and no generator in debug; cand masks events by ground_score, strong is
# swept with "off", and the point is the one with the most found clips, then the fewest FP clips, then the
# loosest verify, then the loosest cand, then strong off or the highest strong.

def test_sweep_selection(tmp_path, capsys):
    run, labels = _write_run(tmp_path, {
        "X-1": _done(_dbg_event(10, 0.6, 0.7)),
        "X-2": _done(_dbg_event(5, 0.3, 0.2)),
    })
    out = tmp_path / "t.json"
    out.write_text(json.dumps({"cand": 0.25, "verify": 0.5, "strong": 0.8, "fallback": 0.5, "top_k": 6}))
    assert _sweep(run, labels, out) == 0
    th = json.loads(out.read_text())
    # found 1 with fp 0 at the loosest verify (0.01) needs cand > 0.3 to drop the clean event: cand 0.35
    assert (th["generator"], th["cand"], th["verify"], th["strong"], th["fallback"], th["top_k"]) == (
        "segments", 0.35, 0.01, None, 0.5, 6)
    text = capsys.readouterr().out
    assert "chosen: verify=0.01 top_k=6 cand=0.35 strong=off  verdict_recall=1/1 found=1/1 fp_clips=0/1" in text
    assert _rows(text, 6, 0.21) == [["6", "0.21", "0.05", "off", "1/1", "1/1", "0/1", "1.000", "1.00"]]
    assert _rows(text, 6, 0.2) == [["6", "0.20", "0.05", "off", "1/1", "1/1", "1/1", "0.500", "1.00"]]


def test_sweep_cand_excludes_clean_event(tmp_path):
    # the clean event passes every verify setting below 0.95, so only cand can exclude it (cand 0.35 > 0.3)
    run, labels = _write_run(tmp_path, {
        "X-1": _done(_dbg_event(10, 0.6, 0.9)),
        "X-2": _done(_dbg_event(5, 0.3, 0.95)),
    })
    out = tmp_path / "t.json"
    assert _sweep(run, labels, out) == 0
    th = json.loads(out.read_text())
    assert (th["cand"], th["verify"], th["strong"]) == (0.35, 0.01, None)


def test_sweep_strong_is_chosen_only_when_it_finds_more(tmp_path, capsys):
    # InternVideo3 says no everywhere: only strong >= 0.55 (above the clean event) and <= 0.9 finds X-1
    run, labels = _write_run(tmp_path, {
        "X-1": _done(_dbg_event(10, 0.9, 0.0)),
        "X-2": _done(_dbg_event(5, 0.5, 0.0)),
    })
    out = tmp_path / "t.json"
    assert _sweep(run, labels, out) == 0
    th = json.loads(out.read_text())
    assert (th["cand"], th["verify"], th["strong"]) == (0.05, 0.01, 0.9)
    text = capsys.readouterr().out
    off = [ln.split() for ln in text.splitlines() if ln.split()[:1] == ["off"]]
    assert off and off[0][5] == "0/1"  # strong off finds nothing
    assert any(ln.split()[:2] == ["0.90", "0.05"] for ln in text.splitlines())


def test_sweep_failed_violation_clip_counts_as_missed(tmp_path, capsys):
    run, labels = _write_run(tmp_path, {
        "X-1": {"status": "failed", "error": "x"},
        "X-2": _done(_dbg_event(5, 0.3, 0.2)),
    })
    out = tmp_path / "t.json"
    assert _sweep(run, labels, out) == 0
    text = capsys.readouterr().out
    assert "recall 1.0 is not reachable" in text and "X-1" in text
    assert json.loads(out.read_text())["verify"] == 0.01  # fp 0 through cand 0.35


def test_sweep_absent_violation_clip_counts_as_missed(tmp_path, capsys):
    run, labels = _write_run(tmp_path, {"X-2": _done(_dbg_event(5, 0.3, 0.2))})
    out = tmp_path / "t.json"
    assert _sweep(run, labels, out) == 0
    assert out.exists() and "recall 1.0 is not reachable" in capsys.readouterr().out


def test_sweep_without_usable_clips_writes_nothing(tmp_path):
    run, labels = _write_run(tmp_path, {"X-1": {"status": "failed", "error": "x"}})
    out = tmp_path / "t.json"
    assert _sweep(run, labels, out) == 1
    assert not out.exists()


def test_sweep_with_real_logic(tmp_path):
    pytest.importorskip("pipeline.logic")
    run, labels = _write_run(tmp_path, {
        "X-1": _done(_dbg_event(10, 0.6, 0.7)), "X-2": _done(_dbg_event(5, 0.3, 0.2))})
    out = tmp_path / "t.json"
    assert ev.main(["sweep", "--run", str(run), "--labels", str(labels), "--out", str(out)]) == 0


# Cells runs: every event counts (no cand mask), strong stays off, top_k is simulated from each event's rank.

CELLS_TH = {"generator": "cells", "cell_sec": 4.0, "rank": "logit", "cand": 0.25, "verify": 0.05, "strong": None,
            "fallback": 0.5, "top_k": 2, "merge_sec": 3, "anchor": "center", "anchor_offset_sec": 0}


def _cells_event(t, verify, rank, ground=0.0):
    return {**_dbg_event(t, ground, verify), "t_start": t - 2, "t_end": t + 2, "rank": rank}


def _cells_done(*events):
    return {"status": "done", "verdict": "x", "violations": [],
            "debug": {"events": list(events), "generator": "cells", "cell_sec": 4.0, "generator_fallback": False,
                      "versions": {"thresholds": json.dumps(CELLS_TH, sort_keys=True)}}}


def _cells_run(tmp_path):
    # X-3's label at 30 s has no event within 2 s, so recall 1.0 cannot be reached. X-1's label is only in its
    # rank-1 cell (a run's ground_score plays no part: 0.0 everywhere). Clean X-2 is flagged at verify <= 0.30,
    # clean X-4 at verify <= 0.33.
    return _write_run(tmp_path, {
        "X-1": _cells_done(_cells_event(2, 0.60, 0), _cells_event(10, 0.40, 1)),
        "X-2": _cells_done(_cells_event(5, 0.30, 0), _cells_event(9, 0.03, 1)),
        "X-3": _cells_done(_cells_event(40, 0.70, 0), _cells_event(44, 0.20, 1)),
        "X-4": _cells_done(_cells_event(3, 0.33, 0)),
    }, extra_rows="X-3,1,30,,,,main,given,\nX-4,0,,,,,main,given,\n")


def test_sweep_cells_prints_the_curve_and_picks_most_found_then_fewest_fp(tmp_path, capsys):
    run, labels = _cells_run(tmp_path)
    out = tmp_path / "t.json"
    out.write_text(json.dumps({**CELLS_TH, "top_k": 6, "verify": 0.5}))
    assert _sweep(run, labels, out) == 0
    text = capsys.readouterr().out
    # top_k verify verdict_recall found fp_clips verdict_acc events/flagged
    assert _rows(text, 2, 0.35) == [["2", "0.35", "2/2", "1/2", "0/2", "1.000", "1.50", "<-", "chosen"]]
    assert _rows(text, 2, 0.4) == [["2", "0.40", "2/2", "1/2", "0/2", "1.000", "1.50"]]
    assert _rows(text, 2, 0.3) == [["2", "0.30", "2/2", "1/2", "2/2", "0.500", "1.25"]]
    assert _rows(text, 2, 0.01) == [["2", "0.01", "2/2", "1/2", "2/2", "0.500", "1.75"]]
    assert _rows(text, 1, 0.35) == [["1", "0.35", "2/2", "0/2", "0/2", "1.000", "1.00"]]  # X-1's label is rank 1
    assert _rows(text, 2, 0.95) == [["2", "0.95", "0/2", "0/2", "0/2", "0.500", "-"]]
    assert not _rows(text, 3, 0.35)  # no top_k above the run's own
    assert len([ln for ln in text.splitlines() if ln.split()[:1] in (["1"], ["2"])]) == 2 * 43
    assert "recall 1.0 is not reachable" in text and "X-3" in text
    th = json.loads(out.read_text())
    assert th == {**CELLS_TH, "verify": 0.35, "top_k": 2, "strong": None}


def test_sweep_cells_events_without_rank_count_at_every_top_k(tmp_path, capsys):
    # the fallback grid has no ranks: its events are kept at every simulated top_k
    run, labels = _write_run(tmp_path, {
        "X-1": _cells_done(_cells_event(2, 0.6, None), _cells_event(10, 0.4, None)),
        "X-2": _cells_done(_cells_event(5, 0.1, 0)),
    })
    out = tmp_path / "t.json"
    assert _sweep(run, labels, out) == 0
    assert _rows(capsys.readouterr().out, 1, 0.4)[0][:5] == ["1", "0.40", "1/1", "1/1", "0/1"]
    th = json.loads(out.read_text())
    assert (th["verify"], th["top_k"]) == (0.11, 2)  # the loosest verify with fp 0; the run's top_k on a tie


def test_sweep_refuses_a_run_with_two_generators(tmp_path):
    run, labels = _write_run(tmp_path, {
        "X-1": _cells_done(_cells_event(10, 0.6, 0)), "X-2": _done(_dbg_event(5, 0.3, 0.2))})
    out = tmp_path / "t.json"
    assert _sweep(run, labels, out) == 2
    assert not out.exists()


def test_summary_counts_failed_and_missing():
    recs = [
        _rec("X-1", 1, [{"t_label": 22, "type_id": ""}], [{"timecode_sec": 22}], "violation"),
        make_record("X-2", 1, [{"t_label": 10, "type_id": ""}], [], None, "timeout", 330.0, []),
        make_record("X-3", 1, [{"t_label": 10, "type_id": ""}], [], None, "missing", None, []),
        _rec("X-4", 0, [], [], "no_violation"),
    ]
    s = summarize(recs)
    assert s["recall_clips"] == pytest.approx(1 / 3)
    assert s["verdict_acc"] == 0.5
    assert s["failed_clips"] == ["X-2"] and s["missing_clips"] == ["X-3"]
    assert s["max_latency_s"] == 330.0


def test_malformed_timecode_does_not_crash():
    body = {"status": "done", "verdict": "violation",
            "violations": [{"timecode_sec": None}, {"timecode_sec": 3.0, "type_id": "other"}],
            "debug": {"events": [{"kept": True, "timecode": None}]}}
    pred = ev._pred_from_body(body)
    assert [p["timecode_sec"] for p in pred] == [3.0]
    assert match_events([{"timecode_sec": None}, {"timecode_sec": "x"}], [{"t_label": 3}]) == []


def test_summary_events_per_flagged_clip():
    recs = [
        _rec("X-1", 1, [{"t_label": 22, "type_id": ""}], [{"timecode_sec": 22}, {"timecode_sec": 40}], "violation"),
        _rec("X-3", 0, [], [{"timecode_sec": 5}], "violation"),
        _rec("X-5", 0, [], [], "no_violation"),
        make_record("X-2", 1, [{"t_label": 10, "type_id": ""}], [], None, "timeout", 330.0, []),
    ]
    s = summarize(recs)
    assert s["events_per_flagged_clip"] == 1.5 and s["verdict_acc"] == 0.5
    assert summarize(recs[2:])["events_per_flagged_clip"] is None
