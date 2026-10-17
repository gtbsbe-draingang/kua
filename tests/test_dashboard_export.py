"""tools/dashboard_export.py: an eval run folder -> one text-only JSON for the results dashboard."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import dashboard_export as dx  # noqa: E402

from eval.score import make_record  # noqa: E402

FAKE_IMG = "QUJDREVGRw=="  # a short fake string, not an image
THRESHOLDS = {"generator": "grid", "cell_sec": 8.0, "top_k": 8, "verify": 0.05, "strong": None,
              "fallback": 0.5, "zoom": False, "verify_mode": "single"}
VERSIONS = {"pipeline": "real", "verifier": "iv3 at http://127.0.0.1:30000", "rules": "r1",
            "thresholds": json.dumps(THRESHOLDS, sort_keys=True)}


def _event(t, verify, kept, type_source="judge"):
    return {"rule_id": "other", "t_start": t - 4, "t_end": t + 4, "timecode": t, "ground_score": 0.0,
            "type_probs": {}, "verify_score": verify, "kept": kept, "reason": "verified" if kept else "rejected",
            "type_source": type_source, "evidence": [{"t": t}]}


def _violation(t, conf, clause="ст. 5", img=FAKE_IMG):
    return {"type": "Касса без присмотра", "timecode_sec": t, "description": "text",
            "evidence_frames": [{"timestamp_sec": t, "image_base64": img}],
            "type_id": "cash_unattended",
            "rule": {"document": "Приказ 1", "clause": clause, "quote": "цитата"}, "confidence": conf}


def _body(verdict, events, violations):
    return {"job_id": "abc", "status": "done", "verdict": verdict, "violations": violations,
            "debug": {"events": events, "timings": {"total": 30.0}, "versions": VERSIONS,
                      "generator": "grid", "cell_sec": 8.0}}


def _run(tmp_path, name="run", clause="ст. 5", latencies=(40.0, 50.0, 60.0)):
    """X-1 violation found, X-2 violation missed, X-3 clean but flagged."""
    run = tmp_path / name
    run.mkdir()
    bodies = {
        "X-1": _body("violation", [_event(10.0, 0.9, True), _event(30.0, 0.01, False)],
                     [_violation(10.0, 0.9, clause)]),
        "X-2": _body("no_violation", [_event(20.0, 0.01, False)], []),
        "X-3": _body("violation", [_event(5.0, 0.4, True, "fallback")], [_violation(5.0, 0.4)]),
    }
    truth = {"X-1": (1, [{"t_label": 11.0, "type_id": ""}]), "X-2": (1, [{"t_label": 20.0, "type_id": ""}]),
             "X-3": (0, [])}
    recs = []
    for (clip, body), lat in zip(bodies.items(), latencies, strict=True):
        (run / f"{clip}.json").write_text(json.dumps(body, ensure_ascii=False))
        hv, tr = truth[clip]
        pred = [{"timecode_sec": v["timecode_sec"], "type_id": v["type_id"]} for v in body["violations"]]
        recs.append(make_record(clip, hv, tr, pred, body["verdict"], "done", lat, []))
    from eval.score import summarize
    (run / "per_clip.json").write_text(json.dumps(recs))
    (run / "summary.json").write_text(json.dumps(summarize(recs)))
    labels = tmp_path / "labels.csv"
    labels.write_text("clip,has_violation,t_label,t_start,t_end,type_id,angle,source,note\n"
                      "X-1,1,11,,,,main,given,\nX-2,1,20,,,,main,given,\nX-3,0,,,,,main,given,\n")
    return run, labels


def _walk(x):
    if isinstance(x, dict):
        for k, v in x.items():
            yield k
            yield from _walk(v)
    elif isinstance(x, list):
        for v in x:
            yield from _walk(v)
    else:
        yield x


def test_export_strips_images(tmp_path):
    run, labels = _run(tmp_path)
    out = dx.export(run, labels)
    text = json.dumps(out, ensure_ascii=False)
    assert FAKE_IMG not in text
    assert not [k for k in _walk(out) if isinstance(k, str) and "image" in k.lower()]
    assert "evidence_frames" not in text
    assert text.isascii() is False  # Russian text survives
    assert out["summary"]["n_clips"] == 3
    assert out["thresholds"]["verify"] == 0.05 and out["versions"]["pipeline"] == "real"


def test_export_found_and_fp(tmp_path):
    run, labels = _run(tmp_path)
    out = dx.export(run, labels)
    rows = {c["clip"]: c for c in out["clips"]}
    assert (rows["X-1"]["label"], rows["X-1"]["found"], rows["X-1"]["fp"]) == ("violation", True, False)
    assert (rows["X-2"]["label"], rows["X-2"]["found"], rows["X-2"]["fp"]) == ("violation", False, False)
    assert (rows["X-3"]["label"], rows["X-3"]["found"], rows["X-3"]["fp"]) == ("clean", False, True)
    assert rows["X-1"]["label_sec"] == [11.0] and rows["X-3"]["label_sec"] == []
    assert rows["X-1"]["verdict"] == "violation" and rows["X-2"]["verdict"] == "no_violation"
    ev = rows["X-1"]["events"]
    assert ev == [{"timecode_sec": 10.0, "type_id": "cash_unattended", "type": "Касса без присмотра",
                   "type_source": "judge", "confidence": 0.9, "document": "Приказ 1", "clause": "ст. 5"}]
    assert rows["X-3"]["events"][0]["type_source"] == "fallback"
    assert rows["X-1"]["cells"] == 2 and rows["X-2"]["cells"] == 1
    assert rows["X-1"]["latency_s"] == 40.0
    assert "quote" not in json.dumps(out, ensure_ascii=False)
    # the sweep reuses eval's own table: one row per verify point, at the run's top_k
    sw = out["sweep"]
    row = next(r for r in sw if r["verify"] == 0.05)
    assert (row["top_k"], row["found"], row["fp_clips"]) == (8, 1, 1)
    assert set(row) == {"verify", "top_k", "found", "fp_clips", "verdict_acc", "events_per_flagged"}
    t = out["timings"]["run"]
    assert (t["n"], t["mean"], t["max"]) == (3, 50.0, 60.0) and t["p95"] == pytest.approx(59.0)


def test_export_c5_block(tmp_path):
    run, labels = _run(tmp_path)
    c5, _ = _run(tmp_path, "run_c5", latencies=(100.0, 200.0, 250.0))
    out = dx.export(run, labels, c5_dir=c5)
    assert out["timings"]["c5"]["max"] == 250.0
    assert out["c5"]["over_deadline"] == ["X-3"]


def test_export_refuses_leftover_image(tmp_path):
    run, labels = _run(tmp_path, clause="x" * 20001)  # a long value that is copied into the output
    with pytest.raises(dx.ExportError, match="20000"):
        dx.export(run, labels)
    with pytest.raises(dx.ExportError, match="image"):
        dx.check_text_only({"versions": {"image_x": "a"}})
    with pytest.raises(dx.ExportError, match="image"):
        dx.check_text_only({"a": [{"Image_Base64": "a"}]})
    dx.check_text_only({"a": ["x" * 20000]})
