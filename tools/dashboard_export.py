"""One text-only JSON file for the results dashboard, from an eval run folder.

    python tools/dashboard_export.py --run RUN [--c5 RUN_C5] [--labels data/labels.csv] --out dashboard.json

RUN is the folder `eval run` wrote (one <clip>.json per clip, summary.json, per_clip.json). The per-clip API
responses hold base64 evidence frames: every image field is dropped before anything is read from them, and the
finished file is checked: a key containing "image" or a value longer than 20000 characters stops the export.
The operating-point table is eval sweep's own (eval.eval.cmd_sweep), not a second implementation.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import eval as ev  # noqa: E402

MAX_VALUE_CHARS = 20000
DEADLINE_SEC = 240.0
EVENT_TOL = 0.06  # a violation and the debug event it came from share a rounded timecode


class ExportError(RuntimeError):
    pass


def _strip_images(x):
    """Copy of x without evidence_frames and without any key naming an image."""
    if isinstance(x, dict):
        return {k: _strip_images(v) for k, v in x.items()
                if k != "evidence_frames" and "image" not in str(k).lower()}
    if isinstance(x, list):
        return [_strip_images(v) for v in x]
    return x


def check_text_only(x, path="$") -> None:
    if isinstance(x, dict):
        for k, v in x.items():
            if "image" in str(k).lower():
                raise ExportError(f"image field left at {path}.{k}")
            check_text_only(v, f"{path}.{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            check_text_only(v, f"{path}[{i}]")
    elif isinstance(x, str) and len(x) > MAX_VALUE_CHARS:
        raise ExportError(f"value longer than {MAX_VALUE_CHARS} characters at {path} ({len(x)})")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _events(body: dict) -> list[dict]:
    """The kept events of a done body: timecode, type, source, confidence and citation document + clause."""
    dbg_events = [e for e in (body.get("debug") or {}).get("events", []) if isinstance(e, dict) and e.get("kept")]
    out = []
    for v in body.get("violations") or []:
        tc = v.get("timecode_sec")
        near = sorted((e for e in dbg_events if isinstance(e.get("timecode"), (int, float))
                       and isinstance(tc, (int, float)) and abs(e["timecode"] - tc) <= EVENT_TOL),
                      key=lambda e: abs(e["timecode"] - tc))
        rule = v.get("rule") or {}
        out.append({"timecode_sec": tc, "type_id": v.get("type_id"), "type": v.get("type"),
                    "type_source": near[0].get("type_source") if near else None,
                    "confidence": v.get("confidence"),
                    "document": rule.get("document"), "clause": rule.get("clause")})
    return out


def _timing(recs: list[dict]) -> dict:
    lat = [r["latency_s"] for r in recs if r.get("latency_s") is not None]
    if not lat:
        return {"n": 0, "mean": None, "p95": None, "max": None}
    return {"n": len(lat), "mean": round(float(np.mean(lat)), 2),
            "p95": round(float(np.percentile(lat, 95)), 2), "max": round(float(max(lat)), 2)}


def _sweep(run_dir, labels) -> list[dict]:
    rows: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        a = SimpleNamespace(run=str(run_dir), labels=str(labels), out=str(Path(tmp) / "t.json"))
        code = ev.cmd_sweep(a, rows=rows)
    if code != 0:
        raise ExportError(f"eval sweep exited {code} on {run_dir}")
    return rows


def export(run_dir, labels, c5_dir=None) -> dict:
    run_dir = Path(run_dir)
    lab = ev.load_labels(labels)
    summary = _read(run_dir / "summary.json")
    recs = {r["clip"]: r for r in _read(run_dir / "per_clip.json")}
    clips, debugs = [], []
    for name in sorted(lab, key=ev._clip_key):
        r = recs.get(name)
        if r is None:
            raise ExportError(f"{name} is in the labels but not in per_clip.json")
        body = {}
        if r["status"] == "done":
            body = _strip_images(_read(run_dir / f"{name}.json"))  # images go first, before any other use
            if isinstance(body.get("debug"), dict):
                debugs.append(body["debug"])
        done = r["status"] == "done"
        clips.append({
            "clip": name,
            "label": "violation" if r["has_violation"] else "clean",
            "label_sec": [t["t_label"] for t in r["truth"]],
            "status": r["status"],
            "verdict": r["verdict"],
            "found": bool(r["found"]),
            "fp": bool(done and not r["has_violation"] and r["verdict"] == "violation"),
            "events": _events(body) if done else [],
            "cells": len((body.get("debug") or {}).get("events", [])) if done else None,
            "latency_s": r["latency_s"],
        })
    versions = next((d["versions"] for d in debugs if isinstance(d.get("versions"), dict)), {})
    out = {"summary": summary, "clips": clips, "sweep": _sweep(run_dir, labels),
           "versions": versions, "thresholds": ev._run_thresholds(debugs),
           "timings": {"run": _timing(list(recs.values()))}}
    if c5_dir is not None:
        c5 = list(_read(Path(c5_dir) / "per_clip.json"))
        out["timings"]["c5"] = _timing(c5)
        out["c5"] = {"deadline_s": DEADLINE_SEC,
                     "jobs": [{"clip": r["clip"], "status": r["status"], "latency_s": r["latency_s"]} for r in c5],
                     "over_deadline": [r["clip"] for r in c5
                                       if r["latency_s"] is not None and r["latency_s"] > DEADLINE_SEC],
                     "not_done": [r["clip"] for r in c5 if r["status"] != "done"]}
    check_text_only(out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dashboard_export")
    ap.add_argument("--run", required=True)
    ap.add_argument("--c5")
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        out = export(a.run, a.labels, a.c5)
    except ExportError as e:
        print(f"dashboard_export: refused: {e}", file=sys.stderr)
        return 1
    text = json.dumps(out, ensure_ascii=False, indent=1)
    Path(a.out).write_text(text, encoding="utf-8")
    print(f"wrote {a.out} ({len(text.encode('utf-8'))} bytes, {len(out['clips'])} clips)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
