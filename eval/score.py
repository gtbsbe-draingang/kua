"""Scoring of per-clip results against labels."""
from __future__ import annotations

import base64
import binascii
import statistics


def match_events(pred: list[dict], truth: list[dict], tol: float = 2.0) -> list[tuple]:
    """One-to-one greedy matching by smallest absolute time error (<= tol).

    Returns (pred_index, truth_index, abs_error) sorted by truth index.
    """
    pairs = sorted(
        (abs(float(p["timecode_sec"]) - float(t["t_label"])), i, j)
        for i, p in enumerate(pred) for j, t in enumerate(truth)
    )
    used_p, used_t, out = set(), set(), []
    for err, i, j in pairs:
        if err > tol:
            break
        if i in used_p or j in used_t:
            continue
        used_p.add(i)
        used_t.add(j)
        out.append((i, j, err))
    return sorted(out, key=lambda m: (m[1], m[0]))


def clip_found(has_violation, truth, pred, verdict) -> bool:
    """Found = verdict is violation and every truth event is matched."""
    if not has_violation or verdict != "violation":
        return False
    return len(match_events(pred, truth)) == len(truth)


def make_record(clip, has_violation, truth, pred, verdict, status, latency_s, schema_errors):
    return {
        "clip": clip, "has_violation": int(has_violation), "truth": truth, "pred": pred,
        "verdict": verdict, "found": clip_found(has_violation, truth, pred, verdict),
        "latency_s": latency_s, "status": status, "schema_errors": list(schema_errors),
    }


def check_schema(body: dict, rules: dict) -> list[str]:
    """Problems in a done body against the brief's response schema (empty = fine)."""
    errs = []
    if body.get("verdict") not in ("violation", "no_violation"):
        errs.append(f"bad verdict {body.get('verdict')!r}")
    vs = body.get("violations")
    if not isinstance(vs, list):
        return errs + ["violations is not a list"]
    for i, v in enumerate(vs):
        if not isinstance(v, dict):
            errs.append(f"violation #{i} is not an object")
            continue
        for k in ("type", "description"):
            if not isinstance(v.get(k), str) or not v.get(k):
                errs.append(f"violation #{i}: missing {k}")
        if not isinstance(v.get("timecode_sec"), (int, float)) or isinstance(v.get("timecode_sec"), bool):
            errs.append(f"violation #{i}: bad timecode_sec")
        if v.get("type_id") not in rules:
            errs.append(f"violation #{i}: type_id {v.get('type_id')!r} not in rules")
        fr = v.get("evidence_frames")
        if not isinstance(fr, list) or not 1 <= len(fr) <= 3:
            errs.append(f"violation #{i}: evidence_frames must hold 1-3 items")
            continue
        for k, f in enumerate(fr):
            if not isinstance(f, dict) or not isinstance(f.get("timestamp_sec"), (int, float)):
                errs.append(f"violation #{i} frame #{k}: bad timestamp_sec")
                continue
            try:
                raw = base64.b64decode(f.get("image_base64") or "", validate=True)
            except (binascii.Error, ValueError, TypeError):
                raw = b""
            if not raw.startswith(b"\xff\xd8"):
                errs.append(f"violation #{i} frame #{k}: image_base64 is not a JPEG")
    return errs


def _share(a, b):
    return a / b if b else None


def summarize(per_clip: list[dict]) -> dict:
    viol = [r for r in per_clip if r["has_violation"]]
    clean = [r for r in per_clip if not r["has_violation"]]
    found = sum(1 for r in viol if clip_found(1, r["truth"], r["pred"], r["verdict"]))
    done = [r for r in per_clip if r["status"] == "done"]
    ok = sum(1 for r in done if r["verdict"] == ("violation" if r["has_violation"] else "no_violation"))
    errors, n_truth, typed, type_ok, type_ok_nj = [], 0, 0, 0, 0
    for r in per_clip:
        n_truth += len(r["truth"])
        for pi, ti, err in match_events(r["pred"], r["truth"]):
            errors.append(err)
            tt = r["truth"][ti].get("type_id")
            if tt:
                typed += 1
                type_ok += r["pred"][pi].get("type_id") == tt
                type_ok_nj += r["pred"][pi].get("type_id_no_judge") == tt
    lat = [r["latency_s"] for r in per_clip if r.get("latency_s") is not None]
    return {
        "n_clips": len(per_clip),
        "recall_clips": _share(found, len(viol)),
        "fp_clips": sum(1 for r in clean if r["status"] == "done" and r["verdict"] == "violation"),
        "verdict_acc": _share(ok, len(done)),
        "tc_median": statistics.median(errors) if errors else None,
        "tc_max": max(errors) if errors else None,
        "tc_within_1s": _share(sum(e <= 1.0 for e in errors), n_truth),
        "tc_within_2s": _share(sum(e <= 2.0 for e in errors), n_truth),
        "type_acc": _share(type_ok, typed),
        "type_acc_no_judge": _share(type_ok_nj, typed),
        "max_latency_s": max(lat) if lat else None,
        "schema_errors": [f"{r['clip']}: {e}" for r in per_clip for e in r.get("schema_errors", [])],
        "failed_clips": [r["clip"] for r in per_clip if r["status"] != "done"],
    }
