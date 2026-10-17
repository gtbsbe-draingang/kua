"""Evaluation CLI: `run` scores the service per clip, `sweep` re-picks thresholds offline."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np

from eval.make_labels import norm_name
from eval.score import check_schema, make_record, match_events, summarize
from rules import load_rules

POLL_SEC = 1.0
TIMEOUT_SEC = 330.0
RETRIES = 3
GRID = [round(0.05 * i, 2) for i in range(1, 20)]


def load_labels(path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            c = out.setdefault(row["clip"], {"has_violation": int(row["has_violation"]), "truth": []})
            if row["t_label"] != "":
                c["truth"].append({"t_label": float(row["t_label"]), "type_id": row.get("type_id") or ""})
    return out


def resolve_clips(directory) -> dict[str, Path]:
    """Latin clip name (X-<n>) -> file; handles Cyrillic Х and any .mov case."""
    return {norm_name(p.name): p for p in sorted(Path(directory).iterdir())
            if p.is_file() and p.suffix.lower() == ".mov"}


def _argmax_type(probs: dict) -> str | None:
    cand = {k: v for k, v in (probs or {}).items() if k != "none"}
    return max(cand, key=cand.get) if cand else None


def _pred_from_body(body: dict) -> list[dict]:
    events = (body.get("debug") or {}).get("events", [])
    pred = []
    for v in body.get("violations", []):
        tc = v.get("timecode_sec")
        near = [e for e in events if e.get("kept") and abs(e.get("timecode", 1e9) - tc) <= 0.06]
        near.sort(key=lambda e: abs(e["timecode"] - tc))
        pred.append({"timecode_sec": tc, "type_id": v.get("type_id"),
                     "type_id_no_judge": _argmax_type(near[0].get("type_probs")) if near else None})
    return pred


def _request(fn, *a, **kw):
    last = None
    for i in range(RETRIES):
        try:
            return fn(*a, **kw)
        except httpx.HTTPError as e:
            last = e
            time.sleep(1.0)
    raise last


def run_clip(url, key, clip, path, out_dir, info, rules) -> dict:
    hdr = {"X-API-Key": key}
    body, status, t0 = {}, "failed", time.time()
    latency = None
    try:
        with httpx.Client(base_url=url, timeout=60.0) as cl:
            def post():
                with open(path, "rb") as f:
                    r = cl.post("/api/v1/analyze", headers=hdr, files={"file": (path.name, f)})
                r.raise_for_status()
                return r.json()["job_id"]
            jid = _request(post)
            while True:
                r = _request(cl.get, f"/api/v1/jobs/{jid}", headers=hdr, params={"debug": 1})
                r.raise_for_status()
                body = r.json()
                if body.get("status") in ("done", "failed"):
                    status = body["status"]
                    break
                if time.time() - t0 > TIMEOUT_SEC:
                    status = "timeout"
                    break
                time.sleep(POLL_SEC)
            latency = round(time.time() - t0, 2)
    except Exception as e:  # network or HTTP failure after retries: record, never crash the run
        body = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
        status = "failed"
    (out_dir / f"{clip}.json").write_text(json.dumps(body, ensure_ascii=False))
    errs = check_schema(body, rules) if status == "done" else []
    pred = _pred_from_body(body) if status == "done" else []
    return make_record(clip, info["has_violation"], info["truth"], pred,
                       body.get("verdict") if status == "done" else None, status, latency, errs)


def _times(items, key):
    return ",".join(f"{x[key]:g}" for x in items) or "-"


def print_table(recs):
    print(f"{'clip':<7}{'truth':<14}{'pred':<14}{'verdict':<13}{'result':<8}{'latency':>8}")
    for r in recs:
        if r["status"] != "done":
            res = r["status"]
        elif r["has_violation"]:
            res = "found" if r["found"] else "missed"
        else:
            res = "flagged" if r["verdict"] == "violation" else "clean"
        lat = "-" if r["latency_s"] is None else f"{r['latency_s']:.1f}s"
        print(f"{r['clip']:<7}{_times(r['truth'], 't_label'):<14}{_times(r['pred'], 'timecode_sec'):<14}"
              f"{str(r['verdict']):<13}{res:<8}{lat:>8}")


def cmd_run(a) -> int:
    labels = load_labels(a.labels)
    files = resolve_clips(a.clips)
    only = {s.strip() for s in a.only.split(",")} if a.only else None
    rules = load_rules(a.rules, strict=False)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = []
    for clip, info in labels.items():
        if only and clip not in only:
            continue
        if clip not in files:
            print(f"skip {clip}: no file in {a.clips}", file=sys.stderr)
            continue
        todo.append((clip, info))
    todo.sort(key=lambda x: int(x[0].split("-")[1]))
    with ThreadPoolExecutor(max_workers=max(1, a.concurrent)) as ex:
        futs = [ex.submit(run_clip, a.url, a.key, c, files[c], out, i, rules) for c, i in todo]
        recs = [f.result() for f in futs]
    print_table(recs)
    s = summarize(recs)
    print(json.dumps(s, indent=2, ensure_ascii=False))
    (out / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=False))
    (out / "per_clip.json").write_text(json.dumps(recs, indent=2, ensure_ascii=False))
    return 0


# ---- sweep ----------------------------------------------------------------

def _load_run(run_dir, labels):
    clips = []
    for p in sorted(Path(run_dir).glob("*.json")):
        if p.stem in labels:
            body = json.loads(p.read_text(encoding="utf-8"))
            if body.get("status") == "done" and "debug" in body:
                clips.append((p.stem, body["debug"]["events"]))
            else:
                print(f"sweep: skipping {p.stem} (status {body.get('status')})", file=sys.stderr)
    return clips


def _keep(kd, ground, verify, vlm_ok, th) -> bool:
    r = kd(ground, verify, vlm_ok, th)
    return bool(r[0] if isinstance(r, tuple) else r)


def cmd_sweep(a) -> int:
    try:
        from pipeline.logic import keep_decision
    except ImportError:
        print("sweep needs pipeline.logic.keep_decision (Task 7), which is not available yet", file=sys.stderr)
        return 2
    labels = load_labels(a.labels)
    clips = _load_run(a.run, labels)
    events = [(ci, e) for ci, (_, evs) in enumerate(clips) for e in evs]
    n = len(events)
    truths = [(ci, t["t_label"]) for ci, (c, _) in enumerate(clips) for t in labels[c]["truth"]]
    ground = np.array([e["ground_score"] for _, e in events], dtype=float)
    # M[e, t]: event e lies within tol of truth t (same clip)
    M = np.zeros((n, len(truths)), dtype=float)
    for k, (ci, e) in enumerate(events):
        for j, (tc, tl) in enumerate(truths):
            if tc == ci and abs(e["timecode"] - tl) <= 2.0:
                M[k, j] = 1.0
    clean_mat = np.zeros((n, len(clips)), dtype=float)
    for k, (ci, _) in enumerate(events):
        if not labels[clips[ci][0]]["has_violation"]:
            clean_mat[k, ci] = 1.0
    truth_clip = np.array([tc for tc, _ in truths], dtype=int)
    viol_clips = [ci for ci, (c, _) in enumerate(clips) if labels[c]["has_violation"]]
    base = json.loads(Path(a.out).read_text()) if Path(a.out).is_file() else {}

    def score(kept):  # kept: (K, n) bool -> recall (K,), fp (K,)
        kf = kept.astype(float)
        covered = (kf @ M) > 0
        fp = ((kf @ clean_mat) > 0).sum(axis=1)
        n_ok = np.zeros(kept.shape[0])
        for ci in viol_clips:
            idx = np.where(truth_clip == ci)[0]
            n_ok += covered[:, idx].all(axis=1) if len(idx) else 0
        rec = n_ok / len(viol_clips) if viol_clips else 1.0
        return rec, fp

    cand_masks = np.stack([ground >= c - 1e-9 for c in GRID])  # (19, n)
    vlm_ok = [e.get("verify_score") is not None for _, e in events]
    best, best_key = None, None
    for v in GRID:
        for s in GRID:
            for f in GRID:
                th = {**base, "verify": v, "strong": s, "fallback": f}
                kd = np.array([_keep(keep_decision, e["ground_score"], e.get("verify_score"), vlm_ok[k], th)
                               for k, (_, e) in enumerate(events)], dtype=bool)
                rec, fp = score(cand_masks & kd[None, :])
                for i, c in enumerate(GRID):
                    if rec[i] >= 1.0 - 1e-9:
                        key = (int(fp[i]), c, v, -s, f)
                        if best_key is None or key < best_key:
                            best_key, best = key, (c, v, s, f, int(fp[i]))
    if best is None:
        print("no setting reaches recall 1.0; best ground_score near each label:")
        for ci in viol_clips:
            for t in labels[clips[ci][0]]["truth"]:
                near = [e["ground_score"] for e in clips[ci][1] if abs(e["timecode"] - t["t_label"]) <= 2.0]
                print(f"  {clips[ci][0]} @ {t['t_label']:g}s: {max(near) if near else 'no event'}")
        return 1
    c, v, s, f, fp = best
    c2, v2 = round(max(c - 0.05, 0.05), 2), round(max(v - 0.05, 0.05), 2)

    def eval_at(c_, v_):
        th = {**base, "verify": v_, "strong": s, "fallback": f}
        kd = np.array([_keep(keep_decision, e["ground_score"], e.get("verify_score"), vlm_ok[k], th)
                       for k, (_, e) in enumerate(events)], dtype=bool)
        rec, fpp = score((ground >= c_ - 1e-9)[None, :] & kd[None, :])
        return float(rec[0]), int(fpp[0])

    print(f"chosen: cand={c} verify={v} strong={s} fallback={f}  recall={eval_at(c, v)[0]:.3f} fp_clips={fp}")
    r2, fp2 = eval_at(c2, v2)
    print(f"loosened: cand={c2} verify={v2}  recall={r2:.3f} fp_clips={fp2}")
    out = {**base, "cand": c2, "verify": v2, "strong": s, "fallback": f}
    Path(a.out).write_text(json.dumps(out, indent=2))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="eval.eval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", required=True)
    r.add_argument("--key", required=True)
    r.add_argument("--labels", default="data/labels.csv")
    r.add_argument("--clips", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--concurrent", type=int, default=1)
    r.add_argument("--only")
    r.add_argument("--rules", default="rules.yaml")
    sw = sub.add_parser("sweep")
    sw.add_argument("--run", required=True)
    sw.add_argument("--labels", default="data/labels.csv")
    sw.add_argument("--out", default="thresholds.json")
    a = ap.parse_args(argv)
    return cmd_run(a) if a.cmd == "run" else cmd_sweep(a)


if __name__ == "__main__":
    sys.exit(main())
