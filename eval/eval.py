"""Evaluation CLI: `run` scores the service per clip, `sweep` re-picks thresholds offline."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np

from eval.make_labels import norm_name
from eval.score import _num, check_schema, make_record, match_events, summarize
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
    for v in body.get("violations") or []:
        tc = v.get("timecode_sec") if isinstance(v, dict) else None
        if not _num(tc):
            continue  # check_schema reports it
        near = [e for e in events if isinstance(e, dict) and e.get("kept")
                and _num(e.get("timecode")) and abs(e["timecode"] - tc) <= 0.06]
        near.sort(key=lambda e: abs(e["timecode"] - tc))
        pred.append({"timecode_sec": tc, "type_id": v.get("type_id"),
                     "type_id_no_judge": _argmax_type(near[0].get("type_probs")) if near else None})
    return pred


def _transient(r: httpx.Response) -> bool:
    return r.status_code >= 500


def run_clip(url, key, clip, path, out_dir, info, rules) -> dict:
    hdr = {"X-API-Key": key}
    body, status, t0 = {}, "failed", time.time()
    try:
        with httpx.Client(base_url=url, timeout=60.0) as cl:
            jid = None
            for attempt in range(RETRIES):  # retry only when the request cannot have reached the server
                try:
                    with open(path, "rb") as f:
                        r = cl.post("/api/v1/analyze", headers=hdr, files={"file": (path.name, f)})
                    r.raise_for_status()  # no retry on 4xx/5xx: a 5xx may already have created a job
                    jid = r.json()["job_id"]
                    break
                except (httpx.ConnectError, httpx.ConnectTimeout):
                    if attempt == RETRIES - 1:
                        raise
                    time.sleep(1.0)
            while True:
                try:
                    r = cl.get(f"/api/v1/jobs/{jid}", headers=hdr, params={"debug": 1})
                    if not _transient(r):
                        r.raise_for_status()  # 4xx (e.g. 401) fails the clip
                        body = r.json()
                        if body.get("status") in ("done", "failed"):
                            status = body["status"]
                            break
                except httpx.TransportError:
                    pass  # transient network error: keep polling within the budget
                if time.time() - t0 > TIMEOUT_SEC:
                    status = "timeout"
                    body = {"status": "timeout"}
                    break
                time.sleep(POLL_SEC)
    except Exception as e:  # recorded, never crashes the run
        body = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
        status = "failed"
    latency = round(time.time() - t0, 2)
    try:
        (out_dir / f"{clip}.json").write_text(json.dumps(body, ensure_ascii=False))
        errs = check_schema(body, rules) if status == "done" else []
        pred = _pred_from_body(body) if status == "done" else []
        return make_record(clip, info["has_violation"], info["truth"], pred,
                           body.get("verdict") if status == "done" else None, status, latency, errs)
    except Exception as e:
        return make_record(clip, info["has_violation"], info["truth"], [], None, "failed", latency,
                           [f"post-processing crashed: {type(e).__name__}: {e}"])


def _clip_key(name: str):
    m = re.search(r"(\d+)$", name)
    return (int(m.group(1)) if m else 10**9, name)


def _times(items, key):
    return ",".join(f"{x[key]:g}" for x in items) or "-"


def print_table(recs):
    print(f"{'clip':<7}{'truth':<14}{'pred':<14}{'verdict':<13}{'result':<8}{'latency':>8}")
    for r in recs:
        if r["status"] != "done":
            res = r["status"]  # failed | timeout | missing
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
    todo, missing = [], []
    for clip, info in labels.items():
        if only and clip not in only:
            continue
        if clip not in files:
            print(f"missing {clip}: no file in {a.clips}", file=sys.stderr)
            missing.append(make_record(clip, info["has_violation"], info["truth"], [], None, "missing", None, []))
            continue
        todo.append((clip, info))
    todo.sort(key=lambda x: _clip_key(x[0]))
    with ThreadPoolExecutor(max_workers=max(1, a.concurrent)) as ex:
        futs = [ex.submit(run_clip, a.url, a.key, c, files[c], out, i, rules) for c, i in todo]
        recs = [f.result() for f in futs] + missing
    recs.sort(key=lambda r: _clip_key(r["clip"]))
    print_table(recs)
    s = summarize(recs)
    print(json.dumps(s, indent=2, ensure_ascii=False))
    (out / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=False))
    (out / "per_clip.json").write_text(json.dumps(recs, indent=2, ensure_ascii=False))
    return 0


# ---- sweep ----------------------------------------------------------------
# Re-decides which recorded candidates a run would keep at other thresholds, without rerunning the models.
# Every point is scored on the per-video verdict and on the timecodes; the whole curve is printed so a person
# can choose, and the point with the most found clips, then the fewest FP clips, then the loosest verify is
# written to --out. Cells runs: no cand mask, strong off, top_k simulated from each candidate's rank. Legacy
# segments runs: cand masks events by ground_score, and strong is swept as well (off or a value).

SKIP_NAMES = {"summary", "per_clip"}
TOL_SEC = 2.0
VERIFY_GRID = [round(0.01 * i, 2) for i in range(1, 31)] + [round(0.05 * i, 2) for i in range(7, 20)]
GRID = [round(0.05 * i, 2) for i in range(1, 20)]  # cand and strong, segments runs only
# How a run made and checked its candidates: verify is chosen on the run's own scores, so these go with it.
RUN_KEYS = ("generator", "cell_sec", "rank", "anchor", "anchor_offset_sec", "merge_sec", "zoom", "verify_mode")
RUN_DEFAULTS = {"zoom": False, "verify_mode": "single"}  # a run recorded before these switches ran like this


def _load_run(run_dir, labels):
    """Returns (clips with usable debug: [(name, debug)], unusable clip -> status, extra files not in labels)."""
    clips, unusable, extra = [], {}, []
    for p in sorted(Path(run_dir).glob("*.json")):
        if p.stem in SKIP_NAMES:
            continue
        if p.stem not in labels:
            extra.append(p.stem)
            continue
        try:
            body = json.loads(p.read_text(encoding="utf-8"))
            dbg = body["debug"] if body.get("status") == "done" else None
            if dbg is not None and not isinstance(dbg["events"], list):
                dbg = None
        except (ValueError, KeyError, TypeError, AttributeError):
            body, dbg = {}, None
        if dbg is None:
            unusable[p.stem] = body.get("status", "unreadable") if isinstance(body, dict) else "unreadable"
        else:
            clips.append((p.stem, dbg))
    for c in labels:
        if c not in unusable and c not in {n for n, _ in clips} and c not in extra:
            unusable[c] = "absent"
    return clips, unusable, extra


def _default_keep():
    try:
        from pipeline.logic import keep_decision
    except ModuleNotFoundError as e:
        if e.name != "pipeline.logic":
            raise
        return None
    return keep_decision


def _keep(keep, ground, verify, vlm_ok, th) -> bool:
    return bool(keep(ground, verify, vlm_ok, th)[0])


def _run_thresholds(debugs) -> dict:
    """The thresholds the run itself used (debug versions), or {}."""
    for d in debugs:
        try:
            th = json.loads(d["versions"]["thresholds"])
        except (KeyError, TypeError, ValueError):
            continue
        if isinstance(th, dict):
            return th
    return {}


def _rank(e) -> int | None:
    r = e.get("rank")
    return r if isinstance(r, int) and not isinstance(r, bool) and r >= 0 else None


def _pos_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 1


def _cell(x, fmt) -> str:
    return "-" if x is None else format(x, fmt)


def cmd_sweep(a, keep=None, rows=None) -> int:
    """rows: a list that receives the printed operating-point table as dicts (for tools/dashboard_export.py)."""
    keep = keep or _default_keep()
    if keep is None:
        print("sweep needs pipeline.logic.keep_decision (Task 7), which is not available yet", file=sys.stderr)
        return 2
    labels = load_labels(a.labels)
    clips, unusable, extra = _load_run(a.run, labels)
    if extra:
        print(f"sweep: in the run but not in the labels: {', '.join(extra)}", file=sys.stderr)
    if unusable:
        print("sweep: in the labels but not usable in the run (counted as missed or wrong): "
              + ", ".join(f"{c} ({st})" for c, st in unusable.items()), file=sys.stderr)
    if not clips:
        print("sweep: no clip of the run has usable results; nothing to choose")
        return 1
    gens = sorted({d.get("generator") or "segments" for _, d in clips})
    if len(gens) > 1:
        print(f"sweep: the run mixes generators ({', '.join(gens)}); sweep one run at a time")
        return 2
    gen = gens[0]
    grid = gen == "grid"
    cells = gen in ("cells", "grid")  # both: no cand mask, strong off
    if grid:
        print("sweep: grid run: events carry no rank, so top_k is not simulated (the run's own top_k is used); "
              "a cell without an answer is kept at every verify")
    run_th = _run_thresholds(d for _, d in clips)
    base = json.loads(Path(a.out).read_text()) if Path(a.out).is_file() else dict(run_th)
    fallback = base.get("fallback", run_th.get("fallback", 0.5))

    names = sorted(labels, key=_clip_key)
    ci_of = {c: i for i, c in enumerate(names)}
    events = [(ci_of[c], e) for c, d in clips for e in d["events"] if isinstance(e, dict) and _num(e.get("timecode"))]
    n = len(events)
    ground = np.array([e["ground_score"] if _num(e.get("ground_score")) else 0.0 for _, e in events], dtype=float)
    verify = [e["verify_score"] if _num(e.get("verify_score")) else None for _, e in events]
    ranks = [_rank(e) for _, e in events]
    has_rank = any(r is not None for r in ranks)
    run_k = run_th.get("top_k")
    if not _pos_int(run_k):
        run_k = max(r for r in ranks if r is not None) + 1 if has_rank else base.get("top_k")
    run_k = run_k if _pos_int(run_k) else 6
    ks = list(range(1, run_k + 1)) if has_rank else [run_k]
    within = {k: np.array([r is None or r < k for r in ranks], dtype=bool) for k in ks}

    viol_idx = [i for i, c in enumerate(names) if labels[c]["has_violation"]]
    usable = {ci_of[c] for c, _ in clips}
    clean_usable = [i for i, c in enumerate(names) if not labels[c]["has_violation"] and i in usable]
    n_clean = len(names) - len(viol_idx)
    truths = [(ci, t["t_label"]) for ci, c in enumerate(names) for t in labels[c]["truth"]]
    truth_of = {ci: [j for j, (tc, _) in enumerate(truths) if tc == ci] for ci in viol_idx}
    near = np.zeros((n, len(truths)), dtype=float)  # event k is within TOL_SEC of truth j (same clip)
    of_clip = np.zeros((n, len(names)), dtype=float)
    for k, (ci, e) in enumerate(events):
        of_clip[k, ci] = 1.0
        for j, (tc, tl) in enumerate(truths):
            if tc == ci and abs(e["timecode"] - tl) <= TOL_SEC:
                near[k, j] = 1.0

    def score(masks):  # (P, n) kept events -> per point metrics and (P, violation clips) found
        kf = masks.astype(float)
        per_clip = kf @ of_clip
        flagged = per_clip > 0
        covered = (kf @ near) > 0
        found = np.zeros((masks.shape[0], len(viol_idx)), dtype=bool)
        for j, ci in enumerate(viol_idx):
            found[:, j] = flagged[:, ci] & (covered[:, truth_of[ci]].all(axis=1) if truth_of[ci] else True)
        vr = flagged[:, viol_idx].sum(axis=1)
        fp = flagged.sum(axis=1) - vr
        acc = (vr + len(clean_usable) - flagged[:, clean_usable].sum(axis=1)) / len(names)
        nflag = flagged.sum(axis=1)
        epf = [float(e) / f if f else None for e, f in zip(per_clip.sum(axis=1), nflag, strict=True)]
        return {"vr": vr, "found": found.sum(axis=1), "fp": fp, "acc": acc, "epf": epf}, found

    def _kept(k, th) -> bool:
        if grid:  # the pipeline keeps a grid cell InternVideo3 did not answer
            return verify[k] is None or verify[k] >= th["verify"]
        return _keep(keep, ground[k], verify[k], verify[k] is not None, th)

    strong_opts = [None] if cells else [None, *GRID]
    cand_opts = [None] if cells else GRID
    cand_mask = {c: np.ones(n, dtype=bool) if c is None else ground >= c - 1e-9 for c in cand_opts}
    points = {}  # (verify, top_k, cand, strong) -> metrics
    for s in strong_opts:
        for v in VERIFY_GRID:
            th = {**base, "verify": v, "strong": s, "fallback": fallback}
            kd = np.array([_kept(k, th) for k in range(n)], dtype=bool)
            combos = [(c, k) for c in cand_opts for k in ks]
            m, _ = score(np.stack([kd & cand_mask[c] & within[k] for c, k in combos]) if n
                         else np.zeros((len(combos), 0), dtype=bool))
            for i, (c, k) in enumerate(combos):
                points[(v, k, c, s)] = {key: m[key][i] for key in m}

    def order(pt):  # most found, fewest FP, loosest verify, largest top_k, loosest cand, strong off or highest
        (v, k, c, s), m = pt
        return (-int(m["found"]), int(m["fp"]), v, -k, -1.0 if c is None else c, (0, 0.0) if s is None else (1, -s))

    best_pt = min(points.items(), key=order)
    (bv, bk, bc, bs), bm = best_pt
    nv = len(viol_idx)

    def metrics(m) -> str:
        vr, found, fp = (f"{int(m[key])}/{of}" for key, of in (("vr", nv), ("found", nv), ("fp", n_clean)))
        return f"{vr:>14} {found:>7} {fp:>8} {m['acc']:>11.3f} {_cell(m['epf'], '.2f'):>14}"

    print(f"sweep: {len(clips)} of {len(names)} labelled clips usable ({nv} violation, {n_clean} clean); "
          f"generator {gen}" + (f", cell {clips[0][1].get('cell_sec')} s" if cells else "")
          + f"; run top_k {run_k}" + ("" if has_rank else " (no ranks recorded: top_k is not simulated)")
          + f"; fallback {fallback:g}")
    print("found: violation clip flagged, every label within +-2 s of a kept event; fp_clips: clean clips flagged; "
          "verdict_acc counts failed or absent clips as wrong; events/flagged: kept events per flagged clip")
    main_cand = None if cells else GRID[0]
    print(f"\noperating points, strong off{'' if cells else f', cand {main_cand:.2f}'}:")
    print(f"{'top_k':>5} {'verify':>6}" + ("" if cells else f" {'cand':>5} {'strong':>6}")
          + f" {'verdict_recall':>14} {'found':>7} {'fp_clips':>8} {'verdict_acc':>11} {'events/flagged':>14}")
    for k in sorted(ks, reverse=True):
        for v in VERIFY_GRID:
            m = points[(v, k, main_cand, None)]
            mark = "  <- chosen" if (v, k, main_cand, None) == best_pt[0] else ""
            if rows is not None:
                rows.append({"verify": v, "top_k": k, "found": int(m["found"]), "fp_clips": int(m["fp"]),
                             "verdict_acc": float(m["acc"]), "events_per_flagged": m["epf"]})
            print(f"{k:>5} {v:>6.2f}" + ("" if cells else f" {main_cand:>5.2f} {'off':>6}")
                  + f" {metrics(m)}{mark}")
    if not cells:
        print("\nlegacy strong options (segments): the best point for each strong value")
        print(f"{'strong':>6} {'cand':>5} {'top_k':>5} {'verify':>6} {'verdict_recall':>14} {'found':>7} "
              f"{'fp_clips':>8} {'verdict_acc':>11} {'events/flagged':>14}")
        for s in strong_opts:
            (v, k, c, _), m = min(((key, m) for key, m in points.items() if key[3] == s), key=order)
            print(f"{_cell(s, '.2f') if s is not None else 'off':>6} {c:>5.2f} {k:>5} {v:>6.2f} {metrics(m)}")

    print(f"\nchosen: verify={bv:.2f} top_k={bk}" + ("" if cells else f" cand={bc:.2f}")
          + f" strong={'off' if bs is None else f'{bs:.2f}'}  verdict_recall={int(bm['vr'])}/{nv} "
          f"found={int(bm['found'])}/{nv} fp_clips={int(bm['fp'])}/{n_clean} verdict_acc={bm['acc']:.3f} "
          f"events_per_flagged_clip={_cell(bm['epf'], '.2f')}")
    if int(bm["found"]) < nv:
        kd = np.array([_kept(k, {**base, "verify": bv, "strong": bs, "fallback": fallback}) for k in range(n)],
                      dtype=bool)
        _, found = score((kd & cand_mask[bc] & within[bk])[None, :])
        missed = [names[ci] + (f" ({unusable[names[ci]]})" if names[ci] in unusable else "")
                  for j, ci in enumerate(viol_idx) if not found[0, j]]
        print(f"recall 1.0 is not reachable on this run: at most {int(bm['found'])}/{nv} violation clips are found "
              f"at any point; not found at the chosen point: {', '.join(missed)}")
    out = dict(base)
    out.update({k: run_th[k] for k in RUN_KEYS if k in run_th})
    if run_th:
        out.update({k: v for k, v in RUN_DEFAULTS.items() if k not in run_th})
    out["generator"] = gen
    if cells and _num(clips[0][1].get("cell_sec")):
        out["cell_sec"] = clips[0][1]["cell_sec"]
    out.update({"verify": bv, "top_k": bk, "strong": bs, "fallback": fallback})
    if not cells:
        out["cand"] = bc
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {a.out}")
    return 0


def main(argv=None, keep=None) -> int:
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
    return cmd_run(a) if a.cmd == "run" else cmd_sweep(a, keep)


if __name__ == "__main__":
    sys.exit(main())
