"""Offline comparison of candidate generators on dumped grounding curves (Task 7C-1). CPU only.

    uv run python tools/analyze_candidates.py --dump DIR [--unpacked DIR] [--out DIR] [--top-k 6]

DIR holds tools/dump_grounding.py's <clip>.json files. Every generator turns one clip's curves into at most
top_k candidates (rule, t_start, t_end, timecode, score). The generators:

  a  current       the pipeline today: logic.segments per rule at `cand`, logic.top_candidates (cross-rule
                   transitive merge), real._candidate for times and the peak timecode
  b  peaks         per rule, no cross-rule merge: runs >= thr, long runs split at local maxima (each peak keeps
                   at most max_len seconds of its run), then time-NMS across rules (timecodes > nms apart)
  c  contrast      each query's curve minus its own clip median (prob or logit units) or z-scored, then the
                   per-rule max of those, then (b)
  d  intervals     VideoJev's own decoded interval per query, its peak as timecode, time-NMS across queries
  e  pruning       queries "always on" in the clean clips (share of cells >= 0.5 over every clean clip) are
                   dropped; the share is recomputed without the held-out clip in leave-one-out
  g  grid          chance baseline: top_k evenly spaced 4 s windows, score 0

Metrics (labels: data/labels.csv through the dump; 22 labels on 21 violation clips, 18 clean clips):
  tc_recall    share of labels with a candidate whose timecode is within +-2 s
  win_recall   share of labels whose verify window [start-2, end+2] of some candidate covers t_label
  cands        mean and max candidates per clip; len = mean candidate length (s)
  auc          ROC-AUC of each label's best-matching (tc +-2 s) candidate score (missing = -inf) against each
               clean clip's top candidate score (no candidate = -inf)
  r0fp         recall at zero FP with the score alone (as if `strong` were the only keep rule): labels (and
               violation clips, all labels) whose best match beats every clean clip's top score

Writes OUT/analysis.json (every setting's metrics), OUT/cands_<name>.json (the candidates of the settings
named in --export) and prints the tables used in the report.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SVC = Path(__file__).resolve().parent.parent
if str(SVC) not in sys.path:
    sys.path.insert(0, str(SVC))

TOL = 2.0  # the plan's timecode tolerance (s)
PAD = 2.0  # real.WINDOW_PAD_SEC
EPS = 1e-6


# ---- data -------------------------------------------------------------------------------------

@dataclass
class Clip:
    name: str
    has_violation: int
    t_labels: list[float]
    duration: float
    step: float
    times: np.ndarray  # (n,)
    bounds: np.ndarray  # (n, 2)
    queries: list[str]
    query_rules: dict[str, list[str]]
    q: dict[str, np.ndarray]  # query -> (n,) probabilities
    rules: dict[str, np.ndarray]  # rule -> (n,) max over its queries (the pipeline's rule curve)
    intervals: dict[str, list[float] | None] = field(default_factory=dict)


def load_dump(directory) -> list[Clip]:
    clips = []
    for p in sorted(Path(directory).glob("*.json"), key=lambda p: clip_key(p.stem)):
        d = json.loads(p.read_text(encoding="utf-8"))
        if "query_curves" not in d:
            continue
        nan0 = lambda xs: np.array([np.nan if x is None else float(x) for x in xs], dtype=float)  # noqa: E731
        q = {k: np.nan_to_num(nan0(v), nan=0.0) for k, v in d["query_curves"].items()}
        rules = {k: np.nan_to_num(nan0(v), nan=0.0) for k, v in d["rule_curves"].items()}
        clips.append(Clip(d["clip"], int(d["has_violation"]), [float(t) for t in d["t_labels"]],
                          float(d["duration"]), float(d["step_sec"]), nan0(d["times"]),
                          np.array(d["bounds"], dtype=float), list(d["queries"]), d["query_rules"], q, rules,
                          d.get("query_intervals") or {}))
    return clips


def clip_key(name: str):
    try:
        return (0, int(name.split("-")[1]))
    except (IndexError, ValueError):
        return (1, name)


# ---- pure helpers (unit-tested) ------------------------------------------------------------------

def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def contrast(p: np.ndarray, kind: str) -> np.ndarray:
    """A query curve relative to its own clip: 'raw' (unchanged), 'median' (p - median p), 'logit'
    (logit p - median logit p) or 'z' ((p - mean) / std, 0 when the curve is constant)."""
    p = np.asarray(p, dtype=float)
    if kind == "raw":
        return p.copy()
    if kind == "median":
        return p - np.median(p)
    if kind == "logit":
        z = logit(p)
        return z - np.median(z)
    if kind == "z":
        sd = float(np.std(p))
        return (p - float(np.mean(p))) / sd if sd > 1e-9 else np.zeros_like(p)
    raise ValueError(f"unknown contrast {kind!r}")


def rule_curves(clip: Clip, kind: str = "raw", keep: set[str] | None = None) -> dict[str, np.ndarray]:
    """Per-rule elementwise max over the rule's (kept) queries, after the per-query contrast."""
    out: dict[str, np.ndarray] = {}
    for qtext, rids in clip.query_rules.items():
        if keep is not None and qtext not in keep:
            continue
        c = contrast(clip.q[qtext], kind)
        for rid in rids:
            out[rid] = c if rid not in out else np.maximum(out[rid], c)
    return out


def runs(x: np.ndarray, thr: float) -> list[tuple[int, int]]:
    """Maximal runs [i0, i1] (inclusive) of x >= thr."""
    out, i, n = [], 0, len(x)
    while i < n:
        if not x[i] >= thr:
            i += 1
            continue
        j = i
        while j + 1 < n and x[j + 1] >= thr:
            j += 1
        out.append((i, j))
        i = j + 1
    return out


def local_maxima(x: np.ndarray, i0: int, i1: int) -> list[int]:
    """Indices in [i0, i1] that are >= both neighbours inside the run (a plateau gives its first cell)."""
    out = []
    for i in range(i0, i1 + 1):
        left = x[i - 1] if i > i0 else -math.inf
        right = x[i + 1] if i < i1 else -math.inf
        if x[i] >= right and x[i] > left:
            out.append(i)
    return out


def peak_segments(x: np.ndarray, thr: float, max_cells: int) -> list[tuple[int, int, int]]:
    """Segments (i0, i1, peak) of the runs of x >= thr. A run of at most max_cells cells is one segment
    around its argmax. A longer run is split at its local maxima: from the highest down, each peak at least
    max_cells // 2 + 1 cells from an already kept one keeps the cells of the run within max_cells // 2."""
    half = max(max_cells // 2, 0)
    out = []
    for a, b in runs(x, thr):
        if b - a + 1 <= max_cells:
            out.append((a, b, a + int(np.argmax(x[a:b + 1]))))
            continue
        kept: list[int] = []
        for p in sorted(local_maxima(x, a, b), key=lambda i: (-x[i], i)):
            if all(abs(p - k) > half for k in kept):
                kept.append(p)
        for p in sorted(kept):
            out.append((max(a, p - half), min(b, p + half), p))
    return out


def nms_time(cands: list[dict], sec: float, top_k: int) -> list[dict]:
    """Greedy by score (ties: earlier timecode): keep a candidate only if its timecode is more than sec
    from every kept one; at most top_k."""
    kept: list[dict] = []
    for c in sorted(cands, key=lambda c: (-c["score"], c["timecode"])):
        if all(abs(c["timecode"] - k["timecode"]) > sec for k in kept):
            kept.append(c)
            if len(kept) == top_k:
                break
    return kept


def auc(pos: list[float], neg: list[float]) -> float | None:
    """P(pos > neg) + P(pos == neg) / 2 over all pairs; None when a side is empty. -inf is allowed."""
    if not pos or not neg:
        return None
    s = 0.0
    for a in pos:
        for b in neg:
            s += 1.0 if a > b else 0.5 if a == b else 0.0
    return s / (len(pos) * len(neg))


def always_on_share(clips: list[Clip], thr: float = 0.5) -> dict[str, float]:
    """Per query: the share of all cells of the given clips with p >= thr."""
    qs = clips[0].queries if clips else []
    return {q: float(np.mean(np.concatenate([c.q[q] >= thr for c in clips]))) for q in qs}


# ---- generators ---------------------------------------------------------------------------------

def _cand(clip: Clip, rid: str, i0: int, i1: int, ip: int, score: float) -> dict:
    return {"rule": rid, "t_start": float(clip.bounds[i0][0]), "t_end": float(clip.bounds[i1][1]),
            "timecode": float(clip.times[ip]), "score": float(score),
            "raw": float(clip.rules[rid][i0:i1 + 1].max()) if rid in clip.rules else float("nan")}


def gen_current(clip: Clip, cand: float, top_k: int) -> list[dict]:
    from pipeline import real
    from pipeline.logic import segments, top_candidates

    tl = real._Timeline(clip.step, [float(t) for t in clip.times], [tuple(map(float, b)) for b in clip.bounds])
    th = {"anchor": "peak", "anchor_offset_sec": 0.0}
    curves = {rid: [float(v) for v in c] for rid, c in clip.rules.items()}
    per_rule = {rid: segments(c, tl.step, cand) for rid, c in curves.items()}
    out = []
    for rid, s, e, sc in top_candidates(per_rule, top_k):
        c = real._candidate(rid, s, e, sc, curves[rid], tl, clip.duration, th)
        out.append({"rule": rid, "t_start": c.t_start, "t_end": c.t_end, "timecode": c.timecode,
                    "score": float(sc), "raw": float(sc)})
    return out


def gen_peaks(clip: Clip, thr: float, max_len: float, nms: float, top_k: int, kind: str = "raw",
              keep: set[str] | None = None) -> list[dict]:
    curves = rule_curves(clip, kind, keep)
    max_cells = max(1, int(round(max_len / clip.step)))
    pool = [_cand(clip, rid, a, b, p, x[p])
            for rid, x in curves.items() for a, b, p in peak_segments(x, thr, max_cells)]
    return nms_time(pool, nms, top_k)


def gen_intervals(clip: Clip, nms: float, top_k: int, keep: set[str] | None = None) -> list[dict]:
    pool = []
    for qtext, iv in clip.intervals.items():
        if not iv or len(iv) != 2 or (keep is not None and qtext not in keep):
            continue
        s, e = float(iv[0]), float(iv[1])
        inside = [i for i in range(len(clip.times)) if s - EPS <= clip.times[i] <= e + EPS]
        if not inside:
            continue
        p = clip.q[qtext]
        ip = max(inside, key=lambda i: (p[i], -i))
        rid = clip.query_rules[qtext][0]
        pool.append({"rule": rid, "t_start": s, "t_end": e, "timecode": float(clip.times[ip]),
                     "score": float(p[ip]), "raw": float(p[ip]), "query": qtext})
    return nms_time(pool, nms, top_k)


def gen_grid(clip: Clip, top_k: int, length: float = 4.0) -> list[dict]:
    k = min(top_k, max(1, int(math.ceil(clip.duration / length))))
    centres = [(i + 0.5) * clip.duration / k for i in range(k)]
    return [{"rule": "other", "t_start": max(0.0, t - length / 2), "t_end": min(clip.duration, t + length / 2),
             "timecode": t, "score": 0.0, "raw": 0.0} for t in centres]


# ---- metrics ------------------------------------------------------------------------------------

def best_match(cands: list[dict], t: float, key: str = "score") -> float:
    m = [c[key] for c in cands if abs(c["timecode"] - t) <= TOL + EPS]
    return max(m) if m else -math.inf


def evaluate(clips: list[Clip], cands: dict[str, list[dict]], key: str = "score") -> dict:
    labels = [(c, t) for c in clips if c.has_violation for t in c.t_labels]
    clean = [c for c in clips if not c.has_violation]
    tc_hit = [any(abs(x["timecode"] - t) <= TOL + EPS for x in cands[c.name]) for c, t in labels]
    win_hit = [any(x["t_start"] - PAD - EPS <= t <= x["t_end"] + PAD + EPS for x in cands[c.name]) for c, t in labels]
    pos = [best_match(cands[c.name], t, key) for c, t in labels]
    neg = [max((x[key] for x in cands[c.name]), default=-math.inf) for c in clean]
    top_clean = max(neg, default=-math.inf)
    r0 = [p > top_clean for p in pos]
    viol = [c for c in clips if c.has_violation]
    clip_r0 = [all(r for (lc, _), r in zip(labels, r0, strict=True) if lc.name == c.name) for c in viol]
    n = [len(cands[c.name]) for c in clips]
    lens = [x["t_end"] - x["t_start"] for c in clips for x in cands[c.name]]
    return {"tc_recall": float(np.mean(tc_hit)), "win_recall": float(np.mean(win_hit)),
            "cands_mean": float(np.mean(n)), "cands_max": int(max(n)),
            "len_mean": float(np.mean(lens)) if lens else 0.0,
            "auc": auc(pos, neg), "r0fp_labels": float(np.mean(r0)), "r0fp_clips": float(np.mean(clip_r0)),
            "missed": [f"{c.name}@{t:g}" for (c, t), h in zip(labels, tc_hit, strict=True) if not h]}


# ---- the sweep ----------------------------------------------------------------------------------

GRIDS = {
    "a": {"cand": [0.25, 0.5, 0.8, 0.9, 0.95]},
    "b": {"thr": [0.25, 0.5, 0.8, 0.9], "max_len": [2.0, 4.0, 6.0, 10.0], "nms": [2.0, 4.0, 6.0]},
    "c_median": {"thr": [0.0, 0.05, 0.1, 0.2, 0.3], "max_len": [2.0, 4.0, 6.0, 10.0], "nms": [2.0, 4.0, 6.0]},
    "c_logit": {"thr": [0.0, 0.5, 1.0, 2.0, 3.0], "max_len": [2.0, 4.0, 6.0, 10.0], "nms": [2.0, 4.0, 6.0]},
    "c_z": {"thr": [0.0, 0.5, 1.0, 1.5, 2.0], "max_len": [2.0, 4.0, 6.0, 10.0], "nms": [2.0, 4.0, 6.0]},
    "d": {"nms": [2.0, 4.0, 6.0]},
    "e_prune": {"prune": [0.3, 0.5, 0.7, 0.9]},
}


def run_setting(clips, family: str, params: dict, top_k: int, keep_by_clip=None) -> dict[str, list[dict]]:
    out = {}
    for c in clips:
        keep = keep_by_clip(c) if keep_by_clip else None
        if family == "a":
            out[c.name] = gen_current(c, params["cand"], top_k)
        elif family == "b" or family.startswith("c_") or family.startswith("e_"):
            kind = params.get("kind", "raw" if family == "b" else family[2:])
            out[c.name] = gen_peaks(c, params["thr"], params["max_len"], params["nms"], top_k, kind, keep)
        elif family == "d":
            out[c.name] = gen_intervals(c, params["nms"], top_k, keep)
        elif family == "g":
            out[c.name] = gen_grid(c, top_k)
        else:
            raise ValueError(family)
    return out


def prune_keep(clips: list[Clip], share: float, exclude: str | None = None) -> set[str]:
    """Queries whose always-on share over the clean clips (minus `exclude`) is below `share`."""
    clean = [c for c in clips if not c.has_violation and c.name != exclude]
    if not clean:
        return set(clips[0].queries) if clips else set()
    on = always_on_share(clean)
    return {q for q, s in on.items() if s < share}


def objective(m: dict) -> tuple:
    """Higher is better: timecode recall, then separability, then fewer candidates, then shorter ones."""
    return (round(m["tc_recall"], 6), round(m["auc"] or 0.0, 6), -m["cands_mean"], -m["len_mean"])


def settings(family: str):
    g = GRIDS[family]
    keys = list(g)
    for vals in itertools.product(*(g[k] for k in keys)):
        yield dict(zip(keys, vals, strict=True))


def fmt_row(name: str, m: dict) -> str:
    a = "-" if m["auc"] is None else f"{m['auc']:.3f}"
    return (f"| {name} | {m['tc_recall']:.3f} | {m['win_recall']:.3f} | {m['cands_mean']:.2f} / {m['cands_max']} | "
            f"{m['len_mean']:.1f} | {a} | {m['r0fp_labels']:.3f} / {m['r0fp_clips']:.3f} | "
            f"{', '.join(m['missed']) or '-'} |")


HEADER = ("| setting | tc_recall | win_recall | cands mean / max | len (s) | auc | r0fp labels / clips | missed |\n"
          "|---|---|---|---|---|---|---|---|")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out")
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--best", type=int, default=5, help="rows per family in the printed tables")
    a = ap.parse_args(argv)
    clips = load_dump(a.dump)
    out = Path(a.out or a.dump)
    out.mkdir(parents=True, exist_ok=True)
    n_lab = sum(len(c.t_labels) for c in clips if c.has_violation)
    print(f"{len(clips)} clips, {sum(c.has_violation for c in clips)} violation clips, {n_lab} labels, "
          f"{sum(1 - c.has_violation for c in clips)} clean; duration median "
          f"{np.median([c.duration for c in clips]):.1f} s (min {min(c.duration for c in clips):.1f}, "
          f"max {max(c.duration for c in clips):.1f})")
    results = []
    for fam in ["g", "a", "b", "c_median", "c_logit", "c_z", "d"]:
        rows = []
        for params in (settings(fam) if fam in GRIDS else [{}]):
            m = evaluate(clips, run_setting(clips, fam, params, a.top_k))
            rows.append((params, m))
            results.append({"family": fam, "params": params, **m})
        rows.sort(key=lambda r: objective(r[1]), reverse=True)
        print(f"\n### {fam}\n{HEADER}")
        for params, m in rows[: a.best]:
            print(fmt_row(json.dumps(params), m))
    (out / "analysis.json").write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
