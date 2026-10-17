"""Pure decision logic of the review pipeline: no I/O, no models."""
from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any, NamedTuple

from pipeline.types import Event

LOGIT_EPS = 1e-6  # probabilities are clipped to [LOGIT_EPS, 1 - LOGIT_EPS] before the logit
CELL_EPS = 1e-6  # seconds: a grounding time this close below a cell boundary belongs to the later cell


def _r(x: float) -> float:
    return round(x, 6)


# ---- the "cells" candidate generator --------------------------------------------------------------
# The clip is cut into fixed cells of cell_sec seconds; each cell is scored by how far the grounding scores
# inside it stand above each query's own clip baseline (logit p - median logit p, maxed over the queries);
# the top_k cells by score are the candidates. A clip of at most top_k cells is covered completely.

def logit(p) -> float:
    """log(p / (1 - p)) with p clipped to [LOGIT_EPS, 1 - LOGIT_EPS]. NaN, None and other non-numbers count
    as 0, like a missing grounding score."""
    try:
        p = float(p)
    except (TypeError, ValueError):
        p = 0.0
    if math.isnan(p):
        p = 0.0
    p = min(max(p, LOGIT_EPS), 1 - LOGIT_EPS)
    return math.log(p / (1 - p))


def logit_contrast(p: Sequence[float]) -> list[float]:
    """logit p[i] minus the median over i of logit p: how far each score stands above the curve's own
    typical level (a query that is always on scores about 0 everywhere)."""
    z = [logit(v) for v in p]
    if not z:
        return []
    m = statistics.median(z)
    return [v - m for v in z]


def rank_curve(curves: Sequence[Sequence[float]]) -> tuple[list[float], list[int]]:
    """x[i] = max over the curves q of logit_contrast(q)[i], and the index of the first curve reaching it.
    All curves must have the same length."""
    if not curves:
        return [], []
    n = len(curves[0])
    if any(len(c) != n for c in curves):
        raise ValueError("rank_curve needs curves of equal length")
    cs = [logit_contrast(c) for c in curves]
    x, arg = [], []
    for i in range(n):
        best = 0
        for q in range(1, len(cs)):
            if cs[q][i] > cs[best][i]:
                best = q
        x.append(cs[best][i])
        arg.append(best)
    return x, arg


class Cell(NamedTuple):
    index: int  # j: the cell is [j * cell_sec, min((j + 1) * cell_sec, duration)]
    t_start: float
    t_end: float
    members: tuple[int, ...]  # grounding indices whose time falls in the cell
    best: int | None  # the member with the highest rank score (the earliest on a tie); None without members
    score: float  # the rank score at best; -inf without members
    rank: int | None  # 0-based rank by score within the clip (ties: the earlier cell); None = not ranked

    @property
    def centre(self) -> float:
        return (self.t_start + self.t_end) / 2


def _cell_bounds(duration: float, cell_sec: float) -> list[tuple[float, float]]:
    if not duration > 0 or not cell_sec > 0:
        raise ValueError("duration and cell_sec must be positive")
    n = max(1, math.ceil(duration / cell_sec - 1e-9))
    return [(j * cell_sec, min((j + 1) * cell_sec, duration)) for j in range(n)]


def cells(times: Sequence[float], x: Sequence[float], duration: float, cell_sec: float) -> list[Cell]:
    """Every cell of the clip in time order, scored by the max of x over its grounding indices and ranked by
    that score (ties: the earlier cell). Each index belongs to exactly one cell: a time on a boundary goes to
    the later cell, a time outside the clip to the first or last cell."""
    if len(times) != len(x):
        raise ValueError("times and x must have the same length")
    bounds = _cell_bounds(duration, cell_sec)
    members: list[list[int]] = [[] for _ in bounds]
    for i, t in enumerate(times):
        members[min(max(math.floor((float(t) + CELL_EPS) / cell_sec), 0), len(bounds) - 1)].append(i)
    best = [max(m, key=lambda i: (x[i], -i)) if m else None for m in members]
    score = [float(x[b]) if b is not None else -math.inf for b in best]
    order = sorted(range(len(bounds)), key=lambda j: (-score[j], j))
    rank = {j: r for r, j in enumerate(order)}
    return [Cell(j, a, b, tuple(members[j]), best[j], score[j], rank[j]) for j, (a, b) in enumerate(bounds)]


def top_cells(all_cells: Sequence[Cell], top_k: int) -> list[Cell]:
    """The top_k cells by rank, in time order."""
    return [c for c in all_cells if c.rank is not None and c.rank < top_k]


def even_cells(duration: float, cell_sec: float, top_k: int) -> list[Cell]:
    """The fallback when there are no grounding scores: top_k cells spread evenly over the clip (every cell
    when there are at most top_k), unscored and unranked."""
    bounds = _cell_bounds(duration, cell_sec)
    n = len(bounds)
    js = range(n) if n <= top_k else [math.floor((i + 0.5) * n / top_k) for i in range(top_k)]
    return [Cell(j, *bounds[j], (), None, -math.inf, None) for j in js]


def segments(probs: list[float], step_sec: float, cand: float) -> list[tuple[float, float, float]]:
    """Maximal runs of consecutive probs >= cand as (start, end, max prob); sample i is at i*step_sec.
    NaN never reaches cand."""
    out: list[tuple[float, float, float]] = []
    i, n = 0, len(probs)
    while i < n:
        if not probs[i] >= cand:
            i += 1
            continue
        j = i
        while j + 1 < n and probs[j + 1] >= cand:
            j += 1
        out.append((_r(i * step_sec), _r((j + 1) * step_sec), max(probs[i:j + 1])))
        i = j + 1
    return out


def top_candidates(per_rule: dict[str, list[tuple]], top_k: int) -> list[tuple[str, float, float, float]]:
    """Pool segments of all rules, merge overlapping/touching ones, keep the top_k by score."""
    pool = sorted(((s, e, sc, rule) for rule, segs in per_rule.items() for s, e, sc in segs),
                  key=lambda x: (x[0], x[1]))
    groups: list[list] = []  # [start, end, best score, best rule]
    for s, e, sc, rule in pool:
        if groups and s <= groups[-1][1]:
            g = groups[-1]
            g[1] = max(g[1], e)
            if sc > g[2]:
                g[2], g[3] = sc, rule
        else:
            groups.append([s, e, sc, rule])
    groups.sort(key=lambda g: -g[2])
    return [(g[3], g[0], g[1], g[2]) for g in groups[:top_k]]


def keep_decision(ground: float, verify: float | None, vlm_ok: bool, th: dict) -> tuple[bool, str]:
    """Keep a candidate when InternVideo3 says yes (p_yes >= verify); when grounding alone is at least
    `strong` (None turns this rule off: it never fires); or, only when InternVideo3 gave no answer, when
    grounding is at least `fallback`."""
    if vlm_ok and verify is not None and verify >= th["verify"]:
        return True, "verified"
    if th["strong"] is not None and ground >= th["strong"]:
        return True, "strong_ground"
    if not vlm_ok and ground >= th["fallback"]:
        return True, "fallback_no_vlm"
    return False, "rejected"


def merge_events(events: list[Event], merge_sec: float) -> list[Event]:
    """Chain-merge kept events of the same rule whose timecodes are within merge_sec."""
    return [e for e, _ in merge_events_indexed(events, merge_sec)]


def merge_events_indexed(events: list[Event], merge_sec: float,
                         best_key: Callable[[Event], Any] | None = None) -> list[tuple[Event, int]]:
    """merge_events, with the index of the input event each output event copies: a merged group's
    highest-scoring member (by best_key, default the grounding score; the earliest on a tie), or the
    (rejected) event itself."""
    passthrough = [(e, i) for i, e in enumerate(events) if not e.kept]
    by_rule: dict[str, list[int]] = {}
    for i, e in enumerate(events):
        if e.kept:
            by_rule.setdefault(e.rule_id, []).append(i)
    merged: list[tuple[Event, int]] = []
    for idx in by_rule.values():
        idx = sorted(idx, key=lambda i: events[i].timecode)
        group = [idx[0]]
        for i in idx[1:]:
            if events[i].timecode - events[group[-1]].timecode <= merge_sec:
                group.append(i)
            else:
                merged.append(_collapse(events, group, best_key))
                group = [i]
        merged.append(_collapse(events, group, best_key))
    return sorted(passthrough + merged, key=lambda x: x[0].timecode)


def _collapse(events: list[Event], group: list[int], best_key=None) -> tuple[Event, int]:
    key = best_key or (lambda e: e.ground_score)
    best = max(group, key=lambda i: key(events[i]))
    return replace(events[best], t_start=min(events[i].t_start for i in group),
                   t_end=max(events[i].t_end for i in group)), best


def pick_timecode(probs: list[float], step_sec: float, t_start: float, t_end: float,
                  anchor: str, offset: float) -> float:
    if anchor == "onset":
        t = t_start
    elif anchor == "peak":
        idx = [i for i in range(len(probs)) if t_start <= _r(i * step_sec) < t_end]
        if idx:
            t = _r(max(idx, key=lambda i: (probs[i], -i)) * step_sec)
        else:
            t = t_start
    else:
        raise ValueError(f"unknown anchor: {anchor!r}")
    return max(t + offset, 0.0)


def judge_candidates(type_probs: dict[str, float], verify_type: str, k: int = 3) -> list[str]:
    ranked = sorted((r for r in type_probs if r != "none"), key=lambda r: (-type_probs[r], r))
    out = ranked[:k]
    if verify_type and verify_type != "none" and verify_type not in out:
        out.append(verify_type)
    if "other" not in out:
        out.append("other")
    return out


def rules_from_hits(clause_ids: list[str], clause_rules: dict[str, list[str]], first: str | None,
                    k: int = 4) -> list[str]:
    """The judge's candidate rules from regulation retrieval: `first` (the check's own type, unless None, "none" or
    "other"), then the catalogue rules citing each hit clause in hit order, without repeats; up to k rules in all,
    then "other" last, always."""
    out: list[str] = []
    if first and first not in ("none", "other"):
        out.append(first)
    for cid in clause_ids:
        for rid in clause_rules.get(cid, []):
            if len(out) >= k:
                break
            if rid != "other" and rid not in out:
                out.append(rid)
    return out[:k] + ["other"]


def accept_judge(rule_id: str | None, candidates: list[str]) -> bool:
    return rule_id is not None and rule_id in candidates


def sample_times(duration: float, fps: float, max_frames: int | None = None) -> list[float]:
    """Frame times VideoJev samples from a clip: k/fps while < duration; above max_frames,
    max_frames frames spread evenly over the clip instead (VideoJev's sampling_grid)."""
    if duration <= 0 or fps <= 0:
        raise ValueError("duration and fps must be positive")
    times = [k / fps for k in range(math.ceil(duration * fps)) if k / fps < duration]
    if max_frames is not None and len(times) > max_frames:
        eff = max_frames / duration
        times = [k / eff for k in range(max_frames)]
    return times


def pair_timeline(frame_times: list[float], duration: float) -> tuple[list[float], list[tuple[float, float]]]:
    """VideoJev scores one value per pair of sampled frames. Score i stands for the midpoint of frames
    2i and 2i+1 (an odd last frame pairs with itself); its cell runs halfway to the neighbouring pairs,
    and the outer cells reach 0 and the duration (VideoJev's pair_timeline)."""
    t = [float(x) for x in frame_times]
    if not t:
        raise ValueError("no frame times")
    last = len(t) - 1
    centres = [(t[i] + t[min(i + 1, last)]) / 2 for i in range(0, len(t), 2)]
    edges = [0.0] + [(t[i] + t[i + 1]) / 2 for i in range(1, last, 2)] + [float(duration)]
    return centres, [(edges[k], edges[k + 1]) for k in range(len(centres))]
