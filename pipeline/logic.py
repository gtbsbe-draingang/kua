"""Pure decision logic of the review pipeline: no I/O, no models."""
from __future__ import annotations

import math
from dataclasses import replace

from pipeline.types import Event


def _r(x: float) -> float:
    return round(x, 6)


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
    if vlm_ok and verify is not None and verify >= th["verify"]:
        return True, "verified"
    if ground >= th["strong"]:
        return True, "strong_ground"
    if not vlm_ok and ground >= th["fallback"]:
        return True, "fallback_no_vlm"
    return False, "rejected"


def merge_events(events: list[Event], merge_sec: float) -> list[Event]:
    """Chain-merge kept events of the same rule whose timecodes are within merge_sec."""
    return [e for e, _ in merge_events_indexed(events, merge_sec)]


def merge_events_indexed(events: list[Event], merge_sec: float) -> list[tuple[Event, int]]:
    """merge_events, with the index of the input event each output event copies: a merged group's
    highest-scoring member, or the (rejected) event itself."""
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
                merged.append(_collapse(events, group))
                group = [i]
        merged.append(_collapse(events, group))
    return sorted(passthrough + merged, key=lambda x: x[0].timecode)


def _collapse(events: list[Event], group: list[int]) -> tuple[Event, int]:
    best = max(group, key=lambda i: events[i].ground_score)
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
