"""Pure decision logic of the review pipeline: no I/O, no models."""
from __future__ import annotations

from dataclasses import replace

from pipeline.types import Event


def _r(x: float) -> float:
    return round(x, 6)


def segments(probs: list[float], step_sec: float, cand: float) -> list[tuple[float, float, float]]:
    """Maximal runs of consecutive probs >= cand as (start, end, max prob); sample i is at i*step_sec."""
    out: list[tuple[float, float, float]] = []
    i, n = 0, len(probs)
    while i < n:
        if probs[i] < cand:
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
    passthrough = [e for e in events if not e.kept]
    by_rule: dict[str, list[Event]] = {}
    for e in events:
        if e.kept:
            by_rule.setdefault(e.rule_id, []).append(e)
    merged: list[Event] = []
    for evs in by_rule.values():
        evs = sorted(evs, key=lambda e: e.timecode)
        group = [evs[0]]
        for e in evs[1:]:
            if e.timecode - group[-1].timecode <= merge_sec:
                group.append(e)
            else:
                merged.append(_collapse(group))
                group = [e]
        merged.append(_collapse(group))
    return sorted(passthrough + merged, key=lambda e: e.timecode)


def _collapse(group: list[Event]) -> Event:
    best = max(group, key=lambda e: e.ground_score)
    return replace(best, t_start=min(e.t_start for e in group), t_end=max(e.t_end for e in group))


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
