import pytest

from pipeline.logic import (accept_judge, judge_candidates, keep_decision, merge_events,
                            pick_timecode, segments, top_candidates)
from pipeline.types import Event

TH = {"verify": 0.5, "strong": 0.8, "fallback": 0.5}


def test_segments_two_runs():
    assert segments([0.1, 0.3, 0.4, 0.1, 0.6, 0.2], 0.25, 0.25) == [(0.25, 0.75, 0.4), (1.0, 1.25, 0.6)]


def test_segments_none_above_cand():
    assert segments([0.1, 0.2], 0.25, 0.5) == []
    assert segments([], 0.25, 0.5) == []


def test_segments_run_touching_end():
    assert segments([0.1, 0.6, 0.7], 0.5, 0.5) == [(0.5, 1.5, 0.7)]


def test_top_candidates_merges_across_rules_and_cuts():
    per_rule = {
        "a": [(1.0, 3.0, 0.5), (20.0, 21.0, 0.4)],
        "b": [(2.5, 4.0, 0.9)],
        "c": [(10.0, 11.0, 0.6)],
    }
    assert top_candidates(per_rule, 2) == [("b", 1.0, 4.0, 0.9), ("c", 10.0, 11.0, 0.6)]
    assert len(top_candidates(per_rule, 10)) == 3


def test_top_candidates_touching_and_transitive():
    per_rule = {"a": [(0.0, 1.0, 0.5), (2.0, 3.0, 0.3)], "b": [(1.0, 2.0, 0.4)]}
    assert top_candidates(per_rule, 5) == [("a", 0.0, 3.0, 0.5)]


def test_keep_strong_overrides_vlm_no():
    assert keep_decision(0.9, 0.1, True, TH) == (True, "strong_ground")


def test_keep_verified():
    assert keep_decision(0.3, 0.7, True, TH) == (True, "verified")


def test_reject():
    assert keep_decision(0.3, 0.2, True, TH) == (False, "rejected")


def test_fallback_when_vlm_down():
    assert keep_decision(0.6, None, False, TH) == (True, "fallback_no_vlm")


def test_no_fallback_when_vlm_ok_but_unverified():
    assert keep_decision(0.6, None, True, TH) == (False, "rejected")


def _ev(rule, tc, ground=0.5, kept=True, t0=None, t1=None):
    return Event(rule_id=rule, t_start=tc - 1 if t0 is None else t0, t_end=tc + 1 if t1 is None else t1,
                 timecode=tc, ground_score=ground, type_probs={}, verify_score=None, kept=kept,
                 reason="verified" if kept else "rejected")


def test_merge_same_type_within_3s():
    evs = [_ev("cash_unattended", 10.0), _ev("cash_unattended", 12.5), _ev("other_rule", 11.0)]
    out = merge_events(evs, 3.0)
    assert [(e.rule_id, e.timecode) for e in out] == [("cash_unattended", 10.0), ("other_rule", 11.0)]


def test_merge_keeps_best_and_widens_without_mutating():
    a, b = _ev("r", 10.0, 0.5), _ev("r", 12.0, 0.9)
    out = merge_events([a, b], 3.0)
    assert len(out) == 1 and out[0].ground_score == 0.9 and out[0].timecode == 12.0
    assert (out[0].t_start, out[0].t_end) == (9.0, 13.0)
    assert (b.t_start, b.t_end) == (11.0, 13.0) and out[0] is not b


def test_merge_chains_and_respects_gap():
    out = merge_events([_ev("r", 0.0), _ev("r", 3.0), _ev("r", 6.0), _ev("r", 20.0)], 3.0)
    assert len(out) == 2 and out[1].timecode == 20.0
    assert (out[0].t_start, out[0].t_end) == (-1.0, 7.0)


def test_merge_leaves_rejected_alone():
    evs = [_ev("r", 10.0), _ev("r", 11.0, kept=False), _ev("r", 11.5, kept=False)]
    out = merge_events(evs, 3.0)
    assert len(out) == 3 and [e.timecode for e in out] == [10.0, 11.0, 11.5]


def test_pick_timecode_peak_onset_offset():
    probs = [0.1, 0.2, 0.9, 0.9, 0.3]
    assert pick_timecode(probs, 0.5, 0.5, 2.0, "peak", 0.0) == 1.0  # first max, index 2
    assert pick_timecode(probs, 0.5, 0.5, 2.0, "onset", 0.0) == 0.5
    assert pick_timecode(probs, 0.5, 0.5, 2.0, "peak", 0.25) == 1.25
    assert pick_timecode(probs, 0.5, 0.0, 0.5, "onset", -1.0) == 0.0  # clamped


def test_pick_timecode_peak_window_exclusive_end_and_empty():
    probs = [0.1, 0.2, 0.9]
    assert pick_timecode(probs, 0.5, 0.0, 1.0, "peak", 0.0) == 0.5  # index 2 (t=1.0) excluded
    assert pick_timecode(probs, 0.5, 5.0, 6.0, "peak", 0.0) == 5.0


def test_pick_timecode_unknown_anchor():
    with pytest.raises(ValueError):
        pick_timecode([0.1], 0.5, 0.0, 1.0, "middle", 0.0)


def test_judge_rejected_if_not_candidate():
    assert accept_judge("x", ["a", "other"]) is False
    assert accept_judge(None, ["a", "other"]) is False
    assert accept_judge("a", ["a", "other"]) is True


def test_judge_candidates_include_verify_type():
    probs = {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.05, "none": 0.9}
    out = judge_candidates(probs, "d")
    assert out == ["a", "b", "c", "d", "other"]
    assert len(out) == len(set(out))


def test_judge_candidates_dedupe_and_empty_verify():
    probs = {"a": 0.4, "b": 0.3, "none": 0.9}
    assert judge_candidates(probs, "a") == ["a", "b", "other"]
    assert judge_candidates(probs, "") == ["a", "b", "other"]
    assert judge_candidates(probs, "none") == ["a", "b", "other"]
    assert judge_candidates({"b": 0.5, "a": 0.5}, "") == ["a", "b", "other"]


def test_segments_nan_is_never_a_candidate():
    nan = float("nan")
    assert segments([nan, 0.5, nan, nan], 1.0, 0.25) == [(1.0, 2.0, 0.5)]
    assert segments([nan, nan], 0.5, 0.25) == []


def test_sample_times_fixed_fps_and_cap():
    from pipeline.logic import sample_times
    assert sample_times(6.0, 2.0) == [k / 2 for k in range(12)]
    assert sample_times(5.9, 2.0) == [k / 2 for k in range(12)]  # 5.5 < 5.9
    assert sample_times(5.5, 2.0) == [k / 2 for k in range(11)]  # 5.5 is not < 5.5
    capped = sample_times(10.0, 4.0, max_frames=8)  # 40 > 8: 8 uniform frames at 0.8 fps
    assert len(capped) == 8 and capped[1] == 1.25 and capped[-1] == 8.75


def test_pair_timeline_even_and_odd():
    from pipeline.logic import pair_timeline
    times, bounds = pair_timeline([0.0, 0.5, 1.0, 1.5], 2.0)
    assert times == [0.25, 1.25] and bounds == [(0.0, 0.75), (0.75, 2.0)]
    times, bounds = pair_timeline([0.0, 0.5, 1.0], 1.5)  # odd: the last pair is one real frame
    assert times == [0.25, 1.0] and bounds == [(0.0, 0.75), (0.75, 1.5)]
    with pytest.raises(ValueError):
        pair_timeline([], 1.0)
