import math

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


def test_merge_events_indexed_links_each_output_to_its_source():
    from pipeline.logic import merge_events_indexed
    evs = [_ev("r", 10.0, 0.5), _ev("x", 11.0, kept=False), _ev("r", 12.0, 0.9), _ev("r", 20.0, 0.4)]
    out = merge_events_indexed(evs, 3.0)
    assert [(e.rule_id, e.timecode, i) for e, i in out] == [("x", 11.0, 1), ("r", 12.0, 2), ("r", 20.0, 3)]
    assert (out[1][0].t_start, out[1][0].t_end) == (9.0, 13.0)
    assert [e for e, _ in out] == merge_events(evs, 3.0)


# ---- the "cells" candidate generator ---------------------------------------------------------------

def test_keep_strong_none_never_fires():
    th = {**TH, "strong": None}
    assert keep_decision(1.0, 0.1, True, th) == (False, "rejected")
    assert keep_decision(1.0, None, False, th) == (True, "fallback_no_vlm")  # the fallback rule still applies
    assert keep_decision(0.0, None, False, th) == (False, "rejected")
    assert keep_decision(0.0, 0.5, True, th) == (True, "verified")


def test_logit_clips_0_1_and_treats_nan_as_0():
    from pipeline.logic import LOGIT_EPS, logit
    lo = math.log(LOGIT_EPS / (1 - LOGIT_EPS))
    assert logit(0.0) == pytest.approx(lo) and logit(-3.0) == pytest.approx(lo)
    assert logit(1.0) == pytest.approx(-lo) and logit(float("inf")) == pytest.approx(-lo)
    assert logit(float("nan")) == logit(0.0) and logit(None) == logit(0.0) and logit("x") == logit(0.0)
    assert logit(0.5) == 0.0 and logit(0.9) == pytest.approx(math.log(9))


def test_logit_contrast_subtracts_the_curve_median():
    from pipeline.logic import logit, logit_contrast
    z = logit_contrast([0.9, 0.9, 0.95, 0.9, 0.6])
    assert z == pytest.approx([0.0, 0.0, logit(0.95) - logit(0.9), 0.0, logit(0.6) - logit(0.9)])
    assert logit_contrast([0.2, 0.4]) == pytest.approx([(logit(0.2) - logit(0.4)) / 2, (logit(0.4) - logit(0.2)) / 2])
    assert logit_contrast([]) == []


def test_rank_curve_is_the_max_contrast_over_queries_and_its_first_query():
    from pipeline.logic import logit, rank_curve
    flat_high = [0.98] * 5  # always on: contrast 0 everywhere
    bump = [0.5, 0.5, 0.9, 0.5, 0.5]
    x, arg = rank_curve([flat_high, bump])
    assert x == pytest.approx([0.0, 0.0, logit(0.9), 0.0, 0.0])
    assert arg == [0, 0, 1, 0, 0]  # ties go to the first query
    assert rank_curve([]) == ([], [])
    with pytest.raises(ValueError):
        rank_curve([[0.5, 0.5], [0.5]])


def _times(duration, step=1.0):
    return [i * step + 0.25 for i in range(int(duration / step))]


def test_cells_cover_a_short_clip_in_time_order():
    from pipeline.logic import cells, top_cells
    x = [0.0] * 20
    x[13] = 2.0
    all_cells = cells(_times(20), x, 20.0, 4.0)
    assert [(c.index, c.t_start, c.t_end, c.centre) for c in all_cells] == [
        (0, 0.0, 4.0, 2.0), (1, 4.0, 8.0, 6.0), (2, 8.0, 12.0, 10.0), (3, 12.0, 16.0, 14.0), (4, 16.0, 20.0, 18.0)]
    assert [c.rank for c in all_cells] == [1, 2, 3, 0, 4]  # the bump's cell first, then ties by time
    assert all_cells[3].best == 13 and all_cells[3].score == 2.0
    assert all_cells[0].members == (0, 1, 2, 3)
    assert top_cells(all_cells, 6) == all_cells  # every cell, in time order


def test_cells_rank_a_long_clip_and_keep_the_top_k_in_time_order():
    from pipeline.logic import cells, top_cells
    x = [0.0] * 40
    x[33], x[9], x[20] = 3.0, 2.0, 1.0  # cells 8, 2, 5
    top = top_cells(cells(_times(40), x, 40.0, 4.0), 2)
    assert [(c.index, c.rank, c.centre, c.best) for c in top] == [(2, 1, 10.0, 9), (8, 0, 34.0, 33)]
    assert [c.index for c in top_cells(cells(_times(40), x, 40.0, 4.0), 3)] == [2, 5, 8]


def test_cells_ties_go_to_the_earlier_cell_and_index():
    from pipeline.logic import cells, top_cells
    x = [1.0] * 40
    top = top_cells(cells(_times(40), x, 40.0, 4.0), 2)
    assert [c.index for c in top] == [0, 1] and top[0].best == 0  # first index of the cell on a tie


def test_cells_last_cell_is_partial():
    from pipeline.logic import cells
    cs = cells(_times(22), [0.0] * 22, 22.0, 4.0)
    assert len(cs) == 6 and (cs[-1].t_start, cs[-1].t_end, cs[-1].centre) == (20.0, 22.0, 21.0)
    assert cs[-1].members == (20, 21)
    assert len(cells([0.25], [0.0], 4.0, 4.0)) == 1  # an exact multiple adds no empty cell


def test_cells_partition_the_indices_and_rank_empty_cells_last():
    from pipeline.logic import cells, top_cells
    times = [3.0, 4.0, 4.5, 30.0]  # 4.0 sits on a boundary: it belongs to the later cell only
    cs = cells(times, [0.5, 0.1, 0.2, 9.0], 12.0, 4.0)  # 30.0 lies past the clip: the last cell takes it
    assert [c.members for c in cs] == [(0,), (1, 2), (3,)]
    assert [c.best for c in cs] == [0, 2, 3]
    sparse = cells([1.0, 9.0], [0.5, 0.2], 12.0, 4.0)  # cell 1 has no grounding index
    assert sparse[1].best is None and sparse[1].score == -math.inf and sparse[1].rank == 2
    assert [c.index for c in top_cells(sparse, 2)] == [0, 2]
    with pytest.raises(ValueError):
        cells([0.25], [0.0, 1.0], 4.0, 4.0)
    with pytest.raises(ValueError):
        cells([0.25], [0.0], 0.0, 4.0)


def test_even_cells_spread_top_k_cells_over_the_clip():
    from pipeline.logic import even_cells
    assert [c.index for c in even_cells(20.0, 4.0, 6)] == [0, 1, 2, 3, 4]
    long = even_cells(142.5, 4.0, 6)  # 36 cells
    assert [c.index for c in long] == [3, 9, 15, 21, 27, 33]
    assert all(c.rank is None and c.best is None and c.members == () for c in long)
    assert (long[0].t_start, long[0].t_end, long[0].centre) == (12.0, 16.0, 14.0)
    assert [(c.t_start, c.t_end) for c in even_cells(22.0, 4.0, 2)] == [(4.0, 8.0), (16.0, 20.0)]


def test_rules_from_hits_order_dedup_and_other_last():
    from pipeline.logic import rules_from_hits
    cr = {"V1:20": ["cash_unattended", "previous_cash_on_desk"], "V1:42": ["deposit_not_counted"]}
    hits = ["V1:20", "V1:42", "V1:20"]
    assert rules_from_hits(hits, cr, None, 4) == [
        "cash_unattended", "previous_cash_on_desk", "deposit_not_counted", "other"]
    assert rules_from_hits(hits, cr, "deposit_not_counted", 2) == ["deposit_not_counted", "cash_unattended", "other"]
    assert rules_from_hits(["V1:42", "V9:9"], {**cr, "V1:42": ["other", "deposit_not_counted"]}, "other") == [
        "deposit_not_counted", "other"]


def test_rules_from_hits_empty_gives_first_and_other():
    from pipeline.logic import rules_from_hits
    assert rules_from_hits([], {}, None) == ["other"]
    assert rules_from_hits([], {}, "none") == ["other"]
    assert rules_from_hits([], {}, "cash_unattended") == ["cash_unattended", "other"]
    assert rules_from_hits(["V1:20"], {}, "cash_unattended") == ["cash_unattended", "other"]
