import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import analyze_candidates as ac  # noqa: E402
import dump_grounding as dg  # noqa: E402


def test_contrast_median_and_logit_remove_the_clip_baseline():
    p = np.array([0.9, 0.9, 0.95, 0.9, 0.6])
    assert ac.contrast(p, "median").tolist() == pytest.approx([0.0, 0.0, 0.05, 0.0, -0.3])
    z = ac.contrast(p, "logit")
    assert z[0] == pytest.approx(0.0) and z[2] == pytest.approx(math.log(19) - math.log(9))
    assert ac.contrast(p, "raw").tolist() == p.tolist()


def test_contrast_z_and_constant_curve():
    p = np.array([0.2, 0.4, 0.6])
    assert ac.contrast(p, "z").tolist() == pytest.approx([-1.2247449, 0.0, 1.2247449])
    assert ac.contrast(np.full(4, 0.97), "z").tolist() == [0.0] * 4
    with pytest.raises(ValueError):
        ac.contrast(p, "nope")


def test_logit_is_finite_at_the_edges():
    assert np.isfinite(ac.logit(np.array([0.0, 1.0]))).all()


def test_runs_and_local_maxima():
    x = np.array([0.1, 0.6, 0.7, 0.6, 0.2, 0.9, 0.9])
    assert ac.runs(x, 0.5) == [(1, 3), (5, 6)]
    assert ac.local_maxima(x, 1, 3) == [2]
    assert ac.local_maxima(x, 5, 6) == [5]  # a plateau gives its first cell


def test_peak_segments_short_run_is_one_segment_around_its_argmax():
    x = np.array([0.0, 0.6, 0.8, 0.7, 0.0])
    assert ac.peak_segments(x, 0.5, 4) == [(1, 3, 2)]


def test_peak_segments_long_run_split_at_local_maxima():
    # one 11-cell run with peaks at 2 (0.9) and 8 (0.95); max 4 cells -> half 2
    x = np.array([0.6, 0.7, 0.9, 0.7, 0.6, 0.6, 0.6, 0.8, 0.95, 0.8, 0.6])
    assert ac.peak_segments(x, 0.5, 4) == [(0, 4, 2), (6, 10, 8)]
    # a lower peak too close to a kept one is dropped
    y = np.array([0.6, 0.9, 0.8, 0.85, 0.6, 0.6, 0.6])
    assert ac.peak_segments(y, 0.5, 4) == [(0, 3, 1)]


def test_nms_time_keeps_best_and_suppresses_near_timecodes():
    c = [{"timecode": 10.0, "score": 0.9}, {"timecode": 11.5, "score": 0.95}, {"timecode": 20.0, "score": 0.5},
         {"timecode": 30.0, "score": 0.4}]
    kept = ac.nms_time(c, 2.0, 2)
    assert [k["timecode"] for k in kept] == [11.5, 20.0]
    assert [k["timecode"] for k in ac.nms_time(c, 2.0, 10)] == [11.5, 20.0, 30.0]


def test_auc_counts_ties_and_minus_inf():
    assert ac.auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert ac.auc([0.5], [0.5]) == 0.5
    assert ac.auc([-math.inf], [0.1]) == 0.0
    assert ac.auc([-math.inf], [-math.inf]) == 0.5
    assert ac.auc([], [0.1]) is None


def _clip(name, viol, labels, curves, rules):
    n = len(next(iter(curves.values())))
    times = np.arange(n) + 0.25
    bounds = np.array([[0.0 if i == 0 else i - 0.25, i + 0.75] for i in range(n)])
    bounds[-1, 1] = float(n)
    q = {k: np.array(v, dtype=float) for k, v in curves.items()}
    rc = {}
    for qt, rids in rules.items():
        for r in rids:
            rc[r] = q[qt] if r not in rc else np.maximum(rc[r], q[qt])
    return ac.Clip(name, viol, labels, float(n), 1.0, times, bounds, list(curves), rules, q, rc, {})


def test_contrast_generator_finds_the_bump_on_a_flat_high_curve():
    flat = [0.98] * 30
    bump = [0.9] * 30
    bump[22] = 0.97
    c = _clip("X-1", 1, [22.0], {"q1": flat, "q2": bump}, {"q1": ["r1"], "q2": ["r2"]})
    raw = ac.gen_peaks(c, 0.8, 4.0, 2.0, 6, "raw")
    assert raw[0]["rule"] == "r1"  # the flat query wins on raw scores
    con = ac.gen_peaks(c, 0.01, 4.0, 2.0, 6, "median")
    assert con[0]["rule"] == "r2" and con[0]["timecode"] == 22.25 and con[0]["raw"] == pytest.approx(0.97)


def test_evaluate_recall_auc_and_zero_fp_recall():
    v = _clip("X-1", 1, [5.0], {"q": [0.1] * 10}, {"q": ["r"]})
    k = _clip("X-2", 0, [], {"q": [0.1] * 10}, {"q": ["r"]})
    cands = {"X-1": [{"timecode": 6.25, "t_start": 5.0, "t_end": 7.0, "score": 0.9}],
             "X-2": [{"timecode": 3.25, "t_start": 3.0, "t_end": 4.0, "score": 0.5}]}
    m = ac.evaluate([v, k], cands)
    assert m["tc_recall"] == 1.0 and m["win_recall"] == 1.0 and m["auc"] == 1.0
    assert m["r0fp_labels"] == 1.0 and m["r0fp_clips"] == 1.0 and m["missed"] == []
    cands["X-2"][0]["score"] = 0.95
    assert ac.evaluate([v, k], cands)["r0fp_labels"] == 0.0


def test_prune_keep_uses_only_clean_clips_and_honours_exclude():
    on, off = [0.9] * 10, [0.1] * 10
    clean = _clip("X-3", 0, [], {"a": on, "b": off}, {"a": ["r"], "b": ["r"]})
    viol = _clip("X-1", 1, [2.0], {"a": off, "b": on}, {"a": ["r"], "b": ["r"]})
    assert ac.prune_keep([clean, viol], 0.5) == {"b"}
    assert ac.prune_keep([clean, viol], 0.5, exclude="X-3") == {"a", "b"}  # no clean clip left: keep all


def test_shared_prefix_stops_at_the_first_difference_or_marker():
    assert dg.shared_prefix([[1, 2, 3, 9, 5, 7], [1, 2, 3, 4, 5, 7], [1, 2, 3, 4, 8, 7]], {7}) == 3
    assert dg.shared_prefix([[1, 2, 3, 7, 5], [1, 2, 3, 7, 6]], {7}) == 3
    assert dg.shared_prefix([[1, 2, 7]], {7}) == 2


def test_cells_cover_a_short_clip_and_rank_a_long_one():
    short = _clip("X-1", 1, [9.0], {"q": [0.5] * 20}, {"q": ["r"]})  # 20 s: 5 cells of 4 s, all kept
    cs = ac.gen_cells(short, 4.0, 6, "logit")
    assert [(c["t_start"], c["t_end"], c["timecode"]) for c in cs] == [
        (0.0, 4.0, 2.0), (4.0, 8.0, 6.0), (8.0, 12.0, 10.0), (12.0, 16.0, 14.0), (16.0, 20.0, 18.0)]
    curve = [0.9] * 40
    curve[33] = 0.99  # 40 s: 10 cells, only the top 2 kept; the bump's cell first by score
    long = _clip("X-2", 1, [33.0], {"q": curve, "p": [0.2] * 40}, {"q": ["r"], "p": ["s"]})
    top = ac.gen_cells(long, 4.0, 2, "logit")
    assert len(top) == 2 and any(c["t_start"] == 32.0 and c["timecode"] == 34.0 and c["rule"] == "r" for c in top)
    assert ac.gen_cells(long, 4.0, 2, "logit", anchor="peak")[-1]["timecode"] == 33.25


def test_keep_table_counts_found_clips_and_fp_clips():
    import verify_candidates as vc

    labels = {"X-1": {"has_violation": 1, "truth": [{"t_label": 10.0}]},
              "X-2": {"has_violation": 0, "truth": []}}
    rows = [{"clip": "X-1", "timecode": 11.0, "p_yes": 0.4}, {"clip": "X-1", "timecode": 20.0, "p_yes": 0.9},
            {"clip": "X-2", "timecode": 3.0, "p_yes": 0.3}, {"clip": "X-2", "timecode": 9.0, "p_yes": None}]
    assert vc.keep_table(rows, labels, [0.2, 0.35, 0.5]) == [(0.2, 1.0, 1, 1), (0.35, 1.0, 1, 0), (0.5, 0.0, 0, 0)]
