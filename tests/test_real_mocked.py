"""The real pipeline end to end with the model calls mocked (ground, choose, verify, verify_multi, judge_answer and
the person box)."""
import io
import json
import logging
import os
import threading
import shutil
import tempfile
import time
from pathlib import Path

import av
import pytest
import yaml
from PIL import Image

import pipeline
from pipeline import frames, real, videojev, zoom
from pipeline.frames import probe
from pipeline.videojev import Curves
from rules import load_rules

ROOT = Path(__file__).resolve().parent.parent
# The legacy generator ("segments"), with the thresholds it ran on before the cells generator.
TH = {"generator": "segments", "cand": 0.25, "verify": 0.5, "strong": 0.8, "fallback": 0.5, "top_k": 6,
      "merge_sec": 3, "anchor": "peak", "anchor_offset_sec": 0}
TH_CELLS = {**TH, "generator": "cells", "cell_sec": 4.0, "rank": "logit", "strong": None, "anchor": "center"}
REAL_JUDGE_ANSWER, REAL_GROUND, REAL_CHOOSE = real.judge_answer, real.ground, real.choose
LOW = 0.05


class Rig:
    """Scripted model answers. curves: rule id -> curve given to that rule's first query (others stay LOW).
    multi: window index -> ({rule id: P(YES)}, observation) for verify_multi (rules left out answer None).
    judge_answer: a (rule_id, justification, confidence) triple or a full clients.judge_answer dict."""

    def __init__(self, rules):
        self.rules = rules
        self.curves, self.verdicts, self.choice, self.multi = {}, {}, {}, {}
        self.judge_answer = (None, "", None)
        self.curves_cls = None
        self.choose_error = None
        self.ground_calls, self.choose_calls, self.verify_calls, self.judge_calls = [], [], [], []
        self.multi_calls, self.box_calls = [], []
        self.ground_coded, self.choose_coded = None, []
        self.ground_timeouts, self.choose_timeouts = [], []

    def ground(self, path, queries):  # design.md's two-argument mock signature
        self.ground_calls.append((path, list(queries)))
        self.ground_timeouts.append(real._MODEL_WAIT.get())
        self.ground_coded = coded(path)
        n = len(next(iter(self.curves.values()), [LOW] * 6))
        firsts = {self.rules[rid]["grounding_queries"][0]: c for rid, c in self.curves.items()}
        out = {q: list(firsts.get(q, [LOW] * n)) for q in queries}
        return self.curves_cls(out) if self.curves_cls else out

    def choose(self, window, options):
        self.choose_calls.append((window, list(options)))
        self.choose_timeouts.append(real._MODEL_WAIT.get())
        self.choose_coded.append(coded(window))
        if self.choose_error:
            raise self.choose_error
        by_option = {("normal cash handling" if rid == "none" else self.rules[rid]["choice_option"]): p
                     for rid, p in self.choice.items()}
        return {o: by_option.get(o, 0.0) for o in options}

    def verify(self, window, rules, timeout):
        info = {"window": window, "ids": [r["id"] for r in rules], "timeout": timeout,
                "exists": os.path.exists(window)}
        if info["exists"]:
            info["coded"] = coded(window)
            info["duration"] = probe(window)[0]
        self.verify_calls.append(info)
        return self.verdicts.get(rules[0]["id"], (None, "", ""))

    def verify_multi(self, window, rules, timeout):
        k = int(Path(window).stem.removeprefix("window"))
        self.multi_calls.append({"k": k, "ids": [r["id"] for r in rules], "timeout": timeout,
                                 "coded": coded(window)})
        ps, obs = self.multi.get(k, ({}, ""))
        return [ps.get(r["id"]) for r in rules], obs

    def judge(self, observation, evidence, candidates, timeout):  # stands in for clients.judge_answer
        self.judge_calls.append({"observation": observation, "evidence": evidence,
                                 "candidates": candidates, "timeout": timeout})
        a = self.judge_answer
        if a is None or isinstance(a, dict):
            return a
        rid, just, conf = a
        return None if rid is None else {"rule_id": rid, "justification": just, "confidence": conf,
                                         "clause_id": None, "violation": None, "citation": None}

    def person_box(self, path, start, end):  # no person found unless a test says otherwise
        self.box_calls.append((path, start, end))
        return None


def coded(path):
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        first = next(c.decode(s))
        return s.codec_context.name, s.codec_context.width, s.codec_context.height, int(first.rotation or 0)


@pytest.fixture
def rules():
    return load_rules(ROOT / "rules.yaml")


def setup_rig(monkeypatch, tmp_path, rules, th=None):
    p = tmp_path / "thresholds.json"
    p.write_text(json.dumps(th or TH))
    monkeypatch.setenv("THRESHOLDS_PATH", str(p))
    monkeypatch.setenv("RULES_PATH", str(ROOT / "rules.yaml"))
    monkeypatch.setattr(real, "_CONFIG", None)
    monkeypatch.setattr(videojev, "GROUND_FPS", 2.0)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    r = Rig(rules)
    r.scratch = scratch
    r.retrieve_calls = []
    r.retrieved = []  # clause ids the fake retrieval returns; [] = retrieval down

    def retrieve(text, k, timeout):
        r.retrieve_calls.append((text, k, timeout))
        return list(r.retrieved)
    monkeypatch.setattr(real.clients, "retrieve", retrieve)
    for name, fake in (("ground", r.ground), ("choose", r.choose), ("verify", r.verify),
                       ("verify_multi", r.verify_multi), ("judge_answer", r.judge)):
        monkeypatch.setattr(real, name, fake)
    monkeypatch.setattr(zoom, "person_box", r.person_box)
    return r


@pytest.fixture
def rig(monkeypatch, tmp_path, rules):
    return setup_rig(monkeypatch, tmp_path, rules)


@pytest.fixture
def clip(portrait_clip, tmp_path):
    d = tmp_path / "job"
    d.mkdir()
    return str(shutil.copy(portrait_clip, d / "input.mov"))


def kept(res):
    return [e for e in res.events if e.kept]


def detectable_ids(rules):
    return [rid for rid, r in rules.items() if r.get("detectable")]


def two_candidates(rig):
    rig.curves = {"cash_unattended": [LOW, 0.6, 0.7, LOW, LOW, LOW],
                  "personal_items_at_workplace": [LOW, LOW, LOW, LOW, 0.4, LOW]}
    rig.verdicts = {"cash_unattended": (0.9, "cash_unattended", "Кассир отходит от стола."),
                    "personal_items_at_workplace": (0.1, "none", "Кассир работает с клиентом.")}
    rig.choice = {"cash_unattended": 0.6, "personal_items_at_workplace": 0.3, "none": 0.1}


def test_real_mocked_end_to_end(rig, clip, rules):
    two_candidates(rig)
    rig.judge_answer = ("cash_unattended", "Это может указывать на нарушение.", 0.7)
    stages = []
    res = real.analyze(clip, time.time() + 240, stages.append)

    assert res.verdict == "violation"
    assert stages == ["decoding", "grounding", "verifying", "writing"]
    assert [(e.rule_id, e.reason) for e in res.events] == [
        ("cash_unattended", "verified"), ("personal_items_at_workplace", "rejected")]
    ev, rej = res.events
    assert (ev.t_start, ev.t_end) == pytest.approx((0.75, 2.75))  # cells of pair scores 1..2 at 2 fps
    assert ev.ground_score == pytest.approx(0.7) and ev.verify_score == pytest.approx(0.9)
    assert 1 <= len(ev.evidence) <= 3 and ev.timecode == ev.evidence[0].t
    assert ev.timecode == pytest.approx(2.25)  # the peak pair's midpoint
    assert ev.type_source == "judge" and ev.judge_rule_id == "cash_unattended"
    assert ev.justification == "Это может указывать на нарушение."
    assert ev.observation == "Кассир отходит от стола."
    assert ev.type_probs["cash_unattended"] == pytest.approx(0.6) and ev.type_probs["none"] == pytest.approx(0.1)
    assert set(ev.type_probs) == set(detectable_ids(rules)) | {"none"}

    # rejected candidates stay in the result, without frames or a judge call
    assert rej.verify_score == pytest.approx(0.1) and rej.evidence == [] and rej.timecode == pytest.approx(4.25)
    assert rej.observation == "Кассир работает с клиентом." and rej.type_source == "fallback"

    # every detectable rule's queries went to one grounding call on the input
    (gpath, queries), = rig.ground_calls
    assert gpath == clip
    want = [q for rid in detectable_ids(rules) for q in rules[rid]["grounding_queries"]]
    assert queries == list(dict.fromkeys(want))

    # verify: one upright window per candidate, grounded rule first, every detectable rule listed
    assert sorted(c["ids"][0] for c in rig.verify_calls) == ["cash_unattended", "personal_items_at_workplace"]
    for c in rig.verify_calls:
        assert sorted(c["ids"]) == sorted(detectable_ids(rules)) and c["exists"]
        assert 59 < c["timeout"] <= 60
    w = next(c for c in rig.verify_calls if c["ids"][0] == "cash_unattended")
    assert w["duration"] == pytest.approx(4.75, abs=0.15)  # [0.75 - 2, 2.75 + 2] clamped at 0
    assert len(rig.choose_calls) == 2

    # judge: kept event only, its frames, its candidates with the rule texts
    j, = rig.judge_calls
    assert j["observation"] == "Кассир отходит от стола." and j["evidence"] == ev.evidence
    assert 29 < j["timeout"] <= 30
    ids = [c["id"] for c in j["candidates"]]
    assert ids[:2] == ["cash_unattended", "personal_items_at_workplace"] and ids[-1] == "other"
    for c in j["candidates"]:
        r = rules[c["id"]]
        assert c["name"] == r["name"] and c["visual_criteria"] == r["visual_criteria"]
        assert c["never_claim"] == r["never_claim"]
        assert [q["quote"] for q in c["citations"]] == [q["quote"] for q in r["citations"]]

    assert {"decoding", "grounding", "choosing", "verifying", "writing", "total"} <= set(res.timings)
    assert res.versions["pipeline"] == "real" and res.versions["rules"].startswith("v1 ")
    assert json.loads(res.versions["thresholds"]) == {**TH, "zoom": False, "verify_mode": "single"}  # the defaults
    assert sorted(os.listdir(Path(clip).parent)) == ["input.mov"]  # nothing written next to the input
    assert os.listdir(rig.scratch) == []  # the work folder is removed


def test_judge_answer_not_a_candidate_falls_back(rig, clip, rules):
    two_candidates(rig)
    rig.choice = {"cash_unattended": 0.2, "previous_cash_on_desk": 0.5, "none": 0.3}
    rig.judge_answer = ("x", "Текст.", 0.9)
    ev, = kept(real.analyze(clip, time.time() + 240, lambda s: None))
    assert ev.type_source == "fallback" and ev.judge_rule_id is None
    assert ev.rule_id == "previous_cash_on_desk"
    assert ev.justification == rules["previous_cash_on_desk"]["why"]


def test_judge_absent_uses_best_choice_type_and_rule_why(rig, clip, rules, monkeypatch):
    monkeypatch.setattr(real, "judge_answer", REAL_JUDGE_ANSWER)
    monkeypatch.setenv("JUDGE_URL", "")
    two_candidates(rig)
    rig.choice = {"cash_unattended": 0.2, "previous_cash_on_desk": 0.5, "none": 0.3}
    ev, = kept(real.analyze(clip, time.time() + 240, lambda s: None))
    assert (ev.rule_id, ev.type_source, ev.judge_rule_id) == ("previous_cash_on_desk", "fallback", None)
    assert ev.justification == rules["previous_cash_on_desk"]["why"]
    assert ev.observation == "Кассир отходит от стола."


def test_deadline_skips_vlm(rig, clip, rules):
    two_candidates(rig)
    rig.curves["personal_items_at_workplace"] = [LOW, LOW, LOW, LOW, 0.45, LOW]
    res = real.analyze(clip, time.time() - 1, lambda s: None)
    assert rig.verify_calls == [] and rig.judge_calls == [] and rig.choose_calls == []
    assert all(e.verify_score is None for e in res.events)
    assert [(e.rule_id, e.kept, e.reason) for e in res.events] == [
        ("cash_unattended", True, "fallback_no_vlm"), ("personal_items_at_workplace", False, "rejected")]
    ev = res.events[0]
    assert ev.type_source == "fallback" and ev.justification == rules["cash_unattended"]["why"]
    assert 1 <= len(ev.evidence) <= 3 and ev.timecode == ev.evidence[0].t
    assert res.verdict == "violation"


def test_close_deadline_shrinks_verify_timeout(rig, clip):
    two_candidates(rig)
    real.analyze(clip, time.time() + 50, lambda s: None)
    assert all(25 < c["timeout"] <= 30 for c in rig.verify_calls)


def test_verify_down_keeps_on_fallback_threshold(rig, clip):
    two_candidates(rig)
    rig.curves["personal_items_at_workplace"] = [LOW, LOW, LOW, LOW, 0.6, LOW]
    rig.verdicts["personal_items_at_workplace"] = (None, "", "")
    rig.choice = {}  # no type preference: each kept event keeps its grounded rule, so the two don't merge
    res = real.analyze(clip, time.time() + 240, lambda s: None)
    assert [(e.ground_score, e.verify_score, e.reason) for e in res.events] == [
        (0.7, 0.9, "verified"), (0.6, None, "fallback_no_vlm")]
    assert all(1 <= len(e.evidence) <= 3 and e.timecode == e.evidence[0].t for e in kept(res))


def test_strong_ground_overrides_no_and_same_rule_merges(rig, clip):
    rig.curves = {"cash_unattended": [0.9, LOW, 0.85, LOW, LOW, LOW]}
    rig.verdicts = {"cash_unattended": (0.1, "none", "")}
    res = real.analyze(clip, time.time() + 240, lambda s: None)
    ev, = res.events
    assert (ev.kept, ev.reason, ev.ground_score) == (True, "strong_ground", pytest.approx(0.9))
    assert (ev.t_start, ev.t_end) == pytest.approx((0.0, 2.75))
    assert ev.timecode == ev.evidence[0].t == pytest.approx(0.25)


def test_no_candidates_is_clean(rig, clip):
    res = real.analyze(clip, time.time() + 240, lambda s: None)
    assert res.verdict == "no_violation" and res.events == []
    assert rig.verify_calls == rig.choose_calls == rig.judge_calls == []


def test_choose_failure_keeps_grounded_type(rig, clip, rules):
    two_candidates(rig)
    rig.choose_error = RuntimeError("cuda")
    rig.judge_answer = ("x", "", None)
    ev, = kept(real.analyze(clip, time.time() + 240, lambda s: None))
    assert ev.type_probs == {} and ev.rule_id == "cash_unattended" and ev.type_source == "fallback"
    assert ev.justification == rules["cash_unattended"]["why"]
    ids = [c["id"] for c in rig.judge_calls[0]["candidates"]]
    assert ids == ["cash_unattended", "other"]


def test_model_timeline_is_used(rig, clip):
    two_candidates(rig)
    rig.curves_cls = lambda d: Curves(d, times=[i + 0.5 for i in range(6)],
                                      bounds=[(float(i), i + 1.0) for i in range(6)], step_sec=1.0)
    ev = kept(real.analyze(clip, time.time() + 240, lambda s: None))[0]
    assert (ev.t_start, ev.t_end) == pytest.approx((1.0, 3.0)) and ev.timecode == pytest.approx(2.5)


def test_onset_anchor_offset_and_clamp(monkeypatch, tmp_path, rules, clip):
    rig = setup_rig(monkeypatch, tmp_path, rules, {**TH, "anchor": "onset", "anchor_offset_sec": 100})
    two_candidates(rig)
    res = real.analyze(clip, time.time() + 240, lambda s: None)
    ev, rej = res.events
    assert rej.timecode == pytest.approx(6.0)  # clamped to the duration
    assert ev.timecode == ev.evidence[0].t and 5.9 < ev.timecode < 6.0
    second = tmp_path / "second"
    second.mkdir()
    rig2 = setup_rig(monkeypatch, second, rules, {**TH, "anchor": "onset"})
    two_candidates(rig2)
    ev = kept(real.analyze(clip, time.time() + 240, lambda s: None))[0]
    assert ev.timecode == pytest.approx(0.75)  # the segment start


def test_rotated_input_is_normalised_for_every_model(rig, rotated_clip, tmp_path):
    clip = str(shutil.copy(rotated_clip, tmp_path / "input.mov"))
    two_candidates(rig)
    res = real.analyze(clip, time.time() + 240, lambda s: None)
    (gpath, _), = rig.ground_calls
    assert gpath != clip and rig.ground_coded == ("h264", 848, 464, 0)
    assert len(rig.verify_calls) == 2 and all(c["coded"] == ("h264", 848, 464, 0) for c in rig.verify_calls)
    assert rig.choose_coded == [("h264", 848, 464, 0)] * 2
    ev = kept(res)[0]
    assert Image.open(io.BytesIO(ev.evidence[0].jpeg)).size == (848, 464)


def test_bad_thresholds_fail_at_warmup(monkeypatch, tmp_path):
    p = tmp_path / "th.json"
    p.write_text(json.dumps({k: v for k, v in TH.items() if k != "strong"}))
    monkeypatch.setenv("THRESHOLDS_PATH", str(p))
    monkeypatch.setenv("RULES_PATH", str(ROOT / "rules.yaml"))
    monkeypatch.setattr(real, "_CONFIG", None)
    with pytest.raises(ValueError, match="strong"):
        real.warmup()
    p.write_text(json.dumps({**TH, "anchor": "middle"}))
    with pytest.raises(ValueError, match="anchor"):
        real.warmup()


def test_duplicate_choice_options_fail_at_warmup(monkeypatch, tmp_path):
    data = yaml.safe_load((ROOT / "rules.yaml").read_text(encoding="utf-8"))
    det = [r for r in data["rules"] if r.get("detectable")]
    det[1]["choice_option"] = det[2]["choice_option"]
    p = tmp_path / "rules.yaml"
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    th = tmp_path / "th.json"
    th.write_text(json.dumps(TH))
    monkeypatch.setenv("RULES_PATH", str(p))
    monkeypatch.setenv("THRESHOLDS_PATH", str(th))
    monkeypatch.setattr(real, "_CONFIG", None)
    with pytest.raises(ValueError, match="choice_option"):
        real.warmup()


def test_warmup_loads_config_and_model(monkeypatch, tmp_path):
    th = tmp_path / "th.json"
    th.write_text(json.dumps(TH))
    monkeypatch.setenv("THRESHOLDS_PATH", str(th))
    monkeypatch.setenv("RULES_PATH", str(ROOT / "rules.yaml"))
    monkeypatch.setattr(real, "_CONFIG", None)
    loaded = []

    class FakeVJ:
        def load(self):
            loaded.append(True)

    monkeypatch.setattr(videojev, "get", lambda: FakeVJ())
    real.warmup()
    assert loaded == [True] and real._CONFIG is not None


def test_get_analyze_real_warms_up(monkeypatch):
    calls = []
    monkeypatch.setenv("PIPELINE", "real")
    monkeypatch.setattr(real, "warmup", lambda: calls.append("warmup"))
    assert pipeline.get_analyze() is real.analyze and calls == ["warmup"]


def test_result_fits_the_api_and_debug_json(rig, clip, rules):
    import api
    from pipeline.types import to_debug
    two_candidates(rig)
    rig.judge_answer = ("other", "Действия могут указывать на нарушение.", 0.5)
    res = real.analyze(clip, time.time() + 240, lambda s: None)
    v, = api.to_api("abc123def456", res, rules)["violations"]
    assert v["type_id"] == "other" and v["type"] == rules["other"]["name"]
    assert 1 <= len(v["evidence_frames"]) <= 3
    assert v["evidence_frames"][0]["timestamp_sec"] == v["timecode_sec"] == 2.2
    assert "Кассир отходит от стола." in v["description"] and "Основание:" in v["description"]
    dbg = json.loads(json.dumps(to_debug(res)))
    assert [e["kept"] for e in dbg["events"]] == [True, False]


def test_same_best_type_merges_across_grounded_rules(rig, clip):
    rig.curves = {"cash_unattended": [LOW, 0.6, LOW, LOW, LOW, LOW],
                  "personal_items_at_workplace": [LOW, LOW, LOW, 0.7, LOW, LOW]}
    rig.verdicts = {"cash_unattended": (0.9, "cash_unattended", "obs-a"),
                    "personal_items_at_workplace": (0.9, "deposit_not_counted", "obs-b")}
    rig.choice = {"cash_unattended": 0.7, "none": 0.3}
    res = real.analyze(clip, time.time() + 240, lambda s: None)
    ev, = res.events  # both candidates retyped to cash_unattended, 2 s apart: one event
    assert (ev.rule_id, ev.kept, ev.reason) == ("cash_unattended", True, "verified")
    assert (ev.t_start, ev.t_end) == pytest.approx((0.75, 3.75))
    assert ev.ground_score == pytest.approx(0.7) and ev.observation == "obs-b"  # the best member's check
    assert ev.timecode == ev.evidence[0].t == pytest.approx(3.25)
    j, = rig.judge_calls
    assert "deposit_not_counted" in [c["id"] for c in j["candidates"]]  # the best member's TYPE


def test_rejected_events_keep_their_grounded_rule(rig, clip):
    two_candidates(rig)
    rig.choice = {"previous_cash_on_desk": 0.9, "none": 0.1}
    res = real.analyze(clip, time.time() + 240, lambda s: None)
    assert [(e.rule_id, e.kept) for e in res.events] == [
        ("previous_cash_on_desk", True), ("personal_items_at_workplace", False)]


def test_model_calls_get_deadline_budgets(rig, clip):
    two_candidates(rig)
    real.analyze(clip, time.time() + 240, lambda s: None)
    gt, = rig.ground_timeouts
    assert 240 - real.GROUND_START_RESERVE_SEC - 2 < gt <= 240 - real.GROUND_START_RESERVE_SEC
    assert len(rig.choose_timeouts) == 2
    assert all(240 - real.CHOOSE_MIN_LEFT_SEC - 3 < t <= 240 - real.CHOOSE_MIN_LEFT_SEC for t in rig.choose_timeouts)


def test_late_grounding_is_logged(rig, clip, caplog):
    two_candidates(rig)
    with caplog.at_level(logging.WARNING, logger="pipeline.real"):
        res = real.analyze(clip, time.time() + 10, lambda s: None)
    assert rig.ground_timeouts == [0.0]  # no time to wait: a try, which succeeds when the model is free
    assert any("too late for the check" in r.getMessage() for r in caplog.records)
    assert res.verdict == "violation" and rig.verify_calls == []


class NoModel:
    def predict_video(self, *a, **k):
        raise AssertionError("the model must not be used while another job holds it")


@pytest.fixture
def busy_model(monkeypatch):
    """A VideoJev whose lock another job holds until the test ends."""
    vj = videojev.VideoJev()
    vj._install(NoModel(), "proc", None, None)
    monkeypatch.setattr(videojev, "_INSTANCE", vj)
    held, release = threading.Event(), threading.Event()

    def hold():
        with vj._lock:
            held.set()
            release.wait(30)

    threading.Thread(target=hold, daemon=True).start()
    held.wait()
    yield vj
    release.set()


def test_choose_skipped_when_model_busy(rig, clip, busy_model, monkeypatch, rules):
    monkeypatch.setattr(real, "choose", REAL_CHOOSE)
    monkeypatch.setattr(real, "CHOOSE_MIN_LEFT_SEC", 238.0)  # a choice may wait about 1.5 s for the model
    two_candidates(rig)
    t = time.monotonic()
    res = real.analyze(clip, time.time() + 240, lambda s: None)
    assert 1.0 < time.monotonic() - t < 5  # waited out one budget, then skipped the other choice
    assert all(e.type_probs == {} for e in res.events)
    ev, = kept(res)
    assert ev.rule_id == "cash_unattended" and ev.justification == rules["cash_unattended"]["why"]


def test_ground_busy_fails_the_job_with_a_clear_error(rig, clip, busy_model, monkeypatch):
    monkeypatch.setattr(real, "ground", REAL_GROUND)
    t = time.monotonic()
    with pytest.raises(real.DeadlineExceeded, match="busy"):
        real.analyze(clip, time.time() + real.GROUND_START_RESERVE_SEC + 0.5, lambda s: None)
    assert time.monotonic() - t < 3


def test_normalize_gets_a_deadline_budget(rig, clip, monkeypatch):
    seen = []

    def fake(path, out_path, timeout=None):
        seen.append(timeout)
        return str(path)

    monkeypatch.setattr(frames, "normalize", fake)
    real.analyze(clip, time.time() + 240, lambda s: None)
    real.analyze(clip, time.time() + 50, lambda s: None)
    assert seen[0] == pytest.approx(real.NORMALIZE_MAX_SEC)
    assert seen[1] == pytest.approx(real.NORMALIZE_MIN_SEC)


def test_normalize_failure_falls_back_to_the_original(rig, rotated_clip, tmp_path, monkeypatch, caplog):
    clip = str(shutil.copy(rotated_clip, tmp_path / "input.mov"))

    def boom(path, out_path, timeout=None):
        raise frames.VideoError("ffmpeg failed: TimeoutExpired")

    monkeypatch.setattr(frames, "normalize", boom)
    two_candidates(rig)
    with caplog.at_level(logging.WARNING, logger="pipeline.real"):
        res = real.analyze(clip, time.time() + 240, lambda s: None)
    assert rig.ground_calls[0][0] == clip and res.verdict == "violation"
    assert any("original" in r.getMessage() for r in caplog.records)


def test_unreadable_input_fails(rig, tmp_path):
    bad = tmp_path / "input.mov"
    bad.write_text("not a video")
    with pytest.raises(frames.VideoError):
        real.analyze(str(bad), time.time() + 240, lambda s: None)
    assert rig.ground_calls == []


def test_work_folder_next_to_the_input_without_tmpdir(rig, clip, monkeypatch):
    for k in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.delenv(k, raising=False)
    two_candidates(rig)
    real.analyze(clip, time.time() + 240, lambda s: None)
    assert all(Path(c["window"]).parent.parent == Path(clip).parent / "work" for c in rig.verify_calls)
    assert sorted(os.listdir(Path(clip).parent)) == ["input.mov"]  # the work folder we made is gone
    assert os.listdir(rig.scratch) == []


def test_evidence_falls_back_to_single_frames(rig, clip, monkeypatch):
    two_candidates(rig)
    jpeg_at = frames.jpeg_at

    def no_evidence(*a, **k):
        raise frames.VideoError("seek failed")

    def picky(path, t, quality=90):
        if abs(t - 2.25) < 1e-6:
            raise frames.VideoError("bad frame")
        return jpeg_at(path, t, quality)

    monkeypatch.setattr(frames, "evidence", no_evidence)
    monkeypatch.setattr(frames, "jpeg_at", picky)
    ev, = kept(real.analyze(clip, time.time() + 240, lambda s: None))
    assert len(ev.evidence) == 1 and ev.timecode == ev.evidence[0].t == pytest.approx(0.75)  # t_start


# ---- the cells generator ----------------------------------------------------------------------------
# The 6 s test clip at 2 fps has 6 grounding scores at 0.25, 1.25, ..., 5.25 s: cells [0, 4] (scores 0-3) and
# [4, 6] (scores 4-5).

@pytest.fixture
def cells_rig(monkeypatch, tmp_path, rules):
    return setup_rig(monkeypatch, tmp_path, rules, TH_CELLS)


def test_cells_end_to_end(cells_rig, clip):
    from pipeline.types import to_debug
    two_candidates(cells_rig)
    res = real.analyze(clip, time.time() + 240, lambda s: None)
    assert (res.generator, res.cell_sec, res.generator_fallback) == ("cells", 4.0, False)
    a, b = res.events
    assert (a.t_start, a.t_end, a.rule_id, a.rank, a.kept, a.reason) == (
        0.0, 4.0, "cash_unattended", 0, True, "verified")
    assert a.ground_score == pytest.approx(0.7)  # the raw max in the cell
    assert a.timecode == a.evidence[0].t == pytest.approx(2.0, abs=0.04)  # the cell centre
    assert (b.t_start, b.t_end, b.rule_id, b.rank, b.kept, b.timecode) == (
        4.0, 6.0, "personal_items_at_workplace", 1, False, pytest.approx(5.0))
    assert b.ground_score == pytest.approx(0.4)
    w = {c["ids"][0]: c for c in cells_rig.verify_calls}  # windows [t_start - 2, t_end + 2] within the clip
    assert w["cash_unattended"]["duration"] == pytest.approx(6.0, abs=0.15)
    assert w["personal_items_at_workplace"]["duration"] == pytest.approx(4.0, abs=0.15)
    dbg = json.loads(json.dumps(to_debug(res)))
    assert (dbg["generator"], dbg["cell_sec"], dbg["generator_fallback"]) == ("cells", 4.0, False)
    assert [e["rank"] for e in dbg["events"]] == [0, 1]
    assert json.loads(res.versions["thresholds"]) == {**TH_CELLS, "zoom": False, "verify_mode": "single"}


def test_cells_rule_is_the_rank_argmax_query_and_ground_is_the_cell_raw_max(monkeypatch, tmp_path, rules, clip):
    rig = setup_rig(monkeypatch, tmp_path, rules, {**TH_CELLS, "top_k": 1})
    # cash_unattended is high everywhere after 0 s (contrast about 0); personal items stand out at 2.25 s
    rig.curves = {"cash_unattended": [LOW, 0.95, 0.95, 0.95, 0.95, 0.95],
                  "personal_items_at_workplace": [LOW, LOW, 0.5, LOW, LOW, LOW]}
    ev, = real.analyze(clip, time.time() + 240, lambda s: None).events
    assert (ev.t_start, ev.t_end, ev.rule_id, ev.rank) == (0.0, 4.0, "personal_items_at_workplace", 0)
    assert ev.ground_score == pytest.approx(0.95)
    assert [c["ids"][0] for c in rig.verify_calls] == ["personal_items_at_workplace"]


def test_cells_top_k_keeps_the_best_ranked_cell(monkeypatch, tmp_path, rules, clip):
    rig = setup_rig(monkeypatch, tmp_path, rules, {**TH_CELLS, "top_k": 1, "anchor_offset_sec": -0.5})
    rig.curves = {"cash_unattended": [LOW, LOW, LOW, LOW, LOW, 0.8]}
    ev, = real.analyze(clip, time.time() + 240, lambda s: None).events
    assert (ev.t_start, ev.t_end, ev.rule_id, ev.rank, ev.timecode) == (
        4.0, 6.0, "cash_unattended", 0, pytest.approx(4.5))


def _ground_fails(exc):
    def ground(path, queries):
        raise exc
    return ground


def test_cells_fall_back_to_an_even_grid_when_grounding_raises(cells_rig, clip, monkeypatch, caplog):
    monkeypatch.setattr(real, "ground", _ground_fails(RuntimeError("CUDA error")))
    cells_rig.verdicts = {"other": (0.1, "none", "Ничего не видно.")}
    with caplog.at_level(logging.WARNING, logger="pipeline.real"):
        res = real.analyze(clip, time.time() + 240, lambda s: None)
    assert (res.generator, res.generator_fallback) == ("cells", True)
    assert [(e.t_start, e.t_end, e.timecode, e.rule_id, e.ground_score, e.rank, e.kept) for e in res.events] == [
        (0.0, 4.0, 2.0, "other", 0.0, None, False), (4.0, 6.0, 5.0, "other", 0.0, None, False)]
    assert [c["ids"][0] for c in cells_rig.verify_calls] == ["other", "other"]
    assert any("evenly spaced" in r.getMessage() for r in caplog.records)
    cells_rig.verdicts = {"other": (0.9, "none", "Кассир отходит.")}  # the keep rule rests on InternVideo3 alone
    res = real.analyze(clip, time.time() + 240, lambda s: None)
    assert res.verdict == "violation" and {e.reason for e in res.events if e.kept} == {"verified"}


def test_cells_fallback_without_the_check_keeps_nothing(cells_rig, clip, monkeypatch):
    monkeypatch.setattr(real, "ground", _ground_fails(RuntimeError("CUDA error")))
    res = real.analyze(clip, time.time() - 1, lambda s: None)  # no time left for InternVideo3
    assert res.generator_fallback and cells_rig.verify_calls == []
    assert res.verdict == "no_violation" and [e.reason for e in res.events] == ["rejected", "rejected"]


def test_cells_fall_back_when_the_model_stays_busy(cells_rig, clip, monkeypatch):
    monkeypatch.setattr(real, "ground", _ground_fails(videojev.ModelBusy("busy")))
    res = real.analyze(clip, time.time() + 240, lambda s: None)
    assert res.generator_fallback and [e.rule_id for e in res.events] == ["other", "other"]


@pytest.mark.parametrize("answer", [{}, "nan"])
def test_cells_fall_back_without_usable_scores(cells_rig, clip, monkeypatch, answer):
    def ground(path, queries):
        return {} if answer == {} else {q: [float("nan")] * 6 for q in queries}
    monkeypatch.setattr(real, "ground", ground)
    res = real.analyze(clip, time.time() + 240, lambda s: None)
    assert res.generator_fallback and [(e.t_start, e.rank) for e in res.events] == [(0.0, None), (4.0, None)]


def test_segments_results_record_rank_and_generator(rig, clip):
    two_candidates(rig)
    res = real.analyze(clip, time.time() + 240, lambda s: None)
    assert (res.generator, res.cell_sec, res.generator_fallback) == ("segments", None, False)
    assert [e.rank for e in res.events] == [0, 1]  # by grounding score


def test_thresholds_strong_null_cells_keys_and_rejections(tmp_path):
    p = tmp_path / "th.json"
    p.write_text(json.dumps(TH_CELLS))
    assert real._load_thresholds(p)["strong"] is None
    p.write_text(json.dumps(TH))  # a segments file needs no cell keys
    assert real._load_thresholds(p)["generator"] == "segments"
    bad = [({"strong": "high"}, "strong"), ({"strong": True}, "strong"), ({"strong": float("nan")}, "strong"),
           ({"generator": "peaks"}, "generator"), ({"rank": "z"}, "rank"), ({"cell_sec": 0}, "cell_sec"),
           ({"cell_sec": None}, "cell_sec"), ({"anchor": "middle"}, "anchor"), ({"zoom": "yes"}, "zoom"),
           ({"zoom": 1}, "zoom"), ({"zoom": None}, "zoom"), ({"verify_mode": "both"}, "verify_mode"),
           ({"verify_mode": None}, "verify_mode"), ({"verify_mode": ["multi"]}, "verify_mode")]
    for change, match in bad:
        p.write_text(json.dumps({**TH_CELLS, **change}))
        with pytest.raises(ValueError, match=match):
            real._load_thresholds(p)
    for key in ("generator", "cell_sec", "rank"):
        p.write_text(json.dumps({k: v for k, v in TH_CELLS.items() if k != key}))
        with pytest.raises(ValueError, match=key):
            real._load_thresholds(p)
    p.write_text(json.dumps({**TH, "anchor": "center"}))  # the legacy generator has no centre anchor
    with pytest.raises(ValueError, match="anchor"):
        real._load_thresholds(p)


def test_committed_thresholds_use_grid_without_strong():
    # top_k 8 checks every cell of a clip up to 32 s; zoom and the per-rule check stay off until measured
    assert real._load_thresholds(ROOT / "thresholds.json") == {
        "generator": "grid", "cell_sec": 4.0, "rank": "logit", "cand": 0.25, "verify": 0.05, "strong": None,
        "fallback": 0.5, "top_k": 8, "merge_sec": 3, "anchor": "center", "anchor_offset_sec": 0,
        "zoom": False, "verify_mode": "single"}


# ---- the switches: zoom and the per-rule check -----------------------------------------------------------
# On the 6 s clip the cells rig has two cells, [0, 4] and [4, 6], so the windows are [0, 6] and [2, 6].

def _spy_cut_window(monkeypatch, fail_cropped=False):
    calls = []
    cut = frames.cut_window

    def spy(path, start, end, out_path, crop=None):
        calls.append({"path": path, "start": start, "end": end, "crop": crop})
        if crop and fail_cropped:
            raise frames.VideoError("ffmpeg failed: crop")
        return cut(path, start, end, out_path, crop=crop)

    monkeypatch.setattr(frames, "cut_window", spy)
    return calls


def test_zoom_crop_passed_to_cut_window(monkeypatch, tmp_path, rules, rotated_clip):
    from pipeline.types import to_debug
    rig = setup_rig(monkeypatch, tmp_path, rules, {**TH_CELLS, "zoom": True})
    clip = str(shutil.copy(rotated_clip, tmp_path / "input.mov"))
    two_candidates(rig)
    boxes = []

    def person_box(path, start, end):
        boxes.append((path, start, end))
        return 100, 100, 500, 300  # inside the 848x464 upright frame

    monkeypatch.setattr(zoom, "person_box", person_box)
    cuts = _spy_cut_window(monkeypatch)
    res = real.analyze(clip, time.time() + 240, lambda s: None)
    (src, _), = rig.ground_calls
    assert src != clip  # the box is found on the upright copy the windows are cut from
    assert sorted(boxes) == [(src, 0.0, pytest.approx(6.0, abs=0.05)), (src, 2.0, pytest.approx(6.0, abs=0.05))]
    assert sorted((c["path"], c["start"], c["end"], c["crop"]) for c in cuts) == [
        (src, s, e, (100, 100, 500, 300)) for _, s, e in sorted(boxes)]
    assert [c["coded"][1:3] for c in rig.verify_calls] == [(896, 448)] * 2  # 400x200 scaled to long side 896
    assert [e.crop for e in res.events] == [[100, 100, 500, 300]] * 2
    dbg = json.loads(json.dumps(to_debug(res)))
    assert [e["crop"] for e in dbg["events"]] == [[100, 100, 500, 300]] * 2
    assert [(e.rule_id, e.reason) for e in res.events] == [
        ("cash_unattended", "verified"), ("personal_items_at_workplace", "rejected")]


def test_zoom_failure_falls_back_to_full_frame(monkeypatch, tmp_path, rules, clip, caplog):
    rig = setup_rig(monkeypatch, tmp_path, rules, {**TH_CELLS, "zoom": True})
    two_candidates(rig)

    def person_box(path, start, end):
        raise RuntimeError("CUDA error: out of memory")

    monkeypatch.setattr(zoom, "person_box", person_box)
    cuts = _spy_cut_window(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="pipeline.real"):
        res = real.analyze(clip, time.time() + 240, lambda s: None)
    assert [c["crop"] for c in cuts] == [None, None]
    assert [c["coded"][1:3] for c in rig.verify_calls] == [(464, 848)] * 2  # the full frame
    assert [e.crop for e in res.events] == [None, None]
    assert len([r for r in caplog.records if "person box failed" in r.getMessage()]) == 2
    assert [(e.rule_id, e.reason) for e in res.events] == [
        ("cash_unattended", "verified"), ("personal_items_at_workplace", "rejected")]


def test_zoom_cropped_cut_failure_cuts_uncropped(monkeypatch, tmp_path, rules, clip, caplog):
    rig = setup_rig(monkeypatch, tmp_path, rules, {**TH_CELLS, "zoom": True})
    two_candidates(rig)
    monkeypatch.setattr(zoom, "person_box", lambda path, start, end: (100, 200, 300, 600))
    cuts = _spy_cut_window(monkeypatch, fail_cropped=True)
    with caplog.at_level(logging.WARNING, logger="pipeline.real"):
        res = real.analyze(clip, time.time() + 240, lambda s: None)
    assert sorted(c["crop"] is None for c in cuts) == [False, False, True, True]  # each window tried cropped first
    assert [c["coded"][1:3] for c in rig.verify_calls] == [(464, 848)] * 2
    assert [e.crop for e in res.events] == [None, None]  # the crop that was not used is not recorded
    assert len([r for r in caplog.records if "uncropped" in r.getMessage()]) == 2


def test_zoom_skipped_close_to_the_deadline(monkeypatch, tmp_path, rules, clip, caplog):
    rig = setup_rig(monkeypatch, tmp_path, rules, {**TH_CELLS, "zoom": True})
    two_candidates(rig)
    monkeypatch.setattr(real, "POOL_SIZE", 1)  # the windows are cut one after the other
    boxes = []

    def slow_person_box(path, start, end):
        boxes.append(start)
        # this detection takes so long that the next window starts with too little time left
        monkeypatch.setattr(real, "WINDOWS_MIN_LEFT_SEC", 1e9)
        return 100, 200, 300, 600

    monkeypatch.setattr(zoom, "person_box", slow_person_box)
    with caplog.at_level(logging.WARNING, logger="pipeline.real"):
        res = real.analyze(clip, time.time() + 240, lambda s: None)
    assert boxes == [0.0]
    assert [e.crop for e in res.events] == [[100, 200, 300, 600], None]
    assert sorted(c["coded"][1:3] for c in rig.verify_calls) == [(448, 896), (464, 848)]
    assert any("too late for the person box" in r.getMessage() for r in caplog.records)


def multi_ids(rules):
    return [rid for rid in detectable_ids(rules) if rid != "other"]


def test_multi_mode_takes_max_rule_and_type(monkeypatch, tmp_path, rules, clip):
    import api
    from pipeline.logic import judge_candidates
    from pipeline.types import to_debug
    rig = setup_rig(monkeypatch, tmp_path, rules, {**TH_CELLS, "verify_mode": "multi"})
    two_candidates(rig)  # VideoJev's best type is cash_unattended
    first = {"cash_unattended": 0.3, "previous_cash_on_desk": 0.8, "deposit_not_counted": 0.6}
    rig.multi = {0: (first, "Кассир отходит от стола."),
                 1: ({"cash_unattended": 0.2, "personal_items_at_workplace": 0.1}, "Кассир работает с клиентом.")}
    res = real.analyze(clip, time.time() + 240, lambda s: None)
    assert rig.verify_calls == []  # no single-mode check
    assert sorted(c["k"] for c in rig.multi_calls) == [0, 1]
    for c in rig.multi_calls:  # every detectable rule but other, in rules.yaml order
        assert c["ids"] == multi_ids(rules) and 59 < c["timeout"] <= 60
    a, b = res.events
    assert (a.verify_score, a.kept, a.reason, a.observation) == (0.8, True, "verified", "Кассир отходит от стола.")
    assert (b.verify_score, b.kept, b.reason, b.rule_id) == (0.2, False, "rejected", "personal_items_at_workplace")
    assert a.p_rules == {rid: first.get(rid) for rid in multi_ids(rules)}
    # the kept event's type is the rule answered YES most surely; without the judge it is the result's type
    assert (a.rule_id, a.type_source, a.justification) == (
        "previous_cash_on_desk", "fallback", rules["previous_cash_on_desk"]["why"])
    j, = rig.judge_calls  # the top 3 rules by P(YES), then other
    ids = [c["id"] for c in j["candidates"]]
    assert ids == ["previous_cash_on_desk", "deposit_not_counted", "cash_unattended", "other"]
    assert ids == judge_candidates(first, "previous_cash_on_desk")
    v, = api.to_api("abc123def456", res, rules)["violations"]
    assert v["type_id"] == "previous_cash_on_desk" and v["confidence"] == 0.8
    dbg = json.loads(json.dumps(to_debug(res)))
    p0 = dbg["events"][0]["p_rules"]
    assert p0["previous_cash_on_desk"] == 0.8 and p0["personal_items_at_workplace"] is None
    assert list(p0) == multi_ids(rules)


def test_multi_mode_below_verify_keeps_the_choice_type(monkeypatch, tmp_path, rules, clip):
    rig = setup_rig(monkeypatch, tmp_path, rules, {**TH_CELLS, "verify_mode": "multi", "strong": 0.65})
    two_candidates(rig)
    rig.multi = {0: ({"previous_cash_on_desk": 0.4, "deposit_not_counted": 0.1}, "obs")}
    a, _ = real.analyze(clip, time.time() + 240, lambda s: None).events
    assert (a.verify_score, a.kept, a.reason) == (0.4, True, "strong_ground")
    assert a.rule_id == "cash_unattended"  # no rule reached verify: VideoJev's best type, as in single mode


def test_multi_all_none_uses_fallback(monkeypatch, tmp_path, rules, clip):
    from pipeline.logic import judge_candidates
    rig = setup_rig(monkeypatch, tmp_path, rules, {**TH_CELLS, "verify_mode": "multi"})
    two_candidates(rig)  # the cells' ground scores are 0.7 and 0.4 against fallback 0.5
    rig.multi = {}  # every answer None: the check failed
    res = real.analyze(clip, time.time() + 240, lambda s: None)
    assert len(rig.multi_calls) == 2
    a, b = res.events
    assert (a.verify_score, a.kept, a.reason, a.rule_id) == (None, True, "fallback_no_vlm", "cash_unattended")
    assert (b.verify_score, b.kept, b.reason) == (None, False, "rejected")
    assert a.p_rules == dict.fromkeys(multi_ids(rules))
    j, = rig.judge_calls  # no answer to rank by: VideoJev's types, as in single mode
    assert [c["id"] for c in j["candidates"]] == judge_candidates(a.type_probs, "")


def test_thresholds_without_new_keys_load_with_defaults(rig, clip, tmp_path):
    p = tmp_path / "old.json"
    p.write_text(json.dumps(TH_CELLS))  # written before the switches existed
    th = real._load_thresholds(p)
    assert (th["zoom"], th["verify_mode"]) == (False, "single")
    assert {k: v for k, v in th.items() if k not in ("zoom", "verify_mode")} == TH_CELLS
    p.write_text(json.dumps({**TH_CELLS, "zoom": True, "verify_mode": "multi"}))
    th = real._load_thresholds(p)
    assert (th["zoom"], th["verify_mode"]) == (True, "multi")
    two_candidates(rig)  # rig runs on TH, an old file: the single-mode check and no person box
    res = real.analyze(clip, time.time() + 240, lambda s: None)
    assert len(rig.verify_calls) == 2 and rig.multi_calls == [] and rig.box_calls == []
    assert all(e.crop is None and e.p_rules is None for e in res.events)


CITATION = {"document": "Правила осуществления кассовых операций", "clause": "п. 22",
            "quote": "Кассовому работнику не допускается покидать рабочее место до окончания обслуживания клиента.",
            "url": "https://adilet.zan.kz/rus/docs/V1900019680#z114"}


def test_judge_citation_reaches_api(rig, clip, rules):
    import api
    two_candidates(rig)
    good = {"rule_id": "cash_unattended", "justification": "Действия могут указывать на нарушение.",
            "confidence": 0.7, "clause_id": "V1900019680:22", "violation": 0.9, "citation": CITATION}
    rig.judge_answer = good
    res = real.analyze(clip, time.time() + 240, lambda s: None)
    ev, = kept(res)
    assert (ev.type_source, ev.judge_rule_id, ev.citation) == ("judge", "cash_unattended", CITATION)
    v, = api.to_api("abc123def456", res, rules)["violations"]
    assert v["type_id"] == "cash_unattended" and v["rule"] == CITATION
    assert v["description"].endswith(f"Основание: {CITATION['document']}, {CITATION['clause']}.")
    # a citation with an empty field, or an answer the pipeline does not accept, leaves the catalogue citation
    for answer in ({**good, "citation": {**CITATION, "quote": " "}}, {**good, "citation": None},
                   {**good, "rule_id": "x"}):
        rig.judge_answer = answer
        res = real.analyze(clip, time.time() + 240, lambda s: None)
        ev, = kept(res)
        assert ev.citation is None
        v, = api.to_api("abc123def456", res, rules)["violations"]
        c = rules[v["type_id"]]["citations"][0]
        assert v["rule"] == {k: c.get(k, "") for k in CITATION}


# ---- the grid generator: evenly spaced cells, no VideoJev ------------------------------------------------

TH_GRID = {k: v for k, v in TH_CELLS.items() if k != "rank"} | {"generator": "grid"}


@pytest.fixture
def grid_rig(monkeypatch, tmp_path, rules):
    r = setup_rig(monkeypatch, tmp_path, rules, TH_GRID)

    def banned(*a, **k):
        raise AssertionError("the grid generator must not touch VideoJev")
    for name in ("ground", "choose"):
        monkeypatch.setattr(real, name, banned)
    monkeypatch.setattr(videojev, "get", banned)
    return r


def long_clip(tmp_path, seconds):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg missing")
    import subprocess
    d = tmp_path / f"long{seconds}"
    d.mkdir()
    out = d / "input.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=96x96:rate=5",
                    "-t", str(seconds), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)], check=True)
    return str(out)


def test_grid_never_calls_videojev(grid_rig, tmp_path):
    from pipeline.types import to_debug
    stages = []
    grid_rig.verdicts = {"other": (0.9, "none", "Кассир отходит.")}
    res = real.analyze(long_clip(tmp_path, 20), time.time() + 240, stages.append)
    assert len(res.events) == 5 and len(grid_rig.verify_calls) == 5
    assert "grounding" not in stages
    assert res.versions["videojev"] == "off" and res.versions["ground_fps"] == "off"
    assert (res.generator, res.cell_sec, res.generator_fallback) == ("grid", 4.0, False)
    assert all((e.rule_id, e.ground_score, e.rank) == ("other", 0.0, None) for e in res.events)
    assert [e.timecode for e in res.events] == pytest.approx([2.0, 6.0, 10.0, 14.0, 18.0], abs=0.25)
    assert {c["ids"][0] for c in grid_rig.verify_calls} == {"other"}
    assert json.loads(json.dumps(to_debug(res)))["generator"] == "grid"


def test_grid_long_clip_takes_top_k_even_cells(monkeypatch, tmp_path, rules):
    from pipeline.logic import even_cells
    grid_rig = setup_rig(monkeypatch, tmp_path, rules, {**TH_GRID, "top_k": 8})
    for name in ("ground", "choose"):
        monkeypatch.setattr(real, name, lambda *a: pytest.fail("VideoJev call"))
    path = long_clip(tmp_path, 70)
    want = even_cells(probe(path)[0], 4.0, 8)
    assert len(want) == 8
    grid_rig.verdicts = {"other": (0.9, "none", "obs")}
    res = real.analyze(path, time.time() + 240, lambda s: None)
    assert [(e.t_start, e.t_end) for e in res.events] == [(c.t_start, c.t_end) for c in want]
    assert len(grid_rig.verify_calls) == 8


def test_grid_unanswered_cell_kept_recall_first(grid_rig, tmp_path):
    path = long_clip(tmp_path, 20)
    grid_rig.verdicts = {}  # no answer from InternVideo3: (None, "", "")
    res = real.analyze(path, time.time() + 240, lambda s: None)
    assert res.verdict == "violation"
    assert {(e.kept, e.reason, e.ground_score, e.verify_score) for e in res.events} == {
        (True, "fallback_no_vlm", 0.0, None)}
    grid_rig.verdicts = {"other": (0.01, "none", "Ничего.")}  # answered below verify: rejected
    res = real.analyze(path, time.time() + 240, lambda s: None)
    assert res.verdict == "no_violation" and {e.reason for e in res.events} == {"rejected"}
    n = len(grid_rig.verify_calls)
    res = real.analyze(path, time.time() - 1, lambda s: None)  # no time for the check: still kept
    assert len(grid_rig.verify_calls) == n and res.verdict == "violation"
    assert {e.reason for e in res.events} == {"fallback_no_vlm"}


def test_grid_single_mode_type_from_check(grid_rig, tmp_path, rules):
    path = long_clip(tmp_path, 20)
    for vtype, want, cands in (("cash_unattended", "cash_unattended", ["cash_unattended", "other"]),
                               ("none", "other", ["other"]), ("bogus", "other", ["other"])):
        grid_rig.verdicts = {"other": (0.9, vtype, "Кассир отходит.")}
        grid_rig.judge_calls.clear()
        res = real.analyze(path, time.time() + 240, lambda s: None)
        assert {e.rule_id for e in res.events if e.kept} == {want}
        assert grid_rig.judge_calls
        assert all([c["id"] for c in j["candidates"]] == cands for j in grid_rig.judge_calls)


def test_clause_rules_built_from_catalogue(monkeypatch, tmp_path, rules):
    setup_rig(monkeypatch, tmp_path, rules, TH_GRID)
    from rag.corpus import find_quote, load_clauses
    cfg = real._config()
    clause = find_quote(load_clauses(), rules["cash_unattended"]["citations"][0]["quote"])
    assert clause is not None and "cash_unattended" in cfg.clause_rules[clause.id]
    assert all(r in cfg.detectable for rs in cfg.clause_rules.values() for r in rs)


def test_clause_rules_failure_gives_empty_and_warning(monkeypatch, tmp_path, rules, caplog):
    import rag.corpus
    setup_rig(monkeypatch, tmp_path, rules, TH_GRID)

    def boom():
        raise OSError("no corpus")
    monkeypatch.setattr(rag.corpus, "load_clauses", boom)
    with caplog.at_level(logging.WARNING):
        assert real._config().clause_rules == {}
    assert sum("clause" in r.getMessage() for r in caplog.records) == 1


def test_grid_single_judge_candidates_from_retrieval(grid_rig, tmp_path):
    from pipeline.logic import rules_from_hits
    cfg = real._config()
    cid = next(c for c, rs in cfg.clause_rules.items() if "cash_unattended" in rs)
    path = long_clip(tmp_path, 8)
    grid_rig.retrieved = [cid]
    grid_rig.verdicts = {"other": (0.9, "none", "Кассир отходит от стола.")}
    res = real.analyze(path, time.time() + 240, lambda s: None)
    want = rules_from_hits([cid], cfg.clause_rules, None)
    assert want[0] == "cash_unattended" and want[-1] == "other"
    assert grid_rig.judge_calls
    for j in grid_rig.judge_calls:
        assert [c["id"] for c in j["candidates"]] == want
    assert {tuple(e.judge_candidates) for e in res.events if e.kept} == {tuple(want)}
    assert {c[:2] for c in grid_rig.retrieve_calls} == {("Кассир отходит от стола.", 8)}
    assert all(0 < c[2] <= 5.0 for c in grid_rig.retrieve_calls)
    # the check's own type goes first
    grid_rig.verdicts = {"other": (0.9, "previous_cash_on_desk", "Кассир отходит от стола.")}
    grid_rig.judge_calls.clear()
    res = real.analyze(path, time.time() + 240, lambda s: None)
    assert {e.judge_candidates[0] for e in res.events if e.kept} == {"previous_cash_on_desk"}


def test_grid_single_retrieval_down_uses_old_candidates(grid_rig, tmp_path):
    grid_rig.retrieved = []
    grid_rig.verdicts = {"other": (0.9, "cash_unattended", "Кассир отходит.")}
    res = real.analyze(long_clip(tmp_path, 8), time.time() + 240, lambda s: None)
    assert grid_rig.retrieve_calls
    assert all([c["id"] for c in j["candidates"]] == ["cash_unattended", "other"] for j in grid_rig.judge_calls)
    assert {tuple(e.judge_candidates) for e in res.events if e.kept} == {("cash_unattended", "other")}


def test_multi_mode_with_answers_does_not_retrieve(monkeypatch, tmp_path, rules):
    rig = setup_rig(monkeypatch, tmp_path, rules, {**TH_GRID, "verify_mode": "multi"})
    rig.retrieved = ["x"]
    rig.multi = {0: ({"cash_unattended": 0.3, "previous_cash_on_desk": 0.8}, "obs")}
    real.analyze(long_clip(tmp_path, 4), time.time() + 240, lambda s: None)
    assert rig.retrieve_calls == [] and rig.judge_calls


def test_grid_multi_mode_type_and_judge_candidates(monkeypatch, tmp_path, rules):
    from pipeline.logic import judge_candidates
    rig = setup_rig(monkeypatch, tmp_path, rules, {**TH_GRID, "verify_mode": "multi"})
    monkeypatch.setattr(real, "ground", lambda *a: pytest.fail("ground"))
    monkeypatch.setattr(real, "choose", lambda *a: pytest.fail("choose"))
    first = {"cash_unattended": 0.3, "previous_cash_on_desk": 0.8}
    rig.multi = {0: (first, "obs")}
    res = real.analyze(long_clip(tmp_path, 8), time.time() + 240, lambda s: None)  # two cells
    a, b = res.events
    assert (a.kept, a.rule_id, a.reason) == (True, "previous_cash_on_desk", "verified")
    assert (b.kept, b.reason) == (True, "fallback_no_vlm")  # unanswered, kept
    ids = [[c["id"] for c in j["candidates"]] for j in rig.judge_calls]
    assert judge_candidates(first, "previous_cash_on_desk") in ids


def test_warmup_skips_videojev_for_grid(monkeypatch, tmp_path):
    th = tmp_path / "th.json"
    th.write_text(json.dumps(TH_GRID))
    monkeypatch.setenv("THRESHOLDS_PATH", str(th))
    monkeypatch.setenv("RULES_PATH", str(ROOT / "rules.yaml"))
    monkeypatch.setattr(real, "_CONFIG", None)

    def banned():
        raise AssertionError("warmup must not touch VideoJev for the grid generator")
    monkeypatch.setattr(videojev, "get", banned)
    real.warmup()
    assert real._CONFIG is not None and real._CONFIG.th["generator"] == "grid"


def test_thresholds_grid_needs_cell_sec(tmp_path):
    p = tmp_path / "th.json"
    p.write_text(json.dumps(TH_GRID))  # no rank: fine
    assert real._load_thresholds(p)["generator"] == "grid"
    p.write_text(json.dumps({k: v for k, v in TH_GRID.items() if k != "cell_sec"}))
    with pytest.raises(ValueError, match="cell_sec"):
        real._load_thresholds(p)
    p.write_text(json.dumps({**TH_GRID, "cell_sec": 0}))
    with pytest.raises(ValueError, match="cell_sec"):
        real._load_thresholds(p)


def test_grid_merge_keeps_best_verified_cell(monkeypatch, tmp_path, rules, clip):
    rig = setup_rig(monkeypatch, tmp_path, rules, {**TH_GRID, "zoom": True})
    for name in ("ground", "choose"):
        monkeypatch.setattr(real, name, lambda *a: pytest.fail("VideoJev call"))

    def verify(window, rs, timeout):  # cell 0 unanswered, cell 1 answered
        return (0.9, "none", "Кассир отходит.") if Path(window).stem == "window1" else (None, "", "")
    monkeypatch.setattr(real, "verify", verify)
    monkeypatch.setattr(zoom, "person_box", lambda p, s, e: (100, 100, 300, 500) if s > 0 else None)
    res = real.analyze(clip, time.time() + 240, lambda s: None)  # cell centres 2 and 5 merge (merge_sec 3)
    ev, = res.events
    assert (ev.verify_score, ev.observation, ev.crop, ev.reason) == (
        0.9, "Кассир отходит.", [100, 100, 300, 500], "verified")
    assert (ev.t_start, ev.t_end) == (0.0, 6.0)
    assert ev.timecode == ev.evidence[0].t == pytest.approx(5.0, abs=0.1)  # the second cell's evidence
    j, = rig.judge_calls
    assert j["observation"] == "Кассир отходит." and j["evidence"] == ev.evidence
