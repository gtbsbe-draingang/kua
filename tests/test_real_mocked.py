"""The real pipeline end to end with the four model calls mocked (ground, choose, verify, judge)."""
import io
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import av
import pytest
import yaml
from PIL import Image

import pipeline
from pipeline import real, videojev
from pipeline.frames import probe
from pipeline.videojev import Curves
from rules import load_rules

ROOT = Path(__file__).resolve().parent.parent
TH = {"cand": 0.25, "verify": 0.5, "strong": 0.8, "fallback": 0.5, "top_k": 6, "merge_sec": 3,
      "anchor": "peak", "anchor_offset_sec": 0}
REAL_JUDGE = real.judge
LOW = 0.05


class Rig:
    """Scripted model answers. curves: rule id -> curve given to that rule's first query (others stay LOW)."""

    def __init__(self, rules):
        self.rules = rules
        self.curves, self.verdicts, self.choice = {}, {}, {}
        self.judge_answer = (None, "", None)
        self.curves_cls = None
        self.choose_error = None
        self.ground_calls, self.choose_calls, self.verify_calls, self.judge_calls = [], [], [], []
        self.ground_coded, self.choose_coded = None, []

    def ground(self, path, queries):
        self.ground_calls.append((path, list(queries)))
        self.ground_coded = coded(path)
        n = len(next(iter(self.curves.values()), [LOW] * 6))
        firsts = {self.rules[rid]["grounding_queries"][0]: c for rid, c in self.curves.items()}
        out = {q: list(firsts.get(q, [LOW] * n)) for q in queries}
        return self.curves_cls(out) if self.curves_cls else out

    def choose(self, window, options):
        self.choose_calls.append((window, list(options)))
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

    def judge(self, observation, evidence, candidates, timeout):
        self.judge_calls.append({"observation": observation, "evidence": evidence,
                                 "candidates": candidates, "timeout": timeout})
        return self.judge_answer


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
    for name in ("ground", "choose", "verify", "judge"):
        monkeypatch.setattr(real, name, getattr(r, name))
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
    assert json.loads(res.versions["thresholds"]) == TH
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
    monkeypatch.setattr(real, "judge", REAL_JUDGE)
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
