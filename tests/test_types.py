import json

from pipeline.types import Event, Frame, PipelineResult, to_debug


def test_to_debug_drops_jpeg():
    ev = Event(
        rule_id="other", t_start=0.0, t_end=2.0, timecode=1.0,
        ground_score=0.9, type_probs={"other": 0.8, "none": 0.2},
        verify_score=None, kept=True, reason="fallback_no_vlm",
        evidence=[Frame(1.0, b"\xff\xd8")],
    )
    res = PipelineResult("violation", [ev], {"total": 1.0}, {"stub": "1"})
    out = to_debug(res)
    text = json.dumps(out)
    assert "jpeg" not in text
    assert out["events"][0]["evidence"] == [{"t": 1.0}]
    assert out["verdict"] == "violation"
    assert out["events"][0]["verify_score"] is None
    assert out["events"][0]["type_source"] == "fallback"


def test_evidence_default_not_shared():
    a = Event("o", 0, 1, 0, 0.1, {}, None, False, "rejected")
    b = Event("o", 0, 1, 0, 0.1, {}, None, False, "rejected")
    a.evidence.append(Frame(0.0, b""))
    assert b.evidence == []
