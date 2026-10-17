"""Shared result types for the analysis pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass
class Frame:
    t: float
    jpeg: bytes


@dataclass
class Event:
    rule_id: str
    t_start: float
    t_end: float
    timecode: float  # == evidence[0].t when evidence is present
    ground_score: float  # max sigmoid over the rule's grounding queries
    type_probs: dict[str, float]  # choice softmax over rule ids + "none"
    verify_score: float | None  # P(YES); None = not run
    kept: bool
    reason: str  # verified | strong_ground | fallback_no_vlm | rejected
    observation: str = ""
    evidence: list[Frame] = field(default_factory=list)
    judge_rule_id: str | None = None
    justification: str = ""
    type_source: str = "fallback"  # judge | fallback


@dataclass
class PipelineResult:
    verdict: Literal["violation", "no_violation"]
    events: list[Event]  # all candidates, including rejected ones
    timings: dict[str, float]
    versions: dict[str, str]


def to_debug(result: PipelineResult) -> dict:
    """JSON-serialisable view of the result; evidence frames keep only their time."""
    out = asdict(result)
    for ev_out, ev in zip(out["events"], result.events):
        ev_out["evidence"] = [{"t": f.t} for f in ev.evidence]
    return out
