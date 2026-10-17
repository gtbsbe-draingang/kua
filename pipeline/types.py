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
    ground_score: float  # raw grounding probability: cells, the max over every query in the cell; segments,
    # the max over the rule's queries in the segment
    type_probs: dict[str, float]  # choice softmax over rule ids + "none"
    verify_score: float | None  # P(YES); None = not run
    kept: bool
    reason: str  # verified | strong_ground | fallback_no_vlm | rejected
    observation: str = ""
    evidence: list[Frame] = field(default_factory=list)
    judge_rule_id: str | None = None
    justification: str = ""
    type_source: str = "fallback"  # judge | fallback
    rank: int | None = None  # 0-based rank of the candidate by grounding within the clip; None = not ranked
    crop: list[int] | None = None  # zoom: the person box [x0, y0, x1, y1] the window was cut to; None = uncropped
    p_rules: dict[str, float | None] | None = None  # multi check: P(YES) per rule (None = no answer); None = not run
    citation: dict | None = None  # the judge's corpus citation {document, clause, quote, url}; None = the catalogue's


@dataclass
class PipelineResult:
    verdict: Literal["violation", "no_violation"]
    events: list[Event]  # all candidates, including rejected ones
    timings: dict[str, float]
    versions: dict[str, str]
    generator: str | None = None  # how candidates were made: cells | segments (None: no generator, e.g. stub)
    cell_sec: float | None = None  # the cell length of the cells generator
    generator_fallback: bool = False  # grounding failed, so the cells were spread evenly instead of ranked


def to_debug(result: PipelineResult) -> dict:
    """JSON-serialisable view of the result; evidence frames keep only their time."""
    out = asdict(result)
    for ev_out, ev in zip(out["events"], result.events):
        ev_out["evidence"] = [{"t": f.t} for f in ev.evidence]
    return out
