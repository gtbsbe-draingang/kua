"""Stub pipeline: one fixed 'violation' at 25% of the clip, with real frames. Same signature as the real one."""
from __future__ import annotations

import time
from collections.abc import Callable

from pipeline.frames import jpeg_at, probe
from pipeline.types import Event, PipelineResult

STUB_NOTE = "[заглушка]"


def analyze(video_path, deadline: float, on_stage: Callable[[str], None]) -> PipelineResult:
    timings: dict[str, float] = {}
    t0 = time.time()
    on_stage("decoding")
    duration, _, _, _ = probe(video_path)
    timings["decoding"] = time.time() - t0

    t0 = time.time()
    on_stage("grounding")
    timecode = round(0.25 * duration, 1)
    t_start, t_end = max(0.0, timecode - 1.0), min(duration, timecode + 1.0)
    timings["grounding"] = time.time() - t0

    t0 = time.time()
    on_stage("verifying")
    evidence = []
    for t in (timecode, t_start, t_end):
        fr = jpeg_at(video_path, t)
        if all(abs(fr.t - e.t) >= 0.5 for e in evidence):
            evidence.append(fr)
    timings["verifying"] = time.time() - t0

    t0 = time.time()
    on_stage("writing")
    ev = Event(
        rule_id="cash_unattended",
        t_start=t_start,
        t_end=t_end,
        timecode=evidence[0].t,
        ground_score=0.9,
        type_probs={"cash_unattended": 1.0},
        verify_score=0.9,
        kept=True,
        reason="verified",
        observation=f"{STUB_NOTE} Наличные на столе без присмотра кассира.",
        evidence=evidence,
        justification=f"{STUB_NOTE} Тестовый результат без анализа видео.",
        type_source="fallback",
    )
    timings["writing"] = time.time() - t0
    return PipelineResult("violation", [ev], timings, {"pipeline": "stub"})
