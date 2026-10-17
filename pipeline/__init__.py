"""Pipeline entry point. Heavy modules are imported lazily so stub mode never loads torch."""
from __future__ import annotations

import os
from collections.abc import Callable

from pipeline.types import PipelineResult


def get_analyze() -> Callable[[str, float, Callable[[str], None]], PipelineResult]:
    if os.environ.get("PIPELINE", "stub") == "stub":
        from pipeline import stub

        return stub.analyze
    from pipeline import real

    real.warmup()  # load the rules, thresholds and VideoJev at server start, not on the first job
    return real.analyze
