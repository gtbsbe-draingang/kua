"""Real review pipeline (design.md section 1, steps 1-7).

VideoJev finds when something may happen and which type it looks like, InternVideo3 checks and describes
each moment, an optional judge picks the rule and words the justification, and evidence frames are cut.
Recall comes first: a failed or late model call can only fall back to the grounding score; it never drops
a moment that grounding alone would keep.

Candidates come from one of two generators, chosen by `generator` in thresholds.json:
- "cells" (the default since Task 7C-2): the clip is cut into cell_sec cells, ranked by how far the grounding
  scores inside each stand above their query's clip median (logic.cells); the top_k cells are checked, each
  with the cell centre as its timecode. If grounding fails or gives no usable scores, top_k cells spread
  evenly over the clip are checked instead, and InternVideo3 alone decides.
- "segments" (the original design): runs of a rule's grounding score >= cand, merged across rules.

Two switches in thresholds.json change how each window is checked; both are off when a file leaves them out:
- `zoom`: each window is cut cropped to the people in it (zoom.person_box on the upright copy). No person, a
  failed detector or too little time left gives the full frame.
- `verify_mode`: "single" asks one YES/NO question listing every rule (clients.verify); "multi" asks one per
  rule but `other` in the same call (clients.verify_multi): p_yes is the highest answer, and its rule, when it
  reaches `verify`, is the check's type.
The judge (clients.judge_answer) may also give a corpus citation for its rule, which then replaces the rule's
catalogue citation in the result.

`ground`, `choose`, `verify`, `verify_multi` and `judge_answer` are the module-level wrappers the tests mock.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from pipeline import clients, frames, videojev, zoom
from pipeline.logic import (accept_judge, cells, even_cells, judge_candidates, keep_decision, merge_events_indexed,
                            pair_timeline, pick_timecode, rank_curve, sample_times, segments, top_candidates,
                            top_cells)
from pipeline.types import Event, Frame, PipelineResult
from rules import load_rules

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
NONE_OPTION = "normal cash handling"
WINDOW_PAD_SEC = 2.0
POOL_SIZE = 6
VERIFY_MAX_SEC, VERIFY_RESERVE_SEC, VERIFY_MIN_SEC = 60.0, 20.0, 5.0
JUDGE_MAX_SEC, JUDGE_RESERVE_SEC, JUDGE_MIN_SEC = 30.0, 10.0, 3.0
# A type choice may wait for the model until this much time is left, so the check and the frames still fit.
CHOOSE_MIN_LEFT_SEC = 30.0
# Grounding may wait for the model (another job's call) until this much time is left: grounding itself (about
# 15-30 s on the longest training clip) plus windows, frames and the result must still fit before the deadline.
GROUND_START_RESERVE_SEC = 45.0
# Windows are cut only while at least one of their users (the check or a type choice) can still run.
WINDOWS_MIN_LEFT_SEC = min(VERIFY_MIN_SEC + VERIFY_RESERVE_SEC, CHOOSE_MIN_LEFT_SEC)
# The upright copy may take up to NORMALIZE_MAX_SEC, but never past GROUND_START_RESERVE_SEC before the
# deadline; it always gets NORMALIZE_MIN_SEC, and on failure the original file is used.
NORMALIZE_MAX_SEC, NORMALIZE_MIN_SEC = 120.0, 5.0
TH_KEYS = ("generator", "cand", "verify", "strong", "fallback", "top_k", "merge_sec", "anchor",
           "anchor_offset_sec")
CELLS_KEYS = ("cell_sec", "rank")  # required by the cells generator
TH_DEFAULTS = {"zoom": False, "verify_mode": "single"}  # the switches, as a file without them runs
VERIFY_MODES = ("single", "multi")
GENERATORS = ("cells", "segments")
RANKS = ("logit",)
ANCHORS = ("peak", "onset", "center")  # cells always use the centre; segments take peak or onset
FALLBACK_RULE = "other"  # the rule of an unscored cell: the generic catch-all


class DeadlineExceeded(RuntimeError):
    """The job cannot produce a result before its deadline."""


# ---- the model calls (mocked in tests) -----------------------------------------------------------
# They keep design.md's signatures. How long ground/choose may wait for the shared VideoJev model (another
# job may be using it) is set by _run through model_wait(), so a two-argument mock still fits.

_MODEL_WAIT: ContextVar[float | None] = ContextVar("model_wait", default=None)


@contextmanager
def model_wait(seconds: float | None):
    token = _MODEL_WAIT.set(seconds)
    try:
        yield
    finally:
        _MODEL_WAIT.reset(token)


def ground(video_path: str, queries: list[str]) -> dict[str, list[float]]:
    return videojev.get().ground(video_path, queries, timeout=_MODEL_WAIT.get())


def choose(window_path: str, options: list[str]) -> dict[str, float]:
    return videojev.get().choose(window_path, options, timeout=_MODEL_WAIT.get())


def verify(window_path: str, rules: list[dict], timeout: float) -> tuple[float | None, str, str]:
    return clients.verify(window_path, rules, timeout)


def verify_multi(window_path: str, rules: list[dict], timeout: float) -> tuple[list[float | None], str]:
    return clients.verify_multi(window_path, rules, timeout)


def judge_answer(observation: str, evidence: list[Frame], candidates: list[dict], timeout: float) -> dict | None:
    return clients.judge_answer(observation, evidence, candidates, timeout)


# ---- configuration, loaded once ------------------------------------------------------------------

@dataclass(frozen=True)
class _Config:
    rules: dict[str, dict]
    detectable: list[str]  # catalogue order
    queries: list[str]  # every detectable rule's grounding queries, without repeats
    query_rules: list[str]  # the rule of each query: the first detectable rule that lists it
    options: list[str]  # choice options: one per detectable rule, then NONE_OPTION
    option_ids: list[str]  # the rule id of each option; "none" for NONE_OPTION
    th: dict
    rules_version: str


_CONFIG: _Config | None = None
_CONFIG_LOCK = threading.Lock()


def _number(x) -> bool:
    return not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x)


def _load_thresholds(path: Path) -> dict:
    th = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(th, dict):
        raise ValueError(f"{path}: thresholds must be a JSON object")
    th = {**TH_DEFAULTS, **th}
    missing = [k for k in TH_KEYS if k not in th]
    if th.get("generator") == "cells":
        missing += [k for k in CELLS_KEYS if k not in th]
    if missing:
        raise ValueError(f"{path}: missing {', '.join(missing)}")
    if th["generator"] not in GENERATORS:
        raise ValueError(f"{path}: generator must be one of {', '.join(GENERATORS)}, not {th['generator']!r}")
    for k in ("cand", "verify", "fallback", "merge_sec", "anchor_offset_sec"):
        if not _number(th[k]):
            raise ValueError(f"{path}: {k} must be a number")
    if th["strong"] is not None and not _number(th["strong"]):
        raise ValueError(f"{path}: strong must be a number, or null to turn it off")
    if "cell_sec" in th and not (_number(th["cell_sec"]) and th["cell_sec"] > 0):
        raise ValueError(f"{path}: cell_sec must be a positive number")
    if "rank" in th and th["rank"] not in RANKS:
        raise ValueError(f"{path}: rank must be one of {', '.join(RANKS)}, not {th['rank']!r}")
    if isinstance(th["top_k"], bool) or not isinstance(th["top_k"], int) or th["top_k"] < 1:
        raise ValueError(f"{path}: top_k must be a positive integer")
    if th["anchor"] not in ANCHORS:
        raise ValueError(f"{path}: anchor must be one of {', '.join(ANCHORS)}, not {th['anchor']!r}")
    if th["anchor"] == "center" and th["generator"] != "cells":
        raise ValueError(f"{path}: anchor 'center' needs the cells generator; segments take 'peak' or 'onset'")
    if not isinstance(th["zoom"], bool):
        raise ValueError(f"{path}: zoom must be true or false")
    if th["verify_mode"] not in VERIFY_MODES:
        raise ValueError(f"{path}: verify_mode must be one of {', '.join(VERIFY_MODES)}, not {th['verify_mode']!r}")
    return th


def _load_config(rules_path: Path, th_path: Path) -> _Config:
    raw = rules_path.read_bytes()
    rules = load_rules(rules_path, strict=True)
    version = (yaml.safe_load(raw) or {}).get("version", "?")
    detectable = [rid for rid, r in rules.items() if r.get("detectable")]
    if not detectable:
        raise ValueError(f"{rules_path}: no detectable rule")
    for rid in detectable:
        r = rules[rid]
        for key in ("name", "choice_option", "verify_prompt", "why"):
            if not str(r.get(key) or "").strip():
                raise ValueError(f"rule {rid!r} is detectable but has no {key}")
        for q in r["grounding_queries"]:
            if not str(q).strip() or "<|" in str(q):
                raise ValueError(f"rule {rid!r}: grounding query {q!r} is empty or has a control token")
    options = [str(rules[rid]["choice_option"]) for rid in detectable] + [NONE_OPTION]
    dup = sorted({o for o in options if options.count(o) > 1})
    if dup:
        raise ValueError(f"each detectable rule needs its own choice_option, other than {NONE_OPTION!r}: {dup}")
    query_rule: dict[str, str] = {}
    for rid in detectable:
        for q in rules[rid]["grounding_queries"]:
            query_rule.setdefault(str(q), rid)
    queries = list(query_rule)
    th = _load_thresholds(th_path)
    digest = hashlib.sha256(raw).hexdigest()[:12]
    return _Config(rules, detectable, queries, [query_rule[q] for q in queries], options, detectable + ["none"],
                   th, f"v{version} sha256:{digest}")


def _config() -> _Config:
    global _CONFIG
    with _CONFIG_LOCK:
        if _CONFIG is None:
            _CONFIG = _load_config(Path(os.environ.get("RULES_PATH") or ROOT / "rules.yaml"),
                                   Path(os.environ.get("THRESHOLDS_PATH") or ROOT / "thresholds.json"))
        return _CONFIG


def warmup() -> None:
    """Load the rules, the thresholds and VideoJev now (at server start), not on the first job."""
    _config()
    log.info("video encoder for internal copies: %s", frames.video_encoder())
    videojev.get().load()


# ---- grounding scores to seconds ---------------------------------------------------------------

@dataclass(frozen=True)
class _Timeline:
    step: float  # index i is at i*step in the index time logic.segments / pick_timecode work in
    times: list[float]  # seconds each score stands for
    bounds: list[tuple[float, float]]  # the cell of seconds each score covers

    def index(self, t: float) -> int:
        return min(max(int(round(t / self.step)), 0), len(self.times) - 1)


def _timeline(curves, n: int, duration: float) -> _Timeline:
    """The model's own score times when ground() returned them; otherwise VideoJev's sampling rebuilt from
    GROUND_FPS; otherwise n equal cells over the clip."""
    times, bounds, step = (getattr(curves, a, None) for a in ("times", "bounds", "step_sec"))
    if times is not None and bounds is not None and len(times) == len(bounds) == n and step and step > 0:
        return _Timeline(float(step), [float(t) for t in times], [(float(a), float(b)) for a, b in bounds])
    ft = sample_times(duration, videojev.GROUND_FPS, videojev.MAX_FRAMES)
    if math.ceil(len(ft) / 2) == n:
        times, bounds = pair_timeline(ft, duration)
        return _Timeline(2 * (ft[1] - ft[0]) if len(ft) > 1 else duration, times, bounds)
    log.warning("grounding gave %d scores for a %.2f s clip at %.2f fps; assuming even cells",
                n, duration, videojev.GROUND_FPS)
    step = duration / n
    return _Timeline(step, [(i + 0.5) * step for i in range(n)], [(i * step, (i + 1) * step) for i in range(n)])


def _clean(p) -> float:
    try:
        p = float(p)
    except (TypeError, ValueError):
        return 0.0
    return p if math.isfinite(p) else 0.0


def _rule_curves(curves: dict, cfg: _Config) -> tuple[dict[str, list[float]], int]:
    """A rule's curve is the elementwise max over its queries' curves."""
    n = max((len(v) for v in curves.values()), default=0)
    out = {}
    for rid in cfg.detectable:
        qs = [curves.get(q) or [] for q in cfg.rules[rid]["grounding_queries"]]
        out[rid] = [max(_clean(c[i]) if i < len(c) else 0.0 for c in qs) for i in range(n)]
    return out, n


# ---- one job ---------------------------------------------------------------------------------

@dataclass
class _Cand:
    rule_id: str
    score: float
    t_start: float
    t_end: float
    timecode: float
    window: str | None = None
    type_probs: dict[str, float] = field(default_factory=dict)
    p_yes: float | None = None
    vtype: str = ""
    observation: str = ""
    rank: int | None = None
    crop: tuple[int, int, int, int] | None = None
    p_rules: dict[str, float | None] | None = None


def _left(deadline: float) -> float:
    return deadline - time.time()


def _clamp(t: float, duration: float) -> float:
    return min(max(float(t), 0.0), duration)


def _candidate(rid: str, s: float, e: float, score: float, curve: list[float], tl: _Timeline,
               duration: float, th: dict) -> _Cand:
    n = len(tl.times)
    i0 = min(max(int(round(s / tl.step)), 0), n - 1)
    i1 = min(max(int(round(e / tl.step)) - 1, i0), n - 1)  # e is the exclusive end in index time
    t_start, t_end = _clamp(tl.bounds[i0][0], duration), _clamp(tl.bounds[i1][1], duration)
    if th["anchor"] == "onset":
        tc = t_start
    else:
        tc = tl.times[tl.index(pick_timecode(curve, tl.step, s, e, "peak", 0.0))]
    return _Cand(rid, float(score), t_start, t_end, _clamp(tc + float(th["anchor_offset_sec"]), duration))


def _finite(p) -> bool:
    try:
        return math.isfinite(float(p))
    except (TypeError, ValueError):
        return False


def _fallback_rule(cfg: _Config) -> str:
    return FALLBACK_RULE if FALLBACK_RULE in cfg.detectable else cfg.detectable[0]


def _cell_candidates(curves, cfg: _Config, duration: float, th: dict) -> tuple[list[_Cand], bool]:
    """The cells generator (logic.cells): the top_k cells by rank score, in time order, each timed at its
    centre (plus anchor_offset_sec) with the rule of the query that ranks it and, as ground_score, the highest
    raw probability of any query in the cell. Without usable grounding scores (curves None, empty or never
    finite), top_k cells spread evenly over the clip, with rule FALLBACK_RULE and ground_score 0. Returns the
    candidates and whether the fallback was used."""
    cell_sec, top_k, off = float(th["cell_sec"]), int(th["top_k"]), float(th["anchor_offset_sec"])
    qidx, raw = [], []
    for qi, q in enumerate(cfg.queries):
        c = list((curves or {}).get(q) or [])
        if any(_finite(v) for v in c):
            qidx.append(qi)
            raw.append(c)
    n = max((len(c) for c in raw), default=0)
    if not n:
        if curves is not None:
            log.warning("grounding returned no usable scores; the candidates are evenly spaced cells")
        return [_Cand(_fallback_rule(cfg), 0.0, c.t_start, c.t_end, _clamp(c.centre + off, duration))
                for c in even_cells(duration, cell_sec, top_k)], True
    cs = [[_clean(c[i]) if i < len(c) else 0.0 for i in range(n)] for c in raw]
    x, arg = rank_curve(cs)
    tl = _timeline(curves, n, duration)
    out = []
    for c in top_cells(cells(tl.times, x, duration, cell_sec), top_k):
        rid = cfg.query_rules[qidx[arg[c.best]]] if c.best is not None else _fallback_rule(cfg)
        ground = max((curve[i] for curve in cs for i in c.members), default=0.0)
        out.append(_Cand(rid, float(ground), c.t_start, c.t_end, _clamp(c.centre + off, duration), rank=c.rank))
    return out, False


def _verify_rules(cfg: _Config, grounded: str) -> list[dict]:
    """Every detectable rule, the grounded one first and the generic 'other' last."""
    rest = [r for r in cfg.detectable if r not in (grounded, "other")]
    tail = ["other"] if grounded != "other" and "other" in cfg.rules and "other" in cfg.detectable else []
    return [cfg.rules[r] for r in [grounded, *rest, *tail]]


def _multi_rules(cfg: _Config) -> list[dict]:
    """The multi check's questions: every detectable rule but the generic 'other', in catalogue order."""
    return [cfg.rules[r] for r in cfg.detectable if r != "other"]


def _take_multi(c: _Cand, rules: list[dict], answer: tuple[list, str], verify_th: float) -> None:
    """Record a multi check's answer on c: P(YES) per rule, the observation, p_yes = the highest answer and
    vtype = its rule when it reaches verify ("none" below). With no answer at all p_yes stays None: the check
    failed, and the keep rule falls back to grounding as in single mode."""
    ps, observation = answer
    p_rules = {str(r["id"]): p for r, p in zip(rules, ps, strict=True)}
    c.p_rules, c.observation = p_rules, observation
    known = {r: p for r, p in p_rules.items() if p is not None}
    if known:
        best = max(known, key=lambda r: (known[r], r == c.rule_id))  # the grounded rule on a tie
        c.p_yes, c.vtype = known[best], best if known[best] >= verify_th else "none"


def _person_box(src: str, start: float, end: float, deadline: float, k: int) -> tuple[int, int, int, int] | None:
    """Window k's person crop box, or None for the full frame: no person in it, the detector failed, or too
    little time is left to wait for it."""
    left = _left(deadline)
    if left < WINDOWS_MIN_LEFT_SEC:
        log.warning("window %d: %.1f s before the deadline, too late for the person box; cut uncropped", k, left)
        return None
    try:
        box = zoom.person_box(src, start, end)
        if box is None:
            return None
        x0, y0, x1, y1 = (int(v) for v in box)
    except Exception as e:
        log.warning("window %d: person box failed (%s: %s); cut uncropped", k, type(e).__name__, e)
        return None
    return x0, y0, x1, y1


def _corpus_citation(x) -> dict | None:
    """The judge's citation when each of its fields (document, clause, quote, url) is a non-empty string; else
    None, and the event keeps its rule's catalogue citation."""
    if isinstance(x, dict) and all(isinstance(x.get(k), str) and x[k].strip() for k in clients.CITATION_FIELDS):
        return {k: x[k] for k in clients.CITATION_FIELDS}
    return None


def _judge_candidate(rule: dict) -> dict:
    return {"id": rule["id"], "name": rule["name"],
            "visual_criteria": list(rule.get("visual_criteria") or []),
            "never_claim": list(rule.get("never_claim") or []),
            "citations": [{"document": c.get("document", ""), "clause": c.get("clause", ""),
                           "quote": c.get("quote", "")} for c in rule.get("citations") or []]}


def _best_type(type_probs: dict[str, float], grounded: str) -> str:
    """VideoJev's most likely rule; the grounded rule when there are no choices or on a tie."""
    probs = {k: v for k, v in type_probs.items() if k != "none"}
    if not probs:
        return grounded
    return max(probs, key=lambda r: (probs[r], r == grounded))


def _evidence(src: str, ev: Event) -> list[Frame]:
    """1-3 frames for a kept event. If those fail, one frame at the timecode, then at the segment start, the
    segment end and the clip start. A kept event is never dropped for want of a frame (recall first); it is
    left without one only if no frame of the clip decodes at all."""
    try:
        return frames.evidence(src, ev.timecode, ev.t_start, ev.t_end)
    except frames.VideoError as e:
        log.warning("evidence frames failed at %.2f s: %s", ev.timecode, e)
    for t in (ev.timecode, ev.t_start, ev.t_end, 0.0):
        try:
            return [frames.jpeg_at(src, t)]
        except frames.VideoError as e:
            log.warning("no frame at %.2f s: %s", t, e)
    log.error("kept event at %.2f s has no evidence frame", ev.timecode)
    return []


def _work_dir(video_path: str) -> tuple[str, str | None]:
    """A fresh folder for the job's windows and upright copy: under the temp folder the environment names
    (env.sh points TMPDIR inside the root on the pod); without one, under a work/ folder next to the input,
    never the system /tmp. Returns the folder and the work/ folder if this call created it."""
    if any(os.environ.get(k) for k in ("TMPDIR", "TEMP", "TMP")):
        return tempfile.mkdtemp(prefix="review-"), None
    parent = Path(video_path).resolve().parent / "work"
    made = None if parent.is_dir() else str(parent)
    parent.mkdir(exist_ok=True)
    return tempfile.mkdtemp(prefix="review-", dir=parent), made


def analyze(video_path, deadline: float, on_stage: Callable[[str], None]) -> PipelineResult:
    """Review one video. deadline is a time.time() value; work files live in a folder removed afterwards."""
    cfg = _config()
    work, made = _work_dir(str(video_path))
    pools: list[ThreadPoolExecutor] = []
    try:
        return _run(str(video_path), deadline, on_stage, cfg, work, pools)
    finally:
        for p in pools:  # never wait for a straggling request; each has its own timeout
            p.shutdown(wait=False, cancel_futures=True)
        if os.environ.get("PIPELINE_KEEP_WORK") != "1":
            shutil.rmtree(work, ignore_errors=True)
            if made:
                try:
                    os.rmdir(made)  # only the work/ folder this job created, and only when empty
                except OSError:
                    pass


def _run(video_path: str, deadline: float, on_stage, cfg: _Config, work: str,
         pools: list[ThreadPoolExecutor]) -> PipelineResult:
    th = cfg.th
    timings: dict[str, float] = {}
    t_all = time.monotonic()

    def lap(name: str, t0: float) -> None:
        timings[name] = round(time.monotonic() - t0, 3)

    # 1. decoding: one upright copy when the input needs it; every later step reads src
    on_stage("decoding")
    t0 = time.monotonic()
    nt = max(NORMALIZE_MIN_SEC, min(NORMALIZE_MAX_SEC, _left(deadline) - GROUND_START_RESERVE_SEC))
    try:
        src = frames.normalize(video_path, os.path.join(work, "norm.mp4"), timeout=nt)
    except frames.VideoError as e:
        log.warning("upright copy failed (%s); using the original file", e)
        src = video_path
    duration = frames.probe(src)[0]  # an unreadable input fails the job here
    lap("decoding", t0)

    # 2. grounding: one VideoJev pass with every query; candidates over all rules. With the cells generator a
    # failed grounding call (an error, or the model busy for too long) leaves evenly spaced cells to check.
    on_stage("grounding")
    t0 = time.monotonic()
    gt = max(0.0, _left(deadline) - GROUND_START_RESERVE_SEC)
    use_cells = th["generator"] == "cells"
    curves = None
    try:
        with model_wait(gt):
            curves = ground(src, cfg.queries)
    except videojev.ModelBusy as e:
        if not use_cells:
            raise DeadlineExceeded(f"VideoJev stayed busy with other jobs, so grounding could not start at least "
                                   f"{GROUND_START_RESERVE_SEC:.0f} s before the deadline") from e
        log.warning("grounding skipped (%s); the candidates are evenly spaced cells", e)
    except Exception as e:
        if not use_cells:
            raise
        log.warning("grounding failed (%s: %s); the candidates are evenly spaced cells", type(e).__name__, e)
    if _left(deadline) - VERIFY_RESERVE_SEC < VERIFY_MIN_SEC:
        log.warning("grounding finished %.1f s before the deadline: too late for the check step; "
                    "events are kept on grounding scores alone", _left(deadline))
    cands: list[_Cand] = []
    grid_fallback = False
    if use_cells:
        cands, grid_fallback = _cell_candidates(curves, cfg, duration, th)
    else:
        rule_curves, n = _rule_curves(curves, cfg)
        if n:
            tl = _timeline(curves, n, duration)
            per_rule = {rid: segments(c, tl.step, float(th["cand"])) for rid, c in rule_curves.items()}
            cands = [_candidate(rid, s, e, sc, rule_curves[rid], tl, duration, th)
                     for rid, s, e, sc in top_candidates(per_rule, int(th["top_k"]))]
            for r, c in enumerate(cands):  # top_candidates gives them best first
                c.rank = r
    lap("grounding", t0)

    pool = ThreadPoolExecutor(max_workers=POOL_SIZE, thread_name_prefix="review")
    pools.append(pool)
    t0 = time.monotonic()
    if cands and _left(deadline) >= WINDOWS_MIN_LEFT_SEC:
        def cut(k: int) -> str | None:
            # with zoom, cropped to the people in it; a cropped cut that fails is retried on the full frame
            c = cands[k]
            start, end = _clamp(c.t_start - WINDOW_PAD_SEC, duration), _clamp(c.t_end + WINDOW_PAD_SEC, duration)
            out = os.path.join(work, f"window{k}.mp4")
            if th["zoom"]:
                c.crop = _person_box(src, start, end, deadline, k)
            if c.crop:
                try:
                    return frames.cut_window(src, start, end, out, crop=c.crop)
                except frames.VideoError as e:
                    log.warning("window %d failed cropped (%s); cutting it uncropped", k, e)
                    c.crop = None
            try:
                return frames.cut_window(src, start, end, out)
            except frames.VideoError as e:
                log.warning("window %d failed: %s", k, e)
                return None

        for c, w in zip(cands, pool.map(cut, range(len(cands))), strict=True):
            c.window = w
    lap("windows", t0)

    # 3. verifying: InternVideo3 on every window in parallel, while VideoJev picks each window's type
    on_stage("verifying")
    t0 = time.monotonic()
    vt = min(VERIFY_MAX_SEC, _left(deadline) - VERIFY_RESERVE_SEC)
    multi = th["verify_mode"] == "multi"
    multi_rules = _multi_rules(cfg) if multi else []
    futures = {}
    if vt >= VERIFY_MIN_SEC:
        futures = {k: pool.submit(verify_multi, c.window, multi_rules, vt) if multi
                   else pool.submit(verify, c.window, _verify_rules(cfg, c.rule_id), vt)
                   for k, c in enumerate(cands) if c.window}
    t1 = time.monotonic()
    for c in cands:
        if c.window is None:
            continue
        budget = _left(deadline) - CHOOSE_MIN_LEFT_SEC  # how long this choice may wait for the model
        if budget <= 0:
            break
        try:
            with model_wait(budget):
                probs = choose(c.window, cfg.options)
            c.type_probs = {rid: _clean(probs.get(opt))
                            for opt, rid in zip(cfg.options, cfg.option_ids, strict=True)}
        except videojev.ModelBusy as e:
            log.warning("type choices skipped: %s", e)
            break
        except Exception as e:
            log.warning("type choice failed: %s: %s", type(e).__name__, e)
    lap("choosing", t1)
    if futures:
        done, _ = wait(futures.values(), timeout=max(0.0, t0 + vt + 5 - time.monotonic()))
        for k, f in futures.items():
            if f not in done:
                continue
            try:
                if multi:
                    _take_multi(cands[k], multi_rules, f.result(), float(th["verify"]))
                else:
                    cands[k].p_yes, cands[k].vtype, cands[k].observation = f.result()
            except Exception as e:
                log.warning("verify failed: %s", type(e).__name__)
    lap("verifying", t0)

    # 4. keep or drop every candidate (rejected ones stay in the result for the threshold sweep). A kept
    # event takes its type BEFORE merging, so same-type events within merge_sec become one: in multi mode the
    # rule the check answered YES to (vtype), otherwise VideoJev's best type.
    events = []
    for c in cands:
        kept, reason = keep_decision(c.score, c.p_yes, c.p_yes is not None, th)
        rid = c.rule_id
        if kept:
            rid = c.vtype if multi and c.vtype in cfg.detectable else _best_type(c.type_probs, c.rule_id)
        events.append(Event(rule_id=rid, t_start=c.t_start, t_end=c.t_end, timecode=c.timecode, ground_score=c.score,
                            type_probs=c.type_probs, verify_score=c.p_yes, kept=kept, reason=reason,
                            observation=c.observation, rank=c.rank, crop=list(c.crop) if c.crop else None,
                            p_rules=c.p_rules))
    merged = merge_events_indexed(events, float(th["merge_sec"]))
    events = [ev for ev, _ in merged]
    kept_evs = [(ev, cands[i]) for ev, i in merged if ev.kept]  # each with its best member's candidate

    # 5. writing: evidence frames first (the judge looks at them), then type and justification
    on_stage("writing")
    t0 = time.monotonic()
    for ev, _ in kept_evs:
        ev.evidence = _evidence(src, ev)
        if ev.evidence:
            ev.timecode = ev.evidence[0].t
    lap("evidence", t0)

    t1 = time.monotonic()

    def candidate_ids(ev: Event, c: _Cand) -> list[str]:
        # multi mode ranks the rules by the check's answers; without any (or in single mode) by VideoJev's choice
        answered = {r: p for r, p in (c.p_rules or {}).items() if p is not None}
        return judge_candidates(answered or ev.type_probs or {c.rule_id: 1.0},
                                c.vtype if c.vtype in cfg.detectable else "")

    cand_ids = [candidate_ids(ev, c) for ev, c in kept_evs]
    answers: dict[int, dict | None] = {}
    jt = min(JUDGE_MAX_SEC, _left(deadline) - JUDGE_RESERVE_SEC)
    if kept_evs and jt >= JUDGE_MIN_SEC:
        jpool = ThreadPoolExecutor(max_workers=min(POOL_SIZE, len(kept_evs)), thread_name_prefix="judge")
        pools.append(jpool)
        jf = {i: jpool.submit(judge_answer, ev.observation, ev.evidence,
                              [_judge_candidate(cfg.rules[r]) for r in cand_ids[i]], jt)
              for i, (ev, _) in enumerate(kept_evs)}
        done, _ = wait(jf.values(), timeout=jt + 2)
        for i, f in jf.items():
            if f in done:
                try:
                    answers[i] = f.result()
                except Exception as e:
                    log.warning("judge failed: %s", type(e).__name__)
    for i, (ev, _) in enumerate(kept_evs):
        a = answers.get(i)
        a = a if isinstance(a, dict) else {}
        rid, just = a.get("rule_id"), a.get("justification")
        if accept_judge(rid, cand_ids[i]) and just:
            ev.type_source, ev.judge_rule_id, ev.justification = "judge", rid, just
            ev.citation = _corpus_citation(a.get("citation"))
        else:
            ev.type_source, ev.judge_rule_id, ev.justification = "fallback", None, cfg.rules[ev.rule_id]["why"]
    lap("judging", t1)
    lap("writing", t0)

    events.sort(key=lambda e: e.timecode)
    timings["total"] = round(time.monotonic() - t_all, 3)
    versions = {
        "pipeline": "real",
        "videojev": videojev.get().version(),
        "ground_fps": f"{videojev.GROUND_FPS:g}",
        "verifier": f"{clients.IV3_MODEL} at {', '.join(clients.iv3_urls())}",
        "judge": (f"{os.environ.get('JUDGE_MODEL', clients.JUDGE_MODEL)} at "
                  f"{os.environ.get('JUDGE_URL', clients.JUDGE_URL)}") if clients.judge_enabled() else "off",
        "zoom": zoom.DETECTOR if th["zoom"] else "off",
        "encoder": frames.video_encoder(),
        "rules": cfg.rules_version,
        "thresholds": json.dumps(th, sort_keys=True),
    }
    verdict = "violation" if any(e.kept for e in events) else "no_violation"
    return PipelineResult(verdict, events, timings, versions, generator=th["generator"],
                          cell_sec=float(th["cell_sec"]) if use_cells else None, generator_fallback=grid_fallback)
