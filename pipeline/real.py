"""Real review pipeline (design.md section 1, steps 1-7).

VideoJev finds when something may happen and which type it looks like, InternVideo3 checks and describes
each moment, an optional judge picks the rule and words the justification, and evidence frames are cut.
Recall comes first: a failed or late model call can only fall back to the grounding score; it never drops
a moment that grounding alone would keep.

`ground`, `choose`, `verify` and `judge` are the module-level wrappers the tests mock.
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
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from pipeline import clients, frames, videojev
from pipeline.logic import (accept_judge, judge_candidates, keep_decision, merge_events, pair_timeline,
                            pick_timecode, sample_times, segments, top_candidates)
from pipeline.types import Event, Frame, PipelineResult
from rules import load_rules

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
NONE_OPTION = "normal cash handling"
WINDOW_PAD_SEC = 2.0
POOL_SIZE = 6
VERIFY_MAX_SEC, VERIFY_RESERVE_SEC, VERIFY_MIN_SEC = 60.0, 20.0, 5.0
JUDGE_MAX_SEC, JUDGE_RESERVE_SEC, JUDGE_MIN_SEC = 30.0, 10.0, 3.0
CHOOSE_MIN_LEFT_SEC = 30.0  # below this, type choices are skipped so the check and the frames still fit
TH_KEYS = ("cand", "verify", "strong", "fallback", "top_k", "merge_sec", "anchor", "anchor_offset_sec")


# ---- the model calls (mocked in tests) -----------------------------------------------------------

def ground(video_path: str, queries: list[str]) -> dict[str, list[float]]:
    return videojev.get().ground(video_path, queries)


def choose(window_path: str, options: list[str]) -> dict[str, float]:
    return videojev.get().choose(window_path, options)


def verify(window_path: str, rules: list[dict], timeout: float) -> tuple[float | None, str, str]:
    return clients.verify(window_path, rules, timeout)


def judge(observation: str, evidence: list[Frame], candidates: list[dict],
          timeout: float) -> tuple[str | None, str, float | None]:
    return clients.judge(observation, evidence, candidates, timeout)


# ---- configuration, loaded once ------------------------------------------------------------------

@dataclass(frozen=True)
class _Config:
    rules: dict[str, dict]
    detectable: list[str]  # catalogue order
    queries: list[str]  # every detectable rule's grounding queries, without repeats
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
    missing = [k for k in TH_KEYS if k not in th]
    if missing:
        raise ValueError(f"{path}: missing {', '.join(missing)}")
    for k in ("cand", "verify", "strong", "fallback", "merge_sec", "anchor_offset_sec"):
        if not _number(th[k]):
            raise ValueError(f"{path}: {k} must be a number")
    if isinstance(th["top_k"], bool) or not isinstance(th["top_k"], int) or th["top_k"] < 1:
        raise ValueError(f"{path}: top_k must be a positive integer")
    if th["anchor"] not in ("peak", "onset"):
        raise ValueError(f"{path}: anchor must be 'peak' or 'onset', not {th['anchor']!r}")
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
    queries = list(dict.fromkeys(str(q) for rid in detectable for q in rules[rid]["grounding_queries"]))
    th = _load_thresholds(th_path)
    digest = hashlib.sha256(raw).hexdigest()[:12]
    return _Config(rules, detectable, queries, options, detectable + ["none"], th, f"v{version} sha256:{digest}")


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


def _verify_rules(cfg: _Config, grounded: str) -> list[dict]:
    """Every detectable rule, the grounded one first and the generic 'other' last."""
    rest = [r for r in cfg.detectable if r not in (grounded, "other")]
    tail = ["other"] if grounded != "other" and "other" in cfg.rules and "other" in cfg.detectable else []
    return [cfg.rules[r] for r in [grounded, *rest, *tail]]


def _judge_candidate(rule: dict) -> dict:
    return {"id": rule["id"], "name": rule["name"],
            "visual_criteria": list(rule.get("visual_criteria") or []),
            "never_claim": list(rule.get("never_claim") or []),
            "citations": [{"document": c.get("document", ""), "clause": c.get("clause", ""),
                           "quote": c.get("quote", "")} for c in rule.get("citations") or []]}


def _best_type(ev: Event) -> str:
    """VideoJev's most likely rule; the grounded rule when there are no choices or on a tie."""
    probs = {k: v for k, v in ev.type_probs.items() if k != "none"}
    if not probs:
        return ev.rule_id
    return max(probs, key=lambda r: (probs[r], r == ev.rule_id))


def _evidence(src: str, ev: Event) -> list[Frame]:
    try:
        return frames.evidence(src, ev.timecode, ev.t_start, ev.t_end)
    except frames.VideoError as e:
        log.warning("evidence frames failed at %.2f s: %s", ev.timecode, e)
    try:
        return [frames.jpeg_at(src, ev.timecode)]
    except frames.VideoError as e:
        log.warning("no evidence frame at %.2f s: %s", ev.timecode, e)
        return []


def analyze(video_path, deadline: float, on_stage: Callable[[str], None]) -> PipelineResult:
    """Review one video. deadline is a time.time() value; work files live in a temporary folder."""
    cfg = _config()
    work = tempfile.mkdtemp(prefix="review-")
    pools: list[ThreadPoolExecutor] = []
    try:
        return _run(str(video_path), deadline, on_stage, cfg, work, pools)
    finally:
        for p in pools:  # never wait for a straggling request; each has its own timeout
            p.shutdown(wait=False, cancel_futures=True)
        if os.environ.get("PIPELINE_KEEP_WORK") != "1":
            shutil.rmtree(work, ignore_errors=True)


def _run(video_path: str, deadline: float, on_stage, cfg: _Config, work: str,
         pools: list[ThreadPoolExecutor]) -> PipelineResult:
    th = cfg.th
    timings: dict[str, float] = {}
    t_all = time.monotonic()

    def lap(name: str, t0: float) -> None:
        timings[name] = round(time.monotonic() - t0, 3)

    # 1. decoding: one upright H.264 copy when the input needs it; every later step reads src
    on_stage("decoding")
    t0 = time.monotonic()
    src = frames.normalize(video_path, os.path.join(work, "norm.mp4"))
    duration = frames.probe(src)[0]
    lap("decoding", t0)

    # 2. grounding: one VideoJev pass with every query; candidates over all rules
    on_stage("grounding")
    t0 = time.monotonic()
    curves = ground(src, cfg.queries)
    rule_curves, n = _rule_curves(curves, cfg)
    cands: list[_Cand] = []
    if n:
        tl = _timeline(curves, n, duration)
        per_rule = {rid: segments(c, tl.step, float(th["cand"])) for rid, c in rule_curves.items()}
        cands = [_candidate(rid, s, e, sc, rule_curves[rid], tl, duration, th)
                 for rid, s, e, sc in top_candidates(per_rule, int(th["top_k"]))]
    lap("grounding", t0)

    pool = ThreadPoolExecutor(max_workers=POOL_SIZE, thread_name_prefix="review")
    pools.append(pool)
    t0 = time.monotonic()
    if cands and _left(deadline) >= min(VERIFY_MIN_SEC + VERIFY_RESERVE_SEC, CHOOSE_MIN_LEFT_SEC):
        def cut(k: int) -> str | None:
            c = cands[k]
            try:
                return frames.cut_window(src, c.t_start - WINDOW_PAD_SEC, c.t_end + WINDOW_PAD_SEC,
                                         os.path.join(work, f"window{k}.mp4"))
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
    futures = {}
    if vt >= VERIFY_MIN_SEC:
        futures = {k: pool.submit(verify, c.window, _verify_rules(cfg, c.rule_id), vt)
                   for k, c in enumerate(cands) if c.window}
    t1 = time.monotonic()
    for c in cands:
        if c.window is None or _left(deadline) < CHOOSE_MIN_LEFT_SEC:
            continue
        try:
            probs = choose(c.window, cfg.options)
            c.type_probs = {rid: _clean(probs.get(opt)) for opt, rid in zip(cfg.options, cfg.option_ids, strict=True)}
        except Exception as e:
            log.warning("type choice failed: %s: %s", type(e).__name__, e)
    lap("choosing", t1)
    if futures:
        done, _ = wait(futures.values(), timeout=max(0.0, t0 + vt + 5 - time.monotonic()))
        for k, f in futures.items():
            if f not in done:
                continue
            try:
                cands[k].p_yes, cands[k].vtype, cands[k].observation = f.result()
            except Exception as e:
                log.warning("verify failed: %s", type(e).__name__)
    lap("verifying", t0)

    # 4. keep or drop every candidate; rejected ones stay in the result for the threshold sweep
    events, vtypes = [], {}
    for c in cands:
        kept, reason = keep_decision(c.score, c.p_yes, c.p_yes is not None, th)
        ev = Event(rule_id=c.rule_id, t_start=c.t_start, t_end=c.t_end, timecode=c.timecode,
                   ground_score=c.score, type_probs=c.type_probs, verify_score=c.p_yes, kept=kept,
                   reason=reason, observation=c.observation)
        vtypes.setdefault((ev.rule_id, ev.timecode, ev.ground_score), c.vtype)
        events.append(ev)
    events = merge_events(events, float(th["merge_sec"]))
    kept_evs = [(ev, vtypes.get((ev.rule_id, ev.timecode, ev.ground_score), "")) for ev in events if ev.kept]

    # 5. writing: evidence frames first (the judge looks at them), then type and justification
    on_stage("writing")
    t0 = time.monotonic()
    for ev, _ in kept_evs:
        ev.evidence = _evidence(src, ev)
        if ev.evidence:
            ev.timecode = ev.evidence[0].t
    lap("evidence", t0)

    t1 = time.monotonic()
    cand_ids = []
    for ev, vtype in kept_evs:
        probs = ev.type_probs or {ev.rule_id: 1.0}
        cand_ids.append(judge_candidates(probs, vtype if vtype in cfg.detectable else ""))
    answers: dict[int, tuple] = {}
    jt = min(JUDGE_MAX_SEC, _left(deadline) - JUDGE_RESERVE_SEC)
    if kept_evs and jt >= JUDGE_MIN_SEC:
        jpool = ThreadPoolExecutor(max_workers=min(POOL_SIZE, len(kept_evs)), thread_name_prefix="judge")
        pools.append(jpool)
        jf = {i: jpool.submit(judge, ev.observation, ev.evidence,
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
        best = _best_type(ev)
        rid, just, _conf = answers.get(i, (None, "", None))
        if accept_judge(rid, cand_ids[i]) and just:
            ev.type_source, ev.judge_rule_id, ev.justification = "judge", rid, just
        else:
            ev.type_source, ev.judge_rule_id, ev.justification = "fallback", None, cfg.rules[best]["why"]
        ev.rule_id = best
    lap("judging", t1)
    lap("writing", t0)

    events.sort(key=lambda e: e.timecode)
    timings["total"] = round(time.monotonic() - t_all, 3)
    versions = {
        "pipeline": "real",
        "videojev": videojev.get().version(),
        "ground_fps": f"{videojev.GROUND_FPS:g}",
        "verifier": f"{clients.IV3_MODEL} at {os.environ.get('IV3_URL', clients.IV3_URL)}",
        "judge": (f"{os.environ.get('JUDGE_MODEL', clients.JUDGE_MODEL)} at "
                  f"{os.environ.get('JUDGE_URL', clients.JUDGE_URL)}") if clients.judge_enabled() else "off",
        "rules": cfg.rules_version,
        "thresholds": json.dumps(th, sort_keys=True),
    }
    verdict = "violation" if any(e.kept for e in events) else "no_violation"
    return PipelineResult(verdict, events, timings, versions)
