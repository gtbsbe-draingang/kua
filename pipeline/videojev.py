"""VideoJev-4B wrapper: temporal grounding and multiple-choice over a video file.

torch and transformers are imported only when the model loads, so the stub pipeline and the laptop tests
never need them. One lock covers loading and every model call: the API's worker threads share one model.

API (model card and remote code of ZhuYuhan/VideoJev-4B, revision 2fc62181d25e):
- `predict_video(processor, video, questions, fps, max_frames, context)` = `prepare_video` (decode on an
  exact fps grid) + `encode_video` + `predict_features`. A question is {"id", "task": "grounding", "query"}
  or {"id", "task": "choice", "query", "options"}.
- grounding returns one independent sigmoid per PAIR of sampled frames (temporal patch 2) plus a decoded
  `interval` = [start, end] in seconds; `interval` is the model's own best segment, not a step size.
  Score i stands for the midpoint of frames 2i and 2i+1; the exact times are `record["score_times"]` and the
  cells `record["score_bounds"]`, which `ground` returns with the probabilities.
- choice returns a softmax over the options in their order, `answer_index` and `answer`.
"""
from __future__ import annotations

import importlib
import logging
import math
import os
import threading
from contextlib import contextmanager

log = logging.getLogger(__name__)

MODEL_ID = os.environ.get("VIDEOJEV_MODEL", "ZhuYuhan/VideoJev-4B")
REVISION = os.environ.get("VIDEOJEV_REVISION") or None
DEVICE = os.environ.get("VIDEOJEV_DEVICE", "cuda:0")
GROUND_FPS = float(os.environ.get("GROUND_FPS", "2.0"))
GROUND_CONTEXT = int(os.environ.get("GROUND_CONTEXT", "147456"))  # USAGE.md grounding setting
CHOICE_FPS = float(os.environ.get("CHOICE_FPS", "2.0"))  # USAGE.md QA setting
CHOICE_CONTEXT = int(os.environ.get("CHOICE_CONTEXT", "262144"))  # USAGE.md QA setting
MAX_FRAMES = int(os.environ.get("VIDEOJEV_MAX_FRAMES", "2048"))
TORCH_THREADS = 4
CHOICE_QUERY = "Which option best describes what happens at the cash desk in this video?"


class ModelBusy(TimeoutError):
    """Another job held the model for longer than this call's timeout."""


class Curves(dict):
    """Grounding probabilities keyed by query (a plain dict to callers), plus the time each index stands for:
    `times[i]` the score's time, `bounds[i]` its cell, `step_sec` the spacing. None when unknown."""

    def __init__(self, probs=(), *, times=None, bounds=None, step_sec=None):
        super().__init__(probs)
        self.times = list(times) if times is not None else None
        self.bounds = [(float(a), float(b)) for a, b in bounds] if bounds is not None else None
        self.step_sec = step_sec


def _label(i: int) -> str:
    """A, B, ..., Z, AA, AB, ...: option letters as in the model card's example."""
    out = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        out = chr(65 + r) + out
    return out


def _sentence(text: str) -> str:
    text = " ".join(str(text).split())
    text = text[:1].upper() + text[1:]
    return text if text.endswith((".", "!", "?")) else text + "."


class VideoJev:
    def __init__(self, model_id: str | None = None, device: str | None = None, revision: str | None = None):
        self.model_id = model_id or MODEL_ID
        self.device = device or DEVICE
        self.revision = revision if revision is not None else REVISION
        self._lock = threading.Lock()
        self._model = None
        self._processor = None
        self._prepare_video = None
        self._tensor = None
        self._snapshot: str | None = None

    # ---- loading --------------------------------------------------------------------------

    def _install(self, model, processor, prepare_video, tensor) -> None:
        self._model, self._processor, self._prepare_video, self._tensor = model, processor, prepare_video, tensor

    @contextmanager
    def _hold(self, timeout: float | None, what: str):
        """The model lock. timeout None waits for ever; <= 0 is a single try; otherwise wait at most that long."""
        if timeout is None:
            ok = self._lock.acquire()
        elif timeout <= 0:
            ok = self._lock.acquire(blocking=False)
        else:
            ok = self._lock.acquire(timeout=timeout)
        if not ok:
            raise ModelBusy(f"VideoJev {what}: the model stayed busy with another job for "
                            f"{max(timeout or 0.0, 0.0):.1f} s")
        try:
            yield
        finally:
            self._lock.release()

    def _source(self) -> str:
        """The local snapshot folder when the weights are cached: loading by repo id can fail offline
        (HF_HUB_OFFLINE=1). Falls back to the id itself (or a local path given as VIDEOJEV_MODEL)."""
        if os.path.isdir(self.model_id):
            return self.model_id
        try:
            from huggingface_hub import snapshot_download

            return snapshot_download(self.model_id, revision=self.revision, local_files_only=True)
        except Exception as e:
            log.warning("no local snapshot of %s (%s); loading by repo id", self.model_id, type(e).__name__)
            return self.model_id

    def load(self) -> None:
        with self._lock:
            self._load_locked()

    def _load_locked(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoProcessor

        torch.set_num_threads(TORCH_THREADS)
        source = self._source()
        kw = {"trust_remote_code": True}
        if self.revision and source == self.model_id:
            kw["revision"] = self.revision
        processor = AutoProcessor.from_pretrained(source, **kw)
        model = AutoModel.from_pretrained(source, dtype=torch.bfloat16, device_map={"": self.device},
                                          attn_implementation="flash_attention_2", **kw)
        self._snapshot = os.path.basename(source.rstrip("/")) if source != self.model_id else None
        model.prepare_for_inference()
        try:  # the remote module imports prepare_video; using it directly exposes the exact score times
            prepare = getattr(importlib.import_module(type(model).__module__), "prepare_video", None)
        except Exception:
            prepare = None
        if prepare is None:
            log.warning("VideoJev remote code has no prepare_video; grounding times fall back to the fps grid")
        self._install(model, processor, prepare, torch.tensor)

    def version(self) -> str:
        rev = (getattr(getattr(self._model, "config", None), "_commit_hash", None) or self._snapshot
               or self.revision or "main")
        return f"{self.model_id}@{rev}"

    # ---- tasks ----------------------------------------------------------------------------

    def ground(self, video_path, queries: list[str], timeout: float | None = None) -> Curves:
        """One grounding pass over the whole video with every query: raw per-index probabilities by query.
        Raises ModelBusy when the model is not free within timeout seconds."""
        queries = list(dict.fromkeys(queries))
        if not queries:
            return Curves()
        ids = [f"g{i}" for i in range(len(queries))]
        questions = [{"id": qid, "task": "grounding", "query": q} for qid, q in zip(ids, queries, strict=True)]
        record = None
        with self._hold(timeout, "grounding"):
            self._load_locked()
            if self._prepare_video is not None:
                pixels, record = self._prepare_video(self._processor, str(video_path), fps=GROUND_FPS,
                                                     max_frames=MAX_FRAMES, context=GROUND_CONTEXT)
                features = self._model.encode_video(pixels, self._tensor([record["grid_thw"]]))
                results = self._model.predict_features(self._processor, features, record, questions,
                                                       context=GROUND_CONTEXT)
            else:
                results = self._model.predict_video(self._processor, str(video_path), questions, fps=GROUND_FPS,
                                                    max_frames=MAX_FRAMES, context=GROUND_CONTEXT)
        by_id = {r["id"]: r for r in results}
        missing = [qid for qid in ids if qid not in by_id]
        if missing:
            raise ValueError(f"VideoJev returned no result for {len(missing)} of {len(ids)} queries")
        probs = {q: [float(p) for p in by_id[qid]["probabilities"]] for qid, q in zip(ids, queries, strict=True)}
        if record is None:
            return Curves(probs)
        return Curves(probs, times=[float(t) for t in record["score_times"]], bounds=record["score_bounds"],
                      step_sec=2.0 / float(record["fps"]))

    def choose(self, window_path, options: list[str], timeout: float | None = None) -> dict[str, float]:
        """VideoJev's single-choice QA over the options: softmax probability per option text.
        Raises ModelBusy when the model is not free within timeout seconds."""
        options = list(options)
        if len(options) < 2:
            raise ValueError("choice needs at least two options")
        if len(set(options)) != len(options):
            raise ValueError("choice options must be distinct")
        labelled = [f"{_label(i)}. {_sentence(o)}" for i, o in enumerate(options)]
        question = {"id": "choice", "task": "choice", "query": CHOICE_QUERY, "options": labelled}
        with self._hold(timeout, "choice"):
            self._load_locked()
            results = self._model.predict_video(self._processor, str(window_path), [question], fps=CHOICE_FPS,
                                                max_frames=MAX_FRAMES, context=CHOICE_CONTEXT)
        probs = [float(p) for p in results[0]["probabilities"]]
        if len(probs) != len(options):
            raise ValueError(f"VideoJev returned {len(probs)} probabilities for {len(options)} options")
        probs = [p if math.isfinite(p) and p > 0 else 0.0 for p in probs]
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        return dict(zip(options, probs, strict=True))


_INSTANCE: VideoJev | None = None
_INSTANCE_LOCK = threading.Lock()


def get() -> VideoJev:
    global _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is None:
            _INSTANCE = VideoJev()
        return _INSTANCE
