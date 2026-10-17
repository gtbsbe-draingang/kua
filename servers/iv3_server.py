"""InternVideo3-8B check server. Runs in env/iv3, from svc/:

    env/iv3/bin/python -m uvicorn servers.iv3_server:app --host 127.0.0.1 --port 30000

GET  /healthz -> {"ok": true}. The port opens only after the model has loaded and
     answered a warm-up request, so a reachable /healthz means ready.
POST /verify {"video_path": str, "prompt": str, "fps": 4}
     -> {"p_yes": float, "type_id": str, "observation": str, "raw": str, ...}
     plus diagnostics: answer, yes_no_mass, frames, grid_thw, prompt_tokens,
     timings. Any error -> HTTP 500 {"error": "..."}.

How a request is answered:
- Frames: PyAV decodes the file and keeps, for each time k/fps (k = 0, 1, ...)
  inside the video, the decoded frame nearest to it. They reach the processor as
  one video with do_sample_frames=False and video_metadata (source fps, source
  frame indices), so the processor neither re-samples them (it would assume
  24 fps and halve them) nor stamps wrong times: InternVideo3 writes
  "<t seconds>" before each merged frame pair from that metadata. The server
  checks that video_grid_thw[0] == ceil(frames / 2) and that the prompt holds
  one <|vision_start|> per merged pair.
- Resolution: video_processor.size exactly as the model's inference_demo.py sets
  it, with min_pixels = max_pixels = 128*32*32 (a total pixel budget over all
  frames, so short windows keep their native size).
- p_yes: the assistant turn is prefilled with "ANSWER:". At the next position,
  p_yes is the softmax over the logits of the YES and NO first tokens.
  YES/Yes/yes and NO/No/no, each with and without a leading space, are all
  single tokens in this tokenizer (checked at startup); the model emits the
  leading-space form after "ANSWER:". The softmax runs over all twelve and the
  six probabilities per side are summed:
      p_yes = sum(exp(logit[y]) for y in YES) / sum(exp(logit[t]) for t in YES + NO)
  yes_no_mass is the share of the full-vocabulary softmax on those twelve
  tokens; a low value means the model did not answer in the format.
- Then greedy decoding (do_sample=False) of up to 200 new tokens continues into
  "TYPE: <id>" and "OBSERVATION: <text>", parsed leniently. raw is "ANSWER:"
  plus the generated text.
- One lock around the model. Decoding and preprocessing run outside it, so they
  overlap with another request on the GPU. A request that waits longer than
  IV3_LOCK_TIMEOUT seconds (default 300) for the lock fails with HTTP 500.
"""
from __future__ import annotations

import logging
import math
import os
import re
import threading
import time
from contextlib import asynccontextmanager

import av
import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

REPO = os.environ.get("IV3_REPO", "yanziang/InternVideo3-8B-Instruct")
MIN_PIXELS = MAX_PIXELS = 128 * 32 * 32  # inference_demo.py
VIDEO_SIZE = {"longest_edge": MAX_PIXELS * 512, "shortest_edge": MIN_PIXELS * 32}  # inference_demo.py
MAX_NEW_TOKENS = 200
MAX_FRAMES = int(os.environ.get("IV3_MAX_FRAMES", "768"))  # the video processor's own max_frames
LOCK_TIMEOUT = float(os.environ.get("IV3_LOCK_TIMEOUT", "300"))
PREFILL = "ANSWER:"
YES = ("YES", " YES", "Yes", " Yes", "yes", " yes")
NO = ("NO", " NO", "No", " No", "no", " no")

torch.set_num_threads(4)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("iv3")

LOCK = threading.Lock()
STATE: dict = {}


# ---------------------------------------------------------------- frames

def decode_frames(path: str, fps: float) -> tuple[np.ndarray, dict]:
    """Frames nearest to t = k/fps (k = 0, 1, ...) for every t inside the video.

    Returns (frames as uint8 [T, H, W, 3], info) where info has the source fps,
    the source frame index and presentation time of each kept frame, the number
    of decoded frames and the video's duration (end of its last frame).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no such file: {path}")
    frames: list[np.ndarray] = []
    indices: list[int] = []
    times: list[float] = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.thread_count = 2
        rate = stream.average_rate or stream.guessed_rate or stream.base_rate
        src_fps = float(rate) if rate else None
        prev = None
        prev_t = prev_i = 0
        origin = None
        k = n = skipped = 0
        end = 0.0

        def keep(frame, t, i):
            if len(frames) >= MAX_FRAMES:
                raise ValueError(f"more than {MAX_FRAMES} frames at {fps} fps: send a shorter window")
            frames.append(frame.to_ndarray(format="rgb24"))
            indices.append(i)
            times.append(t)

        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("a video frame has no timestamp")
            stamp = float(frame.pts * frame.time_base)
            if origin is None:
                origin = stamp
            now = stamp - origin
            if prev is not None and now <= prev_t:
                skipped += 1  # duplicate or out-of-order timestamp
                continue
            while k / fps <= now:
                target = k / fps
                if prev is not None and abs(prev_t - target) <= abs(now - target):
                    keep(prev, prev_t, prev_i)
                else:
                    keep(frame, now, n)
                k += 1
            prev, prev_t, prev_i = frame, now, n
            n += 1
            if frame.duration:
                end = now + float(frame.duration * frame.time_base)
            else:
                end = now + (1.0 / src_fps if src_fps else 0.0)
        if prev is None:
            raise ValueError("the file has no decodable video frames")
        while k / fps < end:
            keep(prev, prev_t, prev_i)
            k += 1
    if src_fps is None:
        src_fps = n / end if end > 0 else fps
    drift = max(abs(i / src_fps - t) for i, t in zip(indices, times))
    info = dict(src_fps=src_fps, indices=indices, times=times, decoded=n, duration=end,
                skipped=skipped, drift=drift)
    return np.stack(frames), info


# ---------------------------------------------------------------- model

def load() -> None:
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoProcessor

    t0 = time.perf_counter()
    # A local snapshot path, not the repo id: transformers 4.57.3 asks the Hub
    # about a repo id's tokenizer even with HF_HUB_OFFLINE=1, and fails.
    path = snapshot_download(REPO, local_files_only=True)
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    processor.video_processor.size = dict(VIDEO_SIZE)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="cuda:0", trust_remote_code=True,
    ).eval()
    tok = processor.tokenizer

    def single_tokens(words):
        ids = []
        for w in words:
            t = tok.encode(w, add_special_tokens=False)
            if len(t) != 1:
                raise RuntimeError(f"{w!r} is not one token in this tokenizer: {t}")
            ids.append(t[0])
        return ids

    yes_ids, no_ids = single_tokens(YES), single_tokens(NO)
    if set(yes_ids) & set(no_ids):
        raise RuntimeError("YES and NO variants share a token")
    STATE.update(model=model, processor=processor, tok=tok, path=path,
                 yes=torch.tensor(yes_ids), no=torch.tensor(no_ids),
                 vision_start=tok.convert_tokens_to_ids("<|vision_start|>"))
    torch.cuda.synchronize()
    log.info("loaded %s (%s) in %.1f s; video size %s; GPU memory allocated %.1f GiB",
             REPO, path, time.perf_counter() - t0, processor.video_processor.size,
             torch.cuda.memory_allocated() / 2**30)


def build_inputs(frames: np.ndarray, info: dict, prompt: str):
    from transformers.video_utils import VideoMetadata

    if "<|" in prompt:
        raise ValueError("the prompt must not contain special tokens")
    processor = STATE["processor"]
    messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + PREFILL
    n, h, w, _ = frames.shape
    meta = VideoMetadata(total_num_frames=info["decoded"], fps=info["src_fps"], width=w, height=h,
                         duration=info["duration"], video_backend="pyav",
                         frames_indices=list(info["indices"]))
    inputs = processor(text=[text], videos=[frames], video_metadata=[meta],
                       do_sample_frames=False, return_tensors="pt")
    grid = inputs["video_grid_thw"][0].tolist()
    if grid[0] != math.ceil(n / 2):
        raise RuntimeError(f"processor re-sampled the frames: {n} frames, video_grid_thw {grid}")
    starts = int((inputs["input_ids"][0] == STATE["vision_start"]).sum())
    if starts != grid[0]:
        raise RuntimeError(f"expected {grid[0]} timestamped frame pairs in the prompt, found {starts}")
    return inputs, grid


ANSWER_RE = re.compile(r"(?<![A-Za-z])ANSWER\s*\**\s*[:：=]\s*\**\s*(YES|NO)\b", re.I)
TYPE_RE = re.compile(r"(?<![A-Za-z])TYPE\s*\**\s*[:：=]\s*\**\s*[`'\"«<\[(]*\s*([A-Za-z][\w\-]*)", re.I)
OBS_RE = re.compile(r"(?<![A-Za-z])OBSERVATION\s*\**\s*[:：=]\s*(.*)", re.I | re.S)
NEXT_LABEL_RE = re.compile(r"\n\s*\**\s*(ANSWER|TYPE|OBSERVATION)\s*\**\s*[:：=]", re.I)


def parse(raw: str) -> tuple[str, str, str]:
    """(answer, type_id, observation) from the model's text; '' where missing."""
    m = ANSWER_RE.search(raw)
    answer = m.group(1).upper() if m else ""
    m = TYPE_RE.search(raw)
    type_id = m.group(1).lower().replace("-", "_") if m else ""
    if type_id in ("null", "n_a", "na", "nothing"):
        type_id = "none"
    observation = ""
    m = OBS_RE.search(raw)
    if m:
        observation = NEXT_LABEL_RE.split(m.group(1), maxsplit=1)[0]
        observation = observation.strip().strip("*").strip().strip("\"'«»`").strip()
    return answer, type_id, observation


def run_model(inputs, max_new_tokens: int = MAX_NEW_TOKENS) -> dict:
    model, tok = STATE["model"], STATE["tok"]
    inputs = inputs.to(model.device)
    prompt_tokens = int(inputs["input_ids"].shape[1])
    try:
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None,
                output_logits=True, return_dict_in_generate=True,
            )
        first = out.logits[0][0].float().cpu()  # raw logits of the first new token
        new_ids = out.sequences[0, prompt_tokens:].cpu()
        del out
    finally:
        del inputs
        torch.cuda.empty_cache()  # the GPU is shared: give back what this request cached
    yes, no = first[STATE["yes"]], first[STATE["no"]]
    p_yes = float(torch.sigmoid(torch.logsumexp(yes, 0) - torch.logsumexp(no, 0)))
    if not math.isfinite(p_yes):
        raise RuntimeError("the model returned non-finite logits")
    logp = torch.log_softmax(first, -1)
    mass = float(torch.logsumexp(logp[torch.cat([STATE["yes"], STATE["no"]])], 0).exp())
    top = torch.topk(logp, 5)
    top5 = [(tok.decode([int(i)]), round(float(v.exp()), 4)) for v, i in zip(top.values, top.indices)]
    raw = PREFILL + tok.decode(new_ids, skip_special_tokens=True)
    return dict(p_yes=p_yes, yes_no_mass=mass, top5=top5, raw=raw,
                prompt_tokens=prompt_tokens, new_tokens=int(new_ids.numel()))


def warmup() -> None:
    """One tiny synthetic request through the whole path (frames, processor, generate)."""
    t0 = time.perf_counter()
    frames = np.full((4, 256, 448, 3), 128, dtype=np.uint8)
    info = dict(src_fps=4.0, indices=[0, 1, 2, 3], times=[0, .25, .5, .75], decoded=4, duration=1.0)
    inputs, _ = build_inputs(frames, info, "Is anything moving in this video? Answer YES or NO.")
    out = run_model(inputs, max_new_tokens=4)
    log.info("warm-up in %.1f s: %r, p_yes %.3f, yes_no_mass %.3f",
             time.perf_counter() - t0, out["raw"], out["p_yes"], out["yes_no_mass"])


# ---------------------------------------------------------------- HTTP

@asynccontextmanager
async def lifespan(_app: FastAPI):
    load()
    warmup()
    yield


app = FastAPI(title="iv3", lifespan=lifespan)


class VerifyRequest(BaseModel):
    video_path: str
    prompt: str
    fps: float = 4.0


def error(message: str) -> JSONResponse:
    return JSONResponse(status_code=500, content={"error": message})


@app.exception_handler(RequestValidationError)
async def bad_request(_request: Request, exc: RequestValidationError):
    return error(f"bad request: {exc}")


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/verify")
def verify(req: VerifyRequest):
    t0 = time.perf_counter()
    try:
        if not (0 < req.fps <= 30):
            raise ValueError(f"fps must be in (0, 30], got {req.fps}")
        if not req.prompt.strip():
            raise ValueError("empty prompt")
        frames, info = decode_frames(req.video_path, req.fps)
        t1 = time.perf_counter()
        inputs, grid = build_inputs(frames, info, req.prompt)
        t2 = time.perf_counter()
        if not LOCK.acquire(timeout=LOCK_TIMEOUT):
            raise TimeoutError(f"the model stayed busy for {LOCK_TIMEOUT:.0f} s")
        try:
            t3 = time.perf_counter()
            out = run_model(inputs)
        finally:
            LOCK.release()
        t4 = time.perf_counter()
    except Exception as exc:  # noqa: BLE001 - every failure becomes a JSON 500
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
        log.exception("verify failed for %s", req.video_path)
        return error(f"{type(exc).__name__}: {exc}")
    answer, type_id, observation = parse(out["raw"])
    timings = dict(decode=t1 - t0, preprocess=t2 - t1, wait=t3 - t2, model=t4 - t3, total=t4 - t0)
    log.info("verify %s | %d frames (grid %s, drift %.3f s) | %d prompt + %d new tokens | "
             "p_yes %.4f answer %s type %s | yes_no_mass %.3f top5 %s | %s",
             os.path.basename(req.video_path), len(frames), grid, info["drift"],
             out["prompt_tokens"], out["new_tokens"], out["p_yes"], answer or "-", type_id or "-",
             out["yes_no_mass"], out["top5"],
             " ".join(f"{k} {v:.2f}s" for k, v in timings.items()))
    return {
        "p_yes": out["p_yes"], "type_id": type_id, "observation": observation, "raw": out["raw"],
        "answer": answer, "yes_no_mass": out["yes_no_mass"], "frames": len(frames),
        "grid_thw": grid, "prompt_tokens": out["prompt_tokens"],
        "timings": {k: round(v, 3) for k, v in timings.items()},
    }
