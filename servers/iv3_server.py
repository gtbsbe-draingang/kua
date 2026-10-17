"""InternVideo3-8B check server. Runs in env/iv3, from svc/:

    env/iv3/bin/python -m uvicorn servers.iv3_server:app --host 127.0.0.1 --port 30000

GET  /healthz -> {"ok": true}. The port opens only after the model has loaded and
     answered a warm-up request, so a reachable /healthz means ready.
POST /verify {"video_path": str, "prompt": str, "fps": 4}
     -> {"p_yes": float, "type_id": str, "observation": str, "raw": str, ...}
     plus diagnostics: answer, yes_no_mass, frames, native_size, decoded_size,
     grid_thw, prompt_tokens, timings. Any error -> HTTP 500 {"error": "..."}.
POST /verify {..., "n_answers": n} with 1 <= n <= 20, for a prompt of n numbered
     YES/NO questions -> {"p_yes": float, "p_yes_list": [float | null] * n,
     "observation": str, "raw": str, ...} plus frames, native_size, decoded_size,
     grid_thw, prompt_tokens, timings. No answer line at all -> HTTP 500
     {"error": "no answers"}.
POST /generate {"prompt": str (1-20000 chars), "system": str = "", "max_new_tokens": 1-512 (384)}
     -> {"text": str, "seconds": float}. Text only: the chat template with the
     system turn (when given), the user turn and the generation prompt, then
     greedy decoding with no logits processor. It takes a request slot and the
     model lock like /verify. Any error -> HTTP 500 {"error": "..."}.

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
  it, with min_pixels = max_pixels = 128*32*32. longest_edge is a pixel budget
  over all frames together (67,108,864), so short windows keep their native size.
  When the budget forces the processor to scale frames down (long or large
  input), PyAV decodes them straight to the size the processor would pick
  itself (its own smart_resize: aspect kept, multiples of 32, never larger than
  native) and the processor is told not to resize again. Otherwise frames are
  decoded at native size and the processor resizes as usual (it only rounds to
  its 32-pixel grid, or scales a tiny input up). Either way a request holds at
  most about 3 x 67 MP of uint8 frames (about 200 MB) plus the processor's
  float32 pixels, never one native frame per sample (a 1080p clip at 768
  frames would be 4.8 GB).
- Concurrency: at most IV3_SLOTS requests (default 2) are past decoding at any
  time, i.e. decoding, preprocessing or holding inputs for the model; the rest
  wait. One lock around the model: one request runs on the GPU while the next
  one decodes and preprocesses. A request that has not reached the model within
  IV3_LOCK_TIMEOUT seconds (default 300) fails with HTTP 500.
- p_yes: the assistant turn is prefilled with "ANSWER:". At the next position,
  p_yes is the softmax over the logits of the YES and NO first tokens.
  YES/Yes/yes and NO/No/no, each with and without a leading space, are all
  single tokens in this tokenizer (checked at startup); the model emits the
  leading-space form after "ANSWER:". The softmax runs over all twelve and the
  six probabilities per side are summed:
      p_yes = sum(exp(logit[y]) for y in YES) / sum(exp(logit[t]) for t in YES + NO)
  yes_no_mass is the share of the full-vocabulary softmax on those twelve
  tokens; a low value means the model did not answer in the format. The logits
  are those of the first generation step only, copied out by a logits processor
  (greedy decoding adds no processor that changes them).
- Then greedy decoding (do_sample=False) of up to 200 new tokens continues into
  "TYPE: <id>" and "OBSERVATION: <text>", parsed leniently. raw is "ANSWER:"
  plus the generated text. type_id is the model's word, lowercased; the server
  does not check it against the rule catalogue. The caller validates it.
- n_answers: no prefill, so raw is the generated text alone. Greedy decoding of
  up to 16 n + 200 new tokens, while a logits processor records
  logsumexp(YES) - logsumexp(NO) over the same twelve tokens at every step.
  Answer k is the sigmoid of that margin at the first YES/NO token that follows
  a line start "<k>." or "<k>:" (answers_from_steps), null where the model wrote
  no such line. p_yes is the max of the answers, so a client that reads only
  p_yes still gets a number. observation is parsed as above.
"""
from __future__ import annotations

import os

# Before torch is imported: the pod's CPU quota is 16 cores, shared.
os.environ.setdefault("OMP_NUM_THREADS", "4")

import logging  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402

import av  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from transformers import LogitsProcessor, LogitsProcessorList  # noqa: E402

REPO = os.environ.get("IV3_REPO", "yanziang/InternVideo3-8B-Instruct")
MIN_PIXELS = MAX_PIXELS = 128 * 32 * 32  # inference_demo.py
VIDEO_SIZE = {"longest_edge": MAX_PIXELS * 512, "shortest_edge": MIN_PIXELS * 32}  # inference_demo.py
MAX_NEW_TOKENS = 200
MAX_FRAMES = int(os.environ.get("IV3_MAX_FRAMES", "768"))  # the video processor's own max_frames
LOCK_TIMEOUT = float(os.environ.get("IV3_LOCK_TIMEOUT", "300"))
SLOTS = threading.BoundedSemaphore(int(os.environ.get("IV3_SLOTS", "2")))
PREFILL = "ANSWER:"
YES = ("YES", " YES", "Yes", " Yes", "yes", " yes")
NO = ("NO", " NO", "No", " No", "no", " no")
YES_NO = frozenset(YES + NO)
MAX_ANSWERS = 20
MAX_PROMPT = 20000  # characters of a /generate prompt

torch.set_num_threads(4)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("iv3")

LOCK = threading.Lock()
STATE: dict = {}


# ---------------------------------------------------------------- frames

def processor_target(n: int, height: int, width: int) -> tuple[int, int] | None:
    """The (height, width) the video processor would give n native frames when
    its pixel budget forces it to scale them down; None when it would not."""
    vp = STATE["processor"].video_processor
    tf, f = vp.temporal_patch_size, vp.patch_size * vp.merge_size
    n = max(n, tf)
    # the processor's own test (smart_resize): budget exceeded on its rounded grid
    t_bar = round(n / tf) * tf
    if t_bar * (round(height / f) * f) * (round(width / f) * f) <= VIDEO_SIZE["longest_edge"]:
        return None
    return STATE["smart_resize"](num_frames=n, height=height, width=width, temporal_factor=tf, factor=f,
                                 min_pixels=VIDEO_SIZE["shortest_edge"], max_pixels=VIDEO_SIZE["longest_edge"])


def decode_frames(path: str, fps: float) -> tuple[np.ndarray, dict]:
    """Frames nearest to t = k/fps (k = 0, 1, ...) for every t inside the video.

    Returns (frames as uint8 [T, H, W, 3], info) where info has the source fps,
    the source frame index and presentation time of each kept frame, the number
    of decoded frames, the video's duration (end of its last frame), the native
    and decoded frame sizes, and do_resize for the processor. Frames are decoded
    straight to the processor's target size when its pixel budget would scale
    them down (planned from the header's duration, checked against the real
    frame count afterwards), and land in one preallocated array.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no such file: {path}")
    indices: list[int] = []
    times: list[float] = []
    buf: np.ndarray | None = None
    count = 0
    native = plan = None  # native (h, w); decode size when the processor would scale down
    convert: dict = {}
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.thread_count = 2
        rate = stream.average_rate or stream.guessed_rate or stream.base_rate
        src_fps = float(rate) if rate else None
        if stream.duration is not None:
            header_s = float(stream.duration * stream.time_base)
        else:
            header_s = container.duration / 1e6 if container.duration else 0.0
        # frames this request will keep; unknown duration plans for the worst case
        n_plan = min(MAX_FRAMES, max(1, math.ceil(header_s * fps))) if header_s > 0 else MAX_FRAMES
        prev = None
        prev_t = prev_i = 0
        origin = None
        k = n = skipped = 0
        end = 0.0

        def keep(frame, t, i):
            nonlocal buf, count, native, plan, convert
            if count >= MAX_FRAMES:
                raise ValueError(f"more than {MAX_FRAMES} frames at {fps} fps: send a shorter window")
            if buf is None:
                native = (frame.height, frame.width)
                plan = processor_target(n_plan, *native)
                convert = dict(format="rgb24")
                if plan is not None:
                    convert.update(height=plan[0], width=plan[1], interpolation="BICUBIC")
                h, w = plan or native
                buf = np.empty((min(MAX_FRAMES, n_plan + 2), h, w, 3), dtype=np.uint8)
            elif count == len(buf):  # the header understated the duration
                grown = np.empty((min(MAX_FRAMES, 2 * len(buf)), *buf.shape[1:]), dtype=np.uint8)
                grown[:count] = buf[:count]
                buf = grown
            arr = frame.to_ndarray(**convert)
            if arr.shape != buf.shape[1:]:
                raise ValueError(f"frame size changed mid-stream: {arr.shape[:2]} after {buf.shape[1:3]}")
            buf[count] = arr
            count += 1
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
    # Frames already at the processor's target for the real count need no second
    # resize. If the plan missed (wrong header), the processor resizes as usual.
    do_resize = plan is None or processor_target(count, *native) != plan
    info = dict(src_fps=src_fps, indices=indices, times=times, decoded=n, duration=end,
                skipped=skipped, drift=drift, native=native, size=buf.shape[1:3], do_resize=do_resize)
    return buf[:count], info


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
    vp = processor.video_processor
    # the resize rule of the video processor class actually loaded (Qwen3VLVideoProcessor)
    smart_resize = getattr(sys.modules[type(vp).__module__], "smart_resize")
    STATE.update(model=model, processor=processor, tok=tok, path=path, smart_resize=smart_resize,
                 yes=torch.tensor(yes_ids), no=torch.tensor(no_ids),
                 vision_start=tok.convert_tokens_to_ids("<|vision_start|>"))
    torch.cuda.synchronize()
    log.info("loaded %s (%s) in %.1f s; video size %s; GPU memory allocated %.1f GiB",
             REPO, path, time.perf_counter() - t0, processor.video_processor.size,
             torch.cuda.memory_allocated() / 2**30)


def build_inputs(frames: np.ndarray, info: dict, prompt: str, prefill: str = PREFILL):
    from transformers.video_utils import VideoMetadata

    if "<|" in prompt:
        raise ValueError("the prompt must not contain special tokens")
    processor = STATE["processor"]
    messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + prefill
    n, h, w, _ = frames.shape
    meta = VideoMetadata(total_num_frames=info["decoded"], fps=info["src_fps"], width=w, height=h,
                         duration=info["duration"], video_backend="pyav",
                         frames_indices=list(info["indices"]))
    inputs = processor(text=[text], videos=[frames], video_metadata=[meta], do_sample_frames=False,
                       do_resize=info.get("do_resize", True), return_tensors="pt")
    grid = inputs["video_grid_thw"][0].tolist()
    if grid[0] != math.ceil(n / 2):
        raise RuntimeError(f"processor re-sampled the frames: {n} frames, video_grid_thw {grid}")
    starts = int((inputs["input_ids"][0] == STATE["vision_start"]).sum())
    if starts != grid[0]:
        raise RuntimeError(f"expected {grid[0]} timestamped frame pairs in the prompt, found {starts}")
    return inputs, grid


def build_text_inputs(prompt: str, system: str = ""):
    """Inputs for a text-only request: the chat template with the system turn (when given), the user turn and the
    generation prompt. No video, no prefill."""
    if "<|" in prompt or "<|" in system:
        raise ValueError("the prompt must not contain special tokens")
    processor = STATE["processor"]
    messages = [{"role": "system", "content": [{"type": "text", "text": system}]}] if system else []
    messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], return_tensors="pt")


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


ANSWER_LINE_RE = re.compile(r"[^\S\n]*(\d+)[.:][^\S\n]*")


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def answers_from_steps(step_texts: list[str], step_margins: list[float], n: int) -> list[float | None]:
    """P(YES) of answer lines 1..n from the generated tokens' texts and the YES/NO margin of each step.

    Answer k is sigmoid(margin) at the first token that is a YES/NO variant and has only "<k>." or
    "<k>:" before it on its line (spaces allowed around the number). The line is the joined text since
    the last newline, so "12." may span several tokens. None where no such line was generated, or
    where its margin is not finite."""
    found: dict[int, float | None] = {}
    line = ""
    for text, margin in zip(step_texts, step_margins):
        if text in YES_NO and (m := ANSWER_LINE_RE.fullmatch(line)) and 1 <= (k := int(m.group(1))) <= n:
            found.setdefault(k, sigmoid(margin) if math.isfinite(margin) else None)
        line = (line + text).rpartition("\n")[2]
    return [found.get(k) for k in range(1, n + 1)]


class FirstStepLogits(LogitsProcessor):
    """Copies the first generation step's scores to the CPU and passes them on.
    With greedy decoding and no penalties, these are the raw float32 logits of
    the token after the prompt (generate adds no processor that changes them)."""

    def __init__(self):
        self.logits: torch.Tensor | None = None

    def __call__(self, input_ids, scores):
        if self.logits is None:
            self.logits = scores[0].detach().float().cpu()
        return scores


class AllStepsYesNo(LogitsProcessor):
    """Records logsumexp(YES) - logsumexp(NO) of every generation step's scores and passes them on.
    yes and no are the token ids on the scores' device; the margins stay there until margins()."""

    def __init__(self, yes: torch.Tensor, no: torch.Tensor):
        self.yes, self.no = yes, no
        self.steps: list[torch.Tensor] = []

    def __call__(self, input_ids, scores):
        s = scores[0].detach().float()
        self.steps.append(torch.logsumexp(s[self.yes], 0) - torch.logsumexp(s[self.no], 0))
        return scores

    def margins(self) -> list[float]:
        return torch.stack(self.steps).cpu().tolist() if self.steps else []


def generate(inputs, max_new_tokens: int, capture: LogitsProcessor | None = None) -> tuple[torch.Tensor, int]:
    """Greedy decoding with capture as the only logits processor (none without it): (new token ids on the CPU,
    prompt tokens)."""
    model = STATE["model"]
    inputs = inputs.to(model.device)
    prompt_tokens = int(inputs["input_ids"].shape[1])
    try:
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None,
                logits_processor=LogitsProcessorList([capture] if capture is not None else []),
            )
        new_ids = out[0, prompt_tokens:].cpu()
        del out
    finally:
        del inputs
        torch.cuda.empty_cache()  # the GPU is shared: give back what this request cached
    return new_ids, prompt_tokens


def run_model(inputs, max_new_tokens: int = MAX_NEW_TOKENS) -> dict:
    tok = STATE["tok"]
    capture = FirstStepLogits()
    new_ids, prompt_tokens = generate(inputs, max_new_tokens, capture)
    first = capture.logits
    if first is None:
        raise RuntimeError("generate produced no step")
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


def run_model_multi(inputs, n: int) -> dict:
    """The answers to an n-question prompt (built without prefill): p_yes_list from the YES/NO margin
    at each answer line, and the generated text."""
    tok, device = STATE["tok"], STATE["model"].device
    capture = AllStepsYesNo(STATE["yes"].to(device), STATE["no"].to(device))
    # 16 tokens per line "<k>. YES" (it takes about 5), and room for a 1-3 sentence OBSERVATION
    new_ids, prompt_tokens = generate(inputs, 16 * n + 200, capture)
    margins = capture.margins()
    if len(margins) != len(new_ids):
        raise RuntimeError(f"{len(margins)} recorded steps for {len(new_ids)} new tokens")
    texts = tok.batch_decode([[int(i)] for i in new_ids], skip_special_tokens=True)
    return dict(p_yes_list=answers_from_steps(texts, margins, n), raw=tok.decode(new_ids, skip_special_tokens=True),
                prompt_tokens=prompt_tokens, new_tokens=int(new_ids.numel()))


def run_text(inputs, max_new_tokens: int) -> dict:
    """The generated text of a text-only request (build_text_inputs), greedy and with no logits processor."""
    new_ids, prompt_tokens = generate(inputs, max_new_tokens)
    return dict(text=STATE["tok"].decode(new_ids, skip_special_tokens=True), prompt_tokens=prompt_tokens,
                new_tokens=int(new_ids.numel()))


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
    n_answers: int | None = None  # set: the prompt asks this many numbered YES/NO questions


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT)
    system: str = ""
    max_new_tokens: int = Field(384, ge=1, le=512)


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
    multi = req.n_answers is not None
    try:
        if not (0 < req.fps <= 30):
            raise ValueError(f"fps must be in (0, 30], got {req.fps}")
        if not req.prompt.strip():
            raise ValueError("empty prompt")
        if multi and not (1 <= req.n_answers <= MAX_ANSWERS):
            raise ValueError(f"n_answers must be in [1, {MAX_ANSWERS}], got {req.n_answers}")
        deadline = t0 + LOCK_TIMEOUT
        if not SLOTS.acquire(timeout=LOCK_TIMEOUT):
            raise TimeoutError(f"no free slot for {LOCK_TIMEOUT:.0f} s (other requests in progress)")
        try:
            t1 = time.perf_counter()
            frames, info = decode_frames(req.video_path, req.fps)
            n_frames = len(frames)
            t2 = time.perf_counter()
            inputs, grid = build_inputs(frames, info, req.prompt, "" if multi else PREFILL)
            del frames
            t3 = time.perf_counter()
            if not LOCK.acquire(timeout=max(0.0, deadline - t3)):
                raise TimeoutError(f"the model stayed busy for {LOCK_TIMEOUT:.0f} s")
            try:
                t4 = time.perf_counter()
                out = run_model_multi(inputs, req.n_answers) if multi else run_model(inputs)
            finally:
                LOCK.release()
            del inputs
            t5 = time.perf_counter()
        finally:
            SLOTS.release()
    except Exception as exc:  # noqa: BLE001 - every failure becomes a JSON 500
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
        log.exception("verify failed for %s", req.video_path)
        return error(f"{type(exc).__name__}: {exc}")
    answer, type_id, observation = parse(out["raw"])
    timings = dict(slot_wait=t1 - t0, decode=t2 - t1, preprocess=t3 - t2, wait=t4 - t3, model=t5 - t4,
                   total=t5 - t0)
    if multi:
        answers = out["p_yes_list"]
        log.info("verify %s | %d frames decoded %sx%s (grid %s) | %d prompt + %d new tokens | answers %s | %s",
                 os.path.basename(req.video_path), n_frames, info["size"][1], info["size"][0], grid,
                 out["prompt_tokens"], out["new_tokens"],
                 " ".join("-" if p is None else f"{p:.3f}" for p in answers),
                 " ".join(f"{k} {v:.2f}s" for k, v in timings.items()))
        known = [p for p in answers if p is not None]
        if not known:
            log.warning("verify %s: no answer line in %r", os.path.basename(req.video_path), out["raw"])
            return error("no answers")
        return {
            "p_yes": max(known), "p_yes_list": answers, "observation": observation, "raw": out["raw"],
            "frames": n_frames, "native_size": list(info["native"]), "decoded_size": list(info["size"]),
            "grid_thw": grid, "prompt_tokens": out["prompt_tokens"],
            "timings": {k: round(v, 3) for k, v in timings.items()},
        }
    log.info("verify %s | %d frames %sx%s -> decoded %sx%s, processor resize %s (grid %s, drift %.3f s) | "
             "%d prompt + %d new tokens | p_yes %.4f answer %s type %s | yes_no_mass %.3f top5 %s | %s",
             os.path.basename(req.video_path), n_frames, info["native"][1], info["native"][0],
             info["size"][1], info["size"][0], info["do_resize"], grid, info["drift"],
             out["prompt_tokens"], out["new_tokens"], out["p_yes"], answer or "-", type_id or "-",
             out["yes_no_mass"], out["top5"],
             " ".join(f"{k} {v:.2f}s" for k, v in timings.items()))
    return {
        "p_yes": out["p_yes"], "type_id": type_id, "observation": observation, "raw": out["raw"],
        "answer": answer, "yes_no_mass": out["yes_no_mass"], "frames": n_frames,
        "native_size": list(info["native"]), "decoded_size": list(info["size"]),
        "grid_thw": grid, "prompt_tokens": out["prompt_tokens"],
        "timings": {k: round(v, 3) for k, v in timings.items()},
    }


@app.post("/generate")
def generate_text(req: GenerateRequest):
    t0 = time.perf_counter()
    try:
        if not req.prompt.strip():
            raise ValueError("empty prompt")
        deadline = t0 + LOCK_TIMEOUT
        if not SLOTS.acquire(timeout=LOCK_TIMEOUT):
            raise TimeoutError(f"no free slot for {LOCK_TIMEOUT:.0f} s (other requests in progress)")
        try:
            t1 = time.perf_counter()
            inputs = build_text_inputs(req.prompt, req.system)
            t2 = time.perf_counter()
            if not LOCK.acquire(timeout=max(0.0, deadline - t2)):
                raise TimeoutError(f"the model stayed busy for {LOCK_TIMEOUT:.0f} s")
            try:
                t3 = time.perf_counter()
                out = run_text(inputs, req.max_new_tokens)
            finally:
                LOCK.release()
            del inputs
            t4 = time.perf_counter()
        finally:
            SLOTS.release()
    except Exception as exc:  # noqa: BLE001 - every failure becomes a JSON 500
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
        log.exception("generate failed")
        return error(f"{type(exc).__name__}: {exc}")
    log.info("generate | %d prompt + %d new tokens | slot_wait %.2fs preprocess %.2fs wait %.2fs model %.2fs "
             "total %.2fs", out["prompt_tokens"], out["new_tokens"], t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0)
    return {"text": out["text"], "seconds": round(t4 - t0, 3)}
