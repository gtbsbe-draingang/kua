"""VideoJev-4B smoke test: grounding output, timings, and the index -> seconds map.

    env/main/bin/python tools/smoke_videojev.py CLIP "QUERY" [--fps 4.0 [--fps 2.0 ...]]
        [--choice WINDOW [--choice-fps 4.0 ...]] [--hold SECONDS] [--json OUT]

CLIP is a path or a clip name in data/clips (X-1, X-14; a Latin X also matches
a Cyrillic Х). The model loads once, as its model card does (AutoModel,
trust_remote_code, bf16, flash_attention_2, prepare_for_inference()). Then for
each --fps:
  run 1  model.predict_video(...), the API call. The first call of the process
         also pays for Triton's JIT compilation of the linear-attention kernels.
  run 2  the three stages predict_video runs (prepare_video, encode_video,
         predict_features), timed one by one. Run 2 also exposes the time each
         probability scores (record["score_times"]), which checks the map below.
It prints len(probabilities), interval, the clip duration and wall times, with
model load and inference apart.

Index -> seconds: frames are sampled at t_k = k / fps (k = 0 .. n-1, every
t_k < duration); probability i scores the pair of frames 2i and 2i+1, at their
midpoint t_i = (2i + 0.5) / fps (0.5 i + 0.125 s at 4 fps, i + 0.25 s at 2 fps),
and len(probabilities) = ceil(n / 2). A longer clip than max_frames / fps
(2048 frames) is sampled uniformly instead; fps above is then record["fps"].

--choice WINDOW also times one `choice` call on that window per --choice-fps.
--hold keeps the model on the GPU for that many seconds at the end, so its
memory can be measured.
"""
from __future__ import annotations

import os

# Before torch is imported: the pod's CPU quota is 16 cores, shared.
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import unicodedata  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = "ZhuYuhan/VideoJev-4B"
CONTEXT = 147456      # the model card's grounding setting
QA_CONTEXT = 262144   # the model card's QA setting
MAX_FRAMES = 2048
ROOT = Path(os.environ.get("MTA_ROOT", "/home/jovyan/model_train_amir"))
CHOICE_QUERY = "What happens with the cash in this clip?"
CHOICE_OPTIONS = [
    "A. The cashier hands cash to the client.",
    "B. The client hands cash to the cashier.",
    "C. Cash lies on the desk and nobody handles it.",
    "D. No cash is visible.",
]


def resolve(clip: str) -> Path:
    p = Path(clip)
    if p.is_file():
        return p

    def norm(name: str) -> str:
        return unicodedata.normalize("NFC", name).replace("Х", "X").replace("х", "x")
    want = norm(clip if clip.endswith(".mov") else f"{clip}.mov")
    for q in sorted((ROOT / "data" / "clips").iterdir()):
        if norm(q.name) == want:
            return q
    raise SystemExit(f"clip not found: {clip}")


def media_duration(path: Path) -> float:
    import av
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        return float(s.duration * s.time_base) if s.duration is not None else c.duration / 1e6


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def gib(n: int) -> str:
    return f"{n / 2**30:.2f} GiB"


def grounding(model, processor, mod, clip: Path, query: str, fps: float) -> dict:
    questions = [{"id": "q", "task": "grounding", "query": query}]
    torch.cuda.reset_peak_memory_stats()
    (r1,), run1 = timed(lambda: model.predict_video(
        processor, str(clip), questions, fps=fps, max_frames=MAX_FRAMES, context=CONTEXT))
    (pixels, record), t_prep = timed(lambda: mod.prepare_video(
        processor, str(clip), fps=fps, max_frames=MAX_FRAMES, context=CONTEXT))
    feats, t_enc = timed(lambda: model.encode_video(pixels, torch.tensor([record["grid_thw"]])))
    (r2,), t_dec = timed(lambda: model.predict_features(processor, feats, record, questions, context=CONTEXT))
    peak = torch.cuda.max_memory_allocated()

    probs = np.asarray(r2["probabilities"])
    times = np.asarray(record["score_times"])
    bounds = np.asarray(record["score_bounds"])
    n, rf = record["frames"], float(record["fps"])
    i = np.arange(len(probs))
    formula = (2 * i + 0.5) / rf
    if n % 2:
        formula[-1] = (n - 1) / rf  # an odd last frame is paired with itself
    step = float(times[1] - times[0]) if len(times) > 1 else float("nan")
    dur = media_duration(clip)
    res = dict(
        clip=clip.name, query=query, fps=fps, record_fps=rf, frames=n, grid_thw=record["grid_thw"],
        visual_tokens=record["tokens"], duration=dur, sampled_duration=record["sampled_duration"],
        len_probabilities=len(probs), interval=r2["interval"], interval_run1=r1["interval"],
        step=step, len_x_step=len(probs) * step,
        map_max_err=float(np.max(np.abs(times - formula))), len_ok=len(probs) == math.ceil(n / 2),
        run1_vs_run2_max_diff=float(np.max(np.abs(np.asarray(r1["probabilities"]) - probs))),
        first=[(int(k), float(times[k]), bounds[k].tolist(), float(probs[k])) for k in range(min(3, len(probs)))],
        last=[(int(k), float(times[k]), bounds[k].tolist(), float(probs[k])) for k in range(max(0, len(probs) - 2), len(probs))],
        top5=[(int(k), float(times[k]), float(probs[k])) for k in np.argsort(-probs)[:5]],
        above_threshold=[float(times[k]) for k in np.flatnonzero(probs >= model.config.decoder["threshold"])],
        wall=dict(predict_video=run1, prepare_video=t_prep, encode_video=t_enc, predict_features=t_dec,
                  staged_total=t_prep + t_enc + t_dec),
        peak_allocated=peak,
    )
    print(f"\n=== grounding {clip.name} @ {fps} fps: {query!r}")
    print(f"duration {dur:.3f} s (sampled {record['sampled_duration']:.3f} s) | frames {n} at {rf} fps | "
          f"grid_thw {record['grid_thw']} | visual tokens {record['tokens']}")
    print(f"len(probabilities) {len(probs)} (ceil(frames/2) {'OK' if res['len_ok'] else 'MISMATCH'}) | "
          f"interval {r2['interval']} (run 1: {r1['interval']})")
    print(f"index -> seconds: t_i = (2i + 0.5) / {rf:g}, max error vs score_times {res['map_max_err']:.2e} | "
          f"step {step:g} s | len x step {len(probs) * step:.3f} s vs duration {dur:.3f} s")
    for k, t, b, p in res["first"] + res["last"]:
        print(f"  i={k:4d}  t={t:8.3f}  cell=[{b[0]:.3f}, {b[1]:.3f}]  p={p:.4f}")
    print("top 5 (i, t, p): " + ", ".join(f"({k}, {t:.3f}, {p:.3f})" for k, t, p in res["top5"]))
    print(f"times with p >= {model.config.decoder['threshold']}: {res['above_threshold']}")
    print(f"run 1 vs run 2 max |dp| {res['run1_vs_run2_max_diff']:.2e}")
    w = res["wall"]
    print(f"wall: predict_video {w['predict_video']:.2f} s (run 1) | run 2: prepare_video {w['prepare_video']:.2f} s"
          f" + encode_video {w['encode_video']:.2f} s + predict_features {w['predict_features']:.2f} s"
          f" = {w['staged_total']:.2f} s | peak allocated {gib(peak)}", flush=True)
    return res


def choice(model, processor, window: Path, fps: float) -> dict:
    questions = [{"id": "c", "task": "choice", "query": CHOICE_QUERY, "options": CHOICE_OPTIONS}]
    walls = []
    for _ in range(2):
        (r,), t = timed(lambda: model.predict_video(
            processor, str(window), questions, fps=fps, max_frames=MAX_FRAMES, context=QA_CONTEXT))
        walls.append(t)
    print(f"\n=== choice {window.name} @ {fps} fps: answer {r['answer']!r} | probabilities "
          f"{[round(p, 4) for p in r['probabilities']]} | wall {walls[0]:.2f} s, then {walls[1]:.2f} s", flush=True)
    return dict(window=window.name, fps=fps, answer=r["answer"], probabilities=r["probabilities"], wall=walls)


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("query")
    ap.add_argument("--fps", type=float, action="append")
    ap.add_argument("--choice", help="a short window for one `choice` call")
    ap.add_argument("--choice-fps", type=float, action="append")
    ap.add_argument("--hold", type=float, default=0, help="seconds to keep the model loaded at the end")
    ap.add_argument("--json", help="write the results here")
    args = ap.parse_args()
    fps_list = args.fps or [4.0]
    clip = resolve(args.clip)
    torch.set_num_threads(4)

    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoProcessor
    import transformers
    import triton
    from importlib.metadata import version

    print(f"torch {torch.__version__} | transformers {transformers.__version__} | triton {triton.__version__} | "
          f"fla {version('flash-linear-attention')} | flash_attn {version('flash-attn')} | CC={os.environ.get('CC')}")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    path = snapshot_download(REPO, local_files_only=True)
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    model = AutoModel.from_pretrained(path, trust_remote_code=True, dtype=torch.bfloat16,
                                      device_map={"": "cuda:0"}, attn_implementation="flash_attention_2")
    model.prepare_for_inference()
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    print(f"model load {load_s:.1f} s | allocated {gib(torch.cuda.memory_allocated())}", flush=True)
    mod = sys.modules[type(model).__module__]  # the checkpoint's own code: prepare_video etc.

    results = dict(load_s=load_s, weights_allocated=torch.cuda.memory_allocated(), grounding=[], choice=[])
    for fps in fps_list:
        results["grounding"].append(grounding(model, processor, mod, clip, args.query, fps))
    if args.choice:
        for fps in args.choice_fps or [4.0]:
            results["choice"].append(choice(model, processor, resolve(args.choice), fps))
    results["reserved"] = torch.cuda.memory_reserved()
    print(f"\nmemory: allocated {gib(torch.cuda.memory_allocated())}, reserved {gib(torch.cuda.memory_reserved())}, "
          f"peak allocated {gib(torch.cuda.max_memory_allocated())}")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1, ensure_ascii=False))
        print(f"results: {args.json}")
    if args.hold > 0:
        print(f"holding the model on the GPU for {args.hold:.0f} s (pid {os.getpid()})", flush=True)
        time.sleep(args.hold)
    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
