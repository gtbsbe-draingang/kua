"""Run the real pipeline in-process on clips and print what it found, with timings.

    env/main/bin/python tools/run_real.py CLIP [CLIP ...] [--fps F] [--out DIR]

CLIP is a path or a clip name in data/clips (X-1, X-14; a Latin X also matches a Cyrillic Х). A clip named
twice runs twice in this process, so the second run shows warm timings (Triton's autotune results for a new
input length are cached under the root). --fps sets GROUND_FPS for this process.

pipeline.real.warmup() runs once (rules, thresholds, encoder choice, VideoJev load). Then each clip gets
deadline = now + 240 s and one analyze(); the tool prints the stage changes, every event (kept and rejected),
the grounding timeline, each rule's highest grounding score, the timings and the versions, and writes the
debug dict (pipeline.types.to_debug, frames as times only) plus the raw grounding curves (under
run.grounding) to DIR/<clip>.json (DIR/<clip>.run<k>.json for the k-th run of the same clip). It never
prints or writes image data.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
import unicodedata
from pathlib import Path

SVC = Path(__file__).resolve().parent.parent
ROOT = Path(os.environ.get("MTA_ROOT", "/home/jovyan/model_train_amir"))
DEADLINE_SEC = 240.0


def latin(name: str) -> str:
    stem = unicodedata.normalize("NFC", name)
    stem = stem[:-4] if stem.lower().endswith(".mov") else stem
    return "X" + stem[1:] if stem[:1] in "XxХх" else stem


def resolve(clip: str) -> Path:
    for p in (Path(clip), ROOT / clip):
        if p.is_file():
            return p
    want = latin(Path(clip).name)
    clips = ROOT / "data" / "clips"
    for q in sorted(clips.iterdir()) if clips.is_dir() else []:
        if q.suffix.lower() == ".mov" and latin(q.name) == want:
            return q
    raise SystemExit(f"clip not found: {clip}")


def fmt(x, nd=3) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--fps", type=float, help="GROUND_FPS for this run (default: the code's default)")
    ap.add_argument("--out", default=str(ROOT / "runs" / "smoke_real"))
    a = ap.parse_args()
    if a.fps is not None:
        os.environ["GROUND_FPS"] = f"{a.fps:g}"  # read when pipeline.videojev is imported
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    sys.stdout.reconfigure(line_buffering=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.path.insert(0, str(SVC))

    from pipeline import frames, real, videojev
    from pipeline.types import to_debug

    paths = [resolve(c) for c in a.clips]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"GROUND_FPS {videojev.GROUND_FPS:g} | JUDGE_URL {os.environ.get('JUDGE_URL', '<unset>')!r} | "
          f"TMPDIR {os.environ.get('TMPDIR')}")

    t0 = time.monotonic()
    encoder = frames.video_encoder()
    t1 = time.monotonic()
    real.warmup()
    t2 = time.monotonic()
    print(f"encoder for internal copies: {encoder} (detected in {t1 - t0:.2f} s)")
    print(f"warmup (rules, thresholds, VideoJev load): {t2 - t1:.2f} s | videojev {videojev.get().version()}")
    import torch  # already loaded by VideoJev

    print(f"GPU allocated after load {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
    cfg = real._config()
    print(f"{len(cfg.detectable)} detectable rules, {len(cfg.queries)} grounding queries, "
          f"{len(cfg.options)} choice options")

    seen: dict[str, int] = {}
    failed = 0
    for path in paths:
        name = latin(path.name)
        seen[name] = seen.get(name, 0) + 1
        k = seen[name]
        captured = {}
        orig_ground = real.ground

        def ground_spy(video_path, queries, _orig=orig_ground, _cap=captured):
            g0 = time.monotonic()
            curves = _orig(video_path, queries)
            _cap["sec"], _cap["curves"] = time.monotonic() - g0, curves
            return curves

        real.ground = ground_spy
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        print(f"\n===== {name} run {k} ({path}) GROUND_FPS {videojev.GROUND_FPS:g} =====")

        def on_stage(s, _start=start):
            print(f"  [{time.time() - _start:7.2f} s] stage {s}")

        try:
            res = real.analyze(str(path), start + DEADLINE_SEC, on_stage)
        except Exception:
            failed += 1
            print(f"FAILED after {time.time() - start:.1f} s")
            traceback.print_exc()
            continue
        finally:
            real.ground = orig_ground
        wall = time.time() - start

        curves = captured.get("curves")
        if curves is not None:
            times, bounds = getattr(curves, "times", None), getattr(curves, "bounds", None)
            n = max((len(v) for v in curves.values()), default=0)
            print(f"grounding call {captured['sec']:.2f} s | {len(curves)} curves x {n} scores | "
                  f"step {getattr(curves, 'step_sec', None)} | times[:3] {times[:3] if times else None} | "
                  f"last cell {bounds[-1] if bounds else None}")
            rc, _ = real._rule_curves(curves, cfg)
            tops = sorted(((max(c), c.index(max(c)), rid) for rid, c in rc.items() if c), reverse=True)
            print("rule max ground score (score @ time): " + ", ".join(
                f"{rid} {m:.3f}@{times[i]:.2f}" if times else f"{rid} {m:.3f}@i{i}" for m, i, rid in tops))

        print(f"verdict {res.verdict} | {len(res.events)} events after merge, "
              f"{sum(e.kept for e in res.events)} kept | wall {wall:.2f} s | "
              f"peak GPU allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
        print(f"{'rule_id':<24}{'t_start':>8}{'t_end':>8}{'timecode':>9}{'ground':>8}{'verify':>8}  "
              f"{'kept':<6}{'reason':<17}{'type_source':<12}{'best_choice':<24}frames")
        for e in res.events:
            best = max((r for r in e.type_probs if r != "none"), key=e.type_probs.get, default="-")
            print(f"{e.rule_id:<24}{e.t_start:8.2f}{e.t_end:8.2f}{e.timecode:9.2f}{e.ground_score:8.3f}"
                  f"{fmt(e.verify_score):>8}  {str(e.kept):<6}{e.reason:<17}{e.type_source:<12}{best:<24}"
                  f"{[round(f.t, 2) for f in e.evidence]}")
        for e in res.events:
            if e.observation:
                print(f"  observation @ {e.timecode:.2f} ({'kept' if e.kept else 'rejected'}): {e.observation[:300]}")
        print(f"timings {json.dumps(res.timings)}")
        print(f"versions {json.dumps(res.versions, ensure_ascii=False)}")
        dest = out / (f"{name}.json" if k == 1 else f"{name}.run{k}.json")
        debug = to_debug(res)
        debug["run"] = {"clip": str(path), "run": k, "wall_sec": round(wall, 3),
                        "ground_call_sec": round(captured["sec"], 3) if "sec" in captured else None,
                        "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        if curves is not None:  # the raw grounding output, for checking candidates offline
            debug["run"]["grounding"] = {
                "step_sec": getattr(curves, "step_sec", None), "times": getattr(curves, "times", None),
                "bounds": getattr(curves, "bounds", None),
                "rule_curves": {rid: [round(p, 4) for p in c] for rid, c in rc.items()},
                "query_curves": {q: [round(float(p), 4) for p in c] for q, c in curves.items()}}
        dest.write_text(json.dumps(debug, ensure_ascii=False, indent=1))
        print(f"debug: {dest}")
    print("DONE" if not failed else f"DONE with {failed} failed runs")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
