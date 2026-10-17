"""Dump VideoJev grounding curves for the labelled clips, exactly as the pipeline grounds them (Task 7C-1).

    $MTA_ROOT/env/main/bin/python tools/dump_grounding.py [--mode packed] [--only X-1,X-14]
        [--labels data/labels.csv] [--clips DIR] [--out DIR] [--skip-existing]

packed (the pipeline's own call): for every clip in the labels, frames.normalize + probe as real._run does,
then real.ground(src, cfg.queries) -- one VideoJev pass with every detectable rule's queries from rules.yaml,
in the pipeline's order, at GROUND_FPS. Spies (no copied logic) keep what ground() drops: VideoJev's own
decoded `interval` per query (from the model's predict_features), the context groups the queries were packed
into (the remote prepare_query_groups) and the sampling record (prepare_video).

Each clip goes to DIR/<clip>.json: the timeline (times, bounds, step), the per-query and per-rule curves
(real._rule_curves), the per-query intervals, the packing groups, the ground-call time and the peak GPU
memory. Non-finite numbers become null and are counted (`nonfinite`); the JSON is written with
allow_nan=False. No image data is read back or written.
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

SVC = Path(__file__).resolve().parent.parent
ROOT = Path(os.environ.get("MTA_ROOT", "/home/jovyan/model_train_amir"))


def finite(x, counter: list[int]):
    """float(x) when finite; None (and one more in counter[0]) otherwise."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        counter[0] += 1
        return None
    if not math.isfinite(v):
        counter[0] += 1
        return None
    return v


def clean_list(xs, counter: list[int]) -> list:
    return [finite(x, counter) for x in xs]


class Spies:
    """Wraps the loaded model's predict_features, the remote prepare_query_groups and prepare_video to keep
    what VideoJev.ground() computes but does not return. Nothing about the call itself changes."""

    def __init__(self, vj):
        self.vj = vj
        self.results: list[dict] = []
        self.groups: list[dict] = []
        self.record: dict | None = None
        model = vj._model
        self._module = importlib.import_module(type(model).__module__)
        self._orig_pf = model.predict_features
        self._orig_groups = self._module.prepare_query_groups
        self._orig_prepare = vj._prepare_video

        def predict_features(*args, **kwargs):
            out = self._orig_pf(*args, **kwargs)
            self.results.extend(out)
            return out

        def prepare_query_groups(processor, record, questions, context):
            for batch, rows in self._orig_groups(processor, record, questions, context):
                self.groups.append({"ids": [r.get("id") for r in rows], "tokens": int(batch["input_ids"].shape[1])})
                yield batch, rows

        def prepare_video(*args, **kwargs):
            pixels, record = self._orig_prepare(*args, **kwargs)
            self.record = record
            return pixels, record

        model.predict_features = predict_features  # instance attribute shadows the bound method
        self._module.prepare_query_groups = prepare_query_groups
        if self._orig_prepare is not None:
            vj._prepare_video = prepare_video

    def reset(self) -> None:
        self.results, self.groups, self.record = [], [], None


def record_summary(record: dict | None) -> dict | None:
    if record is None:
        return None
    keys = ("frames", "fps", "grid_thw", "tokens", "sampled_duration")
    return {k: (list(record[k]) if k == "grid_thw" else record[k]) for k in keys if k in record}


def dump_packed(name: str, path: Path, info: dict, cfg, spies: Spies, out: Path) -> dict:
    import torch

    from pipeline import frames, real, videojev

    work = tempfile.mkdtemp(prefix="dump-")
    try:
        src = frames.normalize(str(path), os.path.join(work, "norm.mp4"))
        duration = frames.probe(src)[0]
        spies.reset()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        curves = real.ground(src, cfg.queries)  # exactly the pipeline's call
        sec = time.monotonic() - t0
    finally:
        shutil.rmtree(work, ignore_errors=True)
    rule_curves, n = real._rule_curves(curves, cfg)
    tl = real._timeline(curves, n, duration)
    # ground() names its questions g0..gN in query order; map the spied results back to query text
    by_id = {r["id"]: r for r in spies.results}
    qid = {q: f"g{i}" for i, q in enumerate(cfg.queries)}
    bad = [0]
    query_rules = {q: [rid for rid in cfg.detectable if q in cfg.rules[rid]["grounding_queries"]]
                   for q in cfg.queries}
    groups = [{"queries": [cfg.queries[int(i[1:])] for i in g["ids"]], "tokens": g["tokens"]}
              for g in spies.groups]
    doc = {
        "clip": name, "file": path.name, "mode": "packed",
        "has_violation": info["has_violation"], "t_labels": [t["t_label"] for t in info["truth"]],
        "duration": duration, "ground_fps": videojev.GROUND_FPS, "context": videojev.GROUND_CONTEXT,
        "videojev": videojev.get().version(), "rules_version": cfg.rules_version,
        "record": record_summary(spies.record),
        "step_sec": tl.step, "times": clean_list(tl.times, bad), "bounds": [clean_list(b, bad) for b in tl.bounds],
        "queries": list(cfg.queries), "query_rules": query_rules, "groups": groups,
        "query_curves": {q: clean_list(curves[q], bad) for q in cfg.queries},
        "query_intervals": {q: clean_list(by_id[qid[q]].get("interval") or [], bad) if qid[q] in by_id else None
                            for q in cfg.queries},
        "rule_curves": {rid: clean_list(c, bad) for rid, c in rule_curves.items()},
        "ground_sec": round(sec, 3), "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    doc["nonfinite"] = bad[0]
    (out / f"{name}.json").write_text(json.dumps(doc, ensure_ascii=False, allow_nan=False))
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["packed"], default="packed")
    ap.add_argument("--labels", default=str(SVC / "data" / "labels.csv"))
    ap.add_argument("--clips", default=str(ROOT / "data" / "clips"))
    ap.add_argument("--out", help="default: $MTA_ROOT/runs/ground_dump/<mode>")
    ap.add_argument("--only", help="comma-separated clip names (Latin X)")
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    sys.stdout.reconfigure(line_buffering=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.path.insert(0, str(SVC))

    from eval.eval import load_labels, resolve_clips
    from pipeline import real, videojev

    labels = load_labels(a.labels)
    files = resolve_clips(a.clips)
    names = [c for c in labels if c in files]
    missing = [c for c in labels if c not in files]
    if missing:
        print(f"labelled clips without a file: {', '.join(missing)}")
    if a.only:
        want = [s.strip() for s in a.only.split(",") if s.strip()]
        names = [c for c in want if c in files and c in labels]
    out = Path(a.out or ROOT / "runs" / "ground_dump" / a.mode)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    real.warmup()
    vj = videojev.get()
    cfg = real._config()
    print(f"warmup {time.monotonic() - t0:.1f} s | {vj.version()} | GROUND_FPS {videojev.GROUND_FPS:g} | "
          f"{len(cfg.detectable)} rules, {len(cfg.queries)} queries | {len(names)} clips -> {out}")
    spies = Spies(vj)
    failed = 0
    for k, name in enumerate(names, 1):
        if a.skip_existing and (out / f"{name}.json").is_file():
            print(f"[{k}/{len(names)}] {name}: exists, skipped")
            continue
        try:
            doc = dump_packed(name, files[name], labels[name], cfg, spies, out)
        except Exception as e:  # keep going; the report lists failures
            failed += 1
            logging.exception("%s failed: %s", name, e)
            continue
        tops = sorted(((max(v for v in c if v is not None), rid) for rid, c in doc["rule_curves"].items()),
                      reverse=True)[:3]
        print(f"[{k}/{len(names)}] {name}: {doc['duration']:.1f} s, {len(doc['times'])} scores, "
              f"{len(doc['groups'])} group(s) {[g['tokens'] for g in doc['groups']]} tokens, "
              f"ground {doc['ground_sec']:.2f} s, peak {doc['peak_gpu_gib']} GiB, nonfinite {doc['nonfinite']} | "
              + ", ".join(f"{rid} {m:.3f}" for m, rid in tops))
    print(f"DONE in {time.monotonic() - t0:.0f} s" + (f", {failed} failed" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
