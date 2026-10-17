"""Dump VideoJev grounding curves for the labelled clips, exactly as the pipeline grounds them (Task 7C-1).

    $MTA_ROOT/env/main/bin/python tools/dump_grounding.py [--mode packed|single|prefix] [--only X-1,X-14]
        [--labels data/labels.csv] [--clips DIR] [--out DIR] [--skip-existing]

Every mode uses the same clip preparation as real._run (frames.normalize + probe), the same queries in the
same order (real._config(): every detectable rule's grounding_queries from rules.yaml), GROUND_FPS,
GROUND_CONTEXT and the model's own sampling (prepare_video), and the same rule curves (real._rule_curves)
and timeline (real._timeline).

packed (the pipeline's own call): real.ground(src, cfg.queries), one VideoJev pass with every query.
  VideoJev packs the queries causally into one sequence after the video (prepare_query_groups), so query k
  also reads queries 0..k-1 (the context bleed of 7B-2). Spies keep what ground() drops: the decoded
  `interval` per query (predict_features), the context groups (prepare_query_groups) and the sampling record.
single (unpacked, reference): the video is decoded and encoded ONCE (prepare_video + encode_video); then
  each query gets its own predict_features call, i.e. its own language-model pass over video + that query.
  Every curve is independent of the other queries; the cost is one LM pass over the video per query.
prefix (unpacked, cheap): the same independent curves from one LM pass over the shared prefix (the video
  and the prompt text every query shares), whose KV cache (full-attention layers) and conv/recurrent
  states (linear-attention layers) are snapshotted; each query then runs only its own suffix (question,
  timestamps, decide token) on the restored cache. The readout (decision head, sigmoid, interval decode)
  is the model's own (head, format_predictions). Check it against `single` (see the report).

Each clip goes to DIR/<clip>.json: the timeline (times, bounds, step), the per-query and per-rule curves,
the per-query intervals, the packing groups (packed), the timings and the peak GPU memory. Non-finite
numbers become null and are counted (`nonfinite`); the JSON is written with allow_nan=False. No image data
is read back or written.
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


def shared_prefix(seqs: list[list[int]], markers: set[int]) -> int:
    """Length of the token prefix all sequences share, ending before any sequence's first readout marker."""
    n = min(len(s) for s in seqs)
    for s in seqs:
        first = next((i for i, t in enumerate(s) if t in markers), len(s))
        n = min(n, first)
        for i in range(n):
            if s[i] != seqs[0][i]:
                n = i
                break
    return n


class Unpacked:
    """Independent per-query grounding on one VideoJev instance: `single` (one LM pass per query) or
    `prefix` (one LM pass over the shared prefix, then each query's suffix on a restored cache)."""

    def __init__(self, vj, spies: Spies):
        self.vj, self.model, self.proc = vj, vj._model, vj._processor
        self.mod = spies._module
        self.pmod = sys.modules[spies._orig_groups.__module__]  # the remote processing_videojev
        self.decoder = dict(self.model.config.decoder)
        tok = self.proc.tokenizer
        self.decide_id = tok.convert_tokens_to_ids(self.pmod.DECIDE)
        self.option_id = tok.convert_tokens_to_ids(self.pmod.OPTION_END)
        self.video_id = self.model.config.backbone_config.video_token_id

    def _encode(self, path: str):
        import torch

        from pipeline import videojev

        t0 = time.monotonic()
        pixels, record = self.vj._prepare_video(self.proc, path, fps=videojev.GROUND_FPS,
                                                max_frames=videojev.MAX_FRAMES, context=videojev.GROUND_CONTEXT)
        t1 = time.monotonic()
        features = self.model.encode_video(pixels, torch.tensor([record["grid_thw"]]))
        torch.cuda.synchronize()
        return record, features, {"prepare": t1 - t0, "encode": time.monotonic() - t1}

    def single(self, path: str, queries: list[str]):
        from pipeline import videojev

        rows = [{"id": f"g{i}", "task": "grounding", "query": q} for i, q in enumerate(queries)]
        with self.vj._hold(None, "single grounding"):
            record, features, tm = self._encode(path)
            t0 = time.monotonic()
            results = []
            for r in rows:
                results.extend(self.model.predict_features(self.proc, features, record, [r],
                                                           context=videojev.GROUND_CONTEXT))
            tm["queries"] = time.monotonic() - t0
        return record, results, tm

    def prefix(self, path: str, queries: list[str]):
        import torch
        from transformers.cache_utils import DynamicCache, LinearAttentionCacheLayerMixin

        from pipeline import videojev

        rows = [{"id": f"g{i}", "task": "grounding", "query": q} for i, q in enumerate(queries)]
        model = self.model
        lm = model.backbone.language_model
        dev = model.get_input_embeddings().weight.device
        amp = torch.autocast("cuda", dtype=torch.bfloat16)
        with self.vj._hold(None, "prefix grounding"), torch.inference_mode():
            record, features, tm = self._encode(path)
            t0 = time.monotonic()
            # each query's own complete single-question input, exactly as prepare_query_groups builds row 0
            seqs = [self.pmod.native_inputs(self.proc, record, [r]) for r in rows]
            for ids, _ in seqs:
                if (ids.shape[1] + 63) // 64 * 64 > videojev.GROUND_CONTEXT:
                    raise ValueError("a single query exceeds the grounding context")
            ids0, types0 = seqs[0]
            n_pre = shared_prefix([ids[0].tolist() for ids, _ in seqs], {self.decide_id, self.option_id})
            for ids, _ in seqs:
                if (ids[0, n_pre:] == self.video_id).any():
                    raise ValueError("a video token lies after the shared prefix")
            grid = torch.tensor([record["grid_thw"]], device=dev)

            def positions(ids, types):
                return model.backbone.compute_3d_position_ids(
                    input_ids=ids.to(dev), inputs_embeds=None, video_grid_thw=grid, attention_mask=None,
                    mm_token_type_ids=types.to(dev))

            pos0 = positions(ids0, types0)
            base = pos0[..., n_pre:n_pre + 1]

            def suffix_positions(n):  # text after the video: every rope component steps by one
                return base + torch.arange(n, device=dev).view(*([1] * (base.ndim - 1)), n)

            for ids, types in (seqs[0], seqs[-1]):  # check that assumption against the exact positions
                exact = positions(ids, types)[..., n_pre:]
                if not torch.equal(exact, suffix_positions(ids.shape[1] - n_pre)):
                    raise ValueError("suffix positions are not a plain continuation of the prefix")

            def fa(nq, nk):
                cq = torch.tensor([0, nq], device=dev, dtype=torch.int32)
                ck = torch.tensor([0, nk], device=dev, dtype=torch.int32)
                return dict(attention_mask={"full_attention": None, "linear_attention": None},
                            cu_seq_lens_q=cq, cu_seq_lens_k=ck, max_length_q=nq, max_length_k=nk)

            cache = DynamicCache(config=lm.config)
            pre_ids = ids0[:, :n_pre].to(dev)
            with amp:
                emb = model.get_input_embeddings()(pre_ids)
                slots = pre_ids == self.video_id
                if int(slots.sum()) != len(features):
                    raise ValueError("every video token needs exactly one feature")
                emb = emb.masked_scatter(slots[..., None].expand_as(emb), features.to(emb))
                lm(inputs_embeds=emb, position_ids=pos0[..., :n_pre], past_key_values=cache, use_cache=True,
                   **fa(n_pre, n_pre))
            saved = []
            for layer in cache.layers:
                if isinstance(layer, LinearAttentionCacheLayerMixin):
                    saved.append(("lin", layer.conv_states[0].clone(), layer.recurrent_states[0].clone()))
                else:
                    saved.append(("kv", layer.keys, layer.values))
            torch.cuda.synchronize()
            tm["prefix"] = time.monotonic() - t0
            t1 = time.monotonic()
            results = []
            for r, (ids, _) in zip(rows, seqs, strict=True):
                for layer, s in zip(cache.layers, saved, strict=True):
                    if s[0] == "lin":
                        layer.conv_states[0].copy_(s[1])
                        layer.recurrent_states[0].copy_(s[2])
                    else:
                        layer.keys, layer.values = s[1], s[2]
                suf = ids[:, n_pre:].to(dev)
                n = suf.shape[1]
                with amp:
                    hidden = lm(inputs_embeds=model.get_input_embeddings()(suf), position_ids=suffix_positions(n),
                                past_key_values=cache, use_cache=True, **fa(n, n_pre + n)).last_hidden_state[0]
                q_idx = (suf[0] == self.decide_id).nonzero().flatten()
                o_idx = (suf[0] == self.option_id).nonzero().flatten()
                if len(q_idx) != 1 or len(o_idx) != record["grid_thw"][0] or not (o_idx < q_idx[0]).all():
                    raise ValueError("unexpected decision/candidate markers in a query suffix")
                with torch.autocast(device_type=dev.type, enabled=False):
                    logits = model.head(hidden[o_idx][None], hidden[q_idx])
                    mask = torch.ones_like(logits, dtype=torch.bool)
                    out = self.mod.VideoJevOutput(logits=logits, probabilities=logits.sigmoid() * mask,
                                                  score_mask=mask)
                results.extend(self.pmod.format_predictions(out, [r], record, self.decoder))
            torch.cuda.synchronize()
            tm["queries"] = time.monotonic() - t1
            tm["prefix_tokens"], tm["suffix_tokens"] = n_pre, sum(ids.shape[1] - n_pre for ids, _ in seqs)
        return record, results, tm


def record_summary(record: dict | None) -> dict | None:
    if record is None:
        return None
    keys = ("frames", "fps", "grid_thw", "tokens", "sampled_duration")
    return {k: (list(record[k]) if k == "grid_thw" else record[k]) for k in keys if k in record}


def dump_clip(name: str, path: Path, info: dict, cfg, spies: Spies, unpacked: Unpacked | None, mode: str,
              out: Path) -> dict:
    import torch

    from pipeline import frames, real, videojev

    work = tempfile.mkdtemp(prefix="dump-")
    timings = {}
    try:
        src = frames.normalize(str(path), os.path.join(work, "norm.mp4"))
        duration = frames.probe(src)[0]
        spies.reset()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        if mode == "packed":
            curves = real.ground(src, cfg.queries)  # exactly the pipeline's call
        else:
            record, results, timings = getattr(unpacked, mode)(src, cfg.queries)
            spies.results, spies.record = list(results), record
            probs = {q: [float(p) for p in r["probabilities"]] for q, r in zip(cfg.queries, results, strict=True)}
            curves = videojev.Curves(probs, times=[float(t) for t in record["score_times"]],
                                     bounds=record["score_bounds"], step_sec=2.0 / float(record["fps"]))
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
        "clip": name, "file": path.name, "mode": mode,
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
        "ground_sec": round(sec, 3), "timings": {k: round(v, 3) for k, v in timings.items()},
        "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    doc["nonfinite"] = bad[0]
    (out / f"{name}.json").write_text(json.dumps(doc, ensure_ascii=False, allow_nan=False))
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["packed", "single", "prefix"], default="packed")
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
    unpacked = None if a.mode == "packed" else Unpacked(vj, spies)
    failed = 0
    for k, name in enumerate(names, 1):
        if a.skip_existing and (out / f"{name}.json").is_file():
            print(f"[{k}/{len(names)}] {name}: exists, skipped")
            continue
        try:
            doc = dump_clip(name, files[name], labels[name], cfg, spies, unpacked, a.mode, out)
        except Exception as e:  # keep going; the report lists failures
            failed += 1
            logging.exception("%s failed: %s", name, e)
            continue
        tops = sorted(((max(v for v in c if v is not None), rid) for rid, c in doc["rule_curves"].items()),
                      reverse=True)[:3]
        print(f"[{k}/{len(names)}] {name}: {doc['duration']:.1f} s, {len(doc['times'])} scores, "
              f"{len(doc['groups'])} group(s) {[g['tokens'] for g in doc['groups']]} tokens, "
              f"ground {doc['ground_sec']:.2f} s {doc['timings'] or ''}, peak {doc['peak_gpu_gib']} GiB, "
              f"nonfinite {doc['nonfinite']} | "
              + ", ".join(f"{rid} {m:.3f}" for m, rid in tops))
    print(f"DONE in {time.monotonic() - t0:.0f} s" + (f", {failed} failed" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
