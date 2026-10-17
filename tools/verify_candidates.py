"""InternVideo3 check of candidate windows with the pipeline's own window cutting and verify call (Task 7C-1).

On the pod (needs the iv3 server and the clips; no VideoJev):

    $MTA_ROOT/env/main/bin/python tools/verify_candidates.py run --cands FILE [--cands FILE ...]
        [--prompt pipeline|rule|multi] [--zoom] [--subset FILE] [--cache FILE] [--labels data/labels.csv]
        [--clips DIR]

  FILE is a cands_<name>.json from tools/analyze_candidates.py --export. For every clip and candidate the
  window [t_start - 2, t_end + 2] is cut from the clip exactly as real._run does (frames.normalize, then
  frames.cut_window with real.WINDOW_PAD_SEC; the pod's encoder is mpeg4) and checked:
    pipeline  clients.verify with the pipeline's prompt: every detectable rule, the candidate's rule first
              (real._verify_rules)
    rule      clients.verify with only the candidate's rule (its verify_prompt in the same template)
    multi     clients.verify_multi: one numbered YES/NO question per detectable rule except "other", in
              catalogue order (MULTI_RULES); p_yes = the max of the answers, type_id = the rule that gives it
  --zoom crops each window to the people in it first: zoom.person_box on the normalised copy over the
  window, then cut_window(crop=...); a detector error or no person cuts the window uncropped, as the pipeline
  does. The record keeps the box, the detector's seconds and any error.
  --subset limits the run to the windows listed in a JSON list of {"clip", "start", "end", "rule"}.
  Results go to the cache (JSON lines, default $MTA_ROOT/runs/verify/cache.jsonl), keyed by clip, window,
  rule and variant (the prompt, plus "+zoom"), so a window is checked once whichever generators share it, and
  a rerun resumes. Observations and the model's raw text are kept; no frame is read back or written.

Anywhere (CPU):

    uv run python tools/verify_candidates.py report --cands FILE --cache FILE [--cache FILE ...]
        [--subset FILE] [--variant V ...] [--json OUT] [--reps 2000]

  joins candidates and results per variant (pipeline, rule, multi, each optionally +zoom) and prints:
  per-candidate ROC-AUC of p_yes for candidates matching a label (timecode within +-2 s) against candidates
  on clean clips; per-clip AUC of max p_yes (violation vs clean); and the keep rule p_yes >= v (alone, and
  with a ground-score override) as recall over violation clips (every label matched by a kept candidate) and
  FP clips (clean clips with a kept candidate), per v. Over complete coverage it adds the decision summary
  (summarize): window and clip AUC, each against two nulls, and the operating points at m = 1, 2 and all.

Nulls (shuffle_null, mean and standard deviation over --reps permutations, seeded):
  no-signal    the window scores permuted over all windows of all clips: no link between score and clip or
               moment, the score distribution and every clip's window count kept. Its clip AUC is the value a
               score without signal gets from the window counts alone (clean clips are longer, so their max
               is drawn from more windows), and its spread is the noise of the metric on these clips.
  within-clip  the scores permuted among each clip's own windows: the clip-level signal kept, the moment
               destroyed. A window AUC or a "found" count above it is localisation; the clip max, and so the
               clip AUC, the FP clips and the verdicts, do not change under it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import statistics
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SVC = Path(__file__).resolve().parent.parent
for _p in (SVC, SVC / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from analyze_candidates import auc  # noqa: E402  (one AUC for every report)

ROOT = Path(os.environ.get("MTA_ROOT", "/home/jovyan/model_train_amir"))
TOL = 2.0
EPS = 1e-6
PAD = 2.0  # real.WINDOW_PAD_SEC, checked in run
VERIFY_TIMEOUT = {"pipeline": 60.0, "rule": 60.0, "multi": 120.0}  # multi decodes up to 16 n + 200 tokens
# eval sweep's verify grid (7C-2): 0.01..0.30 by 0.01, then 0.35..0.95 by 0.05
VS = [round(0.01 * i, 2) for i in range(1, 31)] + [round(0.05 * i, 2) for i in range(7, 20)]
MS = [1, 2, None]  # most kept events per clip; None = all


def variant_name(prompt: str, zoom: bool) -> str:
    return prompt + ("+zoom" if zoom else "")


def multi_rules(cfg) -> list[dict]:
    """The rules a multi-mode check asks: every detectable rule except "other", in catalogue order."""
    return [cfg.rules[r] for r in cfg.detectable if r != "other"]


def window_key(clip: str, start: float, end: float, rule: str, prompt: str) -> str:
    return f"{clip}|{start:.3f}|{end:.3f}|{rule}|{prompt}"


def load_cache(path: Path) -> dict[str, dict]:
    out = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                out[d["key"]] = d
    return out


def candidate_windows(cands: dict, labels: dict) -> list[dict]:
    """One row per candidate: its window (unclamped; the cut clamps), label match and clip truth."""
    rows = []
    for clip, cs in cands["clips"].items():
        info = labels.get(clip)
        if info is None:
            continue
        for k, c in enumerate(cs):
            near = [t["t_label"] for t in info["truth"] if abs(c["timecode"] - t["t_label"]) <= TOL + EPS]
            rows.append({"clip": clip, "k": k, "rule": c["rule"], "t_start": c["t_start"], "t_end": c["t_end"],
                         "timecode": c["timecode"], "score": c["score"], "raw": c.get("raw"),
                         "has_violation": info["has_violation"], "matched": bool(near),
                         "label": near[0] if near else None})
    return rows


class ReplySpy:
    """Wraps clients._post to keep the last JSON reply of the check server (raw text, token counts, sizes),
    so the run records the pipeline client's own values and the server's diagnostics from one call."""

    def __init__(self, clients):
        self.clients, self.orig, self.last = clients, clients._post, None

    def __enter__(self):
        def post(url, body, timeout):
            r = self.orig(url, body, timeout)
            try:
                self.last = r.json() if r.status_code == 200 else {"http_status": r.status_code,
                                                                   "error": r.text[:300]}
            except Exception:  # noqa: BLE001 - the client reports its own failure
                self.last = {"http_status": r.status_code}
            return r
        self.clients._post = post
        return self

    def __exit__(self, *exc):
        self.clients._post = self.orig

    def take(self) -> dict:
        d, self.last = self.last or {}, None
        return d


def check_record(prompt: str, rules: list[dict], p, tid, obs, reply: dict) -> dict:
    """The verdict fields of a cache record. multi: p is the per-rule list aligned with rules; p_yes is the max
    of its known values and type_id the first rule (catalogue order) that gives it; both None when none."""
    rec: dict = {}
    if prompt == "multi":
        p_rules = {str(r["id"]): x for r, x in zip(rules, p, strict=True)}
        known = {k: x for k, x in p_rules.items() if x is not None}
        rec["p_rules"] = p_rules
        rec["p_yes"] = max(known.values()) if known else None
        rec["type_id"] = max(known, key=known.get) if known else None
    else:
        rec["p_yes"], rec["type_id"] = p, tid
    rec["observation"] = obs
    for k in ("raw", "prompt_tokens", "decoded_size", "frames", "yes_no_mass", "http_status", "error"):
        if k in reply:
            rec[k] = reply[k]
    if isinstance(reply.get("timings"), dict):
        rec["model_sec"] = reply["timings"].get("model")
    return rec


def cmd_run(a) -> int:
    sys.stdout.reconfigure(line_buffering=True)
    from eval.eval import load_labels, resolve_clips
    from pipeline import clients, frames, real

    if real.WINDOW_PAD_SEC != PAD:
        raise SystemExit(f"real.WINDOW_PAD_SEC is {real.WINDOW_PAD_SEC}, this tool assumes {PAD}")
    if a.zoom:
        from pipeline import zoom
    cfg = real._config()
    variant = variant_name(a.prompt, a.zoom)
    labels = load_labels(a.labels)
    files = resolve_clips(a.clips)
    cache_path = Path(a.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = load_cache(cache_path)
    want = None
    if a.subset:
        want = {(w["clip"], round(w["start"], 3), round(w["end"], 3), w["rule"])
                for w in json.loads(Path(a.subset).read_text())}
    todo: dict[str, list[tuple[float, float, str, float]]] = {}
    for f in a.cands:
        for r in candidate_windows(json.loads(Path(f).read_text()), labels):
            s, e = r["t_start"] - PAD, r["t_end"] + PAD
            if want is not None and (r["clip"], round(s, 3), round(e, 3), r["rule"]) not in want:
                continue
            if window_key(r["clip"], s, e, r["rule"], variant) in cache:
                continue
            todo.setdefault(r["clip"], [])
            if all((s, e, r["rule"]) != w[:3] for w in todo[r["clip"]]):
                todo[r["clip"]].append((s, e, r["rule"], r["timecode"]))
    n = sum(len(v) for v in todo.values())
    print(f"encoder {frames.video_encoder()} | variant {variant} | {n} windows to check on {len(todo)} clips "
          f"({len(cache)} cached)")
    timeout = VERIFY_TIMEOUT[a.prompt]
    done, t_all = 0, time.monotonic()
    with cache_path.open("a", encoding="utf-8") as sink, ThreadPoolExecutor(max_workers=2) as pool, \
            ReplySpy(clients) as spy:
        for clip, wins in todo.items():
            work = tempfile.mkdtemp(prefix="verify-")
            try:
                src = frames.normalize(str(files[clip]), os.path.join(work, "norm.mp4"))

                def cut(i, _src=src, _work=work, _wins=wins):
                    s, e = _wins[i][:2]
                    z: dict = {}
                    if a.zoom:  # as the pipeline will: the person box on the upright copy, else uncropped
                        t0 = time.monotonic()
                        try:
                            z["crop"] = zoom.person_box(_src, s, e)
                        except Exception as err:  # noqa: BLE001
                            z["crop"], z["zoom_error"] = None, f"{type(err).__name__}: {err}"[:300]
                        z["zoom_sec"] = round(time.monotonic() - t0, 3)
                    t0 = time.monotonic()
                    path = frames.cut_window(_src, s, e, os.path.join(_work, f"w{i}.mp4"), crop=z.get("crop"))
                    z["cut_sec"] = round(time.monotonic() - t0, 3)
                    return path, z

                futs = [pool.submit(cut, i) for i in range(len(wins))]  # cut ahead while iv3 works
                for i, (s, e, rule, tc) in enumerate(wins):
                    try:
                        path, z = futs[i].result()
                    except frames.VideoError as err:
                        print(f"  {clip} [{s:.2f}, {e:.2f}] cut failed: {err}")
                        continue
                    t0 = time.monotonic()
                    if a.prompt == "multi":
                        rules = multi_rules(cfg)
                        p, obs = clients.verify_multi(path, rules, timeout)
                        tid = None
                    else:
                        rules = real._verify_rules(cfg, rule) if a.prompt == "pipeline" else [cfg.rules[rule]]
                        p, tid, obs = clients.verify(path, rules, timeout)
                    sec = time.monotonic() - t0
                    rec = {"key": window_key(clip, s, e, rule, variant), "clip": clip, "start": s, "end": e,
                           "rule": rule, "prompt": variant, "timecode": tc,
                           **check_record(a.prompt, rules, p, tid, obs, spy.take()),
                           "verify_sec": round(sec, 3), **z}
                    sink.write(json.dumps(rec, ensure_ascii=False, allow_nan=False) + "\n")
                    sink.flush()
                    done += 1
                    ps = "None" if rec["p_yes"] is None else f"{rec['p_yes']:.3f}"
                    crop = f" crop {z['crop']}" if z.get("crop") else (" no crop" if a.zoom else "")
                    print(f"[{done}/{n}] {clip} [{s:6.2f}, {e:6.2f}] {rule:<34} p_yes {ps} "
                          f"{rec['type_id'] or '-':<34} {sec:5.2f} s{crop}")
            finally:
                shutil.rmtree(work, ignore_errors=True)
    print(f"DONE {done} windows in {time.monotonic() - t_all:.0f} s")
    if a.zoom:
        import torch
        if torch.cuda.is_available():
            print(f"detector GPU memory: max allocated {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB, "
                  f"reserved {torch.cuda.memory_reserved() / 2**20:.0f} MiB")
    return 0


# ---- report -------------------------------------------------------------------------------------

JOIN_FIELDS = ("p_yes", "type_id", "verify_sec", "p_rules", "zoom_sec", "crop", "zoom_error", "prompt_tokens")


def join(cands: dict, labels: dict, cache: dict, prompt: str) -> list[dict]:
    """One row per candidate with the variant's result fields (None where the window was not checked)."""
    rows = []
    for r in candidate_windows(cands, labels):
        s, e = r["t_start"] - PAD, r["t_end"] + PAD
        hit = cache.get(window_key(r["clip"], s, e, r["rule"], prompt))
        rows.append({**r, **{k: None if hit is None else hit.get(k) for k in JOIN_FIELDS},
                     "checked": hit is not None})
    return rows


def keep_table(rows: list[dict], labels: dict, vs: list[float], strong: float | None = None,
               ground_key: str = "raw") -> list[tuple[float, float, int, int]]:
    """For each v: (v, recall over violation clips, found clips, FP clips). A candidate is kept when its
    p_yes >= v (or, with strong, its ground score >= strong); a missing p_yes is never kept by verify."""
    viol = [c for c, i in labels.items() if i["has_violation"]]
    clean = [c for c, i in labels.items() if not i["has_violation"]]
    out = []
    for v in vs:
        kept = [r for r in rows if (r["p_yes"] is not None and r["p_yes"] >= v)
                or (strong is not None and (r.get(ground_key) or -math.inf) >= strong)]
        found = sum(all(any(r["clip"] == c and abs(r["timecode"] - t["t_label"]) <= TOL + EPS for r in kept)
                        for t in labels[c]["truth"]) for c in viol)
        fp = sum(any(r["clip"] == c for r in kept) for c in clean)
        out.append((v, found / len(viol) if viol else 0.0, found, fp))
    return out


def is_matched(r: dict, labels: dict) -> bool:
    """The row's timecode is within +-2 s of one of its clip's labels."""
    return any(abs(r["timecode"] - t["t_label"]) <= TOL + EPS for t in labels[r["clip"]]["truth"])


def _found(kept_by: dict[str, list[dict]], labels: dict, viol: list[str]) -> int:
    return sum(all(any(abs(r["timecode"] - t["t_label"]) <= TOL + EPS for r in kept_by.get(c, ()))
                   for t in labels[c]["truth"]) for c in viol)


def operating_points(rows: list[dict], labels: dict, vs: list[float], ms: list[int | None],
                     key: str = "p_yes") -> list[dict]:
    """Keep score >= v, at most m events per clip (highest score first; m None = every one). Per (v, m):
    violation clips with a kept event (verdict recall), violation clips with every label matched (found; a
    clip with two labels needs both), clean clips with a kept event (FP), verdict accuracy over all clips,
    and kept events per flagged clip. The score is row[key]; a missing score is never kept."""
    viol = [c for c, i in labels.items() if i["has_violation"]]
    clean = [c for c, i in labels.items() if not i["has_violation"]]
    by_clip: dict[str, list[dict]] = {}
    for r in rows:
        if r.get(key) is not None:
            by_clip.setdefault(r["clip"], []).append(r)
    out = []
    for v in vs:
        ranked = {c: sorted((x for x in rs if x[key] >= v), key=lambda x: -x[key]) for c, rs in by_clip.items()}
        for m in ms:
            kept_by = {c: rs[:m] for c, rs in ranked.items() if rs}
            n_kept = sum(len(rs) for rs in kept_by.values())
            vr = sum(c in kept_by for c in viol)
            fp = sum(c in kept_by for c in clean)
            out.append({"v": v, "m": m, "verdict_recall": vr, "found": _found(kept_by, labels, viol), "fp": fp,
                        "acc": (vr + len(clean) - fp) / len(labels) if labels else 0.0,
                        "events": n_kept / len(kept_by) if kept_by else 0.0})
    return out


def clip_scores(rows: list[dict], key: str = "p_yes") -> dict[str, float]:
    """Each clip's score: the max of its windows' scores (rows without a score left out)."""
    out: dict[str, float] = {}
    for r in rows:
        x = r.get(key)
        if x is not None:
            out[r["clip"]] = max(out.get(r["clip"], -math.inf), x)
    return out


def clip_auc(rows: list[dict], labels: dict, key: str = "p_yes") -> float | None:
    """ROC-AUC of the clip max, violation clips against clean clips, over the clips with a scored window."""
    cs = clip_scores(rows, key)
    return auc([x for c, x in cs.items() if labels[c]["has_violation"]],
               [x for c, x in cs.items() if not labels[c]["has_violation"]])


def window_auc(rows: list[dict], labels: dict, key: str = "p_yes") -> float | None:
    """ROC-AUC of the window score, label-matched windows (+-2 s) against windows of clean clips."""
    got = [r for r in rows if r.get(key) is not None]
    return auc([r[key] for r in got if labels[r["clip"]]["has_violation"] and is_matched(r, labels)],
               [r[key] for r in got if not labels[r["clip"]]["has_violation"]])


def shuffle_null(rows: list[dict], fn, mode: str, reps: int = 2000, seed: int = 0, key: str = "p_yes"):
    """fn over copies of the scored rows with row[key] permuted, either among all rows ("global": the no-signal
    null) or among each clip's own rows ("within"). Returns (mean, standard deviation) over reps permutations,
    or {name: (mean, sd)} when fn returns a dict; None values (an AUC with an empty side) are skipped."""
    if mode not in ("global", "within"):
        raise ValueError(f"unknown shuffle {mode!r}")
    got = [dict(r) for r in rows if r.get(key) is not None]
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(got):
        groups.setdefault("" if mode == "global" else r["clip"], []).append(i)
    rng = random.Random(seed)
    vals: list = []
    for _ in range(reps):
        for idx in groups.values():
            xs = [got[i][key] for i in idx]
            rng.shuffle(xs)
            for i, x in zip(idx, xs, strict=True):
                got[i][key] = x
        vals.append(fn(got))

    def stat(xs):
        xs = [x for x in xs if x is not None]
        return (statistics.fmean(xs), statistics.pstdev(xs)) if xs else (None, None)

    if vals and isinstance(vals[0], dict):
        return {k: stat([v.get(k) for v in vals]) for k in vals[0]}
    return stat(vals)


def pick_verify(points: list[dict]) -> dict:
    """The eval sweep's choice (7C-2) among the m = all points: most found clips, then fewest FP clips, then
    the lowest v."""
    return max((p for p in points if p["m"] is None), key=lambda p: (p["found"], -p["fp"], -p["v"]))


def time_stats(xs: list[float]) -> dict:
    srt = sorted(x for x in xs if x is not None)
    if not srt:
        return {"n": 0, "mean": None, "p95": None}
    return {"n": len(srt), "mean": statistics.fmean(srt), "p95": srt[int(0.95 * (len(srt) - 1))]}


def summarize(rows: list[dict], labels: dict, key: str = "p_yes", sec_key: str = "verify_sec",
              reps: int = 2000, seed: int = 0) -> dict:
    """The decision-table metrics of one variant: window and clip AUC with their no-signal and within-clip
    nulls; the operating points at m = 1, 2 and all for the sweep's verify (pick_verify) and for the most
    accurate verify, each with the within-clip null of found and the no-signal null of found, FP and
    accuracy; the whole m = all curve; times. Over the clips in labels."""
    got = [r for r in rows if r.get(key) is not None]
    curve = operating_points(got, labels, VS, [None], key)
    chosen = pick_verify(curve)
    best = max(curve, key=lambda p: (p["acc"], p["found"], -p["v"]))
    vs = sorted({chosen["v"], best["v"]})

    def metrics(rs):
        out = {"window_auc": window_auc(rs, labels, key), "clip_auc": clip_auc(rs, labels, key)}
        for o in operating_points(rs, labels, vs, MS, key):
            for f in ("found", "fp", "acc"):
                out[f"{o['v']}|{o['m']}|{f}"] = o[f]
        return out

    within = shuffle_null(got, metrics, "within", reps, seed, key)
    nosig = shuffle_null(got, metrics, "global", reps, seed + 1, key)
    points = []
    for o in operating_points(got, labels, vs, MS, key):
        k = f"{o['v']}|{o['m']}|"
        points.append({**o, "found_null_within": within[k + "found"], "found_null_no_signal": nosig[k + "found"],
                       "fp_null_no_signal": nosig[k + "fp"], "acc_null_no_signal": nosig[k + "acc"]})
    types: dict[str, int] = {}
    for r in got:
        t = r.get("type_id") or "-"
        types[t] = types.get(t, 0) + 1
    zoomed = [r for r in rows if "zoom_sec" in r and r.get("zoom_sec") is not None]
    viol = sum(1 for i in labels.values() if i["has_violation"])
    return {
        "windows": len(rows), "scored": len(got), "clips": len(clip_scores(got, key)),
        "violation_clips": viol, "clean_clips": len(labels) - viol,
        "window_auc": window_auc(got, labels, key), "window_auc_null": {"no_signal": nosig["window_auc"],
                                                                        "within_clip": within["window_auc"]},
        "clip_auc": clip_auc(got, labels, key), "clip_auc_null": {"no_signal": nosig["clip_auc"],
                                                                  "within_clip": within["clip_auc"]},
        "verify": chosen["v"], "best_acc_verify": best["v"], "points": points, "curve": curve,
        "types": dict(sorted(types.items(), key=lambda kv: -kv[1])),
        "sec": time_stats([r.get(sec_key) for r in rows]),
        "zoom_sec": time_stats([r.get("zoom_sec") for r in zoomed]),
        "cropped": (sum(1 for r in zoomed if r.get("crop")) / len(zoomed)) if zoomed else None,
        "zoom_errors": sum(1 for r in zoomed if r.get("zoom_error")),
        "prompt_tokens": time_stats([r.get("prompt_tokens") for r in rows]),
        "reps": reps, "seed": seed,
    }


def _f3(x) -> str:
    return "-" if x is None else f"{x:.3f}"


def _null(t) -> str:
    return "-" if t is None or t[0] is None else f"{t[0]:.3f} ± {t[1]:.3f}"


def print_summary(name: str, sm: dict) -> None:
    print(f"\n### decision summary: {name}  ({sm['scored']}/{sm['windows']} windows scored, {sm['clips']} clips; "
          f"nulls: mean ± sd over {sm['reps']} permutations)")
    print(f"window AUC {_f3(sm['window_auc'])} | no-signal null {_null(sm['window_auc_null']['no_signal'])} | "
          f"within-clip null {_null(sm['window_auc_null']['within_clip'])}")
    print(f"clip AUC   {_f3(sm['clip_auc'])} | no-signal null {_null(sm['clip_auc_null']['no_signal'])} | "
          f"within-clip null {_null(sm['clip_auc_null']['within_clip'])} (the clip max cannot change)")
    print(f"verify (most found, then fewest FP, then lowest v; m = all): {sm['verify']:.2f} | most accurate "
          f"verify {sm['best_acc_verify']:.2f}")
    print("| v | m | found | found null within | found null no-signal | FP clips | FP null no-signal | verdict acc | "
          "acc null no-signal | verdict recall | events/flagged |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for o in sm["points"]:
        print(f"| {o['v']:.2f} | {o['m'] or 'all'} | {o['found']}/{sm['violation_clips']} | "
              f"{_null(o['found_null_within'])} | {_null(o['found_null_no_signal'])} | {o['fp']}/{sm['clean_clips']} | "
              f"{_null(o['fp_null_no_signal'])} | {o['acc']:.3f} | {_null(o['acc_null_no_signal'])} | "
              f"{o['verdict_recall']} | {o['events']:.2f} |")
    sec, zs, pt = sm["sec"], sm["zoom_sec"], sm["prompt_tokens"]
    print(f"seconds per window: mean {_f3(sec['mean'])}, p95 {_f3(sec['p95'])} (n {sec['n']}) | detector: mean "
          f"{_f3(zs['mean'])}, p95 {_f3(zs['p95'])} (n {zs['n']}) | cropped {_f3(sm['cropped'])} | detector errors "
          f"{sm['zoom_errors']} | prompt tokens mean {_f3(pt['mean'])}")
    print(f"type counts {sm['types']}")
    print("m = all curve: " + "; ".join(f"{o['v']:.2f}: {o['found']}/{o['fp']}/{o['acc']:.3f}" for o in sm["curve"])
          + "  (v: found/FP/acc)")


def _logit(p: float) -> float:
    from pipeline.logic import logit
    return logit(p)


def cmd_report(a) -> int:
    from eval.eval import load_labels

    labels = load_labels(a.labels)
    cache: dict[str, dict] = {}
    for c in a.cache:
        cache.update(load_cache(Path(c)))
    present = {d.get("prompt") for d in cache.values()}
    variants = a.variant or [v for v in (variant_name(p, z) for p in ("pipeline", "rule", "multi")
                                         for z in (False, True)) if v in present]
    summaries = {}
    want = None
    if a.subset:
        want = {(w["clip"], round(w["start"], 3), round(w["end"], 3), w["rule"])
                for w in json.loads(Path(a.subset).read_text())}
    for f in a.cands:
        cands = json.loads(Path(f).read_text())
        stem = Path(f).stem
        print(f"\n## {Path(f).name}: {cands['family']} {json.dumps(cands['params'])}")
        for prompt in variants:
            rows = join(cands, labels, cache, prompt)
            if want is not None:
                rows = [r for r in rows if (r["clip"], round(r["t_start"] - PAD, 3), round(r["t_end"] + PAD, 3),
                                            r["rule"]) in want]
            got = [r for r in rows if r["p_yes"] is not None]
            checked = sum(r["checked"] for r in rows)
            if not got:
                continue
            pos = [r["p_yes"] for r in got if r["matched"]]
            neg = [r["p_yes"] for r in got if not r["has_violation"]]
            other = [r["p_yes"] for r in got if r["has_violation"] and not r["matched"]]
            clips = sorted({r["clip"] for r in got})
            cmax = {c: max(r["p_yes"] for r in got if r["clip"] == c) for c in clips}
            cpos = [cmax[c] for c in clips if labels[c]["has_violation"]]
            cneg = [cmax[c] for c in clips if not labels[c]["has_violation"]]
            secs = [r["verify_sec"] for r in got if r["verify_sec"] is not None]

            def f3(x):
                return "-" if x is None else f"{x:.3f}"

            print(f"variant {prompt}: {len(got)}/{len(rows)} windows scored ({checked} checked) | candidate AUC "
                  f"matched vs clean {f3(auc(pos, neg))} ({len(pos)} vs {len(neg)}; unmatched windows on "
                  f"violation clips {len(other)}, mean p_yes {f3(sum(other) / len(other) if other else None)}) | "
                  f"clip AUC max p_yes {f3(auc(cpos, cneg))} ({len(cpos)} vs {len(cneg)}) | mean p_yes matched "
                  f"{f3(sum(pos) / len(pos) if pos else None)}, clean {f3(sum(neg) / len(neg) if neg else None)}"
                  f" | verify {sum(secs) / len(secs):.2f} s mean, {max(secs):.2f} s max")
            srt = sorted(secs)
            types = {}
            for r in got:
                types[r["type_id"] or "-"] = types.get(r["type_id"] or "-", 0) + 1
            print(f"  verify time median {srt[len(srt) // 2]:.2f} s, p95 {srt[int(0.95 * (len(srt) - 1))]:.2f} s | "
                  f"TYPE counts {dict(sorted(types.items(), key=lambda kv: -kv[1]))}")
            if want is None and checked == len(rows):  # tables only over complete coverage
                m_pos = [r for r in got if r["matched"]]
                m_neg = [r for r in got if not r["has_violation"]]
                med = {c: sorted(r["p_yes"] for r in got if r["clip"] == c) for c in clips}
                rel = {id(r): r["p_yes"] - med[r["clip"]][len(med[r["clip"]]) // 2] for r in got}
                extra = {"window grounding score": auc([r["score"] for r in m_pos], [r["score"] for r in m_neg]),
                         "p_yes - clip median": auc([rel[id(r)] for r in m_pos], [rel[id(r)] for r in m_neg]),
                         "matched vs unmatched windows on violation clips": auc(pos, other)}
                for alpha in (0.5, 1.0):
                    f = lambda r, al=alpha: _logit(r["p_yes"]) + al * r["score"]  # noqa: E731
                    extra[f"logit(p_yes) + {alpha} * score"] = auc([f(r) for r in m_pos], [f(r) for r in m_neg])
                print("  other window AUCs (matched vs clean unless named): "
                      + "; ".join(f"{k} {f3(x)}" for k, x in extra.items()))
                vs = [round(0.05 * i, 2) for i in range(1, 20)]
                print("| v | recall (clips) | found | FP clips |\n|---|---|---|---|")
                for v, rec, found, fp in keep_table(got, labels, vs):
                    print(f"| {v:.2f} | {rec:.3f} | {found} | {fp} |")
                print("| v | m | verdict recall | found | FP clips | verdict acc | events per flagged clip |\n"
                      "|---|---|---|---|---|---|---|")
                for o in operating_points(got, labels, [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6],
                                          [1, 2, 6]):
                    print(f"| {o['v']:.2f} | {o['m']} | {o['verdict_recall']} | {o['found']} | {o['fp']} | "
                          f"{o['acc']:.3f} | {o['events']:.2f} |")
                sm = summarize(rows, labels, reps=a.reps, seed=a.seed)
                name = f"{stem} {prompt}"
                print_summary(name, sm)
                summaries[name] = sm
        if want is not None:  # the same windows under both prompts
            both = [(x, y) for x, y in zip(join(cands, labels, cache, "pipeline"), join(cands, labels, cache, "rule"),
                                           strict=True)
                    if x["p_yes"] is not None and y["p_yes"] is not None]
            if len(both) > 2:
                xs, ys = [x["p_yes"] for x, _ in both], [y["p_yes"] for _, y in both]
                mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
                cov = sum((u - mx) * (w - my) for u, w in zip(xs, ys, strict=True))
                sd = math.sqrt(sum((u - mx) ** 2 for u in xs) * sum((w - my) ** 2 for w in ys))
                print(f"paired windows {len(both)}: Pearson r of p_yes (pipeline vs rule) {cov / sd:.3f}; TYPE not "
                      f"none: pipeline {sum(x['type_id'] not in ('none', '') for x, _ in both)}, rule "
                      f"{sum(y['type_id'] not in ('none', '') for _, y in both)}")
    if a.json:
        Path(a.json).write_text(json.dumps(summaries, indent=1))
        print(f"\nwrote {a.json}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    default_cache = str(ROOT / "runs" / "verify" / "cache.jsonl")
    for name in ("run", "report"):
        p = sub.add_parser(name)
        p.add_argument("--cands", action="append", required=True)
        p.add_argument("--labels", default=str(SVC / "data" / "labels.csv"))
        p.add_argument("--subset")
        if name == "run":
            p.add_argument("--cache", default=default_cache)
            p.add_argument("--clips", default=str(ROOT / "data" / "clips"))
            p.add_argument("--prompt", choices=["pipeline", "rule", "multi"], default="pipeline")
            p.add_argument("--zoom", action="store_true", help="crop each window to the people in it")
        else:
            p.add_argument("--cache", action="append", help=f"repeatable (default {default_cache})")
            p.add_argument("--variant", action="append", help="pipeline, rule or multi, optionally +zoom")
            p.add_argument("--json", help="write the decision summaries here")
            p.add_argument("--reps", type=int, default=2000, help="permutations per null")
            p.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "report" and not a.cache:
        a.cache = [default_cache]
    return cmd_run(a) if a.cmd == "run" else cmd_report(a)


if __name__ == "__main__":
    sys.exit(main())
