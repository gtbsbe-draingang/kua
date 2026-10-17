"""InternVideo3 check of candidate windows with the pipeline's own window cutting and verify call (Task 7C-1).

On the pod (needs the iv3 server and the clips; no VideoJev):

    $MTA_ROOT/env/main/bin/python tools/verify_candidates.py run --cands FILE [--cands FILE ...]
        [--prompt pipeline|rule] [--subset FILE] [--cache FILE] [--labels data/labels.csv] [--clips DIR]

  FILE is a cands_<name>.json from tools/analyze_candidates.py --export. For every clip and candidate the
  window [t_start - 2, t_end + 2] is cut from the clip exactly as real._run does (frames.normalize, then
  frames.cut_window with real.WINDOW_PAD_SEC; the pod's encoder is mpeg4) and checked with clients.verify:
    pipeline  the pipeline's prompt: every detectable rule, the candidate's rule first (real._verify_rules)
    rule      only the candidate's rule (its verify_prompt in the same template)
  --subset limits the run to the windows listed in a JSON list of {"clip", "start", "end", "rule"}.
  Results go to the cache (JSON lines, default $MTA_ROOT/runs/verify/cache.jsonl), keyed by clip, window,
  rule and prompt, so a window is checked once whichever generators share it, and a rerun resumes.
  Observations are text from the model; no frame is read back or written.

Anywhere (CPU):

    uv run python tools/verify_candidates.py report --cands FILE --cache FILE [--subset FILE]

  joins candidates and results and prints: per-candidate ROC-AUC of p_yes for candidates matching a label
  (timecode within +-2 s) against candidates on clean clips; per-clip AUC of max p_yes (violation vs clean);
  and the keep rule p_yes >= v (alone, and with a ground-score override) as recall over violation clips
  (every label matched by a kept candidate) and FP clips (clean clips with a kept candidate), per v.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SVC = Path(__file__).resolve().parent.parent
ROOT = Path(os.environ.get("MTA_ROOT", "/home/jovyan/model_train_amir"))
TOL = 2.0
EPS = 1e-6
PAD = 2.0  # real.WINDOW_PAD_SEC, checked in run


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


def cmd_run(a) -> int:
    sys.stdout.reconfigure(line_buffering=True)
    sys.path.insert(0, str(SVC))
    from eval.eval import load_labels, resolve_clips
    from pipeline import clients, frames, real

    if real.WINDOW_PAD_SEC != PAD:
        raise SystemExit(f"real.WINDOW_PAD_SEC is {real.WINDOW_PAD_SEC}, this tool assumes {PAD}")
    cfg = real._config()
    labels = load_labels(a.labels)
    files = resolve_clips(a.clips)
    cache_path = Path(a.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = load_cache(cache_path)
    want = None
    if a.subset:
        want = {(w["clip"], round(w["start"], 3), round(w["end"], 3), w["rule"])
                for w in json.loads(Path(a.subset).read_text())}
    todo: dict[str, list[tuple[float, float, str]]] = {}
    for f in a.cands:
        for r in candidate_windows(json.loads(Path(f).read_text()), labels):
            s, e = r["t_start"] - PAD, r["t_end"] + PAD
            if want is not None and (r["clip"], round(s, 3), round(e, 3), r["rule"]) not in want:
                continue
            if window_key(r["clip"], s, e, r["rule"], a.prompt) in cache:
                continue
            todo.setdefault(r["clip"], [])
            if (s, e, r["rule"]) not in todo[r["clip"]]:
                todo[r["clip"]].append((s, e, r["rule"]))
    n = sum(len(v) for v in todo.values())
    print(f"encoder {frames.video_encoder()} | prompt {a.prompt} | {n} windows to check on {len(todo)} clips "
          f"({len(cache)} cached)")
    done, t_all = 0, time.monotonic()
    with cache_path.open("a", encoding="utf-8") as sink, ThreadPoolExecutor(max_workers=2) as pool:
        for clip, wins in todo.items():
            work = tempfile.mkdtemp(prefix="verify-")
            try:
                src = frames.normalize(str(files[clip]), os.path.join(work, "norm.mp4"))

                def cut(i, _src=src, _work=work, _wins=wins):
                    s, e, _ = _wins[i]
                    t0 = time.monotonic()
                    path = frames.cut_window(_src, s, e, os.path.join(_work, f"w{i}.mp4"))
                    return path, time.monotonic() - t0

                futs = [pool.submit(cut, i) for i in range(len(wins))]  # cut ahead while iv3 works
                for i, (s, e, rule) in enumerate(wins):
                    try:
                        path, cut_sec = futs[i].result()
                    except frames.VideoError as err:
                        print(f"  {clip} [{s:.2f}, {e:.2f}] cut failed: {err}")
                        continue
                    rules = real._verify_rules(cfg, rule) if a.prompt == "pipeline" else [cfg.rules[rule]]
                    t0 = time.monotonic()
                    p, tid, obs = clients.verify(path, rules, 60.0)
                    sec = time.monotonic() - t0
                    rec = {"key": window_key(clip, s, e, rule, a.prompt), "clip": clip, "start": s, "end": e,
                           "rule": rule, "prompt": a.prompt, "p_yes": p, "type_id": tid, "observation": obs,
                           "verify_sec": round(sec, 3), "cut_sec": round(cut_sec, 3)}
                    sink.write(json.dumps(rec, ensure_ascii=False, allow_nan=False) + "\n")
                    sink.flush()
                    done += 1
                    ps = "None" if p is None else f"{p:.3f}"
                    print(f"[{done}/{n}] {clip} [{s:6.2f}, {e:6.2f}] {rule:<34} p_yes {ps} {tid or '-':<20} "
                          f"{sec:5.2f} s")
            finally:
                shutil.rmtree(work, ignore_errors=True)
    print(f"DONE {done} windows in {time.monotonic() - t_all:.0f} s")
    return 0


# ---- report -------------------------------------------------------------------------------------

def join(cands: dict, labels: dict, cache: dict, prompt: str) -> list[dict]:
    rows = []
    for r in candidate_windows(cands, labels):
        s, e = r["t_start"] - PAD, r["t_end"] + PAD
        hit = cache.get(window_key(r["clip"], s, e, r["rule"], prompt))
        rows.append({**r, "p_yes": None if hit is None else hit["p_yes"],
                     "type_id": None if hit is None else hit["type_id"],
                     "verify_sec": None if hit is None else hit["verify_sec"]})
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


def operating_points(rows: list[dict], labels: dict, vs: list[float], ms: list[int]) -> list[dict]:
    """Keep p_yes >= v, at most m events per clip (highest p_yes first). Per (v, m): violation clips with a
    kept event (verdict recall), violation clips with every label matched (found), clean clips with a kept
    event (FP), verdict accuracy over all clips, and kept events per flagged clip."""
    viol = [c for c, i in labels.items() if i["has_violation"]]
    clean = [c for c, i in labels.items() if not i["has_violation"]]
    by_clip: dict[str, list[dict]] = {}
    for r in rows:
        if r["p_yes"] is not None:
            by_clip.setdefault(r["clip"], []).append(r)
    out = []
    for v in vs:
        for m in ms:
            kept = [r for rs in by_clip.values()
                    for r in sorted((x for x in rs if x["p_yes"] >= v), key=lambda x: -x["p_yes"])[:m]]
            flagged = {r["clip"] for r in kept}
            vr = sum(c in flagged for c in viol)
            fp = sum(c in flagged for c in clean)
            found = sum(all(any(r["clip"] == c and abs(r["timecode"] - t["t_label"]) <= TOL + EPS for r in kept)
                            for t in labels[c]["truth"]) for c in viol)
            out.append({"v": v, "m": m, "verdict_recall": vr, "found": found, "fp": fp,
                        "acc": (vr + len(clean) - fp) / len(labels) if labels else 0.0,
                        "events": len(kept) / len(flagged) if flagged else 0.0})
    return out


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def cmd_report(a) -> int:
    sys.path.insert(0, str(SVC))
    sys.path.insert(0, str(SVC / "tools"))
    from analyze_candidates import auc
    from eval.eval import load_labels

    labels = load_labels(a.labels)
    cache = load_cache(Path(a.cache))
    want = None
    if a.subset:
        want = {(w["clip"], round(w["start"], 3), round(w["end"], 3), w["rule"])
                for w in json.loads(Path(a.subset).read_text())}
    for f in a.cands:
        cands = json.loads(Path(f).read_text())
        print(f"\n## {Path(f).name}: {cands['family']} {json.dumps(cands['params'])}")
        for prompt in ("pipeline", "rule"):
            rows = join(cands, labels, cache, prompt)
            if want is not None:
                rows = [r for r in rows if (r["clip"], round(r["t_start"] - PAD, 3), round(r["t_end"] + PAD, 3),
                                            r["rule"]) in want]
            got = [r for r in rows if r["p_yes"] is not None]
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

            print(f"prompt {prompt}: {len(got)}/{len(rows)} windows checked | candidate AUC matched vs clean "
                  f"{f3(auc(pos, neg))} ({len(pos)} vs {len(neg)}; unmatched windows on violation clips "
                  f"{len(other)}, mean p_yes {f3(sum(other) / len(other) if other else None)}) | clip AUC max "
                  f"p_yes {f3(auc(cpos, cneg))} ({len(cpos)} vs {len(cneg)}) | mean p_yes matched "
                  f"{f3(sum(pos) / len(pos) if pos else None)}, clean {f3(sum(neg) / len(neg) if neg else None)}"
                  f" | verify {sum(secs) / len(secs):.2f} s mean, {max(secs):.2f} s max")
            srt = sorted(secs)
            types = {}
            for r in got:
                types[r["type_id"] or "-"] = types.get(r["type_id"] or "-", 0) + 1
            print(f"  verify time median {srt[len(srt) // 2]:.2f} s, p95 {srt[int(0.95 * (len(srt) - 1))]:.2f} s | "
                  f"TYPE counts {dict(sorted(types.items(), key=lambda kv: -kv[1]))}")
            if want is None and len(got) == len(rows):  # tables only over complete coverage
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
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "report"):
        p = sub.add_parser(name)
        p.add_argument("--cands", action="append", required=True)
        p.add_argument("--labels", default=str(SVC / "data" / "labels.csv"))
        p.add_argument("--cache", default=str(ROOT / "runs" / "verify" / "cache.jsonl"))
        p.add_argument("--subset")
        if name == "run":
            p.add_argument("--clips", default=str(ROOT / "data" / "clips"))
            p.add_argument("--prompt", choices=["pipeline", "rule"], default="pipeline")
    a = ap.parse_args(argv)
    return cmd_run(a) if a.cmd == "run" else cmd_report(a)


if __name__ == "__main__":
    sys.exit(main())
