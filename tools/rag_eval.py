"""The regulation matcher's verdict on the windows of a multi-mode verify run (Task 3 of the zoom plan).

On the pod (the RAG server on 127.0.0.1:30001; env/main):

    $MTA_ROOT/env/main/bin/python tools/rag_eval.py dump --run RUN --out DUMP.jsonl
        RUN is a verify_candidates.py cache (a folder holding cache.jsonl, or the file) of a multi or multi+zoom
        run. One record per window: clip, window, timecode, the observation, p_rules, the judge candidates the
        pipeline builds in multi mode (logic.judge_candidates over the known p_rules with the argmax rule, as
        real._judge_candidate dicts) and the server's /retrieve hits (top 8, one text per call, none for an
        empty observation: exactly what /v1/chat/completions searches). Text and numbers only.
    $MTA_ROOT/env/main/bin/python tools/rag_eval.py run --backend iv3 --dump DUMP.jsonl --out OUT.jsonl
        each record's clients.judge_request (no frames) to the RAG server through clients.judge_answer, as the
        pipeline sends it; records the answer, the HTTP status and the seconds. Serial, resumable.
    $MTA_ROOT/env/main/bin/python tools/rag_eval.py recall [--queries tests/data/rag_queries.json] [--ks 8,12]
        retrieval recall@k and MRR of the expected clause through /retrieve (a query's "expect" is one clause id or a
        list of them; the first expected clause found counts). The ranking is read to --depth (20, the server's
        cap on k), so MRR counts a clause beyond that as missed.
    $MTA_ROOT/env/main/bin/python tools/rag_eval.py recall --local [--modes dense,lexical,hybrid]
        the same without the server: builds the index in this process (bge-m3 on the pod's GPU or CPU, the
        running server untouched) and scores each mode on one dense matrix: dense (bge-m3 only), lexical (BM25 with
        the glossary only) and hybrid (reciprocal rank fusion of the two). Refuses off the pod.

On the Mac only (measurement stand-in, user decision 2026-10-17; the key comes from ANTHROPIC_API_KEY):

    uv run --group cloud-dev python tools/rag_eval.py run --backend sonnet --dump DUMP.jsonl --out OUT.jsonl
        the same prompt the RAG server builds (matcher.judge_prompt over the dump's hits and the local corpus)
        to claude-sonnet-5-5 with the answer schema as structured output (minimum/maximum removed: the API
        rejects them; parse_answer clips to 0-1). 8 requests at a time, resumable; records the answer, the
        seconds, the token usage and the model that answered. It refuses to start (exit 2, nothing sent) when
        /home/jovyan exists or MTA_ROOT is set, and imports anthropic only here.

Anywhere:

    uv run python tools/rag_eval.py report --out OUT.jsonl [--labels data/labels.csv] [--json FILE]
        window score = the answer's violation, clip score = the max over its windows; the same AUCs, nulls and
        operating points as verify_candidates.py report (verify_candidates.summarize), plus the chosen rule and
        clause counts, and token usage and cost for the sonnet backend.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SVC = Path(__file__).resolve().parent.parent
for _p in (SVC, SVC / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import verify_candidates as vc  # noqa: E402  (one implementation of the AUCs, nulls and operating points)

POD_HOME = Path("/home/jovyan")
RAG_URL = "http://127.0.0.1:30001"
TOP_K = 8  # rag.retrieve.TOP_K, what the server's chat endpoint searches
RECALL_KS = (8, 12)
RECALL_DEPTH = 20  # the largest k the server's /retrieve accepts
RECALL_MODES = ("dense", "lexical", "hybrid")
SONNET = "claude-sonnet-5-5"
# $ per million tokens: input, output, cache read, cache write (5 min)
PRICES = {SONNET: (2.00, 10.00, 0.20, 2.50)}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _done_keys(path: Path) -> set[str]:
    return {d["key"] for d in _read_jsonl(path)}


# ---- dump ---------------------------------------------------------------------------------------

def dump_record(w: dict, cfg, hits: list[dict]) -> dict:
    """A multi-mode cache record plus the judge candidates the pipeline builds from it and the retrieval hits.
    Candidates: logic.judge_candidates over the known p_rules with the argmax rule (the first rule, in catalogue
    order, with the highest P(YES)), each as real._judge_candidate of its rule."""
    from pipeline import logic, real

    known = {k: p for k, p in (w.get("p_rules") or {}).items() if p is not None}
    argmax = max(known, key=known.get) if known else None
    ids = logic.judge_candidates(known, argmax or "")
    return {"key": w["key"], "clip": w["clip"], "start": w["start"], "end": w["end"], "rule": w["rule"],
            "timecode": w["timecode"], "variant": w.get("prompt"), "observation": w.get("observation") or "",
            "p_rules": w.get("p_rules"), "p_yes": w.get("p_yes"), "argmax": argmax, "candidate_ids": ids,
            "candidates": [real._judge_candidate(cfg.rules[i]) for i in ids], "hits": hits}


def _post(url: str, body: dict, timeout: float):
    import httpx
    # trust_env=False: a local service, and a proxy from the environment must not reroute it
    return httpx.post(url, json=body, timeout=timeout, trust_env=False)


def retrieve(url: str, text: str, k: int = TOP_K) -> list[dict]:
    """The server's top k for one text ([] for an empty one, as the chat endpoint skips the search)."""
    if not text.strip():
        return []
    r = _post(url.rstrip("/") + "/retrieve", {"texts": [text], "k": k}, 60.0)
    r.raise_for_status()
    return r.json()["hits"][0]


def cmd_dump(a) -> int:
    from pipeline import real

    cfg = real._config()
    run = Path(a.run)
    cache = run / "cache.jsonl" if run.is_dir() else run
    windows = [d for d in vc.load_cache(cache).values() if str(d.get("prompt", "")).startswith("multi")]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _done_keys(out)
    todo = [w for w in windows if w["key"] not in done]
    print(f"{len(windows)} multi windows in {cache}, {len(done)} already dumped, {len(todo)} to go")
    with out.open("a", encoding="utf-8") as sink:
        for i, w in enumerate(todo, 1):
            # the server searches the observation judge_fields gives back: stripped
            rec = dump_record(w, cfg, retrieve(a.url, (w.get("observation") or "").strip()))
            sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
            sink.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"[{i}/{len(todo)}] {w['clip']} {rec['candidate_ids']} {len(rec['hits'])} hits")
    return 0


# ---- run: iv3 (the RAG server on the pod) ---------------------------------------------------------

def run_iv3(dump: Path, out: Path, url: str, timeout: float) -> int:
    from pipeline import clients

    os.environ["JUDGE_URL"] = url
    recs = [r for r in _read_jsonl(dump) if r["key"] not in _done_keys(out)]
    print(f"{len(recs)} judge requests to {url}")
    with out.open("a", encoding="utf-8") as sink, vc.ReplySpy(clients) as spy:
        for i, r in enumerate(recs, 1):
            t0 = time.monotonic()
            ans = clients.judge_answer(r["observation"], [], r["candidates"], timeout)
            sec = time.monotonic() - t0
            reply = spy.take()
            rec = {"key": r["key"], "clip": r["clip"], "timecode": r["timecode"], "backend": "iv3",
                   "answer": ans, "seconds": round(sec, 3), "http_status": reply.get("http_status", 200),
                   "error": reply.get("error")}
            if ans is not None and ans["rule_id"] not in r["candidate_ids"]:
                rec["answer"], rec["error"] = None, f"rule {ans['rule_id']!r} is not a candidate"
            sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
            sink.flush()
            a = rec["answer"]
            print(f"[{i}/{len(recs)}] {r['clip']} {r['timecode']:6.2f} | "
                  + (f"violation {a['violation']} rule {a['rule_id']} clause {a['clause_id']}" if a
                     else f"FAILED {rec['http_status']} {rec['error'] or ''}") + f" | {sec:.2f} s")
    return 0


# ---- run: sonnet (the Mac only) -----------------------------------------------------------------

def on_pod() -> bool:
    return POD_HOME.exists() or bool(os.environ.get("MTA_ROOT"))


def strip_bounds(schema):
    """The schema without "minimum"/"maximum" anywhere (the Claude API's structured output rejects them)."""
    if isinstance(schema, dict):
        return {k: strip_bounds(v) for k, v in schema.items() if k not in ("minimum", "maximum")}
    if isinstance(schema, list):
        return [strip_bounds(v) for v in schema]
    return schema


def sonnet_prompt(r: dict, index) -> tuple[str, str, list[str], list[str]]:
    """(system, user, rule ids, clause ids) exactly as the RAG server would build them for this record."""
    from rag import matcher
    from rag.retrieve import Hit

    hits = [Hit(h["clause_id"], float(h["score"]), int(h["part"])) for h in r["hits"]]
    system, user, clauses = matcher.judge_prompt(r["observation"].strip(), r["candidates"], hits, index)
    return system, user, [str(c["id"]) for c in r["candidates"]], [c.id for c, _ in clauses]


def run_sonnet(dump: Path, out: Path, concurrency: int) -> int:
    if on_pod():
        print("rag_eval: the sonnet backend refuses to run on the pod (/home/jovyan exists or MTA_ROOT is set); "
              "it is a Mac-only measurement stand-in", file=sys.stderr)
        return 2
    import anthropic  # the cloud-dev group, never in the pod envs

    from rag import matcher
    from rag.corpus import load_clauses

    clauses = load_clauses()
    index = types.SimpleNamespace(clauses=clauses, by_id={c.id: c for c in clauses})
    recs = [r for r in _read_jsonl(dump) if r["key"] not in _done_keys(out)]
    print(f"{len(recs)} requests to {SONNET}, {concurrency} at a time")
    client = anthropic.Anthropic()

    def one(r: dict) -> dict:
        system, user, rule_ids, clause_ids = sonnet_prompt(r, index)
        schema = strip_bounds(matcher.answer_schema(rule_ids, clause_ids))
        rec = {"key": r["key"], "clip": r["clip"], "timecode": r["timecode"], "backend": "sonnet", "answer": None,
               "prompt_chars": len(user), "clauses_offered": len(clause_ids)}
        t0 = time.monotonic()
        try:
            resp = client.beta.messages.create(
                model=SONNET, max_tokens=4000, system=system, messages=[{"role": "user", "content": user}],
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
                betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        except anthropic.APIStatusError as e:
            rec.update(seconds=round(time.monotonic() - t0, 3), error=f"{type(e).__name__} {e.status_code}")
            return rec
        except Exception as e:  # noqa: BLE001 - a connection or SDK error fails this window only
            rec.update(seconds=round(time.monotonic() - t0, 3), error=type(e).__name__)
            return rec
        rec["seconds"] = round(time.monotonic() - t0, 3)
        u = resp.usage
        rec["usage"] = {k: getattr(u, k, None) or 0 for k in ("input_tokens", "output_tokens",
                                                              "cache_read_input_tokens", "cache_creation_input_tokens")}
        rec["model"], rec["stop_reason"] = resp.model, resp.stop_reason
        if resp.stop_reason == "refusal":
            det = getattr(resp, "stop_details", None)
            rec["error"] = f"refusal {getattr(det, 'category', None)}"
            return rec
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        rec["answer"] = matcher.parse_answer(text, rule_ids, clause_ids)
        if rec["answer"] is None:
            rec["error"], rec["text"] = "bad answer", text[:500]
        return rec

    with out.open("a", encoding="utf-8") as sink, ThreadPoolExecutor(max_workers=concurrency) as pool:
        for i, rec in enumerate(pool.map(one, recs), 1):
            sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
            sink.flush()
            a = rec["answer"]
            print(f"[{i}/{len(recs)}] {rec['clip']} {rec['timecode']:6.2f} | "
                  + (f"violation {a['violation']} rule {a['rule_id']} clause {a['clause_id']}" if a
                     else f"FAILED {rec.get('error')}") + f" | {rec.get('seconds')} s")
    return 0


def cmd_run(a) -> int:
    dump, out = Path(a.dump), Path(a.out)
    if a.backend == "sonnet":
        return run_sonnet(dump, out, a.concurrency)
    out.parent.mkdir(parents=True, exist_ok=True)
    return run_iv3(dump, out, a.url, a.timeout)


# ---- recall ---------------------------------------------------------------------------------------

def expected_ids(q: dict) -> list[str]:
    """A query's expected clause ids: "expect" is one id or a list of them."""
    e = q["expect"]
    return [e] if isinstance(e, str) else list(e)


def first_rank(ids: list[str], expect: list[str]) -> int | None:
    """The 1-based rank of the first expected clause in ids, or None."""
    return next((i for i, cid in enumerate(ids, 1) if cid in expect), None)


def summarize_ranks(ranks: list[int | None], ks=RECALL_KS) -> dict:
    """recall@k (the share of queries whose expected clause ranks within k), their counts and the mean reciprocal
    rank (0 for a miss)."""
    n = len(ranks)
    out: dict = {"n": n, "mrr": sum(1 / r for r in ranks if r) / n if n else 0.0}
    for k in ks:
        hit = sum(1 for r in ranks if r is not None and r <= k)
        out[f"hit@{k}"], out[f"recall@{k}"] = hit, hit / n if n else 0.0
    return out


def report_recall(queries: list[dict], rankers: dict, ks=RECALL_KS, echo=print) -> dict:
    """Score every ranker (name -> text -> clause ids, best first) on the queries; one line per query with the rank
    each ranker gave the expected clause, then one summary line per ranker. Returns {name: summary}."""
    ranks: dict[str, list[int | None]] = {name: [] for name in rankers}
    for q in queries:
        expect = expected_ids(q)
        cells = []
        for name, rank in rankers.items():
            r = first_rank(rank(q["text"]), expect)
            ranks[name].append(r)
            cells.append(f"{name} {r or '-':>2}")
        echo(f"{'/'.join(expect)[:34]:<34} " + " | ".join(cells))
    out = {name: summarize_ranks(rs, ks) for name, rs in ranks.items()}
    for name, sm in out.items():
        cols = " | ".join(f"recall@{k} {sm[f'hit@{k}']}/{sm['n']} = {sm[f'recall@{k}']:.3f}" for k in ks)
        echo(f"{name:<8} {cols} | MRR {sm['mrr']:.3f}")
    return out


def mode_rankers(index, modes, depth: int) -> dict:
    """text -> clause ids for each mode on one rag.retrieve.Index: dense and hybrid flip the index's switch around
    the call, lexical reads the BM25 ranking."""
    def ranker(mode: str):
        def rank(text: str) -> list[str]:
            if mode == "lexical":
                return [h.clause_id for h in index.lexical_hits(text)[:depth]]
            was, index.hybrid = index.hybrid, mode == "hybrid"
            try:
                return [h.clause_id for h in index.search(text, depth)]
            finally:
                index.hybrid = was
        return rank
    return {m: ranker(m) for m in modes}


def local_index():
    """The corpus indexed in this process with the real encoder, lexical channel included (the pod's env/main)."""
    from rag.corpus import load_clauses
    from rag.retrieve import Index

    return Index(load_clauses(), hybrid=True)


def cmd_recall(a) -> int:
    queries = json.loads(Path(a.queries).read_text(encoding="utf-8"))
    ks = tuple(int(k) for k in a.ks.split(","))
    depth = max(a.depth, *ks)
    if a.local:
        if not on_pod():
            print("rag_eval: recall --local loads bge-m3 and refuses to run off the pod (no /home/jovyan, no MTA_ROOT)",
                  file=sys.stderr)
            return 2
        modes = [m.strip() for m in a.modes.split(",") if m.strip()]
        bad = [m for m in modes if m not in RECALL_MODES]
        if bad:
            print(f"rag_eval: unknown mode {bad}; choose from {list(RECALL_MODES)}", file=sys.stderr)
            return 2
        index = local_index()
        if not index.hybrid and any(m != "dense" for m in modes):
            print("rag_eval: the lexical channel did not build (see the log above)", file=sys.stderr)
            return 2
        rankers = mode_rankers(index, modes, depth)
    else:
        rankers = {"server": lambda text: [h["clause_id"] for h in retrieve(a.url, text, depth)]}
    report_recall(queries, rankers, ks)
    return 0


# ---- report ---------------------------------------------------------------------------------------

def rows(out: list[dict], labels: dict) -> list[dict]:
    """One row per answered or failed window of a labelled clip: score = the answer's violation (None when the
    matcher failed or gave none), type_id = the chosen rule."""
    res = []
    for d in out:
        if d["clip"] not in labels:
            continue
        a = d.get("answer") or {}
        r = {"clip": d["clip"], "timecode": d["timecode"], "violation": a.get("violation"),
             "type_id": a.get("rule_id"), "seconds": d.get("seconds"),
             "has_violation": labels[d["clip"]]["has_violation"]}
        r["matched"] = vc.is_matched(r, labels)
        res.append(r)
    return res


def cost(out: list[dict]) -> dict:
    by_model: dict[str, dict] = {}
    for d in out:
        if not d.get("usage"):
            continue
        m = by_model.setdefault(d.get("model") or "?", {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                                                       "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0})
        m["calls"] += 1
        for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            m[k] += d["usage"].get(k) or 0
    total = 0.0
    for name, m in by_model.items():
        p = PRICES.get(name)
        m["usd"] = None if p is None else (m["input_tokens"] * p[0] + m["output_tokens"] * p[1]
                                           + m["cache_read_input_tokens"] * p[2]
                                           + m["cache_creation_input_tokens"] * p[3]) / 1e6
        total += m["usd"] or 0.0
    return {"by_model": by_model, "usd": total}


def summarize(out: list[dict], labels: dict, reps: int = 2000, seed: int = 0) -> dict:
    rs = rows(out, labels)
    sm = vc.summarize(rs, labels, key="violation", sec_key="seconds", reps=reps, seed=seed)
    clauses: dict[str, int] = {}
    for d in out:
        a = d.get("answer")
        if a is not None and d["clip"] in labels:
            c = a.get("clause_id") or "none"
            clauses[c] = clauses.get(c, 0) + 1
    sm["failed"] = sum(1 for d in out if d.get("answer") is None and d["clip"] in labels)
    sm["clauses"] = dict(sorted(clauses.items(), key=lambda kv: -kv[1]))
    sm["cost"] = cost(out)
    return sm


def cmd_report(a) -> int:
    from eval.eval import load_labels

    labels = load_labels(a.labels)
    summaries = {}
    for f in a.out:
        out = _read_jsonl(Path(f))
        sm = summarize(out, labels, a.reps, a.seed)
        backend = next((d.get("backend") for d in out if d.get("backend")), "?")
        name = f"rag/{backend} ({Path(f).name})"
        vc.print_summary(name, sm)
        print(f"failed windows {sm['failed']} | chosen clauses {sm['clauses']}")
        for m, c in sm["cost"]["by_model"].items():
            usd = "-" if c["usd"] is None else f"${c['usd']:.2f}"
            print(f"tokens {m}: {c['calls']} calls, input {c['input_tokens']}, output {c['output_tokens']}, cache read "
                  f"{c['cache_read_input_tokens']}, cache write {c['cache_creation_input_tokens']} | {usd}")
        summaries[name] = sm
    if a.json:
        Path(a.json).write_text(json.dumps(summaries, indent=1))
        print(f"\nwrote {a.json}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("dump")
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--url", default=RAG_URL)
    p = sub.add_parser("run")
    p.add_argument("--backend", choices=["iv3", "sonnet"], required=True)
    p.add_argument("--dump", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--url", default=RAG_URL)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--concurrency", type=int, default=8)
    p = sub.add_parser("recall")
    p.add_argument("--queries", default=str(SVC / "tests" / "data" / "rag_queries.json"))
    p.add_argument("--url", default=RAG_URL)
    p.add_argument("--ks", default=",".join(str(k) for k in RECALL_KS))
    p.add_argument("--depth", type=int, default=RECALL_DEPTH)
    p.add_argument("--local", action="store_true")
    p.add_argument("--modes", default=",".join(RECALL_MODES))
    p = sub.add_parser("report")
    p.add_argument("--out", action="append", required=True)
    p.add_argument("--labels", default=str(SVC / "data" / "labels.csv"))
    p.add_argument("--json")
    p.add_argument("--reps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    return {"dump": cmd_dump, "run": cmd_run, "recall": cmd_recall, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
