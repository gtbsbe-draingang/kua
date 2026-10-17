"""Search the regulation index. This is the one function other code should call.

    from regs.search import search
    hits = search("кассир отходит от рабочего места во время обслуживания клиента", k=5)

Each hit carries the verbatim clause text and the citation data (act, clause label, adilet URL
with anchor), so a caller can quote a clause without any model writing the citation.
"""
import functools
import re
import time

import yaml

from . import config, ltr, rerank as reranker, store
from .chunk import passage
from .embed import get_embedder
from .lexical import normalize, to_lex

MODES = ("auto", "ltr", "hybrid", "dense", "bm25")
GLOSSARY_PATH = config.PKG_DIR / "glossary.yaml"


@functools.lru_cache(maxsize=1)
def _glossary() -> list[tuple[re.Pattern, str]]:
    with open(GLOSSARY_PATH, encoding="utf-8") as f:
        terms = yaml.safe_load(f)["terms"]
    return [(re.compile(r"(?<![a-zа-я])(?:" + "|".join(re.escape(normalize(w)) for w in t["when"]) + ")"),
             t["add"]) for t in terms]


def expand(query: str) -> list[str]:
    """Legal wording to add to the BM25 query for the everyday words in `query`."""
    q = normalize(query)
    return [add for pattern, add in _glossary() if pattern.search(q)]


def _fold(hits: list[dict]) -> list[dict]:
    """Keep one result per clause: the best-ranked of its chunks (whole clause, a part, or one
    subclause). The other matched labels are listed in `also_matched`."""
    out, seen = [], {}
    for h in hits:
        key = (h["doc_code"], h["clause"])
        if key in seen:
            seen[key]["also_matched"].append(h["clause_label"])
            continue
        h["also_matched"] = []
        seen[key] = h
        out.append(h)
    return out


def _clause_ranks(hits: list[dict]) -> dict[tuple[str, str], tuple[int, float]]:
    """(act, clause) -> (rank of the clause's best chunk among clauses, that chunk's score)."""
    best: dict[tuple[str, str], tuple[int, float]] = {}
    for h in hits:
        key = (h["doc_code"], h["clause"])
        if key not in best:
            best[key] = (len(best) + 1, round(h["score"], 4))
    return best


def _blend(cands: list[dict], weight: float) -> None:
    """score = weight * rerank_score + (1 - weight) * retrieval score min-max scaled over the pool."""
    lo = min(h["retrieval_score"] for h in cands)
    hi = max(h["retrieval_score"] for h in cands)
    for h in cands:
        retrieval = (h["retrieval_score"] - lo) / (hi - lo) if hi > lo else 1.0
        h["score"] = weight * h["rerank_score"] + (1 - weight) * retrieval


def search(query: str, k: int = 8, mode: str = "hybrid", docs: list[str] | None = None,
           include_repealed: bool = False, explain: bool = False, use_glossary: bool = True,
           rerank: bool = True, rerank_weight: float | None = None, debug: bool = False) -> dict:
    """Retrieve a candidate pool from Milvus, rerank it (if a reranker is configured), and
    return the top k clauses. With reranking, `score` blends the reranker's 0..1 score with the
    scaled retrieval score (config.RERANK_WEIGHT); without it, `score` is the retrieval score
    (RRF / cosine / BM25). debug=True also returns every reranked candidate chunk."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    trained = ltr.load()
    if mode == "auto":
        mode = "ltr" if trained else "hybrid"
    if mode == "ltr" and not trained:
        raise RuntimeError("no LambdaMART model; run `python -m regs train-ltr`")
    rerank = rerank and bool(config.RERANK_MODEL)
    t0 = time.time()
    embedder = get_embedder()  # loaded once per process
    if rerank:
        reranker.get_reranker()
    flt = store.make_filter(docs, include_repealed)
    added = expand(query) if use_glossary else []
    lex = to_lex(" ".join([query] + added))
    vector = embedder.query(query) if mode in ("ltr", "hybrid", "dense") or explain else None
    t_embed = time.time() - t0

    # Several chunks of one clause can come back, so fetch a pool, fold, then cut to k. The pool
    # does not grow with small k, so the top results (and RRF fusion) do not depend on k.
    pool = max(k * 4, 40, config.RERANK_CANDIDATES if rerank else 0)
    if mode == "ltr":
        model, meta = trained
        cands = ltr.candidates(vector, lex, flt, meta["per_list"])
        scores = model.predict(ltr.features(cands)) if cands else []
        for h, s in zip(cands, scores):
            h["score"] = float(s)
        raw = sorted(cands, key=lambda h: -h["score"])
    elif mode == "dense":
        raw = store.search_dense(vector, pool, flt)
    elif mode == "bm25":
        raw = store.search_bm25(lex, pool, flt)
    else:
        raw = store.search_hybrid(vector, lex, pool, flt)
    retrieval = _clause_ranks(raw)

    t_rerank = 0.0
    candidate_clauses: list[str] = []
    debug_cands: list[dict] = []
    if rerank and raw:
        t1 = time.time()
        cands = raw[:config.RERANK_CANDIDATES]
        candidate_clauses = [f"{a}:{c}" for a, c in _clause_ranks(cands)]
        scores = reranker.score(query, [passage(h["doc_short"], h["chapter"], h["section"],
                                                h["clause_label"], h["text"]) for h in cands])
        for h, s in zip(cands, scores):
            h["retrieval_score"], h["rerank_score"] = h["score"], float(s)
        _blend(cands, config.RERANK_WEIGHT if rerank_weight is None else rerank_weight)
        if debug:
            debug_cands = [{"key": f"{h['doc_code']}:{h['clause']}", "id": h["id"],
                            "retrieval_score": h["retrieval_score"], "rerank_score": h["rerank_score"]}
                           for h in cands]
        # The clause's result is its best-scoring chunk: whole clause, part, or one subclause.
        hits = _fold(sorted(cands, key=lambda h: -h["score"]))[:k]
        t_rerank = time.time() - t1
    else:
        hits = _fold(raw)[:k]
    for h in hits:
        h["retrieval_rank"] = retrieval[(h["doc_code"], h["clause"])][0]
        h["reranked"] = rerank

    if explain:  # the clause's best rank in the dense list and in the BM25 list on their own
        dense = _clause_ranks(store.search_dense(vector, pool, flt))
        bm25 = _clause_ranks(store.search_bm25(lex, pool, flt))
        for h in hits:
            key = (h["doc_code"], h["clause"])
            h["dense_rank"], h["dense_score"] = dense.get(key, (None, None))
            h["bm25_rank"], h["bm25_score"] = bm25.get(key, (None, None))

    return {"query": query, "mode": mode, "k": k, "filter": flt, "glossary_added": added,
            "lex_query": lex, "reranker": config.RERANK_MODEL if rerank else None,
            "rerank_candidates": len(candidate_clauses), "candidate_clauses": candidate_clauses,
            "rerank_weight": (config.RERANK_WEIGHT if rerank_weight is None else rerank_weight) if rerank else None,
            "embed_ms": round(t_embed * 1000), "rerank_ms": round(t_rerank * 1000),
            "took_ms": round((time.time() - t0) * 1000), "hits": hits,
            **({"candidates": debug_cands} if debug else {})}


def citation(hit: dict) -> dict:
    """The rules.yaml `citations` entry for a hit (see design.md, section 2). edition_checked is
    left empty: a person fills it in after comparing the clause with the current adilet page."""
    meta = {d["code"]: d for d in store.index_meta().get("documents", [])}
    act = meta.get(hit["doc_code"], {})
    document = act.get("title", hit["doc_short"])
    if act.get("act"):
        document += f" ({act['act']})"
    return {"document": document, "clause": hit["clause_label"], "quote": hit["text"],
            "url": hit["url"], "edition_checked": ""}


class _BlockDumper(yaml.SafeDumper):
    pass


_BlockDumper.add_representer(str, lambda d, s: d.represent_scalar(
    "tag:yaml.org,2002:str", s, style="|" if "\n" in s else None))


def citation_yaml(hit: dict) -> str:
    """citation() as YAML, ready to paste under a rule's `citations:` key."""
    meta = {d["code"]: d for d in store.index_meta().get("documents", [])}
    edition = meta.get(hit["doc_code"], {}).get("latest_amendment") or "unknown"
    note = (f"# Indexed text includes amendments up to {edition}. Open the url, check that the clause\n"
            f"# reads the same in the current edition, then set edition_checked to that date.\n")
    return note + yaml.dump([citation(hit)], Dumper=_BlockDumper, allow_unicode=True, sort_keys=False, width=100)


# --- quality checks -------------------------------------------------------------------------

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ").replace("ё", "е")).strip().lower()


def verify_quotes(rules_path: str) -> int:
    """Check that every citation quote in rules.yaml appears word for word in the indexed corpus.
    Returns the number of quotes that were not found."""
    from .chunk import load_chunks
    corpus = {}
    for ch in load_chunks():
        corpus.setdefault(ch["doc_code"], []).append(_norm(ch["text"]))
    with open(rules_path, encoding="utf-8") as f:
        rules = yaml.safe_load(f).get("rules", [])
    missing = 0
    for rule in rules:
        for cit in rule.get("citations") or []:
            quote = _norm(cit.get("quote") or "")
            code = next((c for c in corpus if c in (cit.get("url") or "")), None)
            texts = corpus.get(code) or [t for ts in corpus.values() for t in ts]
            ok = bool(quote) and any(quote in t for t in texts)
            missing += not ok
            print(f"{'ok  ' if ok else 'MISS'} {rule.get('id')}: {cit.get('clause') or '(no clause)'} {code or '?'}")
    print(f"{missing} quote(s) not found in the corpus")
    return missing


def evaluate(k: int = 5, runs: list[str] | None = None) -> dict:
    """Retrieval quality on eval_queries.yaml, separately for the dev and the holdout set.
    A hit counts if its act and clause number match one of `expect`. For reranked runs `cand`
    is the share of queries whose clause was in the reranker's candidate pool at all, which is
    the ceiling for what reranking can reach."""
    with open(config.EVAL_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    sets = {"dev": data["queries"], "holdout": data.get("holdout") or []}
    all_runs = {  # name -> (mode, glossary, rerank); LambdaMART was trained with the glossary on
        "ltr+glossary": ("ltr", True, False),
        "ltr+glossary+rerank": ("ltr", True, True),
        "hybrid+glossary+rerank": ("hybrid", True, True),
        "hybrid+glossary": ("hybrid", True, False),
        "hybrid": ("hybrid", False, False),
        "dense": ("dense", False, False),
        "bm25+glossary": ("bm25", True, False),
    }
    names = runs or [n for n in all_runs if (config.RERANK_MODEL or "rerank" not in n)
                     and (ltr.load() or not n.startswith("ltr"))]
    if ltr.load():
        print("LambdaMART was trained on the dev queries: its dev rows are in-sample; "
              f"out-of-fold dev = {ltr.load()[1]['cv_dev']}. Judge it on holdout.")
    if config.RERANK_MODEL:
        print(f"reranker: {config.RERANK_MODEL}, {config.RERANK_CANDIDATES} candidate chunks")
    report: dict = {}
    for name in names:
        mode, gloss, rr = all_runs[name]
        for set_name, cases in sets.items():
            if not cases:
                continue
            hits_at = {1: 0, 3: 0, k: 0}
            rr_sum, cand, misses, ms = 0.0, 0, [], 0
            for case in cases:
                expected = {tuple(e.split(":")) for e in case["expect"]}
                res = search(case["q"], k=k, mode=mode, use_glossary=gloss, rerank=rr)
                ms += res["took_ms"]
                got = res["hits"]
                rank = next((i + 1 for i, h in enumerate(got) if (h["doc_code"], h["clause"]) in expected), None)
                for n in hits_at:
                    hits_at[n] += bool(rank and rank <= n)
                rr_sum += 1 / rank if rank else 0
                cand += any(tuple(c.split(":")) in expected for c in res["candidate_clauses"])
                if rank != 1:
                    top = f"{got[0]['doc_code']}:{got[0]['clause']}" if got else "-"
                    misses.append(f"{case['q']}  (rank {rank or '-'}, top-1 {top})")
            n = len(cases)
            row = {**{f"hit@{m}": round(v / n, 3) for m, v in hits_at.items()}, "mrr": round(rr_sum / n, 3)}
            if rr:
                row["cand"] = round(cand / n, 3)
            row["ms"] = round(ms / n)
            report[f"{name}/{set_name}"] = {**row, "not_top1": misses}
            print(f"{name:24s} {set_name:7s} n={n:2d} " + " ".join(f"{key}={val}" for key, val in row.items()))
            for m in misses:
                print(f"{'':33s} not top-1: {m}")
    return report


def sweep(weights: tuple[float, ...] = (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0), k: int = 5) -> dict:
    """Score every eval query with the reranker once, then compute hit@1/hit@3/MRR for each
    rerank weight without running the model again. Pick the weight on `dev`; `holdout` shows
    whether that choice carries over to queries it was not chosen on."""
    if not config.RERANK_MODEL:
        raise SystemExit("sweep tunes the cross-encoder blend; set REGS_RERANK_MODEL first")
    with open(config.EVAL_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    sets = {"dev": data["queries"], "holdout": data.get("holdout") or []}
    print(f"reranker: {config.RERANK_MODEL}, {config.RERANK_CANDIDATES} candidate chunks")
    cached = {name: [(set(case["expect"]), search(case["q"], k=k, debug=True)["candidates"])
                     for case in cases] for name, cases in sets.items() if cases}
    report = {}
    for w in weights:
        row = []
        for name, items in cached.items():
            h1 = h3 = 0
            rr = 0.0
            for expected, cands in items:
                cands = [dict(c) for c in cands]
                _blend(cands, w)
                order = list(dict.fromkeys(c["key"] for c in sorted(cands, key=lambda c: -c["score"])))
                rank = next((i + 1 for i, key in enumerate(order) if key in expected), None)
                h1 += rank == 1
                h3 += bool(rank and rank <= 3)
                rr += 1 / rank if rank else 0
            n = len(items)
            report[(w, name)] = {"hit@1": round(h1 / n, 3), "hit@3": round(h3 / n, 3), "mrr": round(rr / n, 3)}
            row.append(f"{name} hit@1={h1 / n:.3f} hit@3={h3 / n:.3f} mrr={rr / n:.3f}")
        print(f"w={w:<4} " + "   ".join(row))
    return report
