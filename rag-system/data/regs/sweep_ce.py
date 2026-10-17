"""Score dev+holdout once with bge-reranker-v2-m3 over the hybrid pool; compare reranker inputs and blend weights."""
import json, time, yaml
from regs import config, store, rerank
from regs.search import query_inputs, _blend, _fold
from regs.embed import get_embedder
from regs.chunk import passage

POOL = 40
data = yaml.safe_load(open(config.EVAL_PATH, encoding="utf-8"))
out = {}
for set_name in ("queries", "holdout"):
    rows = []
    for case in data[set_name]:
        t = time.time()
        qi = query_inputs(case["q"], rewrite=True)
        raw = store.search_hybrid(get_embedder().query(qi["dense"]), qi["lex"], POOL, "")
        ps = [passage(h["doc_short"], h["chapter"], h["section"], h["clause_label"], h["text"]) for h in raw]
        s_raw = rerank.score(case["q"], ps)
        s_kw = rerank.score(qi["dense"], ps)
        rows.append({"q": case["q"], "expect": case["expect"], "kw": qi["rewritten"],
                     "cands": [{"key": f"{h['doc_code']}:{h['clause']}", "retrieval_score": h["score"],
                                "raw": a, "kw": b} for h, a, b in zip(raw, s_raw, s_kw)]})
        print(set_name, round(time.time() - t, 1), case["q"][:60], flush=True)
    out[set_name] = rows
json.dump(out, open("/data/regs/ce_scores.json", "w"), ensure_ascii=False)
