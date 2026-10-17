"""Command line for the regulation index. Run from svc/:

    python -m regs fetch            download the acts in sources.yaml (needs internet, once)
    python -m regs milvus           start the Milvus Lite server (keep it running)
    python -m regs build            parse -> chunk -> embed -> Milvus (offline)
    python -m regs train-ltr        train LambdaMART (cosine + BM25 features) on the dev queries
    python -m regs search "текст"   query from the terminal
    python -m regs eval             retrieval recall on eval_queries.yaml
    python -m regs sweep            choose REGS_RERANK_WEIGHT (reranker vs retrieval blend)
    python -m regs verify rules.yaml   check that every quote in rules.yaml is in the corpus
    python -m regs serve            web UI on http://127.0.0.1:8010/regs
"""
import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m regs")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--force", action="store_true", help="download again even if the file exists")
    sub.add_parser("parse")
    sub.add_parser("build")
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--mode", default="auto", choices=["auto", "ltr", "hybrid", "dense", "bm25"])
    s.add_argument("--doc", action="append", help="limit to an act code; repeat for several")
    s.add_argument("--no-rerank", action="store_true")
    s.add_argument("--json", action="store_true")
    e = sub.add_parser("eval")
    e.add_argument("-k", type=int, default=5)
    e.add_argument("--run", action="append", help="only these runs, e.g. hybrid+glossary+rerank")
    sub.add_parser("sweep")
    t = sub.add_parser("train-ltr")
    t.add_argument("--per-list", type=int, default=60, help="candidates taken from each of dense and BM25")
    v = sub.add_parser("verify")
    v.add_argument("rules_yaml")
    m = sub.add_parser("milvus")
    m.add_argument("--host", default=None)
    m.add_argument("--port", type=int, default=None)
    w = sub.add_parser("serve")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=8010)
    a = p.parse_args(argv)

    if a.cmd == "fetch":
        from .fetch import fetch_all
        fetch_all(force=a.force)
    elif a.cmd == "parse":
        from .parse import parse_all
        parse_all()
    elif a.cmd == "build":
        from .chunk import build_chunks
        from .embed import get_embedder
        from .parse import parse_all
        from .store import build
        docs = parse_all()
        chunks = build_chunks()
        embedder = get_embedder()
        print(f"embedding {len(chunks)} chunks with {embedder.model_name} on {embedder.device} ...")
        meta = build(chunks, embedder, docs)
        print(f"indexed {meta['chunks']} chunks into {meta['collection']} at {meta['milvus_uri']} "
              f"(embedding took {meta['embed_seconds']} s)")
    elif a.cmd == "search":
        from .search import search
        res = search(a.query, k=a.k, mode=a.mode, docs=a.doc, explain=True, rerank=not a.no_rerank)
        if a.json:
            print(json.dumps(res, ensure_ascii=False, indent=1))
        else:
            print(f"{res['mode']} in {res['took_ms']} ms (rerank {res['rerank_ms']} ms, "
                  f"{res['reranker'] or 'off'}); BM25 terms: {res['lex_query']}")
            for i, h in enumerate(res["hits"], 1):
                print(f"\n{i}. {h['score']:.4f} [retrieval #{h['retrieval_rank']}, dense #{h['dense_rank']}, "
                      f"bm25 #{h['bm25_rank']}] "
                      f"{h['doc_short']}, {h['clause_label']} ({h['status']})\n   {h['url']}")
                print("   " + h["text"][:400].replace("\n", "\n   "))
    elif a.cmd == "eval":
        from .search import evaluate
        evaluate(k=a.k, runs=a.run)
    elif a.cmd == "train-ltr":
        from .ltr import train
        train(per_list=a.per_list)
    elif a.cmd == "sweep":
        from .search import sweep
        sweep()
    elif a.cmd == "verify":
        from .search import verify_quotes
        return 1 if verify_quotes(a.rules_yaml) else 0
    elif a.cmd == "milvus":
        from milvus_lite.adapter.grpc.server import run_server

        from . import config
        config.MILVUS_DATA_DIR.mkdir(parents=True, exist_ok=True)
        host, port = a.host or config.MILVUS_HOST, a.port or config.MILVUS_PORT
        print(f"Milvus Lite serving {config.MILVUS_DATA_DIR} on {host}:{port}")
        run_server(data_dir=str(config.MILVUS_DATA_DIR), host=host, port=port)
    elif a.cmd == "serve":
        import uvicorn
        uvicorn.run("regs.app:app", host=a.host, port=a.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
