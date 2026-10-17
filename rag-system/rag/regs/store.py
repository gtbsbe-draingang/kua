"""The Milvus collection: schema, full rebuild, and the dense, BM25 and hybrid searches.

One collection, one row per chunk:
  dense   FLOAT_VECTOR(dim)    bge-m3 embedding of embed_text, COSINE
  sparse  SPARSE_FLOAT_VECTOR  produced inside Milvus by a BM25 function from `lex`
  the rest are scalar fields returned with each hit (verbatim text, citation data, status)

By default this connects to the Milvus Lite server started with `python -m regs milvus` (see
config.py). AUTOINDEX lets each backend pick its own dense index: flat on Milvus Lite, HNSW on
Milvus Standalone. At a few hundred rows either is exact enough.
"""
import json
import time

from pymilvus import AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, RRFRanker

from . import config

# Field -> max length in bytes. build() checks every row against these.
VARCHAR_FIELDS = {
    "doc_code": 32, "doc_short": 256, "clause": 16, "clause_label": 64, "granularity": 16, "chapter": 1024,
    "section": 1024, "anchor": 16, "url": 512, "status": 16, "notes": 8192, "amended_by": 1024,
    "text": 16384,
}
INT_FIELDS = ["part", "parts", "seq"]
OUTPUT_FIELDS = list(VARCHAR_FIELDS) + INT_FIELDS

_client: MilvusClient | None = None


def client() -> MilvusClient:
    global _client
    if _client is None:
        if "://" not in config.MILVUS_URI:  # embedded Milvus Lite
            config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        _client = MilvusClient(uri=config.MILVUS_URI, token=config.MILVUS_TOKEN)
        # A collection reopened from disk starts released; searches need it loaded.
        if _client.has_collection(config.COLLECTION):
            _client.load_collection(config.COLLECTION)
    return _client


def _schema(dim: int):
    s = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False,
                                   description="NBRK regulation chunks")
    s.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
    for name, length in VARCHAR_FIELDS.items():
        s.add_field(name, DataType.VARCHAR, max_length=length)
    for name in INT_FIELDS:
        s.add_field(name, DataType.INT64)
    # Already stemmed in lexical.py, so the analyzer only needs to split words.
    s.add_field("lex", DataType.VARCHAR, max_length=16384, enable_analyzer=True,
                analyzer_params={"tokenizer": "standard"})
    s.add_field("dense", DataType.FLOAT_VECTOR, dim=dim)
    s.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    s.add_function(Function(name="lex_bm25", function_type=FunctionType.BM25,
                            input_field_names=["lex"], output_field_names=["sparse"]))
    return s


def _index_params(c: MilvusClient):
    ip = c.prepare_index_params()
    ip.add_index("dense", index_type="AUTOINDEX", metric_type="COSINE")
    ip.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25",
                 params={"bm25_k1": 1.2, "bm25_b": 0.75})
    return ip


def _check_lengths(row: dict) -> None:
    for name, limit in list(VARCHAR_FIELDS.items()) + [("lex", 16384)]:
        size = len(row[name].encode("utf-8"))
        if size > limit:
            raise ValueError(f"{row['id']}: field {name} is {size} bytes, limit {limit}; raise it in store.py")


def build(chunks: list[dict], embedder, docs: list[dict]) -> dict:
    """Drop and recreate the collection from scratch. The corpus is small, so a full rebuild is
    simpler than upserts and never leaves stale chunks behind."""
    c = client()
    name = config.COLLECTION
    if c.has_collection(name):
        c.drop_collection(name)
    c.create_collection(name, schema=_schema(embedder.dim), index_params=_index_params(c))

    t0 = time.time()
    vectors = embedder.passages([ch["embed_text"] for ch in chunks])
    embed_sec = time.time() - t0
    rows = []
    for ch, vec in zip(chunks, vectors):
        row = {"id": ch["id"], "lex": ch["lex"], "dense": vec}
        row.update({f: ch[f] for f in OUTPUT_FIELDS})
        _check_lengths(row)
        rows.append(row)
    for i in range(0, len(rows), 64):
        c.insert(name, rows[i:i + 64])

    counts: dict[str, int] = {}
    for ch in chunks:
        counts[ch["doc_code"]] = counts.get(ch["doc_code"], 0) + 1
    meta = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "collection": name, "milvus_uri": config.MILVUS_URI,
        "embed_model": embedder.model_name, "embed_dim": embedder.dim, "embed_device": embedder.device,
        "embed_dtype": embedder.dtype,
        "embed_seconds": round(embed_sec, 1), "chunks": len(chunks),
        "documents": [{
            "code": d["code"], "short": d["short"], "act": d["act"], "title": d["title"],
            "canonical_url": d["canonical_url"], "fetch_url": d["fetch_url"],
            "latest_amendment": d["latest_amendment"], "clauses": len(d["clauses"]),
            "chunks": counts.get(d["code"], 0), "include_chapters": d["include_chapters"],
        } for d in docs],
    }
    config.INDEX_META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


def index_meta() -> dict:
    if not config.INDEX_META_PATH.exists():
        return {}
    return json.loads(config.INDEX_META_PATH.read_text(encoding="utf-8"))


def make_filter(docs: list[str] | None = None, include_repealed: bool = False) -> str:
    parts = []
    if docs:
        parts.append(f"doc_code in {json.dumps(list(docs))}")
    if not include_repealed:
        parts.append('status != "repealed"')
    return " and ".join(parts)


def _hits(result) -> list[dict]:
    return [{"id": h["id"], "score": float(h["distance"]), **h["entity"]} for h in result[0]]


def search_dense(vector: list[float], k: int, flt: str = "") -> list[dict]:
    res = client().search(config.COLLECTION, data=[vector], anns_field="dense", limit=k, filter=flt,
                          output_fields=OUTPUT_FIELDS, search_params={"metric_type": "COSINE"})
    return _hits(res)


def search_bm25(lex_query: str, k: int, flt: str = "") -> list[dict]:
    """Hits best first, `score` = BM25 (>= 0). Milvus Lite 3.2 reports BM25 as a negative
    distance (best = most negative); Milvus Standalone reports it positive. abs() fits both.
    Milvus Lite also computes BM25 statistics over the filtered rows, so scores from searches
    with different filters are not comparable."""
    if not lex_query.strip():
        return []
    res = client().search(config.COLLECTION, data=[lex_query], anns_field="sparse", limit=k, filter=flt,
                          output_fields=OUTPUT_FIELDS, search_params={"metric_type": "BM25"})
    hits = _hits(res)
    for h in hits:
        h["score"] = abs(h["score"])
    return hits


def search_hybrid(vector: list[float], lex_query: str, k: int, flt: str = "", rrf_k: int = 60) -> list[dict]:
    """Milvus-side fusion: each request returns its own top candidates, then reciprocal rank fusion."""
    if not lex_query.strip():
        return search_dense(vector, k, flt)
    pool = max(k * 3, 20)
    reqs = [
        AnnSearchRequest(data=[vector], anns_field="dense", param={"metric_type": "COSINE"}, limit=pool, expr=flt or None),
        AnnSearchRequest(data=[lex_query], anns_field="sparse", param={"metric_type": "BM25"}, limit=pool, expr=flt or None),
    ]
    res = client().hybrid_search(config.COLLECTION, reqs=reqs, ranker=RRFRanker(rrf_k), limit=k,
                                 output_fields=OUTPUT_FIELDS)
    return _hits(res)


def get(ids: list[str]) -> list[dict]:
    rows = client().get(config.COLLECTION, ids=ids, output_fields=OUTPUT_FIELDS)
    return [dict(r) for r in rows]


def neighbours(doc_code: str, seq: int, before: int = 1, after: int = 1) -> list[dict]:
    """Every chunk of the clauses just before and after one chunk's clause, in the same act.
    seq is clause_index * 100 + part_index, so whole clauses are [base, base + 100)."""
    base = (seq // 100) * 100
    flt = (f'doc_code == "{doc_code}" and granularity == "clause" '
           f'and seq >= {base - 100 * before} and seq < {base + 100 * (after + 1)}')
    rows = client().query(config.COLLECTION, filter=flt, output_fields=["id"] + OUTPUT_FIELDS)
    return sorted((dict(r) for r in rows), key=lambda r: r["seq"])


def count() -> int:
    c = client()
    if not c.has_collection(config.COLLECTION):
        return 0
    # count(*) includes unflushed rows; get_collection_stats on Milvus Standalone does not
    return int(c.query(config.COLLECTION, filter="", output_fields=["count(*)"])[0]["count(*)"])
