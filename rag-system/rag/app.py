import json
import os
import secrets
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

from chunk import chunk
from regs import app as regs_app, store
from regs.embed import get_embedder
from regs.rewrite import enabled as rewrite_enabled
from regs.search import search as regs_search
from regs.violation import check as check_violation

env = os.environ
API_KEY = env["RAG_API_KEY"]
LLM_MODEL = env["LLM_MODEL"]
CUSTOM = env.get("MILVUS_COLLECTION", "legal_chunks")   # your own documents, from POST /ingest
CHUNK_CHARS = int(env.get("CHUNK_CHARS", "1500"))
FIELDS = ["document", "clause", "url", "text"]

llm = OpenAI(base_url=env["LLM_BASE_URL"], api_key=env["LLM_API_KEY"])

SYSTEM = ("You answer questions about cash-handling regulations using ONLY the numbered sources. "
          "Cite every claim as [n]. If the sources do not answer the question, say so. "
          "Never invent a document, clause or quote. Answer in the language of the question.")

app = FastAPI(title="rag")
# regulation index UI at /regs and its API at /api/regs/* (key = REGS_API_KEY, set to RAG_API_KEY in compose)
app.include_router(regs_app.router)
app.include_router(regs_app.page_router)


def auth(x_api_key: str = Header("")):
    if not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(401, "bad api key")


def embed(texts: list[str]) -> list[list[float]]:
    return get_embedder().passages(texts)


class Doc(BaseModel):
    document: str          # full document title, becomes the citation
    url: str = ""
    text: str


class IngestReq(BaseModel):
    documents: list[Doc]


class Query(BaseModel):
    query: str
    top_k: int = Field(5, ge=1, le=30)
    # "regs": the NBRK acts (python -m regs build); "custom": documents sent to POST /ingest
    index: Literal["regs", "custom"] = "regs"
    rewrite: bool = True   # regs only: an LLM restates the query in the acts' wording first
    judge: bool = True     # regs only: an LLM keeps only the clauses that apply (replaces the threshold)


@app.get("/health")
def health():
    return {"ok": True, "chunks": store.count(), "rewrite": rewrite_enabled(),
            "collections": store.client().list_collections()}


@app.post("/ingest", dependencies=[Depends(auth)])
def ingest(req: IngestReq):
    """Your own documents, searched with index="custom". Re-ingesting a document replaces its old chunks."""
    milvus, total = store.client(), 0
    for d in req.documents:
        chunks = chunk(d.text, CHUNK_CHARS)
        if not chunks:
            continue
        vecs = embed([c["text"] for c in chunks])
        if not milvus.has_collection(CUSTOM):
            milvus.create_collection(CUSTOM, dimension=len(vecs[0]), metric_type="COSINE", auto_id=True)
        milvus.delete(CUSTOM, filter=f"document == {json.dumps(d.document, ensure_ascii=False)}")
        milvus.insert(CUSTOM, [{"vector": v, "document": d.document, "url": d.url, **c}
                               for c, v in zip(chunks, vecs)])
        total += len(chunks)
    return {"chunks": total}


@app.post("/search", dependencies=[Depends(auth)])
def search(q: Query) -> dict:
    if q.index == "custom":
        milvus = store.client()
        if not milvus.has_collection(CUSTOM):
            return {"rewritten": None, "hits": []}
        res = milvus.search(CUSTOM, data=[get_embedder().query(q.query)], limit=q.top_k, output_fields=FIELDS)
        return {"rewritten": None,
                "hits": [{"score": h["distance"], **{f: h["entity"].get(f, "") for f in FIELDS}} for h in res[0]]}
    if not store.count():
        return {"rewritten": None, "hits": []}
    # one hit per clause, LambdaMART-ranked (the rewrite model when rewrite is on); only hits above
    # the model's relevance threshold, none if the LLM says the acts do not cover the query
    res = regs_search(q.query, k=q.top_k, mode="auto", rewrite=q.rewrite and rewrite_enabled(),
                      judge=q.judge and rewrite_enabled())
    return {"rewritten": res["rewritten"], "out_of_scope": res["out_of_scope"], "min_score": res["min_score"],
            "hits": [{"score": h["score"], "document": h["doc_short"], "clause": h["clause_label"],
                      "url": h["url"], "status": h["status"], "text": h["text"]} for h in res["hits"]]}


class LogReq(BaseModel):
    log: str = Field(min_length=1, max_length=100_000)   # the VLM's per-second log of one video
    top_k: int = Field(3, ge=1, le=20)                   # clauses the LLM compares the summary with


@app.post("/violation", dependencies=[Depends(auth)])
def violation(req: LogReq) -> dict:
    """VLM log -> one/two-sentence summary -> top_k clauses -> the ones it breaks, with citations
    from the index (regs/violation.py)."""
    if not rewrite_enabled():
        raise HTTPException(503, "no LLM configured (LLM_BASE_URL / LLM_MODEL)")
    return check_violation(req.log, k=req.top_k)


@app.post("/answer", dependencies=[Depends(auth)])
def answer(q: Query):
    res = search(q)
    hits = res["hits"]
    if not hits:
        return {"answer": "", "rewritten": res["rewritten"], "sources": []}
    ctx = "\n\n".join(f"[{i}] {h['document']}, {h['clause']}\n{h['text']}" for i, h in enumerate(hits, 1))
    # no temperature: newer Claude models reject it
    r = llm.chat.completions.create(model=LLM_MODEL, messages=[
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Sources:\n{ctx}\n\nQuestion: {q.query}"},
    ])
    # sources are returned verbatim from Milvus; the model never writes a citation itself
    return {"answer": r.choices[0].message.content, "rewritten": res["rewritten"], "sources": hits}
