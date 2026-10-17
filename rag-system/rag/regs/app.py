"""Web API and page for searching the regulation index.

Standalone:   python -m regs serve          -> http://127.0.0.1:8010/regs
Inside api.py (one process, one public URL):
    from regs.app import router as regs_router, page_router as regs_page
    app.include_router(regs_router); app.include_router(regs_page)

Endpoints (all GET; X-API-Key is required only when REGS_API_KEY is set):
    /api/regs/stats                 index metadata: acts, editions, model, chunk counts
    /api/regs/search?q=...          LambdaMART (default) | hybrid | dense | bm25 ranking, optional
                                    neural reranking, with per-hit citation YAML
    /api/regs/chunk/{id}            one chunk plus the clauses around it
"""
import secrets
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse

from . import config, ltr, store
from .embed import get_embedder
from .rerank import get_reranker
from .judge import enabled as judge_enabled
from .rewrite import enabled as rewrite_enabled
from .search import MODES, citation_yaml, search


def _check_key(x_api_key: str | None = Header(default=None)) -> None:
    if config.API_KEY and not secrets.compare_digest(x_api_key or "", config.API_KEY):
        raise HTTPException(status_code=401, detail="missing or wrong X-API-Key")


router = APIRouter(prefix="/api/regs", tags=["regs"], dependencies=[Depends(_check_key)])
page_router = APIRouter()


@router.get("/stats")
def stats() -> dict:
    meta = store.index_meta()
    if not meta:
        raise HTTPException(status_code=503, detail="index not built; run `python -m regs build`")
    return {**meta, "rows": store.count(), "milvus_uri": config.MILVUS_URI, "modes": list(MODES),
            "reranker": config.RERANK_MODEL or None, "rerank_candidates": config.RERANK_CANDIDATES,
            "ltr": ltr.load()[1] if ltr.load() else None,
            "ltr_rewrite": ltr.load(True)[1] if ltr.load(True) else None,
            "rewrite_model": config.REWRITE_MODEL if rewrite_enabled() else None,
            "judge_model": config.REWRITE_MODEL if judge_enabled() else None,
            "api_key_required": bool(config.API_KEY)}


@router.get("/search")
def search_endpoint(q: str = Query(min_length=2, max_length=1000),
                    k: int = Query(default=5, ge=1, le=30),
                    mode: str = Query(default="auto", pattern="^(auto|ltr|hybrid|dense|bm25)$"),
                    doc: list[str] = Query(default=[]),
                    include_repealed: bool = False,
                    glossary: bool = True,
                    rerank: bool = True,
                    rewrite: bool = True,
                    threshold: bool = True,
                    judge: bool = True) -> dict:
    res = search(q, k=k, mode=mode, docs=doc or None, include_repealed=include_repealed,
                 explain=True, use_glossary=glossary, rerank=rerank,
                 rewrite=rewrite and rewrite_enabled(), threshold=threshold,
                 judge=judge and judge_enabled())
    for h in res["hits"]:
        h["citation_yaml"] = citation_yaml(h)
    return res


@router.get("/chunk/{chunk_id}")
def chunk_endpoint(chunk_id: str, before: int = Query(default=1, ge=0, le=5),
                   after: int = Query(default=1, ge=0, le=5)) -> dict:
    rows = store.get([chunk_id])
    if not rows:
        raise HTTPException(status_code=404, detail="no such chunk")
    row = rows[0]
    return {"chunk": row, "context": store.neighbours(row["doc_code"], row["seq"], before, after)}


@page_router.get("/regs", include_in_schema=False)
def page() -> FileResponse:
    return FileResponse(config.STATIC_DIR / "index.html")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_embedder()   # load the models before the first request instead of during it
    get_reranker()   # None when REGS_RERANK_MODEL is empty
    ltr.load()       # None until `python -m regs train-ltr` has run
    store.client()   # fail at startup, not on the first search, if Milvus is not running
    yield


app = FastAPI(title="Regulation search", lifespan=lifespan)
app.include_router(router)
app.include_router(page_router)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/regs")
