"""Regulation matcher server (the pipeline's judge). Runs in env/main, from svc/:

    env/main/bin/python -m uvicorn servers.rag_server:app --host 127.0.0.1 --port 30001

GET  /healthz -> {"ok": true, "clauses": n, "passages": m}. The port opens only after every
     in-force clause has been embedded, so a reachable /healthz means ready.
POST /v1/chat/completions with a clients.judge_request body (the frames are not used)
     -> {"model": "rag+iv3", "choices": [{"message": {"role": "assistant", "content": <JSON>}}]}
     where the content is {violation, rule_id, clause_id, justification, confidence, citation}.
     A body it cannot read -> 400 {"error": "bad request: ..."}; the LLM failing -> 502
     {"error": "llm"}; an answer without a candidate rule_id -> 502 {"error": "bad answer"}.
POST /retrieve {"texts": [str], "k": 1-20 (8)} -> {"hits": [[{"clause_id", "score", "part"}]]},
     one list per text, for measurement.

How a judge request is answered:
- Clauses: the clause of each candidate rule's catalogue quote first, then the top 8 clauses
  retrieved for the observation (none for an empty observation), each once
  (matcher.clause_candidates). Retrieved clauses are dropped from the end while the prompt
  is over the /generate limit (matcher.fit_clauses). matcher.judge_prompt runs both and
  builds the prompt; tools/rag_eval.py calls the same function.
- The LLM picks the rule, the clause or none, and P(violation) from matcher.build_prompt;
  matcher.parse_answer reads its JSON. RAG_LLM_KIND selects the backend:
    iv3 (default)  InternVideo3's text-only POST /generate, at RAG_LLM_URL (default port 30000) or spread
                   over the comma-separated RAG_LLM_URLS;
    openai         an OpenAI-style POST /v1/chat/completions (the Qwen3.5-9B judge on SGLang, tools/judge_llm.sh,
                   default http://127.0.0.1:30005/v1/chat/completions) for the model named by RAG_LLM_MODEL
                   (default qwen3.5-9b), thinking off, any <think> block cut from the reply;
  both with a timeout of RAG_LLM_TIMEOUT seconds. An unknown kind stops the startup.
- The model never writes the citation. For the chosen clause the server takes the act's
  document, "п. <num>" and the url from the corpus, and as the quote the clause part the
  retrieval hit on, or else the catalogue quote that brought the clause in (the chosen
  rule's own first), cut to 600 characters at a word boundary. No clause, no citation.
"""
from __future__ import annotations

import os

# Before torch is imported (it loads with the embedder, at startup): the pod's CPU quota is 16 cores, shared.
os.environ.setdefault("OMP_NUM_THREADS", "4")

import json  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402

import httpx  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from pipeline import clients, replicas  # noqa: E402
from rag import matcher  # noqa: E402
from rag.corpus import Clause, find_quote, load_clauses, passages  # noqa: E402
from rag.retrieve import TOP_K, Index  # noqa: E402

LLM_URL = "http://127.0.0.1:30000/generate"
LLM_KIND = "iv3"
CHAT_URL = "http://127.0.0.1:30005/v1/chat/completions"
CHAT_MODEL = "qwen3.5-9b"
LLM_TIMEOUT = 25.0
MAX_NEW_TOKENS = 384
CONNECT_TIMEOUT = 2.0
MODEL = "rag+iv3"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("rag")

LLM = Callable[[str, str, float], str]  # (system, prompt, timeout) -> the model's text; raises on failure


def post_json(url: str, body: dict, timeout: float) -> dict:
    """POST body as JSON to a local LLM server and return its JSON object; anything else is an exception."""
    # trust_env=False: a local service, and a proxy from the environment must not reroute it
    r = httpx.post(url, json=body, timeout=httpx.Timeout(timeout, connect=min(CONNECT_TIMEOUT, timeout)),
                   trust_env=False)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    d = r.json()
    if not isinstance(d, dict):
        raise ValueError("reply is not a JSON object")
    return d


def http_llm(url: str) -> LLM:
    """An LLM behind POST url, the iv3 server's /generate: {system, prompt, max_new_tokens} -> {"text": ...}."""
    def call(system: str, prompt: str, timeout: float) -> str:
        d = post_json(url, {"system": system, "prompt": prompt, "max_new_tokens": MAX_NEW_TOKENS}, timeout)
        if not isinstance(d.get("text"), str):
            raise ValueError("reply without a text")
        return d["text"]
    return call


def strip_think(text: str) -> str:
    """text without its <think>…</think> blocks, stripped. A lone </think> (the opening tag lives in the chat
    template's prompt) drops everything before it; a <think> never closed (cut off at max_tokens) everything after."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in text:
        text = text.rpartition("</think>")[2]
    return text.partition("<think>")[0].strip()


def openai_llm(url: str, model: str) -> LLM:
    """An LLM behind POST url, an OpenAI-style /v1/chat/completions (SGLang): greedy, thinking off (Qwen3.5 thinks
    by default, which costs seconds the 240 s job deadline cannot spare)."""
    def call(system: str, prompt: str, timeout: float) -> str:
        d = post_json(url, {"model": model,
                            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                            "max_tokens": MAX_NEW_TOKENS, "temperature": 0,
                            "chat_template_kwargs": {"enable_thinking": False}}, timeout)
        try:
            content = d["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            content = None
        if not isinstance(content, str):
            raise ValueError("reply without a message content")
        return strip_think(content)
    return call


def pooled_llm(urls: list[str], backend: Callable[[str], LLM] = http_llm) -> LLM:
    """backend(url) over several replicas: each call goes to the least busy one, and to another once when the
    chosen one refuses the connection."""
    pool = replicas.Pool(urls)
    calls = {u: backend(u) for u in pool.urls}

    def call(system: str, prompt: str, timeout: float) -> str:
        return replicas.call_with_failover(pool, lambda url, left: calls[url](system, prompt, left), timeout)
    return call


def make_llm(kind: str, urls: list[str], model: str) -> LLM:
    """The pooled LLM client for RAG_LLM_KIND: iv3 (/generate) or openai (chat completions for model)."""
    if kind == "iv3":
        return pooled_llm(urls)
    if kind == "openai":
        return pooled_llm(urls, lambda url: openai_llm(url, model))
    raise ValueError(f"RAG_LLM_KIND must be iv3 or openai, not {kind!r}")


def catalogue_quote(clause: Clause, candidates: list[dict], rule_id: str, clauses: list[Clause]) -> str | None:
    """The catalogue quote that brought clause in, whitespace collapsed as the corpus has it: the chosen rule's own
    citations first, then the other candidates' in order."""
    ordered = sorted(candidates, key=lambda c: str(c.get("id")) != rule_id)
    for cand in ordered:
        for cit in cand.get("citations") or []:
            quote = cit.get("quote") if isinstance(cit, dict) else None
            if isinstance(quote, str) and quote.strip():
                found = find_quote(clauses, quote)
                if found is not None and found.id == clause.id:
                    return " ".join(quote.split())
    return None


def citation(clause: Clause, part: int | None, candidates: list[dict], rule_id: str, clauses: list[Clause]) -> dict:
    """{document, clause, quote, url} from the corpus: the quote is the retrieved part, else the catalogue quote."""
    quote = clause.parts[part] if part is not None and 0 <= part < len(clause.parts) else None
    if not quote:
        quote = catalogue_quote(clause, candidates, rule_id, clauses) or matcher.clause_text(clause)
    return {"document": clause.document, "clause": f"п. {clause.num}", "quote": matcher.cut(quote, matcher.QUOTE_MAX),
            "url": clause.url}


class RetrieveRequest(BaseModel):
    texts: list[str]
    k: int = Field(TOP_K, ge=1, le=20)


def create_app(index=None, llm: LLM | None = None, llm_timeout: float = LLM_TIMEOUT, lifespan=None) -> FastAPI:
    """The server over index (rag.retrieve.Index or anything with .clauses, .by_id, .search and .search_many) and
    llm. The module's app passes neither and fills both in its lifespan."""
    app = FastAPI(title="rag", lifespan=lifespan)
    app.state.index, app.state.llm, app.state.llm_timeout = index, llm, llm_timeout

    def bad(message: str) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": f"bad request: {message}"})

    @app.exception_handler(RequestValidationError)
    async def bad_request(_request: Request, exc: RequestValidationError):
        return bad(str(exc))

    @app.get("/healthz")
    def healthz():
        index = app.state.index
        return {"ok": True, "clauses": len(index.clauses), "passages": len(passages(index.clauses))}

    @app.post("/v1/chat/completions")
    def chat_completions(body: dict):
        t0 = time.perf_counter()
        index, llm = app.state.index, app.state.llm
        try:
            observation, candidates = clients.judge_fields(body)
        except ValueError as exc:
            return bad(str(exc))
        rule_ids = [str(c.get("id")) for c in candidates if c.get("id") is not None]
        if not rule_ids:
            return bad("no candidate rules")
        hits = index.search(observation) if observation.strip() else []
        system, prompt, clauses = matcher.judge_prompt(observation, candidates, hits, index)
        clause_ids = [c.id for c, _ in clauses]
        t1 = time.perf_counter()
        try:
            text = llm(system, prompt, app.state.llm_timeout)
        except Exception as exc:  # noqa: BLE001 - every LLM failure is a 502
            log.warning("llm failed after %.1f s: %s: %s", time.perf_counter() - t1, type(exc).__name__, exc)
            return JSONResponse(status_code=502, content={"error": "llm"})
        t2 = time.perf_counter()
        answer = matcher.parse_answer(text, rule_ids, clause_ids)
        if answer is None:
            log.warning("bad answer for rules %s: %r", rule_ids, text[:500])
            return JSONResponse(status_code=502, content={"error": "bad answer"})
        offered = {c.id: (c, part) for c, part in clauses}
        cid = answer["clause_id"]
        answer["citation"] = (citation(*offered[cid], candidates, answer["rule_id"], index.clauses)
                              if cid is not None else None)
        log.info("judge | %d hits, %d clauses offered (%d prompt chars) | rule %s clause %s violation %s "
                 "confidence %s | retrieve %.2fs llm %.2fs", len(hits), len(clauses), len(prompt),
                 answer["rule_id"], cid or "-", answer["violation"], answer["confidence"], t1 - t0, t2 - t1)
        return {"model": MODEL,
                "choices": [{"message": {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)}}]}

    @app.post("/retrieve")
    def retrieve(req: RetrieveRequest):
        found = app.state.index.search_many(req.texts, req.k)
        return {"hits": [[{"clause_id": h.clause_id, "score": h.score, "part": h.part} for h in row]
                         for row in found]}

    return app


@asynccontextmanager
async def lifespan(app: FastAPI):
    import torch

    torch.set_num_threads(4)
    t0 = time.perf_counter()
    clauses = load_clauses()
    app.state.index = Index(clauses)
    log.info("indexed %d clauses, %d passages in %.1f s", len(clauses), len(passages(clauses)),
             time.perf_counter() - t0)
    kind = os.environ.get("RAG_LLM_KIND", LLM_KIND)
    model = os.environ.get("RAG_LLM_MODEL", CHAT_MODEL)
    urls = replicas.parse_urls(os.environ.get("RAG_LLM_URLS"),
                               os.environ.get("RAG_LLM_URL", CHAT_URL if kind == "openai" else LLM_URL))
    app.state.llm = make_llm(kind, urls, model)
    app.state.llm_timeout = float(os.environ.get("RAG_LLM_TIMEOUT", LLM_TIMEOUT))
    log.info("LLM %s%s at %s, timeout %.0f s", kind, f" {model}" if kind == "openai" else "", ", ".join(urls),
             app.state.llm_timeout)
    yield
    app.state.index = app.state.llm = None


app = create_app(lifespan=lifespan)
