"""Clause retrieval: dense bge-m3 search, fused with a lexical BM25 + glossary search (rag/lexical.py) by
reciprocal rank fusion. torch and transformers load lazily, inside embed()."""
from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from typing import NamedTuple

import numpy as np

from rag.corpus import Clause, passages

EMBEDDER = "BAAI/bge-m3"
MAX_TOKENS = 512
BATCH = 32
TOP_K = 8
RRF_K = 60
# What RAG_HYBRID means when it is unset or empty: "1" fuses the lexical channel in, "0" searches dense only.
# "1" since the pod measurement of 2026-10-17 (recall@8 on the 15 queries: dense 10, hybrid 12; see the task report).
HYBRID_DEFAULT = "1"

log = logging.getLogger(__name__)

_lock = threading.Lock()
_model = None  # (tokenizer, model, device)


def _load():
    global _model
    with _lock:
        if _model is None:
            import torch
            from transformers import AutoModel, AutoTokenizer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            tok = AutoTokenizer.from_pretrained(EMBEDDER, local_files_only=True)
            model = AutoModel.from_pretrained(EMBEDDER, local_files_only=True, dtype=dtype)
            _model = (tok, model.to(device).eval(), device)
        return _model


def embed(texts: list[str]) -> np.ndarray:
    """L2-normalised bge-m3 dense vectors (CLS of the last hidden state), shape (n, 1024)."""
    if not texts:
        return np.zeros((0, 1024), dtype=np.float32)
    import torch

    tok, model, device = _load()
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), BATCH):
            enc = tok(texts[i:i + BATCH], padding=True, truncation=True,
                      max_length=MAX_TOKENS, return_tensors="pt").to(device)
            cls = model(**enc).last_hidden_state[:, 0].float()
            out.append(torch.nn.functional.normalize(cls, dim=-1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


class Hit(NamedTuple):
    """One clause. score is the cosine of its best passage in a dense search and the reciprocal-rank-fusion sum in
    a hybrid one; part indexes Clause.parts (the passage the clause was found by)."""
    clause_id: str
    score: float
    part: int


def hybrid_enabled() -> bool:
    """The RAG_HYBRID switch: 0, false, no or off turn the lexical channel off, any other value turns it on, and
    unset or empty means HYBRID_DEFAULT."""
    value = (os.environ.get("RAG_HYBRID") or HYBRID_DEFAULT).strip().lower()
    return value not in ("0", "false", "no", "off")


class Index:
    """The in-force clauses, searched by bge-m3 vectors of their passages (one per non-empty line) and, when hybrid,
    also by BM25 over the same passages. hybrid=None follows RAG_HYBRID. If the lexical side cannot be built (the
    stemmer package is missing, the glossary is broken) the index logs it and serves dense results only: .hybrid
    says which one this is. Tools may flip .hybrid to compare the two on one index."""

    def __init__(self, clauses: list[Clause], embed_fn: Callable[[list[str]], np.ndarray] = embed,
                 hybrid: bool | None = None):
        self.clauses = list(clauses)
        self.by_id = {c.id: c for c in self.clauses}
        self._embed = embed_fn
        self._passages = passages(self.clauses)
        self._order = {c.id: i for i, c in enumerate(self.clauses)}
        texts = [t for _, _, t in self._passages]
        self._matrix = (np.asarray(embed_fn(texts), dtype=np.float32) if texts
                        else np.zeros((0, 1024), dtype=np.float32))
        self._lexicon = None
        if hybrid_enabled() if hybrid is None else hybrid:
            try:  # imported here: a missing stemmer package must not stop rag.retrieve from importing
                from rag import lexical

                self._lexicon = lexical.Lexicon(texts)
            except Exception:  # noqa: BLE001 - the optional channel must never take the dense search down
                log.warning("lexical channel failed to build; searching dense only", exc_info=True)
        self.hybrid = self._lexicon is not None
        mode = f"hybrid (dense + BM25, RRF k={RRF_K})" if self.hybrid else "dense only"
        log.info("retrieval: %s over %d clauses, %d passages", mode, len(self.clauses), len(self._passages))

    def search(self, text: str, k: int = TOP_K) -> list[Hit]:
        return self.search_many([text], k)[0]

    def search_many(self, texts: list[str], k: int = TOP_K) -> list[list[Hit]]:
        live = [i for i, t in enumerate(texts) if t and t.strip()]
        results: list[list[Hit]] = [[] for _ in texts]
        if not live or not len(self._matrix):
            return results
        vecs = np.asarray(self._embed([texts[i] for i in live]), dtype=np.float32)
        sims = vecs @ self._matrix.T
        for row, i in zip(sims, live):
            ranked = self._fold(row)
            if self.hybrid:
                try:
                    ranked = self._fuse(ranked, self.lexical_hits(texts[i]))
                except Exception:  # noqa: BLE001 - one failed lexical query serves its dense ranking
                    log.warning("lexical search failed; serving the dense ranking for this query", exc_info=True)
            results[i] = ranked[:k]
        return results

    def lexical_hits(self, text: str) -> list[Hit]:
        """The clauses with a BM25 match for text (and the glossary words its everyday words bring in), best first,
        scored by their best passage; [] without the lexical channel."""
        if self._lexicon is None or not text.strip():
            return []
        return [h for h in self._fold(self._lexicon.scores(text)) if h.score > 0]

    def _fold(self, scores) -> list[Hit]:
        """One Hit per clause, from its best-scoring passage (the first on a tie), best clause first (corpus order
        on a tie)."""
        best: dict[str, Hit] = {}
        for (cid, part, _), s in zip(self._passages, scores):
            if cid not in best or s > best[cid].score:
                best[cid] = Hit(cid, float(s), part)
        return sorted(best.values(), key=lambda h: (-h.score, self._order[h.clause_id]))

    def _fuse(self, dense: list[Hit], lexical: list[Hit]) -> list[Hit]:
        """Reciprocal rank fusion of the two clause rankings: score = sum over channels of 1 / (RRF_K + rank). The
        part is the one the channel that ranks the clause higher found it by (the dense one when equal)."""
        fused: dict[str, Hit] = {}
        for rank, h in enumerate(dense, 1):
            fused[h.clause_id] = Hit(h.clause_id, 1 / (RRF_K + rank), h.part)
        dense_rank = {h.clause_id: rank for rank, h in enumerate(dense, 1)}
        for rank, h in enumerate(lexical, 1):
            prev = fused[h.clause_id]
            part = h.part if rank < dense_rank[h.clause_id] else prev.part
            fused[h.clause_id] = Hit(h.clause_id, prev.score + 1 / (RRF_K + rank), part)
        return sorted(fused.values(), key=lambda h: (-h.score, self._order[h.clause_id]))
