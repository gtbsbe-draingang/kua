"""Dense clause retrieval with bge-m3. torch and transformers load lazily, inside embed()."""
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import NamedTuple

import numpy as np

from rag.corpus import Clause, passages

EMBEDDER = "BAAI/bge-m3"
MAX_TOKENS = 512
BATCH = 32
TOP_K = 8

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
    clause_id: str
    score: float
    part: int


class Index:
    def __init__(self, clauses: list[Clause], embed_fn: Callable[[list[str]], np.ndarray] = embed):
        self.clauses = list(clauses)
        self.by_id = {c.id: c for c in self.clauses}
        self._embed = embed_fn
        self._passages = passages(self.clauses)
        self._order = {c.id: i for i, c in enumerate(self.clauses)}
        texts = [t for _, _, t in self._passages]
        self._matrix = (np.asarray(embed_fn(texts), dtype=np.float32) if texts
                        else np.zeros((0, 1024), dtype=np.float32))

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
            best: dict[str, Hit] = {}
            for (cid, part, _), s in zip(self._passages, row):
                if cid not in best or s > best[cid].score:
                    best[cid] = Hit(cid, float(s), part)
            ranked = sorted(best.values(), key=lambda h: (-h.score, self._order[h.clause_id]))
            results[i] = ranked[:k]
        return results
