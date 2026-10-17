"""LambdaMART reranking over the two first-stage signals: cosine (bge-m3) and BM25.

Hybrid search fuses the dense and BM25 lists with a fixed rule (RRF). LambdaMART learns the
fusion instead: gradient-boosted regression trees fitted to LambdaRank gradients, so the trees
directly optimise ranking quality (NDCG@k) on labelled queries.

Candidates: the union of the dense top-N and the BM25 top-N chunks. Every candidate gets both
scores exactly. A missing cosine comes from a dense search filtered to those ids (cosine does
not depend on other rows). BM25 comes from one search that returns every matching chunk
(BM25_LIMIT), because Milvus Lite computes BM25 statistics over the filtered rows, so an
id-filtered BM25 search would give scores on a different scale. A chunk absent from that list
shares no stemmed term with the query: BM25 = 0.

Features (all derived from cosine and BM25, per candidate chunk):
  cos        cosine similarity between query and chunk (bge-m3)
  bm25       BM25 score of the stemmed, glossary-expanded query
  cos_rank   rank of the chunk by cosine among the query's candidates (1 = best)
  bm25_rank  same for BM25; chunks with BM25 = 0 share the last rank
  cos_gap    cos minus the best cosine for this query
  bm25_rel   bm25 divided by the best BM25 for this query (raw BM25 is not comparable across queries)
Monotone constraints: the score can only rise with cos, bm25, cos_gap and bm25_rel, and only
fall as the ranks grow. With few labelled queries this is the main guard against overfitting.

Why not LightGBM: its macOS wheel needs Homebrew's libomp, and when it borrows torch's copy
instead the process deadlocks (seen 2026-10-17). These are scikit-learn trees with the same
LambdaMART update (Burges, 2010): lambda gradients weighted by |delta NDCG| of each pair swap,
and Newton leaf values, obtained by fitting the trees to lambda/w with sample weight w.
"""
import functools
import json
import pickle
import time

import numpy as np

from . import config, store

FEATURES = ["cos", "bm25", "cos_rank", "bm25_rank", "cos_gap", "bm25_rel"]
BM25_LIMIT = 2000  # more than the chunk count, so the BM25 list holds every matching chunk
MONOTONE = [1, 1, -1, -1, 1, 1]


# --- candidates and features ----------------------------------------------------------------

def _and(flt: str, extra: str) -> str:
    return f"({flt}) and {extra}" if flt else extra


def candidates(vector: list[float], lex: str, flt: str, per_list: int) -> list[dict]:
    """Union of the dense and BM25 top lists, each with exact `cos` and `bm25` scores."""
    dense = store.search_dense(vector, per_list, flt)
    sparse = store.search_bm25(lex, BM25_LIMIT, flt) if lex.strip() else []
    rows = {h["id"]: h for h in dense}
    for h in sparse[:per_list]:
        rows.setdefault(h["id"], h)
    cos = {h["id"]: h["score"] for h in dense}
    bm = {h["id"]: h["score"] for h in sparse}
    missing = [i for i in rows if i not in cos]
    if missing:
        for h in store.search_dense(vector, len(missing), _and(flt, f"id in {json.dumps(missing)}")):
            cos[h["id"]] = h["score"]
    out = []
    for i, h in rows.items():
        h = dict(h)
        h["cos"], h["bm25"] = float(cos.get(i, 0.0)), float(bm.get(i, 0.0))
        out.append(h)
    return out


def features(cands: list[dict]) -> np.ndarray:
    """One row per candidate, columns in FEATURES order. Also stores them on each candidate."""
    if not cands:
        return np.zeros((0, len(FEATURES)))
    cos = np.array([c["cos"] for c in cands])
    bm = np.array([c["bm25"] for c in cands])
    n = len(cands)
    cos_rank = np.empty(n)
    cos_rank[np.argsort(-cos, kind="stable")] = np.arange(1, n + 1)
    bm_rank = np.empty(n)
    bm_rank[np.argsort(-bm, kind="stable")] = np.arange(1, n + 1)
    bm_rank[bm <= 0] = n
    best_bm = bm.max()
    X = np.column_stack([cos, bm, cos_rank, bm_rank, cos - cos.max(),
                         bm / best_bm if best_bm > 0 else np.zeros(n)])
    for c, row in zip(cands, X):
        c["features"] = {name: round(float(v), 4) for name, v in zip(FEATURES, row)}
    return X


# --- LambdaMART -----------------------------------------------------------------------------

class LambdaMART:
    def __init__(self, n_trees: int = 100, learning_rate: float = 0.1, max_leaf_nodes: int = 8,
                 min_samples_leaf: int = 10, ndcg_at: int = 10, sigma: float = 1.0,
                 monotone: list[int] | None = None, seed: int = 0):
        self.n_trees, self.learning_rate = n_trees, learning_rate
        self.max_leaf_nodes, self.min_samples_leaf = max_leaf_nodes, min_samples_leaf
        self.ndcg_at, self.sigma, self.monotone, self.seed = ndcg_at, sigma, monotone, seed
        self.trees: list = []

    def params(self) -> dict:
        return {"n_trees": self.n_trees, "learning_rate": self.learning_rate,
                "max_leaf_nodes": self.max_leaf_nodes, "min_samples_leaf": self.min_samples_leaf,
                "ndcg_at": self.ndcg_at, "sigma": self.sigma}

    def _lambdas(self, s: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """LambdaRank gradients and Newton weights for one query (binary or graded labels)."""
        n = len(s)
        lam, w = np.zeros(n), np.zeros(n)
        gains = 2.0 ** y - 1
        ideal = np.sort(gains)[::-1][:self.ndcg_at]
        idcg = float(np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2))))
        if idcg == 0:
            return lam, w
        rank = np.empty(n, dtype=int)
        rank[np.argsort(-s, kind="stable")] = np.arange(n)
        disc = np.where(rank < self.ndcg_at, 1.0 / np.log2(rank + 2.0), 0.0)
        hi = y[:, None] > y[None, :]                        # pairs (i, j) where i should rank above j
        if not hi.any():
            return lam, w
        delta = np.abs((gains[:, None] - gains[None, :]) * (disc[:, None] - disc[None, :])) / idcg
        rho = 1.0 / (1.0 + np.exp(self.sigma * (s[:, None] - s[None, :])))
        g = np.where(hi, self.sigma * delta * rho, 0.0)
        h = np.where(hi, self.sigma ** 2 * delta * rho * (1.0 - rho), 0.0)
        lam = g.sum(axis=1) - g.sum(axis=0)
        w = h.sum(axis=1) + h.sum(axis=0)
        return lam, w

    def fit(self, X: np.ndarray, y: np.ndarray, groups: list[np.ndarray]) -> "LambdaMART":
        from sklearn.tree import DecisionTreeRegressor
        F = np.zeros(len(y))
        self.trees = []
        for m in range(self.n_trees):
            lam, w = np.zeros(len(y)), np.zeros(len(y))
            for idx in groups:
                lam[idx], w[idx] = self._lambdas(F[idx], y[idx])
            mask = w > 1e-12
            if not mask.any():
                break
            tree = DecisionTreeRegressor(max_leaf_nodes=self.max_leaf_nodes,
                                         min_samples_leaf=self.min_samples_leaf,
                                         monotonic_cst=self.monotone, random_state=self.seed + m)
            tree.fit(X[mask], lam[mask] / w[mask], sample_weight=w[mask])  # leaf = sum(lam) / sum(w)
            F += self.learning_rate * tree.predict(X)
            self.trees.append(tree)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros(len(X))
        for tree in self.trees:
            out += self.learning_rate * tree.predict(X)
        return out


# --- training ---------------------------------------------------------------------------------

def _query_data(cases: list[dict], per_list: int) -> list[dict]:
    """Candidates, features and labels for each labelled query (label 1 = an expected clause)."""
    from .embed import get_embedder
    from .lexical import to_lex
    from .search import expand
    embedder = get_embedder()
    flt = store.make_filter()
    out = []
    for case in cases:
        expected = {tuple(e.split(":")) for e in case["expect"]}
        lex = to_lex(" ".join([case["q"]] + expand(case["q"])))
        cands = candidates(embedder.query(case["q"]), lex, flt, per_list)
        X = features(cands)
        y = np.array([1.0 if (c["doc_code"], c["clause"]) in expected else 0.0 for c in cands])
        out.append({"q": case["q"], "expected": expected, "cands": cands, "X": X, "y": y})
    return out


def _clause_rank(model: LambdaMART, item: dict) -> int | None:
    scores = model.predict(item["X"])
    order = list(dict.fromkeys((item["cands"][i]["doc_code"], item["cands"][i]["clause"])
                               for i in np.argsort(-scores, kind="stable")))
    return next((r + 1 for r, key in enumerate(order) if key in item["expected"]), None)


def _fit(items: list[dict], params: dict) -> LambdaMART:
    items = [it for it in items if it["y"].any()]  # a query whose clause is not in the pool teaches nothing
    X = np.vstack([it["X"] for it in items])
    y = np.concatenate([it["y"] for it in items])
    groups, start = [], 0
    for it in items:
        groups.append(np.arange(start, start + len(it["y"])))
        start += len(it["y"])
    return LambdaMART(monotone=MONOTONE, **params).fit(X, y, groups)


def _metrics(ranks: list[int | None]) -> dict:
    n = len(ranks)
    return {"hit@1": round(sum(r == 1 for r in ranks) / n, 3),
            "hit@3": round(sum(bool(r and r <= 3) for r in ranks) / n, 3),
            "mrr": round(sum(1 / r for r in ranks if r) / n, 3)}


GRID = [dict(n_trees=t, learning_rate=0.1, max_leaf_nodes=l, min_samples_leaf=m, ndcg_at=k)
        for t in (50, 150) for l in (4, 8) for m in (5, 20) for k in (5, 10)]


def train(per_list: int = 60, folds: int = 5, seed: int = 0) -> dict:
    """Pick hyperparameters by grouped cross-validation on the dev queries (top-1 first, then
    MRR), train on all dev queries, save. The holdout queries are never read here."""
    import yaml
    with open(config.EVAL_PATH, encoding="utf-8") as f:
        dev = yaml.safe_load(f)["queries"]
    t0 = time.time()
    items = _query_data(dev, per_list)
    pool_recall = sum(it["y"].any() for it in items) / len(items)
    print(f"{len(items)} dev queries, {sum(len(it['y']) for it in items)} candidate chunks, "
          f"clause in pool for {pool_recall:.0%}")

    rng = np.random.default_rng(seed)
    fold_of = rng.permutation(len(items)) % folds
    results = []
    for params in GRID:
        ranks: list[int | None] = [None] * len(items)
        for f in range(folds):
            model = _fit([it for it, fo in zip(items, fold_of) if fo != f], params)
            for i, (it, fo) in enumerate(zip(items, fold_of)):
                if fo == f:
                    ranks[i] = _clause_rank(model, it)
        results.append((params, _metrics(ranks)))
    results.sort(key=lambda r: (-r[1]["hit@1"], -r[1]["mrr"], r[0]["n_trees"], r[0]["max_leaf_nodes"]))
    best_params, best_cv = results[0]
    print("cross-validated on dev (5 folds, out-of-fold):")
    for params, m in results[:5]:
        print(f"  {params} -> {m}")

    model = _fit(items, best_params)
    meta = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "features": FEATURES, "monotone": MONOTONE,
        "params": best_params, "cv_dev": best_cv, "per_list": per_list, "folds": folds,
        "train_queries": len(items), "pool_recall_dev": round(pool_recall, 3),
        "index": {k: store.index_meta().get(k) for k in ("built_at", "embed_model", "embed_dtype")},
        "train_seconds": round(time.time() - t0, 1),
    }
    config.LTR_MODEL_PATH.write_bytes(pickle.dumps({"model": model, "meta": meta}))
    config.LTR_META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    load.cache_clear()
    print(f"saved {config.LTR_MODEL_PATH} with {len(model.trees)} trees; params {best_params}")
    return meta


@functools.lru_cache(maxsize=1)
def load() -> tuple[LambdaMART, dict] | None:
    """The trained model and its metadata, or None if `python -m regs train-ltr` has not run.
    Pickle runs code on load: only load a file this project wrote."""
    if not config.LTR_MODEL_PATH.exists():
        return None
    blob = pickle.loads(config.LTR_MODEL_PATH.read_bytes())
    return blob["model"], blob["meta"]
