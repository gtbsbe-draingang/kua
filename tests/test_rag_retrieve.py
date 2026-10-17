import subprocess
import sys

import numpy as np

from rag import retrieve
from rag.corpus import Clause


def clause(cid, *parts):
    code, num = cid.split(":")
    return Clause(cid, code, "s", "d", "a", num, "ch", tuple(parts), "u")


def unit(*v):
    a = np.array(v, dtype=np.float32)
    return a / np.linalg.norm(a)


VECS = {
    "q": unit(1, 0, 0),
    "strong": unit(1, 0.05, 0),
    "med1": unit(1, 1, 0),
    "med2": unit(1, 0, 1),
    "far": unit(0, 1, 0),
}


class Fake:
    def __init__(self):
        self.calls = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        return np.stack([VECS[t] for t in texts])


def test_search_ranks_by_max_passage():
    fake = Fake()
    idx = retrieve.Index([clause("A:1", "far", "strong"), clause("A:2", "med1", "med2")], embed_fn=fake)
    hits = idx.search("q")
    assert [h.clause_id for h in hits] == ["A:1", "A:2"]
    assert hits[0].part == 1
    assert hits[0].score > hits[1].score
    assert len(fake.calls) == 2  # construction, then the query
    assert idx.by_id["A:2"] is idx.clauses[1]


def test_search_ties_go_to_corpus_order_and_k_limits():
    idx = retrieve.Index([clause("A:1", "med1"), clause("A:2", "med1"), clause("A:3", "far")], embed_fn=Fake())
    assert [h.clause_id for h in idx.search("q", k=2)] == ["A:1", "A:2"]


def test_search_empty_text_skips_embedding():
    fake = Fake()
    idx = retrieve.Index([clause("A:1", "strong")], embed_fn=fake)
    n = len(fake.calls)
    assert idx.search("") == [] and idx.search("  \n ") == []
    assert len(fake.calls) == n


def test_search_many_embeds_once():
    fake = Fake()
    idx = retrieve.Index([clause("A:1", "strong"), clause("A:2", "far")], embed_fn=fake)
    n = len(fake.calls)
    out = idx.search_many(["q", " ", "far"])
    assert len(fake.calls) == n + 1 and fake.calls[-1] == ["q", "far"]
    assert out[1] == []
    assert out[0][0].clause_id == "A:1" and out[2][0].clause_id == "A:2"
    assert idx.search_many([" "]) == [[]] and len(fake.calls) == n + 1


def test_retrieve_imports_without_torch():
    code = ("import sys; import rag.retrieve; "
            "bad = [m for m in ('torch', 'transformers') if m in sys.modules]; "
            "sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
