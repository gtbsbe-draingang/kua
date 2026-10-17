import logging
import os
import subprocess
import sys
import zlib

import numpy as np
import pytest

from rag import corpus, lexical, retrieve
from rag.corpus import Clause


COMMITTED_DEFAULT = retrieve.HYBRID_DEFAULT  # before the fixture below replaces it


@pytest.fixture(autouse=True)
def dense_by_default(monkeypatch):
    """The dense tests below do not depend on the environment's choice; the hybrid ones ask for hybrid=True."""
    monkeypatch.delenv("RAG_HYBRID", raising=False)
    monkeypatch.setattr(retrieve, "HYBRID_DEFAULT", "0")


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


# ---- hybrid: dense + BM25 with the glossary, fused with reciprocal rank fusion ---------------------------------

class Table:
    """A dense stand-in: the vector of every text is looked up, so a test says exactly how the dense side ranks."""

    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        return np.stack([self.vectors[t] for t in texts])


def rrf(*ranks, k=60):
    return sum(1 / (k + r) for r in ranks)


def test_hybrid_lexical_channel_lifts_a_clause_the_dense_side_ranks_last():
    # dense order: A:1 > A:2 > A:3 > A:4; only A:4 shares words with the query (the glossary adds the acts' wording)
    embed = Table({"посторонний": unit(1, 0, 0), "шум раз": unit(1, 0.05, 0), "шум два": unit(1, 0.1, 0),
                   "шум три": unit(1, 0.2, 0), "посторонние лица в помещении кассы": unit(1, 1, 0)})
    clauses = [clause("A:1", "шум раз"), clause("A:2", "шум два"), clause("A:3", "шум три"),
               clause("A:4", "посторонние лица в помещении кассы")]
    dense = retrieve.Index(clauses, embed_fn=embed, hybrid=False)
    hybrid = retrieve.Index(clauses, embed_fn=embed, hybrid=True)
    assert [h.clause_id for h in dense.search("посторонний")] == ["A:1", "A:2", "A:3", "A:4"]
    hits = hybrid.search("посторонний")
    assert [h.clause_id for h in hits] == ["A:4", "A:1", "A:2", "A:3"]
    assert hits[0].score == pytest.approx(rrf(4, 1))  # rank 4 in the dense ranking, rank 1 in the lexical one
    assert [h.score for h in hits[1:]] == pytest.approx([rrf(1), rrf(2), rrf(3)])


def test_hybrid_agreement_of_both_channels_beats_the_dense_top_alone():
    # dense order A:1 > A:2 > A:3; the lexical channel knows only A:2
    embed = Table({"сейф": unit(1, 0, 0), "ящик": unit(1, 0.01, 0), "касса сейф": unit(1, 0.2, 0),
                   "прочее": unit(1, 1, 0)})
    clauses = [clause("A:1", "ящик"), clause("A:2", "касса сейф"), clause("A:3", "прочее")]
    hits = retrieve.Index(clauses, embed_fn=embed, hybrid=True).search("сейф")
    assert [h.clause_id for h in hits] == ["A:2", "A:1", "A:3"]
    assert [h.score for h in hits] == pytest.approx([rrf(2, 1), rrf(1), rrf(3)])


def test_hybrid_part_comes_from_the_channel_that_ranks_the_clause_higher():
    embed = Table({
        "телефон": unit(1, 0, 0),
        "шум": unit(1, 0.01, 0),  # a distractor that is the dense top
        "общие положения": unit(1, 0.3, 0),  # dense picks part 0 of B:1 ...
        "средства связи запрещены": unit(1, 1, 0),  # ... while the glossary points at part 1
    })
    clauses = [clause("A:1", "шум"), clause("B:1", "общие положения", "средства связи запрещены")]
    hit = {h.clause_id: h for h in retrieve.Index(clauses, embed_fn=embed, hybrid=True).search("телефон")}
    assert hit["B:1"].part == 1  # lexical rank 1 beats dense rank 2
    assert hit["A:1"].part == 0
    dense_only = {h.clause_id: h for h in retrieve.Index(clauses, embed_fn=embed, hybrid=False).search("телефон")}
    assert dense_only["B:1"].part == 0
    # equal ranks go to the dense part
    embed.vectors["общие положения"] = unit(1, 0, 0)
    embed.vectors["шум"] = unit(0, 1, 0)
    same = {h.clause_id: h for h in retrieve.Index(clauses, embed_fn=embed, hybrid=True).search("телефон")}
    assert same["B:1"].part == 0


def test_hybrid_gives_one_hit_per_clause_and_part_indexes_the_clause():
    embed = Table({"касса": unit(1, 0, 0), "касса раз": unit(1, 0.1, 0), "касса два": unit(1, 0.2, 0),
                   "касса три": unit(1, 0.3, 0), "прочее": unit(0, 1, 0)})
    clauses = [clause("A:1", "касса раз", "касса два", "касса три"), clause("A:2", "прочее")]
    hits = retrieve.Index(clauses, embed_fn=embed, hybrid=True).search("касса", k=8)
    assert [h.clause_id for h in hits] == ["A:1", "A:2"]
    for h in hits:
        assert 0 <= h.part < len(next(c for c in clauses if c.id == h.clause_id).parts)


def test_hybrid_part_index_survives_empty_parts():
    embed = Table({"телефон": unit(1, 0, 0), "шум": unit(0, 1, 0), "средства связи": unit(1, 1, 0)})
    clauses = [clause("A:1", "шум"), clause("B:1", "", "средства связи")]
    hits = retrieve.Index(clauses, embed_fn=embed, hybrid=True).search("телефон")
    best = next(h for h in hits if h.clause_id == "B:1")
    assert best.part == 1 and clauses[1].parts[best.part] == "средства связи"


def test_hybrid_ties_go_to_corpus_order_and_k_limits():
    embed = Table({"q": unit(1, 0, 0), "один": unit(0, 1, 0), "два": unit(0, 1, 0), "три": unit(0, 0, 1)})
    clauses = [clause("A:1", "один"), clause("A:2", "два"), clause("A:3", "три")]
    idx = retrieve.Index(clauses, embed_fn=embed, hybrid=True)
    assert [h.clause_id for h in idx.search("q")] == ["A:1", "A:2", "A:3"]
    assert [h.clause_id for h in idx.search("q", k=2)] == ["A:1", "A:2"]


def test_hybrid_embeds_once_per_batch_and_skips_empty_texts():
    embed = Table({"касса": unit(1, 0, 0), "сейф": unit(0, 1, 0), "касса сейф": unit(1, 1, 0)})
    idx = retrieve.Index([clause("A:1", "касса сейф")], embed_fn=embed, hybrid=True)
    n = len(embed.calls)
    out = idx.search_many(["касса", " ", "сейф"])
    assert len(embed.calls) == n + 1 and embed.calls[-1] == ["касса", "сейф"]
    assert out[1] == [] and out[0][0].clause_id == out[2][0].clause_id == "A:1"
    assert idx.search("") == [] and idx.search_many([" "]) == [[]] and len(embed.calls) == n + 1


def test_hybrid_flag_follows_the_argument_the_environment_and_the_default(monkeypatch):
    texts = Table({"касса": unit(1, 0, 0)})

    def built(**kw):
        return retrieve.Index([clause("A:1", "касса")], embed_fn=texts, **kw)

    assert built().hybrid is False  # the fixture sets the module default to dense
    monkeypatch.setattr(retrieve, "HYBRID_DEFAULT", "1")
    assert built().hybrid is True
    for off in ("0", "false", "OFF", "no"):
        monkeypatch.setenv("RAG_HYBRID", off)
        assert built().hybrid is False, off
    for on in ("1", "true", "yes", "2"):
        monkeypatch.setenv("RAG_HYBRID", on)
        assert built().hybrid is True, on
    monkeypatch.setenv("RAG_HYBRID", "")
    assert built().hybrid is True  # an empty value is the default
    monkeypatch.setenv("RAG_HYBRID", "0")
    assert built(hybrid=True).hybrid is True  # the argument wins


def test_hybrid_is_the_default_when_the_environment_is_silent():
    # the gate of the brief: hybrid recall@8 on the 15 queries was at least dense-only's, so the default is on
    assert COMMITTED_DEFAULT == "1"
    code = ("import numpy as np\n"
            "from rag import retrieve\n"
            "from rag.corpus import Clause\n"
            "c = Clause('A:1', 'A', 's', 'd', 'a', '1', 'ch', ('касса',), 'u')\n"
            "idx = retrieve.Index([c], embed_fn=lambda t: np.ones((len(t), 3), dtype=np.float32))\n"
            "assert idx.hybrid is True\n")
    env = {k: v for k, v in os.environ.items() if k != "RAG_HYBRID"}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr


def test_lexical_build_failure_falls_back_to_dense_and_logs(monkeypatch, caplog):
    def broken(*a, **k):
        raise RuntimeError("no glossary")

    monkeypatch.setattr(lexical, "Lexicon", broken)
    embed = Table({"q": unit(1, 0, 0), "a": unit(1, 0.1, 0), "b": unit(0, 1, 0)})
    clauses = [clause("A:1", "a"), clause("A:2", "b")]
    with caplog.at_level(logging.WARNING, logger="rag.retrieve"):
        idx = retrieve.Index(clauses, embed_fn=embed, hybrid=True)
    assert idx.hybrid is False
    assert any("lexical" in r.getMessage().lower() and r.levelno >= logging.WARNING for r in caplog.records)
    assert idx.search("q") == retrieve.Index(clauses, embed_fn=embed, hybrid=False).search("q")


def test_missing_stemmer_package_falls_back_to_dense():
    code = ("import sys; sys.modules['snowballstemmer'] = None\n"
            "import numpy as np\n"
            "from rag import retrieve\n"
            "from rag.corpus import Clause\n"
            "c = Clause('A:1', 'A', 's', 'd', 'a', '1', 'ch', ('касса',), 'u')\n"
            "idx = retrieve.Index([c], embed_fn=lambda t: np.ones((len(t), 3), dtype=np.float32), hybrid=True)\n"
            "assert idx.hybrid is False and [h.clause_id for h in idx.search('касса')] == ['A:1']\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "lexical" in r.stderr.lower()  # logged


def test_lexical_failure_at_query_time_serves_the_dense_ranking(monkeypatch, caplog):
    embed = Table({"q": unit(1, 0, 0), "a": unit(1, 0.1, 0), "b": unit(0, 1, 0)})
    clauses = [clause("A:1", "a"), clause("A:2", "b")]
    idx = retrieve.Index(clauses, embed_fn=embed, hybrid=True)

    def boom(query):
        raise ValueError("stemmer")

    monkeypatch.setattr(idx._lexicon, "scores", boom)
    with caplog.at_level(logging.WARNING, logger="rag.retrieve"):
        hits = idx.search("q")
    assert idx.hybrid is True  # one bad query does not switch the channel off
    assert [h.clause_id for h in hits] == ["A:1", "A:2"]
    assert hits == retrieve.Index(clauses, embed_fn=embed, hybrid=False).search("q")
    assert any("lexical" in r.getMessage().lower() for r in caplog.records)


def test_lexical_hits_rank_only_clauses_that_match():
    embed = Table({"касса": unit(1, 0, 0), "касса сейф": unit(1, 1, 0), "прочее": unit(0, 1, 0)})
    idx = retrieve.Index([clause("A:1", "прочее"), clause("A:2", "касса сейф")], embed_fn=embed, hybrid=True)
    assert [h.clause_id for h in idx.lexical_hits("касса")] == ["A:2"]
    assert idx.lexical_hits("зебра") == [] and idx.lexical_hits("  ") == []
    assert retrieve.Index([clause("A:1", "касса")], embed_fn=embed, hybrid=False).lexical_hits("касса") == []


def hashing_embed(texts):
    """A deterministic stand-in for bge-m3 on the real corpus: L2-normalised hashed bag of stems, 1024-d."""
    out = np.zeros((len(texts), 1024), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in lexical.tokens(t):
            out[i, zlib.crc32(tok.encode()) % 1024] += 1.0
        out[i, 0] += 1e-3  # never a zero vector
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def test_hybrid_on_the_real_corpus_never_returns_a_suspended_or_repealed_clause():
    in_force = corpus.load_clauses()
    everything = corpus.load_clauses(in_force_only=False)
    ids = {c.id for c in in_force}
    out_of_force = [c for c in everything if c.id not in ids]
    assert {c.id for c in out_of_force} >= {"V2000020075:11", "V2000020075:14", "V2000020075:16", "V1900019680:95"}
    idx = retrieve.Index(in_force, embed_fn=hashing_embed, hybrid=True)
    assert idx.hybrid
    # the text of every clause that is out of force, then the observations the pipeline sees
    queries = [" ".join(c.parts) for c in out_of_force]
    queries += ["каждая касса имеет отдельный вход с закрывающейся на замок дверью", "дверь кассы открыта"]
    for hits in idx.search_many(queries, k=20):
        assert 0 < len(hits) <= 20
        assert {h.clause_id for h in hits} <= ids
        for h in hits:
            assert idx.by_id[h.clause_id].parts[h.part]  # Hit.part indexes the clause's parts, never an empty one
    # the same holds for the lexical ranking on its own
    for q in queries:
        assert {h.clause_id for h in idx.lexical_hits(q)} <= ids
