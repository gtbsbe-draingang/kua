"""rag/lexical.py: the Russian tokenizer, the BM25 scorer and the glossary that widens the lexical query."""
import math
import textwrap
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from rag import lexical


def test_tokens_stem_fold_yo_and_drop_stop_words():
    assert lexical.tokens("Кассира") == lexical.tokens("кассир") == lexical.tokens("КАССИРОМ")
    assert lexical.tokens("посторонних лиц") == lexical.tokens("посторонние лица")
    assert lexical.tokens("отошёл") == lexical.tokens("отошел")
    # stop words and punctuation go; numbers and Latin words stay as they are
    assert lexical.tokens("и в на, или!") == []
    assert lexical.tokens("пункт 19-1 USB") == [lexical.tokens("пункт")[0], "19-1", "usb"]
    assert lexical.tokens("") == []


def test_tokens_are_stable_across_threads():
    # Snowball stemmer objects keep state and are not thread-safe: every thread needs its own
    words = "кассовый работник покидать рабочее место посторонние лица средства связи сейфовая комната".split() * 40
    serial = [lexical.tokens(w) for w in words]
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(lexical.tokens, words)) == serial


def test_bm25_matches_the_formula():
    docs = [["a", "b"], ["a"], ["c"]]
    bm = lexical.BM25(docs, k1=1.2, b=0.75)
    idf = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))
    avg = 4 / 3
    want = idf * 1 * (1.2 + 1) / (1 + 1.2 * (1 - 0.75 + 0.75 * 2 / avg))
    got = bm.scores(["b"])
    assert got == pytest.approx([want, 0.0, 0.0])
    # the term in two documents is rarer-than-not: lower idf, and the shorter document scores higher
    s = bm.scores(["a"])
    assert s[1] > s[0] > 0 and s[2] == 0


def test_bm25_counts_a_repeated_query_term_once():
    bm = lexical.BM25([["a", "b"], ["b"]])
    assert bm.scores(["a", "a", "a"]) == pytest.approx(bm.scores(["a"]))


def test_bm25_unknown_terms_and_empty_inputs():
    bm = lexical.BM25([["a"], []])
    assert list(bm.scores(["zzz"])) == [0.0, 0.0]
    assert list(bm.scores([])) == [0.0, 0.0]
    empty = lexical.BM25([])
    assert len(empty.scores(["a"])) == 0


def test_bm25_term_frequency_saturates():
    bm = lexical.BM25([["a"] * 1 + ["x"] * 9, ["a"] * 5 + ["x"] * 5, ["z"] * 10])
    s = bm.scores(["a"])
    assert s[1] > s[0]
    assert s[1] < 5 * s[0]  # five occurrences are worth far less than five times one


GLOSSARY = textwrap.dedent("""\
    version: 1
    terms:
      - when: [кассир, teller]
        add: "кассовый работник"
      - when: [на виду, смартфон]
        add: "средства связи"
""")


def write_glossary(tmp_path, text=GLOSSARY, name="glossary.yaml"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_glossary_expands_word_prefixes(tmp_path):
    g = lexical.load_glossary(write_glossary(tmp_path))
    assert lexical.expand("Кассир берёт деньги", g) == ["кассовый работник"]
    assert lexical.expand("кассирша", g) == ["кассовый работник"]  # a prefix of the word
    assert lexical.expand("инкассатор", g) == []  # not at a word start
    assert lexical.expand("Teller", g) == ["кассовый работник"]
    assert lexical.expand("деньги лежат на виду", g) == ["средства связи"]
    assert lexical.expand("кассир со смартфоном", g) == ["кассовый работник", "средства связи"]
    assert lexical.expand("", g) == []


def test_glossary_accepts_an_empty_term_list(tmp_path):
    assert lexical.load_glossary(write_glossary(tmp_path, "version: 1\nterms: []\n")) == ()


@pytest.mark.parametrize("text", [
    "- not a mapping\n",
    "terms: {a: 1}\n",
    "terms:\n  - when: [а]\n",  # no add
    "terms:\n  - add: x\n",  # no when
    "terms:\n  - when: []\n    add: x\n",
    "terms:\n  - when: [1]\n    add: x\n",
    "terms:\n  - when: [а]\n    add: ''\n",
])
def test_glossary_rejects_malformed_files(tmp_path, text):
    with pytest.raises(ValueError):
        lexical.load_glossary(write_glossary(tmp_path, text))


def test_shipped_glossary_is_valid():
    g = lexical.load_glossary()
    assert len(g) >= 10
    added = " ".join(lexical.expand("Кассир достал телефон, посторонний вошёл в кассовое помещение", g))
    for word in ("кассовый работник", "средства связи", "третьи лица", "кассовый узел"):
        assert word in added


def test_lexicon_scores_use_the_glossary(tmp_path):
    g = write_glossary(tmp_path)
    texts = ["Кассовые работники не покидают рабочее место", "Деньги хранятся в сейфе", "Работник обязан быть в форме"]
    plain = lexical.Lexicon(texts, glossary=write_glossary(tmp_path, "terms: []\n", "none.yaml"))
    wide = lexical.Lexicon(texts, glossary=g)
    # "кассир" never occurs in the passages, "кассовый работник" does
    assert not plain.scores("кассир").any()
    s = wide.scores("кассир")
    assert s.argmax() == 0 and s[0] > 0 and s[1] == 0
    assert wide.scores("   ").tolist() == [0.0, 0.0, 0.0]
    assert wide.scores("zzz").tolist() == [0.0, 0.0, 0.0]


def test_lexicon_scores_have_one_value_per_passage():
    lex = lexical.Lexicon(["раз два", "три четыре", "пять"])
    out = lex.scores("три")
    assert isinstance(out, np.ndarray) and out.shape == (3,) and out.argmax() == 1
