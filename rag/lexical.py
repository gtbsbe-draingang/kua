"""The lexical channel of the hybrid retrieval: Snowball-stemmed BM25 over the clause passages, with a small
glossary that adds the acts' wording to the query for the everyday words a video observation uses.

Pure Python and numpy, no model. The tokenizer, the stop words and the glossary come from the retrieval index of
our teammates (their stemming happens on the client for the same reason: the stem of a query word and of a passage
word must be the same string).
"""
from __future__ import annotations

import math
import re
import threading
from pathlib import Path

import numpy as np
import snowballstemmer
import yaml

GLOSSARY_PATH = Path(__file__).parent / "glossary.yaml"
K1 = 1.2
B = 0.75

# Snowball stemmer objects keep state between calls and are not thread-safe (concurrent requests raised
# IndexError inside stemWord for our teammates), and the server answers from a thread pool: one per thread.
_local = threading.local()

_TOKEN = re.compile(r"\d+(?:-\d+)?|[a-z]+|[а-яәғқңөұүһі]+")
_CYRILLIC = re.compile(r"[а-яәғқңөұүһі]")

STOP_WORDS = frozenset("""
и в во не на с со по к ко о об от до за из у для при а или либо но что это этот эта эти его ее их
как так же также если то ли бы был была были быть есть который которая которые которого которой
настоящих настоящим данный данного том числе ним них нему ней она он оно они
""".split())


def _stemmer():
    stemmer = getattr(_local, "russian", None)
    if stemmer is None:
        stemmer = _local.russian = snowballstemmer.stemmer("russian")
    return stemmer


def normalize(text: str) -> str:
    return text.lower().replace("ё", "е")


def tokens(text: str) -> list[str]:
    """Lowercase words, ё written as е, stop words dropped, Cyrillic words reduced to their Russian Snowball stem.
    Numbers (also "19-1") and Latin words stay as they are."""
    out = []
    for tok in _TOKEN.findall(normalize(text)):
        if tok in STOP_WORDS:
            continue
        out.append(_stemmer().stemWord(tok) if _CYRILLIC.match(tok) else tok)
    return out


class BM25:
    """Okapi BM25 over tokenised documents: idf = ln(1 + (N - df + 0.5) / (df + 0.5)), which is never negative, and
    tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avg len)). Each distinct query term counts once. The per-document
    weight of every term is computed here, so a query only adds up the arrays of its terms."""

    def __init__(self, docs: list[list[str]], k1: float = K1, b: float = B):
        self.n = len(docs)
        lengths = np.array([len(d) for d in docs], dtype=np.float64)
        avg = float(lengths.mean()) if self.n and lengths.sum() else 1.0
        postings: dict[str, dict[int, int]] = {}
        for i, doc in enumerate(docs):
            for term in doc:
                counts = postings.setdefault(term, {})
                counts[i] = counts.get(i, 0) + 1
        self._weights: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for term, counts in postings.items():
            idx = np.fromiter(counts, dtype=np.int64, count=len(counts))
            tf = np.fromiter(counts.values(), dtype=np.float64, count=len(counts))
            idf = math.log(1 + (self.n - len(counts) + 0.5) / (len(counts) + 0.5))
            norm = k1 * (1 - b + b * lengths[idx] / avg)
            self._weights[term] = (idx, idf * tf * (k1 + 1) / (tf + norm))

    def scores(self, query: list[str]) -> np.ndarray:
        """One score per document, 0 where no query term occurs."""
        out = np.zeros(self.n, dtype=np.float64)
        for term in set(query):
            entry = self._weights.get(term)
            if entry is not None:
                out[entry[0]] += entry[1]
        return out


def load_glossary(path: Path = GLOSSARY_PATH) -> tuple[tuple[re.Pattern, str], ...]:
    """(pattern, words to add) per glossary entry: the pattern matches a normalised query that has a word starting
    with any of the entry's `when` prefixes. Raises ValueError for a file that is not a list of {when, add}."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    terms = data.get("terms") if isinstance(data, dict) else None
    if not isinstance(terms, list):
        raise ValueError(f"{Path(path).name}: expected a mapping with a list `terms`")
    entries = []
    for i, term in enumerate(terms):
        when = term.get("when") if isinstance(term, dict) else None
        add = term.get("add") if isinstance(term, dict) else None
        if (not isinstance(when, list) or not when or not all(isinstance(w, str) and w.strip() for w in when)
                or not isinstance(add, str) or not add.strip()):
            raise ValueError(f"{Path(path).name}: term {i} needs a non-empty `when` list of strings and an `add` string")
        prefixes = "|".join(re.escape(normalize(w.strip())) for w in when)
        entries.append((re.compile(r"(?<![a-zа-я])(?:" + prefixes + ")"), add.strip()))
    return tuple(entries)


def expand(query: str, glossary: tuple[tuple[re.Pattern, str], ...]) -> list[str]:
    """The acts' wording to add to the lexical query for the everyday words in query."""
    q = normalize(query)
    return [add for pattern, add in glossary if pattern.search(q)]


class Lexicon:
    """BM25 over the passages (one document per passage, in the order given), queried through the glossary."""

    def __init__(self, texts: list[str], glossary: Path = GLOSSARY_PATH):
        self._glossary = load_glossary(glossary)
        self._bm25 = BM25([tokens(t) for t in texts])

    def expand(self, query: str) -> list[str]:
        return expand(query, self._glossary)

    def scores(self, query: str) -> np.ndarray:
        """The BM25 score of every passage for query and the glossary words its everyday words bring in."""
        return self._bm25.scores(tokens(" ".join([query, *self.expand(query)])))
