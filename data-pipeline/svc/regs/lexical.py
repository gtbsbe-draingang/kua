"""Text for the BM25 field: lowercased, ё -> е, stop words removed, Snowball-stemmed.

Stemming happens here, on the client, because Milvus Lite accepts a "stemmer" analyzer filter but
does not apply it (checked with milvus-lite 3.2.2: "видеонаблюдение" did not match
"видеонаблюдения"). Stemming the same way at index and query time gives the same results on
Milvus Lite and on a Milvus server, whose analyzer then only needs to split on whitespace.
"""
import re

import snowballstemmer

_RU = snowballstemmer.stemmer("russian")
_EN = snowballstemmer.stemmer("english")
_TOKEN = re.compile(r"\d+(?:-\d+)?|[a-z]+|[а-яәғқңөұүһі]+")
_CYRILLIC = re.compile(r"[а-яәғқңөұүһі]")

STOP_WORDS = set("""
и в во не на с со по к ко о об от до за из у для при а или либо но что это этот эта эти его ее их
как так же также если то ли бы был была были быть есть который которая которые которого которой
настоящих настоящим данный данного том том числе их его ее ним них нему ней она он оно они
the a an of to in on and or for with by is are be at as from that this it
""".split())


def normalize(text: str) -> str:
    return text.lower().replace("ё", "е")


def tokens(text: str) -> list[str]:
    out = []
    for tok in _TOKEN.findall(normalize(text)):
        if tok in STOP_WORDS:
            continue
        if tok[0].isdigit():
            out.append(tok)
        elif _CYRILLIC.match(tok):
            out.append(_RU.stemWord(tok))
        else:
            out.append(_EN.stemWord(tok))
    return out


def to_lex(text: str) -> str:
    return " ".join(tokens(text))
