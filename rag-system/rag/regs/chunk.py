"""Turn parsed clauses into retrieval chunks.

One clause (пункт) is one chunk, because a clause is the unit a citation names ("п. 22").
A clause over MAX_CHUNK_WORDS is split at subclause boundaries, and every part repeats the
clause's opening line, so a part like "2) принимает банкноты с полистным пересчетом..." still
says who must do it. Clauses are never merged, so a hit always maps to one clause.

A clause that is a list of duties ("31. Кассовый работник ... осуществляет следующие действия:
1) ... 6) ...") also gets one extra chunk per subclause: the opening line plus that subclause.
Otherwise the one duty a query is about is diluted by the other five. search() folds the clause
and its subclause hits back into one result, labelled with the subclause ("п. 31, пп. 4)").

Each chunk keeps three texts:
  text        the clause wording, verbatim. This is what gets quoted.
  embed_text  "<act>. <chapter>. <paragraph>. п. N." + text, for the dense vector
  lex         embed_text normalised and stemmed, for BM25
"""
import json
import re

from . import config
from .lexical import to_lex
from .parse import load_parsed

SUBCLAUSE_RE = re.compile(r"^(\d{1,2}(?:-\d)?)\)")
SENTENCE_RE = re.compile(r"(?<=[.;:])\s+(?=[А-ЯA-Z0-9])")


def _words(text: str) -> int:
    return len(text.split())


def _split_long_line(line: str, limit: int) -> list[str]:
    """Split one very long paragraph at sentence ends into pieces of at most `limit` words."""
    pieces, cur = [], []
    for sent in SENTENCE_RE.split(line):
        if cur and _words(" ".join(cur + [sent])) > limit:
            pieces.append(" ".join(cur))
            cur = []
        cur.append(sent)
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def _groups(lead: str, rest: list[str], limit: int) -> list[list[str]]:
    """Pack the lines after the opening line into groups that fit next to the opening line."""
    budget = max(limit - _words(lead), limit // 2)
    lines = [p for line in rest for p in (_split_long_line(line, budget) if _words(line) > budget else [line])]
    groups, cur = [], []
    for line in lines:
        if cur and _words(" ".join(cur + [line])) > budget:
            groups.append(cur)
            cur = []
        cur.append(line)
    if cur:
        groups.append(cur)
    return groups


def _subclause_range(lines: list[str]) -> str:
    nums = [m.group(1) for m in (SUBCLAUSE_RE.match(line) for line in lines) if m]
    if not nums:
        return ""
    return f"пп. {nums[0]})" if len(nums) == 1 else f"пп. {nums[0]})–{nums[-1]})"


def passage(doc_short: str, chapter: str, section: str, label: str, text: str) -> str:
    """The text a model sees for a chunk: where it sits in the act, then the verbatim wording.
    Used for the dense embedding here and for the reranker in search.py."""
    header = ". ".join(x for x in (doc_short, chapter, section) if x)
    return f"{header}. {label}.\n{text}"


def _subclauses(lines: list[str]) -> list[tuple[str, list[str]]]:
    """[(number, lines)] for each "N) ..." subclause after the opening line, with any
    continuation paragraphs that follow it. Empty if the clause is not a list."""
    subs: list[tuple[str, list[str]]] = []
    for line in lines[1:]:
        m = SUBCLAUSE_RE.match(line)
        if m:
            subs.append((m.group(1), [line]))
        elif subs:
            subs[-1][1].append(line)
        else:
            return []  # a paragraph before the list: not a plain list of duties
    return subs if len(subs) >= 2 else []


def clause_chunks(doc: dict, clause: dict, limit: int = config.MAX_CHUNK_WORDS) -> list[dict]:
    lines = clause["lines"]
    if _words(" ".join(lines)) <= limit:
        parts = [lines]
    elif len(lines) == 1:
        parts = [[p] for p in _split_long_line(lines[0], limit)]
    else:
        parts = [[lines[0]] + group for group in _groups(lines[0], lines[1:], limit)]

    url = doc["canonical_url"] + (f"#{clause['anchor']}" if clause["anchor"] else "")
    base_id = f"{doc['code']}:{clause['num']}"

    def make(chunk_id: str, label: str, part_lines: list[str], part: int, parts: int,
             granularity: str, seq: int) -> dict:
        text = "\n".join(part_lines)
        embed_text = passage(doc["short"], clause["chapter"], clause["section"], label, text)
        return {
            "id": chunk_id, "doc_code": doc["code"], "doc_short": doc["short"],
            "clause": clause["num"], "part": part, "parts": parts, "clause_label": label,
            "granularity": granularity, "chapter": clause["chapter"], "section": clause["section"],
            "anchor": clause["anchor"] or "", "url": url, "status": clause["status"],
            "notes": "\n".join(clause["notes"]), "amended_by": "; ".join(clause["amendments"]),
            "text": text, "embed_text": embed_text, "lex": to_lex(embed_text), "seq": seq,
        }

    out = []
    for i, part in enumerate(parts):
        label = f"п. {clause['num']}"
        sub = _subclause_range(part[1:]) if len(parts) > 1 else ""
        if sub:
            label += f", {sub}"
        elif len(parts) > 1:
            label += f" (часть {i + 1} из {len(parts)})"
        chunk_id = base_id + (f":{i + 1}" if len(parts) > 1 else "")
        out.append(make(chunk_id, label, part, i + 1, len(parts), "clause", clause["seq"] * 100 + i))
    for num, sub_lines in _subclauses(lines):
        out.append(make(f"{base_id}:pp{num}", f"п. {clause['num']}, пп. {num})", [lines[0]] + sub_lines,
                        1, 1, "subclause", clause["seq"] * 100 + 50))
    return out


def build_chunks() -> list[dict]:
    chunks = []
    for doc in load_parsed():
        for clause in doc["clauses"]:
            chunks.extend(clause_chunks(doc, clause))
    config.CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    split = sum(1 for c in chunks if c["parts"] > 1)
    subs = sum(1 for c in chunks if c["granularity"] == "subclause")
    longest = max(_words(c["embed_text"]) for c in chunks)
    print(f"chunks: {len(chunks)} ({len(chunks) - subs} clause-level, {split} of them parts of split "
          f"clauses; {subs} subclause-level), longest {longest} words")
    return chunks


def load_chunks() -> list[dict]:
    with open(config.CHUNKS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]
