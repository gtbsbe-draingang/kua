"""The rule and clause matcher: which clauses to offer, the LLM prompt, the answer schema and the answer parser.

The model chooses a rule, a clause id and P(violation); it never writes a citation. The RAG server fills that from
the corpus by clause id."""
from __future__ import annotations

import json
import math
import re

from pipeline.clients import CANDIDATES_LABEL, JUDGE_SYSTEM, NO_OBSERVATION, OBSERVATION_LABEL
from rag.corpus import Clause, find_quote
from rag.retrieve import Hit

CLAUSE_MAX = 1500  # characters of a clause's text in the prompt
QUOTE_MAX = 600  # characters of a citation quote
PROMPT_MAX = 20000  # characters of the user prompt the iv3 /generate endpoint accepts
NONE = "none"

MATCH_SYSTEM = JUDGE_SYSTEM + """
clause_id must be one of the listed clause ids, or none when no listed clause states the requirement that was broken.
violation: your probability from 0 to 1 that the observation shows a violation of rule_id at all; near 0 when it describes normal work.
Answer with one JSON object only."""
CLAUSES_LABEL = "\n\nRegulation clauses, each headed by its clause_id in brackets:\n"


def answer_schema(rule_ids: list[str], clause_ids: list[str]) -> dict:
    unit = {"type": "number", "minimum": 0, "maximum": 1}
    return {"type": "object",
            "properties": {"violation": dict(unit),
                           "rule_id": {"type": "string", "enum": list(rule_ids)},
                           "clause_id": {"type": "string", "enum": list(clause_ids) + [NONE]},
                           "justification": {"type": "string"},
                           "confidence": dict(unit)},
            "required": ["violation", "rule_id", "clause_id", "justification", "confidence"],
            "additionalProperties": False}


def cut(text: str, limit: int) -> str:
    """text when it has at most limit characters; else its prefix up to the last space at or before limit, plus
    "…" (a hard cut at limit when there is no space)."""
    if len(text) <= limit:
        return text
    at = text.rfind(" ", 0, limit + 1)
    return (text[:at].rstrip() if at > 0 else text[:limit]) + "…"


def clause_text(c: Clause) -> str:
    return " ".join(p for p in c.parts if p)


def clause_candidates(hits: list[Hit], candidates: list[dict], index) -> list[tuple[Clause, int | None]]:
    """The clauses to offer: first the clause of each candidate rule's citation quote (part None), then the
    retrieval hits not listed yet (with the hit's part), each clause once, in order. A quote that is empty or not
    in the corpus (a suspended clause, for one) is skipped."""
    out: list[tuple[Clause, int | None]] = []
    seen: set[str] = set()
    for cand in candidates:
        for cit in cand.get("citations") or []:
            quote = cit.get("quote") if isinstance(cit, dict) else None
            if not isinstance(quote, str) or not quote.strip():
                continue
            c = find_quote(index.clauses, quote)
            if c is not None and c.id not in seen:
                seen.add(c.id)
                out.append((c, None))
    for h in hits:
        c = index.by_id.get(h.clause_id)
        if c is not None and c.id not in seen:
            seen.add(c.id)
            out.append((c, h.part))
    return out


def build_prompt(observation: str, candidates: list[dict],
                 clauses: list[tuple[Clause, int | None]]) -> tuple[str, str]:
    """(system, user). The user text frames the observation and the candidate rules as judge_request does, then
    lists each clause as "[<id>] <short>, п. <num>" and its text, cut to CLAUSE_MAX characters."""
    blocks = [f"[{c.id}] {c.short}, п. {c.num}\n{cut(clause_text(c), CLAUSE_MAX)}" for c, _ in clauses]
    user = (f"{OBSERVATION_LABEL}{observation.strip() or NO_OBSERVATION}"
            f"{CANDIDATES_LABEL}{json.dumps(candidates, ensure_ascii=False, indent=1)}"
            f"{CLAUSES_LABEL}" + ("\n\n".join(blocks) if blocks else "(none)"))
    return MATCH_SYSTEM, user


def fit_clauses(observation: str, candidates: list[dict], clauses: list[tuple[Clause, int | None]],
                limit: int = PROMPT_MAX) -> list[tuple[Clause, int | None]]:
    """clauses without the last retrieved ones while the user prompt is longer than limit. Catalogue clauses
    (part None, listed first) always stay, so the result can still be over the limit."""
    out = list(clauses)
    while out and out[-1][1] is not None and len(build_prompt(observation, candidates, out)[1]) > limit:
        out.pop()
    return out


def judge_prompt(observation: str, candidates: list[dict], hits: list[Hit],
                 index) -> tuple[str, str, list[tuple[Clause, int | None]]]:
    """What the matcher sends the LLM for one judge request: clause_candidates over the retrieval hits and the
    candidates' catalogue quotes, fit_clauses to the /generate limit, then build_prompt. Returns (system, user,
    the clauses offered). The RAG server and tools/rag_eval.py both call this, so their prompts cannot drift."""
    clauses = fit_clauses(observation, candidates, clause_candidates(hits, candidates, index))
    system, user = build_prompt(observation, candidates, clauses)
    return system, user, clauses


def _first_object(text: str) -> dict | None:
    """The first JSON object in text (surrounding prose or a ```json fence allowed)."""
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text, m.start())
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _pick(x, ids: list[str]) -> str | None:
    """x as one of ids: surrounding spaces, quotes and brackets removed, compared exactly, then ignoring case."""
    if not isinstance(x, str):
        return None
    s = x.strip().strip("[]`'\"«» ")
    if s in ids:
        return s
    return {i.lower(): i for i in ids}.get(s.lower())


def _unit(x) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        return None
    return min(1.0, max(0.0, float(x)))


def parse_answer(text: str, rule_ids: list[str], clause_ids: list[str]) -> dict | None:
    """{violation, rule_id, clause_id, justification, confidence} from the model's text, or None when there is no
    JSON object or its rule_id is not one of rule_ids. An unknown clause_id, or none, becomes None; violation and
    confidence are clipped to [0, 1], None when missing or not numbers."""
    d = _first_object(text or "")
    if d is None:
        return None
    rule_id = _pick(d.get("rule_id"), rule_ids)
    if rule_id is None:
        return None
    just = d.get("justification")
    return {"violation": _unit(d.get("violation")), "rule_id": rule_id,
            "clause_id": _pick(d.get("clause_id"), clause_ids),
            "justification": " ".join(just.split()) if isinstance(just, str) else "",
            "confidence": _unit(d.get("confidence"))}
