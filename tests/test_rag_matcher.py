"""The matcher's answer schema, clause candidates, prompt and answer parser, without a model."""
import json
from pathlib import Path

from pipeline import clients
from rag import corpus, matcher
from rag.corpus import Clause
from rag.retrieve import Hit
from rules import load_rules

ROOT = Path(__file__).resolve().parent.parent


def clause(cid, *parts, short="Правила (НБРК № 1)"):
    code, num = cid.split(":")
    return Clause(cid, code, short, f"Документ {code}", f"Постановление {code}", num, "Глава 1", tuple(parts),
                  f"https://adilet.zan.kz/rus/docs/{code}#z{num}")


class FakeIndex:
    def __init__(self, clauses):
        self.clauses = list(clauses)
        self.by_id = {c.id: c for c in self.clauses}


B9 = clause("B:9", "Порядок хранения ключей определяется банком.")
A1 = clause("A:1", "Кассир хранит наличные в сейфе.", "Сейф закрывается на ключ.")
A2 = clause("A:2", "Кассовому работнику не допускается покидать рабочее место.")
B7 = clause("B:7", "Посторонние лица в кассу не допускаются.", "Вход в кассу закрыт.")
INDEX = FakeIndex([B9, A1, A2, B7])  # B:9 first: an empty quote would "match" it


def cand(rid, *quotes):
    return {"id": rid, "name": rid, "visual_criteria": [], "never_claim": [],
            "citations": [{"document": "Doc", "clause": "", "quote": q} for q in quotes]}


# ---- answer_schema ------------------------------------------------------------------------

def test_answer_schema_enums_include_none():
    s = matcher.answer_schema(["cash_unattended", "other"], ["A:1", "B:7"])
    assert s["type"] == "object" and s["additionalProperties"] is False
    p = s["properties"]
    assert set(p) == set(s["required"]) == {"violation", "rule_id", "clause_id", "justification", "confidence"}
    assert p["rule_id"] == {"type": "string", "enum": ["cash_unattended", "other"]}
    assert p["clause_id"] == {"type": "string", "enum": ["A:1", "B:7", "none"]}
    assert p["justification"] == {"type": "string"}
    for k in ("violation", "confidence"):
        assert p[k] == {"type": "number", "minimum": 0, "maximum": 1}
    assert matcher.answer_schema(["other"], [])["properties"]["clause_id"]["enum"] == ["none"]


# ---- clause_candidates --------------------------------------------------------------------

def test_clause_candidates_catalogue_first_dedup():
    cands = [cand("empty", "", "   "),  # empty quotes are skipped, not matched against every clause
             cand("r_safe", "Сейф закрывается на ключ.", "Кассир хранит наличные"),  # both in A:1
             cand("r_leave", "не допускается покидать\n  рабочее   место"),  # whitespace collapsed
             cand("other", "текста нет в корпусе")]
    hits = [Hit("B:7", 0.9, 1), Hit("A:1", 0.8, 0), Hit("B:9", 0.5, 0), Hit("Z:1", 0.4, 0)]
    out = matcher.clause_candidates(hits, cands, INDEX)
    assert [(c.id, part) for c, part in out] == [("A:1", None), ("A:2", None), ("B:7", 1), ("B:9", 0)]
    assert out[0][0] is A1
    assert matcher.clause_candidates([], [], INDEX) == []
    assert matcher.clause_candidates([], [{"id": "x"}], INDEX) == []  # no citations


def test_clause_candidates_skips_quotes_outside_the_in_force_corpus():
    rules = load_rules(ROOT / "rules.yaml")
    r = rules["unauthorized_person_in_cash_area"]
    c = {"id": r["id"], "citations": [{"quote": x["quote"]} for x in r["citations"]]}
    out = matcher.clause_candidates([], [c], FakeIndex(corpus.load_clauses()))
    assert [(x.id, part) for x, part in out] == [("V2000020075:8", None)]  # п. 16 is suspended


# ---- build_prompt -------------------------------------------------------------------------

SYSTEM_LINES = (
    "clause_id must be one of the listed clause ids, or none when no listed clause states the requirement that was "
    "broken.\n"
    "violation: your probability from 0 to 1 that the observation shows a violation of rule_id at all; near 0 when "
    "it describes normal work.\n"
    "Answer with one JSON object only.")


def test_build_prompt_lists_ids_and_cuts_long_clauses():
    words = [f"слово{i:04d}" for i in range(400)]
    long_text = " ".join(words)
    assert len(long_text) > 3000
    long = clause("C:12", long_text[:1599], long_text[1600:], short="Длинные правила (НБРК № 9)")
    assert " ".join(long.parts) == long_text
    cands = [cand("r_safe", "Сейф закрывается на ключ."), cand("other", "текста нет")]
    system, user = matcher.build_prompt("Кассир уходит.\nДеньги лежат.", cands, [(A1, None), (long, 3), (B7, 0)])

    assert system == clients.JUDGE_SYSTEM + "\n" + SYSTEM_LINES
    assert user.startswith("Observation from a video model (may be empty): Кассир уходит.\nДеньги лежат.\n\n")
    assert json.dumps(cands, ensure_ascii=False, indent=1) in user  # the dicts judge_request sends
    lines = user.split("\n\nRegulation clauses, each headed by its clause_id in brackets:\n")[1].splitlines()
    heads = [ln for ln in lines if ln.startswith("[")]
    assert heads == ["[A:1] Правила (НБРК № 1), п. 1", "[C:12] Длинные правила (НБРК № 9), п. 12",
                     "[B:7] Правила (НБРК № 1), п. 7"]
    assert lines[lines.index(heads[0]) + 1] == "Кассир хранит наличные в сейфе. Сейф закрывается на ключ."
    assert lines[lines.index(heads[2]) + 1] == "Посторонние лица в кассу не допускаются. Вход в кассу закрыт."
    body = lines[lines.index(heads[1]) + 1]
    assert body.endswith("…") and len(body) <= 1501
    kept = body[:-1]
    assert long_text.startswith(kept) and long_text[len(kept)] == " "  # cut on a word boundary
    assert len(kept) > 1400
    assert user.index("[A:1]") > user.index("Candidate rules (JSON):")


def test_build_prompt_empty_observation_and_no_clauses():
    system, user = matcher.build_prompt("  ", [cand("other")], [])
    assert user.startswith("Observation from a video model (may be empty): (none)\n\n")
    assert "[" not in user.split("Regulation clauses")[1]


def test_cut_keeps_short_text_and_cuts_at_the_last_space():
    assert matcher.cut("abc def", 7) == "abc def"
    assert matcher.cut("abc def", 6) == "abc…"
    assert matcher.cut("abc def ghi", 7) == "abc def…"  # the space right at the limit counts
    assert matcher.cut("abcdefghij", 4) == "abcd…"  # no space: a hard cut
    assert matcher.cut("", 5) == ""


def test_fit_clauses_drops_retrieved_clauses_from_the_end():
    big = [clause(f"D:{i}", "x " * 600) for i in range(30)]
    cands = [cand("r_safe", "Сейф закрывается на ключ.")]
    items = [(A1, None)] + [(c, 0) for c in big]
    fitted = matcher.fit_clauses("obs", cands, items)
    assert fitted == items[:len(fitted)] and fitted[0] == (A1, None)
    assert len(matcher.build_prompt("obs", cands, fitted)[1]) <= matcher.PROMPT_MAX
    assert len(matcher.build_prompt("obs", cands, items[:len(fitted) + 1])[1]) > matcher.PROMPT_MAX
    assert matcher.fit_clauses("obs", cands, items[:3]) == items[:3]
    # catalogue clauses always stay, even over the limit
    catalogue = [(c, None) for c in big]
    assert matcher.fit_clauses("obs", cands, catalogue) == catalogue


# ---- parse_answer -------------------------------------------------------------------------

RULES = ["cash_unattended", "other"]
CLAUSES = ["V1900019680:20", "V2000020075:8"]
ANSWER = {"violation": 0.8, "rule_id": "cash_unattended", "clause_id": "V1900019680:20",
          "justification": " Действия  могут\nуказывать на нарушение. ", "confidence": 0.6}


def test_parse_answer_valid():
    assert matcher.parse_answer(json.dumps(ANSWER, ensure_ascii=False), RULES, CLAUSES) == {
        "violation": 0.8, "rule_id": "cash_unattended", "clause_id": "V1900019680:20",
        "justification": "Действия могут указывать на нарушение.", "confidence": 0.6}


def test_parse_answer_fenced():
    text = "Ответ:\n```json\n" + json.dumps(ANSWER, ensure_ascii=False, indent=2) + "\n```\nГотово."
    assert matcher.parse_answer(text, RULES, CLAUSES)["clause_id"] == "V1900019680:20"
    # the first object that parses wins; a broken one before it is skipped
    text = '{"rule_id": oops} then ' + json.dumps({**ANSWER, "rule_id": "other"}) + " " + json.dumps(ANSWER)
    assert matcher.parse_answer(text, RULES, CLAUSES)["rule_id"] == "other"


def test_parse_answer_unknown_rule_is_none():
    for rid in ("made_up", None, 3, ""):
        assert matcher.parse_answer(json.dumps({**ANSWER, "rule_id": rid}), RULES, CLAUSES) is None
    assert matcher.parse_answer(json.dumps({k: v for k, v in ANSWER.items() if k != "rule_id"}), RULES,
                                CLAUSES) is None
    for text in ("", "no json here", "[1, 2]", '"cash_unattended"', "{broken"):
        assert matcher.parse_answer(text, RULES, CLAUSES) is None
    # case and stray quotes around a known id are tolerated
    assert matcher.parse_answer(json.dumps({**ANSWER, "rule_id": " `Cash_Unattended` "}), RULES,
                                CLAUSES)["rule_id"] == "cash_unattended"


def test_parse_answer_unknown_clause_becomes_none():
    for cid in ("none", "NONE", "V1900019680:99", "", None, 20):
        a = matcher.parse_answer(json.dumps({**ANSWER, "clause_id": cid}), RULES, CLAUSES)
        assert a["clause_id"] is None and a["rule_id"] == "cash_unattended", cid
    a = matcher.parse_answer(json.dumps({**ANSWER, "clause_id": "[v2000020075:8]"}), RULES, CLAUSES)
    assert a["clause_id"] == "V2000020075:8"  # the prompt shows ids in brackets
    assert matcher.parse_answer(json.dumps({**ANSWER, "clause_id": "none"}), RULES, [])["clause_id"] is None


def test_parse_answer_clips_probabilities():
    a = matcher.parse_answer(json.dumps({**ANSWER, "violation": 1.7, "confidence": -0.2}), RULES, CLAUSES)
    assert (a["violation"], a["confidence"]) == (1.0, 0.0)
    a = matcher.parse_answer(json.dumps({**ANSWER, "violation": 1, "confidence": 0}), RULES, CLAUSES)
    assert (a["violation"], a["confidence"]) == (1.0, 0.0)
    for bad in ("0.7", None, True, [0.5], {"p": 0.5}):
        a = matcher.parse_answer(json.dumps({**ANSWER, "violation": bad, "confidence": bad}), RULES, CLAUSES)
        assert (a["violation"], a["confidence"]) == (None, None), bad
    a = matcher.parse_answer('{"rule_id": "other", "violation": NaN, "confidence": Infinity}', RULES, CLAUSES)
    assert (a["violation"], a["confidence"]) == (None, None)
    a = matcher.parse_answer('{"rule_id": "other"}', RULES, CLAUSES)
    assert a == {"violation": None, "rule_id": "other", "clause_id": None, "justification": "", "confidence": None}
