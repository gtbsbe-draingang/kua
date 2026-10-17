import copy
import csv
import re
from pathlib import Path

import pytest
import yaml

from rag import corpus
from rules import RuleError, citation_text, load_rules

ROOT = Path(__file__).resolve().parent.parent


def cit(**kw):
    c = {"document": "Doc", "clause": "п. 1", "quote": "text",
         "url": "https://adilet.zan.kz/rus/docs/X", "edition_checked": "2026-10-17"}
    c.update(kw)
    return c


def rule(rid="r1", **kw):
    r = {"id": rid, "name": "N", "seen_in_training": True, "detectable": True,
         "severity": "high", "omission": False, "visual_criteria": ["a"],
         "never_claim": ["b"], "grounding_queries": ["q"], "choice_option": "c",
         "verify_prompt": "p", "why": "w", "citations": [cit()]}
    r.update(kw)
    return r


def other():
    return rule("other", name="Other")


def write(tmp_path, rules):
    p = tmp_path / "rules.yaml"
    p.write_text(yaml.safe_dump({"version": 1, "rules": rules}, allow_unicode=True))
    return p


def test_valid_fixture_loads(tmp_path):
    assert set(load_rules(write(tmp_path, [rule(), other()]))) == {"r1", "other"}


def test_catalogue_loads_strict():
    rules = load_rules(ROOT / "rules.yaml", strict=True)
    assert "other" in rules
    assert "cash_unattended" in rules  # the stub pipeline emits this id
    for rid, r in rules.items():
        if r.get("detectable"):
            assert 2 <= len(r["grounding_queries"]) <= 4, rid


# "left" is also the past tense of "leave" ("cash is left unattended"), which describes an
# action, not a place. Every other use of "left" is treated as a screen position, so a
# spatial "left side" or "on the left" fails. Only these verb phrases are allowed.
POSITION_WORDS = re.compile(
    r"\b(right|top|bottom|upper|lower|corner|foreground|background)\b", re.I)
LEFT = re.compile(r"\bleft\b", re.I)
LEFT_VERB_PHRASES = re.compile(
    r"\b(cashier|client|customer|person|he|she|they|has|have|had)\s+left\b"
    r"|\b(?!(?:the|on|to|at|from|in|far)\b)\w+\s+left\s+"
    r"(open|unattended|unlocked|lying|waiting|out|in|on|without|alone|behind|visible)\b",
    re.I)


def left_positions(text):
    """Occurrences of "left" that are not one of the allowed verb phrases."""
    return LEFT.findall(LEFT_VERB_PHRASES.sub(" ", text))


def _prompt_strings(r):
    out = list(r.get("grounding_queries") or []) + list(r.get("visual_criteria") or [])
    out += list(r.get("never_claim") or [])
    out += [r.get("choice_option") or "", r.get("verify_prompt") or ""]
    return out


def test_queries_have_no_screen_positions():
    for rid, r in load_rules(ROOT / "rules.yaml", strict=True).items():
        for text in _prompt_strings(r):
            m = POSITION_WORDS.search(str(text))
            assert not m, f"{rid}: {m.group(0)!r} in {text!r}"
            assert not left_positions(str(text)), f"{rid}: positional 'left' in {text!r}"


@pytest.mark.parametrize("text", ["on the left", "left side of the desk", "the left window",
                                  "left of the cashier", "cashier at far left"])
def test_left_guard_rejects_spatial_use(text):
    assert left_positions(text)


@pytest.mark.parametrize("text", ["the cashier left", "safe left open", "keys left in the lock",
                                  "after the client has left", "cash is left unattended"])
def test_left_guard_allows_verb_use(text):
    assert not left_positions(text)


def test_detectable_why_is_hedged():
    for rid, r in load_rules(ROOT / "rules.yaml", strict=True).items():
        if r.get("detectable"):
            assert "может" in r["why"], f"{rid}: why must be hedged with 'может'"


def test_undetectable_rules_state_a_limitation():
    for rid, r in load_rules(ROOT / "rules.yaml", strict=True).items():
        if not r.get("detectable"):
            assert str(r.get("limitation") or "").strip(), f"{rid}: empty limitation"


def test_every_seen_rule_has_a_clip():
    with open(ROOT / "data" / "labels.csv", encoding="utf-8", newline="") as f:
        used = {row["type_id"] for row in csv.DictReader(f)}
    for rid, r in load_rules(ROOT / "rules.yaml", strict=True).items():
        if r.get("seen_in_training"):
            assert rid in used, rid


REPEALED = re.compile(r"(58\s+(от|of)\s+.*2001|394\s+(от|of)\s+.*1999|V14Y0010204)", re.I)


def test_no_cited_repealed_acts():
    for rid, r in load_rules(ROOT / "rules.yaml", strict=True).items():
        for c in r.get("citations") or []:
            for field in ("document", "url"):
                v = str(c.get(field) or "")
                assert not REPEALED.search(v), f"{rid}: {field} {v!r}"
                assert "V14Y0010204" not in v, rid
                assert not re.search(r"№\s*(58|394)\b", v), f"{rid}: {v!r}"


# Citation urls that may point at a document outside the corpus (none for now). Any other
# citation must name an act of the corpus, so a mistyped act code in a url fails the test.
OUTSIDE_CORPUS_URLS = frozenset()


def corpus_citation_problems(rules, in_force, codes, outside=frozenset()):
    """Citations that do not resolve to an in-force clause of a corpus act.

    `in_force` is what rag/corpus.py loads (suspended and repealed clauses are dropped);
    `codes` are the act codes of the whole corpus. A citation whose url does not name one of
    them is a problem unless the url is in `outside`.
    """
    problems = []
    for rid, r in rules.items():
        for c in r.get("citations") or []:
            m = re.search(r"/docs/(\w+)", c["url"])
            code = m.group(1) if m else None
            if code not in codes:
                if c["url"] not in outside:
                    problems.append(f"{rid}: {c['url']!r} names no act of the corpus")
                continue
            hit = corpus.find_quote(in_force, c["quote"], code)
            if hit is None:
                problems.append(f"{rid}: no in-force clause of {code} contains {c['quote'][:60]!r}")
                continue
            label = re.match(r"п\. (\d+(?:-\d+)?)", c.get("clause") or "")
            if label and label.group(1) != hit.num:
                problems.append(f"{rid}: clause {c['clause']!r} but the quote is in п. {hit.num}")
    return problems


def _corpus_problems(rules, outside=OUTSIDE_CORPUS_URLS):
    codes = {c.code for c in corpus.load_clauses(in_force_only=False)}
    return corpus_citation_problems(rules, corpus.load_clauses(), codes, outside)


def test_catalogue_citations_resolve_to_in_force_clauses():
    # a suspended or repealed clause (rag/corpus.py drops them) must not be cited again
    assert _corpus_problems(load_rules(ROOT / "rules.yaml", strict=True)) == []


def test_citing_a_suspended_clause_is_reported():
    # НБРК № 14 п. 16 is suspended until 2027-01-01: the text is in the act but not in force
    quote = "каждая касса имеет отдельный вход с закрывающейся на замок дверью"
    assert corpus.find_quote(corpus.load_clauses(in_force_only=False), quote, "V2000020075")
    bad = rule(citations=[cit(clause="п. 16, подп. 1)", quote=quote,
                              url="https://adilet.zan.kz/rus/docs/V2000020075#z121")])
    problems = _corpus_problems({"r1": bad})
    assert len(problems) == 1 and "V2000020075" in problems[0]


def test_wrong_clause_number_is_reported():
    quote = "Кассовый узел подлежит изолированию от других производственных, подсобных, подвальных и служебных помещений одним закрывающимся служебным входом."
    good = rule(citations=[cit(clause="п. 8, ч. 1", quote=quote,
                               url="https://adilet.zan.kz/rus/docs/V2000020075#z49")])
    assert _corpus_problems({"r1": good}) == []
    wrong = rule(citations=[cit(clause="п. 9, ч. 1", quote=quote,
                                url="https://adilet.zan.kz/rus/docs/V2000020075#z49")])
    assert len(_corpus_problems({"r1": wrong})) == 1


def test_hyphenated_clause_numbers_are_matched_whole():
    quote = "Банк, инкассаторская организация, использующие спецконтейнеры, принимают от клиента банкноты"
    url = "https://adilet.zan.kz/rus/docs/V1900019680#z463"
    assert _corpus_problems({"r1": rule(citations=[cit(clause="п. 63-1, ч. 2", quote=quote, url=url)])}) == []
    prefix_only = rule(citations=[cit(clause="п. 63", quote=quote, url=url)])
    assert len(_corpus_problems({"r1": prefix_only})) == 1


@pytest.mark.parametrize("url", ["https://adilet.zan.kz/rus/docs/V2000020076#z49",  # mistyped act code
                                 "https://adilet.zan.kz/rus/",                      # no /docs/
                                 "https://www.nationalbank.kz/ru/news/x"])
def test_citation_of_a_document_outside_the_corpus_must_be_allowlisted(url):
    r = {"r1": rule(citations=[cit(url=url, quote="not in any act")])}
    problems = _corpus_problems(r)
    assert len(problems) == 1 and url in problems[0]
    assert _corpus_problems(r, outside=frozenset({url})) == []


def test_catalogue_cites_no_document_outside_the_corpus():
    assert OUTSIDE_CORPUS_URLS == frozenset()


def test_strict_rejects_todo_quote(tmp_path):
    p = write(tmp_path, [rule(citations=[cit(quote="TODO")]), other()])
    with pytest.raises(RuleError, match="r1"):
        load_rules(p, strict=True)
    assert "r1" in load_rules(p, strict=False)


def test_strict_rejects_empty_quote(tmp_path):
    with pytest.raises(RuleError, match="r1"):
        load_rules(write(tmp_path, [rule(citations=[cit(quote="")]), other()]))


def test_rejects_duplicate_id(tmp_path):
    with pytest.raises(RuleError, match="r1"):
        load_rules(write(tmp_path, [rule(), rule(), other()]))


def test_rejects_missing_other(tmp_path):
    with pytest.raises(RuleError, match="other"):
        load_rules(write(tmp_path, [rule()]))


def test_rejects_two_other(tmp_path):
    with pytest.raises(RuleError, match="other"):
        load_rules(write(tmp_path, [other(), other()]))


def test_rejects_detectable_without_query(tmp_path):
    with pytest.raises(RuleError, match="r1"):
        load_rules(write(tmp_path, [rule(grounding_queries=[]), other()]))


def test_rejects_detectable_without_citation(tmp_path):
    with pytest.raises(RuleError, match="r1"):
        load_rules(write(tmp_path, [rule(citations=[]), other()]))


def test_rejects_foreign_url(tmp_path):
    p = write(tmp_path, [rule(citations=[cit(url="https://example.com/x")]), other()])
    with pytest.raises(RuleError, match="r1"):
        load_rules(p)


def test_rejects_non_https_url(tmp_path):
    p = write(tmp_path, [rule(citations=[cit(url="http://adilet.zan.kz/x")]), other()])
    with pytest.raises(RuleError, match="r1"):
        load_rules(p)


@pytest.mark.parametrize("host", ["adilet.zan.kz", "www.adilet.zan.kz",
                                  "nationalbank.kz", "www.nationalbank.kz"])
def test_allowed_hosts(tmp_path, host):
    p = write(tmp_path, [rule(citations=[cit(url=f"https://{host}/a")]), other()])
    assert "r1" in load_rules(p)


def test_lookalike_host_rejected(tmp_path):
    p = write(tmp_path, [rule(citations=[cit(url="https://adilet.zan.kz.evil.com/a")]), other()])
    with pytest.raises(RuleError, match="r1"):
        load_rules(p)


def test_undetectable_needs_no_query(tmp_path):
    p = write(tmp_path, [rule("lim", detectable=False, grounding_queries=[], citations=[]), other()])
    assert "lim" in load_rules(p)


def test_citation_text_with_clause():
    assert citation_text(rule()) == "Doc, п. 1"


def test_citation_text_without_clause():
    r = rule(citations=[cit(clause="")])
    assert citation_text(r) == "Doc"


def test_citation_text_uses_first_citation():
    r = rule(citations=[cit(document="First"), cit(document="Second")])
    assert citation_text(r) == "First, п. 1"
