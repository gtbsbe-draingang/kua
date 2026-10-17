import copy
import csv
import re
from pathlib import Path

import pytest
import yaml

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
