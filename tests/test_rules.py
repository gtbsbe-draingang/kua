import copy
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


def test_loads_v0_non_strict():
    assert set(load_rules(ROOT / "rules.yaml", strict=False)) == {"other", "cash_unattended"}


def test_v0_strict_fails():
    with pytest.raises(RuleError):
        load_rules(ROOT / "rules.yaml")


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
