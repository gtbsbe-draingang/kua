import json
from pathlib import Path

import pytest

from rag import corpus
from rules import load_rules

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def clauses():
    return corpus.load_clauses()


def test_load_clauses_counts(clauses):
    assert len(clauses) == 247
    per = {}
    for c in clauses:
        per[c.code] = per.get(c.code, 0) + 1
    assert per == {"V1900019680": 152, "V2000020075": 20, "V1900018545": 36, "V1700016120": 39}


def test_clause_ids_unique_and_urls_canonical(clauses):
    assert len({c.id for c in clauses}) == len(clauses)
    no_anchor = []
    for c in clauses:
        assert c.id == f"{c.code}:{c.num}"
        base = "https://adilet.zan.kz/rus/docs/" + c.code
        if c.url == base:
            no_anchor.append(c.id)
        else:
            assert c.url.startswith(base + "#z")
    assert no_anchor == ["V1900019680:53"]  # the parsed act has no anchor for this clause


def test_parts_are_lines_without_notes(clauses):
    raw = json.loads((corpus.ACTS_DIR / "V1900018545.json").read_text(encoding="utf-8"))
    src = next(c for c in raw["clauses"] if c["num"] == "41")
    c = next(c for c in clauses if c.id == "V1900018545:41")
    assert c.parts == tuple(" ".join(s.split()) for s in src["lines"])
    assert src["notes"]
    for note in src["notes"]:
        assert all(note not in p for p in c.parts)
    assert c.url == "https://adilet.zan.kz/rus/docs/V1900018545#" + src["anchor"]


def test_passages_skip_empty_parts():
    c = corpus.Clause("X:1", "X", "s", "d", "a", "1", "ch", ("one", "", "two"), "u")
    assert corpus.passages([c]) == [("X:1", 0, "one"), ("X:1", 2, "two")]


def test_load_clauses_rejects_duplicates_and_missing_fields(tmp_path):
    act = json.loads((corpus.ACTS_DIR / "V1900018545.json").read_text(encoding="utf-8"))
    dup = dict(act, clauses=[act["clauses"][0], act["clauses"][0]])
    d = tmp_path / "dup"
    d.mkdir()
    (d / "A.json").write_text(json.dumps(dup), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        corpus.load_clauses(d)

    bad = dict(act, clauses=[{k: v for k, v in act["clauses"][0].items() if k != "anchor"}])
    d2 = tmp_path / "bad"
    d2.mkdir()
    (d2 / "A.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="anchor"):
        corpus.load_clauses(d2)


def test_catalogue_quotes_are_in_corpus():
    # one quote (п. 16 of НБРК №14) sits in a clause the act marks as suspended until 2027
    everything = corpus.load_clauses(in_force_only=False)
    in_force = corpus.load_clauses()
    suspended_only = []
    rules = load_rules(ROOT / "rules.yaml")
    n = 0
    for rid, r in rules.items():
        for c in r.get("citations") or []:
            code = c["url"].split("/docs/")[1].split("#")[0]
            assert corpus.find_quote(everything, c["quote"], code) is not None, (rid, c["quote"][:60])
            if corpus.find_quote(in_force, c["quote"], code) is None:
                suspended_only.append(c["quote"])
            n += 1
    assert n == 35
    assert suspended_only == ["каждая касса имеет отдельный вход с закрывающейся на замок дверью"]


def test_find_quote_scopes_to_act(clauses):
    q = "В помещениях операционных касс обменных пунктов не допускается присутствие посторонних лиц"
    assert corpus.find_quote(clauses, q).id == "V1900018545:41"
    assert corpus.find_quote(clauses, q, "V1900018545").id == "V1900018545:41"
    assert corpus.find_quote(clauses, q, "V2000020075") is None
    assert corpus.find_quote(clauses, "  В   помещениях\nопераци" + "онных касс", "V1900018545")


def test_rag_queries_expect_existing_clauses(clauses):
    queries = json.loads((ROOT / "tests" / "data" / "rag_queries.json").read_text(encoding="utf-8"))
    ids = {c.id for c in clauses}
    assert len(queries) == 15
    for q in queries:
        assert q["text"].strip() and q["expect"] in ids
    assert sum(q["expect"] == "V1900018545:41" for q in queries) >= 2
    rules = load_rules(ROOT / "rules.yaml")
    first = [
        corpus.find_quote(clauses, r["citations"][0]["quote"],
                          r["citations"][0]["url"].split("/docs/")[1].split("#")[0]).id
        for r in rules.values() if r.get("detectable") and r["id"] != "other"
    ]
    assert [q["expect"] for q in queries[:13]] == first
