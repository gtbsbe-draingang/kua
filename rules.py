"""Loader and validator for the rules catalogue (rules.yaml)."""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import yaml

ALLOWED_HOSTS = {"adilet.zan.kz", "www.adilet.zan.kz", "nationalbank.kz", "www.nationalbank.kz"}


class RuleError(ValueError):
    pass


def _check_citation(rid: str, i: int, c: dict, strict: bool) -> None:
    if not isinstance(c, dict):
        raise RuleError(f"rule {rid!r}: citation #{i} is not a mapping")
    if strict:
        quote = str(c.get("quote") or "").strip()
        if not quote:
            raise RuleError(f"rule {rid!r}: citation #{i} has an empty quote")
        if quote == "TODO":
            raise RuleError(f"rule {rid!r}: citation #{i} quote is still TODO")
        url = str(c.get("url") or "")
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
            raise RuleError(
                f"rule {rid!r}: citation #{i} url {url!r} must be https on one of "
                f"{sorted(ALLOWED_HOSTS)}"
            )


def load_rules(path, strict: bool = True) -> dict[str, dict]:
    with open(Path(path), encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or not isinstance(data.get("rules"), list):
        raise RuleError("rules file must be a mapping with a 'rules' list")
    rules: dict[str, dict] = {}
    for r in data["rules"]:
        rid = r.get("id") if isinstance(r, dict) else None
        if not rid or not isinstance(rid, str):
            raise RuleError(f"rule without a string id: {r!r}")
        if rid in rules:
            raise RuleError(f"duplicate rule id {rid!r}")
        citations = r.get("citations") or []
        if r.get("detectable"):
            if not r.get("grounding_queries"):
                raise RuleError(f"rule {rid!r} is detectable but has no grounding query")
            if not citations:
                raise RuleError(f"rule {rid!r} is detectable but has no citation")
        for i, c in enumerate(citations):
            _check_citation(rid, i, c, strict)
        rules[rid] = r
    n_other = sum(1 for rid in rules if rid == "other")
    if n_other != 1:
        raise RuleError(f"exactly one rule with id 'other' is required, found {n_other}")
    return rules


def citation_text(rule: dict) -> str:
    c = rule["citations"][0]
    clause = (c.get("clause") or "").strip()
    return f"{c['document']}, {clause}" if clause else c["document"]
