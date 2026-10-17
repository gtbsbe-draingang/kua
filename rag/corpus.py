"""The regulation corpus: in-force clauses of the four parsed adilet acts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ACTS_DIR = Path(__file__).parent / "acts"

_ACT_FIELDS = ("code", "short", "title", "act", "canonical_url", "clauses")
_CLAUSE_FIELDS = ("num", "anchor", "chapter", "lines", "status", "seq")


@dataclass(frozen=True)
class Clause:
    id: str
    code: str
    short: str
    document: str
    act: str
    num: str
    chapter: str
    parts: tuple[str, ...]
    url: str


def _collapse(text: str) -> str:
    return " ".join(text.split())


def load_clauses(acts_dir: Path = ACTS_DIR, in_force_only: bool = True) -> list[Clause]:
    clauses: list[Clause] = []
    seen: set[str] = set()
    for path in sorted(Path(acts_dir).glob("*.json"), key=lambda p: _read(p)["code"]):
        act = _read(path)
        for f in _ACT_FIELDS:
            if f not in act:
                raise ValueError(f"{path.name}: missing field {f!r}")
        for raw in sorted(act["clauses"], key=lambda c: c.get("seq", 0)):
            for f in _CLAUSE_FIELDS:
                if f not in raw:
                    raise ValueError(f"{path.name}: clause {raw.get('num')!r} is missing field {f!r}")
            if in_force_only and raw["status"] != "in_force":
                continue
            cid = f"{act['code']}:{raw['num']}"
            if cid in seen:
                raise ValueError(f"duplicate clause id {cid}")
            seen.add(cid)
            # one in-force clause (V1900019680:53) has no anchor: link the act itself
            url = f"{act['canonical_url']}#{raw['anchor']}" if raw["anchor"] else act["canonical_url"]
            clauses.append(Clause(
                id=cid, code=act["code"], short=act["short"], document=act["title"],
                act=act["act"], num=raw["num"], chapter=raw["chapter"],
                parts=tuple(_collapse(s) for s in raw["lines"]),
                url=url,
            ))
    return clauses


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def passages(clauses: list[Clause]) -> list[tuple[str, int, str]]:
    return [(c.id, i, p) for c in clauses for i, p in enumerate(c.parts) if p]


def find_quote(clauses: list[Clause], quote: str, code: str | None = None) -> Clause | None:
    needle = _collapse(quote)
    for c in clauses:
        if code is not None and c.code != code:
            continue
        if needle in " ".join(c.parts):
            return c
    return None
