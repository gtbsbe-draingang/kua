"""Turn the adilet-format HTML of one act into a list of clauses (пункты) with their structure.

The page keeps adilet's markup inside <article>:
  <h3>                     the title of the Rules, "Глава N. ...", "Параграф N. ..."
  <p id="zNN">             one paragraph: a clause start "20. ...", a subclause "1) ...",
                           or a continuation paragraph of the current clause
  <span class="note">...   a footnote ("Сноска. ..."), split around <a> links and ended by <br>;
                           a repealed clause appears here as "17. Исключен ..."
  <font color="#FF0000">   an editor's note ("Примечание ИЗПИ!"), e.g. that a clause is suspended
                           and which wording applies meanwhile
The resolution text before the Rules and the annexes (forms) after them are skipped.
"""
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

from . import config
from .fetch import load_sources, raw_path

CLAUSE_RE = re.compile(r"^(\d{1,3}(?:-\d{1,2})?)\.\s+(.*)$", re.S)
REPEALED_RE = re.compile(r"^(\d{1,3}(?:-\d{1,2})?)\.\s*Исключен", re.I)
AMENDMENT_RE = re.compile(r"от\s+(\d{2})\.(\d{2})\.(\d{2,4})\s*(?:г\.|года)?\s*№\s*(\d+)")
SUSPENDED_RE = re.compile(r"Действие\s+(под)?пункт[а-я]*\s+(\d{1,3}(?:-\d{1,2})?\)?)", re.I)
INLINE_TAGS = {"span", "a", "font", "b", "i", "u", "sup", "sub", "strong", "em"}


@dataclass
class Clause:
    num: str                      # "20", "10-1"
    anchor: str | None            # adilet anchor of the first paragraph, e.g. "z111"
    chapter: str
    section: str                  # "Параграф ..." or ""
    lines: list[str]              # first line starts with "20. "; then subclauses / paragraphs verbatim
    notes: list[str] = field(default_factory=list)       # footnotes and editor's notes, verbatim
    amendments: list[str] = field(default_factory=list)  # "№ 71 от 10.11.2025", oldest first
    status: str = "in_force"      # in_force | repealed | suspended (interim wording applies)
    seq: int = 0                  # order inside the document

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _logical_lines(article: Tag):
    """Yield (kind, text, anchor) for each logical line of the article in document order.

    kind is heading | text | note | editor | table. Inline runs (footnotes split around links,
    editor's notes) are joined until the next <br> or block element.
    """
    buf: list = []

    def flush():
        if not buf:
            return None
        text = _clean("".join(n.get_text() if isinstance(n, Tag) else str(n) for n in buf))
        tags = [n for n in buf if isinstance(n, Tag)]
        anchor = None
        for t in tags:  # an inline line has no id; adilet marks it with <a name="zNN">
            named = [t] if t.name == "a" and t.get("name") else t.find_all("a", attrs={"name": True})
            if named:
                anchor = named[0]["name"]
                break
        if any(t.name == "font" for t in tags):
            kind = "editor"
        elif any("note" in (t.get("class") or []) for t in tags) or text.startswith("Сноска"):
            kind = "note"
        else:
            kind = "text"
        buf.clear()
        return (kind, text, anchor) if text else None

    for node in article.children:
        if isinstance(node, NavigableString):
            if node.strip():
                buf.append(node)
            continue
        if not isinstance(node, Tag):
            continue
        if node.name in INLINE_TAGS:
            buf.append(node)
            continue
        line = flush()
        if line:
            yield line
        if node.name == "br":
            continue
        text = _clean(node.get_text())
        if not text:
            continue
        if node.name == "h3":
            yield "heading", text, node.get("id")
        elif node.find("table") is not None or node.name == "table":
            yield "table", text, node.get("id")
        elif "note" in (node.get("class") or []) or text.startswith("Сноска"):
            yield "note", text, node.get("id")
        else:
            yield "text", text, node.get("id")
    line = flush()
    if line:
        yield line


def _num_key(num: str) -> tuple[int, int]:
    major, _, minor = num.partition("-")
    return int(major), int(minor or 0)


def _amendments(note: str) -> list[tuple[str, str]]:
    """[(iso_date, "№ 71 от 10.11.2025"), ...] for every amending act mentioned in a note."""
    out = []
    for d, m, y, n in AMENDMENT_RE.findall(note):
        y = y if len(y) == 4 else "20" + y
        out.append((f"{y}-{m}-{d}", f"№ {n} от {d}.{m}.{y}"))
    return out


def parse_document(doc: dict) -> dict:
    html = raw_path(doc["code"]).read_text(encoding="utf-8")
    article = BeautifulSoup(html, "html.parser").find("article")
    if article is None:
        raise RuntimeError(f"{doc['code']}: no <article> element")
    include = set(doc.get("include_chapters") or [])

    clauses: list[Clause] = []
    chapter = section = rules_title = ""
    started = False
    current: Clause | None = None
    pending_editor: list[str] = []
    pending_anchor: str | None = None  # the interim wording after an editor's note has no id
    all_amendments: list[tuple[str, str]] = []

    def chapter_included() -> bool:
        if not include:
            return True
        m = re.match(r"Глава\s+(\d+(?:-\d+)?)", chapter)
        return bool(m and m.group(1) in include)

    def attach_notes(c: Clause, notes: list[str]) -> None:
        for note in notes:
            c.notes.append(note)
            found = _amendments(note)
            c.amendments.extend(label for _, label in found if label not in c.amendments)
            all_amendments.extend(found)

    def add_clause(num: str, anchor: str | None, first_line: str, status: str = "in_force") -> Clause:
        nonlocal current, pending_anchor
        c = Clause(num=num, anchor=anchor or pending_anchor, chapter=chapter, section=section,
                   lines=[first_line], status=status, seq=len(clauses))
        if pending_editor:
            m = SUSPENDED_RE.search(" ".join(pending_editor))
            if m and not m.group(1) and m.group(2) == num:
                c.status = "suspended"
            attach_notes(c, pending_editor)
            pending_editor.clear()
        pending_anchor = None
        clauses.append(c)
        current = c
        return c

    for kind, text, anchor in _logical_lines(article):
        if not started:
            if kind == "heading" and text.startswith("Правила"):
                started, rules_title = True, text
            continue
        if kind in ("text", "table") and text.startswith("Приложение"):
            break  # annexes (forms) start here
        if kind == "heading":
            if text.startswith("Глава"):
                chapter, section, current = text, "", None
            elif text.startswith("Параграф"):
                section, current = text, None
            continue
        if not chapter_included():
            continue
        if kind == "table":
            continue
        if kind == "editor":
            if text.startswith("Примечание ИЗПИ"):
                pending_anchor = anchor
            else:
                pending_editor.append(text)
            continue
        if kind == "note":
            m = REPEALED_RE.match(text)
            if m and (current is None or _num_key(m.group(1)) > _num_key(current.num)):
                c = add_clause(m.group(1), anchor, text, status="repealed")
                found = _amendments(text)
                c.amendments = [label for _, label in found]
                all_amendments.extend(found)
                continue
            if current is not None:  # otherwise it is a footnote on a chapter or paragraph heading
                attach_notes(current, [text])
            continue
        # kind == "text"
        m = CLAUSE_RE.match(text)
        # Clause numbers rise through the whole act, so a smaller "1. " is a list inside a clause.
        if m and (current is None or _num_key(m.group(1)) > _num_key(current.num)):
            add_clause(m.group(1), anchor, text)
        elif current is not None:
            if pending_editor:  # a note on a subclause, e.g. "Действие подпункта 3) приостановлено"
                attach_notes(current, pending_editor)
                pending_editor.clear()
            current.lines.append(text)

    if not started:
        raise RuntimeError(f"{doc['code']}: the Rules title heading was not found")
    latest = max(all_amendments, default=("", ""))
    return {
        "code": doc["code"], "short": doc["short"], "title": doc["title"], "act": doc["act"],
        "canonical_url": doc["canonical_url"], "fetch_url": doc["fetch_url"],
        "rules_title": rules_title, "include_chapters": sorted(include),
        "latest_amendment": latest[1], "latest_amendment_date": latest[0],
        "clauses": [asdict(c) for c in clauses],
    }


def parsed_path(code: str) -> Path:
    return config.PARSED_DIR / f"{code}.json"


def parse_all() -> list[dict]:
    config.PARSED_DIR.mkdir(parents=True, exist_ok=True)
    docs = []
    for src in load_sources():
        doc = parse_document(src)
        parsed_path(src["code"]).write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        statuses = {}
        for c in doc["clauses"]:
            statuses[c["status"]] = statuses.get(c["status"], 0) + 1
        print(f"parsed {src['code']}: {len(doc['clauses'])} clauses {statuses}, "
              f"latest amendment {doc['latest_amendment'] or '-'}")
        docs.append(doc)
    return docs


def load_parsed() -> list[dict]:
    return [json.loads(parsed_path(s["code"]).read_text(encoding="utf-8")) for s in load_sources()]
