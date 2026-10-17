"""Ingest zakon.uchet.kz HTML dumps (raw/*.html + *.meta.json) into the RAG API.

usage: RAG_API_KEY=... python3 rag/ingest_raw.py raw [http://localhost:8080]
"""
import html
import json
import os
import re
import sys
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

BLOCK = {"p", "br", "div", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "article"}


class Body(HTMLParser):
    """Text of <section itemprop="articleBody">, one line per block element; amendment notes dropped."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title, self.parts = "", []
        self.depth = 0          # >0 while inside articleBody (counts nested <section>)
        self.note = 0           # >0 while inside <span class="note">
        self.in_h1 = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "h1" and a.get("itemprop") == "name":
            self.in_h1 = True
        if tag == "section" and (self.depth or a.get("itemprop") == "articleBody"):
            self.depth += 1
        if self.depth and tag == "span" and (self.note or "note" in (a.get("class") or "").split()):
            self.note += 1
        if self.depth and tag in BLOCK:
            self.parts.append("\n")
        if self.depth and tag == "td":
            self.parts.append(" | ")

    def handle_endtag(self, tag):
        if tag == "h1":
            self.in_h1 = False
        if self.depth and tag == "section":
            self.depth -= 1
        if self.note and tag == "span":
            self.note -= 1
        if self.depth and tag in BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.in_h1:
            self.title += data
        elif self.depth and not self.note:
            self.parts.append(data)


def parse(path: Path) -> tuple[str, str]:
    p = Body()
    p.feed(path.read_text(encoding="utf-8"))
    text = html.unescape("".join(p.parts)).replace("\xa0", " ")
    lines = (re.sub(r"[ \t]+", " ", l).strip(" |") for l in text.split("\n"))
    # blank line between blocks so the chunker sees each paragraph separately
    return " ".join(p.title.split()), "\n\n".join(l for l in lines if l)


def main(folder: str, api: str = "http://localhost:8080"):
    docs = []
    for f in sorted(Path(folder).glob("*.html")):
        meta = json.loads(f.with_suffix(".meta.json").read_text())
        title, text = parse(f)
        docs.append({"document": f"{title} ({meta['code']})", "url": meta["fetch_url"], "text": text})
        print(f"{f.name}: {len(text)} chars  {title[:70]}")
    for d in docs:  # one request per document keeps each call short
        req = urllib.request.Request(f"{api}/ingest", json.dumps({"documents": [d]}).encode(),
                                     {"Content-Type": "application/json", "X-API-Key": os.environ["RAG_API_KEY"]})
        print(d["url"], json.load(urllib.request.urlopen(req, timeout=600)))


if __name__ == "__main__":
    main(*sys.argv[1:])
