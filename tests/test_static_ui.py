"""The result page's own assets: every file it references is served, with a type the browser accepts, and
nothing is loaded from the internet (the service must work offline)."""
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api
from pipeline import stub
from rules import load_rules

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
TEXT_ASSETS = ["index.html", "ui.css", "ui.js", "fonts/fonts.css"]
SVG_NS = "http://www.w3.org/2000/svg"  # the only URL allowed: an XML namespace name, never fetched


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    rules = load_rules(ROOT / "rules.yaml", strict=False)
    app = api.create_app(tmp_path_factory.mktemp("jobs"), "k", stub.analyze, rules, workers=1)
    with TestClient(app) as c:
        yield c


def page_refs() -> list[str]:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    return [m.split("?")[0] for m in re.findall(r'(?:href|src)="(/static/[^"]+)"', html)]


def test_index_references_its_assets():
    refs = page_refs()
    for needed in ("/static/ui.css", "/static/ui.js", "/static/fonts/fonts.css"):
        assert needed in refs


def test_every_referenced_asset_is_served_with_a_usable_type(client):
    assert client.get("/").headers["content-type"].startswith("text/html")
    for ref in page_refs():
        r = client.get(ref)
        assert r.status_code == 200, ref
        assert r.content == (STATIC / ref.removeprefix("/static/")).read_bytes(), ref
        ctype = r.headers["content-type"]
        if ref.endswith(".css"):
            assert ctype.startswith("text/css"), (ref, ctype)
        elif ref.endswith(".js"):
            assert "javascript" in ctype, (ref, ctype)


def test_every_font_in_fonts_css_exists_and_is_served(client):
    css = (STATIC / "fonts" / "fonts.css").read_text(encoding="utf-8")
    fonts = re.findall(r"url\(([^)]+\.woff2)\)", css)
    assert len(fonts) == 6
    for name in fonts:
        assert (STATIC / "fonts" / name).is_file(), name
        r = client.get(f"/static/fonts/{name}")
        assert r.status_code == 200 and r.content[:4] == b"wOF2", name


def test_no_remote_resources():
    for name in TEXT_ASSETS:
        text = (STATIC / name).read_text(encoding="utf-8")
        urls = set(re.findall(r"(?i)(?:https?:)?//[a-z0-9-]+(?:\.[a-z0-9-]+)+[^\s\"')]*", text))
        assert urls <= {SVG_NS}, (name, urls - {SVG_NS})
        assert "@import" not in text, name
