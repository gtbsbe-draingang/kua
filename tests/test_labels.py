from pathlib import Path

import pytest

from eval.make_labels import build_labels, parse_seconds

XLSX = Path("../open/open_labels.xlsx")
CLIPS = Path("../open/videos")
real = pytest.mark.skipif(not CLIPS.exists(), reason="no ../open/videos")


def test_parse_seconds():
    assert parse_seconds("22 секунда") == 22
    assert parse_seconds("1минута 8 секунда") == 68
    assert parse_seconds("1 минута 8 секунд") == 68


@pytest.fixture(scope="module")
def rows():
    return build_labels(XLSX, CLIPS)


@real
def test_clip_counts(rows):
    clips = {r["clip"] for r in rows}
    assert len(clips) == 40
    pos = {r["clip"] for r in rows if r["has_violation"] == 1}
    neg = {r["clip"] for r in rows if r["has_violation"] == 0}
    assert (len(pos), len(neg)) == (21, 19)


@real
def test_x4_two_events(rows):
    assert [r["t_label"] for r in rows if r["clip"] == "X-4"] == [57, 68]


@real
def test_dupes_absent_and_names_latin(rows):
    names = {r["clip"] for r in rows}
    assert "X-47" not in names and "X-48" not in names
    assert all(n.startswith("X-") and n.isascii() for n in names)


@real
def test_constant_columns(rows):
    assert all(r["angle"] == "main" and r["source"] == "given" for r in rows)
    assert all(r["t_start"] == "" and r["t_end"] == "" and r["type_id"] == "" for r in rows)
