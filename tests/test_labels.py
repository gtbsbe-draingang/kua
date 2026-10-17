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


def _xlsx(tmp_path, rows):
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["файл", "таймстемп", "вердикт"])
    for r in rows:
        ws.append(r)
    p = tmp_path / "l.xlsx"
    wb.save(p)
    return p


def test_violation_without_timestamp_raises(tmp_path):
    x = _xlsx(tmp_path, [("X-1", None, "нарушение")])
    with pytest.raises(ValueError, match="X-1"):
        build_labels(x, tmp_path)


def test_unrecognised_verdict_raises(tmp_path):
    x = _xlsx(tmp_path, [("X-2", "5 секунда", "возможно")])
    with pytest.raises(ValueError, match="X-2"):
        build_labels(x, tmp_path)


def test_verdict_case_whitespace_insensitive(tmp_path):
    x = _xlsx(tmp_path, [("X-3", "5 секунда", " Нарушение "), ("X-4", None, "Без  нарушения")])
    assert [(r["clip"], r["has_violation"]) for r in build_labels(x, tmp_path)] == [("X-3", 1), ("X-4", 0)]
