import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import augment_labels as al  # noqa: E402

HEAD = "clip,has_violation,t_label,t_start,t_end,type_id,angle,source,note\n"


def test_twelve_rows(tmp_path):
    src = tmp_path / "labels.csv"
    src.write_text(HEAD + "X-1,1,22,,,,main,given,\nX-3,0,,,,,main,given,\n"
                   "X-4,1,57,,,,main,given,merged\n")
    dst = tmp_path / "out" / "aug.csv"
    assert al.write_labels(src, dst) == 12
    rows = list(csv.DictReader(open(dst)))
    assert list(rows[0]) == HEAD.strip().split(",")
    assert [r["clip"] for r in rows[:4]] == ["X-1-tr", "X-1-hf", "X-1-crop", "X-1-dark"]
    assert all(r["angle"] == "other" and r["source"] == "augmented" for r in rows)
    assert [r["note"] for r in rows[:4]] == list(al.VARIANTS)
    assert rows[0]["t_label"] == "22" and rows[4]["has_violation"] == "0"
    assert src.read_text().count("\n") == 4  # source untouched


def test_only_and_cyrillic(tmp_path):
    d = tmp_path / "c"; d.mkdir()
    (d / "Х-3.mov").write_bytes(b"")
    assert al.resolve(d, "X-3").name == "Х-3.mov"
    rows = [{"clip": "X-1", "angle": "main", "source": "given", "note": ""},
            {"clip": "X-3", "angle": "main", "source": "given", "note": ""}]
    assert len(al.augment_rows(rows, ["X-3"])) == 4
