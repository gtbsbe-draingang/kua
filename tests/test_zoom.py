import pytest

from pipeline import frames, zoom
from pipeline.types import Frame


def test_union_box_pads_and_clamps():
    boxes = [(100, 200, 300, 600), (250, 150, 400, 500)]
    assert zoom.union_box(boxes, 464, 848) == (70, 104, 430, 646)
    # padded past the frame edge: clamps to it
    assert zoom.union_box([(2, 300, 100, 500)], 464, 848) == (0, 280, 110, 520)
    assert zoom.union_box([(300, 400, 462, 846)], 464, 848) == (282, 354, 464, 848)


def test_union_box_none_when_empty():
    assert zoom.union_box([], 464, 848) is None


def test_union_box_none_when_nearly_full_frame():
    assert zoom.union_box([(0, 0, 464, 848)], 464, 848) is None
    assert zoom.union_box([(20, 20, 440, 820)], 464, 848) is None


def _frame(t):
    from io import BytesIO
    from PIL import Image
    buf = BytesIO()
    Image.new("RGB", (464, 848)).save(buf, "JPEG")
    return Frame(t=t, jpeg=buf.getvalue())


def test_zoom_box_none_when_no_person(monkeypatch):
    monkeypatch.setattr(frames, "jpeg_at", lambda path, t, quality=90: _frame(t))
    monkeypatch.setattr(zoom, "_detect", lambda images: [[("chair", 0.9, (10, 10, 100, 100))] for _ in images])
    assert zoom.person_box("clip.mp4", 0.0, 8.0) is None


def test_zoom_box_uses_upright_frames(monkeypatch):
    calls = []

    def fake_jpeg_at(path, t, quality=90):
        calls.append((path, t))
        return _frame(t)

    monkeypatch.setattr(frames, "jpeg_at", fake_jpeg_at)
    monkeypatch.setattr(zoom, "_detect", lambda images: [[
        ("person", 0.9, (100, 200, 200, 400)), ("person", 0.3, (0, 0, 460, 840))] for _ in images])
    box = zoom.person_box("upright.mp4", 2.0, 10.0)
    assert calls == [("upright.mp4", 2.0), ("upright.mp4", 6.0), ("upright.mp4", 10.0)]
    assert box == (90, 180, 210, 420)  # the 0.3 detection is below SCORE_MIN
