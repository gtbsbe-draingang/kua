import io

import pytest
from PIL import Image

from pipeline.frames import VideoError, jpeg_at, probe


def test_probe_plain(portrait_clip):
    dur, fps, w, h = probe(portrait_clip)
    assert abs(dur - 6) < 0.2 and abs(fps - 30) < 0.5 and (w, h) == (464, 848)


def test_probe_reports_display_size_for_rotated(rotated_clip):
    _, _, w, h = probe(rotated_clip)
    assert (w, h) == (848, 464)


def test_jpeg_at_is_upright_for_rotated(rotated_clip):
    img = Image.open(io.BytesIO(jpeg_at(rotated_clip, 1.0).jpeg))
    assert img.size == (848, 464)


def test_jpeg_at_plain_and_clamped(portrait_clip):
    fr = jpeg_at(portrait_clip, 999)
    assert fr.t < 6.0 and Image.open(io.BytesIO(fr.jpeg)).size == (464, 848)
    assert jpeg_at(portrait_clip, -5).t == 0.0


def test_probe_garbage(tmp_path):
    p = tmp_path / "t.mov"
    p.write_text("nope")
    with pytest.raises(VideoError):
        probe(p)
