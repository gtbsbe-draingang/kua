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


def test_window_clamped_at_edges(portrait_clip, tmp_path):
    from pipeline.frames import cut_window
    out = cut_window(portrait_clip, -2, 3, str(tmp_path / "w.mp4"))
    dur, _, w, h = probe(out)
    assert out == str(tmp_path / "w.mp4") and abs(dur - 3) < 0.3 and (w, h) == (464, 848)


def test_window_end_beyond_duration(portrait_clip, tmp_path):
    from pipeline.frames import cut_window
    out = cut_window(portrait_clip, 4, 99, str(tmp_path / "w.mp4"))
    assert abs(probe(out)[0] - 2) < 0.3


def test_window_empty_raises(portrait_clip, tmp_path):
    from pipeline.frames import cut_window
    with pytest.raises(VideoError):
        cut_window(portrait_clip, 10, 12, str(tmp_path / "w.mp4"))


def test_window_rotated_is_upright(rotated_clip, tmp_path):
    from pipeline.frames import cut_window
    out = cut_window(rotated_clip, 1, 3, str(tmp_path / "w.mp4"))
    _, _, w, h = probe(out)
    assert (w, h) == (848, 464)


def test_evidence_clamped_and_deduped(portrait_clip):
    from pipeline.frames import evidence
    fr = evidence(portrait_clip, 5.9, 5.8, 7.0)
    assert 1 <= len(fr) <= 2 and all(f.t <= 6.0 for f in fr) and abs(fr[0].t - 5.9) < 0.05


def test_evidence_three_distinct(portrait_clip):
    from pipeline.frames import evidence
    fr = evidence(portrait_clip, 3.0, 1.0, 5.0)
    assert len(fr) == 3 and abs(fr[0].t - 3.0) < 0.05
    assert all(f.jpeg[:2] == b"\xff\xd8" for f in fr)
