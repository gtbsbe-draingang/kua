import io
import subprocess

import av
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


def test_evidence_half_second_apart_is_kept_despite_float_error(portrait_clip):
    from pipeline.frames import evidence
    assert abs(0.2 - 0.7) < 0.5  # 0.49999999999999994 in binary floating point
    fr = evidence(portrait_clip, 0.7, 0.2, 1.2)
    assert [round(f.t, 6) for f in fr] == [0.7, 0.2, 1.2]


def _failing_ffmpeg(calls):
    def run(cmd, **kw):
        calls.append(cmd)
        with open(cmd[-1], "wb") as f:
            f.write(b"partial")
        raise subprocess.CalledProcessError(1, cmd, stderr=b"boom")
    return run


def test_window_failure_removes_partial_output(portrait_clip, tmp_path, monkeypatch):
    from pipeline import frames
    calls = []
    monkeypatch.setattr(frames.subprocess, "run", _failing_ffmpeg(calls))
    out = tmp_path / "w.mp4"
    with pytest.raises(VideoError, match="boom"):
        frames.cut_window(portrait_clip, 1, 3, str(out))
    assert not out.exists() and "-nostdin" in calls[0]


def _coded(path):
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        first = next(c.decode(s))
        return s.codec_context.name, s.codec_context.width, s.codec_context.height, int(first.rotation or 0)


def test_normalize_keeps_upright_h264(portrait_clip, tmp_path):
    from pipeline.frames import normalize
    out = tmp_path / "norm.mp4"
    assert normalize(portrait_clip, str(out)) == str(portrait_clip)
    assert not out.exists()


def test_normalize_rotated_gives_upright_h264_copy(rotated_clip, tmp_path):
    from pipeline.frames import normalize
    out = str(tmp_path / "norm.mp4")
    assert normalize(rotated_clip, out) == out
    assert _coded(out) == ("h264", 848, 464, 0)
    assert abs(probe(out)[0] - 6.0) < 0.2


def test_normalize_transcodes_other_codecs(tmp_path):
    from pipeline.frames import normalize
    src = tmp_path / "mpeg4.mov"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30",
                    "-t", "2", "-c:v", "mpeg4", str(src)], check=True)
    out = str(tmp_path / "norm.mp4")
    assert normalize(src, out) == out and _coded(out) == ("h264", 320, 240, 0)


def test_normalize_garbage_and_failure_cleanup(rotated_clip, tmp_path, monkeypatch):
    from pipeline import frames
    bad = tmp_path / "bad.mov"
    bad.write_text("nope")
    with pytest.raises(VideoError):
        frames.normalize(bad, str(tmp_path / "n1.mp4"))
    calls = []
    monkeypatch.setattr(frames.subprocess, "run", _failing_ffmpeg(calls))
    out = tmp_path / "n2.mp4"
    with pytest.raises(VideoError, match="boom"):
        frames.normalize(rotated_clip, str(out))
    assert not out.exists() and "-nostdin" in calls[0]
