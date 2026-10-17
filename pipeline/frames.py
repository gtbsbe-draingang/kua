"""Video access: probe and single-frame JPEG extraction (more helpers land in a later task)."""
from __future__ import annotations

import io
from fractions import Fraction

import av

from pipeline.types import Frame


class VideoError(Exception):
    pass


def _fps(stream) -> float:
    for rate in (stream.average_rate, stream.guessed_rate):
        if rate:
            return float(rate)
    return 30.0


def _duration(container, stream) -> float:
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if container.duration:
        return container.duration / av.time_base
    return 0.0


def probe(path) -> tuple[float, float, int, int]:
    """Return (duration, fps, width, height). Raises VideoError for anything unreadable."""
    try:
        with av.open(str(path)) as c:
            if not c.streams.video:
                raise VideoError("no video stream")
            s = c.streams.video[0]
            dur = _duration(c, s)
            fps = _fps(s)
            w, h = s.codec_context.width, s.codec_context.height
    except VideoError:
        raise
    except Exception as e:  # PyAV raises many distinct error types
        raise VideoError(f"cannot open video: {type(e).__name__}: {e}") from e
    if not dur or dur <= 0:
        raise VideoError("video has zero duration")
    if not w or not h:
        raise VideoError("video has no dimensions")
    return dur, fps, w, h


def jpeg_at(path, t: float, quality: int = 90) -> Frame:
    """JPEG of the frame displayed at t (clamped to [0, duration - 1/fps])."""
    dur, fps, _, _ = probe(path)
    t = min(max(float(t), 0.0), max(dur - 1.0 / fps, 0.0))
    try:
        with av.open(str(path)) as c:
            s = c.streams.video[0]
            tb = s.time_base
            c.seek(int(Fraction(t) / tb), stream=s, backward=True, any_frame=False)
            chosen = None
            for fr in c.decode(s):
                ft = float(fr.pts * tb) if fr.pts is not None else None
                if ft is None:
                    continue
                if chosen is not None and ft > t + 1e-6:
                    break
                chosen = fr
            if chosen is None:
                raise VideoError("no frame decoded")
            buf = io.BytesIO()
            chosen.to_image().save(buf, format="JPEG", quality=quality)
    except VideoError:
        raise
    except Exception as e:
        raise VideoError(f"cannot decode frame: {type(e).__name__}: {e}") from e
    return Frame(t=t, jpeg=buf.getvalue())
