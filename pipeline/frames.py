"""Video access: probe, single-frame JPEG extraction, window cutting and evidence frames."""
from __future__ import annotations

import io
import subprocess
from fractions import Fraction

import av
from PIL import Image

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


_TRANSPOSE = {90: Image.Transpose.ROTATE_90, 180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_270}


def _rotation(frame) -> int:
    """Counter-clockwise degrees (0/90/180/270) to apply to make the frame upright."""
    try:
        return int(round(frame.rotation or 0)) % 360
    except Exception:
        return 0


def _upright(img: Image.Image, rot: int) -> Image.Image:
    if rot in _TRANSPOSE:
        return img.transpose(_TRANSPOSE[rot])
    if rot:
        return img.rotate(rot, expand=True)
    return img


def probe(path) -> tuple[float, float, int, int]:
    """Return (duration, fps, width, height) with display dimensions (rotation applied).
    Raises VideoError for anything unreadable."""
    try:
        with av.open(str(path)) as c:
            if not c.streams.video:
                raise VideoError("no video stream")
            s = c.streams.video[0]
            dur = _duration(c, s)
            fps = _fps(s)
            w, h = s.codec_context.width, s.codec_context.height
            first = next(c.decode(s), None)
            if first is not None and _rotation(first) in (90, 270):
                w, h = h, w
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
            _upright(chosen.to_image(), _rotation(chosen)).save(buf, format="JPEG", quality=quality)
    except VideoError:
        raise
    except Exception as e:
        raise VideoError(f"cannot decode frame: {type(e).__name__}: {e}") from e
    return Frame(t=t, jpeg=buf.getvalue())


def cut_window(path, start: float, end: float, out_path: str) -> str:
    """Re-encode [start, end] (clamped to [0, duration]) to out_path as an upright H.264 clip."""
    dur, _, _, _ = probe(path)
    start = max(float(start), 0.0)
    end = min(float(end), dur)
    if end - start <= 0.01:
        raise VideoError(f"empty window [{start:.2f}, {end:.2f}] in a {dur:.2f}s video")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(path),
           "-t", f"{end - start:.3f}", "-an", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
    except subprocess.CalledProcessError as e:
        tail = (e.stderr or b"").decode("utf-8", "replace").strip()[-500:]
        raise VideoError(f"ffmpeg failed: {tail}") from e
    except (subprocess.TimeoutExpired, OSError) as e:
        raise VideoError(f"ffmpeg failed: {type(e).__name__}: {e}") from e
    return out_path


def evidence(path, timecode: float, t_start: float, t_end: float) -> list[Frame]:
    """Up to 3 JPEG frames at the timecode, window start and window end; the first is the timecode's.
    Times are clamped to [0, duration - 1/fps] and any within 0.5 s of a kept one is dropped."""
    dur, fps, _, _ = probe(path)
    hi = max(dur - 1.0 / fps, 0.0)
    kept: list[float] = []
    for t in (timecode, t_start, t_end):
        t = min(max(float(t), 0.0), hi)
        if all(abs(t - k) >= 0.5 for k in kept):
            kept.append(t)
        if len(kept) == 3:
            break
    return [jpeg_at(path, t, 90) for t in kept]
