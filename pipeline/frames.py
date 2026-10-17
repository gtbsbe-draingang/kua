"""Video access: probe, upright normalisation, single-frame JPEG extraction, window cutting and evidence frames."""
from __future__ import annotations

import io
import os
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


ENCODER_THREADS = "4"  # the pod has a 16-CPU quota; x264's default thread count follows the host's cores


def _ffmpeg(cmd: list[str], out_path, timeout: float) -> None:
    """Run ffmpeg; on any failure remove the partial out_path and raise VideoError with the stderr tail."""
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
        return
    except subprocess.CalledProcessError as e:
        tail = (e.stderr or b"").decode("utf-8", "replace").strip()[-500:]
        err = VideoError(f"ffmpeg failed: {tail}")
        cause: Exception = e
    except (subprocess.TimeoutExpired, OSError) as e:
        err, cause = VideoError(f"ffmpeg failed: {type(e).__name__}: {e}"), e
    try:
        os.unlink(out_path)
    except OSError:
        pass
    raise err from cause


def normalize(path, out_path: str) -> str:
    """Return path itself when it is H.264 with no rotation tag that starts at t=0. Otherwise transcode it
    once to an upright H.264 copy at out_path and return out_path. VideoJev and InternVideo3 decode with
    PyAV and ignore rotation tags, and a start offset would shift their times against the frame times here."""
    try:
        with av.open(str(path)) as c:
            if not c.streams.video:
                raise VideoError("no video stream")
            s = c.streams.video[0]
            codec = s.codec_context.name
            first = next(c.decode(s), None)
            if first is None:
                raise VideoError("no frame decoded")
            rot = _rotation(first)
            t0 = float(first.pts * s.time_base) if first.pts is not None and s.time_base else 0.0
    except VideoError:
        raise
    except Exception as e:
        raise VideoError(f"cannot open video: {type(e).__name__}: {e}") from e
    if codec == "h264" and rot == 0 and abs(t0) < 0.05:
        return str(path)
    cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(path), "-map", "0:v:0", "-an", "-sn",
           "-dn", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "18", "-pix_fmt", "yuv420p", "-threads", ENCODER_THREADS, "-movflags", "+faststart",
           str(out_path)]
    _ffmpeg(cmd, out_path, timeout=180)
    return str(out_path)


def cut_window(path, start: float, end: float, out_path: str) -> str:
    """Re-encode [start, end] (clamped to [0, duration]) to out_path as an upright H.264 clip."""
    dur, _, _, _ = probe(path)
    start = max(float(start), 0.0)
    end = min(float(end), dur)
    if end - start <= 0.01:
        raise VideoError(f"empty window [{start:.2f}, {end:.2f}] in a {dur:.2f}s video")
    cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(path),
           "-t", f"{end - start:.3f}", "-an", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "20", "-pix_fmt", "yuv420p", "-threads", ENCODER_THREADS, "-movflags", "+faststart",
           str(out_path)]
    _ffmpeg(cmd, out_path, timeout=60)
    return out_path


def evidence(path, timecode: float, t_start: float, t_end: float) -> list[Frame]:
    """Up to 3 JPEG frames at the timecode, window start and window end; the first is the timecode's.
    Times are clamped to [0, duration - 1/fps] and any less than 0.5 s from a kept one is dropped
    (exactly 0.5 s apart counts as apart, whatever the float rounding)."""
    dur, fps, _, _ = probe(path)
    hi = max(dur - 1.0 / fps, 0.0)
    kept: list[float] = []
    for t in (timecode, t_start, t_end):
        t = min(max(float(t), 0.0), hi)
        if all(abs(t - k) >= 0.5 - 1e-6 for k in kept):
            kept.append(t)
        if len(kept) == 3:
            break
    return [jpeg_at(path, t, 90) for t in kept]
