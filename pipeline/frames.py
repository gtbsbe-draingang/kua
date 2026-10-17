"""Video access: probe, upright normalisation, single-frame JPEG extraction, window cutting and evidence frames."""
from __future__ import annotations

import functools
import io
import logging
import os
import subprocess
from fractions import Fraction

import av
from PIL import Image

from pipeline.types import Frame

log = logging.getLogger(__name__)

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


ENCODER_THREADS = "4"  # the pod has a 16-CPU quota; an encoder's default thread count follows the host's cores
# Encoders for the internal copies (windows, upright copies), best first. They are model inputs only: the UI
# plays the user's own file. mpeg4 is FFmpeg's native encoder and always present (the pod's ffmpeg has no
# libx264, and its libopenh264 is listed but does not load).
ENCODER_ORDER = ("libx264", "libopenh264", "mpeg4")
ENCODER_CODEC = {"libx264": "h264", "libopenh264": "h264", "mpeg4": "mpeg4"}


def ffmpeg_bin() -> str:
    return os.environ.get("FFMPEG") or "ffmpeg"


@functools.lru_cache(maxsize=None)
def _usable_encoders(ffmpeg: str) -> frozenset[str]:
    """Encoders from ENCODER_ORDER that this ffmpeg lists AND that encode a test frame (a listed encoder
    can still fail to load its library)."""
    try:
        listing = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True,
                                 timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    listed = {parts[1] for parts in (line.split() for line in listing.splitlines()) if len(parts) > 1}
    usable = set()
    for enc in ENCODER_ORDER:
        if enc not in listed:
            continue
        cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
               "testsrc=size=64x64:rate=1", "-frames:v", "1", *_encoder_args(enc, 20), "-f", "null", "-"]
        try:
            if subprocess.run(cmd, capture_output=True, timeout=30).returncode == 0:
                usable.add(enc)
        except (OSError, subprocess.SubprocessError):
            pass
    return frozenset(usable)


def video_encoder() -> str:
    """VIDEO_ENCODER when set, else the first usable of ENCODER_ORDER; mpeg4 as the last resort."""
    forced = os.environ.get("VIDEO_ENCODER", "").strip()
    if forced:
        if forced not in ENCODER_CODEC:
            raise VideoError(f"VIDEO_ENCODER={forced!r} is not one of {', '.join(ENCODER_ORDER)}")
        return forced
    usable = _usable_encoders(ffmpeg_bin())
    return next((e for e in ENCODER_ORDER if e in usable), "mpeg4")


def _encoder_args(encoder: str, crf: int) -> list[str]:
    if encoder == "libx264":
        args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf)]
    elif encoder == "libopenh264":
        args = ["-c:v", "libopenh264", "-b:v", "4M"]
    else:
        args = ["-c:v", "mpeg4", "-q:v", "2"]
    return args + ["-pix_fmt", "yuv420p", "-threads", ENCODER_THREADS]


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


def normalize(path, out_path: str, timeout: float = 180.0) -> str:
    """Return path itself when it is H.264 (or already the encoder's codec) with no rotation tag and starts
    at t=0. Otherwise transcode it once to an upright copy at out_path and return out_path. VideoJev and
    InternVideo3 decode with PyAV and ignore rotation tags, and a start offset would shift their times
    against the frame times here."""
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
    encoder = video_encoder()
    if codec in ("h264", ENCODER_CODEC[encoder]) and rot == 0 and abs(t0) < 0.05:
        return str(path)
    cmd = [ffmpeg_bin(), "-nostdin", "-y", "-loglevel", "error", "-i", str(path), "-map", "0:v:0", "-an", "-sn",
           "-dn", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", *_encoder_args(encoder, 18),
           "-movflags", "+faststart", str(out_path)]
    _ffmpeg(cmd, out_path, timeout=timeout)
    return str(out_path)


def _crop_filter(crop: tuple[int, int, int, int], long_side: int) -> str:
    """`crop=w:h:x:y,scale=W:H` for a crop box (x0, y0, x1, y1): the longer side becomes long_side, both even."""
    x0, y0, x1, y1 = crop
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        raise VideoError(f"empty crop box {crop}")
    k = long_side / max(w, h)
    sw, sh = (long_side, max(2 * round(h * k / 2), 2)) if w >= h else (max(2 * round(w * k / 2), 2), long_side)
    return f"crop={w}:{h}:{x0}:{y0},scale={sw}:{sh}"


def cut_window(path, start: float, end: float, out_path: str,
               crop: tuple[int, int, int, int] | None = None, long_side: int = 896) -> str:
    """Re-encode [start, end] (clamped to [0, duration]) to out_path as an upright clip (video_encoder()).
    With crop (x0, y0, x1, y1) in upright pixels, the clip is cropped to it and scaled to long_side."""
    dur, _, _, _ = probe(path)
    start = max(float(start), 0.0)
    end = min(float(end), dur)
    if end - start <= 0.01:
        raise VideoError(f"empty window [{start:.2f}, {end:.2f}] in a {dur:.2f}s video")
    vf = ["-vf", _crop_filter(crop, long_side)] if crop else []
    cmd = [ffmpeg_bin(), "-nostdin", "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(path),
           "-t", f"{end - start:.3f}", "-an", *vf, *_encoder_args(video_encoder(), 20),
           "-movflags", "+faststart", str(out_path)]
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
