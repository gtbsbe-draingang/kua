import shutil
import subprocess
from pathlib import Path

import pytest


def _make_clip(path: Path, size: str) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
         f"testsrc=size={size}:rate=30", "-t", "6", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


@pytest.fixture(scope="session")
def portrait_clip(tmp_path_factory):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg missing")
    return _make_clip(tmp_path_factory.mktemp("clips") / "portrait.mov", "464x848")


@pytest.fixture(scope="session")
def landscape_clip(tmp_path_factory):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg missing")
    return _make_clip(tmp_path_factory.mktemp("clips") / "landscape.mov", "1920x1080")


@pytest.fixture(scope="session")
def rotated_clip(portrait_clip, tmp_path_factory):
    """The 464x848 portrait clip re-muxed with a 90 degree display rotation (displays as 848x464)."""
    out = tmp_path_factory.mktemp("rot") / "rotated.mov"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-display_rotation", "90", "-i", str(portrait_clip),
         "-c", "copy", str(out)],
        check=True,
    )
    return out
