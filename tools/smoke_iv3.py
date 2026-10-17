"""Smoke test for the InternVideo3 check server (servers/iv3_server.py).

    env/iv3/bin/python tools/smoke_iv3.py [--url http://127.0.0.1:30000]

Cuts two 6 s windows with ffmpeg into cache/smoke/: X-1 around its label at
22 s (19-25 s) and the middle of the clean clip X-3. POSTs each to /verify with
a check prompt that asks for ANSWER / TYPE / OBSERVATION, forbids naming or
identifying people and requires «кассир», «клиент» or «инкассатор». Prints
p_yes, type, observation and latency. Also checks /healthz and that a missing
file gives HTTP 500 with a JSON error.

Exits 1 unless every window got a numeric p_yes, a TYPE and an OBSERVATION.
Standard library only (plus pyyaml for the rule ids), so env/iv3 or env/main
can run it.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("MTA_ROOT", "/home/jovyan/model_train_amir"))
SVC = Path(__file__).resolve().parent.parent
FFMPEG = os.environ.get("FFMPEG", "/opt/conda/bin/ffmpeg")
WINDOW = 6.0

PROMPT = """This is a short clip from a CCTV camera above a bank cash desk.
Question: {question}

Reply in exactly three lines:
ANSWER: YES or NO
TYPE: the id of the matching violation type from this list, or none: {ids}
OBSERVATION: one or two sentences in Russian describing what happens in the clip.

In OBSERVATION describe only actions with cash, documents and objects. Never name, identify or \
describe any person: no names, faces, appearance, clothing, age or gender. Refer to people only \
as «кассир», «клиент» or «инкассатор»."""

QUESTION = ("Does the clip show a violation of cash-handling rules at the cash desk, "
            "such as cash left without control or cash handed over outside normal service?")


def find_clip(stem: str) -> Path:
    """data/clips/<stem>.mov, matching a Cyrillic Х in the file name as a Latin X."""
    def norm(name: str) -> str:
        return unicodedata.normalize("NFC", name).replace("Х", "X").replace("х", "x")
    for p in sorted((ROOT / "data" / "clips").iterdir()):
        if norm(p.name) == f"{stem}.mov":
            return p
    raise SystemExit(f"clip {stem}.mov not found in {ROOT / 'data' / 'clips'}")


def duration(path: Path) -> float:
    import av
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        if s.duration is not None:
            return float(s.duration * s.time_base)
        return c.duration / 1e6


def cut(src: Path, start: float, out: Path) -> Path:
    """Re-encode [start, start + WINDOW), video only.

    MPEG-4 part 2 at -q:v 2 (near-lossless at this size): the pod's ffmpeg 4.3
    has no libx264, and its libopenh264 fails to load ("Incorrect library
    version loaded")."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [FFMPEG, "-nostdin", "-hide_banner", "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", str(src),
           "-t", f"{WINDOW:.3f}", "-an", "-c:v", "mpeg4", "-q:v", "2", "-pix_fmt", "yuv420p",
           "-threads", "4", str(out)]
    subprocess.run(cmd, check=True)
    return out


def call(url: str, payload: dict | None = None, timeout: float = 600) -> tuple[int, dict]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--fps", type=float, default=4.0)
    args = ap.parse_args()

    rules = yaml.safe_load((SVC / "rules.yaml").read_text())["rules"]
    ids = ", ".join(r["id"] for r in rules if r.get("detectable", True))
    prompt = PROMPT.format(question=QUESTION, ids=ids)

    status, body = call(f"{args.url}/healthz", timeout=10)
    print(f"healthz: HTTP {status} {body}")
    if status != 200 or body != {"ok": True}:
        return 1

    x1, x3 = find_clip("X-1"), find_clip("X-3")
    d3 = duration(x3)
    s3 = max(0.0, d3 / 2 - WINDOW / 2)
    smoke = ROOT / "cache" / "smoke"
    windows = [
        ("X-1 around the label at 22 s", cut(x1, 19.0, smoke / "X-1_19-25.mp4")),
        (f"X-3 (clean), middle of {d3:.1f} s", cut(x3, s3, smoke / f"X-3_{s3:.0f}-{s3 + WINDOW:.0f}.mp4")),
    ]

    ok = True
    for name, path in windows:
        t0 = time.perf_counter()
        status, body = call(f"{args.url}/verify", {"video_path": str(path), "prompt": prompt, "fps": args.fps})
        latency = time.perf_counter() - t0
        print(f"\n=== {name}: {path.name}")
        if status != 200:
            print(f"HTTP {status}: {body}")
            ok = False
            continue
        p = body.get("p_yes")
        good = isinstance(p, (int, float)) and bool(body.get("type_id")) and bool(body.get("observation"))
        ok &= good
        print(f"p_yes        {p:.4f}" if isinstance(p, (int, float)) else f"p_yes        {p!r}")
        print(f"answer       {body.get('answer')!r}  (yes_no_mass {body.get('yes_no_mass', float('nan')):.3f})")
        print(f"type_id      {body.get('type_id')!r}")
        print(f"observation  {body.get('observation')!r}")
        print(f"raw          {body.get('raw')!r}")
        print(f"frames {body.get('frames')}, grid_thw {body.get('grid_thw')}, prompt tokens {body.get('prompt_tokens')}")
        print(f"latency      {latency:.2f} s (server {body.get('timings')})")
        print("format       " + ("OK" if good else "MISSING p_yes, TYPE or OBSERVATION"))

    status, body = call(f"{args.url}/verify", {"video_path": str(smoke / "missing.mp4"), "prompt": prompt})
    err_ok = status == 500 and "error" in body
    print(f"\nmissing file: HTTP {status} {body} -> {'OK' if err_ok else 'expected HTTP 500 with an error'}")
    ok &= err_ok
    print("\nRESULT: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
