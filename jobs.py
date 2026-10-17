"""Job folders: <jobs_dir>/<id>/{input.mov,state.json,result.json,debug.json}."""
from __future__ import annotations

import json
import os
import secrets
import tempfile
from pathlib import Path


def new_job(jobs_dir) -> str:
    jobs_dir = Path(jobs_dir)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    while True:
        jid = secrets.token_hex(6)
        try:
            (jobs_dir / jid).mkdir()
            return jid
        except FileExistsError:
            continue


def write_json_atomic(path, obj) -> None:
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_state(job_dir) -> dict:
    with open(Path(job_dir) / "state.json", encoding="utf-8") as f:
        return json.load(f)


def write_state(job_dir, **fields) -> None:
    job_dir = Path(job_dir)
    try:
        state = read_state(job_dir)
    except FileNotFoundError:
        state = {}
    state.update(fields)
    write_json_atomic(job_dir / "state.json", state)


def pending_jobs(jobs_dir) -> list[str]:
    jobs_dir = Path(jobs_dir)
    found: list[tuple[float, str]] = []
    if not jobs_dir.is_dir():
        return []
    for d in jobs_dir.iterdir():
        try:
            s = read_state(d)
        except (OSError, ValueError):
            continue
        if s.get("status") == "processing":
            found.append((float(s.get("created_at") or 0), d.name))
    return [name for _, name in sorted(found)]
