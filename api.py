"""HTTP API: upload a video, poll the job. The pipeline is injected as `analyze`."""
from __future__ import annotations

import base64
import hmac
import os
import queue
import re
import shutil
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

import jobs
from pipeline.types import PipelineResult, to_debug
from rules import citation_text, load_rules

CHUNK = 1024 * 1024
DEADLINE_SEC = 240
JOB_ID_RE = re.compile(r"^[0-9a-f]{12}$")
STATIC_INDEX = Path(__file__).resolve().parent / "static" / "index.html"


def _r1(x: float) -> float:
    return round(float(x), 1)


def to_api(job_id: str, result: PipelineResult, rules: dict) -> dict:
    violations = []
    for ev in sorted((e for e in result.events if e.kept), key=lambda e: e.timecode):
        rid = ev.judge_rule_id if ev.type_source == "judge" and ev.judge_rule_id in rules else ev.rule_id
        rule = rules[rid]
        c = rule["citations"][0]
        desc = " ".join(
            p.strip()
            for p in (ev.observation, ev.justification, f"Основание: {citation_text(rule)}.")
            if p and p.strip()
        )
        score = ev.verify_score if ev.verify_score is not None else ev.ground_score
        violations.append({
            "type": rule["name"],
            "timecode_sec": _r1(ev.timecode),
            "description": desc,
            "evidence_frames": [
                {"timestamp_sec": _r1(f.t), "image_base64": base64.b64encode(f.jpeg).decode("ascii")}
                for f in ev.evidence[:3]
            ],
            "type_id": rid,
            "rule": {k: c.get(k, "") for k in ("document", "clause", "quote", "url")},
            "confidence": round(float(score), 2),
        })
    return {"job_id": job_id, "status": "done", "verdict": result.verdict, "violations": violations}


def create_app(jobs_dir: Path, api_key: str, analyze, rules: dict, workers: int = 2) -> FastAPI:
    jobs_dir = Path(jobs_dir)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = int(os.environ.get("MAX_UPLOAD_BYTES", 200 * 1024 * 1024))
    q: queue.Queue[str] = queue.Queue()

    def run_job(jid: str) -> None:
        d = jobs_dir / jid
        try:
            state = jobs.read_state(d)
            if not (d / "input.mov").is_file():
                raise FileNotFoundError("input.mov is missing")
            deadline = float(state["created_at"]) + DEADLINE_SEC
            result = analyze(str(d / "input.mov"), deadline, lambda s: jobs.write_state(d, stage=s))
            jobs.write_json_atomic(d / "result.json", to_api(jid, result, rules))
            jobs.write_json_atomic(d / "debug.json", to_debug(result))
            jobs.write_state(d, status="done")
        except Exception as e:
            try:
                jobs.write_state(d, status="failed", error=f"{type(e).__name__}: {e}")
            except Exception:
                pass

    def worker() -> None:
        while True:
            jid = q.get()
            try:
                run_job(jid)
            finally:
                q.task_done()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        for i in range(workers):
            threading.Thread(target=worker, name=f"worker-{i}", daemon=True).start()
        for jid in jobs.pending_jobs(jobs_dir):
            q.put(jid)
        yield

    app = FastAPI(lifespan=lifespan)

    def check_key(x_api_key: str | None = Header(default=None)) -> None:
        if x_api_key is None or not hmac.compare_digest(x_api_key.encode(), api_key.encode()):
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.get("/")
    def index():
        if not STATIC_INDEX.is_file():
            raise HTTPException(status_code=404, detail="UI not installed")
        return FileResponse(STATIC_INDEX)

    @app.post("/api/v1/analyze", dependencies=[Depends(check_key)])
    async def analyze_route(file: UploadFile = File(...)):
        # check_key runs before the body is parsed (dependencies resolve first)
        jid = jobs.new_job(jobs_dir)
        d = jobs_dir / jid
        total = 0
        try:
            with open(d / "input.mov", "wb") as out:
                while chunk := await file.read(CHUNK):
                    total += len(chunk)
                    if total > max_bytes:
                        raise HTTPException(status_code=413, detail="file too large")
                    out.write(chunk)
        except BaseException:
            shutil.rmtree(d, ignore_errors=True)
            raise
        jobs.write_state(d, status="processing", stage="queued", created_at=time.time(), error=None)
        q.put(jid)
        return {"job_id": jid, "status": "processing"}

    @app.middleware("http")
    async def reject_oversize(request: Request, call_next):
        if request.url.path == "/api/v1/analyze" and request.method == "POST":
            cl = request.headers.get("content-length")
            if cl and cl.isdigit() and int(cl) > max_bytes + CHUNK:
                from fastapi.responses import JSONResponse

                return JSONResponse({"detail": "file too large"}, status_code=413)
        return await call_next(request)

    @app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(check_key)])
    def get_job(job_id: str, debug: int = 0):
        if not JOB_ID_RE.match(job_id):
            raise HTTPException(status_code=404, detail="unknown job")
        d = jobs_dir / job_id
        try:
            state = jobs.read_state(d)
        except (OSError, ValueError):
            raise HTTPException(status_code=404, detail="unknown job")
        status = state.get("status")
        if status == "processing":
            return {"job_id": job_id, "status": "processing", "stage": state.get("stage")}
        if status == "failed":
            return {"job_id": job_id, "status": "failed", "error": state.get("error") or "unknown error"}
        import json

        body = json.loads((d / "result.json").read_text(encoding="utf-8"))
        if debug:
            body["debug"] = json.loads((d / "debug.json").read_text(encoding="utf-8"))
        return body

    return app


_app = None


def __getattr__(name: str):
    global _app
    if name == "app":
        if _app is None:
            key = os.environ.get("API_KEY", "")
            if not key:
                raise RuntimeError("API_KEY environment variable is required")
            from pipeline import get_analyze

            rules = load_rules(os.environ.get("RULES_PATH", "rules.yaml"),
                               strict=os.environ.get("RULES_STRICT") != "0")
            _app = create_app(Path(os.environ.get("JOBS_DIR", "jobs")), key, get_analyze(), rules)
        return _app
    raise AttributeError(name)
