"""HTTP API: upload a video, poll the job. The pipeline is injected as `analyze`."""
from __future__ import annotations

import base64
import hmac
import json
import logging
import os
import queue
import re
import shutil
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from python_multipart.multipart import MultipartParser, parse_options_header

import jobs
from pipeline.types import PipelineResult, to_debug
from rules import citation_text, load_rules

log = logging.getLogger(__name__)
CHUNK = 1024 * 1024
DEADLINE_SEC = 240
JOB_ID_RE = re.compile(r"^[0-9a-f]{12}$")
STATIC_INDEX = Path(__file__).resolve().parent / "static" / "index.html"


def _r1(x: float) -> float:
    return round(float(x), 1)


def to_api(job_id: str, result: PipelineResult, rules: dict) -> dict:
    violations = []
    for ev in sorted((e for e in result.events if e.kept), key=lambda e: e.timecode):
        judged = ev.type_source == "judge" and ev.judge_rule_id in rules
        rid = ev.judge_rule_id if judged else ev.rule_id
        rule = rules[rid]
        # the judge's corpus citation when it gave one for its rule, else the rule's first catalogue citation
        cit = ev.citation if judged and ev.citation else None
        c = cit or rule["citations"][0]
        basis = f"{cit['document']}, {cit['clause']}" if cit else citation_text(rule)
        desc = " ".join(
            p.strip()
            for p in (ev.observation, ev.justification, f"Основание: {basis}.")
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
                log.exception("could not record failure of job %s", jid)

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
            jobs.write_state(jobs_dir / jid, stage="queued")
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

    async def receive_upload(request: Request, out_path: Path) -> None:
        """Stream the multipart body, writing only the `file` part to out_path."""
        ctype, params = parse_options_header(request.headers.get("content-type", ""))
        boundary = params.get(b"boundary")
        if ctype.lower() != b"multipart/form-data" or not boundary:
            raise HTTPException(status_code=400, detail="expected multipart/form-data with a boundary")
        st = {"total": 0, "exceeded": False, "found": False, "done": False, "out": None,
              "field": b"", "value": b"", "headers": {}}

        def on_part_begin():
            st["headers"] = {}
            st["field"] = st["value"] = b""

        def on_header_field(data, start, end):
            st["field"] += data[start:end]

        def on_header_value(data, start, end):
            st["value"] += data[start:end]

        def on_header_end():
            st["headers"][st["field"].lower()] = st["value"]
            st["field"] = st["value"] = b""

        def on_headers_finished():
            disp = st["headers"].get(b"content-disposition", b"")
            _, dparams = parse_options_header(disp)
            if dparams.get(b"name") == b"file" and not st["found"]:
                st["found"] = True
                st["out"] = open(out_path, "wb")

        def on_part_data(data, start, end):
            if st["out"] is None:
                return
            st["total"] += end - start
            if st["total"] > max_bytes:
                st["exceeded"] = True
                return
            st["out"].write(data[start:end])

        def on_part_end():
            if st["out"] is not None:
                st["out"].close()
                st["out"] = None
                st["done"] = True

        parser = MultipartParser(boundary, {
            "on_part_begin": on_part_begin, "on_header_field": on_header_field,
            "on_header_value": on_header_value, "on_header_end": on_header_end,
            "on_headers_finished": on_headers_finished, "on_part_data": on_part_data,
            "on_part_end": on_part_end,
        })
        try:
            async for chunk in request.stream():
                parser.write(chunk)
                if st["exceeded"]:
                    raise HTTPException(status_code=413, detail="file too large")
            parser.finalize()
        except HTTPException:
            raise
        except OSError:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"malformed multipart body: {e}")
        finally:
            if st["out"] is not None:
                st["out"].close()
        if not st["done"]:
            raise HTTPException(status_code=422, detail="missing multipart field 'file'")

    @app.post("/api/v1/analyze", dependencies=[Depends(check_key)])
    async def analyze_route(request: Request):
        # No body parameters are declared, so the key is checked before any body byte is read.
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > max_bytes + CHUNK:
            raise HTTPException(status_code=413, detail="file too large")
        jid = jobs.new_job(jobs_dir)
        d = jobs_dir / jid
        try:
            await receive_upload(request, d / "input.mov")
        except BaseException:
            shutil.rmtree(d, ignore_errors=True)
            raise
        jobs.write_state(d, status="processing", stage="queued", created_at=time.time(), error=None)
        q.put(jid)
        return {"job_id": jid, "status": "processing"}

    @app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(check_key)])
    def get_job(job_id: str, debug: bool = False):
        if not JOB_ID_RE.match(job_id):
            raise HTTPException(status_code=404, detail="unknown job")
        d = jobs_dir / job_id
        try:
            state = jobs.read_state(d)
        except (OSError, ValueError):
            raise HTTPException(status_code=404, detail="unknown job")
        status = state.get("status")
        if status == "failed":
            return {"job_id": job_id, "status": "failed", "error": state.get("error") or "unknown error"}
        if status != "done":
            return {"job_id": job_id, "status": "processing", "stage": state.get("stage")}
        try:
            body = json.loads((d / "result.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            return {"job_id": job_id, "status": "failed",
                    "error": f"result unavailable: {type(e).__name__}: {e}"}
        if debug:
            try:
                body["debug"] = json.loads((d / "debug.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
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
