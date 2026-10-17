"""HTTP clients for the InternVideo3 check server and the optional rule judge. Neither ever raises:
any error, timeout, non-200 reply or malformed answer gives None / empty values."""
from __future__ import annotations

import base64
import json
import logging
import math
import os
import re

import httpx

from pipeline.types import Frame

log = logging.getLogger(__name__)

IV3_URL = "http://127.0.0.1:30000/verify"
IV3_MODEL = "yanziang/InternVideo3-8B-Instruct"  # served by servers/iv3_server.py; recorded in versions only
JUDGE_URL = "http://127.0.0.1:30001"
JUDGE_MODEL = "Qwen/Qwen3.5-9B"
VERIFY_FPS = 4
CONNECT_TIMEOUT = 2.0  # an absent server fails fast
OBSERVATION_MAX = 1000
ROLES = "«кассир», «клиент» or «инкассатор»"

VERIFY_TEMPLATE = """This is a short clip from a CCTV camera at a bank cash desk.
Does the clip show ANY of these possible violations? The first one is the most likely, but answer YES if any one of them is clearly visible.
{rules}

How to answer:
- Use only what is visible in the clip. Do not guess amounts of money, documents, signatures, account postings or anything else that cannot be seen.
- Do not name or try to identify any person, and do not describe faces or appearance. Refer to people only as {roles}.

Reply in exactly this format, three lines:
ANSWER: YES or ANSWER: NO
TYPE: <the id of the violation you see, or none>
OBSERVATION: <1-3 sentences in Russian about the visible actions only>"""

JUDGE_SYSTEM = f"""You check one flagged moment from a CCTV camera at a bank cash desk and decide which rule it may violate. The result is only a hint for a human reviewer.
- Use only facts stated in the observation or visible in the frames.
- Never state anything listed under a rule's "never_claim".
- Do not name or try to identify any person, and do not describe faces or appearance. Refer to people only as {ROLES}.
- rule_id must be one of the candidate ids. Choose "other" when no specific candidate fits.
- justification: 1-3 sentences in Russian, worded as a possibility (for example "Действия могут указывать на нарушение ..."), citing the visible facts.
- confidence: your probability from 0 to 1 that rule_id is right."""


def _timeout(seconds: float) -> httpx.Timeout:
    return httpx.Timeout(seconds, connect=min(CONNECT_TIMEOUT, seconds))


def _post(url: str, body: dict, timeout: float):
    # trust_env=False: these are local services, and a proxy from the environment must not reroute them
    return httpx.post(url, json=body, timeout=_timeout(timeout), trust_env=False)


def _prob(x) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    x = float(x)
    return x if math.isfinite(x) and 0.0 <= x <= 1.0 else None


def _token(x) -> str:
    """Lower-case id with surrounding quotes, backticks and punctuation removed."""
    return re.sub(r"^[^\w]+|[^\w]+$", "", str(x or "").strip()).lower()


# ---- InternVideo3 check ------------------------------------------------------------------------

def verify_prompt(rules: list[dict]) -> str:
    lines = "\n".join(f"- {r['id']}: {' '.join(str(r.get('verify_prompt') or '').split())}" for r in rules)
    return VERIFY_TEMPLATE.format(rules=lines, roles=ROLES)


def verify(window_path, rules: list[dict], timeout: float) -> tuple[float | None, str, str]:
    """POST the window to the InternVideo3 server: (P(YES), TYPE id among the rules or "none" or "",
    OBSERVATION). (None, "", "") on any failure."""
    try:
        if not timeout or timeout <= 0:
            return None, "", ""
        body = {"video_path": os.path.abspath(str(window_path)), "prompt": verify_prompt(rules),
                "fps": VERIFY_FPS}
        r = _post(os.environ.get("IV3_URL", IV3_URL), body, timeout)
        if r.status_code != 200:
            log.warning("verify: HTTP %s", r.status_code)
            return None, "", ""
        d = r.json()
        p = _prob(d.get("p_yes")) if isinstance(d, dict) else None
        if p is None:
            log.warning("verify: reply without a valid p_yes")
            return None, "", ""
        tid = _token(d.get("type_id"))
        tid = tid if tid == "none" or tid in {str(x["id"]).lower() for x in rules} else ""
        obs = " ".join(str(d.get("observation") or "").split())[:OBSERVATION_MAX]
        return p, tid, obs
    except Exception as e:
        log.warning("verify failed: %s", type(e).__name__)
        return None, "", ""


# ---- judge -------------------------------------------------------------------------------------

def judge_enabled() -> bool:
    return bool(os.environ.get("JUDGE_URL", JUDGE_URL).strip())


def _judge_endpoint() -> str:
    base = os.environ.get("JUDGE_URL", JUDGE_URL).strip().rstrip("/")
    return base if base.endswith("/chat/completions") else base + "/v1/chat/completions"


def judge_request(observation: str, evidence: list[Frame], candidates: list[dict]) -> dict:
    ids = [str(c["id"]) for c in candidates]
    text = (f"Observation from a video model (may be empty): {observation.strip() or '(none)'}\n\n"
            f"Candidate rules (JSON):\n{json.dumps(candidates, ensure_ascii=False, indent=1)}\n\n"
            f"{len(evidence)} frame(s) from the moment follow.")
    content = [{"type": "text", "text": text}] + [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(f.jpeg).decode()}}
        for f in evidence]
    schema = {"type": "object",
              "properties": {"rule_id": {"type": "string", "enum": ids},
                             "justification": {"type": "string"},
                             "confidence": {"type": "number", "minimum": 0, "maximum": 1}},
              "required": ["rule_id", "justification", "confidence"],
              "additionalProperties": False}
    return {"model": os.environ.get("JUDGE_MODEL", JUDGE_MODEL),
            "messages": [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": content}],
            "temperature": 0, "max_tokens": 512,
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "rule_choice", "schema": schema, "strict": True}},
            "chat_template_kwargs": {"enable_thinking": False}}


def _json_object(text: str):
    text = text.strip()
    m = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.S)
    return json.loads(m.group(1) if m else text)


def judge(observation: str, evidence: list[Frame], candidates: list[dict],
          timeout: float) -> tuple[str | None, str, float | None]:
    """Ask the judge which candidate rule fits: (rule_id as answered, Russian justification, confidence).
    The caller checks rule_id against the candidates. (None, "", None) when JUDGE_URL is empty or on failure."""
    try:
        if not judge_enabled() or not candidates or not timeout or timeout <= 0:
            return None, "", None
        r = _post(_judge_endpoint(), judge_request(observation or "", evidence, candidates), timeout)
        if r.status_code != 200:
            log.warning("judge: HTTP %s", r.status_code)
            return None, "", None
        d = _json_object(r.json()["choices"][0]["message"]["content"])
        rid = d.get("rule_id") if isinstance(d, dict) else None
        if not isinstance(rid, str):
            log.warning("judge: answer without a rule_id")
            return None, "", None
        just = d.get("justification")
        just = " ".join(just.split()) if isinstance(just, str) else ""
        return rid.strip(), just, _prob(d.get("confidence"))
    except Exception as e:
        log.warning("judge failed: %s", type(e).__name__)
        return None, "", None
