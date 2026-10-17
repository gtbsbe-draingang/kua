"""HTTP clients for the InternVideo3 check server and the optional rule judge. Neither ever raises:
any error, timeout, non-200 reply or malformed answer gives None / empty values. judge_fields, the
RAG server's reader of the judge request, is the one function here that raises (ValueError)."""
from __future__ import annotations

import base64
import json
import logging
import math
import os
import re
import threading

import httpx

from pipeline import replicas
from pipeline.prompts_zh import MULTI_TEMPLATE_ZH, VERIFY_PROMPTS_ZH
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
GUIDANCE = f"""How to answer:
- Use only what is visible in the clip. Do not guess amounts of money, documents, signatures, account postings or anything else that cannot be seen.
- Do not name or try to identify any person, and do not describe faces or appearance. Refer to people only as {ROLES}."""
OBSERVATION = "1-3 sentences in Russian about the visible actions only"

VERIFY_TEMPLATE = """This is a short clip from a CCTV camera at a bank cash desk.
Does the clip show ANY of these possible violations? The first one is the most likely, but answer YES if any one of them is clearly visible.
{rules}

{guidance}

Reply in exactly this format, three lines:
ANSWER: YES or ANSWER: NO
TYPE: <the id of the violation you see, or none>
OBSERVATION: <{observation}>"""

MULTI_TEMPLATE = """This is a short clip from a CCTV camera at a bank cash desk.
Answer each of these {n} questions about the clip with YES or NO. Answer YES only if it is clearly visible.
{questions}

{guidance}

Reply in exactly this format, {lines} lines: one line per question in the same order, then the observation:
{answers}
OBSERVATION: <{observation}, including who is present (by role only) and where the cash is>"""

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


_pool: replicas.Pool | None = None
_pool_lock = threading.Lock()


def iv3_urls() -> list[str]:
    """The InternVideo3 /verify replicas: IV3_URLS (comma-separated), else [IV3_URL], else the default."""
    return replicas.parse_urls(os.environ.get("IV3_URLS"), os.environ.get("IV3_URL", IV3_URL))


def _reset_pool() -> None:
    global _pool
    with _pool_lock:
        _pool = None


def _iv3_pool() -> replicas.Pool:
    """The process's pool, created on first use. It is rebuilt when the environment names other URLs."""
    global _pool
    urls = iv3_urls()
    with _pool_lock:
        pool = _pool
        if pool is None or pool.urls != urls:
            pool = _pool = replicas.Pool(urls)
        return pool


def _post_iv3(body: dict, timeout: float):
    return replicas.call_with_failover(_iv3_pool(), lambda url, left: _post(url, body, left), timeout)


def _prob(x) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    x = float(x)
    return x if math.isfinite(x) and 0.0 <= x <= 1.0 else None


def _token(x) -> str:
    """Lower-case id with surrounding quotes, backticks and punctuation removed."""
    return re.sub(r"^[^\w]+|[^\w]+$", "", str(x or "").strip()).lower()


# ---- InternVideo3 check ------------------------------------------------------------------------

def _question(rule: dict) -> str:
    return " ".join(str(rule.get("verify_prompt") or "").split())


def verify_prompt(rules: list[dict]) -> str:
    lines = "\n".join(f"- {r['id']}: {_question(r)}" for r in rules)
    return VERIFY_TEMPLATE.format(rules=lines, guidance=GUIDANCE, observation=OBSERVATION)


def multi_prompt(rules: list[dict], lang: str = "en") -> str:
    """Every rule's verify_prompt as question k = 1, 2, ..., answered on a line "<k>. YES" or "<k>. NO".
    lang "zh": the Chinese template and each rule's Chinese question (prompts_zh; the English verify_prompt for
    an id it lacks), with the same answer lines and a Russian observation."""
    n = len(rules)
    if lang == "zh":
        template = MULTI_TEMPLATE_ZH
        questions = [" ".join(VERIFY_PROMPTS_ZH[str(r["id"])].split()) if str(r.get("id")) in VERIFY_PROMPTS_ZH
                     else _question(r) for r in rules]
    elif lang == "en":
        template, questions = MULTI_TEMPLATE, [_question(r) for r in rules]
    else:
        raise ValueError(f"unknown prompt language {lang!r}")
    return template.format(
        n=n, lines=n + 1, guidance=GUIDANCE, observation=OBSERVATION,
        questions="\n".join(f"{k}. {q}" for k, q in enumerate(questions, 1)),
        answers="\n".join(f"{k}. YES or {k}. NO" for k in range(1, n + 1)))


def verify(window_path, rules: list[dict], timeout: float) -> tuple[float | None, str, str]:
    """POST the window to the InternVideo3 server: (P(YES), TYPE id among the rules or "none" or "",
    OBSERVATION). (None, "", "") on any failure."""
    try:
        if not timeout or timeout <= 0:
            return None, "", ""
        body = {"video_path": os.path.abspath(str(window_path)), "prompt": verify_prompt(rules),
                "fps": VERIFY_FPS}
        r = _post_iv3(body, timeout)
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


def verify_multi(window_path, rules: list[dict], timeout: float,
                 lang: str = "en") -> tuple[list[float | None], str]:
    """One POST asks every rule (multi_prompt in lang): (P(YES) per rule, aligned with rules; OBSERVATION).
    An entry that is not a probability is None for its rule. ([None] * len(rules), "") on any other
    failure, including a p_yes_list of the wrong length."""
    try:
        if not rules or not timeout or timeout <= 0:
            return [None] * len(rules), ""
        body = {"video_path": os.path.abspath(str(window_path)), "prompt": multi_prompt(rules, lang),
                "fps": VERIFY_FPS, "n_answers": len(rules)}
        r = _post_iv3(body, timeout)
        if r.status_code != 200:
            log.warning("verify_multi: HTTP %s", r.status_code)
            return [None] * len(rules), ""
        d = r.json()
        ps = d.get("p_yes_list") if isinstance(d, dict) else None
        if not isinstance(ps, list) or len(ps) != len(rules):
            log.warning("verify_multi: reply without a p_yes_list of %d values", len(rules))
            return [None] * len(rules), ""
        obs = " ".join(str(d.get("observation") or "").split())[:OBSERVATION_MAX]
        return [_prob(p) for p in ps], obs
    except Exception as e:
        log.warning("verify_multi failed: %s", type(e).__name__)
        return [None] * len(rules), ""


# ---- judge -------------------------------------------------------------------------------------

def judge_enabled() -> bool:
    return bool(os.environ.get("JUDGE_URL", JUDGE_URL).strip())


def _judge_endpoint() -> str:
    base = os.environ.get("JUDGE_URL", JUDGE_URL).strip().rstrip("/")
    return base if base.endswith("/chat/completions") else base + "/v1/chat/completions"


# The judge request's text: OBSERVATION_LABEL + observation (NO_OBSERVATION when empty) + CANDIDATES_LABEL + the
# candidates as indented JSON + "\n\n<n> frame(s) from the moment follow." judge_fields reads it back.
OBSERVATION_LABEL = "Observation from a video model (may be empty): "
NO_OBSERVATION = "(none)"
CANDIDATES_LABEL = "\n\nCandidate rules (JSON):\n"
FRAMES_RE = re.compile(r"\n\n\d+ frame\(s\) from the moment follow\.")


def judge_request(observation: str, evidence: list[Frame], candidates: list[dict]) -> dict:
    ids = [str(c["id"]) for c in candidates]
    text = (f"{OBSERVATION_LABEL}{observation.strip() or NO_OBSERVATION}"
            f"{CANDIDATES_LABEL}{json.dumps(candidates, ensure_ascii=False, indent=1)}\n\n"
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


def judge_fields(body: dict) -> tuple[str, list[dict]]:
    """(observation, candidates) back from a judge_request body; the frames are not read. The observation is the
    one judge_request wrote: stripped, and "" for its "(none)". ValueError when the body is not laid out that way.

    The observation may hold anything, the candidate label included: the label that counts is the last one, because
    the indented JSON after it never holds a blank line (json.dumps escapes newlines inside strings)."""
    messages = body.get("messages") if isinstance(body, dict) else None
    if not isinstance(messages, list):
        raise ValueError("no messages")
    user = next((m for m in messages if isinstance(m, dict) and m.get("role") == "user"), None)
    if user is None:
        raise ValueError("no user message")
    content = user.get("content")
    if isinstance(content, str):
        text = content
    else:
        parts = content if isinstance(content, list) else []
        text = next((p.get("text") for p in parts if isinstance(p, dict) and p.get("type") == "text"), None)
    if not isinstance(text, str) or not text.startswith(OBSERVATION_LABEL):
        raise ValueError("the user text does not start with the observation")
    at = text.rfind(CANDIDATES_LABEL)
    if at < len(OBSERVATION_LABEL):
        raise ValueError("no candidate rules in the user text")
    candidates, end = json.JSONDecoder().raw_decode(text, at + len(CANDIDATES_LABEL))
    if not FRAMES_RE.fullmatch(text, end):
        raise ValueError("unexpected text after the candidate rules")
    if not isinstance(candidates, list) or not all(isinstance(c, dict) for c in candidates):
        raise ValueError("the candidate rules are not a list of objects")
    observation = text[len(OBSERVATION_LABEL):at]
    return ("" if observation == NO_OBSERVATION else observation), candidates


def _json_object(text: str):
    text = text.strip()
    m = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.S)
    return json.loads(m.group(1) if m else text)


CITATION_FIELDS = ("document", "clause", "quote", "url")


def _citation(x) -> dict | None:
    if not isinstance(x, dict) or not all(isinstance(x.get(k), str) for k in CITATION_FIELDS):
        return None
    return {k: x[k] for k in CITATION_FIELDS}


def judge_answer(observation: str, evidence: list[Frame], candidates: list[dict], timeout: float) -> dict | None:
    """Ask the judge which candidate rule fits, with the request judge sends. Returns rule_id as answered (stripped),
    justification (whitespace collapsed, "" when missing), confidence, and what the RAG server adds: clause_id,
    violation and citation ({document, clause, quote, url}). Those four are None when absent or invalid. The caller
    checks rule_id against the candidates. None when JUDGE_URL is empty or on failure."""
    try:
        if not judge_enabled() or not candidates or not timeout or timeout <= 0:
            return None
        r = _post(_judge_endpoint(), judge_request(observation or "", evidence, candidates), timeout)
        if r.status_code != 200:
            log.warning("judge: HTTP %s", r.status_code)
            return None
        d = _json_object(r.json()["choices"][0]["message"]["content"])
        rid = d.get("rule_id") if isinstance(d, dict) else None
        if not isinstance(rid, str):
            log.warning("judge: answer without a rule_id")
            return None
        just = d.get("justification")
        clause = d.get("clause_id")
        clause = clause.strip() if isinstance(clause, str) else ""
        return {"rule_id": rid.strip(), "justification": " ".join(just.split()) if isinstance(just, str) else "",
                "confidence": _prob(d.get("confidence")),
                "clause_id": clause if clause and clause.lower() != "none" else None,
                "violation": _prob(d.get("violation")), "citation": _citation(d.get("citation"))}
    except Exception as e:
        log.warning("judge failed: %s", type(e).__name__)
        return None


def judge(observation: str, evidence: list[Frame], candidates: list[dict],
          timeout: float) -> tuple[str | None, str, float | None]:
    """Ask the judge which candidate rule fits: (rule_id as answered, Russian justification, confidence).
    The caller checks rule_id against the candidates. (None, "", None) when JUDGE_URL is empty or on failure."""
    a = judge_answer(observation, evidence, candidates, timeout)
    return (None, "", None) if a is None else (a["rule_id"], a["justification"], a["confidence"])
