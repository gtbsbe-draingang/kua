"""InternVideo3 check and judge clients against a local HTTP server: request shape, parsing, never raising."""
import base64
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from pipeline import clients
from pipeline.types import Frame

RULES = [
    {"id": "cash_unattended", "verify_prompt": "Does the cashier leave cash unattended?"},
    {"id": "personal_items_at_workplace", "verify_prompt": "Does the cashier use a phone?"},
    {"id": "other", "verify_prompt": "Is cash handled in an uncontrolled way?"},
]
CANDS = [
    {"id": "cash_unattended", "name": "Наличные без присмотра", "visual_criteria": ["cash lies exposed"],
     "never_claim": ["amount of cash"], "citations": [{"document": "Doc 231", "clause": "п. 20", "quote": "Q-20"}]},
    {"id": "other", "name": "Иное", "visual_criteria": [], "never_claim": ["posting"],
     "citations": [{"document": "Doc 231", "clause": "", "quote": "Q-doc"}]},
]
JPEG = b"\xff\xd8\xff\xe0fake"


@pytest.fixture
def server():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            srv.requests.append((self.path, json.loads(body)))
            time.sleep(srv.delay)
            status, payload = srv.reply
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except OSError:
                pass  # the client gave up first

        def log_message(self, *args):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.requests, srv.delay, srv.reply = [], 0.0, (200, {})
    srv.url = f"http://127.0.0.1:{srv.server_address[1]}"
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


def unused_url():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"http://127.0.0.1:{port}"


# ---- verify -------------------------------------------------------------------------------

def test_verify_prompt_lists_rules_in_order_and_sets_the_format():
    p = clients.verify_prompt(RULES)
    lines = [ln for ln in p.splitlines() if ln.startswith("- ") and ":" in ln and "_" in ln.split(":")[0]]
    assert [ln[2:].split(":")[0] for ln in lines][:2] == ["cash_unattended", "personal_items_at_workplace"]
    assert "Does the cashier leave cash unattended?" in p and "other:" in p
    for s in ("ANSWER: YES", "ANSWER: NO", "TYPE:", "OBSERVATION:", "Russian",
              "«кассир»", "«клиент»", "«инкассатор»", "ANY"):
        assert s in p, s
    assert "identify" in p and "amounts" in p


def test_verify_prompt_text_is_pinned():
    assert clients.verify_prompt(RULES) == (
        "This is a short clip from a CCTV camera at a bank cash desk.\n"
        "Does the clip show ANY of these possible violations? The first one is the most likely, but answer YES if "
        "any one of them is clearly visible.\n"
        "- cash_unattended: Does the cashier leave cash unattended?\n"
        "- personal_items_at_workplace: Does the cashier use a phone?\n"
        "- other: Is cash handled in an uncontrolled way?\n"
        "\n"
        "How to answer:\n"
        "- Use only what is visible in the clip. Do not guess amounts of money, documents, signatures, account "
        "postings or anything else that cannot be seen.\n"
        "- Do not name or try to identify any person, and do not describe faces or appearance. Refer to people only "
        "as «кассир», «клиент» or «инкассатор».\n"
        "\n"
        "Reply in exactly this format, three lines:\n"
        "ANSWER: YES or ANSWER: NO\n"
        "TYPE: <the id of the violation you see, or none>\n"
        "OBSERVATION: <1-3 sentences in Russian about the visible actions only>")


def test_verify_ok(server, monkeypatch, tmp_path):
    monkeypatch.setenv("IV3_URL", server.url + "/verify")
    server.reply = (200, {"p_yes": 0.73, "type_id": " `Cash_Unattended`.", "observation": " Кассир уходит. ",
                          "raw": "ANSWER: YES"})
    w = tmp_path / "w.mp4"
    w.write_bytes(b"x")
    assert clients.verify(str(w), RULES, 10) == (0.73, "cash_unattended", "Кассир уходит.")
    path, body = server.requests[0]
    assert path == "/verify" and body["fps"] == 4 and body["video_path"] == str(w.resolve())
    assert body["prompt"] == clients.verify_prompt(RULES)
    assert set(body) == {"video_path", "prompt", "fps"}  # no n_answers: the server's single mode


@pytest.mark.parametrize("type_id,expected", [("none", "none"), ("NONE.", "none"), ("made_up", ""), (None, "")])
def test_verify_type_must_be_offered(server, monkeypatch, type_id, expected):
    monkeypatch.setenv("IV3_URL", server.url + "/verify")
    server.reply = (200, {"p_yes": 0.2, "type_id": type_id, "observation": "x"})
    assert clients.verify("/w.mp4", RULES, 10) == (0.2, expected, "x")


@pytest.mark.parametrize("reply", [
    (500, {"error": "boom"}), (200, b"not json"), (200, {"p_yes": "high"}), (200, {"p_yes": 1.5}),
    (200, {"p_yes": None}), (200, [1, 2]), (200, {"p_yes": True}),
])
def test_verify_errors_give_none(server, monkeypatch, reply):
    monkeypatch.setenv("IV3_URL", server.url + "/verify")
    server.reply = reply
    assert clients.verify("/w.mp4", RULES, 10) == (None, "", "")


def test_verify_nan_gives_none(server, monkeypatch):
    monkeypatch.setenv("IV3_URL", server.url + "/verify")
    server.reply = (200, b'{"p_yes": NaN, "type_id": "none", "observation": ""}')
    assert clients.verify("/w.mp4", RULES, 10) == (None, "", "")


def test_verify_connection_refused_is_fast(monkeypatch):
    monkeypatch.setenv("IV3_URL", unused_url() + "/verify")
    t = time.monotonic()
    assert clients.verify("/w.mp4", RULES, 30) == (None, "", "")
    assert time.monotonic() - t < 3


def test_verify_times_out(server, monkeypatch):
    monkeypatch.setenv("IV3_URL", server.url + "/verify")
    server.delay = 2.0
    t = time.monotonic()
    assert clients.verify("/w.mp4", RULES, 0.3) == (None, "", "")
    assert time.monotonic() - t < 1.5


def test_verify_without_time_sends_nothing(server, monkeypatch):
    monkeypatch.setenv("IV3_URL", server.url + "/verify")
    assert clients.verify("/w.mp4", RULES, 0) == (None, "", "")
    assert server.requests == []


# ---- verify, one answer per rule ----------------------------------------------------------

def test_multi_prompt_numbers_every_rule():
    rules = RULES + [{"id": "split", "verify_prompt": "Is the cash\n   drawer open?\n"}]
    p = clients.multi_prompt(rules)
    lines = p.splitlines()
    questions = ["1. Does the cashier leave cash unattended?", "2. Does the cashier use a phone?",
                 "3. Is cash handled in an uncontrolled way?", "4. Is the cash drawer open?"]
    answers = ["1. YES or 1. NO", "2. YES or 2. NO", "3. YES or 3. NO", "4. YES or 4. NO"]
    at = [lines.index(s) for s in questions + answers]
    assert at == sorted(at) and at[4:] == list(range(at[4], at[4] + 4))
    assert lines[at[-1] + 1].startswith("OBSERVATION: <1-3 sentences in Russian about the visible actions only")
    assert lines[at[-1] + 1].endswith(">") and lines[at[-1] + 1] == lines[-1]
    for s in ("who is present", "where the cash is", "«кассир»", "«клиент»", "«инкассатор»", "identify", "amounts"):
        assert s in p, s
    assert "<|" not in p and "ANSWER:" not in p and "TYPE:" not in p and "cash_unattended" not in p


def test_verify_multi_aligns_and_handles_errors(server, monkeypatch, tmp_path):
    monkeypatch.setenv("IV3_URL", server.url + "/verify")
    w = tmp_path / "w.mp4"
    w.write_bytes(b"x")
    server.reply = (200, {"p_yes": 0.9, "p_yes_list": [0.9, None, 0.25], "observation": " Кассир  считает\nденьги. "})
    assert clients.verify_multi(str(w), RULES, 10) == ([0.9, None, 0.25], "Кассир считает деньги.")
    path, body = server.requests[0]
    assert path == "/verify"
    assert body == {"video_path": str(w.resolve()), "prompt": clients.multi_prompt(RULES), "fps": 4, "n_answers": 3}
    server.reply = (200, {"p_yes_list": [0.9, 1.5, "high"], "observation": "o"})  # a bad value is None for its rule
    assert clients.verify_multi(str(w), RULES, 10) == ([0.9, None, None], "o")
    for reply in [(200, {"p_yes": 0.9, "p_yes_list": [0.9, 0.1], "observation": "o"}),  # wrong length
                  (200, {"p_yes": 0.9, "p_yes_list": [0.9, 0.1, 0.2, 0.3]}), (500, {"error": "no answers"}),
                  (200, b"not json"), (200, [0.9, 0.1, 0.2]), (200, {"p_yes": 0.9}),
                  (200, {"p_yes_list": "0.9 0.1 0.2"})]:
        server.reply = reply
        assert clients.verify_multi(str(w), RULES, 10) == ([None, None, None], ""), reply


def _multi_rules():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import verify_candidates as vc

    from pipeline import real
    return vc.multi_rules(real._config())


def test_multi_prompt_en_unchanged():
    import hashlib

    rules = _multi_rules()
    p = clients.multi_prompt(rules)
    assert len(rules) == 13 and clients.multi_prompt(rules, lang="en") == p
    # the English prompt of the 13 catalogue rules, pinned before the Chinese variant was added
    assert hashlib.md5(p.encode()).hexdigest() == "3ddc7ebae68d4497480826af06d64fcf" and len(p) == 2310


def test_multi_prompt_zh_numbers_every_rule_and_keeps_yes_no():
    from pipeline.prompts_zh import VERIFY_PROMPTS_ZH

    rules = _multi_rules()
    p = clients.multi_prompt(rules, lang="zh")
    lines = p.splitlines()
    assert lines[0] == "这是银行收银台监控摄像头拍摄的一段短视频。"
    assert lines[1] == "请用 YES 或 NO 回答以下 13 个关于该视频的问题。只有在视频中清楚可见时才回答 YES。"
    questions = [f"{k}. {VERIFY_PROMPTS_ZH[r['id']]}" for k, r in enumerate(rules, 1)]
    answers = [f"{k}. YES or {k}. NO" for k in range(1, 14)]
    at = [lines.index(q) for q in questions + answers]
    assert at == sorted(at) and at[:13] == list(range(2, 15)) and at[13:] == list(range(at[13], at[13] + 13))
    assert "共 14 行" in p and "答案只写英文 YES 或 NO" in p
    assert lines[-1] == ("OBSERVATION: <用俄语（русский язык）写 1-3 句话，只描述可见的动作，"
                         "包括在场人员（只写角色）以及现金所在位置>")
    assert lines[at[-1] + 1] == lines[-1]
    for s in ("«кассир»", "«клиент»", "«инкассатор»"):
        assert s in p, s
    assert set(VERIFY_PROMPTS_ZH) == {r["id"] for r in rules}
    # an id without a Chinese question keeps its English one; the answer lines are built the same way
    extra = clients.multi_prompt([{"id": "split", "verify_prompt": "Is the cash\n   drawer open?\n"}], lang="zh")
    assert "1. Is the cash drawer open?" in extra.splitlines() and "1. YES or 1. NO" in extra.splitlines()


def test_verify_multi_zh_sends_the_chinese_prompt(server, monkeypatch, tmp_path):
    monkeypatch.setenv("IV3_URL", server.url + "/verify")
    w = tmp_path / "w.mp4"
    w.write_bytes(b"x")
    server.reply = (200, {"p_yes": 0.9, "p_yes_list": [0.9, None, 0.25], "observation": "Кассир считает деньги."})
    assert clients.verify_multi(str(w), RULES, 10, lang="zh") == ([0.9, None, 0.25], "Кассир считает деньги.")
    assert server.requests[0][1]["prompt"] == clients.multi_prompt(RULES, lang="zh")
    assert server.requests[0][1]["n_answers"] == 3


def test_verify_multi_without_time_or_server(server, monkeypatch):
    monkeypatch.setenv("IV3_URL", server.url + "/verify")
    assert clients.verify_multi("/w.mp4", RULES, 0) == ([None, None, None], "")
    assert clients.verify_multi("/w.mp4", [], 10) == ([], "")
    assert server.requests == []
    monkeypatch.setenv("IV3_URL", unused_url() + "/verify")
    t = time.monotonic()
    assert clients.verify_multi("/w.mp4", RULES, 30) == ([None, None, None], "")
    assert time.monotonic() - t < 3


# ---- judge --------------------------------------------------------------------------------

def judge_reply(obj):
    content = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return 200, {"choices": [{"message": {"role": "assistant", "content": content}}]}


def test_judge_request_shape_and_answer(server, monkeypatch):
    monkeypatch.setenv("JUDGE_URL", server.url)
    server.reply = judge_reply({"rule_id": "cash_unattended", "justification": " Это может указывать. ",
                                "confidence": 0.8})
    ev = [Frame(1.0, JPEG), Frame(2.0, JPEG)]
    assert clients.judge("Кассир уходит.", ev, CANDS, 10) == ("cash_unattended", "Это может указывать.", 0.8)
    path, body = server.requests[0]
    assert path == "/v1/chat/completions"
    assert body["temperature"] == 0 and body["chat_template_kwargs"] == {"enable_thinking": False}
    rf = body["response_format"]
    assert rf["type"] == "json_schema"
    schema = rf["json_schema"]["schema"]
    assert schema["properties"]["rule_id"]["enum"] == ["cash_unattended", "other"]
    assert set(schema["required"]) == {"rule_id", "justification", "confidence"}
    content = [part for m in body["messages"] for part in (m["content"] if isinstance(m["content"], list) else
                                                           [{"type": "text", "text": m["content"]}])]
    images = [p["image_url"]["url"] for p in content if p["type"] == "image_url"]
    assert images == ["data:image/jpeg;base64," + base64.b64encode(JPEG).decode()] * 2
    text = "\n".join(p["text"] for p in content if p["type"] == "text")
    for s in ("Кассир уходит.", "Q-20", "Q-doc", "amount of cash", "cash lies exposed", "Russian",
              "«кассир»", "never"):
        assert s in text, s


def test_judge_full_endpoint_url_is_used_as_is(server, monkeypatch):
    monkeypatch.setenv("JUDGE_URL", server.url + "/v1/chat/completions")
    server.reply = judge_reply({"rule_id": "other", "justification": "j", "confidence": 0.5})
    assert clients.judge("", [Frame(0.0, JPEG)], CANDS, 10) == ("other", "j", 0.5)
    assert server.requests[0][0] == "/v1/chat/completions"


def test_judge_returns_what_the_model_said_and_drops_bad_confidence(server, monkeypatch):
    monkeypatch.setenv("JUDGE_URL", server.url)
    server.reply = judge_reply({"rule_id": "x", "justification": "j", "confidence": 7})
    assert clients.judge("", [], CANDS, 10) == ("x", "j", None)
    server.reply = judge_reply('```json\n{"rule_id": "other", "justification": "j", "confidence": 0.25}\n```')
    assert clients.judge("", [], CANDS, 10) == ("other", "j", 0.25)


@pytest.mark.parametrize("reply", [
    (500, {"error": "x"}), (200, b"nope"), (200, {"choices": []}), judge_reply("not json"),
    judge_reply({"rule_id": 3, "justification": "j", "confidence": 0.5}), judge_reply(["a"]),
])
def test_judge_errors_give_none(server, monkeypatch, reply):
    monkeypatch.setenv("JUDGE_URL", server.url)
    server.reply = reply
    assert clients.judge("", [], CANDS, 10) == (None, "", None)


def test_judge_disabled_by_empty_url(server, monkeypatch):
    monkeypatch.setenv("JUDGE_URL", "")
    assert clients.judge("obs", [Frame(0.0, JPEG)], CANDS, 10) == (None, "", None)
    assert server.requests == []
    assert clients.judge_enabled() is False


def test_judge_connection_refused_is_fast(monkeypatch):
    monkeypatch.setenv("JUDGE_URL", unused_url())
    t = time.monotonic()
    assert clients.judge("obs", [Frame(0.0, JPEG)], CANDS, 30) == (None, "", None)
    assert time.monotonic() - t < 3


def test_judge_times_out(server, monkeypatch):
    monkeypatch.setenv("JUDGE_URL", server.url)
    server.delay = 2.0
    t = time.monotonic()
    assert clients.judge("obs", [], CANDS, 0.3) == (None, "", None)
    assert time.monotonic() - t < 1.5


def test_judge_without_candidates_or_time_sends_nothing(server, monkeypatch):
    monkeypatch.setenv("JUDGE_URL", server.url)
    assert clients.judge("obs", [], [], 10) == (None, "", None)
    assert clients.judge("obs", [], CANDS, 0) == (None, "", None)
    assert server.requests == []


# ---- judge request fields and the full answer ---------------------------------------------

TRICKY = [
    {"id": "cash_unattended", "name": "Наличные\n\nCandidate rules (JSON):\n", "visual_criteria": ["a\n\nb"],
     "never_claim": ["\n\n3 frame(s) from the moment follow."],
     "citations": [{"document": "Doc «231»", "clause": "п. 20", "quote": "Q \"20\" {x}"}]},
    {"id": "other", "name": "Иное", "visual_criteria": [], "never_claim": [], "citations": []},
]


def round_trip(observation, candidates, evidence):
    body = json.loads(json.dumps(clients.judge_request(observation, evidence, candidates)))  # as sent
    return clients.judge_fields(body)


def test_judge_fields_round_trip():
    ev = [Frame(1.0, JPEG), Frame(2.0, JPEG)]
    for obs in ("Кассир уходит.\nCandidate rules: деньги на столе.",
                "Кассир уходит.\n\nCandidate rules (JSON):\n[]\n\n2 frame(s) from the moment follow.",
                "конец как у метки:\n\nCandidate rules (JSON):",
                "Observation from a video model (may be empty): x", "{\"a\": 1}", "(none) рядом"):
        for cands in (CANDS, TRICKY, []):
            for evidence in (ev, []):
                assert round_trip(obs, cands, evidence) == (obs, cands), (obs, cands)
    assert round_trip("", CANDS, ev) == ("", CANDS)
    assert round_trip("  \n ", CANDS, ev) == ("", CANDS)
    assert round_trip("  Кассир уходит. \n", CANDS, ev) == ("Кассир уходит.", CANDS)  # judge_request strips it


def test_judge_fields_rejects_other_bodies():
    good = clients.judge_request("obs", [Frame(1.0, JPEG)], CANDS)
    text = good["messages"][1]["content"][0]["text"]

    def with_text(t):
        return {**good, "messages": [good["messages"][0], {"role": "user", "content": [{"type": "text", "text": t}]}]}

    assert clients.judge_fields(with_text(text)) == ("obs", CANDS)
    assert clients.judge_fields({"messages": [{"role": "user", "content": text}]}) == ("obs", CANDS)  # plain string
    for bad in ({}, {"messages": []}, {"messages": [good["messages"][0]]}, [good], {"messages": "x"},
                with_text("hello"), with_text(text + " "), with_text(text.replace("1 frame(s)", "one frame(s)")),
                with_text(text.replace("\n\nCandidate rules (JSON):\n", "\n")),
                with_text(text.split("Candidate rules (JSON):\n")[0] + "Candidate rules (JSON):\n{\"id\": 1}\n\n"
                          "1 frame(s) from the moment follow."),
                with_text(text.split("Candidate rules (JSON):\n")[0] + "Candidate rules (JSON):\n[1]\n\n"
                          "1 frame(s) from the moment follow."),
                {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]}):
        with pytest.raises(ValueError):
            clients.judge_fields(bad)


CITATION = {"document": "Правила кассовых операций", "clause": "п. 20",
            "quote": "Все ранее принятые наличные деньги хранятся в специальных устройствах",
            "url": "https://adilet.zan.kz/rus/docs/V1900019680#z112"}


class MockPost:
    """Stands in for httpx.post: records each call and answers with the next reply."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, url, json, timeout, trust_env):
        self.calls.append((url, json, timeout, trust_env))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def chat(obj, status=200):
    content = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return httpx.Response(status, json={"choices": [{"message": {"role": "assistant", "content": content}}]})


def test_judge_answer_returns_clause_and_citation(monkeypatch):
    monkeypatch.setenv("JUDGE_URL", "http://127.0.0.1:30001")
    post = MockPost(chat({"violation": 0.9, "rule_id": " cash_unattended ", "clause_id": "V1900019680:20",
                          "justification": " Может  указывать\nна нарушение. ", "confidence": 0.8,
                          "citation": {**CITATION, "extra": "dropped"}}))
    monkeypatch.setattr(clients.httpx, "post", post)
    ev = [Frame(1.0, JPEG)]
    assert clients.judge_answer("Кассир уходит.", ev, CANDS, 10) == {
        "rule_id": "cash_unattended", "justification": "Может указывать на нарушение.", "confidence": 0.8,
        "clause_id": "V1900019680:20", "violation": 0.9, "citation": CITATION}
    url, body, timeout, trust_env = post.calls[0]
    assert url == "http://127.0.0.1:30001/v1/chat/completions" and trust_env is False
    assert body == clients.judge_request("Кассир уходит.", ev, CANDS)  # the same request judge sends
    assert timeout.read == 10


def test_judge_answer_absent_fields_and_failures(monkeypatch):
    monkeypatch.setenv("JUDGE_URL", "http://127.0.0.1:30001")
    # a judge that knows nothing of clauses: the new fields are None
    monkeypatch.setattr(clients.httpx, "post", MockPost(chat({"rule_id": "other", "justification": "j",
                                                              "confidence": 0.5})))
    assert clients.judge_answer("", [], CANDS, 10) == {"rule_id": "other", "justification": "j", "confidence": 0.5,
                                                      "clause_id": None, "violation": None, "citation": None}
    for extra in ({"clause_id": "none", "violation": 1.5, "citation": None},
                  {"clause_id": "", "violation": "high", "citation": {**CITATION, "url": None}},
                  {"clause_id": 20, "violation": True, "citation": {k: v for k, v in CITATION.items() if k != "url"}},
                  {"clause_id": None, "violation": None, "citation": "п. 20"}):
        monkeypatch.setattr(clients.httpx, "post", MockPost(chat({"rule_id": "other", "justification": "j",
                                                                  "confidence": 0.5, **extra})))
        a = clients.judge_answer("", [], CANDS, 10)
        assert (a["clause_id"], a["violation"], a["citation"]) == (None, None, None), extra
    for reply in (chat("not json"), chat({"rule_id": 3}), chat(["a"]), httpx.Response(502, json={"error": "llm"}),
                  httpx.Response(200, content=b"nope"), httpx.ConnectError("refused")):
        monkeypatch.setattr(clients.httpx, "post", MockPost(reply))
        assert clients.judge_answer("obs", [], CANDS, 10) is None, reply
    post = MockPost()
    monkeypatch.setattr(clients.httpx, "post", post)
    assert clients.judge_answer("obs", [], [], 10) is None and clients.judge_answer("obs", [], CANDS, 0) is None
    monkeypatch.setenv("JUDGE_URL", "")
    assert clients.judge_answer("obs", [], CANDS, 10) is None
    assert post.calls == []
