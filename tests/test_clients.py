"""InternVideo3 check and judge clients against a local HTTP server: request shape, parsing, never raising."""
import base64
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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
