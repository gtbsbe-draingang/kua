"""The RAG server's LLM backends: InternVideo3's /generate ("iv3", the default) and an OpenAI-style chat completions
server ("openai", the Qwen3.5-9B judge on SGLang), picked by RAG_LLM_KIND. HTTP is mocked at httpx.post, plus one test
against a real local socket for what goes over the wire."""
import json
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from fastapi.testclient import TestClient

from pipeline import clients
from pipeline.types import Frame
from rag.corpus import Clause
from rag.retrieve import Hit
from servers import rag_server

CHAT_URL = "http://127.0.0.1:30005/v1/chat/completions"
JPEG = b"\xff\xd8\xff\xe0fake"
CLAUSE = Clause("V2:8", "V2", "Правила V2", "Документ V2", "Постановление V2", "8", "Глава 1",
                ("Посторонние лица в помещение кассы не допускаются.",), "https://adilet.zan.kz/rus/docs/V2#z8")
CANDS = [{"id": "other", "name": "Иное", "visual_criteria": [], "never_claim": [], "citations": []}]
ANSWER = {"violation": 0.85, "rule_id": "other", "clause_id": "V2:8", "justification": "Может указывать.",
          "confidence": 0.7}


def chat_reply(content, status=200):
    return httpx.Response(status, json={"choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                                     "finish_reason": "stop"}]})


@pytest.fixture
def posts(monkeypatch):
    """Replace httpx.post: .sent records (url, json, timeout, trust_env), .replies is consumed one per call."""
    p = types.SimpleNamespace(sent=[], replies=[])

    def post(url, json, timeout, trust_env):
        p.sent.append((url, json, timeout, trust_env))
        reply = p.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(rag_server.httpx, "post", post)
    return p


# ---- openai_llm: the request ---------------------------------------------------------------

def test_openai_llm_posts_a_chat_completion(posts):
    posts.replies.append(chat_reply("ответ"))
    llm = rag_server.openai_llm(CHAT_URL, "qwen3.5-9b")
    assert llm("системное", "вопрос", 12.0) == "ответ"
    url, body, timeout, trust_env = posts.sent[0]
    assert url == CHAT_URL
    assert body == {"model": "qwen3.5-9b",
                    "messages": [{"role": "system", "content": "системное"}, {"role": "user", "content": "вопрос"}],
                    "max_tokens": rag_server.MAX_NEW_TOKENS, "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False}}
    assert body["max_tokens"] == 384
    assert (timeout.read, timeout.connect, trust_env) == (12.0, 2.0, False)  # as http_llm
    posts.replies.append(chat_reply("ok"))
    llm("s", "p", 0.5)
    assert posts.sent[1][2].connect == 0.5  # never longer than the call's whole timeout


# ---- openai_llm: the answer ----------------------------------------------------------------

@pytest.mark.parametrize("content, text", [
    ("  ответ \n", "ответ"),
    ("<think>\nрассуждение\n</think>\n\nответ", "ответ"),
    ("<think>\n\n</think>\n\n{\"a\": 1}", "{\"a\": 1}"),  # what the template's empty think block leaves
    ("<think>раз</think>один<think>два\nстроки</think> два", "один два"),  # every block
    ("<think>a\n</think>\n<think>b</think>", ""),  # nothing but thinking
    ("рассуждение без начала</think>\n\nответ", "ответ"),  # the opening tag lives in the prompt
    ("ответ<think>оборвано max_tokens", "ответ"),  # never closed
    ("", ""),
])
def test_openai_llm_removes_think_blocks(posts, content, text):
    posts.replies.append(chat_reply(content))
    assert rag_server.openai_llm(CHAT_URL, "m")("s", "p", 5.0) == text


@pytest.mark.parametrize("reply", [
    httpx.Response(500, json={"error": "x"}),
    httpx.Response(503, content=b"loading"),
    httpx.Response(200, content=b"nope"),
    httpx.Response(200, json=["choices"]),
    httpx.Response(200, json={"text": "iv3 shape"}),
    httpx.Response(200, json={"choices": []}),
    httpx.Response(200, json={"choices": [None]}),
    httpx.Response(200, json={"choices": [{}]}),
    httpx.Response(200, json={"choices": [{"message": None}]}),
    httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": None}}]}),
    httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": ["x"]}}]}),
], ids=repr)
def test_openai_llm_raises_on_a_bad_reply(posts, reply):
    posts.replies.append(reply)
    with pytest.raises((RuntimeError, ValueError)):  # the server maps every failure to one 502, as for http_llm
        rag_server.openai_llm(CHAT_URL, "m")("s", "p", 5.0)


def test_openai_llm_lets_transport_errors_through(posts):
    for exc in (httpx.ConnectError("down"), httpx.ReadTimeout("slow")):
        posts.replies.append(exc)
        with pytest.raises(type(exc)):
            rag_server.openai_llm(CHAT_URL, "m")("s", "p", 5.0)


# ---- pooling and kind selection -------------------------------------------------------------

def test_pooled_openai_llm_fails_over_on_connect_error(posts):
    urls = ["http://127.0.0.1:1/v1/chat/completions", "http://127.0.0.1:2/v1/chat/completions"]
    posts.replies += [httpx.ConnectError("down"), chat_reply("<think>x</think>ответ")]
    llm = rag_server.pooled_llm(urls, lambda url: rag_server.openai_llm(url, "m"))
    assert llm("s", "p", 10) == "ответ"
    assert [s[0] for s in posts.sent] == urls
    assert all(s[1]["model"] == "m" for s in posts.sent)


def test_pooled_llm_defaults_to_the_generate_backend(posts):
    posts.replies.append(httpx.Response(200, json={"text": "ответ"}))
    assert rag_server.pooled_llm(["http://127.0.0.1:1/generate"])("sys", "user", 10) == "ответ"
    assert posts.sent[0][1] == {"system": "sys", "prompt": "user", "max_new_tokens": 384}


def test_make_llm_selects_by_kind(posts):
    posts.replies += [httpx.Response(200, json={"text": "iv3"}), chat_reply("qwen")]
    assert rag_server.make_llm("iv3", ["http://127.0.0.1:30000/generate"], "ignored")("s", "p", 5) == "iv3"
    assert posts.sent[0][1].keys() == {"system", "prompt", "max_new_tokens"}
    assert rag_server.make_llm("openai", [CHAT_URL], "qwen3.5-9b")("s", "p", 5) == "qwen"
    assert posts.sent[1][0] == CHAT_URL and posts.sent[1][1]["model"] == "qwen3.5-9b"
    with pytest.raises(ValueError, match="RAG_LLM_KIND.*vllm"):
        rag_server.make_llm("vllm", [CHAT_URL], "m")


# ---- through the server ---------------------------------------------------------------------

class OneClauseIndex:
    clauses = [CLAUSE]
    by_id = {CLAUSE.id: CLAUSE}
    hybrid = True

    def search(self, text, k=8):
        return [Hit("V2:8", 0.9, 0)]

    def search_many(self, texts, k=8):
        return [[Hit("V2:8", 0.9, 0)] for _ in texts]


def judge_body():
    return json.loads(json.dumps(clients.judge_request("Кассир уходит.", [Frame(1.0, JPEG)], CANDS)))


def test_rag_server_answers_through_the_openai_backend(posts):
    posts.replies.append(chat_reply("<think>\n\n</think>\n\n```json\n" + json.dumps(ANSWER, ensure_ascii=False)
                                    + "\n```"))
    app = rag_server.create_app(OneClauseIndex(), rag_server.openai_llm(CHAT_URL, "qwen3.5-9b"), llm_timeout=7.5)
    r = TestClient(app).post("/v1/chat/completions", json=judge_body())
    assert r.status_code == 200, r.text
    a = json.loads(r.json()["choices"][0]["message"]["content"])
    assert (a["rule_id"], a["clause_id"], a["violation"]) == ("other", "V2:8", 0.85)
    assert a["citation"]["clause"] == "п. 8"
    _, sent, timeout, _ = posts.sent[0]
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]
    assert "other" in sent["messages"][1]["content"] and timeout.read == 7.5


def test_response_model_label_is_the_apps_label(posts):
    posts.replies += [chat_reply(json.dumps(ANSWER)), chat_reply(json.dumps(ANSWER))]
    llm = rag_server.openai_llm(CHAT_URL, "m")
    default = TestClient(rag_server.create_app(OneClauseIndex(), llm))
    assert default.post("/v1/chat/completions", json=judge_body()).json()["model"] == "rag+iv3"
    labelled = TestClient(rag_server.create_app(OneClauseIndex(), llm, model_label="rag+m"))
    assert labelled.post("/v1/chat/completions", json=judge_body()).json()["model"] == "rag+m"


@pytest.mark.parametrize("kind, model, name, label", [
    ("iv3", "qwen3.5-9b", "iv3", "rag+iv3"),  # the model name belongs to the openai backend only
    ("openai", "qwen3.5-9b", "openai qwen3.5-9b", "rag+qwen3.5-9b"),
    ("openai", "other-model", "openai other-model", "rag+other-model"),
])
def test_llm_name_and_label_say_which_backend_answers(kind, model, name, label):
    assert rag_server.llm_name(kind, model) == name
    assert rag_server.llm_label(kind, model) == label


def test_rag_server_openai_failures_are_a_502(posts):
    posts.replies += [httpx.Response(500, json={}), httpx.ReadTimeout("slow"), chat_reply("<think>только мысли"),
                      chat_reply("не JSON")]
    client = TestClient(rag_server.create_app(OneClauseIndex(), rag_server.openai_llm(CHAT_URL, "m")))
    got = [client.post("/v1/chat/completions", json=judge_body()) for _ in range(4)]
    assert [(r.status_code, r.json()) for r in got] == [
        (502, {"error": "llm"}), (502, {"error": "llm"}), (502, {"error": "bad answer"}), (502, {"error": "bad answer"})]


def test_openai_llm_over_a_real_socket():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            seen.append((self.path, self.headers["Content-Type"], json.loads(raw)))
            out = json.dumps({"choices": [{"message": {"role": "assistant",
                                                       "content": "<think>\n\n</think>\n\nпривет"}}]}).encode()
            status = 500 if len(seen) > 1 else 200
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        llm = rag_server.openai_llm(f"http://127.0.0.1:{server.server_port}/v1/chat/completions", "qwen3.5-9b")
        assert llm("система", "вопрос", 5.0) == "привет"
        with pytest.raises(RuntimeError, match="HTTP 500"):
            llm("система", "вопрос", 5.0)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
    path, content_type, body = seen[0]
    assert path == "/v1/chat/completions" and content_type == "application/json"
    assert body["messages"][0] == {"role": "system", "content": "система"}  # UTF-8 on the wire, read back intact
    assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["temperature"] == 0


# ---- startup: RAG_LLM_KIND, RAG_LLM_MODEL, RAG_LLM_URL(S) -----------------------------------

@pytest.fixture
def started(monkeypatch, posts):
    """Run the real app's lifespan with the index class and torch replaced; yields a TestClient."""
    torch = types.ModuleType("torch")
    torch.set_num_threads = lambda n: None
    monkeypatch.setitem(sys.modules, "torch", torch)

    class Index(OneClauseIndex):
        def __init__(self, clauses):
            pass

    monkeypatch.setattr(rag_server, "Index", Index)
    monkeypatch.setattr(rag_server, "load_clauses", lambda: [CLAUSE])
    for name in ("RAG_LLM_KIND", "RAG_LLM_MODEL", "RAG_LLM_URL", "RAG_LLM_URLS", "RAG_LLM_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def judge_once(client):
    return client.post("/v1/chat/completions", json=judge_body())


def test_startup_defaults_to_the_generate_backend(started, posts):
    posts.replies.append(httpx.Response(200, json={"text": json.dumps(ANSWER)}))
    with TestClient(rag_server.app) as client:
        r = judge_once(client)
        assert r.status_code == 200 and r.json()["model"] == "rag+iv3"
        assert client.get("/healthz").json() == {"ok": True, "clauses": 1, "passages": 1, "llm": "iv3",
                                                 "retrieval": "hybrid"}
    url, sent, _, _ = posts.sent[0]
    assert url == "http://127.0.0.1:30000/generate" and sent.keys() == {"system", "prompt", "max_new_tokens"}


def test_startup_openai_kind(started, posts):
    started.setenv("RAG_LLM_KIND", "openai")
    posts.replies.append(chat_reply(json.dumps(ANSWER)))
    with TestClient(rag_server.app) as client:
        r = judge_once(client)
        assert r.status_code == 200 and r.json()["model"] == "rag+qwen3.5-9b"  # labelled by the backend's model
        assert client.get("/healthz").json()["llm"] == "openai qwen3.5-9b"
    url, sent, _, _ = posts.sent[0]
    assert url == CHAT_URL  # the judge LLM's own address is the default for this kind
    assert sent["model"] == "qwen3.5-9b" and "messages" in sent


def test_startup_openai_kind_reads_model_and_urls(started, posts):
    started.setenv("RAG_LLM_KIND", "openai")
    started.setenv("RAG_LLM_MODEL", "other-model")
    started.setenv("RAG_LLM_URLS", "http://127.0.0.1:7/v1/chat/completions")
    posts.replies.append(chat_reply(json.dumps(ANSWER)))
    with TestClient(rag_server.app) as client:
        r = judge_once(client)
        assert r.status_code == 200 and r.json()["model"] == "rag+other-model"
        assert client.get("/healthz").json()["llm"] == "openai other-model"
    assert posts.sent[0][0] == "http://127.0.0.1:7/v1/chat/completions" and posts.sent[0][1]["model"] == "other-model"


def test_startup_rejects_an_unknown_kind(started):
    started.setenv("RAG_LLM_KIND", "vllm")
    with pytest.raises(ValueError, match="RAG_LLM_KIND"):
        with TestClient(rag_server.app):
            pass
