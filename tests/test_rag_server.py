"""The RAG server with a fake index and a fake LLM: a judge request in, a rule, a clause and a citation from the
corpus out. The real app's startup runs once, with the index class and torch replaced."""
import json
import re
import subprocess
import sys
import threading
import types
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from pipeline import clients
from pipeline.types import Frame
from rag import corpus, matcher
from rag.corpus import Clause
from rag.retrieve import Hit
from servers import rag_server

ROOT = Path(__file__).resolve().parent.parent
JPEG = b"\xff\xd8\xff\xe0fake"


def clause(cid, *parts):
    code, num = cid.split(":")
    return Clause(cid, code, f"Правила {code}", f"Документ {code}", f"Постановление {code}", num, "Глава 1",
                  tuple(parts), f"https://adilet.zan.kz/rus/docs/{code}#z{num}")


A20 = clause("V1:20", "На рабочем месте находятся наличные деньги клиента.",
             "Все ранее принятые наличные деньги хранятся в сейфе.")
A22 = clause("V1:22", "Кассовому работнику не допускается покидать рабочее место до окончания обслуживания клиента.")
B8 = clause("V2:8", "Вход в помещение кассы оборудуется дверью.", "Посторонние лица в помещение кассы не допускаются.")
CANDS = [
    {"id": "cash_unattended", "name": "Наличные без присмотра", "visual_criteria": ["cash lies exposed"],
     "never_claim": ["amount of cash"],
     "citations": [{"document": "Doc 231", "clause": "п. 20, ч. 2",
                    "quote": "Все ранее принятые наличные деньги\n   хранятся в сейфе."}]},
    {"id": "cashier_leaves_during_service", "name": "Уход кассира", "visual_criteria": [], "never_claim": [],
     "citations": [{"document": "Doc 231", "clause": "п. 22", "quote": "не допускается покидать рабочее место"},
                   {"document": "Doc 231", "clause": "п. 20", "quote": "находятся наличные деньги клиента"}]},
    {"id": "other", "name": "Иное", "visual_criteria": [], "never_claim": ["posting"],
     "citations": [{"document": "Doc 231", "clause": "", "quote": "этого текста нет в корпусе"}]},
]
OBS = "Кассир уходит, деньги лежат на столе."
HITS = (Hit("V2:8", 0.91, 1), Hit("V1:20", 0.8, 0))


class FakeIndex:
    hybrid = True  # rag.retrieve.Index.hybrid: the lexical channel is in use

    def __init__(self, clauses, hits=HITS):
        self.clauses = list(clauses)
        self.by_id = {c.id: c for c in self.clauses}
        self.hits = list(hits)
        self.searched, self.many = [], []

    def search(self, text, k=8):
        self.searched.append(text)
        return self.hits[:k]

    def search_many(self, texts, k=8):
        self.many.append((list(texts), k))
        return [self.hits[:k] if t.strip() else [] for t in texts]


class FakeLLM:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def __call__(self, system, prompt, timeout):
        self.calls.append((system, prompt, timeout))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def answer(**kw):
    a = {"violation": 0.85, "rule_id": "other", "clause_id": "V2:8", "justification": "Может указывать на нарушение.",
         "confidence": 0.7}
    return json.dumps({**a, **kw}, ensure_ascii=False)


def body(observation=OBS, candidates=CANDS):
    # sent as JSON, exactly as the pipeline's judge client sends it
    return json.loads(json.dumps(clients.judge_request(observation, [Frame(1.0, JPEG), Frame(2.0, JPEG)], candidates)))


@pytest.fixture
def make():
    def make(reply, hits=HITS, clauses=(A20, A22, B8)):
        index, llm = FakeIndex(clauses, hits), FakeLLM(reply)
        return TestClient(rag_server.create_app(index, llm, llm_timeout=7.5)), index, llm
    return make


def content(r) -> dict:
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["model"] == "rag+iv3" and d["choices"][0]["message"]["role"] == "assistant"
    return json.loads(d["choices"][0]["message"]["content"])


def cite(c, quote):
    return {"document": c.document, "clause": f"п. {c.num}", "quote": quote, "url": c.url}


# ---- /v1/chat/completions -----------------------------------------------------------------

def test_rag_server_chat_completions(make):
    client, index, llm = make("Ответ:\n```json\n" + answer() + "\n```")
    a = content(client.post("/v1/chat/completions", json=body()))
    assert a == {"violation": 0.85, "rule_id": "other", "clause_id": "V2:8",
                 "justification": "Может указывать на нарушение.", "confidence": 0.7,
                 "citation": cite(B8, "Посторонние лица в помещение кассы не допускаются.")}  # the hit's part
    assert a["citation"]["url"] == "https://adilet.zan.kz/rus/docs/V2#z8"
    assert index.searched == [OBS]
    # catalogue clauses first (each once), then the hits not listed yet; the frames are not used
    expected = matcher.build_prompt(OBS, CANDS, [(A20, None), (A22, None), (B8, 1)])
    assert llm.calls == [(expected[0], expected[1], 7.5)]


def test_rag_server_catalogue_clause_cites_the_rule_quote(make):
    client, _, _ = make(answer(rule_id="cashier_leaves_during_service", clause_id="V1:20"))
    a = content(client.post("/v1/chat/completions", json=body()))
    assert a["citation"] == cite(A20, "находятся наличные деньги клиента")  # the chosen rule's own quote
    client, _, _ = make(answer(rule_id="other", clause_id="V1:20"))
    a = content(client.post("/v1/chat/completions", json=body()))
    assert a["citation"] == cite(A20, "Все ранее принятые наличные деньги хранятся в сейфе.")  # first rule citing it


def test_rag_server_long_quote_is_cut(make):
    words = " ".join(f"слово{i:03d}" for i in range(100))
    long = clause("V3:5", words)
    client, _, _ = make(answer(clause_id="V3:5"), hits=(Hit("V3:5", 0.9, 0),), clauses=(A20, A22, B8, long))
    quote = content(client.post("/v1/chat/completions", json=body()))["citation"]["quote"]
    assert quote.endswith("…") and len(quote) <= 601
    assert words.startswith(quote[:-1]) and words[len(quote) - 1] == " "


def test_rag_server_empty_observation_skips_search(make):
    reply = answer(rule_id="cash_unattended", clause_id="none", violation=0.05, justification="Обычная работа.")
    for obs in ("", "   "):
        client, index, llm = make(reply)
        a = content(client.post("/v1/chat/completions", json=body(obs)))
        assert index.searched == []
        prompt = llm.calls[0][1]
        assert "(may be empty): (none)\n" in prompt
        assert "[V1:20]" in prompt and "[V1:22]" in prompt and "[V2:8]" not in prompt  # catalogue clauses only
        assert a["clause_id"] is None and a["citation"] is None and a["violation"] == 0.05


def test_rag_server_bad_answer_502(make):
    for reply in ("not json", "", answer(rule_id="made_up"), json.dumps(["other"]), '{"clause_id": "V2:8"}'):
        client, _, _ = make(reply)
        r = client.post("/v1/chat/completions", json=body())
        assert (r.status_code, r.json()) == (502, {"error": "bad answer"}), reply
    # an unknown clause is no failure: no clause and no citation
    client, _, _ = make(answer(clause_id="V9:1"))
    a = content(client.post("/v1/chat/completions", json=body()))
    assert a["clause_id"] is None and a["citation"] is None and a["rule_id"] == "other"
    for exc in (RuntimeError("HTTP 500"), httpx.ReadTimeout("slow")):
        client, _, _ = make(exc)
        r = client.post("/v1/chat/completions", json=body())
        assert (r.status_code, r.json()) == (502, {"error": "llm"})


def test_rag_server_rejects_requests_it_cannot_read(make):
    client, index, llm = make(answer())
    good = body()
    text = good["messages"][1]["content"][0]["text"]
    for bad in ({}, {"messages": []}, {"messages": [{"role": "user", "content": "hello"}]},
                {"messages": [{"role": "user", "content": [{"type": "text", "text": text + " extra"}]}]},
                [1, 2], body(candidates=[])):
        r = client.post("/v1/chat/completions", json=bad)
        assert r.status_code == 400 and r.json()["error"].startswith("bad request"), bad
    assert index.searched == [] and llm.calls == []


def test_rag_server_drops_retrieved_clauses_over_the_prompt_limit(make):
    big = [clause(f"V4:{i}", "слово " * 240) for i in range(30)]
    hits = tuple(Hit(c.id, 0.5, 0) for c in big)
    client, _, llm = make(answer(clause_id="V4:29"), hits=hits, clauses=(A20, A22, B8, *big))
    a = content(client.post("/v1/chat/completions", json=body()))
    prompt = llm.calls[0][1]
    assert len(prompt) <= matcher.PROMPT_MAX
    assert "[V1:20]" in prompt and "[V1:22]" in prompt and "[V4:0]" in prompt and "[V4:29]" not in prompt
    assert a["clause_id"] is None and a["citation"] is None  # the model named a clause it was not shown


# ---- /retrieve and /healthz ---------------------------------------------------------------

def test_rag_server_retrieve(make):
    client, index, _ = make("")
    r = client.post("/retrieve", json={"texts": ["деньги", " ", "касса"], "k": 1})
    assert r.status_code == 200
    one = [{"clause_id": "V2:8", "score": 0.91, "part": 1}]
    assert r.json() == {"hits": [one, [], one]}
    assert index.many == [(["деньги", " ", "касса"], 1)]
    r = client.post("/retrieve", json={"texts": ["деньги"]})
    assert r.json()["hits"][0] == one + [{"clause_id": "V1:20", "score": 0.8, "part": 0}]
    assert index.many[-1][1] == 8
    for bad in ({"texts": ["a"], "k": 0}, {"texts": ["a"], "k": 21}, {"texts": "a"}, {}):
        r = client.post("/retrieve", json=bad)
        assert r.status_code == 400 and r.json()["error"].startswith("bad request"), bad


def test_rag_server_healthz(make):
    client, _, _ = make("")
    assert client.get("/healthz").json() == {"ok": True, "clauses": 3, "passages": 5, "llm": "iv3",
                                             "retrieval": "hybrid"}


def test_rag_server_healthz_says_what_is_serving():
    index = FakeIndex((A20, A22, B8))
    index.hybrid = False  # dense only: RAG_HYBRID=0, or the lexical channel could not be built
    client = TestClient(rag_server.create_app(index, FakeLLM(""), backend="openai qwen3.5-9b",
                                              model_label="rag+qwen3.5-9b"))
    d = client.get("/healthz").json()
    assert (d["llm"], d["retrieval"]) == ("openai qwen3.5-9b", "dense")
    index.hybrid = True  # read from the index on every call (tools flip it to compare the two modes)
    assert client.get("/healthz").json()["retrieval"] == "hybrid"


# ---- the real app: import, startup, LLM client ---------------------------------------------

def test_header_documents_every_rag_environment_variable():
    """The module header is the one place that lists the RAG_* switches; one the code reads must be in it."""
    names = set()
    for module in ("servers/rag_server.py", "rag/retrieve.py"):
        names |= set(re.findall(r'environ\.get\("(RAG_[A-Z_]+)"', (ROOT / module).read_text()))
    assert {"RAG_LLM_KIND", "RAG_LLM_MODEL", "RAG_HYBRID"} <= names
    assert [n for n in sorted(names) if n not in rag_server.__doc__] == []


def test_rag_server_imports_without_torch():
    code = ("import sys; import servers.rag_server as s; "
            "bad = [m for m in ('torch', 'transformers') if m in sys.modules]; "
            "sys.exit(1 if bad or s.app.state.index is not None else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr


def test_http_llm_posts_the_generate_body(monkeypatch):
    sent, replies = [], []

    def post(url, json, timeout, trust_env):
        sent.append((url, json, timeout, trust_env))
        return replies.pop(0)

    monkeypatch.setattr(rag_server.httpx, "post", post)
    llm = rag_server.http_llm("http://127.0.0.1:30000/generate")
    replies.append(httpx.Response(200, json={"text": "ответ", "seconds": 1.2}))
    assert llm("sys", "user", 12.0) == "ответ"
    url, sent_body, timeout, trust_env = sent[0]
    assert url == "http://127.0.0.1:30000/generate"
    assert sent_body == {"system": "sys", "prompt": "user", "max_new_tokens": 384}
    assert (timeout.read, timeout.connect, trust_env) == (12.0, 2.0, False)
    for reply in (httpx.Response(500, json={"error": "x"}), httpx.Response(200, json={"seconds": 1.0}),
                  httpx.Response(200, content=b"nope"), httpx.Response(200, json=["text"])):
        replies.append(reply)
        with pytest.raises(Exception):
            llm("sys", "user", 12.0)


def test_rag_server_startup_builds_the_index_and_reads_env(monkeypatch):
    threads, built, sent = [], [], []
    torch = types.ModuleType("torch")
    torch.set_num_threads = threads.append
    monkeypatch.setitem(sys.modules, "torch", torch)

    class Index:
        hybrid = False

        def __init__(self, clauses):
            built.append(len(clauses))
            self.clauses, self.by_id = list(clauses), {c.id: c for c in clauses}

    def post(url, json, timeout, trust_env):
        sent.append((url, timeout.read))
        return httpx.Response(200, json={"text": "{}", "seconds": 0.1})

    monkeypatch.setattr(rag_server, "Index", Index)
    monkeypatch.setattr(rag_server.httpx, "post", post)
    monkeypatch.setenv("RAG_LLM_URL", "http://127.0.0.1:9/generate")
    monkeypatch.setenv("RAG_LLM_TIMEOUT", "3.5")
    n = len(corpus.load_clauses())
    with TestClient(rag_server.app) as client:
        assert threads == [4] and built == [n]
        assert client.get("/healthz").json() == {"ok": True, "clauses": n, "llm": "iv3", "retrieval": "dense",
                                                 "passages": len(corpus.passages(corpus.load_clauses()))}
        r = client.post("/v1/chat/completions", json=body("", CANDS[-1:]))
        assert (r.status_code, r.json()) == (502, {"error": "bad answer"})  # "{}" names no rule
    assert sent == [("http://127.0.0.1:9/generate", 3.5)]
    assert rag_server.app.state.index is None  # released at shutdown


def test_rag_server_spreads_llm_calls(monkeypatch):
    urls = ["http://127.0.0.1:1/generate", "http://127.0.0.1:2/generate"]
    seen, lock = [], threading.Lock()
    both = threading.Barrier(2, timeout=5)  # each call holds its replica until the other has arrived

    def post(url, json, timeout, trust_env):
        with lock:
            seen.append(url)
        both.wait()
        return httpx.Response(200, json={"text": answer(), "seconds": 0.1})

    monkeypatch.setattr(rag_server.httpx, "post", post)
    client = TestClient(rag_server.create_app(FakeIndex((A20, A22, B8)), rag_server.pooled_llm(urls)))
    results = []
    threads = [threading.Thread(target=lambda: results.append(client.post("/v1/chat/completions", json=body())))
               for _ in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert [r.status_code for r in results] == [200, 200]
    assert sorted(seen) == urls


def test_rag_server_llm_fails_over_on_connect_error(monkeypatch):
    urls = ["http://127.0.0.1:1/generate", "http://127.0.0.1:2/generate"]
    seen = []

    def post(url, json, timeout, trust_env):
        seen.append(url)
        if url == urls[0]:
            raise httpx.ConnectError("down")
        return httpx.Response(200, json={"text": "ответ"})

    monkeypatch.setattr(rag_server.httpx, "post", post)
    assert rag_server.pooled_llm(urls)("s", "p", 10) == "ответ"
    assert seen == urls


# ---- the judge client against this server -------------------------------------------------

def test_judge_answer_through_rag_server(make, monkeypatch):
    client, _, _ = make(answer())

    def post(url, json, timeout, trust_env):
        assert url == "http://127.0.0.1:30001/v1/chat/completions"
        return client.post("/v1/chat/completions", json=json)

    monkeypatch.setenv("JUDGE_URL", "http://127.0.0.1:30001")
    monkeypatch.setattr(clients.httpx, "post", post)
    ev = [Frame(1.0, JPEG)]
    a = clients.judge_answer(OBS, ev, CANDS, 10)
    assert a == {"rule_id": "other", "justification": "Может указывать на нарушение.", "confidence": 0.7,
                 "clause_id": "V2:8", "violation": 0.85,
                 "citation": cite(B8, "Посторонние лица в помещение кассы не допускаются.")}
    assert clients.judge(OBS, ev, CANDS, 10) == ("other", "Может указывать на нарушение.", 0.7)
