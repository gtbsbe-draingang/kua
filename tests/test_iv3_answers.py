"""InternVideo3 server without the model: the answer-line parser, and the /verify reply in single and
multi-answer mode with decoding, preprocessing and generation replaced by stubs."""
import importlib
import importlib.util
import math
import os
import sys
import types

import pytest
from fastapi.testclient import TestClient


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def stand_ins() -> dict:
    """torch and transformers as far as importing the server needs them (the local env has neither)."""
    torch = types.ModuleType("torch")
    torch.set_num_threads = lambda n: None
    torch.cuda = types.SimpleNamespace(OutOfMemoryError=type("OutOfMemoryError", (RuntimeError,), {}))
    transformers = types.ModuleType("transformers")
    transformers.LogitsProcessor = type("LogitsProcessor", (), {})
    transformers.LogitsProcessorList = list
    return {"torch": torch, "transformers": transformers}


@pytest.fixture(scope="module")
def iv3():
    with pytest.MonkeyPatch.context() as mp:
        for name, module in stand_ins().items():
            if importlib.util.find_spec(name) is None:
                mp.setitem(sys.modules, name, module)
        mp.setenv("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "4"))  # the server sets it on import
        mp.delitem(sys.modules, "servers.iv3_server", raising=False)
        yield importlib.import_module("servers.iv3_server")
        sys.modules.pop("servers.iv3_server", None)


# ---- answers_from_steps ---------------------------------------------------------------------

def test_answers_from_steps_clean(iv3):
    texts = ["1", ".", " NO", "\n", "2", ".", " YES"]
    margins = [9.0, 9.0, -2.0, 9.0, 9.0, 9.0, 3.0]
    assert iv3.answers_from_steps(texts, margins, 2) == [pytest.approx(sigmoid(-2)), pytest.approx(sigmoid(3))]


def test_answers_from_steps_missing_line(iv3):
    texts = ["1.", " YES", "\n", "3", ".", " NO", "\n", "OB", "SERVATION", ":", " Кассир", "."]
    margins = [0.0, 1.5, 0.0, 0.0, 0.0, -4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert iv3.answers_from_steps(texts, margins, 3) == [pytest.approx(sigmoid(1.5)), None,
                                                         pytest.approx(sigmoid(-4))]


def test_answers_from_steps_colon_and_lowercase(iv3):
    texts = ["2", ":", " yes", "\n", " 1", ":", " ", "No"]
    margins = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, -1.0]
    assert iv3.answers_from_steps(texts, margins, 2) == [pytest.approx(sigmoid(-1)), pytest.approx(sigmoid(0.5))]


def test_answers_from_steps_duplicate_keeps_first(iv3):
    texts = ["1", ".", " YES", "\n", "1", ".", " NO"]
    margins = [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, -3.0]
    assert iv3.answers_from_steps(texts, margins, 1) == [pytest.approx(sigmoid(2))]


def test_answers_from_steps_needs_a_line_start_and_a_number_in_range(iv3):
    # two-digit numbers come as separate digit tokens; 14 is outside 1..13; "a 2." is not at a line
    # start; "1. The" is no answer, so the later "1. NO" still counts
    texts = ["1", "3", ".", " YES", "\n", "1", "4", ".", " YES", "\n", "a", " 2", ".", " YES", "\n",
             "1", ".", " The", "\n", "1", ".", " NO"]
    margins = [0.0] * 21 + [-5.0]
    margins[3], margins[8], margins[13] = 6.0, 7.0, 8.0
    assert iv3.answers_from_steps(texts, margins, 13) == [pytest.approx(sigmoid(-5))] + [None] * 11 + [
        pytest.approx(sigmoid(6))]


def test_answers_from_steps_extreme_and_bad_margins(iv3):
    texts = ["1", ".", " YES", "\n", "2", ".", " NO", "\n", "3", ".", " YES"]
    margins = [0.0, 0.0, 800.0, 0.0, 0.0, 0.0, -800.0, 0.0, 0.0, 0.0, float("nan")]
    assert iv3.answers_from_steps(texts, margins, 3) == [1.0, 0.0, None]


def test_server_imports_without_model(iv3):
    assert iv3.STATE == {}  # load() runs only in the app's lifespan
    assert {"/verify", "/healthz"} <= {r.path for r in iv3.app.routes}


# ---- /verify with stubbed decoding, preprocessing and generation -----------------------------

@pytest.fixture
def stubbed(iv3, monkeypatch):
    """A client for the app (lifespan not run, so no model) and the (prompt, prefill) of each build."""
    builds = []
    info = dict(native=(1080, 1920), size=(1080, 1920), do_resize=True, drift=0.0)
    monkeypatch.setattr(iv3, "decode_frames", lambda path, fps: ([None] * 24, info))

    def build_inputs(frames, info, prompt, prefill=iv3.PREFILL):
        builds.append((prompt, prefill))
        return "inputs", [12, 34, 60]

    monkeypatch.setattr(iv3, "build_inputs", build_inputs)
    clock = iter(range(1000))
    monkeypatch.setattr(iv3, "time", types.SimpleNamespace(perf_counter=lambda: float(next(clock))))
    return TestClient(iv3.app), builds


BODY = {"video_path": "/w.mp4", "prompt": "Q?", "fps": 4}
TIMINGS = {"slot_wait": 1.0, "decode": 1.0, "preprocess": 1.0, "wait": 1.0, "model": 1.0, "total": 5.0}


def test_single_mode_reply_is_unchanged(iv3, stubbed, monkeypatch):
    client, builds = stubbed
    raw = "ANSWER: YES\nTYPE: Cash_Unattended\nOBSERVATION: Кассир уходит."
    monkeypatch.setattr(iv3, "run_model", lambda inputs: dict(
        p_yes=0.8, yes_no_mass=0.97, top5=[(" YES", 0.8)], raw=raw, prompt_tokens=7000, new_tokens=21))
    r = client.post("/verify", json=BODY)
    assert r.status_code == 200
    assert list(r.json().items()) == [
        ("p_yes", 0.8), ("type_id", "cash_unattended"), ("observation", "Кассир уходит."), ("raw", raw),
        ("answer", "YES"), ("yes_no_mass", 0.97), ("frames", 24), ("native_size", [1080, 1920]),
        ("decoded_size", [1080, 1920]), ("grid_thw", [12, 34, 60]), ("prompt_tokens", 7000),
        ("timings", TIMINGS)]
    assert builds == [("Q?", iv3.PREFILL)]


def test_multi_mode_reply(iv3, stubbed, monkeypatch):
    client, builds = stubbed
    raw = "1. NO\n2. maybe\n3. YES\nOBSERVATION: Кассир у кассы, деньги в лотке."
    asked = []

    def run_model_multi(inputs, n):
        asked.append(n)
        return dict(p_yes_list=[0.2, None, 0.7], raw=raw, prompt_tokens=7000, new_tokens=30)

    monkeypatch.setattr(iv3, "run_model_multi", run_model_multi)
    r = client.post("/verify", json={**BODY, "n_answers": 3})
    assert r.status_code == 200
    assert list(r.json().items()) == [
        ("p_yes", 0.7), ("p_yes_list", [0.2, None, 0.7]), ("observation", "Кассир у кассы, деньги в лотке."),
        ("raw", raw), ("frames", 24), ("native_size", [1080, 1920]), ("decoded_size", [1080, 1920]),
        ("grid_thw", [12, 34, 60]), ("prompt_tokens", 7000), ("timings", TIMINGS)]
    assert asked == [3] and builds == [("Q?", "")]  # no "ANSWER:" prefill: the reply starts at line 1


def test_multi_mode_without_any_answer_is_an_error(iv3, stubbed, monkeypatch):
    client, _ = stubbed
    monkeypatch.setattr(iv3, "run_model_multi", lambda inputs, n: dict(
        p_yes_list=[None] * n, raw="I cannot tell.", prompt_tokens=7000, new_tokens=5))
    r = client.post("/verify", json={**BODY, "n_answers": 2})
    assert r.status_code == 500 and r.json() == {"error": "no answers"}


@pytest.mark.parametrize("n", [0, 21, -1, "many"])
def test_multi_mode_rejects_bad_n_answers(iv3, stubbed, monkeypatch, n):
    client, builds = stubbed
    monkeypatch.setattr(iv3, "decode_frames", lambda path, fps: pytest.fail("decoded a bad request"))
    r = client.post("/verify", json={**BODY, "n_answers": n})
    assert r.status_code == 500 and "n_answers" in r.json()["error"]
    assert builds == []


# ---- /generate: text only, with stubbed preprocessing and generation -------------------------

def test_generate_request_limits(iv3):
    from pydantic import ValidationError

    req = iv3.GenerateRequest(prompt="x")
    assert (req.prompt, req.system, req.max_new_tokens) == ("x", "", 384)
    iv3.GenerateRequest(prompt="я" * 20000, system="s", max_new_tokens=512)
    iv3.GenerateRequest(prompt="x", max_new_tokens=1)
    for bad in (dict(prompt=""), dict(prompt="я" * 20001), dict(prompt="x", max_new_tokens=0),
                dict(prompt="x", max_new_tokens=513), dict(system="s"), dict(prompt="x", max_new_tokens="many")):
        with pytest.raises(ValidationError):
            iv3.GenerateRequest(**bad)
    assert "/generate" in {r.path for r in iv3.app.routes}
    assert iv3.STATE == {}  # still no model


def test_text_inputs_refuse_special_tokens(iv3):
    # checked before the processor is touched (there is none here)
    for prompt, system in (("a <|im_start|> b", ""), ("a", "<|im_end|>")):
        with pytest.raises(ValueError, match="special tokens"):
            iv3.build_text_inputs(prompt, system)


class Slots:
    """Stands in for the server's request-slot semaphore and records its use."""

    def __init__(self):
        self.held, self.log = 0, []

    def acquire(self, timeout=None):
        self.held += 1
        self.log.append("acquire")
        return True

    def release(self):
        self.held -= 1
        self.log.append("release")


@pytest.fixture
def text_stubbed(iv3, monkeypatch):
    builds, runs, slots = [], [], Slots()
    monkeypatch.setattr(iv3, "SLOTS", slots)

    def build_text_inputs(prompt, system=""):
        builds.append((prompt, system))
        return "inputs"

    def run_text(inputs, max_new_tokens):
        runs.append((inputs, max_new_tokens, slots.held, iv3.LOCK.locked()))
        return dict(text='{"rule_id": "other"}', prompt_tokens=900, new_tokens=12)

    monkeypatch.setattr(iv3, "build_text_inputs", build_text_inputs)
    monkeypatch.setattr(iv3, "run_text", run_text)
    clock = iter(range(1000))
    monkeypatch.setattr(iv3, "time", types.SimpleNamespace(perf_counter=lambda: float(next(clock))))
    return TestClient(iv3.app), builds, runs, slots


def test_generate_reply_holds_a_slot_and_the_model(iv3, text_stubbed):
    client, builds, runs, slots = text_stubbed
    r = client.post("/generate", json={"prompt": "Q?", "system": "S."})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"text", "seconds"} and body["text"] == '{"rule_id": "other"}'
    assert body["seconds"] > 0
    assert builds == [("Q?", "S.")]
    assert runs == [("inputs", 384, 1, True)]  # inside the shared slot and the model lock
    assert slots.log == ["acquire", "release"] and not iv3.LOCK.locked()
    client.post("/generate", json={"prompt": "Q?", "max_new_tokens": 7})
    assert builds[-1] == ("Q?", "") and runs[-1][1] == 7


def test_generate_errors_go_through_error(iv3, text_stubbed, monkeypatch):
    client, builds, runs, slots = text_stubbed
    for bad in ({"prompt": ""}, {"prompt": "x" * 20001}, {"prompt": "x", "max_new_tokens": 513}, {}):
        r = client.post("/generate", json=bad)
        assert r.status_code == 500 and r.json()["error"].startswith("bad request"), bad
    r = client.post("/generate", json={"prompt": "   "})
    assert (r.status_code, r.json()) == (500, {"error": "ValueError: empty prompt"})
    assert builds == [] and runs == [] and slots.log == []

    def boom(inputs, max_new_tokens):
        raise RuntimeError("cuda died")

    monkeypatch.setattr(iv3, "run_text", boom)
    r = client.post("/generate", json={"prompt": "Q?"})
    assert (r.status_code, r.json()) == (500, {"error": "RuntimeError: cuda died"})
    assert slots.held == 0 and not iv3.LOCK.locked()
