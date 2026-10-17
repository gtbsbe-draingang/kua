"""VideoJev wrapper with a fake model: the request shapes, the result mapping and the lock. No torch needed."""
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from pipeline import videojev
from pipeline.videojev import Curves, VideoJev

ROOT = Path(__file__).resolve().parent.parent
RECORD = {"frames": 6, "fps": 2.0, "grid_thw": [3, 2, 2], "score_times": [0.25, 1.25, 2.5],
          "score_bounds": [[0.0, 0.75], [0.75, 1.75], [1.75, 3.0]], "sampled_duration": 3.0}


class FakeModel:
    def __init__(self, choice_probs=None, delay=0.0):
        self.calls = []
        self.choice_probs = choice_probs
        self.delay = delay
        self.active = self.max_active = 0
        self._count_lock = threading.Lock()

    def _enter(self):
        with self._count_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(self.delay)
        with self._count_lock:
            self.active -= 1

    def encode_video(self, pixels, grid):
        self.calls.append(("encode", pixels, grid))
        return "features"

    def predict_features(self, processor, features, record, questions, *, context, threshold=None):
        self._enter()
        self.calls.append(("features", features, questions, context))
        n = len(record["score_times"])
        out = [{"id": q["id"], "task": q["task"], "probabilities": [0.1 * (k + 1)] * n, "interval": [0.0, 1.0]}
               for k, q in enumerate(questions)]
        return out[::-1]  # the wrapper must map by id, not by position

    def predict_video(self, processor, video, questions, *, fps, max_frames, context, threshold=None):
        self._enter()
        self.calls.append(("video", video, questions, fps, max_frames, context))
        q = questions[0]
        if q["task"] == "grounding":
            return [{"id": x["id"], "task": "grounding", "probabilities": [0.5, 0.6], "interval": [0.0, 1.0]}
                    for x in questions]
        probs = self.choice_probs or [1.0 / len(q["options"])] * len(q["options"])
        return [{"id": q["id"], "task": "choice", "probabilities": probs, "answer_index": 0,
                 "answer": q["options"][0]}]


def make(model, prepare=True):
    vj = VideoJev()
    seen = []

    def fake_prepare(processor, path, *, fps, max_frames, context):
        seen.append((path, fps, max_frames, context))
        return "pixels", dict(RECORD)

    vj._install(model, processor="proc", prepare_video=fake_prepare if prepare else None, tensor=lambda x: x)
    return vj, seen


def test_ground_one_call_mapped_by_id_with_model_times():
    model = FakeModel()
    vj, seen = make(model)
    out = vj.ground("/v.mp4", ["a cashier walks away", "cash on the desk", "a cashier walks away"])
    assert isinstance(out, Curves) and isinstance(out, dict)
    assert list(out) == ["a cashier walks away", "cash on the desk"]  # duplicates collapse
    assert out["a cashier walks away"] == pytest.approx([0.1] * 3)
    assert out["cash on the desk"] == pytest.approx([0.2] * 3)
    assert out.times == [0.25, 1.25, 2.5] and out.bounds == [(0.0, 0.75), (0.75, 1.75), (1.75, 3.0)]
    assert out.step_sec == pytest.approx(1.0)  # one score per frame pair: 2 / fps
    assert seen == [("/v.mp4", videojev.GROUND_FPS, videojev.MAX_FRAMES, videojev.GROUND_CONTEXT)]
    (_, _, grid), (_, feats, questions, context) = model.calls
    assert grid == [RECORD["grid_thw"]] and feats == "features" and context == videojev.GROUND_CONTEXT
    assert [q["task"] for q in questions] == ["grounding", "grounding"]
    assert len({q["id"] for q in questions}) == 2


def test_ground_without_prepare_video_uses_predict_video():
    model = FakeModel()
    vj, _ = make(model, prepare=False)
    out = vj.ground("/v.mp4", ["q1"])
    assert out == {"q1": [0.5, 0.6]} and out.times is None and out.bounds is None
    kind, video, questions, fps, max_frames, context = model.calls[0]
    assert (kind, video, fps, max_frames, context) == (
        "video", "/v.mp4", videojev.GROUND_FPS, videojev.MAX_FRAMES, videojev.GROUND_CONTEXT)


def test_ground_no_queries_skips_model():
    model = FakeModel()
    vj, _ = make(model)
    assert vj.ground("/v.mp4", []) == {} and model.calls == []


def test_choose_lettered_options_keyed_by_original_text():
    model = FakeModel(choice_probs=[0.2, 0.2, 0.4])  # renormalised to sum 1
    vj, _ = make(model)
    opts = ["cash left unattended", "personal phone at the desk", "normal cash handling"]
    out = vj.choose("/w.mp4", opts)
    assert out == pytest.approx({opts[0]: 0.25, opts[1]: 0.25, opts[2]: 0.5})
    kind, video, questions, fps, max_frames, context = model.calls[0]
    assert (kind, video, fps, context) == ("video", "/w.mp4", videojev.CHOICE_FPS, videojev.CHOICE_CONTEXT)
    q, = questions
    assert q["task"] == "choice" and q["query"] == videojev.CHOICE_QUERY
    assert q["options"] == ["A. Cash left unattended.", "B. Personal phone at the desk.", "C. Normal cash handling."]


def test_choose_bad_output_and_bad_options():
    vj, _ = make(FakeModel(choice_probs=[float("nan"), 1.0]))
    assert vj.choose("/w.mp4", ["x", "y"]) == {"x": 0.0, "y": 1.0}
    vj, _ = make(FakeModel(choice_probs=[1.0]))  # wrong length
    with pytest.raises(ValueError):
        vj.choose("/w.mp4", ["x", "y"])
    with pytest.raises(ValueError):
        vj.choose("/w.mp4", ["only one"])
    with pytest.raises(ValueError):
        vj.choose("/w.mp4", ["same", "same"])


def test_option_labels_past_z():
    assert videojev._label(0) == "A" and videojev._label(25) == "Z" and videojev._label(26) == "AA"


def test_model_calls_are_serialised():
    model = FakeModel(delay=0.05)
    vj, _ = make(model)
    threads = [threading.Thread(target=vj.ground, args=("/v.mp4", ["q"])) for _ in range(3)]
    threads += [threading.Thread(target=vj.choose, args=("/w.mp4", ["x", "y"])) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert model.max_active == 1


def test_light_imports_do_not_load_torch():
    code = ("import sys, pipeline, pipeline.stub, pipeline.real, pipeline.videojev, pipeline.clients\n"
            "from pipeline.videojev import VideoJev\n"
            "VideoJev()\n"
            "bad = [m for m in ('torch', 'transformers') if m in sys.modules]\n"
            "assert not bad, bad\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def _hold(lock, seconds):
    held = threading.Event()

    def run():
        with lock:
            held.set()
            time.sleep(seconds)

    threading.Thread(target=run, daemon=True).start()
    held.wait()


@pytest.mark.parametrize("call", ["ground", "choose"])
def test_busy_model_raises_after_the_acquire_timeout(call):
    model = FakeModel()
    vj, _ = make(model)
    _hold(vj._lock, 1.0)
    t = time.monotonic()
    with pytest.raises(videojev.ModelBusy):
        if call == "ground":
            vj.ground("/v.mp4", ["q"], timeout=0.2)
        else:
            vj.choose("/w.mp4", ["x", "y"], timeout=0.2)
    assert 0.15 < time.monotonic() - t < 0.8 and model.calls == []


def test_zero_timeout_is_a_try_and_none_waits():
    vj, _ = make(FakeModel())
    assert vj.ground("/v.mp4", ["q"], timeout=0) == {"q": pytest.approx([0.1] * 3)}  # free: proceeds
    _hold(vj._lock, 0.3)
    with pytest.raises(videojev.ModelBusy):
        vj.choose("/w.mp4", ["x", "y"], timeout=0)
    assert vj.choose("/w.mp4", ["x", "y"]) == {"x": 0.5, "y": 0.5}  # no timeout: waits for the holder
