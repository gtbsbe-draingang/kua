"""tools/rag_eval.py: the dump's judge candidates are the pipeline's, the Sonnet backend never runs on the pod,
and the report scores a clip by its best window."""
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import rag_eval  # noqa: E402
import verify_candidates as vc  # noqa: E402

from pipeline import logic, real  # noqa: E402


def _window(p_rules, obs="Кассир уходит, деньги лежат на столе."):
    return {"key": "X-1|17.000|29.000|cash_unattended|multi+zoom", "clip": "X-1", "start": 17.0, "end": 29.0,
            "rule": "cash_unattended", "prompt": "multi+zoom", "timecode": 22.0, "p_rules": p_rules,
            "p_yes": max(p for p in p_rules.values() if p is not None), "observation": obs}


def test_rag_eval_dump_record_candidates_match_pipeline():
    cfg = real._config()
    p_rules = {r["id"]: 0.1 for r in vc.multi_rules(cfg)}
    p_rules.update({"cashier_leaves_during_service": 0.62, "previous_cash_on_desk": 0.4,
                    "cash_unattended": 0.55, "deposit_not_counted": None})
    hits = [{"clause_id": "V1900019680:22", "score": 0.71, "part": 0}]
    rec = rag_eval.dump_record(_window(p_rules), cfg, hits)
    known = {k: v for k, v in p_rules.items() if v is not None}
    ids = logic.judge_candidates(known, "cashier_leaves_during_service")
    assert ids == ["cashier_leaves_during_service", "cash_unattended", "previous_cash_on_desk", "other"]
    assert rec["candidate_ids"] == ids and rec["argmax"] == "cashier_leaves_during_service"
    assert rec["candidates"] == [real._judge_candidate(cfg.rules[i]) for i in ids]
    assert rec["hits"] == hits and rec["observation"] == "Кассир уходит, деньги лежат на столе."
    assert (rec["clip"], rec["timecode"], rec["p_yes"]) == ("X-1", 22.0, 0.62)
    # every answer missing (the check failed): no rule to rank, so only the generic candidate
    w = _window({"cash_unattended": 0.3}, obs="")
    w["p_rules"], w["p_yes"] = {"cash_unattended": None}, None
    none = rag_eval.dump_record(w, cfg, [])
    assert none["candidate_ids"] == logic.judge_candidates({}, "") == ["other"] and none["argmax"] is None


def test_rag_eval_sonnet_refuses_on_pod(tmp_path, monkeypatch, capsys):
    built = []
    fake = types.ModuleType("anthropic")
    fake.Anthropic = lambda *a, **k: built.append(1)  # building a client would be recorded here
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    dump = tmp_path / "dump.jsonl"
    dump.write_text(json.dumps({"key": "k", "observation": "x", "candidates": [], "hits": []}) + "\n")
    out = tmp_path / "out.jsonl"
    monkeypatch.setenv("MTA_ROOT", "/home/jovyan/model_train_amir")
    assert rag_eval.main(["run", "--backend", "sonnet", "--dump", str(dump), "--out", str(out)]) == 2
    assert not built and not out.exists()
    assert "refus" in capsys.readouterr().err.lower()
    # a pod home folder alone is enough
    monkeypatch.delenv("MTA_ROOT")
    monkeypatch.setattr(rag_eval, "POD_HOME", tmp_path)
    assert rag_eval.run_sonnet(dump, out, 8) == 2 and not built and not out.exists()


def test_rag_eval_report_uses_clip_max():
    labels = {"X-1": {"has_violation": 1, "truth": [{"t_label": 6.0}]},
              "X-2": {"has_violation": 1, "truth": [{"t_label": 2.0}]},
              "X-3": {"has_violation": 0, "truth": []}}

    def rec(clip, tc, v, clause="V1900019680:20"):
        return {"key": f"{clip}|{tc}", "clip": clip, "timecode": tc, "seconds": 1.0,
                "answer": None if v is None else {"violation": v, "rule_id": "cash_unattended", "clause_id": clause,
                                                  "justification": "", "confidence": 0.5}}

    out = [rec("X-1", 2.0, 0.1), rec("X-1", 6.0, 0.9), rec("X-2", 2.0, 0.6),
           rec("X-3", 2.0, 0.7, None), rec("X-3", 6.0, 0.0), rec("X-3", 10.0, None)]
    sm = rag_eval.summarize(out, labels, reps=50)
    assert sm["clip_auc"] == 0.5  # max 0.9 and 0.6 against 0.7 (by the mean it would be 1.0)
    assert sm["window_auc"] == vc.window_auc(rag_eval.rows(out, labels), labels, key="violation")
    assert sm["scored"] == 5 and sm["failed"] == 1
    assert sm["clauses"] == {"V1900019680:20": 4, "none": 1}


# ---- recall: several expected clauses, recall@k, MRR, and the dense / lexical / hybrid comparison --------------

def test_rag_eval_first_rank_takes_any_expected_clause():
    assert rag_eval.expected_ids({"expect": "A:1"}) == ["A:1"]
    assert rag_eval.expected_ids({"expect": ["A:1", "B:2"]}) == ["A:1", "B:2"]
    assert rag_eval.first_rank(["X:1", "B:2", "A:1"], ["A:1", "B:2"]) == 2
    assert rag_eval.first_rank(["X:1"], ["A:1"]) is None
    assert rag_eval.first_rank([], ["A:1"]) is None


def test_rag_eval_summarize_ranks():
    sm = rag_eval.summarize_ranks([1, 2, 9, None, 12, 13], ks=(8, 12))
    assert sm["n"] == 6 and (sm["hit@8"], sm["hit@12"]) == (2, 4)
    assert sm["recall@8"] == pytest.approx(2 / 6) and sm["recall@12"] == pytest.approx(4 / 6)
    assert sm["mrr"] == pytest.approx((1 + 1 / 2 + 1 / 9 + 0 + 1 / 12 + 1 / 13) / 6)
    assert rag_eval.summarize_ranks([], ks=(8,)) == {"n": 0, "mrr": 0.0, "hit@8": 0, "recall@8": 0.0}


def test_rag_eval_report_recall_lines_and_summaries():
    queries = [{"text": "a", "expect": "A:1"}, {"text": "b", "expect": ["B:1", "B:2"]}]
    answers = {"dense": {"a": ["A:1"], "b": ["Z:1"] * 9 + ["B:2"]}, "hybrid": {"a": ["Z:1", "A:1"], "b": ["B:1"]}}
    rankers = {name: (lambda text, m=m: m[text]) for name, m in answers.items()}
    lines = []
    out = rag_eval.report_recall(queries, rankers, ks=(8, 12), echo=lines.append)
    assert out["dense"]["hit@8"] == 1 and out["dense"]["hit@12"] == 2
    assert out["hybrid"]["hit@8"] == 2 and out["hybrid"]["mrr"] == pytest.approx((1 / 2 + 1) / 2)
    assert "dense 10" in lines[1] and "hybrid  1" in lines[1] and "B:1/B:2" in lines[1]
    assert any(l.startswith("hybrid") and "recall@8 2/2 = 1.000" in l and "MRR 0.750" in l for l in lines)


def _tiny_index():
    from rag.corpus import Clause
    from rag.retrieve import Index

    def clause(cid, *parts):
        code, num = cid.split(":")
        return Clause(cid, code, "s", "d", "a", num, "ch", tuple(parts), "u")

    vecs = {"телефон": (1, 0), "шум": (1, 0.01), "средства связи запрещены": (1, 1), "прочее": (0, 1)}

    def embed(texts):
        v = np.array([vecs[t] for t in texts], dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    return Index([clause("A:1", "шум"), clause("B:1", "средства связи запрещены"), clause("C:1", "прочее")],
                 embed_fn=embed, hybrid=True)


def test_rag_eval_mode_rankers_compare_dense_lexical_hybrid_on_one_index():
    index = _tiny_index()
    rankers = rag_eval.mode_rankers(index, ["dense", "lexical", "hybrid"], 3)
    assert rankers["dense"]("телефон") == ["A:1", "B:1", "C:1"]
    assert rankers["lexical"]("телефон") == ["B:1"]
    assert rankers["hybrid"]("телефон") == ["B:1", "A:1", "C:1"]
    assert index.hybrid is True  # flipped around each call and put back
    assert rankers["dense"]("телефон") == ["A:1", "B:1", "C:1"]
    assert rag_eval.mode_rankers(index, ["hybrid"], 1)["hybrid"]("телефон") == ["B:1"]


def test_rag_eval_recall_local_refuses_off_the_pod(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("MTA_ROOT", raising=False)
    monkeypatch.setattr(rag_eval, "POD_HOME", tmp_path / "no-such-home")
    monkeypatch.setattr(rag_eval, "local_index", lambda: pytest.fail("the encoder would have been loaded"))
    assert rag_eval.main(["recall", "--local"]) == 2
    assert "refuses" in capsys.readouterr().err


def test_rag_eval_recall_local_on_the_pod(tmp_path, monkeypatch, capsys):
    q = tmp_path / "q.json"
    q.write_text(json.dumps([{"text": "телефон", "expect": ["B:1", "Z:9"]}]), encoding="utf-8")
    monkeypatch.setenv("MTA_ROOT", "/home/jovyan/model_train_amir")
    monkeypatch.setattr(rag_eval, "local_index", _tiny_index)
    assert rag_eval.main(["recall", "--local", "--queries", str(q), "--ks", "1,2"]) == 0
    out = capsys.readouterr().out
    assert "dense  2 | lexical  1 | hybrid  1" in out
    assert "dense    recall@1 0/1 = 0.000 | recall@2 1/1 = 1.000 | MRR 0.500" in out
    assert "hybrid   recall@1 1/1 = 1.000" in out
    assert rag_eval.main(["recall", "--local", "--queries", str(q), "--modes", "dense,bogus"]) == 2
    assert "bogus" in capsys.readouterr().err


def test_rag_eval_recall_local_needs_the_lexical_channel_for_the_other_modes(tmp_path, monkeypatch, capsys):
    from rag import lexical

    q = tmp_path / "q.json"
    q.write_text(json.dumps([{"text": "телефон", "expect": "B:1"}]), encoding="utf-8")
    monkeypatch.setenv("MTA_ROOT", "/home/jovyan/model_train_amir")
    monkeypatch.setattr(lexical, "Lexicon", lambda *a, **k: 1 / 0)
    monkeypatch.setattr(rag_eval, "local_index", _tiny_index)
    assert rag_eval.main(["recall", "--local", "--queries", str(q)]) == 2
    assert "lexical channel did not build" in capsys.readouterr().err
    assert rag_eval.main(["recall", "--local", "--queries", str(q), "--modes", "dense"]) == 0


def test_rag_eval_recall_through_the_server(tmp_path, monkeypatch, capsys):
    q = tmp_path / "q.json"
    q.write_text(json.dumps([{"text": "a", "expect": "A:1"}, {"text": "b", "expect": ["B:1"]}]), encoding="utf-8")
    asked = []

    def fake_retrieve(url, text, k):
        asked.append((url, text, k))
        return [{"clause_id": "X:1"}, {"clause_id": "A:1" if text == "a" else "B:2"}]

    monkeypatch.setattr(rag_eval, "retrieve", fake_retrieve)
    assert rag_eval.main(["recall", "--queries", str(q), "--url", "http://x"]) == 0
    assert asked == [("http://x", "a", 20), ("http://x", "b", 20)]
    out = capsys.readouterr().out
    assert "server   recall@8 1/2 = 0.500 | recall@12 1/2 = 0.500 | MRR 0.250" in out


def test_rag_holdout_expects_existing_in_force_clauses():
    from rag import corpus

    ids = {c.id for c in corpus.load_clauses()}
    queries = json.loads((Path(__file__).resolve().parent / "data" / "rag_holdout.json").read_text(encoding="utf-8"))
    assert len(queries) == 18
    for q in queries:
        assert q["text"].strip() and q["expect"] and set(q["expect"]) <= ids, q
        assert "V2000020075:16" not in q["expect"]  # suspended until 2027: the teammate's expectation is dropped
