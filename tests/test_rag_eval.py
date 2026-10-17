"""tools/rag_eval.py: the dump's judge candidates are the pipeline's, the Sonnet backend never runs on the pod,
and the report scores a clip by its best window."""
import json
import sys
import types
from pathlib import Path


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
