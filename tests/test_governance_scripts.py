"""Slice 10: the fairness audit and report-table scripts are deterministic and read only run outputs."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fa = _load("fairness_audit", "scripts/fairness_audit.py")
analysis = _load("analysis", "evaluation/analysis.py")


def _ticket(tid, tier, fluency, words, route, intent="api_key_issue", channel="email", region="europe"):
    return {"ticket_id": tid, "channel": channel, "customer_tier": tier, "language_fluency": fluency,
            "customer_region": region, "subject": "s", "body": " ".join(["word"] * words),
            "labels": {"intent": intent, "expected_route": route, "expected_doc_ids": ["DOC-A"],
                       "must_not_auto_respond": route != "auto_respond"}}


def _decision(tid, action, intent="api_key_issue", doc="DOC-A", score=0.5, codes=()):
    return {"ticket_id": tid, "action_taken": action, "prediction": {"value": intent},
            "sources_used": [{"doc_id": doc, "score": score}], "reason_codes": list(codes)}


@pytest.fixture
def fixture():
    tickets = [_ticket("T1", "enterprise", "fluent", 5, "auto_respond"),
               _ticket("T2", "enterprise", "non_fluent", 10, "auto_respond"),
               _ticket("T3", "standard", "fluent", 20, "auto_respond"),
               _ticket("T4", "standard", "non_fluent", 30, "escalate", intent="security_incident"),
               {"ticket_id": "T5", "body": "unlabelled tickets are ignored"}]
    decisions = [_decision("T1", "auto_respond"),
                 _decision("T2", "auto_respond"),
                 _decision("T3", "escalate", score=0.2, codes=["no_relevant_documentation"]),
                 _decision("T4", "escalate", intent="security_incident", codes=["restricted_intent"]),
                 _decision("T5", "escalate")]
    return tickets, decisions


def test_fairness_audit_values_on_a_small_fixture(fixture):
    rep = fa.audit(*fixture)
    assert rep["tickets"] == 4 and rep["length_split_median_body_words"] == 15
    tier = rep["segments"]["customer_tier"]
    ent, std = tier["groups"]["enterprise"], tier["groups"]["standard"]
    assert (ent["tickets"], ent["resolution_rate"], ent["quality_routing_accuracy"]) == (2, 1.0, 1.0)
    assert (std["resolution_rate"], std["quality_routing_accuracy"], std["auto_precision"]) == (0.0, 0.5, None)
    assert std["resolution_rate_points_below_best"] == 100.0 and std["missed_expected_auto_by_reason"] == {"no_relevant_documentation": 1}
    assert std["label_ceiling_expected_auto"] == 0.5 and std["retrieval_hit_top1"] == 1.0
    assert tier["variation_points"]["resolution_rate"] == 100.0 and tier["meets_under_5_points"]["resolution_rate"] is False
    length = rep["segments"]["length"]["groups"]
    assert length["short"]["tickets"] == 2 and length["short"]["resolution_rate"] == 1.0   # 5 and 10 words <= median 15
    assert rep["largest_gap"]["points"] == 100.0


def test_fairness_audit_cli_is_byte_identical_across_runs(fixture, tmp_path):
    tickets, decisions = fixture
    run = tmp_path / "run"
    run.mkdir()
    (run / "decisions.json").write_text(json.dumps(decisions), encoding="utf-8")
    (run / "metrics.json").write_text(json.dumps({"run": {"run_id": "run-x", "date": "2026-01-01"}}), encoding="utf-8")
    tf = tmp_path / "tickets.json"
    tf.write_text(json.dumps(tickets), encoding="utf-8")
    outs = []
    for i in (1, 2):
        out = tmp_path / f"out{i}" / "audit"
        fa.main(["--run", str(run), "--tickets", str(tf), "--out", str(out)])
        outs.append((out.with_suffix(".json").read_bytes(), out.with_suffix(".md").read_bytes()))
    assert outs[0] == outs[1]
    md = outs[0][1].decode()
    assert "NOT met" in md and "| enterprise | 2 | 100.0%" in md
    report = json.loads(outs[0][0])
    assert report["run_id"] == "run-x" and report["run"] == "run"            # never a local absolute path
    assert str(tmp_path) not in outs[0][0].decode() + md


def test_report_tables_from_a_real_harness_run_mark_missing_runs_and_match_metrics(tmp_path, monkeypatch):
    from cloudserve.evaluation import main as harness
    monkeypatch.setenv("CLOUDSERVE_KILL_SWITCH_FILE", str(tmp_path / "no_switch"))
    tickets = json.loads((ROOT / "data/validation_tickets.json").read_text(encoding="utf-8-sig"))[:6]
    tf = tmp_path / "six.json"
    tf.write_text(json.dumps(tickets), encoding="utf-8")
    results = tmp_path / "results"
    harness(["--input", str(tf), "--output", str(results / "final_offline_default")])
    m = json.loads((results / "final_offline_default" / "metrics.json").read_text(encoding="utf-8"))

    text = analysis.build("test-version", results_dir=results)
    assert text == analysis.build("test-version", results_dir=results)          # deterministic
    assert "git `test-version`" in text and "never available" in text
    assert text.count("missing at generation time") == len(analysis.RUNS) - 1
    b = m["business"]
    assert f"{b['first_contact_resolution_if_kill_switch_released_pct']}% if released" in text
    g = m["governance"]
    assert f"decisions logged {g['decisions_logged']} / processed {g['tickets_processed']}" in text
    assert "IN-SAMPLE" in text                                                   # the dev run is labelled even when missing
    assert "duplicate development tickets" in text and str(tmp_path) not in text
