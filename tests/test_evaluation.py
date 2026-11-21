"""Unattended harness (A9, A10): arbitrary paths, real pipeline, DB-backed reconciliation."""
import json
import sqlite3
from cloudserve.evaluation import calibration_table, main, per_class


def run(tmp_path, items, *extra):
    inp = tmp_path / "in.json"
    inp.write_text(json.dumps(items), encoding="utf-8")
    out = tmp_path / "out"
    main(["--input", str(inp), "--output", str(out), *extra])
    return json.loads((out / "metrics.json").read_text()), out


def test_harness_runs_real_graph_on_unseen_file_with_bad_rows(tmp_path, validation):
    items = validation[:12] + [{"ticket_id": "BAD", "channel": "bad", "body": "x"}, 42]
    m, out = run(tmp_path, items)
    assert m["run"]["engine"] == "langgraph" and m["volume"]["processed"] == 14
    db_rows = sqlite3.connect(out / "decisions.sqlite").execute("select count(*) from decision_log").fetchone()[0]
    assert db_rows == m["governance"]["decisions_logged"] == 14 and m["governance"]["decision_log_reconciles"]
    for group in ("volume", "business", "technical", "governance", "routing", "fairness_segments"):
        assert group in m
    assert (out / "metrics_summary.md").read_text().startswith("# Evaluation summary")


def test_rerun_into_same_folder_does_not_accumulate_rows(tmp_path, validation):
    run(tmp_path, validation[:3])
    m, out = run(tmp_path, validation[:3])
    assert m["governance"]["decisions_logged"] == 3


def test_full_validation_metrics_include_required_figures_and_safety(tmp_path, validation):
    m, _ = run(tmp_path, validation)
    t = m["technical"]
    assert t["intent_precision_recall_by_class"]["billing_query"]["precision"] is not None
    assert t["latency_seconds"]["p95"] is not None and t["retrieval_hit_rate_any_expected_doc"] is not None
    assert m["routing"]["must_not_auto_respond_violations"] == 0
    assert m["governance"]["private_data_in_outbound_responses"] == 0
    assert m["volume"]["auto_responded_sent"] == 0      # kill switch is the default
    assert m["governance"]["confidence_calibration"]["bands"]


def test_auto_send_flag_releases_answers(tmp_path, validation):
    m, _ = run(tmp_path, validation[:20], "--auto-send")
    assert m["volume"]["auto_responded_sent"] > 0 and m["volume"]["auto_respond_decisions_held_by_kill_switch"] == 0
    assert m["business"]["first_contact_resolution_pct"] > 0


def test_unlabelled_file_still_produces_volume_and_governance_metrics(tmp_path, validation):
    items = [{k: v for k, v in t.items() if k not in ("labels", "history")} for t in validation[:5]]
    m, _ = run(tmp_path, items)
    assert m["labelled_tickets"] == 0 and "routing" not in m and m["governance"]["decision_log_reconciles"]


def test_metric_helpers():
    assert per_class(["a"], ["b"])["a"] == {"support": 1, "predicted": 0, "precision": None, "recall": 0.0}
    rows = calibration_table([(0.9, True), (0.9, False), (0.1, False)])
    assert rows[-1]["observed"] == 0.5 and rows[-1]["gap_points"] == 40.0


def test_training_overlap_caveat_is_computed_for_the_actual_input(tmp_path, validation):
    unseen = [{**t, "body": t["body"] + " (rephrased)"} for t in validation[:4]]
    m, _ = run(tmp_path, validation[:6] + unseen)
    ov = m["run"]["training_overlap"]
    assert ov["tickets"] == 10 and ov["exact_body_matches"] == sum(
        t["body"].strip().lower() in {d["body"].strip().lower() for d in __import__("json").load(open("data/development_tickets.json"))}
        for t in validation[:6])
    assert f"of 10 ticket bodies" in m["limitations"][0]


def test_business_figures_match_hand_computation(tmp_path, validation):
    m, out = run(tmp_path, validation[:30], "--auto-send")
    rs = json.loads((out / "decisions.json").read_text())
    sent = sum(r["answer_sent"] for r in rs)
    assert m["business"]["first_contact_resolution_pct"] == round(100 * sent / 30, 1)
    assert m["business"]["escalation_rate_pct"] == round(100 * (30 - sent) / 30, 1)
