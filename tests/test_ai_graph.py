"""End-to-end LangGraph workflow: actions, kill switch, guardrail blocking, failures, audit."""
import json
import pytest
from cloudserve.logging_store import GOVERNANCE_FIELDS, count_run, open_log, persist
from cloudserve.generate import EvidenceSelection
from cloudserve.runtime import run_batch
from conftest import LOGIN, FixedClassifier


class SpySelector:
    name, version = "spy", "test"

    def __init__(self, result=None, exc=None):
        self.calls, self.result, self.exc = 0, result, exc

    def select(self, ticket, evidence):
        self.calls += 1
        if self.exc:
            raise self.exc
        return self.result or EvidenceSelection(evidence[0]["doc_id"], "x" * 10)


class BrokenClassifier:
    name, version = "broken", "test"

    def classify(self, ticket):
        raise TimeoutError("provider timed out")


def test_graph_is_a_real_compiled_langgraph_with_a_route_branch(make_engine):
    e = make_engine()
    assert e.implementation == "langgraph"
    g = e.graph.get_graph()
    assert {"route", "select_evidence", "validate", "finalize"} <= set(g.nodes)
    assert {(x.source, x.target) for x in g.edges} >= {("route", "select_evidence"), ("route", "finalize")}


def test_kill_switch_off_holds_answer_for_review(make_engine):
    r = make_engine(auto_send=False).process(LOGIN)
    assert (r["action_taken"], r["status"], r["answer_sent"]) == ("auto_respond", "held_for_review", False)
    assert r["response"] and r["citations"] == ["DOC-AUTH-001#resolution"] and "kill switch" in r["reason"]


def test_kill_switch_released_sends_answer(make_engine):
    r = make_engine(auto_send=True).process(LOGIN)
    assert (r["status"], r["delivery"], r["answer_sent"]) == ("auto_responded", "sent", True)
    assert r["handoff"] is None


def test_engineered_guardrail_ticket_is_blocked_even_with_sending_on_a7(make_engine):
    bad = SpySelector(EvidenceSelection("DOC-AUTH-001", "Your account has been unlocked and we have issued a refund for the trouble."))
    r = make_engine(selector=bad, auto_send=True).process(LOGIN)
    assert (r["action_taken"], r["status"], r["answer_sent"], r["response"]) == ("block", "blocked", False, "")
    assert "quote_not_supported_by_source" in r["guardrail_failures"]
    assert r["guardrail_results"]["grounding"] == "fail" and r["handoff"]["why_escalated"]


def test_escalation_skips_the_evidence_model_call(make_engine):
    spy = SpySelector()
    r = make_engine(classifier=FixedClassifier("security_incident", 1.0), selector=spy).process(LOGIN)
    assert r["action_taken"] == "escalate" and spy.calls == 0


def test_classifier_outage_escalates_with_reason(make_engine):
    r = make_engine(classifier=BrokenClassifier(), auto_send=True).process(LOGIN)
    assert r["action_taken"] == "escalate" and r["stage"] == "classification"
    assert r["prediction"]["error_type"] == "TimeoutError" and "FR-10" in r["requirement_ids"]


def test_selector_outage_escalates_not_crashes(make_engine):
    r = make_engine(selector=SpySelector(exc=ConnectionError("down")), auto_send=True).process(LOGIN)
    assert r["action_taken"] == "escalate" and r["stage"] == "generation" and not r["answer_sent"]


@pytest.mark.parametrize("raw", [{"ticket_id": "B1", "channel": "email", "body": ""}, "garbage", None,
                                 {"channel": "pigeon", "body": "hi"}])
def test_malformed_input_produces_logged_escalation(make_engine, raw):
    r = make_engine().process(raw)
    assert r["action_taken"] == "escalate" and r["stage"] == "ingest"
    assert all(f in r for f in GOVERNANCE_FIELDS)


def test_every_path_writes_a_complete_governance_record(make_engine, tmp_path):
    engines = [make_engine(), make_engine(auto_send=True),
               make_engine(selector=SpySelector(EvidenceSelection("DOC-AUTH-001", "y" * 60))),
               make_engine(classifier=BrokenClassifier())]
    con = open_log(tmp_path / "a.sqlite")
    for i, e in enumerate(engines):
        for raw in (LOGIN, {"ticket_id": "B", "channel": "x"}):
            persist(con, e.process(raw), "run-t")
    rows = [json.loads(x[0]) for x in con.execute("select record_json from decision_log")]
    assert count_run(con, "run-t") == 8 and {r["action_taken"] for r in rows} == {"auto_respond", "block", "escalate"}
    for r in rows:
        assert all(f in r for f in GOVERNANCE_FIELDS) and isinstance(r["reason"], str) and r["reason"]
    con.close()
    con = open_log(tmp_path / "a.sqlite")        # survives a restart
    assert count_run(con, "run-t") == 8
    con.close()


def test_persist_rejects_incomplete_records(tmp_path):
    con = open_log(tmp_path / "b.sqlite")
    with pytest.raises(ValueError):
        persist(con, {"ticket_id": "x"}, "r")
    con.close()


def test_batch_continues_past_bad_rows_and_reconciles(make_engine, validation, tmp_path):
    items = validation[:5] + ["bad", {"ticket_id": "E", "channel": "email", "body": ""}] + validation[5:8]
    results, logged = run_batch(make_engine(), items, tmp_path / "c.sqlite", "run-x")
    assert len(results) == logged == len(items)


def test_no_restricted_ticket_is_auto_answered_on_validation(make_engine, validation):
    e = make_engine(auto_send=True)
    rs = {t["ticket_id"]: e.process(t) for t in validation}
    restricted = [t["ticket_id"] for t in validation if t["labels"]["must_not_auto_respond"]]
    assert restricted and all(rs[i]["action_taken"] != "auto_respond" for i in restricted)
    assert rs["VAL-0043"]["action_taken"] == "escalate" and rs["VAL-0009"]["action_taken"] == "escalate"


def test_same_ticket_same_decision_across_runs_a5(make_engine, validation):
    a = [make_engine().process(t) for t in validation]
    b = [make_engine().process(t) for t in validation]
    key = lambda r: (r["ticket_id"], r["action_taken"], r["reason_codes"], r["prediction"]["value"], r["response"])
    assert [key(r) for r in a] == [key(r) for r in b]


def test_langgraph_and_sequential_engines_agree(make_engine, validation):
    g, s = make_engine(), make_engine(use_langgraph=False)
    for t in validation:
        assert (g.process(t)["action_taken"], g.process(t)["reason_codes"]) == \
               (s.process(t)["action_taken"], s.process(t)["reason_codes"])


def test_decision_independent_of_ground_truth_labels(make_engine):
    e = make_engine()
    a = e.process({**LOGIN, "labels": {"intent": "x", "expected_route": "escalate"}})
    b = e.process({**LOGIN, "labels": {"intent": "y", "expected_route": "auto_respond"}})
    assert (a["action_taken"], a["prediction"]) == (b["action_taken"], b["prediction"])


def test_engineered_injection_ticket_is_blocked_end_to_end_via_cli(tmp_path):
    """A7 as the assessor runs it: submit the ticket through the documented command, open the log."""
    import sqlite3
    from cloudserve.cli import main
    main(["--input", "examples/guardrail_ticket.json", "--output", str(tmp_path / "g.json"),
          "--db", str(tmp_path / "g.sqlite"), "--auto-send"])
    row = sqlite3.connect(tmp_path / "g.sqlite").execute(
        "select action_taken, status, answer_sent, record_json from decision_log").fetchone()
    rec = json.loads(row[3])
    assert row[:3] == ("block", "blocked", 0)
    assert rec["guardrail_results"]["instruction_integrity"] == "fail"
    assert "Ignore all previous instructions" in rec["flagged_input_for_review"] and rec["response"] == ""


def test_held_answer_carries_a_handoff_for_the_agent(make_engine):
    r = make_engine(auto_send=False).process(LOGIN)
    assert r["status"] == "held_for_review" and r["handoff"]["sources"]


def test_llm_runs_are_marked_as_using_an_unvalidated_confidence_floor(make_engine):
    nb_rec = make_engine().process(LOGIN)
    other = make_engine(classifier=FixedClassifier()).process(LOGIN)
    assert nb_rec["thresholds"]["confidence_floor_validated_for_this_classifier"] is True
    assert other["thresholds"]["confidence_floor_validated_for_this_classifier"] is False
