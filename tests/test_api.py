"""Slice 9: HTTP API, Prometheus metrics and concurrency safety (TestClient, threads, local only)."""
import json
import os
import sqlite3
import threading
import time
import pytest
from fastapi.testclient import TestClient
from conftest import LOGIN, ROOT
from cloudserve import killswitch
from cloudserve.api import create_app, parse_args
from cloudserve.graph import DecisionEngine
from cloudserve.logging_store import GOVERNANCE_FIELDS, count_run, open_log, persist_one
from cloudserve.metrics import Metrics
from cloudserve.providers import ProviderCircuitOpen, ProviderGuard

INJECTION = json.loads((ROOT / "examples/guardrail_ticket.json").read_text())
INJECTION = INJECTION[0] if isinstance(INJECTION, list) else INJECTION


@pytest.fixture
def api(retrieval, nb, tmp_path):
    kb, policy, info = retrieval

    def _make(auto_send=False):
        engine = DecisionEngine(kb, policy, nb, auto_send=auto_send, retrieval_info=info,
                                kill_switch_file=tmp_path / "KILL_SWITCH")
        db = tmp_path / "decisions.sqlite"
        return TestClient(create_app(engine, db)), db, engine
    return _make


def metric(text: str, name: str, **labels) -> float:
    want = ",".join(f'{k}="{v}"' for k, v in labels.items())
    for line in text.splitlines():
        if line.startswith(name + ("{" + want + "}" if want else " ")) or (want and line.startswith(name + "{") and all(
                f'{k}="{v}"' in line for k, v in labels.items())):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


def test_post_ticket_returns_a_governance_complete_record_and_logs_it(api):
    client, db, _ = api()
    r = client.post("/tickets", json=LOGIN)
    assert r.status_code == 200
    rec = r.json()
    assert all(f in rec for f in GOVERNANCE_FIELDS)
    assert rec["action_taken"] == "auto_respond" and rec["status"] == "held_for_review"   # auto-send off by default
    con = sqlite3.connect(db)
    row = con.execute("SELECT ticket_id, action_taken, run_id FROM decision_log WHERE decision_id = ?",
                      (rec["decision_id"],)).fetchone()
    assert row == ("T-LOGIN", "auto_respond", rec["run_id"]) and rec["run_id"].startswith("api-")


def test_auto_send_releases_and_kill_switch_overrides_via_api(api, tmp_path):
    client, _, _ = api(auto_send=True)
    assert client.post("/tickets", json=LOGIN).json()["status"] == "auto_responded"
    killswitch.engage("api test", tmp_path / "KILL_SWITCH")
    held = client.post("/tickets", json=LOGIN).json()
    assert held["status"] == "held_for_review" and "kill switch was engaged" in held["reason"]
    h = client.get("/health").json()
    assert h["kill_switch"]["engaged"] and h["answers_released"] is False and h["auto_send_enabled"]


def test_health_reports_state_without_secrets_env_values_or_absolute_paths(api, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-health-secret")
    client, _, _ = api()
    client.post("/tickets", json=LOGIN)
    h = client.get("/health")
    body = h.text + client.get("/metrics").text
    assert h.json()["kill_switch"] == {"engaged": False, "flag_file": "KILL_SWITCH", "engaged_at": None}
    assert h.json()["policy_version"] == "RP-1.0" and h.json()["breaker"] is None
    assert "sk-ant-health-secret" not in body and str(ROOT) not in body and "/Users/" not in body
    assert os.path.expanduser("~") not in body


def test_metrics_expose_counters_that_increment(api):
    client, _, _ = api()
    before = client.get("/metrics").text
    assert metric(before, "tickets_processed_total", channel="email", outcome="auto_respond") == 0
    client.post("/tickets", json=LOGIN)
    client.post("/tickets", json=INJECTION)
    after = client.get("/metrics").text
    assert metric(after, "tickets_processed_total", channel="email", outcome="auto_respond") == 1
    assert metric(after, "tickets_processed_total", channel=INJECTION["channel"], outcome="block") == 1
    assert metric(after, "guardrail_blocks_total", guardrail="untrusted_instruction_in_ticket") == 1
    assert metric(after, "response_seconds_count") == 2 and metric(after, "classification_confidence_count") == 2
    for name in ("kill_switch_engaged", "breaker_open", "llm_cost_usd_total", "answers_total"):
        assert name in after
    assert metric(after, "kill_switch_engaged") == 0


def test_non_object_body_is_rejected_with_422_and_not_logged(api):
    client, db, _ = api()
    assert client.post("/tickets", content=b"not json", headers={"content-type": "application/json"}).status_code == 422
    assert client.post("/tickets", json=[LOGIN]).status_code == 422
    assert not db.exists() or sqlite3.connect(db).execute("SELECT COUNT(*) FROM decision_log").fetchone()[0] == 0


def test_invalid_ticket_object_is_a_logged_escalation_not_a_crash(api):
    client, db, _ = api()
    r = client.post("/tickets", json={"ticket_id": "BAD", "channel": "fax", "body": ""})
    assert r.status_code == 200 and r.json()["action_taken"] == "escalate"
    assert sqlite3.connect(db).execute("SELECT COUNT(*) FROM decision_log WHERE ticket_id='BAD'").fetchone()[0] == 1


def test_concurrent_requests_are_all_logged(api):
    client, db, _ = api()
    errors = []

    def send(i):
        try:
            assert client.post("/tickets", json={**LOGIN, "ticket_id": f"C-{i}"}).status_code == 200
        except Exception as exc:          # surfaced below
            errors.append(exc)
    threads = [threading.Thread(target=send, args=(i,)) for i in range(16)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    assert sqlite3.connect(db).execute("SELECT COUNT(DISTINCT ticket_id) FROM decision_log").fetchone()[0] == 16


def test_concurrent_writes_to_the_decision_log_lose_nothing(tmp_path, make_engine):
    rec = make_engine().process(LOGIN)
    db = tmp_path / "log.sqlite"

    def write(i):
        persist_one(db, {**rec, "decision_id": f"d-{i}"}, "run-x")
    threads = [threading.Thread(target=write, args=(i,)) for i in range(40)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert count_run(open_log(db), "run-x") == 40


def test_half_open_breaker_admits_exactly_one_trial_call_across_threads():
    clock = {"t": 0.0}
    g = ProviderGuard(max_consecutive_failures=1, deadline=5, cooldown=10, clock=lambda: clock["t"])
    with pytest.raises(ConnectionError):
        g.call(lambda: (_ for _ in ()).throw(ConnectionError("down")))
    clock["t"] = 11
    release, entered, results = threading.Event(), [], []

    def slow_trial():
        entered.append(1)
        release.wait(5)
        return "ok"

    def caller():
        try:
            results.append(g.call(slow_trial))
        except ProviderCircuitOpen:
            results.append("fast-fail")
    threads = [threading.Thread(target=caller) for _ in range(8)]
    [t.start() for t in threads]
    deadline = time.time() + 5
    while len(results) < 7 and time.time() < deadline:      # 7 fail fast while the trial is in flight
        time.sleep(0.01)
    assert results.count("fast-fail") == 7 and len(entered) == 1 and g.state == "half_open"
    release.set()
    [t.join() for t in threads]
    assert results.count("ok") == 1 and g.state == "closed" and g.trials == 1


def test_metrics_record_provider_failures_and_breaker_state(make_engine):
    class Broken:
        name, version = "ollama:gemma4:e2b", "PR-01 v1.1"
        guard = ProviderGuard(max_consecutive_failures=1, deadline=5)

        def classify(self, ticket):
            return self.guard.call(lambda: (_ for _ in ()).throw(TimeoutError("slow")))
    engine = make_engine(classifier=Broken())
    m = Metrics(engine)
    m.observe(engine.process(LOGIN))
    text = m.exposition().decode()
    assert metric(text, "provider_failures_total", provider="ollama", error="TimeoutError") == 1
    assert metric(text, "breaker_open") == 1


def test_api_binds_localhost_by_default_and_reads_engine_options_from_env(monkeypatch):
    monkeypatch.setenv("CLOUDSERVE_ANSWER_MODE", "generated")
    monkeypatch.setenv("CLOUDSERVE_CLASSIFIER", "llm")
    a = parse_args([])
    assert a.host == "127.0.0.1" and a.port == 8000 and a.answer_mode == "generated" and a.classifier == "llm"
    assert a.llm_cache == "off" and a.auto_send is False
    assert parse_args(["--classifier", "offline"]).classifier == "offline"     # flags still win


def test_harness_metrics_port_serves_live_metrics(tmp_path):
    import socket
    import urllib.request
    from cloudserve.evaluation import main
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    main(["--input", str(ROOT / "examples/four_channels.json"), "--output", str(tmp_path / "out"),
          "--metrics-port", str(port)])
    text = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5).read().decode()
    assert sum(metric(text, "tickets_processed_total", channel=c, outcome=o)
               for c in ("email", "chat", "docs_comment", "forum") for o in ("auto_respond", "escalate", "block")) == 4



def test_dashboard_has_the_setup_guide_panels_and_uses_only_exposed_metrics(make_engine):
    import re
    dash = json.loads((ROOT / "monitoring/grafana/dashboards/cloudserve.json").read_text())
    titles = [p["title"] for p in dash["panels"]]
    assert titles == ["Tickets per hour by channel", "Tickets per hour by outcome", "Answered automatically vs escalated",
                      "Response latency p50 / p95", "Guardrail activations by guardrail", "Classification confidence distribution"]
    exposed = Metrics(make_engine()).exposition().decode()
    for p in dash["panels"]:
        for t in p["targets"]:
            for name in re.findall(r"\b([a-z_]+(?:_total|_bucket))\b", t["expr"]):
                assert f"# TYPE {re.sub(r'_(total|bucket)$', '', name)}" in exposed, name
    compose = (ROOT / "monitoring/docker-compose.yml").read_text()
    assert "${GRAFANA_ADMIN_PASSWORD:-admin}" in compose and '"127.0.0.1:3000:3000"' in compose
    assert "host.docker.internal:8000" in (ROOT / "monitoring/prometheus.yml").read_text()
