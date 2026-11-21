"""Slice 8: runtime kill switch, LLM cache, tokens and cost, half-open breaker, 429 retries,
PII masking for hosted providers. Fakes and local sockets only (no external network)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import pytest
from conftest import LOGIN, FixedClassifier
from cloudserve import killswitch
from cloudserve.classify import LangChainClassifier
from cloudserve.evaluation import TICKETS_PER_MONTH, usage_metrics
from cloudserve.generate import LangChainAnswerGenerator
from cloudserve.graph import DecisionEngine
from cloudserve.model_calls import CallLog, LLMCache, cache_key, load_prices
from cloudserve.providers import ProviderCircuitOpen, ProviderGuard, StructuredResult, _unwrap

RES = "DOC-AUTH-001#resolution"
ANSWER = {"cannot_answer": False, "reason": "covered", "sentences": [
    {"text": "If the CLI is failing, run `cloudserve auth logout` followed by `cloudserve auth login`.",
     "citations": [RES], "quote": "run `cloudserve auth logout` followed by `cloudserve auth login`"}]}
CLASSIFIED = {"intent": "authentication_failure", "urgency": "medium", "confidence": 0.95,
              "alternatives": [], "reason": "login failure"}


class FakeModel:
    def __init__(self, output, usage=None):
        self.output, self.usage, self.calls = output, usage, []

    def invoke(self, messages):
        self.calls.append(messages)
        if isinstance(self.output, Exception):
            raise self.output
        return StructuredResult(self.output, self.usage) if self.usage else self.output


@pytest.fixture
def engine_with(retrieval, tmp_path):
    kb, policy, info = retrieval

    def _make(fake, cache_mode="read_write", provider=None, model_name="fake", auto_send=False, ks=None):
        calls = CallLog(LLMCache(cache_mode, tmp_path / "cache"))
        gen = LangChainAnswerGenerator(provider=provider, model=fake, model_name=model_name, calls=calls)
        return DecisionEngine(kb, policy, FixedClassifier(), auto_send=auto_send, retrieval_info=info,
                              answer_mode="generated", generator=gen,
                              kill_switch_file=ks or tmp_path / "KILL_SWITCH")
    return _make


# ---- 1. runtime kill switch ----
def test_kill_switch_engaged_mid_batch_holds_later_tickets_even_with_auto_send(make_engine, retrieval, nb, tmp_path):
    kb, policy, info = retrieval
    flag = tmp_path / "KILL_SWITCH"
    engine = DecisionEngine(kb, policy, nb, auto_send=True, retrieval_info=info, kill_switch_file=flag)
    before = engine.process(LOGIN)
    assert before["status"] == "auto_responded" and before["kill_switch"]["file_flag_engaged"] is False
    killswitch.engage("bad billing answers", flag)
    during = engine.process(LOGIN)
    assert during["status"] == "held_for_review" and not during["answer_sent"]
    assert during["action_taken"] == "auto_respond" and "kill switch was engaged" in during["reason"]
    assert during["kill_switch"]["file_flag_engaged"] and during["kill_switch"]["reason"] == "bad billing answers"
    assert before["status"] == "auto_responded"          # already-finalised decisions are unaffected
    killswitch.release(flag)
    assert engine.process(LOGIN)["status"] == "auto_responded"


def test_kill_switch_is_read_at_finalize_for_tickets_in_flight(retrieval, nb, tmp_path):
    kb, policy, info = retrieval
    flag = tmp_path / "KILL_SWITCH"

    class EngageWhileClassifying:          # the switch is thrown while this ticket is in flight
        name, version = "fixed", "test"

        def classify(self, ticket):
            killswitch.engage("in flight", flag)
            return FixedClassifier().classify(ticket)
    r = DecisionEngine(kb, policy, EngageWhileClassifying(), auto_send=True, retrieval_info=info,
                       kill_switch_file=flag).process(LOGIN)
    assert r["status"] == "held_for_review"


def test_any_file_at_the_flag_path_engages_the_switch(tmp_path):
    flag = tmp_path / "KILL_SWITCH"
    flag.write_text("")
    assert killswitch.state(flag)["engaged"]


def test_kill_switch_cli_on_status_off(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CLOUDSERVE_KILL_SWITCH_FILE", str(tmp_path / "KS"))
    killswitch.main(["on", "--reason", "incident 42"])
    assert json.loads(capsys.readouterr().out)["engaged"]
    killswitch.main(["status"])
    assert json.loads(capsys.readouterr().out)["reason"] == "incident 42"
    killswitch.main(["off"])
    assert json.loads(capsys.readouterr().out)["engaged"] is False


def test_records_never_carry_an_absolute_flag_path(tmp_path):
    assert "/" not in killswitch.state(tmp_path / "KS")["flag_file"]
    assert killswitch.state()["flag_file"] == "storage/KILL_SWITCH"


# ---- 2. LLM response cache ----
def test_second_run_makes_zero_model_calls_and_yields_identical_decisions(engine_with, validation):
    tickets = [LOGIN] + validation[:4]
    first_fake, second_fake = FakeModel(ANSWER), FakeModel(ANSWER)
    first = [engine_with(first_fake).process(t) for t in tickets]
    second = [engine_with(second_fake).process(t) for t in tickets]
    assert len(first_fake.calls) >= 1 and second_fake.calls == []
    key = lambda r: (r["ticket_id"], r["action_taken"], r["status"], r["response"], r["citations"], r["reason_codes"])
    assert [key(r) for r in first] == [key(r) for r in second]
    called = [r for r in second if r["model_calls"]]
    assert called and all(r["llm_cache"] == "hit" and r["usage"]["live_calls"] == 0 for r in called)
    assert all(r["llm_cache"] == "miss" for r in first if r["model_calls"])


def test_read_only_never_writes_and_off_never_reads(engine_with, tmp_path):
    engine_with(FakeModel(ANSWER), cache_mode="read_only").process(LOGIN)
    assert not (tmp_path / "cache").exists()
    engine_with(FakeModel(ANSWER)).process(LOGIN)                     # populate
    fake = FakeModel(ANSWER)
    r = engine_with(fake, cache_mode="off").process(LOGIN)
    assert len(fake.calls) == 1 and r["llm_cache"] == "off"


def test_cache_holds_no_secrets_and_no_ticket_text(engine_with, tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-secret-value")
    engine_with(FakeModel(ANSWER)).process(LOGIN)
    stored = "".join(p.read_text() for p in (tmp_path / "cache").rglob("*.json"))
    assert stored and "sk-ant-test-secret-value" not in stored and LOGIN["body"] not in stored


def test_cache_key_changes_with_prompt_version_model_and_messages():
    from langchain_core.messages import HumanMessage
    from cloudserve.generate import answer_schema
    schema, m = answer_schema(), [HumanMessage(content="a")]
    base = cache_key("ollama", "gemma4:e2b", "PR-02 v1.0", m, schema)
    assert base == cache_key("ollama", "gemma4:e2b", "PR-02 v1.0", m, schema)
    assert len({base, cache_key("ollama", "other", "PR-02 v1.0", m, schema),
                cache_key("ollama", "gemma4:e2b", "PR-02 v1.1", m, schema),
                cache_key("ollama", "gemma4:e2b", "PR-02 v1.0", [HumanMessage(content="b")], schema)}) == 4


def test_failed_calls_are_not_cached(engine_with):
    engine_with(FakeModel(TimeoutError("down"))).process(LOGIN)
    fake = FakeModel(ANSWER)
    engine_with(fake).process(LOGIN)
    assert len(fake.calls) == 1


# ---- 3. tokens and cost ----
def test_tokens_and_cost_are_recorded_per_decision(engine_with):
    fake = FakeModel(ANSWER, usage={"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200})
    r = engine_with(fake, provider="anthropic", model_name="claude-haiku-4-5").process(LOGIN)
    call = r["model_calls"][0]
    assert call["component"] == "generator" and call["prompt"] == "PR-02 v1.0" and call["cache"] == "miss"
    assert (call["input_tokens"], call["output_tokens"]) == (1000, 200)
    assert call["cost_usd"] == pytest.approx(1000 / 1e6 * 1 + 200 / 1e6 * 5)
    assert r["usage"] == {"model_calls": 1, "live_calls": 1, "cache_hits": 0, "input_tokens": 1000,
                          "output_tokens": 200, "estimated_cost_usd": 0.002, "cost_if_live_usd": 0.002}
    hit = engine_with(FakeModel(ANSWER), provider="anthropic", model_name="claude-haiku-4-5").process(LOGIN)
    assert hit["usage"]["estimated_cost_usd"] == 0 and hit["usage"]["cost_if_live_usd"] == 0.002


def test_local_providers_cost_zero_and_unknown_hosted_models_are_not_guessed(engine_with):
    usage = {"input_tokens": 500, "output_tokens": 50}
    local = engine_with(FakeModel(ANSWER, usage), provider="ollama", model_name="gemma4:e2b").process(LOGIN)
    assert local["usage"]["estimated_cost_usd"] == 0
    unknown = engine_with(FakeModel(ANSWER, usage), provider="openai", model_name="gpt-x").process(LOGIN)
    assert unknown["model_calls"][0]["price_known"] is False and unknown["usage"]["estimated_cost_usd"] is None


def test_usage_metrics_project_monthly_cost_with_a_stated_volume(engine_with):
    fake = FakeModel(ANSWER, usage={"input_tokens": 1000, "output_tokens": 200})
    rows = [engine_with(fake, provider="anthropic", model_name="claude-haiku-4-5", cache_mode="off").process(LOGIN)
            for _ in range(2)]
    m = usage_metrics(rows, "2026-11-22")
    assert m["live_calls"] == 2 and m["estimated_cost_per_ticket_if_live_usd"] == 0.002
    proj = m["projected_monthly_cost_usd"]
    assert proj["tickets_per_month"] == TICKETS_PER_MONTH == 2167
    assert "five hundred tickets a week" in proj["volume_source"] and proj["cost_usd"] == round(0.002 * 2167, 4)


def test_price_table_has_source_and_date():
    p = load_prices()
    assert p["as_of"] and "source" in p and p["models"]["anthropic:claude-haiku-4-5"] == {
        "input_per_million": 1.0, "output_per_million": 5.0}


def test_include_raw_output_is_unwrapped_with_usage():
    from langchain_core.messages import AIMessage
    raw = AIMessage(content="", usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10})
    out = _unwrap({"raw": raw, "parsed": {"x": 1}, "parsing_error": None})
    assert out.parsed == {"x": 1} and out.usage["input_tokens"] == 7
    with pytest.raises(ValueError):
        _unwrap({"raw": raw, "parsed": None, "parsing_error": ValueError("bad json")})
    with pytest.raises(Exception, match="no structured output"):
        _unwrap({"raw": raw, "parsed": None, "parsing_error": None})


# ---- 4. half-open circuit breaker ----
class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _fail():
    raise ConnectionError("down")


def test_breaker_half_opens_after_cooldown_and_closes_on_success():
    clock = Clock()
    g = ProviderGuard(max_consecutive_failures=2, deadline=5, cooldown=30, clock=clock)
    for _ in range(2):
        with pytest.raises(ConnectionError):
            g.call(_fail)
    assert g.state == "open"
    clock.t = 29.9
    with pytest.raises(ProviderCircuitOpen):
        g.call(lambda: "ok")                     # still cooling down: provider not called
    clock.t = 30.0
    assert g.call(lambda: "ok") == "ok" and g.state == "closed" and g.trials == 1


def test_breaker_reopens_when_the_trial_call_fails():
    clock = Clock()
    g = ProviderGuard(max_consecutive_failures=2, deadline=5, cooldown=30, clock=clock)
    for _ in range(2):
        with pytest.raises(ConnectionError):
            g.call(_fail)
    clock.t = 31
    with pytest.raises(ConnectionError):
        g.call(_fail)                            # the single trial fails
    assert g.state == "open" and g.opened_at == 31
    calls = []
    clock.t = 40
    with pytest.raises(ProviderCircuitOpen):
        g.call(lambda: calls.append(1))
    assert calls == []                           # a new cooldown started from the failed trial


def test_breaker_cooldown_is_configurable(monkeypatch):
    monkeypatch.setenv("CLOUDSERVE_BREAKER_COOLDOWN_SECONDS", "5")
    assert ProviderGuard().cooldown == 5


# ---- 5. rate limits (429) from a local OpenAI-compatible server (llamacpp provider) ----
def _server(fail_times):
    state = {"requests": 0}
    body = json.dumps({"id": "x", "object": "chat.completion", "created": 0, "model": "local-model",
                       "choices": [{"index": 0, "finish_reason": "stop",
                                    "message": {"role": "assistant", "content": json.dumps(CLASSIFIED)}}],
                       "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}}).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length", 0)))
            state["requests"] += 1
            if state["requests"] <= fail_times:
                self.send_response(429)
                self.send_header("retry-after-ms", "10")
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error": {"message": "rate limited", "type": "rate_limit"}}')
                return
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, state


@pytest.fixture
def no_proxy(monkeypatch):
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("CLOUDSERVE_MAX_RETRIES", "2")
    monkeypatch.setenv("CLOUDSERVE_TIMEOUT_SECONDS", "5")


def test_429_is_retried_then_succeeds(no_proxy):
    srv, state = _server(fail_times=2)
    try:
        clf = LangChainClassifier(provider="llamacpp", base_url=f"http://127.0.0.1:{srv.server_port}/v1",
                                  calls=CallLog(LLMCache("off")))
        out = clf.classify(type("T", (), {"ticket_id": "T", "channel": "email", "subject": "s", "body": "b"})())
    finally:
        srv.shutdown()
    assert out["intent"] == "authentication_failure" and state["requests"] == 3      # 2 retries, then success
    usage = clf.calls.drain()[0]
    assert (usage["input_tokens"], usage["output_tokens"]) == (120, 30)


def test_persistent_429_escalates_cleanly_with_bounded_retries(no_proxy, retrieval):
    kb, policy, info = retrieval
    srv, state = _server(fail_times=10 ** 6)
    try:
        clf = LangChainClassifier(provider="llamacpp", base_url=f"http://127.0.0.1:{srv.server_port}/v1")
        r = DecisionEngine(kb, policy, clf, retrieval_info=info).process(LOGIN)
    finally:
        srv.shutdown()
    assert r["action_taken"] == "escalate" and "RateLimitError" in r["prediction"]["error_type"]
    assert state["requests"] == 3                # max_retries=2: one try plus two retries, never more


# ---- 6. PII masking before hosted providers ----
PII_TICKET = {**LOGIN, "body": LOGIN["body"] + " Reach me at jane.doe@example.com or +44 20 7946 0958."}


def _ticket():
    from cloudserve.ingest import normalize
    return normalize(PII_TICKET)


@pytest.mark.parametrize("provider,masked", [("anthropic", True), ("openai", True), ("ollama", False), ("llamacpp", False)])
def test_ticket_pii_is_masked_only_for_hosted_providers(provider, masked):
    clf = LangChainClassifier(provider=provider, model=FakeModel(CLASSIFIED))
    gen = LangChainAnswerGenerator(provider=provider, model=FakeModel(ANSWER))
    for msgs in (clf.messages(_ticket()), gen.messages(_ticket(), [{"passage_id": RES, "title": "t", "passage": "p"}])):
        text = " ".join(m.content for m in msgs)
        assert ("jane.doe@example.com" in text) is not masked


def test_record_states_pii_masked_for_provider(engine_with):
    fake = FakeModel(ANSWER)
    r = engine_with(fake, provider="anthropic", model_name="claude-haiku-4-5").process(PII_TICKET)
    assert r["pii_masked_for_provider"] is True
    assert "jane.doe@example.com" not in " ".join(m.content for m in fake.calls[0])
    local = engine_with(FakeModel(ANSWER), provider="ollama", model_name="gemma4:e2b", cache_mode="off").process(PII_TICKET)
    assert local["pii_masked_for_provider"] is False


def test_offline_records_have_no_model_calls(make_engine):
    r = make_engine().process(LOGIN)
    assert r["model_calls"] == [] and r["llm_cache"] == "not_used" and r["pii_masked_for_provider"] is None


def test_cache_defaults_off_for_the_cli_and_read_write_for_evaluation():
    import argparse
    from cloudserve.runtime import add_engine_arguments
    cli, ev = argparse.ArgumentParser(), argparse.ArgumentParser()
    add_engine_arguments(cli)
    add_engine_arguments(ev, llm_cache_default="read_write")
    assert cli.parse_args([]).llm_cache == "off" and ev.parse_args([]).llm_cache == "read_write"
    src = (__import__("conftest").ROOT / "src/cloudserve/evaluation.py").read_text()
    assert 'add_engine_arguments(p, llm_cache_default="read_write")' in src
