"""Prometheus metrics (Setup Guide §06), updated from each finished decision record.

Names follow the Setup Guide where it gives them (tickets_processed_total{channel,outcome},
response_seconds, guardrail_blocks_total{guardrail}); the rest serve the dashboard and the
risks (R-06 provider failures, R-08 cost, Governance §5 kill switch). Each `Metrics` owns its
own registry, so the API and the harness (`--metrics-port`) expose only their own series.
Label values are bounded (known channels, actions and codes), never ticket text.
"""
from __future__ import annotations
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest, start_http_server
from prometheus_client import CONTENT_TYPE_LATEST  # noqa: F401  (re-exported for the API)
from . import killswitch
from .ingest import CHANNELS

OUTCOMES = ("auto_respond", "escalate", "block")
CONFIDENCE_BUCKETS = tuple(round(i / 10, 1) for i in range(1, 11))
LATENCY_BUCKETS = (0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60)
FAILURE_STAGES = {"evidence_selection_failed": "evidence_selector", "generation_failed": "generator"}


class Metrics:
    def __init__(self, engine=None, registry: CollectorRegistry | None = None):
        r = self.registry = registry or CollectorRegistry()
        self.tickets = Counter("tickets_processed_total", "Tickets processed", ["channel", "outcome"], registry=r)
        self.answers = Counter("answers_total", "Auto-respond decisions by delivery (sent or held)", ["delivery"], registry=r)
        self.latency = Histogram("response_seconds", "End to end decision time (live decisions; LLM cache replays excluded)",
                                 buckets=LATENCY_BUCKETS, registry=r)
        self.blocks = Counter("guardrail_blocks_total", "Responses blocked", ["guardrail"], registry=r)
        self.activations = Counter("guardrail_activations_total", "Guardrail checks that failed (block or escalate)",
                                   ["guardrail", "action"], registry=r)
        self.confidence = Histogram("classification_confidence", "Classifier confidence (drift shows here first)",
                                    buckets=CONFIDENCE_BUCKETS, registry=r)
        self.provider_failures = Counter("provider_failures_total", "Model calls that failed", ["provider", "error"], registry=r)
        self.cost = Counter("llm_cost_usd_total", "Estimated model spend in USD (config/model_prices.json)", registry=r)
        self.cache_hits = Counter("llm_cache_hits_total", "Model calls answered from the LLM cache", registry=r)
        self.kill_switch = Gauge("kill_switch_engaged", "1 while the runtime kill switch flag file is present", registry=r)
        self.breaker = Gauge("breaker_open", "1 while the provider circuit breaker is open or half-open", registry=r)
        path = getattr(engine, "kill_switch_file", None)
        self.kill_switch.set_function(lambda: float(killswitch.state(path)["engaged"]))
        guard = _guard_of(engine)
        self.breaker.set_function(lambda: float(guard is not None and guard.state != "closed"))
        for c in sorted(CHANNELS) + ["other"]:          # pre-create so rates start at zero
            for o in OUTCOMES:
                self.tickets.labels(channel=c, outcome=o)

    def observe(self, record: dict) -> None:
        channel = record.get("channel") if record.get("channel") in CHANNELS else "other"
        action = record.get("action_taken") if record.get("action_taken") in OUTCOMES else "escalate"
        self.tickets.labels(channel=channel, outcome=action).inc()
        if action == "auto_respond":
            self.answers.labels(delivery=record.get("delivery", "unknown")).inc()
        if record.get("llm_cache") not in ("hit", "mixed"):
            self.latency.observe(float(record.get("latency_seconds") or 0.0))
        for g in record.get("guardrail_failures", []):
            self.activations.labels(guardrail=g, action=action).inc()
            if action == "block":
                self.blocks.labels(guardrail=g).inc()
        pred = record.get("prediction") or {}
        if pred.get("confidence_type") not in ("model_failed", "not_classified") and isinstance(pred.get("confidence"), (int, float)):
            self.confidence.observe(float(pred["confidence"]))
        models = record.get("model") or {}
        if pred.get("confidence_type") == "model_failed":
            self.provider_failures.labels(provider=_provider(models.get("classifier")),
                                          error=str(pred.get("error_type") or "unknown")[:60]).inc()
        codes = record.get("reason_codes") or []
        for i, code in enumerate(codes):
            if code in FAILURE_STAGES:
                err = codes[i + 1] if i + 1 < len(codes) else "unknown"
                self.provider_failures.labels(provider=_provider(models.get(FAILURE_STAGES[code])), error=str(err)[:60]).inc()
        usage = record.get("usage") or {}
        if isinstance(usage.get("estimated_cost_usd"), (int, float)):
            self.cost.inc(usage["estimated_cost_usd"])
        if usage.get("cache_hits"):
            self.cache_hits.inc(usage["cache_hits"])

    def exposition(self) -> bytes:
        return generate_latest(self.registry)

    def serve(self, port: int, addr: str = "127.0.0.1"):
        """Expose /metrics on its own port (the unattended harness, `--metrics-port`)."""
        return start_http_server(port, addr=addr, registry=self.registry)


def _provider(model: dict | None) -> str:
    name = (model or {}).get("name", "")
    return name.split(":", 1)[0] if ":" in name else "unknown"


def _guard_of(engine):
    for a in (getattr(engine, "classifier", None), getattr(engine, "selector", None), getattr(engine, "generator", None)):
        if getattr(a, "guard", None) is not None:
            return a.guard
    return None
