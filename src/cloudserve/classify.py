"""Intent and urgency classification with confidence (FR-02).

Two classifiers share one output contract: an offline naive Bayes over word presence (document-frequency likelihoods, present words only; the score is a relative posterior, not a calibrated probability)
(the default, no network) and PR-01 through any LLM provider (LangChainClassifier).

The offline classifier is trained ONLY on development tickets (data/development_tickets.json,
which the Dataset Guide permits for "developing your classifier"). It never reads validation or
hidden-set labels: `classify` receives a normalised Ticket, from which labels were stripped at ingestion.

The posterior is reported as `confidence`. It is a model probability, not a calibrated one;
calibration is measured by the evaluation harness (confidence bands vs observed accuracy).
"""
from __future__ import annotations
import hashlib
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from .ingest import Ticket, load_json, tokens
from .providers import ProviderGuard, build_model, for_provider, load_prompt, structured

INTENTS = (
    "account_access", "api_key_issue", "api_usage_question", "authentication_failure",
    "billing_query", "compliance_request", "configuration_help", "data_export",
    "data_residency", "database_issue", "deployment_failure", "feature_request",
    "integration_help", "onboarding", "performance_degradation", "quota_or_overage",
    "rate_limit", "rollback_request", "security_incident", "sso_configuration",
    "unclear_request", "webhook_issue",
)
UNKNOWN = "unknown"
URGENCIES = ("low", "medium", "high")
# Urgency is NOT learnable from the text in this dataset (identical bodies carry different
# urgency labels; see docs/architecture.md §8). A transparent keyword rule is used and routing
# deliberately does not depend on it.
_HIGH_URGENCY = ("urgent", "asap", "immediately", "production down", "outage", "cannot access",
                 "security breach", "all customers", "data loss", "right now")


def urgency_rule(text: str) -> str:
    t = text.lower()
    return "high" if any(w in t for w in _HIGH_URGENCY) else "medium"


class NaiveBayesIntentClassifier:
    name = "naive_bayes_intent"
    ALPHA = 0.5

    def __init__(self, examples: list[tuple[str, str]]):
        self.class_counts: Counter = Counter()
        self.word_counts: dict[str, Counter] = defaultdict(Counter)
        self.vocab: set[str] = set()
        for text, intent in examples:
            if intent not in INTENTS:
                continue
            self.class_counts[intent] += 1
            for w in tokens(text):
                self.word_counts[intent][w] += 1
                self.vocab.add(w)
        if not self.class_counts:
            raise ValueError("no training examples with a known intent")
        digest = hashlib.sha256(repr(sorted((t, i) for t, i in examples)).encode()).hexdigest()[:12]
        self.version = f"nb-{digest}"   # identifies the exact training data in audit records

    @classmethod
    def from_development_file(cls, path: str | Path):
        rows = load_json(path)
        return cls([(f"{r.get('subject') or ''} {r.get('body') or ''}", r["labels"]["intent"])
                    for r in rows if isinstance(r, dict) and isinstance(r.get("labels"), dict)])

    def posterior(self, text: str) -> list[tuple[str, float]]:
        feats = tokens(text) & self.vocab
        if not feats:
            return []
        total = sum(self.class_counts.values())
        logp = {}
        for y, n in self.class_counts.items():
            s = math.log(n / total)
            for w in feats:
                s += math.log((self.word_counts[y][w] + self.ALPHA) / (n + 2 * self.ALPHA))
            logp[y] = s
        top = max(logp.values())
        z = sum(math.exp(v - top) for v in logp.values())
        return sorted(((y, math.exp(v - top) / z) for y, v in logp.items()), key=lambda p: (-p[1], p[0]))

    def classify(self, ticket: Ticket) -> dict:
        ranked = self.posterior(ticket.text)
        if not ranked:
            return {"intent": UNKNOWN, "urgency": urgency_rule(ticket.text), "confidence": 0.0,
                    "alternatives": [], "reason": "No known vocabulary in the ticket.",
                    "confidence_type": "naive_bayes_posterior"}
        (intent, p), rest = ranked[0], ranked[1:4]
        return {"intent": intent, "urgency": urgency_rule(ticket.text), "confidence": round(p, 4),
                "alternatives": [{"intent": y, "confidence": round(q, 4)} for y, q in rest],
                "reason": "Highest posterior among the 22 allowed intents.",
                "confidence_type": "naive_bayes_posterior"}


IntentLabel = Literal[INTENTS + (UNKNOWN,)]


class Classification(BaseModel):
    """PR-01 v1.1 output contract: intent restricted to the 22 dataset intents or 'unknown'."""
    model_config = ConfigDict(extra="forbid")
    intent: IntentLabel
    urgency: Literal["low", "medium", "high"]
    confidence: float = Field(ge=0, le=1)
    alternatives: list[IntentLabel] = Field(default_factory=list, max_length=3)
    reason: str = Field(min_length=1, max_length=400)


class LangChainClassifier:
    """PR-01 classification through any provider; same contract as the offline classifier."""

    def __init__(self, provider: str | None = None, model_name: str | None = None,
                 base_url: str | None = None, model=None, guard: ProviderGuard | None = None, calls=None):
        self.provider = provider or os.getenv("CLOUDSERVE_PROVIDER", "ollama")
        self.schema, self.calls = Classification, calls
        self.prompt = load_prompt("PR-01_classification.md")
        self.guard = guard or ProviderGuard()
        if model is not None:                       # injected (tests): used as-is
            self.runnable, self.model_name = model, model_name or "injected"
        else:
            chat, self.model_name, extra = build_model(self.provider, model_name, base_url)
            self.runnable = structured(chat, Classification, self.provider, extra)
        self.name = f"{self.provider}:{self.model_name}"
        self.version = self.prompt["label"]

    def messages(self, ticket) -> list:
        from langchain_core.messages import HumanMessage, SystemMessage
        system = self.prompt["system"].replace("{allowed_intents}", ", ".join(INTENTS + (UNKNOWN,)))
        values = {"ticket_id": for_provider(self.provider, str(ticket.ticket_id)[:64]),
                  "channel": for_provider(self.provider, str(ticket.channel)[:32]),
                  "subject": for_provider(self.provider, ticket.subject[:300]),
                  "body": for_provider(self.provider, ticket.body[:4000])}
        # Single pass, so placeholder-like text inside the ticket is never substituted.
        user = re.sub(r"\{(ticket_id|channel|subject|body)\}", lambda m: values[m.group(1)], self.prompt["user"])
        return [SystemMessage(content=system), HumanMessage(content=user)]

    def classify(self, ticket) -> dict:
        from .model_calls import call_model
        v = call_model(self, "classifier", self.messages(ticket))
        alts = [a for a in dict.fromkeys(v.alternatives) if a != v.intent]
        return {"intent": v.intent, "urgency": v.urgency, "confidence": v.confidence,
                # The model names alternatives but gives no per-alternative score.
                "alternatives": [{"intent": a, "confidence": None} for a in alts],
                "reason": v.reason, "confidence_type": "uncalibrated_model_score"}
