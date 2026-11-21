"""Offline intent classifier: multinomial naive Bayes over ticket words (standard library only).

Trained ONLY on development tickets (data/development_tickets.json, which the Dataset Guide
permits for "developing your classifier"). It never reads validation or hidden-set labels:
`predict` receives a normalised Ticket, from which labels were stripped at ingestion.

The posterior is reported as `confidence`. It is a model probability, not a calibrated one;
calibration is measured by the evaluation harness (confidence bands vs observed accuracy).
"""
from __future__ import annotations
import hashlib
import math
from collections import Counter, defaultdict
from pathlib import Path
from .pipeline import Ticket, load_json, tokens

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
# urgency labels; see docs/ARCHITECTURE.md §8). A transparent keyword rule is used and routing
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
