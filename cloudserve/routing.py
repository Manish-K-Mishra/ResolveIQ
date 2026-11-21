"""Deterministic routing policy (FR-04, A5). Pure function: same inputs, same decision.

The policy file (config/routing_policy.json) is produced by scripts/derive_routing_policy.py
from DEVELOPMENT data only, and its version is written into every decision record.
"""
from __future__ import annotations
import json
from pathlib import Path
from .classifier import UNKNOWN
from .pipeline import RISK_PATTERN, Ticket, contains_injection

DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[1] / "config" / "routing_policy.json"

# Plain-language reasons a support manager can read (Build Spec §03 Route).
REASON_TEXT = {
    "untrusted_instruction_in_ticket": "The ticket contains text that tries to instruct the system, so a person must review it.",
    "sensitive_request_manual_review": "The ticket mentions a legal, breach, exposed-credential or refund matter, which only staff may handle.",
    "restricted_intent": "This type of request ({intent}) is never answered automatically by policy.",
    "intent_unknown": "The system could not tell what the customer needs.",
    "confidence_below_threshold": "The system was only {confidence:.0%} sure of the request type; the policy requires at least {threshold:.0%}.",
    "classification_model_failed": "The classification model failed or was unavailable, so the ticket was escalated without a prediction.",
    "no_relevant_documentation": "No documentation article was relevant enough to answer from.",
    "no_matching_documentation_for_intent": "The documentation found does not cover this type of request (best match {doc_id} at {score:.2f}; needs one of the approved articles at {threshold:.2f} or more).",
    "eligible_for_automatic_answer": "Known low-risk request type, confident classification and an approved matching article.",
}


def load_policy(path: str | Path | None = None) -> dict:
    policy = json.loads(Path(path or DEFAULT_POLICY_PATH).read_text(encoding="utf-8"))
    for key in ("policy_version", "restricted_intents", "intent_documents",
                "classification_confidence_min", "retrieval_score_min"):
        if key not in policy:
            raise ValueError(f"routing policy missing {key!r}")
    return policy


def route(ticket: Ticket, classification: dict, matches: list[dict], policy: dict) -> dict:
    """Return {'route': 'answer'|'escalate', 'codes', 'reason', 'evidence', thresholds}.

    Order matters only for the reason text; every failing rule is recorded. A missing
    confidence is treated as zero (Governance §4: a missing score is not a high one).
    """
    conf_min = float(policy["classification_confidence_min"])
    ret_min = float(policy["retrieval_score_min"])
    intent = classification.get("intent") or UNKNOWN
    confidence = classification.get("confidence")
    confidence = float(confidence) if isinstance(confidence, (int, float)) else 0.0
    codes, details = [], {}

    if contains_injection(ticket.text):
        codes.append("untrusted_instruction_in_ticket")
    if RISK_PATTERN.search(ticket.text):
        codes.append("sensitive_request_manual_review")
    if classification.get("confidence_type") == "model_failed":
        codes.append("classification_model_failed")
    elif intent == UNKNOWN:
        codes.append("intent_unknown")
    elif intent in policy["restricted_intents"]:
        codes.append("restricted_intent")
    if "classification_model_failed" not in codes and confidence < conf_min:
        codes.append("confidence_below_threshold")

    approved_docs = set(policy["intent_documents"].get(intent, []))
    evidence = [m for m in matches if m["doc_id"] in approved_docs and m["score"] >= ret_min]
    if not matches:
        codes.append("no_relevant_documentation")
    elif not evidence:
        codes.append("no_matching_documentation_for_intent")
        details.update(doc_id=matches[0]["doc_id"], score=matches[0]["score"])

    decision = "escalate" if codes else "answer"
    codes = codes or ["eligible_for_automatic_answer"]
    fmt = {"intent": intent, "confidence": confidence, "doc_id": "-", "score": 0.0, **details}
    reason = " ".join(REASON_TEXT[c].format(**fmt, threshold=ret_min if c == "no_matching_documentation_for_intent" else conf_min)
                      for c in codes)
    return {"route": decision, "codes": codes, "reason": reason, "evidence": evidence if decision == "answer" else [],
            "threshold_applied": {"classification_confidence_min": conf_min, "retrieval_score_min": ret_min},
            "policy_version": policy["policy_version"]}
