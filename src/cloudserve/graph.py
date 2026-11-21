"""The CloudServe decision workflow, orchestrated with LangGraph.

    ingest -> classify -> retrieve -> route --(answer)--> select_evidence -> validate -> finalize
                                          `--(escalate)-----------------------------------^

Critical decisions are deterministic Python (route.route, guardrails.validate_response);
models only classify and select quotes behind adapters. Every path, including failures,
ends in `finalize`, which builds one governance-schema decision record.

Actions (Governance §1): auto_respond | escalate | block.
Kill switch (Governance §5): when automatic sending is disabled (the default), an
auto_respond decision is HELD for human review (status 'held_for_review', answer_sent false).
"""
from __future__ import annotations
import os
import time
import uuid
from typing import Any, TypedDict
from .classify import UNKNOWN
from .generate import select_from_document
from .guardrails import mask_pii, validate_response
from .ingest import normalize, utc_now
from .retrieve import KnowledgeBase
from .route import route as route_policy

BASE_REQUIREMENTS = ["FR-01", "FR-02", "FR-03", "FR-04", "FR-07"]


def auto_send_enabled_from_env() -> bool:
    return os.getenv("CLOUDSERVE_AUTO_SEND", "false").strip().lower() in ("1", "true", "yes", "on")


class FlowState(TypedDict, total=False):
    raw: dict
    ticket: Any
    classification: dict
    matches: list[dict]
    routing: dict
    selection: Any
    selection_errors: list[str]
    validation: dict
    record: dict


class DecisionEngine:
    """Holds the run's dependencies; `process(raw)` never raises and always returns a record."""

    def __init__(self, kb: KnowledgeBase, policy: dict, classifier, selector=None,
                 auto_send: bool = False, use_langgraph: bool = True):
        self.kb, self.policy, self.classifier, self.selector = kb, policy, classifier, selector
        self.auto_send = auto_send
        self.implementation = "langgraph" if use_langgraph else "sequential"
        self.graph = self._build_graph() if use_langgraph else None

    # ---------------- nodes ----------------
    def ingest(self, s: FlowState):
        return {"ticket": normalize(s["raw"])}

    def classify(self, s: FlowState):
        try:
            return {"classification": self.classifier.classify(s["ticket"])}
        except Exception as exc:   # provider outage, timeout, invalid output, open circuit
            return {"classification": {"intent": UNKNOWN, "urgency": "medium", "confidence": 0.0,
                                       "alternatives": [], "reason": "classifier unavailable",
                                       "confidence_type": "model_failed", "error_type": type(exc).__name__}}

    def retrieve(self, s: FlowState):
        return {"matches": self.kb.search(s["ticket"].text)}

    def route(self, s: FlowState):
        return {"routing": route_policy(s["ticket"], s["classification"], s["matches"], self.policy)}

    def select_evidence(self, s: FlowState):
        evidence = s["routing"]["evidence"]
        if self.selector is None:
            return {"selection": select_from_document(evidence), "selection_errors": []}
        try:
            return {"selection": self.selector.select(s["ticket"], evidence), "selection_errors": []}
        except Exception as exc:
            return {"selection": None, "selection_errors": ["evidence_selection_failed", type(exc).__name__]}

    def validate(self, s: FlowState):
        return {"validation": validate_response(s.get("selection"), s["routing"]["evidence"])}

    def finalize(self, s: FlowState):
        return {"record": self._record(s)}

    # ---------------- orchestration ----------------
    def _build_graph(self):
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:
            raise RuntimeError("LangGraph is not installed: pip install -r requirements.txt") from exc
        g = StateGraph(FlowState)
        for name in ("ingest", "classify", "retrieve", "route", "select_evidence", "validate", "finalize"):
            g.add_node(name, getattr(self, name))
        g.add_edge(START, "ingest")
        g.add_edge("ingest", "classify")
        g.add_edge("classify", "retrieve")
        g.add_edge("retrieve", "route")
        g.add_conditional_edges("route", lambda s: s["routing"]["route"],
                                {"answer": "select_evidence", "escalate": "finalize"})
        g.add_edge("select_evidence", "validate")
        g.add_edge("validate", "finalize")
        g.add_edge("finalize", END)
        return g.compile()

    def _run_sequential(self, raw: dict) -> dict:
        s: FlowState = {"raw": raw}
        for step in (self.ingest, self.classify, self.retrieve, self.route):
            s.update(step(s))
        if s["routing"]["route"] == "answer":
            s.update(self.select_evidence(s))
            s.update(self.validate(s))
        return self.finalize(s)["record"]

    def process(self, raw: dict) -> dict:
        t0, started = time.perf_counter(), utc_now()
        try:
            record = self.graph.invoke({"raw": raw})["record"] if self.graph else self._run_sequential(raw)
        except Exception as exc:     # malformed ticket or unexpected internal error: never crash the batch
            record = self._fallback_record(raw, exc)
        record["started_at"], record["completed_at"] = started, utc_now()
        record["latency_seconds"] = round(time.perf_counter() - t0, 4)
        return record

    # ---------------- records ----------------
    def _models(self) -> dict:
        c = {"name": getattr(self.classifier, "name", type(self.classifier).__name__),
             "version": getattr(self.classifier, "version", "unversioned")}
        sel = ({"name": self.selector.name, "version": self.selector.version} if self.selector
               else {"name": "deterministic_resolution_section", "version": "1.0"})
        return {"classifier": c, "evidence_selector": sel}

    def _prompt_versions(self) -> list[str]:
        versions = [v for v in (getattr(self.classifier, "version", ""), getattr(self.selector, "version", ""))
                    if str(v).startswith("PR-")]
        return versions or ["none (no model prompt used: offline classifier and deterministic selector)"]

    def _record(self, s: FlowState) -> dict:
        t, c, r = s["ticket"], s["classification"], s["routing"]
        v = s.get("validation")
        sel_errors = s.get("selection_errors", [])
        conf_min = r["threshold_applied"]["classification_confidence_min"]
        conf = c.get("confidence") if isinstance(c.get("confidence"), (int, float)) else 0.0
        guardrails = {"pii": "not_run", "grounding": "not_run", "instruction_integrity": "not_run",
                      "scope": "not_run", "confidence_floor": "pass" if conf >= conf_min else "fail"}
        reqs = list(BASE_REQUIREMENTS)
        response, failures, flagged_input = "", [], None
        if "untrusted_instruction_in_ticket" in r["codes"]:
            # Governance §4 instruction integrity: block, escalate and record the input for review.
            guardrails["instruction_integrity"] = "fail"
            failures.append("untrusted_instruction_in_ticket")
            flagged_input = mask_pii(t.text)[:2000]
        if "sensitive_request_manual_review" in r["codes"]:
            guardrails["scope"] = "fail"          # legal/breach/refund matters are out of scope
            failures.append("sensitive_request_manual_review")
        if r["route"] == "escalate" and flagged_input is not None:
            action, stage, reason, codes = "block", "validation", r["reason"], r["codes"]
            reqs += ["FR-06", "FR-08"]
        elif r["route"] == "escalate":
            action, stage, reason, codes = "escalate", "classification" if c.get("confidence_type") == "model_failed" else "routing", r["reason"], r["codes"]
            reqs += ["FR-08"] + (["FR-10"] if c.get("confidence_type") == "model_failed" else [])
        elif sel_errors:
            action, stage, codes = "escalate", "generation", sel_errors
            reason = "The evidence-selection model failed or was unavailable, so the ticket was escalated."
            reqs += ["FR-05", "FR-08", "FR-10"]
        elif not v["passed"]:
            guardrails.update(v["guardrail_results"])
            action, stage, codes, failures = "block", "validation", v["failures"], failures + v["failures"]
            reason = ("A pre-release check blocked the automatic answer (" + ", ".join(v["failures"]) +
                      "); the ticket was escalated to a person.")
            reqs += ["FR-05", "FR-06", "FR-08"]
        else:
            guardrails.update(v["guardrail_results"])
            action, stage, codes, response = "auto_respond", "validation", r["codes"], v["response"]
            reason = r["reason"] + " All pre-release checks passed."
            reqs += ["FR-05", "FR-06"]
        if action == "auto_respond":
            delivery = "sent" if self.auto_send else "held_kill_switch"
            status = "auto_responded" if self.auto_send else "held_for_review"
            if not self.auto_send:
                reason += " Automatic sending is switched off (kill switch), so the answer is held for an agent."
        else:
            delivery, status = "not_applicable", {"escalate": "escalated", "block": "blocked"}[action]
        sources = [{"doc_id": m["doc_id"], "title": m.get("title", ""), "score": m["score"]} for m in s["matches"]]
        summary = self._summary(t)
        return {
            "decision_id": uuid.uuid4().hex, "timestamp": utc_now(), "ticket_id": t.ticket_id,
            "channel": t.channel, "stage": stage, "input_summary": summary,
            "model": self._models(),
            "prediction": {"value": c["intent"], "confidence": conf, "urgency": c.get("urgency"),
                           "confidence_type": c.get("confidence_type"), "error_type": c.get("error_type")},
            "alternatives": c.get("alternatives", []), "sources_used": sources,
            "threshold_applied": conf_min,
            "thresholds": {**r["threshold_applied"], "confidence_floor_validated_for_this_classifier":
                           self.policy.get("confidence_threshold_validated_for") == self._models()["classifier"]["name"]},
            "action_taken": action, "status": status, "delivery": delivery, "answer_sent": delivery == "sent",
            "reason": reason, "reason_codes": codes, "guardrail_results": guardrails,
            "guardrail_failures": sorted(set(failures)), "response": response, "flagged_input_for_review": flagged_input,
            "citations": [s["selection"].doc_id] if response else [],
            "handoff": None if delivery == "sent" else {
                "summary": summary, "intent": c["intent"], "urgency": c.get("urgency"),
                "sources": [x["doc_id"] for x in sources], "why_escalated": reason},
            "prompt_version": self._prompt_versions(), "policy_version": r["policy_version"],
            "requirement_ids": sorted(set(reqs)), "implementation": self.implementation,
        }

    @staticmethod
    def _summary(t) -> str:
        text = mask_pii(t.text)
        return f"{t.channel} ticket from a {t.customer_tier} customer: {text[:160]}{'…' if len(text) > 160 else ''}"

    def _fallback_record(self, raw, exc: Exception) -> dict:
        tid = str(raw.get("ticket_id") or "<missing>") if isinstance(raw, dict) else "<invalid>"
        channel = raw.get("channel") if isinstance(raw, dict) else None
        reason = f"The ticket could not be processed ({type(exc).__name__}: {str(exc)[:120]}); it was escalated to a person."
        return {
            "decision_id": uuid.uuid4().hex, "timestamp": utc_now(), "ticket_id": tid,
            "channel": channel if isinstance(channel, str) else None, "stage": "ingest",
            "input_summary": "unreadable or invalid ticket", "model": self._models(),
            "prediction": {"value": UNKNOWN, "confidence": 0.0, "urgency": None,
                           "confidence_type": "not_classified", "error_type": type(exc).__name__},
            "alternatives": [], "sources_used": [],
            "threshold_applied": self.policy["classification_confidence_min"],
            "thresholds": {"classification_confidence_min": self.policy["classification_confidence_min"],
                           "retrieval_score_min": self.policy["retrieval_score_min"]},
            "action_taken": "escalate", "status": "escalated", "delivery": "not_applicable", "answer_sent": False,
            "reason": reason, "reason_codes": ["invalid_ticket_or_internal_error", type(exc).__name__],
            "guardrail_results": {k: "not_run" for k in ("pii", "grounding", "instruction_integrity", "scope", "confidence_floor")},
            "guardrail_failures": [], "response": "", "flagged_input_for_review": None, "citations": [],
            "handoff": {"summary": "unreadable or invalid ticket", "intent": UNKNOWN, "urgency": None,
                        "sources": [], "why_escalated": reason},
            "prompt_version": self._prompt_versions(), "policy_version": self.policy["policy_version"],
            "requirement_ids": ["FR-01", "FR-07", "FR-08", "FR-10"], "implementation": self.implementation,
        }
