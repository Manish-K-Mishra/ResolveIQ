"""The CloudServe decision workflow, orchestrated with LangGraph.

    ingest -> classify -> retrieve -> route --(answer)--> select_evidence -> validate -> finalize
                                          `--(escalate)-----------------------------------^

In generated answer mode (`answer_mode="generated"`) a `generate` node (PR-02, cited sentences
over the approved passages) takes the place of `select_evidence`; the shape is otherwise the same,
and escalated tickets never reach the generator.

Critical decisions are deterministic Python (route.route, guardrails.validate_response and
guardrails.validate_generated); models only classify, select quotes or draft cited sentences
behind adapters. Every path, including failures,
ends in `finalize`, which builds one governance-schema decision record.

Actions (Governance §1): auto_respond | escalate | block.
Kill switch (Governance §5): when automatic sending is disabled (the default), or the runtime
flag file is present (cloudserve.killswitch, read at finalize for every ticket), an auto_respond
decision is HELD for human review (status 'held_for_review', answer_sent false).
"""
from __future__ import annotations
import os
import time
import uuid
from pathlib import Path
from typing import Any, TypedDict
from .classify import UNKNOWN
from .generate import select_from_document
from .guardrails import mask_pii, validate_generated, validate_response
from . import killswitch
from .ingest import normalize, utc_now
from .model_calls import CallLog, cache_status, summarise
from .retrieve import KnowledgeBase
from .route import route as route_policy

BASE_REQUIREMENTS = ["FR-01", "FR-02", "FR-03", "FR-04", "FR-07"]
ANSWER_MODES = ("extractive", "generated")
CANNOT_ANSWER_REASON = "the documentation does not answer this"


def auto_send_enabled_from_env() -> bool:
    return os.getenv("CLOUDSERVE_AUTO_SEND", "false").strip().lower() in ("1", "true", "yes", "on")


class FlowState(TypedDict, total=False):
    raw: dict
    ticket: Any
    classification: dict
    matches: list[dict]
    routing: dict
    answer_passages: list[dict]
    selection: Any
    selection_errors: list[str]
    generated: Any
    validation: dict
    record: dict


class DecisionEngine:
    """Holds the run's dependencies; `process(raw)` never raises and always returns a record."""

    def __init__(self, kb: KnowledgeBase, policy: dict, classifier, selector=None,
                 auto_send: bool = False, use_langgraph: bool = True, retrieval_info: dict | None = None,
                 answer_mode: str = "extractive", generator=None, kill_switch_file=None):
        if answer_mode not in ANSWER_MODES:
            raise ValueError(f"answer_mode must be one of {ANSWER_MODES}")
        if answer_mode == "generated" and generator is None:
            raise ValueError("generated answer mode needs a generator (an LLM provider)")
        self.kb, self.policy, self.classifier, self.selector = kb, policy, classifier, selector
        self.answer_mode, self.generator = answer_mode, generator
        # Which retriever actually ran (and whether it is the fallback), written into every record.
        self.retrieval_info = retrieval_info or {**kb.describe(), "fallback": False}
        self.auto_send = auto_send
        self.kill_switch_file = Path(kill_switch_file) if kill_switch_file else killswitch.flag_path()
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
        # The complete resolution section of each approved matching article, best match first.
        candidates = self.kb.answer_passages(s["routing"]["evidence"])
        if self.selector is None:
            return {"answer_passages": candidates, "selection": select_from_document(candidates), "selection_errors": []}
        try:
            return {"answer_passages": candidates, "selection": self.selector.select(s["ticket"], candidates),
                    "selection_errors": []}
        except Exception as exc:
            return {"answer_passages": candidates, "selection": None,
                    "selection_errors": ["evidence_selection_failed", type(exc).__name__]}

    def generate(self, s: FlowState):
        # Approved passages only (with passage ids), never whole articles: retrieve.generation_passages.
        passages = self.kb.generation_passages(s["routing"]["evidence"])
        try:
            return {"answer_passages": passages, "generated": self.generator.generate(s["ticket"], passages),
                    "selection_errors": []}
        except Exception as exc:   # outage, timeout, open circuit, malformed / schema-invalid output
            return {"answer_passages": passages, "generated": None,
                    "selection_errors": ["generation_failed", type(exc).__name__]}

    def validate(self, s: FlowState):
        if self.answer_mode == "generated":
            if s.get("generated") is None:      # generation failed: _record escalates on selection_errors
                return {"validation": None}
            return {"validation": validate_generated(s["generated"], s.get("answer_passages", []))}
        return {"validation": validate_response(s.get("selection"), s.get("answer_passages", []))}

    def finalize(self, s: FlowState):
        return {"record": self._record(s)}

    # ---------------- orchestration ----------------
    def _build_graph(self):
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:
            raise RuntimeError("LangGraph is not installed: pip install -r requirements.txt") from exc
        g = StateGraph(FlowState)
        answer_node = self._answer_node
        for name in ("ingest", "classify", "retrieve", "route", answer_node, "validate", "finalize"):
            g.add_node(name, getattr(self, name))
        g.add_edge(START, "ingest")
        g.add_edge("ingest", "classify")
        g.add_edge("classify", "retrieve")
        g.add_edge("retrieve", "route")
        g.add_conditional_edges("route", lambda s: s["routing"]["route"],
                                {"answer": answer_node, "escalate": "finalize"})
        g.add_edge(answer_node, "validate")
        g.add_edge("validate", "finalize")
        g.add_edge("finalize", END)
        return g.compile()

    @property
    def _answer_node(self) -> str:
        return "generate" if self.answer_mode == "generated" else "select_evidence"

    def _run_sequential(self, raw: dict) -> dict:
        s: FlowState = {"raw": raw}
        for step in (self.ingest, self.classify, self.retrieve, self.route):
            s.update(step(s))
        if s["routing"]["route"] == "answer":
            s.update(getattr(self, self._answer_node)(s))
            s.update(self.validate(s))
        return self.finalize(s)["record"]

    def _call_logs(self) -> list:
        logs = [getattr(a, "calls", None) for a in (self.classifier, self.selector, self.generator)]
        return list({id(x): x for x in logs if isinstance(x, CallLog)}.values())

    def _drain_calls(self) -> list[dict]:
        return [e for log in self._call_logs() for e in log.drain()]

    def process(self, raw: dict) -> dict:
        t0, started = time.perf_counter(), utc_now()
        self._drain_calls()          # nothing from an earlier ticket is attributed to this one
        try:
            record = self.graph.invoke({"raw": raw})["record"] if self.graph else self._run_sequential(raw)
        except Exception as exc:     # malformed ticket or unexpected internal error: never crash the batch
            record = self._fallback_record(raw, exc)
        calls = self._drain_calls()
        record["model_calls"], record["usage"] = calls, summarise(calls)
        record["llm_cache"] = cache_status(calls)
        record["pii_masked_for_provider"] = (any(c["pii_masked_for_provider"] for c in calls) if calls else None)
        record["started_at"], record["completed_at"] = started, utc_now()
        record["latency_seconds"] = round(time.perf_counter() - t0, 4)
        return record

    # ---------------- records ----------------
    def _models(self) -> dict:
        c = {"name": getattr(self.classifier, "name", type(self.classifier).__name__),
             "version": getattr(self.classifier, "version", "unversioned")}
        sel = ({"name": self.selector.name, "version": self.selector.version} if self.selector
               else {"name": "deterministic_resolution_section", "version": "1.0"})
        out = {"classifier": c, "evidence_selector": sel}
        if self.answer_mode == "generated":
            out["evidence_selector"] = {"name": "not_used (generated answer mode)", "version": "n/a"}
            out["generator"] = {"name": getattr(self.generator, "name", type(self.generator).__name__),
                                "version": getattr(self.generator, "version", "unversioned")}
        return out

    def _prompt_versions(self) -> list[str]:
        answerer = self.generator if self.answer_mode == "generated" else self.selector
        versions = [v for v in (getattr(self.classifier, "version", ""), getattr(answerer, "version", ""))
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
        if self.answer_mode == "generated":
            guardrails.update({"citations": "not_run", "support": "not_run"})
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
            what = "answer-generation" if self.answer_mode == "generated" else "evidence-selection"
            reason = f"The {what} model failed, was unavailable or returned invalid output, so the ticket was escalated."
            reqs += ["FR-05", "FR-08", "FR-10"]
        elif v.get("cannot_answer"):
            # The model said the approved passages do not answer the ticket: escalate, do not block.
            action, stage, codes = "escalate", "generation", ["documentation_does_not_answer"]
            reason = (f"Escalated to a person: {CANNOT_ANSWER_REASON}"
                      + (f" (model: {v['reason'][:200]})" if v.get("reason") else "") + ".")
            reqs += ["FR-05", "FR-08"]
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
        ks = self._kill_switch()          # read now, at finalize, for every ticket
        if action == "auto_respond":
            release = self.auto_send and not ks["file_flag_engaged"]
            delivery = "sent" if release else "held_kill_switch"
            status = "auto_responded" if release else "held_for_review"
            if ks["file_flag_engaged"]:
                reason += (" The kill switch was engaged (" + ks["flag_file"] +
                           (f", since {ks['engaged_at']}" if ks.get("engaged_at") else "") +
                           "), so no answer is released; the answer is held for an agent.")
            elif not self.auto_send:
                reason += " Automatic sending is switched off (kill switch), so the answer is held for an agent."
        else:
            delivery, status = "not_applicable", {"escalate": "escalated", "block": "blocked"}[action]
        sources = [{"passage_id": m.get("passage_id"), "doc_id": m["doc_id"], "title": m.get("title", ""),
                    "section": m.get("section"), "score": m["score"], "start": m.get("start"), "end": m.get("end")}
                   for m in s["matches"]]
        citations, spans = [], []
        if response and self.answer_mode == "generated":
            by_id = {p["passage_id"]: p for p in s["answer_passages"]}
            spans = [{**sp, "matched_passage_id": by_id[sp["passage_id"]].get("matched_passage_id"),
                      "score": by_id[sp["passage_id"]].get("score")} for sp in v["citation_spans"]]
            citations = list(dict.fromkeys(c for e in v["sentences"] for c in e["citations"]))
        elif response:
            sel = s["selection"]
            # Same lookup as the grounding check (guardrails.validate_response), which already passed.
            cited = next(m for m in s["answer_passages"] if m["doc_id"] == sel.doc_id
                         and (sel.passage_id is None or m.get("passage_id") == sel.passage_id))
            citations = [sel.citation]
            # Exact character span of the quote inside the article content (DOC content[start:end]).
            at = cited["passage"].find(sel.quote)
            spans = [{"passage_id": sel.citation, "doc_id": sel.doc_id,
                      "matched_passage_id": cited.get("matched_passage_id"), "score": cited["score"],
                      "start": cited["start"] + at if at >= 0 and cited.get("start") is not None else None,
                      "end": cited["start"] + at + len(sel.quote) if at >= 0 and cited.get("start") is not None else None}]
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
            "citations": citations, "citation_spans": spans, "retrieval": self.retrieval_info,
            "answer_mode": self.answer_mode, "kill_switch": ks,
            "unsupported_claims": (v or {}).get("unsupported_claims", []),
            "generation": self._generation_block(s),
            "handoff": None if delivery == "sent" else {
                "summary": summary, "intent": c["intent"], "urgency": c.get("urgency"),
                "sources": [x["passage_id"] or x["doc_id"] for x in sources], "why_escalated": reason},
            "prompt_version": self._prompt_versions(), "policy_version": r["policy_version"],
            "requirement_ids": sorted(set(reqs)), "implementation": self.implementation,
        }

    def _generation_block(self, s: FlowState) -> dict | None:
        """What the generator was given and returned, plus the per-sentence checks (generated mode)."""
        if self.answer_mode != "generated" or "answer_passages" not in s:
            return None
        g, v = s.get("generated"), s.get("validation") or {}
        return {"passages_given": [p["passage_id"] for p in s["answer_passages"]],
                "cannot_answer": bool(g and g.cannot_answer), "model_reason": g.reason if g else None,
                "sentences": v.get("sentences", []), "error": s.get("selection_errors") or None}

    def _kill_switch(self) -> dict:
        st = killswitch.state(self.kill_switch_file)
        return {"auto_send_enabled": self.auto_send, "file_flag_engaged": st["engaged"],
                "flag_file": st["flag_file"], "engaged_at": st.get("engaged_at"), "reason": st.get("reason")}

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
            "citation_spans": [], "retrieval": self.retrieval_info,
            "answer_mode": self.answer_mode, "unsupported_claims": [], "generation": None,
            "kill_switch": self._kill_switch(),
            "handoff": {"summary": "unreadable or invalid ticket", "intent": UNKNOWN, "urgency": None,
                        "sources": [], "why_escalated": reason},
            "prompt_version": self._prompt_versions(), "policy_version": self.policy["policy_version"],
            "requirement_ids": ["FR-01", "FR-07", "FR-08", "FR-10"], "implementation": self.implementation,
        }
