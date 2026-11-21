"""Evidence selection and pre-release validation (FR-05, FR-06, A6, A7).

The response is EXTRACTIVE: a fixed sentence around a verbatim quote from an approved,
retrieved article, cited by doc_id. A model may select the quote (PR-04); it may not write
free text. The validator runs on every candidate response before release and can block it.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from .pipeline import INJECTION_PATTERN, PII_PATTERN

DOC_ID = re.compile(r"^DOC-[A-Z0-9-]+$")
# Tone/scope: the system must not claim completed account actions or make commitments.
SCOPE_PATTERN = re.compile(
    r"(?i)\b(?:we\s+have\s+(?:issued|refunded|credited|fixed|rotated|changed|reset|restored)"
    r"|we\s+(?:will|guarantee\s+to)\s+(?:refund|credit|compensate)|guaranteed?)\b")
_RESOLUTION = re.compile(r"(?im)^## (?:Resolution|Steps|Workaround|How to|Fix|Recommended actions)\b[^\n]*\n")


@dataclass(frozen=True)
class EvidenceSelection:
    doc_id: str
    quote: str


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def select_from_document(evidence: list[dict]) -> EvidenceSelection | None:
    """Deterministic selector: the resolution section of the best approved article."""
    for m in evidence:
        section = _RESOLUTION.search(m["passage"])
        if not section:
            continue
        quote = re.split(r"(?m)^## ", m["passage"][section.end():], maxsplit=1)[0].strip()[:1200].strip()
        if len(quote) >= 40:
            return EvidenceSelection(doc_id=m["doc_id"], quote=quote)
    return None


def render_response(selection: EvidenceSelection) -> str:
    """Only deterministic text surrounds the source-exact quotation."""
    return (f"CloudServe documentation contains these steps:\n\n{selection.quote}\n\n"
            f"Source: [{selection.doc_id}]")


def validate_response(selection: EvidenceSelection | None, evidence: list[dict]) -> dict:
    """Pre-release checks. Returns {'passed', 'failures', 'guardrail_results', 'response'}.

    grounding: cited doc must be one of the approved retrieved articles and the quote must occur
    verbatim in it. pii: no private-data patterns in the outbound text. instruction_integrity:
    no instruction-like text in the outbound text. scope: no completed-action claims/commitments.
    """
    results = {"grounding": "pass", "pii": "pass", "instruction_integrity": "pass", "scope": "pass"}
    failures: list[str] = []
    if selection is None:
        results = {k: "not_run" for k in results}
        results["grounding"] = "fail"
        return {"passed": False, "failures": ["no_evidence_selection"], "guardrail_results": results, "response": ""}
    source = next((m["passage"] for m in evidence if m["doc_id"] == selection.doc_id), None)
    if not DOC_ID.fullmatch(selection.doc_id):
        failures.append("invalid_document_id")
    if source is None:
        failures.append("citation_not_in_retrieval")
    if not selection.quote or len(selection.quote.strip()) < 40 or len(selection.quote) > 1500:
        failures.append("quote_invalid_length")
    elif source is not None and _normalize(selection.quote) not in _normalize(source):
        failures.append("quote_not_supported_by_source")
    if failures:
        results["grounding"] = "fail"
    response = render_response(selection)
    if PII_PATTERN.search(response):
        failures.append("possible_private_data"); results["pii"] = "fail"
    if INJECTION_PATTERN.search(response):
        failures.append("suspicious_source_instructions"); results["instruction_integrity"] = "fail"
    if SCOPE_PATTERN.search(response):
        failures.append("out_of_scope_commitment"); results["scope"] = "fail"
    passed = not failures
    return {"passed": passed, "failures": sorted(set(failures)), "guardrail_results": results,
            "response": response if passed else ""}


class LangChainEvidenceSelector:
    """PR-04: a provider-independent model selects a verbatim quote (it never writes free text)."""

    def __init__(self, provider: str | None = None, model_name: str | None = None,
                 base_url: str | None = None, model=None, guard=None):
        from .ai import ProviderGuard, build_model, load_prompt, structured
        from pydantic import BaseModel, ConfigDict, Field

        class Selected(BaseModel):
            model_config = ConfigDict(extra="forbid")
            doc_id: str = Field(min_length=1, max_length=80)
            quote: str = Field(max_length=1500)

        self.schema = Selected
        self.prompt = load_prompt("PR-04_evidence_selection.md")
        self.guard = guard or ProviderGuard()
        self.provider = provider or "injected"
        if model is not None:
            self.runnable, self.model_name = model, model_name or "injected"
        else:
            chat, self.model_name, extra = build_model(self.provider, model_name, base_url)
            self.runnable = structured(chat, Selected, self.provider, extra)
        self.name = f"{self.provider}:{self.model_name}"
        self.version = self.prompt["label"]

    def select(self, ticket, evidence: list[dict]) -> EvidenceSelection:
        from langchain_core.messages import HumanMessage, SystemMessage
        payload = json.dumps({"ticket_subject": ticket.subject[:300], "ticket_body": ticket.body[:3000],
                              "retrieved_documents": [{"doc_id": m["doc_id"], "passage": m["passage"][:4500]}
                                                      for m in evidence[:3]]}, ensure_ascii=False)
        msgs = [SystemMessage(content=self.prompt["system"]),
                HumanMessage(content=self.prompt["user"].replace("{payload_json}", payload))]
        result = self.guard.call(self.runnable.invoke, msgs)
        sel = self.schema.model_validate(result.model_dump() if hasattr(result, "model_dump") else result)
        return EvidenceSelection(sel.doc_id, sel.quote)
