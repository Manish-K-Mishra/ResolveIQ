"""Answer generation with citations (FR-05): evidence selection and response rendering.

The response is EXTRACTIVE: a fixed sentence around a verbatim quote from an approved,
retrieved article, cited by doc_id. A model may select the quote (PR-04); it may not write
free text. guardrails.validate_response runs on every candidate response before release.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass

_RESOLUTION = re.compile(r"(?im)^## (?:Resolution|Steps|Workaround|How to|Fix|Recommended actions)\b[^\n]*\n")


@dataclass(frozen=True)
class EvidenceSelection:
    doc_id: str
    quote: str


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


class LangChainEvidenceSelector:
    """PR-04: a provider-independent model selects a verbatim quote (it never writes free text)."""

    def __init__(self, provider: str | None = None, model_name: str | None = None,
                 base_url: str | None = None, model=None, guard=None):
        from .providers import ProviderGuard, build_model, load_prompt, structured
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
