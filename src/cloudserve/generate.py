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
    passage_id: str | None = None    # DOC-XXX#<slug>; resolves to exact corpus text

    @property
    def citation(self) -> str:
        return self.passage_id or self.doc_id


def select_from_document(evidence: list[dict]) -> EvidenceSelection | None:
    """Deterministic selector: the resolution steps of the best-matching approved passage.

    `evidence` is best first (KnowledgeBase.answer_passages): each article's resolution section.
    """
    for m in evidence:
        section = _RESOLUTION.search(m["passage"])
        if not section:
            continue
        quote = re.split(r"(?m)^## ", m["passage"][section.end():], maxsplit=1)[0].strip()[:1200].strip()
        if len(quote) >= 40:
            return EvidenceSelection(doc_id=m["doc_id"], quote=quote, passage_id=m.get("passage_id"))
    return None


def render_response(selection: EvidenceSelection) -> str:
    """Only deterministic text surrounds the source-exact quotation."""
    return (f"CloudServe documentation contains these steps:\n\n{selection.quote}\n\n"
            f"Source: [{selection.citation}]")


class LangChainEvidenceSelector:
    """PR-04: a provider-independent model selects a verbatim quote (it never writes free text)."""

    def __init__(self, provider: str | None = None, model_name: str | None = None,
                 base_url: str | None = None, model=None, guard=None, calls=None):
        from .providers import ProviderGuard, build_model, load_prompt, structured
        from pydantic import BaseModel, ConfigDict, Field
        self.calls = calls

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
        from .model_calls import call_model
        from .providers import for_provider
        payload = json.dumps({"ticket_subject": for_provider(self.provider, ticket.subject[:300]),
                              "ticket_body": for_provider(self.provider, ticket.body[:3000]),
                              "retrieved_documents": [{"doc_id": m["doc_id"], "passage": m["passage"][:4500]}
                                                      for m in evidence[:3]]}, ensure_ascii=False)
        msgs = [SystemMessage(content=self.prompt["system"]),
                HumanMessage(content=self.prompt["user"].replace("{payload_json}", payload))]
        sel = call_model(self, "evidence_selector", msgs)
        # One candidate passage per article, so the returned doc_id identifies the passage.
        passage_id = next((m.get("passage_id") for m in evidence[:3] if m["doc_id"] == sel.doc_id), None)
        return EvidenceSelection(sel.doc_id, sel.quote, passage_id)


# ---------------- grounded generation (PR-02) ----------------
MAX_SENTENCES = 6
MAX_CITATIONS = 3


def answer_schema():
    """PR-02 output contract (extra fields forbidden; bounded sizes)."""
    from pydantic import BaseModel, ConfigDict, Field

    class GeneratedSentence(BaseModel):
        model_config = ConfigDict(extra="forbid")
        text: str = Field(min_length=1, max_length=400)
        citations: list[str] = Field(min_length=1, max_length=MAX_CITATIONS)
        quote: str = Field(min_length=1, max_length=500)

    class GeneratedAnswer(BaseModel):
        model_config = ConfigDict(extra="forbid")
        cannot_answer: bool
        reason: str = Field(default="", max_length=300)
        sentences: list[GeneratedSentence] = Field(default_factory=list, max_length=MAX_SENTENCES)

    return GeneratedAnswer


@dataclass(frozen=True)
class GeneratedAnswer:
    """Parsed generator output: plain data, independent of the model library."""
    cannot_answer: bool
    reason: str
    sentences: tuple      # of dicts {text, citations, quote}


def _terminated(text: str) -> str:
    """Sentence with closing punctuation, so rendered answers read as sentences."""
    text = text.strip()
    return text if not text or text[-1] in ".!?:" else text + "."


def render_generated(answer: GeneratedAnswer, passages: list[dict]) -> str:
    """Each sentence keeps its own citation marker; a Sources line lists every cited passage."""
    titles = {p["passage_id"]: p.get("title", "") for p in passages}
    body = " ".join(f"{_terminated(s['text'])} [{', '.join(s['citations'])}]" for s in answer.sentences)
    cited = list(dict.fromkeys(c for s in answer.sentences for c in s["citations"]))
    sources = "; ".join(f"{c} ({titles[c]})" if titles.get(c) else c for c in cited)
    return f"{body}\n\nSources: {sources}"


class LangChainAnswerGenerator:
    """PR-02: a provider-independent model writes a short answer from approved passages only.

    The ticket is placed in a delimited untrusted block in the user message; the system message
    holds only the rules. Output is parsed into a strict schema; anything else raises, and the
    workflow escalates. guardrails.validate_generated decides whether the answer can be released.
    """

    def __init__(self, provider: str | None = None, model_name: str | None = None,
                 base_url: str | None = None, model=None, guard=None, calls=None):
        from .providers import ProviderGuard, build_model, load_prompt, structured
        self.schema, self.calls = answer_schema(), calls
        self.prompt = load_prompt("PR-02_grounded_answer.md")
        self.guard = guard or ProviderGuard()
        self.provider = provider or "injected"
        if model is not None:                       # injected (tests): used as-is
            self.runnable, self.model_name = model, model_name or "injected"
        else:
            chat, self.model_name, extra = build_model(self.provider, model_name, base_url)
            self.runnable = structured(chat, self.schema, self.provider, extra)
        self.name = f"{self.provider}:{self.model_name}"
        self.version = self.prompt["label"]

    def messages(self, ticket, passages: list[dict]) -> list:
        from langchain_core.messages import HumanMessage, SystemMessage
        payload = json.dumps([{"passage_id": p["passage_id"], "title": p.get("title", ""), "text": p["passage"][:2500]}
                              for p in passages], ensure_ascii=False, indent=1)
        from .providers import for_provider
        values = {"passages_json": payload, "ticket_id": for_provider(self.provider, str(ticket.ticket_id)[:64]),
                  "channel": for_provider(self.provider, str(ticket.channel)[:32]),
                  "subject": for_provider(self.provider, ticket.subject[:300]),
                  "body": for_provider(self.provider, ticket.body[:4000])}
        # Single pass, so placeholder-like text inside the ticket or passages is never substituted.
        user = re.sub(r"\{(passages_json|ticket_id|channel|subject|body)\}", lambda m: values[m.group(1)],
                      self.prompt["user"])
        return [SystemMessage(content=self.prompt["system"]), HumanMessage(content=user)]

    def generate(self, ticket, passages: list[dict]) -> GeneratedAnswer:
        from .model_calls import call_model
        v = call_model(self, "generator", self.messages(ticket, passages))
        return GeneratedAnswer(v.cannot_answer, v.reason,
                               tuple({"text": s.text, "citations": list(s.citations), "quote": s.quote} for s in v.sentences))
