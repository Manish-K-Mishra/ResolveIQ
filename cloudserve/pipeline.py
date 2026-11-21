"""Ingestion, lexical retrieval and shared safety patterns (deterministic, no model calls)."""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

CHANNELS = {"email", "chat", "docs_comment", "forum"}
# Evaluation-only fields. They are removed at ingestion so they can never reach inference.
EVALUATION_ONLY_FIELDS = ("labels", "history")

TOKEN = re.compile(r"[a-z0-9_]+")
STOP = set("the and for with this that from have has are was were you your our can not when what where which into using user error issue help please need because would should about does after before then them just been more some there than their they cant dont unable getting while also same last week now again".split())

# --- Guardrail patterns (defence in depth; intent-based routing is the primary control) ---
PII_PATTERN = re.compile(
    r"(?i)(?:\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b"            # email address
    r"|\b(?:\+?\d[\d ()-]{8,}\d)\b"                       # phone-like number
    r"|\b(?:sk|pk|rk)[-_](?:live|test)[-_][a-z0-9]{8,}\b"  # API-key shapes
    r"|\bAKIA[0-9A-Z]{16}\b"                              # cloud access key id
    r"|\b(?:\d[ -]?){13,16}\b)")                          # card-number-like digit run
INJECTION_PATTERN = re.compile(
    r"(?i)(?:\b(?:ignore|disregard|forget|override)\s+(?:(?:all|any|the|your|my|these|those|previous|prior|above|earlier|system|of)\s+)*"
    r"(?:instructions?|rules?|prompts?|guidelines?|directions?|directives?)\b"
    r"|\bforget\s+(?:everything|all)\s+(?:you\s+)?(?:were|have\s+been|was)\s+told"
    r"|\b(?:reveal|print|show|tell|display|repeat|output|leak)\s+(?:me\s+)?(?:the\s+|your\s+)?(?:hidden\s+|secret\s+|initial\s+|original\s+)?"
    r"(?:system\s+prompt|instructions|api\s+key)"
    r"|\bbypass\s+(?:the\s+|your\s+)?(?:guardrails|filters|safety)"
    r"|\byou\s+are\s+now\s+(?:a|an|the|dan|in)\b"
    r"|\b(?:unrestricted|unfiltered|jailbroken)\s+(?:ai|assistant|mode|model)\b"
    r"|\bact\s+as\s+(?:an?\s+|the\s+)?(?:admin|administrator|developer|system|root|superuser|dan)\b"
    r"|\bnew\s+(?:system\s+)?instructions?\s*:"
    r"|<\s*/?\s*system\s*>"
    r"|\bignora\s+(?:todas\s+)?(?:las\s+)?instrucciones)")
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"))
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "@": "a", "$": "s"})


def contains_injection(text: str) -> bool:
    """Instruction-integrity screen on untrusted ticket text (Governance §4).

    Checks the text as written and a normalised copy (NFKC, zero-width characters removed,
    common digit-for-letter swaps undone) so trivial obfuscation does not slip past. This is
    still pattern matching: paraphrased injections can pass (docs/ARCHITECTURE.md §10).
    The answer released to the customer is a verified documentation excerpt either way.
    """
    import unicodedata
    plain = unicodedata.normalize("NFKC", text or "").translate(_ZERO_WIDTH)
    return bool(INJECTION_PATTERN.search(plain) or INJECTION_PATTERN.search(plain.translate(_LEET)))


RISK_PATTERN = re.compile(
    r"(?i)\b(?:legal\s+action|lawsuit|lawyers?|attorneys?|sue|suing|subpoena"
    r"|breach(?:ed|es)?|data\s+leak|leaked|compromised"
    r"|credentials?\s+(?:may\s+have\s+been\s+|have\s+been\s+|were\s+|was\s+)?exposed"
    r"|exposed\s+(?:credentials?|keys?|secrets?|data)"
    r"|refund(?:ed|s)?|chargeback)\b")


def tokens(text: str) -> set[str]:
    return {w for w in TOKEN.findall(text.lower()) if len(w) > 2 and w not in STOP}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: str | Path):
    """BOM-tolerant JSON loader (an unseen evaluation file may carry a UTF-8 BOM)."""
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_tickets(path: str | Path) -> list:
    raw = load_json(path)
    return raw if isinstance(raw, list) else [raw]


@dataclass(frozen=True)
class Ticket:
    ticket_id: str
    channel: str
    subject: str
    body: str
    customer_tier: str
    metadata: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return f"{self.subject} {self.body}".strip()


class ValidationError(ValueError):
    pass


def normalize(data: dict) -> Ticket:
    """One internal representation for all four channels; evaluation labels are dropped."""
    if not isinstance(data, dict):
        raise ValidationError("ticket must be a JSON object")
    channel = str(data.get("channel") or "").strip().lower()
    if channel not in CHANNELS:
        raise ValidationError("unsupported or missing channel")
    tid = str(data.get("ticket_id") or "").strip()
    if not tid:
        raise ValidationError("ticket_id is required")
    subject = str(data.get("subject") or "").strip()
    body = str(data.get("body") or "").strip()
    if not (body or subject):
        raise ValidationError("ticket text is empty")
    # Allow-list of non-label metadata; EVALUATION_ONLY_FIELDS (labels, history) never pass through.
    metadata = {k: data.get(k) for k in ("received_at", "customer_region", "language_fluency")}
    return Ticket(tid, channel, subject, body, str(data.get("customer_tier") or "unknown").lower(), metadata)


class KnowledgeBase:
    """Lexical retrieval over whole articles.

    Chunking decision: each of the 29 articles is one passage. Articles are short and follow a
    fixed Symptoms/Causes/Resolution/Notes structure; splitting inside a resolution sequence
    yields passages that retrieve well but read as incomplete (Dataset Guide §2). The citation
    unit is therefore the article (doc_id); the quoted span inside it is verified verbatim.
    """
    MIN_SCORE = 0.26        # share of query terms found in the article
    MIN_OVERLAP = 2         # at least two shared terms, so single-word matches return nothing

    def __init__(self, docs: list[dict]):
        self.docs = []
        for d in docs:
            if not d.get("doc_id") or not d.get("content"):
                continue
            target = " ".join([str(d.get("title", "")), str(d.get("category", "")), str(d["content"])])
            self.docs.append((d, tokens(target)))
        self.by_id = {d["doc_id"]: d for d, _ in self.docs}

    @classmethod
    def from_path(cls, path: str | Path):
        return cls(load_json(path))

    def search(self, question: str, limit: int = 3) -> list[dict]:
        q = tokens(question)
        if not q:
            return []
        ranked = []
        for d, dt in self.docs:
            overlap = q & dt
            score = len(overlap) / len(q)
            if score >= self.MIN_SCORE and len(overlap) >= self.MIN_OVERLAP:
                ranked.append({"doc_id": d["doc_id"], "title": d.get("title", ""),
                               "score": round(score, 4), "passage": d["content"]})
        return sorted(ranked, key=lambda x: (-x["score"], x["doc_id"]))[:limit]


def mask_pii(text: str) -> str:
    return PII_PATTERN.sub("[redacted]", text)
