"""Ingestion (FR-01): one internal Ticket for all four channels, plus shared text helpers.

Deterministic, no model calls. Evaluation-only fields are dropped here so they never reach inference.
"""
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
