"""Ingestion (A2) and retrieval (A4)."""
import json
import pytest
from cloudserve.ingest import ValidationError, load_tickets, normalize
from cloudserve.retrieve import KnowledgeBase


def ticket(channel="email", **kw):
    return {"ticket_id": "T-1", "channel": channel, "subject": "Login help",
            "body": "The console says invalid credentials", **kw}


@pytest.mark.parametrize("channel", ["email", "chat", "docs_comment", "forum"])
def test_each_channel_normalises_to_one_representation(channel):
    t = normalize(ticket(channel))
    assert (t.channel, t.ticket_id, t.body) == (channel, "T-1", "The console says invalid credentials")


def test_chat_ticket_without_subject_and_unicode_body():
    t = normalize({"ticket_id": "C-1", "channel": "CHAT", "subject": None, "body": "Connexion échouée 🔒 — 401"})
    assert t.channel == "chat" and t.subject == "" and "🔒" in t.text


@pytest.mark.parametrize("bad", [{"ticket_id": "X", "channel": "fax", "body": "hi"},
                                 {"ticket_id": "", "channel": "email", "body": "hi"},
                                 {"ticket_id": "X", "channel": "email", "body": "  ", "subject": ""},
                                 "not-an-object"])
def test_invalid_tickets_raise_validation_error(bad):
    with pytest.raises(ValidationError):
        normalize(bad)


def test_evaluation_labels_and_history_are_stripped_at_ingestion():
    t = normalize(ticket(labels={"intent": "rate_limit"}, history={"csat_rating": 1}))
    assert "labels" not in t.metadata and "history" not in t.metadata
    assert "rate_limit" not in repr(t)


def test_bom_encoded_input_file_loads(tmp_path):
    p = tmp_path / "bom.json"
    p.write_bytes("﻿".encode("utf-8") + json.dumps([ticket()]).encode("utf-8"))
    assert load_tickets(p)[0]["ticket_id"] == "T-1"


def test_retrieval_returns_real_identifiable_passages(kb):
    hits = kb.search("invalid credentials console account locked login")
    assert hits and hits[0]["doc_id"] == "DOC-AUTH-001"
    for h in hits:
        assert h["passage"] == kb.by_id[h["doc_id"]]["content"][h["start"]:h["end"]] == kb.resolve(h["passage_id"])
        assert h["passage_id"].startswith(h["doc_id"] + "#") and 0 < h["score"] <= 1


@pytest.mark.parametrize("q", ["hyperquantum zebra orbit magnet", "", "the and for"])
def test_retrieval_returns_nothing_when_nothing_is_relevant(kb, q):
    assert kb.search(q) == []


def test_knowledge_base_skips_documents_without_content():
    kb = KnowledgeBase([{"doc_id": "DOC-A-1", "content": ""}, {"title": "no id", "content": "x"}])
    assert kb.docs == []
