"""Persistent decision log (FR-07, A8). One row per processed ticket, keyed by run.

Each row carries the Governance Framework minimum record (decision_id, timestamp, ticket_id,
stage, input_summary, model, prediction, alternatives, sources_used, threshold_applied,
action_taken, reason, guardrail_results, prompt_version, requirement_ids) in `record_json`,
with the fields used for reconciliation also stored as columns.
"""
from __future__ import annotations
import json
import sqlite3
import threading
import uuid
from pathlib import Path
from .ingest import utc_now

GOVERNANCE_FIELDS = ("decision_id", "timestamp", "ticket_id", "stage", "input_summary", "model",
                     "prediction", "alternatives", "sources_used", "threshold_applied", "action_taken",
                     "reason", "guardrail_results", "prompt_version", "requirement_ids")


def new_run_id() -> str:
    return f"run-{utc_now()[:19].replace(':', '').replace('-', '')}-{uuid.uuid4().hex[:6]}"


# One writer at a time in this process (API worker threads); `timeout` covers other processes.
_WRITE_LOCK = threading.Lock()


def open_log(path: str | Path) -> sqlite3.Connection:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p, timeout=30)
    with _WRITE_LOCK:
        _create(con)
    return con


def _create(con: sqlite3.Connection) -> None:
    con.execute("""CREATE TABLE IF NOT EXISTS decision_log (
        decision_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, ticket_id TEXT NOT NULL, channel TEXT,
        stage TEXT NOT NULL, action_taken TEXT NOT NULL, status TEXT NOT NULL,
        answer_sent INTEGER NOT NULL, policy_version TEXT, prompt_version TEXT,
        reason TEXT NOT NULL, record_json TEXT NOT NULL, created_at TEXT NOT NULL)""")
    con.commit()


def persist(con: sqlite3.Connection, record: dict, run_id: str) -> None:
    missing = [f for f in GOVERNANCE_FIELDS if f not in record]
    if missing:
        raise ValueError(f"decision record missing governance fields: {missing}")
    with _WRITE_LOCK, con:
        con.execute("INSERT INTO decision_log VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (record["decision_id"], run_id, record["ticket_id"], record.get("channel"),
                     record["stage"], record["action_taken"], record["status"], int(bool(record["answer_sent"])),
                     record.get("policy_version"), json.dumps(record["prompt_version"]), record["reason"],
                     json.dumps(record, ensure_ascii=False), utc_now()))


def count_run(con: sqlite3.Connection, run_id: str) -> int:
    return con.execute("SELECT COUNT(*) FROM decision_log WHERE run_id = ?", (run_id,)).fetchone()[0]


def persist_one(path: str | Path, record: dict, run_id: str) -> None:
    """Write one decision on its own connection (safe from any thread, e.g. API requests)."""
    con = open_log(path)
    try:
        persist(con, record, run_id)
    finally:
        con.close()
