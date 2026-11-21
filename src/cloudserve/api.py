"""HTTP service for the decision workflow (Setup Guide §08 api.py; Stage 4 B-12).

    python -m cloudserve.api                      # http://127.0.0.1:8000  (the Build Spec's "python -m src.api")
    python -m cloudserve.api --classifier llm --provider ollama --answer-mode generated

    POST /tickets   one ticket (dataset schema) -> the governance decision record, persisted
    GET  /health    policy, models, kill switch, breaker (no secrets, env values or local paths)
    GET  /metrics   Prometheus exposition (cloudserve.metrics)

The engine is built once at startup from the same options as the CLI; each option can also be
set by environment variable (CLOUDSERVE_CLASSIFIER, CLOUDSERVE_ANSWER_MODE, CLOUDSERVE_PROVIDER,
CLOUDSERVE_AUTO_SEND, CLOUDSERVE_LLM_CACHE, CLOUDSERVE_DB, ...). Automatic sending is off unless
asked for, and the runtime kill switch applies to every request.

Malformed input: a body that is not a JSON object is rejected with 422 and is not logged (it is
not a ticket). A JSON object that is not a valid ticket (missing or empty fields, unknown
channel) is processed like the batch path: a logged escalation, returned with 200.
"""
from __future__ import annotations
import argparse
import os
import threading
from pathlib import Path
from . import ROOT
from .logging_store import new_run_id, persist_one
from .metrics import CONTENT_TYPE_LATEST, Metrics

DEFAULT_DB = ROOT / "storage" / "decisions.sqlite"
ENV_DEFAULTS = {"classifier": "CLOUDSERVE_CLASSIFIER", "evidence_selector": "CLOUDSERVE_EVIDENCE_SELECTOR",
                "answer_mode": "CLOUDSERVE_ANSWER_MODE", "llm_cache": "CLOUDSERVE_LLM_CACHE",
                "engine": "CLOUDSERVE_ENGINE"}


def _relative(path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return Path(path).name


def health(engine, db_path, cache_mode: str | None = None) -> dict:
    """Operational state for humans and probes. Only names, versions and booleans: never keys,
    environment values, base URLs or absolute paths."""
    from .metrics import _guard_of
    models = engine._models()
    guard = _guard_of(engine)
    ks = engine._kill_switch()
    r = engine.retrieval_info
    return {
        "status": "ok", "implementation": engine.implementation,
        "policy_version": engine.policy.get("policy_version"), "answer_mode": engine.answer_mode,
        "models": {k: {"name": v.get("name"), "version": v.get("version")} for k, v in models.items()},
        "prompt_versions": engine._prompt_versions(),
        "auto_send_enabled": engine.auto_send,
        "kill_switch": {"engaged": ks["file_flag_engaged"], "flag_file": ks["flag_file"], "engaged_at": ks.get("engaged_at")},
        "answers_released": bool(engine.auto_send and not ks["file_flag_engaged"]),
        "breaker": None if guard is None else {"state": guard.state, "consecutive_failures": guard.consecutive_failures,
                                               "total_failures": guard.total_failures, "trials": guard.trials},
        "llm_cache": cache_mode if engine._call_logs() else "not_used",
        "retrieval": {"retriever": r.get("retriever"), "name": r.get("name"), "chunking": r.get("chunking"),
                      "fallback": bool(r.get("fallback"))},
        "decision_log": _relative(db_path),
    }


def create_app(engine, db_path=DEFAULT_DB, metrics: Metrics | None = None, cache_mode: str | None = None):
    from fastapi import Body, FastAPI
    from fastapi.responses import Response
    app = FastAPI(title="CloudServe support decisions", version="1.0")
    metrics = metrics or Metrics(engine)
    run_id = "api-" + new_run_id()
    app.state.engine, app.state.metrics, app.state.db_path, app.state.run_id = engine, metrics, db_path, run_id
    count = {"n": 0}
    lock = threading.Lock()

    @app.post("/tickets")
    def post_ticket(ticket: dict = Body(..., description="one ticket with the dataset schema")):
        # A plain `def` runs on FastAPI's thread pool; engine, breaker, call log and the decision
        # log are safe under concurrent requests.
        record = engine.process(ticket)
        record["run_id"] = run_id
        persist_one(db_path, record, run_id)
        metrics.observe(record)
        with lock:
            count["n"] += 1
        return record

    @app.get("/health")
    def get_health():
        return {**health(engine, db_path, cache_mode), "run_id": run_id, "decisions_this_process": count["n"]}

    @app.get("/metrics")
    def get_metrics():
        return Response(metrics.exposition(), media_type=CONTENT_TYPE_LATEST)

    return app


def parse_args(argv=None) -> argparse.Namespace:
    from .runtime import add_engine_arguments
    p = argparse.ArgumentParser(description="Serve the CloudServe decision workflow over HTTP")
    p.add_argument("--host", default=os.getenv("CLOUDSERVE_API_HOST", "127.0.0.1"),
                   help="bind address (default 127.0.0.1: local only)")
    p.add_argument("--port", type=int, default=int(os.getenv("CLOUDSERVE_API_PORT", "8000")))
    p.add_argument("--db", default=os.getenv("CLOUDSERVE_DB", str(DEFAULT_DB)), help="decision log (SQLite)")
    add_engine_arguments(p)
    # Environment provides defaults for the engine options; command-line flags still win.
    p.set_defaults(**{k: os.environ[v] for k, v in ENV_DEFAULTS.items() if os.getenv(v)})
    return p.parse_args(argv)


def main(argv=None):
    import uvicorn
    from .runtime import build_engine
    a = parse_args(argv)
    engine = build_engine(a)
    app = create_app(engine, a.db, cache_mode=a.llm_cache)
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()
