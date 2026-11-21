"""Shared construction of the decision engine and batch runner (used by CLI and harness)."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from . import ROOT
from .classify import NaiveBayesIntentClassifier
from .graph import DecisionEngine, auto_send_enabled_from_env
from .logging_store import count_run, open_log, persist
from .retrieve import KnowledgeBase
from .route import DEFAULT_POLICY_PATH, load_policy


def load_env() -> None:
    """Read .env without printing it; OS environment variables take precedence."""
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
    except ImportError:
        pass


def add_engine_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--docs", default=str(ROOT / "data/documentation.json"), help="knowledge-base corpus")
    p.add_argument("--train", default=str(ROOT / "data/development_tickets.json"),
                   help="labelled DEVELOPMENT tickets used to train the offline classifier")
    p.add_argument("--policy", default=str(DEFAULT_POLICY_PATH), help="versioned routing policy")
    p.add_argument("--engine", choices=["graph", "sequential"], default="graph",
                   help="graph = LangGraph workflow (default); sequential = same nodes without LangGraph")
    p.add_argument("--classifier", choices=["offline", "llm"], default="offline",
                   help="offline = naive Bayes (no network); llm = PR-01 through --provider")
    p.add_argument("--evidence-selector", choices=["extractive", "llm"], default="extractive",
                   help="extractive = deterministic; llm = PR-04 verbatim quote selection")
    p.add_argument("--provider", choices=["ollama", "llamacpp", "anthropic", "openai"], default=None,
                   help="model backend (default: CLOUDSERVE_PROVIDER or ollama)")
    p.add_argument("--model", default=None, help="model name (default: CLOUDSERVE_MODEL or provider default)")
    p.add_argument("--base-url", default=None, help="local model server URL (ollama/llamacpp only)")
    p.add_argument("--auto-send", action="store_true",
                   help="release automatic answers (kill switch off); also CLOUDSERVE_AUTO_SEND=true")


def build_engine(a: argparse.Namespace) -> DecisionEngine:
    load_env()
    provider = a.provider or os.getenv("CLOUDSERVE_PROVIDER", "ollama")
    if a.base_url and provider in ("anthropic", "openai"):
        raise SystemExit("--base-url is only supported for local providers (ollama, llamacpp)")
    kb = KnowledgeBase.from_path(a.docs)
    policy = load_policy(a.policy)
    try:
        return _build(a, provider, kb, policy)
    except Exception as exc:
        from .providers import ModelUnavailable
        if isinstance(exc, ModelUnavailable):   # e.g. missing API key: clear message, no traceback
            raise SystemExit(f"Model provider not usable: {exc}. Use --classifier offline to run without a model.")
        if isinstance(exc, ValueError) and str(exc).startswith("CLOUDSERVE_"):   # bad numeric setting
            raise SystemExit(f"Configuration error: {exc}")
        raise


def _build(a, provider, kb, policy) -> DecisionEngine:
    guard = None
    if a.classifier == "llm" or a.evidence_selector == "llm":
        from .providers import ProviderGuard
        guard = ProviderGuard()       # one circuit breaker shared by both model calls in this run
    if a.classifier == "llm":
        from .classify import LangChainClassifier
        classifier = LangChainClassifier(provider=provider, model_name=a.model, base_url=a.base_url, guard=guard)
    else:
        classifier = NaiveBayesIntentClassifier.from_development_file(a.train)
    selector = None
    if a.evidence_selector == "llm":
        from .generate import LangChainEvidenceSelector
        selector = LangChainEvidenceSelector(provider=provider, model_name=a.model, base_url=a.base_url, guard=guard)
    return DecisionEngine(kb, policy, classifier, selector,
                          auto_send=a.auto_send or auto_send_enabled_from_env(),
                          use_langgraph=a.engine == "graph")


def run_batch(engine: DecisionEngine, items: list, db_path: str | Path, run_id: str) -> tuple[list[dict], int]:
    """Process every item; persist each decision immediately so a crash loses nothing logged."""
    con = open_log(db_path)
    results = []
    try:
        for item in items:
            record = engine.process(item)
            record["run_id"] = run_id
            persist(con, record, run_id)
            results.append(record)
        logged = count_run(con, run_id)
    finally:
        con.close()
    return results, logged
