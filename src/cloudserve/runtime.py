"""Shared construction of the decision engine and batch runner (used by CLI and harness)."""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from . import ROOT
from .classify import NaiveBayesIntentClassifier
from .graph import DecisionEngine, auto_send_enabled_from_env
from .logging_store import count_run, open_log, persist
from .ingest import load_json
from .retrieve import DEFAULT_MODEL_DIR, EmbeddingModelUnavailable, KnowledgeBase, OnnxSentenceEmbedder
from .route import DEFAULT_POLICY_PATH, load_policy


def load_env() -> None:
    """Read .env without printing it; OS environment variables take precedence."""
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
    except ImportError:
        pass


def add_engine_arguments(p: argparse.ArgumentParser, llm_cache_default: str = "off") -> None:
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
    p.add_argument("--answer-mode", choices=["extractive", "generated"], default="extractive",
                   help="extractive = verbatim resolution quote (offline default); generated = PR-02 cited "
                        "sentences written by the model through --provider, checked by validate_generated")
    p.add_argument("--provider", choices=["ollama", "llamacpp", "anthropic", "openai"], default=None,
                   help="model backend (default: CLOUDSERVE_PROVIDER or ollama)")
    p.add_argument("--model", default=None, help="model name (default: CLOUDSERVE_MODEL or provider default)")
    p.add_argument("--base-url", default=None, help="local model server URL (ollama/llamacpp only)")
    p.add_argument("--llm-cache", choices=["read_write", "read_only", "off"], default=llm_cache_default,
                   help=f"LLM response cache under storage/llm_cache (default here: {llm_cache_default}; "
                        "the evaluation harness defaults to read_write for reproducible runs, the CLI to off)")
    p.add_argument("--auto-send", action="store_true",
                   help="release automatic answers (kill switch off); also CLOUDSERVE_AUTO_SEND=true")


def build_knowledge_base(docs: list[dict], policy: dict, model_dir=None) -> tuple[KnowledgeBase, dict, dict]:
    """Build the retriever the policy was derived for; return (kb, effective policy, retrieval info).

    If the policy names the embedding retriever and the model cannot be loaded, fall back to the
    policy's lexical fallback with ITS thresholds, and say so in every record (never silently).
    """
    cfg = policy.get("retrieval") or {"retriever": "lexical", "chunking": "article"}
    if cfg["retriever"] == "lexical":
        kb = KnowledgeBase(docs, chunking=cfg["chunking"], retriever="lexical")
        return kb, policy, {**kb.describe(), "fallback": False}
    try:
        embedder = OnnxSentenceEmbedder(model_dir or os.getenv("CLOUDSERVE_EMBEDDING_MODEL_DIR") or DEFAULT_MODEL_DIR)
        kb = KnowledgeBase(docs, chunking=cfg["chunking"], retriever="embedding", embedder=embedder)
        return kb, policy, {**kb.describe(), "fallback": False}
    except EmbeddingModelUnavailable as exc:
        fb = policy.get("retrieval_fallback")
        if not fb:
            raise SystemExit(f"Embedding retriever unavailable ({exc}) and the policy defines no fallback")
        kb = KnowledgeBase(docs, chunking=fb["chunking"], retriever="lexical")
        effective = {**policy, "classification_confidence_min": fb["classification_confidence_min"],
                     "retrieval_score_min": fb["retrieval_score_min"]}
        print(f"WARNING: embedding retriever unavailable ({exc}); using the lexical fallback", file=sys.stderr)
        return kb, effective, {**kb.describe(), "fallback": True, "fallback_reason": str(exc),
                               "intended_retriever": cfg.get("name", cfg["retriever"])}


def build_engine(a: argparse.Namespace) -> DecisionEngine:
    load_env()
    provider = a.provider or os.getenv("CLOUDSERVE_PROVIDER", "ollama")
    if a.base_url and provider in ("anthropic", "openai"):
        raise SystemExit("--base-url is only supported for local providers (ollama, llamacpp)")
    kb, policy, retrieval_info = build_knowledge_base(load_json(a.docs), load_policy(a.policy))
    try:
        return _build(a, provider, kb, policy, retrieval_info)
    except Exception as exc:
        from .providers import ModelUnavailable
        if isinstance(exc, ModelUnavailable):   # e.g. missing API key: clear message, no traceback
            raise SystemExit(f"Model provider not usable: {exc}. Use --classifier offline to run without a model.")
        if isinstance(exc, ValueError) and str(exc).startswith("CLOUDSERVE_"):   # bad numeric setting
            raise SystemExit(f"Configuration error: {exc}")
        raise


def _build(a, provider, kb, policy, retrieval_info) -> DecisionEngine:
    answer_mode = getattr(a, "answer_mode", "extractive")
    if answer_mode == "generated" and a.evidence_selector == "llm":
        raise SystemExit("--answer-mode generated replaces evidence selection; drop --evidence-selector llm")
    guard = calls = None
    if a.classifier == "llm" or a.evidence_selector == "llm" or answer_mode == "generated":
        from .model_calls import CallLog, LLMCache
        from .providers import ProviderGuard
        guard = ProviderGuard()       # one circuit breaker shared by every model call in this run
        calls = CallLog(LLMCache(getattr(a, "llm_cache", "off")))   # shared cache, usage and cost log
    if a.classifier == "llm":
        from .classify import LangChainClassifier
        classifier = LangChainClassifier(provider=provider, model_name=a.model, base_url=a.base_url, guard=guard, calls=calls)
    else:
        classifier = NaiveBayesIntentClassifier.from_development_file(a.train)
    selector = None
    if a.evidence_selector == "llm":
        from .generate import LangChainEvidenceSelector
        selector = LangChainEvidenceSelector(provider=provider, model_name=a.model, base_url=a.base_url, guard=guard, calls=calls)
    generator = None
    if answer_mode == "generated":    # needs an LLM provider; build_model raises ModelUnavailable otherwise
        from .generate import LangChainAnswerGenerator
        generator = LangChainAnswerGenerator(provider=provider, model_name=a.model, base_url=a.base_url, guard=guard, calls=calls)
    return DecisionEngine(kb, policy, classifier, selector,
                          auto_send=a.auto_send or auto_send_enabled_from_env(),
                          use_langgraph=a.engine == "graph", retrieval_info=retrieval_info,
                          answer_mode=answer_mode, generator=generator)


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
