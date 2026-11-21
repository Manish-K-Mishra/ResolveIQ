"""Model-independent LLM adapters (Ollama, llama.cpp, Anthropic Claude, OpenAI).

Every provider gets the same prompt (PR-01 from prompts/), the same strict schema, a bounded
request timeout and bounded retries. A run-level circuit breaker (ProviderGuard) stops calling
a provider that keeps failing, so an outage degrades the run instead of stalling it (A11).
Model output never makes routing decisions; cloudserve/routing.py does.
"""
from __future__ import annotations
import os
import re
import threading
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from .classifier import INTENTS, UNKNOWN

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
PROVIDERS = ("ollama", "llamacpp", "anthropic", "openai")
DEFAULT_MODELS = {
    "ollama": "gemma4:e2b",        # the model verified on the build machine; use `ollama list`
    "llamacpp": "local-model",
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-4o-mini",
}
IntentLabel = Literal[INTENTS + (UNKNOWN,)]


class Classification(BaseModel):
    """PR-01 v1.1 output contract: intent restricted to the 22 dataset intents or 'unknown'."""
    model_config = ConfigDict(extra="forbid")
    intent: IntentLabel
    urgency: Literal["low", "medium", "high"]
    confidence: float = Field(ge=0, le=1)
    alternatives: list[IntentLabel] = Field(default_factory=list, max_length=3)
    reason: str = Field(min_length=1, max_length=400)


class ModelUnavailable(RuntimeError):
    pass


class ProviderCircuitOpen(ModelUnavailable):
    pass


def load_prompt(filename: str) -> dict:
    """Parse a versioned prompt file: YAML-like front matter, then '## system' / '## user'."""
    text = (PROMPT_DIR / filename).read_text(encoding="utf-8")
    _, front, body = text.split("---", 2)
    meta = dict(re.findall(r'(?m)^(id|version):\s*"?([^"\n]+)"?\s*$', front))
    sections = dict(re.findall(r"(?ms)^## (system|user)\n(.*?)(?=^## |\Z)", body))
    return {"id": meta["id"], "version": meta["version"], "label": f"{meta['id']} v{meta['version']}",
            "system": sections["system"].strip(), "user": sections["user"].strip()}


def _env_number(name: str, default: float, kind=float, minimum: float = 0.0, strict: bool = True):
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return kind(default)
    try:
        value = kind(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None
    if value < minimum or (strict and value == minimum):
        raise ValueError(f"{name} must be {'>' if strict else '>='} {minimum}, got {raw!r}")
    return value


def runtime_settings() -> dict:
    timeout = _env_number("CLOUDSERVE_TIMEOUT_SECONDS", 30)
    retries = _env_number("CLOUDSERVE_MAX_RETRIES", 2, int, strict=False)
    # Hard wall-clock limit per model call, covering every retry. The per-request timeout alone
    # is per-read: a server that streams slowly would never trip it.
    deadline = _env_number("CLOUDSERVE_CALL_DEADLINE_SECONDS", timeout * (retries + 1) + 5)
    return {"timeout": timeout, "max_retries": retries, "deadline": deadline}


def build_model(provider: str, model_name: str | None = None, base_url: str | None = None,
                timeout: float | None = None, max_retries: int | None = None):
    """Create a chat model lazily (no network call here) with bounded timeout and retries."""
    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported provider {provider!r}; choose from {PROVIDERS}")
    s = runtime_settings()
    timeout = s["timeout"] if timeout is None else timeout
    max_retries = s["max_retries"] if max_retries is None else max_retries
    # CLOUDSERVE_MODEL belongs to CLOUDSERVE_PROVIDER; never send e.g. a Claude model name to Ollama.
    env_model = os.getenv("CLOUDSERVE_MODEL") if os.getenv("CLOUDSERVE_PROVIDER", "ollama") == provider else None
    name = model_name or env_model or DEFAULT_MODELS[provider]
    try:
        if provider == "ollama":
            from langchain_ollama import ChatOllama
            # reasoning=False: no hidden "thinking" tokens, which only add latency here.
            # num_predict caps output length so a runaway generation cannot run unbounded.
            return ChatOllama(model=name, temperature=0, reasoning=False, num_predict=512,
                              base_url=base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                              client_kwargs={"timeout": timeout}), name, max_retries
        if provider == "llamacpp":
            from langchain_openai import ChatOpenAI
            url = base_url or os.getenv("LLAMACPP_BASE_URL", "http://localhost:8080/v1")
            # Placeholder key for the local OpenAI-compatible transport only (not a secret).
            return ChatOpenAI(model=name, base_url=url, api_key="local-no-key", temperature=0,
                              timeout=timeout, max_retries=max_retries), name, 0
        if provider == "anthropic":
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise ModelUnavailable("ANTHROPIC_API_KEY is required for Claude (set it in .env)")
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=name, temperature=0, timeout=timeout,
                                 max_retries=max_retries), name, 0
        if not os.getenv("OPENAI_API_KEY"):
            raise ModelUnavailable("OPENAI_API_KEY is required for OpenAI")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=name, temperature=0, timeout=timeout, max_retries=max_retries), name, 0
    except ImportError as exc:
        raise ModelUnavailable("Missing model packages: pip install -r requirements.txt") from exc


def structured(model, schema, provider: str, extra_retries: int):
    """Structured-output runnable; Ollama has no SDK retry, so add a bounded LangChain retry."""
    method = "json_schema" if provider in ("ollama", "llamacpp") else "function_calling"
    runnable = model.with_structured_output(schema, method=method)
    if extra_retries > 0:
        runnable = runnable.with_retry(stop_after_attempt=extra_retries + 1, wait_exponential_jitter=True)
    return runnable


class ProviderDeadlineExceeded(ModelUnavailable):
    pass


def call_with_deadline(fn, deadline: float, *args, **kwargs):
    """Run fn in a daemon thread and give up after `deadline` seconds.

    A daemon thread is used (not ThreadPoolExecutor, whose workers are joined at interpreter
    exit) so an abandoned call can never keep the process alive after the run ends.
    """
    box: dict = {}

    def target():
        try:
            box["result"] = fn(*args, **kwargs)
        except BaseException as exc:      # re-raised in the caller's thread
            box["error"] = exc
    worker = threading.Thread(target=target, daemon=True, name="cloudserve-model-call")
    worker.start()
    worker.join(deadline)
    if worker.is_alive():
        raise ProviderDeadlineExceeded(f"model call exceeded {deadline:.0f}s wall-clock deadline")
    if "error" in box:
        raise box["error"]
    return box["result"]


class ProviderGuard:
    """Run-level circuit breaker plus wall-clock deadline for every model call.

    After N consecutive failures (including deadline overruns) the provider is not called again
    in this run; tickets then escalate immediately, so an outage cannot stall the batch.
    """

    def __init__(self, max_consecutive_failures: int = 3, deadline: float | None = None):
        self.max_failures = max_consecutive_failures
        self.deadline = runtime_settings()["deadline"] if deadline is None else deadline
        self.consecutive_failures = 0
        self.is_open = False
        self.total_failures = 0

    def call(self, fn, *args, **kwargs):
        if self.is_open:
            raise ProviderCircuitOpen("provider disabled for this run after repeated failures")
        try:
            result = call_with_deadline(fn, self.deadline, *args, **kwargs)
        except Exception:
            self.consecutive_failures += 1
            self.total_failures += 1
            if self.consecutive_failures >= self.max_failures:
                self.is_open = True
            raise
        self.consecutive_failures = 0
        return result


class LangChainClassifier:
    """PR-01 classification through any provider; same contract as the offline classifier."""

    def __init__(self, provider: str | None = None, model_name: str | None = None,
                 base_url: str | None = None, model=None, guard: ProviderGuard | None = None):
        self.provider = provider or os.getenv("CLOUDSERVE_PROVIDER", "ollama")
        self.prompt = load_prompt("PR-01_classification.md")
        self.guard = guard or ProviderGuard()
        if model is not None:                       # injected (tests): used as-is
            self.runnable, self.model_name = model, model_name or "injected"
        else:
            chat, self.model_name, extra = build_model(self.provider, model_name, base_url)
            self.runnable = structured(chat, Classification, self.provider, extra)
        self.name = f"{self.provider}:{self.model_name}"
        self.version = self.prompt["label"]

    def messages(self, ticket) -> list:
        from langchain_core.messages import HumanMessage, SystemMessage
        system = self.prompt["system"].replace("{allowed_intents}", ", ".join(INTENTS + (UNKNOWN,)))
        user = (self.prompt["user"].replace("{ticket_id}", ticket.ticket_id)
                .replace("{channel}", ticket.channel)
                .replace("{subject}", ticket.subject[:300])
                .replace("{body}", ticket.body[:4000]))
        return [SystemMessage(content=system), HumanMessage(content=user)]

    def classify(self, ticket) -> dict:
        result = self.guard.call(self.runnable.invoke, self.messages(ticket))
        v = Classification.model_validate(result.model_dump() if hasattr(result, "model_dump") else result)
        alts = [a for a in dict.fromkeys(v.alternatives) if a != v.intent]
        return {"intent": v.intent, "urgency": v.urgency, "confidence": v.confidence,
                # The model names alternatives but gives no per-alternative score.
                "alternatives": [{"intent": a, "confidence": None} for a in alts],
                "reason": v.reason, "confidence_type": "uncalibrated_model_score"}
