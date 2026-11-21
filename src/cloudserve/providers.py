"""Model-independent LLM plumbing (Ollama, llama.cpp, Anthropic Claude, OpenAI).

Every model call uses a versioned prompt from prompts/build/, a strict schema, a bounded request
timeout and bounded retries. A run-level circuit breaker (ProviderGuard) stops calling a provider
that keeps failing, so an outage degrades the run instead of stalling it (A11).
Model output never makes routing decisions; route.py does.
"""
from __future__ import annotations
import os
import re
import threading
import time
from dataclasses import dataclass
from . import ROOT

PROMPT_DIR = ROOT / "prompts" / "build"
PROVIDERS = ("ollama", "llamacpp", "anthropic", "openai")
# Hosted providers receive PII-masked ticket text (Governance §4 private data); local ones run
# on this machine and receive the ticket as written.
HOSTED_PROVIDERS = ("anthropic", "openai")
DEFAULT_MODELS = {
    "ollama": "gemma4:e2b",        # the model verified on the build machine; use `ollama list`
    "llamacpp": "local-model",
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-4o-mini",
}


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
    cooldown = _env_number("CLOUDSERVE_BREAKER_COOLDOWN_SECONDS", 30, strict=False)
    return {"timeout": timeout, "max_retries": retries, "deadline": deadline, "breaker_cooldown": cooldown}


def for_provider(provider: str, text: str) -> str:
    """Ticket text as it may be sent to `provider`: PII-masked for hosted providers."""
    if provider in HOSTED_PROVIDERS:
        from .guardrails import mask_pii
        return mask_pii(text)
    return text


@dataclass(frozen=True)
class StructuredResult:
    """A parsed structured output plus the token usage the provider reported for it."""
    parsed: object
    usage: dict | None


def _unwrap(out):
    """include_raw=True output -> StructuredResult; parsing failures raise (and are retried)."""
    if out.get("parsing_error") is not None:
        raise out["parsing_error"]
    if out.get("parsed") is None:
        from langchain_core.exceptions import OutputParserException
        raise OutputParserException("the model returned no structured output")
    raw = out.get("raw")
    return StructuredResult(out["parsed"], dict(getattr(raw, "usage_metadata", None) or {}) or None)


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
    from langchain_core.runnables import RunnableLambda
    method = "json_schema" if provider in ("ollama", "llamacpp") else "function_calling"
    # include_raw keeps the provider message so its usage_metadata (tokens) can be recorded.
    runnable = model.with_structured_output(schema, method=method, include_raw=True) | RunnableLambda(_unwrap)
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
    """Run-level circuit breaker (closed / open / half-open) plus wall-clock deadline per call.

    After N consecutive failures (including deadline overruns) the breaker opens and tickets
    escalate immediately without calling the provider, so an outage cannot stall the batch.
    After `cooldown` seconds one trial call is allowed (half-open): success closes the breaker,
    failure re-opens it for another cooldown. So a recovered provider is used again in the run.
    """

    def __init__(self, max_consecutive_failures: int = 3, deadline: float | None = None,
                 cooldown: float | None = None, clock=time.monotonic):
        s = runtime_settings()
        self.max_failures = max_consecutive_failures
        self.deadline = s["deadline"] if deadline is None else deadline
        self.cooldown = s["breaker_cooldown"] if cooldown is None else cooldown
        self.clock = clock
        self.consecutive_failures = 0
        self.total_failures = 0
        self.state = "closed"
        self.opened_at = None
        self.trials = 0

    @property
    def is_open(self) -> bool:
        return self.state == "open"

    def call(self, fn, *args, **kwargs):
        if self.state == "open":
            if self.clock() - self.opened_at < self.cooldown:
                raise ProviderCircuitOpen("provider disabled after repeated failures (circuit open)")
            self.state, self.trials = "half_open", self.trials + 1     # one trial call
        try:
            result = call_with_deadline(fn, self.deadline, *args, **kwargs)
        except Exception:
            self.consecutive_failures += 1
            self.total_failures += 1
            if self.state == "half_open" or self.consecutive_failures >= self.max_failures:
                self.state, self.opened_at = "open", self.clock()
            raise
        self.consecutive_failures, self.state = 0, "closed"
        return result
