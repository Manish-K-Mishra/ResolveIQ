"""One place every model call goes through: response cache, token usage and cost (A5, R-08).

    call_model(adapter, component, messages) -> validated schema object

* Cache (A5 in LLM mode): `storage/llm_cache/<k[:2]>/<k>.json`, key = sha256 of provider, model,
  prompt id+version, the rendered messages and the output schema (name + JSON schema). The key
  material holds no credentials, base URLs or environment values. A cache file stores only the
  validated output and the usage reported when it was made, never the messages. Modes:
  read_write (lookup, store misses), read_only (lookup, never store), off.
* Usage: tokens from the provider's usage_metadata (`StructuredResult`), cost from
  config/model_prices.json. Every call is appended to a `CallLog` that the engine drains per
  ticket into the decision record, with `cache: hit|miss|off`, so a replayed answer is never
  reported as live latency or live spend.
"""
from __future__ import annotations
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from . import ROOT
from .providers import HOSTED_PROVIDERS

CACHE_MODES = ("read_write", "read_only", "off")
DEFAULT_CACHE_DIR = ROOT / "storage" / "llm_cache"
PRICES_PATH = ROOT / "config" / "model_prices.json"
LOCAL_PROVIDERS = ("ollama", "llamacpp")


def load_prices(path: str | Path | None = None) -> dict:
    return json.loads(Path(path or PRICES_PATH).read_text(encoding="utf-8"))


def price_for(prices: dict, provider: str, model: str) -> dict | None:
    if provider in prices.get("free_local_providers", LOCAL_PROVIDERS):
        return {"input_per_million": 0.0, "output_per_million": 0.0}
    return prices.get("models", {}).get(f"{provider}:{model}")


def cost_usd(price: dict | None, input_tokens, output_tokens) -> float | None:
    if price is None or input_tokens is None or output_tokens is None:
        return None
    return round(input_tokens / 1e6 * price["input_per_million"] + output_tokens / 1e6 * price["output_per_million"], 8)


def _schema_fingerprint(schema) -> str:
    js = schema.model_json_schema() if hasattr(schema, "model_json_schema") else {}
    return f"{getattr(schema, '__name__', 'schema')}:{hashlib.sha256(json.dumps(js, sort_keys=True).encode()).hexdigest()[:16]}"


def cache_key(provider: str, model: str, prompt_label: str, messages: list, schema) -> str:
    material = {"provider": provider, "model": model, "prompt": prompt_label, "schema": _schema_fingerprint(schema),
                "messages": [[getattr(m, "type", type(m).__name__), getattr(m, "content", str(m))] for m in messages]}
    return hashlib.sha256(json.dumps(material, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


class LLMCache:
    def __init__(self, mode: str = "read_write", directory: str | Path | None = None):
        if mode not in CACHE_MODES:
            raise ValueError(f"cache mode must be one of {CACHE_MODES}")
        self.mode = mode
        self.dir = Path(directory or os.getenv("CLOUDSERVE_LLM_CACHE_DIR") or DEFAULT_CACHE_DIR)

    def _path(self, key: str) -> Path:
        return self.dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        if self.mode == "off":
            return None
        try:
            return json.loads(self._path(key).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def put(self, key: str, entry: dict) -> None:
        if self.mode != "read_write":
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)          # atomic: a crashed run never leaves a half-written entry


class CallLog:
    """Shared by the run's model adapters; the engine drains it once per ticket."""

    def __init__(self, cache: LLMCache | None = None, prices: dict | None = None):
        self.cache = cache
        self.prices = prices if prices is not None else load_prices()
        # Per thread: concurrent API requests each run a ticket on their own worker thread, so a
        # thread's entries are exactly its current ticket's calls.
        self._local = threading.local()

    @property
    def entries(self) -> list[dict]:
        if not hasattr(self._local, "entries"):
            self._local.entries = []
        return self._local.entries

    def drain(self) -> list[dict]:
        out, self._local.entries = self.entries, []
        return out


def _usage_tokens(usage: dict | None):
    if not usage:
        return None, None
    return usage.get("input_tokens"), usage.get("output_tokens")


def call_model(adapter, component: str, messages: list):
    """Run one structured model call for `adapter` (classifier, selector or generator)."""
    log = getattr(adapter, "calls", None)
    log = log if isinstance(log, CallLog) else None
    cache = log.cache if log else None
    key = cache_key(adapter.provider, adapter.model_name, adapter.version, messages, adapter.schema) if cache else None
    hit = cache.get(key) if cache else None
    t0 = time.perf_counter()
    if hit is not None:
        validated, usage, status = adapter.schema.model_validate(hit["output"]), hit.get("usage"), "hit"
    else:
        result = adapter.guard.call(adapter.runnable.invoke, messages)
        parsed, usage = (result.parsed, result.usage) if hasattr(result, "parsed") and hasattr(result, "usage") else (result, None)
        validated = adapter.schema.model_validate(parsed.model_dump() if hasattr(parsed, "model_dump") else parsed)
        status = "miss" if cache and cache.mode != "off" else "off"
        if cache:
            cache.put(key, {"output": validated.model_dump(), "usage": usage, "prompt": adapter.version,
                            "provider": adapter.provider, "model": adapter.model_name})
    if log is not None:
        tin, tout = _usage_tokens(usage)
        price = price_for(log.prices, adapter.provider, adapter.model_name)
        cost = cost_usd(price, tin, tout)
        log.entries.append({
            "component": component, "prompt": adapter.version, "provider": adapter.provider,
            "model": adapter.model_name, "cache": status, "input_tokens": tin, "output_tokens": tout,
            # A cache hit costs nothing now; cost_if_live is what the same call costs when live.
            "cost_usd": 0.0 if status == "hit" else cost, "cost_if_live_usd": cost,
            "price_known": price is not None, "latency_seconds": round(time.perf_counter() - t0, 4),
            "pii_masked_for_provider": adapter.provider in HOSTED_PROVIDERS})
    return validated


def summarise(entries: list[dict]) -> dict:
    """Per-decision usage block."""
    def total(k):
        vals = [e[k] for e in entries]
        return None if any(v is None for v in vals) else round(sum(vals), 8) if vals else 0
    hits = sum(e["cache"] == "hit" for e in entries)
    return {"model_calls": len(entries), "live_calls": len(entries) - hits, "cache_hits": hits,
            "input_tokens": total("input_tokens"), "output_tokens": total("output_tokens"),
            "estimated_cost_usd": total("cost_usd"), "cost_if_live_usd": total("cost_if_live_usd")}


def cache_status(entries: list[dict]) -> str:
    if not entries:
        return "not_used"
    kinds = {e["cache"] for e in entries}
    return "hit" if kinds == {"hit"} else "mixed" if "hit" in kinds else "miss" if "miss" in kinds else "off"
