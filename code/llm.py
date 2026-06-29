"""Async LLM client for OpenRouter with response caching, retries, and cost
accounting. OpenRouter speaks the OpenAI Chat Completions API, so we use the
official `openai` async SDK pointed at the OpenRouter base URL.

A single shared AsyncOpenAI client + a global semaphore bound concurrency. Each
call is cached on disk keyed by a hash of (model, messages, params) so reruns and
resumes are free and deterministic-where-possible.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_random_exponential)

from config import (BACKEND, CACHE_DIR, MAX_CONCURRENCY, MAX_RETRIES, MODELS,
                    REQUEST_TIMEOUT)

load_dotenv(Path(__file__).resolve().parent / ".env")

if BACKEND == "openrouter":
    _API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
    _BASE_URL = "https://openrouter.ai/api/v1"
else:  # local vLLM OpenAI-compatible server (Colab). No real key needed.
    _API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")
    _BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")

_client: AsyncOpenAI | None = None
_sem: asyncio.Semaphore | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        if BACKEND == "openrouter" and not _API_KEY:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Copy code/.env.example to "
                "code/.env and paste your key (see README).")
        _client = AsyncOpenAI(
            api_key=_API_KEY, base_url=_BASE_URL, timeout=REQUEST_TIMEOUT,
            default_headers={
                "HTTP-Referer": os.environ.get("OPENROUTER_APP_URL", ""),
                "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "agent-rot"),
            },
        )
    return _client


def _get_sem() -> asyncio.Semaphore:
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(MAX_CONCURRENCY)
    return _sem


# --------------------------------------------------------------- cost tracking
@dataclass
class CostMeter:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usd: float = 0.0
    calls: int = 0
    cache_hits: int = 0

    def add(self, model_key: str, p_tok: int, c_tok: int) -> None:
        m = MODELS[model_key]
        self.prompt_tokens += p_tok
        self.completion_tokens += c_tok
        self.usd += p_tok / 1e6 * m.price_in + c_tok / 1e6 * m.price_out
        self.calls += 1

    def summary(self) -> str:
        return (f"calls={self.calls} cache_hits={self.cache_hits} "
                f"prompt_tok={self.prompt_tokens:,} "
                f"compl_tok={self.completion_tokens:,} "
                f"est_usd=${self.usd:.4f}")


METER = CostMeter()


# --------------------------------------------------------------- caching
def _cache_key(model_route: str, messages: list[dict], params: dict) -> str:
    payload = json.dumps(
        {"m": model_route, "msgs": messages, "p": params},
        sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def _cache_get(key: str) -> dict | None:
    p = _cache_path(key)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _cache_put(key: str, value: dict) -> None:
    try:
        _cache_path(key).write_text(
            json.dumps(value, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


# --------------------------------------------------------------- the call
class TransientLLMError(Exception):
    pass


@retry(retry=retry_if_exception_type(TransientLLMError),
       wait=wait_random_exponential(min=1, max=30),
       stop=stop_after_attempt(MAX_RETRIES), reraise=True)
async def _raw_call(model_route: str, messages: list[dict], params: dict) -> dict:
    client = _get_client()
    try:
        resp = await client.chat.completions.create(
            model=model_route, messages=messages, **params)
    except Exception as e:  # noqa: BLE001 - classify then re-raise
        msg = str(e).lower()
        transient = any(s in msg for s in (
            "rate limit", "429", "timeout", "timed out", "overloaded",
            "502", "503", "504", "connection", "temporarily"))
        if transient:
            raise TransientLLMError(str(e)) from e
        raise
    choice = resp.choices[0]
    usage = resp.usage
    return {
        "content": choice.message.content or "",
        "finish_reason": choice.finish_reason,
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
    }


async def chat(model_key: str, messages: list[dict], *, temperature: float,
               top_p: float, max_tokens: int, seed: int | None = None,
               use_cache: bool = True) -> dict:
    """One chat completion. Returns dict with content, finish_reason, usage.

    Cached on disk; cache hits cost nothing and are counted separately.
    """
    model = MODELS[model_key]
    params = {"temperature": temperature, "top_p": top_p,
              "max_tokens": max_tokens}
    if seed is not None:
        params["seed"] = seed
    key = _cache_key(model.route, messages, params)

    if use_cache:
        cached = _cache_get(key)
        if cached is not None:
            METER.cache_hits += 1
            return cached

    async with _get_sem():
        t0 = time.monotonic()
        out = await _raw_call(model.route, messages, params)
        out["latency_s"] = round(time.monotonic() - t0, 3)

    METER.add(model_key, out["prompt_tokens"], out["completion_tokens"])
    if use_cache:
        _cache_put(key, out)
    return out


async def ping(model_key: str = "llama8b") -> str:
    """Smoke test that the key + route work. Returns the model's reply text."""
    out = await chat(model_key,
                     [{"role": "user", "content": "Reply with exactly: OK"}],
                     temperature=0.0, top_p=1.0, max_tokens=8, use_cache=False)
    return out["content"].strip()
