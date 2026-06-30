"""Central configuration: model registry, design grid, paths.

Backend is selected by the AGENT_ROT_BACKEND env var:
  * "vllm"  (default)  -> local vLLM OpenAI-compatible server (Google Colab).
                          Inference is free; `price_*` fields are 0 and used only
                          to keep the cost meter uniform.
  * "openrouter"       -> hosted aggregator. `price_*` are approximate list prices
                          (USD / 1M tokens) for the running cost estimate only;
                          the authoritative cost is whatever the host bills.

The harness is OpenAI-API-compatible, so the same task/runner/analysis code runs
against either backend; only the base URL + key + model `route` differ.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- paths
# AGENT_ROT_OUT lets Colab redirect results/cache to a persistent Google Drive
# folder so checkpoints survive session restarts. Defaults to the repo dir.
ROOT = Path(__file__).resolve().parent
OUT = Path(os.environ.get("AGENT_ROT_OUT", str(ROOT)))
RESULTS_DIR = OUT / "results"
RAW_DIR = RESULTS_DIR / "raw"           # one JSONL per (model,family,regime)
CACHE_DIR = OUT / ".cache"              # response cache
FIG_DIR = Path(os.environ.get("AGENT_ROT_OUT", str(ROOT.parent))) / "figures" \
    if os.environ.get("AGENT_ROT_OUT") else ROOT.parent / "figures"
for _d in (RESULTS_DIR, RAW_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- backend
BACKEND = os.environ.get("AGENT_ROT_BACKEND", "vllm").lower()


# ---------------------------------------------------------------- models
@dataclass(frozen=True)
class Model:
    key: str                 # short name used in our data
    route: str               # backend model id (HF repo for vLLM; route for hosts)
    params_b: float          # parameter count in billions (the capability axis)
    vendor: str              # family, for the generality claim
    price_in: float          # USD / 1M prompt tokens (0 for local vLLM)
    price_out: float         # USD / 1M completion tokens (0 for local vLLM)
    quant: str = ""          # vLLM quantization flag ("awq" etc.); "" = fp16


# Colab/vLLM ladder: UNGATED public open models that fit a single T4 (16 GB) and
# need NO Hugging Face token or license acceptance (run-as-is). The <=3.8B run in
# fp16; the 7B uses a pre-quantized AWQ-INT4 checkpoint to fit with an 8k KV cache.
# A 0.5B->7B capability ladder; Qwen gives a clean within-family ladder and Phi a
# cross-family point. `route` is the HuggingFace repo id served by the vLLM server.
_VLLM_MODELS = [
    Model("qwen0_5b", "Qwen/Qwen2.5-0.5B-Instruct",        0.5, "Alibaba",   0.0, 0.0),
    Model("qwen1_5b", "Qwen/Qwen2.5-1.5B-Instruct",        1.5, "Alibaba",   0.0, 0.0),
    Model("phi3_5",   "microsoft/Phi-3.5-mini-instruct",   3.8, "Microsoft", 0.0, 0.0),
    Model("qwen7b",   "Qwen/Qwen2.5-7B-Instruct-AWQ",      7.6, "Alibaba",   0.0, 0.0, "awq"),
]

# Hosted ladder (OpenRouter): the full study runs here. A 1B->671B capability
# ladder across Meta / Alibaba / DeepSeek. Prices (USD/1M) are approximate, for the
# cost meter only; routes re-confirmed against the live /models list at runtime.
# Five models that reliably follow the structured JSON protocol, spanning 1B->671B
# across three vendor families. (Llama-3.2-3B and Gemma-3-4B / Phi-4-mini were
# evaluated but excluded: their OpenRouter routes do not adhere to the protocol —
# empty tool-call responses or conversational replies — see DEVIATIONS.)
_NAN = float("nan")
_OPENROUTER_MODELS = [
    # Open models, known parameter counts -> the capability-scaling ladder.
    Model("llama1b",  "meta-llama/llama-3.2-1b-instruct",  1.2,  "Meta",     0.005, 0.01),
    Model("qwen7b",   "qwen/qwen-2.5-7b-instruct",         7.6,  "Alibaba",  0.04,  0.10),
    Model("llama8b",  "meta-llama/llama-3.1-8b-instruct",  8.0,  "Meta",     0.02,  0.03),
    Model("llama70b", "meta-llama/llama-3.3-70b-instruct", 70.0, "Meta",     0.12,  0.30),
    Model("qwen72b",  "qwen/qwen-2.5-72b-instruct",        72.0, "Alibaba",  0.36,  0.40),
    Model("deepseek", "deepseek/deepseek-chat",            671.0,"DeepSeek",  0.30,  0.88),
    # Proprietary deployed models (parameter counts undisclosed -> NaN; included for
    # generality and the law, excluded from the params-scaling correlation).
    Model("gpt4omini","openai/gpt-4o-mini",                _NAN, "OpenAI",    0.15,  0.60),
    Model("gemini",   "google/gemini-2.5-flash-lite",      _NAN, "Google",    0.10,  0.40),
    Model("haiku",    "anthropic/claude-3-haiku",          _NAN, "Anthropic", 0.25,  1.25),
]

_ACTIVE = _OPENROUTER_MODELS if BACKEND == "openrouter" else _VLLM_MODELS
MODELS: dict[str, Model] = {m.key: m for m in _ACTIVE}

# Combined metadata for ALL models across both backends, so analysis can map any
# model key -> (params_b, vendor) for the capability-scaling view regardless of
# which backend produced the rows. (Same key in both lists => identical meta.)
MODEL_META: dict[str, Model] = {}
for _m in _VLLM_MODELS + _OPENROUTER_MODELS:
    MODEL_META.setdefault(_m.key, _m)

# Hybrid study: small open models (1B-8B) run free on Colab/vLLM; these 1-2 large
# models run on OpenRouter (~$5) to test whether the degradation law holds up the
# capability ladder. llama70b extends the Meta within-family ladder (1B/3B/8B/70B);
# deepseek adds a cross-family large point. Run via: AGENT_ROT_BACKEND=openrouter.
FRONTIER_MODELS = ["llama70b", "deepseek"]

# A small model used to debug the harness before the full sweep.
DEBUG_MODEL = "qwen0_5b" if BACKEND == "vllm" else "llama1b"

# ---------------------------------------------------------------- design grid
TASK_FAMILIES = ["ledger", "refchain", "cipher"]

# Horizon = number of dependent operations / agent steps.
HORIZONS = [2, 4, 8, 16, 32]

# Regimes that disentangle step-count from context-length (see DESIGN.md §3).
#   natural    : N agent turns, full history accumulates (steps + context grow).
#   compressed : N agent turns, per-turn context bounded (isolates step count).
#   padded     : few turns, context inflated to a high-step token budget
#                (isolates context length).
REGIMES = ["natural", "compressed", "padded"]

PILOT_TRIALS = 8
FULL_TRIALS = 25   # lean full sweep: 4 models x 3 families x 5 horizons x 3 regimes x 25

# Generation params (fixed for the main run; one temp-sensitivity sub-check later).
TEMPERATURE = 0.2
TOP_P = 1.0
MAX_COMPLETION_TOKENS = 1024
# A trajectory is capped at this many agent turns as a runaway guard. The cap is
# set generously above the largest horizon so it never truncates a healthy run.
MAX_TURNS_SLACK = 8   # max_turns = horizon + slack

# ---------------------------------------------------------------- concurrency
# Trajectories run concurrently and vLLM continuous-batches them on the GPU, so a
# high client-side concurrency keeps the batcher saturated (vLLM admission-controls
# its own queue, so over-subscription is safe). This is the main throughput lever.
MAX_CONCURRENCY = int(os.environ.get("AGENT_ROT_CONCURRENCY", "48"))
REQUEST_TIMEOUT = 120.0
MAX_RETRIES = 6
