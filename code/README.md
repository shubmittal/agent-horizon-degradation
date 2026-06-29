# Agent-rot harness — long-horizon degradation in LLM agents

Reproducible code for *How Fast Do Agents Rot? An Empirical Study of Long-Horizon
Degradation in LLM Agents.* Three synthetic, oracle-verifiable agent task families
with a parametric horizon, run under three context regimes that disentangle
step-count from context-length, across a ladder of open models.

## What's here

| File | Role |
|------|------|
| `config.py` | model registry, design grid, paths, backend switch |
| `llm.py` | async OpenAI-compatible client (vLLM **or** OpenRouter), caching, cost meter |
| `tasks.py` | the three task families (`ledger`, `refchain`, `cipher`) + oracles |
| `runner.py` | ReAct-style trajectory runner, regime/context handling, per-step scoring, checkpointing, sweep CLI |
| `analysis.py` | cell tables (Wilson CIs), decay-curve fits + model selection, per-step hazard, regime-contrast disentangling, benchmark-gap projection, figures |
| `test_harness.py` | offline validation with a mock perfect/faulty agent (no GPU/key) |
| `build_notebook.py` | regenerates the one-click Colab notebook from these modules |

## How it runs

The harness is **OpenAI-API-compatible**, so inference comes from either backend
(`AGENT_ROT_BACKEND`):

- **`vllm`** (default) — a local vLLM OpenAI server. This is the **Google Colab**
  path: open `../agent_rot_colab.ipynb`, pick a T4 GPU, *Run all*. Results
  checkpoint to Google Drive and resume across sessions.
- **`openrouter`** — a hosted aggregator. Copy `.env.example` to `.env`, paste an
  OpenRouter key, then run `runner.py` locally.

## Quick checks (no GPU, no key)

```bash
pip install -r requirements.txt
python tasks.py          # oracle self-test for all families
python test_harness.py   # end-to-end harness validation with a mock agent
```

## Running a sweep

```bash
# pilot: 8 trials/cell across all models x families x horizons x regimes
python runner.py --pilot
# full run
python runner.py --trials 40
# a single model/family while iterating
python runner.py --models llama1b --families ledger --regimes natural
python analysis.py       # writes results/summary.json, *.csv, and ../figures/*.png
```

Sweeps are **resumable**: each trajectory is checkpointed by
`(model, family, regime, horizon, trial)` in `results/raw/*.jsonl` and re-runs skip
completed cells.

## Design

Horizon `H` ∈ {2,4,8,16,32} (number of dependent steps). Regimes: `natural`
(H turns, long context), `compressed` (H turns, bounded context via carried state),
`padded` (1 turn, all H ops in one long prompt). The paired regime contrast at
fixed H isolates context-length from the multi-step process. See `../DESIGN.md`
for the full pre-registration (hypotheses, power, analysis plan).

> Reproducibility: temperature fixed at 0.2, per-trajectory seeds recorded; residual
> provider/GPU non-determinism is noted in the paper. No fabricated data — every
> number traces to `results/raw/`.
