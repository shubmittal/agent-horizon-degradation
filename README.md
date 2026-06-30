# How Fast Do Agents Rot? — Long-Horizon Degradation in LLM Agents

An empirical study measuring how LLM agents degrade as task **horizon** (number of
dependent steps) grows, and disentangling whether the cause is step count or context
length. Everything here is reproducible from the released code, prompts, seeds, and
raw trajectories.

**Paper:** [`paper/How_Fast_Do_Agents_Rot.docx`](paper/How_Fast_Do_Agents_Rot.docx)
(also `paper/main.tex` + `paper/references.bib`).
**Pre-registration:** [`DESIGN.md`](DESIGN.md). **Evidence trail:** [`RESULTS.md`](RESULTS.md).

## Headline findings (N = 10,664 trajectories; 9 models 1.2B–671B + deployed proprietary)
- Task success follows a **geometric law**, `P(success) ≈ a·r^H` (best fit in 28/36
  model×task cells), where `r` is a per-step reliability.
- `r` **rises with model scale** (Pearson +0.36 vs. log-params) and saturates near 0.7
  for strong open and deployed models — never reaching 1 on non-trivial tasks, so
  geometric compounding guarantees long-horizon collapse at every scale.
- **On a genuine agentic tool-use loop (`toolqa`), every model collapses from ~1.0 at
  H=2 to near 0 by H=16** — including GPT-4o-mini, Gemini, and Claude.
- Degradation **accelerates** within a trajectory (per-step accuracy 0.58 → 0.44).
- The driver is **step count, not context length**: bounding context *steepens* decay
  (p = 3×10⁻⁶) — the opposite of lost-in-the-middle. Naive context truncation backfires.
- **Benchmark gap:** projecting measured `r` onto benchmark horizons gives success
  0.42 (GAIA-length) → 0.30 (SWE-bench-length) → 0.24 (100-step production).

## Repository layout
```
code/            harness (tasks, runner, async LLM client, analysis) + results/
  tasks.py       3 synthetic oracle-verifiable task families (ledger/refchain/cipher)
  runner.py      ReAct-style trajectory runner, regimes, checkpointing, sweep CLI
  llm.py         async OpenAI-compatible client (OpenRouter), caching, cost meter
  analysis.py    curve fits, disentangling GLM, capability scaling, figures
  test_harness.py  offline validation with a mock agent (no API key needed)
  config.py      model registry, design grid, paths
  results/       summary.json, cell_table.csv, curve_fits.csv, raw/*.jsonl (all data)
figures/         the three paper figures
paper/           main.tex, references.bib, How_Fast_Do_Agents_Rot.docx
DESIGN.md        pre-registration (hypotheses, power, analysis plan)
RESULTS.md       every reported number → source
```

## Reproduce

### 0. Install
```bash
cd code
pip install -r requirements.txt
```

### 1. Sanity checks (no API key, no GPU)
```bash
python tasks.py          # oracle self-test for all three task families
python test_harness.py   # end-to-end harness validation with a mock agent
```

### 2. Re-run the study
Inference goes through an OpenAI-compatible endpoint. We used **OpenRouter**:
```bash
cp .env.example .env      # then paste your key: OPENROUTER_API_KEY=sk-or-...
AGENT_ROT_BACKEND=openrouter python runner.py --trials 25
AGENT_ROT_BACKEND=openrouter python analysis.py   # writes results/ + figures/
```
The sweep is **resumable** (checkpointed per `(model,family,regime,horizon,seed)`) and
**cached** (re-runs reuse stored responses). Full study ≈ 10,664 trajectories, ≈ $6.7 of
hosted inference. The agentic family is run separately:
`python runner.py --families toolqa --regimes natural --trials 15`. Temperature 0.2;
seeds recorded; residual provider non-determinism noted in the paper.

To reproduce the analysis/figures **without spending anything**, run step 2's
`analysis.py` directly on the released `code/results/raw/*.jsonl`.

## Models
Open (1.2B–671B): Llama-3.2-1B, Qwen2.5-7B, Llama-3.1-8B, Llama-3.3-70B, Qwen2.5-72B,
DeepSeek-V3. Proprietary deployed: GPT-4o-mini, Gemini-2.5-Flash-Lite, Claude-3-Haiku.
Six vendor families total. Three smaller candidates (Llama-3.2-3B, Gemma-3-4B,
Phi-4-mini) were excluded for not adhering to the structured protocol via their hosted
routes (see `RESULTS.md`).

## License / data
Released for reproducibility. Raw trajectories contain task instructions, model
outputs, and oracle scores only — no personal or sensitive data.
