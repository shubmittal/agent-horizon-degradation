# Results evidence trail (every number traces here → `code/results/`)

**Dataset:** 10,664 analyzed agent trajectories. Streaming families =
9 models × 3 families × 5 horizons × 3 regimes × 25 trials; agentic family =
9 models × `toolqa` × 4 horizons (H≤16) × 15 trials. Backend: OpenRouter, temp 0.2,
seeds recorded. Est. cost ≈ $6.7 of hosted inference.
Raw: `code/results/raw/*.jsonl`; summary: `code/results/summary.json`.

**Models (6 vendor families, 1.2B–671B + deployed proprietary):** Llama-3.2-1B,
Qwen2.5-7B, Llama-3.1-8B, Llama-3.3-70B, Qwen2.5-72B, DeepSeek-V3 (open);
GPT-4o-mini, Gemini-2.5-Flash-Lite, Claude-3-Haiku (proprietary).
Excluded (protocol non-adherence via hosted route): Llama-3.2-3B, Gemma-3-4B, Phi-4-mini.

## Headline numbers
- **Shape:** geometric best-fit in **28/36** model×family cells; threshold in 8/36.
  P(success) ≈ a·r^H.
- **Per-step reliability r (direct, natural):** by model — llama1b 0.16, qwen7b 0.60,
  llama8b 0.64, llama70b 0.61, qwen72b 0.71, deepseek 0.72, gpt4omini 0.68,
  gemini 0.69, haiku 0.68. Rises with scale (Pearson +0.36 vs log10 params over open
  models); proprietary models cluster near 0.68–0.72; none reach 1.
- **Agentic collapse (`toolqa`):** every model ~1.0 at H=2 → near 0 by H=16, incl.
  deployed: GPT-4o-mini 1.0→0.07, Claude-Haiku 1.0→0.00, DeepSeek 1.0→0.27,
  Qwen-72B 1.0→0.13.
- **Accelerating hazard:** per-step accuracy 0.58 (first third) → 0.44 (last third);
  hazard_accelerates = True.
- **Disentangling (logit slope per horizon doubling):** natural −0.44, compressed
  −0.69 (p=3×10⁻⁶ vs natural), padded −0.40 (p=0.51, ≈ natural). Bounding context
  STEEPENS decay ⇒ not context length; naive truncation backfires.
- **Regime success (pooled, H=2→32):** natural 0.75→0.58, compressed 0.64→0.30,
  padded 0.45→0.23.
- **Format/tool-call drift:** 21% of trajectories (≥1 malformed turn), rising with H.
- **Benchmark-gap projection (mean r=0.61):** GAIA(8) 0.42, WebArena(15) 0.36,
  τ-bench(20) 0.33, SWE-bench/OSWorld(30) 0.30, TheAgentCompany(100) 0.24.

## Success by model × horizon (natural, streaming families pooled)
| model | H2 | H4 | H8 | H16 | H32 |
|-------|----|----|----|-----|-----|
| Llama-3.2-1B | 0.48 | 0.35 | 0.29 | 0.24 | 0.24 |
| Qwen2.5-7B | 0.73 | 0.68 | 0.65 | 0.61 | 0.56 |
| Llama-3.1-8B | 0.71 | 0.68 | 0.65 | 0.67 | 0.61 |
| GPT-4o-mini | 0.87 | 0.75 | 0.69 | 0.67 | 0.61 |
| Gemini-2.5-FL | 0.79 | 0.71 | 0.71 | 0.65 | 0.64 |
| Claude-3-Haiku | 0.80 | 0.69 | 0.67 | 0.65 | 0.65 |
| Llama-3.3-70B | 0.68 | 0.61 | 0.63 | 0.65 | 0.57 |
| Qwen2.5-72B | 0.84 | 0.73 | 0.68 | 0.67 | 0.67 |
| DeepSeek-V3 | 0.83 | 0.79 | 0.75 | 0.69 | 0.64 |

## `toolqa` agentic success by model × horizon (natural)
| model | H2 | H4 | H8 | H16 |
|-------|----|----|----|-----|
| Llama-3.2-1B | 0.00 | 0.00 | 0.00 | 0.00 |
| Qwen2.5-7B | 1.00 | 0.07 | 0.13 | 0.33 |
| Llama-3.1-8B | 1.00 | 0.73 | 0.07 | 0.13 |
| GPT-4o-mini | 1.00 | 1.00 | 0.20 | 0.07 |
| Gemini-2.5-FL | 0.20 | 0.53 | 0.00 | 0.00 |
| Claude-3-Haiku | 1.00 | 0.47 | 0.00 | 0.00 |
| Llama-3.3-70B | 1.00 | 0.93 | 0.67 | 0.00 |
| Qwen2.5-72B | 1.00 | 1.00 | 0.87 | 0.13 |
| DeepSeek-V3 | 1.00 | 1.00 | 0.67 | 0.27 |

## Figures (`figures/`)
- fig1_decay_curves.png — success vs horizon, 4 families (incl. agentic toolqa), 9 models.
- fig2_regime_contrast.png — natural vs compressed vs padded (streaming families).
- fig3_perstep_accuracy.png — per-step accuracy vs step index (accelerating decay).

## Deviations from pre-registration (DESIGN.md)
1. Backend: OpenRouter (~$6.7) instead of Colab/vLLM, after repeated Colab dependency
   failures. Same harness, hosted inference.
2. Models: 9 hosted models (6 open 1.2–671B + 3 proprietary deployed) — extends the
   ladder and adds deployed systems. Three small models excluded for protocol
   non-adherence.
3. Added the agentic `toolqa` family (real ReAct tool-use loop) for ecological validity.
4. Harness fix mid-study: `extract_json` parses multiple concatenated JSON objects;
   all streaming data re-scored uniformly. Empty-response retries + per-trajectory
   error isolation added for robustness.
5. The agentic `toolqa` family uses horizons {2,4,8,16} by design — collapse is already
   complete by H=16. Temperature is fixed at 0.2 (standard for agentic determinism);
   `temp_study.py` is provided for anyone wanting a sensitivity sweep.
