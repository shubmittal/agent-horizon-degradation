# Results evidence trail (every number traces here → `code/results/`)

**Dataset:** 5,625 agent trajectories = 5 models × 3 families × 5 horizons × 3
regimes × 25 trials. Backend: OpenRouter. Temp 0.2, seeds recorded. Est. cost ≈ $3.
Raw: `code/results/raw/*.jsonl`; summary: `code/results/summary.json`.

**Models (1.2B–671B, 3 vendor families):** Llama-3.2-1B, Qwen2.5-7B,
Llama-3.1-8B, Llama-3.3-70B (Meta/Alibaba), DeepSeek-V3.
(Excluded: Llama-3.2-3B, Gemma-3-4B, Phi-4-mini — OpenRouter routes did not adhere
to the structured protocol; see DEVIATIONS below.)

## Headline numbers
- **Shape:** geometric best-fit in 10/15 (model×family) cells; threshold in 5/15.
  P(success) ≈ a·r^H.
- **Per-step reliability r (direct, natural regime):** median 0.74; by model —
  llama1b 0.16, qwen7b 0.60, llama8b 0.64, llama70b 0.61, deepseek 0.72.
- **Capability scaling:** Pearson(r, log10 params) = +0.38 (rises with scale, saturating).
- **Accelerating hazard:** per-step accuracy 0.49 (first third of trajectory) →
  0.35 (last third); `hazard_accelerates = True`.
- **Disentangling (logit slope per horizon doubling):** natural −0.37, compressed
  −0.62 (p=0.0007 vs natural), padded −0.53 (p=0.023). BOTH interventions are
  *steeper* than natural ⇒ neither flattens decay ⇒ degradation is intrinsic to
  step count, NOT context length (bounding context hurts).
- **Regime success (pooled, H=2→32):** natural 0.69→0.53, compressed 0.54→0.21,
  padded 0.46→0.22.
- **Format/tool-call drift (≥1 malformed turn):** 21% overall, rising with horizon.
- **Benchmark-gap projection (mean per-step r=0.55):** GAIA(8 steps) 0.31,
  WebArena(15) 0.24, τ-bench(20) 0.21, SWE-bench/OSWorld(30) 0.18,
  TheAgentCompany(100) 0.10.

## Success by model × horizon (natural)
| model | H2 | H4 | H8 | H16 | H32 |
|-------|----|----|----|-----|-----|
| llama1b  | 0.48 | 0.35 | 0.29 | 0.24 | 0.24 |
| qwen7b   | 0.73 | 0.68 | 0.65 | 0.61 | 0.56 |
| llama8b  | 0.71 | 0.68 | 0.65 | 0.67 | 0.61 |
| llama70b | 0.68 | 0.61 | 0.63 | 0.65 | 0.57 |
| deepseek | 0.83 | 0.79 | 0.75 | 0.69 | 0.64 |

## Per-step r by model × family (natural) — task difficulty spans the range
| model | cipher | ledger | refchain |
|-------|--------|--------|----------|
| llama1b  | 0.00 | 0.10 | 0.38 |
| qwen7b   | 0.06 | 0.74 | 0.99 |
| llama8b  | 0.02 | 0.91 | 1.00 |
| llama70b | 0.09 | 0.79 | 0.94 |
| deepseek | 0.25 | 0.94 | 0.98 |

cipher (procedural string-edit) is hard per-step; refchain (variable tracking)
easy; ledger (arithmetic state) medium — the geometric law holds across all.

## Figures (`figures/`)
- fig1_decay_curves.png — success vs horizon, per family, per model (Wilson CIs).
- fig2_regime_contrast.png — natural vs compressed vs padded, per family.
- fig3_perstep_accuracy.png — per-step accuracy vs step index (accelerating decay).

## Deviations from pre-registration (DESIGN.md)
1. Backend: pivoted from Colab/vLLM (free) to OpenRouter (~$3) after repeated
   Colab dependency failures (CUDA13/12, transformers 5.0, protobuf/TF). Same
   harness, hosted inference.
2. Models: 5 hosted open models (1.2–671B) instead of the 4 Colab models; this
   *extends* the capability ladder. Three candidate small models excluded for
   protocol non-adherence (logged above).
3. Trials: 25 (lean) as planned. Harness fix mid-study: `extract_json` now parses
   multiple concatenated JSON objects (some models replay the exchange); all data
   re-scored uniformly with the fixed parser.
