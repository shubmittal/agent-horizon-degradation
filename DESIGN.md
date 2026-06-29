# Pre-Registration & Experimental Design — Step (1) Deliverable

**Paper:** *How Fast Do Agents Rot? An Empirical Study of Long-Horizon Degradation
in LLM Agents.*
**Status:** Draft design for author sign-off. Written before the full run
(informal pre-registration). Pilot results may refine N and horizon levels; any
deviation from this plan will be logged in `DEVIATIONS.md`.

---

## 1. Hypotheses (the measurable spine)

**H1 — Existence & shape.** Per-trajectory task success probability
`P(success | H)` declines monotonically as task horizon `H` (number of dependent
steps) grows. We characterize the *functional form* by fitting and model-selecting
among three candidate shapes:

- **Geometric / exponential:** `P = a · r^H` — constant per-step failure hazard;
  reliability *compounds*. Predicts `log P` linear in `H`. (This is the survey's
  F6 "cascading failure" prediction.)
- **Threshold / cliff:** `P ≈ const`, then a sharp drop near a capacity limit
  `H*` (working-memory / context saturation). Logistic-in-`H` with steep slope.
- **Linear / gradual:** `P = a − bH`.

*H1a (directional):* the geometric form dominates, i.e. per-step reliability `r`
is the fundamental quantity. *Key sub-question:* is per-step hazard **constant**
(pure geometric) or **rising with H** (super-geometric / accelerating rot)?

**H2 — Cause (the disentangling control, mandatory).** Degradation is driven by
the **multi-step process** (many sequential agent turns + accumulated state) vs.
**raw context length** (tokens to attend over) — or it is intrinsic per-step
compounding invariant to either. At a *fixed* horizon H the number of operations
is held equal across three regimes, which vary only the two axes of interest:

| Regime | agent turns | context length |
|--------|-------------|----------------|
| natural | H | long |
| compressed | H | **short** (state carried, history windowed) |
| padded | **1** | long (all H ops in one prompt) |

We fit `logit P(success) = β0 + β_H·log₂H · C(regime) + (model) + (family)` and
read off the **per-regime decay slope** `d logit / d log₂H`. Paired contrasts
(identified because operation count is held fixed):
- **natural vs compressed** (same turns, differ only in context length) → isolates
  the **context-length** effect. Compressed flatter ⇒ context length drove the rot.
- **natural vs padded** (same context, differ in number of turns) → isolates the
  **multi-step-process** effect.

Decision rule: whichever manipulation *flattens* the decay is the driver; if
neither flattens it, degradation is intrinsic per-step compounding. (An earlier
plan to regress on `log(turns)+log(tokens)` jointly was dropped: without holding H
fixed those proxies both encode H and split its effect spuriously.)

**H3 — Generality.** The qualitative shape (H1) and dominant cause (H2) are
consistent across ≥3 task families and ≥3 models; slope magnitudes may differ,
sign/shape do not. Tested via mixed-effects variance components.

**H4 — Benchmark gap (the framing payoff / thesis).** Benchmarks sample the flat
top of the decay curve. Using fitted per-step hazard `r`, a benchmark of horizon
`H_b` reports ≈ `r^{H_b}`; production horizons are `k×` longer. We quantify: "at
median benchmark horizon success is X%; the *same* fitted hazard at production
horizon (k×) gives Y%" — the gap benchmarks hide.

---

## 2. Factors & levels (full factorial core + disentangling sub-study)

| Factor | Levels | Notes |
|--------|--------|-------|
| **Model** | 6 (hybrid) | **Free on Colab/vLLM:** Llama-3.2-1B, Llama-3.2-3B, Qwen2.5-7B, Llama-3.1-8B. **~$5 on OpenRouter:** Llama-3.3-70B (extends the Meta within-family ladder 1B→3B→8B→70B) + DeepSeek-V3 (cross-family large). Capability ladder ⇒ tests whether rot *scales with capability*. |
| **Task family** | 3 | ledger / refchain / cipher (§3) |
| **Horizon H** | 5: {2, 4, 8, 16, 32} steps | geometric spacing; extend to 64 if signal warrants |
| **Context regime** | 3: natural / compressed / padded | the H2 control |
| **Trial** | pilot 8 → full 25 | power-based (§5); seeds recorded |

**Core curve cells:** Model × Family × Horizon × *natural* = 4×3×5 = 60 cells.
**Disentangling sub-study:** at matched token budgets, contrast
{high-step·natural, high-step·compressed, low-step·padded} per Model × Family.
Pilot decides exactly which horizons enter the regime contrast (cost control).

---

## 3. Task families (synthetic, oracle-verifiable, parametric horizon)

All three are tool-using agent loops (ReAct-style: model emits a tool call →
simulated environment responds → model continues), with **ground-truth
verification by the simulator** (no LLM judge → no judge noise, low cost). Each
exercises a distinct locus of the survey's failure taxonomy and supports all
three context regimes.

1. **Ledger / state-tracking** *(memory locus, F3).* A simulated account system:
   the agent processes a stream of `N` transactions (deposit / transfer /
   conditional ops) delivered through tool responses, maintains balances, and
   answers a final query (e.g. final balance of account *j*, or first account to
   go negative). Horizon = `N` transactions.

2. **Reference-chain resolution** *(retrieval / lost-in-the-middle locus, F2).*
   `N` chained variable definitions (`x_{i} = f(x_{i-1})`), some defined early and
   queried late, interleaved with distractors; the agent resolves a target
   variable via tool lookups. Horizon = chain length. Stresses long-range
   retrieval and dependency tracking.

3. **Multi-hop tool composition / planning** *(planning + tool-use locus, F1).*
   A goal in a small simulated world (graph navigation / staged assembly) reachable
   only by composing `N` dependent primitive tool calls; success = goal state
   reached. Horizon = required plan length. Stresses planning depth and tool-call
   correctness over a long plan.

**Context regimes (the H2 knob), per family:**
- **Natural** — full trajectory history accumulates in context (context grows ∝ steps).
- **Compressed** — environment returns a compact running-state summary each turn,
  holding context ≈ constant while steps grow ⇒ isolates *step-count* effect.
- **Padded** — step count held low but context inflated with task-relevant filler
  to match a high-step token budget ⇒ isolates *context-length* effect.

Synthetic-by-design buys a clean horizon knob + exact verification; the cost is
ecological validity, addressed in Limitations and (optionally) one realistic
spot-check on a public agent task.

---

## 4. Metrics

- **Primary:** per-trajectory binary task success → success rate per cell
  (Wilson 95% CI).
- **Per-step accuracy / first-error step** where decomposable → drives the
  compounding analysis and the per-step hazard estimate.
- **Trajectory length actually taken, token counts, tool-call validity rate,
  termination type** (correct / wrong-confident / loop / give-up) → mechanism
  characterization mapped to F1/F4/F5.
- Cost + latency per cell (for the cost-aware reporting the survey calls for).

---

## 5. Power & sample size

Binary outcome; primary inference is the **decay curve**, which pools across 5
horizons × trials, so slope/shape power far exceeds any single pairwise cell.
Pairwise sanity (two-proportion, α=.05, power=.8): detect 0.9→0.7 ⇒ ~43/grp;
0.8→0.6 ⇒ ~60/grp; 0.8→0.65 ⇒ ~110/grp. Plan: **pilot 8/cell** to estimate
observed per-step hazard, re-solve N to a pre-registered MDE (target: 80% power to
detect a 15pp adjacent-cell gap and a per-step-hazard difference of interest),
then **full 40/cell** (top up boundary cells near 50% where variance is maximal).
Report effect sizes + CIs throughout; bootstrap (BCa) CIs on fitted curve
parameters.

---

## 6. Analysis plan (pre-registered)

1. Per-cell success + Wilson CI; heatmaps (model × horizon, per family).
2. Curve fit per (model, family): geometric vs threshold vs linear; AIC +
   bootstrap model-selection; report `r` (per-step reliability) with BCa CIs.
3. Compounding test: constant vs rising per-step hazard (LR test, pure-geometric
   vs hazard-increasing).
4. Disentangling: the per-regime decay-slope model of §H2; report each regime's
   `d logit / d log₂H`, the context-length and multi-step-process contrasts with
   interaction p-values, and the regime-contrast figure.
5. Generality: mixed-effects model (model, family as groups); variance components;
   is shape consistent? **Capability scaling:** correlate per-step reliability `r`
   with log model size (1B→671B) — does rot scale with capability?
6. Framing (H4): project fitted hazard onto **representative real-benchmark
   horizons** (GAIA/WebArena/τ-bench/SWE-bench/OSWorld/TheAgentCompany, sourced and
   cited) vs production horizons; produce the gap figure + headline number.
7. Mechanism decomposition (mapped to the survey taxonomy): format/tool-call drift
   rate (F1), first-error position, per-step hazard constant-vs-accelerating (F6),
   and silent-failure / termination type (F5).
8. Mitigation read-out: the `compressed` regime doubles as an intervention —
   report the reliability recovered by bounded-context state-carry.

---

## 7. Reproducibility & determinism

- Temperature fixed (low, e.g. 0–0.2) for the main run; one temperature-sensitivity
  sub-check. Seeds recorded per trajectory; provider non-determinism noted.
- Results checkpointed keyed by `(model, family, horizon, regime, seed)`; runs are
  resumable and skip completed work. Prompt caching on shared long prefixes.
- Release: harness code, all task generators, prompts, seeds, raw trajectories
  (license permitting), analysis notebooks, and this design doc.

---

## 8. Must-cite anchors (verified in companion survey unless marked ⏳)

Shared, already web-verified with the companion survey (see
`../SoK-Agent-Failures/SOURCES.md`):

- **Agent benchmarks (horizon framing):** SWE-bench (Jimenez et al., ICLR 2024);
  τ-bench (Yao et al., 2024) & τ²-bench (Barres et al., 2025); WebArena (Zhou et
  al., NeurIPS 2023); GAIA (Mialon et al., 2023); AgentBench (Liu et al., ICLR
  2024); OSWorld (Xie et al., NeurIPS 2024); AppWorld (Trivedi et al., ACL 2024);
  Mind2Web (Deng et al., NeurIPS 2023); TheAgentCompany (Xu et al., 2024); BFCL
  (Patil et al., ICML 2025).
- **Long-horizon / time-horizon:** METR — Measuring AI Ability to Complete Long
  Tasks (Kwa et al., 2025).
- **Context-length degradation:** Lost in the Middle (Liu et al., TACL 2024);
  Same Task, More Tokens (Levy et al., ACL 2024).
- **Reliability / compounding / eval-validity:** AI Agents That Matter (Kapoor et
  al., 2024); Establishing Best Practices for Rigorous Agentic Benchmarks (Zhu et
  al., NeurIPS 2025); Why Do Multi-Agent LLM Systems Fail? / MAST (Cemri et al.,
  2025).
- **Methods:** ReAct (Yao et al., ICLR 2023); Reflexion (Shinn et al., NeurIPS
  2023).

To verify and add during related-work (each web-verified before inclusion):
- ⏳ RULER: long-context benchmark (Hsieh et al., 2024).
- ⏳ A "context rot" / long-context-degradation empirical report.
- ⏳ NoLiMa or a needle-in-haystack long-context retrieval reference.
- ⏳ A compounding-error / horizon-length theory or measurement paper.
- ⏳ A statistical-rigor-in-ML-evaluation reference (for the power/CI framing).

Target count at submission: 18–24 verified references.
