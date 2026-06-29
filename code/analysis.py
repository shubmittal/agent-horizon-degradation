"""Analysis pipeline: load raw trajectories -> cell success tables (Wilson CIs)
-> decay-curve fits + model selection -> per-step hazard / compounding ->
step-vs-context disentangling GLM -> benchmark-gap projection -> figures.

Curve model selection uses statsmodels GLM(Binomial) under three link functions:
  * log     link -> P = a * r^H        (GEOMETRIC: constant per-step hazard)
  * identity     -> P = a - b*H        (LINEAR)
  * logit        -> sigmoid(b0+b1*H)   (THRESHOLD / cliff)
AIC selects among them; the geometric fit yields r (per-step reliability) + CI.

Run:  python analysis.py            # reads results/raw, writes results/summary.json + figures
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

# Separated cells (success all-0 or all-1) make the GLM link fits diverge; we guard
# the results explicitly, so silence the expected numeric noise.
warnings.filterwarnings("ignore")
np.seterr(all="ignore")

from config import FIG_DIR, MODEL_META, RAW_DIR, RESULTS_DIR

FIG_DIR.mkdir(parents=True, exist_ok=True)

# Representative horizons (in agent steps) for the §H4 benchmark-gap projection.
# These are order-of-magnitude task lengths from the benchmark papers, used to show
# WHERE on the decay curve each benchmark samples; exact values are re-sourced and
# cited at drafting. The projection curve over ALL horizons is also reported so a
# reader can place any benchmark precisely.
BENCHMARK_HORIZONS = {
    "GAIA": 8,              # short general-assistant tasks
    "WebArena": 15,         # web navigation action sequences
    "tau-bench": 20,        # customer-service tool-use dialogues
    "SWE-bench": 30,        # multi-step code-repair trajectories
    "OSWorld": 30,          # computer-use tasks
    "TheAgentCompany": 100, # very-long simulated company workflows
}
BENCH_HORIZON = 15      # representative bounded-benchmark task length (steps)
PROD_HORIZON = 100      # representative production agent horizon (steps)


# ---------------------------------------------------------------- load
def load_results() -> pd.DataFrame:
    rows = []
    for p in sorted(RAW_DIR.glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue   # skip a partially-written line during a live sweep
    if not rows:
        raise SystemExit(f"No results found in {RAW_DIR}. Run the sweep first.")
    df = pd.DataFrame(rows)
    df["success"] = df["success"].astype(int)
    return df


# ---------------------------------------------------------------- Wilson CI
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def cell_table(df: pd.DataFrame) -> pd.DataFrame:
    g = (df.groupby(["model", "family", "regime", "horizon"])["success"]
         .agg(["sum", "count"]).reset_index())
    ci = g.apply(lambda r: wilson(int(r["sum"]), int(r["count"])), axis=1)
    g[["rate", "lo", "hi"]] = pd.DataFrame(ci.tolist(), index=g.index)
    return g


# ---------------------------------------------------------------- curve fits
def _fit_link(sub: pd.DataFrame, link) -> dict | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = sm.GLM(sub["success"],
                       sm.add_constant(sub[["horizon"]].astype(float)),
                       family=sm.families.Binomial(link)).fit()
        return {"aic": m.aic, "params": m.params.to_dict(),
                "bse": m.bse.to_dict()}
    except Exception:  # noqa: BLE001 separation / non-convergence
        return None


def fit_curves(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, family) on the NATURAL regime: fit 3 shapes, pick by AIC,
    and report geometric per-step reliability r with a CI."""
    out = []
    nat = df[df["regime"] == "natural"]
    for (model, family), sub in nat.groupby(["model", "family"]):
        links = {"geometric": sm.families.links.Log(),
                 "linear": sm.families.links.Identity(),
                 "threshold": sm.families.links.Logit()}
        fits = {name: _fit_link(sub, lk) for name, lk in links.items()}
        fits = {k: v for k, v in fits.items() if v is not None}
        if not fits:
            continue
        best = min(fits, key=lambda k: fits[k]["aic"])
        meta = MODEL_META.get(model)
        row = {"model": model, "family": family, "best_shape": best,
               "params_b": meta.params_b if meta else float("nan"),
               "vendor": meta.vendor if meta else "?", "n": len(sub)}
        for name, f in fits.items():
            row[f"aic_{name}"] = round(f["aic"], 1)
        if "geometric" in fits:
            b = fits["geometric"]["params"]["horizon"]
            se = fits["geometric"]["bse"]["horizon"]
            a = fits["geometric"]["params"]["const"]
            r = float(np.exp(b))
            # Guard separated/degenerate fits: r in (0,1]; a (intercept success) <= 1.
            if np.isfinite(r) and 0 < r <= 1.0 and np.isfinite(se) and se < 5:
                row["r_perstep"] = r
                row["r_lo"] = float(np.exp(b - 1.96 * se))
                row["r_hi"] = float(min(1.0, np.exp(b + 1.96 * se)))
                row["a_intercept"] = float(min(1.0, np.exp(a)))
        out.append(row)
    return pd.DataFrame(out)


def perstep_reliability(df: pd.DataFrame) -> pd.DataFrame:
    """Direct, fit-free per-step reliability r per (model, family): the fraction of
    individual steps answered correctly in the natural regime. Robust to the
    all-0/all-1 separation that breaks the GLM curve fits, and the natural quantity
    for the compounding law P(success) ~ r^H."""
    nat = df[df["regime"] == "natural"]
    out = []
    for (model, family), sub in nat.groupby(["model", "family"]):
        steps = [int(x) for arr in sub["per_step_correct"] for x in arr]
        if steps:
            meta = MODEL_META.get(model)
            out.append({"model": model, "family": family,
                        "r_direct": float(np.mean(steps)), "n_steps": len(steps),
                        "params_b": meta.params_b if meta else float("nan"),
                        "vendor": meta.vendor if meta else "?"})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- per-step hazard
def perstep_curves(df: pd.DataFrame) -> pd.DataFrame:
    """Empirical per-step accuracy vs step index (interactive regimes), pooled
    per (model, family, regime). Evidence on whether hazard is constant."""
    out = []
    inter = df[df["regime"].isin(["natural", "compressed"])]
    for (model, family, regime), sub in inter.groupby(["model", "family", "regime"]):
        maxlen = int(sub["horizon"].max())
        acc = [[] for _ in range(maxlen)]
        for arr in sub["per_step_correct"]:
            for i, ok in enumerate(arr):
                acc[i].append(1 if ok else 0)
        for i, vals in enumerate(acc):
            if vals:
                out.append({"model": model, "family": family, "regime": regime,
                            "step": i + 1, "acc": float(np.mean(vals)),
                            "n": len(vals)})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- capability scaling
def capability_scaling(rel: pd.DataFrame) -> dict:
    """Does per-step reliability rise with model size? Correlate the direct r with
    log10(params_b) across (model, family). Tests 'rot scales with capability'."""
    if rel.empty:
        return {}
    c = rel.dropna(subset=["r_direct", "params_b"])
    c = c[c["params_b"] > 0]
    if c["params_b"].nunique() < 2:
        return {"note": "need >=2 distinct model sizes for scaling"}
    x = np.log10(c["params_b"].to_numpy())
    y = c["r_direct"].to_numpy()
    r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else float("nan")
    by_model = (c.groupby(["model", "params_b"])["r_direct"].mean()
                .reset_index().sort_values("params_b"))
    return {
        "pearson_r_vs_log10params": round(r, 3),
        "r_perstep_by_model": [{"model": m, "params_b": p, "r_direct": round(v, 4)}
                               for m, p, v in by_model.itertuples(index=False)],
    }


# ---------------------------------------------------------------- mechanism
def mechanism(df: pd.DataFrame) -> dict:
    """Failure-mechanism decomposition from data we already capture, mapped to the
    survey taxonomy: format/tool-call drift (F1), where the first error lands, and
    whether per-step hazard is constant or ACCELERATING (compounding, F6)."""
    inter = df[df["regime"].isin(["natural", "compressed"])]
    out = {"format_drift_rate": round(float((df["malformed_count"] > 0).mean()), 4)}

    # first-error position as a fraction of horizon (does it fail early or late?)
    fe = inter.dropna(subset=["first_error_step"])
    if len(fe):
        frac = (fe["first_error_step"] + 1) / fe["horizon"]
        out["first_error_frac_mean"] = round(float(frac.mean()), 3)

    # accelerating hazard: per-step accuracy in the LAST third vs FIRST third of a
    # trajectory (pooled over long horizons). Lower late accuracy => accelerating.
    long = inter[inter["horizon"] >= 8]
    early, late = [], []
    for arr, H in zip(long["per_step_correct"], long["horizon"]):
        if not arr:
            continue
        k = max(1, len(arr) // 3)
        early += [int(x) for x in arr[:k]]
        late += [int(x) for x in arr[-k:]]
    if early and late:
        out["perstep_acc_first_third"] = round(float(np.mean(early)), 3)
        out["perstep_acc_last_third"] = round(float(np.mean(late)), 3)
        out["hazard_accelerates"] = bool(np.mean(late) < np.mean(early) - 0.02)
    return out


# ---------------------------------------------------------------- disentangle
def disentangle(df: pd.DataFrame) -> dict:
    """Identify the driver of degradation via PAIRED regime contrasts at fixed
    horizon. The operation count H is held equal across regimes at a given level,
    so the regimes vary only the two axes of interest:

        natural    : H turns, long context
        compressed : H turns, SHORT context   (vs natural -> context-length axis)
        padded     : 1 turn,  long context    (vs natural -> multi-turn-process axis)

    We fit  logit(success) ~ log2(H) * C(regime) + C(model) + C(family)  and read
    off the per-regime decay slope d logit / d log2(H). A FLATTER slope under a
    manipulation means that manipulation removed degradation:
      * compressed flatter than natural  => context length was driving the rot.
      * padded flatter than natural       => the multi-step process was driving it.
      * no regime difference               => intrinsic per-step compounding,
                                              invariant to presentation.
    """
    d = df.copy()
    d = d[d["horizon"] > 0].copy()
    d["log2H"] = np.log2(d["horizon"].astype(float))
    base = [r for r in ["natural", "compressed", "padded"] if r in set(d["regime"])]
    d["regime"] = pd.Categorical(d["regime"], categories=base, ordered=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = smf.glm("success ~ log2H * C(regime, Treatment('natural')) "
                    "+ C(model) + C(family)",
                    data=d, family=sm.families.Binomial()).fit()

    def slope(regime: str) -> float:
        b = float(m.params.get("log2H", 0.0))
        if regime != "natural":
            b += float(m.params.get(
                f"log2H:C(regime, Treatment('natural'))[T.{regime}]", 0.0))
        return b

    slopes = {r: slope(r) for r in base}
    s_nat = slopes.get("natural", 0.0)
    # Slopes are negative (decay). A regime whose slope is LESS negative than
    # natural has a flatter curve => that manipulation removed degradation.
    #   context_effect > 0  <=> compressed flatter than natural <=> context drove rot
    #   process_effect > 0  <=> padded flatter than natural   <=> multi-turn drove rot
    context_effect = slopes.get("compressed", s_nat) - s_nat
    process_effect = slopes.get("padded", s_nat) - s_nat
    pvals = {r: float(m.pvalues.get(
        f"log2H:C(regime, Treatment('natural'))[T.{r}]", float("nan")))
        for r in base if r != "natural"}

    THRESH = 0.05
    if context_effect < THRESH and process_effect < THRESH:
        driver = "intrinsic per-step compounding (no regime flattening)"
    elif context_effect >= process_effect:
        driver = "context length"
    else:
        driver = "multi-step process"

    return {
        "per_regime_logit_slope_per_doubling": {k: round(v, 4) for k, v in slopes.items()},
        "context_length_effect": round(context_effect, 4),
        "multistep_process_effect": round(process_effect, 4),
        "interaction_pvalues": {k: round(v, 6) for k, v in pvals.items()},
        "n": int(len(d)),
        "driver": driver,
    }


# ---------------------------------------------------------------- framing gap
def benchmark_gap(rel: pd.DataFrame) -> dict:
    """Project the robust per-step reliability r (P(success) ~ r^H) onto benchmark
    vs production horizons. Uses the direct r per (model, family)."""
    if rel.empty:
        return {}
    g = rel.copy()
    g["p_bench"] = g["r_direct"].clip(upper=1.0) ** BENCH_HORIZON
    g["p_prod"] = g["r_direct"].clip(upper=1.0) ** PROD_HORIZON
    mean_r = float(g["r_direct"].mean())
    by_bench = {name: round(float((g["r_direct"] ** h).mean()), 4)
                for name, h in sorted(BENCHMARK_HORIZONS.items(), key=lambda kv: kv[1])}
    return {
        "bench_horizon": BENCH_HORIZON, "prod_horizon": PROD_HORIZON,
        "mean_p_bench": round(float(g["p_bench"].mean()), 4),
        "mean_p_prod": round(float(g["p_prod"].mean()), 4),
        "mean_per_step_r": round(mean_r, 4),
        "projected_success_at_benchmark_horizons": by_bench,
        "per_cell": g[["model", "family", "r_direct", "p_bench", "p_prod"]].to_dict(orient="records"),
    }


# ---------------------------------------------------------------- figures
def make_figures(df: pd.DataFrame, cells: pd.DataFrame, steps: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = sorted(df["family"].unique())
    models = sorted(df["model"].unique())

    # Fig 1: decay curves (natural) per family, one line per model, Wilson CIs.
    fig, axes = plt.subplots(1, len(families), figsize=(5 * len(families), 4),
                             squeeze=False)
    for ax, fam in zip(axes[0], families):
        sub = cells[(cells["family"] == fam) & (cells["regime"] == "natural")]
        for model in models:
            s = sub[sub["model"] == model].sort_values("horizon")
            if s.empty:
                continue
            ax.errorbar(s["horizon"], s["rate"],
                        yerr=[s["rate"] - s["lo"], s["hi"] - s["rate"]],
                        marker="o", capsize=3, label=model)
        ax.set_xscale("log", base=2)
        ax.set_title(f"{fam}")
        ax.set_xlabel("horizon (steps)")
        ax.set_ylabel("task success rate")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, alpha=0.3)
    axes[0][-1].legend(fontsize=8)
    fig.suptitle("Long-horizon degradation (natural regime)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_decay_curves.png", dpi=150)
    plt.close(fig)

    # Fig 2: regime comparison (the disentangling), pooled across models.
    fig, axes = plt.subplots(1, len(families), figsize=(5 * len(families), 4),
                             squeeze=False)
    for ax, fam in zip(axes[0], families):
        sub = cells[cells["family"] == fam]
        for reg in ["natural", "compressed", "padded"]:
            s = (sub[sub["regime"] == reg].groupby("horizon")["rate"]
                 .mean().reset_index().sort_values("horizon"))
            if s.empty:
                continue
            ax.plot(s["horizon"], s["rate"], marker="s", label=reg)
        ax.set_xscale("log", base=2)
        ax.set_title(f"{fam}")
        ax.set_xlabel("horizon (steps)")
        ax.set_ylabel("success rate (mean over models)")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, alpha=0.3)
    axes[0][-1].legend(fontsize=8)
    fig.suptitle("Step-count vs context-length (regime contrast)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_regime_contrast.png", dpi=150)
    plt.close(fig)

    # Fig 3: per-step accuracy vs step index (natural).
    if not steps.empty:
        fig, axes = plt.subplots(1, len(families), figsize=(5 * len(families), 4),
                                 squeeze=False)
        for ax, fam in zip(axes[0], families):
            sub = steps[(steps["family"] == fam) & (steps["regime"] == "natural")]
            for model in models:
                s = sub[sub["model"] == model].sort_values("step")
                if s.empty:
                    continue
                ax.plot(s["step"], s["acc"], marker=".", label=model)
            ax.set_title(f"{fam}")
            ax.set_xlabel("step index")
            ax.set_ylabel("per-step accuracy")
            ax.set_ylim(-0.03, 1.03)
            ax.grid(True, alpha=0.3)
        axes[0][-1].legend(fontsize=8)
        fig.suptitle("Per-step accuracy across the trajectory (natural regime)")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig3_perstep_accuracy.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> None:
    df = load_results()
    cells = cell_table(df)
    curves = fit_curves(df)
    rel = perstep_reliability(df)
    steps = perstep_curves(df)
    dis = disentangle(df)
    gap = benchmark_gap(rel)
    scaling = capability_scaling(rel)
    mech = mechanism(df)

    cells.to_csv(RESULTS_DIR / "cell_table.csv", index=False)
    if not curves.empty:
        curves.to_csv(RESULTS_DIR / "curve_fits.csv", index=False)
    make_figures(df, cells, steps)

    summary = {
        "n_trajectories": int(len(df)),
        "models": sorted(df["model"].unique().tolist()),
        "families": sorted(df["family"].unique().tolist()),
        "horizons": sorted(int(h) for h in df["horizon"].unique()),
        "regimes": sorted(df["regime"].unique().tolist()),
        "best_shape_counts": (curves["best_shape"].value_counts().to_dict()
                              if not curves.empty else {}),
        "direct_perstep_r": (
            {"min": round(float(rel["r_direct"].min()), 4),
             "median": round(float(rel["r_direct"].median()), 4),
             "max": round(float(rel["r_direct"].max()), 4)}
            if not rel.empty else {}),
        "geometric_r_summary": (
            {"min": round(float(curves["r_perstep"].min()), 4),
             "median": round(float(curves["r_perstep"].median()), 4),
             "max": round(float(curves["r_perstep"].max()), 4)}
            if "r_perstep" in curves and curves["r_perstep"].notna().any() else {}),
        "disentangle": dis,
        "capability_scaling": scaling,
        "mechanism": mech,
        "benchmark_gap": {k: v for k, v in gap.items() if k != "per_cell"},
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {RESULTS_DIR/'summary.json'}, cell_table.csv, curve_fits.csv "
          f"and figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
