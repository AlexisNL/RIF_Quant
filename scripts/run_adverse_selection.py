"""
Adverse Selection Switching Pipeline — Tick-by-Tick, Train/Test Split
======================================================================

Methodological design
---------------------
All parameter choices come from optimize_adverse_selection_parameters.py
(walk-forward CV on TRAIN); this script only evaluates on TEST.

Split: TRAIN = first TRAIN_RATIO of ticks | TEST = remaining ticks.
This is a strict temporal split — no future data ever reaches the training
phase.

No-lookahead guarantee
----------------------
Wasserstein features use the CAUSAL variant:
    W(series[i-2w : i-w], series[i-w : i])
Only data strictly before tick i is used at position i.

The switching logistic regression is fitted on TRAIN observations only.
AUC is evaluated on TEST observations only.

Requires
--------
Run scripts/optimize_adverse_selection_parameters.py first to produce:
    data/results/best_parameters_adverse_selection.csv

Run
---
    MPLBACKEND=Agg PYTHONUTF8=1 PYTHONPATH=/c/Users/Alexis/Desktop/Quant/RIF \\
    C:/Users/Alexis/anaconda3/envs/kedro-environment/python.exe -u \\
    scripts/run_adverse_selection.py
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from src.config import TICKERS, RESULTS_DIR, FIGURES_DIR, validate_config
from src.data.loader import load_ticker_ticks
from src.features.wasserstein import compute_tick_wasserstein_causal
from src.models.hmm_optimal import LocalHMM
from src.realtime.dual_clock_monitor import (
    RegimeCoefficients,
    calibrate_regime_coefficients,
    test_coefficient_homogeneity,
)
from src.analysis.adverse_selection import (
    build_adverse_selection_dataset,
    generate_latex_tables,
)

# ============================================================================
# Fixed parameters (not tuned — pure backtest settings)
# ============================================================================

TRAIN_RATIO = 0.70       # first 70% of ticks: train; last 30%: test
WASS_STRIDE = 50         # causal Wasserstein stride (performance)
HORIZON_TICKS = 100      # H-tick-ahead price impact for labeling
IMPACT_THRESHOLD = 2.0   # |impact| > threshold * tick_size => informed
TICK_SIZE = 0.01

# Final HMM fitting settings (heavier than CV, run once on full TRAIN)
FINAL_N_INIT = 3
FINAL_N_ITER = 500


# ============================================================================
# Helpers
# ============================================================================

def load_best_params(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "best_parameters_adverse_selection.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            "Run scripts/optimize_adverse_selection_parameters.py first."
        )
    return pd.read_csv(path)


def compute_causal_wass_features(
    tick_df: pd.DataFrame,
    window: int,
    stride: int = WASS_STRIDE,
) -> np.ndarray:
    """
    Compute causal Wasserstein (OFI, OBI) features for a tick DataFrame.

    Returns np.ndarray of shape (len(tick_df) - 2*window, 2).
    Output row j corresponds to original tick j + 2*window.
    """
    ofi_w = compute_tick_wasserstein_causal(
        tick_df["ofi"].values, window=window, stride=stride
    )
    obi_w = compute_tick_wasserstein_causal(
        tick_df["obi"].values, window=window, stride=stride
    )
    return np.column_stack([ofi_w, obi_w])


def align_lob_to_wass(tick_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Return the slice of tick_df that corresponds to the valid range of
    causal Wasserstein output.

    Causal Wasserstein output[j] corresponds to original tick j + 2*window,
    so the aligned LOB data starts at row 2*window.
    """
    return tick_df.iloc[2 * window :].reset_index(drop=True)


def compute_pi_test(
    ofi: np.ndarray,
    obi: np.ndarray,
    states: np.ndarray,
    coeffs: dict,
) -> np.ndarray:
    """Apply TRAIN-fitted switching logistic to TEST observations."""
    pi = np.zeros(len(ofi))
    for k, c in coeffs.items():
        mask = states == k
        if mask.sum() == 0:
            continue
        logit = c.alpha + c.beta_ofi * ofi[mask] + c.gamma_obi * obi[mask]
        pi[mask] = 1.0 / (1.0 + np.exp(-logit))
    return pi


def plot_results(
    df_test: pd.DataFrame,
    pi_test: np.ndarray,
    pi_vpin_test: np.ndarray,
    pi_pooled_test: np.ndarray,
    coeffs_train: dict,
    n_regimes: int,
    ticker: str,
) -> None:
    """Four-panel TEST-set evaluation figure."""
    y = df_test["is_informed"].values
    states = df_test["regime"].values

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: ROC curves
    ax = axes[0, 0]
    for label, pi, color, ls in [
        (f"Switching (AUC={roc_auc_score(y, pi_test):.3f})",   pi_test,    "green",     "-"),
        (f"Pooled (AUC={roc_auc_score(y, pi_pooled_test):.3f})", pi_pooled_test, "steelblue", "--"),
        (f"VPIN-like (AUC={roc_auc_score(y, pi_vpin_test):.3f})", pi_vpin_test, "orange", "-."),
    ]:
        fpr, tpr, _ = roc_curve(y, pi)
        ax.plot(fpr, tpr, label=label, linewidth=2, color=color, linestyle=ls)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random (0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC — TEST set ({ticker})")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2: AUC bar (TRAIN vs TEST decomposition by model)
    ax = axes[0, 1]
    models = ["PIN static", "VPIN-like", "Pooled logistic", "Switching"]
    aucs = [0.5, roc_auc_score(y, pi_vpin_test),
            roc_auc_score(y, pi_pooled_test), roc_auc_score(y, pi_test)]
    colors = ["gray", "orange", "steelblue", "green"]
    bars = ax.bar(range(len(models)), aucs, color=colors, alpha=0.8)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("AUC (TEST)")
    ax.set_title(f"Model Comparison — TEST ({ticker})")
    ax.axhline(0.5, color="k", linestyle="--", alpha=0.3)
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

    # Panel 3: Regime-heterogeneous coefficients (TRAIN estimates)
    ax = axes[1, 0]
    regimes = sorted(coeffs_train.keys())
    betas  = [coeffs_train[r].beta_ofi  for r in regimes]
    gammas = [coeffs_train[r].gamma_obi for r in regimes]
    x = np.arange(len(regimes))
    w = 0.35
    ax.bar(x - w / 2, betas,  w, label="beta (OFI)", color="steelblue")
    ax.bar(x + w / 2, gammas, w, label="gamma (OBI)", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Regime {r}" for r in regimes])
    ax.set_ylabel("Coefficient (TRAIN estimate)")
    ax.set_title(f"Regime Coefficients — TRAIN fit ({ticker})")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Panel 4: pi(t) distribution on TEST by regime
    ax = axes[1, 1]
    for k in range(n_regimes):
        mask = states == k
        if mask.sum() == 0:
            continue
        ax.hist(pi_test[mask], bins=30, alpha=0.5,
                label=f"Regime {k}", density=True)
    ax.set_xlabel("pi(t) — adverse selection probability (TEST)")
    ax.set_ylabel("Density")
    ax.set_title(f"pi(t) by Regime — TEST ({ticker})")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle(
        f"Adverse Selection Switching — {ticker}\n"
        f"(TRAIN={TRAIN_RATIO:.0%} fit, TEST={1-TRAIN_RATIO:.0%} evaluation, "
        f"causal Wasserstein)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    out = FIGURES_DIR / f"adverse_selection_{ticker}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Figure: {out.name}")


# ============================================================================
# Main pipeline
# ============================================================================

def main():
    validate_config()

    print("\n" + "=" * 70)
    print("ADVERSE SELECTION SWITCHING — TICK-BY-TICK BACKTEST")
    print("=" * 70)
    print(f"Tickers   : {TICKERS}")
    print(f"Split     : {TRAIN_RATIO:.0%} TRAIN / {1-TRAIN_RATIO:.0%} TEST")
    print(f"Horizon   : {HORIZON_TICKS} ticks")
    print(f"Threshold : {IMPACT_THRESHOLD} ticks = ${IMPACT_THRESHOLD * TICK_SIZE:.2f}")
    print("=" * 70 + "\n")

    best_params = load_best_params(RESULTS_DIR)
    t_start = time.time()
    all_comparisons = []

    for ticker in TICKERS:
        print(f"\n{'='*70}")
        print(f"  {ticker}")
        print(f"{'='*70}")
        t0 = time.time()

        # --- Load parameters optimised on TRAIN only ----------------------
        row = best_params[best_params["ticker"] == ticker]
        if row.empty:
            print(f"  WARN: no parameters found for {ticker} — skipping")
            continue
        row = row.iloc[0]
        window        = int(row["tick_wass_window"])
        n_regimes     = int(row["n_regimes"])
        persistence   = float(row["local_persistence"])
        smooth_window = int(row["local_smoothing"])
        print(f"  Params: window={window}, K={n_regimes}, "
              f"persist={persistence}, smooth={smooth_window}")
        print(f"  CV score: mean_ARI={row['mean_ari']:.3f}  "
              f"std_ARI={row['std_ari']:.3f}")

        # --- Load tick data -----------------------------------------------
        try:
            tick_df = load_ticker_ticks(ticker)
        except FileNotFoundError as e:
            print(f"  WARN: {e} — skipping")
            continue

        n_raw   = len(tick_df)
        n_train = int(n_raw * TRAIN_RATIO)
        n_test  = n_raw - n_train
        print(f"  Ticks: {n_raw:,}  TRAIN: {n_train:,}  TEST: {n_test:,} "
              f"({time.time()-t0:.1f}s)")

        # ===================================================================
        # TRAIN phase — all computation uses only tick_df[:n_train]
        # ===================================================================

        train_df_raw = tick_df.iloc[:n_train]

        # Causal Wasserstein on TRAIN ticks only
        print("  [TRAIN] Causal Wasserstein ...")
        t1 = time.time()
        X_train = compute_causal_wass_features(train_df_raw, window)
        print(f"  [TRAIN] {len(X_train):,} valid obs ({time.time()-t1:.1f}s)")

        if len(X_train) < n_regimes * 50:
            print(f"  WARN: TRAIN too short after Wasserstein offset — skipping")
            continue

        # Fit HMM on TRAIN
        print(f"  [TRAIN] Fitting HMM ({n_regimes} regimes, {FINAL_N_INIT} restarts) ...")
        t1 = time.time()
        model = LocalHMM(
            n_regimes=n_regimes,
            persistence=persistence,
            smooth_window=smooth_window,
            n_init=FINAL_N_INIT,
            n_iter=FINAL_N_ITER,
            verbose=False,
        )
        model.fit(X_train)
        print(f"  [TRAIN] HMM fitted ({time.time()-t1:.1f}s)")

        # Align TRAIN LOB data to Wasserstein valid range and build dataset
        train_lob = align_lob_to_wass(train_df_raw, window)
        df_train = build_adverse_selection_dataset(
            lob_data=train_lob,
            states=model.states_,
            horizon=HORIZON_TICKS,
            impact_threshold_ticks=IMPACT_THRESHOLD,
            tick_size=TICK_SIZE,
        )
        df_train["split"] = "train"

        # Fit switching logistic regression on TRAIN
        coeffs_train = calibrate_regime_coefficients(
            df_train["ofi"].values,
            df_train["obi"].values,
            df_train["is_informed"].values,
            df_train["regime"].values,
            n_regimes,
        )

        # Fit benchmark models on TRAIN (pooled + VPIN-like)
        X_bl = np.column_stack([df_train["ofi"].values, df_train["obi"].values])
        y_tr = df_train["is_informed"].values
        model_pooled = LogisticRegression(max_iter=1000).fit(X_bl, y_tr)
        model_vpin   = LogisticRegression(max_iter=1000).fit(
            df_train["ofi"].values.reshape(-1, 1), y_tr
        )

        # Homogeneity test on TRAIN (for paper)
        homogeneity_test = test_coefficient_homogeneity(
            df_train["ofi"].values, df_train["obi"].values,
            y_tr, df_train["regime"].values, n_regimes,
        )

        # ===================================================================
        # TEST phase — HMM and logistic models are FROZEN from TRAIN
        # ===================================================================

        test_df_raw = tick_df.iloc[n_train:]

        # Causal Wasserstein on TEST ticks only (uses only test data)
        print("  [TEST] Causal Wasserstein ...")
        t1 = time.time()
        X_test = compute_causal_wass_features(test_df_raw, window)
        print(f"  [TEST] {len(X_test):,} valid obs ({time.time()-t1:.1f}s)")

        if len(X_test) < n_regimes * 10:
            print(f"  WARN: TEST too short after Wasserstein offset — skipping")
            continue

        # Infer regimes using TRAIN-fitted model (Viterbi, no re-fitting)
        states_test = model.predict(X_test)

        # Align TEST LOB data and build dataset
        test_lob = align_lob_to_wass(test_df_raw, window)
        df_test = build_adverse_selection_dataset(
            lob_data=test_lob,
            states=states_test,
            horizon=HORIZON_TICKS,
            impact_threshold_ticks=IMPACT_THRESHOLD,
            tick_size=TICK_SIZE,
        )
        df_test["split"] = "test"

        if len(df_test) < 200 or df_test["is_informed"].sum() < 10:
            print(f"  WARN: TEST dataset too sparse — skipping")
            continue

        # Evaluate models on TEST — TRAIN coefficients applied to TEST data
        y_test  = df_test["is_informed"].values
        ofi_tst = df_test["ofi"].values
        obi_tst = df_test["obi"].values
        reg_tst = df_test["regime"].values

        pi_switching_test = compute_pi_test(ofi_tst, obi_tst, reg_tst, coeffs_train)
        pi_pooled_test    = model_pooled.predict_proba(
            np.column_stack([ofi_tst, obi_tst])
        )[:, 1]
        pi_vpin_test      = model_vpin.predict_proba(
            ofi_tst.reshape(-1, 1)
        )[:, 1]

        auc_sw     = roc_auc_score(y_test, pi_switching_test)
        auc_pooled = roc_auc_score(y_test, pi_pooled_test)
        auc_vpin   = roc_auc_score(y_test, pi_vpin_test)

        comparison_df = pd.DataFrame([
            {"Model": "PIN static (Easley 1996)",     "AUC": 0.500,       "N_params": 1},
            {"Model": "VPIN-like (OFI only)",          "AUC": auc_vpin,    "N_params": 2},
            {"Model": "Pooled logistic (OFI+OBI)",     "AUC": auc_pooled,  "N_params": 3},
            {"Model": "Switching (OFI+OBI | regime)",  "AUC": auc_sw,
             "N_params": 3 * n_regimes},
        ])

        print(f"\n  TEST SET RESULTS")
        print(f"  {'Model':<36} {'AUC':>6}")
        for _, r in comparison_df.iterrows():
            marker = " <-- proposed" if "Switching" in r["Model"] else ""
            print(f"  {r['Model']:<36} {r['AUC']:>6.3f}{marker}")
        print(f"  Gain vs PIN static: +{(auc_sw-0.5)/0.5*100:.1f}% AUC")

        # Plot and LaTeX tables
        plot_results(df_test, pi_switching_test, pi_vpin_test,
                     pi_pooled_test, coeffs_train, n_regimes, ticker)
        generate_latex_tables(comparison_df, homogeneity_test, coeffs_train, ticker=ticker)

        comparison_df["Ticker"] = ticker
        all_comparisons.append(comparison_df)
        print(f"  Done in {time.time()-t0:.1f}s")

    # =======================================================================
    # Cross-ticker summary
    # =======================================================================
    if all_comparisons:
        full = pd.concat(all_comparisons, ignore_index=True)
        full.to_csv(RESULTS_DIR / "adverse_selection_all_tickers.csv", index=False)

        print("\n" + "=" * 70)
        print("CROSS-TICKER SUMMARY — TEST SET AUC")
        print("=" * 70)
        summary = (
            full.groupby("Model")["AUC"]
            .agg(["mean", "std"])
            .rename(columns={"mean": "Mean AUC", "std": "Std AUC"})
        )
        print(summary.to_string())
        print("=" * 70)

    elapsed = time.time() - t_start
    print(f"\nPipeline complete in {elapsed:.1f}s ({elapsed/60:.1f} min)\n")


if __name__ == "__main__":
    main()
