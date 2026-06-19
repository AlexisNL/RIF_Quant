"""
Adverse Selection Switching Pipeline — Tick-by-Tick
=====================================================

Methodological note
-------------------
The main pipeline (run_hierarchical_contagion.py) resamples LOB events
to 500 ms bars to synchronize multiple tickers for cross-asset contagion
analysis.  For adverse selection, resampling is *wrong*: it conflates
informed trades with noise events and destroys the tick-level microstructure
signal that distinguishes the two.

This script therefore:
  - Loads raw LOBSTER data **per ticker**, without resampling
  - Computes Wasserstein distances over a sliding window of ticks
    (with stride to keep computation tractable on millions of events)
  - Fits a per-ticker HMM on tick-level features — completely independent
    of the cross-asset 500 ms pipeline
  - Labels each tick as "informed" based on H-tick-ahead price impact
  - Estimates the switching logistic regression P(informed|OFI,OBI,regime=k)

This pipeline is self-contained: it does NOT require run_hierarchical_contagion.py
to be run first.

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

from src.config import TICKERS, N_REGIMES, RESULTS_DIR, validate_config
from src.data.loader import load_ticker_ticks
from src.features.wasserstein import compute_tick_wasserstein
from src.models.hmm_optimal import LocalHMM
from src.analysis.adverse_selection import (
    build_adverse_selection_dataset,
    estimate_adverse_selection_switching,
    plot_adverse_selection_results,
    generate_latex_tables,
)

# ============================================================================
# Parameters
# ============================================================================

# Wasserstein sliding window (in ticks).
# 100 ticks on AAPL ~ 1-5 seconds of market activity.
TICK_WASS_WINDOW = 100

# Stride between Wasserstein evaluations.
# stride=50 computes W every 50 ticks and forward-fills between positions.
# This reduces computation 50x with minimal impact on regime detection
# (regimes are persistent; a lag of 50 ticks << typical regime duration).
TICK_WASS_STRIDE = 50

# Price impact horizon (in ticks) for labeling informed trades.
# 100 ticks ~ 1-10 seconds for liquid US equities.
HORIZON_TICKS = 100

# A tick is labeled "informed" when |impact| > threshold * tick_size.
IMPACT_THRESHOLD_TICKS = 2.0
TICK_SIZE = 0.01  # $0.01 minimum price increment

# HMM parameters (per-ticker tick-level model)
HMM_N_REGIMES = N_REGIMES
HMM_PERSISTENCE = 0.90   # slightly higher than 500 ms (ticks change faster)
HMM_SMOOTH_WINDOW = 20   # in ticks (~0.2-1s); fewer than the 500 ms model
HMM_N_INIT = 3
HMM_N_ITER = 500


def compute_tick_hmm_states(
    tick_df: pd.DataFrame,
    window: int = TICK_WASS_WINDOW,
    stride: int = TICK_WASS_STRIDE,
    n_regimes: int = HMM_N_REGIMES,
    persistence: float = HMM_PERSISTENCE,
    smooth_window: int = HMM_SMOOTH_WINDOW,
    ticker: str = '',
) -> tuple:
    """
    Compute per-ticker HMM states at tick resolution.

    Wasserstein features are computed for OFI and OBI over a sliding
    tick window (with stride), then a GaussianHMM is fitted on those
    two-dimensional features.

    Returns
    -------
    states : np.ndarray, shape (len(tick_df) - 2*window,)
        Regime per tick for the valid range [window, N-window).
    wass_ofi : np.ndarray
    wass_obi : np.ndarray
    """
    t0 = time.time()

    ofi = tick_df['ofi'].values
    obi = tick_df['obi'].values

    print(f"  Computing tick Wasserstein (window={window}, stride={stride}) ...")
    wass_ofi = compute_tick_wasserstein(ofi, window=window, stride=stride)
    wass_obi = compute_tick_wasserstein(obi, window=window, stride=stride)

    n_wass = len(wass_ofi)
    print(f"  Wasserstein features: {n_wass:,} ticks ({time.time()-t0:.1f}s)")

    if n_wass < n_regimes * 50:
        raise ValueError(
            f"Too few observations ({n_wass}) after Wasserstein window "
            f"for {n_regimes} regimes. Reduce window or check data."
        )

    X = np.column_stack([wass_ofi, wass_obi])

    print(f"  Fitting HMM ({n_regimes} regimes, {HMM_N_INIT} restarts) ...")
    t1 = time.time()
    model = LocalHMM(
        n_regimes=n_regimes,
        persistence=persistence,
        smooth_window=smooth_window,
        n_init=HMM_N_INIT,
        n_iter=HMM_N_ITER,
    )
    model.fit(X)
    states = model.states_

    regime_counts = {r: int((states == r).sum()) for r in range(n_regimes)}
    print(f"  HMM fitted in {time.time()-t1:.1f}s | "
          f"regime counts: { {r: f'{c:,}' for r, c in regime_counts.items()} }")

    return states, wass_ofi, wass_obi


def main():
    validate_config()

    print("\n" + "=" * 70)
    print("ADVERSE SELECTION SWITCHING — TICK-BY-TICK PIPELINE")
    print("=" * 70)
    print(f"Tickers       : {TICKERS}")
    print(f"Wass window   : {TICK_WASS_WINDOW} ticks  stride={TICK_WASS_STRIDE}")
    print(f"Horizon       : {HORIZON_TICKS} ticks")
    print(f"Threshold     : {IMPACT_THRESHOLD_TICKS} ticks = ${IMPACT_THRESHOLD_TICKS * TICK_SIZE:.2f}")
    print(f"HMM regimes   : {HMM_N_REGIMES}  persistence={HMM_PERSISTENCE}")
    print("=" * 70 + "\n")

    t_start = time.time()
    all_comparisons = []

    for ticker in TICKERS:
        print(f"\n{'='*70}")
        print(f"  {ticker}")
        print(f"{'='*70}")

        # ------------------------------------------------------------------
        # 1. Load tick-by-tick data (no resampling)
        # ------------------------------------------------------------------
        t0 = time.time()
        try:
            tick_df = load_ticker_ticks(ticker)
        except FileNotFoundError as e:
            print(f"  WARN: {e} — skipping {ticker}")
            continue

        n_raw = len(tick_df)
        print(f"  Raw ticks: {n_raw:,}  ({time.time()-t0:.1f}s)")

        # ------------------------------------------------------------------
        # 2. Per-ticker tick-level HMM on Wasserstein features
        # ------------------------------------------------------------------
        try:
            states, wass_ofi, wass_obi = compute_tick_hmm_states(tick_df, ticker=ticker)
        except ValueError as e:
            print(f"  WARN: {e} — skipping {ticker}")
            continue

        # ------------------------------------------------------------------
        # 3. Align tick data to the valid Wasserstein range [W, N-W)
        #
        # wass_ofi/obi have length N - 2*W, corresponding to tick_df rows
        # [W, N-W).  We slice tick_df to this range so that:
        #   - aligned_lob[t] has regime states[t]
        #   - aligned_lob[t + HORIZON] gives the future price for labeling
        #
        # We further restrict to [0, len(wass) - HORIZON) so that the
        # horizon look-ahead stays within the valid range.
        # ------------------------------------------------------------------
        W = TICK_WASS_WINDOW
        H = HORIZON_TICKS

        # Rows of tick_df used for features
        lob_valid = tick_df.iloc[W : n_raw - W].reset_index(drop=True)

        # lob_valid has len(wass_ofi) rows.
        # The price at t+H must also lie within lob_valid, so we cap at
        # len(lob_valid) - H to keep the horizon within the valid window.
        n_valid = len(lob_valid) - H
        if n_valid < 200:
            print(f"  WARN: not enough ticks after alignment ({n_valid}) — skipping")
            continue

        lob_aligned = lob_valid.iloc[:n_valid]
        states_aligned = states[:n_valid]

        # ------------------------------------------------------------------
        # 4. Build adverse selection dataset at tick level
        # ------------------------------------------------------------------
        df = build_adverse_selection_dataset(
            lob_data=lob_valid,   # full valid range; build_* handles horizon internally
            states=states,
            horizon=H,
            impact_threshold_ticks=IMPACT_THRESHOLD_TICKS,
            tick_size=TICK_SIZE,
        )

        if len(df) < 500:
            print(f"  WARN: dataset too small ({len(df)}) — skipping")
            continue

        # ------------------------------------------------------------------
        # 5. Estimate switching logistic regression + benchmarks
        # ------------------------------------------------------------------
        coeffs, comparison_df, homogeneity_test = estimate_adverse_selection_switching(
            df, n_regimes=HMM_N_REGIMES
        )

        # ------------------------------------------------------------------
        # 6. Figures and LaTeX tables
        # ------------------------------------------------------------------
        plot_adverse_selection_results(
            df, coeffs, comparison_df, HMM_N_REGIMES, ticker=ticker
        )
        generate_latex_tables(comparison_df, homogeneity_test, coeffs, ticker=ticker)

        comparison_df['Ticker'] = ticker
        all_comparisons.append(comparison_df)

    # ------------------------------------------------------------------
    # Cross-ticker summary
    # ------------------------------------------------------------------
    if all_comparisons:
        full = pd.concat(all_comparisons, ignore_index=True)
        full.to_csv(RESULTS_DIR / 'adverse_selection_all_tickers.csv', index=False)

        print("\n" + "=" * 70)
        print("CROSS-TICKER SUMMARY  (AUC across tickers)")
        print("=" * 70)
        summary = full.groupby('Model')['AUC'].agg(['mean', 'std'])
        summary.columns = ['Mean AUC', 'Std AUC']
        print(summary.to_string())
        print("=" * 70)

    elapsed = time.time() - t_start
    print(f"\nPipeline complete in {elapsed:.1f}s ({elapsed/60:.1f} min)\n")


if __name__ == "__main__":
    main()
