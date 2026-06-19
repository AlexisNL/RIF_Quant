"""
Adverse Selection Switching Pipeline
======================================

Standalone extension to the HMM-Wasserstein framework.

Requires: run_hierarchical_contagion.py to have been executed first
(reads hierarchical_states_local.csv from data/results/).

For each ticker:
  1. Load LOBSTER data (micro_price, ofi, obi at 500ms)
  2. Load per-ticker local HMM states from saved CSV
  3. Align states to raw LOB timeline
  4. Build adverse selection dataset (label = price impact horizon H)
  5. Estimate switching logistic regression per regime
  6. Compare to benchmarks (PIN static, VPIN-like, pooled)
  7. LR test of coefficient homogeneity across regimes
  8. Generate figures and LaTeX tables

Run:
    MPLBACKEND=Agg PYTHONUTF8=1 PYTHONPATH=/c/Users/Alexis/Desktop/Quant/RIF \\
    python scripts/run_adverse_selection.py
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import time
import numpy as np
import pandas as pd

from src.config import TICKERS, N_REGIMES, WASSERSTEIN_WINDOW, RESULTS_DIR, validate_config
from src.data.loader import load_and_sync_all_tickers
from src.analysis.adverse_selection import (
    build_adverse_selection_dataset,
    estimate_adverse_selection_switching,
    plot_adverse_selection_results,
    generate_latex_tables,
)

# ============================================================================
# Parameters
# ============================================================================

HORIZON = 60                    # price impact horizon (bars at 500ms = 30 seconds)
IMPACT_THRESHOLD_TICKS = 2.0   # |impact| > 2 ticks => informed
TICK_SIZE = 0.01               # $0.01 per tick


def load_local_states(results_dir: Path) -> pd.DataFrame:
    """
    Load per-ticker local HMM states saved by run_hierarchical_contagion.py.

    Returns DataFrame with columns: timestamp, state_AAPL, state_INTC, ...
    """
    states_file = results_dir / 'hierarchical_states_local.csv'
    if not states_file.exists():
        raise FileNotFoundError(
            f"{states_file} not found.\n"
            "Run scripts/run_hierarchical_contagion.py first."
        )
    df = pd.read_csv(states_file, parse_dates=['timestamp'])
    print(f"  Loaded states: {len(df)} rows, columns: {list(df.columns)}")
    return df


def align_states_to_lob(
    lob_data: pd.DataFrame,
    states_df: pd.DataFrame,
    ticker: str,
) -> np.ndarray:
    """
    Align HMM states (indexed by Wasserstein-window timestamps) to the
    LOBSTER 500ms index using nearest-match merge.

    The states cover indices [window, N] of the original LOB data.
    We join on timestamp and forward-fill gaps.

    Returns np.ndarray of length len(lob_data), dtype=int.
    """
    col = f'state_{ticker}'
    if col not in states_df.columns:
        raise KeyError(f"Column {col} not in states CSV. Available: {list(states_df.columns)}")

    states_sub = states_df[['timestamp', col]].rename(columns={col: 'regime'})
    states_sub = states_sub.set_index('timestamp')

    # Merge on datetime index using reindex + nearest fill
    merged = states_sub.reindex(
        states_sub.index.union(lob_data.index)
    ).sort_index().ffill().bfill()

    aligned = merged.reindex(lob_data.index)['regime'].fillna(0).astype(int).values
    return aligned


def main():
    validate_config()

    print("\n" + "=" * 70)
    print("ADVERSE SELECTION SWITCHING PIPELINE")
    print("=" * 70)
    print(f"Tickers   : {TICKERS}")
    print(f"Horizon   : {HORIZON} bars ({HORIZON * 0.5:.0f}s)")
    print(f"Threshold : {IMPACT_THRESHOLD_TICKS} ticks")
    print("=" * 70 + "\n")

    t_start = time.time()

    # ------------------------------------------------------------------
    # 1. Load LOBSTER data (raw OFI, OBI, micro_price)
    # ------------------------------------------------------------------
    print("[1/3] Loading LOBSTER data...")
    synced_data = load_and_sync_all_tickers(TICKERS)
    print(f"  OK - {len(synced_data)} tickers, {len(next(iter(synced_data.values())))} bars each\n")

    # ------------------------------------------------------------------
    # 2. Load HMM local states
    # ------------------------------------------------------------------
    print("[2/3] Loading HMM local states from results/...")
    states_df = load_local_states(RESULTS_DIR)
    print()

    # ------------------------------------------------------------------
    # 3. Per-ticker adverse selection estimation
    # ------------------------------------------------------------------
    print("[3/3] Estimating adverse selection switching model per ticker...\n")

    all_comparisons = []

    for ticker in TICKERS:
        print(f"--- {ticker} " + "-" * 50)

        lob_data = synced_data[ticker]

        # Align states
        try:
            states = align_states_to_lob(lob_data, states_df, ticker)
        except KeyError as e:
            print(f"  WARN: {e} — skipping {ticker}")
            continue

        # Build dataset
        df = build_adverse_selection_dataset(
            lob_data=lob_data,
            states=states,
            horizon=HORIZON,
            impact_threshold_ticks=IMPACT_THRESHOLD_TICKS,
            tick_size=TICK_SIZE,
        )

        if len(df) < 100:
            print(f"  WARN: too few observations for {ticker}, skipping")
            continue

        # Estimate switching model
        coeffs, comparison_df, homogeneity_test = estimate_adverse_selection_switching(
            df, n_regimes=N_REGIMES
        )

        # Visualize
        plot_adverse_selection_results(df, coeffs, comparison_df, N_REGIMES, ticker=ticker)

        # LaTeX tables
        generate_latex_tables(comparison_df, homogeneity_test, coeffs, ticker=ticker)

        comparison_df['Ticker'] = ticker
        all_comparisons.append(comparison_df)
        print()

    # ------------------------------------------------------------------
    # Summary across tickers
    # ------------------------------------------------------------------
    if all_comparisons:
        full = pd.concat(all_comparisons, ignore_index=True)
        full.to_csv(RESULTS_DIR / 'adverse_selection_all_tickers.csv', index=False)

        print("=" * 70)
        print("CROSS-TICKER SUMMARY  (mean AUC)")
        print("=" * 70)
        summary = full.groupby('Model')['AUC'].agg(['mean', 'std'])
        print(summary.to_string())
        print("=" * 70)

    elapsed = time.time() - t_start
    print(f"\nPipeline complete in {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
