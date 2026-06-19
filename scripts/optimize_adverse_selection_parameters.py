"""
Walk-Forward Cross-Validation for Adverse Selection HMM Parameters
====================================================================

Avoids lookahead bias by:
  1. Restricting all parameter search to the TRAIN split (first TRAIN_RATIO
     of ticks per ticker).
  2. Using an expanding-window walk-forward CV within TRAIN, so each
     validation fold is always strictly after its training fold.
  3. Computing only CAUSAL Wasserstein features (W uses series[i-2w:i-w]
     and series[i-w:i], no future data at position i).

Scoring criterion per fold
--------------------------
  ARI(HMM_val, KMeans_val) — label-invariant clustering consistency.
  KMeans is fitted independently on the val fold → truly out-of-sample.

  A small stability penalty λ·std(ARI) rewards parameters that are
  consistently good across folds rather than lucky on one fold.

Combined score (per parameter combination, per ticker):
  score = mean(ARI_val) - LAMBDA_STD * std(ARI_val)

Outputs
-------
  data/results/best_parameters_adverse_selection.csv
  data/results/cv_results_adverse_selection.csv

Run
---
    MPLBACKEND=Agg PYTHONUTF8=1 PYTHONPATH=/c/Users/Alexis/Desktop/Quant/RIF \\
    C:/Users/Alexis/anaconda3/envs/kedro-environment/python.exe -u \\
    scripts/optimize_adverse_selection_parameters.py
"""

import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from src.config import TICKERS, RESULTS_DIR, validate_config
from src.data.loader import load_ticker_ticks
from src.features.wasserstein import compute_tick_wasserstein_causal
from src.models.hmm_optimal import LocalHMM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

TRAIN_RATIO = 0.70
N_FOLDS = 5
LAMBDA_STD = 0.5       # penalty weight for ARI instability across folds
WASS_STRIDE = 50       # stride for causal Wasserstein (performance)
MIN_FOLD_OBS = 200     # skip fold if training portion has fewer observations

PARAM_GRID = {
    "tick_wass_window": [50, 100, 200],
    "local_persistence": [0.85, 0.90, 0.95],
    "local_smoothing":   [10, 20, 50],
    "n_regimes":         [2, 3, 4],
}

# Light HMM settings during grid search (fast); final fit uses heavier settings
CV_HMM_N_INIT = 2
CV_HMM_N_ITER = 100


# ============================================================================
# Walk-forward fold generator
# ============================================================================

def walk_forward_folds(n_obs: int, n_folds: int = N_FOLDS):
    """
    Expanding-window fold indices for a series of length n_obs.

    The series is cut into (n_folds + 1) equal segments.
    Fold k uses:
      - train: [0, (k+1) * seg]
      - val  : [(k+1) * seg, (k+2) * seg]

    Returns list of (train_end, val_start, val_end) tuples.
    Always strictly chronological: val is always after train.
    """
    seg = n_obs // (n_folds + 1)
    folds = []
    for k in range(n_folds):
        train_end = (k + 1) * seg
        val_start = train_end
        val_end = min((k + 2) * seg, n_obs)
        if train_end >= MIN_FOLD_OBS and val_end > val_start:
            folds.append((train_end, val_start, val_end))
    return folds


# ============================================================================
# Fold-level scoring
# ============================================================================

def _score_fold(
    X_full: np.ndarray,
    train_end: int,
    val_start: int,
    val_end: int,
    n_regimes: int,
    persistence: float,
    smooth_window: int,
) -> float:
    """
    Fit HMM on X_full[:train_end], predict on X_full[val_start:val_end],
    return ARI(HMM_val, KMeans_val).

    Returns np.nan if fitting fails or val fold is too small.
    """
    X_tr = X_full[:train_end]
    X_val = X_full[val_start:val_end]

    if len(X_tr) < MIN_FOLD_OBS or len(X_val) < n_regimes * 10:
        return np.nan

    # Fit HMM on train fold
    try:
        model = LocalHMM(
            n_regimes=n_regimes,
            persistence=persistence,
            smooth_window=smooth_window,
            n_init=CV_HMM_N_INIT,
            n_iter=CV_HMM_N_ITER,
            verbose=False,
        )
        model.fit(X_tr)
    except RuntimeError:
        return np.nan

    # Predict regimes on val fold using the fitted model (no lookahead)
    try:
        states_val = model.predict(X_val)
    except Exception:
        return np.nan

    if len(np.unique(states_val)) < 2:
        return 0.0

    # KMeans on val fold as reference partition (label-invariant via ARI)
    scaler = StandardScaler()
    X_val_sc = scaler.fit_transform(X_val)
    km = KMeans(n_clusters=n_regimes, random_state=42, n_init=5, max_iter=100)
    km_val = km.fit_predict(X_val_sc)

    return float(adjusted_rand_score(states_val, km_val))


# ============================================================================
# Per-ticker evaluation worker
# ============================================================================

def _evaluate_combination(
    params: dict,
    wass_cache: dict,
    n_train_valid: int,
) -> dict | None:
    """
    Score one parameter combination using walk-forward CV.

    wass_cache[window] = np.ndarray of shape (n_train_valid, 2)
    where column 0 = causal OFI Wasserstein, column 1 = causal OBI Wasserstein.
    n_train_valid = len(wass_cache[window]) (after 2*window offset).
    """
    window = params["tick_wass_window"]
    if window not in wass_cache:
        return None

    X_full = wass_cache[window]  # (n_train_valid, 2)
    n_obs = len(X_full)
    folds = walk_forward_folds(n_obs, N_FOLDS)

    if not folds:
        return None

    ari_scores = []
    for train_end, val_start, val_end in folds:
        ari = _score_fold(
            X_full, train_end, val_start, val_end,
            params["n_regimes"],
            params["local_persistence"],
            params["local_smoothing"],
        )
        if not np.isnan(ari):
            ari_scores.append(ari)

    if len(ari_scores) < 2:
        return None

    mean_ari = float(np.mean(ari_scores))
    std_ari = float(np.std(ari_scores))
    score = mean_ari - LAMBDA_STD * std_ari

    return {
        **params,
        "n_folds_ok": len(ari_scores),
        "mean_ari": mean_ari,
        "std_ari": std_ari,
        "score": score,
    }


# ============================================================================
# Per-ticker optimization
# ============================================================================

def optimize_ticker(ticker: str, tick_df: pd.DataFrame) -> list[dict]:
    """
    Run the full grid search for one ticker.

    Returns list of result dicts (one per parameter combination).
    """
    n_raw = len(tick_df)
    n_train = int(n_raw * TRAIN_RATIO)
    logger.info("%s: %d total ticks, using first %d (%.0f%%) as TRAIN",
                ticker, n_raw, n_train, TRAIN_RATIO * 100)

    # Pre-compute causal Wasserstein for each window on the TRAIN ticks only.
    # Using causal version: output[j] uses series[j : j+2w], which is a purely
    # retrospective computation — no future data beyond j+2w-1 is used.
    windows = sorted(set(PARAM_GRID["tick_wass_window"]))
    wass_cache: dict[int, np.ndarray] = {}

    for w in windows:
        ofi_train = tick_df["ofi"].values[:n_train]
        obi_train = tick_df["obi"].values[:n_train]
        logger.info("%s: computing causal Wasserstein (window=%d, stride=%d) ...",
                    ticker, w, WASS_STRIDE)
        ofi_wass = compute_tick_wasserstein_causal(ofi_train, window=w, stride=WASS_STRIDE)
        obi_wass = compute_tick_wasserstein_causal(obi_train, window=w, stride=WASS_STRIDE)
        if len(ofi_wass) == 0:
            logger.warning("%s: window=%d too large for TRAIN, skipping", ticker, w)
            continue
        wass_cache[w] = np.column_stack([ofi_wass, obi_wass])
        logger.info("%s: window=%d -> %d valid obs", ticker, w, len(ofi_wass))

    if not wass_cache:
        logger.error("%s: no valid Wasserstein features, skipping ticker", ticker)
        return []

    # Grid search (parallelised over combinations)
    param_names = list(PARAM_GRID.keys())
    combinations = [
        dict(zip(param_names, vals))
        for vals in product(*[PARAM_GRID[k] for k in param_names])
    ]
    logger.info("%s: testing %d parameter combinations x %d folds ...",
                ticker, len(combinations), N_FOLDS)

    n_valid_per_window = {w: len(arr) for w, arr in wass_cache.items()}
    results = []

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(
                _evaluate_combination,
                combo,
                wass_cache,
                n_valid_per_window.get(combo["tick_wass_window"], 0),
            ): combo
            for combo in combinations
        }
        for fut in as_completed(futures):
            res = fut.result()
            if res is not None:
                res["ticker"] = ticker
                results.append(res)

    logger.info("%s: %d/%d combinations succeeded", ticker, len(results), len(combinations))
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    validate_config()

    logger.info("=" * 70)
    logger.info("ADVERSE SELECTION HMM — WALK-FORWARD PARAMETER OPTIMIZATION")
    logger.info("=" * 70)
    logger.info("Tickers      : %s", TICKERS)
    logger.info("Train ratio  : %.0f%%  |  CV folds: %d (expanding window)",
                TRAIN_RATIO * 100, N_FOLDS)
    logger.info("Lambda_std   : %.1f  (ARI stability penalty)", LAMBDA_STD)
    logger.info("Grid size    : %d combinations",
                len(list(product(*PARAM_GRID.values()))))
    logger.info("=" * 70)

    all_results = []

    for ticker in TICKERS:
        logger.info("\n--- %s ---", ticker)
        try:
            tick_df = load_ticker_ticks(ticker)
        except FileNotFoundError as e:
            logger.warning("%s: %s — skipping", ticker, e)
            continue

        results = optimize_ticker(ticker, tick_df)
        all_results.extend(results)

    if not all_results:
        logger.error("No results produced. Check data paths.")
        return

    cv_df = pd.DataFrame(all_results).sort_values(
        ["ticker", "score"], ascending=[True, False]
    ).reset_index(drop=True)

    # Best parameters per ticker (highest score)
    best_rows = []
    for ticker in TICKERS:
        sub = cv_df[cv_df["ticker"] == ticker]
        if sub.empty:
            continue
        best_rows.append(sub.iloc[0].to_dict())

    best_df = pd.DataFrame(best_rows)

    cv_df.to_csv(RESULTS_DIR / "cv_results_adverse_selection.csv", index=False)
    best_df.to_csv(RESULTS_DIR / "best_parameters_adverse_selection.csv", index=False)

    logger.info("\n" + "=" * 70)
    logger.info("BEST PARAMETERS PER TICKER")
    logger.info("=" * 70)
    display_cols = [
        "ticker", "tick_wass_window", "n_regimes",
        "local_persistence", "local_smoothing",
        "mean_ari", "std_ari", "score",
    ]
    logger.info("\n%s", best_df[[c for c in display_cols if c in best_df.columns]].to_string(index=False))
    logger.info("=" * 70)
    logger.info("Saved: cv_results_adverse_selection.csv and "
                "best_parameters_adverse_selection.csv")


if __name__ == "__main__":
    main()
