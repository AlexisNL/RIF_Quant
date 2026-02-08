"""
Per-ticker optimization for hierarchical HMM pipeline.

Modes (independent):
1) Local HMM optimization per ticker
2) Global Meta-HMM (on local probs)
3) Global direct HMM (on concatenated temporal features)

By default, runs all modes. Use CLI flags to select.
"""

from itertools import product
from typing import Dict, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import argparse

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from src.config import (
    TICKERS,
    ANALYSIS_DATE,
    RAW_DATA_DIR,
    RESULTS_DIR,
    RESAMPLE_FREQ,
    HMM_REQUIRE_CONVERGENCE,
    HMM_MIN_REGIME_SHARE,
    MMD_WINDOW,
    MMD_STEP,
    MMD_PENALTY_WEIGHT,
    MMD_PENALTY_TARGET_CORR,
)
from src.data.loader import load_and_sync_all_tickers
from src.features.mad_normalizer import normalize_innovations_mad
from src.features.wasserstein import compute_wasserstein_temporal_features
from src.models.hmm_optimal import fit_optimized_hmm_with_probs, fit_optimized_hmm
from src.models.meta_hmm import fit_hierarchical_hmm_pipeline


PARAM_GRID = {
    "mad_window": [50, 100, 150],
    "wasserstein_window": [50, 100, 150],
    "local_persistence": [0.85, 0.90, 0.95],
    "local_smoothing": [10, 20, 30],
    "n_regimes": [3],
}

GLOBAL_PARAM_GRID = {
    "global_persistence": [0.85, 0.90, 0.95],
    "global_smoothing": [10, 20, 30],
}

METRICS = ["Price", "OFI", "OBI"]


def _rbf_mmd(x: np.ndarray, y: np.ndarray, gamma: float = None) -> float:
    if len(x) == 0 or len(y) == 0:
        return np.nan
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    if gamma is None:
        all_vals = np.concatenate([x, y], axis=0)
        dists = np.abs(all_vals - all_vals.T)
        med = np.median(dists[dists > 0])
        if not np.isfinite(med) or med == 0:
            gamma = 1.0
        else:
            gamma = 1.0 / (2 * med * med)
    k_xx = np.exp(-gamma * (x - x.T) ** 2)
    k_yy = np.exp(-gamma * (y - y.T) ** 2)
    k_xy = np.exp(-gamma * (x - y.T) ** 2)
    return float(k_xx.mean() + k_yy.mean() - 2 * k_xy.mean())


def _compute_mmd_series(series: np.ndarray, states: np.ndarray) -> pd.DataFrame:
    rows = []
    n = len(series)
    for start in range(0, n - MMD_WINDOW + 1, MMD_STEP):
        end = start + MMD_WINDOW
        window_states = states[start:end]
        window_vals = series[start:end]
        x = window_vals[window_states == 0]
        y = window_vals[window_states == 1]
        mmd_val = _rbf_mmd(x, y)
        toxic = float(np.mean(np.abs(window_vals)))
        rows.append(
            {
                "start_idx": start,
                "end_idx": end,
                "mmd_r0_r1": mmd_val,
                "metric_toxicity": toxic,
            }
        )
    return pd.DataFrame(rows)


def _evaluate_one_ticker(params, ticker, wass_X):
    mad_window = params["mad_window"]
    wass_window = params["wasserstein_window"]
    local_persist = params["local_persistence"]
    local_smooth = params["local_smoothing"]
    n_regimes = params["n_regimes"]

    try:
        ticker_cols = []
        for metric in METRICS:
            col_name = f"{ticker}_{metric}"
            if col_name in wass_X.columns:
                ticker_cols.append(col_name)
        if not ticker_cols:
            return None

        wass_X_ticker = wass_X[ticker_cols]
        model, states, _ = fit_optimized_hmm_with_probs(
            wass_X_ticker,
            n_components=n_regimes,
            persistence=local_persist,
            smooth_window=local_smooth,
        )

        if hasattr(model, "monitor_") and HMM_REQUIRE_CONVERGENCE:
            if not bool(model.monitor_.converged):
                return None

        regime_counts = pd.Series(states).value_counts(normalize=True)
        if regime_counts.min() < HMM_MIN_REGIME_SHARE:
            return None

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(wass_X_ticker)
        kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        kmeans_states = kmeans.fit_predict(X_scaled)
        ari_local = float(adjusted_rand_score(states, kmeans_states))

        # MMD penalty proxy (OFI only)
        mmd_penalty = 0.0
        ofi_col = f"{ticker}_OFI"
        if ofi_col in wass_X.columns:
            ofi_series = wass_X[ofi_col].values
            mmd_df = _compute_mmd_series(ofi_series, states)
            if len(mmd_df) > 1:
                corr = np.corrcoef(mmd_df["mmd_r0_r1"], mmd_df["metric_toxicity"])[0, 1]
                if not np.isfinite(corr):
                    corr = 0.0
                mmd_penalty = max(0.0, MMD_PENALTY_TARGET_CORR - corr) * MMD_PENALTY_WEIGHT

        score = ari_local - mmd_penalty

        return {
            "ticker": ticker,
            "mad_window": mad_window,
            "wasserstein_window": wass_window,
            "local_persistence": local_persist,
            "local_smoothing": local_smooth,
            "n_regimes": n_regimes,
            "ari_local": ari_local,
            "mmd_penalty": mmd_penalty,
            "score": score,
            "success": True,
        }
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Optimize hierarchical HMM pipeline")
    parser.add_argument("--locals-only", action="store_true", help="Run only local HMM optimization")
    parser.add_argument("--meta-only", action="store_true", help="Run only global Meta-HMM")
    parser.add_argument("--direct-only", action="store_true", help="Run only global direct HMM")
    args = parser.parse_args()

    run_locals = True
    run_meta = True
    run_direct = True
    if args.locals_only or args.meta_only or args.direct_only:
        run_locals = args.locals_only
        run_meta = args.meta_only
        run_direct = args.direct_only

    print("=" * 70)
    print("OPTIMIZE HIERARCHICAL PARAMETERS (PER TICKER)")
    print("=" * 70)
    print(f"Analysis date: {ANALYSIS_DATE}")
    print(f"Tickers: {', '.join(TICKERS)}\n")

    print("=" * 70)
    print("LOAD DATA")
    print("=" * 70)
    synced_data = load_and_sync_all_tickers(TICKERS, RESAMPLE_FREQ, RAW_DATA_DIR)
    print(f"Loaded {len(synced_data[TICKERS[0]])} observations per ticker\n")

    print("=" * 70)
    print("PRECOMPUTE FEATURES")
    print("=" * 70)
    mad_windows = sorted(PARAM_GRID["mad_window"])
    wass_windows = sorted(PARAM_GRID["wasserstein_window"])

    innov_cache: Dict[int, Dict] = {}
    for mad_window in mad_windows:
        innov_cache[mad_window] = normalize_innovations_mad(
            synced_data,
            TICKERS,
            window=mad_window,
            min_periods=max(30, mad_window // 2),
        )

    feature_cache: Dict[Tuple[int, int], pd.DataFrame] = {}
    for mad_window in mad_windows:
        for wass_window in wass_windows:
            wass_X = compute_wasserstein_temporal_features(
                innov_cache[mad_window],
                TICKERS,
                window=wass_window,
            )
            feature_cache[(int(mad_window), int(wass_window))] = wass_X

    results_df = None
    best_df = None

    if run_locals:
        print("=" * 70)
        print("RUN PER-TICKER OPTIMIZATION")
        print("=" * 70)

        param_names = list(PARAM_GRID.keys())
        all_combinations = list(product(*[PARAM_GRID[k] for k in param_names]))

        max_workers = os.cpu_count() or 1
        print(f"Running in parallel with {max_workers} workers\n")

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for ticker in TICKERS:
                for combination in all_combinations:
                    params = dict(zip(param_names, combination))
                    cache_key = (params["mad_window"], params["wasserstein_window"])
                    wass_X = feature_cache[cache_key]
                    futures[
                        executor.submit(_evaluate_one_ticker, params, ticker, wass_X)
                    ] = (ticker, params)

            for future in as_completed(futures):
                result = future.result()
                if result is not None and result.get("success"):
                    results.append(result)

        results_df = pd.DataFrame(results)
        if results_df.empty:
            print("No successful combinations for locals.")
        else:
            results_df = results_df.sort_values("score", ascending=False).reset_index(drop=True)
            output_file = RESULTS_DIR / "optimization_hierarchical_results_per_ticker.csv"
            results_df.to_csv(output_file, index=False)
            print(f"Saved per-ticker results to {output_file}")

            best_rows = []
            for ticker in TICKERS:
                sub = results_df[results_df["ticker"] == ticker]
                if sub.empty:
                    continue
                best_rows.append(sub.iloc[0])

            best_df = pd.DataFrame(best_rows).reset_index(drop=True)
            output_file = RESULTS_DIR / "best_parameters_hierarchical_per_ticker.csv"
            best_df.to_csv(output_file, index=False)
            print(f"Saved best params per ticker to {output_file}")

            # Also save TXT (legacy-friendly)
            txt_path = RESULTS_DIR / "best_parameters_hierarchical_per_ticker.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("# Best parameters per ticker (generated)\n\n")
                for _, row in best_df.iterrows():
                    f.write(f"[{row['ticker']}]\n")
                    f.write(f"mad_window = {int(row['mad_window'])}\n")
                    f.write(f"wasserstein_window = {int(row['wasserstein_window'])}\n")
                    f.write(f"local_persistence = {float(row['local_persistence'])}\n")
                    f.write(f"local_smoothing = {int(row['local_smoothing'])}\n")
                    f.write(f"n_regimes = {int(row['n_regimes'])}\n")
                    f.write(f"ari_local = {float(row['ari_local'])}\n")
                    f.write(f"mmd_penalty = {float(row['mmd_penalty'])}\n")
                    f.write(f"score = {float(row['score'])}\n\n")
            print(f"Saved best params per ticker to {txt_path}")

            print("\nBEST PER TICKER")
            print(
                best_df[
                    [
                        "ticker",
                        "mad_window",
                        "wasserstein_window",
                        "local_persistence",
                        "local_smoothing",
                        "n_regimes",
                        "ari_local",
                        "mmd_penalty",
                        "score",
                    ]
                ].to_string(index=False)
            )

    if run_meta or run_direct:
        if best_df is None:
            best_path = RESULTS_DIR / "best_parameters_hierarchical_per_ticker.csv"
            if best_path.exists():
                best_df = pd.read_csv(best_path)
            else:
                print("Missing best_parameters_hierarchical_per_ticker.csv. Run locals or provide the file.")
                return

    if run_meta:
        print("\n" + "=" * 70)
        print("FIT GLOBAL META-HMM WITH PER-TICKER PARAMS")
        print("=" * 70)

        local_states = {}
        local_state_probs = {}
        for _, row in best_df.iterrows():
            ticker = row["ticker"]
            cache_key = (int(row["mad_window"]), int(row["wasserstein_window"]))
            wass_X = feature_cache[cache_key]
            ticker_cols = [f"{ticker}_{m}" for m in METRICS if f"{ticker}_{m}" in wass_X.columns]
            if not ticker_cols:
                continue

            model, states, state_probs = fit_optimized_hmm_with_probs(
                wass_X[ticker_cols],
                n_components=int(row["n_regimes"]),
                persistence=float(row["local_persistence"]),
                smooth_window=int(row["local_smoothing"]),
            )
            local_states[ticker] = states
            local_state_probs[ticker] = state_probs

        if len(local_state_probs) == 0:
            print("No local models available for global fit.")
            return

        n_regimes = int(best_df["n_regimes"].iloc[0])
        meta_hmm, global_states, global_probs, _ = fit_hierarchical_hmm_pipeline(
            local_state_probs,
            local_states,
            list(local_states.keys()),
            n_global_regimes=n_regimes,
            persistence=0.90,
            smooth_window=20,
        )

        kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        global_km = kmeans.fit_predict(global_probs)
        ari_global = float(adjusted_rand_score(global_states, global_km))

        meta_vs_local = []
        for t in local_states:
            ls = local_states[t]
            m = min(len(global_states), len(ls))
            meta_vs_local.append(adjusted_rand_score(global_states[:m], ls[:m]))
        ari_meta_vs_local = float(np.mean(meta_vs_local)) if meta_vs_local else np.nan

        print(f"Global ARI (HMM vs KMeans) = {ari_global:.3f}")
        print(f"Global ARI vs locals (mean) = {ari_meta_vs_local:.3f}")

        # MMD diagnostics for meta global (Price/OFI/OBI)
        meta_mmd_stats = {}
        common_key = (int(best_df["mad_window"].min()), int(best_df["wasserstein_window"].min()))
        wass_X_global = feature_cache[common_key]
        global_cols = [c for c in wass_X_global.columns if any(c.endswith(suf) for suf in ["_Price", "_OFI", "_OBI"])]
        X_global = wass_X_global[global_cols]
        for metric in ["Price", "OFI", "OBI"]:
            cols = [c for c in X_global.columns if c.endswith(f"_{metric}")]
            if not cols:
                continue
            series = X_global[cols].mean(axis=1).values
            mmd_df = _compute_mmd_series(series, global_states[: len(series)])
            meta_mmd_stats[f"meta_mmd_{metric.lower()}_mean"] = float(mmd_df["mmd_r0_r1"].mean())
        if meta_mmd_stats:
            print("Meta global MMD means:", meta_mmd_stats)

    if run_direct:
        print("\n" + "=" * 70)
        print("FIT GLOBAL DIRECT HMM (ALL TICKER METRICS)")
        print("=" * 70)

        common_key = (int(best_df["mad_window"].min()), int(best_df["wasserstein_window"].min()))
        wass_X_global = feature_cache[common_key]
        global_cols = [c for c in wass_X_global.columns if any(c.endswith(suf) for suf in ["_Price", "_OFI", "_OBI"])]
        X_global = wass_X_global[global_cols]

        n_regimes = int(best_df["n_regimes"].iloc[0])
        global_direct_rows = []
        for gp in GLOBAL_PARAM_GRID["global_persistence"]:
            for gs in GLOBAL_PARAM_GRID["global_smoothing"]:
                _, states_g = fit_optimized_hmm(
                    X_global,
                    n_components=n_regimes,
                    persistence=gp,
                    smooth_window=gs,
                )
                X_scaled = StandardScaler().fit_transform(X_global)
                km = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
                km_states = km.fit_predict(X_scaled)
                ari_direct = float(adjusted_rand_score(states_g, km_states))

                global_direct_rows.append(
                    {
                        "global_persistence": gp,
                        "global_smoothing": gs,
                        "ari_direct": ari_direct,
                    }
                )

        global_direct_df = pd.DataFrame(global_direct_rows).sort_values("ari_direct", ascending=False)
        output_file = RESULTS_DIR / "optimization_global_direct.csv"
        global_direct_df.to_csv(output_file, index=False)
        print(f"Saved direct-global optimization to {output_file}")

        best_direct = global_direct_df.iloc[0]
        print(
            "Best direct-global: "
            f"p={best_direct['global_persistence']}, s={best_direct['global_smoothing']}, "
            f"ARI={best_direct['ari_direct']:.3f}"
        )


if __name__ == "__main__":
    main()
