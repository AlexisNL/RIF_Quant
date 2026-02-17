# -*- coding: utf-8 -*-
"""
Hierarchical Contagion Detection Pipeline (Two-Level Architecture)
=================================================================

Key idea: second-order HMM for sector-level contagion.

Architecture:
-------------
LEVEL 1 (Local)  : HMM per asset -> P(regime | asset)
LEVEL 2 (Global) : Meta-HMM over all probabilities -> sector regime

Key benefits:
-------------
- Resolves label switching (aligns regime semantics)
- Filters noise (ignores isolated transitions)
- Detects contagion (co-movements of regimes)
- Identifies the "Patient Zero" (Transfer Entropy)

Usage::

    pipeline = ContagionPipeline(tickers=TICKERS, ...).run()

    # Or step by step:
    pipeline = ContagionPipeline(tickers=TICKERS, ...)
    pipeline.load_data().extract_features().fit_local_hmms().fit_global_hmms()
    pipeline.analyze_leadlag().analyze_contagion()
    pipeline.plot_visualizations().generate_outputs().print_summary()
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Imports du projet
from src.config import (
    TICKERS,
    ANALYSIS_DATE,
    RAW_DATA_DIR,
    RESULTS_DIR,
    N_REGIMES,
    HMM_PERSISTENCE_LOCAL,
    HMM_SMOOTHING_LOCAL,
    HMM_PERSISTENCE_GLOBAL,
    HMM_SMOOTHING_GLOBAL,
    HMM_COV_FULL_CORR_THRESHOLD,
    MMD_WINDOW,
    MMD_STEP,
    LEADLAG_MAX_LAG,
    LEADLAG_QUANTILES,
    WASSERSTEIN_WINDOW,
)
from src.data.loader import load_all_tickers
from src.features.mad_normalizer import normalize_innovations_mad
from src.features.wasserstein import (
    compute_wasserstein_temporal_features,
    _compute_temporal_wasserstein_series,
)
from src.models.hmm_optimal import fit_optimized_hmm_with_probs
from src.models.meta_hmm import MetaHMM, fit_hierarchical_hmm_pipeline
from src.analysis.contagion_metrics import (
    compute_transfer_entropy_matrix,
    compute_transfer_entropy_matrix_significance,
    compute_regime_correlation,
    identify_patient_zero,
    visualize_contagion_network,
)
from src.analysis.leadlag import (
    analyze_multimetric_leadlag_by_model,
    analyze_interticker_leadlag_by_metric_quantile,
)
from src.analysis.event_study import analyze_event_goog_spike
from src.visualization.regime_plots import plot_regime_statistics


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

PAPER_DIR = Path("paper")
PAPER_FIGURES_DIR = PAPER_DIR / "figures"
PAPER_TABLES_DIR = PAPER_DIR / "tables"
PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)

_GREEN, _BLUE, _RED = "#88cc88", "#7799dd", "#dd6666"
_LOCAL_REGIME_COLORS = {
    "AAPL": [_GREEN, _BLUE, _RED],
    "INTC": [_GREEN, _BLUE, _RED],
    "GOOG": [_GREEN, _RED, _BLUE],
    "AMZN": [_GREEN, _RED, _BLUE],
    "MSFT": [_RED, _BLUE, _GREEN],
}
_LOCAL_REGIME_LABELS = {
    "AAPL": ["Regime 0 (Calm)", "Regime 1 (Intermediate)", "Regime 2 (Stressed)"],
    "INTC": ["Regime 0 (Calm)", "Regime 1 (Intermediate)", "Regime 2 (Stressed)"],
    "GOOG": ["Regime 0 (Calm)", "Regime 1 (Stressed)", "Regime 2 (Intermediate)"],
    "AMZN": ["Regime 0 (Calm)", "Regime 1 (Stressed)", "Regime 2 (Intermediate)"],
    "MSFT": ["Regime 0 (Stressed)", "Regime 1 (Intermediate)", "Regime 2 (Calm)"],
}


# ---------------------------------------------------------------------------
# Module-level utility functions (stateless helpers)
# ---------------------------------------------------------------------------

def _save_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    """Write a DataFrame to a LaTeX table file with caption and label."""
    latex = df.to_latex(index=False, escape=False, float_format="%.4f")
    out = (
        "\\begin{table}[H]\n"
        "\\centering\\small\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{latex}"
        "\\end{table}\n"
    )
    path.write_text(out, encoding="utf-8")


def _rbf_mmd(x: np.ndarray, y: np.ndarray, gamma: float = None) -> float:
    """Compute RBF-kernel MMD between two 1D arrays."""
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


def _compute_mmd_series(
    series: np.ndarray, states: np.ndarray, mmd_window: int, mmd_step: int
) -> pd.DataFrame:
    """Compute MMD statistics per regime for a given series."""
    rows = []
    n = len(series)
    for start in range(0, n - mmd_window + 1, mmd_step):
        end = start + mmd_window
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


def _plot_state_timeline(states: np.ndarray, title: str, path: Path) -> None:
    """Plot a regime timeline and save it to disk."""
    plt.figure(figsize=(12, 2.5))
    plt.plot(states, linewidth=0.8)
    plt.title(title)
    plt.xlabel("Time (obs)")
    plt.ylabel("State")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def _plot_state_hist(states: np.ndarray, title: str, path: Path) -> None:
    """Plot a regime histogram and save it to disk."""
    values, counts = np.unique(states, return_counts=True)
    plt.figure(figsize=(6, 4))
    plt.bar(values, counts, color="steelblue", alpha=0.8)
    plt.title(title)
    plt.xlabel("State")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def _plot_feature_by_regime(
    df: pd.DataFrame, states: np.ndarray, title: str, path: Path
) -> None:
    """Plot feature distributions by regime and save."""
    plot_df = df.copy()
    plot_df["regime"] = states[: len(plot_df)]
    melted = plot_df.melt(id_vars="regime", var_name="feature", value_name="value")
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=melted, x="feature", y="value", showfliers=False)
    plt.title(title)
    plt.xlabel("Feature")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def _plot_posterior_stacked(
    probs: np.ndarray,
    title: str,
    path: Path,
    colors: list = None,
    regime_labels: list = None,
) -> None:
    """Stacked area plot of posterior regime probabilities."""
    default_colors = ["#ff9999", "#99cc99", "#9999ff"]
    n_regimes = probs.shape[1]
    use_colors = (colors if colors else default_colors)[:n_regimes]
    use_labels = regime_labels if regime_labels else [f"Regime {k}" for k in range(n_regimes)]
    fig, ax = plt.subplots(figsize=(14, 3.5))
    x = np.arange(probs.shape[0])
    ax.stackplot(
        x,
        *[probs[:, k] for k in range(n_regimes)],
        labels=use_labels[:n_regimes],
        colors=use_colors,
        alpha=0.85,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(regime)")
    ax.set_xlabel("Time (obs)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def _heatmap_annotate(ax, mat_values, annot_strings, fontsize=10):
    """Add text annotations to heatmap cells manually."""
    nr, nc = mat_values.shape
    vmax = max(abs(np.nanmin(mat_values)), abs(np.nanmax(mat_values)), 1e-9)
    for i in range(nr):
        for j in range(nc):
            v = mat_values[i, j]
            txt = annot_strings[i][j]
            if pd.isna(v) or txt == "":
                continue
            color = "white" if abs(v) / vmax > 0.6 else "black"
            ax.text(
                j + 0.5, i + 0.5, txt,
                ha="center", va="center",
                fontsize=fontsize, color=color, fontweight="bold",
            )


def _plot_regime_characteristics(
    df: pd.DataFrame, states: np.ndarray, title: str, path: Path
) -> pd.DataFrame:
    """Plot per-regime metric profiles for a ticker."""
    metrics = df.columns.tolist()
    rows = []
    for regime in np.unique(states):
        mask = states == regime
        row = {"regime": int(regime)}
        for m in metrics:
            row[f"{m}_mean"] = float(df.loc[mask, m].mean())
            row[f"{m}_std"] = float(df.loc[mask, m].std())
        rows.append(row)
    stats_df = pd.DataFrame(rows)

    plt.figure(figsize=(10, 5))
    for m in metrics:
        plt.errorbar(
            stats_df["regime"],
            stats_df[f"{m}_mean"],
            yerr=stats_df[f"{m}_std"],
            marker="o",
            label=m,
        )
    plt.title(title)
    plt.xlabel("Regime")
    plt.ylabel("Mean (+-std)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return stats_df


def _compute_leadlag_significant(
    source: np.ndarray,
    target: np.ndarray,
    quantiles: list,
    max_lag: int,
    alpha: float,
    min_obs: int,
) -> pd.DataFrame:
    """Compute significant lead-lag relationships with p-values."""
    lags = np.arange(-max_lag, max_lag + 1)
    rows = []

    for q in quantiles:
        threshold = np.percentile(source, q * 100)
        q_label = f"Q{int(q * 100)}"
        mask = source <= threshold if q < 0.5 else source >= threshold

        source_sub = np.array(source)[mask]
        target_sub = np.array(target)[mask]

        for lag in lags:
            if len(source_sub) <= abs(lag) or len(source_sub) < min_obs:
                continue
            if lag < 0:
                r, p = stats.spearmanr(source_sub[-lag:], target_sub[:lag], nan_policy="omit")
            elif lag > 0:
                r, p = stats.spearmanr(source_sub[:-lag], target_sub[lag:], nan_policy="omit")
            else:
                r, p = stats.spearmanr(source_sub, target_sub, nan_policy="omit")
            if p < alpha:
                rows.append(
                    {
                        "quantile": q_label,
                        "lag_obs": int(lag),
                        "lag_seconds": lag * 0.5,
                        "correlation": float(r),
                        "p_value": float(p),
                        "n_obs": int(len(source_sub)),
                    }
                )

    return pd.DataFrame(rows)


def _plot_leadlag_from_df(df_sig: pd.DataFrame, title: str, path: Path) -> bool:
    """Plot lead-lag curves from a significant-results DataFrame."""
    if df_sig is None or df_sig.empty:
        return False
    plt.figure(figsize=(8, 4))
    for q_label in sorted(df_sig["quantile"].unique()):
        sub = df_sig[df_sig["quantile"] == q_label]
        plt.plot(sub["lag_seconds"], sub["correlation"], marker="o", label=q_label, alpha=0.7)
    plt.axvline(0, color="red", linestyle="--", linewidth=1)
    plt.axhline(0, color="gray", linestyle=":", alpha=0.5)
    plt.title(title)
    plt.xlabel("Lag (seconds)")
    plt.ylabel("Correlation (significant only)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return True


def _leadlag_corr(
    series_local: np.ndarray, series_global: np.ndarray, max_lag: int
):
    """Compute lagged correlations between two series."""
    lags = range(-max_lag, max_lag + 1)
    corrs = []
    for lag in lags:
        if lag < 0:
            x = series_local[-lag:]
            y = series_global[: len(series_global) + lag]
        elif lag > 0:
            x = series_local[: len(series_local) - lag]
            y = series_global[lag:]
        else:
            x, y = series_local, series_global
        if len(x) < 10:
            corrs.append(np.nan)
        else:
            corrs.append(stats.spearmanr(x, y, nan_policy="omit").correlation)
    return np.array(list(lags)), np.array(corrs)


def _leadlag_pvalues(
    series_local: np.ndarray, series_global: np.ndarray, lags: np.ndarray
) -> np.ndarray:
    """Compute p-values for lagged correlations."""
    pvals = []
    for lag in lags:
        if lag < 0:
            x = series_local[-lag:]
            y = series_global[: len(series_global) + lag]
        elif lag > 0:
            x = series_local[: len(series_local) - lag]
            y = series_global[lag:]
        else:
            x, y = series_local, series_global
        if len(x) < 10:
            pvals.append(np.nan)
        else:
            pvals.append(stats.spearmanr(x, y, nan_policy="omit").pvalue)
    return np.array(pvals)


def _best_leadlag(
    series_a: np.ndarray,
    series_b: np.ndarray,
    max_lag: int,
    alpha: float,
    min_obs: int,
    *,
    require_sig: bool = True,
) -> Optional[Dict]:
    """Select the best significant lag given alpha and minimum observations."""
    lags, corrs = _leadlag_corr(series_a, series_b, max_lag)
    pvals = _leadlag_pvalues(series_a, series_b, lags)
    finite_mask = np.isfinite(corrs)
    if not np.any(finite_mask):
        return None
    if require_sig:
        sig_mask = (pvals < alpha) & finite_mask
        if not np.any(sig_mask):
            return None
        sel_corrs = corrs[sig_mask]
        sel_lags = lags[sig_mask]
        sel_pvals = pvals[sig_mask]
    else:
        sel_corrs = corrs[finite_mask]
        sel_lags = lags[finite_mask]
        sel_pvals = pvals[finite_mask]
    best_idx = int(np.nanargmax(np.abs(sel_corrs)))
    return {
        "best_lag_obs": int(sel_lags[best_idx]),
        "best_lag_seconds": float(sel_lags[best_idx] * 0.5),
        "best_corr": float(sel_corrs[best_idx]),
        "best_pval": float(sel_pvals[best_idx]),
    }


def _kmeans_labels(X: np.ndarray, n_clusters: int) -> np.ndarray:
    """Fit KMeans and return cluster labels."""
    Xs = StandardScaler().fit_transform(X)
    return KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit_predict(Xs)


# ---------------------------------------------------------------------------
# ContagionPipeline class
# ---------------------------------------------------------------------------

class ContagionPipeline:
    """
    End-to-end hierarchical contagion detection pipeline.

    Encapsulates all steps:
      0. Data loading and Wasserstein feature extraction
      1. Local HMMs per ticker
      2. Meta-HMM (hierarchical global) + Direct global HMM
      3. Lead-lag analyses (local->global, ticker->ticker)
      4. Transfer Entropy + regime correlation + Patient Zero
      5. Visualizations + LaTeX reporting

    Parameters
    ----------
    tickers : list of str
    analysis_date : str
    raw_data_dir : Path
    results_dir : Path
    n_regimes : int
    hmm_persistence_local, hmm_smoothing_local : float, int
    hmm_persistence_global, hmm_smoothing_global : float, int
    hmm_cov_full_corr_threshold : float
    mmd_window, mmd_step : int
    leadlag_max_lag : int
    leadlag_quantiles : list of float
    wasserstein_window : int
    """

    def __init__(
        self,
        tickers: List[str] = None,
        analysis_date: str = None,
        raw_data_dir: Path = None,
        results_dir: Path = None,
        n_regimes: int = N_REGIMES,
        hmm_persistence_local: float = HMM_PERSISTENCE_LOCAL,
        hmm_smoothing_local: int = HMM_SMOOTHING_LOCAL,
        hmm_persistence_global: float = HMM_PERSISTENCE_GLOBAL,
        hmm_smoothing_global: int = HMM_SMOOTHING_GLOBAL,
        hmm_cov_full_corr_threshold: float = HMM_COV_FULL_CORR_THRESHOLD,
        mmd_window: int = MMD_WINDOW,
        mmd_step: int = MMD_STEP,
        leadlag_max_lag: int = LEADLAG_MAX_LAG,
        leadlag_quantiles: list = None,
        wasserstein_window: int = WASSERSTEIN_WINDOW,
    ) -> None:
        self.tickers = tickers if tickers is not None else TICKERS
        self.analysis_date = analysis_date if analysis_date is not None else ANALYSIS_DATE
        self.raw_data_dir = raw_data_dir if raw_data_dir is not None else RAW_DATA_DIR
        self.results_dir = results_dir if results_dir is not None else RESULTS_DIR
        self.n_regimes = n_regimes
        self.hmm_persistence_local = hmm_persistence_local
        self.hmm_smoothing_local = hmm_smoothing_local
        self.hmm_persistence_global = hmm_persistence_global
        self.hmm_smoothing_global = hmm_smoothing_global
        self.hmm_cov_full_corr_threshold = hmm_cov_full_corr_threshold
        self.mmd_window = mmd_window
        self.mmd_step = mmd_step
        self.leadlag_max_lag = leadlag_max_lag
        self.leadlag_quantiles = leadlag_quantiles if leadlag_quantiles is not None else LEADLAG_QUANTILES
        self.wasserstein_window = wasserstein_window

        # Results — set after each step (sklearn-style attr_ suffix)
        self.synced_data_: Optional[Dict] = None
        self.per_ticker_params_: Optional[pd.DataFrame] = None
        self.use_per_ticker_: bool = False
        self.best_global_params_: Dict = {}
        self.best_direct_params_: Dict = {}
        self.innov_cache_: Dict = {}
        self.feature_cache_: Dict = {}
        self.ticker_feature_blocks_: Dict = {}
        self.wass_X_all_: Optional[pd.DataFrame] = None
        self.wass_X_decomposed_: Optional[Dict] = None
        self.local_models_: Optional[Dict] = None
        self.local_states_: Optional[Dict] = None
        self.local_state_probs_: Optional[Dict] = None
        self.meta_hmm_ = None
        self.global_states_: Optional[np.ndarray] = None
        self.global_probs_: Optional[np.ndarray] = None
        self.sync_df_: Optional[pd.DataFrame] = None
        self.global_direct_states_: Optional[np.ndarray] = None
        self.global_direct_probs_: Optional[np.ndarray] = None
        self.global_temporal_df_: Optional[pd.DataFrame] = None
        self.leadlag_df_: Optional[pd.DataFrame] = None
        self.heatmap_df_: Optional[pd.DataFrame] = None
        self.leadlag_pairs_df_: Optional[pd.DataFrame] = None
        self.te_matrix_: Optional[pd.DataFrame] = None
        self.te_k_summary_: Optional[pd.DataFrame] = None
        self.regime_corr_df_: Optional[pd.DataFrame] = None
        self.patient_zero_info_: Optional[Dict] = None
        self.ari_summary_df_: Optional[pd.DataFrame] = None
        self.ari_local_df_: Optional[pd.DataFrame] = None
        self.mmd_local_df_: Optional[pd.DataFrame] = None
        self.mmd_global_df_: Optional[pd.DataFrame] = None
        self.X_meta_: Optional[np.ndarray] = None
        self.leadlag_sig_df_: Optional[pd.DataFrame] = None
        self.leadlag_ticker_metric_df_: Optional[pd.DataFrame] = None
        self.event_results_ = None

    # ------------------------------------------------------------------
    # Step 0: Data loading + feature extraction
    # ------------------------------------------------------------------

    def load_data(self) -> "ContagionPipeline":
        """Load LOBSTER data and parameter files."""
        print("=" * 80)
        print("ETAPE 0 : PREPARATION DES DONNEES")
        print("=" * 80)

        print(f"\n[1/3] Chargement des donnees LOBSTER...")
        self.synced_data_ = load_all_tickers(
            self.tickers, self.analysis_date, self.raw_data_dir
        )
        print(f"OK {len(self.synced_data_[self.tickers[0]])} observations par ticker")

        self._load_params()
        return self

    def extract_features(self) -> "ContagionPipeline":
        """MAD normalization + Wasserstein temporal features."""
        self._check_fitted("synced_data_", "load_data")

        print(f"\n[2/3] Normalisation MAD (fenetre glissante robuste)...")
        innov_dict = normalize_innovations_mad(
            self.synced_data_,
            self.tickers,
            window=self.wasserstein_window,
            min_periods=max(50, self.wasserstein_window // 2),
        )
        print(f"OK Innovations normalisees pour {len(innov_dict)} series")

        print(f"\n[3/3] Calcul des distances de Wasserstein (temporal)...")

        # Build caches for all required (mad_window, wass_window) combos
        per_ticker = self.per_ticker_params_
        use_per_ticker = self.use_per_ticker_

        mad_windows = (
            sorted(per_ticker["mad_window"].unique()) if use_per_ticker else [self.wasserstein_window]
        )
        wass_windows = (
            sorted(per_ticker["wasserstein_window"].unique()) if use_per_ticker else [self.wasserstein_window]
        )

        for mad_w in mad_windows:
            self.innov_cache_[mad_w] = normalize_innovations_mad(
                self.synced_data_,
                self.tickers,
                window=int(mad_w),
                min_periods=max(50, int(mad_w) // 2),
            )

        for mad_w in mad_windows:
            for wass_w in wass_windows:
                wass_X = compute_wasserstein_temporal_features(
                    self.innov_cache_[mad_w],
                    self.tickers,
                    window=int(wass_w),
                )
                self.feature_cache_[(int(mad_w), int(wass_w))] = wass_X

        # Per-ticker feature blocks
        for ticker in self.tickers:
            if use_per_ticker:
                row = per_ticker[per_ticker["ticker"] == ticker].iloc[0]
                mad_w = int(row["mad_window"])
                wass_w = int(row["wasserstein_window"])
            else:
                mad_w = wass_w = self.wasserstein_window
            wass_X = self.feature_cache_[(mad_w, wass_w)]
            cols = [f"{ticker}_Price", f"{ticker}_OFI", f"{ticker}_OBI"]
            cols = [c for c in cols if c in wass_X.columns]
            self.ticker_feature_blocks_[ticker] = wass_X[cols].copy()

        # Align on common index
        common_index = None
        for df in self.ticker_feature_blocks_.values():
            common_index = df.index if common_index is None else common_index.intersection(df.index)
        for t in list(self.ticker_feature_blocks_.keys()):
            self.ticker_feature_blocks_[t] = self.ticker_feature_blocks_[t].loc[common_index]

        self.wass_X_all_ = pd.concat(self.ticker_feature_blocks_.values(), axis=1)
        print(f"Matrice Wasserstein (temporal) : {self.wass_X_all_.shape}")

        output_file = self.results_dir / "hierarchical_temporal_features.csv"
        self.wass_X_all_.to_csv(output_file, index=True)
        print(f"Temporal features sauvegardees dans {output_file}")

        # Build decomposed dict for lead-lag
        self.wass_X_decomposed_ = {m: {t: [] for t in self.tickers} for m in ["price_ret", "obi", "ofi"]}
        for ticker in self.tickers:
            df = self.ticker_feature_blocks_[ticker]
            if f"{ticker}_Price" in df.columns:
                self.wass_X_decomposed_["price_ret"][ticker] = df[f"{ticker}_Price"].values.tolist()
            if f"{ticker}_OBI" in df.columns:
                self.wass_X_decomposed_["obi"][ticker] = df[f"{ticker}_OBI"].values.tolist()
            if f"{ticker}_OFI" in df.columns:
                self.wass_X_decomposed_["ofi"][ticker] = df[f"{ticker}_OFI"].values.tolist()

        return self

    # ------------------------------------------------------------------
    # Step 1: Local HMMs
    # ------------------------------------------------------------------

    def fit_local_hmms(self) -> "ContagionPipeline":
        """Fit one HMM per ticker and extract state probabilities."""
        self._check_fitted("wass_X_all_", "extract_features")

        print("\n" + "=" * 80)
        print("ETAPE 1 : HMM LOCAUX (NIVEAU 1 - PAR ACTIF)")
        print("=" * 80)
        print("Objectif : Extraire P(regime | actif) pour chaque actif\n")

        self.local_models_ = {}
        self.local_states_ = {}
        self.local_state_probs_ = {}
        selected_metrics = ["Price", "OFI", "OBI"]
        print(f"Metriques selectionnees : {selected_metrics}")

        for ticker in self.tickers:
            local_persist, local_smooth, local_regimes = self._ticker_hmm_params(ticker)
            print(f"\n[{ticker}] Fitting HMM local...")

            ticker_cols = [
                f"{ticker}_{m}"
                for m in selected_metrics
                if f"{ticker}_{m}" in self.wass_X_all_.columns
            ]
            if not ticker_cols:
                print(f"  WARN Aucune colonne trouvee pour {ticker}, skip")
                continue

            wass_X_ticker = self.wass_X_all_[ticker_cols]

            # Covariance type: switch to 'full' if high correlation
            covariance_type = "diag"
            if wass_X_ticker.shape[1] > 1:
                corr = wass_X_ticker.corr().abs()
                max_corr = corr.where(~np.eye(corr.shape[0], dtype=bool)).max().max()
                mean_corr = corr.where(~np.eye(corr.shape[0], dtype=bool)).mean().mean()
                if pd.notna(mean_corr):
                    print(f"  -> Mean |corr|={mean_corr:.2f} (max |corr|={max_corr:.2f})")
                if pd.notna(max_corr) and max_corr >= self.hmm_cov_full_corr_threshold:
                    covariance_type = "full"
                    print(
                        f"  -> High corr detected (>= {self.hmm_cov_full_corr_threshold:.2f}), "
                        "using covariance='full'"
                    )

            model, states, state_probs = fit_optimized_hmm_with_probs(
                wass_X_ticker,
                n_components=local_regimes,
                persistence=local_persist,
                smooth_window=local_smooth,
                covariance_type=covariance_type,
            )

            self.local_models_[ticker] = model
            self.local_states_[ticker] = states
            self.local_state_probs_[ticker] = state_probs
            print(f"  OK Probabilites extraites : shape = {state_probs.shape}")

        # Save local states CSV
        states_local_df = pd.DataFrame({"timestamp": self.wass_X_all_.index})
        for ticker in self.tickers:
            if ticker in self.local_states_:
                states_local_df[f"state_{ticker}"] = self.local_states_[ticker]
        output_file = self.results_dir / "hierarchical_states_local.csv"
        states_local_df.to_csv(output_file, index=False)
        print(f"\nOK Etats locaux sauvegardes dans {output_file}")

        return self

    # ------------------------------------------------------------------
    # Step 2: Global HMMs (meta + direct)
    # ------------------------------------------------------------------

    def fit_global_hmms(self) -> "ContagionPipeline":
        """Fit meta-HMM (hierarchical) and direct global HMM."""
        self._check_fitted("local_state_probs_", "fit_local_hmms")

        print("\n" + "=" * 80)
        print("ETAPE 2 : META-HMM GLOBAL (NIVEAU 2 - SECTORIEL)")
        print("=" * 80)
        print("Objectif : Detecter regimes sectoriels a partir des probas locales\n")

        # Meta-HMM
        (
            self.meta_hmm_,
            self.global_states_,
            self.global_probs_,
            self.sync_df_,
        ) = fit_hierarchical_hmm_pipeline(
            local_state_probs=self.local_state_probs_,
            local_states=self.local_states_,
            tickers=self.tickers,
            n_global_regimes=self.n_regimes,
            persistence=float(
                self.best_global_params_.get("HMM_PERSISTENCE_GLOBAL", self.hmm_persistence_global)
            ),
            smooth_window=int(
                self.best_global_params_.get("HMM_SMOOTHING_GLOBAL", self.hmm_smoothing_global)
            ),
        )

        # Diagnostics
        print("\n" + "=" * 80)
        print("DIAGNOSTICS : PROBAS GLOBALES & SYNCHRONISATION")
        print("=" * 80)
        global_prob_max = self.global_probs_.max(axis=1)
        global_prob_entropy = -np.sum(
            self.global_probs_ * np.log(self.global_probs_ + 1e-12), axis=1
        )
        entropy_max = np.log(self.global_probs_.shape[1])
        entropy_ratio = float(np.mean(global_prob_entropy) / entropy_max)
        print(f"Global probs: mean(max P) = {global_prob_max.mean():.3f}")
        print(
            f"Global probs: mean entropy = {global_prob_entropy.mean():.3f} "
            f"(ratio {entropy_ratio:.3f} of max)"
        )
        if entropy_ratio > 0.90:
            print("  -> Warning: probabilities are very flat (weak global signal).")
        elif entropy_ratio > 0.75:
            print("  -> Note: probabilities are quite flat (moderate global signal).")

        sync_mean = float(self.sync_df_["sync_rate"].mean())
        sync_median = float(self.sync_df_["sync_rate"].median())
        print(f"Sync mean = {sync_mean:.3f}, median = {sync_median:.3f}")
        if sync_mean < 0.10:
            print("  -> Warning: low synchronization (global signal likely weak).")

        # Save global states
        states_global_df = pd.DataFrame(
            {"timestamp": self.wass_X_all_.index, "global_state": self.global_states_}
        )
        for i in range(self.global_probs_.shape[1]):
            states_global_df[f"global_prob_regime_{i}"] = self.global_probs_[:, i]
        states_global_df.to_csv(self.results_dir / "hierarchical_states_global.csv", index=False)
        self.sync_df_.to_csv(self.results_dir / "hierarchical_synchronization.csv", index=False)
        print(f"\nOK Etats globaux sauvegardes")

        # Direct global HMM
        print("\n" + "=" * 80)
        print("HMM GLOBAL DIRECT (WASSERSTEIN GLOBAL)")
        print("=" * 80)

        direct_persist = float(
            self.best_direct_params_.get(
                "global_persistence",
                self.best_global_params_.get("HMM_PERSISTENCE_GLOBAL", self.hmm_persistence_global),
            )
        )
        direct_smooth = int(
            self.best_direct_params_.get(
                "global_smoothing",
                self.best_global_params_.get("HMM_SMOOTHING_GLOBAL", self.hmm_smoothing_global),
            )
        )
        _, self.global_direct_states_, self.global_direct_probs_ = fit_optimized_hmm_with_probs(
            self.wass_X_all_,
            n_components=self.n_regimes,
            persistence=direct_persist,
            smooth_window=direct_smooth,
            covariance_type="diag",
        )

        states_global_direct_df = pd.DataFrame(
            {
                "timestamp": self.wass_X_all_.index,
                "global_direct_state": self.global_direct_states_,
            }
        )
        for i in range(self.global_direct_probs_.shape[1]):
            states_global_direct_df[f"global_direct_prob_regime_{i}"] = self.global_direct_probs_[:, i]
        states_global_direct_df.to_csv(
            self.results_dir / "hierarchical_states_global_direct.csv", index=False
        )
        print(f"OK Etats globaux (direct) sauvegardes")

        # Temporal Wasserstein on global stress probability
        if self.global_probs_.shape[1] == 3:
            global_stress = self.global_probs_[:, 1] + self.global_probs_[:, 2]
        else:
            global_stress = self.global_probs_[:, -1]

        global_wass_temporal = _compute_temporal_wasserstein_series(
            global_stress, window=self.wasserstein_window
        )
        w = self.wasserstein_window
        global_temporal_index = self.wass_X_all_.index[w:-w]
        self.global_temporal_df_ = pd.DataFrame(
            {"global_stress_wass_temporal": global_wass_temporal},
            index=global_temporal_index,
        )
        self.global_temporal_df_.to_csv(
            self.results_dir / "hierarchical_global_temporal_wass.csv", index=True
        )
        print(f"OK Global temporal Wasserstein sauvegarde")

        return self

    # ------------------------------------------------------------------
    # Step 3: Lead-lag analyses
    # ------------------------------------------------------------------

    def analyze_leadlag(self) -> "ContagionPipeline":
        """Run both lead-lag analyses (local->global and ticker->ticker)."""
        self._analyze_local_vs_global_leadlag()
        self._analyze_ticker_ticker_leadlag()
        return self

    def _analyze_local_vs_global_leadlag(self) -> None:
        """Lead-lag: local ticker Wasserstein vs global stress (temporal)."""
        self._check_fitted("global_temporal_df_", "fit_global_hmms")

        print("\n" + "=" * 80)
        print("LEAD-LAG LOCAL VS GLOBAL (TEMPORAL WASSERSTEIN)")
        print("=" * 80)

        alpha_threshold = 0.05
        min_obs = 30
        global_series = self.global_temporal_df_["global_stress_wass_temporal"].values
        base_n = min(len(global_series), len(self.global_states_))
        global_series = global_series[:base_n]
        global_states_aligned = self.global_states_[:base_n]

        leadlag_rows = []
        heatmap_rows = []

        for ticker in self.tickers:
            cols = [c for c in self.wass_X_all_.columns if c.startswith(f"{ticker}_")]
            if not cols:
                continue
            local_series = self.wass_X_all_[cols].mean(axis=1).values
            n = min(len(local_series), base_n)
            local_series = local_series[:n]
            global_series_aligned = global_series[:n]
            global_states_local = global_states_aligned[:n]

            for regime in np.unique(global_states_local):
                regime_mask = global_states_local == regime
                if regime_mask.sum() < min_obs:
                    continue
                for q in self.leadlag_quantiles:
                    threshold = np.percentile(local_series[regime_mask], q * 100)
                    q_mask = local_series <= threshold if q < 0.5 else local_series >= threshold
                    q_label = f"Q{int(q * 100)}"
                    mask = regime_mask & q_mask
                    if mask.sum() < min_obs:
                        continue
                    best = _best_leadlag(
                        local_series[mask],
                        global_series_aligned[mask],
                        self.leadlag_max_lag,
                        alpha_threshold,
                        min_obs,
                    )
                    if best is None:
                        continue
                    leadlag_rows.append(
                        {"ticker": ticker, "global_regime": int(regime), "quantile": q_label,
                         **best, "n_obs": int(mask.sum())}
                    )
                    heatmap_rows.append(
                        {"ticker": ticker, "global_regime": int(regime), "quantile": q_label,
                         "best_lag_seconds": best["best_lag_seconds"], "best_corr": best["best_corr"]}
                    )

        self.leadlag_df_ = pd.DataFrame(leadlag_rows).sort_values(
            ["ticker", "global_regime", "quantile", "best_pval"],
            ascending=[True, True, True, True],
        )
        output_file = self.results_dir / "hierarchical_leadlag_local_vs_global_quantile.csv"
        self.leadlag_df_.to_csv(output_file, index=False)
        print(f"Lead-lag local vs global sauvegarde dans {output_file}")

        self.heatmap_df_ = pd.DataFrame(heatmap_rows)
        self._plot_local_global_heatmaps(global_series, global_states_aligned, base_n, alpha_threshold, min_obs)

    def _plot_local_global_heatmaps(
        self, global_series, global_states_aligned, base_n, alpha_threshold, min_obs
    ) -> None:
        """Save local->global lead-lag heatmaps (ranked + paper-quality)."""
        heatmap_df = self.heatmap_df_
        leadlag_df = self.leadlag_df_

        if not heatmap_df.empty:
            max_heatmaps = 4
            heatmap_scores = (
                heatmap_df.groupby(["global_regime", "quantile"])["best_corr"]
                .apply(lambda x: np.nanmax(np.abs(x)) if np.isfinite(x).any() else -np.inf)
                .reset_index(name="score")
                .sort_values(["score"], ascending=False)
            )
            selected_keys = set(
                (int(r["global_regime"]), r["quantile"])
                for _, r in heatmap_scores.head(max_heatmaps).iterrows()
            )
            for regime in sorted(heatmap_df["global_regime"].unique()):
                for q_label in sorted(heatmap_df["quantile"].unique()):
                    if (int(regime), q_label) not in selected_keys:
                        continue
                    sub = heatmap_df[
                        (heatmap_df["global_regime"] == regime) & (heatmap_df["quantile"] == q_label)
                    ]
                    if sub.empty:
                        continue
                    pivot = sub.pivot(index="ticker", columns="quantile", values="best_lag_seconds")
                    plt.figure(figsize=(6, 4))
                    sns.heatmap(pivot, cmap="coolwarm", center=0, annot=True, fmt=".1f")
                    plt.title(f"Lead-lag local->global (Regime {regime}, {q_label})")
                    plt.xlabel("Quantile")
                    plt.ylabel("Ticker")
                    out = self.results_dir / f"hierarchical_leadlag_local_vs_global_R{regime}_{q_label}.png"
                    plt.tight_layout()
                    plt.savefig(out, dpi=300)
                    plt.close()

        # Paper-quality heatmap (all cells filled, manual annotations)
        tickers_list = self.tickers
        q_priority = ["Q90", "Q50", "Q10"]
        regimes_present = sorted(leadlag_df["global_regime"].unique()) if not leadlag_df.empty else []
        mat_corr = np.full((len(tickers_list), len(regimes_present)), np.nan)

        for ti, ticker in enumerate(tickers_list):
            for ri, regime in enumerate(regimes_present):
                for q_try in q_priority:
                    sub = leadlag_df[
                        (leadlag_df["ticker"] == ticker)
                        & (leadlag_df["global_regime"] == regime)
                        & (leadlag_df["quantile"] == q_try)
                    ]
                    if not sub.empty:
                        mat_corr[ti, ri] = sub.iloc[0]["best_corr"]
                        break
                else:
                    cols = [c for c in self.wass_X_all_.columns if c.startswith(f"{ticker}_")]
                    if not cols:
                        continue
                    ls = self.wass_X_all_[cols].mean(axis=1).values
                    n = min(len(ls), base_n)
                    ls = ls[:n]
                    gs = global_series[:n]
                    gsa = global_states_aligned[:n]
                    rmask = gsa == regime
                    if rmask.sum() < 30:
                        continue
                    res = _best_leadlag(ls[rmask], gs[rmask], self.leadlag_max_lag,
                                       alpha_threshold, 30, require_sig=False)
                    if res is not None:
                        mat_corr[ti, ri] = res["best_corr"]

        col_labels = [f"Regime {r}" for r in regimes_present]
        annot_lg = [
            [("" if pd.isna(mat_corr[i, j]) else f"{mat_corr[i, j]:.2f}") for j in range(len(regimes_present))]
            for i in range(len(tickers_list))
        ]
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(
            pd.DataFrame(mat_corr, index=tickers_list, columns=col_labels),
            cmap="coolwarm", center=0, annot=False, ax=ax,
        )
        _heatmap_annotate(ax, mat_corr, annot_lg, fontsize=11)
        ax.set_title("Local to Global Lead-lag Correlation")
        ax.set_ylabel("Ticker")
        ax.set_xlabel("Global Regime")
        plt.tight_layout()
        plt.savefig(PAPER_FIGURES_DIR / "leadlag_local_global_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("Paper local->global heatmap saved.")

    def _analyze_ticker_ticker_leadlag(self) -> None:
        """Lead-lag: pairwise ticker Wasserstein distances."""
        self._check_fitted("wass_X_all_", "extract_features")

        print("\n" + "=" * 80)
        print("LEAD-LAG ENTRE TICKERS (TEMPORAL WASSERSTEIN)")
        print("=" * 80)

        alpha_threshold = 0.05
        min_obs = 30
        tickers_list = self.tickers
        series_by_ticker = {
            t: self.wass_X_all_[
                [c for c in self.wass_X_all_.columns if c.startswith(f"{t}_")]
            ].mean(axis=1).values
            for t in tickers_list
        }

        pair_rows = []
        heatmaps_by_q: Dict = {}
        scores_by_q: Dict = {}

        for q in self.leadlag_quantiles:
            q_label = f"Q{int(q * 100)}"
            heatmap = np.full((len(tickers_list), len(tickers_list)), np.nan, dtype=float)

            for i, t1 in enumerate(tickers_list):
                s1 = series_by_ticker[t1]
                threshold = np.percentile(s1, q * 100)
                mask = s1 <= threshold if q < 0.5 else s1 >= threshold
                s1_sub = s1[mask]

                for j, t2 in enumerate(tickers_list):
                    if t1 == t2:
                        continue
                    s2 = series_by_ticker[t2][mask]
                    if len(s1_sub) < min_obs:
                        continue
                    best = _best_leadlag(s1_sub, s2, self.leadlag_max_lag, alpha_threshold, min_obs)
                    if best is None:
                        continue
                    pair_rows.append(
                        {"quantile": q_label, "ticker1": t1, "ticker2": t2,
                         **best, "n_obs": int(len(s1_sub))}
                    )
                    heatmap[i, j] = best["best_corr"]

            heatmaps_by_q[q_label] = heatmap
            scores_by_q[q_label] = (
                float(np.nanmax(np.abs(heatmap))) if np.isfinite(heatmap).any() else -np.inf
            )

        # Plot top-2 quantile heatmaps
        ranked_q = sorted(scores_by_q.items(), key=lambda x: x[1], reverse=True)
        selected_q = [q for q, _ in ranked_q[:2] if np.isfinite(scores_by_q[q])]
        for q_label in selected_q:
            heatmap = heatmaps_by_q[q_label]
            plt.figure(figsize=(8, 6))
            sns.heatmap(heatmap, cmap="coolwarm", center=0,
                        xticklabels=tickers_list, yticklabels=tickers_list, annot=False)
            plt.title(f"Lead-lag inter-tickers (best corr, {q_label})")
            plt.xlabel("Target ticker")
            plt.ylabel("Source ticker")
            out = self.results_dir / f"hierarchical_leadlag_between_tickers_{q_label}.png"
            plt.tight_layout()
            plt.savefig(out, dpi=300)
            plt.close()
            print(f"Heatmap lead-lag inter-tickers sauvegardee dans {out}")

        self.leadlag_pairs_df_ = pd.DataFrame(pair_rows).sort_values(
            ["quantile", "best_pval", "best_corr"], ascending=[True, True, False]
        )
        out = self.results_dir / "hierarchical_leadlag_between_tickers_quantile.csv"
        self.leadlag_pairs_df_.to_csv(out, index=False)
        print(f"Lead-lag ticker-ticker sauvegarde dans {out}")

        # Paper-quality inter-ticker heatmap
        if not self.leadlag_pairs_df_.empty:
            n = len(tickers_list)
            sym_mat = np.full((n, n), np.nan)
            for _, row in self.leadlag_pairs_df_.iterrows():
                i = tickers_list.index(row["ticker1"])
                j = tickers_list.index(row["ticker2"])
                val = row["best_corr"]
                if pd.isna(sym_mat[i, j]) or abs(val) > abs(sym_mat[i, j]):
                    sym_mat[i, j] = val
                if pd.isna(sym_mat[j, i]) or abs(val) > abs(sym_mat[j, i]):
                    sym_mat[j, i] = val
            annot_strs = [
                [("" if pd.isna(sym_mat[i, j]) else f"{sym_mat[i, j]:.2f}") for j in range(n)]
                for i in range(n)
            ]
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                pd.DataFrame(sym_mat, index=tickers_list, columns=tickers_list),
                cmap="coolwarm", center=0, annot=False, ax=ax,
            )
            _heatmap_annotate(ax, sym_mat, annot_strs, fontsize=11)
            ax.set_title("Inter-ticker Lead-lag Correlation (best across quantiles)")
            ax.set_xlabel("Ticker")
            ax.set_ylabel("Ticker")
            plt.tight_layout()
            plt.savefig(PAPER_FIGURES_DIR / "leadlag_interticker_heatmap.png", dpi=300, bbox_inches="tight")
            plt.close()
            print("Paper inter-ticker heatmap saved.")

    # ------------------------------------------------------------------
    # Step 4: Contagion analysis (TE + regime correlation + Patient Zero)
    # ------------------------------------------------------------------

    def analyze_contagion(self) -> "ContagionPipeline":
        """Transfer Entropy, regime correlation, and Patient Zero identification."""
        self._check_fitted("local_state_probs_", "fit_local_hmms")

        print("\n" + "=" * 80)
        print("ETAPE 3 : TRANSFER ENTROPY (CAUSALITE DIRIGEE)")
        print("=" * 80)

        k_grid = list(range(1, 11))
        self.te_matrix_, self.te_k_summary_ = compute_transfer_entropy_matrix_significance(
            self.local_state_probs_,
            self.tickers,
            k_grid=k_grid,
            bins=10,
            n_surrogates=100,
            block_size=30,
            alpha=0.05,
        )
        self.te_matrix_.to_csv(self.results_dir / "hierarchical_transfer_entropy.csv")
        self.te_k_summary_.to_csv(
            self.results_dir / "hierarchical_transfer_entropy_k_summary.csv", index=False
        )
        print(f"\nOK Matrice TE sauvegardee")

        print("\n" + "=" * 80)
        print("ETAPE 4 : CORRELATION DE REGIMES")
        print("=" * 80)

        self.regime_corr_df_ = compute_regime_correlation(
            self.local_state_probs_, self.tickers, max_lag=10
        )
        self.regime_corr_df_.to_csv(
            self.results_dir / "hierarchical_regime_correlation.csv", index=False
        )
        print(f"\nOK Correlations de regimes sauvegardees")

        print("\n" + "=" * 80)
        print("ETAPE 5 : IDENTIFICATION DU 'PATIENT ZERO'")
        print("=" * 80)

        self.patient_zero_info_ = identify_patient_zero(self.te_matrix_, self.sync_df_)
        pz = self.patient_zero_info_

        out = self.results_dir / "hierarchical_patient_zero.txt"
        with open(out, "w", encoding="utf-8") as f:
            f.write("PATIENT ZERO DE LA CONTAGION\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Actif identifie : {pz['patient_zero']}\n")
            f.write(f"Contagion Score : {pz['contagion_score']:.3f}\n")
            f.write(f"Transfer Entropy sortante : {pz['te_outgoing']:.4f} nats\n")
            f.write(f"Leadership Score : {pz['leadership_score']:.3f}\n\n")
            f.write("RANKING COMPLET :\n")
            f.write(
                pz["ranking"][["ticker", "contagion_score", "te_outgoing", "leadership_score"]].to_string(index=False)
            )
        print(f"\nOK Patient Zero sauvegarde dans {out}")

        return self

    # ------------------------------------------------------------------
    # Step 5: Visualizations
    # ------------------------------------------------------------------

    def plot_visualizations(self) -> "ContagionPipeline":
        """Generate all 6 standard visualizations."""
        self._check_fitted("global_states_", "fit_global_hmms")

        # Viz 1: regime hierarchy
        print("\n" + "=" * 80)
        print("VISUALISATION 1 : HIERARCHIE DES REGIMES")
        print("=" * 80)
        fig = self.meta_hmm_.visualize_regime_hierarchy(
            self.local_states_,
            self.global_states_,
            self.tickers,
            timestamps=self.wass_X_all_.index,
            save_path=self.results_dir / "hierarchical_regime_hierarchy.png",
        )
        plt.close(fig)

        # Viz 2: contagion network
        print("\n" + "=" * 80)
        print("VISUALISATION 2 : RESEAU DE CONTAGION")
        print("=" * 80)
        self._check_fitted("te_matrix_", "analyze_contagion")
        fig = visualize_contagion_network(
            self.te_matrix_,
            self.patient_zero_info_,
            save_path=self.results_dir / "hierarchical_contagion_network.png",
        )
        if fig:
            plt.close(fig)

        # Viz 3: TE heatmap
        print("\n" + "=" * 80)
        print("VISUALISATION 3 : HEATMAP TRANSFER ENTROPY")
        print("=" * 80)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            self.te_matrix_, annot=True, fmt=".4f", cmap="YlOrRd", ax=ax,
            cbar_kws={"label": "Transfer Entropy (nats)"},
        )
        ax.set_title("Matrice de Transfer Entropy (Causalite Dirigee)", fontweight="bold", fontsize=12)
        ax.set_xlabel("Target (Effet)", fontweight="bold")
        ax.set_ylabel("Source (Cause)", fontweight="bold")
        plt.tight_layout()
        out = self.results_dir / "hierarchical_te_heatmap.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"OK Heatmap TE sauvegardee dans {out}")

        # Viz 4: timeline with probabilities
        print("\n" + "=" * 80)
        print("VISUALISATION 4 : TIMELINE DES PROBABILITES")
        print("=" * 80)
        fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
        time_indices = np.arange(len(self.global_states_))

        ax = axes[0]
        for regime in range(self.n_regimes):
            ax.plot(time_indices, self.global_probs_[:, regime],
                    label=f"Regime Global {regime}", alpha=0.7, linewidth=1)
        ax.set_ylabel("P(Regime Global)", fontweight="bold")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_title("Timeline des Probabilites de Regimes (Architecture Hierarchique)",
                     fontweight="bold", fontsize=14)

        ax = axes[1]
        for ticker in self.tickers:
            if ticker in self.local_state_probs_:
                probs = self.local_state_probs_[ticker]
                stress_prob = probs[:, 1] + probs[:, 2] if probs.shape[1] == 3 else probs[:, -1]
                ax.plot(time_indices, stress_prob, label=ticker, alpha=0.7, linewidth=1)
        ax.set_ylabel("P(Stress Local)", fontweight="bold")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        wass_mean = self.wass_X_all_.mean(axis=1).values
        ax.plot(time_indices, wass_mean, linewidth=1, color="purple", alpha=0.7)
        ax.fill_between(time_indices, 0, wass_mean, alpha=0.3, color="purple")
        ax.set_ylabel("Wasserstein\nmoyen", fontweight="bold")
        ax.set_xlabel("Temps (observations)", fontweight="bold")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = self.results_dir / "hierarchical_timeline_probabilities.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"OK Timeline sauvegardee dans {out}")

        # Viz 5: concordance matrices
        print("\n" + "=" * 80)
        print("VISUALISATION 5 : CONCORDANCE LOCAL vs GLOBAL")
        print("=" * 80)
        fig, axes = plt.subplots(1, len(self.tickers), figsize=(20, 4))
        if len(self.tickers) == 1:
            axes = [axes]
        for i, (ticker, ax) in enumerate(zip(self.tickers, axes)):
            if ticker not in self.local_states_:
                continue
            concordance = np.zeros((self.n_regimes, self.n_regimes))
            for global_r in range(self.n_regimes):
                for local_r in range(self.n_regimes):
                    mask = (self.global_states_ == global_r) & (self.local_states_[ticker] == local_r)
                    concordance[global_r, local_r] = mask.sum()
            row_sums = concordance.sum(axis=1, keepdims=True)
            concordance_norm = concordance / (row_sums + 1e-9)
            sns.heatmap(
                concordance_norm, annot=True, fmt=".2f",
                cmap="RdYlGn", vmin=0, vmax=1, ax=ax,
                xticklabels=[f"L{j}" for j in range(self.n_regimes)],
                yticklabels=[f"G{j}" for j in range(self.n_regimes)],
                cbar_kws={"label": "Probabilite"},
            )
            ax.set_title(f"{ticker}", fontweight="bold", fontsize=12)
            ax.set_xlabel("Regime Local", fontweight="bold")
            ax.set_ylabel("Regime Global" if i == 0 else "", fontweight="bold")
        fig.suptitle("Concordance Regime Global -> Regimes Locaux", fontweight="bold", fontsize=14, y=1.02)
        plt.tight_layout()
        out = self.results_dir / "hierarchical_concordance_matrices.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"OK Matrices de concordance sauvegardees dans {out}")

        # Viz 6: lead-lag analysis
        print("\n" + "=" * 80)
        print("VISUALISATION 6 : LEAD-LAG ANALYSIS")
        print("=" * 80)
        self._check_fitted("wass_X_decomposed_", "extract_features")

        self.leadlag_sig_df_ = analyze_multimetric_leadlag_by_model(
            self.wass_X_decomposed_,
            self.tickers,
            max_lag=self.leadlag_max_lag,
            cross_metric_only=True,
            max_pairs_to_plot=6,
        )
        if self.leadlag_sig_df_ is not None and not self.leadlag_sig_df_.empty:
            out = self.results_dir / "leadlag_multimetric_quantile_significant.csv"
            self.leadlag_sig_df_.to_csv(out, index=False)
            print(f"OK Lead-lag significatifs sauvegardes dans {out}")
        else:
            print("WARN Aucun lead-lag significatif trouve pour l'analyse multi-metrique.")

        self.leadlag_ticker_metric_df_ = analyze_interticker_leadlag_by_metric_quantile(
            self.wass_X_decomposed_,
            self.tickers,
            max_lag=self.leadlag_max_lag,
            max_heatmaps_per_metric=1,
        )
        if self.leadlag_ticker_metric_df_ is not None and not self.leadlag_ticker_metric_df_.empty:
            out = self.results_dir / "leadlag_tickers_by_metric_quantile_significant.csv"
            self.leadlag_ticker_metric_df_.to_csv(out, index=False)
            print(f"OK Lead-lag inter-tickers (par metrique/quantile) sauvegarde dans {out}")
        else:
            print("WARN Aucun lead-lag inter-ticker significatif trouve.")

        return self

    # ------------------------------------------------------------------
    # Step 6: Event study + reporting (tables + figures)
    # ------------------------------------------------------------------

    def generate_outputs(self) -> "ContagionPipeline":
        """Event study, LaTeX tables, and all remaining figures."""
        self._run_event_study()
        self._compute_ari_mmd()
        self._save_all_tables()
        self._save_all_figures()
        self._save_leadlag_sig_plots()
        return self

    def _run_event_study(self) -> None:
        """GOOG spike event study."""
        print("\n" + "=" * 80)
        print("EVENT STUDY : GOOG CRASH (21 JUIN 2012)")
        print("=" * 80)
        self.event_results_ = analyze_event_goog_spike(
            self.synced_data_, self.global_states_, self.wasserstein_window
        )
        if self.event_results_ is not None:
            out = self.results_dir / "hierarchical_event_study_goog.csv"
            if isinstance(self.event_results_, dict) and "results_df" in self.event_results_:
                self.event_results_["results_df"].to_csv(out, index=False)
            elif hasattr(self.event_results_, "to_csv"):
                self.event_results_.to_csv(out, index=False)
            else:
                print("WARN Event study format unexpected, skipping CSV export")
            print(f"\nOK Event study sauvegarde dans {out}")

    def _compute_ari_mmd(self) -> None:
        """ARI diagnostics and MMD statistics."""
        # Build X_meta
        prob_arrays = [self.local_state_probs_[t] for t in self.tickers if t in self.local_state_probs_]
        min_len = min(arr.shape[0] for arr in prob_arrays)
        prob_arrays = [arr[:min_len] for arr in prob_arrays]
        self.X_meta_ = np.hstack(prob_arrays)

        meta_kmeans = _kmeans_labels(self.X_meta_, self.n_regimes)
        direct_kmeans = _kmeans_labels(self.wass_X_all_.values, self.n_regimes)

        ari_meta_kmeans = adjusted_rand_score(self.global_states_[:min_len], meta_kmeans)
        ari_direct_kmeans = adjusted_rand_score(
            self.global_direct_states_[: len(direct_kmeans)], direct_kmeans
        )
        n_align = min(len(self.global_states_), len(self.global_direct_states_))
        ari_meta_direct = adjusted_rand_score(
            self.global_states_[:n_align], self.global_direct_states_[:n_align]
        )

        ari_local_rows = []
        for ticker in self.tickers:
            if ticker not in self.local_states_:
                continue
            n = min(len(self.global_states_), len(self.local_states_[ticker]))
            ari_local_rows.append(
                {
                    "ticker": ticker,
                    "ari_meta_vs_local": adjusted_rand_score(
                        self.global_states_[:n], self.local_states_[ticker][:n]
                    ),
                }
            )
        self.ari_local_df_ = pd.DataFrame(ari_local_rows)
        self.ari_summary_df_ = pd.DataFrame(
            [
                {"comparison": "meta_vs_kmeans", "ari": ari_meta_kmeans},
                {"comparison": "direct_vs_kmeans", "ari": ari_direct_kmeans},
                {"comparison": "meta_vs_direct", "ari": ari_meta_direct},
            ]
        )

        # MMD diagnostics
        mmd_rows = []
        for ticker in self.tickers:
            if ticker not in self.local_states_:
                continue
            for metric in ["Price", "OFI", "OBI"]:
                col = f"{ticker}_{metric}"
                if col not in self.wass_X_all_.columns:
                    continue
                series = self.wass_X_all_[col].values
                states = self.local_states_[ticker][: len(series)]
                mmd_df = _compute_mmd_series(series, states, self.mmd_window, self.mmd_step)
                if len(mmd_df) == 0:
                    continue
                mmd_rows.append(
                    {
                        "ticker": ticker,
                        "metric": metric,
                        "mmd_mean": float(mmd_df["mmd_r0_r1"].mean()),
                        "toxicity_mean": float(mmd_df["metric_toxicity"].mean()),
                    }
                )

        global_mmd_rows = []
        for metric in ["Price", "OFI", "OBI"]:
            cols = [c for c in self.wass_X_all_.columns if c.endswith(f"_{metric}")]
            if not cols:
                continue
            series = self.wass_X_all_[cols].mean(axis=1).values
            meta_mmd_df = _compute_mmd_series(series, self.global_states_[: len(series)], self.mmd_window, self.mmd_step)
            direct_mmd_df = _compute_mmd_series(series, self.global_direct_states_[: len(series)], self.mmd_window, self.mmd_step)
            global_mmd_rows.append(
                {
                    "metric": metric,
                    "mmd_meta_mean": float(meta_mmd_df["mmd_r0_r1"].mean()),
                    "mmd_direct_mean": float(direct_mmd_df["mmd_r0_r1"].mean()),
                }
            )

        self.mmd_local_df_ = pd.DataFrame(mmd_rows)
        self.mmd_global_df_ = pd.DataFrame(global_mmd_rows)

    def _save_all_tables(self) -> None:
        """Save all LaTeX tables."""
        print("\n" + "=" * 80)
        print("REPORTING (TABLES LaTeX + FIGURES)")
        print("=" * 80)

        # Regime distribution per ticker
        regime_rows = []
        for ticker in self.tickers:
            if ticker not in self.local_states_:
                continue
            values, counts = np.unique(self.local_states_[ticker], return_counts=True)
            total = counts.sum()
            for v, c in zip(values, counts):
                regime_rows.append({"ticker": ticker, "regime": int(v),
                                    "count": int(c), "pct": float(c / total)})
        _save_latex_table(
            pd.DataFrame(regime_rows),
            PAPER_TABLES_DIR / "local_regime_distribution.tex",
            "Local regime distribution per ticker",
            "tab:local_regime_distribution",
        )

        # Local HMM params
        if self.use_per_ticker_ and self.per_ticker_params_ is not None:
            params_df = self.per_ticker_params_[
                ["ticker", "mad_window", "wasserstein_window",
                 "local_persistence", "local_smoothing", "n_regimes"]
            ].copy()
        else:
            params_df = pd.DataFrame(
                [
                    {
                        "ticker": t,
                        "mad_window": self.wasserstein_window,
                        "wasserstein_window": self.wasserstein_window,
                        "local_persistence": self.hmm_persistence_local,
                        "local_smoothing": self.hmm_smoothing_local,
                        "n_regimes": self.n_regimes,
                    }
                    for t in self.tickers
                ]
            )
        _save_latex_table(
            params_df, PAPER_TABLES_DIR / "local_hmm_params.tex",
            "Local HMM optimized parameters", "tab:local_hmm_params",
        )

        # Synchronization
        _save_latex_table(
            self.sync_df_.rename(columns={"ticker": "Ticker", "sync_rate": "Sync Rate"}),
            PAPER_TABLES_DIR / "local_global_sync.tex",
            "Local-to-global synchronization rate", "tab:local_global_sync",
        )

        # Lead-lag tables
        _save_latex_table(
            self.leadlag_df_, PAPER_TABLES_DIR / "leadlag_local_global_top.tex",
            "Lead-lag local to global (top results)", "tab:leadlag_local_global",
        )
        _save_latex_table(
            self.leadlag_pairs_df_, PAPER_TABLES_DIR / "leadlag_between_tickers_top.tex",
            "Lead-lag between tickers (top results)", "tab:leadlag_between_tickers",
        )

        if self.leadlag_sig_df_ is not None and not self.leadlag_sig_df_.empty:
            _save_latex_table(
                self.leadlag_sig_df_,
                PAPER_TABLES_DIR / "leadlag_multimetric_quantile_significant.tex",
                "Significant lead-lag by quantile (ticker + global)",
                "tab:leadlag_quantile_sig",
            )
        if self.leadlag_ticker_metric_df_ is not None and not self.leadlag_ticker_metric_df_.empty:
            _save_latex_table(
                self.leadlag_ticker_metric_df_,
                PAPER_TABLES_DIR / "leadlag_tickers_metric_quantile_significant.tex",
                "Significant inter-ticker lead-lag by metric and quantile",
                "tab:leadlag_ticker_metric_sig",
            )

        # TE tables
        if self.te_k_summary_ is not None and not self.te_k_summary_.empty:
            _save_latex_table(
                self.te_k_summary_,
                PAPER_TABLES_DIR / "transfer_entropy_k_summary.tex",
                "Transfer Entropy k selection summary",
                "tab:transfer_entropy_k_summary",
            )
        te_long = (
            self.te_matrix_.stack()
            .reset_index()
            .rename(columns={"level_0": "source", "level_1": "target", 0: "te"})
            .sort_values("te", ascending=False)
        )
        _save_latex_table(
            te_long.head(10),
            PAPER_TABLES_DIR / "transfer_entropy_top.tex",
            "Transfer Entropy (top)", "tab:transfer_entropy",
        )

        # ARI + MMD + entropy tables
        _save_latex_table(
            self.ari_summary_df_,
            PAPER_TABLES_DIR / "ari_global_comparisons.tex",
            "ARI diagnostics (global comparisons)", "tab:ari_global",
        )
        _save_latex_table(
            self.ari_local_df_,
            PAPER_TABLES_DIR / "ari_meta_vs_local.tex",
            "ARI meta vs local HMMs", "tab:ari_meta_local",
        )
        _save_latex_table(
            self.mmd_local_df_,
            PAPER_TABLES_DIR / "mmd_local.tex",
            "Local MMD statistics (per ticker/metric)", "tab:mmd_local",
        )
        _save_latex_table(
            self.mmd_global_df_,
            PAPER_TABLES_DIR / "mmd_global.tex",
            "Global MMD (meta vs direct)", "tab:mmd_global",
        )

        meta_entropy = -np.sum(self.global_probs_ * np.log(self.global_probs_ + 1e-12), axis=1)
        direct_entropy = -np.sum(
            self.global_direct_probs_ * np.log(self.global_direct_probs_ + 1e-12), axis=1
        )
        _save_latex_table(
            pd.DataFrame([
                {"model": "meta", "entropy_mean": float(meta_entropy.mean())},
                {"model": "direct", "entropy_mean": float(direct_entropy.mean())},
            ]),
            PAPER_TABLES_DIR / "entropy_global.tex",
            "Mean entropy of global posteriors", "tab:entropy_global",
        )

        n_align = min(len(self.global_states_), len(self.global_direct_states_))
        sync_meta_direct = float(
            np.mean(self.global_states_[:n_align] == self.global_direct_states_[:n_align])
        )
        _save_latex_table(
            pd.DataFrame([{"comparison": "meta_vs_direct_state_sync", "sync_rate": sync_meta_direct}]),
            PAPER_TABLES_DIR / "sync_global_comparison.tex",
            "Synchronization meta vs direct", "tab:sync_global_comparison",
        )

        # Unified robustness table
        robustness_rows = [
            {"Metric": "ARI meta vs K-Means", "Value": f"{self.ari_summary_df_.loc[self.ari_summary_df_['comparison']=='meta_vs_kmeans','ari'].iloc[0]:.4f}"},
            {"Metric": "ARI direct vs K-Means", "Value": f"{self.ari_summary_df_.loc[self.ari_summary_df_['comparison']=='direct_vs_kmeans','ari'].iloc[0]:.4f}"},
            {"Metric": "ARI meta vs direct", "Value": f"{self.ari_summary_df_.loc[self.ari_summary_df_['comparison']=='meta_vs_direct','ari'].iloc[0]:.4f}"},
        ]
        for _, row in self.ari_local_df_.iterrows():
            robustness_rows.append({"Metric": f"ARI meta vs local ({row['ticker']})", "Value": f"{row['ari_meta_vs_local']:.4f}"})
        for _, row in self.mmd_global_df_.iterrows():
            robustness_rows.append({"Metric": f"MMD meta ({row['metric']})", "Value": f"{row['mmd_meta_mean']:.4f}"})
            robustness_rows.append({"Metric": f"MMD direct ({row['metric']})", "Value": f"{row['mmd_direct_mean']:.4f}"})
        robustness_rows.append({"Metric": "Entropy mean (meta)", "Value": f"{float(meta_entropy.mean()):.4f}"})
        robustness_rows.append({"Metric": "Entropy mean (direct)", "Value": f"{float(direct_entropy.mean()):.4f}"})
        robustness_rows.append({"Metric": "State sync (meta vs direct)", "Value": f"{sync_meta_direct:.4f}"})
        _save_latex_table(
            pd.DataFrame(robustness_rows),
            PAPER_TABLES_DIR / "robustness_unified.tex",
            "Unified robustness diagnostics (ARI, MMD, entropy)", "tab:robustness_unified",
        )

        # Significant lead-lag per ticker + HMM (full detail)
        self._save_detailed_leadlag_tables()

    def _save_detailed_leadlag_tables(self) -> None:
        """Compute and save per-ticker, per-regime significant lead-lag tables."""
        alpha = 0.05
        min_obs = 30
        quantiles = [0.1, 0.5, 0.9]
        metric_keys = ["price_ret", "obi", "ofi"]
        metric_names = {"price_ret": "Price", "obi": "OBI", "ofi": "OFI"}

        leadlag_sig_rows = []
        leadlag_sig_store: Dict = {}

        for ticker in self.tickers:
            if ticker not in self.local_states_:
                continue
            for src in metric_keys:
                for tgt in metric_keys:
                    source_series = np.array(self.wass_X_decomposed_[src][ticker])
                    target_series = np.array(self.wass_X_decomposed_[tgt][ticker])

                    for regime in np.unique(self.local_states_[ticker]):
                        mask = self.local_states_[ticker][: len(source_series)] == regime
                        if mask.sum() < min_obs:
                            continue
                        df_sig = _compute_leadlag_significant(
                            source_series[mask], target_series[mask],
                            quantiles, self.leadlag_max_lag, alpha, min_obs,
                        )
                        if not df_sig.empty:
                            df_sig = df_sig.copy()
                            df_sig["ticker"] = ticker
                            df_sig["hmm"] = "local"
                            df_sig["regime"] = int(regime)
                            df_sig["source_metric"] = metric_names[src]
                            df_sig["target_metric"] = metric_names[tgt]
                            leadlag_sig_rows.append(df_sig)
                            key = ("local", ticker, int(regime), metric_names[src], metric_names[tgt])
                            leadlag_sig_store[key] = df_sig

                    for regime in np.unique(self.global_states_):
                        mask = self.global_states_[: len(source_series)] == regime
                        if mask.sum() < min_obs:
                            continue
                        df_sig = _compute_leadlag_significant(
                            source_series[mask], target_series[mask],
                            quantiles, self.leadlag_max_lag, alpha, min_obs,
                        )
                        if not df_sig.empty:
                            df_sig = df_sig.copy()
                            df_sig["ticker"] = ticker
                            df_sig["hmm"] = "meta"
                            df_sig["regime"] = int(regime)
                            df_sig["source_metric"] = metric_names[src]
                            df_sig["target_metric"] = metric_names[tgt]
                            leadlag_sig_rows.append(df_sig)
                            key = ("meta", ticker, int(regime), metric_names[src], metric_names[tgt])
                            leadlag_sig_store[key] = df_sig

        if leadlag_sig_rows:
            full_df = pd.concat(leadlag_sig_rows, ignore_index=True)
            full_df = full_df.sort_values(["p_value", "correlation"], ascending=[True, False])
            for ticker in self.tickers:
                for hmm in ["local", "meta"]:
                    sub = full_df[(full_df["ticker"] == ticker) & (full_df["hmm"] == hmm)]
                    if sub.empty:
                        continue
                    _save_latex_table(
                        sub,
                        PAPER_TABLES_DIR / f"leadlag_significant_{hmm}_{ticker}.tex",
                        f"Significant lead-lag ({hmm}) - {ticker}",
                        f"tab:leadlag_sig_{hmm}_{ticker}",
                    )
            _save_latex_table(
                full_df,
                PAPER_TABLES_DIR / "leadlag_significant_global_top.tex",
                "Significant lead-lag (global)", "tab:leadlag_sig_global",
            )

        self._leadlag_sig_store_ = leadlag_sig_store

    def _save_all_figures(self) -> None:
        """Generate and save all remaining figures (per-ticker + global)."""
        # Local HMM figures
        for ticker in self.tickers:
            if ticker not in self.local_states_:
                continue
            ticker_cols = [c for c in self.wass_X_all_.columns if c.startswith(f"{ticker}_")]
            wass_X_ticker = self.wass_X_all_[ticker_cols]
            _plot_state_timeline(
                self.local_states_[ticker], f"Local HMM Timeline - {ticker}",
                PAPER_FIGURES_DIR / f"hmm_local_{ticker}_timeline.png",
            )
            _plot_state_hist(
                self.local_states_[ticker], f"Local HMM Regime Histogram - {ticker}",
                PAPER_FIGURES_DIR / f"hmm_local_{ticker}_hist.png",
            )
            _plot_feature_by_regime(
                wass_X_ticker, self.local_states_[ticker],
                f"Local HMM Features by Regime - {ticker}",
                PAPER_FIGURES_DIR / f"hmm_local_{ticker}_features.png",
            )
            _plot_posterior_stacked(
                self.local_state_probs_[ticker],
                f"Posterior Regime Probabilities - {ticker}",
                PAPER_FIGURES_DIR / f"hmm_local_{ticker}_posterior.png",
                colors=_LOCAL_REGIME_COLORS.get(ticker),
                regime_labels=_LOCAL_REGIME_LABELS.get(ticker),
            )

        # Global meta and direct figures
        _plot_state_timeline(
            self.global_states_, "Meta-HMM Timeline (Global)",
            PAPER_FIGURES_DIR / "hmm_meta_timeline.png",
        )
        _plot_state_hist(
            self.global_states_, "Meta-HMM Regime Histogram (Global)",
            PAPER_FIGURES_DIR / "hmm_meta_hist.png",
        )
        _plot_feature_by_regime(
            pd.DataFrame(self.X_meta_, columns=[f"p_{i}" for i in range(self.X_meta_.shape[1])]),
            self.global_states_[: len(self.X_meta_)],
            "Meta-HMM Features by Regime (Global)",
            PAPER_FIGURES_DIR / "hmm_meta_features.png",
        )
        _plot_state_timeline(
            self.global_direct_states_, "Direct Global HMM Timeline",
            PAPER_FIGURES_DIR / "hmm_direct_timeline.png",
        )
        _plot_state_hist(
            self.global_direct_states_, "Direct Global HMM Regime Histogram",
            PAPER_FIGURES_DIR / "hmm_direct_hist.png",
        )
        _plot_feature_by_regime(
            self.wass_X_all_, self.global_direct_states_,
            "Direct Global HMM Features by Regime",
            PAPER_FIGURES_DIR / "hmm_direct_features.png",
        )
        _plot_posterior_stacked(
            self.global_probs_, "Meta-HMM Global Posterior Probabilities",
            PAPER_FIGURES_DIR / "hmm_meta_posterior.png",
        )
        _plot_posterior_stacked(
            self.global_direct_probs_, "Direct Global HMM Posterior Probabilities",
            PAPER_FIGURES_DIR / "hmm_direct_posterior.png",
        )

        # Sync + entropy
        plt.figure(figsize=(8, 4))
        sns.barplot(data=self.sync_df_, x="ticker", y="sync_rate")
        plt.title("Synchronisation local -> global")
        plt.tight_layout()
        plt.savefig(PAPER_FIGURES_DIR / "sync_local_global.png", dpi=300)
        plt.close()

        meta_entropy = -np.sum(self.global_probs_ * np.log(self.global_probs_ + 1e-12), axis=1)
        direct_entropy = -np.sum(
            self.global_direct_probs_ * np.log(self.global_direct_probs_ + 1e-12), axis=1
        )
        plt.figure(figsize=(8, 4))
        sns.kdeplot(meta_entropy, label="meta")
        sns.kdeplot(direct_entropy, label="direct")
        plt.title("Distribution de l'entropie (global)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PAPER_FIGURES_DIR / "entropy_global.png", dpi=300)
        plt.close()

        # Temporal regime comparison (per ticker)
        for ticker in self.tickers:
            if ticker not in self.local_states_:
                continue
            n = min(len(self.local_states_[ticker]),
                    len(self.global_states_), len(self.global_direct_states_))
            fig, axes = plt.subplots(3, 1, figsize=(12, 5), sharex=True)
            axes[0].plot(self.local_states_[ticker][:n], linewidth=0.8)
            axes[0].set_ylabel(f"{ticker} local")
            axes[1].plot(self.global_states_[:n], linewidth=0.8, color="tab:orange")
            axes[1].set_ylabel("meta")
            axes[2].plot(self.global_direct_states_[:n], linewidth=0.8, color="tab:green")
            axes[2].set_ylabel("direct")
            axes[2].set_xlabel("Time (obs)")
            fig.suptitle(f"Temporal Regime Comparison - {ticker}", y=0.98)
            plt.tight_layout()
            plt.savefig(PAPER_FIGURES_DIR / f"temporal_regime_comparison_{ticker}.png", dpi=300)
            plt.close()

        # Regime characteristics
        for ticker in self.tickers:
            if ticker not in self.local_states_:
                continue
            ticker_cols = [c for c in self.wass_X_all_.columns if c.startswith(f"{ticker}_")]
            df_local = self.wass_X_all_[ticker_cols].copy()
            _plot_regime_characteristics(
                df_local, self.local_states_[ticker][: len(df_local)],
                f"Microstructure Regime Characteristics (Local) - {ticker}",
                PAPER_FIGURES_DIR / f"regime_characteristics_local_{ticker}.png",
            )
        _plot_regime_characteristics(
            self.wass_X_all_, self.global_direct_states_[: len(self.wass_X_all_)],
            "Microstructure Regime Characteristics (Global Direct)",
            PAPER_FIGURES_DIR / "regime_characteristics_global_direct.png",
        )
        _plot_regime_characteristics(
            pd.DataFrame(self.X_meta_, columns=[f"p_{i}" for i in range(self.X_meta_.shape[1])]),
            self.global_states_[: len(self.X_meta_)],
            "Microstructure Regime Characteristics (Meta-HMM)",
            PAPER_FIGURES_DIR / "regime_characteristics_meta.png",
        )

        # Stress decomposition
        self._stress_decomposition_plot(
            self.global_states_, "Stress Decomposition (Overlay: Meta-HMM)",
            "stress_decomposition_meta.png",
        )
        self._stress_decomposition_plot(
            self.global_direct_states_,
            "Stress Decomposition (Overlay: Direct Global HMM)",
            "stress_decomposition_direct.png",
            log_scale=True,
        )

    def _save_leadlag_sig_plots(self) -> None:
        """Save top-5 significant lead-lag plots per ticker and HMM."""
        if not hasattr(self, "_leadlag_sig_store_"):
            return
        leadlag_sig_store = self._leadlag_sig_store_
        for ticker in self.tickers:
            for hmm in ["local", "meta"]:
                candidates = []
                for key, df_sig in leadlag_sig_store.items():
                    hmm_k, ticker_k, regime_k, src_k, tgt_k = key
                    if hmm_k != hmm or ticker_k != ticker:
                        continue
                    min_p = float(df_sig["p_value"].min())
                    max_abs_corr = float(df_sig["correlation"].abs().max())
                    candidates.append((min_p, -max_abs_corr, key, df_sig))
                if not candidates:
                    continue
                candidates.sort()
                for _, _, key, df_sig in candidates[:5]:
                    hmm_k, ticker_k, regime_k, src_k, tgt_k = key
                    title = f"Lead-lag ({hmm_k} HMM) {ticker_k}: {src_k} -> {tgt_k} (Regime {regime_k})"
                    filename = f"leadlag_{hmm_k}_{ticker_k}_{src_k}_{tgt_k}_R{regime_k}.png"
                    _plot_leadlag_from_df(df_sig, title, PAPER_FIGURES_DIR / filename)

    # ------------------------------------------------------------------
    # Summary + full pipeline runner
    # ------------------------------------------------------------------

    def print_summary(self) -> "ContagionPipeline":
        """Print final pipeline summary to stdout."""
        self._check_fitted("patient_zero_info_", "analyze_contagion")
        pz = self.patient_zero_info_

        print("\n" + "=" * 80)
        print("RESUME FINAL - ARCHITECTURE HIERARCHIQUE")
        print("=" * 80)
        print("\nOK PIPELINE COMPLETE AVEC SUCCES !\n")
        print("RESULTATS CLES :")
        print(f"  1. Patient Zero identifie : {pz['patient_zero']}")
        print(f"     -> Contagion Score = {pz['contagion_score']:.3f}")
        print(f"     -> TE sortante = {pz['te_outgoing']:.4f} nats")
        print(f"  2. Regimes globaux : {self.n_regimes} etats sectoriels detectes")
        te_vals = self.te_matrix_.values
        te_nonzero = te_vals[te_vals > 0]
        print(f"  3. Synchronisation moyenne : {self.sync_df_['sync_rate'].mean():.1%}")
        print(f"  4. TE moyen : {te_nonzero.mean():.4f} nats" if te_nonzero.size else "  4. TE moyen : (no non-zero TE)")
        print("\n" + "=" * 80)
        print("ARCHITECTURE HIERARCHIQUE - VERSION 2.0")
        print("=" * 80)
        return self

    def run(self) -> "ContagionPipeline":
        """Execute the full pipeline end-to-end."""
        print("=" * 80)
        print("PIPELINE HIERARCHIQUE DE DETECTION DE CONTAGION")
        print("=" * 80)
        print(f"\nDate d'analyse: {self.analysis_date}")
        print(f"Tickers: {', '.join(self.tickers)}")
        print(f"Regimes locaux: {self.n_regimes}")
        print(f"Regimes globaux: {self.n_regimes}")
        print(f"\nNOUVEAUTES :")
        print(f"  - Normalisation MAD (robuste) au lieu de GARCH")
        print(f"  - HMM hierarchique (local + global)")
        print(f"  - Transfer Entropy pour causalite dirigee")
        print(f"  - Identification du Patient Zero")

        return (
            self.load_data()
                .extract_features()
                .fit_local_hmms()
                .fit_global_hmms()
                .analyze_leadlag()
                .analyze_contagion()
                .plot_visualizations()
                .generate_outputs()
                .print_summary()
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_params(self) -> None:
        """Load optional per-ticker and global optimized parameters from disk."""
        per_ticker_csv = self.results_dir / "best_parameters_hierarchical_per_ticker.csv"
        self.use_per_ticker_ = per_ticker_csv.exists()
        if self.use_per_ticker_:
            self.per_ticker_params_ = pd.read_csv(per_ticker_csv)
            print(f"Parameters per ticker loaded: {per_ticker_csv}")
        else:
            print("Parametres par ticker non trouves, fallback sur config global")

        global_path = self.results_dir / "best_parameters_hierarchical.txt"
        if global_path.exists():
            with open(global_path, "r", encoding="utf-8") as f:
                for line in f:
                    if "=" in line and not line.strip().startswith("#"):
                        key, val = line.split("=", 1)
                        self.best_global_params_[key.strip()] = val.strip()

        direct_path = self.results_dir / "optimization_global_direct.csv"
        if direct_path.exists():
            try:
                df = pd.read_csv(direct_path)
                if "ari_direct" in df.columns:
                    df = df.sort_values("ari_direct", ascending=False)
                row = df.iloc[0]
                self.best_direct_params_ = {
                    "global_persistence": float(row.get("global_persistence")),
                    "global_smoothing": int(row.get("global_smoothing")),
                }
                print(f"Parameters direct-global loaded: {direct_path}")
            except Exception:
                self.best_direct_params_ = {}

    def _ticker_hmm_params(self, ticker: str):
        """Return (persistence, smoothing, n_regimes) for a ticker."""
        if self.use_per_ticker_ and self.per_ticker_params_ is not None:
            row = self.per_ticker_params_[self.per_ticker_params_["ticker"] == ticker].iloc[0]
            return float(row["local_persistence"]), int(row["local_smoothing"]), int(row["n_regimes"])
        return self.hmm_persistence_local, self.hmm_smoothing_local, self.n_regimes

    def _stress_decomposition_plot(
        self, states: np.ndarray, title: str, filename: str, log_scale: bool = False
    ) -> None:
        """Plot stress decomposition with regime overlays."""
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        metric_names_list = ["Price", "OBI", "OFI"]
        metric_keys_list = ["price_ret", "obi", "ofi"]
        for i, (m_name, m_key) in enumerate(zip(metric_names_list, metric_keys_list)):
            for ticker in self.tickers:
                stress_series = self.wass_X_decomposed_[m_key][ticker]
                axes[i].plot(range(len(stress_series)), stress_series,
                             label=ticker, alpha=0.7, linewidth=1.2)
            for regime in np.unique(states):
                mask = states == regime
                axes[i].fill_between(
                    range(len(states)), 0, axes[i].get_ylim()[1],
                    where=mask, alpha=0.12,
                    label=f"Regime {regime}" if i == 0 else "",
                )
            axes[i].set_ylabel(f"{m_name} Stress", fontweight="bold")
            if log_scale:
                axes[i].set_yscale("log")
            axes[i].grid(alpha=0.3)
        axes[2].set_xlabel("Time (index)")
        axes[0].legend(loc="upper right", ncol=6)
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
        plt.tight_layout()
        plt.savefig(PAPER_FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
        plt.close()

    def _check_fitted(self, attr: str, step: str) -> None:
        """Raise RuntimeError if a required attribute is not yet set."""
        if getattr(self, attr) is None:
            raise RuntimeError(
                f"'{attr}' is None. Call .{step}() before this step."
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ContagionPipeline().run()
