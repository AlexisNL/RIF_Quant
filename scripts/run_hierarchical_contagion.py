from __future__ import annotations

import logging
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
from src.features.mad_normalizer import MADNormalizer
from src.features.wasserstein import WassersteinExtractor
from src.models.hmm_optimal import LocalHMM
from src.models.meta_hmm import fit_hierarchical_hmm_pipeline
from src.analysis.contagion_metrics import ContagionAnalyzer
from src.analysis.leadlag import LeadLagAnalyzer
from src.analysis.event_study import analyze_event_goog_spike

PAPER_DIR = Path("paper")
PAPER_FIGURES_DIR = PAPER_DIR / "figures"
PAPER_TABLES_DIR = PAPER_DIR / "tables"
PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

_GREEN, _BLUE, _RED = "#88cc88", "#7799dd", "#dd6666"
_SEMANTIC = [(_GREEN, "Calm"), (_BLUE, "Intermediate"), (_RED, "Stressed")]


def _compute_regime_order(states: np.ndarray, orig_data: pd.DataFrame) -> list:
    """Return regime ids sorted from calmest to most stressed.

    Uses the composite LOB stress proxy
        norm99(|Δp/p|) + norm99(|OBI|) + norm99(|OFI|)
    to characterise regime *type* (what the market looks like), consistent
    with the paper caption.  norm99 clips each feature to [0, 1] via robust
    min-max scaling at the 1st–99th percentile.

    ``orig_data`` must contain columns ``micro_price``, ``obi``, ``ofi``.
    """
    n = len(states)
    price_ret_abs = orig_data["micro_price"].pct_change().abs().fillna(0).values[:n]
    obi_abs = orig_data["obi"].abs().values[:n]
    ofi_abs = orig_data["ofi"].abs().values[:n]

    def _norm01(x: np.ndarray) -> np.ndarray:
        p1, p99 = np.percentile(x, 1), np.percentile(x, 99)
        return np.clip((x - p1) / (p99 - p1 + 1e-12), 0.0, 1.0)

    stress_proxy = _norm01(price_ret_abs) + _norm01(obi_abs) + _norm01(ofi_abs)
    unique = np.unique(states)
    regime_stress = {r: float(np.mean(stress_proxy[states == r])) for r in unique}
    return sorted(unique.tolist(), key=lambda r: regime_stress[r])


def _states_to_onehot(states: np.ndarray, n_regimes: int) -> np.ndarray:
    """Convert the Viterbi state sequence to one-hot encoding (0/1 per regime).

    This keeps posterior-style plots consistent with ARI, MMD, and regime-share
    summaries, which are all computed from Viterbi states.
    """
    onehot = np.zeros((len(states), n_regimes))
    onehot[np.arange(len(states)), states] = 1.0
    return onehot


def _build_regime_maps(order: list) -> tuple:
    """Build (colors, labels) arrays indexed by regime id from a calmest-first order."""
    n = len(order)
    colors = [None] * n
    labels = [None] * n
    for sem_idx, regime_id in enumerate(order):
        col, name = _SEMANTIC[sem_idx]
        colors[regime_id] = col
        labels[regime_id] = f"Regime {regime_id} ({name})"
    return colors, labels


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
    plot_order = list(range(n_regimes))
    if regime_labels:
        semantic_rank = {"calm": 0, "intermediate": 1, "stressed": 2}
        ranked = []
        for idx, lbl in enumerate(use_labels):
            low = str(lbl).lower()
            rank = next((v for k, v in semantic_rank.items() if k in low), 99)
            ranked.append((rank, idx))
        plot_order = [idx for _, idx in sorted(ranked)]
    fig, ax = plt.subplots(figsize=(14, 3.5))
    x = np.arange(probs.shape[0])
    ax.stackplot(
        x,
        *[probs[:, k] for k in plot_order],
        labels=[use_labels[k] for k in plot_order],
        colors=[use_colors[k] for k in plot_order],
        alpha=0.85,
    )
    ax.set_xlim(0, max(len(x) - 1, 1))
    ax.margins(x=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(regime)")
    ax.set_xlabel("Time index")
    ax.set_title(title)
    ax.legend(
        loc="upper right",
        ncol=n_regimes,
        fontsize=8,
        frameon=True,
    )
    fig.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


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


def _leadlag_corr_pval(
    series_local: np.ndarray, series_global: np.ndarray, max_lag: int
):
    """Compute lagged Spearman correlations and p-values in a single pass."""
    lags_list = list(range(-max_lag, max_lag + 1))
    corrs = []
    pvals = []
    for lag in lags_list:
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
            pvals.append(np.nan)
        else:
            res = stats.spearmanr(x, y, nan_policy="omit")
            corrs.append(res.correlation)
            pvals.append(res.pvalue)
    return np.array(lags_list), np.array(corrs), np.array(pvals)


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
    lags, corrs, pvals = _leadlag_corr_pval(series_a, series_b, max_lag)
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
        tickers: List[str] = TICKERS,
        analysis_date: str = ANALYSIS_DATE,
        raw_data_dir: Path = RAW_DATA_DIR,
        results_dir: Path = RESULTS_DIR,
        n_regimes: int = N_REGIMES,
        hmm_persistence_local: float = HMM_PERSISTENCE_LOCAL,
        hmm_smoothing_local: int = HMM_SMOOTHING_LOCAL,
        hmm_persistence_global: float = HMM_PERSISTENCE_GLOBAL,
        hmm_smoothing_global: int = HMM_SMOOTHING_GLOBAL,
        hmm_cov_full_corr_threshold: float = HMM_COV_FULL_CORR_THRESHOLD,
        mmd_window: int = MMD_WINDOW,
        mmd_step: int = MMD_STEP,
        leadlag_max_lag: int = LEADLAG_MAX_LAG,
        leadlag_quantiles: list = LEADLAG_QUANTILES,
        wasserstein_window: int = WASSERSTEIN_WINDOW,
    ) -> None:
        self.tickers = list(tickers)
        self.analysis_date = analysis_date
        self.raw_data_dir = raw_data_dir
        self.results_dir = results_dir
        self.n_regimes = n_regimes
        self.hmm_persistence_local = hmm_persistence_local
        self.hmm_smoothing_local = hmm_smoothing_local
        self.hmm_persistence_global = hmm_persistence_global
        self.hmm_smoothing_global = hmm_smoothing_global
        self.hmm_cov_full_corr_threshold = hmm_cov_full_corr_threshold
        self.mmd_window = mmd_window
        self.mmd_step = mmd_step
        self.leadlag_max_lag = leadlag_max_lag
        self.leadlag_quantiles = list(leadlag_quantiles)
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
        self.regime_stats_df_: Optional[pd.DataFrame] = None


    def load_data(self) -> "ContagionPipeline":
        """Load LOBSTER data and parameter files."""
        logger.info("--- step 0: data preparation ---")

        logger.info("[1/3] loading LOBSTER data...")
        self.synced_data_ = load_all_tickers(
            self.tickers, self.analysis_date, self.raw_data_dir
        )
        logger.info("%d observations per ticker", len(self.synced_data_[self.tickers[0]]))

        self._load_params()
        return self

    def extract_features(self) -> "ContagionPipeline":
        """MAD normalization + Wasserstein temporal features."""
        self._check_fitted("synced_data_", "load_data")

        logger.info("[2/3] MAD normalisation (robust sliding window)...")
        innov_dict = MADNormalizer(
            window=self.wasserstein_window,
            min_periods=max(50, self.wasserstein_window // 2),
        ).fit_transform(self.synced_data_, self.tickers)
        logger.info("%d normalised series", len(innov_dict))

        logger.info("[3/3] computing temporal Wasserstein distances...")

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
            self.innov_cache_[mad_w] = MADNormalizer(
                window=int(mad_w),
                min_periods=max(50, int(mad_w) // 2),
            ).fit_transform(self.synced_data_, self.tickers)

        for mad_w in mad_windows:
            for wass_w in wass_windows:
                wass_X = WassersteinExtractor(window=int(wass_w)).compute_features(
                    self.innov_cache_[mad_w], self.tickers
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
        logger.info("Wasserstein feature matrix: %s", self.wass_X_all_.shape)

        output_file = self.results_dir / "hierarchical_temporal_features.csv"
        self.wass_X_all_.to_csv(output_file, index=True)
        logger.info("temporal features saved to %s", output_file)

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


    def fit_local_hmms(self) -> "ContagionPipeline":
        """Fit one HMM per ticker and extract state probabilities."""
        self._check_fitted("wass_X_all_", "extract_features")

        logger.info("--- step 1: local HMMs (level 1 — per ticker) ---")

        self.local_models_ = {}
        self.local_states_ = {}
        self.local_state_probs_ = {}
        selected_metrics = ["Price", "OFI", "OBI"]
        logger.info("metrics: %s", selected_metrics)

        for ticker in self.tickers:
            local_persist, local_smooth, local_regimes = self._ticker_hmm_params(ticker)
            logger.info("[%s] fitting local HMM...", ticker)

            ticker_cols = [
                f"{ticker}_{m}"
                for m in selected_metrics
                if f"{ticker}_{m}" in self.wass_X_all_.columns
            ]
            if not ticker_cols:
                logger.warning("no columns found for %s, skipping", ticker)
                continue

            wass_X_ticker = self.wass_X_all_[ticker_cols]

            # Covariance type: switch to 'full' if high correlation
            covariance_type = "diag"
            if wass_X_ticker.shape[1] > 1:
                corr = wass_X_ticker.corr().abs()
                max_corr = corr.where(~np.eye(corr.shape[0], dtype=bool)).max().max()
                mean_corr = corr.where(~np.eye(corr.shape[0], dtype=bool)).mean().mean()
                if pd.notna(mean_corr):
                    logger.debug("  mean |corr|=%.2f (max |corr|=%.2f)", mean_corr, max_corr)
                if pd.notna(max_corr) and max_corr >= self.hmm_cov_full_corr_threshold:
                    covariance_type = "full"
                    logger.info(
                        "  high correlation detected (>= %.2f), using covariance=full",
                        self.hmm_cov_full_corr_threshold,
                    )

            _hmm = LocalHMM(
                n_regimes=local_regimes,
                persistence=local_persist,
                smooth_window=local_smooth,
                covariance_type=covariance_type,
            ).fit(wass_X_ticker)

            self.local_models_[ticker] = _hmm.model_
            self.local_states_[ticker] = _hmm.states_
            self.local_state_probs_[ticker] = _hmm.probs_
            logger.info("  posterior probabilities extracted: shape=%s", _hmm.probs_.shape)

        # Save local states CSV
        states_local_df = pd.DataFrame({"timestamp": self.wass_X_all_.index})
        for ticker in self.tickers:
            if ticker in self.local_states_:
                states_local_df[f"state_{ticker}"] = self.local_states_[ticker]
        output_file = self.results_dir / "hierarchical_states_local.csv"
        states_local_df.to_csv(output_file, index=False)
        logger.info("local states saved to %s", output_file)

        # Save regime descriptive stats (original microstructure features per regime)
        self._save_regime_stats()

        return self

    def _save_regime_stats(self) -> None:
        """Compute and save descriptive stats of original LOB features per regime per ticker.

        For each (ticker, regime) pair computes: observation count, share of day,
        price return volatility, absolute OBI mean, absolute OFI mean, return
        kurtosis, and a human-readable stress label derived from
        ``_compute_regime_order``.  Results are written to
        ``hierarchical_regime_stats.csv``.
        """
        rows = []

        for ticker in self.tickers:
            if ticker not in self.local_states_:
                continue

            orig_aligned, states_aligned = self._get_aligned_local_data_and_states(ticker)
            n_total = len(states_aligned)
            if n_total == 0:
                continue

            # Determine semantic labels via return-volatility stress proxy (original features)
            ticker_cols = [c for c in self.wass_X_all_.columns if c.startswith(f"{ticker}_")]
            wass_df = self.wass_X_all_[ticker_cols].loc[states_aligned.index]
            order = _compute_regime_order(states_aligned.values, orig_aligned)
            label_map = {}
            semantic = ["Calm", "Intermediate", "Stressed"]
            for sem_idx, regime_id in enumerate(order):
                label_map[regime_id] = semantic[sem_idx]

            # Identify individual Wasserstein columns by metric name
            col_ret = next((c for c in ticker_cols if "price_ret" in c), None)
            col_obi = next((c for c in ticker_cols if "_obi" in c), None)
            col_ofi = next((c for c in ticker_cols if "_ofi" in c), None)

            price_ret = orig_aligned["price_ret"]

            for regime in sorted(states_aligned.unique()):
                mask = states_aligned == regime
                sub_ret = price_ret[mask].dropna()
                sub_obi = orig_aligned.loc[mask, "obi"]
                sub_ofi = orig_aligned.loc[mask, "ofi"]
                n = int(mask.sum())
                rows.append({
                    "ticker": ticker,
                    "regime": int(regime),
                    "label": label_map[regime],
                    "n": n,
                    "pct": round(n / n_total * 100, 1),
                    "wass_ret_mean": round(float(wass_df.loc[mask, col_ret].mean()), 4) if col_ret else None,
                    "wass_obi_mean": round(float(wass_df.loc[mask, col_obi].mean()), 4) if col_obi else None,
                    "wass_ofi_mean": round(float(wass_df.loc[mask, col_ofi].mean()), 4) if col_ofi else None,
                    "ret_std_bps": round(float(sub_ret.std()) * 1e4, 4),
                    "ret_abs_mean_bps": round(float(sub_ret.abs().mean()) * 1e4, 4),
                    "ret_kurt": round(float(sub_ret.kurt()), 1),
                    "obi_mean": round(float(sub_obi.mean()), 4),
                    "obi_abs_mean": round(float(sub_obi.abs().mean()), 4),
                    "ofi_mean": round(float(sub_ofi.mean()), 1),
                    "ofi_abs_mean": round(float(sub_ofi.abs().mean()), 1),
                })

        stats_df = pd.DataFrame(rows)
        self.regime_stats_df_ = stats_df
        out = self.results_dir / "hierarchical_regime_stats.csv"
        stats_df.to_csv(out, index=False)
        logger.info("regime stats saved to %s", out)

    def _get_aligned_local_data_and_states(self, ticker: str) -> tuple[pd.DataFrame, pd.Series]:
        """Align original ticker data and local states on their common time index.

        Mirrors the alignment logic from _tmp_regime_stats.py.
        """
        df = self.synced_data_[ticker].copy()
        df["price_ret"] = df["micro_price"].pct_change()

        states = self.local_states_[ticker]
        state_index = self.wass_X_all_.index[: len(states)]
        states_series = pd.Series(states, index=state_index, name=f"state_{ticker}")

        common_idx = df.index.intersection(states_series.index)
        df_aligned = df.loc[common_idx]
        states_aligned = states_series.loc[common_idx]
        return df_aligned, states_aligned


    def fit_global_hmms(self) -> "ContagionPipeline":
        """Fit meta-HMM (hierarchical) and direct global HMM."""
        self._check_fitted("local_state_probs_", "fit_local_hmms")

        logger.info("--- step 2: global meta-HMM (level 2 — sector) ---")

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
        global_prob_max = self.global_probs_.max(axis=1)
        global_prob_entropy = -np.sum(
            self.global_probs_ * np.log(self.global_probs_ + 1e-12), axis=1
        )
        entropy_max = np.log(self.global_probs_.shape[1])
        entropy_ratio = float(np.mean(global_prob_entropy) / entropy_max)
        logger.info(
            "global probs: mean(max P)=%.3f, mean entropy=%.3f (ratio %.3f of max)",
            global_prob_max.mean(), global_prob_entropy.mean(), entropy_ratio,
        )
        if entropy_ratio > 0.90:
            logger.warning("very flat probabilities (weak global signal)")
        elif entropy_ratio > 0.75:
            logger.info("fairly flat probabilities (moderate global signal)")

        sync_mean = float(self.sync_df_["sync_rate"].mean())
        sync_median = float(self.sync_df_["sync_rate"].median())
        logger.info("sync rate: mean=%.3f, median=%.3f", sync_mean, sync_median)
        if sync_mean < 0.10:
            logger.warning("low synchronisation (global signal likely weak)")

        # Save global states
        states_global_df = pd.DataFrame(
            {"timestamp": self.wass_X_all_.index, "global_state": self.global_states_}
        )
        for i in range(self.global_probs_.shape[1]):
            states_global_df[f"global_prob_regime_{i}"] = self.global_probs_[:, i]
        states_global_df.to_csv(self.results_dir / "hierarchical_states_global.csv", index=False)
        self.sync_df_.to_csv(self.results_dir / "hierarchical_synchronization.csv", index=False)
        logger.info("global states saved")

        # Direct global HMM
        logger.info("--- direct global HMM (Wasserstein global) ---")

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
        _hmm_direct = LocalHMM(
            n_regimes=self.n_regimes,
            persistence=direct_persist,
            smooth_window=direct_smooth,
            covariance_type="diag",
        ).fit(self.wass_X_all_)
        self.global_direct_states_ = _hmm_direct.states_
        self.global_direct_probs_ = _hmm_direct.probs_

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
        logger.info("direct global states saved")

        # Temporal Wasserstein on global stress probability
        if self.global_probs_.shape[1] == 3:
            global_stress = self.global_probs_[:, 1] + self.global_probs_[:, 2]
        else:
            global_stress = self.global_probs_[:, -1]

        global_wass_temporal = WassersteinExtractor(
            window=self.wasserstein_window
        ).compute_temporal_series(global_stress)
        w = self.wasserstein_window
        global_temporal_index = self.wass_X_all_.index[w:-w]
        self.global_temporal_df_ = pd.DataFrame(
            {"global_stress_wass_temporal": global_wass_temporal},
            index=global_temporal_index,
        )
        self.global_temporal_df_.to_csv(
            self.results_dir / "hierarchical_global_temporal_wass.csv", index=True
        )
        logger.info("global temporal Wasserstein saved")

        return self


    def analyze_leadlag(self) -> "ContagionPipeline":
        """Run both lead-lag analyses (local->global and ticker->ticker)."""
        self._analyze_local_vs_global_leadlag()
        self._analyze_ticker_ticker_leadlag()
        return self

    def _analyze_local_vs_global_leadlag(self) -> None:
        """Lead-lag: local ticker Wasserstein vs global stress (temporal)."""
        self._check_fitted("global_temporal_df_", "fit_global_hmms")

        logger.info("--- lead-lag local vs global (temporal Wasserstein) ---")

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
        logger.info("lead-lag local vs global saved to %s", output_file)

        self.heatmap_df_ = pd.DataFrame(heatmap_rows)
        self._plot_local_global_heatmaps(global_series, global_states_aligned, base_n, alpha_threshold, min_obs)

    def _plot_local_global_heatmaps(
        self, global_series, global_states_aligned, base_n, alpha_threshold, min_obs
    ) -> None:
        """Save paper-quality local->global lead-lag heatmap."""
        leadlag_df = self.leadlag_df_

        tickers_list = self.tickers
        q_priority = ["Q90", "Q50", "Q10"]
        regimes_present = sorted(leadlag_df["global_regime"].unique()) if not leadlag_df.empty else []
        mat_corr = np.full((len(tickers_list), len(regimes_present)), np.nan)
        mat_lag  = np.full((len(tickers_list), len(regimes_present)), np.nan)
        mat_q    = np.full((len(tickers_list), len(regimes_present)), "", dtype=object)

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
                        mat_lag[ti, ri]  = sub.iloc[0]["best_lag_seconds"]
                        mat_q[ti, ri]    = q_try
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
                        mat_lag[ti, ri]  = res["best_lag_seconds"]

        col_labels = [f"Regime {r}" for r in regimes_present]
        nr, nc = len(tickers_list), len(regimes_present)
        vmax = max(abs(np.nanmin(mat_corr)), abs(np.nanmax(mat_corr)), 1e-9)

        fig, ax = plt.subplots(figsize=(max(8, nc * 2.5), max(5, nr * 1.2)))
        im = ax.imshow(mat_corr, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(nc))
        ax.set_yticks(range(nr))
        ax.set_xticklabels(col_labels, fontsize=11)
        ax.set_yticklabels(tickers_list, fontsize=11)
        for i in range(nr):
            for j in range(nc):
                corr = mat_corr[i, j]
                lag  = mat_lag[i, j]
                if not np.isfinite(corr):
                    continue
                text_color = "white" if abs(corr) / vmax > 0.6 else "black"
                q_label_cell = mat_q[i, j]
                q_line = f"\n{q_label_cell}" if q_label_cell else ""
                ax.text(
                    j, i,
                    f"{corr:.2f}\n({lag:.0f}s){q_line}",
                    ha="center", va="center",
                    fontsize=10, color=text_color, fontweight="bold",
                    linespacing=1.4,
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman r")
        ax.set_title(
            "Local -> Global lead-lag (best quantile available)\n(correlation / optimal lag)",
            fontweight="bold",
        )
        ax.set_ylabel("Ticker")
        ax.set_xlabel("Global regime")
        plt.tight_layout()
        plt.savefig(PAPER_FIGURES_DIR / "leadlag_local_global_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("local->global lead-lag heatmap saved")

    def _analyze_ticker_ticker_leadlag(self) -> None:
        """Lead-lag: pairwise ticker Wasserstein distances."""
        self._check_fitted("wass_X_all_", "extract_features")

        logger.info("--- inter-ticker lead-lag (temporal Wasserstein) ---")

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
        heatmaps_by_q:     Dict = {}
        lag_heatmaps_by_q: Dict = {}

        for q in self.leadlag_quantiles:
            q_label = f"Q{int(q * 100)}"
            heatmap     = np.full((len(tickers_list), len(tickers_list)), np.nan, dtype=float)
            lag_heatmap = np.full((len(tickers_list), len(tickers_list)), np.nan, dtype=float)

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
                    # Heatmap: always show best corr (no significance filter)
                    best_any = _best_leadlag(
                        s1_sub, s2, self.leadlag_max_lag, alpha_threshold, min_obs,
                        require_sig=False,
                    )
                    if best_any is None:
                        continue
                    heatmap[i, j]     = best_any["best_corr"]
                    lag_heatmap[i, j] = best_any["best_lag_seconds"]
                    # CSV: significant pairs only
                    if best_any["best_pval"] < alpha_threshold:
                        pair_rows.append(
                            {"quantile": q_label, "ticker1": t1, "ticker2": t2,
                             **best_any, "n_obs": int(len(s1_sub))}
                        )

            heatmaps_by_q[q_label]     = heatmap
            lag_heatmaps_by_q[q_label] = lag_heatmap

        self.leadlag_pairs_df_ = pd.DataFrame(pair_rows).sort_values(
            ["quantile", "best_pval", "best_corr"], ascending=[True, True, False]
        )
        out = self.results_dir / "hierarchical_leadlag_between_tickers_quantile.csv"
        self.leadlag_pairs_df_.to_csv(out, index=False)
        logger.info("ticker-ticker lead-lag saved to %s", out)

        # Paper-quality inter-ticker heatmap: Q10 vs Q90 only (shared colorbar)
        calm_q  = "Q10" if "Q10" in heatmaps_by_q else None
        stress_q = "Q90" if "Q90" in heatmaps_by_q else None
        plot_pairs = [
            (ql, label)
            for ql, label in [(calm_q, "Q10"), (stress_q, "Q90")]
            if ql is not None
        ]
        if plot_pairs:
            n = len(tickers_list)
            n_plots = len(plot_pairs)
            vmin, vmax = -0.5, 0.5

            # Reserve the rightmost column for the colorbar (width_ratios)
            width_ratios = [5] * n_plots + [0.3]
            fig, all_axes = plt.subplots(
                1, n_plots + 1,
                figsize=(5.5 * n_plots + 1.2, 5.2),
                gridspec_kw={"width_ratios": width_ratios},
            )
            axes = list(all_axes[:n_plots])
            cbar_ax = all_axes[-1]

            cmap = plt.cm.coolwarm.copy()
            cmap.set_bad(color="#d4d4d4")   # NaN cells → light grey

            im = None
            for ax, (q_label, title) in zip(axes, plot_pairs):
                hm     = heatmaps_by_q[q_label]
                lag_hm = lag_heatmaps_by_q[q_label]

                # Mask NaN so imshow renders them in the bad-color
                hm_masked = np.ma.masked_invalid(hm)
                im = ax.imshow(hm_masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")

                # Axis labels — set major ticks explicitly before minor
                ax.set_xticks(range(n))
                ax.set_xticklabels(tickers_list, fontsize=10)
                ax.set_yticks(range(n))
                ax.set_yticklabels(tickers_list, fontsize=10)
                ax.set_title(title, fontweight="bold", fontsize=12, pad=6)
                ax.set_ylabel("Source" if ax is axes[0] else "", fontsize=10)

                # Cell annotations: corr (bold) + lag seconds below
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            ax.text(j, i, "─", ha="center", va="center",
                                    fontsize=11, color="black")
                            continue
                        corr = hm[i, j]
                        lag  = lag_hm[i, j]
                        if np.isfinite(corr):
                            text_color = "white" if abs(corr) >= 0.3 else "black"
                            lag_str = f"({lag:+.0f}s)" if np.isfinite(lag) else ""
                            ax.text(
                                j, i,
                                f"{corr:.2f}\n{lag_str}",
                                ha="center", va="center",
                                fontsize=8, color=text_color, fontweight="bold",
                                linespacing=1.5,
                            )
                        else:
                            ax.text(j, i, "ns", ha="center", va="center",
                                    fontsize=8, color="#888888")

                # Minor grid lines (cell borders)
                ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
                ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
                ax.tick_params(which="minor", bottom=False, left=False)

            fig.colorbar(im, cax=cbar_ax, label="Spearman ρ")
            plt.suptitle(
                "Inter-ticker lead-lag: best correlation by stress quantile",
                fontsize=13, fontweight="bold", y=1.02,
            )
            plt.tight_layout()
            plt.savefig(
                PAPER_FIGURES_DIR / "leadlag_interticker_heatmap.png",
                dpi=300, bbox_inches="tight",
            )
            plt.close()
            logger.info("inter-ticker lead-lag heatmap saved")


    def analyze_contagion(self) -> "ContagionPipeline":
        """Transfer Entropy, regime correlation, and Patient Zero identification."""
        self._check_fitted("local_state_probs_", "fit_local_hmms")

        logger.info("--- step 3: transfer entropy (directed causality) ---")

        k_grid = list(range(1, 11))
        _ca = ContagionAnalyzer(n_bins=10, n_surrogates=100, block_size=30, alpha=0.05)
        self.te_matrix_, self.te_k_summary_ = _ca.compute_te_matrix_significance(
            self.local_state_probs_, self.tickers, k_grid=k_grid
        )
        self.te_matrix_.to_csv(self.results_dir / "hierarchical_transfer_entropy.csv")
        self.te_k_summary_.to_csv(
            self.results_dir / "hierarchical_transfer_entropy_k_summary.csv", index=False
        )
        logger.info("TE matrix saved")

        logger.info("--- step 4: regime correlation ---")

        self.regime_corr_df_ = ContagionAnalyzer().compute_regime_correlation(
            self.local_state_probs_, self.tickers, max_lag=10
        )
        self.regime_corr_df_.to_csv(
            self.results_dir / "hierarchical_regime_correlation.csv", index=False
        )
        logger.info("regime correlations saved")

        logger.info("--- step 5: patient zero identification ---")

        self.patient_zero_info_ = ContagionAnalyzer().identify_patient_zero(
            sync_df=self.sync_df_, te_matrix=self.te_matrix_
        )
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
        logger.info("patient zero saved to %s", out)

        return self


    def plot_visualizations(self) -> "ContagionPipeline":
        """Generate all 6 standard visualizations."""
        self._check_fitted("global_states_", "fit_global_hmms")

        # Viz 1: regime hierarchy
        logger.info("--- viz 1: regime hierarchy ---")
        fig = self.meta_hmm_.visualize_regime_hierarchy(
            self.local_states_,
            self.global_states_,
            self.tickers,
            timestamps=self.wass_X_all_.index,
            save_path=self.results_dir / "hierarchical_regime_hierarchy.png",
        )
        plt.close(fig)

        # Viz 2: contagion network
        logger.info("--- viz 2: contagion network ---")
        self._check_fitted("te_matrix_", "analyze_contagion")
        fig = ContagionAnalyzer().plot_network(
            te_matrix=self.te_matrix_,
            patient_zero_info=self.patient_zero_info_,
            save_path=self.results_dir / "hierarchical_contagion_network.png",
        )
        if fig:
            plt.close(fig)

        # Viz 3: TE heatmap
        logger.info("--- viz 3: TE heatmap ---")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            self.te_matrix_, annot=True, fmt=".4f", cmap="YlOrRd", ax=ax,
            cbar_kws={"label": "Transfer Entropy (nats)"},
        )
        ax.set_title("Matrice de Transfer Entropy (Causalite Dirigee)", fontweight="bold", fontsize=12)
        ax.set_xlabel("Target (Effect)", fontweight="bold")
        ax.set_ylabel("Source (Cause)", fontweight="bold")
        plt.tight_layout()
        out = self.results_dir / "hierarchical_te_heatmap.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("TE heatmap saved to %s", out)

        # Viz 4: timeline with probabilities
        logger.info("--- viz 4: probability timeline ---")
        fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
        time_indices = np.arange(len(self.global_states_))

        ax = axes[0]
        for regime in range(self.n_regimes):
            ax.plot(time_indices, self.global_probs_[:, regime],
                    label=f"Regime Global {regime}", alpha=0.7, linewidth=1)
        ax.set_ylabel("P(Regime Global)", fontweight="bold")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_title("Regime Probability Timeline (Hierarchical Architecture)",
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
        ax.set_ylabel("Mean\nWasserstein", fontweight="bold")
        ax.set_xlabel("Time (observations)", fontweight="bold")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = self.results_dir / "hierarchical_timeline_probabilities.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("probability timeline saved to %s", out)

        # Viz 5: concordance matrices
        logger.info("--- viz 5: local vs global concordance ---")
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
                cbar_kws={"label": "Probability"},
            )
            ax.set_title(f"{ticker}", fontweight="bold", fontsize=12)
            ax.set_xlabel("Regime Local", fontweight="bold")
            ax.set_ylabel("Regime Global" if i == 0 else "", fontweight="bold")
        fig.suptitle("Concordance Regime Global -> Regimes Locaux", fontweight="bold", fontsize=14, y=1.02)
        plt.tight_layout()
        out = self.results_dir / "hierarchical_concordance_matrices.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("concordance matrices saved to %s", out)

        # Viz 6: lead-lag analysis
        logger.info("--- viz 6: lead-lag analysis ---")
        self._check_fitted("wass_X_decomposed_", "extract_features")

        self.leadlag_sig_df_ = LeadLagAnalyzer(
            max_lag=self.leadlag_max_lag,
            quantiles=list(self.leadlag_quantiles),
        ).fit_by_model(
            self.wass_X_decomposed_, self.tickers,
            cross_metric_only=True, max_pairs_to_plot=6,
            max_models_to_plot=len(self.tickers) + 1,
        )
        if self.leadlag_sig_df_ is not None and not self.leadlag_sig_df_.empty:
            out = self.results_dir / "leadlag_multimetric_quantile_significant.csv"
            self.leadlag_sig_df_.to_csv(out, index=False)
            logger.info("significant lead-lag results saved to %s", out)
        else:
            logger.warning("no significant multi-metric lead-lag results found")

        self.leadlag_ticker_metric_df_ = LeadLagAnalyzer(
            max_lag=self.leadlag_max_lag,
            quantiles=list(self.leadlag_quantiles),
        ).fit_inter_ticker(
            self.wass_X_decomposed_, self.tickers,
        )
        if self.leadlag_ticker_metric_df_ is not None and not self.leadlag_ticker_metric_df_.empty:
            out = self.results_dir / "leadlag_tickers_by_metric_quantile_significant.csv"
            self.leadlag_ticker_metric_df_.to_csv(out, index=False)
            logger.info("inter-ticker lead-lag results saved to %s", out)
        else:
            logger.warning("no significant inter-ticker lead-lag results found")

        return self


    def generate_outputs(self) -> "ContagionPipeline":
        """Event study, LaTeX tables, and all remaining figures."""
        self._run_event_study()
        self._compute_ari_mmd()
        self._save_all_tables()
        self._save_all_figures()
        return self

    def _run_event_study(self) -> None:
        """GOOG spike event study."""
        logger.info("--- event study: GOOG crash (21 June 2012) ---")
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
                logger.warning("event study format unexpected, skipping CSV export")
            logger.info("event study saved to %s", out)

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
        logger.info("--- reporting (LaTeX tables + figures) ---")

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

        # Per-ticker regime descriptive stats (Tables 6-10 in paper)
        self._write_regime_stats_latex_tables()

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

    def _write_regime_stats_latex_tables(self) -> None:
        """Write per-ticker regime descriptive stats as LaTeX tables (Tables 6-10).

        Generates ``paper/tables/regime_stats_{TICKER}.tex`` for each ticker,
        matching the format used in the paper appendix.  Reads from
        ``self.regime_stats_df_`` (set by :meth:`_save_regime_stats`).
        """
        if self.regime_stats_df_ is None or self.regime_stats_df_.empty:
            logger.warning("regime_stats_df_ not available, skipping LaTeX table generation")
            return

        label_order = {"Calm": 0, "Intermediate": 1, "Stressed": 2}

        for ticker in self.tickers:
            sub = self.regime_stats_df_[self.regime_stats_df_["ticker"] == ticker].copy()
            if sub.empty:
                continue
            sub["_order"] = sub["label"].map(label_order)
            sub = sub.sort_values("_order").drop(columns="_order")

            lines = [
                r"\begin{table}[H]",
                r"\centering\small",
                (r"\caption{Regime descriptive statistics -- " + ticker + r".}"),
                r"\label{tab:regime_stats_" + ticker.lower() + "}",
                r"\begin{tabular}{llrrrrrrr}",
                r"\toprule",
                (r"Regime & Label & $N$ (\%) & "
                 r"$\sigma_r$ (bp) & $|\bar{r}|$ (bp) & Kurt & "
                 r"$|\overline{\text{OBI}}|$ & $|\overline{\text{OFI}}|$ \\"),
                r"\midrule",
            ]
            for _, row in sub.iterrows():
                n_fmt = f"{int(row['n']):,}".replace(",", "{,}")
                n_str = f"{n_fmt} ({row['pct']:.1f}\\%)"
                lines.append(
                    f"R{int(row['regime'])} & {row['label']:<13} & "
                    f"{n_str} & "
                    f"{row['ret_std_bps']:.2f} & "
                    f"{row['ret_abs_mean_bps']:.2f} & "
                    f"{row['ret_kurt']:.1f} & "
                    f"{row['obi_abs_mean']:.2f} & "
                    f"{row['ofi_abs_mean']:.0f} \\\\"
                )
            lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

            out_path = PAPER_TABLES_DIR / f"regime_stats_{ticker}.tex"
            out_path.write_text("\n".join(lines), encoding="utf-8")
            logger.info("regime stats LaTeX saved: %s", out_path)

    def _save_all_figures(self) -> None:
        """Generate and save paper figures (posteriors + stress decomposition)."""
        wass_index = self.wass_X_all_.index

        # Cross-ticker average of original features (for global HMM ordering)
        orig_global = pd.DataFrame({
            "micro_price": pd.concat(
                [self.synced_data_[t]["micro_price"] for t in self.tickers], axis=1
            ).reindex(wass_index).mean(axis=1),
            "obi": pd.concat(
                [self.synced_data_[t]["obi"] for t in self.tickers], axis=1
            ).reindex(wass_index).mean(axis=1),
            "ofi": pd.concat(
                [self.synced_data_[t]["ofi"] for t in self.tickers], axis=1
            ).reindex(wass_index).mean(axis=1),
        })

        # Local HMM posteriors (one per ticker)
        for ticker in self.tickers:
            if ticker not in self.local_states_:
                continue
            states = self.local_states_[ticker]
            orig_aligned, states_aligned = self._get_aligned_local_data_and_states(ticker)
            local_order = _compute_regime_order(states_aligned.values, orig_aligned)
            local_colors, local_labels = _build_regime_maps(local_order)
            n_local_regimes = self.local_state_probs_[ticker].shape[1]
            _plot_posterior_stacked(
                _states_to_onehot(states, n_local_regimes),
                f"Regime Sequence (Viterbi) - {ticker}",
                PAPER_FIGURES_DIR / f"hmm_local_{ticker}_posterior.png",
                colors=local_colors,
                regime_labels=local_labels,
            )

        # Global posteriors (meta + direct) — ordered by σr of cross-ticker average
        n_meta = min(len(self.global_states_), len(orig_global))
        meta_order = _compute_regime_order(self.global_states_[:n_meta], orig_global.iloc[:n_meta])
        meta_colors, meta_labels = _build_regime_maps(meta_order)
        n_direct = min(len(self.global_direct_states_), len(orig_global))
        direct_order = _compute_regime_order(
            self.global_direct_states_[:n_direct], orig_global.iloc[:n_direct]
        )
        direct_colors, direct_labels = _build_regime_maps(direct_order)
        _plot_posterior_stacked(
            _states_to_onehot(self.global_states_, self.n_regimes),
            "Regime Sequence (Viterbi) - Meta-HMM Global",
            PAPER_FIGURES_DIR / "hmm_meta_posterior.png",
            colors=meta_colors,
            regime_labels=meta_labels,
        )
        _plot_posterior_stacked(
            _states_to_onehot(self.global_direct_states_, self.n_regimes),
            "Regime Sequence (Viterbi) - Direct Global HMM",
            PAPER_FIGURES_DIR / "hmm_direct_posterior.png",
            colors=direct_colors,
            regime_labels=direct_labels,
        )

        # Stress decomposition (direct global HMM, log scale)
        self._stress_decomposition_plot(
            self.global_direct_states_,
            "Stress Decomposition (Overlay: Direct Global HMM)",
            "stress_decomposition_direct.png",
            log_scale=True,
        )
        # Per-ticker format (3 metric rows, log scale)
        self._stress_decomposition_by_ticker_plot()


    def print_summary(self) -> "ContagionPipeline":
        """Print final pipeline summary to stdout."""
        self._check_fitted("patient_zero_info_", "analyze_contagion")
        pz = self.patient_zero_info_

        te_vals = self.te_matrix_.values
        te_nonzero = te_vals[te_vals > 0]
        te_mean_str = f"{te_nonzero.mean():.4f} nats" if te_nonzero.size else "n/a (no non-zero TE)"
        logger.info(
            "--- pipeline complete ---\n"
            "  patient zero: %s (contagion_score=%.3f, te_outgoing=%.4f nats)\n"
            "  global regimes: %d\n"
            "  mean sync rate: %.1f%%\n"
            "  mean TE: %s",
            pz["patient_zero"], pz["contagion_score"], pz["te_outgoing"],
            self.n_regimes,
            self.sync_df_["sync_rate"].mean() * 100,
            te_mean_str,
        )
        return self

    def run(self) -> "ContagionPipeline":
        """Execute the full pipeline end-to-end."""
        logger.info(
            "--- hierarchical contagion detection pipeline ---\n"
            "  analysis date: %s | tickers: %s | regimes: %d",
            self.analysis_date, ", ".join(self.tickers), self.n_regimes,
        )

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


    def _load_params(self) -> None:
        """Load optional per-ticker and global optimized parameters from disk."""
        per_ticker_csv = self.results_dir / "best_parameters_hierarchical_per_ticker.csv"
        self.use_per_ticker_ = per_ticker_csv.exists()
        if self.use_per_ticker_:
            self.per_ticker_params_ = pd.read_csv(per_ticker_csv)
            logger.info("per-ticker params loaded from %s", per_ticker_csv)
        else:
            logger.info("no per-ticker params found, falling back to global config")

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
                logger.info("direct-global params loaded from %s", direct_path)
            except Exception:
                self.best_direct_params_ = {}

    def _ticker_hmm_params(self, ticker: str):
        """Return (persistence, smoothing, n_regimes) for a ticker."""
        if self.use_per_ticker_ and self.per_ticker_params_ is not None:
            row = self.per_ticker_params_[self.per_ticker_params_["ticker"] == ticker].iloc[0]
            return float(row["local_persistence"]), int(row["local_smoothing"]), int(row["n_regimes"])
        return self.hmm_persistence_local, self.hmm_smoothing_local, self.n_regimes

    def _stress_decomposition_by_ticker_plot(self) -> None:
        """Stress decomposition: 3 metric rows, 5 ticker lines each, log scale.

        Same layout as stress_decomposition_direct.png but using Meta-HMM regimes
        for background coloring (vs Direct HMM in stress_decomposition_direct.png).
        """
        self._check_fitted("global_states_", "fit_global_hmms")
        self._check_fitted("wass_X_decomposed_", "extract_features")
        self._stress_decomposition_plot(
            self.global_states_,
            "Stress Decomposition by Ticker (Meta-HMM Regimes, Log Scale)",
            "stress_decomposition_by_ticker_log.png",
            log_scale=True,
        )
        logger.info("stress decomposition by ticker (log) saved")

    def _stress_decomposition_plot(
        self, states: np.ndarray, title: str, filename: str, log_scale: bool = False
    ) -> None:
        """Plot stress decomposition with semantically-colored regime overlays."""
        n = min(len(states), len(self.wass_X_all_))
        wass_index = self.wass_X_all_.index
        orig_global = pd.DataFrame({
            "micro_price": pd.concat(
                [self.synced_data_[t]["micro_price"] for t in self.tickers], axis=1
            ).reindex(wass_index).mean(axis=1),
            "obi": pd.concat(
                [self.synced_data_[t]["obi"] for t in self.tickers], axis=1
            ).reindex(wass_index).mean(axis=1),
            "ofi": pd.concat(
                [self.synced_data_[t]["ofi"] for t in self.tickers], axis=1
            ).reindex(wass_index).mean(axis=1),
        })
        order = _compute_regime_order(states[:n], orig_global.iloc[:n])
        colors_map, labels_map = _build_regime_maps(order)

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
                    color=colors_map[regime],
                    label=labels_map[regime] if i == 0 else "",
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

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    ContagionPipeline().run()
