from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.config import MAX_LAG, QUANTILES, ALPHA_SIGNIFICANCE, FIGURES_DIR


class LeadLagAnalyzer:
    """
    Lead-lag relationships between LOB stress metrics.

    Parameters shared across all analysis methods are set once at
    construction.  Results from the last ``fit_*`` call are stored as
    ``attr_`` attributes.

    Parameters
    ----------
    max_lag : int
        Maximum lag in observations (symmetric: ``±max_lag``).
    alpha : float
        Significance threshold for Spearman p-values.
    quantiles : list of float
        Quantile thresholds used to subset the data.
    min_obs : int
        Minimum observations required in a quantile subset to compute
        a correlation.

    Attributes
    ----------
    results_by_model_ : pd.DataFrame or None
        Output of the last ``fit_by_model`` call.
    results_cross_metric_ : pd.DataFrame or None
        Output of the last ``fit_cross_metric`` call.
    results_inter_ticker_ : pd.DataFrame or None
        Output of the last ``fit_inter_ticker`` call.
    """

    def __init__(
        self,
        max_lag: int = MAX_LAG,
        alpha: float = ALPHA_SIGNIFICANCE,
        quantiles: List[float] = None,
        min_obs: int = 30,
    ) -> None:
        self.max_lag = max_lag
        self.alpha = alpha
        self.quantiles = quantiles if quantiles is not None else list(QUANTILES)
        self.min_obs = min_obs
        self.results_by_model_:     Optional[pd.DataFrame] = None
        self.results_cross_metric_: Optional[pd.DataFrame] = None
        self.results_inter_ticker_: Optional[pd.DataFrame] = None

    def fit_by_model(
        self,
        wass_decomposed: Dict,
        tickers: List[str],
        cross_metric_only: bool = False,
        max_models_to_plot: int = 3,
        always_plot_global: bool = True,
        max_pairs_to_plot: int = 6,
    ) -> pd.DataFrame:
        """
        Lead-lag per model (each ticker + GLOBAL average).

        For each model, selects top metric pairs by significance count and
        saves one grid figure per model.

        Returns
        -------
        pd.DataFrame
            Strongest significant result per model / quantile / metric pair.
            Columns: model, source_metric, target_metric, quantile,
            best_lag_obs, best_lag_seconds, best_corr, best_pval, n_obs.
        """
        metrics = ["price_ret", "obi", "ofi"]
        if cross_metric_only:
            base_pairs = list(combinations(metrics, 2))
        else:
            base_pairs = list(combinations(metrics, 2)) + [(m, m) for m in metrics]

        all_pairs = []
        for m1, m2 in base_pairs:
            all_pairs.append((m1, m2))
            if m1 != m2:
                all_pairs.append((m2, m1))

        models: Dict[str, Dict[str, np.ndarray]] = {}
        for t in tickers:
            models[t] = {m: np.asarray(wass_decomposed[m][t], dtype=float) for m in metrics}
        models["GLOBAL"] = {
            m: np.mean([models[t][m] for t in tickers], axis=0) for m in metrics
        }

        lags = np.arange(-self.max_lag, self.max_lag + 1)
        results = []
        counts: Dict[str, int] = {}

        for model_name, series_map in models.items():
            model_count = 0
            for src_m, tgt_m in all_pairs:
                for q in self.quantiles:
                    sub_src, sub_tgt = self._quantile_subset(
                        series_map[src_m], series_map[tgt_m], q
                    )
                    row = self._best_sig_lag(sub_src, sub_tgt, lags)
                    if row is not None:
                        best_lag, best_corr, best_pval = row
                        results.append(
                            {
                                "model":           model_name,
                                "source_metric":   src_m,
                                "target_metric":   tgt_m,
                                "quantile":        f"Q{int(q * 100)}",
                                "best_lag_obs":    best_lag,
                                "best_lag_seconds": best_lag * 0.5,
                                "best_corr":       best_corr,
                                "best_pval":       best_pval,
                                "n_obs":           int(len(sub_src)),
                            }
                        )
                        model_count += 1
            counts[model_name] = model_count

        ordered = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        plot_models = [m for m, _ in ordered[:max_models_to_plot]]
        if always_plot_global and "GLOBAL" in models and "GLOBAL" not in plot_models:
            plot_models = ["GLOBAL"] + plot_models[:-1]

        for model_name in plot_models:
            series_map = models[model_name]
            model_results = [r for r in results if r["model"] == model_name]

            pair_scores: Dict[Tuple, float] = {}
            for r in model_results:
                pair = (r["source_metric"], r["target_metric"])
                pair_scores[pair] = max(pair_scores.get(pair, 0.0), abs(r["best_corr"]))

            if pair_scores:
                ranked = sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)
                selected = [p for p, _ in ranked[:max_pairs_to_plot]]
            else:
                selected = all_pairs[:max_pairs_to_plot]

            nrows, ncols = self._grid_shape(len(selected))
            fig, axes = plt.subplots(nrows, ncols, figsize=(20, 12 if nrows <= 2 else 16))
            axes = np.array(axes).reshape(-1)

            for idx, (src_m, tgt_m) in enumerate(selected):
                src_s = series_map[src_m]
                tgt_s = series_map[tgt_m]
                for q in self.quantiles:
                    sub_src, sub_tgt = self._quantile_subset(src_s, tgt_s, q)
                    corrs = [
                        self._spearman_at_lag(sub_src, sub_tgt, lag)[0] for lag in lags
                    ]
                    axes[idx].plot(
                        lags * 0.5, corrs,
                        marker="o", label=f"Q{int(q * 100)}",
                        alpha=0.8 if q >= 0.5 else 0.5, linewidth=1.5,
                    )
                axes[idx].axvline(0, color="red", linestyle="--", linewidth=2)
                axes[idx].axhline(0, color="gray", linestyle=":", alpha=0.5)
                axes[idx].set_title(f"{src_m.upper()} → {tgt_m.upper()}", fontsize=12, fontweight="bold")
                axes[idx].set_xlabel("Lag (seconds)", fontsize=10)
                axes[idx].set_ylabel("Correlation", fontsize=10)
                axes[idx].legend(fontsize=8)
                axes[idx].grid(alpha=0.3)

            for j in range(idx + 1, len(axes)):
                axes[j].axis("off")

            plt.suptitle(f"Multi-Metric Lead-Lag by Quantile ({model_name})", fontsize=16, fontweight="bold", y=0.995)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"leadlag_multimetric_grid_{model_name}.png", dpi=300, bbox_inches="tight")
            plt.close()
            print(f"\nOK Lead-lag grid saved for {model_name}")

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values(
                ["model", "quantile", "source_metric", "target_metric"]
            ).reset_index(drop=True)
        self.results_by_model_ = df
        return df

    def fit_cross_metric(
        self,
        wass_decomposed: Dict,
        tickers: List[str],
    ) -> pd.DataFrame:
        """
        Cross-metric lead-lag with 2×3 grid (significant correlations only).

        Produces ``leadlag_crossmetric_significant.png`` in ``FIGURES_DIR``.

        Returns
        -------
        pd.DataFrame
            All significant (source, target, quantile, lag, corr, p_value) rows.
        """
        metrics = ["price_ret", "obi", "ofi"]
        pairs = [(m1, m2) for m1, m2 in combinations(metrics, 2)]
        all_pairs = [(m1, m2) for m1, m2 in pairs] + [(m2, m1) for m1, m2 in pairs]

        lags = np.arange(-self.max_lag, self.max_lag + 1)
        sig_rows = []

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()

        for idx, (src_m, tgt_m) in enumerate(all_pairs):
            src_agg = np.mean([wass_decomposed[src_m][t] for t in tickers], axis=0)
            tgt_agg = np.mean([wass_decomposed[tgt_m][t] for t in tickers], axis=0)

            for q in self.quantiles:
                sub_src, sub_tgt = self._quantile_subset(src_agg, tgt_agg, q)
                corrs, pvals = [], []
                for lag in lags:
                    if len(sub_src) <= abs(lag) or len(sub_src) < self.min_obs:
                        corrs.append(np.nan)
                        pvals.append(1.0)
                        continue
                    r, p = self._spearman_at_lag(sub_src, sub_tgt, lag)
                    corrs.append(r)
                    pvals.append(p)
                    if p < self.alpha:
                        sig_rows.append(
                            {
                                "source":      src_m,
                                "target":      tgt_m,
                                "quantile":    f"Q{int(q * 100)}",
                                "lag_obs":     int(lag),
                                "lag_seconds": lag * 0.5,
                                "correlation": r,
                                "p_value":     p,
                                "n_obs":       int(len(sub_src)),
                            }
                        )

                corrs = np.array(corrs)
                pvals = np.array(pvals)
                sig_mask = pvals < self.alpha
                if np.any(sig_mask):
                    axes[idx].plot(
                        lags[sig_mask] * 0.5, corrs[sig_mask],
                        marker="o", label=f"Q{int(q * 100)}",
                        alpha=0.8 if q >= 0.5 else 0.5, linewidth=1.5, markersize=4,
                    )

            axes[idx].axvline(0, color="red", linestyle="--", linewidth=2)
            axes[idx].axhline(0, color="gray", linestyle=":", alpha=0.5)
            axes[idx].set_title(f"{src_m.upper()} → {tgt_m.upper()}", fontsize=12, fontweight="bold")
            axes[idx].set_xlabel("Lag (seconds)", fontsize=10)
            axes[idx].set_ylabel("Correlation", fontsize=10)
            axes[idx].legend(fontsize=8)
            axes[idx].grid(alpha=0.3)

        plt.suptitle(
            f"Lead-Lag Analysis: Significant Correlations Only (p < {self.alpha})",
            fontsize=16, fontweight="bold", y=0.995,
        )
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "leadlag_crossmetric_significant.png", dpi=300, bbox_inches="tight")
        plt.close()

        df = pd.DataFrame(sig_rows)
        if not df.empty:
            df.to_csv(FIGURES_DIR / "leadlag_significant_results.csv", index=False)
            print(f"\nOK Significant correlations saved ({len(df)} results)")
        else:
            print(f"\n  No significant correlations found (α < {self.alpha})")

        self.results_cross_metric_ = df
        return df

    def fit_inter_ticker(
        self,
        wass_decomposed: Dict,
        tickers: List[str],
        max_heatmaps_per_metric: int = 1,
    ) -> pd.DataFrame:
        """
        Inter-ticker lead-lag by metric and quantile.

        Generates one heatmap per metric (best quantile) and returns a
        DataFrame of all significant results.

        Returns
        -------
        pd.DataFrame
            Columns: metric, quantile, ticker1, ticker2, best_lag_obs,
            best_lag_seconds, best_corr, best_pval, n_obs.
        """
        metrics = ["price_ret", "obi", "ofi"]
        lags = np.arange(-self.max_lag, self.max_lag + 1)
        results = []

        for metric in metrics:
            heatmaps: Dict[str, np.ndarray] = {}
            scores:   Dict[str, float]      = {}

            for q in self.quantiles:
                q_label = f"Q{int(q * 100)}"
                heatmap = np.full((len(tickers), len(tickers)), np.nan)

                for i, t1 in enumerate(tickers):
                    s1 = np.asarray(wass_decomposed[metric][t1], dtype=float)
                    sub_s1 = self._quantile_mask(s1, q)

                    for j, t2 in enumerate(tickers):
                        if t1 == t2:
                            continue
                        s2 = np.asarray(wass_decomposed[metric][t2], dtype=float)
                        s2_sub = s2[self._quantile_index(s1, q)]

                        row = self._best_sig_lag_from_arrays(sub_s1, s2_sub, lags)
                        if row is not None:
                            best_lag, best_corr, best_pval = row
                            heatmap[i, j] = best_corr
                            results.append(
                                {
                                    "metric":          metric,
                                    "quantile":        q_label,
                                    "ticker1":         t1,
                                    "ticker2":         t2,
                                    "best_lag_obs":    best_lag,
                                    "best_lag_seconds": best_lag * 0.5,
                                    "best_corr":       best_corr,
                                    "best_pval":       best_pval,
                                    "n_obs":           int(len(sub_s1)),
                                }
                            )

                heatmaps[q_label] = heatmap
                scores[q_label] = (
                    float(np.nanmax(np.abs(heatmap)))
                    if np.isfinite(heatmap).any()
                    else -np.inf
                )

            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for q_label, _ in ranked[:max_heatmaps_per_metric]:
                if not np.isfinite(scores[q_label]):
                    continue
                heatmap = heatmaps[q_label]
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(heatmap, cmap="coolwarm", vmin=-0.3, vmax=0.3)
                ax.set_xticks(range(len(tickers)))
                ax.set_yticks(range(len(tickers)))
                ax.set_xticklabels(tickers, rotation=45, ha="right")
                ax.set_yticklabels(tickers)
                ax.set_title(f"{metric.upper()} Lead-Lag (Quantile {q_label})")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Best corr (sig)")
                plt.tight_layout()
                plt.savefig(
                    FIGURES_DIR / f"leadlag_tickers_{metric}_{q_label}.png",
                    dpi=300, bbox_inches="tight",
                )
                plt.close()

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values(["metric", "quantile", "ticker1", "ticker2"]).reset_index(drop=True)
        self.results_inter_ticker_ = df
        return df

    def _spearman_at_lag(
        self, s1: np.ndarray, s2: np.ndarray, lag: int
    ) -> Tuple[float, float]:
        """Spearman correlation between s1 and s2 shifted by ``lag``."""
        if len(s1) <= abs(lag) or len(s1) < self.min_obs:
            return np.nan, 1.0
        if lag < 0:
            a, b = s1[-lag:], s2[:lag]
        elif lag > 0:
            a, b = s1[:-lag], s2[lag:]
        else:
            a, b = s1, s2
        r, p = spearmanr(a, b, nan_policy="omit")
        return float(r), float(p)

    def _quantile_subset(
        self, s1: np.ndarray, s2: np.ndarray, q: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (s1[mask], s2[mask]) where mask selects the q-quantile tail of s1."""
        threshold = np.percentile(s1, q * 100)
        mask = s1 <= threshold if q < 0.5 else s1 >= threshold
        return np.asarray(s1)[mask], np.asarray(s2)[mask]

    def _quantile_mask(self, s: np.ndarray, q: float) -> np.ndarray:
        """Return the quantile-subset of s."""
        threshold = np.percentile(s, q * 100)
        mask = s <= threshold if q < 0.5 else s >= threshold
        return s[mask]

    def _quantile_index(self, s: np.ndarray, q: float) -> np.ndarray:
        """Return boolean index array for the quantile-subset of s."""
        threshold = np.percentile(s, q * 100)
        return s <= threshold if q < 0.5 else s >= threshold

    def _best_sig_lag(
        self, s1: np.ndarray, s2: np.ndarray, lags: np.ndarray
    ) -> Optional[Tuple[int, float, float]]:
        """Return (best_lag, best_corr, best_pval) for the most significant lag, or None."""
        corrs, pvals = [], []
        for lag in lags:
            r, p = self._spearman_at_lag(s1, s2, lag)
            corrs.append(r)
            pvals.append(p)
        return self._pick_best(np.array(corrs), np.array(pvals), lags)

    def _best_sig_lag_from_arrays(
        self, s1: np.ndarray, s2: np.ndarray, lags: np.ndarray
    ) -> Optional[Tuple[int, float, float]]:
        """Like _best_sig_lag but uses pre-subsetted arrays (s1, s2 already masked)."""
        corrs, pvals = [], []
        for lag in lags:
            if len(s1) <= abs(lag) or len(s1) < self.min_obs:
                corrs.append(np.nan)
                pvals.append(1.0)
                continue
            r, p = self._spearman_at_lag(s1, s2, lag)
            corrs.append(r)
            pvals.append(p)
        return self._pick_best(np.array(corrs), np.array(pvals), lags)

    def _pick_best(
        self, corrs: np.ndarray, pvals: np.ndarray, lags: np.ndarray
    ) -> Optional[Tuple[int, float, float]]:
        """Pick the lag with highest |corr| among significant ones."""
        sig = pvals < self.alpha
        if not np.any(sig):
            return None
        sig_c = corrs[sig]
        sig_l = lags[sig]
        sig_p = pvals[sig]
        best  = int(np.nanargmax(np.abs(sig_c)))
        return int(sig_l[best]), float(sig_c[best]), float(sig_p[best])

    @staticmethod
    def _grid_shape(n: int) -> Tuple[int, int]:
        """Choose a compact (rows, cols) grid for n subplots."""
        if n <= 1: return 1, 1
        if n == 2: return 1, 2
        if n == 3: return 1, 3
        if n == 4: return 2, 2
        if n <= 6: return 2, 3
        return 3, 3
