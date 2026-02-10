"""
Analyse lead-lag entre métriques de stress
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
from scipy.stats import spearmanr
from itertools import combinations

from src.config import MAX_LAG, QUANTILES, ALPHA_SIGNIFICANCE, FIGURES_DIR


def analyze_multimetric_leadlag_significant_crossmetric(
    """Analyze multimetric leadlag significant crossmetric."""
    wass_decomposed: Dict,
    tickers: List[str],
    quantiles: List[float] = QUANTILES,
    max_lag: int = MAX_LAG,
    alpha: float = ALPHA_SIGNIFICANCE,
    min_obs: int = 30
) -> pd.DataFrame:
    """
    Analyse lead-lag avec focus sur corrélations significatives (p < 0.05).
    
    Génère un graphique 2x3 montrant uniquement les paires croisées
    avec affichage sélectif des corrélations significatives.
    
    Parameters
    ----------
    wass_decomposed : Dict
        Distances Wasserstein
    tickers : List[str]
        Liste des tickers
    quantiles : List[float]
        Quantiles de stress
    max_lag : int
        Lag maximal
    alpha : float
        Seuil de significativité (p-value)
    min_obs : int
        Nombre minimum d'observations requises
    
    Returns
    -------
    pd.DataFrame
        Résultats significatifs avec colonnes:
        source, target, quantile, lag_obs, lag_seconds, correlation, p_value, n_obs
    """
    
    metrics = ['price_ret', 'obi', 'ofi']
    metric_pairs = list(combinations(metrics, 2))
    
    all_pairs = []
    for m1, m2 in metric_pairs:
        all_pairs.append((m1, m2))
        all_pairs.append((m2, m1))
    
    lags = np.arange(-max_lag, max_lag + 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    significance_results = []
    
    for idx, (source_metric, target_metric) in enumerate(all_pairs):
        
        source_stress = np.mean([wass_decomposed[source_metric][t] for t in tickers], axis=0)
        target_stress = np.mean([wass_decomposed[target_metric][t] for t in tickers], axis=0)
        
        for q in quantiles:
            threshold = np.percentile(source_stress, q * 100)
            
            if q < 0.5:
                mask = source_stress <= threshold
                label = f"Q{int(q*100)}"
                alpha_val = 0.5
            else:
                mask = source_stress >= threshold
                label = f"Q{int(q*100)}"
                alpha_val = 0.8
            
            source_sub = np.array(source_stress)[mask]
            target_sub = np.array(target_stress)[mask]
            
            corrs = []
            pvals = []
            
            for lag in lags:
                if len(source_sub) <= abs(lag) or len(source_sub) < min_obs:
                    corrs.append(np.nan)
                    pvals.append(1.0)
                    continue
                
                if lag < 0:
                    r, p = spearmanr(source_sub[-lag:], target_sub[:lag], nan_policy="omit")
                elif lag > 0:
                    r, p = spearmanr(source_sub[:-lag], target_sub[lag:], nan_policy="omit")
                else:
                    r, p = spearmanr(source_sub, target_sub, nan_policy="omit")
                
                corrs.append(r)
                pvals.append(p)
                
                if p < alpha:
                    significance_results.append({
                        'source': source_metric,
                        'target': target_metric,
                        'quantile': f"Q{int(q*100)}",
                        'lag_obs': lag,
                        'lag_seconds': lag * 0.5,
                        'correlation': r,
                        'p_value': p,
                        'n_obs': len(source_sub)
                    })
            
            corrs = np.array(corrs)
            pvals = np.array(pvals)
            
            # Affichage uniquement des points significatifs
            sig_mask = pvals < alpha
            
            if np.any(sig_mask):
                lags_sig = lags[sig_mask] * 0.5
                corrs_sig = corrs[sig_mask]
                
                axes[idx].plot(lags_sig, corrs_sig, 
                              marker='o', label=label,
                              alpha=alpha_val, linewidth=1.5, markersize=4)
        
        axes[idx].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[idx].axhline(0, color='gray', linestyle=':', alpha=0.5)
        axes[idx].set_title(f'{source_metric.upper()} → {target_metric.upper()}', 
                           fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Lag (seconds)', fontsize=10)
        axes[idx].set_ylabel('Correlation', fontsize=10)
        axes[idx].legend(fontsize=8)
        axes[idx].grid(alpha=0.3)
    
    plt.suptitle(f'Lead-Lag Analysis: Significant Correlations Only (p < {alpha})', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'leadlag_crossmetric_significant.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    df_sig = pd.DataFrame(significance_results)
    
    if len(df_sig) > 0:
        df_sig.to_csv(FIGURES_DIR / 'leadlag_significant_results.csv', index=False)
        print(f"\n✓ Significant correlations saved ({len(df_sig)} results)")
    else:
        print(f"\n⚠️  No significant correlations found (α < {alpha})")
    
    return df_sig


def analyze_multimetric_leadlag_by_model(
    """Analyze multimetric leadlag by model."""
    wass_decomposed: Dict,
    tickers: List[str],
    quantiles: List[float] = QUANTILES,
    max_lag: int = MAX_LAG,
    alpha: float = ALPHA_SIGNIFICANCE,
    min_obs: int = 30,
    cross_metric_only: bool = False,
    max_models_to_plot: int = 3,
    always_plot_global: bool = True,
    max_pairs_to_plot: int = 6,
) -> pd.DataFrame:
    """
    Lead-lag multi-métrique par modèle (chaque ticker + global).

    - Génère un graphique 3x3 de courbes par quantile pour chaque ticker et pour le global.
    - Retourne un tableau des lead-lag les plus forts et significatifs par modèle/quantile/paire.

    Parameters
    ----------
    wass_decomposed : Dict
        Distances Wasserstein par métrique et ticker
    tickers : List[str]
        Liste des tickers
    quantiles : List[float]
        Quantiles de stress à analyser (ex: [0.1, 0.5, 0.9])
    max_lag : int
        Lag maximal en observations (±max_lag)
    alpha : float
        Seuil de significativité (p-value)
    min_obs : int
        Nombre minimum d'observations requises

    Returns
    -------
    pd.DataFrame
        Résultats significatifs (plus forts) par modèle/quantile/paire de métriques.
    """

    metrics = ["price_ret", "obi", "ofi"]
    if cross_metric_only:
        metric_pairs = list(combinations(metrics, 2))
    else:
        metric_pairs = list(combinations(metrics, 2)) + [(m, m) for m in metrics]

    all_pairs = []
    for m1, m2 in metric_pairs:
        all_pairs.append((m1, m2))
        if m1 != m2:
            all_pairs.append((m2, m1))

    # Build model series: each ticker + GLOBAL (mean across tickers)
    models = {}
    for t in tickers:
        models[t] = {m: np.asarray(wass_decomposed[m][t], dtype=float) for m in metrics}

    global_series = {}
    for m in metrics:
        global_series[m] = np.mean([models[t][m] for t in tickers], axis=0)
    models["GLOBAL"] = global_series

    lags = np.arange(-max_lag, max_lag + 1)
    results = []

    n_pairs = len(all_pairs)
    if n_pairs <= 6:
        nrows, ncols = 2, 3
    else:
        nrows, ncols = 3, 3

    # Pass 1: compute results and counts (no plotting)
    counts = {}
    for model_name, series_map in models.items():
        model_count = 0
        for source_metric, target_metric in all_pairs:
            source_stress = series_map[source_metric]
            target_stress = series_map[target_metric]

            for q in quantiles:
                threshold = np.percentile(source_stress, q * 100)
                if q < 0.5:
                    mask = source_stress <= threshold
                else:
                    mask = source_stress >= threshold

                source_sub = np.array(source_stress)[mask]
                target_sub = np.array(target_stress)[mask]

                corrs = []
                pvals = []
                for lag in lags:
                    if len(source_sub) <= abs(lag) or len(source_sub) < min_obs:
                        corrs.append(np.nan)
                        pvals.append(1.0)
                        continue

                    if lag < 0:
                        r, p = spearmanr(source_sub[-lag:], target_sub[:lag], nan_policy="omit")
                    elif lag > 0:
                        r, p = spearmanr(source_sub[:-lag], target_sub[lag:], nan_policy="omit")
                    else:
                        r, p = spearmanr(source_sub, target_sub, nan_policy="omit")

                    corrs.append(r)
                    pvals.append(p)

                corrs = np.array(corrs)
                pvals = np.array(pvals)

                sig_mask = pvals < alpha
                if np.any(sig_mask):
                    sig_corrs = corrs[sig_mask]
                    sig_lags = lags[sig_mask]
                    sig_pvals = pvals[sig_mask]
                    best_idx = int(np.nanargmax(np.abs(sig_corrs)))
                    best_corr = float(sig_corrs[best_idx])
                    best_lag = int(sig_lags[best_idx])
                    best_pval = float(sig_pvals[best_idx])

                    results.append(
                        {
                            "model": model_name,
                            "source_metric": source_metric,
                            "target_metric": target_metric,
                            "quantile": f"Q{int(q*100)}",
                            "best_lag_obs": best_lag,
                            "best_lag_seconds": best_lag * 0.5,
                            "best_corr": best_corr,
                            "best_pval": best_pval,
                            "n_obs": int(len(source_sub)),
                        }
                    )
                    model_count += 1

        counts[model_name] = model_count

    # Select models to plot
    ordered_models = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    plot_models = [m for m, _ in ordered_models[:max_models_to_plot]]
    if always_plot_global and "GLOBAL" in models and "GLOBAL" not in plot_models:
        plot_models = ["GLOBAL"] + plot_models[:-1]

    def _grid_shape(n_items: int):
        """Helper function for grid shape."""
        if n_items <= 1:
            return 1, 1
        if n_items == 2:
            return 1, 2
        if n_items == 3:
            return 1, 3
        if n_items == 4:
            return 2, 2
        if n_items <= 6:
            return 2, 3
        return 3, 3

    # Pass 2: plot only selected models and only most important pairs
    for model_name in plot_models:
        series_map = models[model_name]

        # Select top pairs by max |corr| across quantiles (significant only)
        model_results = [r for r in results if r["model"] == model_name]
        pair_scores = {}
        for r in model_results:
            pair = (r["source_metric"], r["target_metric"])
            score = abs(r["best_corr"])
            pair_scores[pair] = max(pair_scores.get(pair, 0.0), score)

        if pair_scores:
            ranked_pairs = sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)
            selected_pairs = [p for p, _ in ranked_pairs[:max_pairs_to_plot]]
        else:
            selected_pairs = all_pairs[:max_pairs_to_plot]

        n_pairs_plot = len(selected_pairs)
        nrows, ncols = _grid_shape(n_pairs_plot)
        fig, axes = plt.subplots(nrows, ncols, figsize=(20, 12 if nrows <= 2 else 16))
        axes = np.array(axes).reshape(-1)

        for idx, (source_metric, target_metric) in enumerate(selected_pairs):
            source_stress = series_map[source_metric]
            target_stress = series_map[target_metric]

            for q in quantiles:
                threshold = np.percentile(source_stress, q * 100)
                if q < 0.5:
                    mask = source_stress <= threshold
                    label = f"Q{int(q*100)}"
                    alpha_val = 0.5
                else:
                    mask = source_stress >= threshold
                    label = f"Q{int(q*100)}"
                    alpha_val = 0.8

                source_sub = np.array(source_stress)[mask]
                target_sub = np.array(target_stress)[mask]

                corrs = []
                for lag in lags:
                    if len(source_sub) <= abs(lag) or len(source_sub) < min_obs:
                        corrs.append(np.nan)
                        continue

                    if lag < 0:
                        r, _ = spearmanr(source_sub[-lag:], target_sub[:lag], nan_policy="omit")
                    elif lag > 0:
                        r, _ = spearmanr(source_sub[:-lag], target_sub[lag:], nan_policy="omit")
                    else:
                        r, _ = spearmanr(source_sub, target_sub, nan_policy="omit")

                    corrs.append(r)

                axes[idx].plot(
                    lags * 0.5,
                    corrs,
                    marker="o",
                    label=label,
                    alpha=alpha_val,
                    linewidth=1.5,
                )

            axes[idx].axvline(0, color="red", linestyle="--", linewidth=2)
            axes[idx].axhline(0, color="gray", linestyle=":", alpha=0.5)
            axes[idx].set_title(
                f"{source_metric.upper()} → {target_metric.upper()}",
                fontsize=12,
                fontweight="bold",
            )
            axes[idx].set_xlabel("Lag (seconds)", fontsize=10)
            axes[idx].set_ylabel("Correlation", fontsize=10)
            axes[idx].legend(fontsize=8)
            axes[idx].grid(alpha=0.3)

        for j in range(idx + 1, len(axes)):
            axes[j].axis("off")

        plt.suptitle(
            f"Multi-Metric Lead-Lag by Quantile ({model_name})",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )
        plt.tight_layout()
        plt.savefig(
            FIGURES_DIR / f"leadlag_multimetric_grid_{model_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print(f"\n✓ Lead-lag grid saved for {model_name}")

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(
            ["model", "quantile", "source_metric", "target_metric"],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)
    return df


def analyze_interticker_leadlag_by_metric_quantile(
    """Analyze interticker leadlag by metric quantile."""
    wass_decomposed: Dict,
    tickers: List[str],
    quantiles: List[float] = QUANTILES,
    max_lag: int = MAX_LAG,
    alpha: float = ALPHA_SIGNIFICANCE,
    min_obs: int = 30,
    max_heatmaps_per_metric: int = 1,
) -> pd.DataFrame:
    """
    Lead-lag entre tickers par métrique et quantile.

    - Génère une heatmap par métrique et quantile avec la corrélation la plus forte (significative).
    - Retourne un tableau des lead-lag les plus forts et significatifs par métrique/quantile/pair.
    """

    metrics = ["price_ret", "obi", "ofi"]
    lags = np.arange(-max_lag, max_lag + 1)
    results = []

    for metric in metrics:
        heatmaps_by_quantile = {}
        scores_by_quantile = {}
        for q in quantiles:
            heatmap = np.full((len(tickers), len(tickers)), np.nan, dtype=float)

            for i, t1 in enumerate(tickers):
                s1 = np.asarray(wass_decomposed[metric][t1], dtype=float)
                threshold = np.percentile(s1, q * 100)
                if q < 0.5:
                    mask = s1 <= threshold
                else:
                    mask = s1 >= threshold

                s1_sub = s1[mask]

                for j, t2 in enumerate(tickers):
                    if t1 == t2:
                        continue
                    s2 = np.asarray(wass_decomposed[metric][t2], dtype=float)
                    s2_sub = s2[mask]

                    corrs = []
                    pvals = []
                    for lag in lags:
                        if len(s1_sub) <= abs(lag) or len(s1_sub) < min_obs:
                            corrs.append(np.nan)
                            pvals.append(1.0)
                            continue

                        if lag < 0:
                            r, p = spearmanr(s1_sub[-lag:], s2_sub[:lag], nan_policy="omit")
                        elif lag > 0:
                            r, p = spearmanr(s1_sub[:-lag], s2_sub[lag:], nan_policy="omit")
                        else:
                            r, p = spearmanr(s1_sub, s2_sub, nan_policy="omit")

                        corrs.append(r)
                        pvals.append(p)

                    corrs = np.array(corrs)
                    pvals = np.array(pvals)
                    sig_mask = pvals < alpha
                    if np.any(sig_mask):
                        sig_corrs = corrs[sig_mask]
                        sig_lags = lags[sig_mask]
                        best_idx = int(np.nanargmax(np.abs(sig_corrs)))
                        best_corr = float(sig_corrs[best_idx])
                        best_lag = int(sig_lags[best_idx])
                        best_pval = float(pvals[sig_mask][best_idx])

                        heatmap[i, j] = best_corr
                        results.append(
                            {
                                "metric": metric,
                                "quantile": f"Q{int(q*100)}",
                                "ticker1": t1,
                                "ticker2": t2,
                                "best_lag_obs": best_lag,
                                "best_lag_seconds": best_lag * 0.5,
                                "best_corr": best_corr,
                                "best_pval": best_pval,
                                "n_obs": int(len(s1_sub)),
                            }
                        )

            q_label = f"Q{int(q*100)}"
            heatmaps_by_quantile[q_label] = heatmap
            if np.isfinite(heatmap).any():
                scores_by_quantile[q_label] = float(np.nanmax(np.abs(heatmap)))
            else:
                scores_by_quantile[q_label] = -np.inf

        # Plot only the most important quantiles per metric
        ranked = sorted(scores_by_quantile.items(), key=lambda x: x[1], reverse=True)
        selected = [q for q, _ in ranked[:max_heatmaps_per_metric] if np.isfinite(scores_by_quantile[q])]
        for q_label in selected:
            heatmap = heatmaps_by_quantile[q_label]
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
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(
            ["metric", "quantile", "ticker1", "ticker2"],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)
    return df
