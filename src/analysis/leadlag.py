"""
Analyse lead-lag entre métriques de stress
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
from scipy.stats import pearsonr
from itertools import combinations

from src.config import MAX_LAG, QUANTILES, ALPHA_SIGNIFICANCE, FIGURES_DIR


def analyze_multimetric_leadlag_full(
    wass_decomposed: Dict,
    tickers: List[str],
    quantiles: List[float] = QUANTILES,
    max_lag: int = MAX_LAG
) -> None:
    """
    Analyse lead-lag complète (9 paires incluant auto-corrélations).
    
    Génère un graphique 3x3 montrant toutes les combinaisons de métriques.
    
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
    """
    
    metrics = ['price_ret', 'obi', 'ofi']
    metric_pairs = list(combinations(metrics, 2)) + [(m, m) for m in metrics]
    
    all_pairs = []
    for m1, m2 in metric_pairs:
        all_pairs.append((m1, m2))
        if m1 != m2:
            all_pairs.append((m2, m1))
    
    lags = np.arange(-max_lag, max_lag + 1)
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    axes = axes.flatten()
    
    for idx, (source_metric, target_metric) in enumerate(all_pairs[:9]):
        
        source_stress = np.mean([wass_decomposed[source_metric][t] for t in tickers], axis=0)
        target_stress = np.mean([wass_decomposed[target_metric][t] for t in tickers], axis=0)
        
        for q in quantiles:
            threshold = np.percentile(source_stress, q * 100)
            
            if q < 0.5:
                mask = source_stress <= threshold
                label = f"Q{int(q*100)}"
                alpha = 0.5
            else:
                mask = source_stress >= threshold
                label = f"Q{int(q*100)}"
                alpha = 0.8
            
            source_sub = np.array(source_stress)[mask]
            target_sub = np.array(target_stress)[mask]
            
            corrs = []
            for lag in lags:
                if len(source_sub) <= abs(lag):
                    corrs.append(np.nan)
                    continue
                    
                if lag < 0:
                    r, _ = pearsonr(source_sub[-lag:], target_sub[:lag])
                elif lag > 0:
                    r, _ = pearsonr(source_sub[:-lag], target_sub[lag:])
                else:
                    r, _ = pearsonr(source_sub, target_sub)
                
                corrs.append(r)
            
            axes[idx].plot(lags * 0.5, corrs, marker='o', label=label, alpha=alpha, linewidth=1.5)
        
        axes[idx].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[idx].axhline(0, color='gray', linestyle=':', alpha=0.5)
        axes[idx].set_title(f'{source_metric.upper()} → {target_metric.upper()}', 
                           fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Lag (seconds)', fontsize=10)
        axes[idx].set_ylabel('Correlation', fontsize=10)
        axes[idx].legend(fontsize=8)
        axes[idx].grid(alpha=0.3)
    
    plt.suptitle('Multi-Metric Lead-Lag Analysis by Stress Quantile', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'leadlag_multimetric_grid.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Lead-lag grid (9 panels) saved")


def analyze_multimetric_leadlag_significant_crossmetric(
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
                    r, p = pearsonr(source_sub[-lag:], target_sub[:lag])
                elif lag > 0:
                    r, p = pearsonr(source_sub[:-lag], target_sub[lag:])
                else:
                    r, p = pearsonr(source_sub, target_sub)
                
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
