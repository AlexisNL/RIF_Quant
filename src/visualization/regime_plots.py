"""
Visualisation et statistiques des régimes HMM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List

from src.config import TICKERS, FIGURES_DIR


def plot_regime_statistics(
    wass_X: np.ndarray,
    states: np.ndarray,
    tickers: List[str] = TICKERS,
    selected_vars: List[str] = None
) -> pd.DataFrame:
    """
    Calcule et visualise les statistiques descriptives par régime.
    
    Parameters
    ----------
    wass_X : np.ndarray
        Matrice des features Wasserstein (selected variables only)
    states : np.ndarray
        États de régimes (HMM)
    tickers : List[str]
        Liste des tickers
    selected_vars : List[str], optional
        Variables sélectionnées (e.g., ['Price', 'OBI'])
        If None, assumes all three variables
    
    Returns
    -------
    pd.DataFrame
        Statistiques par régime
    """
    
    if selected_vars is None:
        selected_vars = ['Price', 'OBI', 'OFI']
    
    n_tickers = len(tickers)
    n_vars = len(selected_vars)
    
    # Map variable names to slices in wass_X
    var_slices = {}
    offset = 0
    for var in selected_vars:
        var_slices[var] = slice(offset, offset + n_tickers)
        offset += n_tickers
    
    results = []
    
    for regime in np.unique(states):
        mask = (states == regime)
        
        row = {
            'regime': int(regime),
            'n_obs': int(np.sum(mask)),
            'pct': float(np.sum(mask)/len(states)*100)
        }
        
        # Extract stress for each selected variable
        for var in selected_vars:
            if var in var_slices:
                var_data = wass_X[mask, var_slices[var]]
                row[f'{var.lower()}_mean'] = float(np.mean(var_data))
                row[f'{var.lower()}_std'] = float(np.std(var_data))
        
        # Compute median duration (approximate)
        regime_changes = np.diff(np.concatenate([[False], mask, [False]]).astype(int))
        starts = np.where(regime_changes == 1)[0]
        ends = np.where(regime_changes == -1)[0]
        
        if len(starts) > 0 and len(ends) > 0:
            durations = ends - starts
            median_dur = float(np.median(durations)) * 0.5  # 500ms per obs
        else:
            median_dur = 0.0
        
        row['median_duration'] = median_dur
        
        results.append(row)
    
    df_stats = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("STATISTIQUES DESCRIPTIVES PAR RÉGIME")
    print("="*80)
    print(df_stats.to_string(index=False))
    
    # Sauvegarde CSV
    df_stats.to_csv(FIGURES_DIR / 'regime_statistics.csv', index=False)
    
    # Visualisation
    _plot_regime_bars(df_stats)
    
    return df_stats


def _plot_regime_bars(df_stats: pd.DataFrame) -> None:
    """Graphique à barres des statistiques de régimes."""
    
    # Detect which variables are present
    available_vars = []
    for var in ['Price', 'OBI', 'OFI']:
        if f'{var.lower()}_mean' in df_stats.columns:
            available_vars.append(var)
    
    if not available_vars:
        print("⚠️  No variables found for plotting")
        return
    
    n_vars = len(available_vars)
    fig, axes = plt.subplots(1, n_vars, figsize=(6*n_vars, 5))
    
    # Handle single variable case
    if n_vars == 1:
        axes = [axes]
    
    for i, metric in enumerate(available_vars):
        col_mean = f'{metric.lower()}_mean'
        col_std = f'{metric.lower()}_std'
        
        means = df_stats[col_mean].values
        stds = df_stats[col_std].values if col_std in df_stats.columns else None
        
        axes[i].bar(
            df_stats['regime'], 
            means, 
            yerr=stds,
            capsize=5, 
            alpha=0.7, 
            color=['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(means)]
        )
        
        axes[i].set_xlabel('Regime', fontsize=12, fontweight='bold')
        axes[i].set_ylabel(f'{metric} Stress (mean)', fontsize=12, fontweight='bold')
        axes[i].set_title(f'{metric} Stress Distribution', fontsize=14, fontweight='bold')
        axes[i].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'regime_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Regime statistics plot saved")


def plot_stress_decomposition(
    wass_decomposed: Dict,
    states: np.ndarray,
    tickers: List[str] = TICKERS
) -> None:
    """
    Visualise le stress décomposé par ticker et métrique.
    
    Génère un graphique 3x1 montrant l'évolution temporelle du stress
    pour chaque métrique, avec overlay des régimes HMM.
    
    Parameters
    ----------
    wass_decomposed : Dict
        Distances Wasserstein par métrique et ticker
    states : np.ndarray
        États de régimes
    tickers : List[str]
        Liste des tickers
    """
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    metric_names = ['Price', 'OBI', 'OFI']
    metric_keys = ['price_ret', 'obi', 'ofi']
    
    for i, (m_name, m_key) in enumerate(zip(metric_names, metric_keys)):
        # Plot du stress par ticker
        for ticker in tickers:
            stress_series = wass_decomposed[m_key][ticker]
            axes[i].plot(
                range(len(stress_series)), 
                stress_series,
                label=ticker, 
                alpha=0.7, 
                linewidth=1.5
            )
        
        # Overlay des régimes
        for regime in np.unique(states):
            mask = (states == regime)
            axes[i].fill_between(
                range(len(states)), 
                0, 
                axes[i].get_ylim()[1],
                where=mask, 
                alpha=0.15,
                label=f'Regime {regime}' if i == 0 else ''
            )
        
        axes[i].set_ylabel(f'{m_name} Stress', fontsize=12, fontweight='bold')
        axes[i].legend(loc='upper right', ncol=6)
        axes[i].grid(alpha=0.3)
    
    axes[2].set_xlabel('Time (index)', fontsize=12)
    fig.suptitle(
        'Stress Decomposition by Ticker and Metric', 
        fontsize=16, 
        fontweight='bold', 
        y=0.995
    )
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'stress_decomposition_by_ticker.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Stress decomposition plot saved")