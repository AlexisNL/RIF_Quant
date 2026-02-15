"""
Event Study: Analyse du spike GOOG et réallocation cross-sectionnelle
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple

from src.config import (
    EVENT_SPIKE_TIME, EVENT_WINDOW_WIDE, EVENT_WINDOW_NARROW,
    FIGURE_DPI, FONT_SIZE_TITLE, FONT_SIZE_LABEL
)


def analyze_event_goog_spike(
    synced_data: Dict[str, pd.DataFrame],
    regime_states: np.ndarray,
    wasserstein_window: int
) -> Dict:
    """
    Analyse l'événement du spike GOOG et la réallocation cross-sectionnelle.
    
    Parameters
    ----------
    synced_data : dict
        Données synchronisées par ticker
    regime_states : np.ndarray
        États de régime détectés par le HMM
    wasserstein_window : int
        Fenêtre Wasserstein (pour ajuster l'offset des régimes)
    
    Returns
    -------
    dict
        Résultats avec delta_ofi, absorption MSFT, régime, figure path
    """
    
    from src.config import TICKERS, FIGURES_DIR
    
    tickers = TICKERS
    output_path = FIGURES_DIR
    
    spike_time = EVENT_SPIKE_TIME
    window_wide = EVENT_WINDOW_WIDE
    window_narrow = EVENT_WINDOW_NARROW
    
    # ========================================================================
    # CALCUL DES DELTAS OFI
    # ========================================================================
    
    delta_ofi = {}
    
    for ticker in tickers:
        ofi_before = synced_data[ticker]['ofi'].iloc[
            spike_time - window_narrow : spike_time
        ].mean()
        
        ofi_after = synced_data[ticker]['ofi'].iloc[
            spike_time : spike_time + window_narrow
        ].mean()
        
        delta_ofi[ticker] = ofi_after - ofi_before
    
    # ========================================================================
    # STATISTIQUES DE RÉALLOCATION
    # ========================================================================
    
    # GOOG est la source (perte de liquidité)
    goog_loss = delta_ofi['GOOG']
    
    # Autres tickers sont les absorbeurs
    other_tickers = [t for t in tickers if t != 'GOOG']
    reallocation = {t: delta_ofi[t] for t in other_tickers}
    total_reallocation = sum(reallocation.values())
    
    # Pourcentages d'absorption
    absorption_pct = {
        t: (delta_ofi[t] / total_reallocation * 100) 
        for t in other_tickers
    }
    
    msft_absorption_pct = absorption_pct['MSFT']
    
    # ========================================================================
    # RÉGIME ACTIF AU MOMENT DU SPIKE
    # ========================================================================
    
    # Le spike se produit à l'index spike_time dans synced_data
    # Les régimes commencent après WASSERSTEIN_WINDOW observations
    # Il faut donc ajuster l'index
    from src.config import WASSERSTEIN_WINDOW
    regime_index = spike_time - WASSERSTEIN_WINDOW
    
    if 0 <= regime_index < len(regime_states):
        regime_at_spike = int(regime_states[regime_index])
    else:
        regime_at_spike = None
    
    # ========================================================================
    # VISUALISATION
    # ========================================================================
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # ========================================================================
    # PANEL A: VUE LARGE (±16 minutes)
    # ========================================================================
    
    for ticker in tickers:
        ofi_series = synced_data[ticker]['ofi'].iloc[
            spike_time - window_wide : spike_time + window_wide
        ]
        axes[0].plot(
            range(len(ofi_series)), 
            ofi_series.values, 
            label=ticker, 
            alpha=0.7, 
            linewidth=1.5
        )
    
    # Ligne verticale au moment du spike
    axes[0].axvline(
        window_wide, 
        color='red', 
        linestyle='--', 
        linewidth=2, 
        label='GOOG Spike'
    )
    
    axes[0].set_title(
        'Context: OFI Evolution Around GOOG Spike (±16 min)', 
        fontsize=FONT_SIZE_TITLE, 
        fontweight='bold'
    )
    axes[0].set_ylabel('OFI', fontsize=FONT_SIZE_LABEL)
    axes[0].legend(fontsize=10, ncol=6)
    axes[0].grid(alpha=0.3)
    
    # ========================================================================
    # PANEL B: ZOOM (±4 minutes)
    # ========================================================================
    
    for ticker in tickers:
        ofi_series = synced_data[ticker]['ofi'].iloc[
            spike_time - window_narrow : spike_time + window_narrow
        ]
        axes[1].plot(
            range(len(ofi_series)), 
            ofi_series.values, 
            label=ticker, 
            linewidth=2
        )
    
    # Ligne verticale au moment du spike
    axes[1].axvline(
        window_narrow, 
        color='red', 
        linestyle='--', 
        linewidth=2, 
        label='Spike Time'
    )
    
    # Annotation du régime si disponible
    if regime_at_spike is not None:
        axes[1].text(
            0.02, 0.98,
            f'Regime {regime_at_spike}',
            transform=axes[1].transAxes,
            fontsize=12,
            fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
    
    axes[1].set_title(
        'Zoom: Cross-Ticker OFI Reallocation (±4 min)', 
        fontsize=FONT_SIZE_TITLE, 
        fontweight='bold'
    )
    axes[1].set_xlabel(
        'Time (observations, 500ms each)', 
        fontsize=FONT_SIZE_LABEL
    )
    axes[1].set_ylabel('OFI', fontsize=FONT_SIZE_LABEL)
    axes[1].legend(fontsize=10, ncol=6)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarde
    figure_path = output_path / 'event_study_goog_spike_optimal.png'
    plt.savefig(figure_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # TABLEAU RÉCAPITULATIF
    # ========================================================================
    
    results_df = pd.DataFrame([
        {
            'Ticker': 'GOOG',
            'Delta_OFI': goog_loss,
            'Absorption_Pct': np.nan,
            'Role': 'Liquidity source'
        }
    ] + [
        {
            'Ticker': t,
            'Delta_OFI': delta_ofi[t],
            'Absorption_Pct': absorption_pct[t],
            'Role': 'Primary absorber' if t == 'MSFT' else 'Secondary absorber'
        }
        for t in sorted(other_tickers, key=lambda x: absorption_pct[x], reverse=True)
    ])
    
    # Sauvegarde CSV
    csv_path = output_path / 'event_study_reallocation.csv'
    results_df.to_csv(csv_path, index=False, float_format='%.2f')
    
    # ========================================================================
    # AFFICHAGE CONSOLE
    # ========================================================================
    
    print("\n" + "="*80)
    print("EVENT STUDY: GOOG SPIKE ANALYSIS")
    print("="*80)
    print(f"\nSpike time: t = {spike_time} (~11:33 AM)")
    print(f"Analysis window: ±{window_narrow} obs (±{window_narrow * 0.5 / 60:.1f} min)")
    
    if regime_at_spike is not None:
        print(f"Regime at spike: {regime_at_spike}")
    
    print("\nCross-ticker OFI reallocation:")
    print(results_df.to_string(index=False))
    
    print(f"\n✓ MSFT absorption: {msft_absorption_pct:.1f}% of total reallocation")
    print(f"✓ Figure saved: {figure_path.name}")
    print(f"✓ CSV saved: {csv_path.name}")
    print("="*80)
    
    # ========================================================================
    # RETOUR DES RÉSULTATS
    # ========================================================================
    
    return {
        'delta_ofi': delta_ofi,
        'absorption_pct': absorption_pct,
        'msft_absorption_pct': msft_absorption_pct,
        'regime_at_spike': regime_at_spike,
        'goog_loss': goog_loss,
        'total_reallocation': total_reallocation,
        'figure_path': str(figure_path),
        'csv_path': str(csv_path),
        'results_df': results_df
    }


def compute_event_summary_statistics(
    synced_data: Dict[str, pd.DataFrame],
    spike_time: int,
    window: int,
    tickers: list
) -> pd.DataFrame:
    """
    Calcule les statistiques récapitulatives pour l'event study.
    
    Parameters
    ----------
    synced_data : dict
        Données synchronisées
    spike_time : int
        Index temporel du spike
    window : int
        Taille de fenêtre (observations)
    tickers : list
        Liste des tickers
    
    Returns
    -------
    pd.DataFrame
        Statistiques par ticker (mean, std, min, max avant/après)
    """
    
    stats = []
    
    for ticker in tickers:
        ofi = synced_data[ticker]['ofi']
        
        ofi_before = ofi.iloc[spike_time - window : spike_time]
        ofi_after = ofi.iloc[spike_time : spike_time + window]
        
        stats.append({
            'Ticker': ticker,
            'Mean_Before': ofi_before.mean(),
            'Std_Before': ofi_before.std(),
            'Min_Before': ofi_before.min(),
            'Max_Before': ofi_before.max(),
            'Mean_After': ofi_after.mean(),
            'Std_After': ofi_after.std(),
            'Min_After': ofi_after.min(),
            'Max_After': ofi_after.max(),
            'Delta_Mean': ofi_after.mean() - ofi_before.mean(),
            'Delta_Std': ofi_after.std() - ofi_before.std()
        })
    
    return pd.DataFrame(stats)
