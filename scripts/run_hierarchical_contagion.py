# -*- coding: latin-1 -*-
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

Changes vs. the previous pipeline:
----------------------------------
1. GARCH -> MAD normalization (more robust to outliers)
2. Labels -> state probabilities (continuous variables)
3. Single HMM -> hierarchical HMM (local + global)
4. Price correlation -> regime correlation
5. Simple lead-lag -> Transfer Entropy (directed causality)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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
from src.features.mad_normalizer import normalize_innovations_mad  # NOUVEAU : MAD au lieu de GARCH
from src.features.wasserstein import (
    compute_wasserstein_temporal_features,
    _compute_temporal_wasserstein_series,
)
from src.models.hmm_optimal import fit_optimized_hmm_with_probs  # NOUVEAU : avec probabilités
from src.models.meta_hmm import MetaHMM, fit_hierarchical_hmm_pipeline  # NOUVEAU : Méta-HMM
from src.analysis.contagion_metrics import (  # NOUVEAU : Métriques de contagion
    compute_transfer_entropy_matrix,
    compute_transfer_entropy_matrix_significance,
    compute_regime_correlation,
    identify_patient_zero,
    visualize_contagion_network
)
from src.analysis.leadlag import (
    analyze_multimetric_leadlag_by_model,
    analyze_interticker_leadlag_by_metric_quantile,
)
from src.analysis.event_study import analyze_event_goog_spike
from src.visualization.regime_plots import plot_regime_statistics


PAPER_DIR = Path("paper")
PAPER_FIGURES_DIR = PAPER_DIR / "figures"
PAPER_TABLES_DIR = PAPER_DIR / "tables"
PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)


def _save_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    """Write a DataFrame to a LaTeX table file with caption and label."""
    df.to_latex(path, index=False, caption=caption, label=label, escape=False)


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


def _compute_mmd_series(series: np.ndarray, states: np.ndarray) -> pd.DataFrame:
    """Compute MMD statistics per regime for a given series."""
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


def _plot_feature_by_regime(df: pd.DataFrame, states: np.ndarray, title: str, path: Path) -> None:
    """Plot feature distributions by regime and save."""
    plot_df = df.copy()
    plot_df["regime"] = states[: len(plot_df)]
    melted = plot_df.melt(id_vars="regime", var_name="feature", value_name="value")
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=melted, x="feature", y="value", hue="regime", showfliers=False)
    plt.title(title)
    plt.xlabel("Feature")
    plt.ylabel("Value")
    plt.legend(title="Regime", loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def _plot_regime_characteristics(df: pd.DataFrame,states: np.ndarray,title: str,path: Path,) -> pd.DataFrame:
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
    plt.ylabel("Mean (±std)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return stats_df


def _compute_leadlag_significant(source: np.ndarray,target: np.ndarray,quantiles: list,max_lag: int,alpha: float,min_obs: int,) -> pd.DataFrame:
    """Compute significant lead-lag relationships with p-values."""
    lags = np.arange(-max_lag, max_lag + 1)
    rows = []

    for q in quantiles:
        threshold = np.percentile(source, q * 100)
        if q < 0.5:
            mask = source <= threshold
            q_label = f"Q{int(q*100)}"
        else:
            mask = source >= threshold
            q_label = f"Q{int(q*100)}"

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
        plt.plot(
            sub["lag_seconds"],
            sub["correlation"],
            marker="o",
            label=q_label,
            alpha=0.7,
        )
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

print("="*80)
print("PIPELINE HIERARCHIQUE DE DETECTION DE CONTAGION")
print("="*80)
print(f"\nDate d'analyse: {ANALYSIS_DATE}")
print(f"Tickers: {', '.join(TICKERS)}")
print(f"Regimes locaux: {N_REGIMES}")
print(f"Regimes globaux: {N_REGIMES}")
print(f"\nNOUVEAUTES :")
print(f"  - Normalisation MAD (robuste) au lieu de GARCH")
print(f"  - HMM hierarchique (local + global)")
print(f"  - Transfer Entropy pour causalite dirigee")
print(f"  - Identification du Patient Zero")


# ============================================================================
# ÉTAPE 0 : PRÉPARATION DES DONNÉES
# ============================================================================
print("\n" + "="*80)
print("ÉTAPE 0 : PRÉPARATION DES DONNÉES")
print("="*80)

print("\n[1/3] Chargement des données LOBSTER...")
synced_data = load_all_tickers(TICKERS, ANALYSIS_DATE, RAW_DATA_DIR)
print(f"✓ {len(synced_data[TICKERS[0]])} observations par ticker")

print("\n[2/3] Normalisation MAD (fenêtre glissante robuste)...")
# NOUVEAU : MAD au lieu de GARCH
innov_dict = normalize_innovations_mad(
    synced_data,
    TICKERS,
    window=WASSERSTEIN_WINDOW,  # Fenêtre de normalisation
    min_periods=max(50, WASSERSTEIN_WINDOW//2)
)
print(f"✓ Innovations normalisées pour {len(innov_dict)} séries")

print("\n[3/3] Calcul des distances de Wasserstein (temporal, avant vs apres)...")

# Use per-ticker optimized parameters if available (CSV is the single source of truth)
per_ticker_params_csv = RESULTS_DIR / "best_parameters_hierarchical_per_ticker.csv"
use_per_ticker = per_ticker_params_csv.exists()
per_ticker_params = None

# Optional global params (for meta-HMM)
best_global_params_path = RESULTS_DIR / 'best_parameters_hierarchical.txt'
best_global_params = {}
if best_global_params_path.exists():
    with open(best_global_params_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '=' in line and not line.strip().startswith('#'):
                key, val = line.split('=', 1)
                best_global_params[key.strip()] = val.strip()

# Optional direct-global params (for direct HMM)
best_direct_params = {}
best_direct_path = RESULTS_DIR / "optimization_global_direct.csv"
if best_direct_path.exists():
    try:
        best_direct_df = pd.read_csv(best_direct_path)
        if "ari_direct" in best_direct_df.columns:
            best_direct_df = best_direct_df.sort_values("ari_direct", ascending=False)
        best_direct_row = best_direct_df.iloc[0]
        best_direct_params = {
            "global_persistence": float(best_direct_row.get("global_persistence")),
            "global_smoothing": int(best_direct_row.get("global_smoothing")),
        }
        print(f"Parameters direct-global loaded: {best_direct_path}")
    except Exception:
        best_direct_params = {}

if use_per_ticker:
    per_ticker_params = pd.read_csv(per_ticker_params_csv)
    print(f"Parameters per ticker loaded: {per_ticker_params_csv}")
else:
    print("Parametres par ticker non trouves, fallback sur config global")

# Cache innovations/features for required windows
mad_windows = (
    sorted(per_ticker_params["mad_window"].unique())
    if use_per_ticker
    else [WASSERSTEIN_WINDOW]
)
wass_windows = (
    sorted(per_ticker_params["wasserstein_window"].unique())
    if use_per_ticker
    else [WASSERSTEIN_WINDOW]
)

innov_cache = {}
for mad_window in mad_windows:
    innov_cache[mad_window] = normalize_innovations_mad(
        synced_data,
        TICKERS,
        window=int(mad_window),
        min_periods=max(50, int(mad_window) // 2),
    )

feature_cache = {}
for mad_window in mad_windows:
    for wass_window in wass_windows:
        wass_X = compute_wasserstein_temporal_features(
            innov_cache[mad_window],
            TICKERS,
            window=int(wass_window),
        )
        feature_cache[(int(mad_window), int(wass_window))] = wass_X

# Build per-ticker feature blocks and align on common index
ticker_feature_blocks = {}
for ticker in TICKERS:
    if use_per_ticker:
        row = per_ticker_params[per_ticker_params["ticker"] == ticker].iloc[0]
        mad_w = int(row["mad_window"])
        wass_w = int(row["wasserstein_window"])
    else:
        mad_w = WASSERSTEIN_WINDOW
        wass_w = WASSERSTEIN_WINDOW
    wass_X = feature_cache[(mad_w, wass_w)]
    cols = [f"{ticker}_Price", f"{ticker}_OFI", f"{ticker}_OBI"]
    cols = [c for c in cols if c in wass_X.columns]
    ticker_feature_blocks[ticker] = wass_X[cols].copy()

# Align all tickers on common index
common_index = None
for df in ticker_feature_blocks.values():
    common_index = df.index if common_index is None else common_index.intersection(df.index)

for t in list(ticker_feature_blocks.keys()):
    ticker_feature_blocks[t] = ticker_feature_blocks[t].loc[common_index]

wass_X_all = pd.concat(ticker_feature_blocks.values(), axis=1)
print(f"Matrice Wasserstein (temporal) : {wass_X_all.shape}")
output_file = RESULTS_DIR / "hierarchical_temporal_features.csv"
wass_X_all.to_csv(output_file, index=True)
print(f"Temporal features sauvegardees dans {output_file}")

# Build decomposed dict for lead-lag (metric -> ticker -> list)
wass_X_decomposed = {m: {t: [] for t in TICKERS} for m in ["price_ret", "obi", "ofi"]}
for ticker in TICKERS:
    df = ticker_feature_blocks[ticker]
    if f"{ticker}_Price" in df.columns:
        wass_X_decomposed["price_ret"][ticker] = df[f"{ticker}_Price"].values.tolist()
    if f"{ticker}_OBI" in df.columns:
        wass_X_decomposed["obi"][ticker] = df[f"{ticker}_OBI"].values.tolist()
    if f"{ticker}_OFI" in df.columns:
        wass_X_decomposed["ofi"][ticker] = df[f"{ticker}_OFI"].values.tolist()

# ÉTAPE 1 : HMM LOCAUX (PAR ACTIF) → PROBABILITÉS D'ÉTAT
# ============================================================================
print("\n" + "="*80)
print("ÉTAPE 1 : HMM LOCAUX (NIVEAU 1 - PAR ACTIF)")
print("="*80)
print("Objectif : Extraire P(régime | actif) pour chaque actif\n")

# Sélection des métriques (ajout de Price pour capter la dynamique prix)
selected_metrics = ['Price', 'OFI', 'OBI']
print(f"Métriques sélectionnées : {selected_metrics}")

local_models = {}
local_states = {}
local_state_probs = {}  # NOUVEAU : Probabilités au lieu de labels seulement

for ticker in TICKERS:
    # Per-ticker params if available
    if use_per_ticker:
        row = per_ticker_params[per_ticker_params['ticker'] == ticker].iloc[0]
        local_persist = float(row['local_persistence'])
        local_smooth = int(row['local_smoothing'])
        local_regimes = int(row['n_regimes'])
    else:
        local_persist = HMM_PERSISTENCE_LOCAL
        local_smooth = HMM_SMOOTHING_LOCAL
        local_regimes = N_REGIMES
    print(f"\n[{ticker}] Fitting HMM local...")

    # Sélection des colonnes pour ce ticker
    ticker_cols = []
    for metric in selected_metrics:
        col_name = f'{ticker}_{metric}'
        if col_name in wass_X_all.columns:
            ticker_cols.append(col_name)

    if len(ticker_cols) == 0:
        print(f"  ⚠ Aucune colonne trouvée pour {ticker}, skip")
        continue

    wass_X_ticker = wass_X_all[ticker_cols]

    # Fallback covariance: switch to 'full' if high correlation between features
    covariance_type = "diag"
    if wass_X_ticker.shape[1] > 1:
        corr = wass_X_ticker.corr().abs()
        max_corr = corr.where(~np.eye(corr.shape[0], dtype=bool)).max().max()
        mean_corr = corr.where(~np.eye(corr.shape[0], dtype=bool)).mean().mean()
        if pd.notna(mean_corr):
            print(f"  -> Mean |corr|={mean_corr:.2f} (max |corr|={max_corr:.2f})")
        if pd.notna(max_corr) and max_corr >= HMM_COV_FULL_CORR_THRESHOLD:
            covariance_type = "full"
            print(
                f"  -> High corr detected (>= {HMM_COV_FULL_CORR_THRESHOLD:.2f}), "
                "using covariance='full'"
            )

    # Fit HMM avec extraction des probabilités
    # NOUVEAU : fit_optimized_hmm_with_probs au lieu de fit_optimized_hmm
    model, states, state_probs = fit_optimized_hmm_with_probs(
        wass_X_ticker,
        n_components=local_regimes,
        persistence=local_persist,
        smooth_window=local_smooth,
        covariance_type=covariance_type,
    )

    local_models[ticker] = model
    local_states[ticker] = states
    local_state_probs[ticker] = state_probs  # Shape: (n_obs, n_regimes)

    print(f"  ✓ Probabilités extraites : shape = {state_probs.shape}")

# Sauvegarde des états locaux
states_local_df = pd.DataFrame({
    'timestamp': wass_X_all.index
})
for ticker in TICKERS:
    # Per-ticker params if available
    if use_per_ticker:
        row = per_ticker_params[per_ticker_params['ticker'] == ticker].iloc[0]
        local_persist = float(row['local_persistence'])
        local_smooth = int(row['local_smoothing'])
        local_regimes = int(row['n_regimes'])
    else:
        local_persist = HMM_PERSISTENCE_LOCAL
        local_smooth = HMM_SMOOTHING_LOCAL
        local_regimes = N_REGIMES
    if ticker in local_states:
        states_local_df[f'state_{ticker}'] = local_states[ticker]

output_file = RESULTS_DIR / 'hierarchical_states_local.csv'
states_local_df.to_csv(output_file, index=False)
print(f"\n✓ États locaux sauvegardés dans {output_file}")


# ============================================================================
# ÉTAPE 2 : MÉTA-HMM GLOBAL → RÉGIMES SECTORIELS
# ============================================================================
print("\n" + "="*80)
print("ÉTAPE 2 : MÉTA-HMM GLOBAL (NIVEAU 2 - SECTORIEL)")
print("="*80)
print("Objectif : Détecter régimes sectoriels à partir des probas locales\n")

# Pipeline complet hiérarchique
# NOUVEAU : fit_hierarchical_hmm_pipeline
meta_hmm, global_states, global_probs, sync_df = fit_hierarchical_hmm_pipeline(
    local_state_probs=local_state_probs,
    local_states=local_states,
    tickers=TICKERS,
    n_global_regimes=N_REGIMES,
    persistence=float(best_global_params.get('HMM_PERSISTENCE_GLOBAL', HMM_PERSISTENCE_GLOBAL)),
    smooth_window=int(best_global_params.get('HMM_SMOOTHING_GLOBAL', HMM_SMOOTHING_GLOBAL)),
)

# Diagnostics: global probabilities + synchronization strength
print("\n" + "="*80)
print("DIAGNOSTICS : PROBAS GLOBALES & SYNCHRONISATION")
print("="*80)

# 1) Global probabilities "flatness"
global_prob_max = global_probs.max(axis=1)
global_prob_entropy = -np.sum(
    global_probs * np.log(global_probs + 1e-12), axis=1
)
entropy_max = np.log(global_probs.shape[1])
entropy_ratio = float(np.mean(global_prob_entropy) / entropy_max)

print(f"Global probs: mean(max P) = {global_prob_max.mean():.3f}")
print(
    "Global probs: mean entropy = "
    f"{global_prob_entropy.mean():.3f} (ratio {entropy_ratio:.3f} of max)"
)
if entropy_ratio > 0.90:
    print("  -> Warning: probabilities are very flat (weak global signal).")
elif entropy_ratio > 0.75:
    print("  -> Note: probabilities are quite flat (moderate global signal).")

# 2) Synchronization strength
sync_mean = float(sync_df["sync_rate"].mean())
sync_median = float(sync_df["sync_rate"].median())
print(f"Sync mean = {sync_mean:.3f}, median = {sync_median:.3f}")
if sync_mean < 0.10:
    print("  -> Warning: low synchronization (global signal likely weak).")

# Sauvegarde
states_global_df = pd.DataFrame({
    'timestamp': wass_X_all.index,
    'global_state': global_states
})

# Ajouter les probas globales
for i in range(global_probs.shape[1]):
    states_global_df[f'global_prob_regime_{i}'] = global_probs[:, i]

output_file = RESULTS_DIR / 'hierarchical_states_global.csv'
states_global_df.to_csv(output_file, index=False)
print(f"\n✓ États globaux sauvegardés dans {output_file}")

# Sauvegarde de la synchronisation
output_file = RESULTS_DIR / 'hierarchical_synchronization.csv'
sync_df.to_csv(output_file, index=False)
print(f"✓ Synchronisation sauvegardée dans {output_file}")


# ============================================================================
# HMM GLOBAL DIRECT (SUR FEATURES WASSERSTEIN)
# ============================================================================
print("\n" + "="*80)
print("HMM GLOBAL DIRECT (WASSERSTEIN GLOBAL)")
print("="*80)

direct_persist = float(
    best_direct_params.get("global_persistence", best_global_params.get("HMM_PERSISTENCE_GLOBAL", HMM_PERSISTENCE_GLOBAL))
)
direct_smooth = int(
    best_direct_params.get("global_smoothing", best_global_params.get("HMM_SMOOTHING_GLOBAL", HMM_SMOOTHING_GLOBAL))
)
global_direct_model, global_direct_states, global_direct_probs = fit_optimized_hmm_with_probs(
    wass_X_all,
    n_components=N_REGIMES,
    persistence=direct_persist,
    smooth_window=direct_smooth,
    covariance_type="diag",
)

states_global_direct_df = pd.DataFrame(
    {
        "timestamp": wass_X_all.index,
        "global_direct_state": global_direct_states,
    }
)
for i in range(global_direct_probs.shape[1]):
    states_global_direct_df[f"global_direct_prob_regime_{i}"] = global_direct_probs[:, i]

output_file = RESULTS_DIR / "hierarchical_states_global_direct.csv"
states_global_direct_df.to_csv(output_file, index=False)
print(f"✓ États globaux (direct) sauvegardés dans {output_file}")

# Temporal Wasserstein on global stress probability (for lead-lag comparison)
if global_probs.shape[1] == 3:
    global_stress = global_probs[:, 1] + global_probs[:, 2]
else:
    global_stress = global_probs[:, -1]

global_wass_temporal = _compute_temporal_wasserstein_series(
    global_stress,
    window=WASSERSTEIN_WINDOW,
)
global_temporal_index = wass_X_all.index[WASSERSTEIN_WINDOW:-WASSERSTEIN_WINDOW]
global_temporal_df = pd.DataFrame(
    {"global_stress_wass_temporal": global_wass_temporal},
    index=global_temporal_index,
)
output_file = RESULTS_DIR / 'hierarchical_global_temporal_wass.csv'
global_temporal_df.to_csv(output_file, index=True)
print(f"âœ“ Global temporal Wasserstein sauvegardÃ© dans {output_file}")

# Lead-lag local vs global based on temporal Wasserstein series
print("\n" + "="*80)
print("LEAD-LAG LOCAL VS GLOBAL (TEMPORAL WASSERSTEIN)")
print("="*80)

def _leadlag_corr(series_local: np.ndarray, series_global: np.ndarray, max_lag: int):
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
            x = series_local
            y = series_global
        if len(x) < 10:
            corrs.append(np.nan)
        else:
            corrs.append(stats.spearmanr(x, y, nan_policy="omit").correlation)
    return np.array(list(lags)), np.array(corrs)


def _leadlag_pvalues(series_local: np.ndarray, series_global: np.ndarray, lags: np.ndarray):
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
            x = series_local
            y = series_global
        if len(x) < 10:
            pvals.append(np.nan)
        else:
            pvals.append(stats.spearmanr(x, y, nan_policy="omit").pvalue)
    return np.array(pvals)


def _best_leadlag(series_a: np.ndarray, series_b: np.ndarray, max_lag: int, alpha: float, min_obs: int):
    """Select the best significant lag given alpha and minimum observations."""
    lags, corrs = _leadlag_corr(series_a, series_b, max_lag)
    pvals = _leadlag_pvalues(series_a, series_b, lags)
    sig_mask = (pvals < alpha) & np.isfinite(corrs)
    if not np.any(sig_mask):
        return None
    sig_corrs = corrs[sig_mask]
    sig_lags = lags[sig_mask]
    sig_pvals = pvals[sig_mask]
    best_idx = int(np.nanargmax(np.abs(sig_corrs)))
    return {
        "best_lag_obs": int(sig_lags[best_idx]),
        "best_lag_seconds": float(sig_lags[best_idx] * 0.5),
        "best_corr": float(sig_corrs[best_idx]),
        "best_pval": float(sig_pvals[best_idx]),
    }


leadlag_rows = []
heatmap_rows = []
global_series = global_temporal_df["global_stress_wass_temporal"].values
alpha_threshold = 0.05
min_obs = 30

base_n = min(len(global_series), len(global_states))
global_series = global_series[:base_n]
global_states_aligned = global_states[:base_n]

for ticker in TICKERS:
    cols = [c for c in wass_X_all.columns if c.startswith(f"{ticker}_")]
    if not cols:
        continue
    local_series = wass_X_all[cols].mean(axis=1).values
    n = min(len(local_series), base_n)
    local_series = local_series[:n]
    global_series_aligned = global_series[:n]
    global_states_local = global_states_aligned[:n]

    for regime in np.unique(global_states_local):
        regime_mask = global_states_local == regime
        if regime_mask.sum() < min_obs:
            continue
        for q in LEADLAG_QUANTILES:
            threshold = np.percentile(local_series[regime_mask], q * 100)
            if q < 0.5:
                q_mask = local_series <= threshold
                q_label = f"Q{int(q*100)}"
            else:
                q_mask = local_series >= threshold
                q_label = f"Q{int(q*100)}"

            mask = regime_mask & q_mask
            if mask.sum() < min_obs:
                continue

            best = _best_leadlag(
                local_series[mask],
                global_series_aligned[mask],
                LEADLAG_MAX_LAG,
                alpha_threshold,
                min_obs,
            )
            if best is None:
                continue

            leadlag_rows.append(
                {
                    "ticker": ticker,
                    "global_regime": int(regime),
                    "quantile": q_label,
                    **best,
                    "n_obs": int(mask.sum()),
                }
            )

            heatmap_rows.append(
                {
                    "ticker": ticker,
                    "global_regime": int(regime),
                    "quantile": q_label,
                    "best_lag_seconds": best["best_lag_seconds"],
                    "best_corr": best["best_corr"],
                }
            )

leadlag_df = pd.DataFrame(leadlag_rows).sort_values(
    ["ticker", "global_regime", "quantile", "best_pval"], ascending=[True, True, True, True]
)
output_file = RESULTS_DIR / "hierarchical_leadlag_local_vs_global_quantile.csv"
leadlag_df.to_csv(output_file, index=False)
print(f"Lead-lag local vs global (quantile+regime) sauvegarde dans {output_file}")

heatmap_df = pd.DataFrame(heatmap_rows)
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
            sub = heatmap_df[(heatmap_df["global_regime"] == regime) & (heatmap_df["quantile"] == q_label)]
            if sub.empty:
                continue
            pivot = sub.pivot(index="ticker", columns="quantile", values="best_lag_seconds")
            plt.figure(figsize=(6, 4))
            sns.heatmap(pivot, cmap="coolwarm", center=0, annot=True, fmt=".1f")
            plt.title(f"Lead-lag local->global (Regime {regime}, {q_label})")
            plt.xlabel("Quantile")
            plt.ylabel("Ticker")
            output_file = RESULTS_DIR / f"hierarchical_leadlag_local_vs_global_R{regime}_{q_label}.png"
            plt.tight_layout()
            plt.savefig(output_file, dpi=300)
            plt.close()


# ============================================================================
# LEAD-LAG ENTRE TICKERS (TEMPORAL WASSERSTEIN)
# ============================================================================
print("\n" + "="*80)
print("LEAD-LAG ENTRE TICKERS (TEMPORAL WASSERSTEIN)")
print("="*80)
pair_rows = []
series_by_ticker = {
    t: wass_X_all[[c for c in wass_X_all.columns if c.startswith(f"{t}_")]].mean(axis=1).values
    for t in TICKERS
}

tickers_list = list(series_by_ticker.keys())
alpha_threshold = 0.05
min_obs = 30

for q in LEADLAG_QUANTILES:
    q_label = f"Q{int(q*100)}"
    heatmap = np.full((len(tickers_list), len(tickers_list)), np.nan, dtype=float)

    for i, t1 in enumerate(tickers_list):
        s1 = series_by_ticker[t1]
        threshold = np.percentile(s1, q * 100)
        if q < 0.5:
            mask = s1 <= threshold
        else:
            mask = s1 >= threshold

        s1_sub = s1[mask]
        for j, t2 in enumerate(tickers_list):
            if t1 == t2:
                continue
            s2 = series_by_ticker[t2][mask]
            if len(s1_sub) < min_obs:
                continue

            best = _best_leadlag(s1_sub, s2, LEADLAG_MAX_LAG, alpha_threshold, min_obs)
            if best is None:
                continue

            pair_rows.append(
                {
                    "quantile": q_label,
                    "ticker1": t1,
                    "ticker2": t2,
                    **best,
                    "n_obs": int(len(s1_sub)),
                }
            )
            heatmap[i, j] = best["best_corr"]

    if "heatmaps_by_q" not in locals():
        heatmaps_by_q = {}
        scores_by_q = {}
    heatmaps_by_q[q_label] = heatmap
    scores_by_q[q_label] = (
        float(np.nanmax(np.abs(heatmap))) if np.isfinite(heatmap).any() else -np.inf
    )

max_quantile_heatmaps = 2
ranked_q = sorted(scores_by_q.items(), key=lambda x: x[1], reverse=True)
selected_q = [q for q, _ in ranked_q[:max_quantile_heatmaps] if np.isfinite(scores_by_q[q])]
for q_label in selected_q:
    heatmap = heatmaps_by_q[q_label]
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        heatmap,
        cmap="coolwarm",
        center=0,
        xticklabels=tickers_list,
        yticklabels=tickers_list,
        annot=False,
    )
    plt.title(f"Lead-lag inter-tickers (best corr, {q_label})")
    plt.xlabel("Target ticker")
    plt.ylabel("Source ticker")
    output_file = RESULTS_DIR / f"hierarchical_leadlag_between_tickers_{q_label}.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Heatmap lead-lag inter-tickers sauvegardee dans {output_file}")

leadlag_pairs_df = pd.DataFrame(pair_rows).sort_values(
    ["quantile", "best_pval", "best_corr"], ascending=[True, True, False]
)
output_file = RESULTS_DIR / "hierarchical_leadlag_between_tickers_quantile.csv"
leadlag_pairs_df.to_csv(output_file, index=False)
print(f"Lead-lag ticker-ticker (quantiles) sauvegarde dans {output_file}")


# ============================================================================
# ETAPE 3 : TRANSFER ENTROPY -> CAUSALITE DIRIGEE
# ============================================================================
print("\n" + "="*80)
print("ETAPE 3 : TRANSFER ENTROPY (CAUSALITE DIRIGEE)")
print("="*80)
print("Objectif : Mesurer qui cause qui (information dirigee)\n")

# Calcul de la matrice de Transfer Entropy
# NOUVEAU : compute_transfer_entropy_matrix_significance
k_grid = list(range(1, 11))
te_matrix, te_k_summary = compute_transfer_entropy_matrix_significance(
    local_state_probs,
    TICKERS,
    k_grid=k_grid,
    bins=10,
    n_surrogates=100,
    block_size=30,
    alpha=0.05,
)

# Sauvegarde
output_file = RESULTS_DIR / 'hierarchical_transfer_entropy.csv'
te_matrix.to_csv(output_file)
print(f"\nOK Matrice TE sauvegardee dans {output_file}")
output_file = RESULTS_DIR / 'hierarchical_transfer_entropy_k_summary.csv'
te_k_summary.to_csv(output_file, index=False)
print(f"OK Resume k TE sauvegarde dans {output_file}")


# ============================================================================
# ÉTAPE 4 : CORRÉLATION DE RÉGIMES → CO-MOUVEMENTS
# ============================================================================
print("\n" + "="*80)
print("ÉTAPE 4 : CORRÉLATION DE RÉGIMES")
print("="*80)
print("Objectif : Mesurer les co-mouvements de régimes\n")

# Corrélation croisée des probabilités
# NOUVEAU : compute_regime_correlation
regime_corr_df = compute_regime_correlation(
    local_state_probs,
    TICKERS,
    max_lag=10
)

# Sauvegarde
output_file = RESULTS_DIR / 'hierarchical_regime_correlation.csv'
regime_corr_df.to_csv(output_file, index=False)
print(f"\n✓ Corrélations de régimes sauvegardées dans {output_file}")


# ============================================================================
# ÉTAPE 5 : IDENTIFICATION DU "PATIENT ZÉRO"
# ============================================================================
print("\n" + "="*80)
print("ÉTAPE 5 : IDENTIFICATION DU 'PATIENT ZÉRO'")
print("="*80)
print("Objectif : Qui initie la contagion\n")

# NOUVEAU : identify_patient_zero
patient_zero_info = identify_patient_zero(te_matrix, sync_df)

# Sauvegarde
output_file = RESULTS_DIR / 'hierarchical_patient_zero.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("PATIENT ZÉRO DE LA CONTAGION\n")
    f.write("="*50 + "\n\n")
    f.write(f"Actif identifié : {patient_zero_info['patient_zero']}\n")
    f.write(f"Contagion Score : {patient_zero_info['contagion_score']:.3f}\n")
    f.write(f"Transfer Entropy sortante : {patient_zero_info['te_outgoing']:.4f} nats\n")
    f.write(f"Leadership Score : {patient_zero_info['leadership_score']:.3f}\n\n")
    f.write("RANKING COMPLET :\n")
    f.write(patient_zero_info['ranking'][['ticker', 'contagion_score', 'te_outgoing', 'leadership_score']].to_string(index=False))

print(f"\n✓ Patient zéro sauvegardé dans {output_file}")


# ============================================================================
# VISUALISATION 1 : HIÉRARCHIE DES RÉGIMES
# ============================================================================
print("\n" + "="*80)
print("VISUALISATION 1 : HIÉRARCHIE DES RÉGIMES")
print("="*80)

fig = meta_hmm.visualize_regime_agreement(
    local_states,
    global_states,
    TICKERS,
    timestamps=wass_X_all.index,
    save_path=RESULTS_DIR / 'hierarchical_regime_hierarchy.png'
)
plt.close(fig)


# ============================================================================
# VISUALISATION 2 : RÉSEAU DE CONTAGION (TRANSFER ENTROPY)
# ============================================================================
print("\n" + "="*80)
print("VISUALISATION 2 : RÉSEAU DE CONTAGION")
print("="*80)

# NOUVEAU : visualize_contagion_network
fig = visualize_contagion_network(
    te_matrix,
    patient_zero_info,
    save_path=RESULTS_DIR / 'hierarchical_contagion_network.png'
)
if fig:
    plt.close(fig)


# ============================================================================
# VISUALISATION 3 : HEATMAP TRANSFER ENTROPY
# ============================================================================
print("\n" + "="*80)
print("VISUALISATION 3 : HEATMAP TRANSFER ENTROPY")
print("="*80)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(te_matrix, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax,
            cbar_kws={'label': 'Transfer Entropy (nats)'})
ax.set_title('Matrice de Transfer Entropy (Causalité Dirigée)', fontweight='bold', fontsize=12)
ax.set_xlabel('Target (Effet)', fontweight='bold')
ax.set_ylabel('Source (Cause)', fontweight='bold')
plt.tight_layout()
output_file = RESULTS_DIR / 'hierarchical_te_heatmap.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Heatmap TE sauvegardée dans {output_file}")
plt.close()


# ============================================================================
# VISUALISATION 4 : TIMELINE AVEC PROBABILITÉS
# ============================================================================
print("\n" + "="*80)
print("VISUALISATION 4 : TIMELINE DES PROBABILITÉS")
print("="*80)

fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)

time_indices = np.arange(len(global_states))

# 1. Régime global
ax = axes[0]
for regime in range(N_REGIMES):
    ax.plot(time_indices, global_probs[:, regime],
           label=f'Régime Global {regime}', alpha=0.7, linewidth=1)
ax.set_ylabel('P(Régime Global)', fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_title('Timeline des Probabilités de Régimes (Architecture Hiérarchique)',
            fontweight='bold', fontsize=14)

# 2. Probabilités de stress (régime 1+2) par actif
ax = axes[1]
for ticker in TICKERS:
    # Per-ticker params if available
    if use_per_ticker:
        row = per_ticker_params[per_ticker_params['ticker'] == ticker].iloc[0]
        local_persist = float(row['local_persistence'])
        local_smooth = int(row['local_smoothing'])
        local_regimes = int(row['n_regimes'])
    else:
        local_persist = HMM_PERSISTENCE_LOCAL
        local_smooth = HMM_SMOOTHING_LOCAL
        local_regimes = N_REGIMES
    if ticker in local_state_probs:
        if local_state_probs[ticker].shape[1] == 3:
            stress_prob = local_state_probs[ticker][:, 1] + local_state_probs[ticker][:, 2]
        else:
            stress_prob = local_state_probs[ticker][:, -1]
        ax.plot(time_indices, stress_prob, label=ticker, alpha=0.7, linewidth=1)
ax.set_ylabel('P(Stress Local)', fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# 3. Distance Wasserstein moyenne
ax = axes[2]
wass_mean = wass_X_all.mean(axis=1).values
ax.plot(time_indices, wass_mean, linewidth=1, color='purple', alpha=0.7)
ax.fill_between(time_indices, 0, wass_mean, alpha=0.3, color='purple')
ax.set_ylabel('Wasserstein\nmoyen', fontweight='bold')
ax.set_xlabel('Temps (observations)', fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_file = RESULTS_DIR / 'hierarchical_timeline_probabilities.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Timeline sauvegardée dans {output_file}")
plt.close()


# ============================================================================
# VISUALISATION 5 : CONCORDANCE LOCAL vs GLOBAL
# ============================================================================
print("\n" + "="*80)
print("VISUALISATION 5 : CONCORDANCE LOCAL vs GLOBAL")
print("="*80)

fig, axes = plt.subplots(1, len(TICKERS), figsize=(20, 4))
if len(TICKERS) == 1:
    axes = [axes]

for i, (ticker, ax) in enumerate(zip(TICKERS, axes)):
    if ticker not in local_states:
        continue

    # Matrice de concordance
    concordance = np.zeros((N_REGIMES, N_REGIMES))
    for global_r in range(N_REGIMES):
        for local_r in range(N_REGIMES):
            mask = (global_states == global_r) & (local_states[ticker] == local_r)
            concordance[global_r, local_r] = mask.sum()

    # Normalisation
    row_sums = concordance.sum(axis=1, keepdims=True)
    concordance_norm = concordance / (row_sums + 1e-9)

    # Heatmap
    sns.heatmap(concordance_norm, annot=True, fmt='.2f',
                cmap='RdYlGn', vmin=0, vmax=1, ax=ax,
                xticklabels=[f'L{j}' for j in range(N_REGIMES)],
                yticklabels=[f'G{j}' for j in range(N_REGIMES)],
                cbar_kws={'label': 'Probabilité'})
    ax.set_title(f'{ticker}', fontweight='bold', fontsize=12)
    ax.set_xlabel('Régime Local', fontweight='bold')
    if i == 0:
        ax.set_ylabel('Régime Global', fontweight='bold')
    else:
        ax.set_ylabel('')

fig.suptitle('Concordance Régime Global → Régimes Locaux',
            fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
output_file = RESULTS_DIR / 'hierarchical_concordance_matrices.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Matrices de concordance sauvegardées dans {output_file}")
plt.close()


# ============================================================================
# VISUALISATION 6 : LEAD-LAG ANALYSIS (BONUS)
# ============================================================================
print("\n" + "="*80)
print("VISUALISATION 6 : LEAD-LAG ANALYSIS (BONUS)")
print("="*80)

leadlag_sig_df = analyze_multimetric_leadlag_by_model(
    wass_X_decomposed,
    TICKERS,
    max_lag=LEADLAG_MAX_LAG,
    cross_metric_only=True,
    max_pairs_to_plot=6,
)

if leadlag_sig_df is not None and not leadlag_sig_df.empty:
    output_file = RESULTS_DIR / "leadlag_multimetric_quantile_significant.csv"
    leadlag_sig_df.to_csv(output_file, index=False)
    print(f"✓ Lead-lag significatifs sauvegardés dans {output_file}")
else:
    print("⚠️  Aucun lead-lag significatif trouvé pour l'analyse multi-métrique.")

leadlag_ticker_metric_df = analyze_interticker_leadlag_by_metric_quantile(
    wass_X_decomposed,
    TICKERS,
    max_lag=LEADLAG_MAX_LAG,
    max_heatmaps_per_metric=1,
)

if leadlag_ticker_metric_df is not None and not leadlag_ticker_metric_df.empty:
    output_file = RESULTS_DIR / "leadlag_tickers_by_metric_quantile_significant.csv"
    leadlag_ticker_metric_df.to_csv(output_file, index=False)
    print(f"✓ Lead-lag inter-tickers (par métrique/quantile) sauvegardé dans {output_file}")
else:
    print("⚠️  Aucun lead-lag inter-ticker significatif trouvé par métrique/quantile.")


# ============================================================================
# EVENT STUDY : GOOG CRASH (BONUS)
# ============================================================================
print("\n" + "="*80)
print("EVENT STUDY : GOOG CRASH (21 JUIN 2012)")
print("="*80)

event_results = analyze_event_goog_spike(
    synced_data,
    global_states,
    WASSERSTEIN_WINDOW
)

if event_results is not None:
    output_file = RESULTS_DIR / 'hierarchical_event_study_goog.csv'
    if isinstance(event_results, dict) and "results_df" in event_results:
        event_results["results_df"].to_csv(output_file, index=False)
    elif hasattr(event_results, "to_csv"):
        event_results.to_csv(output_file, index=False)
    else:
        print("âš  Event study format unexpected, skipping CSV export")
    print(f"\n✓ Event study sauvegardé dans {output_file}")


# ============================================================================
# REPORTING (TABLES LaTeX + FIGURES)
# ============================================================================
print("\n" + "="*80)
print("REPORTING (TABLES LaTeX + FIGURES)")
print("="*80)

# Regime distribution per ticker
regime_rows = []
for ticker in TICKERS:
    if ticker not in local_states:
        continue
    values, counts = np.unique(local_states[ticker], return_counts=True)
    total = counts.sum()
    for v, c in zip(values, counts):
        regime_rows.append(
            {
                "ticker": ticker,
                "regime": int(v),
                "count": int(c),
                "pct": float(c / total),
            }
        )
regime_dist_df = pd.DataFrame(regime_rows)
_save_latex_table(
    regime_dist_df,
    PAPER_TABLES_DIR / "local_regime_distribution.tex",
    "Répartition des régimes par ticker",
    "tab:local_regime_distribution",
)

# Local HMM parameters table
if use_per_ticker and per_ticker_params is not None:
    params_df = per_ticker_params[
        ["ticker", "mad_window", "wasserstein_window", "local_persistence", "local_smoothing", "n_regimes"]
    ].copy()
else:
    params_df = pd.DataFrame(
        [
            {
                "ticker": t,
                "mad_window": WASSERSTEIN_WINDOW,
                "wasserstein_window": WASSERSTEIN_WINDOW,
                "local_persistence": HMM_PERSISTENCE_LOCAL,
                "local_smoothing": HMM_SMOOTHING_LOCAL,
                "n_regimes": N_REGIMES,
            }
            for t in TICKERS
        ]
    )
_save_latex_table(
    params_df,
    PAPER_TABLES_DIR / "local_hmm_params.tex",
    "Paramètres HMM locaux",
    "tab:local_hmm_params",
)

# Synchronization table
_save_latex_table(
    sync_df,
    PAPER_TABLES_DIR / "local_global_sync.tex",
    "Synchronisation local → global",
    "tab:local_global_sync",
)

# Lead-lag tables
leadlag_top_n = 10
_save_latex_table(
    leadlag_df,
    PAPER_TABLES_DIR / "leadlag_local_global_top.tex",
    "Lead-lag local → global (top)",
    "tab:leadlag_local_global",
)
_save_latex_table(
    leadlag_pairs_df,
    PAPER_TABLES_DIR / "leadlag_between_tickers_top.tex",
    "Lead-lag ticker ↔ ticker (top)",
    "tab:leadlag_between_tickers",
)

# Lead-lag quantiles (per ticker + global) - strongest significant per pair
if "leadlag_sig_df" in globals() and leadlag_sig_df is not None and not leadlag_sig_df.empty:
    _save_latex_table(
        leadlag_sig_df,
        PAPER_TABLES_DIR / "leadlag_multimetric_quantile_significant.tex",
        "Lead-lag significatifs par quantile (ticker + global)",
        "tab:leadlag_quantile_sig",
    )
if "leadlag_ticker_metric_df" in globals() and leadlag_ticker_metric_df is not None and not leadlag_ticker_metric_df.empty:
    _save_latex_table(
        leadlag_ticker_metric_df,
        PAPER_TABLES_DIR / "leadlag_tickers_metric_quantile_significant.tex",
        "Lead-lag inter-tickers par métrique et quantile (significatifs)",
        "tab:leadlag_ticker_metric_sig",
    )

# Transfer Entropy k summary
if 'te_k_summary' in globals() and te_k_summary is not None and not te_k_summary.empty:
    _save_latex_table(
        te_k_summary,
        PAPER_TABLES_DIR / 'transfer_entropy_k_summary.tex',
        'Transfer Entropy k selection summary',
        'tab:transfer_entropy_k_summary',
    )

# Transfer Entropy top N
te_long = (
    te_matrix.stack()
    .reset_index()
    .rename(columns={"level_0": "source", "level_1": "target", 0: "te"})
    .sort_values("te", ascending=False)
)
_save_latex_table(
    te_long.head(leadlag_top_n),
    PAPER_TABLES_DIR / "transfer_entropy_top.tex",
    "Transfer Entropy (top)",
    "tab:transfer_entropy",
)

# MMD diagnostics (local + global)
mmd_rows = []
for ticker in TICKERS:
    if ticker not in local_states:
        continue
    for metric in ["Price", "OFI", "OBI"]:
        col = f"{ticker}_{metric}"
        if col not in wass_X_all.columns:
            continue
        series = wass_X_all[col].values
        states = local_states[ticker][: len(series)]
        mmd_df = _compute_mmd_series(series, states)
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
    cols = [c for c in wass_X_all.columns if c.endswith(f"_{metric}")]
    if not cols:
        continue
    series = wass_X_all[cols].mean(axis=1).values
    meta_mmd_df = _compute_mmd_series(series, global_states[: len(series)])
    direct_mmd_df = _compute_mmd_series(series, global_direct_states[: len(series)])
    global_mmd_rows.append(
        {
            "metric": metric,
            "mmd_meta_mean": float(meta_mmd_df["mmd_r0_r1"].mean()),
            "mmd_direct_mean": float(direct_mmd_df["mmd_r0_r1"].mean()),
        }
    )

mmd_local_df = pd.DataFrame(mmd_rows)
mmd_global_df = pd.DataFrame(global_mmd_rows)
_save_latex_table(
    mmd_local_df,
    PAPER_TABLES_DIR / "mmd_local.tex",
    "MMD local (par ticker/métrique)",
    "tab:mmd_local",
)
_save_latex_table(
    mmd_global_df,
    PAPER_TABLES_DIR / "mmd_global.tex",
    "MMD global (méta vs direct)",
    "tab:mmd_global",
)

# ARI diagnostics and comparisons
def _kmeans_labels(X: np.ndarray, n_clusters: int) -> np.ndarray:
    """Fit KMeans and return cluster labels."""
    Xs = StandardScaler().fit_transform(X)
    return KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit_predict(Xs)

# Build meta input matrix
prob_arrays = [local_state_probs[t] for t in TICKERS if t in local_state_probs]
min_len = min(arr.shape[0] for arr in prob_arrays)
prob_arrays = [arr[:min_len] for arr in prob_arrays]
X_meta = np.hstack(prob_arrays)

meta_kmeans_labels = _kmeans_labels(X_meta, N_REGIMES)
direct_kmeans_labels = _kmeans_labels(wass_X_all.values, N_REGIMES)

ari_meta_kmeans = adjusted_rand_score(global_states[:min_len], meta_kmeans_labels)
ari_direct_kmeans = adjusted_rand_score(global_direct_states[: len(direct_kmeans_labels)], direct_kmeans_labels)
ari_meta_direct = adjusted_rand_score(
    global_states[: min(len(global_states), len(global_direct_states))],
    global_direct_states[: min(len(global_states), len(global_direct_states))],
)

ari_local_rows = []
for ticker in TICKERS:
    if ticker not in local_states:
        continue
    n = min(len(global_states), len(local_states[ticker]))
    ari_local_rows.append(
        {
            "ticker": ticker,
            "ari_meta_vs_local": adjusted_rand_score(global_states[:n], local_states[ticker][:n]),
        }
    )
ari_local_df = pd.DataFrame(ari_local_rows)

ari_summary_df = pd.DataFrame(
    [
        {"comparison": "meta_vs_kmeans", "ari": ari_meta_kmeans},
        {"comparison": "direct_vs_kmeans", "ari": ari_direct_kmeans},
        {"comparison": "meta_vs_direct", "ari": ari_meta_direct},
    ]
)

_save_latex_table(
    ari_summary_df,
    PAPER_TABLES_DIR / "ari_global_comparisons.tex",
    "ARI diagnostics (globaux)",
    "tab:ari_global",
)
_save_latex_table(
    ari_local_df,
    PAPER_TABLES_DIR / "ari_meta_vs_local.tex",
    "ARI méta vs locaux",
    "tab:ari_meta_local",
)

# Entropy and synchronization comparisons (meta vs direct)
meta_entropy = -np.sum(global_probs * np.log(global_probs + 1e-12), axis=1)
direct_entropy = -np.sum(global_direct_probs * np.log(global_direct_probs + 1e-12), axis=1)
entropy_df = pd.DataFrame(
    [
        {"model": "meta", "entropy_mean": float(meta_entropy.mean())},
        {"model": "direct", "entropy_mean": float(direct_entropy.mean())},
    ]
)
_save_latex_table(
    entropy_df,
    PAPER_TABLES_DIR / "entropy_global.tex",
    "Entropie moyenne des probas globales",
    "tab:entropy_global",
)

sync_meta_direct = float(
    np.mean(
        global_states[: min(len(global_states), len(global_direct_states))]
        == global_direct_states[: min(len(global_states), len(global_direct_states))]
    )
)
sync_compare_df = pd.DataFrame(
    [
        {"comparison": "meta_vs_direct_state_sync", "sync_rate": sync_meta_direct},
    ]
)
_save_latex_table(
    sync_compare_df,
    PAPER_TABLES_DIR / "sync_global_comparison.tex",
    "Synchronisation méta vs direct",
    "tab:sync_global_comparison",
)

# FIGURES: local HMMs
for ticker in TICKERS:
    if ticker not in local_states:
        continue
    ticker_cols = [c for c in wass_X_all.columns if c.startswith(f"{ticker}_")]
    wass_X_ticker = wass_X_all[ticker_cols]
    _plot_state_timeline(
        local_states[ticker],
        f"Local HMM Timeline - {ticker}",
        PAPER_FIGURES_DIR / f"hmm_local_{ticker}_timeline.png",
    )
    _plot_state_hist(
        local_states[ticker],
        f"Local HMM Regime Histogram - {ticker}",
        PAPER_FIGURES_DIR / f"hmm_local_{ticker}_hist.png",
    )
    _plot_feature_by_regime(
        wass_X_ticker,
        local_states[ticker],
        f"Local HMM Features by Regime - {ticker}",
        PAPER_FIGURES_DIR / f"hmm_local_{ticker}_features.png",
    )

# FIGURES: global meta and direct
_plot_state_timeline(
    global_states,
    "Meta-HMM Timeline (Global)",
    PAPER_FIGURES_DIR / "hmm_meta_timeline.png",
)
_plot_state_hist(
    global_states,
    "Meta-HMM Regime Histogram (Global)",
    PAPER_FIGURES_DIR / "hmm_meta_hist.png",
)
_plot_feature_by_regime(
    pd.DataFrame(X_meta, columns=[f"p_{i}" for i in range(X_meta.shape[1])]),
    global_states[: len(X_meta)],
    "Meta-HMM Features by Regime (Global)",
    PAPER_FIGURES_DIR / "hmm_meta_features.png",
)

_plot_state_timeline(
    global_direct_states,
    "Direct Global HMM Timeline",
    PAPER_FIGURES_DIR / "hmm_direct_timeline.png",
)
_plot_state_hist(
    global_direct_states,
    "Direct Global HMM Regime Histogram",
    PAPER_FIGURES_DIR / "hmm_direct_hist.png",
)
_plot_feature_by_regime(
    wass_X_all,
    global_direct_states,
    "Direct Global HMM Features by Regime",
    PAPER_FIGURES_DIR / "hmm_direct_features.png",
)

# FIGURES: synchronization and entropy
plt.figure(figsize=(8, 4))
sns.barplot(data=sync_df, x="ticker", y="sync_rate")
plt.title("Synchronisation local → global")
plt.tight_layout()
plt.savefig(PAPER_FIGURES_DIR / "sync_local_global.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 4))
sns.kdeplot(meta_entropy, label="meta")
sns.kdeplot(direct_entropy, label="direct")
plt.title("Distribution de l'entropie (global)")
plt.legend()
plt.tight_layout()
plt.savefig(PAPER_FIGURES_DIR / "entropy_global.png", dpi=300)
plt.close()

# Temporal regime comparison (local vs meta vs direct)
for ticker in TICKERS:
    if ticker not in local_states:
        continue
    n = min(len(local_states[ticker]), len(global_states), len(global_direct_states))
    fig, axes = plt.subplots(3, 1, figsize=(12, 5), sharex=True)
    axes[0].plot(local_states[ticker][:n], linewidth=0.8)
    axes[0].set_ylabel(f"{ticker} local")
    axes[1].plot(global_states[:n], linewidth=0.8, color="tab:orange")
    axes[1].set_ylabel("meta")
    axes[2].plot(global_direct_states[:n], linewidth=0.8, color="tab:green")
    axes[2].set_ylabel("direct")
    axes[2].set_xlabel("Time (obs)")
    fig.suptitle(f"Temporal Regime Comparison - {ticker}", y=0.98)
    plt.tight_layout()
    plt.savefig(PAPER_FIGURES_DIR / f"temporal_regime_comparison_{ticker}.png", dpi=300)
    plt.close()

# Microstructure regime characteristics (local per ticker)
for ticker in TICKERS:
    if ticker not in local_states:
        continue
    ticker_cols = [c for c in wass_X_all.columns if c.startswith(f"{ticker}_")]
    df_local = wass_X_all[ticker_cols].copy()
    _plot_regime_characteristics(
        df_local,
        local_states[ticker][: len(df_local)],
        f"Microstructure Regime Characteristics (Local) - {ticker}",
        PAPER_FIGURES_DIR / f"regime_characteristics_local_{ticker}.png",
    )

# Microstructure regime characteristics (global meta and direct)
_plot_regime_characteristics(
    wass_X_all,
    global_direct_states[: len(wass_X_all)],
    "Microstructure Regime Characteristics (Global Direct)",
    PAPER_FIGURES_DIR / "regime_characteristics_global_direct.png",
)
_plot_regime_characteristics(
    pd.DataFrame(X_meta, columns=[f"p_{i}" for i in range(X_meta.shape[1])]),
    global_states[: len(X_meta)],
    "Microstructure Regime Characteristics (Meta-HMM)",
    PAPER_FIGURES_DIR / "regime_characteristics_meta.png",
)

# Stress decomposition by ticker/metric with global overlays
def _stress_decomposition_plot(states: np.ndarray, title: str, filename: str):
    """Plot stress decomposition with regime overlays."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    metric_names = ['Price', 'OBI', 'OFI']
    metric_keys = ['price_ret', 'obi', 'ofi']
    for i, (m_name, m_key) in enumerate(zip(metric_names, metric_keys)):
        for ticker in TICKERS:
            stress_series = wass_X_decomposed[m_key][ticker]
            axes[i].plot(
                range(len(stress_series)),
                stress_series,
                label=ticker,
                alpha=0.7,
                linewidth=1.2
            )
        for regime in np.unique(states):
            mask = (states == regime)
            axes[i].fill_between(
                range(len(states)),
                0,
                axes[i].get_ylim()[1],
                where=mask,
                alpha=0.12,
                label=f"Regime {regime}" if i == 0 else ''
            )
        axes[i].set_ylabel(f'{m_name} Stress', fontweight='bold')
        axes[i].grid(alpha=0.3)
    axes[2].set_xlabel('Time (index)')
    axes[0].legend(loc='upper right', ncol=6)
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(PAPER_FIGURES_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()

_stress_decomposition_plot(global_states, "Stress Decomposition (Overlay: Meta-HMM)", "stress_decomposition_meta.png")
_stress_decomposition_plot(global_direct_states, "Stress Decomposition (Overlay: Direct Global HMM)", "stress_decomposition_direct.png")

# Lead-lag multi-metric by quantile per ticker and HMM (local/meta), only significant
quantiles = [0.1, 0.5, 0.9]
alpha = 0.05
min_obs = 30
metric_keys = ["price_ret", "obi", "ofi"]
metric_names = {"price_ret": "Price", "obi": "OBI", "ofi": "OFI"}

leadlag_sig_rows = []
leadlag_sig_store = {}

for ticker in TICKERS:
    if ticker not in local_states:
        continue
    for src in metric_keys:
        for tgt in metric_keys:
            source_series = np.array(wass_X_decomposed[src][ticker])
            target_series = np.array(wass_X_decomposed[tgt][ticker])

            # Local HMM regimes: plot only if significant in any regime
            for regime in np.unique(local_states[ticker]):
                mask = local_states[ticker][: len(source_series)] == regime
                if mask.sum() < min_obs:
                    continue
                df_sig = _compute_leadlag_significant(
                    source_series[mask],
                    target_series[mask],
                    quantiles,
                    max_lag=LEADLAG_MAX_LAG,
                    alpha=alpha,
                    min_obs=min_obs,
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

            # Meta HMM regimes: use global states as mask
            for regime in np.unique(global_states):
                mask = global_states[: len(source_series)] == regime
                if mask.sum() < min_obs:
                    continue
                df_sig = _compute_leadlag_significant(
                    source_series[mask],
                    target_series[mask],
                    quantiles,
                    max_lag=LEADLAG_MAX_LAG,
                    alpha=alpha,
                    min_obs=min_obs,
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

# Save LaTeX tables: significant lead-lag per ticker and regime (full)
if leadlag_sig_rows:
    leadlag_sig_df = pd.concat(leadlag_sig_rows, ignore_index=True)
    leadlag_sig_df = leadlag_sig_df.sort_values(["p_value", "correlation"], ascending=[True, False])
    for ticker in TICKERS:
        for hmm in ["local", "meta"]:
            sub = leadlag_sig_df[(leadlag_sig_df["ticker"] == ticker) & (leadlag_sig_df["hmm"] == hmm)]
            if sub.empty:
                continue
            _save_latex_table(
                sub,
                PAPER_TABLES_DIR / f"leadlag_significant_{hmm}_{ticker}.tex",
                f"Lead-lag significatifs ({hmm}) - {ticker}",
                f"tab:leadlag_sig_{hmm}_{ticker}",
            )

    _save_latex_table(
        leadlag_sig_df,
        PAPER_TABLES_DIR / "leadlag_significant_global_top.tex",
        "Lead-lag significatifs (global)",
        "tab:leadlag_sig_global",
    )

# Limit number of figures: top 5 significant combinations per ticker and HMM
for ticker in TICKERS:
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
        top = candidates[:5]
        for _, _, key, df_sig in top:
            hmm_k, ticker_k, regime_k, src_k, tgt_k = key
            title = f"Lead-lag ({hmm_k} HMM) {ticker_k}: {src_k} → {tgt_k} (Regime {regime_k})"
            filename = f"leadlag_{hmm_k}_{ticker_k}_{src_k}_{tgt_k}_R{regime_k}.png"
            _plot_leadlag_from_df(df_sig, title, PAPER_FIGURES_DIR / filename)


# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================
print("\n" + "="*80)
print("RÉSUMÉ FINAL - ARCHITECTURE HIÉRARCHIQUE")
print("="*80)

print("\n✓ PIPELINE COMPLÉTÉ AVEC SUCCÈS !\n")

print("📊 RÉSULTATS CLÉS :")
print(f"  1. Patient Zéro identifié : {patient_zero_info['patient_zero']}")
print(f"     → Contagion Score = {patient_zero_info['contagion_score']:.3f}")
print(f"     → TE sortante = {patient_zero_info['te_outgoing']:.4f} nats")
print(f"  2. Régimes globaux : {N_REGIMES} états sectoriels détectés")
print(f"  3. Synchronisation moyenne : {sync_df['sync_rate'].mean():.1%}")
print(f"  4. TE moyen : {te_matrix.values[te_matrix.values > 0].mean():.4f} nats")

print("\n📁 FICHIERS GÉNÉRÉS :")
print("  Données CSV :")
print("    - hierarchical_states_local.csv")
print("    - hierarchical_states_global.csv")
print("    - hierarchical_synchronization.csv")
print("    - hierarchical_transfer_entropy.csv")
print("    - hierarchical_transfer_entropy_k_summary.csv")
print("    - hierarchical_regime_correlation.csv")
print("    - hierarchical_patient_zero.txt")
print("    - hierarchical_event_study_goog.csv")

print("\n  Visualisations PNG :")
print("    - hierarchical_regime_hierarchy.png")
print("    - hierarchical_contagion_network.png")
print("    - hierarchical_te_heatmap.png")
print("    - hierarchical_timeline_probabilities.png")
print("    - hierarchical_concordance_matrices.png")
print("    - hierarchical_leadlag_local_vs_global_quantile.csv")
print("    - hierarchical_leadlag_local_vs_global_R{regime}_Q{quantile}.png")
print("    - hierarchical_leadlag_between_tickers_quantile.csv")
print("    - hierarchical_leadlag_between_tickers_Q{quantile}.png")
print("    - leadlag_multimetric_grid_{TICKER|GLOBAL}.png")
print("    - leadlag_multimetric_quantile_significant.csv")
print("    - leadlag_tickers_{metric}_Q{quantile}.png")
print("    - leadlag_tickers_by_metric_quantile_significant.csv")

print("\n🎯 INNOVATIONS IMPLÉMENTÉES :")
print("  ✓ HMM hiérarchique (local + global)")
print("  ✓ Normalisation MAD (robuste aux outliers)")
print("  ✓ Transfer Entropy (causalité dirigée)")
print("  ✓ Identification du Patient Zéro")
print("  ✓ Résolution du Label Switching")
print("  ✓ Filtrage du bruit par consensus")

print("\n" + "="*80)
print("ARCHITECTURE HIÉRARCHIQUE - VERSION 2.0")
print("="*80)
