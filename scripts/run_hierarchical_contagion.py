"""
Pipeline Hiérarchique de Détection de Contagion - Architecture à 2 niveaux
===========================================================================

INNOVATION MAJEURE : HMM de second ordre pour contagion sectorielle

Architecture :
--------------
NIVEAU 1 (Local)  : HMM par actif → P(régime | actif)
NIVEAU 2 (Global) : Méta-HMM observe toutes les probas → Régime sectoriel

Avantages clés :
----------------
✓ Résout le "Label Switching" (réaligne les sémantiques)
✓ Filtre le bruit (ignore transitions isolées)
✓ Détecte la contagion (co-mouvements de régimes)
✓ Identifie le "Patient Zéro" (Transfer Entropy)

Changements vs ancien pipeline :
---------------------------------
1. GARCH → Normalisation MAD (plus robuste aux outliers)
2. Labels → Probabilités d'état (variables continues)
3. HMM unique → HMM hiérarchique (local + global)
4. Corrélation prix → Corrélation de régimes
5. Lead-lag simple → Transfer Entropy (causalité dirigée)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
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
    compute_regime_correlation,
    identify_patient_zero,
    visualize_contagion_network
)
from src.analysis.leadlag import analyze_multimetric_leadlag_full
from src.analysis.event_study import analyze_event_goog_spike
from src.visualization.regime_plots import plot_regime_statistics


print("="*80)
print("PIPELINE HIÉRARCHIQUE DE DÉTECTION DE CONTAGION")
print("="*80)
print(f"\n📅 Date d'analyse: {ANALYSIS_DATE}")
print(f"📊 Tickers: {', '.join(TICKERS)}")
print(f"🔬 Régimes locaux: {N_REGIMES}")
print(f"🌐 Régimes globaux: {N_REGIMES}")
print(f"\n🆕 NOUVEAUTÉS :")
print(f"  - Normalisation MAD (robuste) au lieu de GARCH")
print(f"  - HMM hiérarchique (local + global)")
print(f"  - Transfer Entropy pour causalité dirigée")
print(f"  - Identification du 'Patient Zéro'")


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

# Use per-ticker optimized parameters if available
per_ticker_params_path = RESULTS_DIR / "best_parameters_hierarchical_per_ticker.txt"
per_ticker_params_csv = RESULTS_DIR / "best_parameters_hierarchical_per_ticker.csv"
use_per_ticker = per_ticker_params_path.exists() or per_ticker_params_csv.exists()
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

if use_per_ticker:
    if per_ticker_params_path.exists():
        # Parse INI-like txt
        current = None
        rows = []
        with open(per_ticker_params_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("[") and line.endswith("]"):
                    current = {"ticker": line.strip("[]")}
                    rows.append(current)
                elif "=" in line and current is not None:
                    k, v = [s.strip() for s in line.split("=", 1)]
                    current[k] = v
        per_ticker_params = pd.DataFrame(rows)
        for col in ["mad_window", "wasserstein_window", "local_smoothing", "n_regimes"]:
            if col in per_ticker_params.columns:
                per_ticker_params[col] = per_ticker_params[col].astype(int)
        for col in ["local_persistence", "ari_local", "mmd_penalty", "score"]:
            if col in per_ticker_params.columns:
                per_ticker_params[col] = per_ticker_params[col].astype(float)
        print(f"Parameters per ticker loaded: {per_ticker_params_path}")
    else:
        per_ticker_params = pd.read_csv(per_ticker_params_csv)
        print(f"Parameters per ticker loaded: {per_ticker_params_csv}")
else:
    print("??? Param??tres par ticker non trouv??s, fallback sur config global")

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
print(f"??? Matrice Wasserstein (temporal) : {wass_X_all.shape}")
output_file = RESULTS_DIR / "hierarchical_temporal_features.csv"
wass_X_all.to_csv(output_file, index=True)
print(f"??? Temporal features sauvegard??es dans {output_file}")

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
            corrs.append(np.corrcoef(x, y)[0, 1])
    return np.array(list(lags)), np.array(corrs)

def _leadlag_pvalues(series_local: np.ndarray, series_global: np.ndarray, lags: np.ndarray):
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
            # Pearson correlation p-value
            r = np.corrcoef(x, y)[0, 1]
            df = len(x) - 2
            if df <= 0 or np.isnan(r):
                pvals.append(np.nan)
            else:
                t_stat = r * np.sqrt(df / (1 - r**2 + 1e-12))
                pvals.append(2 * (1 - stats.t.cdf(abs(t_stat), df)))
    return np.array(pvals)

leadlag_rows = []
heatmap_rows = []
global_series = global_temporal_df["global_stress_wass_temporal"].values

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
    cols = [c for c in wass_X_all.columns if c.startswith(f"{ticker}_")]
    if not cols:
        continue
    local_series = wass_X_all[cols].mean(axis=1).values

    # Align lengths just in case
    n = min(len(local_series), len(global_series))
    local_series = local_series[:n]
    global_series_aligned = global_series[:n]

    lags, corrs = _leadlag_corr(local_series, global_series_aligned, LEADLAG_MAX_LAG)
    pvals = _leadlag_pvalues(local_series, global_series_aligned, lags)
    if np.all(np.isnan(corrs)):
        continue

    max_idx = int(np.nanargmax(np.abs(corrs)))
    best_lag = int(lags[max_idx])
    best_corr = float(corrs[max_idx])
    best_pval = float(pvals[max_idx]) if np.isfinite(pvals[max_idx]) else np.nan

    pos_corr = np.nanmax(corrs[lags > 0]) if np.any(lags > 0) else np.nan
    neg_corr = np.nanmax(corrs[lags < 0]) if np.any(lags < 0) else np.nan
    pos_pval = np.nanmin(pvals[lags > 0]) if np.any(lags > 0) else np.nan
    neg_pval = np.nanmin(pvals[lags < 0]) if np.any(lags < 0) else np.nan
    alpha_score = float(
        (pos_corr if np.isfinite(pos_corr) else 0.0)
        - (neg_corr if np.isfinite(neg_corr) else 0.0)
    )

    leadlag_rows.append(
        {
            "ticker": ticker,
            "best_lag_obs": best_lag,
            "best_lag_seconds": best_lag * 0.5,
            "best_corr": best_corr,
            "best_pval": best_pval,
            "max_corr_pos_lag": pos_corr,
            "max_corr_neg_lag": neg_corr,
            "min_pval_pos_lag": pos_pval,
            "min_pval_neg_lag": neg_pval,
            "alpha_score": alpha_score,
        }
    )

    for lag, corr, pval in zip(lags, corrs, pvals):
        heatmap_rows.append(
            {
                "ticker": ticker,
                "lag_obs": int(lag),
                "lag_seconds": lag * 0.5,
                "corr": corr,
                "pval": pval,
            }
        )

leadlag_df = pd.DataFrame(leadlag_rows).sort_values("alpha_score", ascending=False)
output_file = RESULTS_DIR / "hierarchical_leadlag_local_vs_global.csv"
leadlag_df.to_csv(output_file, index=False)
print(f"âœ“ Lead-lag local vs global sauvegardÃ© dans {output_file}")

# Log with p-value threshold
alpha_threshold = 0.05
sig_df = leadlag_df[leadlag_df["best_pval"] < alpha_threshold]
print(f"\nTop 3 alpha (local leads global, p < {alpha_threshold}):")
print(sig_df.head(3).to_string(index=False))

# Heatmap of correlations
heatmap_df = pd.DataFrame(heatmap_rows)
heatmap_pivot = heatmap_df.pivot(index="ticker", columns="lag_seconds", values="corr")
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_pivot, cmap="coolwarm", center=0, annot=False)
plt.title("Lead-lag Correlations (local vs global)")
plt.xlabel("Lag (seconds)")
plt.ylabel("Ticker")
output_file = RESULTS_DIR / "hierarchical_leadlag_local_vs_global_heatmap.png"
plt.tight_layout()
plt.savefig(output_file, dpi=300)
plt.close()
print(f"âœ“ Heatmap lead-lag sauvegardÃ©e dans {output_file}")


# ============================================================================
# ÉTAPE 3 : TRANSFER ENTROPY → CAUSALITÉ DIRIGÉE
# ============================================================================
print("\n" + "="*80)
print("ÉTAPE 3 : TRANSFER ENTROPY (CAUSALITÉ DIRIGÉE)")
print("="*80)
print("Objectif : Mesurer qui cause qui (information dirigée)\n")

# Calcul de la matrice de Transfer Entropy
# NOUVEAU : compute_transfer_entropy_matrix
te_matrix = compute_transfer_entropy_matrix(
    local_state_probs,
    TICKERS,
    k=2,  # Lag de 2 obs = 1 seconde
    bins=10
)

# Sauvegarde
output_file = RESULTS_DIR / 'hierarchical_transfer_entropy.csv'
te_matrix.to_csv(output_file)
print(f"\n✓ Matrice TE sauvegardée dans {output_file}")


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
print("Objectif : Qui initie la contagion ?\n")

# NOUVEAU : identify_patient_zero
patient_zero_info = identify_patient_zero(te_matrix, sync_df)

# Sauvegarde
output_file = RESULTS_DIR / 'hierarchical_patient_zero.txt'
with open(output_file, 'w') as f:
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

fig_leadlag = analyze_multimetric_leadlag_full(
    wass_X_decomposed,
    TICKERS,
    max_lag=20
)
output_file = RESULTS_DIR / 'hierarchical_leadlag_grid.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Lead-lag analysis sauvegardé dans {output_file}")
plt.close()


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
print("    - hierarchical_regime_correlation.csv")
print("    - hierarchical_patient_zero.txt")
print("    - hierarchical_event_study_goog.csv")

print("\n  Visualisations PNG :")
print("    - hierarchical_regime_hierarchy.png")
print("    - hierarchical_contagion_network.png")
print("    - hierarchical_te_heatmap.png")
print("    - hierarchical_timeline_probabilities.png")
print("    - hierarchical_concordance_matrices.png")
print("    - hierarchical_leadlag_grid.png")

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
