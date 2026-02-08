"""
Configuration centralisée pour le projet LOB Contagion Regimes
"""

import os
from pathlib import Path

# ============================================================================
# CHEMINS DE BASE
# ============================================================================

# Répertoire racine du projet
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Chemins de données
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"

# Chemins de sortie
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"
SUPPLEMENTARY_DIR = PROJECT_ROOT / "paper" / "supplementary"

# Créer les répertoires s'ils n'existent pas
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, 
                  FIGURES_DIR, SUPPLEMENTARY_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# PARAMÈTRES DE DONNÉES
# ============================================================================

# Tickers analysés
TICKERS = ["AAPL", "INTC", "GOOG", "AMZN", "MSFT"]

# Date d'analyse
ANALYSIS_DATE = "2012-06-21"
START_TIME = "34200000"  # 9:30 AM EST in milliseconds since midnight
END_TIME = "57600000"    # 4:00 PM EST

# Fréquence de resampling
RESAMPLE_FREQ = "500ms"  # 500 milliseconds = 2 Hz

# Nombre de niveaux LOB
N_LEVELS = 5

# ============================================================================
# PARAMÈTRES WASSERSTEIN
# ============================================================================

# Taille de la fenêtre glissante (en observations)
WASSERSTEIN_WINDOW = 100  # 100 obs × 500ms = 50 secondes

# Métriques à calculer
METRICS = ['price_ret', 'obi', 'ofi']

# ============================================================================
# PARAMÈTRES HMM OPTIMISÉ
# ============================================================================

# Nombre de régimes
N_REGIMES = 3

# Type de covariance
HMM_COVARIANCE_TYPE = "diag"  # 'diag' pour éviter sur-ajustement

# Persistance forcée (probabilité de rester dans le même état)
# Local (HMM par actif)
HMM_PERSISTENCE_LOCAL = 0.85  # 90%
# Global (Méta-HMM)
HMM_PERSISTENCE_GLOBAL = 0.85  # plus persistant au niveau global

# Fenêtre de lissage post-estimation (en observations)
# Local (HMM par actif)
HMM_SMOOTHING_LOCAL = 10  # 20 obs × 500ms = 10 secondes
# Global (Méta-HMM)
HMM_SMOOTHING_GLOBAL = 15  # plus agressif au niveau global

# Rétro-compatibilité (anciens scripts)
HMM_PERSISTENCE = HMM_PERSISTENCE_LOCAL
HMM_SMOOTHING_WINDOW = HMM_SMOOTHING_LOCAL

# Seuil de corrélation pour basculer en covariance 'full'
HMM_COV_FULL_CORR_THRESHOLD = 0.70

# Convergence threshold (HMM)
HMM_REQUIRE_CONVERGENCE = True
HMM_MIN_REGIME_SHARE = 0.01

# MMD diagnostics/penalty
MMD_WINDOW = 1000
MMD_STEP = 100
MMD_PENALTY_WEIGHT = 0.10
MMD_PENALTY_TARGET_CORR = 0.20

# Nombre d'itérations EM
HMM_N_ITER = 1000

# Seed aléatoire pour reproductibilité
RANDOM_STATE = 42

# ============================================================================
# PARAMÈTRES LEAD-LAG ANALYSIS
# ============================================================================

# Plage de lags à tester (en observations)
LEADLAG_MAX_LAG = 20  # ±20 obs × 500ms = ±10 secondes
MAX_LAG = LEADLAG_MAX_LAG  # Alias pour compatibilité

# Quantiles à analyser
LEADLAG_QUANTILES = [0.1, 0.5, 0.9]  # Q10, Q50, Q90
QUANTILES = LEADLAG_QUANTILES  # Alias pour compatibilité

# Seuil de significativité statistique
ALPHA_SIGNIFICANCE = 0.05  # p < 0.05

# Nombre minimum d'observations pour calcul corrélation
MIN_OBS_CORRELATION = 30

# ============================================================================
# PARAMÈTRES EVENT STUDY
# ============================================================================

# Temps du spike GOOG (en index d'observation)
EVENT_SPIKE_TIME = 15000  # ~11:33 AM

# Fenêtres d'analyse
EVENT_WINDOW_WIDE = 2000   # ±16 minutes pour contexte
EVENT_WINDOW_NARROW = 500  # ±4 minutes pour zoom détaillé

# ============================================================================
# PARAMÈTRES VISUALISATION
# ============================================================================

# DPI pour figures
FIGURE_DPI = 300

# Style seaborn
SEABORN_STYLE = "whitegrid"

# Palette de couleurs pour régimes
REGIME_COLORS = {
    0: '#FF6B6B',  # Rouge clair
    1: '#4ECDC4',  # Turquoise
    2: '#45B7D1'   # Bleu ciel
}

# Couleurs pour quantiles lead-lag
QUANTILE_COLORS = {
    0.1: '#1f77b4',  # Bleu
    0.5: '#ff7f0e',  # Orange
    0.9: '#2ca02c'   # Vert
}

# Taille de police
FONT_SIZE_TITLE = 14
FONT_SIZE_LABEL = 12
FONT_SIZE_LEGEND = 10

# ============================================================================
# PARAMÈTRES MULTIPROCESSING
# ============================================================================

# Nombre de workers (None = utiliser tous les CPUs disponibles -1)
N_JOBS = None

# ============================================================================
# FORMATS DE FICHIERS
# ============================================================================

# Format des fichiers LOBSTER
LOBSTER_ORDERBOOK_FORMAT = "{ticker}_{date}_{start}_{end}_orderbook_{levels}.csv"
LOBSTER_MESSAGE_FORMAT = "{ticker}_{date}_{start}_{end}_message_{levels}.csv"

# Format de sauvegarde des résultats
RESULTS_FORMAT = {
    'regime_stats': 'regime_statistics_optimal.csv',
    'contagion_matrix': 'contagion_matrix_optimal.png',
    'leadlag_grid': 'leadlag_multimetric_grid_optimal.png',
    'leadlag_significant': 'leadlag_crossmetric_significant_only.png',
    'event_study': 'event_study_goog_spike_optimal.png',
    'stress_decomposition': 'stress_decomposition_by_ticker_optimal.png',
    'robustness_confusion': 'confusion_matrix_regimes.png',
    'robustness_temporal': 'temporal_regime_comparison_obi_ofi.png'
}

# ============================================================================
# LOGGING
# ============================================================================

# Niveau de log
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR

# Format du log
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ============================================================================
# HELPERS
# ============================================================================

def get_lobster_filepath(ticker: str, file_type: str = "orderbook") -> Path:
    """Construit le chemin vers un fichier LOBSTER."""
    if file_type == "orderbook":
        filename = LOBSTER_ORDERBOOK_FORMAT.format(
            ticker=ticker,
            date=ANALYSIS_DATE.replace("-", "_"),
            start=START_TIME,
            end=END_TIME,
            levels=N_LEVELS
        )
    elif file_type == "message":
        filename = LOBSTER_MESSAGE_FORMAT.format(
            ticker=ticker,
            date=ANALYSIS_DATE.replace("-", "_"),
            start=START_TIME,
            end=END_TIME,
            levels=N_LEVELS
        )
    else:
        raise ValueError(f"Unknown file_type: {file_type}")
    
    return RAW_DATA_DIR / filename

def get_result_filepath(result_key: str) -> Path:
    """Retourne le chemin vers un fichier de résultat."""
    filename = RESULTS_FORMAT.get(result_key)
    if filename is None:
        raise ValueError(f"Unknown result_key: {result_key}")
    
    if filename.endswith('.png') or filename.endswith('.pdf'):
        return FIGURES_DIR / filename
    else:
        return RESULTS_DIR / "final_optimal" / filename

# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Valide la cohérence de la configuration."""
    # Vérifier que les tickers sont uniques
    assert len(TICKERS) == len(set(TICKERS)), "Duplicate tickers found"
    
    # Vérifier les plages de valeurs
    assert 0 < HMM_PERSISTENCE < 1, "HMM_PERSISTENCE must be in (0, 1)"
    assert 0 < ALPHA_SIGNIFICANCE < 1, "ALPHA_SIGNIFICANCE must be in (0, 1)"
    assert WASSERSTEIN_WINDOW > 0, "WASSERSTEIN_WINDOW must be positive"
    assert N_REGIMES >= 2, "N_REGIMES must be at least 2"
    
    # Vérifier que les répertoires existent
    assert DATA_DIR.exists(), f"Data directory not found: {DATA_DIR}"
    
    print("✓ Configuration validated successfully")

if __name__ == "__main__":
    # Test de validation
    validate_config()
    
    # Affichage de la configuration
    print("\n" + "="*80)
    print("CONFIGURATION SUMMARY")
    print("="*80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Tickers: {', '.join(TICKERS)}")
    print(f"Analysis date: {ANALYSIS_DATE}")
    print(f"N regimes: {N_REGIMES}")
    print(f"HMM persistence: {HMM_PERSISTENCE}")
    print(f"Wasserstein window: {WASSERSTEIN_WINDOW} obs ({WASSERSTEIN_WINDOW * 0.5}s)")
    print(f"Results directory: {RESULTS_DIR}")
    print("="*80)
