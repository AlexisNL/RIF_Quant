"""
Centralized configuration for the LOB Contagion Regimes project.
"""

import os
from pathlib import Path

# ============================================================================
# BASE PATHS
# ============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"

# Output directories
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"
SUPPLEMENTARY_DIR = PROJECT_ROOT / "paper" / "supplementary"

# Create directories if they do not exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR,
                  FIGURES_DIR, SUPPLEMENTARY_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA PARAMETERS
# ============================================================================

# Analyzed tickers
TICKERS = ["AAPL", "INTC", "GOOG", "AMZN", "MSFT"]

# Analysis date
ANALYSIS_DATE = "2012-06-21"
START_TIME = "34200000"  # 9:30 AM EST in milliseconds since midnight
END_TIME = "57600000"    # 4:00 PM EST

# Resampling frequency
RESAMPLE_FREQ = "500ms"  # 500 milliseconds = 2 Hz

# Number of LOB levels
N_LEVELS = 5

# ============================================================================
# WASSERSTEIN PARAMETERS
# ============================================================================

# Sliding window length (in observations)
WASSERSTEIN_WINDOW = 100  # 100 obs x 500ms = 50 seconds


# Metrics to compute
METRICS = ['price_ret', 'obi', 'ofi']

# ============================================================================
# OPTIMIZED HMM PARAMETERS
# ============================================================================

# Number of regimes
N_REGIMES = 3

# Covariance type
HMM_COVARIANCE_TYPE = "diag"  # 'diag' to reduce overfitting

# Forced persistence (probability of staying in the same state)
# Local (per-asset HMM)
HMM_PERSISTENCE_LOCAL = 0.85
# Global (Meta-HMM)
HMM_PERSISTENCE_GLOBAL = 0.85

# Post-estimation smoothing window (in observations)
# Local (per-asset HMM)
HMM_SMOOTHING_LOCAL = 10  # 20 obs x 500ms = 10 seconds
# Global (Meta-HMM)
HMM_SMOOTHING_GLOBAL = 15  # more aggressive smoothing globally

# Backward compatibility (legacy scripts)
HMM_PERSISTENCE = HMM_PERSISTENCE_LOCAL
HMM_SMOOTHING_WINDOW = HMM_SMOOTHING_LOCAL

# Correlation threshold to switch to 'full' covariance
HMM_COV_FULL_CORR_THRESHOLD = 0.70

# Convergence threshold (HMM)
HMM_REQUIRE_CONVERGENCE = True
HMM_MIN_REGIME_SHARE = 0.01

# MMD diagnostics/penalty
MMD_WINDOW = 1000
MMD_STEP = 100
MMD_PENALTY_WEIGHT = 0.10
MMD_PENALTY_TARGET_CORR = 0.20

# Number of EM iterations
HMM_N_ITER = 1000

# Random seed for reproducibility
RANDOM_STATE = 42

# ============================================================================
# LEAD-LAG ANALYSIS PARAMETERS
# ============================================================================

# Lag range to test (in observations)
LEADLAG_MAX_LAG = 20  # +/-20 obs x 500ms = +/-10 seconds
MAX_LAG = LEADLAG_MAX_LAG  # Compatibility alias

# Quantiles to analyze
LEADLAG_QUANTILES = [0.1, 0.5, 0.9]  # Q10, Q50, Q90
QUANTILES = LEADLAG_QUANTILES  # Compatibility alias

# Statistical significance threshold
ALPHA_SIGNIFICANCE = 0.05  # p < 0.05

# Minimum number of observations for correlation computation
MIN_OBS_CORRELATION = 30

# ============================================================================
# EVENT STUDY PARAMETERS
# ============================================================================

# GOOG spike time (observation index)
EVENT_SPIKE_TIME = 15000  # ~11:33 AM

# Analysis windows
EVENT_WINDOW_WIDE = 2000   # +/-16 minutes for context
EVENT_WINDOW_NARROW = 500  # +/-4 minutes for detailed zoom

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================

# Figure DPI
FIGURE_DPI = 300

# Seaborn style
SEABORN_STYLE = "whitegrid"

# Color palette for regimes
REGIME_COLORS = {
    0: '#FF6B6B',  # Light red
    1: '#4ECDC4',  # Turquoise
    2: '#45B7D1'   # Sky blue
}

# Colors for lead-lag quantiles
QUANTILE_COLORS = {
    0.1: '#1f77b4',  # Blue
    0.5: '#ff7f0e',  # Orange
    0.9: '#2ca02c'   # Green
}

# Font sizes
FONT_SIZE_TITLE = 14
FONT_SIZE_LABEL = 12
FONT_SIZE_LEGEND = 10

# ============================================================================
# MULTIPROCESSING PARAMETERS
# ============================================================================

# Number of workers (None = all available CPUs minus one)
N_JOBS = None

# ============================================================================
# FILE FORMATS
# ============================================================================

# LOBSTER filename formats
LOBSTER_ORDERBOOK_FORMAT = "{ticker}_{date}_{start}_{end}_orderbook_{levels}.csv"
LOBSTER_MESSAGE_FORMAT = "{ticker}_{date}_{start}_{end}_message_{levels}.csv"

# Saved results filename mapping
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

# Log level
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR

# Log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ============================================================================
# HELPERS
# ============================================================================

def get_lobster_filepath(ticker: str, file_type: str = "orderbook") -> Path:
    """Get a LOBSTER filepath."""
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
    """Validate config."""
