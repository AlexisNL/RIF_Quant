# Quickstart Guide

This guide covers the current hierarchical pipeline:
local HMMs + global Meta-HMM + direct global HMM.

## 1) Installation

```bash
pip install -r requirements.txt
```

## 2) Required Data

Place LOBSTER files in `data/raw/`:

```text
data/raw/
AAPL_2012-06-21_34200000_57600000_orderbook_5.csv
AAPL_2012-06-21_34200000_57600000_message_5.csv
INTC_2012-06-21_34200000_57600000_orderbook_5.csv
INTC_2012-06-21_34200000_57600000_message_5.csv
... (GOOG, AMZN, MSFT)
```

Default experiment parameters are in `src/config.py`:
`TICKERS`, `ANALYSIS_DATE`, `RESAMPLE_FREQ`, `WASSERSTEIN_WINDOW`, `N_REGIMES`, etc.

## 3) Standard Run

```bash
# Optional but recommended: optimize parameters first
python scripts/optimize_hierarchical_parameters.py

# Run full pipeline
python scripts/run_hierarchical_contagion.py
```

## 4) Targeted Optimization

```bash
# Local HMM optimization only
python scripts/optimize_hierarchical_parameters.py --locals-only

# Global meta-HMM only
python scripts/optimize_hierarchical_parameters.py --meta-only

# Direct global HMM only
python scripts/optimize_hierarchical_parameters.py --direct-only
```

## 5) Main Outputs

### `data/results/`

- `optimization_hierarchical_results_per_ticker.csv`
- `best_parameters_hierarchical_per_ticker.csv`
- `optimization_global_direct.csv`
- `hierarchical_temporal_features.csv`
- `hierarchical_states_local.csv`
- `hierarchical_states_global.csv`
- `hierarchical_states_global_direct.csv`
- `hierarchical_synchronization.csv`
- `hierarchical_leadlag_local_vs_global_quantile.csv`
- `hierarchical_leadlag_between_tickers_quantile.csv`
- `hierarchical_transfer_entropy.csv`
- `hierarchical_regime_stats.csv`
- `hierarchical_event_study_goog.csv`

### Paper artifacts

- Figures: `paper/figures/`
- LaTeX tables: `paper/tables/`

## 6) Build Paper

From `paper/`:

```bash
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

## Troubleshooting

- `ImportError` or missing package:
  `pip install -r requirements.txt`
- Missing/empty outputs:
  verify files in `data/raw/` and the date/ticker settings in `src/config.py`
- Undefined citations/references in paper:
  run `biber main` between LaTeX passes

---
Version: 2.1  
Date: 2026-02-23
