# Regime Change Detection and Contagion in HFT

This repository implements a microstructure regime detection framework using HMMs optimized on order-book metrics (Price, OFI, OBI) transformed by temporal Wasserstein distance. The approach is applied to five tech tickers to estimate local regimes, then a global meta-HMM (and a direct global HMM on the aggregated features) to analyze contagion, lead-lag, and directed causality.

## Core Idea

1. Robust MAD normalization with rolling windows.
2. Regime-change features via temporal Wasserstein (before vs. after, same asset/metric).
3. Local HMMs per ticker with per-ticker optimized parameters.
4. Global meta-HMM on local state probabilities.
5. Direct global HMM on aggregated Wasserstein features.
6. ARI vs. KMeans and MMD diagnostics for robustness and overfitting control.
7. Lead-lag and contagion analyses (local?global and ticker?ticker).

## Scripts

- `scripts/optimize_hierarchical_parameters.py`
  - Optimizes local HMM parameters per ticker.
  - Optionally optimizes the global meta-HMM and direct global HMM.
  - Saves best parameters as `.txt` + `.csv` in `data/results/`.
- `scripts/run_hierarchical_contagion.py`
  - Runs the full hierarchical pipeline.
  - Loads best parameters from `.txt` files (if present).
  - Produces all CSV/PNG outputs in `data/results/`.

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) Speed up Wasserstein
pip install numba

# 1) Optimize parameters (default: local + global meta + global direct)
python scripts/optimize_hierarchical_parameters.py

# 2) Run the full pipeline
python scripts/run_hierarchical_contagion.py
```

## Data

LOBSTER files must be placed in `data/raw/`.
Expected format (example):

```
data/raw/
AAPL_2012-06-21_34200000_57600000_orderbook_5.csv
AAPL_2012-06-21_34200000_57600000_message_5.csv
INTC_2012-06-21_34200000_57600000_orderbook_5.csv
INTC_2012-06-21_34200000_57600000_message_5.csv
... (GOOG, AMZN, MSFT)
```

## Configuration

`src/config.py` centralizes key parameters:

- `TICKERS`, `ANALYSIS_DATE`
- `MAD_WINDOW`
- `WASSERSTEIN_WINDOW`
- `N_REGIMES_LOCAL`, `N_REGIMES_GLOBAL`
- `HMM_PERSISTENCE_LOCAL`, `HMM_PERSISTENCE_GLOBAL`
- `HMM_SMOOTHING_LOCAL`, `HMM_SMOOTHING_GLOBAL`
- `HMM_COV_FULL_CORR_THRESHOLD`
- MMD parameters (`MMD_WINDOW`, `MMD_STEP`, etc.)

Optimized parameters are automatically loaded from `data/results/best_parameters_hierarchical*.txt`.

## Main Outputs

Outputs are saved in `data/results/` (gitignored):

- Local and global states
- Local?global synchronization
- Local vs. global lead-lag
- Ticker?ticker lead-lag (with p-values)
- Transfer Entropy
- MMD diagnostics (local and global)
- GOOG event study
- Visualizations (heatmaps, timelines, concordance)

## Validation Metrics

- **ARI vs KMeans**: agreement between HMM segmentation and clustering.
- **MMD**: distribution separation across regimes, per ticker and globally.

## Repository Structure

See `REPO_STRUCTURE.md` for an up-to-date overview.

## Notes

- The pipeline is designed for one HFT day (2012-06-21) and five tech tickers.
- Results depend strongly on parameters (MAD, Wasserstein, persistence/smoothing).

---
Last update: 2026-02-07
