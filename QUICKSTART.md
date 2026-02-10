# Quickstart Guide

This guide covers the current hierarchical pipeline (local HMMs + global meta-HMM + direct global HMM).

## Installation

```bash
pip install -r requirements.txt

# Optional: Wasserstein acceleration
pip install numba
```

## Standard Run

```bash
# Parameter optimization (local + global meta + global direct)
python scripts/optimize_hierarchical_parameters.py

# Full hierarchical pipeline
python scripts/run_hierarchical_contagion.py
```

## Targeted Optimization

The optimization script can run a subset:

```bash
# Local HMMs only
python scripts/optimize_hierarchical_parameters.py --locals-only

# Global meta-HMM only
python scripts/optimize_hierarchical_parameters.py --meta-only

# Direct global HMM only
python scripts/optimize_hierarchical_parameters.py --direct-only
```

## Generated Outputs

Outputs are created in `data/results/`:

- Optimized parameters in `.txt` and `.csv`
- Local and global states
- Synchronization and leadership
- Local?global and ticker?ticker lead-lag (p-values + heatmaps)
- Local/global MMD diagnostics
- Transfer Entropy
- GOOG event study

## Required Data

Place LOBSTER files in `data/raw/`:

```
data/raw/
AAPL_2012-06-21_34200000_57600000_orderbook_5.csv
AAPL_2012-06-21_34200000_57600000_message_5.csv
... (INTC, GOOG, AMZN, MSFT)
```

## Troubleshooting

- ImportError: check `pip install -r requirements.txt`
- Slow run: install `numba` and reduce `WASSERSTEIN_WINDOW` in `src/config.py`
- Missing data: verify `data/raw/` and `ANALYSIS_DATE`

---
Version: 2.0
Date: 2026-02-07
