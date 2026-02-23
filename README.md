# Regime Identification Framework (RIF)

Hierarchical Wasserstein-HMM pipeline for regime detection and contagion analysis in high-frequency limit order books.

**Paper:** [paper/main.pdf](paper/main.pdf) · **Live site:** [GitHub Pages](https://alexisnl.github.io/RIF_Quant/)

---

## Architecture

The pipeline is implemented as six OOP classes:

| Class | Module | Role |
|---|---|---|
| `MADNormalizer` | `src/features/mad_normalizer.py` | Rolling MAD normalization of LOB metrics |
| `WassersteinExtractor` | `src/features/wasserstein.py` | Temporal 1-Wasserstein distance features |
| `LocalHMM` | `src/models/hmm_optimal.py` | Per-ticker Gaussian HMM — multi-restart + persistence forcing |
| `ContagionAnalyzer` | `src/analysis/contagion_metrics.py` | Transfer Entropy matrix, regime correlation, Patient Zero |
| `LeadLagAnalyzer` | `src/analysis/leadlag.py` | Quantile-conditioned cross-ticker lead-lag |
| `ContagionPipeline` | `scripts/run_hierarchical_contagion.py` | End-to-end orchestration via method chaining |

### LocalHMM — multi-restart design

Each local model runs `n_init=5` independent EM restarts of `n_iter=200` iterations each — same total budget as `n_init=1, n_iter=1000` — with the best restart selected by log-likelihood. This is equivalent to BIC selection under fixed model complexity: the penalty term `d·log(T)` is constant across restarts of the same architecture and cancels in all pairwise comparisons.

### ContagionPipeline — method chaining

```python
ContagionPipeline().run()
# Equivalent to:
(ContagionPipeline()
    .load_data()
    .extract_features()
    .fit_local_hmms()
    .fit_global_hmms()
    .analyze_leadlag()
    .analyze_contagion()
    .plot_visualizations()
    .generate_outputs()
    .print_summary())
```

---

## Quickstart

```bash
pip install -r requirements.txt
pip install numba          # JIT-compiled Wasserstein speedup (strongly recommended)

# (Optional) Per-ticker parameter optimization
python scripts/optimize_hierarchical_parameters.py

# Full pipeline
python scripts/run_hierarchical_contagion.py
```

Optimization subsets:
```bash
python scripts/optimize_hierarchical_parameters.py --locals-only
python scripts/optimize_hierarchical_parameters.py --meta-only
python scripts/optimize_hierarchical_parameters.py --direct-only
```

---

## Data

Place raw LOBSTER files in `data/raw/`:

```
data/raw/
  AAPL_2012-06-21_34200000_57600000_orderbook_5.csv
  AAPL_2012-06-21_34200000_57600000_message_5.csv
  INTC_2012-06-21_34200000_57600000_orderbook_5.csv
  INTC_2012-06-21_34200000_57600000_message_5.csv
  ... (GOOG, AMZN, MSFT)
```

---

## Configuration — `src/config.py`

| Parameter | Default | Description |
|---|---|---|
| `TICKERS` | `[AAPL, INTC, GOOG, AMZN, MSFT]` | Assets to process |
| `ANALYSIS_DATE` | `2012-06-21` | Trading day |
| `N_REGIMES` | `3` | Hidden states per HMM |
| `WASSERSTEIN_WINDOW` | `100` | Default Wasserstein window (observations) |
| `HMM_PERSISTENCE_LOCAL` | `0.90` | Default local persistence |
| `HMM_SMOOTHING_LOCAL` | `20` | Majority-vote smoothing half-window |
| `LEADLAG_MAX_LAG` | `20` | Max lead-lag in observations |
| `LEADLAG_QUANTILES` | `[0.10, 0.50, 0.90]` | Stress quantiles for conditioned analysis |

---

## Scripts

### `scripts/optimize_hierarchical_parameters.py`

Grid-search for per-ticker local HMM parameters:

- MAD window: {50, 100, 150}
- Wasserstein window: {50, 100, 150}
- Persistence: {0.85, 0.90, 0.95}
- Smoothing window: {10, 20, 30}

Criterion: `ARI(HMM, K-means) − 0.1 × MMD_penalty`

Writes to `results/`:
- `best_parameters_hierarchical_per_ticker.csv`
- `optimization_hierarchical_results_per_ticker.csv`
- `optimization_global_direct.csv`

### `scripts/run_hierarchical_contagion.py`

Full end-to-end pipeline. Auto-loads optimized parameters when present; falls back to `src/config.py` defaults.

---

## Outputs

### `results/`

| File | Description |
|---|---|
| `hierarchical_temporal_features.csv` | Wasserstein distance time series (all tickers × metrics) |
| `hierarchical_states_local.csv` | Per-ticker Viterbi state sequences |
| `hierarchical_states_global.csv` | Meta-HMM global state sequence |
| `hierarchical_states_global_direct.csv` | Direct Global HMM state sequence |
| `hierarchical_synchronization.csv` | Local-global co-transition analysis |
| `hierarchical_leadlag_between_tickers_quantile.csv` | Cross-ticker lead-lag by quantile |
| `hierarchical_leadlag_local_vs_global_quantile.csv` | Local-to-global lead-lag |
| `hierarchical_transfer_entropy.csv` | Full N×N Transfer Entropy matrix |
| `hierarchical_regime_stats.csv` | Per-regime LOB stats + Wasserstein means per metric |

### `paper/figures/`

| Figure | Description |
|---|---|
| `hmm_local_{TICKER}_posterior.png` | Per-ticker Viterbi regime sequence |
| `hmm_meta_posterior.png` | Meta-HMM global sequence |
| `hmm_direct_posterior.png` | Direct Global HMM sequence |
| `stress_decomposition_by_ticker_log.png` | Wasserstein stress by ticker + Meta-HMM overlay |
| `stress_decomposition_direct.png` | Same with Direct HMM overlay |
| `leadlag_interticker_heatmap.png` | Cross-ticker lead-lag heatmap (all metrics) |
| `leadlag_interticker_{metric}.png` | Per-metric inter-ticker lead-lag by quantile |
| `leadlag_local_global_heatmap.png` | Local-to-global lead-lag heatmap |
| `leadlag_multimetric_grid.png` | Multi-metric lead-lag grid by quantile |
| `event_study_goog_spike_optimal.png` | GOOG OFI event study with cross-ticker responses |

### `paper/tables/`

Auto-generated LaTeX tables for all paper sections. Per-ticker regime stats (`regime_stats_{TICKER}.tex`) include both Wasserstein distance columns (rate of distributional change) and LOB level columns (economic characterization).

---

## Build Paper

```bash
cd paper
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

---

## Notes

- **Optimization is optional.** The pipeline falls back to `src/config.py` defaults when `best_parameters_hierarchical_per_ticker.csv` is absent.
- **`LocalHMM` defaults:** `n_init=5, n_iter=200`. The optimization grid search uses `n_init=1` on the outer parameter loop for speed.
- **Logging.** All output uses Python `logging`; configure with `logging.basicConfig(level=logging.INFO)`. Windows users: `OK`/`WARN` are used instead of `✓`/`⚠` (cp1252 console codec).
- **Numba.** JIT acceleration for Wasserstein is optional but strongly recommended (~10× speedup). Results are numerically identical to the pure-Python path.
