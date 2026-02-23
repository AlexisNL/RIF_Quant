---
layout: default
---

## Full Paper (PDF): **[Open `paper/main.pdf`](paper/main.pdf)**

# Regime Detection and Contagion in High-Frequency Markets
### A Hierarchical Temporal Wasserstein-HMM Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Working Paper](https://img.shields.io/badge/Status-Working_Paper-orange.svg)]()

> **Abstract.** This project studies how intraday microstructure stress propagates across US tech stocks using a two-level regime framework. I combine robust MAD normalization, temporal 1-Wasserstein distances (before-vs-after distributions), local per-ticker HMMs, a global Meta-HMM, and directed Transfer Entropy. The central result is a **state-dependent and asymmetric** contagion structure: local regimes are highly heterogeneous, global synchronization is sparse on this non-crisis day, and directed information flows concentrate through a few channels.

> **Dataset.** LOBSTER high-frequency data (June 21, 2012), five tickers (AAPL, INTC, GOOG, AMZN, MSFT), synchronized at 500 ms (~46,400 observations per ticker).

---

## Why this framework

Classical correlation-based diagnostics miss the distributional ruptures that dominate high-frequency stress episodes. This pipeline targets those ruptures directly:

1. **MAD normalization** removes outlier sensitivity without strong parametric assumptions.
2. **Temporal Wasserstein features** detect when the distribution of each microstructure variable changes.
3. **Hierarchical HMMs** separate local microstructure regimes from sector-level regimes.
4. **Transfer Entropy + lead-lag** reveal directionality and stress-conditioned topology shifts.

Regime labels are assigned with the composite stress proxy
`sigma_tilde_r = norm99(|Delta p/p|) + norm99(|OBI|) + norm99(|OFI|)`
using percentile-clipped robust scaling (1st-99th percentile).

---

## Core Contributions

- **Temporal OT signal for regime detection.** I use before-vs-after Wasserstein distances per ticker and metric instead of cross-sectional distances.
- **Hierarchical architecture for multi-asset microstructure.** Local HMMs capture ticker-specific geometry, then Meta-HMM captures global coordination.
- **Directed contagion mapping.** Transfer Entropy on regime dynamics complements synchronization and lead-lag evidence.
- **Stress-conditioned dependency map.** Quantile lead-lag (Q10/Q50/Q90) highlights topology inversion between calm and stressed conditions.

---

## Pipeline at a glance

1. Build and synchronize LOB features (`price_ret`, `OBI`, `OFI`).
2. Apply per-ticker MAD normalization.
3. Compute temporal Wasserstein distances per metric.
4. Fit local Gaussian HMMs (multi-restart: **5 x 200** EM iterations).
5. Aggregate local posteriors with Meta-HMM.
6. Compute synchronization, Transfer Entropy, and lead-lag diagnostics.

---

## Results

### 1) Local HMM regime signatures

Below, each ticker is labeled with the stress proxy ordering and interpreted using the descriptive statistics in `paper/tables/regime_stats_*.tex`.

- **AAPL** (`R2` Calm 9.4%, `R0` Intermediate 56.0%, `R1` Stressed 34.6%).  
  The stressed state is clearly separated by return risk (`sigma_r = 0.73 bp`, `|r_bar| = 0.37 bp`) and much higher kurtosis (62.1), while OBI is relatively stable across states.

<p align="center">
  <img src="img/hmm_local_AAPL_posterior.png" width="920"><br>
</p>

- **INTC** (`R0` Calm 80.6%, `R2` Intermediate 14.2%, `R1` Stressed 5.2%).  
  The stress ordering is driven more by order-book pressure than by volatility monotonicity: `R1` has the highest `|OBI|` and elevated `|OFI|`, with fat tails in all states.

<p align="center">
  <img src="img/hmm_local_INTC_posterior.png" width="920"><br>
</p>

- **GOOG** (`R1` Calm 1.3%, `R2` Intermediate 3.6%, `R0` Stressed 95.1%).  
  GOOG is almost entirely in one dominant stressed-labeled state within its own intraday scale, consistent with persistent high microstructure activity rather than constant crisis conditions.

<p align="center">
  <img src="img/hmm_local_GOOG_posterior.png" width="920"><br>
</p>

- **AMZN** (`R2` Calm 23.2%, `R0` Intermediate 65.6%, `R1` Stressed 11.2%).  
  The stressed label is mainly OFI-driven (`|OFI| = 241`), while return volatility peaks in the intermediate state, indicating that directional flow pressure and price variability are not perfectly aligned.

<p align="center">
  <img src="img/hmm_local_AMZN_posterior.png" width="920"><br>
</p>

- **MSFT** (`R2` Calm 2.4%, `R0` Intermediate 67.4%, `R1` Stressed 30.2%).  
  MSFT shows a clean monotonic increase in volatility and flow/imbalance intensity from Calm to Stressed, with a rare calm pocket and a substantial stressed occupancy.

<p align="center">
  <img src="img/hmm_local_MSFT_posterior.png" width="920"><br>
</p>

**Interpretation.** Local regime geometry remains strongly heterogeneous across names, which supports per-ticker parameter optimization and motivates the hierarchical aggregation step rather than a single shared local model.

---

### 2) Global regimes and synchronization

<p align="center">
  <img src="img/hmm_meta_posterior.png" width="920"><br>
  <img src="img/stress_decomposition_by_ticker_log.png" width="1020"><br>
  <img src="img/leadlag_local_global_heatmap.png" width="920"><br>
</p>

From `paper/tables/local_global_sync.tex`:

- **Total global transitions:** 51.
- **Co-transition sync rates:**
  - AAPL: 0/51 (0.0000)
  - GOOG: 7/51 (0.1373)
  - AMZN: 1/51 (0.0196)
  - MSFT: 1/51 (0.0196)
  - INTC: 0/51 (0.0000)
- **Leadership score:** AAPL 1.0000, GOOG 1.0000, AMZN 1.0000, MSFT 0.1111, INTC 0.0000.

**Interpretation.** On this day, synchronization exists but remains sparse. GOOG is the most directly synchronized ticker with global transitions, while AAPL and INTC show no direct co-transition events.

---

### 3) Lead-lag topology is regime-dependent

<p align="center">
  <img src="img/leadlag_interticker_heatmap.png" width="920"><br>
  <img src="img/leadlag_multimetric_grid.png" width="1020"><br>
  <img src="img/leadlag_interticker_price_ret.png" width="920"><br>
  <img src="img/leadlag_interticker_obi.png" width="920"><br>
  <img src="img/leadlag_interticker_ofi.png" width="920"><br>
</p>

From `paper/tables/leadlag_between_tickers_top.tex`:

- **Calm (Q10):** GOOG -> AAPL is strongest (`r = 0.393`, +2.0s).
- **Stressed (Q90):** AAPL -> INTC is strongest (`r = 0.526`, -7.5s), with strong MSFT -> INTC (`r = 0.387`, -0.5s) and INTC -> AAPL (`r = 0.381`, +10.0s).

**Interpretation.** The dependency graph is not static: leadership and lag structure shift materially across stress quantiles.

---

### 4) Transfer Entropy and directed contagion

From `paper/tables/transfer_entropy_top.tex` (top links):

| Source -> Target | TE |
|---|---:|
| AMZN -> MSFT | 0.0035 |
| INTC -> MSFT | 0.0022 |
| AAPL -> MSFT | 0.0017 |
| GOOG -> MSFT | 0.0013 |
| MSFT -> INTC | 0.0013 |

**Interpretation.** In current outputs, the strongest directed channels converge on **MSFT**, which acts as the main recipient node among top TE flows.

---

### 5) Robustness and model agreement

From `paper/tables/robustness_unified.tex`:

- ARI meta vs K-means: **0.2123**
- ARI direct vs K-means: **0.2587**
- ARI meta vs direct: **0.3711**
- Entropy mean: meta **0.0006**, direct **0.0004**
- State sync (meta vs direct): **0.7571**
- MMD (Direct vs Meta):
  - Price: 0.1879 vs 0.1742
  - OFI: 0.7257 vs 0.7099
  - OBI: 0.7688 vs 0.6024

**Interpretation.** Both global models are highly confident (near-deterministic posteriors). Direct Global HMM is stronger on K-means alignment and MMD separation in this run, while Meta-HMM remains structurally consistent and interpretable.

---

### 6) Event study (microstructure reallocation)

<p align="center">
  <img src="img/event_study_goog_spike_optimal.png" width="1020"><br>
</p>

A GOOG OFI spike is followed by synchronous OFI reactions in peers, consistent with rapid cross-sectional liquidity reallocation.

---

## Quickstart

```bash
git clone https://github.com/AlexisNL/RIF_Quant.git
cd RIF_Quant
pip install -r requirements.txt

# optional: per-ticker/grid optimization
python scripts/optimize_hierarchical_parameters.py

# full hierarchical pipeline
python scripts/run_hierarchical_contagion.py
```

Optimization subsets:

```bash
python scripts/optimize_hierarchical_parameters.py --locals-only
python scripts/optimize_hierarchical_parameters.py --meta-only
python scripts/optimize_hierarchical_parameters.py --direct-only
```

---

## Repository map

- Main manuscript: `paper/main.tex`
- Final PDF: `paper/main.pdf`
- Generated figures: `paper/figures/`
- Website images: `img/`
- Main pipeline: `scripts/run_hierarchical_contagion.py`

Working paper, February 2026.
