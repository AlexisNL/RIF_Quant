# Regime Detection & Contagion in High-Frequency Markets
### A Hierarchical Wasserstein-HMM Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Research in Progress](https://img.shields.io/badge/Status-Research_in_Progress-orange.svg)]()

> **Abstract:** This repository implements a novel framework to detect non-linear regime switches and stress propagation in Limit Order Books (LOB). By leveraging **Optimal Transport (Wasserstein Distance)** and **Hierarchical Hidden Markov Models**, we identify directed contagion pathways across the US Tech sector.

---

## 💡 Core Methodology

Traditional correlation-based methods often fail to capture the distributional "ruptures" characteristic of HFT stress. Our pipeline addresses this through:

1. **Robust Pre-processing:** Non-parametric **MAD normalization** to isolate standardized innovations from microstructure noise.
2. **Distributional Features:** Implementation of **Temporal 1-Wasserstein Distance** (before vs. after) to quantify shifts in the full LOB distribution (Micro-price, OBI, OFI).
3. **Hierarchical Modeling:** * **Local HMMs:** Individually optimized per ticker (AAPL, INTC, GOOG, AMZN, MSFT).
    * **Meta-HMM:** A consensus layer aggregating local state probabilities to identify sector-wide regimes.
4. **Causality & Contagion:** Measurement of directed information flow using **Transfer Entropy** and Lead-Lag Quantile analysis to identify "Patient Zero" events.



---

## 📊 Empirical Highlights (Sample Results)

* **Global Synchronization:** 72% agreement between Meta-HMM and Direct Global HMM architectures.
* **Contagion Pathways:** Identification of directed causal links (e.g., AAPL $\rightarrow$ INTC) during liquidity shocks.
* **Robustness:** Validated via **Adjusted Rand Index (ARI)** against K-Means and **Maximum Mean Discrepancy (MMD)** diagnostics.

---

## 🚀 Quickstart

### Installation
```bash
# Clone the repository
git clone [https://github.com/AlexisNL/RIF_Quant.git](https://github.com/AlexisNL/RIF_Quant.git)
cd RIF_Quant

# Install dependencies
pip install -r requirements.txt
pip install numba  # Strongly recommended for Wasserstein speedup