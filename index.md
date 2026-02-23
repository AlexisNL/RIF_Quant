---
layout: default
---

## Full Paper (PDF): **[Open `paper/main.pdf`](paper/main.pdf)**

# Regime Detection & Contagion in High-Frequency Markets
### A Hierarchical Wasserstein-HMM Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Research in Progress](https://img.shields.io/badge/Status-Research_in_Progress-orange.svg)]()

> **Abstract.** This repository implements a hierarchical framework for detecting regime-dependent contagion in high-frequency limit order books. By combining **Temporal Wasserstein Distances** with **Hierarchical Hidden Markov Models** — each fitted with multiple random restarts selected by BIC-equivalent log-likelihood — the pipeline identifies directed contagion pathways across the US Tech sector, revealing a layered causal structure in which assets play distinct anticipatory, co-transitioning, amplifying, or following roles under market stress.

> **Data:** LOBSTER high-frequency data, June 21, 2012 — AAPL, INTC, GOOG, AMZN, MSFT — synchronized on a 500 ms grid (~46,400 observations per asset).

---

## Methodology

Traditional correlation-based methods fail to capture the distributional ruptures characteristic of HFT stress. The pipeline operates in four stages:

1. **MAD Normalization** — Median Absolute Deviation pre-processing isolates robust standardized innovations from microstructure noise without distributional assumptions (breakdown point 0.5). The window is optimized independently per ticker.

2. **Temporal 1-Wasserstein Distance** — For each ticker and metric (Price returns, OBI, OFI), the 1-Wasserstein distance between the *before* and *after* distributions at each time point measures distributional ruptures rather than level or volatility changes. The signal spikes at genuine regime transitions and is self-contained per ticker, making it directly suitable as HMM input.

3. **Hierarchical HMMs with multi-restart selection** — Per-ticker local Gaussian HMMs with individually optimized parameters extract posterior state probabilities via the forward-backward algorithm. To mitigate EM local optima, each model runs **5 independent restarts × 200 iterations** (same total budget as 1 × 1000), with the best restart selected by **log-likelihood — equivalent to BIC** under fixed model complexity (the penalty term cancels across restarts of the same architecture). A **Meta-HMM** then aggregates local posteriors into sector-wide regimes; a parallel **Direct Global HMM** on concatenated Wasserstein features serves as benchmark.

4. **Transfer Entropy + quantile-conditioned Lead-Lag** — Directed information flow from regime posteriors identifies the "Patient Zero". Lead-lag correlations are estimated separately within Wasserstein stress quantiles (Q10 = calm, Q50 = normal, Q90 = stressed), mapping the causal topology as it evolves across market conditions.

**Regime labelling** is based on the composite LOB stress proxy applied to original microstructure features:

> σ̃ᵣ = norm₉₉(|Δp/p|) + norm₉₉(|OBI|) + norm₉₉(|OFI|)

where norm₉₉ denotes robust min-max scaling clipped to the 1st–99th percentile. The regime with the lowest mean σ̃ᵣ is labelled **Calm** (green), the middle **Intermediate** (blue), the highest **Stressed** (red).

---

## 1) Per-Ticker Local HMM Regime Signatures

Each ticker follows its own microstructure geometry with different optimal parameters. The Viterbi sequence plots show one-hot-encoded regime assignments throughout the trading day (9:30–16:00 ET).

---

### AAPL — Calm R2 (52.0%) · Intermediate R0 (13.4%) · Stressed R1 (34.6%)

<p align="center">
  <img src="img/hmm_local_AAPL_posterior.png" width="900"><br>
  <em>Viterbi regime sequence for AAPL. Green = Calm (R2, 52.0%), Blue = Intermediate (R0, 13.4%), Red = Stressed (R1, 34.6%).</em>
</p>

Return volatility is the primary discriminant: Stressed R1 carries substantially higher σᵣ (0.73 bp) and |r̄| (0.37 bp) than Calm R2 (0.49 and 0.24 bp), while |OBI̅| and |OFI̅| differ more modestly. The elevated kurtosis of Stressed R1 (62.2 vs 15.5 in Calm) indicates that the stressed regime is driven by sparse, large return spikes rather than sustained elevated volatility.

**Contagion role — Pure Anticipatory Initiator.** AAPL's local transitions consistently precede global sector shifts without directly co-transitioning (0% co-transition rate). It is the dominant information hub in the Transfer Entropy network, receiving the largest inflows from all peers, with its strongest outgoing channel directed toward MSFT.

---

### INTC — Calm R0 (81.1%) · Intermediate R2 (12.8%) · Stressed R1 (6.1%)

<p align="center">
  <img src="img/hmm_local_INTC_posterior.png" width="900"><br>
  <em>Viterbi regime sequence for INTC. Green = Calm (R0, 81.1%), Blue = Intermediate (R2, 12.8%), Red = Stressed (R1, 6.1%).</em>
</p>

Return volatility is non-monotonic across regimes (σᵣ = 0.60 → 0.64 → 0.40 bp), so order-book imbalance drives the stress ranking: Stressed R1 carries the highest |OBI̅| (0.12) and elevated |OFI̅| (385). Uniformly high kurtosis across all regimes (94–132) reflects INTC's thinner market structure — fat-tailed returns regardless of regime.

**Contagion role — Secondary Anticipatory Amplifier.** INTC is a secondary early-warning indicator (leadership score 0.5, zero co-transitions) whose causal influence intensifies markedly under stress. In the Q90 lead-lag network it becomes the **dominant sector leader**, anticipating other names by several seconds — a pattern entirely invisible to static correlation analysis.

---

### GOOG — Calm R2 (1.3%) · Intermediate R1 (3.6%) · Stressed R0 (95.1%)

<p align="center">
  <img src="img/hmm_local_GOOG_posterior.png" width="900"><br>
  <em>Viterbi regime sequence for GOOG. Green = Calm (R2, 1.3%), Blue = Intermediate (R1, 3.6%), Red = Stressed (R0, 95.1%).</em>
</p>

R0 (Stressed) covers 95.1% of the trading day because it carries the highest σᵣ (0.68 bp) and |OBI̅| (0.31) *within GOOG's own intraday distribution* — not reflecting persistent distress in absolute terms, but GOOG's characteristically high microstructure intensity as a large-cap with continuous price discovery. Calm R2 (1.3%) marks the most quiescent episodes (σᵣ = 0.37 bp, |OFI̅| = 145); Intermediate R1 (3.6%) stands out through elevated directional order-flow (|OFI̅| = 219).

**Contagion role — Direct Co-Transition Partner.** GOOG combines leadership (its transitions precede global shifts) with the highest co-transition synchronization rate (17.9%, 7 out of 39 global transitions), indicating that it both anticipates and directly coincides with sector-wide regime changes. The GOOG→AAPL directed flow is the largest single Transfer Entropy link in the network. An intraday GOOG OFI spike triggers synchronous responses across all other names — consistent with active cross-sectional reallocation rather than passive contagion.

---

### AMZN — Calm R1 (22.4%) · Intermediate R0 (66.3%) · Stressed R2 (11.3%)

<p align="center">
  <img src="img/hmm_local_AMZN_posterior.png" width="900"><br>
  <em>Viterbi regime sequence for AMZN. Green = Calm (R1, 22.4%), Blue = Intermediate (R0, 66.3%), Red = Stressed (R2, 11.3%).</em>
</p>

Order-flow drives the stress ordering rather than return volatility: Stressed R2 achieves its label through the highest |OFI̅| (242 vs 182 for Intermediate and 174 for Calm), while Intermediate R0 paradoxically carries the highest σᵣ (0.90 bp) and |r̄| (0.40 bp). This non-monotonic σᵣ pattern suggests that AMZN's stressed regime corresponds to directional institutional order-slicing rather than large price moves.

**Contagion role — Net Follower.** Leadership score 0.0: AMZN's local transitions lag rather than lead global sector shifts. Its moderate Transfer Entropy outgoing is offset by the absence of any anticipatory signal in the synchronization analysis.

---

### MSFT — Calm R0 (1.4%) · Intermediate R2 (96.6%) · Stressed R1 (2.0%)

<p align="center">
  <img src="img/hmm_local_MSFT_posterior.png" width="900"><br>
  <em>Viterbi regime sequence for MSFT. Green = Calm (R0, 1.4%), Blue = Intermediate (R2, 96.6%), Red = Stressed (R1, 2.0%).</em>
</p>

MSFT is the most interpretable ticker: a clear monotonic gradient in return volatility (σᵣ = 0.04 → 0.63 → 0.74 bp), with both |OBI̅| and |OFI̅| also increasing monotonically across regimes (0.05 → 0.08 → 0.11 and 379 → 412 → 420). The Calm regime R0 is an exceptionally quiescent episode with near-zero volatility (σᵣ = 0.04 bp).

**Contagion role — Primary TE Recipient.** AAPL→MSFT is AAPL's dominant outgoing Transfer Entropy channel, positioning MSFT as the main downstream absorber of sector-wide stress. MSFT is bidirectional with AAPL (also sending back), consistent with its position as the most broadly held institutional name in the dataset.

---

## 2) Global Regime — Meta-HMM

<p align="center">
  <img src="img/hmm_meta_posterior.png" width="900"><br>
  <em>Meta-HMM global Viterbi regime sequence throughout the trading day (39 global transitions). Assignments are near-deterministic (mean entropy ≈ 0.001), confirming that regime boundaries reflect genuine structural changes rather than probabilistic ambiguity.</em>
</p>

The Meta-HMM aggregates local posterior probabilities across all five tickers into sector-wide regimes, resolving the label-switching problem inherent in flat multi-asset HMMs through a consensus mechanism that registers global transitions only when multiple tickers exhibit coordinated probability shifts. It captures the broader cluster geometry of the sector-wide feature space (ARI = 0.476 vs K-means); a parallel Direct Global HMM achieves stronger distributional separation between its regimes at the cost of lower non-parametric alignment — the two models are complementary rather than redundant.

Local-global synchronization varies substantially by ticker:

| Ticker | Co-transition Sync | Leadership Score | Role |
|:------:|:------------------:|:----------------:|------|
| **AAPL** | 0% (0/39) | 1.0 | Pure anticipatory — always precedes, never co-transitions |
| **GOOG** | 17.9% (7/39) | 1.0 | Direct co-transition trigger — anticipates and coincides |
| **MSFT** | 10.3% (4/39) | 0.333 | Moderate coupling, mixed lead/lag |
| **INTC** | 0% (0/39) | 0.5 | Secondary anticipatory — systematically precedes |
| **AMZN** | 2.6% (1/39) | 0.0 | Net follower |

<p align="center">
  <img src="img/leadlag_local_global_heatmap.png" width="820"><br>
  <em>Local-to-global lead-lag correlation heatmap. Best Spearman correlation between each ticker's local Wasserstein stress signal and the Meta-HMM global signal (quantile fallback Q90 → Q50 → Q10). Coupling is modest on this non-crisis day, consistent with partial regime-dependent sector coherence rather than permanent lockstep.</em>
</p>

---

## 3) Lead-Lag Dynamics: Topology Inversion Between Calm and Stress

Lead-lag correlations are estimated separately within Wasserstein stress quantiles. The causal network topology changes fundamentally between calm and stressed regimes — a structural shift invisible to static correlation matrices.

<p align="center">
  <img src="img/leadlag_multimetric_grid.png" width="980"><br>
  <em>Multi-metric lead-lag by stress quantile (Q10 = calm, Q50 = normal, Q90 = stressed). OFI autocorrelation exceeds 0.8 within ±5 s — consistent with algorithmic order-slicing arriving in persistent bursts. Q90 correlations substantially exceed Q10, confirming that microstructure interdependencies amplify nonlinearly under stress.</em>
</p>

### Calm regime (Q10): GOOG leads the sector

In low-stress conditions GOOG is the dominant leading indicator, consistent with informed price discovery where the most liquid large-cap sets the directional tone for the sector:
- **GOOG leads AAPL** by +2 s (*r* = 0.393)
- **GOOG leads MSFT** by +8 s (*r* = 0.278)
- **GOOG leads INTC** (*r* = 0.160)

### Stressed regime (Q90): INTC takes over

Under stress the network inverts sharply — **INTC becomes the dominant anticipatory leader**:
- **INTC leads AAPL** by +7.5 s (*r* = **0.526** — the strongest pairwise correlation across all quantiles and metric channels)
- **INTC leads MSFT** near-instantaneously (*r* = 0.387)
- **AMZN leads AAPL** at +2.5 s (*r* = 0.372)

This topology inversion — GOOG leading in calm, INTC under stress — is the core lead-lag finding. It is consistent with the Transfer Entropy structure: INTC's early-warning signal materializes as a dominant predictor only when stress is elevated, linked to its OBI-driven stress channel and thinner market structure.

### Aggregate inter-ticker dependency map

<p align="center">
  <img src="img/leadlag_interticker_heatmap.png" width="820"><br>
  <em>Strongest Spearman correlation across quantiles for each ticker pair. Dependence is selective — certain pairs dominate the aggregate level while others (AAPL–GOOG, INTC–AMZN) show near-zero correlation, confirming heterogeneous contagion channels within the sector rather than uniform co-movement.</em>
</p>

### OBI channel: GOOG→AAPL is the single strongest pairwise link

<p align="center">
  <img src="img/leadlag_interticker_obi.png" width="820"><br>
  <em>OBI inter-ticker lead-lag by quantile (Q10 left, Q90 right). GOOG→AAPL at Q90 (<em>r</em> = 0.54, −4 s) is the strongest pairwise correlation across all metric channels and quantiles, directly supporting GOOG's direct co-transition role. GOOG→AMZN is the second strongest at Q90 (<em>r</em> = 0.34, −9 s). Under calm conditions (Q10) OBI dependencies are weak throughout.</em>
</p>

The OFI channel shows near-instantaneous (≤1 s) stressed-regime coupling consistent with simultaneous algorithmic order-routing; the price-return channel shows a clear topology inversion between Q10 and Q90 with several correlations changing sign.

---

## 4) Transfer Entropy & Patient Zero

Transfer Entropy from regime posterior probabilities maps the directed information network. **AAPL emerges as the dominant information hub**: the four largest directed flows all target AAPL. Among AAPL's outgoing flows, **AAPL→MSFT** is the strongest, identifying MSFT as the primary downstream recipient.

| Ticker | Contagion Score | Leadership | Co-transition | Role |
|:------:|:--------------:|:----------:|:-------------:|------|
| **AAPL** | 2.00 | 1.0 | 0% | Anticipatory initiator — precedes, never co-transitions |
| **GOOG** | 1.47 | 1.0 | 17.9% | Direct co-transition partner — anticipates and coincides |
| **INTC** | 1.46 | 0.5 | 0% | Secondary amplifier — leads sector under high stress |
| **MSFT** | 0.33 | 0.333 | 10.3% | Primary TE recipient (AAPL→MSFT dominant outgoing) |
| **AMZN** | 0.93 | 0.0 | 2.6% | Net follower |

**Top directed TE flows:**

| Source → Target | TE (nats) | Reading |
|:---:|:---:|---|
| GOOG → AAPL | 5.4 × 10⁻⁴ | Largest single flow in the network |
| MSFT → AAPL | 4.0 × 10⁻⁴ | Bidirectional with AAPL |
| AMZN → AAPL | 3.3 × 10⁻⁴ | — |
| **AAPL → MSFT** | **3.3 × 10⁻⁴** | AAPL's dominant outgoing channel |
| INTC → AAPL | 3.3 × 10⁻⁴ | — |

The causal reading: AAPL fires *before* global shifts (pure anticipatory, 0% co-transition) while GOOG directly co-transitions with the sector (17.9% synchronization). AAPL→MSFT is the primary downstream propagation channel. The layered structure — *anticipatory initiator → direct co-transition partner → stress amplifier → follower* — reflects organized asymmetric directionality rather than uniform sector co-movement.

---

## Quickstart

### Installation
```bash
git clone https://github.com/AlexisNL/RIF_Quant.git
cd RIF_Quant
pip install -r requirements.txt
pip install numba          # Strongly recommended for Wasserstein speedup
```

### Run
```bash
# (Optional) Per-ticker parameter optimization
python scripts/optimize_hierarchical_parameters.py

# Full end-to-end pipeline
python scripts/run_hierarchical_contagion.py
```

Optimization subsets:
```bash
python scripts/optimize_hierarchical_parameters.py --locals-only
python scripts/optimize_hierarchical_parameters.py --meta-only
python scripts/optimize_hierarchical_parameters.py --direct-only
```

The pipeline auto-loads optimized parameters from `results/best_parameters_hierarchical_per_ticker.csv` when present, falling back to defaults in `src/config.py`.

---

*Working Paper — February 2026. Comments welcome: anoirluhalwe@yahoo.com*
