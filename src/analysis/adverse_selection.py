"""
Adverse Selection Estimation — Switching Logistic Regression
=============================================================

Estimates P(informed_i = 1 | OFI_i, OBI_i, regime=k) = sigmoid(alpha_k + beta_k*OFI + gamma_k*OBI)

where coefficients (alpha_k, beta_k, gamma_k) are estimated separately
per HMM regime k. The regime is NOT a regressor: it changes the SENSITIVITY
to OFI/OBI, eliminating redundancy with Wasserstein-derived regime features.

Pipeline:
1. Label each 500ms bar as "informed" if |price_impact(t+H)| > threshold
2. Estimate switching logistic regression per regime
3. Compare to PIN-static, VPIN-like, and pooled benchmarks (AUC)
4. Likelihood ratio test for coefficient homogeneity across regimes
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from src.config import RESULTS_DIR, FIGURES_DIR
from src.realtime.dual_clock_monitor import (
    RegimeCoefficients,
    calibrate_regime_coefficients,
    test_coefficient_homogeneity,
)


def build_adverse_selection_dataset(
    lob_data: pd.DataFrame,
    states: np.ndarray,
    horizon: int = 60,
    impact_threshold_ticks: float = 2.0,
    tick_size: float = 0.01,
) -> pd.DataFrame:
    """
    Build (OFI, OBI, regime, label) dataset from aligned LOBSTER data + HMM states.

    Parameters
    ----------
    lob_data : pd.DataFrame
        Must contain columns: micro_price, ofi, obi.
        Already time-aligned and resampled at 500ms.
    states : np.ndarray, shape (len(lob_data),)
        HMM regime assignment per bar, already aligned to lob_data index.
    horizon : int
        Future horizon (in 500ms bars) to measure price impact.
    impact_threshold_ticks : float
        Bars with |impact| > threshold * tick_size are labeled informed.
    tick_size : float

    Returns
    -------
    pd.DataFrame with columns: ofi, obi, regime, price_impact, is_informed
    """
    price = lob_data['micro_price'].values
    ofi = lob_data['ofi'].values
    obi = lob_data['obi'].values
    n = len(lob_data) - horizon

    records = []
    for t in range(n):
        impact = price[t + horizon] - price[t]
        records.append({
            't': t,
            'ofi': ofi[t],
            'obi': obi[t],
            'regime': int(states[t]),
            'price_impact': impact,
            'is_informed': float(abs(impact) > impact_threshold_ticks * tick_size),
        })

    df = pd.DataFrame(records)
    informed_rate = df['is_informed'].mean() * 100
    print(f"  Dataset: {len(df)} obs, informed rate: {informed_rate:.1f}%")
    return df


def estimate_adverse_selection_switching(
    df: pd.DataFrame,
    n_regimes: int,
) -> Tuple[Dict[int, RegimeCoefficients], pd.DataFrame, Dict]:
    """
    Estimate the switching model and compare to benchmarks.

    Returns
    -------
    coeffs : Dict[int, RegimeCoefficients]
    comparison_df : pd.DataFrame  — AUC table
    homogeneity_test : Dict       — LR test result
    """
    ofi = df['ofi'].values
    obi = df['obi'].values
    y = df['is_informed'].values
    states = df['regime'].values

    # Switching model
    coeffs = calibrate_regime_coefficients(ofi, obi, y, states, n_regimes)

    pi_switching = np.zeros(len(df))
    for k in range(n_regimes):
        mask = (states == k)
        if mask.sum() == 0:
            continue
        c = coeffs[k]
        logit = c.alpha + c.beta_ofi * ofi[mask] + c.gamma_obi * obi[mask]
        pi_switching[mask] = 1.0 / (1.0 + np.exp(-logit))

    auc_switching = roc_auc_score(y, pi_switching)

    # Benchmark: pooled logistic (OFI + OBI, no regime)
    X = np.column_stack([ofi, obi])
    model_pooled = LogisticRegression(max_iter=1000).fit(X, y)
    auc_pooled = roc_auc_score(y, model_pooled.predict_proba(X)[:, 1])

    # Benchmark: VPIN-like (OFI only)
    model_ofi = LogisticRegression(max_iter=1000).fit(ofi.reshape(-1, 1), y)
    auc_ofi = roc_auc_score(y, model_ofi.predict_proba(ofi.reshape(-1, 1))[:, 1])

    comparison_df = pd.DataFrame([
        {'Model': 'PIN static (Easley 1996)', 'AUC': 0.500, 'N_params': 1},
        {'Model': 'VPIN-like (OFI only)',     'AUC': auc_ofi,       'N_params': 2},
        {'Model': 'Pooled logistic (OFI+OBI)', 'AUC': auc_pooled,   'N_params': 3},
        {'Model': 'Switching (OFI+OBI | regime)', 'AUC': auc_switching,
         'N_params': 3 * n_regimes},
    ])

    print("\n" + "=" * 70)
    print("ADVERSE SELECTION MODEL COMPARISON")
    print("=" * 70)
    print(comparison_df.to_string(index=False))
    print(f"\nSwitching vs PIN static: +{(auc_switching - 0.5)/0.5*100:.1f}% AUC")
    print("=" * 70 + "\n")

    homogeneity_test = test_coefficient_homogeneity(ofi, obi, y, states, n_regimes)

    comparison_df.to_csv(RESULTS_DIR / 'adverse_selection_model_comparison.csv', index=False)
    pd.DataFrame([
        {'Regime': r, 'Alpha': c.alpha, 'Beta_OFI': c.beta_ofi, 'Gamma_OBI': c.gamma_obi}
        for r, c in coeffs.items()
    ]).to_csv(RESULTS_DIR / 'adverse_selection_coefficients.csv', index=False)

    return coeffs, comparison_df, homogeneity_test


def plot_adverse_selection_results(
    df: pd.DataFrame,
    coeffs: Dict[int, RegimeCoefficients],
    comparison_df: pd.DataFrame,
    n_regimes: int,
    ticker: str = '',
) -> None:
    """Four-panel figure: ROC curves, AUC bar, coefficient bar, pi(t) distributions."""
    ofi = df['ofi'].values
    obi = df['obi'].values
    y = df['is_informed'].values
    states = df['regime'].values

    pi_switching = np.zeros(len(df))
    for k in range(n_regimes):
        mask = (states == k)
        if mask.sum() == 0:
            continue
        c = coeffs[k]
        logit = c.alpha + c.beta_ofi * ofi[mask] + c.gamma_obi * obi[mask]
        pi_switching[mask] = 1.0 / (1.0 + np.exp(-logit))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    title_suffix = f' — {ticker}' if ticker else ''

    # Panel 1: ROC curves
    ax = axes[0, 0]
    fpr_sw, tpr_sw, _ = roc_curve(y, pi_switching)
    auc_sw = roc_auc_score(y, pi_switching)
    ax.plot(fpr_sw, tpr_sw, label=f'Switching (AUC={auc_sw:.3f})',
            linewidth=2.5, color='green')

    model_ofi = LogisticRegression(max_iter=1000).fit(ofi.reshape(-1, 1), y)
    pi_ofi = model_ofi.predict_proba(ofi.reshape(-1, 1))[:, 1]
    fpr_ofi, tpr_ofi, _ = roc_curve(y, pi_ofi)
    ax.plot(fpr_ofi, tpr_ofi,
            label=f'VPIN-like, OFI only (AUC={roc_auc_score(y, pi_ofi):.3f})',
            linewidth=2, color='orange', linestyle='--')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random (AUC=0.500)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curves{title_suffix}')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    # Panel 2: AUC bar chart
    ax = axes[0, 1]
    colors = ['gray', 'orange', 'steelblue', 'green']
    bars = ax.bar(range(len(comparison_df)), comparison_df['AUC'],
                  color=colors, alpha=0.8)
    ax.set_xticks(range(len(comparison_df)))
    ax.set_xticklabels(comparison_df['Model'], rotation=25, ha='right', fontsize=9)
    ax.set_ylabel('AUC')
    ax.set_title(f'Model Comparison: AUC{title_suffix}')
    ax.axhline(0.5, color='black', linestyle='--', alpha=0.3)
    ax.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, comparison_df['AUC']):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')

    # Panel 3: Coefficients per regime
    ax = axes[1, 0]
    regimes = list(coeffs.keys())
    betas = [coeffs[r].beta_ofi for r in regimes]
    gammas = [coeffs[r].gamma_obi for r in regimes]
    x = np.arange(len(regimes))
    w = 0.35
    ax.bar(x - w/2, betas, w, label='beta (OFI sensitivity)', color='steelblue')
    ax.bar(x + w/2, gammas, w, label='gamma (OBI sensitivity)', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Regime {r}' for r in regimes])
    ax.set_ylabel('Coefficient')
    ax.set_title(f'Heterogeneous Coefficients by Regime{title_suffix}')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Panel 4: pi(t) distribution per regime
    ax = axes[1, 1]
    for k in range(n_regimes):
        mask = (states == k)
        if mask.sum() == 0:
            continue
        ax.hist(pi_switching[mask], bins=30, alpha=0.5,
                label=f'Regime {k}', density=True)
    ax.set_xlabel('pi(t) — adverse selection probability')
    ax.set_ylabel('Density')
    ax.set_title(f'pi(t) Distribution by Regime{title_suffix}')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle(f'Adverse Selection Switching Model{title_suffix}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    suffix = f'_{ticker}' if ticker else ''
    out = FIGURES_DIR / f'adverse_selection{suffix}.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {out.name}")


def generate_latex_tables(
    comparison_df: pd.DataFrame,
    homogeneity_test: Dict,
    coeffs: Dict[int, RegimeCoefficients],
    ticker: str = '',
) -> None:
    """Generate LaTeX tables for comparison and regime coefficients."""
    suffix = f'_{ticker}' if ticker else ''

    # Table 1: model comparison
    rows = []
    for _, row in comparison_df.iterrows():
        model = row['Model']
        auc = f"{row['AUC']:.3f}"
        n_p = int(row['N_params'])
        if 'Switching' in model:
            model = f"\\textbf{{{model}}}"
            auc = f"\\textbf{{{auc}}}"
        rows.append(f"    {model} & {auc} & {n_p} \\\\")

    p_val = homogeneity_test['p_value']
    p_str = "< 0.001" if (not np.isnan(p_val) and p_val < 0.001) else \
            (f"{p_val:.4f}" if not np.isnan(p_val) else "N/A")
    reject_str = "Rejected" if homogeneity_test.get('reject_homogeneity') else "Not rejected"

    latex_cmp = (
        r"\begin{table}[htbp]" "\n"
        r"\centering" "\n"
        r"\caption{Adverse Selection Model Comparison}" "\n"
        r"\label{tab:adverse_selection_comparison}" "\n"
        r"\begin{tabular}{lcc}" "\n"
        r"\toprule" "\n"
        r"\textbf{Model} & \textbf{AUC} & \textbf{Parameters} \\" "\n"
        r"\midrule" "\n"
        + "\n".join(rows) + "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\begin{tablenotes}\small" "\n"
        r"\item LR homogeneity test (H$_0$: identical coefficients across regimes):"
        f" statistic = {homogeneity_test['lr_statistic']:.2f},"
        f" $p$ = {p_str} ({reject_str})."
        "\n" r"\end{tablenotes}" "\n"
        r"\end{table}" "\n"
    )

    out1 = RESULTS_DIR / f'table_adverse_selection_comparison{suffix}.tex'
    out1.write_text(latex_cmp, encoding='utf-8')

    # Table 2: regime coefficients
    coeff_rows = []
    for r, c in coeffs.items():
        coeff_rows.append(
            f"    {r} & {c.alpha:+.3f} & {c.beta_ofi:+.3f} & {c.gamma_obi:+.3f} \\\\"
        )

    latex_coef = (
        r"\begin{table}[htbp]" "\n"
        r"\centering" "\n"
        r"\caption{Regime-Dependent Adverse Selection Coefficients}" "\n"
        r"\label{tab:regime_coefficients}" "\n"
        r"\begin{tabular}{cccc}" "\n"
        r"\toprule" "\n"
        r"\textbf{Regime} & $\alpha_k$ & $\beta_k$ (OFI) & $\gamma_k$ (OBI) \\" "\n"
        r"\midrule" "\n"
        + "\n".join(coeff_rows) + "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\begin{tablenotes}\small" "\n"
        r"\item $P(\text{informed}=1 \mid \text{OFI}, \text{OBI}, \varepsilon=k) ="
        r" \sigma(\alpha_k + \beta_k \cdot \text{OFI} + \gamma_k \cdot \text{OBI})$."
        "\n" r"\end{tablenotes}" "\n"
        r"\end{table}" "\n"
    )

    out2 = RESULTS_DIR / f'table_adverse_selection_coefficients{suffix}.tex'
    out2.write_text(latex_coef, encoding='utf-8')

    print(f"  LaTeX tables saved: {out1.name}, {out2.name}")
