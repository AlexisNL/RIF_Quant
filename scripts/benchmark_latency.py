"""
Latency Benchmark for Dual-Clock Architecture
================================================

Valide empiriquement les affirmations de latence faites dans l'article :
- FAST PATH ~150-200ns
- SLOW PATH ~10us
- Latence amortie ~250-340ns (proche de VPIN ~100ns)

Compare aussi à un calcul "naïf" qui recalculerait Wasserstein+HMM
à CHAQUE tick (l'approche initiale, sans dual-clock).
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from typing import Dict

from src.config import RESULTS_DIR, FIGURES_DIR
from src.realtime.dual_clock_monitor import (
    DualClockAdverseSelectionMonitor,
    RegimeCoefficients,
    WassersteinStreamingEstimator,
    OnlineHMMFilter,
    update_obi_ofi_incremental
)


def simulate_lob_stream(n_events: int, seed: int = 42) -> list:
    """
    Génère un flux simulé d'événements LOB pour le benchmark.

    Inclut une anomalie artificielle au milieu (simule un flash crash)
    pour tester le trigger adaptatif.
    """
    rng = np.random.default_rng(seed)

    events = []
    bid, ask = 100.00, 100.01
    bid_vol, ask_vol = 1000.0, 1000.0

    crash_start = n_events // 2
    crash_duration = 200

    for i in range(n_events):

        if crash_start <= i < crash_start + crash_duration:
            # Anomalie simulée : volumes très déséquilibrés
            bid_vol = max(10.0, bid_vol + rng.normal(-50, 30))
            ask_vol = max(10.0, ask_vol + rng.normal(80, 40))
        else:
            # Régime normal : faible bruit
            bid_vol = max(100.0, bid_vol + rng.normal(0, 15))
            ask_vol = max(100.0, ask_vol + rng.normal(0, 15))

        prev_bid, prev_ask = bid, ask
        bid = round(bid + rng.normal(0, 0.001), 4)
        ask = round(max(bid + 0.01, ask + rng.normal(0, 0.001)), 4)

        events.append({
            'bid': bid, 'ask': ask,
            'bid_vol': bid_vol, 'ask_vol': ask_vol,
            'prev_bid': prev_bid, 'prev_ask': prev_ask
        })

    return events


def benchmark_naive_approach(events: list) -> Dict:
    """
    Approche NAÏVE : recalcule Wasserstein + HMM à CHAQUE tick.
    C'est ce que vous craigniez initialement — trop lent pour le HFT.
    """

    n_regimes = 3
    window = 100

    wass_ofi = WassersteinStreamingEstimator(window=window)
    wass_obi = WassersteinStreamingEstimator(window=window)

    transmat = np.array([[0.85, 0.10, 0.05],
                          [0.08, 0.82, 0.10],
                          [0.15, 0.05, 0.80]])
    means = np.array([[0.3, 0.3], [0.6, 0.6], [1.2, 1.0]])
    covars = np.array([np.eye(2) * 0.2 for _ in range(n_regimes)])

    hmm_filter = OnlineHMMFilter(transmat, means, covars)

    prev_bid_vol, prev_ask_vol = 1000.0, 1000.0

    latencies_ns = []

    for event in events:
        t0 = time.perf_counter_ns()

        obi, ofi = update_obi_ofi_incremental(
            event['bid_vol'], event['ask_vol'],
            prev_bid_vol, prev_ask_vol,
            event['prev_bid'], event['prev_ask'],
            event['bid'], event['ask']
        )

        # TOUJOURS recalculer Wasserstein + HMM (approche naïve)
        wass_ofi.push(ofi)
        wass_obi.push(obi)

        if wass_ofi.is_ready():
            w_ofi = wass_ofi.compute()
            w_obi = wass_obi.compute()
            features = np.array([w_ofi, w_obi])
            hmm_filter.update(features)

        prev_bid_vol, prev_ask_vol = event['bid_vol'], event['ask_vol']

        t1 = time.perf_counter_ns()
        latencies_ns.append(t1 - t0)

    return {
        'latencies_ns': np.array(latencies_ns),
        'mean_ns': np.mean(latencies_ns),
        'median_ns': np.median(latencies_ns),
        'p99_ns': np.percentile(latencies_ns, 99)
    }


def benchmark_dual_clock_approach(events: list) -> Dict:
    """
    Approche DUAL-CLOCK : FAST PATH systématique + SLOW PATH adaptatif.
    """

    n_regimes = 3

    coeffs = {
        0: RegimeCoefficients(alpha=-1.2, beta_ofi=0.3, gamma_obi=-0.2),
        1: RegimeCoefficients(alpha=-0.8, beta_ofi=0.6, gamma_obi=-0.5),
        2: RegimeCoefficients(alpha=-0.3, beta_ofi=1.2, gamma_obi=-0.9),
    }

    transmat = np.array([[0.85, 0.10, 0.05],
                          [0.08, 0.82, 0.10],
                          [0.15, 0.05, 0.80]])
    means = np.array([[0.3, 0.3], [0.6, 0.6], [1.2, 1.0]])
    covars = np.array([np.eye(2) * 0.2 for _ in range(n_regimes)])

    monitor = DualClockAdverseSelectionMonitor(
        regime_coeffs=coeffs,
        hmm_transmat=transmat,
        hmm_means=means,
        hmm_covars=covars,
        wasserstein_window=100,
        anomaly_zscore_threshold=3.0
    )

    latencies_ns = []
    slow_path_triggers = []

    for i, event in enumerate(events):
        t0 = time.perf_counter_ns()

        result = monitor.on_lob_event(event)

        t1 = time.perf_counter_ns()
        latencies_ns.append(t1 - t0)
        slow_path_triggers.append(result['slow_path_triggered'])

    return {
        'latencies_ns': np.array(latencies_ns),
        'mean_ns': np.mean(latencies_ns),
        'median_ns': np.median(latencies_ns),
        'p99_ns': np.percentile(latencies_ns, 99),
        'slow_path_trigger_rate': np.mean(slow_path_triggers),
        'stats': monitor.stats,
        'slow_path_triggers': np.array(slow_path_triggers)
    }


def benchmark_vpin_baseline(events: list) -> Dict:
    """
    Benchmark VPIN simplifié (O(1) pur) pour comparaison de référence.
    """

    latencies_ns = []
    buy_vol_bucket, sell_vol_bucket = 0.0, 0.0

    for event in events:
        t0 = time.perf_counter_ns()

        # VPIN simplifié : classification bulk volume O(1)
        delta_bid = event['bid_vol']
        delta_ask = event['ask_vol']

        buy_vol_bucket += max(0, delta_bid - delta_ask)
        sell_vol_bucket += max(0, delta_ask - delta_bid)

        total = buy_vol_bucket + sell_vol_bucket + 1e-9
        vpin = abs(buy_vol_bucket - sell_vol_bucket) / total

        t1 = time.perf_counter_ns()
        latencies_ns.append(t1 - t0)

    return {
        'latencies_ns': np.array(latencies_ns),
        'mean_ns': np.mean(latencies_ns),
        'median_ns': np.median(latencies_ns),
        'p99_ns': np.percentile(latencies_ns, 99)
    }


def run_full_latency_benchmark(n_events: int = 50000) -> pd.DataFrame:
    """
    Exécute le benchmark complet et génère le rapport comparatif.

    C'est la fonction à appeler pour produire les chiffres exacts
    de l'article (remplacer les estimations théoriques par des
    mesures empiriques réelles sur votre machine).
    """

    print("\n" + "=" * 80)
    print("BENCHMARK DE LATENCE — VALIDATION EMPIRIQUE")
    print("=" * 80)
    print(f"Nombre d'événements simulés : {n_events}")
    print("Inclut une anomalie simulée (flash crash) au milieu du flux\n")

    events = simulate_lob_stream(n_events)

    print("[1/3] Benchmark VPIN baseline (O(1) pur)...")
    results_vpin = benchmark_vpin_baseline(events)

    print("[2/3] Benchmark approche NAÏVE (Wasserstein+HMM à chaque tick)...")
    results_naive = benchmark_naive_approach(events)

    print("[3/3] Benchmark architecture DUAL-CLOCK (votre contribution)...")
    results_dual = benchmark_dual_clock_approach(events)

    # Tableau comparatif
    comparison = pd.DataFrame([
        {
            'Method': 'VPIN baseline',
            'Mean_ns': results_vpin['mean_ns'],
            'Median_ns': results_vpin['median_ns'],
            'P99_ns': results_vpin['p99_ns'],
            'Slow_path_rate': 0.0
        },
        {
            'Method': 'Naive (Wasserstein+HMM every tick)',
            'Mean_ns': results_naive['mean_ns'],
            'Median_ns': results_naive['median_ns'],
            'P99_ns': results_naive['p99_ns'],
            'Slow_path_rate': 1.0
        },
        {
            'Method': 'Dual-clock (adaptive trigger)',
            'Mean_ns': results_dual['mean_ns'],
            'Median_ns': results_dual['median_ns'],
            'P99_ns': results_dual['p99_ns'],
            'Slow_path_rate': results_dual['slow_path_trigger_rate']
        }
    ])

    comparison['Ratio_vs_VPIN'] = comparison['Mean_ns'] / results_vpin['mean_ns']

    print("\n" + "=" * 80)
    print("RÉSULTATS DU BENCHMARK")
    print("=" * 80)
    print(comparison.to_string(index=False))

    print(f"\n{results_dual['stats'].summary()}")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    ratio_naive = comparison.loc[1, 'Ratio_vs_VPIN']
    ratio_dual = comparison.loc[2, 'Ratio_vs_VPIN']
    print(f"Approche naïve  : {ratio_naive:.1f}x plus lente que VPIN")
    print(f"Dual-clock      : {ratio_dual:.1f}x plus lente que VPIN")
    print(f"Gain dual-clock vs naïve : {ratio_naive/ratio_dual:.1f}x plus rapide")
    print("=" * 80 + "\n")

    # Sauvegarde
    comparison.to_csv(RESULTS_DIR / 'latency_benchmark_results.csv', index=False)

    # Visualisation
    _plot_latency_benchmark(results_vpin, results_naive, results_dual, comparison)

    return comparison


def _plot_latency_benchmark(results_vpin, results_naive, results_dual, comparison):
    """Génère le graphique de comparaison des latences."""

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1 : Barres comparatives (échelle log)
    ax1 = axes[0]
    colors = ['orange', 'red', 'green']
    bars = ax1.bar(comparison['Method'], comparison['Mean_ns'], color=colors, alpha=0.8)
    ax1.set_yscale('log')
    ax1.set_ylabel('Latence moyenne (ns, échelle log)', fontweight='bold')
    ax1.set_title('Comparaison de latence par méthode', fontweight='bold')
    ax1.set_xticklabels(comparison['Method'], rotation=20, ha='right')
    ax1.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, comparison['Mean_ns']):
        ax1.text(bar.get_x() + bar.get_width()/2, val * 1.1,
                  f'{val:.0f} ns', ha='center', fontweight='bold')

    # Panel 2 : Latence dual-clock dans le temps (montre le trigger sur anomalie)
    ax2 = axes[1]
    latencies = results_dual['latencies_ns']
    triggers = results_dual['slow_path_triggers']

    ax2.scatter(np.arange(len(latencies))[~triggers], latencies[~triggers],
                s=2, alpha=0.3, color='green', label='Fast path')
    ax2.scatter(np.arange(len(latencies))[triggers], latencies[triggers],
                s=8, alpha=0.8, color='red', label='Slow path (anomalie détectée)')

    ax2.set_yscale('log')
    ax2.set_xlabel('Index événement', fontweight='bold')
    ax2.set_ylabel('Latence (ns, échelle log)', fontweight='bold')
    ax2.set_title('Architecture dual-clock : latence au fil du temps\n'
                   '(pic = trigger sur anomalie simulée type flash-crash)',
                   fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'latency_benchmark.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("OK Latency benchmark plot saved")


if __name__ == "__main__":
    run_full_latency_benchmark(n_events=50000)
