"""
Normalisation robuste par MAD (Median Absolute Deviation)
==========================================================

Approche robuste aux outliers.

MAD (Median Absolute Deviation) :
    MAD = median(|X - median(X)|)
    Z-score robuste = (X - median(X)) / (1.4826 * MAD)

Avantages :
- Plus robuste aux outliers
- Plus rapide (pas d'optimisation itérative)
- Fenêtre glissante simple
- Pas de problèmes de convergence

Fenêtre glissante optimisable (nouveau paramètre).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')


def compute_mad(x: np.ndarray) -> float:
    """
    Calcule le MAD (Median Absolute Deviation).

    MAD = median(|X - median(X)|)

    Args:
        x: Série temporelle

    Returns:
        MAD value (robuste aux outliers)
    """
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    return mad


def normalize_series_mad(
    series: pd.Series,
    window: int = 100,
    min_periods: int = 50
) -> pd.Series:
    """
    Normalise une série par Z-score robuste (MAD).

    Z-score robuste = (X - median) / (1.4826 * MAD)

    Le facteur 1.4826 rend MAD équivalent à l'écart-type
    pour une distribution normale.

    Args:
        series: Série temporelle à normaliser
        window: Taille de la fenêtre glissante
        min_periods: Nombre minimum d'observations

    Returns:
        Série normalisée (Z-scores robustes)
    """
    # Calcul de la médiane glissante
    rolling_median = series.rolling(
        window=window,
        min_periods=min_periods,
        center=False
    ).median()

    # Calcul du MAD glissant
    def rolling_mad(x):
        if len(x) < min_periods:
            return np.nan
        return compute_mad(x.values)

    rolling_mad_values = series.rolling(
        window=window,
        min_periods=min_periods,
        center=False
    ).apply(rolling_mad, raw=False)

    # Z-score robuste
    # Facteur 1.4826 pour équivalence avec std sous normalité
    z_score = (series - rolling_median) / (1.4826 * rolling_mad_values + 1e-9)

    return z_score


def normalize_innovations_mad(
    synced_data: Dict[str, pd.DataFrame],
    tickers: List[str],
    window: int = 100,
    min_periods: int = 50,
    n_jobs: int = None
) -> Dict[str, pd.DataFrame]:
    """
    Normalise toutes les séries (price, OBI, OFI) par MAD.

    Parallélisé pour performance.

    Args:
        synced_data: Dict {ticker: DataFrame} avec colonnes price_ret, obi, ofi
        tickers: Liste des tickers
        window: Fenêtre glissante pour MAD
        min_periods: Observations minimum
        n_jobs: Nombre de workers (None = auto)

    Returns:
        Dict {ticker: DataFrame} avec innovations normalisées
    """
    print(f"  Normalisation MAD avec fenêtre = {window} ({window*0.5:.1f}s)")

    # Prepare tasks (n_tickers x 3 metrics)
    tasks = []
    for ticker in tickers:
        df = synced_data[ticker]

        if "price_ret" in df.columns:
            price_ret = df["price_ret"]
        elif "micro_price" in df.columns:
            price_ret = np.log(df["micro_price"]).diff() * 100
        else:
            raise KeyError(f"{ticker}: missing price_ret or micro_price")

        metrics_map = {
            "price_ret": price_ret,
            "obi": df["obi"],
            "ofi": df["ofi"],
        }

        for metric, series in metrics_map.items():
            tasks.append((ticker, metric, series, window, min_periods))

    # Parallélisation
    if n_jobs is None:
        n_jobs = min(len(tasks), 8)  # Max 8 workers

    results = {}

    def process_series(task):
        ticker, metric, series, win, min_per = task
        normalized = normalize_series_mad(series, window=win, min_periods=min_per)
        return (ticker, metric, normalized)

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(process_series, task) for task in tasks]

        for future in futures:
            ticker, metric, normalized = future.result()

            if ticker not in results:
                results[ticker] = {}

            results[ticker][metric] = normalized

    # Conversion en DataFrames
    innov_dict = {}
    for ticker in tickers:
        innov_dict[ticker] = pd.DataFrame({
            'price_ret': results[ticker]['price_ret'],
            'obi': results[ticker]['obi'],
            'ofi': results[ticker]['ofi']
        })

    # Statistiques
    n_total = len(tickers) * 3
    n_valid = sum([innov_dict[t][m].notna().sum() > 0
                   for t in tickers for m in ['price_ret', 'obi', 'ofi']])

    print(f"  ✓ {n_valid}/{n_total} séries normalisées avec succès")

    return innov_dict




def validate_mad_stationarity(
    innov_dict: Dict[str, pd.DataFrame],
    tickers: List[str]
) -> pd.DataFrame:
    """
    Valide la stationnarité des innovations MAD (test ADF).

    Args:
        innov_dict: Innovations normalisées
        tickers: Liste des tickers

    Returns:
        DataFrame avec résultats ADF
    """
    from statsmodels.tsa.stattools import adfuller

    print("\nValidation de la stationnarité (test ADF)...")

    results = []

    for ticker in tickers:
        for metric in ['price_ret', 'obi', 'ofi']:
            series = innov_dict[ticker][metric].dropna()

            if len(series) > 100:
                try:
                    adf_result = adfuller(series, maxlag=20)

                    results.append({
                        'ticker': ticker,
                        'metric': metric,
                        'adf_stat': adf_result[0],
                        'p_value': adf_result[1],
                        'stationary': adf_result[1] < 0.05,
                        'n_obs': len(series)
                    })
                except:
                    pass

    results_df = pd.DataFrame(results)

    n_stationary = results_df['stationary'].sum()
    n_total = len(results_df)

    print(f"  ✓ {n_stationary}/{n_total} séries stationnaires (p < 0.05)")

    return results_df


if __name__ == "__main__":
    """Test du module."""

    # Test sur série synthétique
    np.random.seed(42)

    # Série avec outliers
    n = 1000
    x = np.random.randn(n)
    x[100] = 10  # Outlier
    x[500] = -8  # Outlier

    series = pd.Series(x)

    # Normalisation
    normalized = normalize_series_mad(series, window=100)

    print("Test sur série synthétique :")
    print(f"  MAD = {compute_mad(x):.3f}")
    print(f"  Std = {np.std(x):.3f}")
    print(f"  Valeurs normalisées : mean={normalized.mean():.3f}, std={normalized.std():.3f}")
    print(f"  Outliers réduits : max={normalized.max():.2f}, min={normalized.min():.2f}")
