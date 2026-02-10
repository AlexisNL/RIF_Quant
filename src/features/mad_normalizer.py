"""
Robust MAD normalization (Median Absolute Deviation)
===================================================

Outlier-robust normalization.

MAD (Median Absolute Deviation):
    MAD = median(|X - median(X)|)
    Robust z-score = (X - median(X)) / (1.4826 * MAD)

Benefits:
- More robust to outliers
- Faster (no iterative optimization)
- Simple rolling window
- No convergence issues

Rolling window is tunable (new parameter).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')


def compute_mad(x: np.ndarray) -> float:
    """
    Compute MAD (Median Absolute Deviation).

    MAD = median(|X - median(X)|)

    Args:
        x: Time series

    Returns:
        MAD value (outlier-robust)
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
    Normalize a series using a robust z-score (MAD).

    Robust z-score = (X - median) / (1.4826 * MAD)

    The 1.4826 factor makes MAD comparable to standard deviation
    under a normal distribution.

    Args:
        series: Time series to normalize
        window: Rolling window size
        min_periods: Minimum observations

    Returns:
        Normalized series (robust z-scores)
    """
    # Rolling median
    rolling_median = series.rolling(
        window=window,
        min_periods=min_periods,
        center=False
    ).median()

    # Rolling MAD
    def rolling_mad(x):
        if len(x) < min_periods:
            return np.nan
        return compute_mad(x.values)

    rolling_mad_values = series.rolling(
        window=window,
        min_periods=min_periods,
        center=False
    ).apply(rolling_mad, raw=False)

    # Robust z-score
    # 1.4826 factor for equivalence with std under normality
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
    Normalize all series (price, OBI, OFI) using MAD.

    Parallelized for performance.

    Args:
        synced_data: Dict {ticker: DataFrame} with columns price_ret, obi, ofi
        tickers: List of tickers
        window: Rolling window for MAD
        min_periods: Minimum observations
        n_jobs: Number of workers (None = auto)

    Returns:
        Dict {ticker: DataFrame} with normalized innovations
    """
    print(f"  MAD normalization with window = {window} ({window*0.5:.1f}s)")

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

    # Parallelization
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

    # Convert to DataFrames
    innov_dict = {}
    for ticker in tickers:
        innov_dict[ticker] = pd.DataFrame({
            'price_ret': results[ticker]['price_ret'],
            'obi': results[ticker]['obi'],
            'ofi': results[ticker]['ofi']
        })

    # Statistics
    n_total = len(tickers) * 3
    n_valid = sum([innov_dict[t][m].notna().sum() > 0
                   for t in tickers for m in ['price_ret', 'obi', 'ofi']])

    print(f"  OK {n_valid}/{n_total} series normalized successfully")

    return innov_dict



def validate_mad_stationarity(
    innov_dict: Dict[str, pd.DataFrame],
    tickers: List[str]
) -> pd.DataFrame:
    """
    Validate stationarity of MAD innovations (ADF test).

    Args:
        innov_dict: Normalized innovations
        tickers: List of tickers

    Returns:
        DataFrame with ADF results
    """
    from statsmodels.tsa.stattools import adfuller

    print("
Stationarity validation (ADF test)...")

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
                except Exception:
                    pass

    results_df = pd.DataFrame(results)

    n_stationary = results_df['stationary'].sum()
    n_total = len(results_df)

    print(f"  OK {n_stationary}/{n_total} stationary series (p < 0.05)")

    return results_df


if __name__ == "__main__":
    """Module test."""

    # Test on synthetic series
    np.random.seed(42)

    # Series with outliers
    n = 1000
    x = np.random.randn(n)
    x[100] = 10  # Outlier
    x[500] = -8  # Outlier

    series = pd.Series(x)

    # Normalization
    normalized = normalize_series_mad(series, window=100)

    print("Test on synthetic series:")
    print(f"  MAD = {compute_mad(x):.3f}")
    print(f"  Std = {np.std(x):.3f}")
    print(f"  Normalized values: mean={normalized.mean():.3f}, std={normalized.std():.3f}")
    print(f"  Reduced outliers: max={normalized.max():.2f}, min={normalized.min():.2f}")
