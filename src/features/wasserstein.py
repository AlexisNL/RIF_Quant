"""
Wasserstein distances with optional Numba acceleration.
"""

from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

# Try Numba
try:
    from src.features.wasserstein_optimized import compute_pairwise_wasserstein_numba

    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False


def compute_wasserstein_distances(
    innov_dict: Dict,
    tickers: List[str],
    window: int = 100,
    use_numba: bool = True,
) -> Tuple[np.ndarray, Dict, pd.Index]:
    """
    Compute Wasserstein distances with optional Numba (faster, same results).
    """
    if use_numba and NUMBA_AVAILABLE:
        print("  -> Numba acceleration enabled")
        return _compute_numba(innov_dict, tickers, window)
    return _compute_scipy(innov_dict, tickers, window)


def compute_wasserstein_features(
    innov_dict: Dict,
    tickers: List[str],
    window: int = 100,
    use_numba: bool = True,
) -> pd.DataFrame:
    """
    Return a feature DataFrame with columns "{ticker}_{Metric}".

    Metrics are mapped as:
    - price_ret -> Price
    - obi -> OBI
    - ofi -> OFI
    """
    wass_X, _, idx = compute_wasserstein_distances(
        innov_dict,
        tickers,
        window=window,
        use_numba=use_numba,
    )

    metric_order = ["price_ret", "obi", "ofi"]
    metric_label = {"price_ret": "Price", "obi": "OBI", "ofi": "OFI"}
    columns = []
    for metric in metric_order:
        label = metric_label[metric]
        for ticker in tickers:
            columns.append(f"{ticker}_{label}")

    return pd.DataFrame(wass_X, index=idx, columns=columns)


def compute_wasserstein_temporal_features(
    innov_dict: Dict,
    tickers: List[str],
    window: int = 100,
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute temporal Wasserstein distances (before vs after) per ticker/metric.

    For each series, at time t we compute:
        W( series[t-window:t], series[t:t+window] )
    """
    if metrics is None:
        metrics = ["price_ret", "obi", "ofi"]

    metric_label = {"price_ret": "Price", "obi": "OBI", "ofi": "OFI"}

    # Build aligned DataFrames and a common index
    metric_dfs = {}
    for m in metrics:
        metric_dfs[m] = pd.DataFrame({t: innov_dict[t][m] for t in tickers}).dropna()

    common_index = metric_dfs[metrics[0]].index
    for m in metrics[1:]:
        common_index = common_index.intersection(metric_dfs[m].index)

    if len(common_index) < 2 * window + 1:
        raise ValueError("Not enough observations for temporal Wasserstein window.")

    data_by_metric = {m: metric_dfs[m].loc[common_index] for m in metrics}

    features = {}
    for m in metrics:
        df = data_by_metric[m]
        for t in tickers:
            series = df[t].values
            temporal_vals = _compute_temporal_wasserstein_series(series, window)
            col_name = f"{t}_{metric_label.get(m, m)}"
            features[col_name] = temporal_vals

    # Index aligns to the center time t
    out_index = common_index[window:-window]
    return pd.DataFrame(features, index=out_index)


def compute_wasserstein_temporal_decomposed(
    innov_dict: Dict,
    tickers: List[str],
    window: int = 100,
    metrics: Optional[List[str]] = None,
) -> Dict:
    """
    Return decomposed temporal Wasserstein distances by metric and ticker.

    Structure: decomposed[metric][ticker] -> list of distances.
    """
    if metrics is None:
        metrics = ["price_ret", "obi", "ofi"]

    metric_dfs = {}
    for m in metrics:
        metric_dfs[m] = pd.DataFrame({t: innov_dict[t][m] for t in tickers}).dropna()

    common_index = metric_dfs[metrics[0]].index
    for m in metrics[1:]:
        common_index = common_index.intersection(metric_dfs[m].index)

    if len(common_index) < 2 * window + 1:
        raise ValueError("Not enough observations for temporal Wasserstein window.")

    decomposed = {m: {t: [] for t in tickers} for m in metrics}
    for m in metrics:
        df = metric_dfs[m].loc[common_index]
        for t in tickers:
            series = df[t].values
            decomposed[m][t] = _compute_temporal_wasserstein_series(series, window).tolist()

    return decomposed


def _compute_temporal_wasserstein_series(series: np.ndarray, window: int) -> np.ndarray:
    """
    Compute temporal Wasserstein distance for a single series.
    """
    n = len(series)
    if n < 2 * window + 1:
        return np.array([])

    out = np.zeros(n - 2 * window)
    for i in range(window, n - window):
        past = series[i - window : i]
        future = series[i : i + window]
        out[i - window] = wasserstein_distance(past, future)
    return out


def decompose_wasserstein_by_ticker(
    innov_dict: Dict,
    tickers: List[str],
    window: int = 100,
    use_numba: bool = True,
) -> Dict:
    """
    Return decomposed Wasserstein distances by metric and ticker.

    Structure: decomposed[metric][ticker] -> list of distances.
    """
    _, decomposed, _ = compute_wasserstein_distances(
        innov_dict,
        tickers,
        window=window,
        use_numba=use_numba,
    )
    return decomposed


def _compute_scipy(innov_dict, tickers, window):
    """Scipy implementation."""
    metric_dfs = {}
    for m in ["price_ret", "obi", "ofi"]:
        metric_dfs[m] = pd.DataFrame({t: innov_dict[t][m] for t in tickers}).dropna()

    common_index = metric_dfs["price_ret"].index
    for m in ["obi", "ofi"]:
        common_index = common_index.intersection(metric_dfs[m].index)

    all_features = []
    decomposed = {m: {t: [] for t in tickers} for m in ["price_ret", "obi", "ofi"]}

    for m in ["price_ret", "obi", "ofi"]:
        df = metric_dfs[m].loc[common_index]
        n = len(df)
        m_wass = []

        for i in range(window, n):
            slice_df = df.iloc[i - window : i]
            row = []

            for t1 in tickers:
                dist = np.mean(
                    [
                        wasserstein_distance(slice_df[t1], slice_df[t2])
                        for t2 in tickers
                        if t1 != t2
                    ]
                )
                row.append(dist)
                decomposed[m][t1].append(dist)

            m_wass.append(row)

        all_features.append(np.array(m_wass))

    wass_X = np.hstack(all_features)
    return wass_X, decomposed, common_index[window:]


def _compute_numba(innov_dict, tickers, window):
    """Numba implementation (faster)."""
    from src.features.wasserstein_optimized import compute_pairwise_wasserstein_numba

    metric_dfs = {}
    for m in ["price_ret", "obi", "ofi"]:
        metric_dfs[m] = pd.DataFrame({t: innov_dict[t][m] for t in tickers}).dropna()

    common_index = metric_dfs["price_ret"].index
    for m in ["obi", "ofi"]:
        common_index = common_index.intersection(metric_dfs[m].index)

    all_features = []
    decomposed = {m: {t: [] for t in tickers} for m in ["price_ret", "obi", "ofi"]}

    for m in ["price_ret", "obi", "ofi"]:
        data = metric_dfs[m].loc[common_index].values
        n = len(data)
        m_wass = []

        for i in range(window, n):
            window_data = data[i - window : i, :]
            avg_dist = compute_pairwise_wasserstein_numba(window_data)

            m_wass.append(avg_dist)
            for j, t in enumerate(tickers):
                decomposed[m][t].append(avg_dist[j])

        all_features.append(np.array(m_wass))

    wass_X = np.hstack(all_features)
    return wass_X, decomposed, common_index[window:]
