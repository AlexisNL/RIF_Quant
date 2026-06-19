from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

# Internal metric key -> output column label (e.g., "AAPL_Price")
_METRICS: Dict[str, str] = {
    "price_ret": "Price",
    "obi": "OBI",
    "ofi": "OFI",
}


class WassersteinExtractor:
    """
    Temporal Wasserstein distances as a LOB stress proxy.

    For each ticker and metric at time t:

        W(series[t-window:t], series[t:t+window])

    This captures local distribution shifts without relying on correlation.

    Parameters
    ----------
    window : int
        Rolling half-window size (observations).
    """

    def __init__(self, window: int = 100) -> None:
        self.window = window

    def compute_features(
        self,
        innov_dict: Dict[str, pd.DataFrame],
        tickers: List[str],
        metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute temporal Wasserstein features for all tickers and metrics.

        Parameters
        ----------
        innov_dict : dict
            {ticker: DataFrame} with columns price_ret, obi, ofi.
        tickers : list of str
            Ordered ticker list.
        metrics : list of str or None
            Optional subset of ["price_ret", "obi", "ofi"].
            Defaults to all metrics.

        Returns
        -------
        pd.DataFrame
            Shape (T - 2*window, n_tickers * n_metrics).
            Columns: "<ticker>_<Label>" (e.g., "AAPL_Price").
        """
        if metrics is None:
            metrics = list(_METRICS.keys())

        metric_dfs = self._build_metric_dfs(innov_dict, tickers, metrics)
        common_index = self._common_index(metric_dfs, metrics)

        if len(common_index) < 2 * self.window + 1:
            raise ValueError(
                f"Not enough observations ({len(common_index)}) for "
                f"temporal Wasserstein window={self.window}."
            )

        features: Dict[str, np.ndarray] = {}
        for metric in metrics:
            df = metric_dfs[metric].loc[common_index]
            label = _METRICS.get(metric, metric)
            for ticker in tickers:
                features[f"{ticker}_{label}"] = self.compute_temporal_series(
                    df[ticker].values
                )

        out_index = common_index[self.window : -self.window]
        return pd.DataFrame(features, index=out_index)

    def compute_temporal_series(self, series: np.ndarray) -> np.ndarray:
        """
        Compute temporal before-vs-after Wasserstein distance for one series.

        At index i (for i in [window, n - window)):
            W(series[i-window:i], series[i:i+window])

        Returns
        -------
        np.ndarray, shape (n - 2*window,)
        """
        n = len(series)
        w = self.window
        if n < 2 * w + 1:
            return np.array([])

        out = np.zeros(n - 2 * w)
        for i in range(w, n - w):
            out[i - w] = wasserstein_distance(series[i - w : i], series[i : i + w])
        return out

    def _build_metric_dfs(
        self,
        innov_dict: Dict[str, pd.DataFrame],
        tickers: List[str],
        metrics: List[str],
    ) -> Dict[str, pd.DataFrame]:
        return {
            metric: pd.DataFrame({ticker: innov_dict[ticker][metric] for ticker in tickers}).dropna()
            for metric in metrics
        }

    @staticmethod
    def _common_index(
        metric_dfs: Dict[str, pd.DataFrame],
        metrics: List[str],
    ) -> pd.Index:
        idx = metric_dfs[metrics[0]].index
        for metric in metrics[1:]:
            idx = idx.intersection(metric_dfs[metric].index)
        return idx


def compute_tick_wasserstein_causal(
    series: np.ndarray,
    window: int = 100,
    stride: int = 50,
) -> np.ndarray:
    """
    Causal (one-sided) Wasserstein distance at tick resolution with stride.

    At each sampled position i (i >= 2*window):
        W(series[i - 2w : i - w], series[i - w : i])

    Uses **only past data** up to tick i — no lookahead.
    Suitable for backtesting and live inference.

    Output alignment
    ----------------
    output[j] is computed at original tick index j + 2*window.
    Output length = n - 2*window (same as the non-causal variant so that
    downstream code can swap between the two without alignment changes).

    Parameters
    ----------
    series : np.ndarray
        Raw tick-level signal (OFI, OBI, or price_ret).
    window : int
        Half-window size in ticks.
    stride : int
        Step between consecutive evaluations; forward-filled between positions.

    Returns
    -------
    np.ndarray, shape (n - 2*window,)
    """
    n = len(series)
    valid_n = n - 2 * window
    if valid_n <= 0:
        return np.array([])

    out = np.full(valid_n, np.nan)
    for pos in range(0, valid_n, stride):
        i = pos + 2 * window
        out[pos] = wasserstein_distance(
            series[i - 2 * window : i - window],
            series[i - window : i],
        )

    last_val = out[0] if not np.isnan(out[0]) else 0.0
    for k in range(valid_n):
        if np.isnan(out[k]):
            out[k] = last_val
        else:
            last_val = out[k]

    return out


def compute_tick_wasserstein(
    series: np.ndarray,
    window: int = 100,
    stride: int = 50,
) -> np.ndarray:
    """
    Compute temporal Wasserstein distances at tick resolution with stride.

    At each sampled position i (stepped by ``stride``):
        W(series[i-window:i], series[i:i+window])

    Values between sampled positions are forward-filled, so the output
    has the same length as the valid range ``(n - 2*window)`` of the
    input, with one Wasserstein computation per ``stride`` ticks.

    This reduces O(N) Wasserstein calls to O(N/stride) while keeping
    the output aligned to the original tick index.

    Parameters
    ----------
    series : np.ndarray
        Raw tick-level signal (OFI, OBI, or price_ret).
    window : int
        Half-window size in ticks.
    stride : int
        Step between consecutive Wasserstein evaluations.
        stride=1 gives the exact series; stride=50 reduces computation
        50x at the cost of states lagging by at most stride ticks.

    Returns
    -------
    np.ndarray, shape (n - 2*window,)
        Wasserstein distances, forward-filled between stride positions.
    """
    n = len(series)
    if n < 2 * window + 1:
        return np.array([])

    valid_n = n - 2 * window
    out = np.full(valid_n, np.nan)

    for pos in range(0, valid_n, stride):
        i = pos + window
        out[pos] = wasserstein_distance(series[i - window: i], series[i: i + window])

    # Forward-fill: propagate each computed value to the next stride positions
    last_val = out[0] if not np.isnan(out[0]) else 0.0
    for k in range(valid_n):
        if np.isnan(out[k]):
            out[k] = last_val
        else:
            last_val = out[k]

    return out
