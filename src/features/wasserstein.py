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
