from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

try:
    from src.features.wasserstein_optimized import compute_pairwise_wasserstein_numba
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

_METRICS = ["price_ret", "obi", "ofi"]
_METRIC_LABELS = {"price_ret": "Price", "obi": "OBI", "ofi": "OFI"}


class WassersteinExtractor:
    """
    Temporal and cross-sectional Wasserstein distance features.

    For each ticker and metric at time *t*, the temporal distance is::

        W( series[t-window:t], series[t:t+window] )

    which measures how much the local distribution has shifted — used as a
    stress proxy without relying on price correlation.

    Parameters
    ----------
    window : int
        Rolling half-window size (observations).
    backend : str
        Computation backend: ``"auto"`` (Numba if available, else SciPy),
        ``"numba"``, or ``"scipy"``.

    Attributes
    ----------
    backend : str
        Resolved backend (``"numba"`` or ``"scipy"``).
    """

    def __init__(
        self,
        window: int = 100,
        backend: str = "auto",
    ) -> None:
        self.window = window
        self.backend = self._resolve_backend(backend)

    def compute_features(
        self,
        innov_dict: Dict[str, pd.DataFrame],
        tickers: List[str],
        metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Temporal Wasserstein features for all tickers and metrics.

        Parameters
        ----------
        innov_dict : dict
            ``{ticker: DataFrame}`` with columns ``price_ret``, ``obi``,
            ``ofi``.
        tickers : list of str
            Ordered ticker list.
        metrics : list of str or None
            Subset of ``["price_ret", "obi", "ofi"]``.  Defaults to all.

        Returns
        -------
        pd.DataFrame
            Shape ``(T - 2*window, n_tickers * n_metrics)``.
            Columns: ``"<ticker>_<Metric>"`` (e.g. ``"AAPL_Price"``).
        """
        if metrics is None:
            metrics = _METRICS

        metric_dfs = self._build_metric_dfs(innov_dict, tickers, metrics)
        common_index = self._common_index(metric_dfs, metrics)

        if len(common_index) < 2 * self.window + 1:
            raise ValueError(
                f"Not enough observations ({len(common_index)}) for "
                f"temporal Wasserstein window={self.window}."
            )

        features: Dict[str, np.ndarray] = {}
        for m in metrics:
            df = metric_dfs[m].loc[common_index]
            for t in tickers:
                col = f"{t}_{_METRIC_LABELS.get(m, m)}"
                features[col] = self.compute_temporal_series(df[t].values)

        out_index = common_index[self.window : -self.window]
        return pd.DataFrame(features, index=out_index)

    def compute_temporal_series(self, series: np.ndarray) -> np.ndarray:
        """
        Temporal before-vs-after Wasserstein distance for a single array.

        At position *i* (for ``i`` in ``[window, n - window)``):
            ``W( series[i-window:i], series[i:i+window] )``

        Parameters
        ----------
        series : np.ndarray, shape (n,)
            Un-normalised (or normalised) 1-D time series.

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
            out[i - w] = wasserstein_distance(
                series[i - w : i], series[i : i + w]
            )
        return out

    def compute_cross_sectional_features(
        self,
        innov_dict: Dict[str, pd.DataFrame],
        tickers: List[str],
    ) -> pd.DataFrame:
        """
        Cross-sectional (inter-ticker) Wasserstein features.

        For each metric, at each time *t*, computes the average pairwise
        Wasserstein distance between tickers over the rolling window.

        Returns
        -------
        pd.DataFrame
            Columns: ``"<ticker>_<Metric>"`` (cross-sectional distances).
        """
        if self.backend == "numba":
            wass_X, _, idx = self._compute_numba(innov_dict, tickers)
        else:
            wass_X, _, idx = self._compute_scipy(innov_dict, tickers)

        columns = [
            f"{ticker}_{_METRIC_LABELS[m]}"
            for m in _METRICS
            for ticker in tickers
        ]
        return pd.DataFrame(wass_X, index=idx, columns=columns)

    def decompose_by_ticker(
        self,
        innov_dict: Dict[str, pd.DataFrame],
        tickers: List[str],
    ) -> Dict[str, Dict[str, list]]:
        """
        Cross-sectional Wasserstein distances decomposed by metric and ticker.

        Returns
        -------
        dict
            ``decomposed[metric][ticker]`` → list of distances.
        """
        if self.backend == "numba":
            _, decomposed, _ = self._compute_numba(innov_dict, tickers)
        else:
            _, decomposed, _ = self._compute_scipy(innov_dict, tickers)
        return decomposed

    def decompose_temporal_by_ticker(
        self,
        innov_dict: Dict[str, pd.DataFrame],
        tickers: List[str],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, list]]:
        """
        Temporal Wasserstein distances decomposed by metric and ticker.

        Returns
        -------
        dict
            ``decomposed[metric][ticker]`` → list of temporal distances.
        """
        if metrics is None:
            metrics = _METRICS

        metric_dfs = self._build_metric_dfs(innov_dict, tickers, metrics)
        common_index = self._common_index(metric_dfs, metrics)

        if len(common_index) < 2 * self.window + 1:
            raise ValueError("Not enough observations for temporal Wasserstein window.")

        decomposed: Dict[str, Dict[str, list]] = {
            m: {t: [] for t in tickers} for m in metrics
        }
        for m in metrics:
            df = metric_dfs[m].loc[common_index]
            for t in tickers:
                decomposed[m][t] = self.compute_temporal_series(df[t].values).tolist()

        return decomposed

    def _resolve_backend(self, backend: str) -> str:
        if backend == "auto":
            return "numba" if NUMBA_AVAILABLE else "scipy"
        if backend == "numba" and not NUMBA_AVAILABLE:
            warnings.warn(
                "Numba backend requested but not available. Falling back to SciPy.",
                RuntimeWarning,
            )
            return "scipy"
        return backend

    def _build_metric_dfs(
        self,
        innov_dict: Dict[str, pd.DataFrame],
        tickers: List[str],
        metrics: List[str],
    ) -> Dict[str, pd.DataFrame]:
        return {
            m: pd.DataFrame({t: innov_dict[t][m] for t in tickers}).dropna()
            for m in metrics
        }

    @staticmethod
    def _common_index(
        metric_dfs: Dict[str, pd.DataFrame],
        metrics: List[str],
    ) -> pd.Index:
        idx = metric_dfs[metrics[0]].index
        for m in metrics[1:]:
            idx = idx.intersection(metric_dfs[m].index)
        return idx

    def _compute_scipy(
        self,
        innov_dict: Dict[str, pd.DataFrame],
        tickers: List[str],
    ) -> Tuple[np.ndarray, Dict, pd.Index]:
        """SciPy-based cross-sectional implementation."""
        metric_dfs = self._build_metric_dfs(innov_dict, tickers, _METRICS)
        common_index = self._common_index(metric_dfs, _METRICS)

        all_features: list = []
        decomposed: Dict[str, Dict[str, list]] = {
            m: {t: [] for t in tickers} for m in _METRICS
        }

        for m in _METRICS:
            df = metric_dfs[m].loc[common_index]
            n = len(df)
            m_wass: list = []

            for i in range(self.window, n):
                slice_df = df.iloc[i - self.window : i]
                row = []
                for t1 in tickers:
                    dist = float(
                        np.mean(
                            [
                                wasserstein_distance(slice_df[t1], slice_df[t2])
                                for t2 in tickers
                                if t1 != t2
                            ]
                        )
                    )
                    row.append(dist)
                    decomposed[m][t1].append(dist)
                m_wass.append(row)

            all_features.append(np.array(m_wass))

        wass_X = np.hstack(all_features)
        return wass_X, decomposed, common_index[self.window :]

    def _compute_numba(
        self,
        innov_dict: Dict[str, pd.DataFrame],
        tickers: List[str],
    ) -> Tuple[np.ndarray, Dict, pd.Index]:
        """Numba-accelerated cross-sectional implementation."""
        from src.features.wasserstein_optimized import compute_pairwise_wasserstein_numba

        metric_dfs = self._build_metric_dfs(innov_dict, tickers, _METRICS)
        common_index = self._common_index(metric_dfs, _METRICS)

        all_features: list = []
        decomposed: Dict[str, Dict[str, list]] = {
            m: {t: [] for t in tickers} for m in _METRICS
        }

        for m in _METRICS:
            data = metric_dfs[m].loc[common_index].values
            n = len(data)
            m_wass: list = []

            for i in range(self.window, n):
                window_data = data[i - self.window : i, :]
                avg_dist = compute_pairwise_wasserstein_numba(window_data)
                m_wass.append(avg_dist)
                for j, t in enumerate(tickers):
                    decomposed[m][t].append(float(avg_dist[j]))

            all_features.append(np.array(m_wass))

        wass_X = np.hstack(all_features)
        return wass_X, decomposed, common_index[self.window :]
