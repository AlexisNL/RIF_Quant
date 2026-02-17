from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def compute_mad(x: np.ndarray) -> float:
    """
    Compute the Median Absolute Deviation.

        MAD = median(|X - median(X)|)

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    float
        MAD value (outlier-robust scatter estimate).
    """
    median = np.median(x)
    return float(np.median(np.abs(x - median)))


class MADNormalizer:
    """
    Rolling MAD-based robust z-score normalizer for LOB innovations.

    Replaces GARCH normalization with a simpler, outlier-robust approach that
    requires no iterative optimization and has no convergence issues.

    Parameters
    ----------
    window : int
        Rolling window size (observations).
    min_periods : int
        Minimum observations required to compute a valid MAD value.
    n_jobs : int or None
        Number of parallel workers for ``fit_transform``.
        ``None`` uses ``min(n_series, 8)``.

    Attributes
    ----------
    innov_dict_ : dict or None
        Normalized innovations ``{ticker: DataFrame}`` after ``fit_transform``.
    stationarity_ : pd.DataFrame or None
        ADF stationarity report after ``validate_stationarity``.
    """

    def __init__(
        self,
        window: int = 100,
        min_periods: int = 50,
        n_jobs: Optional[int] = None,
    ) -> None:
        self.window = window
        self.min_periods = min_periods
        self.n_jobs = n_jobs
        self.innov_dict_: Optional[Dict[str, pd.DataFrame]] = None
        self.stationarity_: Optional[pd.DataFrame] = None

    def fit_transform(
        self,
        synced_data: Dict[str, pd.DataFrame],
        tickers: List[str],
    ) -> Dict[str, pd.DataFrame]:
        """
        Normalize price_ret, OBI, and OFI for every ticker using rolling MAD.

        Results are stored in ``self.innov_dict_`` and also returned.

        Parameters
        ----------
        synced_data : dict
            ``{ticker: DataFrame}`` with columns ``price_ret`` (or
            ``micro_price``), ``obi``, and ``ofi``.
        tickers : list of str
            Ordered list of ticker symbols to process.

        Returns
        -------
        dict
            ``{ticker: DataFrame}`` with columns ``price_ret``, ``obi``,
            ``ofi`` containing robust z-scores.
        """
        print(f"  MAD normalization with window = {self.window} ({self.window * 0.5:.1f}s)")

        tasks = self._build_tasks(synced_data, tickers)
        n_jobs = self.n_jobs if self.n_jobs is not None else min(len(tasks), 8)
        raw_results: Dict[str, Dict[str, pd.Series]] = {}

        def _process(task):
            ticker, metric, series = task
            return ticker, metric, self.transform_series(series)

        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            for ticker, metric, normalized in executor.map(_process, tasks):
                raw_results.setdefault(ticker, {})[metric] = normalized

        innov_dict: Dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            innov_dict[ticker] = pd.DataFrame(
                {m: raw_results[ticker][m] for m in ("price_ret", "obi", "ofi")}
            )

        n_total = len(tickers) * 3
        n_valid = sum(
            innov_dict[t][m].notna().any()
            for t in tickers
            for m in ("price_ret", "obi", "ofi")
        )
        print(f"  OK {n_valid}/{n_total} series normalized successfully")

        self.innov_dict_ = innov_dict
        return innov_dict

    def transform_series(self, series: pd.Series) -> pd.Series:
        """
        Compute the rolling MAD robust z-score for a single series.

            z = (X - rolling_median) / (1.4826 * rolling_MAD)

        Parameters
        ----------
        series : pd.Series
            Raw time series.

        Returns
        -------
        pd.Series
            Robust z-scores aligned with the input index.
        """
        w, mp = self.window, self.min_periods

        rolling_median = series.rolling(window=w, min_periods=mp, center=False).median()

        def _rolling_mad(x: pd.Series) -> float:
            if len(x) < mp:
                return np.nan
            return compute_mad(x.values)

        rolling_mad_vals = series.rolling(window=w, min_periods=mp, center=False).apply(
            _rolling_mad, raw=False
        )

        return (series - rolling_median) / (1.4826 * rolling_mad_vals + 1e-9)

    def validate_stationarity(
        self,
        innov_dict: Optional[Dict[str, pd.DataFrame]] = None,
        tickers: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Run ADF tests on each ticker / metric series to check stationarity.

        Parameters
        ----------
        innov_dict : dict or None
            Normalized innovations. Defaults to ``self.innov_dict_`` if not
            provided.
        tickers : list of str or None
            Tickers to test. Defaults to the keys of ``innov_dict``.

        Returns
        -------
        pd.DataFrame
            Columns: ``ticker``, ``metric``, ``adf_stat``, ``p_value``,
            ``stationary``, ``n_obs``.
        """
        from statsmodels.tsa.stattools import adfuller

        if innov_dict is None:
            if self.innov_dict_ is None:
                raise RuntimeError("No innov_dict available. Call fit_transform first.")
            innov_dict = self.innov_dict_

        if tickers is None:
            tickers = list(innov_dict.keys())

        print("\nStationarity validation (ADF test)...")

        rows = []
        for ticker in tickers:
            for metric in ("price_ret", "obi", "ofi"):
                series = innov_dict[ticker][metric].dropna()
                if len(series) <= 100:
                    continue
                try:
                    stat, pval, *_ = adfuller(series, maxlag=20)
                    rows.append(
                        {
                            "ticker": ticker,
                            "metric": metric,
                            "adf_stat": stat,
                            "p_value": pval,
                            "stationary": pval < 0.05,
                            "n_obs": len(series),
                        }
                    )
                except Exception:
                    pass

        result = pd.DataFrame(rows)
        self.stationarity_ = result

        n_stat = result["stationary"].sum()
        print(f"  OK {n_stat}/{len(result)} stationary series (p < 0.05)")

        return result

    def _build_tasks(
        self,
        synced_data: Dict[str, pd.DataFrame],
        tickers: List[str],
    ) -> list:
        """Build (ticker, metric, series) task list for parallel processing."""
        tasks = []
        for ticker in tickers:
            df = synced_data[ticker]

            if "price_ret" in df.columns:
                price_ret = df["price_ret"]
            elif "micro_price" in df.columns:
                price_ret = np.log(df["micro_price"]).diff() * 100
            else:
                raise KeyError(f"{ticker}: missing 'price_ret' or 'micro_price' column")

            for metric, series in (
                ("price_ret", price_ret),
                ("obi", df["obi"]),
                ("ofi", df["ofi"]),
            ):
                tasks.append((ticker, metric, series))
        return tasks


if __name__ == "__main__":
    np.random.seed(42)
    n = 1000
    x = np.random.randn(n)
    x[100] = 10
    x[500] = -8
    series = pd.Series(x)

    normalizer = MADNormalizer(window=100)
    normalized = normalizer.transform_series(series)

    print("Test on synthetic series:")
    print(f"  MAD = {compute_mad(x):.3f}")
    print(f"  Std = {np.std(x):.3f}")
    print(f"  Normalized: mean={normalized.mean():.3f}, std={normalized.std():.3f}")
    print(f"  Outliers reduced: max={normalized.max():.2f}, min={normalized.min():.2f}")
