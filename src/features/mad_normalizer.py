from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_mad(x: np.ndarray) -> float:
    """
    Deviation Absolue Mediane.

        MAD = median(|X - median(X)|)
    """
    median = np.median(x)
    return float(np.median(np.abs(x - median)))


class MADNormalizer:
    """
    Normalisation robuste par z-score MAD glissant pour les innovations LOB.

    Remplace la normalisation GARCH par une approche robuste aux outliers,
    sans optimisation iterative ni risque de non-convergence.

    Parameters
    ----------
    window : int
        Taille de la fenetre glissante (observations).
    min_periods : int
        Nombre minimum d'observations pour calculer un MAD valide.
    n_jobs : int or None
        Nombre de workers paralleles pour ``fit_transform``.
        ``None`` utilise ``min(n_series, 8)``.

    Attributes
    ----------
    innov_dict_ : dict or None
        Innovations normalisees ``{ticker: DataFrame}`` apres ``fit_transform``.
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

    def fit_transform(
        self,
        synced_data: Dict[str, pd.DataFrame],
        tickers: List[str],
    ) -> Dict[str, pd.DataFrame]:
        """
        Normalise price_ret, OBI et OFI pour chaque ticker par MAD glissant.

        Resultats stockes dans ``self.innov_dict_`` et retournes.

        Parameters
        ----------
        synced_data : dict
            ``{ticker: DataFrame}`` avec colonnes ``price_ret`` (ou
            ``micro_price``), ``obi``, et ``ofi``.
        tickers : list of str
            Liste ordonnee des tickers.

        Returns
        -------
        dict
            ``{ticker: DataFrame}`` avec colonnes ``price_ret``, ``obi``,
            ``ofi`` contenant les z-scores robustes.
        """
        logger.info("MAD normalisation window=%d (%.1fs)", self.window, self.window * 0.5)

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
        logger.info("%d/%d series normalised", n_valid, n_total)

        self.innov_dict_ = innov_dict
        return innov_dict

    def transform_series(self, series: pd.Series) -> pd.Series:
        """
        Z-score MAD robuste glissant pour une serie unique.

            z = (X - mediane_glissante) / (1.4826 * MAD_glissant)

        Returns
        -------
        pd.Series
            Z-scores robustes alignes sur l'index d'entree.
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

    def _build_tasks(
        self,
        synced_data: Dict[str, pd.DataFrame],
        tickers: List[str],
    ) -> list:
        """Construit la liste de taches (ticker, metrique, serie) pour le parallelisme."""
        tasks = []
        for ticker in tickers:
            df = synced_data[ticker]

            if "price_ret" in df.columns:
                price_ret = df["price_ret"]
            elif "micro_price" in df.columns:
                price_ret = np.log(df["micro_price"]).diff() * 100
            else:
                raise KeyError(f"{ticker}: colonne 'price_ret' ou 'micro_price' manquante")

            for metric, series in (
                ("price_ret", price_ret),
                ("obi", df["obi"]),
                ("ofi", df["ofi"]),
            ):
                tasks.append((ticker, metric, series))
        return tasks
