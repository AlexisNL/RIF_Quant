"""
LOBSTER data loading and synchronization utilities.

This module provides a fast Polars-based loader for raw LOBSTER files and
helpers to align multiple tickers on a common resampled timeline.
"""

import polars as pl
import pandas as pd
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor

from src.config import (
    RAW_DATA_DIR, TICKERS, N_LEVELS, RESAMPLE_FREQ,
    ANALYSIS_DATE, START_TIME, END_TIME, N_JOBS
)


def process_lobster_data_fast(ticker: str, base_path: Path = RAW_DATA_DIR) -> pl.DataFrame:
    """Load one ticker's LOBSTER files and compute core microstructure features.

    The function reads order book and message CSV files with Polars lazy scans,
    computes `micro_price`, `obi`, and `ofi`, and returns a materialized
    Polars DataFrame containing only the required columns.

    Parameters
    ----------
    ticker : str
        Ticker symbol used in LOBSTER file naming.
    base_path : Path, default=RAW_DATA_DIR
        Directory containing the raw LOBSTER CSV files.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns: `time`, `micro_price`, `obi`, and `ofi`.

    Raises
    ------
    FileNotFoundError
        If the order book or message file for the ticker is missing.
    """

    book_file = base_path / f"{ticker}_{ANALYSIS_DATE}_{START_TIME}_{END_TIME}_orderbook_{N_LEVELS}.csv"
    msg_file = base_path / f"{ticker}_{ANALYSIS_DATE}_{START_TIME}_{END_TIME}_message_{N_LEVELS}.csv"

    if not book_file.exists():
        raise FileNotFoundError(f"File not found: {book_file}")
    if not msg_file.exists():
        raise FileNotFoundError(f"File not found: {msg_file}")

    # Columns
    columns = []
    for i in range(1, N_LEVELS + 1):
        columns.extend([f'ask_p_{i}', f'ask_s_{i}', f'bid_p_{i}', f'bid_s_{i}'])

    msg_cols = ['time', 'type', 'order_id', 'size', 'price', 'dir']

    # Lazy scan
    df_book = pl.scan_csv(str(book_file), has_header=False, new_columns=columns)
    df_msgs = pl.scan_csv(str(msg_file), has_header=False, new_columns=msg_cols)

    # Horizontal concat with pl.concat (Polars 1.0+)
    df = pl.concat([
        df_msgs.select(['time', 'type', 'price', 'size', 'dir']),
        df_book
    ], how="horizontal")

    # LOB metrics (Polars expressions)
    bid_weights = sum([pl.col(f'bid_s_{i}') for i in range(1, N_LEVELS + 1)])
    ask_weights = sum([pl.col(f'ask_s_{i}') for i in range(1, N_LEVELS + 1)])
    weighted_bid = sum([pl.col(f'bid_p_{i}') * pl.col(f'ask_s_{i}') for i in range(1, N_LEVELS + 1)])
    weighted_ask = sum([pl.col(f'ask_p_{i}') * pl.col(f'bid_s_{i}') for i in range(1, N_LEVELS + 1)])

    micro_price = (weighted_bid + weighted_ask) / (bid_weights + ask_weights)
    obi = (bid_weights - ask_weights) / (bid_weights + ask_weights)

    # OFI
    ofi_exprs = []
    for i in range(1, N_LEVELS + 1):
        bp, bs, ap, as_ = f'bid_p_{i}', f'bid_s_{i}', f'ask_p_{i}', f'ask_s_{i}'
        delta_bid = (
            pl.when(pl.col(bp) > pl.col(bp).shift(1)).then(pl.col(bs))
            .when(pl.col(bp) == pl.col(bp).shift(1)).then(pl.col(bs) - pl.col(bs).shift(1))
            .otherwise(-pl.col(bs).shift(1))
        )
        delta_ask = (
            pl.when(pl.col(ap) < pl.col(ap).shift(1)).then(pl.col(as_))
            .when(pl.col(ap) == pl.col(ap).shift(1)).then(pl.col(as_) - pl.col(as_).shift(1))
            .otherwise(-pl.col(as_).shift(1))
        )
        ofi_exprs.append(delta_bid - delta_ask)

    ofi = pl.sum_horizontal(*ofi_exprs)

    return df.with_columns([
        micro_price.alias("micro_price"),
        obi.alias("obi"),
        ofi.alias("ofi")
    ]).select(['time', 'micro_price', 'obi', 'ofi']).drop_nulls().collect()


def load_and_sync_all_tickers(
    tickers: List[str] = TICKERS,
    resample_freq: str = RESAMPLE_FREQ,
    base_path: Path = RAW_DATA_DIR
) -> Dict[str, pd.DataFrame]:
    """Load, resample, and time-align multiple tickers.

    Each ticker is processed in parallel with `process_lobster_data_fast`, then
    converted to pandas, resampled at `resample_freq`, and forward-filled. The
    returned dictionary is restricted to the common time index shared by all
    tickers.

    Parameters
    ----------
    tickers : List[str], default=TICKERS
        Ticker symbols to load.
    resample_freq : str, default=RESAMPLE_FREQ
        Pandas resampling frequency (for example, `"500ms"`).
    base_path : Path, default=RAW_DATA_DIR
        Directory containing the raw LOBSTER CSV files.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Mapping from ticker to synchronized pandas DataFrame indexed by datetime.
    """

    synced = {}

    with ThreadPoolExecutor(max_workers=N_JOBS) as executor:
        futures = {executor.submit(process_lobster_data_fast, t, base_path): t for t in tickers}
        for f in futures:
            t = futures[f]
            df = f.result().to_pandas()
            df['dt'] = pd.to_datetime(df['time'], unit='s', origin='2024-01-01')
            synced[t] = df.set_index('dt').resample(resample_freq).last().ffill()

    common_idx = synced[tickers[0]].index
    for t in tickers[1:]:
        common_idx = common_idx.intersection(synced[t].index)

    return {t: synced[t].loc[common_idx] for t in tickers}


def load_ticker_ticks(
    ticker: str,
    base_path: Path = RAW_DATA_DIR,
) -> pd.DataFrame:
    """Load raw tick-by-tick LOBSTER data for a single ticker.

    Unlike :func:`load_and_sync_all_tickers`, this function does **not**
    resample to 500 ms and does **not** synchronize with other tickers.
    Use this for analyses that require event-level granularity, such as
    adverse selection estimation.

    Parameters
    ----------
    ticker : str
        Ticker symbol used in LOBSTER file naming.
    base_path : Path, default=RAW_DATA_DIR
        Directory containing the raw LOBSTER CSV files.

    Returns
    -------
    pd.DataFrame
        DateTime-indexed DataFrame with columns:
        ``micro_price``, ``obi``, ``ofi``, ``price_ret``.
        One row per LOB event (no resampling).
    """
    df_polars = process_lobster_data_fast(ticker, base_path)
    df = df_polars.to_pandas()
    df['dt'] = pd.to_datetime(df['time'], unit='s', origin='2024-01-01')
    df = df.set_index('dt').drop(columns=['time'])
    df['price_ret'] = df['micro_price'].pct_change().fillna(0.0)
    return df


def load_all_tickers(
    tickers: List[str] = TICKERS,
    analysis_date: str = ANALYSIS_DATE,
    base_path: Path = RAW_DATA_DIR,
    resample_freq: str = RESAMPLE_FREQ,
) -> Dict[str, pd.DataFrame]:
    """Backward-compatible wrapper around `load_and_sync_all_tickers`.

    This helper keeps compatibility with older scripts that still pass an
    `analysis_date` argument. The argument is currently ignored because the
    effective analysis date comes from `src.config.ANALYSIS_DATE`.

    Parameters
    ----------
    tickers : List[str], default=TICKERS
        Ticker symbols to load.
    analysis_date : str, default=ANALYSIS_DATE
        Legacy parameter kept for API compatibility.
    base_path : Path, default=RAW_DATA_DIR
        Directory containing the raw LOBSTER CSV files.
    resample_freq : str, default=RESAMPLE_FREQ
        Pandas resampling frequency.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Mapping from ticker to synchronized pandas DataFrame.
    """
    if analysis_date != ANALYSIS_DATE:
        print(
            "Warning: load_all_tickers ignores analysis_date and uses config.ANALYSIS_DATE"
        )
    return load_and_sync_all_tickers(tickers, resample_freq, base_path)
