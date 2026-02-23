import sys
sys.path.insert(0, r'C:\Users\Alexis\Desktop\Quant\RIF')
import pandas as pd
import numpy as np

from src.data.loader import load_and_sync_all_tickers
from src.config import TICKERS

# Load states
states_df = pd.read_csv(
    r'C:\Users\Alexis\Desktop\Quant\RIF\data\results\hierarchical_states_local.csv',
    index_col=0, parse_dates=True
)

# Load original LOBSTER data
print("Loading LOBSTER data...")
synced = load_and_sync_all_tickers(TICKERS)
print(f"Loaded. Index length: {len(list(synced.values())[0])}")
print(f"States index length: {len(states_df)}")

REGIME_LABELS = {0: "R0", 1: "R1", 2: "R2"}

rows = []
for ticker in TICKERS:
    df = synced[ticker].copy()
    df["price_ret"] = df["micro_price"].pct_change()

    # Align states with data using common index
    common_idx = df.index.intersection(states_df.index)
    df_aligned = df.loc[common_idx]
    states_aligned = states_df.loc[common_idx, f"state_{ticker}"]

    for regime in sorted(states_aligned.unique()):
        mask = states_aligned == regime
        sub = df_aligned[mask]
        n = len(sub)
        pct = n / len(states_aligned) * 100

        price_ret = sub["price_ret"].dropna()
        obi = sub["obi"]
        ofi = sub["ofi"]

        rows.append({
            "ticker": ticker,
            "regime": int(regime),
            "n": n,
            "pct": round(pct, 1),
            # price_ret
            "ret_mean_bps": round(price_ret.mean() * 1e4, 4),
            "ret_std_bps": round(price_ret.std() * 1e4, 4),
            "ret_abs_mean_bps": round(price_ret.abs().mean() * 1e4, 4),
            "ret_kurt": round(price_ret.kurt(), 1),
            # OBI
            "obi_mean": round(obi.mean(), 4),
            "obi_abs_mean": round(obi.abs().mean(), 4),
            "obi_std": round(obi.std(), 4),
            # OFI
            "ofi_mean": round(ofi.mean(), 1),
            "ofi_abs_mean": round(ofi.abs().mean(), 1),
            "ofi_std": round(ofi.std(), 1),
        })

result = pd.DataFrame(rows)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:.4f}".format)
print("\n=== STATS DESCRIPTIVES PAR REGIME ET TICKER ===")
print(result.to_string(index=False))

# Save
result.to_csv(r'C:\Users\Alexis\Desktop\Quant\RIF\data\results\regime_stats_original_features.csv', index=False)
print("\nSaved to regime_stats_original_features.csv")
