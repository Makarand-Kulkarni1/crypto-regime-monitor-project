"""
features.py
Shared feature engineering logic. Used by both the exploration notebook
(02_feature_engineering.ipynb) and the live prediction script (predict.py) so
training-time and inference-time features never drift apart.
"""

import pandas as pd
import numpy as np


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical/statistical features from raw OHLCV data.

    Expects df with columns: open, high, low, close, volume
    (indexed by datetime, sorted ascending).
    """
    out = df.copy()

    # --- Returns ---
    out["log_return"] = np.log(out["close"] / out["close"].shift(1))

    # --- Rolling volatility (realized vol) ---
    out["vol_short"] = out["log_return"].rolling(window=8).std()   # ~2 hours on 15m candles
    out["vol_long"] = out["log_return"].rolling(window=48).std()   # ~12 hours

    # --- Moving averages & trend ---
    out["ma_fast"] = out["close"].rolling(window=12).mean()
    out["ma_slow"] = out["close"].rolling(window=48).mean()
    out["ma_cross"] = out["ma_fast"] - out["ma_slow"]
    out["trend_strength"] = out["ma_cross"] / out["close"]

    # --- RSI (14-period) ---
    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    out["rsi"] = 100 - (100 / (1 + rs))

    # --- Volume anomaly ---
    out["vol_ma"] = out["volume"].rolling(window=48).mean()
    out["volume_ratio"] = out["volume"] / out["vol_ma"]

    # --- ATR (True Range based) ---
    prev_close = out["close"].shift(1)
    tr = pd.concat([
        out["high"] - out["low"],
        (out["high"] - prev_close).abs(),
        (out["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    out["atr"] = tr.rolling(window=14).mean()
    out["atr_pct"] = out["atr"] / out["close"]

    # --- Volatility ratio (short vs long) ---
    out["vol_ratio"] = out["vol_short"] / out["vol_long"]

    return out
