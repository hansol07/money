from __future__ import annotations

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    close = frame["Close"]

    frame["ma20"] = close.rolling(20).mean()
    frame["ma60"] = close.rolling(60).mean()
    frame["ma120"] = close.rolling(120).mean()
    frame["volume_ma20"] = frame["Volume"].rolling(20).mean()
    frame["rsi"] = RSIIndicator(close=close, window=14).rsi()

    macd = MACD(close=close, window_fast=12, window_slow=26, window_sign=9)
    frame["macd"] = macd.macd()
    frame["macd_signal"] = macd.macd_signal()
    frame["macd_diff"] = macd.macd_diff()

    frame["return_20d"] = close.pct_change(20) * 100
    frame["return_60d"] = close.pct_change(60) * 100
    frame["volume_ratio"] = frame["Volume"] / frame["volume_ma20"]
    return frame.dropna().reset_index(drop=True)
