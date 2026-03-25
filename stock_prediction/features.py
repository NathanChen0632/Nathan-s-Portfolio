"""
features.py
-----------
Transforms raw OHLCV data into ML-ready features and a binary target label.

All indicators are computed using only past / present data to avoid
look-ahead bias (information leakage).

Features computed
-----------------
- Daily return
- High-low range (normalised by close)
- Moving averages: 10-day, 20-day, 50-day (as ratio to close)
- MA crossover signals: Close/MA10, Close/MA20, Close/MA50
- RSI (14-day)
- MACD line and signal line
- Rolling 10-day volatility (std of returns)
- Volume ratio (today vs 20-day avg)

Target
------
  1  if tomorrow's close > today's close
  0  otherwise
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Individual indicator helpers
# ---------------------------------------------------------------------------

def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a raw OHLCV DataFrame, return a new DataFrame containing all
    engineered features and the binary target column 'Target'.

    Rows with NaN values (warm-up period) are dropped before returning.
    """
    feat = pd.DataFrame(index=df.index)

    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    # --- Returns & price range ---
    feat["daily_return"]    = close.pct_change()
    feat["hl_range"]        = (high - low) / close          # normalised H-L range
    feat["open_close_diff"] = (close - df["Open"]) / close  # intraday direction

    # --- Moving averages (ratio to close to make them scale-free) ---
    for window in [10, 20, 50]:
        ma = close.rolling(window).mean()
        feat[f"ma{window}_ratio"] = close / ma              # >1 means above MA

    # --- MA crossover: short vs long ---
    feat["ma10_ma20_cross"] = close.rolling(10).mean() / close.rolling(20).mean()
    feat["ma20_ma50_cross"] = close.rolling(20).mean() / close.rolling(50).mean()

    # --- RSI ---
    feat["rsi14"] = _compute_rsi(close, 14)

    # --- MACD ---
    macd_line, signal_line = _compute_macd(close)
    feat["macd"]        = macd_line
    feat["macd_signal"] = signal_line
    feat["macd_hist"]   = macd_line - signal_line

    # --- Rolling volatility ---
    feat["volatility10"] = feat["daily_return"].rolling(10).std()
    feat["volatility20"] = feat["daily_return"].rolling(20).std()

    # --- Volume indicators ---
    vol_ma20 = vol.rolling(20).mean()
    feat["volume_ratio"] = vol / vol_ma20                   # relative volume

    # --- Target: 1 if tomorrow's close > today's close ---
    feat["Target"] = (close.shift(-1) > close).astype(int)

    # Drop warm-up NaN rows and the final row (no next-day label)
    feat.dropna(inplace=True)

    return feat


def get_feature_columns(feat_df: pd.DataFrame) -> list:
    """Return the list of feature column names (excludes 'Target')."""
    return [c for c in feat_df.columns if c != "Target"]
