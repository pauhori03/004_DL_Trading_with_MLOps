"""
ft_engineering.py
----------------------
Technical indicator feature creation and sequence maker for the CNN.

We aim for >= 20 features spanning momentum, trend, volatility, and volume.
We try pandas_ta first (richer), fall back to ta if not available.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple
from utils import get_logger

logger = get_logger("feature_engineering")

# Try to import pandas_ta; if not, switch to ta
try:
    import ta
    USE_PANDAS_TA = True
except Exception:
    from ta.momentum import RSIIndicator
    from ta.trend import MACD, SMAIndicator
    from ta.volatility import BollingerBands
    from ta.volume import OnBalanceVolumeIndicator
    USE_PANDAS_TA = False


def _add_features_pandas_ta(df: pd.DataFrame) -> pd.DataFrame:
    """Build a rich set of indicators using pandas_ta."""
    out = pd.DataFrame(index=df.index)
    out["Close"] = df["Close"]

    # Momentum / trend
    out["rsi_14"] = ta.rsi(df["Close"], length=14)
    out["roc_5"] = ta.roc(df["Close"], length=5)
    out["roc_10"] = ta.roc(df["Close"], length=10)
    out["mom_10"] = ta.mom(df["Close"], length=10)

    macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    out["macd"] = macd["MACD_12_26_9"]
    out["macd_signal"] = macd["MACDs_12_26_9"]
    out["macd_hist"] = macd["MACDh_12_26_9"]

    # Volatility
    out["atr_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    out["std_10"] = df["Close"].rolling(10).std()
    out["std_20"] = df["Close"].rolling(20).std()

    bb = ta.bbands(df["Close"], length=20, std=2.0)
    out["bb_bandwidth"] = (bb["BBU_20_2.0"] - bb["BBL_20_2.0"]) / bb["BBM_20_2.0"]

    # Volume features
    out["obv"] = ta.obv(df["Close"], df["Volume"])
    out["vroc_10"] = ta.roc(df["Volume"].replace(0, np.nan), length=10)

    # Returns & rolling stats
    out["log_ret_1"] = np.log(df["Close"]).diff()
    out["log_ret_5"] = np.log(df["Close"]).diff(5)
    out["ret_1"] = df["Close"].pct_change()
    out["ret_mean_5"] = out["ret_1"].rolling(5).mean()
    out["ret_mean_10"] = out["ret_1"].rolling(10).mean()
    out["ret_std_10"] = out["ret_1"].rolling(10).std()

    # Moving averages
    out["sma_10"] = df["Close"].rolling(10).mean()
    out["sma_20"] = df["Close"].rolling(20).mean()
    out["sma_50"] = df["Close"].rolling(50).mean()
    out["sma_ratio_10_20"] = out["sma_10"] / out["sma_20"]
    out["sma_ratio_20_50"] = out["sma_20"] / out["sma_50"]

    return out


def _add_features_ta(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback features using the 'ta' library."""
    out = pd.DataFrame(index=df.index)
    out["Close"] = df["Close"]

    # Momentum / trend
    out["rsi_14"] = RSIIndicator(close=df["Close"], window=14).rsi()
    out["roc_5"] = df["Close"].pct_change(5)
    out["roc_10"] = df["Close"].pct_change(10)
    out["mom_10"] = df["Close"].diff(10)

    macd = MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_hist"] = macd.macd_diff()

    # Volatility
    bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
    out["bb_bandwidth"] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    out["std_10"] = df["Close"].rolling(10).std()
    out["std_20"] = df["Close"].rolling(20).std()

    # Volume
    out["obv"] = OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"]).on_balance_volume()
    out["vroc_10"] = df["Volume"].replace(0, np.nan).pct_change(10)

    # Returns & rolling stats
    out["log_ret_1"] = np.log(df["Close"]).diff()
    out["log_ret_5"] = np.log(df["Close"]).diff(5)
    out["ret_1"] = df["Close"].pct_change()
    out["ret_mean_5"] = out["ret_1"].rolling(5).mean()
    out["ret_mean_10"] = out["ret_1"].rolling(10).mean()
    out["ret_std_10"] = out["ret_1"].rolling(10).std()

    # Moving averages
    out["sma_10"] = SMAIndicator(close=df["Close"], window=10).sma_indicator()
    out["sma_20"] = SMAIndicator(close=df["Close"], window=20).sma_indicator()
    out["sma_50"] = SMAIndicator(close=df["Close"], window=50).sma_indicator()
    out["sma_ratio_10_20"] = out["sma_10"] / out["sma_20"]
    out["sma_ratio_20_50"] = out["sma_20"] / out["sma_50"]

    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a wide set of indicators/features. Drops rows with NaN at the end.

    Returns
    -------
    pd.DataFrame
        Feature matrix including 'Close' column (kept for backtesting).
    """
    logger.info("Building technical indicator features")
    if USE_PANDAS_TA:
        feats = _add_features_pandas_ta(df)
    else:
        feats = _add_features_ta(df)
    feats = feats.dropna()
    logger.info(f"Feature matrix shape: {feats.shape} (>=20 features)")
    return feats


def make_sequences(X: pd.DataFrame, seq_len: int) -> np.ndarray:
    """
    Convert a 2D tabular matrix into 3D sequences for CNN input.

    Parameters
    ----------
    X : pd.DataFrame
        Rows must be time-ordered.
    seq_len : int
        Lookback window length.

    Returns
    -------
    np.ndarray
        Shape: (n_samples, seq_len, n_features)
    """
    arr = X.values.astype(np.float32)
    n = arr.shape[0]
    n_features = arr.shape[1]
    if n < seq_len + 1:
        raise ValueError("Not enough rows to build sequences.")
    # Build rolling windows without look-ahead leakage
    out = []
    for i in range(seq_len, n):
        out.append(arr[i - seq_len:i, :])
    return np.stack(out, axis=0)
