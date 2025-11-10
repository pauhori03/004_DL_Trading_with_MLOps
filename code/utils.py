"""
utils.py
--------
Small utilities used across the trading pipeline.
Removed directory creation per request.
"""

from __future__ import annotations
import random
import logging
import numpy as np
import pandas as pd


def set_seed(seed: int) -> None:
    """
    Make runs deterministic across random, numpy, and tensorflow.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        # Skip if TensorFlow not available at import time
        pass


def get_logger(name: str) -> logging.Logger:
    """
    Create a simple console logger at INFO level.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter("[%(asctime)s] %(name)s - %(levelname)s - %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


def to_numpy(df: pd.DataFrame) -> np.ndarray:
    """
    Safe conversion to numpy float32 array.
    """
    return df.values.astype(np.float32)


def max_drawdown(equity: np.ndarray) -> tuple[float, int, int]:
    """
    Compute maximum drawdown and start/end indices using cumulative max.
    Returns (mdd, start_idx, end_idx) where mdd <= 0 (as a return).
    """
    peaks = np.maximum.accumulate(equity)
    drawdowns = equity / peaks - 1.0
    end = int(np.argmin(drawdowns))
    start = int(np.argmax(equity[: end + 1]))
    mdd = float(drawdowns[end])
    return mdd, start, end
