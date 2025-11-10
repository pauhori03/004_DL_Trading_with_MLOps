"""
target_generator.py
-------------------
Create labels {-1, 0, 1} using forward returns aligned to avoid look-ahead.
"""

from __future__ import annotations
import pandas as pd
import numpy as np


def make_target_from_forward_returns(
    prices: pd.Series, horizon: int, up_th: float, down_th: float
) -> pd.Series:
    """
    Compute forward percentage return over 'horizon' days and produce
    {-1, 0, 1} labels. The label at time t is based on prices from t+1 ... t+horizon.

    Parameters
    ----------
    prices : pd.Series
        Close prices indexed by date.
    horizon : int
        Forward horizon in days.
    up_th : float
        Threshold above which we label +1 (long).
    down_th : float
        Threshold below which we label -1 (short).

    Returns
    -------
    pd.Series
        Aligned labels with same index as input prices (with trailing NaN dropped).
    """
    # Forward return: (P_{t+h} / P_t - 1)
    fwd = prices.shift(-horizon) / prices - 1.0
    label = pd.Series(0, index=prices.index, dtype=int)
    label[fwd > up_th] = 1
    label[fwd < down_th] = -1
    # Remove last 'horizon' rows where forward return uses future we don't have
    label = label.iloc[:-horizon]
    return label
