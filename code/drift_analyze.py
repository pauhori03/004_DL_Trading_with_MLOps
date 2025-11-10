"""
drift_analyzer.py
-----------------
Kolmogorov–Smirnov drift tests between train/val/test feature distributions.
"""

from __future__ import annotations
import os
import pandas as pd
from scipy.stats import ks_2samp
from typing import Tuple
from utils import get_logger

logger = get_logger("drift")


def ks_test(train: pd.Series, other: pd.Series) -> float:
    """
    Two-sample KS test p-value. Low p means distributions are different.
    """
    train = train.dropna()
    other = other.dropna()
    if len(train) < 5 or len(other) < 5:
        return 1.0  # not enough data to judge drift
    return float(ks_2samp(train, other).pvalue)


def drift_table(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each feature in train_df, compute p-values vs val and vs test,
    and mark drift if p < 0.05.
    """
    rows = []
    for col in train_df.columns:
        p_val = ks_test(train_df[col], val_df[col]) if col in val_df.columns else 1.0
        p_test = ks_test(train_df[col], test_df[col]) if col in test_df.columns else 1.0
        rows.append({
            "feature": col,
            "p_train_vs_val": p_val,
            "p_train_vs_test": p_test,
            "drift_val": p_val < 0.05,
            "drift_test": p_test < 0.05
        })
    dt = pd.DataFrame(rows).sort_values("p_train_vs_test")
    os.makedirs("data/processed", exist_ok=True)
    dt.to_csv("data/processed/drift_table.csv", index=False)
    logger.info("Saved drift table to data/processed/drift_table.csv")
    return dt
