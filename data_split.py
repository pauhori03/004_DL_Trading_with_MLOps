"""
data_splitter.py
----------------
Chronological split and scaling logic. Scaler is fit on train only to prevent
any look-ahead leakage. Splits are persisted to CSV and scaler is saved.
"""

from __future__ import annotations
import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from utils import get_logger

logger = get_logger("data_splitter")


def chronological_split(
    X: pd.DataFrame, y: pd.Series, ratios: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Chronologically split into train/val/test using ratios.

    Returns
    -------
    (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    n = len(X)
    tr = int(n * ratios["train_ratio"])
    va = int(n * (ratios["train_ratio"] + ratios["val_ratio"]))

    X_train = X.iloc[:tr].copy()
    y_train = y.iloc[:tr].copy()

    X_val = X.iloc[tr:va].copy()
    y_val = y.iloc[tr:va].copy()

    X_test = X.iloc[va:].copy()
    y_test = y.iloc[va:].copy()

    logger.info(f"Split sizes -> train:{len(X_train)}, val:{len(X_val)}, test:{len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_with_train_only(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit StandardScaler on train only. Transform val and test with the same scaler.
    """
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train.values.astype(float))
    Xva = scaler.transform(X_val.values.astype(float))
    Xte = scaler.transform(X_test.values.astype(float))
    return Xtr, Xva, Xte, scaler


def persist_splits_and_scaler(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series, y_val: pd.Series, y_test: pd.Series,
    splits_dir: str, scaler, scaler_path: str
) -> None:
    """
    Save splits as CSV and the scaler with joblib.
    """
    os.makedirs(splits_dir, exist_ok=True)
    X_train.to_csv(os.path.join(splits_dir, "X_train.csv"))
    X_val.to_csv(os.path.join(splits_dir, "X_val.csv"))
    X_test.to_csv(os.path.join(splits_dir, "X_test.csv"))
    y_train.to_csv(os.path.join(splits_dir, "y_train.csv"))
    y_val.to_csv(os.path.join(splits_dir, "y_val.csv"))
    y_test.to_csv(os.path.join(splits_dir, "y_test.csv"))

    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved splits to {splits_dir} and scaler to {scaler_path}")
