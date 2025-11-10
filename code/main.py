"""
main.py
-------
Linear orchestrator that runs the entire pipeline:
1) set seeds and ensure directories
2) download & load data
3) build features
4) generate targets
5) align, split, scale
6) make sequences
7) build/train CNN with MLflow logging
8) evaluate and save predictions/signals
9) backtest and save metrics/equity
10) compute drift table
"""

from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
from tensorflow import keras

from config import CONFIG
from utils import set_seed, ensure_dirs, get_logger
from data import download_data, load_price_data
from feature_engineering import build_features, make_sequences
from target_generator import make_target_from_forward_returns
from data_splitter import chronological_split, scale_with_train_only, persist_splits_and_scaler
from model import build_cnn
from trainer import train_with_mlflow, evaluate_model
from backtesting import probs_to_signals, run_backtest, performance_metrics
from drift_analyzer import drift_table

logger = get_logger("main")


def main():
    # 1) Seeds and directories
    set_seed(CONFIG["runtime"]["seed"])
    ensure_dirs([
        "data/raw", "data/processed", "data/splits",
        "models/best_model",
        "results/predictions", "results/backtest",
        "mlruns"
    ])

    # 2) Download raw data
    raw_csv_path = os.path.join("data", "raw", f"{CONFIG['data']['ticker']}.csv")
    if not os.path.exists(raw_csv_path):
        download_data(
            ticker=CONFIG["data"]["ticker"],
            start=CONFIG["data"]["start_date"],
            end=CONFIG["data"]["end_date"],
            out_path=raw_csv_path
        )
    else:
        logger.info(f"Raw file already exists at {raw_csv_path}")

    # 3) Load prices and build features
    prices = load_price_data(raw_csv_path)
    feats = build_features(prices)

    # 4) Generate target from forward returns (use Close column)
    y_all = make_target_from_forward_returns(
        prices=feats["Close"],
        horizon=CONFIG["data"]["horizon_days"],
        up_th=CONFIG["data"]["up_threshold"],
        down_th=CONFIG["data"]["down_threshold"]
    )

    # 5) Align X and y and drop NaN
    X_all = feats.loc[y_all.index].drop(columns=["Close"])
    X_all = X_all.dropna()
    y_all = y_all.loc[X_all.index]
    logger.info(f"Aligned shapes -> X:{X_all.shape}, y:{y_all.shape}")

    # 6) Chronological split and save splits + scaler
    X_train, X_val, X_test, y_train, y_val, y_test = chronological_split(X_all, y_all, CONFIG["splits"])
    # Persist raw splits (pre-scaling) for drift dashboard
    persist_splits_and_scaler(X_train, X_val, X_test, y_train, y_val, y_test,
                              splits_dir="data/splits",
                              scaler=None, scaler_path="models/best_model/scaler.pkl")  # scaler saved below

    # 7) Scale with train-only
    Xtr_scaled, Xva_scaled, Xte_scaled, scaler = scale_with_train_only(X_train, X_val, X_test)
    # Overwrite scaler file with real scaler
    import joblib
    joblib.dump(scaler, "models/best_model/scaler.pkl")

    # 8) Make sequences for CNN (we must trim y accordingly)
    seq_len = CONFIG["model"]["sequence_length"]

    def seq_pack(X_scaled: np.ndarray, y: pd.Series):
        # After making sequences, we lose the first 'seq_len' rows.
        # Align labels by dropping first 'seq_len' y values.
        X_seq = []
        for i in range(seq_len, len(X_scaled)):
            X_seq.append(X_scaled[i - seq_len:i, :])
        X_seq = np.stack(X_seq, axis=0).astype(np.float32)
        y_seq = y.iloc[seq_len:].values.astype(int)
        return X_seq, y_seq

    Xtr_seq, ytr_seq = seq_pack(Xtr_scaled, y_train)
    Xva_seq, yva_seq = seq_pack(Xva_scaled, y_val)
    Xte_seq, yte_seq = seq_pack(Xte_scaled, y_test)

    input_shape = (seq_len, Xtr_seq.shape[-1])

    # 9) Build and train model
    model = build_cnn(
        input_shape=input_shape,
        n_classes=3,
        learning_rate=CONFIG["model"]["learning_rate"],
        conv_filters=tuple(CONFIG["model"]["conv_filters"]),
        kernel_size=CONFIG["model"]["kernel_size"],
        dropout=CONFIG["model"]["dropout"]
    )
    best_model_path = "models/best_model/best_cnn.h5"
    _ = train_with_mlflow(
        model, Xtr_seq, ytr_seq, Xva_seq, yva_seq,
        config=CONFIG, out_model_path=best_model_path
    )

    # Load best weights (ModelCheckpoint already saved)
    model = keras.models.load_model(best_model_path)

    # 10) Evaluate on test and save predictions/signals
    metrics = evaluate_model(model, Xte_seq, yte_seq)

    probs = model.predict(Xte_seq, verbose=0)
    signals = probs.argmax(axis=1)  # indices 0,1,2
    idx_to_class = {0: -1, 1: 0, 2: 1}
    mapped_signals = np.vectorize(idx_to_class.get)(signals)

    # Save probabilities and signals with test dates aligned to last part of X_test
    test_index = X_test.index[seq_len:]  # aligned with Xte_seq rows
    probs_df = pd.DataFrame(probs, index=test_index, columns=["prob_short", "prob_hold", "prob_long"])
    sig_df = pd.DataFrame(mapped_signals, index=test_index, columns=["signal"])

    os.makedirs("results/predictions", exist_ok=True)
    probs_df.to_csv("results/predictions/test_probs.csv")
    sig_df.to_csv("results/predictions/test_signals.csv")
    logger.info("Saved test probabilities and signals")

    # 11) Backtest on test period using Close from feats
    close_test = feats.loc[test_index, "Close"]
    bt = run_backtest(close_test, mapped_signals, CONFIG)
    os.makedirs("results/backtest", exist_ok=True)
    bt.to_csv("results/backtest/equity.csv")

    perf = performance_metrics(bt["equity"])
    with open("results/backtest/metrics.json", "w") as f:
        json.dump(perf, f, indent=2)
    logger.info(f"Backtest metrics: {perf}")

    # 12) Drift table (on raw splits without scaling)
    drift_table(X_train, X_val, X_test)
    logger.info("Pipeline finished successfully.")


if __name__ == "__main__":
    main()
