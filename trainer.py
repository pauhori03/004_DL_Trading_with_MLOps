"""
trainer.py
----------
Training helpers with MLflow logging. Uses class weights to handle imbalance.
"""

from __future__ import annotations
import os
import numpy as np
import mlflow
from typing import Dict, Tuple
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tensorflow import keras
from utils import get_logger

logger = get_logger("trainer")


def compute_class_weights(y_train: np.ndarray) -> dict[int, float]:
    """
    Compute balanced class weights for labels {-1, 0, 1}. The Keras API expects
    mapping for indices 0..n-1, so we map {-1, 0, 1} -> {0,1,2}.
    """
    label_map = {-1: 0, 0: 1, 1: 2}
    mapped = np.vectorize(label_map.get)(y_train)
    classes = np.array([0, 1, 2])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=mapped)
    return {i: float(w) for i, w in enumerate(weights)}


def _to_one_hot(y: np.ndarray) -> np.ndarray:
    """
    Map labels {-1,0,1} to indices {0,1,2} and one-hot encode.
    """
    label_map = {-1: 0, 0: 1, 1: 2}
    idx = np.vectorize(label_map.get)(y)
    oh = keras.utils.to_categorical(idx, num_classes=3, dtype="float32")
    return oh


def train_with_mlflow(
    model: keras.Model,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    config: dict, out_model_path: str
) -> Dict[str, float]:
    """
    Train model with EarlyStopping & ModelCheckpoint, log to MLflow, and save best model.

    Returns a metrics dictionary.
    """
    os.makedirs(os.path.dirname(out_model_path), exist_ok=True)

    cw = compute_class_weights(y_train)
    ytr = _to_one_hot(y_train)
    yva = _to_one_hot(y_val)

    callbacks = [
        keras.callbacks.ModelCheckpoint(out_model_path, monitor="val_accuracy",
                                        save_best_only=True, mode="max", verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5,
                                      restore_best_weights=True, mode="max")
    ]

    # Configure MLflow
    tracking_uri = config["mlflow"]["tracking_uri"]
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run():
        # Log key params
        for k, v in config["model"].items():
            mlflow.log_param(k, v)

        history = model.fit(
            X_train, ytr,
            validation_data=(X_val, yva),
            epochs=config["model"]["epochs"],
            batch_size=config["model"]["batch_size"],
            class_weight=cw,
            callbacks=callbacks,
            verbose=1
        )

        # Training metrics
        tr_acc = float(history.history["accuracy"][-1])
        tr_loss = float(history.history["loss"][-1])
        va_acc = float(max(history.history["val_accuracy"]))
        va_loss = float(min(history.history["val_loss"]))

        mlflow.log_metric("train_accuracy", tr_acc)
        mlflow.log_metric("train_loss", tr_loss)
        mlflow.log_metric("val_accuracy", va_acc)
        mlflow.log_metric("val_loss", va_loss)

        # Log model artifact
        mlflow.log_artifact(out_model_path)

    metrics = {"train_accuracy": tr_acc, "train_loss": tr_loss,
               "val_accuracy": va_acc, "val_loss": va_loss}
    logger.info(f"Training done. Val acc={va_acc:.4f}")
    return metrics


def evaluate_model(model: keras.Model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Compute accuracy and (macro) F1 with labels {-1,0,1}. Returns simple metrics.
    """
    # Predict class indices (0,1,2) and map back to {-1,0,1}
    probs = model.predict(X, verbose=0)
    idx = probs.argmax(axis=1)
    inv_map = {0: -1, 1: 0, 2: 1}
    y_pred = np.vectorize(inv_map.get)(idx)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, labels=[-1, 0, 1], average="macro")
    cm = confusion_matrix(y, y_pred, labels=[-1, 0, 1])
    logger.info(f"Test accuracy={acc:.4f}, F1={f1:.4f}, CM=\n{cm}")
    return {"accuracy": float(acc), "f1_macro": float(f1)}
