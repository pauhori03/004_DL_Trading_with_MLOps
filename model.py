"""
model.py
--------
Build a simple 1D CNN for classification into {-1, 0, 1}.
"""

from __future__ import annotations
from typing import Tuple
from tensorflow import keras
from tensorflow.keras import layers


def build_cnn(input_shape: tuple[int, int], n_classes: int = 3, learning_rate: float = 1e-3,
              conv_filters=(32, 64), kernel_size: int = 5, dropout: float = 0.25) -> keras.Model:
    """
    Construct and compile a small 1D CNN.

    Parameters
    ----------
    input_shape : (timesteps, n_features)
    n_classes : int
    learning_rate : float
    conv_filters : tuple[int, int]
    kernel_size : int
    dropout : float

    Returns
    -------
    keras.Model
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(conv_filters[0], kernel_size, activation="relu", padding="same"),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(dropout),
        layers.Conv1D(conv_filters[1], kernel_size, activation="relu", padding="same"),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(dropout),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(n_classes, activation="softmax")
    ])

    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model