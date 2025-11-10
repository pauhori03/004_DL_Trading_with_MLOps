"""
app.py (FastAPI)
----------------
Loads the saved best CNN and scaler, exposes a POST /predict endpoint.
The request must contain a 2D list named 'latest_sequence' with shape
(sequence_length, n_features) in the SAME feature order used in training.

We purposely keep the API small and explicit.
"""

from __future__ import annotations
import os
import joblib
import numpy as np
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from tensorflow import keras
from config import CONFIG

# Initialize FastAPI app
app = FastAPI(title="CNN Trading Signals API", version="1.0")

# Load artifacts at startup
MODEL_PATH = "models/best_model/best_cnn.h5"
SCALER_PATH = "models/best_model/scaler.pkl"

if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
    # We avoid crashing on import; raise at runtime in endpoint for clearer message.
    model = None
    scaler = None
else:
    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)


class PredictRequest(BaseModel):
    """
    A single most-recent sequence of features for prediction.
    """
    latest_sequence: list[list[float]] = Field(..., description="2D list of shape (sequence_length, n_features)")


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Scale, reshape to (1, seq_len, n_features), run model, and map to {-1,0,1}.
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or scaler not found. Train the pipeline first (python main.py).")

    seq = np.array(req.latest_sequence, dtype=np.float32)
    if seq.ndim != 2:
        raise HTTPException(status_code=400, detail="latest_sequence must be 2D (seq_len, n_features)")

    # Scale each timestep row using the same scaler learned on training set
    # We flatten -> scale -> reshape back to (seq_len, n_features)
    seq_len, n_features = seq.shape
    flat = seq.reshape(-1, n_features)
    flat_scaled = scaler.transform(flat)
    seq_scaled = flat_scaled.reshape(1, seq_len, n_features)

    probs = model.predict(seq_scaled, verbose=0)[0]
    # Map index->class
    mapping = {0: -1, 1: 0, 2: 1}
    signal = int(mapping[int(np.argmax(probs))])
    resp = {
        "signal": signal,
        "probabilities": {
            "short": float(probs[0]),
            "hold": float(probs[1]),
            "long": float(probs[2])
        }
    }
    return resp
