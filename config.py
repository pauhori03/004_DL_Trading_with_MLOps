"""
Configuration for Deep Learning Trading MLOps project.
"""

CONFIG = {
    "data": {
        "ticker": "DIS",
        "start_date": "2010-01-01",
        "end_date": "2025-01-01",
        "horizon_days": 5,
        "up_threshold": 0.01,     # forward return > +1% => long
        "down_threshold": -0.01   # forward return < -1% => short
    },
    "splits": {"train_ratio": 0.6, "val_ratio": 0.2, "test_ratio": 0.2},
    "model": {
        "epochs": 30,
        "batch_size": 64,
        "learning_rate": 0.001,
        "conv_filters": [32, 64],
        "kernel_size": 5,
        "dropout": 0.25,
        "sequence_length": 32     # lookback window (timesteps)
    },
    "backtest": {
        "commission": 0.00125,        # 0.125%
        "borrow_rate_annual": 0.0025, # 0.25% annualized
        "take_profit": 0.03,          # optional, can be None
        "stop_loss": 0.02,            # optional, can be None
        "position_size": 1            # shares
    },
    "mlflow": {"experiment_name": "cnn_trading_signals", "tracking_uri": None},
    "api": {"host": "127.0.0.1", "port": 8000},
    "runtime": {"seed": 42}
}