"""LSTM sequence model (PyTorch).

Same target choice as XGBoost — next-day log return, not price level. A network
trained on min-max scaled *prices* has the mirror-image of the tree problem:
the scaler is fitted on the training range, so every test price maps to a value
> 1 and the network extrapolates into a region it never saw.

Architecture: 2-layer LSTM over a 30-day window of the stationary feature set,
dropout between layers, a single linear head. Standardisation statistics come
from the training fold only and are stored on the object so that serving uses
identical scaling.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import MODELS
from ..features import feature_columns
from .base import Forecaster, returns_to_prices

log = logging.getLogger(__name__)


def _build_sequences(X: np.ndarray, y: np.ndarray | None, lookback: int):
    """Windows ending at t predict the label at t (which is the t+1 return)."""
    n = len(X)
    if n < lookback:
        raise ValueError(f"Need at least {lookback} rows, got {n}")
    idx = np.arange(lookback - 1, n)
    seqs = np.stack([X[i - lookback + 1: i + 1] for i in idx]).astype(np.float32)
    labels = None if y is None else y[idx].astype(np.float32)
    return seqs, labels, idx


class _Net:
    """Lazily-constructed torch module (kept out of module scope so that the
    package imports cleanly on machines without torch installed)."""

    @staticmethod
    def make(n_features: int, hidden: int, layers: int, dropout: float):
        import torch.nn as nn

        class LSTMNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=n_features, hidden_size=hidden, num_layers=layers,
                    batch_first=True, dropout=dropout if layers > 1 else 0.0,
                )
                self.head = nn.Sequential(
                    nn.Dropout(dropout), nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1)
                )

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :]).squeeze(-1)

        return LSTMNet()


class LSTMForecaster(Forecaster):
    name = "lstm"

    def __init__(self, **params):
        self.hp = {**MODELS.lstm_params, **params}
        self.min_history = int(self.hp["lookback"])
        self.model_ = None
        self.features_: list[str] = []
        self.mu_: np.ndarray | None = None
        self.sd_: np.ndarray | None = None
        self.y_mu_ = 0.0
        self.y_sd_ = 1.0
        self.history_: list[dict] = []

    # -- internals ---------------------------------------------------------
    def _scale(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features_].to_numpy(dtype=float)
        return (X - self.mu_) / self.sd_

    # -- interface ---------------------------------------------------------
    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.hp["seed"])
        np.random.seed(self.hp["seed"])

        self.features_ = feature_columns(train)
        raw_tr = train[self.features_].to_numpy(dtype=float)
        self.mu_ = raw_tr.mean(axis=0)
        self.sd_ = raw_tr.std(axis=0)
        self.sd_[self.sd_ == 0] = 1.0

        y_tr_raw = train["target_return"].to_numpy(dtype=float)
        self.y_mu_, self.y_sd_ = float(y_tr_raw.mean()), float(y_tr_raw.std() or 1.0)

        lookback = int(self.hp["lookback"])
        Xtr, ytr, _ = _build_sequences(self._scale(train),
                                       (y_tr_raw - self.y_mu_) / self.y_sd_, lookback)

        loader = DataLoader(
            TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
            batch_size=int(self.hp["batch_size"]), shuffle=True, drop_last=False,
        )

        net = _Net.make(len(self.features_), int(self.hp["hidden_size"]),
                        int(self.hp["num_layers"]), float(self.hp["dropout"]))
        opt = torch.optim.Adam(net.parameters(), lr=float(self.hp["lr"]))
        loss_fn = torch.nn.HuberLoss(delta=1.0)

        val_seq = None
        if val is not None and len(val) > 0:
            # Prepend the last lookback-1 training rows so that the first
            # validation day still gets a full window. The scaler is the
            # training-fold scaler, so no validation statistics leak in.
            ctx = pd.concat([train.tail(lookback - 1), val], ignore_index=True)
            y_ctx = (ctx["target_return"].to_numpy(dtype=float) - self.y_mu_) / self.y_sd_
            Xv, yv, _ = _build_sequences(self._scale(ctx), y_ctx, lookback)
            assert len(Xv) == len(val), f"{len(Xv)} windows for {len(val)} val rows"
            val_seq = (torch.from_numpy(Xv), torch.from_numpy(yv))

        best_loss, best_state, bad_epochs = np.inf, None, 0
        for epoch in range(int(self.hp["epochs"])):
            net.train()
            total = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(net(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                total += float(loss.detach()) * len(xb)
            train_loss = total / len(loader.dataset)

            if val_seq is not None:
                net.eval()
                with torch.no_grad():
                    v_loss = float(loss_fn(net(val_seq[0]), val_seq[1]))
            else:
                v_loss = train_loss

            self.history_.append({"epoch": epoch, "train_loss": train_loss, "val_loss": v_loss})

            if v_loss < best_loss - 1e-6:
                best_loss, bad_epochs = v_loss, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= int(self.hp["patience"]):
                    log.info("LSTM early stop at epoch %d (best val loss %.5f)", epoch, best_loss)
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self.model_ = net
        log.info("LSTM trained: %d epochs, best val loss %.5f", len(self.history_), best_loss)
        return self

    def predict_returns(self, context: pd.DataFrame, n_out: int) -> np.ndarray:
        """``context`` must end with the rows to predict, preceded by lookback-1 rows."""
        import torch

        lookback = int(self.hp["lookback"])
        X, _, _ = _build_sequences(self._scale(context), None, lookback)
        with torch.no_grad():
            out = self.model_(torch.from_numpy(X)).numpy().astype(float)
        return out[-n_out:] * self.y_sd_ + self.y_mu_

    def backtest(self, history: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        lookback = int(self.hp["lookback"])
        ctx = pd.concat([history.tail(lookback - 1), test], ignore_index=True)
        preds = self.predict_returns(ctx, len(test))
        return returns_to_prices(test["Close"].to_numpy(dtype=float), preds)

    def params(self) -> dict:
        return {
            **{f"lstm_{k}": v for k, v in self.hp.items()},
            "target": "log_return",
            "n_features": len(self.features_),
            "epochs_run": len(self.history_),
        }

    # -- pickling ----------------------------------------------------------
    # The nn.Module class is created inside a factory function, so it has no
    # importable qualname and cannot be pickled directly. We persist the
    # weights as plain arrays and rebuild the graph on load, which also makes
    # the artefact portable across torch versions.
    def __getstate__(self):
        state = self.__dict__.copy()
        net = state.pop("model_", None)
        state["_weights"] = (
            None if net is None
            else {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
        )
        return state

    def __setstate__(self, state):
        weights = state.pop("_weights", None)
        self.__dict__.update(state)
        self.model_ = None
        if weights is not None:
            import torch

            net = _Net.make(len(self.features_), int(self.hp["hidden_size"]),
                            int(self.hp["num_layers"]), float(self.hp["dropout"]))
            net.load_state_dict({k: torch.as_tensor(v) for k, v in weights.items()})
            net.eval()
            self.model_ = net
