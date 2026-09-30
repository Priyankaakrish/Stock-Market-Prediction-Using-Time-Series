"""
src/models/lstm_model.py
-------------------------
Bidirectional LSTM with Self-Attention for NVDA stock price forecasting.

Backend: PyTorch  (pip install torch --index-url https://download.pytorch.org/whl/cpu)
Fallback: TensorFlow/Keras  (if torch is absent and TF is working)

Architecture:
  Input  (batch, lookback, n_features)
    -> BiLSTM layer 1  (return_sequences=True)
    -> Scaled dot-product Attention
    -> BiLSTM layer 2  (return last hidden)
    -> Dense + Dropout
    -> Dense output  (horizon steps)

Training:
  * Adam + ReduceLROnPlateau + EarlyStopping
  * Huber loss (robust to outliers)
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    FORECAST_DAYS,
    LSTM_BATCH_SIZE,
    LSTM_DENSE_UNITS,
    LSTM_DROPOUT,
    LSTM_EPOCHS,
    LSTM_LOOKBACK,
    LSTM_LR,
    LSTM_PATIENCE,
    LSTM_RECURRENT_DROP,
    LSTM_UNITS,
    MODELS_DIR,
    TARGET_COL,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _detect_backend() -> str:
    """Return 'torch', 'tensorflow', or 'none'."""
    try:
        import torch          # noqa: F401
        return "torch"
    except ImportError:
        pass
    except OSError as exc:    # e.g. Windows WinError 1114 (missing VC++ runtime)
        logger.warning(
            "PyTorch is installed but failed to load (%s). LSTM will be skipped. "
            "On Windows, install the Microsoft Visual C++ Redistributable (x64): "
            "https://aka.ms/vs/17/release/vc_redist.x64.exe", exc)
    try:
        import tensorflow as tf  # noqa: F401
        # Quick smoke-test — the Python 3.9 typing bug surfaces on import
        from tensorflow.keras.layers import LSTM  # noqa: F401
        return "tensorflow"
    except Exception:
        pass
    return "none"


BACKEND = _detect_backend()
logger.debug("LSTM backend: %s", BACKEND)


# ===========================================================================
# PyTorch implementation  (preferred)
# ===========================================================================

class _TorchLSTM:
    """Bidirectional LSTM + attention built with PyTorch."""

    def __init__(self, n_features, lookback, horizon, units, dropout,
                 dense_units, lr, batch_size, epochs, patience):
        self.n_features  = n_features
        self.lookback    = lookback
        self.horizon     = horizon
        self.units       = units
        self.dropout     = dropout
        self.dense_units = dense_units
        self.lr          = lr
        self.batch_size  = batch_size
        self.epochs      = epochs
        self.patience    = patience
        self._net        = None
        self.history_    = {"loss": [], "val_loss": []}

    # ---- build ----

    def _build_net(self):
        import torch
        import torch.nn as nn

        class BiLSTMAttention(nn.Module):
            def __init__(self, n_feat, units, n_dense, dropout, horizon):
                super().__init__()
                h0 = units[0]
                self.lstm1 = nn.LSTM(n_feat, h0, batch_first=True,
                                     bidirectional=True, dropout=dropout
                                     if len(units) > 1 else 0.0)
                h1 = units[1] if len(units) > 1 else h0
                self.lstm2 = nn.LSTM(h0 * 2, h1, batch_first=True,
                                     bidirectional=True)
                self.attn_w = nn.Linear(h0 * 2, 1)
                self.drop   = nn.Dropout(dropout)
                self.fc1    = nn.Linear(h1 * 2, n_dense)
                self.relu   = nn.ReLU()
                self.fc2    = nn.Linear(n_dense, horizon)

            def forward(self, x):
                out1, _ = self.lstm1(x)                    # (B, T, 2*h0)
                # Scaled dot-product attention over lstm1 outputs
                score = self.attn_w(out1).squeeze(-1)      # (B, T)
                w     = score.softmax(dim=-1).unsqueeze(-1) # (B, T, 1)
                ctx   = (out1 * w).sum(dim=1, keepdim=True).expand_as(out1)
                out2, _ = self.lstm2(ctx)                   # (B, T, 2*h1)
                z = out2[:, -1, :]                          # last step
                z = self.drop(self.relu(self.fc1(z)))
                return self.fc2(z)                          # (B, horizon)

        self._net = BiLSTMAttention(
            self.n_features, self.units, self.dense_units,
            self.dropout, self.horizon
        )
        return self._net

    # ---- fit ----

    def fit(self, X_train, y_train, X_val, y_val):
        import torch
        import torch.nn as nn
        from torch.optim import Adam
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        net = self._build_net()
        opt = Adam(net.parameters(), lr=self.lr)
        sch = ReduceLROnPlateau(opt, factor=0.5, patience=self.patience // 2,
                                min_lr=1e-6)
        loss_fn = nn.HuberLoss()

        def to_t(arr):
            return torch.tensor(arr, dtype=torch.float32)

        Xtr, ytr = to_t(X_train), to_t(y_train)
        Xva, yva = to_t(X_val),   to_t(y_val)

        ds     = torch.utils.data.TensorDataset(Xtr, ytr)
        loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size,
                                              shuffle=False)

        best_val, best_state, wait = np.inf, None, 0

        for epoch in range(self.epochs):
            net.train()
            ep_loss = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                pred = net(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                ep_loss += loss.item() * len(xb)
            ep_loss /= len(Xtr)

            net.eval()
            with torch.no_grad():
                val_loss = loss_fn(net(Xva), yva).item()

            sch.step(val_loss)
            self.history_["loss"].append(ep_loss)
            self.history_["val_loss"].append(val_loss)

            if val_loss < best_val - 1e-6:
                best_val   = val_loss
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
                wait       = 0
            else:
                wait += 1
                if wait >= self.patience:
                    logger.info("Early stopping at epoch %d (val_loss=%.6f)", epoch+1, best_val)
                    break

            if (epoch + 1) % 10 == 0:
                logger.info("Epoch %3d/%d | loss=%.6f | val_loss=%.6f",
                            epoch+1, self.epochs, ep_loss, val_loss)

        if best_state is not None:
            net.load_state_dict(best_state)
        self._net = net
        logger.info("PyTorch LSTM training complete — best val_loss=%.6f", best_val)
        return self.history_

    # ---- predict ----

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        self._net.eval()
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32)
            return self._net(t).numpy()

    def forecast_future(self, last_sequence: np.ndarray, steps: int) -> np.ndarray:
        seq   = last_sequence.copy()
        preds = []
        for _ in range(steps):
            inp    = seq[-self.lookback:][np.newaxis].astype("float32")
            y_hat  = float(self.predict(inp)[0, 0])
            preds.append(y_hat)
            new_row    = seq[-1].copy()
            new_row[0] = y_hat
            seq = np.vstack([seq, new_row])
        return np.array(preds)

    # ---- save / load ----

    def save(self, path: Path):
        import torch
        torch.save({
            "state_dict": self._net.state_dict(),
            "history":    self.history_,
            "meta": {
                "n_features":  self.n_features,
                "lookback":    self.lookback,
                "horizon":     self.horizon,
                "units":       self.units,
                "dropout":     self.dropout,
                "dense_units": self.dense_units,
                "lr":          self.lr,
            },
        }, str(path))
        logger.info("PyTorch LSTM saved -> %s", path)

    @classmethod
    def load(cls, path: Path) -> "_TorchLSTM":
        import torch
        ckpt = torch.load(str(path), map_location="cpu")
        meta = ckpt["meta"]
        obj  = cls(**meta, batch_size=LSTM_BATCH_SIZE,
                   epochs=LSTM_EPOCHS, patience=LSTM_PATIENCE)
        obj._build_net()
        obj._net.load_state_dict(ckpt["state_dict"])
        obj._net.eval()
        obj.history_ = ckpt.get("history", {"loss": [], "val_loss": []})
        logger.info("PyTorch LSTM loaded <- %s", path)
        return obj


# ===========================================================================
# TensorFlow/Keras fallback implementation
# ===========================================================================

class _TFLSTMForecaster:
    """Original TF/Keras BiLSTM — used only when PyTorch is absent."""

    def __init__(self, n_features, lookback, horizon, units, dropout,
                 recurrent_drop, dense_units, lr, batch_size, epochs, patience):
        self.n_features     = n_features
        self.lookback       = lookback
        self.horizon        = horizon
        self.units          = units
        self.dropout        = dropout
        self.recurrent_drop = recurrent_drop
        self.dense_units    = dense_units
        self.lr             = lr
        self.batch_size     = batch_size
        self.epochs         = epochs
        self.patience       = patience
        self._model         = None
        self.history_       = {"loss": [], "val_loss": []}

    def _build(self):
        import tensorflow as tf
        from tensorflow.keras.layers import (Bidirectional, Dense, Dropout,
                                              GlobalAveragePooling1D, Input, LSTM)
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam

        inp = Input(shape=(self.lookback, self.n_features))
        x   = Bidirectional(LSTM(self.units[0], return_sequences=True,
                                  dropout=self.dropout,
                                  recurrent_dropout=self.recurrent_drop))(inp)
        if len(self.units) > 1:
            x = Bidirectional(LSTM(self.units[1], return_sequences=False,
                                    dropout=self.dropout,
                                    recurrent_dropout=self.recurrent_drop))(x)
        else:
            x = GlobalAveragePooling1D()(x)
        x   = Dense(self.dense_units, activation="relu")(x)
        x   = Dropout(self.dropout)(x)
        out = Dense(self.horizon)(x)
        mdl = Model(inp, out)
        mdl.compile(optimizer=Adam(self.lr), loss="huber", metrics=["mae"])
        self._model = mdl
        return mdl

    def fit(self, X_train, y_train, X_val, y_val):
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        self._build()
        cbs = [
            EarlyStopping(monitor="val_loss", patience=self.patience,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=self.patience // 2, min_lr=1e-6),
        ]
        hist = self._model.fit(
            X_train, y_train, validation_data=(X_val, y_val),
            epochs=self.epochs, batch_size=self.batch_size,
            callbacks=cbs, verbose=1, shuffle=False,
        )
        self.history_ = hist.history
        return self.history_

    def predict(self, X):
        return self._model.predict(X, batch_size=self.batch_size, verbose=0)

    def forecast_future(self, last_sequence, steps):
        seq, preds = last_sequence.copy(), []
        for _ in range(steps):
            inp    = seq[-self.lookback:][np.newaxis].astype("float32")
            y_hat  = float(self.predict(inp)[0, 0])
            preds.append(y_hat)
            nr = seq[-1].copy(); nr[0] = y_hat
            seq = np.vstack([seq, nr])
        return np.array(preds)

    def save(self, path: Path):
        self._model.save(str(path))
        logger.info("TF LSTM saved -> %s", path)

    @classmethod
    def load(cls, path: Path) -> "_TFLSTMForecaster":
        import tensorflow as tf
        obj = cls(n_features=1, lookback=LSTM_LOOKBACK, horizon=1,
                  units=LSTM_UNITS, dropout=LSTM_DROPOUT,
                  recurrent_drop=LSTM_RECURRENT_DROP, dense_units=LSTM_DENSE_UNITS,
                  lr=LSTM_LR, batch_size=LSTM_BATCH_SIZE,
                  epochs=LSTM_EPOCHS, patience=LSTM_PATIENCE)
        obj._model = tf.keras.models.load_model(str(path))
        return obj


# ===========================================================================
# Public facade — automatically picks the right backend
# ===========================================================================

class LSTMForecaster:
    """
    Public LSTM API — wraps PyTorch or TF depending on what is installed.

    Usage:
        lf = LSTMForecaster(n_features=50)
        lf.build()
        history = lf.fit(X_train, y_train, X_val, y_val)
        preds   = lf.predict(X_test)
        lf.save()
    """

    def __init__(
        self,
        n_features:     int   = 1,
        lookback:       int   = LSTM_LOOKBACK,
        horizon:        int   = 1,
        units:          List[int] = LSTM_UNITS,
        dropout:        float = LSTM_DROPOUT,
        recurrent_drop: float = LSTM_RECURRENT_DROP,
        dense_units:    int   = LSTM_DENSE_UNITS,
        lr:             float = LSTM_LR,
        batch_size:     int   = LSTM_BATCH_SIZE,
        epochs:         int   = LSTM_EPOCHS,
        patience:       int   = LSTM_PATIENCE,
    ):
        self.n_features     = n_features
        self.lookback       = lookback
        self.horizon        = horizon
        self.units          = units
        self.dropout        = dropout
        self.recurrent_drop = recurrent_drop
        self.dense_units    = dense_units
        self.lr             = lr
        self.batch_size     = batch_size
        self.epochs         = epochs
        self.patience       = patience
        self._impl          = None
        self._backend       = BACKEND

    # ---- build ----

    def build(self) -> "LSTMForecaster":
        if self._backend == "torch":
            self._impl = _TorchLSTM(
                self.n_features, self.lookback, self.horizon,
                self.units, self.dropout, self.dense_units,
                self.lr, self.batch_size, self.epochs, self.patience,
            )
        elif self._backend == "tensorflow":
            self._impl = _TFLSTMForecaster(
                self.n_features, self.lookback, self.horizon,
                self.units, self.dropout, self.recurrent_drop,
                self.dense_units, self.lr, self.batch_size,
                self.epochs, self.patience,
            )
        else:
            raise RuntimeError(
                "No deep-learning backend found.\n"
                "Install PyTorch (recommended for Python 3.9 Windows):\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
                "Or TensorFlow 2.12:\n"
                "  pip install tensorflow==2.12.0"
            )
        logger.info("LSTM backend: %s", self._backend)
        return self

    # ---- fit / predict / forecast ----

    def fit(self, X_train, y_train, X_val, y_val) -> Dict:
        self.history_ = self._impl.fit(X_train, y_train, X_val, y_val)
        return self.history_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._impl.predict(X)

    def forecast_future(self, last_sequence: np.ndarray,
                        steps: int = FORECAST_DAYS) -> np.ndarray:
        return self._impl.forecast_future(last_sequence, steps)

    @property
    def history_(self):
        return getattr(self._impl, "history_", {"loss": [], "val_loss": []})

    @history_.setter
    def history_(self, v):
        if self._impl is not None:
            self._impl.history_ = v

    # ---- save / load ----

    def save(self, path: Optional[Path] = None) -> Path:
        suffix = ".pt" if self._backend == "torch" else ".h5"
        path   = Path(path or MODELS_DIR / f"lstm_model{suffix}")

        self._impl.save(path)

        meta = {
            "backend":      self._backend,
            "n_features":   self.n_features,
            "lookback":     self.lookback,
            "horizon":      self.horizon,
            "units":        self.units,
            "dropout":      self.dropout,
            "recurrent_drop": self.recurrent_drop,
            "dense_units":  self.dense_units,
            "lr":           self.lr,
        }
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(meta, f, indent=2)

        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "LSTMForecaster":
        # Try .pt first (torch), then .h5 (TF)
        if path is None:
            pt = MODELS_DIR / "lstm_model.pt"
            h5 = MODELS_DIR / "lstm_model.h5"
            path = pt if pt.exists() else h5

        path = Path(path)
        meta_path = path.with_suffix(".json")

        obj = cls()
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            for k, v in meta.items():
                if hasattr(obj, k):
                    setattr(obj, k, v)
            obj._backend = meta.get("backend", BACKEND)

        if obj._backend == "torch" or path.suffix == ".pt":
            obj._impl = _TorchLSTM.load(path)
        else:
            obj._impl = _TFLSTMForecaster.load(path)

        logger.info("LSTM model loaded <- %s (backend=%s)", path, obj._backend)
        return obj