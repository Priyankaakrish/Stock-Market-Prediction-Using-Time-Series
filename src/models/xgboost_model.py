"""
src/models/xgboost_model.py
────────────────────────────
XGBoost + LightGBM gradient-boosting models for tabular feature forecasting.

Features:
  • Optuna-based hyperparameter optimisation (TPE sampler)
  • Time-series aware cross-validation (TimeSeriesSplit)
  • SHAP-based feature importance analysis
  • Both XGBoost and LightGBM supported (selected via model_type param)
  • Joblib serialisation
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from config import (
    FORECAST_DAYS,
    LGBM_N_TRIALS,
    MODELS_DIR,
    TARGET_COL,
    XGB_CV_FOLDS,
    XGB_EARLY_STOP,
    XGB_N_TRIALS,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class GBForecaster:
    """
    Gradient-Boosting forecaster supporting XGBoost and LightGBM backends.

    Usage:
        gb = GBForecaster(model_type="xgboost")
        gb.optimise(X_train, y_train)          # Optuna HPO
        gb.fit(X_train, y_train)
        preds = gb.predict(X_test)
        gb.shap_importance(X_test)
        gb.save()
    """

    def __init__(
        self,
        model_type:  str = "xgboost",   # "xgboost" | "lightgbm"
        n_trials:    int = XGB_N_TRIALS,
        n_splits:    int = XGB_CV_FOLDS,
        early_stop:  int = XGB_EARLY_STOP,
        random_state:int = 42,
    ):
        assert model_type in ("xgboost", "lightgbm"), \
            "model_type must be 'xgboost' or 'lightgbm'"
        self.model_type   = model_type
        self.n_trials     = n_trials
        self.n_splits     = n_splits
        self.early_stop   = early_stop
        self.random_state = random_state
        self._model       = None
        self.best_params_ : Dict = {}
        self.feature_names_: List[str] = []

    # ── Optuna HPO ────────────────────────────────────────────────────────────

    def optimise(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        Run Optuna hyperparameter search with TimeSeriesSplit CV.
        Sets self.best_params_ and returns the params dict.
        """
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        if feature_names is not None:
            self.feature_names_ = feature_names

        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        def objective(trial: "optuna.Trial") -> float:
            if self.model_type == "xgboost":
                params = self._xgb_space(trial)
                model  = self._make_xgb(params)
            else:
                params = self._lgbm_space(trial)
                model  = self._make_lgbm(params)

            scores = []
            for tr_idx, va_idx in tscv.split(X):
                X_tr, X_va = X[tr_idx], X[va_idx]
                y_tr, y_va = y[tr_idx], y[va_idx]

                if self.model_type == "xgboost":
                    import xgboost as xgb
                    dtr = xgb.DMatrix(X_tr, label=y_tr,
                                      feature_names=self.feature_names_ or None)
                    dva = xgb.DMatrix(X_va, label=y_va,
                                      feature_names=self.feature_names_ or None)
                    res = {}
                    xgb.train(params, dtr,
                               num_boost_round=500,
                               evals=[(dva, "val")],
                               early_stopping_rounds=self.early_stop,
                               evals_result=res,
                               verbose_eval=False)
                    scores.append(min(res["val"]["rmse"]))
                else:
                    import lightgbm as lgb
                    dtr = lgb.Dataset(X_tr, label=y_tr)
                    dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
                    cb_log = lgb.log_evaluation(period=-1)
                    cb_early = lgb.early_stopping(self.early_stop, verbose=False)
                    booster = lgb.train(
                        params, dtr,
                        num_boost_round=500,
                        valid_sets=[dva],
                        callbacks=[cb_log, cb_early],
                    )
                    preds = booster.predict(X_va)
                    scores.append(np.sqrt(np.mean((preds - y_va) ** 2)))

            return float(np.mean(scores))

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        self.best_params_ = study.best_params
        logger.info(
            "Optuna optimisation done — best RMSE=%.4f, params=%s",
            study.best_value,
            self.best_params_,
        )
        return self.best_params_

    # ── Hyper-parameter search spaces ─────────────────────────────────────────

    def _xgb_space(self, trial) -> Dict:
        return {
            "objective":        "reg:squarederror",
            "eval_metric":      "rmse",
            "tree_method":      "hist",
            "eta":              trial.suggest_float("eta", 0.01, 0.3, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "lambda":           trial.suggest_float("lambda", 1e-3, 10, log=True),
            "alpha":            trial.suggest_float("alpha", 1e-3, 10, log=True),
            "seed":             self.random_state,
        }

    def _lgbm_space(self, trial) -> Dict:
        return {
            "objective":       "regression",
            "metric":          "rmse",
            "verbosity":       -1,
            "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves":      trial.suggest_int("num_leaves", 20, 300),
            "max_depth":       trial.suggest_int("max_depth", 3, 12),
            "min_child_samples":trial.suggest_int("min_child_samples", 5, 100),
            "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":       trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda":      trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "seed":            self.random_state,
        }

    # ── Model factories ───────────────────────────────────────────────────────

    def _make_xgb(self, params):
        import xgboost as xgb
        return xgb.XGBRegressor(
            **{k: v for k, v in params.items()
               if k not in ("objective", "eval_metric", "tree_method")},
            n_estimators=500,
            early_stopping_rounds=self.early_stop,
            random_state=self.random_state,
            verbosity=0,
        )

    def _make_lgbm(self, params):
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            **{k: v for k, v in params.items() if k not in ("objective", "metric")},
            n_estimators=500,
            random_state=self.random_state,
        )

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   Optional[np.ndarray] = None,
        y_val:   Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "GBForecaster":
        """Fit the final model using best_params_ (or defaults if not optimised)."""
        if feature_names:
            self.feature_names_ = feature_names

        params = self.best_params_ or {}

        if self.model_type == "xgboost":
            import xgboost as xgb
            self._model = xgb.XGBRegressor(
                n_estimators=1000,
                early_stopping_rounds=self.early_stop if X_val is not None else None,
                random_state=self.random_state,
                verbosity=0,
                **{k: v for k, v in params.items()
                   if k not in ("objective", "eval_metric", "tree_method", "seed")},
            )
            eval_set = [(X_val, y_val)] if X_val is not None else None
            self._model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False,
            )
        else:
            import lightgbm as lgb
            self._model = lgb.LGBMRegressor(
                n_estimators=1000,
                random_state=self.random_state,
                **{k: v for k, v in params.items()
                   if k not in ("objective", "metric", "verbosity", "seed")},
            )
            callbacks = []
            if X_val is not None:
                callbacks = [
                    lgb.early_stopping(self.early_stop, verbose=False),
                    lgb.log_evaluation(period=-1),
                ]
            self._model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)] if X_val is not None else None,
                callbacks=callbacks or None,
            )

        logger.info("%s model fitted.", self.model_type.upper())
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        return self._model.predict(X)

    def predict_multi_step(
        self,
        X_last: np.ndarray,
        steps: int = FORECAST_DAYS,
    ) -> np.ndarray:
        """
        Recursive multi-step forecast.
        X_last : (1, n_features) — the last available feature row
        """
        preds   = []
        x_curr  = X_last.copy()
        for _ in range(steps):
            y_hat = float(self.predict(x_curr)[0])
            preds.append(y_hat)
            # Slide window: shift lag features
            x_curr = np.roll(x_curr, -1, axis=1)
            x_curr[0, -1] = y_hat
        return np.array(preds)

    # ── SHAP Feature Importance ───────────────────────────────────────────────

    def shap_importance(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        Compute SHAP values and return a feature-importance DataFrame.
        """
        try:
            import shap

            names = feature_names or self.feature_names_ or \
                    [f"f{i}" for i in range(X.shape[1])]

            explainer   = shap.TreeExplainer(self._model)
            shap_values = explainer.shap_values(X)

            importance = pd.DataFrame({
                "feature":    names,
                "mean_|shap|": np.abs(shap_values).mean(axis=0),
            }).sort_values("mean_|shap|", ascending=False).head(top_n)

            logger.info("SHAP importance computed for top %d features.", top_n)
            return importance

        except ImportError:
            logger.warning("shap not installed — returning built-in feature importance")
            if hasattr(self._model, "feature_importances_"):
                names = feature_names or self.feature_names_ or \
                        [f"f{i}" for i in range(len(self._model.feature_importances_))]
                return pd.DataFrame({
                    "feature":    names,
                    "importance": self._model.feature_importances_,
                }).sort_values("importance", ascending=False).head(top_n)
            return pd.DataFrame()

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> Path:
        path = Path(path or MODELS_DIR / f"{self.model_type}_model.pkl")
        joblib.dump({
            "model":        self._model,
            "model_type":   self.model_type,
            "best_params":  self.best_params_,
            "feature_names":self.feature_names_,
        }, path)
        logger.info("%s model saved -> %s", self.model_type.upper(), path)
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None, model_type: str = "xgboost") -> "GBForecaster":
        path = Path(path or MODELS_DIR / f"{model_type}_model.pkl")
        data = joblib.load(path)
        obj  = cls(model_type=data["model_type"])
        obj._model        = data["model"]
        obj.best_params_  = data["best_params"]
        obj.feature_names_= data["feature_names"]
        logger.info("%s model loaded ← %s", model_type.upper(), path)
        return obj
