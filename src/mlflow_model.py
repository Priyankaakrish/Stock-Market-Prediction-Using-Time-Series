"""MLflow ``pyfunc`` wrapper for the selected forecaster.

Logging a bare pickle as an artifact records the file but does not create an
entry the Model Registry can version or stage. Wrapping the forecaster in a
``PythonModel`` gives us a real registered model with a signature, so the
serving layer can pull ``models:/nvda-forecaster/Production`` instead of
hard-coding a filesystem path.
"""
from __future__ import annotations

import mlflow.pyfunc
import pandas as pd


class NVDAForecastModel(mlflow.pyfunc.PythonModel):
    """Takes a feature frame ending at day *t*, returns the day *t+1* close."""

    def load_context(self, context):
        from src.models.base import Forecaster

        self.model = Forecaster.load(context.artifacts["forecaster"])

    def predict(self, context, model_input: pd.DataFrame, params=None):
        df = model_input.copy()
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        history, target = df.iloc[:-1], df.iloc[-1:]
        preds = self.model.backtest(history, target)
        return pd.DataFrame(
            {
                "predicted_close": preds,
                "last_close": target["Close"].to_numpy(dtype=float),
                "model": self.model.name,
            }
        )


def log_and_register(pickle_path: str, registered_name: str, input_example=None):
    """Log the wrapper and register it. Returns the ModelVersion or None."""
    kwargs = {
        "python_model": NVDAForecastModel(),
        "artifacts": {"forecaster": str(pickle_path)},
        "code_paths": ["src"],
        "registered_model_name": registered_name,
    }
    if input_example is not None:
        kwargs["input_example"] = input_example

    try:                                  # MLflow >= 3
        return mlflow.pyfunc.log_model(name="best_model", **kwargs)
    except TypeError:                     # MLflow 2.x signature
        return mlflow.pyfunc.log_model(artifact_path="best_model", **kwargs)
