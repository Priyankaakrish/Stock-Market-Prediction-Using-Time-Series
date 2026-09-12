import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
logging.getLogger("src.models.xgb_model").setLevel(logging.WARNING)
logging.getLogger("src.models.arima_model").setLevel(logging.WARNING)

from scipy import stats

from src.models.arima_model import ARIMAForecaster
from src.models.baselines import DriftForecaster, MovingAverageForecaster, NaiveForecaster
from src.models.xgb_model import XGBForecaster
from src.split import chronological_split
from src.walkforward import summarise, walk_forward

feats = pd.read_csv("data/processed/nvda_features.csv", parse_dates=["Date"])
splits = chronological_split(feats, save=False)
n_initial = len(splits.train) + len(splits.val)

models = [NaiveForecaster(), MovingAverageForecaster(5), DriftForecaster(),
          ARIMAForecaster(order=(2, 1, 2)), XGBForecaster()]

folds = walk_forward(feats, models, n_initial=n_initial, step=63)
summary = summarise(folds)

base = folds[folds.model == "naive"].set_index("fold")["rmse"]
print("\nmodel      wins   binom p")
for m in ["drift", "xgboost", "arima", "ma5"]:
    r = folds[folds.model == m].set_index("fold")["rmse"]
    w = int((r < base).sum())
    print(f"{m:9} {w:>3}/{len(r):<3} {stats.binomtest(w, len(r), 0.5).pvalue:8.3f}")

summary.to_csv("reports/walkforward_summary.csv", index=False)
pd.set_option("display.width", 200)
print("\n" + summary.round(3).to_string(index=False))