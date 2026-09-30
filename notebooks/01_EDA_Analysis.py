"""
notebooks/01_EDA_Analysis.py
─────────────────────────────
Comprehensive Exploratory Data Analysis for NVIDIA (NVDA) Stock.

Run as a script to generate all plots + a summary HTML report:
    cd nvda_stock_prediction
    python notebooks/01_EDA_Analysis.py

Or open as a Jupyter notebook:
    jupytext --to notebook notebooks/01_EDA_Analysis.py
    jupyter notebook notebooks/01_EDA_Analysis.ipynb

Sections:
  1.  Data Overview & Source Comparison
  2.  Price History (26 years)
  3.  Returns Distribution & Fat Tails
  4.  Volatility Analysis (Rolling, GARCH-like)
  5.  Seasonality & Calendar Patterns
  6.  Technical Indicators Deep-Dive
  7.  Volume Analysis
  8.  Stationarity Tests (ADF, KPSS, PP)
  9.  Autocorrelation (ACF / PACF)
  10. Feature Correlation Heatmap
  11. Train / Val / Test Split Visualisation
"""

# %% [markdown]
# # NVDA Stock Price — Exploratory Data Analysis
# **NVIDIA Corporation (NASDAQ: NVDA) | 1999 – Present**

# %%  Imports
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from src.data.loader import load_all_sources
from src.data.preprocessor import clean, temporal_split
from src.features.engineer import build_feature_set, get_feature_columns

# Style
plt.rcParams.update({
    "figure.dpi": 120,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})
PALETTE = ["#003366", "#E31837", "#00897B", "#FFA726", "#7B1FA2"]
OUTPUT_DIR = Path(__file__).parent / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)


def savefig(name: str):
    p = OUTPUT_DIR / f"{name}.png"
    plt.savefig(p, bbox_inches="tight")
    print(f"  Saved: {p}")
    plt.show()
    plt.close()


# ── 1. Load data ──────────────────────────────────────────────────────────────
print("=" * 60)
print("Section 1: Data Overview")
print("=" * 60)

df_raw   = load_all_sources()
df_clean = clean(df_raw)
df_feat  = build_feature_set(df_clean)

print(f"\nShape          : {df_raw.shape}")
print(f"Date range     : {df_raw.index.min().date()} → {df_raw.index.max().date()}")
print(f"Trading days   : {len(df_raw):,}")
print(f"\nHead:\n{df_raw.head()}")
print(f"\nDescribe:\n{df_raw[['Open','High','Low','Close','Volume']].describe().round(2)}")
print(f"\nMissing values:\n{df_raw.isna().sum()}")

# ── 2. Price history ──────────────────────────────────────────────────────────
print("\nSection 2: Price History")
fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

axes[0].plot(df_clean.index, df_clean["Close"], color=PALETTE[0], linewidth=0.8)
axes[0].set_ylabel("Price (USD)")
axes[0].set_title("NVIDIA (NVDA) — Closing Price 1999–Present", fontsize=14, fontweight="bold")
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

# Volume bars
axes[1].bar(df_clean.index, df_clean["Volume"] / 1e6,
            color=PALETTE[2], alpha=0.6, width=1.0)
axes[1].set_ylabel("Volume (M shares)")
axes[1].set_title("Daily Trading Volume")

# Daily returns
axes[2].plot(df_clean.index, df_clean["Return"] * 100,
             color=PALETTE[1], linewidth=0.5, alpha=0.7)
axes[2].axhline(0, color="black", linewidth=0.5)
axes[2].set_ylabel("Return (%)")
axes[2].set_title("Daily Returns")
axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
savefig("01_price_history")

# ── 3. Returns distribution ───────────────────────────────────────────────────
print("\nSection 3: Returns Distribution")
returns = df_clean["Return"].dropna()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Histogram + KDE vs Normal
axes[0].hist(returns, bins=120, density=True, color=PALETTE[0], alpha=0.6,
             label="Actual returns")
x = np.linspace(returns.min(), returns.max(), 500)
axes[0].plot(x, stats.norm.pdf(x, returns.mean(), returns.std()),
             color=PALETTE[1], linewidth=2, label="Normal fit")
axes[0].plot(x, stats.t.pdf(x, *stats.t.fit(returns)),
             color=PALETTE[2], linewidth=2, linestyle="--", label="Student-t fit")
axes[0].set_xlabel("Daily Return")
axes[0].set_title("Return Distribution")
axes[0].legend()

# Q-Q plot
(osm, osr), (slope, intercept, r) = stats.probplot(returns, dist="norm")
axes[1].scatter(osm, osr, s=2, color=PALETTE[0], alpha=0.4)
axes[1].plot(osm, slope * np.array(osm) + intercept, color=PALETTE[1], linewidth=2)
axes[1].set_xlabel("Theoretical quantiles")
axes[1].set_ylabel("Sample quantiles")
axes[1].set_title("Q-Q Plot (vs Normal)")

# Rolling skewness & kurtosis
roll = returns.rolling(63)
axes[2].plot(df_clean.index[63:], roll.skew().dropna(),
             label="Skewness (63-day)", color=PALETTE[0])
axes[2].plot(df_clean.index[63:], roll.kurt().dropna(),
             label="Excess Kurtosis (63-day)", color=PALETTE[1])
axes[2].axhline(0, color="gray", linewidth=0.5)
axes[2].set_title("Rolling Skewness & Kurtosis")
axes[2].legend()
axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
savefig("02_returns_distribution")

print(f"\nReturn stats:")
print(f"  Mean         : {returns.mean()*252:.2%} (annualised)")
print(f"  Volatility   : {returns.std()*np.sqrt(252):.2%} (annualised)")
print(f"  Skewness     : {returns.skew():.3f}")
print(f"  Kurtosis     : {returns.kurtosis():.3f}")
print(f"  Min daily    : {returns.min():.2%}")
print(f"  Max daily    : {returns.max():.2%}")

# ── 4. Volatility ─────────────────────────────────────────────────────────────
print("\nSection 4: Volatility Analysis")
log_ret = df_clean["Log_Return"].dropna()

fig, ax = plt.subplots(figsize=(16, 5))
for w, col in zip([21, 63, 126], PALETTE):
    hv = log_ret.rolling(w).std() * np.sqrt(252) * 100
    ax.plot(hv.index, hv, linewidth=0.8, alpha=0.85, color=col,
            label=f"HV {w}-day (%)")
ax.set_ylabel("Historical Volatility (%)")
ax.set_title("NVIDIA — Historical Volatility")
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()
savefig("03_volatility")

# ── 5. Seasonality ────────────────────────────────────────────────────────────
print("\nSection 5: Seasonality Patterns")
df_cal = df_clean.copy()
df_cal["Month"] = df_cal.index.month
df_cal["DayOfWeek"] = df_cal.index.dayofweek
df_cal["Return"] = df_cal["Return"] * 100  # in %

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
month_names = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
month_means = df_cal.groupby("Month")["Return"].mean()
axes[0].bar(range(1, 13), month_means.values,
            color=[PALETTE[1] if v < 0 else PALETTE[0] for v in month_means.values])
axes[0].set_xticks(range(1, 13))
axes[0].set_xticklabels(month_names)
axes[0].axhline(0, color="black", linewidth=0.5)
axes[0].set_ylabel("Mean Daily Return (%)")
axes[0].set_title("Monthly Seasonality")

day_names = ["Mon","Tue","Wed","Thu","Fri"]
day_means = df_cal.groupby("DayOfWeek")["Return"].mean()
axes[1].bar(range(5), day_means.values,
            color=[PALETTE[1] if v < 0 else PALETTE[0] for v in day_means.values])
axes[1].set_xticks(range(5))
axes[1].set_xticklabels(day_names)
axes[1].axhline(0, color="black", linewidth=0.5)
axes[1].set_ylabel("Mean Daily Return (%)")
axes[1].set_title("Day-of-Week Seasonality")

plt.tight_layout()
savefig("04_seasonality")

# ── 6. Technical indicators ───────────────────────────────────────────────────
print("\nSection 6: Technical Indicators")
recent = df_feat.tail(252)   # last year

fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=True)

# Price + Bollinger Bands
axes[0].plot(recent.index, recent["Close"], color=PALETTE[0], linewidth=1.2, label="Close")
axes[0].plot(recent.index, recent["BB_Upper"], color=PALETTE[3], linewidth=0.7,
             linestyle="--", label="BB Upper")
axes[0].plot(recent.index, recent["BB_Lower"], color=PALETTE[3], linewidth=0.7,
             linestyle="--", label="BB Lower")
axes[0].fill_between(recent.index, recent["BB_Upper"], recent["BB_Lower"],
                     alpha=0.1, color=PALETTE[3])
axes[0].plot(recent.index, recent["SMA_50"], color=PALETTE[2], linewidth=0.9,
             label="SMA 50")
axes[0].plot(recent.index, recent["SMA_200"], color=PALETTE[4], linewidth=0.9,
             label="SMA 200")
axes[0].set_ylabel("Price (USD)")
axes[0].set_title("Price + Bollinger Bands + Moving Averages", fontweight="bold")
axes[0].legend(ncol=3, fontsize=8)

# RSI
axes[1].plot(recent.index, recent["RSI"], color=PALETTE[0], linewidth=1)
axes[1].axhline(70, color=PALETTE[1], linewidth=0.8, linestyle="--", label="Overbought (70)")
axes[1].axhline(30, color=PALETTE[2], linewidth=0.8, linestyle="--", label="Oversold (30)")
axes[1].fill_between(recent.index, 70, recent["RSI"].clip(upper=70),
                     alpha=0.2, color=PALETTE[1])
axes[1].fill_between(recent.index, recent["RSI"].clip(lower=30), 30,
                     alpha=0.2, color=PALETTE[2])
axes[1].set_ylabel("RSI")
axes[1].set_ylim(0, 100)
axes[1].legend(fontsize=8)
axes[1].set_title("RSI (14-day)")

# MACD
axes[2].plot(recent.index, recent["MACD"], color=PALETTE[0], linewidth=1, label="MACD")
axes[2].plot(recent.index, recent["MACD_Signal"], color=PALETTE[1], linewidth=1,
             linestyle="--", label="Signal")
axes[2].bar(recent.index, recent["MACD_Hist"],
            color=[PALETTE[0] if v >= 0 else PALETTE[1] for v in recent["MACD_Hist"]],
            alpha=0.5, width=1.0, label="Histogram")
axes[2].axhline(0, color="gray", linewidth=0.5)
axes[2].set_ylabel("MACD")
axes[2].legend(fontsize=8)
axes[2].set_title("MACD")

# Volume
if "Volume" in recent.columns:
    axes[3].bar(recent.index, recent["Volume"] / 1e6, color=PALETTE[2], alpha=0.7)
    axes[3].plot(recent.index, recent.get("Volume_SMA_20", recent["Volume"]) / 1e6,
                 color=PALETTE[1], linewidth=1.2, label="Volume SMA 20")
    axes[3].set_ylabel("Volume (M)")
    axes[3].set_title("Volume")
    axes[3].legend(fontsize=8)

plt.tight_layout()
savefig("05_technical_indicators")

# ── 7. Stationarity tests ─────────────────────────────────────────────────────
print("\nSection 7: Stationarity Tests")
close = df_clean["Close"].dropna()
ret   = df_clean["Return"].dropna()

for name, series in [("Close Price (Level)", close), ("Daily Return", ret)]:
    adf = adfuller(series, autolag="AIC")
    try:
        # nlags="auto" triggers a KeyError on statsmodels < 0.14 — use int fallback
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            kp = kpss(series, regression="c", nlags=int(12 * (len(series) / 100) ** 0.25))
        kpss_str = f"KPSS Stat={kp[0]:.4f}, p={kp[1]:.4f}"
    except Exception:
        kpss_str = "KPSS unavailable"
    print(f"\n  {name}:")
    print(f"    ADF  Stat={adf[0]:.4f}, p={adf[1]:.4f} {'[STATIONARY]' if adf[1]<0.05 else '[NON-STATIONARY]'}")
    print(f"    {kpss_str}")

# ── 8. ACF / PACF ─────────────────────────────────────────────────────────────
print("\nSection 8: ACF / PACF")
fig, axes = plt.subplots(2, 2, figsize=(16, 8))

plot_acf(close,  lags=40, ax=axes[0, 0], title="ACF — Close Price")
plot_pacf(close, lags=40, ax=axes[0, 1], method="ywm",
          title="PACF — Close Price")
plot_acf(ret,    lags=40, ax=axes[1, 0], title="ACF — Daily Returns")
plot_pacf(ret,   lags=40, ax=axes[1, 1], method="ywm",
          title="PACF — Daily Returns")

plt.tight_layout()
savefig("06_acf_pacf")

# ── 9. Feature correlation heatmap ────────────────────────────────────────────
print("\nSection 9: Feature Correlation Heatmap")
feat_cols = get_feature_columns(df_feat)
# Select top 20 most-correlated features with Close
corr_with_close = df_feat[feat_cols].corr()["Close"].abs().sort_values(ascending=False)
top20 = corr_with_close.index[:21].tolist()   # include Close itself

corr_matrix = df_feat[top20].corr()
fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(
    corr_matrix,
    ax=ax,
    cmap="RdBu_r",
    center=0,
    vmin=-1, vmax=1,
    annot=True, fmt=".2f",
    annot_kws={"size": 7},
    linewidths=0.5,
)
ax.set_title("Feature Correlation Matrix (Top 20 features by |corr| with Close)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
savefig("07_correlation_heatmap")

# ── 10. Train/Val/Test split ──────────────────────────────────────────────────
print("\nSection 10: Data Split Visualisation")
train_df, val_df, test_df = temporal_split(df_clean)

fig, ax = plt.subplots(figsize=(16, 5))
for subset, label, color in [
    (train_df, f"Train ({len(train_df):,} days)", PALETTE[0]),
    (val_df,   f"Val   ({len(val_df):,} days)",   PALETTE[2]),
    (test_df,  f"Test  ({len(test_df):,} days)",  PALETTE[1]),
]:
    ax.plot(subset.index, subset["Close"], color=color, linewidth=0.9, label=label)

ax.axvline(val_df.index.min(), color="gray", linewidth=1.0, linestyle="--")
ax.axvline(test_df.index.min(), color="gray", linewidth=1.0, linestyle="--")
ax.set_ylabel("Close Price (USD)")
ax.set_title("Train / Validation / Test Split", fontsize=13, fontweight="bold")
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
plt.tight_layout()
savefig("08_data_split")

print(f"\n  Train : {train_df.index.min().date()} → {train_df.index.max().date()} ({len(train_df):,} days)")
print(f"  Val   : {val_df.index.min().date()} → {val_df.index.max().date()} ({len(val_df):,} days)")
print(f"  Test  : {test_df.index.min().date()} → {test_df.index.max().date()} ({len(test_df):,} days)")

print("\n" + "=" * 60)
print("EDA complete. Plots saved to:", OUTPUT_DIR)
print("=" * 60)
