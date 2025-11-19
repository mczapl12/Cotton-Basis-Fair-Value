# cotton_basis_fair_value_single_region_2charts.py
# Two charts, same data:
#  1) Daily z (raw, last 120d)
#  2) 7-day average of daily z (trend view)

import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
import matplotlib.dates as mdates

# -------- SETTINGS --------
DATA_PATH = os.path.expanduser("/Users/michalczaplinski/Downloads/Cotton Project/cotton_cash_basis_dummy.csv")
REGION    = "West_TX"        # must match CSV exactly
ROLL_Z    = 10               # residual rolling std window
MIN_PTS   = max(3, ROLL_Z//2)

SHOW_LAST_DAYS_DAILY  = 120  # for the raw daily chart
SHOW_LAST_DAYS_TREND  = 120  # window for the trend chart (same period)
TREND_WINDOW = "7D"          # rolling window for trend (calendar days)

# Optional: lock betas to pre-shock history (None = fit on all data)
LOCK_BETAS_BEFORE_DATE = None
# LOCK_BETAS_BEFORE_DATE = "2025-10-31"

# -------- OUTPUT FOLDER --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "outputs_single")
os.makedirs(OUT_DIR, exist_ok=True)

# -------- LOAD & CLEAN --------
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"], errors="coerce", infer_datetime_format=True)
df = df.dropna(subset=["date"])
df.columns = [c.strip() for c in df.columns]
df["region"] = df["region"].astype(str).str.strip()
for c in ["Cash_centslb","ICE_centslb","Carry_centslb","LaneFreight_centslb"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["Cash_centslb","ICE_centslb","Carry_centslb","LaneFreight_centslb"])

sub = df[df["region"]==REGION].sort_values("date").reset_index(drop=True)
if sub.empty:
    raise SystemExit(f"No rows for region '{REGION}'.")

# -------- FIT (Cash ≈ a + b1*ICE + b2*Carry + b3*Lane) --------
def fit_betas(frame):
    X = np.column_stack([np.ones(len(frame)),
                         frame[["ICE_centslb","Carry_centslb","LaneFreight_centslb"]].to_numpy()])
    y = frame["Cash_centslb"].to_numpy()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef

if LOCK_BETAS_BEFORE_DATE:
    cutoff = pd.to_datetime(LOCK_BETAS_BEFORE_DATE)
    train = sub[sub["date"] <= cutoff].copy()
    if len(train) < 5:
        raise SystemExit(f"LOCK_BETAS_BEFORE_DATE={cutoff.date()} but too few rows to train.")
    coef = fit_betas(train)
    mode = f"LOCKED (≤ {cutoff.date()})"
else:
    coef = fit_betas(sub)
    mode = "FULL-SAMPLE"

a, b_ice, b_carry, b_lane = coef

# Predictions/residuals/z
X_all = np.column_stack([np.ones(len(sub)),
                         sub[["ICE_centslb","Carry_centslb","LaneFreight_centslb"]].to_numpy()])
yhat  = X_all.dot(coef)
resid = sub["Cash_centslb"].to_numpy() - yhat

res_series = pd.Series(resid, index=sub.index)
exp_std  = res_series.expanding(min_periods=3).std()
roll_std = res_series.rolling(ROLL_Z, min_periods=MIN_PTS).std()
resid_std = np.where(roll_std.notna(), roll_std, exp_std)
z = resid / resid_std

out = sub[["date","region"]].copy()
out["z"] = z

# -------- CHART 1: DAILY z (raw) --------
daily_df = out.copy()
if SHOW_LAST_DAYS_DAILY is not None:
    cutoff2 = daily_df["date"].max() - pd.Timedelta(days=SHOW_LAST_DAYS_DAILY)
    daily_df = daily_df[daily_df["date"] >= cutoff2]

fig1, ax1 = plt.subplots(figsize=(8,5))
ax1.plot(daily_df["date"], daily_df["z"], marker="o", markersize=3, linewidth=1)
ax1.axhline( 2.0, linestyle="--"); ax1.axhline(-2.0, linestyle="--")
ax1.axhline( 1.0, linestyle=":");  ax1.axhline(-1.0, linestyle=":")
ax1.set_title(f"{REGION} — Daily z (raw, last {SHOW_LAST_DAYS_DAILY} days)\nBasis fair-value: ICE + Carry + Lane [{mode}]")
ax1.set_ylabel("z"); ax1.grid(True, alpha=0.25)
loc1 = mdates.AutoDateLocator(minticks=4, maxticks=8)
ax1.xaxis.set_major_locator(loc1); ax1.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc1))
fig1.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"basis_z_{REGION}.png"))
plt.close(fig1)

# -------- CHART 2: 7-DAY AVERAGE OF DAILY z (trend view) --------
trend_df = out.copy()
if SHOW_LAST_DAYS_TREND is not None:
    cutoff_t = trend_df["date"].max() - pd.Timedelta(days=SHOW_LAST_DAYS_TREND)
    trend_df = trend_df[trend_df["date"] >= cutoff_t]

trend = (trend_df.set_index("date")["z"]
         .rolling(TREND_WINDOW, min_periods=2)   # same daily z, lightly smoothed
         .mean()
         .reset_index(name="z_trend"))

fig2, ax2 = plt.subplots(figsize=(8,5))
ax2.plot(trend["date"], trend["z_trend"], linewidth=2)
ax2.axhline( 2.0, linestyle="--"); ax2.axhline(-2.0, linestyle="--")
ax2.axhline( 1.0, linestyle=":");  ax2.axhline(-1.0, linestyle=":")
ax2.set_title(f"{REGION} — 7-day avg of daily z (trend view, last {SHOW_LAST_DAYS_TREND} days)\nBasis fair-value: ICE + Carry + Lane [{mode}]")
ax2.set_ylabel("z"); ax2.grid(True, alpha=0.25)
loc2 = mdates.AutoDateLocator(minticks=4, maxticks=8)
ax2.xaxis.set_major_locator(loc2); ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc2))
fig2.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"basis_z_{REGION}_pretty.png"))
plt.close(fig2)

print("Saved charts to:", OUT_DIR)
