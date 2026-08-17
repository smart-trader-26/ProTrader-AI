"""
Study 1 -- long-horizon, real-data evaluation of the deployed Dynamic Fusion
Framework's uncertainty-softmax weighting mechanism, restricted to the two
sources with genuine multi-year history: Technical (price/indicator) and
Volatility (India VIX).

This is an offline, statistically-powered re-implementation of the exact
fusion mathematics found in models/fusion_framework.py:
    w_i = exp(-sigma_i^2) / sum_j exp(-sigma_j^2)
where sigma_i^2 is the mean of the last MAX_ERROR_WINDOW=10 squared errors
of source i (config/settings.py:160), applied to real OHLCV technical
features (utils/technical_indicators.py) and real India VIX features
(models/volatility_expert.py: VIX close, VIX/MA20 ratio, 20D stock vol),
exactly as consumed by models/technical_expert.py and
models/volatility_expert.py. The two TensorFlow experts (GRU / MLP) are
replaced here by Ridge regression on the identical feature sets, refit
quarterly on an expanding window, because retraining two deep nets on every
quarter x 45 tickers over 10 years is computationally prohibitive and does
not change the object under test (the fusion weighting rule itself).

Universe: NIFTY50_TICKERS (models/cross_sectional_trainer.py), the real
production ticker list. Data: yfinance, 2016-01-01 to present (real, no
synthetic rows). Target: next-day simple return (Returns.shift(-1)),
matching fusion_framework.py:72 exactly.

Outputs (all real, computed from the run below):
    results/study1_daily.csv       -- date, source preds, weights, realized return
    results/study1_summary.json    -- headline metrics + per-year table
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

START = "2016-01-01"
END = None  # today
MAX_ERROR_WINDOW = 10          # config/settings.py:160 (real constant)
VIX_TICKER = "^INDIAVIX"       # config/settings.py:231 (real constant)

NIFTY50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "BAJFINANCE.NS", "WIPRO.NS", "ASIANPAINT.NS", "MARUTI.NS", "HCLTECH.NS",
    "AXISBANK.NS", "LT.NS", "SUNPHARMA.NS", "TITAN.NS", "ONGC.NS",
    "ULTRACEMCO.NS", "POWERGRID.NS", "NTPC.NS", "JSWSTEEL.NS", "TATAMOTORS.NS",
    "TATASTEEL.NS", "ADANIPORTS.NS", "COALINDIA.NS", "BAJAJFINSV.NS", "NESTLEIND.NS",
    "TECHM.NS", "GRASIM.NS", "DIVISLAB.NS", "CIPLA.NS", "DRREDDY.NS",
    "EICHERMOT.NS", "BPCL.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "APOLLOHOSP.NS",
    "TATACONSUM.NS", "UPL.NS", "BRITANNIA.NS", "SHREECEM.NS", "INDUSINDBK.NS",
]

TECH_FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "MA5", "MA20", "MA50", "MA_Ratio_5_20",
    "Volatility_5D", "Volatility_20D", "ATR",
    "Volume_Ratio", "RSI", "MACD", "MACD_Histogram",
    "Price_vs_MA20", "Price_vs_MA50", "Gap",
]
VOL_FEATURE_COLS = ["VIX_Close", "VIX_vs_MA20", "Volatility_20D"]


def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(prices: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, macd_line - signal_line


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def build_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors utils/technical_indicators.py::calculate_technical_indicators."""
    df = df.copy()
    df["Returns"] = df["Close"].pct_change()
    df["MA5"] = df["Close"].rolling(5, min_periods=5).mean()
    df["MA20"] = df["Close"].rolling(20, min_periods=20).mean()
    df["MA50"] = df["Close"].rolling(50, min_periods=50).mean()
    df["MA_Ratio_5_20"] = df["MA5"] / df["MA20"]
    df["Volatility_5D"] = df["Returns"].rolling(5, min_periods=1).std()
    df["Volatility_20D"] = df["Returns"].rolling(20, min_periods=1).std()
    df["ATR"] = atr(df)
    df["Volume_MA20"] = df["Volume"].rolling(20, min_periods=20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA20"]
    df["RSI"] = rsi(df["Close"])
    df["MACD"], df["MACD_Histogram"] = macd(df["Close"])
    df["Price_vs_MA20"] = df["Close"] / df["MA20"] - 1
    df["Price_vs_MA50"] = df["Close"] / df["MA50"] - 1
    df["Gap"] = df["Open"] / df["Close"].shift(1) - 1
    df["Target"] = df["Returns"].shift(-1)  # fusion_framework.py:72
    return df


def main():
    print(f"Downloading {len(NIFTY50_TICKERS)} tickers + VIX from yfinance ({START}..today)...")
    raw = yf.download(NIFTY50_TICKERS, start=START, end=END, group_by="ticker",
                       auto_adjust=True, threads=True, progress=False)
    vix_raw = yf.download(VIX_TICKER, start=START, end=END, auto_adjust=True,
                           progress=False)
    if isinstance(vix_raw.columns, pd.MultiIndex):
        vix_close = vix_raw[("Close", VIX_TICKER)] if (("Close", VIX_TICKER) in vix_raw.columns) else vix_raw["Close"].iloc[:, 0]
    else:
        vix_close = vix_raw["Close"]
    vix_close = vix_close.rename("VIX_Close").dropna()
    vix_ma20 = vix_close.rolling(20, min_periods=20).mean()
    vix_vs_ma20 = (vix_close / vix_ma20).rename("VIX_vs_MA20")
    vix_df = pd.concat([vix_close, vix_vs_ma20], axis=1)

    panels = []
    n_ok = 0
    for tkr in NIFTY50_TICKERS:
        try:
            sub = raw[tkr].dropna(how="all")
        except (KeyError, TypeError):
            print(f"  skip {tkr}: no data")
            continue
        if sub.empty or len(sub) < 300:
            print(f"  skip {tkr}: insufficient history ({len(sub)} rows)")
            continue
        sub = build_technical_features(sub)
        sub = sub.join(vix_df, how="left")
        sub["Volatility_20D_stock"] = sub["Volatility_20D"]
        sub["Ticker"] = tkr
        panels.append(sub)
        n_ok += 1
    print(f"Built feature panels for {n_ok} tickers.")

    panel = pd.concat(panels)
    panel = panel.reset_index().rename(columns={"index": "Date", "Datetime": "Date"})
    panel["Date"] = pd.to_datetime(panel["Date"])

    needed = TECH_FEATURE_COLS + ["VIX_Close", "VIX_vs_MA20", "Target", "Date", "Ticker"]
    panel = panel.dropna(subset=needed)
    panel = panel.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    print(f"Full pooled panel: {len(panel)} ticker-day rows, "
          f"{panel['Date'].min().date()} .. {panel['Date'].max().date()}, "
          f"{panel['Ticker'].nunique()} tickers.")

    panel["VOL_Close_feat"] = panel["VIX_Close"]
    panel["VOL_vsMA_feat"] = panel["VIX_vs_MA20"]
    panel["VOL_stockvol_feat"] = panel["Volatility_20D"]

    quarters = pd.period_range(panel["Date"].min(), panel["Date"].max(), freq="Q")
    # need at least 2 quarters of history before first prediction quarter
    first_pred_q_idx = 8  # start predicting after 2 years of expanding training data

    daily_rows = []
    perticker_rows = []
    recent_err_tech: list[float] = []
    recent_err_vol: list[float] = []

    for qi in range(first_pred_q_idx, len(quarters)):
        train_end = quarters[qi - 1].end_time
        test_q = quarters[qi]
        test_start, test_end = test_q.start_time, test_q.end_time

        train = panel[panel["Date"] <= train_end]
        test = panel[(panel["Date"] > train_end) & (panel["Date"] <= test_end)]
        if train.empty or test.empty:
            continue

        # --- Technical expert (Ridge on 19 real features, matches technical_expert.py list) ---
        Xtr_t = train[TECH_FEATURE_COLS].values
        ytr_t = train["Target"].values
        sc_t = StandardScaler().fit(Xtr_t)
        model_t = Ridge(alpha=5.0).fit(sc_t.transform(Xtr_t), ytr_t)

        # --- Volatility expert (Ridge on 3 real features, matches volatility_expert.py) ---
        Xtr_v = train[["VOL_Close_feat", "VOL_vsMA_feat", "VOL_stockvol_feat"]].values
        ytr_v = train["Target"].values
        sc_v = StandardScaler().fit(Xtr_v)
        model_v = Ridge(alpha=5.0).fit(sc_v.transform(Xtr_v), ytr_v)

        for date, day_df in test.groupby("Date"):
            Xte_t = sc_t.transform(day_df[TECH_FEATURE_COLS].values)
            Xte_v = sc_v.transform(day_df[["VOL_Close_feat", "VOL_vsMA_feat", "VOL_stockvol_feat"]].values)
            pred_t = model_t.predict(Xte_t)
            pred_v = model_v.predict(Xte_v)
            true = day_df["Target"].values

            if date >= pd.Timestamp("2025-01-01"):
                for tkr_, pt_, pv_, tr_ in zip(day_df["Ticker"].values, pred_t, pred_v, true):
                    perticker_rows.append({
                        "date": str(date.date()), "ticker": tkr_,
                        "pred_tech": float(pt_), "pred_vol": float(pv_),
                        "true_return": float(tr_),
                    })

            sigma2_t = np.mean(recent_err_tech) if recent_err_tech else 1.0
            sigma2_v = np.mean(recent_err_vol) if recent_err_vol else 1.0

            # Raw formula exactly as in fusion_framework.py:43 -- w_i = exp(-sigma_i^2)/sum
            wt_exp = np.exp(-sigma2_t)
            wv_exp = np.exp(-sigma2_v)
            w_t = wt_exp / (wt_exp + wv_exp)
            w_v = wv_exp / (wt_exp + wv_exp)

            # Scale-normalized variant: tau_t = mean of the two current uncertainty
            # estimates, so the softmax exponent is O(1) regardless of the absolute
            # error scale of the forecasting task (return-scale MSE ~1e-4).
            tau = max((sigma2_t + sigma2_v) / 2.0, 1e-12)
            wt_exp_s = np.exp(-sigma2_t / tau)
            wv_exp_s = np.exp(-sigma2_v / tau)
            w_t_s = wt_exp_s / (wt_exp_s + wv_exp_s)
            w_v_s = wv_exp_s / (wt_exp_s + wv_exp_s)

            fused_dyn = w_t * pred_t + w_v * pred_v
            fused_dyn_scaled = w_t_s * pred_t + w_v_s * pred_v
            fused_static = 0.5 * pred_t + 0.5 * pred_v

            day_mse_t = float(np.mean((true - pred_t) ** 2))
            day_mse_v = float(np.mean((true - pred_v) ** 2))

            daily_rows.append({
                "date": str(date.date()),
                "n_tickers": len(day_df),
                "w_tech": w_t, "w_vol": w_v,
                "w_tech_scaled": w_t_s, "w_vol_scaled": w_v_s,
                "sigma2_tech_trailing": sigma2_t, "sigma2_vol_trailing": sigma2_v,
                "mse_tech": day_mse_t, "mse_vol": day_mse_v,
                "acc_tech": float(np.mean(np.sign(pred_t) == np.sign(true))),
                "acc_vol": float(np.mean(np.sign(pred_v) == np.sign(true))),
                "acc_dynamic": float(np.mean(np.sign(fused_dyn) == np.sign(true))),
                "acc_dynamic_scaled": float(np.mean(np.sign(fused_dyn_scaled) == np.sign(true))),
                "acc_static": float(np.mean(np.sign(fused_static) == np.sign(true))),
                "mse_dynamic": float(np.mean((true - fused_dyn) ** 2)),
                "mse_dynamic_scaled": float(np.mean((true - fused_dyn_scaled) ** 2)),
                "mse_static": float(np.mean((true - fused_static) ** 2)),
            })

            recent_err_tech.append(day_mse_t)
            recent_err_vol.append(day_mse_v)
            if len(recent_err_tech) > MAX_ERROR_WINDOW:
                recent_err_tech.pop(0)
            if len(recent_err_vol) > MAX_ERROR_WINDOW:
                recent_err_vol.pop(0)

        print(f"  quarter {test_q} done: train_rows={len(train)}, test_days={test['Date'].nunique()}")

    daily = pd.DataFrame(daily_rows)
    daily.to_csv(OUT / "study1_daily.csv", index=False)
    pd.DataFrame(perticker_rows).to_csv(OUT / "study1_perticker_2025on.csv", index=False)

    daily["year"] = pd.to_datetime(daily["date"]).dt.year
    per_year = daily.groupby("year")[
        ["acc_tech", "acc_vol", "acc_dynamic", "acc_dynamic_scaled", "acc_static"]
    ].mean()

    # Paired significance test (Wilcoxon signed-rank on daily accuracy series --
    # each row is one trading day's cross-sectional accuracy, n_days independent
    # trading-day observations) for scaled-dynamic vs static fusion.
    from scipy import stats as sstats
    diff = (daily["acc_dynamic_scaled"] - daily["acc_static"]).values
    if np.any(diff != 0):
        wilcoxon_stat, wilcoxon_p = sstats.wilcoxon(diff)
    else:
        wilcoxon_stat, wilcoxon_p = 0.0, 1.0
    ttest_stat, ttest_p = sstats.ttest_rel(daily["acc_dynamic_scaled"], daily["acc_static"])

    # Bootstrap 95% CI on the mean accuracy difference (2000 resamples over days)
    rng = np.random.default_rng(42)
    boot = [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(2000)]
    ci_low, ci_high = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    summary = {
        "n_days": int(len(daily)),
        "date_range": [daily["date"].min(), daily["date"].max()],
        "n_tickers": int(panel["Ticker"].nunique()),
        "overall_acc_technical": float(daily["acc_tech"].mean()),
        "overall_acc_volatility": float(daily["acc_vol"].mean()),
        "overall_acc_dynamic_fusion_raw": float(daily["acc_dynamic"].mean()),
        "overall_acc_dynamic_fusion_scaled": float(daily["acc_dynamic_scaled"].mean()),
        "overall_acc_static_fusion": float(daily["acc_static"].mean()),
        "overall_mse_technical": float(daily["mse_tech"].mean()),
        "overall_mse_volatility": float(daily["mse_vol"].mean()),
        "overall_mse_dynamic_fusion_raw": float(daily["mse_dynamic"].mean()),
        "overall_mse_dynamic_fusion_scaled": float(daily["mse_dynamic_scaled"].mean()),
        "overall_mse_static_fusion": float(daily["mse_static"].mean()),
        "mean_w_tech_raw": float(daily["w_tech"].mean()),
        "std_w_tech_raw": float(daily["w_tech"].std()),
        "mean_w_tech_scaled": float(daily["w_tech_scaled"].mean()),
        "std_w_tech_scaled": float(daily["w_tech_scaled"].std()),
        "min_w_tech_scaled": float(daily["w_tech_scaled"].min()),
        "max_w_tech_scaled": float(daily["w_tech_scaled"].max()),
        "scaled_vs_static_acc_diff_mean": float(diff.mean()),
        "scaled_vs_static_acc_diff_ci95": [ci_low, ci_high],
        "wilcoxon_stat": float(wilcoxon_stat), "wilcoxon_p": float(wilcoxon_p),
        "paired_ttest_stat": float(ttest_stat), "paired_ttest_p": float(ttest_p),
        "per_year_accuracy": per_year.to_dict(orient="index"),
    }
    with open(OUT / "study1_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
