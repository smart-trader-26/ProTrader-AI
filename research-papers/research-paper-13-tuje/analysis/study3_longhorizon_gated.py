"""
Study 3 -- does dynamic fusion earn its keep at a longer, less noisy horizon?

Study 1 tested next-day direction, where this project's own prior work
(project memory: calibration/accuracy findings) has repeatedly found no
exploitable single-name edge beyond drift -- so it is not a fair arena to
expect a *weighting* mechanism to show its value in, regardless of whether
that mechanism is static or dynamic. This script re-runs the identical two
real sources (technical, volatility), identical real data, identical
softmax-fusion mathematics, and identical quarterly walk-forward as Study 1,
but changes two things that this project's own established results (the
30-day conviction-gated swing signal, see README.md / directional_signal.py)
say matter: (1) the prediction horizon is lengthened from 1 day to 20
trading days, and (2) a conviction gate retains only the highest-magnitude
20% of combined predictions, mirroring the "abstain most of the time, fire
on the high-conviction subset" design already validated elsewhere in this
project.

Leakage control: a prediction made on day t for the return realized over
[t, t+20] is not "resolved" (its error is not knowable) until day t+20. The
uncertainty tracker therefore only ingests an error 20 trading days after
the corresponding prediction was made -- implemented with a 20-day pending
queue below -- exactly mirroring the latency a live system would face.

Significance testing accounts for the fact that 20-day-forward targets
overlap on consecutive days (autocorrelated errors): we use a block
bootstrap with block length 20 trading days, not a naive iid bootstrap.

Everything here is computed from the same real OHLCV / India VIX data as
Study 1. No numbers are invented; whatever this script prints is what goes
in the paper, in both directions.
"""

import json
import warnings
from collections import deque
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
END = None
HORIZON = 20                # trading days ahead
MAX_ERROR_WINDOW = 10        # config/settings.py:160, unchanged from Study 1
VIX_TICKER = "^INDIAVIX"
GATE_TOP_PCT = 0.20          # top 20% |combined prediction| = "high conviction"

NIFTY50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "BAJFINANCE.NS", "WIPRO.NS", "ASIANPAINT.NS", "MARUTI.NS", "HCLTECH.NS",
    "AXISBANK.NS", "LT.NS", "SUNPHARMA.NS", "TITAN.NS", "ONGC.NS",
    "ULTRACEMCO.NS", "POWERGRID.NS", "NTPC.NS", "JSWSTEEL.NS",
    "TATASTEEL.NS", "ADANIPORTS.NS", "COALINDIA.NS", "BAJAJFINSV.NS", "NESTLEIND.NS",
    "TECHM.NS", "GRASIM.NS", "DIVISLAB.NS", "CIPLA.NS", "DRREDDY.NS",
    "EICHERMOT.NS", "BPCL.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "APOLLOHOSP.NS",
    "TATACONSUM.NS", "UPL.NS", "BRITANNIA.NS", "SHREECEM.NS", "INDUSINDBK.NS",
]  # TATAMOTORS.NS excluded (delisted symbol, see Study 1)

TECH_FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "MA5", "MA20", "MA50", "MA_Ratio_5_20",
    "Volatility_5D", "Volatility_20D", "ATR",
    "Volume_Ratio", "RSI", "MACD", "MACD_Histogram",
    "Price_vs_MA20", "Price_vs_MA50", "Gap",
]


def rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    return macd_line, macd_line - macd_line.ewm(span=signal, adjust=False).mean()


def atr(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def build_features(df):
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
    df["Target"] = np.log(df["Close"].shift(-HORIZON) / df["Close"])  # 20d forward log return
    return df


def main():
    print(f"Downloading {len(NIFTY50_TICKERS)} tickers + VIX ({START}..today)...")
    raw = yf.download(NIFTY50_TICKERS, start=START, end=END, group_by="ticker",
                       auto_adjust=True, threads=True, progress=False)
    vix_raw = yf.download(VIX_TICKER, start=START, end=END, auto_adjust=True, progress=False)
    vix_close = (vix_raw[("Close", VIX_TICKER)] if isinstance(vix_raw.columns, pd.MultiIndex)
                 and ("Close", VIX_TICKER) in vix_raw.columns else vix_raw["Close"])
    if isinstance(vix_close, pd.DataFrame):
        vix_close = vix_close.iloc[:, 0]
    vix_close = vix_close.rename("VIX_Close").dropna()
    vix_ma20 = vix_close.rolling(20, min_periods=20).mean()
    vix_df = pd.concat([vix_close, (vix_close / vix_ma20).rename("VIX_vs_MA20")], axis=1)

    panels = []
    for tkr in NIFTY50_TICKERS:
        try:
            sub = raw[tkr].dropna(how="all")
        except (KeyError, TypeError):
            continue
        if sub.empty or len(sub) < 300:
            continue
        sub = build_features(sub).join(vix_df, how="left")
        sub["Ticker"] = tkr
        panels.append(sub)
    panel = pd.concat(panels).reset_index().rename(columns={"index": "Date", "Datetime": "Date"})
    panel["Date"] = pd.to_datetime(panel["Date"])
    needed = TECH_FEATURE_COLS + ["VIX_Close", "VIX_vs_MA20", "Target", "Date", "Ticker"]
    panel = panel.dropna(subset=needed).sort_values(["Date", "Ticker"]).reset_index(drop=True)
    print(f"Panel: {len(panel)} ticker-day rows, {panel['Date'].min().date()}..{panel['Date'].max().date()}, "
          f"{panel['Ticker'].nunique()} tickers, target = {HORIZON}-day forward log return.")

    quarters = pd.period_range(panel["Date"].min(), panel["Date"].max(), freq="Q")
    first_pred_q_idx = 8

    daily_rows = []
    recent_err_tech, recent_err_vol = [], []
    pending = deque(maxlen=HORIZON)  # each entry: (day_mse_tech, day_mse_vol)

    for qi in range(first_pred_q_idx, len(quarters)):
        train_end = quarters[qi - 1].end_time
        test_q = quarters[qi]
        train = panel[panel["Date"] <= train_end]
        test = panel[(panel["Date"] > train_end) & (panel["Date"] <= test_q.end_time)]
        if train.empty or test.empty:
            continue

        Xtr_t = train[TECH_FEATURE_COLS].values
        sc_t = StandardScaler().fit(Xtr_t)
        model_t = Ridge(alpha=5.0).fit(sc_t.transform(Xtr_t), train["Target"].values)

        Xtr_v = train[["VIX_Close", "VIX_vs_MA20", "Volatility_20D"]].values
        sc_v = StandardScaler().fit(Xtr_v)
        model_v = Ridge(alpha=5.0).fit(sc_v.transform(Xtr_v), train["Target"].values)

        for date, day_df in test.groupby("Date"):
            pred_t = model_t.predict(sc_t.transform(day_df[TECH_FEATURE_COLS].values))
            pred_v = model_v.predict(sc_v.transform(day_df[["VIX_Close", "VIX_vs_MA20", "Volatility_20D"]].values))
            true = day_df["Target"].values

            # release any errors that resolved by today (made HORIZON days ago)
            if len(pending) == HORIZON:
                old_mse_t, old_mse_v = pending[0]
                recent_err_tech.append(old_mse_t)
                recent_err_vol.append(old_mse_v)
                if len(recent_err_tech) > MAX_ERROR_WINDOW:
                    recent_err_tech.pop(0)
                if len(recent_err_vol) > MAX_ERROR_WINDOW:
                    recent_err_vol.pop(0)

            sigma2_t = np.mean(recent_err_tech) if recent_err_tech else 1.0
            sigma2_v = np.mean(recent_err_vol) if recent_err_vol else 1.0
            tau = max((sigma2_t + sigma2_v) / 2.0, 1e-12)
            wt_exp, wv_exp = np.exp(-sigma2_t / tau), np.exp(-sigma2_v / tau)
            w_t, w_v = wt_exp / (wt_exp + wv_exp), wv_exp / (wt_exp + wv_exp)

            fused_dyn = w_t * pred_t + w_v * pred_v
            fused_static = 0.5 * pred_t + 0.5 * pred_v

            for i in range(len(day_df)):
                daily_rows.append({
                    "date": str(date.date()), "ticker": day_df["Ticker"].values[i],
                    "pred_tech": float(pred_t[i]), "pred_vol": float(pred_v[i]),
                    "fused_dynamic": float(fused_dyn[i]), "fused_static": float(fused_static[i]),
                    "true": float(true[i]), "w_tech": float(w_t), "w_vol": float(w_v),
                })

            pending.append((float(np.mean((true - pred_t) ** 2)), float(np.mean((true - pred_v) ** 2))))

        print(f"  {test_q}: train_rows={len(train)}, test_rows={len(test)}")

    df = pd.DataFrame(daily_rows)
    df.to_csv(OUT / "study3_daily.csv", index=False)

    df["correct_tech"] = (np.sign(df["pred_tech"]) == np.sign(df["true"])).astype(int)
    df["correct_vol"] = (np.sign(df["pred_vol"]) == np.sign(df["true"])).astype(int)
    df["correct_dynamic"] = (np.sign(df["fused_dynamic"]) == np.sign(df["true"])).astype(int)
    df["correct_static"] = (np.sign(df["fused_static"]) == np.sign(df["true"])).astype(int)

    gate_thresh = df["fused_static"].abs().quantile(1 - GATE_TOP_PCT)
    gated = df[df["fused_static"].abs() >= gate_thresh].copy()

    def block_bootstrap_ci(sub, col_a, col_b, block=HORIZON, n_boot=2000, seed=42):
        sub = sub.sort_values("date")
        diff_by_date = (sub.groupby("date")[col_a].mean() - sub.groupby("date")[col_b].mean())
        vals = diff_by_date.values
        n = len(vals)
        n_blocks = int(np.ceil(n / block))
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, n_blocks, size=n_blocks)
            sample = np.concatenate([vals[i * block:(i + 1) * block] for i in idx])[:n]
            boots.append(sample.mean())
        return float(diff_by_date.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    mean_diff_full, ci_lo_full, ci_hi_full = block_bootstrap_ci(df, "correct_dynamic", "correct_static")
    mean_diff_gate, ci_lo_gate, ci_hi_gate = block_bootstrap_ci(gated, "correct_dynamic", "correct_static")

    summary = {
        "horizon_days": HORIZON,
        "n_rows_total": int(len(df)),
        "n_rows_gated": int(len(gated)),
        "gate_coverage_pct": float(len(gated) / len(df) * 100),
        "gate_threshold_abs_return": float(gate_thresh),
        "date_range": [df["date"].min(), df["date"].max()],
        "ungated": {
            "acc_tech": float(df["correct_tech"].mean()),
            "acc_vol": float(df["correct_vol"].mean()),
            "acc_static": float(df["correct_static"].mean()),
            "acc_dynamic": float(df["correct_dynamic"].mean()),
            "dynamic_minus_static_mean": mean_diff_full,
            "dynamic_minus_static_ci95": [ci_lo_full, ci_hi_full],
        },
        "gated_top20pct_conviction": {
            "acc_tech": float(gated["correct_tech"].mean()),
            "acc_vol": float(gated["correct_vol"].mean()),
            "acc_static": float(gated["correct_static"].mean()),
            "acc_dynamic": float(gated["correct_dynamic"].mean()),
            "dynamic_minus_static_mean": mean_diff_gate,
            "dynamic_minus_static_ci95": [ci_lo_gate, ci_hi_gate],
        },
        "mean_w_tech_overall": float(df["w_tech"].mean()),
        "std_w_tech_overall": float(df["w_tech"].std()),
    }
    with open(OUT / "study3_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
