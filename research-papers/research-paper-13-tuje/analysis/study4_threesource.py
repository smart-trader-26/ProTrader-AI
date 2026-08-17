"""
Study 4 -- does adding a genuinely differentiated third source change the
dynamic-vs-static picture, at either horizon?

Studies 1 and 3 used only technical and volatility experts -- and Section 4.1
of the paper speculates that these two are a poor test of the mechanism
because they are "both derived from the same underlying price process."
This script adds a real third source that is NOT derived from the ticker's
own price series at all: a macro/FX/commodity expert built from USD/INR
(INR=X) and WTI crude (CL=F), using the exact real feature convention already
shipped in this project's own code, data/macro.py::extract_macro_features
(1-day and 5-day log returns, clipped to +-0.2/+-0.4) -- not a new source
invented for this paper. Both tickers have deep, genuine multi-year yfinance
history, unlike sentiment.

Run twice, via the HORIZON constant below: HORIZON=1 reproduces Study 1's
next-day setting with a third source added; HORIZON=20 reproduces Study 3's
longer, conviction-gated setting with a third source added. Same quarterly
expanding-window walk-forward, same MAX_ERROR_WINDOW=10, same softmax fusion
math (now normalized over three sources), same 20-day-lagged pending queue
for the H=20 case to prevent look-ahead. Whatever comes out goes in the
paper as-is.
"""

import json
import sys
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

HORIZON = int(sys.argv[1]) if len(sys.argv) > 1 else 1   # 1 or 20
GATE_TOP_PCT = 0.20
START = "2016-01-01"
END = None
MAX_ERROR_WINDOW = 10
VIX_TICKER = "^INDIAVIX"
MACRO_TICKERS = {"USDINR": "INR=X", "Crude": "CL=F"}  # data/macro.py convention

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
VOL_FEATURE_COLS = ["VIX_Close", "VIX_vs_MA20", "Volatility_20D"]
MACRO_FEATURE_COLS = ["USDINR_1d_chg", "USDINR_5d_chg", "Crude_1d_chg", "Crude_5d_chg"]


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
    if HORIZON == 1:
        df["Target"] = df["Returns"].shift(-1)
    else:
        df["Target"] = np.log(df["Close"].shift(-HORIZON) / df["Close"])
    return df


def fetch_close(ticker):
    raw = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)
    close = raw[("Close", ticker)] if isinstance(raw.columns, pd.MultiIndex) and (("Close", ticker) in raw.columns) else raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.rename(f"{ticker}_Close").dropna()


def main():
    print(f"=== Study 4, HORIZON={HORIZON} ===")
    print(f"Downloading {len(NIFTY50_TICKERS)} tickers + VIX + macro ({START}..today)...")
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

    # -- real macro data, same tickers + same log-return-clip convention as data/macro.py --
    macro_closes = {}
    for label, sym in MACRO_TICKERS.items():
        c = fetch_close(sym)
        macro_closes[label] = c.rename(f"{label}_Close")
    macro_df = pd.concat(macro_closes.values(), axis=1)
    macro_feat = pd.DataFrame(index=macro_df.index)
    for label in MACRO_TICKERS:
        s = macro_df[f"{label}_Close"]
        macro_feat[f"{label}_1d_chg"] = np.log(s / s.shift(1)).clip(-0.2, 0.2)
        macro_feat[f"{label}_5d_chg"] = np.log(s / s.shift(5)).clip(-0.4, 0.4)
    macro_feat = macro_feat.fillna(0.0)
    print(f"Macro panel: {len(macro_feat)} rows, {macro_feat.index.min().date()}..{macro_feat.index.max().date()}")

    panels = []
    for tkr in NIFTY50_TICKERS:
        try:
            sub = raw[tkr].dropna(how="all")
        except (KeyError, TypeError):
            continue
        if sub.empty or len(sub) < 300:
            continue
        sub = build_features(sub).join(vix_df, how="left").join(macro_feat, how="left")
        sub["Ticker"] = tkr
        panels.append(sub)
    panel = pd.concat(panels).reset_index().rename(columns={"index": "Date", "Datetime": "Date"})
    panel["Date"] = pd.to_datetime(panel["Date"])
    needed = TECH_FEATURE_COLS + VOL_FEATURE_COLS[:2] + MACRO_FEATURE_COLS + ["Volatility_20D", "Target", "Date", "Ticker"]
    panel = panel.dropna(subset=needed).sort_values(["Date", "Ticker"]).reset_index(drop=True)
    print(f"Panel: {len(panel)} ticker-day rows, {panel['Date'].min().date()}..{panel['Date'].max().date()}, "
          f"{panel['Ticker'].nunique()} tickers, horizon={HORIZON}d.")

    quarters = pd.period_range(panel["Date"].min(), panel["Date"].max(), freq="Q")
    first_pred_q_idx = 8

    daily_rows = []
    recent_err = {"tech": [], "vol": [], "macro": []}
    pending = deque(maxlen=HORIZON) if HORIZON > 1 else None

    for qi in range(first_pred_q_idx, len(quarters)):
        train_end = quarters[qi - 1].end_time
        test_q = quarters[qi]
        train = panel[panel["Date"] <= train_end]
        test = panel[(panel["Date"] > train_end) & (panel["Date"] <= test_q.end_time)]
        if train.empty or test.empty:
            continue

        models = {}
        for name, cols in [("tech", TECH_FEATURE_COLS),
                            ("vol", ["VIX_Close", "VIX_vs_MA20", "Volatility_20D"]),
                            ("macro", MACRO_FEATURE_COLS)]:
            Xtr = train[cols].values
            sc = StandardScaler().fit(Xtr)
            m = Ridge(alpha=5.0).fit(sc.transform(Xtr), train["Target"].values)
            models[name] = (sc, m)

        for date, day_df in test.groupby("Date"):
            preds = {}
            for name, cols in [("tech", TECH_FEATURE_COLS),
                                ("vol", ["VIX_Close", "VIX_vs_MA20", "Volatility_20D"]),
                                ("macro", MACRO_FEATURE_COLS)]:
                sc, m = models[name]
                preds[name] = m.predict(sc.transform(day_df[cols].values))
            true = day_df["Target"].values

            if HORIZON == 1:
                for name in preds:
                    recent_err[name].append(float(np.mean((true - preds[name]) ** 2)))
                    if len(recent_err[name]) > MAX_ERROR_WINDOW:
                        recent_err[name].pop(0)
            else:
                if len(pending) == HORIZON:
                    old = pending[0]
                    for name in preds:
                        recent_err[name].append(old[name])
                        if len(recent_err[name]) > MAX_ERROR_WINDOW:
                            recent_err[name].pop(0)

            sigma2 = {name: (np.mean(recent_err[name]) if recent_err[name] else 1.0) for name in preds}
            tau = max(np.mean(list(sigma2.values())), 1e-12)
            exp_w = {name: np.exp(-sigma2[name] / tau) for name in sigma2}
            total = sum(exp_w.values())
            w = {name: exp_w[name] / total for name in exp_w}

            fused_dyn = sum(w[name] * preds[name] for name in preds)
            fused_static = np.mean(list(preds.values()), axis=0)

            for i in range(len(day_df)):
                daily_rows.append({
                    "date": str(date.date()), "ticker": day_df["Ticker"].values[i],
                    "pred_tech": float(preds["tech"][i]), "pred_vol": float(preds["vol"][i]),
                    "pred_macro": float(preds["macro"][i]),
                    "fused_dynamic": float(fused_dyn[i]), "fused_static": float(fused_static[i]),
                    "true": float(true[i]),
                    "w_tech": float(w["tech"]), "w_vol": float(w["vol"]), "w_macro": float(w["macro"]),
                })

            if HORIZON > 1:
                pending.append({name: float(np.mean((true - preds[name]) ** 2)) for name in preds})

        print(f"  {test_q}: train_rows={len(train)}, test_rows={len(test)}")

    df = pd.DataFrame(daily_rows)
    suffix = f"_h{HORIZON}"
    df.to_csv(OUT / f"study4_daily{suffix}.csv", index=False)

    df["correct_tech"] = (np.sign(df["pred_tech"]) == np.sign(df["true"])).astype(int)
    df["correct_vol"] = (np.sign(df["pred_vol"]) == np.sign(df["true"])).astype(int)
    df["correct_macro"] = (np.sign(df["pred_macro"]) == np.sign(df["true"])).astype(int)
    df["correct_dynamic"] = (np.sign(df["fused_dynamic"]) == np.sign(df["true"])).astype(int)
    df["correct_static"] = (np.sign(df["fused_static"]) == np.sign(df["true"])).astype(int)

    def block_bootstrap(sub, block, n_boot=2000, seed=42):
        sub = sub.sort_values("date")
        diff_by_date = (sub.groupby("date")["correct_dynamic"].mean() - sub.groupby("date")["correct_static"].mean())
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

    block = HORIZON if HORIZON > 1 else 1
    mean_diff, ci_lo, ci_hi = block_bootstrap(df, block)

    summary = {
        "horizon_days": HORIZON,
        "n_rows_total": int(len(df)),
        "date_range": [df["date"].min(), df["date"].max()],
        "acc_tech": float(df["correct_tech"].mean()),
        "acc_vol": float(df["correct_vol"].mean()),
        "acc_macro": float(df["correct_macro"].mean()),
        "acc_static": float(df["correct_static"].mean()),
        "acc_dynamic": float(df["correct_dynamic"].mean()),
        "dynamic_minus_static_mean": mean_diff,
        "dynamic_minus_static_ci95": [ci_lo, ci_hi],
        "mean_w_tech": float(df["w_tech"].mean()), "std_w_tech": float(df["w_tech"].std()),
        "mean_w_vol": float(df["w_vol"].mean()), "std_w_vol": float(df["w_vol"].std()),
        "mean_w_macro": float(df["w_macro"].mean()), "std_w_macro": float(df["w_macro"].std()),
    }

    if HORIZON > 1:
        gate_thresh = df["fused_static"].abs().quantile(1 - GATE_TOP_PCT)
        gated = df[df["fused_static"].abs() >= gate_thresh].copy()
        mean_diff_g, ci_lo_g, ci_hi_g = block_bootstrap(gated, block)
        summary["gated_top20pct"] = {
            "n_rows": int(len(gated)),
            "acc_tech": float(gated["correct_tech"].mean()),
            "acc_vol": float(gated["correct_vol"].mean()),
            "acc_macro": float(gated["correct_macro"].mean()),
            "acc_static": float(gated["correct_static"].mean()),
            "acc_dynamic": float(gated["correct_dynamic"].mean()),
            "dynamic_minus_static_mean": mean_diff_g,
            "dynamic_minus_static_ci95": [ci_lo_g, ci_hi_g],
        }

    with open(OUT / f"study4_summary{suffix}.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
