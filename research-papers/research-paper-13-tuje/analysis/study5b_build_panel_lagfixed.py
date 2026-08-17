"""
Study 5b -- corrected panel build with a real timezone-alignment bug fixed.

CRITICAL FIX vs study5_build_panel.py (and, it turns out, vs Study 4 as
published): every macro ticker used (INR=X, CL=F, GC=F, ^TNX, ^GSPC, ^VIX)
settles its "date t" daily bar during the US/London evening session, which
in IST lands in the early hours of calendar day t+1 -- verified directly:

    INR=X  index tz Europe/London,      CL=F/^GSPC index tz America/New_York

India's own trading day t closes at 15:30 IST, hours *before* the US session
for "date t" even opens (~19:00 IST). Joining macro_feat's "date t" row onto
the India panel's "date t" row (as study5_build_panel.py and the published
Study 4 both do) therefore uses information that does not exist yet at the
time a day-t prediction would be made -- a genuine look-ahead leak, not the
legitimate "yesterday's US close predicts today's Indian open" spillover
effect it was meant to capture.

Fix: shift every macro feature series forward by one row (`.shift(1)`)
before joining, so the value joined onto India's date-t row is the macro
ticker's OWN previous trading day's figure -- which settled hours before
India's date-t session even opens, under any reasonable timezone accounting.
This is the conservative, provably-non-leaking construction.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
CACHE = HERE / "results" / "study5b_panel_cache_lagfixed.pkl"
CACHE.parent.mkdir(exist_ok=True)

START = "2016-01-01"
END = None
VIX_TICKER = "^INDIAVIX"

# All real macro tickers from data/macro.py (MACRO_TICKERS), same convention.
ALL_MACRO_TICKERS = {
    "USDINR": "INR=X",
    "Crude": "CL=F",
    "US10Y": "^TNX",
    "Gold": "GC=F",
    "SP500": "^GSPC",
    "USVIX": "^VIX",
}

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


def build_technical(df):
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
    # Multi-horizon raw next-day return is computed later per-horizon by the
    # sweep script (Target_h1, Target_h5, ... columns added there); here we
    # only keep the un-shifted Close so any horizon can be derived downstream.
    return df


def fetch_close(ticker):
    raw = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)
    close = raw[("Close", ticker)] if isinstance(raw.columns, pd.MultiIndex) and (("Close", ticker) in raw.columns) else raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.rename(f"{ticker}_Close").dropna()


def main():
    print(f"Downloading {len(NIFTY50_TICKERS)} tickers + VIX + {len(ALL_MACRO_TICKERS)} macro series...")
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

    macro_closes = {}
    for label, sym in ALL_MACRO_TICKERS.items():
        print(f"  fetching {label} ({sym})...")
        macro_closes[label] = fetch_close(sym).rename(f"{label}_Close")
    macro_df = pd.concat(macro_closes.values(), axis=1)
    macro_feat = pd.DataFrame(index=macro_df.index)
    for label in ALL_MACRO_TICKERS:
        s = macro_df[f"{label}_Close"]
        macro_feat[f"{label}_1d_chg"] = np.log(s / s.shift(1)).clip(-0.2, 0.2)
        macro_feat[f"{label}_5d_chg"] = np.log(s / s.shift(5)).clip(-0.4, 0.4)
    macro_feat = macro_feat.fillna(0.0)
    # THE FIX: shift every macro feature forward by one row before it is joined
    # onto the India panel, so "date t" in the join carries the macro ticker's
    # OWN previous trading day's figure -- settled well before India's date-t
    # session even opens. Without this, "date t" carries a same-calendar-day
    # US/London close that doesn't exist until the early hours of date t+1 IST.
    macro_feat = macro_feat.shift(1)
    print(f"Macro panel (lag-corrected): {len(macro_feat)} rows, {macro_feat.index.min().date()}..{macro_feat.index.max().date()}")

    panels = []
    for tkr in NIFTY50_TICKERS:
        try:
            sub = raw[tkr].dropna(how="all")
        except (KeyError, TypeError):
            continue
        if sub.empty or len(sub) < 300:
            continue
        sub = build_technical(sub).join(vix_df, how="left").join(macro_feat, how="left")
        sub["Ticker"] = tkr
        panels.append(sub)
    panel = pd.concat(panels).reset_index().rename(columns={"index": "Date", "Datetime": "Date"})
    panel["Date"] = pd.to_datetime(panel["Date"])
    print(f"Raw panel (pre-target, pre-dropna): {len(panel)} rows, {panel['Ticker'].nunique()} tickers")

    panel.to_pickle(CACHE)
    print(f"Cached -> {CACHE}")


if __name__ == "__main__":
    main()
