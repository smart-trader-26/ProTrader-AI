"""Step 3: align headlines to trading sessions and build the daily panel.

Timing convention (strictly causal, no look-ahead):
  every headline carries a UTC publication stamp; it is converted to
  America/New_York and assigned to the session whose 16:00 ET close first
  follows it.  Sentiment of session d therefore contains only information
  public before that close, and is used to predict the return realised over
  session d+1 onwards.

Prices come from Yahoo Finance (split/dividend adjusted).  The market factor
is SPY; per-symbol betas are fitted on the training window only.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

TRAIN_END = "2016-12-31"


def load_prices(symbols: list[str], start: str, end: str,
                cache: str) -> dict[str, pd.DataFrame]:
    """Adjusted close plus the high/low needed for a volatility proxy."""
    if os.path.exists(cache):
        d = pd.read_parquet(cache)
        return {f: d[f] for f in d.columns.get_level_values(0).unique()}
    import yfinance as yf

    tick = sorted(set(symbols) | {"SPY"})
    raw = yf.download(tick, start=start, end=end, auto_adjust=True,
                      progress=False, threads=True)
    keep = [f for f in ("Close", "High", "Low") if f in
            raw.columns.get_level_values(0)]
    raw = raw[keep]
    raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
    raw = raw.dropna(axis=1, how="all")
    raw.to_parquet(cache)
    return {f: raw[f] for f in keep}


def assign_session(ts_utc: pd.Series, sessions: pd.DatetimeIndex) -> pd.Series:
    """Map each UTC stamp to the first session closing at/after it."""
    et = ts_utc.dt.tz_convert("America/New_York")
    # a headline is "before the close" if stamped strictly before 16:00 ET
    day = et.dt.normalize().dt.tz_localize(None)
    before_close = (et.dt.hour < 16)
    eff = day.where(before_close, day + pd.Timedelta(days=1))
    idx = sessions.searchsorted(eff.values, side="left")
    ok = idx < len(sessions)
    out = pd.Series(pd.NaT, index=ts_utc.index, dtype="datetime64[ns]")
    out.loc[ok] = sessions[idx[ok]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2024-01-01")
    args = ap.parse_args()

    df = pd.read_parquet(os.path.join(args.indir, "corpus_scored.parquet"))
    df["ts"] = pd.to_datetime(df["Date"], format="mixed", utc=True,
                              errors="coerce")
    df = df.dropna(subset=["ts"])
    df = df[(df["ts"] >= args.start) & (df["ts"] < args.end)]
    symbols = sorted(df["Stock_symbol"].unique().tolist())
    print(f"headlines in window: {len(df):,}  symbols: {len(symbols)}")

    fields = load_prices(symbols, args.start, args.end,
                         os.path.join(args.indir, "prices.parquet"))
    px = fields["Close"]
    hi_df, lo_df = fields.get("High"), fields.get("Low")
    sessions = pd.DatetimeIndex(px.index)
    symbols = [s for s in symbols if s in px.columns]
    print(f"symbols with prices: {len(symbols)}  sessions: {len(sessions)}")

    df["session"] = assign_session(df["ts"], sessions)
    df = df.dropna(subset=["session"])
    df = df[df["Stock_symbol"].isin(symbols)]

    agg = (df.groupby(["Stock_symbol", "session"])
             .agg(m1=("m1_finbert", "mean"), m2=("m2_lm", "mean"),
                  m3=("m3_vader", "mean"), m4=("m4_hiv4", "mean"),
                  n_news=("m1_finbert", "size"))
             .reset_index()
             .rename(columns={"Stock_symbol": "symbol"}))
    print(f"symbol-session cells with news: {len(agg):,}")

    # returns
    lr = np.log(px[symbols + ["SPY"]]).diff()
    mkt = lr["SPY"]
    train = lr.index <= TRAIN_END

    # Parkinson range volatility: an efficient, standard intraday proxy that
    # needs only the session high and low.
    if hi_df is not None and lo_df is not None:
        rv = (np.log(hi_df[symbols] / lo_df[symbols]) ** 2) / (4 * np.log(2))
        lrv = np.log(rv.clip(lower=1e-10))
    else:
        lrv = None

    rows = []
    for s in symbols:
        r = lr[s]
        good = r.notna() & mkt.notna()
        b = np.polyfit(mkt[good & train], r[good & train], 1)[0] if \
            (good & train).sum() > 250 else 1.0
        d = pd.DataFrame({"symbol": s, "session": lr.index,
                          "ret": r.values,
                          "ret_adj": (r - b * mkt).values,
                          "beta": b})
        if lrv is not None and s in lrv.columns:
            v = lrv[s]
            # volatility innovation: deviation from the trailing 21-session
            # mean, so the target is the news-driven surprise in volatility
            d["lrv"] = v.values
            d["lrv_innov"] = (v - v.rolling(21, min_periods=10)
                              .mean().shift(1)).values
        rows.append(d)
    pan = pd.concat(rows, ignore_index=True).dropna(subset=["ret_adj"])

    pan = pan.merge(agg, on=["symbol", "session"], how="left")
    pan["n_news"] = pan["n_news"].fillna(0)
    pan["has_news"] = (pan["n_news"] > 0).astype(np.int8)

    # standardise each sentiment measure per symbol on TRAIN statistics only
    for c in ("m1", "m2", "m3", "m4"):
        tr = pan[pan["session"] <= TRAIN_END]
        st = tr.groupby("symbol")[c].agg(["mean", "std"])
        pan = pan.merge(st.rename(columns={"mean": f"{c}_mu",
                                           "std": f"{c}_sd"}),
                        on="symbol", how="left")
        pan[f"{c}_z"] = (pan[c] - pan[f"{c}_mu"]) / pan[f"{c}_sd"]
        # a session with no headline carries no sentiment innovation
        pan[f"{c}_z"] = pan[f"{c}_z"].fillna(0.0).clip(-6, 6)
        pan = pan.drop(columns=[f"{c}_mu", f"{c}_sd"])

    pan = pan.sort_values(["symbol", "session"]).reset_index(drop=True)
    pan["ret_adj"] = pan["ret_adj"].clip(-0.25, 0.25)
    if "lrv_innov" in pan.columns:
        pan["lrv_innov"] = pan["lrv_innov"].clip(-5, 5)

    meta = {
        "n_headlines": int(len(df)),
        "n_unique": int(df["Article_title"].nunique()),
        "n_symbols_news": int(df["Stock_symbol"].nunique()),
        "n_symbols_panel": int(pan["symbol"].nunique()),
        "n_sessions": int(pan["session"].nunique()),
        "date_min": str(df["session"].min().date()),
        "date_max": str(df["session"].max().date()),
        "coverage": float((pan["n_news"] > 0).mean()),
        "median_news_per_covered_session": float(
            pan.loc[pan["n_news"] > 0, "n_news"].median()),
        "n_publishers": int(df["Publisher"].nunique())
        if "Publisher" in df.columns else None,
    }
    import json as _json
    with open(os.path.join(args.indir, "dataset.json"), "w") as fh:
        _json.dump(meta, fh, indent=1)
    print("dataset:", meta)
    out = os.path.join(args.indir, "panel.parquet")
    pan.to_parquet(out, index=False)
    print(f"wrote {out}  rows={len(pan):,}")
    print(pan.groupby(pan["session"].dt.year)["has_news"].agg(["size", "mean"]))


if __name__ == "__main__":
    main()
