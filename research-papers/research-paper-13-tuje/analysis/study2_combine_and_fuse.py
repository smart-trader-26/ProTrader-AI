"""
Study 2, step 2 -- combine real FinBERT sentiment (study2_headline_sentiment.json)
with the real technical/volatility expert predictions already produced by
Study 1 for 2025-01-01 onward (study1_perticker_2025on.csv), and run the
three-source dynamic-fusion mechanism end to end on genuine data.

IMPORTANT SCOPE NOTE (kept honest on purpose): this is a small-sample,
descriptive case study, not a second statistically-powered walk-forward
test like Study 1. yfinance's `.news` endpoint returns only each ticker's
~10 most recent stories, so real sentiment coverage here is sparse (most
ticker-days have zero headlines) and concentrated in a recent window. We
report what these headlines' real sentiment scores actually correlate
with, and show the fusion mechanism's real, computed behaviour on days
where a genuine sentiment signal exists -- we do not claim OOS predictive
significance from ~100 events.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

HERE = Path(__file__).parent
OUT = HERE / "results"

MAX_ERROR_WINDOW = 10


def to_effective_date(pub_date_utc: str, trading_dates: set) -> str | None:
    ts = pd.Timestamp(pub_date_utc)
    ist = ts.tz_convert("Asia/Kolkata") if ts.tzinfo else ts.tz_localize("UTC").tz_convert("Asia/Kolkata")
    d = ist.normalize().tz_localize(None)
    after_close = ist.hour >= 15 and ist.minute >= 30
    if after_close:
        d = d + pd.Timedelta(days=1)
    # roll forward to the next real trading date present in this ticker's calendar
    sorted_days = sorted(trading_dates)
    for td in sorted_days:
        if pd.Timestamp(td) >= d:
            return td
    return None


def sentiment_features_legacy(signed_values: list) -> list:
    """Exact port of sentiment_expert.py's legacy-list branch (lines 79-89)."""
    if signed_values:
        return [
            float(np.mean(signed_values)),
            float(np.std(signed_values)) if len(signed_values) > 1 else 0.0,
            len([v for v in signed_values if v > 0]) / len(signed_values),
            len([v for v in signed_values if v < 0]) / len(signed_values),
            float(np.max(signed_values)),
            0.0, 0.0, 0.0,
        ]
    return [0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]


def main():
    with open(OUT / "study2_headline_sentiment.json", encoding="utf-8") as f:
        headlines = json.load(f)
    perticker = pd.read_csv(OUT / "study1_perticker_2025on.csv", parse_dates=["date"])

    print(f"Real headlines loaded: {len(headlines)}")
    print(f"Per-ticker technical/volatility predictions loaded: {len(perticker)} rows, "
          f"{perticker['ticker'].nunique()} tickers, {perticker['date'].min().date()}..{perticker['date'].max().date()}")

    cal_by_ticker = {t: set(perticker.loc[perticker.ticker == t, "date"]) for t in perticker["ticker"].unique()}

    rows = []
    for h in headlines:
        tkr = h["ticker"]
        cal = cal_by_ticker.get(tkr)
        if not cal:
            continue
        eff = to_effective_date(h["pub_date_utc"], cal)
        if eff is None:
            continue
        rows.append({**h, "effective_date": eff})
    hdf = pd.DataFrame(rows)
    hdf["effective_date"] = pd.to_datetime(hdf["effective_date"])
    print(f"Headlines successfully mapped onto a real trading date in the study window: {len(hdf)}")

    agg = []
    for (tkr, dt), grp in hdf.groupby(["ticker", "effective_date"]):
        feats = sentiment_features_legacy(grp["signed_sentiment"].tolist())
        agg.append({
            "ticker": tkr, "date": dt, "n_headlines": len(grp),
            "sent_mean": feats[0], "sent_std": feats[1],
            "sent_frac_pos": feats[2], "sent_frac_neg": feats[3], "sent_max": feats[4],
        })
    sent_daily = pd.DataFrame(agg)
    print(f"Real (ticker, trading-day) events with at least one headline: {len(sent_daily)}")

    merged = sent_daily.merge(perticker, on=["ticker", "date"], how="left").dropna(subset=["true_return"])
    print(f"Merged with real technical/volatility predictions + realized returns: {len(merged)} events")

    # Descriptive correlation: real signed sentiment vs real realized next-day return.
    r, p = sstats.pearsonr(merged["sent_mean"], merged["true_return"])
    print(f"Pearson r(sentiment_mean, next-day realized return) = {r:.4f}, p = {p:.4f}, n = {len(merged)}")

    # In-sample OLS calibration of a linear sentiment->return mapping (disclosed as
    # in-sample / descriptive -- the sample is too small for a held-out split).
    slope, intercept, r2, p2, se = sstats.linregress(merged["sent_mean"], merged["true_return"])
    merged["pred_sent"] = intercept + slope * merged["sent_mean"]
    print(f"OLS (in-sample): return = {intercept:.6f} + {slope:.6f} * sentiment_mean, "
          f"R = {r2:.4f}, p = {p2:.4f}")

    # Run the real three-source softmax fusion mechanism, scale-normalized variant,
    # sequentially over these real event rows ordered by date.
    merged = merged.sort_values("date").reset_index(drop=True)
    recent_err = {"tech": [], "sent": [], "vol": []}
    weight_rows = []
    for _, row in merged.iterrows():
        preds = {"tech": row["pred_tech"], "sent": row["pred_sent"], "vol": row["pred_vol"]}
        true = row["true_return"]
        sigma2 = {k: (np.mean(v) if v else 1.0) for k, v in recent_err.items()}
        tau = max(np.mean(list(sigma2.values())), 1e-12)
        exp_w = {k: np.exp(-sigma2[k] / tau) for k in sigma2}
        total = sum(exp_w.values())
        w = {k: exp_w[k] / total for k in exp_w}
        fused = sum(w[k] * preds[k] for k in preds)
        fused_static = np.mean(list(preds.values()))

        weight_rows.append({
            "date": row["date"], "ticker": row["ticker"], "n_headlines": row["n_headlines"],
            "sent_mean": row["sent_mean"],
            "pred_tech": preds["tech"], "pred_sent": preds["sent"], "pred_vol": preds["vol"],
            "w_tech": w["tech"], "w_sent": w["sent"], "w_vol": w["vol"],
            "fused_dynamic": fused, "fused_static": fused_static, "true_return": true,
            "correct_dynamic": int(np.sign(fused) == np.sign(true)),
            "correct_static": int(np.sign(fused_static) == np.sign(true)),
        })
        for k in preds:
            recent_err[k].append(float((true - preds[k]) ** 2))
            if len(recent_err[k]) > MAX_ERROR_WINDOW:
                recent_err[k].pop(0)

    wdf = pd.DataFrame(weight_rows)
    wdf.to_csv(OUT / "study2_fused_events.csv", index=False)

    summary = {
        "n_headlines_real": len(headlines),
        "n_headlines_mapped_to_trading_day": len(hdf),
        "n_ticker_day_events": len(sent_daily),
        "n_events_with_predictions": len(merged),
        "pearson_r_sentiment_vs_return": float(r),
        "pearson_p": float(p),
        "ols_slope": float(slope), "ols_intercept": float(intercept),
        "ols_r2": float(r2), "ols_p": float(p2),
        "mean_w_tech": float(wdf["w_tech"].mean()), "std_w_tech": float(wdf["w_tech"].std()),
        "mean_w_sent": float(wdf["w_sent"].mean()), "std_w_sent": float(wdf["w_sent"].std()),
        "mean_w_vol": float(wdf["w_vol"].mean()), "std_w_vol": float(wdf["w_vol"].std()),
        "acc_dynamic": float(wdf["correct_dynamic"].mean()),
        "acc_static": float(wdf["correct_static"].mean()),
        "date_range": [str(wdf["date"].min().date()), str(wdf["date"].max().date())],
    }
    with open(OUT / "study2_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
