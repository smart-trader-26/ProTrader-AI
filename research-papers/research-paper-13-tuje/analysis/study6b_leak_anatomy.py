"""
Study 6b -- complete the leak-anatomy 2x2 for the paper's decomposition table.

The published Study 4 (H=1, tech+vol+macro[USDINR,Crude]) contained two
independent look-ahead defects:

  Bug T (timezone): same-calendar-date macro join feeds the macro expert a
        US/London close that settles only in the early hours of India's next
        session (fixed by macro_feat.shift(1) -- study5b).
  Bug O (ordering): today's own realized error is appended to the trailing
        window BEFORE today's fusion weights are computed (fixed by
        computing weights first -- study5d).

Cells already computed elsewhere:
  (T present, O present)  = published Study 4      -> verified here from its
                            own raw daily CSV (p-values recomputed).
  (T fixed,   O present)  = study5c "usdinr_crude" -> +0.040pp.
  (T fixed,   O fixed)    = study5d               -> +0.012pp, null.

This script adds the missing cell:
  (T present, O fixed)    = run on the ORIGINAL (non-lag-fixed) panel cache
                            with the corrected weight-before-error ordering.

All four cells use the identical walk-forward, window, fusion math, daily
aggregation, and test battery, so differences are attributable to the two
bugs alone. These four numbers are diagnostics of broken protocols, not
findings; only the (T fixed, O fixed) cell is a result of the paper.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
RES = HERE / "results"
MAX_ERROR_WINDOW = 10

TECH_FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "MA5", "MA20", "MA50", "MA_Ratio_5_20",
    "Volatility_5D", "Volatility_20D", "ATR",
    "Volume_Ratio", "RSI", "MACD", "MACD_Histogram",
    "Price_vs_MA20", "Price_vs_MA50", "Gap",
]
VOL_FEATURE_COLS = ["VIX_Close", "VIX_vs_MA20", "Volatility_20D"]
MACRO_COLS = ["USDINR_1d_chg", "USDINR_5d_chg", "Crude_1d_chg", "Crude_5d_chg"]


def stats_from_daily_diff(diff):
    vals = diff.values
    rng = np.random.default_rng(42)
    boots = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(5000)]
    ci95 = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    wp = float(sstats.wilcoxon(vals)[1]) if np.abs(vals).sum() > 0 else 1.0
    tp = float(sstats.ttest_1samp(vals, 0.0)[1])
    return {"mean_diff_pp": float(vals.mean() * 100),
            "ci95_pp": [ci95[0] * 100, ci95[1] * 100],
            "wilcoxon_p": wp, "paired_t_p": tp, "n_days": int(len(vals))}


def verify_published_cell():
    """Recompute the (T present, O present) cell's stats from the raw daily
    CSV the published numbers came from -- independent verification."""
    df = pd.read_csv(RES / "study4_daily_h1.csv")
    df["correct_dynamic"] = (np.sign(df["fused_dynamic"]) == np.sign(df["true"])).astype(int)
    df["correct_static"] = (np.sign(df["fused_static"]) == np.sign(df["true"])).astype(int)
    diff = df.groupby("date")["correct_dynamic"].mean() - df.groupby("date")["correct_static"].mean()
    out = stats_from_daily_diff(diff)
    out["n_rows"] = int(len(df))
    out["acc_static"] = float(df["correct_static"].mean())
    out["acc_dynamic"] = float(df["correct_dynamic"].mean())
    return out


def run_missing_cell():
    """(T present, O fixed): original same-date macro join, corrected ordering."""
    panel_raw = pd.read_pickle(RES / "study5_panel_cache.pkl")  # NON-lag-fixed
    experts = {"tech": TECH_FEATURE_COLS, "vol": VOL_FEATURE_COLS, "macro": MACRO_COLS}
    panel = panel_raw.sort_values(["Ticker", "Date"]).copy()
    panel["Target"] = panel.groupby("Ticker")["Close"].transform(lambda s: s.pct_change().shift(-1))
    needed = sorted(set(c for cols in experts.values() for c in cols))
    panel = panel.dropna(subset=needed + ["Target"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    quarters = pd.period_range(panel["Date"].min(), panel["Date"].max(), freq="Q")
    rows = []
    recent_err = {n: [] for n in experts}
    for qi in range(8, len(quarters)):
        train_end = quarters[qi - 1].end_time
        test_q = quarters[qi]
        train = panel[panel["Date"] <= train_end]
        test = panel[(panel["Date"] > train_end) & (panel["Date"] <= test_q.end_time)]
        if train.empty or test.empty:
            continue
        models = {}
        for name, cols in experts.items():
            sc = StandardScaler().fit(train[cols].values)
            m = Ridge(alpha=5.0).fit(sc.transform(train[cols].values), train["Target"].values)
            models[name] = (sc, m)
        for date, day_df in test.groupby("Date"):
            preds = {n: models[n][1].predict(models[n][0].transform(day_df[experts[n]].values)) for n in experts}
            true = day_df["Target"].values
            # corrected ordering: weights BEFORE today's error
            sigma2 = {n: (np.mean(recent_err[n]) if recent_err[n] else 1.0) for n in preds}
            tau = max(np.mean(list(sigma2.values())), 1e-12)
            ew = {n: np.exp(-sigma2[n] / tau) for n in sigma2}
            tot = sum(ew.values())
            w = {n: ew[n] / tot for n in ew}
            fd = sum(w[n] * preds[n] for n in preds)
            fs = np.mean(list(preds.values()), axis=0)
            for i in range(len(day_df)):
                rows.append({"date": str(date.date()), "fused_dynamic": float(fd[i]),
                             "fused_static": float(fs[i]), "true": float(true[i])})
            for n in preds:
                recent_err[n].append(float(np.mean((true - preds[n]) ** 2)))
                if len(recent_err[n]) > MAX_ERROR_WINDOW:
                    recent_err[n].pop(0)

    df = pd.DataFrame(rows)
    df["correct_dynamic"] = (np.sign(df["fused_dynamic"]) == np.sign(df["true"])).astype(int)
    df["correct_static"] = (np.sign(df["fused_static"]) == np.sign(df["true"])).astype(int)
    diff = df.groupby("date")["correct_dynamic"].mean() - df.groupby("date")["correct_static"].mean()
    out = stats_from_daily_diff(diff)
    out["n_rows"] = int(len(df))
    out["acc_static"] = float(df["correct_static"].mean())
    out["acc_dynamic"] = float(df["correct_dynamic"].mean())
    return out


def main():
    print("Verifying published (both-leaks) cell from raw CSV...")
    cell_tt = verify_published_cell()
    print(json.dumps(cell_tt, indent=2))
    print("\nRunning missing cell (timezone leak present, ordering fixed)...")
    cell_to = run_missing_cell()
    print(json.dumps(cell_to, indent=2))
    out = {"T_present_O_present_verified_from_csv": cell_tt,
           "T_present_O_fixed": cell_to}
    with open(RES / "study6b_leak_anatomy.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {RES / 'study6b_leak_anatomy.json'}")


if __name__ == "__main__":
    main()
