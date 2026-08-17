"""
Study 5d -- BOTH bugs fixed:

Bug 1 (found first): every macro ticker's "date t" close settles during
the US/London evening, landing in the early hours of India's date t+1 --
fixed in study5b by shifting all macro features forward by one row before
joining (already baked into study5b_panel_cache_lagfixed.pkl, reused here).

Bug 2 (found second, while double-checking on the user's prompt): in the
horizon==1 branch of study4_threesource.py (the script that produced the
PUBLISHED Study 4 numbers) and in study5/5b/5c's run_config, today's own
realized squared error is appended to `recent_err` BEFORE sigma2/weights
are computed for fusing TODAY's prediction -- i.e. today's fusion weight
depends on today's own prediction error, which requires already knowing
today's true outcome. Study 1's script (study1_technical_volatility.py)
does NOT have this bug -- it computes sigma2/weights first, and only
appends the realized error at the very end of that iteration, after the
fused prediction is built. This script matches Study 1's correct ordering
for horizon==1. The horizon>1 pending-queue path was checked and is NOT
affected (the queued error is deliberately released only `horizon` steps
later, and today's own error is pushed onto the queue only after fused_dyn
is computed) -- so the H=3/5/10/15 null results already reported stand.

Rerunning every H=1 configuration with BOTH fixes applied.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy import stats as sstats

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
RES = HERE / "results"
CACHE = RES / "study5b_panel_cache_lagfixed.pkl"  # already has the timezone fix baked in
MAX_ERROR_WINDOW = 10

TECH_FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "MA5", "MA20", "MA50", "MA_Ratio_5_20",
    "Volatility_5D", "Volatility_20D", "ATR",
    "Volume_Ratio", "RSI", "MACD", "MACD_Histogram",
    "Price_vs_MA20", "Price_vs_MA50", "Gap",
]
VOL_FEATURE_COLS = ["VIX_Close", "VIX_vs_MA20", "Volatility_20D"]

MACRO_CONFIGS = {
    "usdinr_crude": ["USDINR_1d_chg", "USDINR_5d_chg", "Crude_1d_chg", "Crude_5d_chg"],
    "gold": ["Gold_1d_chg", "Gold_5d_chg"],
    "us10y": ["US10Y_1d_chg", "US10Y_5d_chg"],
    "sp500": ["SP500_1d_chg", "SP500_5d_chg"],
    "usvix": ["USVIX_1d_chg", "USVIX_5d_chg"],
    "all6": ["USDINR_1d_chg", "USDINR_5d_chg", "Crude_1d_chg", "Crude_5d_chg",
             "Gold_1d_chg", "Gold_5d_chg", "US10Y_1d_chg", "US10Y_5d_chg",
             "SP500_1d_chg", "SP500_5d_chg", "USVIX_1d_chg", "USVIX_5d_chg"],
    "remaining4": ["Gold_1d_chg", "Gold_5d_chg", "US10Y_1d_chg", "US10Y_5d_chg",
                   "SP500_1d_chg", "SP500_5d_chg", "USVIX_1d_chg", "USVIX_5d_chg"],
}


def run(panel_raw, experts: dict, label: str, m_total=7):
    panel = panel_raw.sort_values(["Ticker", "Date"]).copy()
    panel["Target"] = panel.groupby("Ticker")["Close"].transform(lambda s: s.pct_change().shift(-1))
    needed = sorted(set(c for cols in experts.values() for c in cols))
    panel = panel.dropna(subset=needed + ["Target"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    quarters = pd.period_range(panel["Date"].min(), panel["Date"].max(), freq="Q")
    daily_rows = []
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

            # FIX 2: compute sigma2/weights from recent_err BEFORE touching it with today's error
            sigma2 = {n: (np.mean(recent_err[n]) if recent_err[n] else 1.0) for n in preds}
            tau = max(np.mean(list(sigma2.values())), 1e-12)
            exp_w = {n: np.exp(-sigma2[n] / tau) for n in sigma2}
            tot = sum(exp_w.values())
            w = {n: exp_w[n] / tot for n in exp_w}

            fused_dyn = sum(w[n] * preds[n] for n in preds)
            fused_static = np.mean(list(preds.values()), axis=0)

            for i in range(len(day_df)):
                daily_rows.append({"date": str(date.date()),
                                    "fused_dynamic": float(fused_dyn[i]), "fused_static": float(fused_static[i]),
                                    "true": float(true[i])})

            # today's own error is appended ONLY NOW, after it has been used for nothing
            # in this iteration -- it becomes available starting next iteration only.
            for n in preds:
                recent_err[n].append(float(np.mean((true - preds[n]) ** 2)))
                if len(recent_err[n]) > MAX_ERROR_WINDOW:
                    recent_err[n].pop(0)

    df = pd.DataFrame(daily_rows)
    df["correct_dynamic"] = (np.sign(df["fused_dynamic"]) == np.sign(df["true"])).astype(int)
    df["correct_static"] = (np.sign(df["fused_static"]) == np.sign(df["true"])).astype(int)

    diff_daily = df.groupby("date")["correct_dynamic"].mean() - df.groupby("date")["correct_static"].mean()
    wstat, wp = sstats.wilcoxon(diff_daily) if diff_daily.abs().sum() > 0 else (0.0, 1.0)
    tstat, tp = sstats.ttest_rel(df.groupby("date")["correct_dynamic"].mean(), df.groupby("date")["correct_static"].mean())

    def ci(level, n_boot=5000, seed=42):
        vals = diff_daily.values
        rng = np.random.default_rng(seed)
        boots = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)]
        lo, hi = (1 - level) / 2 * 100, (1 - (1 - level) / 2) * 100
        return float(np.percentile(boots, lo)), float(np.percentile(boots, hi))

    lvl_bonf = 1 - 0.05 / m_total
    ci95 = ci(0.95)
    ci_bonf = ci(lvl_bonf)

    result = {
        "label": label, "n": int(len(df)),
        "acc_static": float(df["correct_static"].mean()), "acc_dynamic": float(df["correct_dynamic"].mean()),
        "mean_diff_pp": float(diff_daily.mean() * 100),
        "ci95_pp": [ci95[0] * 100, ci95[1] * 100],
        "wilcoxon_p": float(wp), "paired_t_p": float(tp),
        "ci_bonferroni_pp": [ci_bonf[0] * 100, ci_bonf[1] * 100],
        "excludes_zero_95": bool(ci95[0] > 0 or ci95[1] < 0),
        "excludes_zero_bonferroni": bool(ci_bonf[0] > 0 or ci_bonf[1] < 0),
    }
    print(json.dumps(result, indent=2))
    return result


def main():
    panel_raw = pd.read_pickle(CACHE)
    print(f"Loaded lag-fixed cached panel: {len(panel_raw)} rows\n")
    all_r = []
    for macro_name, macro_cols in MACRO_CONFIGS.items():
        experts = {"tech": TECH_FEATURE_COLS, "vol": VOL_FEATURE_COLS, "macro": macro_cols}
        all_r.append(run(panel_raw, experts, f"H1_3src_{macro_name}"))

    experts4 = {
        "tech": TECH_FEATURE_COLS, "vol": VOL_FEATURE_COLS,
        "macro1": MACRO_CONFIGS["usdinr_crude"], "macro2": MACRO_CONFIGS["remaining4"],
    }
    all_r.append(run(panel_raw, experts4, "H1_4src_split_macro"))

    with open(RES / "study5d_doublefixed_results.json", "w") as f:
        json.dump(all_r, f, indent=2)
    print(f"\nSaved {len(all_r)} configs -> study5d_doublefixed_results.json")


if __name__ == "__main__":
    main()
