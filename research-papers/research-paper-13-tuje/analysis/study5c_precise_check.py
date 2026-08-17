"""Rerun the two most important lag-fixed configs and save daily-level data
so we can compute both the standard 95% CI and a Bonferroni-adjusted CI
(m=7, matching the paper's full multiple-comparisons count) precisely."""

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
CACHE = RES / "study5b_panel_cache_lagfixed.pkl"
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
    "all6": ["USDINR_1d_chg", "USDINR_5d_chg", "Crude_1d_chg", "Crude_5d_chg",
             "Gold_1d_chg", "Gold_5d_chg", "US10Y_1d_chg", "US10Y_5d_chg",
             "SP500_1d_chg", "SP500_5d_chg", "USVIX_1d_chg", "USVIX_5d_chg"],
}


def run(panel_raw, macro_cols, label):
    experts = {"tech": TECH_FEATURE_COLS, "vol": VOL_FEATURE_COLS, "macro": macro_cols}
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
            for n in preds:
                recent_err[n].append(float(np.mean((true - preds[n]) ** 2)))
                if len(recent_err[n]) > MAX_ERROR_WINDOW:
                    recent_err[n].pop(0)
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

    df = pd.DataFrame(daily_rows)
    df["correct_dynamic"] = (np.sign(df["fused_dynamic"]) == np.sign(df["true"])).astype(int)
    df["correct_static"] = (np.sign(df["fused_static"]) == np.sign(df["true"])).astype(int)
    df.to_csv(RES / f"study5c_daily_{label}.csv", index=False)

    diff_daily = df.groupby("date")["correct_dynamic"].mean() - df.groupby("date")["correct_static"].mean()
    wstat, wp = sstats.wilcoxon(diff_daily)
    tstat, tp = sstats.ttest_rel(df.groupby("date")["correct_dynamic"].mean(), df.groupby("date")["correct_static"].mean())

    def ci(level, n_boot=5000, seed=42):
        vals = diff_daily.values
        rng = np.random.default_rng(seed)
        boots = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)]
        lo, hi = (1 - level) / 2 * 100, (1 - (1 - level) / 2) * 100
        return float(np.percentile(boots, lo)), float(np.percentile(boots, hi))

    m = 7
    lvl_bonf = 1 - 0.05 / m
    ci95 = ci(0.95)
    ci_bonf = ci(lvl_bonf)

    result = {
        "label": label, "n": len(df),
        "acc_static": float(df["correct_static"].mean()), "acc_dynamic": float(df["correct_dynamic"].mean()),
        "mean_diff_pp": float(diff_daily.mean() * 100),
        "ci95_pp": [ci95[0] * 100, ci95[1] * 100],
        "wilcoxon_p": float(wp), "paired_t_p": float(tp),
        f"bonferroni_m{m}_level_pct": lvl_bonf * 100,
        "ci_bonferroni_pp": [ci_bonf[0] * 100, ci_bonf[1] * 100],
        "excludes_zero_95": bool(ci95[0] > 0),
        "excludes_zero_bonferroni": bool(ci_bonf[0] > 0),
    }
    print(json.dumps(result, indent=2))
    return result


def main():
    panel_raw = pd.read_pickle(CACHE)
    all_r = []
    for label, cols in MACRO_CONFIGS.items():
        all_r.append(run(panel_raw, cols, label))
    with open(RES / "study5c_precise_results.json", "w") as f:
        json.dump(all_r, f, indent=2)


if __name__ == "__main__":
    main()
