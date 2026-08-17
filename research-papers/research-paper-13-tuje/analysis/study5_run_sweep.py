"""
Study 5, step 1 -- exploratory sweep over source composition and horizon,
using the cached real panel from study5_build_panel.py (no re-download).

This is NOT paper content -- it's an honest, broad search for whether any
combination of (more/different real sources) x (horizon) improves on
Study 4's result, run at the user's explicit request ("try all different
methods... whatever adds up to improving the results"). Every configuration
tried is reported below, not just the best-looking one, precisely to avoid
turning this into undisclosed multiple-comparisons fishing.

Each configuration reruns the same quarterly expanding-window walk-forward,
same MAX_ERROR_WINDOW=10, same scale-normalized softmax fusion, same 20-day-
style lagged pending queue for horizon>1, as Studies 1/3/4 -- only the set
of experts and the horizon change.
"""

import json
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
RES = HERE / "results"
CACHE = RES / "study5_panel_cache.pkl"

MAX_ERROR_WINDOW = 10
GATE_TOP_PCT = 0.20

TECH_FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "MA5", "MA20", "MA50", "MA_Ratio_5_20",
    "Volatility_5D", "Volatility_20D", "ATR",
    "Volume_Ratio", "RSI", "MACD", "MACD_Histogram",
    "Price_vs_MA20", "Price_vs_MA50", "Gap",
]
VOL_FEATURE_COLS = ["VIX_Close", "VIX_vs_MA20", "Volatility_20D"]

MACRO_SINGLE = {
    "macro_usdinr_crude": ["USDINR_1d_chg", "USDINR_5d_chg", "Crude_1d_chg", "Crude_5d_chg"],
    "macro_gold": ["Gold_1d_chg", "Gold_5d_chg"],
    "macro_us10y": ["US10Y_1d_chg", "US10Y_5d_chg"],
    "macro_sp500": ["SP500_1d_chg", "SP500_5d_chg"],
    "macro_usvix": ["USVIX_1d_chg", "USVIX_5d_chg"],
    "macro_all6": ["USDINR_1d_chg", "USDINR_5d_chg", "Crude_1d_chg", "Crude_5d_chg",
                   "Gold_1d_chg", "Gold_5d_chg", "US10Y_1d_chg", "US10Y_5d_chg",
                   "SP500_1d_chg", "SP500_5d_chg", "USVIX_1d_chg", "USVIX_5d_chg"],
    "macro_remaining4": ["Gold_1d_chg", "Gold_5d_chg", "US10Y_1d_chg", "US10Y_5d_chg",
                         "SP500_1d_chg", "SP500_5d_chg", "USVIX_1d_chg", "USVIX_5d_chg"],
}


def run_config(panel_raw, horizon, experts: dict, label: str):
    """experts: {name: [feature_cols]}. Returns a summary dict."""
    panel = panel_raw.copy()
    needed_cols = sorted(set(c for cols in experts.values() for c in cols))
    # Build target per-ticker to avoid cross-ticker leakage at panel edges
    panel = panel.sort_values(["Ticker", "Date"])
    panel["Target"] = panel.groupby("Ticker")["Close"].transform(
        lambda s: np.log(s.shift(-horizon) / s) if horizon > 1 else s.pct_change().shift(-1)
    )
    panel = panel.dropna(subset=needed_cols + ["Target"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    quarters = pd.period_range(panel["Date"].min(), panel["Date"].max(), freq="Q")
    first_pred_q_idx = 8
    if first_pred_q_idx >= len(quarters):
        return {"label": label, "error": "insufficient history"}

    daily_rows = []
    recent_err = {name: [] for name in experts}
    pending = deque(maxlen=horizon) if horizon > 1 else None

    for qi in range(first_pred_q_idx, len(quarters)):
        train_end = quarters[qi - 1].end_time
        test_q = quarters[qi]
        train = panel[panel["Date"] <= train_end]
        test = panel[(panel["Date"] > train_end) & (panel["Date"] <= test_q.end_time)]
        if train.empty or test.empty:
            continue

        models = {}
        for name, cols in experts.items():
            Xtr = train[cols].values
            sc = StandardScaler().fit(Xtr)
            m = Ridge(alpha=5.0).fit(sc.transform(Xtr), train["Target"].values)
            models[name] = (sc, m)

        for date, day_df in test.groupby("Date"):
            preds = {}
            for name, cols in experts.items():
                sc, m = models[name]
                preds[name] = m.predict(sc.transform(day_df[cols].values))
            true = day_df["Target"].values

            if horizon == 1:
                for name in preds:
                    recent_err[name].append(float(np.mean((true - preds[name]) ** 2)))
                    if len(recent_err[name]) > MAX_ERROR_WINDOW:
                        recent_err[name].pop(0)
            else:
                if len(pending) == horizon:
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
                    "date": str(date.date()),
                    "fused_dynamic": float(fused_dyn[i]), "fused_static": float(fused_static[i]),
                    "true": float(true[i]),
                })

            if horizon > 1:
                pending.append({name: float(np.mean((true - preds[name]) ** 2)) for name in preds})

    df = pd.DataFrame(daily_rows)
    if df.empty:
        return {"label": label, "error": "no test rows"}
    df["correct_dynamic"] = (np.sign(df["fused_dynamic"]) == np.sign(df["true"])).astype(int)
    df["correct_static"] = (np.sign(df["fused_static"]) == np.sign(df["true"])).astype(int)

    def block_bootstrap(sub, block, n_boot=1500, seed=42):
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

    block = horizon if horizon > 1 else 1
    mean_diff, ci_lo, ci_hi = block_bootstrap(df, block)

    result = {
        "label": label, "horizon": horizon, "experts": list(experts.keys()),
        "n_rows": int(len(df)),
        "acc_static": float(df["correct_static"].mean()),
        "acc_dynamic": float(df["correct_dynamic"].mean()),
        "dynamic_minus_static_mean": mean_diff,
        "dynamic_minus_static_ci95": [ci_lo, ci_hi],
        "excludes_zero": bool(ci_lo > 0 or ci_hi < 0),
    }
    print(json.dumps(result, indent=2))
    return result


def main():
    panel_raw = pd.read_pickle(CACHE)
    print(f"Loaded cached panel: {len(panel_raw)} rows")

    all_results = []

    # --- Part A: macro-factor sweep at H=1 (which single/combined macro source helps?) ---
    for macro_name, macro_cols in MACRO_SINGLE.items():
        experts = {"tech": TECH_FEATURE_COLS, "vol": VOL_FEATURE_COLS, macro_name: macro_cols}
        r = run_config(panel_raw, horizon=1, experts=experts, label=f"H1_3src_{macro_name}")
        all_results.append(r)

    # --- Part B: horizon sweep at 3, 5, 10, 15 with the baseline macro pair (1 and 20 already known) ---
    for h in [3, 5, 10, 15]:
        experts = {"tech": TECH_FEATURE_COLS, "vol": VOL_FEATURE_COLS,
                   "macro_usdinr_crude": MACRO_SINGLE["macro_usdinr_crude"]}
        r = run_config(panel_raw, horizon=h, experts=experts, label=f"H{h}_3src_usdinr_crude")
        all_results.append(r)

    # --- Part C: 4-source (tech, vol, macro_usdinr_crude, macro_remaining4) at H=1 ---
    experts4 = {
        "tech": TECH_FEATURE_COLS, "vol": VOL_FEATURE_COLS,
        "macro_usdinr_crude": MACRO_SINGLE["macro_usdinr_crude"],
        "macro_remaining4": MACRO_SINGLE["macro_remaining4"],
    }
    r = run_config(panel_raw, horizon=1, experts=experts4, label="H1_4src_split_macro")
    all_results.append(r)

    with open(RES / "study5_sweep_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved {len(all_results)} configs -> {RES / 'study5_sweep_results.json'}")


if __name__ == "__main__":
    main()
