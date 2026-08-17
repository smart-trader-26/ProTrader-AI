"""
Study 6 -- the two runs still missing after the study5b/5d leak corrections,
with the protocol fixed in advance of seeing any result.

Part A: H=20 three-source (tech, vol, macro USDINR+Crude), ungated and
top-20%-conviction gated, re-run on the LAG-FIXED macro panel
(study5b_panel_cache_lagfixed.pkl). The published Study 4 H=20 numbers came
from study4_threesource.py, whose macro join has the timezone leak (Bug 1);
its pending-queue error feedback was already correct (Bug 2 absent at H>1).
This run replaces those numbers with fully leak-free ones.

Part B: the tuned-learning-rate experiment the paper's Future Work section
already commits to. The scale-normalized softmax w_i ~ exp(-sigma_i^2/tau_t)
is a Hedge-style update with learning rate eta=1 after normalization; this
part asks whether ANY fixed eta = k/tau_t in a pre-specified grid beats
static fusion, using a disclosed validation/test split so the choice of k is
not fitted to the evaluation sample.

  PRE-SPECIFIED PROTOCOL (written before running):
  - Grid: k in {0.5, 1, 2, 5, 10, 20, 50, 100}. k=1 is the paper's tau rule;
    k->infinity approaches follow-the-leader; k=0 would be static fusion.
  - Configs: (i) 2-source tech+vol (Study 1 setting); (ii) 3-source
    tech+vol+macro(USDINR,Crude) (Study 4 setting). Both H=1, both on the
    lag-fixed panel, both with the corrected error-feedback ordering
    (weights computed BEFORE today's error is appended -- study5d ordering).
  - Selection: mean daily (dynamic - static) accuracy difference over
    prediction days in 2018-01-01..2019-12-31 picks k* per config.
  - Evaluation: k* frozen, reported once on 2020-01-01..end with bootstrap
    95% CI, Wilcoxon, paired t. Full grid saved to JSON for transparency;
    the paper reports the selection rule and the frozen k* test result.
  - The one-pass trick: per-day expert predictions and trailing sigma^2 do
    not depend on k (errors are per-expert, not per-fusion), so the walk-
    forward runs once per config and the k sweep is pure post-processing.
"""

import json
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
RES = HERE / "results"
CACHE = RES / "study5b_panel_cache_lagfixed.pkl"

MAX_ERROR_WINDOW = 10
GATE_TOP_PCT = 0.20
K_GRID = [0.5, 1, 2, 5, 10, 20, 50, 100]
SELECT_END = "2019-12-31"   # selection window: first prediction day .. this
                             # test window: everything after

TECH_FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "MA5", "MA20", "MA50", "MA_Ratio_5_20",
    "Volatility_5D", "Volatility_20D", "ATR",
    "Volume_Ratio", "RSI", "MACD", "MACD_Histogram",
    "Price_vs_MA20", "Price_vs_MA50", "Gap",
]
VOL_FEATURE_COLS = ["VIX_Close", "VIX_vs_MA20", "Volatility_20D"]
MACRO_COLS = ["USDINR_1d_chg", "USDINR_5d_chg", "Crude_1d_chg", "Crude_5d_chg"]


def prepare_panel(panel_raw, experts, horizon):
    panel = panel_raw.sort_values(["Ticker", "Date"]).copy()
    if horizon == 1:
        panel["Target"] = panel.groupby("Ticker")["Close"].transform(lambda s: s.pct_change().shift(-1))
    else:
        panel["Target"] = panel.groupby("Ticker")["Close"].transform(lambda s: np.log(s.shift(-horizon) / s))
    needed = sorted(set(c for cols in experts.values() for c in cols))
    return panel.dropna(subset=needed + ["Target"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)


def walk_forward(panel, experts, horizon):
    """One pass. Returns (rows_df, sigma_df). Weights are NOT computed here --
    only per-expert predictions and the per-day trailing sigma^2 each expert
    would report at prediction time (correct ordering: sigma^2 uses errors
    up to and including the previous resolvable day only)."""
    quarters = pd.period_range(panel["Date"].min(), panel["Date"].max(), freq="Q")
    rows, sig_rows = [], []
    recent_err = {n: [] for n in experts}
    pending = deque(maxlen=horizon) if horizon > 1 else None

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

            if horizon > 1 and len(pending) == horizon:
                old = pending[0]
                for n in preds:
                    recent_err[n].append(old[n])
                    if len(recent_err[n]) > MAX_ERROR_WINDOW:
                        recent_err[n].pop(0)

            # sigma^2 at prediction time -- BEFORE today's own error exists
            sigma2 = {n: (np.mean(recent_err[n]) if recent_err[n] else 1.0) for n in preds}
            sig_rows.append({"date": str(date.date()), **{f"s2_{n}": sigma2[n] for n in sigma2}})

            for i in range(len(day_df)):
                row = {"date": str(date.date()), "true": float(true[i])}
                for n in preds:
                    row[f"pred_{n}"] = float(preds[n][i])
                rows.append(row)

            day_err = {n: float(np.mean((true - preds[n]) ** 2)) for n in preds}
            if horizon == 1:
                for n in preds:
                    recent_err[n].append(day_err[n])
                    if len(recent_err[n]) > MAX_ERROR_WINDOW:
                        recent_err[n].pop(0)
            else:
                pending.append(day_err)

    return pd.DataFrame(rows), pd.DataFrame(sig_rows).set_index("date")


def fuse(rows, sigma, expert_names, k):
    """Vectorized fusion for one k. Returns rows with fused_dynamic/static."""
    s2 = sigma[[f"s2_{n}" for n in expert_names]].values  # (days, E)
    tau = np.maximum(s2.mean(axis=1), 1e-12)
    ew = np.exp(-k * s2 / tau[:, None])
    w = ew / ew.sum(axis=1, keepdims=True)                # (days, E)
    wdf = pd.DataFrame(w, index=sigma.index, columns=[f"w_{n}" for n in expert_names])
    out = rows.join(wdf, on="date")
    preds = out[[f"pred_{n}" for n in expert_names]].values
    wm = out[[f"w_{n}" for n in expert_names]].values
    out["fused_dynamic"] = (preds * wm).sum(axis=1)
    out["fused_static"] = preds.mean(axis=1)
    return out


def daily_diff(out):
    cd = (np.sign(out["fused_dynamic"]) == np.sign(out["true"])).astype(int)
    cs = (np.sign(out["fused_static"]) == np.sign(out["true"])).astype(int)
    g = pd.DataFrame({"date": out["date"], "d": cd, "s": cs}).groupby("date").mean()
    return g["d"] - g["s"], float(cd.mean()), float(cs.mean())


def test_stats(diff, block=1, n_boot=5000, seed=42, m_total=9):
    vals = diff.values
    rng = np.random.default_rng(seed)
    n = len(vals)
    if block == 1:
        boots = [rng.choice(vals, size=n, replace=True).mean() for _ in range(n_boot)]
    else:
        n_blocks = int(np.ceil(n / block))
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, n_blocks, size=n_blocks)
            boots.append(np.concatenate([vals[i * block:(i + 1) * block] for i in idx])[:n].mean())
    ci95 = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    lvl = (0.05 / m_total) / 2 * 100
    cib = (float(np.percentile(boots, lvl)), float(np.percentile(boots, 100 - lvl)))
    wp = float(sstats.wilcoxon(vals)[1]) if np.abs(vals).sum() > 0 else 1.0
    tp = float(sstats.ttest_1samp(vals, 0.0)[1])
    return {
        "mean_diff_pp": float(vals.mean() * 100),
        "ci95_pp": [ci95[0] * 100, ci95[1] * 100],
        "ci_bonferroni_m9_pp": [cib[0] * 100, cib[1] * 100],
        "wilcoxon_p": wp, "paired_t_p": tp,
        "excludes_zero_95": bool(ci95[0] > 0 or ci95[1] < 0),
        "excludes_zero_bonferroni": bool(cib[0] > 0 or cib[1] < 0),
        "n_days": int(n),
    }


def part_a(panel_raw):
    print("=== Part A: H=20 three-source, lag-fixed panel ===")
    experts = {"tech": TECH_FEATURE_COLS, "vol": VOL_FEATURE_COLS, "macro": MACRO_COLS}
    panel = prepare_panel(panel_raw, experts, 20)
    rows, sigma = walk_forward(panel, experts, 20)
    out = fuse(rows, sigma, list(experts), k=1)

    diff, acc_d, acc_s = daily_diff(out)
    res = {"label": "H20_3src_usdinr_crude_lagfixed", "n_rows": int(len(out)),
           "acc_static": acc_s, "acc_dynamic": acc_d,
           **test_stats(diff, block=20)}

    thr = out["fused_static"].abs().quantile(1 - GATE_TOP_PCT)
    gated = out[out["fused_static"].abs() >= thr]
    gdiff, gacc_d, gacc_s = daily_diff(gated)
    res["gated_top20pct"] = {"n_rows": int(len(gated)), "acc_static": gacc_s, "acc_dynamic": gacc_d,
                             **test_stats(gdiff, block=20)}
    # per-expert accuracies for the paper's table
    for n in experts:
        res[f"acc_{n}_alone"] = float((np.sign(out[f"pred_{n}"]) == np.sign(out["true"])).mean())
        res[f"acc_{n}_alone_gated"] = float((np.sign(gated[f"pred_{n}"]) == np.sign(gated["true"])).mean())
    print(json.dumps(res, indent=2))
    return res


def part_b(panel_raw):
    print("\n=== Part B: tuned learning rate, disclosed split ===")
    configs = {
        "2src_tech_vol": {"tech": TECH_FEATURE_COLS, "vol": VOL_FEATURE_COLS},
        "3src_usdinr_crude": {"tech": TECH_FEATURE_COLS, "vol": VOL_FEATURE_COLS, "macro": MACRO_COLS},
    }
    results = {}
    for cname, experts in configs.items():
        panel = prepare_panel(panel_raw, experts, 1)
        rows, sigma = walk_forward(panel, experts, 1)
        grid = []
        for k in K_GRID:
            out = fuse(rows, sigma, list(experts), k)
            diff, acc_d, acc_s = daily_diff(out)
            sel = diff[diff.index <= SELECT_END]
            tst = diff[diff.index > SELECT_END]
            grid.append({
                "k": k,
                "sel_mean_diff_pp": float(sel.mean() * 100), "sel_n_days": int(len(sel)),
                "test_mean_diff_pp": float(tst.mean() * 100),
                "test_stats": test_stats(tst),
                "acc_dynamic_full": acc_d, "acc_static_full": acc_s,
            })
        k_star = max(grid, key=lambda g: g["sel_mean_diff_pp"])
        results[cname] = {
            "k_grid": K_GRID, "selection_end": SELECT_END,
            "grid": grid,
            "k_star": k_star["k"],
            "k_star_selection_diff_pp": k_star["sel_mean_diff_pp"],
            "k_star_test_result": k_star["test_stats"],
        }
        print(f"\n[{cname}] k* = {k_star['k']} "
              f"(selection diff {k_star['sel_mean_diff_pp']:+.4f}pp) -> "
              f"test diff {k_star['test_stats']['mean_diff_pp']:+.4f}pp, "
              f"CI95 {k_star['test_stats']['ci95_pp']}, "
              f"wilcoxon p={k_star['test_stats']['wilcoxon_p']:.4f}")
        for g in grid:
            print(f"    k={g['k']:>5}: sel {g['sel_mean_diff_pp']:+.4f}pp | "
                  f"test {g['test_mean_diff_pp']:+.4f}pp (p_w={g['test_stats']['wilcoxon_p']:.3f})")
    return results


def main():
    panel_raw = pd.read_pickle(CACHE)
    print(f"Loaded lag-fixed cached panel: {len(panel_raw)} rows")
    out = {"part_a_h20_lagfixed": part_a(panel_raw), "part_b_tuned_eta": part_b(panel_raw)}
    with open(RES / "study6_corrected_final.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {RES / 'study6_corrected_final.json'}")


if __name__ == "__main__":
    main()
