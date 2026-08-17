"""
Study 7 -- does dynamic fusion reduce mean squared error, even where it does
not flip enough signs to move directional accuracy?

Motivation (written before looking at any MSE number): the scale-normalized
softmax w_i ~ exp(-sigma_i^2/tau_t) is, to first order, an inverse-variance
weighting -- the classical minimum-variance unbiased combination of several
noisy, conditionally independent estimates (the same logic behind fixed-
effect meta-analysis pooling). That result is about squared error, not about
how often the combined estimate's SIGN is correct. Studies 1/3/4 and the
leak-corrected re-runs (study5d, study6) all evaluate *directional accuracy*,
a coarse binary statistic that a small, real reduction in forecast variance
need not move at all if it is not large enough to flip predictions that were
already going to land on the correct side of zero. This script asks the
mechanism's more natural question directly: is the combined forecast's
squared error lower under dynamic weighting than under static weighting,
on the same real data and the same leak-free protocols already used
elsewhere in this paper?

Four leak-free comparisons, matching the paper's existing structure exactly:
  (a) Study 1 setting, 2-source, H=1 -- reuses the already-saved daily MSE
      columns in study1_daily.csv (no re-run needed).
  (b) Study 3 setting, 2-source, H=20 -- computed from the already-saved
      per-ticker-day predictions in study3_daily.csv (no re-run needed).
  (c) Study 4 setting, 3-source, H=1, BOTH leaks fixed -- one fresh
      walk-forward pass (identical protocol to study5d/study6 Part B, k=1).
  (d) Study 4 setting, 3-source, H=20, lag-fixed -- one fresh walk-forward
      pass (identical protocol to study6 Part A), ungated and top-20%-gated.

All four use the same paired daily/blocked bootstrap machinery already used
throughout the paper. Nothing here is fit to come out positive: the k=1,
W=10, quarterly-refit, 44-ticker protocol is fixed by the rest of the paper,
not chosen for this test.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

from study6_corrected_final import (
    GATE_TOP_PCT, MACRO_COLS, TECH_FEATURE_COLS, VOL_FEATURE_COLS,
    fuse, prepare_panel, walk_forward,
)

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
RES = HERE / "results"


def block_bootstrap_diff(diff_by_date, block=1, n_boot=5000, seed=42, m_total=9):
    vals = diff_by_date.sort_index().values
    n = len(vals)
    rng = np.random.default_rng(seed)
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
        "mean_diff": float(vals.mean()),
        "ci95": [ci95[0], ci95[1]],
        "ci_bonferroni_m9": [cib[0], cib[1]],
        "wilcoxon_p": wp, "paired_t_p": tp,
        "excludes_zero_95": bool(ci95[0] > 0 or ci95[1] < 0),
        "excludes_zero_bonferroni": bool(cib[0] > 0 or cib[1] < 0),
        "n": int(n),
    }


def study1_mse():
    df = pd.read_csv(RES / "study1_daily.csv", parse_dates=["date"])
    diff = (df["mse_static"] - df["mse_dynamic_scaled"]).rename("d")
    diff.index = df["date"]
    r = block_bootstrap_diff(diff, block=1)
    r["mse_static_mean"] = float(df["mse_static"].mean())
    r["mse_dynamic_mean"] = float(df["mse_dynamic_scaled"].mean())
    print("Study 1 (2-source, H=1) MSE reduction (static - dynamic):", json.dumps(r, indent=2))
    return r


def study3_mse():
    df = pd.read_csv(RES / "study3_daily.csv", parse_dates=["date"])
    df["se_dynamic"] = (df["fused_dynamic"] - df["true"]) ** 2
    df["se_static"] = (df["fused_static"] - df["true"]) ** 2
    day = df.groupby("date").agg(se_dynamic=("se_dynamic", "mean"), se_static=("se_static", "mean"),
                                  abs_static=("fused_static", lambda s: s.abs().mean()))
    diff = (day["se_static"] - day["se_dynamic"]).rename("d")
    r = block_bootstrap_diff(diff, block=20)
    r["mse_static_mean"] = float(df["se_static"].mean())
    r["mse_dynamic_mean"] = float(df["se_dynamic"].mean())

    thr = df["fused_static"].abs().quantile(1 - GATE_TOP_PCT)
    gated = df[df["fused_static"].abs() >= thr]
    gday = gated.groupby("date").agg(se_dynamic=("se_dynamic", "mean"), se_static=("se_static", "mean"))
    gdiff = (gday["se_static"] - gday["se_dynamic"]).rename("d")
    rg = block_bootstrap_diff(gdiff, block=20)
    rg["mse_static_mean"] = float(gated["se_static"].mean())
    rg["mse_dynamic_mean"] = float(gated["se_dynamic"].mean())
    print("Study 3 (2-source, H=20, ungated) MSE reduction:", json.dumps(r, indent=2))
    print("Study 3 (2-source, H=20, gated) MSE reduction:", json.dumps(rg, indent=2))
    return r, rg


def source3_mse(horizon, gate=False):
    panel_raw = pd.read_pickle(RES / "study5b_panel_cache_lagfixed.pkl")
    experts = {"tech": TECH_FEATURE_COLS, "vol": VOL_FEATURE_COLS, "macro": MACRO_COLS}
    panel = prepare_panel(panel_raw, experts, horizon)
    rows, sigma = walk_forward(panel, experts, horizon)
    out = fuse(rows, sigma, list(experts), k=1)
    out["se_dynamic"] = (out["fused_dynamic"] - out["true"]) ** 2
    out["se_static"] = (out["fused_static"] - out["true"]) ** 2

    block = horizon if horizon > 1 else 1
    day = out.groupby("date").agg(se_dynamic=("se_dynamic", "mean"), se_static=("se_static", "mean"))
    diff = (day["se_static"] - day["se_dynamic"]).rename("d")
    r = block_bootstrap_diff(diff, block=block)
    r["mse_static_mean"] = float(out["se_static"].mean())
    r["mse_dynamic_mean"] = float(out["se_dynamic"].mean())
    r["n_rows"] = int(len(out))

    result = {"ungated": r}
    if gate:
        thr = out["fused_static"].abs().quantile(1 - GATE_TOP_PCT)
        gated = out[out["fused_static"].abs() >= thr]
        gday = gated.groupby("date").agg(se_dynamic=("se_dynamic", "mean"), se_static=("se_static", "mean"))
        gdiff = (gday["se_static"] - gday["se_dynamic"]).rename("d")
        rg = block_bootstrap_diff(gdiff, block=block)
        rg["mse_static_mean"] = float(gated["se_static"].mean())
        rg["mse_dynamic_mean"] = float(gated["se_dynamic"].mean())
        rg["n_rows"] = int(len(gated))
        result["gated_top20pct"] = rg
    print(f"3-source H={horizon} MSE reduction:", json.dumps(result, indent=2))
    return result


def main():
    out = {}
    out["study1_2src_h1"] = study1_mse()
    r3, r3g = study3_mse()
    out["study3_2src_h20_ungated"] = r3
    out["study3_2src_h20_gated"] = r3g
    out["study4_3src_h1_leakfree"] = source3_mse(1, gate=False)
    out["study4_3src_h20_leakfree"] = source3_mse(20, gate=True)

    with open(RES / "study7_mse_analysis.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {RES / 'study7_mse_analysis.json'}")


if __name__ == "__main__":
    main()
