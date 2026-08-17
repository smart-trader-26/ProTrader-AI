"""
Study 6c -- two remaining numbers for the corrected paper's tables.

1. Per-expert (tech / vol / macro alone) directional accuracies for the
   leak-free H=1 three-source run (same walk-forward as study5d / study6
   part B; per-expert predictions are unaffected by the fusion rule, so a
   single pass suffices).
2. Study 3's ungated two-source 20-day effect re-tested at the paper's new
   comparison-family size: m=9 Bonferroni block-bootstrap CI (block=20)
   from the already-saved study3_daily.csv.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from study6_corrected_final import (
    MACRO_COLS, TECH_FEATURE_COLS, VOL_FEATURE_COLS,
    prepare_panel, walk_forward,
)

HERE = Path(__file__).parent
RES = HERE / "results"


def per_expert_h1():
    panel_raw = pd.read_pickle(RES / "study5b_panel_cache_lagfixed.pkl")
    experts = {"tech": TECH_FEATURE_COLS, "vol": VOL_FEATURE_COLS, "macro": MACRO_COLS}
    panel = prepare_panel(panel_raw, experts, 1)
    rows, _sigma = walk_forward(panel, experts, 1)
    out = {"n_rows": int(len(rows))}
    for n in experts:
        out[f"acc_{n}_alone"] = float((np.sign(rows[f"pred_{n}"]) == np.sign(rows["true"])).mean())
    return out


def study3_m9():
    df = pd.read_csv(RES / "study3_daily.csv")
    df["correct_dynamic"] = (np.sign(df["fused_dynamic"]) == np.sign(df["true"])).astype(int)
    df["correct_static"] = (np.sign(df["fused_static"]) == np.sign(df["true"])).astype(int)
    diff = (df.groupby("date")["correct_dynamic"].mean()
            - df.groupby("date")["correct_static"].mean()).sort_index()
    vals = diff.values
    n = len(vals)
    block = 20
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(20000):
        idx = rng.integers(0, n_blocks, size=n_blocks)
        boots.append(np.concatenate([vals[i * block:(i + 1) * block] for i in idx])[:n].mean())
    lvl = (0.05 / 9) / 2 * 100
    ci95 = [float(np.percentile(boots, 2.5)) * 100, float(np.percentile(boots, 97.5)) * 100]
    ci_m9 = [float(np.percentile(boots, lvl)) * 100, float(np.percentile(boots, 100 - lvl)) * 100]
    return {"n_days": int(n), "mean_diff_pp": float(vals.mean() * 100),
            "ci95_pp_blk20": ci95, "ci_bonferroni_m9_pp_blk20": ci_m9,
            "excludes_zero_m9": bool(ci_m9[0] > 0 or ci_m9[1] < 0)}


def main():
    out = {"h1_3src_leakfree_per_expert": per_expert_h1(), "study3_ungated_m9": study3_m9()}
    print(json.dumps(out, indent=2))
    with open(RES / "study6c_table_numbers.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
