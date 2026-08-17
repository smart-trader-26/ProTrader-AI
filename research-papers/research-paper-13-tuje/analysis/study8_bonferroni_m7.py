"""
Study 8 -- recompute the paper's standing Bonferroni family (m=7: Study 1;
Study 2; Study 3 ungated and gated; Study 4 at next-day horizon, and at
20-day horizon ungated and gated -- the same seven comparisons the paper
has used throughout) at the corrected, leak-free Study 4 numbers.

Study 1, Study 2, and Study 3 are unaffected by the macro-join and
error-ordering bugs (they never touch macro data, and Study 3's pending-
queue ordering was already checked clean in study5d's docstring) so their
m=7 Bonferroni intervals are unchanged from the numbers already in the
paper. Only Study 4's two cells (H=1 leak-free, H=20 leak-free
ungated/gated) need the interval recomputed at the m=7 level instead of
the ad-hoc m=9 used while this was still being explored in study6/study6c.
"""

import json
from pathlib import Path

import numpy as np

from study6_corrected_final import (
    MACRO_COLS, TECH_FEATURE_COLS, VOL_FEATURE_COLS, GATE_TOP_PCT,
    prepare_panel, walk_forward, fuse, daily_diff,
)

HERE = Path(__file__).parent
RES = HERE / "results"
M = 7


def ci_at_level(diff, block, level, n_boot=20000, seed=42):
    vals = diff.sort_index().values
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
    lo, hi = (1 - level) / 2 * 100, (1 - (1 - level) / 2) * 100
    return [float(np.percentile(boots, lo)) * 100, float(np.percentile(boots, hi)) * 100]


def main():
    panel_raw = __import__("pandas").read_pickle(RES / "study5b_panel_cache_lagfixed.pkl")
    experts = {"tech": TECH_FEATURE_COLS, "vol": VOL_FEATURE_COLS, "macro": MACRO_COLS}

    out = {}

    # H=1 leak-free
    panel1 = prepare_panel(panel_raw, experts, 1)
    rows1, sigma1 = walk_forward(panel1, experts, 1)
    out1 = fuse(rows1, sigma1, list(experts), k=1)
    diff1, _, _ = daily_diff(out1)
    out["h1_leakfree"] = {
        "ci95_pp": ci_at_level(diff1, 1, 0.95),
        "ci_bonferroni_m7_pp": ci_at_level(diff1, 1, 1 - 0.05 / M),
    }

    # H=20 leak-free
    panel20 = prepare_panel(panel_raw, experts, 20)
    rows20, sigma20 = walk_forward(panel20, experts, 20)
    out20 = fuse(rows20, sigma20, list(experts), k=1)
    diff20, _, _ = daily_diff(out20)
    out["h20_leakfree_ungated"] = {
        "ci95_pp": ci_at_level(diff20, 20, 0.95),
        "ci_bonferroni_m7_pp": ci_at_level(diff20, 20, 1 - 0.05 / M),
    }
    thr = out20["fused_static"].abs().quantile(1 - GATE_TOP_PCT)
    gated20 = out20[out20["fused_static"].abs() >= thr]
    gdiff20, _, _ = daily_diff(gated20)
    out["h20_leakfree_gated"] = {
        "ci95_pp": ci_at_level(gdiff20, 20, 0.95),
        "ci_bonferroni_m7_pp": ci_at_level(gdiff20, 20, 1 - 0.05 / M),
    }

    print(json.dumps(out, indent=2))
    with open(RES / "study8_bonferroni_m7.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
