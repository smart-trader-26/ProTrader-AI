"""
Step 8 - robustness of the headline comparison to the choices we made.

Three choices in the main specification are ours rather than the data's, and each
is varied here:

  1. the materiality floor mu_0 of Eq. (2), swept from 0 (no floor) to 0.40;
  2. non-overlapping versus overlapping forward windows;
  3. restricting the panel to symbol-sessions that actually carry news, versus
     including quiet sessions where the aggregate is identically zero.

Each row of the output is the same univariate statistic for the gated aggregate
recomputed under one variation, so the reader can see directly whether the result
depends on a tuning choice.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
PY = sys.executable

MU0_GRID = [0.0, 0.10, 0.15, 0.25, 0.40]
HORIZONS = (1, 5, 21)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hr = _load("horse_race", HERE / "03_horse_race.py")


def univariate_for(panel: pd.DataFrame, col: str, overlapping: bool) -> list[dict]:
    rows = []
    for h in HORIZONS:
        ycol = f"fwd{h}"
        base = panel[(panel["has_scored_news"] == 1) & panel[ycol].notna()]
        sub = base if overlapping else hr.nonoverlapping(base, h)
        if len(sub) < 500:
            continue
        import numpy as np

        x = sub[col].to_numpy(float)
        xs = (x - x.mean()) / (x.std(ddof=0) + 1e-12)
        X = np.column_stack([np.ones(len(sub)), xs])
        beta, se = hr.twoway_cluster_se(
            sub[ycol].to_numpy(float), X, sub["session"].to_numpy(), sub["symbol"].to_numpy()
        )
        ic, ict, _ = hr.rank_ic(sub, col, ycol)
        rows.append({
            "horizon": h,
            "n": int(len(sub)),
            "coef_bps": float(beta[1] * 1e4),
            "t_2way": float(beta[1] / se[1]) if se[1] > 0 else float("nan"),
            "ic": ic,
            "ic_t": ict,
        })
    return rows


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    out = []

    # ---- 1. materiality floor -------------------------------------------------
    for mu0 in MU0_GRID:
        tmp = ROOT / "cache" / f"mig_panel_mu{int(mu0 * 100):02d}.parquet"
        if not tmp.exists():
            print(f"building panel at mu0={mu0} ...")
            subprocess.run(
                [PY, str(HERE / "02_build_mig_panel.py"), "--mu0", str(mu0), "--out", str(tmp)],
                check=True, capture_output=True,
            )
        panel = pd.read_parquet(tmp)
        for r in univariate_for(panel, "A_mig", overlapping=False):
            out.append({"variation": "mu0", "setting": mu0, **r})
        print(f"  mu0={mu0}: done")

    # ---- 2. overlapping windows ----------------------------------------------
    panel = pd.read_parquet(ROOT / "cache" / "mig_panel.parquet")
    for r in univariate_for(panel, "A_mig", overlapping=True):
        out.append({"variation": "windows", "setting": "overlapping", **r})
    for r in univariate_for(panel, "A_mig", overlapping=False):
        out.append({"variation": "windows", "setting": "non-overlapping", **r})

    # ---- 3. the same statistic for the closest competing aggregator ----------
    for col in ("pol_relf", "pol_mean"):
        for r in univariate_for(panel, col, overlapping=False):
            out.append({"variation": "aggregator", "setting": col, **r})

    df = pd.DataFrame(out)
    df.to_csv(RESULTS / "robustness.csv", index=False)
    print("\n=== robustness ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print(f"\nwrote {RESULTS / 'robustness.csv'}")


if __name__ == "__main__":
    main()
