"""
Step 7 - how reliable is the measurement instrument?

The three axes are judgements produced by a language model, so before any of them
is used as a regressor it is fair to ask how much of their variance is signal and
how much is the model's own noise.  We score a fixed subsample a second time with
the cache bypassed and report, per axis:

  * ICC(1,1), the one-way random-effects intraclass correlation \\citep{shrout1979},
    which is the share of total variance attributable to genuine between-headline
    differences rather than to disagreement between the two passes;
  * the Spearman rank correlation between passes;
  * exact agreement on the reporting grid, and mean absolute deviation.

A high ICC does not make the scores *valid* - it makes them repeatable.  Validity
is what the predictive tests in steps 3 and 4 are for.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"


def icc11(x: np.ndarray, y: np.ndarray) -> float:
    """One-way random-effects ICC for two measurements of the same target.

    Targets are headlines; the two 'raters' are the two independent scoring
    passes.  MSB is the between-headline mean square, MSW the within-headline
    mean square; ICC = (MSB - MSW) / (MSB + (k-1) MSW) with k = 2.
    """
    m = np.column_stack([x, y]).astype(float)
    n, k = m.shape
    if n < 3:
        return float("nan")
    target_means = m.mean(axis=1)
    grand = m.mean()
    ss_between = k * np.sum((target_means - grand) ** 2)
    ss_within = np.sum((m - target_means[:, None]) ** 2)
    ms_between = ss_between / (n - 1)
    ms_within = ss_within / (n * (k - 1))
    denom = ms_between + (k - 1) * ms_within
    return float((ms_between - ms_within) / denom) if denom > 0 else float("nan")


def main() -> None:
    con = sqlite3.connect(str(ROOT / "cache" / "scores.db"))
    first = pd.read_sql("SELECT k, nu, mu, s FROM scores", con)
    second = pd.read_sql("SELECT k, nu, mu, s FROM retest WHERE pass = 2", con)
    con.close()

    if second.empty:
        print("no retest rows yet - run: 01_score_axes.py --retest 2000")
        return

    second = second.drop_duplicates("k")
    both = first.merge(second, on="k", suffixes=("_1", "_2"))
    print(f"headlines scored twice: {len(both):,}")

    out: dict = {"n": int(len(both))}
    for axis in ("nu", "mu", "s"):
        a = both[f"{axis}_1"].to_numpy(float)
        b = both[f"{axis}_2"].to_numpy(float)
        out[axis] = {
            "icc": icc11(a, b),
            "spearman": float(pd.Series(a).corr(pd.Series(b), method="spearman")),
            "pearson": float(np.corrcoef(a, b)[0, 1]),
            "exact_agreement": float(np.mean(np.abs(a - b) < 1e-9)),
            "mad": float(np.mean(np.abs(a - b))),
            "mean_1": float(a.mean()),
            "mean_2": float(b.mean()),
        }
        r = out[axis]
        print(
            f"  {axis}: ICC={r['icc']:.3f}  rho={r['spearman']:.3f}  "
            f"exact={100*r['exact_agreement']:.1f}%  MAD={r['mad']:.3f}"
        )

    # the quantity that actually matters downstream is the reliability of the
    # product, since that is what enters the aggregate
    a1 = both["s_1"] * both["nu_1"] * both["mu_1"]
    a2 = both["s_2"] * both["nu_2"] * both["mu_2"]
    out["event_signal"] = {
        "icc": icc11(a1.to_numpy(float), a2.to_numpy(float)),
        "spearman": float(a1.corr(a2, method="spearman")),
        "mad": float(np.mean(np.abs(a1 - a2))),
    }
    print(f"  event signal a=s*nu*mu: ICC={out['event_signal']['icc']:.3f}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "reliability.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {RESULTS / 'reliability.json'}")


if __name__ == "__main__":
    main()
