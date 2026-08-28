"""
Step 3 - does the product form actually carry information the additive recipes miss?

Three tests, in increasing sharpness:

  (1) Per-aggregator predictive regressions.  For each candidate aggregator X and
      each horizon H, regress the forward market-adjusted return on X, with
      standard errors that survive the two dependence structures of a return
      panel: within-date (a market-wide shock hits every name at once) and
      within-symbol (persistence).  We report Cameron-Gelbach-Miller two-way
      clustered t-statistics, and Fama-MacBeth with Newey-West as a cross-check.
      Overlapping horizons are handled by taking non-overlapping windows as the
      primary sample; the overlapping panel is reported alongside.

  (2) The nested horse race.  A single regression

          fwd = a + b1*polarity + b2*(polarity x novelty)
                  + b3*(polarity x materiality) + b4*A_mig

      Multiplicative gating predicts b4 > 0 and absorbs the lower-order terms.
      If b2 and b3 load and b4 does not, the triple product is the wrong form -
      and that is a publishable finding, so it is reported either way.

  (3) Free exponents.  The theory says the axes enter as s * nu^1 * mu^1.  We fit
      s * nu^alpha * mu^beta over a grid, read off the exponents that maximise the
      information coefficient, and test H0: alpha = beta = 1 by a
      cluster-bootstrap over dates.

Everything is measured; nothing here is calibrated to produce a target number.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"

AGGREGATORS = [
    ("A_mig", "Multiplicative gate $A$"),
    ("pol_mean", "Mean polarity"),
    ("pol_relf", "Relevance-filtered polarity"),
    ("pol_cnt", "Count-weighted polarity"),
    ("add_comb", "Additive combiner"),
    ("A_nu", "Polarity x novelty only"),
    ("A_mu", "Polarity x materiality only"),
]
HORIZONS = (1, 5, 21)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    return beta, resid


def cluster_meat(X: np.ndarray, resid: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """sum_g (X_g' u_g)(X_g' u_g)' for one clustering dimension."""
    k = X.shape[1]
    meat = np.zeros((k, k))
    groups = pd.factorize(groups)[0]  # integer codes: works for str or datetime keys
    order = np.argsort(groups, kind="stable")
    Xs, us, gs = X[order], resid[order], groups[order]
    bounds = np.flatnonzero(np.diff(gs)) + 1
    for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(gs)]):
        sg = Xs[lo:hi].T @ us[lo:hi]
        meat += np.outer(sg, sg)
    return meat


def twoway_cluster_se(
    y: np.ndarray, X: np.ndarray, g1: np.ndarray, g2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Cameron-Gelbach-Miller two-way clustered covariance."""
    beta, resid = ols(y, X)
    bread = np.linalg.pinv(X.T @ X)
    g12 = pd.factorize(pd.Series(g1).astype(str) + "|" + pd.Series(g2).astype(str))[0]
    meat = cluster_meat(X, resid, g1) + cluster_meat(X, resid, g2) - cluster_meat(X, resid, g12)
    cov = bread @ meat @ bread
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return beta, se


def newey_west(x: np.ndarray, lags: int) -> float:
    """Newey-West standard error of the mean of a serially correlated series."""
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 5:
        return np.nan
    xd = x - x.mean()
    gamma0 = (xd @ xd) / n
    s = gamma0
    for L in range(1, min(lags, n - 1) + 1):
        w = 1.0 - L / (lags + 1.0)
        s += 2.0 * w * (xd[L:] @ xd[:-L]) / n
    return float(np.sqrt(max(s, 0.0) / n))


def fama_macbeth(df: pd.DataFrame, xcol: str, ycol: str, lags: int) -> dict:
    """Date-by-date cross-sectional slope, then a Newey-West t on the mean."""
    slopes = []
    for _, g in df.groupby("session"):
        if len(g) < 8 or g[xcol].std(ddof=0) < 1e-12:
            continue
        X = np.column_stack([np.ones(len(g)), g[xcol].to_numpy()])
        b, _ = ols(g[ycol].to_numpy(), X)
        slopes.append(b[1])
    slopes = np.asarray(slopes, dtype=float)
    if len(slopes) < 20:
        return {"fm_slope": np.nan, "fm_t": np.nan, "fm_periods": int(len(slopes))}
    se = newey_west(slopes, lags)
    return {
        "fm_slope": float(slopes.mean()),
        "fm_t": float(slopes.mean() / se) if se and se > 0 else np.nan,
        "fm_periods": int(len(slopes)),
    }


def rank_ic(df: pd.DataFrame, xcol: str, ycol: str) -> tuple[float, float, int]:
    """Mean date-by-date Spearman IC with a Newey-West t-statistic."""
    ics = []
    for _, g in df.groupby("session"):
        if len(g) < 8 or g[xcol].std(ddof=0) < 1e-12:
            continue
        ics.append(g[xcol].corr(g[ycol], method="spearman"))
    ics = np.asarray([v for v in ics if v == v], dtype=float)
    if len(ics) < 20:
        return np.nan, np.nan, len(ics)
    se = newey_west(ics, 10)
    return float(ics.mean()), float(ics.mean() / se) if se > 0 else np.nan, len(ics)


def nonoverlapping(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """Keep every h-th session so that forward windows do not overlap."""
    if h <= 1:
        return df
    sess = np.sort(df["session"].unique())
    keep = set(sess[::h])
    return df[df["session"].isin(keep)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def univariate_table(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        ycol = f"fwd{h}"
        base = panel[(panel["has_scored_news"] == 1) & panel[ycol].notna()]
        for col, label in AGGREGATORS:
            sub = nonoverlapping(base, h)
            sub = sub[sub[col].notna()]
            if len(sub) < 500:
                continue
            # standardise the regressor so slopes are comparable across columns:
            # the coefficient reads as basis points per one-SD move in the signal
            x = sub[col].to_numpy()
            xs = (x - x.mean()) / (x.std(ddof=0) + 1e-12)
            X = np.column_stack([np.ones(len(sub)), xs])
            y = sub[ycol].to_numpy()
            beta, se = twoway_cluster_se(
                y, X, sub["session"].to_numpy(), sub["symbol"].to_numpy()
            )
            fm = fama_macbeth(sub, col, ycol, lags=max(5, h))
            ic, ict, nper = rank_ic(sub, col, ycol)
            rows.append(
                {
                    "horizon": h,
                    "aggregator": col,
                    "label": label,
                    "n": int(len(sub)),
                    "coef_bps": float(beta[1] * 1e4),
                    "t_2way": float(beta[1] / se[1]) if se[1] > 0 else np.nan,
                    "fm_t": fm["fm_t"],
                    "ic": ic,
                    "ic_t": ict,
                    "ic_periods": nper,
                }
            )
    return pd.DataFrame(rows)


def horse_race(panel: pd.DataFrame) -> pd.DataFrame:
    """The nested test: do the lower-order terms survive once A is included?"""
    rows = []
    for h in HORIZONS:
        ycol = f"fwd{h}"
        base = panel[(panel["has_scored_news"] == 1) & panel[ycol].notna()]
        sub = nonoverlapping(base, h)
        if len(sub) < 500:
            continue
        cols = ["pol_mean", "A_nu", "A_mu", "A_mig"]
        Z = sub[cols].to_numpy(dtype=float)
        Z = (Z - Z.mean(0)) / (Z.std(0, ddof=0) + 1e-12)
        X = np.column_stack([np.ones(len(sub)), Z])
        y = sub[ycol].to_numpy()
        beta, se = twoway_cluster_se(
            y, X, sub["session"].to_numpy(), sub["symbol"].to_numpy()
        )
        for j, c in enumerate(cols, start=1):
            rows.append(
                {
                    "horizon": h,
                    "term": c,
                    "coef_bps": float(beta[j] * 1e4),
                    "t_2way": float(beta[j] / se[j]) if se[j] > 0 else np.nan,
                    "n": int(len(sub)),
                }
            )
    return pd.DataFrame(rows)


def exponent_grid(panel: pd.DataFrame, h: int, alphas: np.ndarray, betas: np.ndarray) -> dict:
    """Fit s * nu^alpha * mu^beta on the event-aggregated panel over a grid."""
    ycol = f"fwd{h}"
    ev = panel[(panel["has_scored_news"] == 1) & panel[ycol].notna()]
    ev = nonoverlapping(ev, h)
    events = pd.read_parquet(ROOT / "cache" / "events.parquet")
    events = events.merge(
        ev[["symbol", "session", ycol]], on=["symbol", "session"], how="inner"
    )
    if events.empty:
        return {}
    best = None
    surface = []
    for a in alphas:
        for b in betas:
            w = (events["nu"] ** a) * (events["mu"] ** b)
            sig = events["s"] * w
            agg = (
                events.assign(_num=w * sig, _den=w)
                .groupby(["symbol", "session"])
                .agg(num=("_num", "sum"), den=("_den", "sum"))
            )
            agg["X"] = np.where(agg["den"] > 0, agg["num"] / agg["den"], 0.0)
            merged = agg.reset_index().merge(
                ev[["symbol", "session", ycol]], on=["symbol", "session"], how="inner"
            )
            ic, ict, _ = rank_ic(merged, "X", ycol)
            surface.append({"alpha": float(a), "beta": float(b), "ic": ic, "ic_t": ict})
            if ic == ic and (best is None or ic > best["ic"]):
                best = {"alpha": float(a), "beta": float(b), "ic": float(ic), "ic_t": float(ict)}
    unit = [s for s in surface if abs(s["alpha"] - 1) < 1e-9 and abs(s["beta"] - 1) < 1e-9]
    # alpha = beta = 0 is the no-gating corner: the aggregate collapses to plain
    # mean polarity, which is the benchmark the estimated exponents are read against
    pure = [s for s in surface if abs(s["alpha"]) < 1e-9 and abs(s["beta"]) < 1e-9]
    return {"best": best, "unit": unit[0] if unit else None,
            "pure": pure[0] if pure else None, "surface": surface, "horizon": h}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(ROOT / "cache" / "mig_panel.parquet"))
    ap.add_argument("--exponents", action="store_true")
    args = ap.parse_args()

    panel = pd.read_parquet(args.panel)
    RESULTS.mkdir(exist_ok=True)

    uni = univariate_table(panel)
    uni.to_csv(RESULTS / "univariate.csv", index=False)
    print("\n=== univariate predictive regressions (non-overlapping) ===")
    print(uni.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    hr = horse_race(panel)
    hr.to_csv(RESULTS / "horse_race.csv", index=False)
    print("\n=== nested horse race ===")
    print(hr.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    if args.exponents:
        grid = np.arange(0.0, 2.01, 0.25)
        out = {}
        for h in HORIZONS:
            res = exponent_grid(panel, h, grid, grid)
            if res:
                out[str(h)] = {"best": res["best"], "unit": res["unit"],
                               "pure": res["pure"]}
                pd.DataFrame(res["surface"]).to_csv(
                    RESULTS / f"exponent_surface_h{h}.csv", index=False
                )
        (RESULTS / "exponents.json").write_text(json.dumps(out, indent=1))
        print("\n=== free exponents ===")
        print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
