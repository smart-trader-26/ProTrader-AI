"""
Step 4 - the selective forecasting experiment, run with and without the gated
text feature A.

This is the experiment the earlier version of this study did not have.  The
conviction gate there was built on price features alone, so the headline number
could not speak to the text decomposition at all.  Here the *same* gate is fitted
twice - once on price features, once on price features plus A - and the reported
quantity is the difference.  If A adds nothing, that is what the table will say.

Protocol
  * Walk-forward by calendar year.  For test year Y, the model sees only sessions
    strictly before 1 January Y.  The last 20% of the training sessions are held
    out to fit the probability calibrator and to choose the conviction threshold,
    so neither is ever chosen on test data.
  * Learner: histogram gradient boosting on the feature block, then isotonic
    calibration of the out-of-fold score, exactly the repair described in the
    methods (fit the calibrator on the distribution it will be applied to).
  * Gate: tau* is the smallest threshold whose fired bucket reaches the target
    precision on the *calibration* fold while still firing at least the minimum
    fraction of the time.
  * Reported: fired-bucket precision against the always-up base rate, the firing
    rate, and the area under the risk-coverage curve (AURC), which removes the
    dependence on any single operating point.
  * Uncertainty: a block bootstrap that resamples whole *dates* with replacement,
    which respects the cross-sectional correlation of returns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"

PRICE_FEATURES = ["mom5", "mom21", "mom63", "trend", "reversal", "lrv", "lrv_innov", "vol21"]
TEXT_FEATURES = ["A_mig", "nu_bar", "mu_bar", "n_news_scored", "frac_material"]
POLARITY_ONLY = ["pol_mean", "n_news_scored"]
RELFILT_ONLY = ["pol_relf", "frac_material", "n_news_scored"]

TARGET_PRECISION = 0.60
MIN_FIRE = 0.05


def add_price_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Backward-looking price features only; every one uses information at or
    before the session whose aggregate we pair it with."""
    p = panel.sort_values(["symbol", "session"]).copy()
    g = p.groupby("symbol")["ret"]
    p["mom5"] = g.transform(lambda x: x.rolling(5).sum())
    p["mom21"] = g.transform(lambda x: x.rolling(21).sum())
    p["mom63"] = g.transform(lambda x: x.rolling(63).sum())
    p["reversal"] = g.transform(lambda x: x.shift(0))
    p["vol21"] = g.transform(lambda x: x.rolling(21).std())
    p["trend"] = p["mom21"] - p["mom63"] / 3.0
    return p


def risk_coverage(prob: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Selective risk as a function of coverage, and the area under it.

    Sorting by confidence and sweeping the threshold traces the risk the system
    accepts for each fraction of cases it agrees to answer.  AURC is the mean
    selective risk over all coverages; lower is better and it needs no threshold.
    """
    order = np.argsort(-prob, kind="stable")
    ys = y[order]
    n = len(ys)
    errors = np.cumsum(1.0 - ys)
    counts = np.arange(1, n + 1)
    risk = errors / counts
    coverage = counts / n
    return coverage, risk, float(risk.mean())


def choose_tau(prob: np.ndarray, y: np.ndarray) -> float:
    """Eq. (4): smallest tau meeting the precision target at a minimum fire rate."""
    grid = np.unique(np.round(np.quantile(prob, np.linspace(0.50, 0.999, 200)), 4))
    best = None
    for tau in grid:
        fired = prob >= tau
        phi = fired.mean()
        if phi < MIN_FIRE or fired.sum() < 30:
            continue
        prec = y[fired].mean()
        if prec >= TARGET_PRECISION:
            best = float(tau)
            break
    if best is None:  # target unreachable on this fold: fall back to the top decile
        best = float(np.quantile(prob, 1.0 - MIN_FIRE))
    return best


def fit_variant(
    train: pd.DataFrame, calib: pd.DataFrame, test: pd.DataFrame, feats: list[str], ycol: str
) -> dict:
    Xtr, ytr = train[feats].to_numpy(float), train[ycol].to_numpy(float)
    Xca, yca = calib[feats].to_numpy(float), calib[ycol].to_numpy(float)
    Xte, yte = test[feats].to_numpy(float), test[ycol].to_numpy(float)
    if len(np.unique(ytr)) < 2 or len(test) < 50:
        return {}

    clf = HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.06, max_depth=4, min_samples_leaf=60,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        random_state=0,
    )
    clf.fit(Xtr, ytr)

    raw_ca = clf.predict_proba(Xca)[:, 1]
    raw_te = clf.predict_proba(Xte)[:, 1]
    # calibrate on the fold the map will actually be applied to
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_ca, yca)
    p_ca = iso.predict(raw_ca)
    p_te = iso.predict(raw_te)

    tau = choose_tau(p_ca, yca)
    fired = p_te >= tau
    cov, risk, aurc = risk_coverage(p_te, yte)
    base = float(yte.mean())
    out = {
        "n_test": int(len(test)),
        "base_rate": base,
        "tau": tau,
        "fire_rate": float(fired.mean()),
        "n_fired": int(fired.sum()),
        "fired_precision": float(yte[fired].mean()) if fired.sum() > 0 else np.nan,
        "edge_pp": float((yte[fired].mean() - base) * 100) if fired.sum() > 0 else np.nan,
        "aurc": aurc,
        "brier": float(np.mean((p_te - yte) ** 2)),
    }
    # precision at a fixed 10% coverage, so variants are compared like for like
    k = max(1, int(0.10 * len(p_te)))
    top = np.argsort(-p_te, kind="stable")[:k]
    out["prec_at_10pct"] = float(yte[top].mean())
    out["_probs"] = p_te
    out["_y"] = yte
    out["_dates"] = test["session"].to_numpy()
    return out


def block_bootstrap_diff(
    a: dict, b: dict, n_boot: int = 2000, coverage: float = 0.10, seed: int = 7
) -> dict:
    """Bootstrap the precision-at-fixed-coverage gap between two variants by
    resampling whole dates, which keeps same-day cross-sectional correlation."""
    rng = np.random.default_rng(seed)
    dates = np.asarray(a["_dates"])
    uniq = np.unique(dates)
    idx_by_date = {d: np.flatnonzero(dates == d) for d in uniq}
    pa, pb, y = a["_probs"], b["_probs"], a["_y"]

    def prec(p, sel):
        k = max(1, int(coverage * len(sel)))
        top = sel[np.argsort(-p[sel], kind="stable")[:k]]
        return y[top].mean()

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_by_date[d] for d in pick])
        diffs[i] = prec(pa, sel) - prec(pb, sel)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # Ties matter here: at a fixed coverage the two variants often select the very
    # same rows, so a large share of resampled differences is exactly zero and
    # both tail masses can exceed one half.  Doubling the smaller tail would then
    # return a "p-value" above 1, so the doubled tail is capped.
    p_two = 2.0 * min(float((diffs <= 0).mean()), float((diffs >= 0).mean()))
    return {
        "diff_mean_pp": float(diffs.mean() * 100),
        "ci_lo_pp": float(lo * 100),
        "ci_hi_pp": float(hi * 100),
        "p_two_sided": min(p_two, 1.0),
        "share_exact_ties": float((np.abs(diffs) < 1e-12).mean()),
    }


def run(panel: pd.DataFrame, horizon: int, news_only: bool) -> tuple[pd.DataFrame, dict]:
    ycol = f"up{horizon}_raw"
    p = add_price_features(panel)
    p = p[p[ycol].notna()].copy()
    if news_only:
        p = p[p["has_scored_news"] == 1].copy()
    feats_all = sorted(set(PRICE_FEATURES + TEXT_FEATURES + POLARITY_ONLY + RELFILT_ONLY))
    p = p.dropna(subset=PRICE_FEATURES)
    for c in feats_all:
        if c in p.columns:
            p[c] = p[c].astype(float)

    variants = {
        "price_only": PRICE_FEATURES,
        "price_plus_A": PRICE_FEATURES + TEXT_FEATURES,
        "price_plus_polarity": PRICE_FEATURES + POLARITY_ONLY,
        "price_plus_relfilt": PRICE_FEATURES + RELFILT_ONLY,
        "A_only": TEXT_FEATURES,
    }

    years = sorted(p["session"].dt.year.unique())
    test_years = [y for y in years if y >= years[0] + 4]
    rows, pooled = [], {k: {"probs": [], "y": [], "dates": []} for k in variants}

    for Y in test_years:
        tr_all = p[p["session"] < f"{Y}-01-01"]
        te = p[(p["session"] >= f"{Y}-01-01") & (p["session"] < f"{Y + 1}-01-01")]
        if len(tr_all) < 2000 or len(te) < 100:
            continue
        cut = tr_all["session"].quantile(0.80)
        tr, ca = tr_all[tr_all["session"] <= cut], tr_all[tr_all["session"] > cut]
        if len(ca) < 300:
            continue
        for name, feats in variants.items():
            res = fit_variant(tr, ca, te, feats, ycol)
            if not res:
                continue
            pooled[name]["probs"].append(res.pop("_probs"))
            pooled[name]["y"].append(res.pop("_y"))
            pooled[name]["dates"].append(res.pop("_dates"))
            rows.append({"horizon": horizon, "test_year": int(Y), "variant": name, **res})

    per_year = pd.DataFrame(rows)

    pooled_res = {}
    for name in variants:
        if not pooled[name]["probs"]:
            continue
        pr = np.concatenate(pooled[name]["probs"])
        yy = np.concatenate(pooled[name]["y"])
        dd = np.concatenate(pooled[name]["dates"])
        cov, risk, aurc = risk_coverage(pr, yy)
        k = max(1, int(0.10 * len(pr)))
        top = np.argsort(-pr, kind="stable")[:k]
        pooled_res[name] = {
            "n": int(len(pr)),
            "base_rate": float(yy.mean()),
            "aurc": aurc,
            "prec_at_10pct": float(yy[top].mean()),
            "brier": float(np.mean((pr - yy) ** 2)),
            "_probs": pr, "_y": yy, "_dates": dd,
        }

    comparisons = {}
    if "price_plus_A" in pooled_res and "price_only" in pooled_res:
        comparisons["A_vs_price"] = block_bootstrap_diff(
            pooled_res["price_plus_A"], pooled_res["price_only"]
        )
    if "price_plus_A" in pooled_res and "price_plus_relfilt" in pooled_res:
        comparisons["A_vs_relfilt"] = block_bootstrap_diff(
            pooled_res["price_plus_A"], pooled_res["price_plus_relfilt"]
        )
    if "price_plus_A" in pooled_res and "price_plus_polarity" in pooled_res:
        comparisons["A_vs_polarity"] = block_bootstrap_diff(
            pooled_res["price_plus_A"], pooled_res["price_plus_polarity"]
        )

    curves = {}
    for name, r in pooled_res.items():
        cov, risk, _ = risk_coverage(r["_probs"], r["_y"])
        step = max(1, len(cov) // 400)
        curves[name] = {"coverage": cov[::step].tolist(), "risk": risk[::step].tolist()}
        for k in ("_probs", "_y", "_dates"):
            r.pop(k, None)

    return per_year, {
        "horizon": horizon,
        "news_only": news_only,
        "pooled": pooled_res,
        "comparisons": comparisons,
        "curves": curves,
        "test_years": [int(y) for y in sorted(per_year["test_year"].unique())]
        if not per_year.empty else [],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(ROOT / "cache" / "mig_panel.parquet"))
    ap.add_argument("--horizons", default="1,5,21")
    ap.add_argument("--all-rows", action="store_true", help="include sessions with no news")
    args = ap.parse_args()

    panel = pd.read_parquet(args.panel)
    RESULTS.mkdir(exist_ok=True)
    all_years, summary = [], {}
    for h in [int(x) for x in args.horizons.split(",")]:
        py, summ = run(panel, h, news_only=not args.all_rows)
        all_years.append(py)
        summary[str(h)] = summ
        print(f"\n=== horizon {h} sessions ===")
        if not py.empty:
            keep = ["test_year", "variant", "n_test", "base_rate", "fire_rate",
                    "fired_precision", "edge_pp", "aurc", "prec_at_10pct"]
            print(py[keep].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
        print(json.dumps({k: v for k, v in summ["pooled"].items()}, indent=1))
        print("comparisons:", json.dumps(summ["comparisons"], indent=1))

    tag = "allrows" if args.all_rows else "newsrows"
    pd.concat(all_years, ignore_index=True).to_csv(RESULTS / f"gate_by_year_{tag}.csv", index=False)
    (RESULTS / f"gate_summary_{tag}.json").write_text(json.dumps(summary, indent=1))
    print(f"\nwrote results to {RESULTS}")


if __name__ == "__main__":
    main()
