"""Step 5: out-of-sample comparison of temporal aggregation filters.

Every filter maps the last K sessions of measured sentiment to one feature
f(t) = sum_k h(k) m1(t-k), which is then mapped to a return forecast by a
scale fitted on the training window only.  Filters differ solely in h.

  latest        h = delta(k)                     one session of news
  uniform-W     h = 1/W on k < W                 the usual fixed window
  exp-H         h ~ 2^(-k/H)                     exponential decay
  lagprofile    h ~ per-lag univariate slopes    reading the correlogram
  corrected     h ~ g_til                        noise-corrected response
  wiener        h ~ R_mm^{-1} c                  generalised matched filter

The identity R_mm^{-1} R_SS g_til = R_mm^{-1} c makes the last row the
noise-whitened matched filter for the corrected kernel.

Reported out of sample: pooled information coefficient, directional
accuracy, Newey-West t statistic, Diebold-Mariano test against the best
fixed window, and a decile long-short spread.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

TRAIN_END = "2016-12-31"


def lag_matrix(df: pd.DataFrame, col: str, K: int) -> np.ndarray:
    g = df.groupby("symbol")[col]
    return np.column_stack([g.shift(k).to_numpy() for k in range(K)])


def forward_target(df: pd.DataFrame, h: int, col: str) -> np.ndarray:
    g = df.groupby("symbol")[col]
    s = sum(g.shift(-i) for i in range(1, h + 1))
    return (s / h if col != "ret_adj" else s).to_numpy()


def newey_west_t(x: np.ndarray, lags: int) -> float:
    """t statistic for H0: E[x]=0 with a Newey-West long-run variance."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 10:
        return np.nan
    mu = float(x.mean())
    z = x - mu
    s = float(z @ z / n)
    for L in range(1, min(lags, n - 1) + 1):
        c = float(z[L:] @ z[:-L] / n)
        s += 2 * (1 - L / (lags + 1)) * c
    se = np.sqrt(max(s, 1e-30) / n)
    return float(mu / se) if se > 0 else np.nan


def build_filters(K: int, kern: dict) -> dict[str, np.ndarray]:
    f = {}
    d = np.zeros(K)
    d[0] = 1.0
    f["latest"] = d
    for W in (2, 3, 5, 10):
        if W <= K:
            h = np.zeros(K)
            h[:W] = 1.0 / W
            f[f"uniform-{W}"] = h
    for H in (1, 2, 3, 5):
        h = 2.0 ** (-np.arange(K) / H)
        f[f"exp-{H}"] = h / h.sum()
    f["lagprofile"] = np.array(kern["lag_profile"])
    f["corrected"] = np.array(kern["corrected"])
    f["wiener"] = np.array(kern["deconvolved"])
    return {k: v / (np.linalg.norm(v) + 1e-12) for k, v in f.items()}


def count_weighted(X: np.ndarray, N: np.ndarray, W: int) -> np.ndarray:
    """Headline-count weighted average over the last W sessions.

    This is the stronger practitioner baseline: a plain session average
    treats a session carrying one headline the same as one carrying twenty,
    whereas this weights each session by how much news it actually held.
    """
    num = (X[:, :W] * N[:, :W]).sum(1)
    den = N[:, :W].sum(1)
    out = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return out


def decile_spread(df: pd.DataFrame, f: np.ndarray, y: np.ndarray,
                  ok: np.ndarray, hz: int) -> tuple[float, float]:
    """Daily cross-sectional top-minus-bottom decile mean return.

    Only names carrying a live signal enter the ranking; ranking names whose
    feature is exactly zero because no news arrived would measure sorting on
    the absence of news rather than on sentiment.
    """
    live = ok & (np.abs(f) > 1e-12)
    d = pd.DataFrame({"session": df["session"].to_numpy()[live],
                      "f": f[live], "y": y[live]})
    out = []
    for _, grp in d.groupby("session"):
        if len(grp) < 20 or grp["f"].std() == 0:
            continue
        q = grp["f"].rank(pct=True)
        top = grp.loc[q > 0.9, "y"].mean()
        bot = grp.loc[q < 0.1, "y"].mean()
        if np.isfinite(top) and np.isfinite(bot):
            out.append(top - bot)
    if len(out) < 50:
        return np.nan, np.nan
    a = np.array(out)
    ann = float(a.mean() * 252 / hz)
    ir = float(a.mean() / a.std() * np.sqrt(252 / hz))
    return ann, ir


def evaluate(name, f, y, df, ok_tr, ok_te, nw_lags, hz):
    """Score a filter through its training-fitted forecast.

    The scale (and sign) of every filter is fixed by a regression on the
    training window, so all filters are compared as forecasts of the same
    response.  Scoring the raw feature instead would flip the sign of any
    filter whose taps are negative -- as they are for volatility, where
    good news lowers the response -- and make the filters incomparable.
    """
    b = float(np.polyfit(f[ok_tr], y[ok_tr], 1)[0])
    pred = b * f
    ic_tr = float(np.corrcoef(pred[ok_tr], y[ok_tr])[0, 1])
    ic = float(np.corrcoef(pred[ok_te], y[ok_te])[0, 1])
    live = ok_te & (np.abs(f) > 1e-12)
    acc = float(np.mean(np.sign(pred[live]) == np.sign(y[live]))) \
        if live.sum() > 100 else np.nan
    prod = (pred[ok_te] - pred[ok_te].mean()) * (y[ok_te] - y[ok_te].mean())
    t = newey_west_t(prod, nw_lags)
    err = (y[ok_te] - pred[ok_te]) ** 2
    ann, ir = decile_spread(df, pred, y, ok_te, hz)
    return {"filter": name, "ic": ic, "ic_train": ic_tr, "ic_live": float(
        np.corrcoef(pred[live], y[live])[0, 1]) if live.sum() > 100 else np.nan,
        "n_live": int(live.sum()), "dir_acc": acc, "nw_t": t,
        "mse": float(err.mean()), "ls_ann": ann, "ls_ir": ir,
        "beta_train": b}, err


def dm_test(e1: np.ndarray, e2: np.ndarray, lags: int) -> float:
    return newey_west_t(e1 - e2, lags)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--horizons", default="1,5")
    ap.add_argument("--target", default="ret_adj",
                    choices=["ret_adj", "lrv_innov"])
    args = ap.parse_args()

    pan = pd.read_parquet(os.path.join(args.indir, "panel.parquet"))
    pan = pan.sort_values(["symbol", "session"]).reset_index(drop=True)
    tag = "" if args.target == "ret_adj" else "_" + args.target
    kern = json.load(open(os.path.join(args.outdir,
                                       f"kernel{tag}.json")))
    K = args.K

    X = lag_matrix(pan, "m1_z", K)
    N = lag_matrix(pan, "n_news", K)
    filters = build_filters(K, kern)

    all_rows = []
    for hz in [int(x) for x in args.horizons.split(",")]:
        y = forward_target(pan, hz, args.target)
        fin = np.isfinite(X).all(1) & np.isfinite(y) & np.isfinite(N).all(1)
        tr = fin & (pan["session"] <= TRAIN_END).to_numpy()
        te = fin & (pan["session"] > TRAIN_END).to_numpy()
        print(f"horizon {hz}: train {tr.sum():,}  test {te.sum():,}")

        feats = {nm: X @ h for nm, h in filters.items()}
        for W in (2, 3, 5, 10):
            feats[f"cwin-{W}"] = count_weighted(X, N, W)

        errs, rows = {}, []
        for nm, f in feats.items():
            r, e = evaluate(nm, f, y, pan, tr, te, 5 * hz, hz)
            r["horizon"] = hz
            r["target"] = args.target
            rows.append(r)
            errs[nm] = e
        fixed = [r for r in rows
                 if r["filter"].startswith(("uniform", "cwin", "exp"))
                 or r["filter"] == "latest"]
        # honest baseline: the window a practitioner would pick, chosen on
        # the training window alone
        base = max(fixed, key=lambda r: (abs(r["ic_train"])
                                         if np.isfinite(r["ic_train"]) else -9))
        # transparency: the window that turns out best on the test window,
        # which no practitioner could have selected in advance
        oracle = max(fixed, key=lambda r: (r["ic"] if np.isfinite(r["ic"])
                                           else -9))
        for r in rows:
            r["dm_vs_base"] = (
                np.nan if r["filter"] == base["filter"]
                else dm_test(errs[base["filter"]], errs[r["filter"]], 5 * hz))
            r["dm_vs_oracle"] = (
                np.nan if r["filter"] == oracle["filter"]
                else dm_test(errs[oracle["filter"]], errs[r["filter"]], 5 * hz))
            r["base_fixed_train"] = base["filter"]
            r["oracle_fixed_test"] = oracle["filter"]
        all_rows += rows
        for r in sorted(rows, key=lambda r: -(r["ic"] if np.isfinite(r["ic"])
                                              else -9)):
            mark = ("<-train-pick" if r["filter"] == base["filter"] else
                    ("<-oracle" if r["filter"] == oracle["filter"] else ""))
            print(f"  {r['filter']:<12} ICtr={r['ic_train']:+.4f} "
                  f"IC={r['ic']:+.4f} acc={r['dir_acc']:.4f}"
                  f" t={r['nw_t']:+.2f} IR={r['ls_ir']:+.2f} "
                  f"DMbase={r['dm_vs_base']:+.2f} {mark}")

    out = pd.DataFrame(all_rows)
    out.to_csv(os.path.join(args.outdir,
                            f"filter_comparison{tag}.csv"), index=False)
    print("wrote filter_comparison.csv")


if __name__ == "__main__":
    main()
