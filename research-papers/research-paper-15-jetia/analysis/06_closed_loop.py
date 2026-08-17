"""Step 6: closed-loop (LMS) adaptation of the aggregation filter.

The response kernel and the scorer reliability both drift, so the filter is
run as a feedback loop over the evaluation window,

        e(t) = y(t) - w(t)' x(t)
        w(t+1) = w(t) + mu e(t) x(t)

with x(t) the measured sentiment lag vector.  Because the loop sees the
noisy measurement, its input power is tr(R_xx) = tr(R_SS)/lambda, so

  * the mean-square stability bound  0 < mu < 2/tr(R_xx)  tightens by
    the factor lambda when the input carries measurement noise, and
  * the gain that minimises steady-state excess error under drift is
        mu* = sqrt( tr(Q) / (sigma^2 tr(R_xx)) ),
    which scales as sqrt(lambda): a noisier scorer demands a slower loop.

This script measures lambda, tr(Q) and sigma^2 on the training window,
predicts mu*, then sweeps mu out of sample to test the prediction, and
compares the closed loop against the fixed open-loop filter.
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


def lms(X: np.ndarray, y: np.ndarray, w0: np.ndarray, mu: float,
        sess_id: np.ndarray, delay: int = 1, normalized: bool = False):
    """Run the loop with causally delayed updates.

    The label attached to session s is the return realised between the
    closes of sessions s and s+delay, so it is unknown until the close of
    session s+delay.  Rows of session s are therefore predicted with weights
    that have absorbed observations up to session s-delay only.  Updating
    inside the current session would feed a not-yet-observable return back
    into predictions for the other symbols of that same session.
    """
    w = w0.astype(np.float64).copy()
    n = X.shape[0]
    pred = np.full(n, np.nan)
    wnorm = np.full(n, np.nan)
    order = np.argsort(sess_id, kind="stable")
    bounds = np.flatnonzero(np.diff(sess_id[order])) + 1
    groups = np.split(order, bounds)

    for gi, g in enumerate(groups):
        for t in g:                       # predict with the current weights
            pred[t] = float(w @ X[t])
        j = gi - delay                    # now-observable session
        if j >= 0:
            for t in groups[j]:
                e = y[t] - float(w @ X[t])
                step = mu / (1e-8 + float(X[t] @ X[t])) if normalized else mu
                w = w + step * e * X[t]
            nrm = float(w @ w)
            if not np.isfinite(nrm) or nrm > 1e12:
                for gg in groups[gi + 1:]:
                    pred[gg] = np.nan
                return pred, w, wnorm, False
        wnorm[g] = float(w @ w)
    return pred, w, wnorm, True


def yearly_kernels(pan, X, y, fin, K):
    """Batch kernel per calendar year, used to measure drift tr(Q)."""
    out = {}
    yr = pd.DatetimeIndex(pan["session"]).year.to_numpy()
    for u in np.unique(yr[fin]):
        m = fin & (yr == u)
        if m.sum() < 5000:
            continue
        A, b = X[m], y[m]
        R = A.T @ A / len(b)
        c = A.T @ b / len(b)
        try:
            out[int(u)] = np.linalg.solve(R + 1e-9 * np.eye(K), c)
        except np.linalg.LinAlgError:
            pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--K", type=int, default=12)
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
    y = forward_target(pan, 1, args.target)
    sess_all = pd.factorize(pan["session"], sort=True)[0]

    fin = np.isfinite(X).all(1) & np.isfinite(y)
    tr = fin & (pan["session"] <= TRAIN_END).to_numpy()
    te = fin & (pan["session"] > TRAIN_END).to_numpy()

    R_xx = X[tr].T @ X[tr] / tr.sum()
    R_SS = np.load(os.path.join(args.outdir, f"R_SS{tag}.npy"))
    tr_Rxx = float(np.trace(R_xx))
    tr_RSS = float(np.trace(R_SS))
    lam = tr_RSS / tr_Rxx
    sigma2 = float(np.var(y[tr]))

    ky = yearly_kernels(pan, X, y, tr, K)
    years = sorted(ky)
    diffs = [ky[b] - ky[a] for a, b in zip(years, years[1:])]
    per_year_Q = float(np.mean([d @ d for d in diffs])) if diffs else np.nan
    steps_per_year = tr.sum() / max(len(years), 1)
    trQ = per_year_Q / steps_per_year

    mu_max = 2.0 / tr_Rxx
    mu_star = float(np.sqrt(trQ / (sigma2 * tr_Rxx)))

    print(f"tr(R_xx)={tr_Rxx:.4f}  tr(R_SS)={tr_RSS:.4f}  lambda={lam:.4f}")
    print(f"sigma^2={sigma2:.3e}  tr(Q)/step={trQ:.3e}")
    print(f"stability bound mu_max={mu_max:.4f}   predicted mu*={mu_star:.5f}")

    w_open = np.array(kern["deconvolved"])
    Xte, yte, ste = X[te], y[te], sess_all[te]

    def run(mu: float, Xa, ya, sa):
        pred, _, wn, ok = lms(Xa, ya, w_open, mu, sa)
        m = np.isfinite(pred)
        if m.sum() < 1000 or np.std(pred[m]) == 0:
            return None
        return {"mse": float(np.mean((ya[m] - pred[m]) ** 2)),
                "ic": float(np.corrcoef(pred[m], ya[m])[0, 1]),
                "stable": bool(ok and mu < mu_max),
                "wnorm_end": float(wn[m][-1]) if np.isfinite(wn[m]).any()
                else np.nan}

    # headline number: the gain is fixed from training quantities only
    head = run(mu_star, Xte, yte, ste)

    grid = sorted({float(f"{g:.8g}") for g in
                   np.concatenate([np.geomspace(mu_max * 1e-4, mu_max * 4, 25),
                                   [mu_star]])})
    rows = []
    for mu in grid:
        r = run(mu, Xte, yte, ste)
        rows.append({"mu": mu, **(r or {"mse": np.nan, "ic": np.nan,
                                        "stable": False})})

    df = pd.DataFrame(rows)
    ok = df[df["stable"] & df["mse"].notna()]
    mu_hat = float(ok.loc[ok["mse"].idxmin(), "mu"]) if len(ok) else np.nan

    # open loop reference on the same window
    p_open = Xte @ w_open
    mse_open = float(np.mean((yte - p_open) ** 2))
    ic_open = float(np.corrcoef(p_open, yte)[0, 1])

    # sqrt(lambda) scaling test across news-intensity strata
    strata = []
    nn = pan["n_news"].to_numpy()
    for lo, hi in [(1, 1), (2, 3), (4, 8), (9, 10 ** 6)]:
        sel = te & (nn >= lo) & (nn <= hi)
        if sel.sum() < 4000:
            continue
        Rs = X[sel].T @ X[sel] / sel.sum()
        a = pan["m1_z"].to_numpy()[sel]
        b = pan["m2_z"].to_numpy()[sel]
        lam_s = float(np.cov(a, b)[0, 1] / np.var(a))
        best, bmse = np.nan, np.inf
        for mu in np.geomspace(2 / np.trace(Rs) * 1e-4, 2 / np.trace(Rs), 20):
            r = run(mu, X[sel], y[sel], sess_all[sel])
            if r and r["mse"] < bmse:
                bmse, best = r["mse"], mu
        strata.append({"n_lo": lo, "n_hi": hi, "n_obs": int(sel.sum()),
                       "lambda": lam_s, "tr_R": float(np.trace(Rs)),
                       "mu_emp": best,
                       "mu_pred": float(np.sqrt(trQ / (sigma2 * np.trace(Rs))))})

    res = {"tr_Rxx": tr_Rxx, "tr_RSS": tr_RSS, "lambda": lam,
           "sigma2": sigma2, "trQ_per_step": trQ, "mu_max": mu_max,
           "mu_star_pred": mu_star, "mu_hat_emp": mu_hat,
           "mse_open": mse_open, "ic_open": ic_open,
           "mse_at_mu_star": head["mse"] if head else np.nan,
           "ic_at_mu_star": head["ic"] if head else np.nan,
           "mse_best_closed": float(ok["mse"].min()) if len(ok) else np.nan,
           "ic_best_closed": float(ok.loc[ok["mse"].idxmin(), "ic"])
           if len(ok) else np.nan,
           "years": years, "strata": strata}
    with open(os.path.join(args.outdir, f"closed_loop{tag}.json"),
              "w") as f:
        json.dump(res, f, indent=1)
    df.to_csv(os.path.join(args.outdir, f"loop_gain_sweep{tag}.csv"),
              index=False)
    print(json.dumps(res, indent=1)[:2000])


if __name__ == "__main__":
    main()
