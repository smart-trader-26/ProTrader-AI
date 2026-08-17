"""Step 4: response-kernel estimation with and without noise correction.

Model.  Let S(t) be the latent sentiment of session t and r(t) the market
adjusted return of the following session.  The predictive response kernel g
satisfies

        r(t+1) = sum_k g(k) S(t-k) + eps(t),   k = 0 .. K-1

The scorers observe m_i(t) = a_i S(t) + e_i(t) with mutually uncorrelated
errors.  Three estimators are compared:

  lag profile   p(k) = cov(m1(t-k), r(t+1)) / var(m1)      (per-lag OLS)
  deconvolved   g_hat = R_mm^{-1} c                        (Wiener-Hopf)
  corrected     g_til = R_SS^{-1} c                        (errors-in-vars)

with c(k) = cov(m1(t-k), r(t+1)), R_mm(j,k) = cov(m1(t-j), m1(t-k)) and
R_SS(j,k) = 0.5[cov(m1(t-j), m2(t-k)) + cov(m2(t-j), m1(t-k))].  The last
identity holds because the two scorer errors are uncorrelated with each
other and with returns, so the cross-covariance of two independent
measurements estimates the latent autocovariance at every lag.

Everything is estimated on the training window only.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

TRAIN_END = "2016-12-31"


def lag_matrix(df: pd.DataFrame, col: str, K: int) -> np.ndarray:
    """Columns k = 0..K-1 hold col shifted forward by k within each symbol."""
    g = df.groupby("symbol")[col]
    return np.column_stack([g.shift(k).to_numpy() for k in range(K)])


def forward_target(df: pd.DataFrame, h: int, col: str) -> np.ndarray:
    """Aggregate of the response over sessions t+1 .. t+h.

    Returns accumulate, so they are summed; the volatility innovation is a
    level, so it is averaged.
    """
    g = df.groupby("symbol")[col]
    s = sum(g.shift(-i) for i in range(1, h + 1))
    return (s / h if col != "ret_adj" else s).to_numpy()


def demean_by_symbol(x: np.ndarray, sym: np.ndarray) -> np.ndarray:
    out = x.astype(np.float64).copy()
    df = pd.DataFrame(out)
    df["s"] = sym
    return (df.groupby("s").transform(lambda v: v - v.mean())).to_numpy()


def smoothness_penalty(K: int) -> np.ndarray:
    """D'D for the second-difference operator D.

    An impulse response of a physical system is smooth in the lag index;
    penalising its curvature is the standard Tikhonov regulariser for FIR
    identification and keeps the estimate from chasing noise at long lags.
    """
    D = np.zeros((max(K - 2, 1), K))
    for i in range(K - 2):
        D[i, i:i + 3] = (1.0, -2.0, 1.0)
    return D.T @ D


def estimate(X1: np.ndarray, X2: np.ndarray, y: np.ndarray,
             alpha: float = 0.0) -> dict:
    """Return the three kernel estimates from centred design matrices."""
    n = len(y)
    c = X1.T @ y / n
    R_mm = X1.T @ X1 / n
    C12 = X1.T @ X2 / n
    R_SS = 0.5 * (C12 + C12.T)

    K = X1.shape[1]
    P = smoothness_penalty(K) + 1e-6 * np.eye(K)
    p = c / np.diag(R_mm)                       # per-lag univariate slope
    g_hat = np.linalg.solve(R_mm + alpha * P, c)
    g_til = np.linalg.solve(R_SS + alpha * P, c)
    return {"lag_profile": p, "deconvolved": g_hat, "corrected": g_til,
            "c": c, "R_mm": R_mm, "R_SS": R_SS}


def select_alpha(X1, X2, y, sess, grid, n_folds: int = 5) -> dict:
    """Pick the penalty by forward-chaining validation inside training.

    Folds are contiguous in time and always validate on a period after the
    one used to fit, so the penalty is never chosen with future data.
    """
    edges = np.quantile(sess, np.linspace(0, 1, n_folds + 2))
    best = {}
    for key in ("deconvolved", "corrected"):
        scores = []
        for a in grid:
            errs = []
            for i in range(1, n_folds + 1):
                fit = sess <= edges[i]
                val = (sess > edges[i]) & (sess <= edges[i + 1])
                if fit.sum() < 5000 or val.sum() < 2000:
                    continue
                g = estimate(X1[fit], X2[fit], y[fit], a)[key]
                pv = X1[val] @ g
                if np.std(pv) < 1e-15:
                    continue
                b = float(np.polyfit(pv, y[val], 1)[0])
                errs.append(float(np.mean((y[val] - b * pv) ** 2)))
            scores.append(np.mean(errs) if errs else np.inf)
        best[key] = float(grid[int(np.argmin(scores))])
        best[key + "_cv"] = [float(s) for s in scores]
    return best


def reliability(cols: dict[str, np.ndarray]) -> dict:
    """Three-indicator reliability of each scorer, over all triples."""
    names = list(cols)
    V = {a: float(np.var(cols[a])) for a in names}
    C = {(a, b): float(np.cov(cols[a], cols[b])[0, 1])
         for a in names for b in names if a != b}
    out = {}
    for a in names:
        vals = []
        for b in names:
            for d in names:
                if len({a, b, d}) < 3:
                    continue
                den = V[a] * C[(b, d)]
                if abs(den) < 1e-12:
                    continue
                lam = C[(a, b)] * C[(a, d)] / den
                if 0 < lam <= 1.5:
                    vals.append(lam)
        if vals:
            out[a] = {"lambda_median": float(np.median(vals)),
                      "lambda_min": float(np.min(vals)),
                      "lambda_max": float(np.max(vals)),
                      "n_triples": len(vals)}
    out["_corr"] = {f"{a}|{b}": round(C[(a, b)] / np.sqrt(V[a] * V[b]), 4)
                    for a in names for b in names if a < b}
    return out


def block_bootstrap(df: pd.DataFrame, X1, X2, y, K, n_boot, seed):
    """Resample whole symbols (cluster bootstrap) for kernel uncertainty."""
    rng = np.random.default_rng(seed)
    syms = df["symbol"].to_numpy()
    uniq = np.unique(syms)
    idx_by_sym = {s: np.flatnonzero(syms == s) for s in uniq}
    boots = {"deconvolved": [], "corrected": []}
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_sym[s] for s in pick])
        est = estimate(X1[idx], X2[idx], y[idx])
        boots["deconvolved"].append(est["deconvolved"])
        boots["corrected"].append(est["corrected"])
    return {k: np.array(v) for k, v in boots.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260725)
    ap.add_argument("--target", default="ret_adj",
                    choices=["ret_adj", "lrv_innov"])
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    pan = pd.read_parquet(os.path.join(args.indir, "panel.parquet"))
    pan = pan.sort_values(["symbol", "session"]).reset_index(drop=True)

    K = args.K
    X1_all = lag_matrix(pan, "m1_z", K)
    X2_all = lag_matrix(pan, "m2_z", K)
    y_all = forward_target(pan, 1, args.target)

    tr = (pan["session"] <= TRAIN_END).to_numpy()
    ok = np.isfinite(X1_all).all(1) & np.isfinite(X2_all).all(1) & \
        np.isfinite(y_all)
    m = tr & ok
    sym = pan["symbol"].to_numpy()[m]
    X1 = demean_by_symbol(X1_all[m], sym)
    X2 = demean_by_symbol(X2_all[m], sym)
    y = demean_by_symbol(y_all[m].reshape(-1, 1), sym).ravel()
    print(f"training observations: {len(y):,}  symbols: {len(np.unique(sym))}")

    sess_tr = pd.factorize(pan.loc[m, "session"], sort=True)[0]
    scale = float(np.mean(np.diag(X1.T @ X1 / len(y))))
    grid = np.concatenate([[0.0], np.geomspace(1e-4, 1e2, 13) * scale])
    alph = select_alpha(X1, X2, y, sess_tr, grid)
    print(f"selected penalty: deconvolved={alph['deconvolved']:.4g}  "
          f"corrected={alph['corrected']:.4g}  (scale={scale:.3g})")

    est = estimate(X1, X2, y, alph["corrected"])
    est_d = estimate(X1, X2, y, alph["deconvolved"])
    est["deconvolved"] = est_d["deconvolved"]

    # reliability of the session-level sentiment measures
    cols = {}
    for c, nm in [("m1_z", "finbert"), ("m2_z", "lm"),
                  ("m3_z", "vader"), ("m4_z", "hiv4")]:
        if c in pan.columns:
            v = pan.loc[m, c].to_numpy()
            if np.isfinite(v).all() and np.var(v) > 0:
                cols[nm] = v
    rel = reliability(cols) if len(cols) >= 3 else {}

    # reliability by news intensity: averaging n headlines divides the
    # scorer error variance by n, so lambda must rise with n.
    strata = []
    nn = pan.loc[m, "n_news"].to_numpy()
    for lo, hi in [(1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 10**6)]:
        sel = (nn >= lo) & (nn <= hi)
        if sel.sum() < 5000:
            continue
        a = pan.loc[m, "m1_z"].to_numpy()[sel]
        b = pan.loc[m, "m2_z"].to_numpy()[sel]
        va, cab = float(np.var(a)), float(np.cov(a, b)[0, 1])
        strata.append({"n_lo": lo, "n_hi": hi, "n_obs": int(sel.sum()),
                       "var_m1": va, "cov_m1m2": cab,
                       "rel_index": cab / va if va > 0 else np.nan})

    boots = block_bootstrap(pan.loc[m], X1, X2, y, K, args.boot, args.seed)

    res = {
        "K": K,
        "target": args.target,
        "alpha": alph,
        "n_obs": int(len(y)),
        "n_symbols": int(len(np.unique(sym))),
        "train_end": TRAIN_END,
        "lag_profile": est["lag_profile"].tolist(),
        "deconvolved": est["deconvolved"].tolist(),
        "corrected": est["corrected"].tolist(),
        "c": est["c"].tolist(),
        "R_mm_diag": np.diag(est["R_mm"]).tolist(),
        "R_SS_diag": np.diag(est["R_SS"]).tolist(),
        "R_mm_row0": est["R_mm"][0].tolist(),
        "R_SS_row0": est["R_SS"][0].tolist(),
        "reliability": rel,
        "reliability_by_news": strata,
        "boot_corrected_lo": np.percentile(boots["corrected"], 2.5, 0).tolist(),
        "boot_corrected_hi": np.percentile(boots["corrected"], 97.5, 0).tolist(),
        "boot_deconv_lo": np.percentile(boots["deconvolved"], 2.5, 0).tolist(),
        "boot_deconv_hi": np.percentile(boots["deconvolved"], 97.5, 0).tolist(),
    }
    # R_SS is estimated from a cross-covariance rather than from a sum of
    # squares, so unlike R_mm it is not positive definite by construction.
    # Report its spectrum: a negative eigenvalue would mean the identifying
    # assumption is violated badly enough to matter.
    ev_ss = np.linalg.eigvalsh(est["R_SS"])
    ev_mm = np.linalg.eigvalsh(est["R_mm"])
    res["eig_R_SS_min"] = float(ev_ss.min())
    res["eig_R_SS_max"] = float(ev_ss.max())
    res["eig_R_mm_min"] = float(ev_mm.min())
    res["R_SS_pos_def"] = bool(ev_ss.min() > 0)
    print(f"eig(R_SS) in [{ev_ss.min():.5g}, {ev_ss.max():.5g}]  "
          f"pos-def={ev_ss.min() > 0}")

    tag = "" if args.target == "ret_adj" else "_" + args.target
    np.save(os.path.join(args.outdir, f"R_mm{tag}.npy"), est["R_mm"])
    np.save(os.path.join(args.outdir, f"R_SS{tag}.npy"), est["R_SS"])
    np.save(os.path.join(args.outdir, f"c_vec{tag}.npy"), est["c"])
    with open(os.path.join(args.outdir, f"kernel{tag}.json"), "w") as f:
        json.dump(res, f, indent=1)

    np.set_printoptions(precision=5, suppress=True)
    print("lag profile :", est["lag_profile"])
    print("deconvolved :", est["deconvolved"])
    print("corrected   :", est["corrected"])
    print("reliability :", json.dumps(rel, indent=1)[:1200])
    print("by news intensity:")
    for s in strata:
        print("  ", s)


if __name__ == "__main__":
    main()
