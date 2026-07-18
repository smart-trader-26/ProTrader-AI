"""Estimate calibration drift (gamma, beta) from the live prediction ledger.

Design: in ProTrader the hybrid model is re-fit at `made_at`, so a forecast
whose target_date is k days later is evaluated at model staleness t = k days.
Interval calibration is measured as coverage of the nominal-90% conformal band
by staleness bucket; the gap g(t) = 0.9 - coverage(t) is the interval
miscalibration analogue of ECE. We fit g(t) = gamma * t^beta by weighted
least squares and also report directional hit-rate decay and the
probabilistic (prob_up) subset where available.

Outputs: results_ledger.json + coverage_by_age.csv (for figures).
"""
import json
import sqlite3
from datetime import datetime, date
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy import stats

HERE = Path(__file__).parent
DB = r"c:\Users\divya\Desktop\finance\data\ledger\predictions.sqlite"

con = sqlite3.connect(DB)
rows = con.execute(
    """SELECT ticker, made_at, target_date, pred_dir, pred_price, anchor_price,
              ci_low, ci_high, confidence_level, prob_up, horizon_days,
              actual_price, hit
       FROM predictions WHERE actual_price IS NOT NULL"""
).fetchall()
print(f"resolved rows: {len(rows)}")

recs = []
for (tk, made_at, tgt, pdir, ppx, anchor, lo, hi, conf, pup, hz, actual, hit) in rows:
    made = datetime.fromisoformat(made_at).date()
    tgt_d = date.fromisoformat(tgt)
    age = (tgt_d - made).days
    if age <= 0 or lo is None or hi is None or anchor is None:
        continue
    covered = 1 if (lo <= actual <= hi) else 0
    recs.append(dict(ticker=tk, age=age, covered=covered, conf=conf,
                     pup=pup, hit=hit, pdir=pdir,
                     outcome_up=1 if actual > anchor else 0,
                     pred_price=ppx, actual=actual, anchor=anchor))

ages = np.array([r["age"] for r in recs])
cov = np.array([r["covered"] for r in recs])
conf_nom = np.array([r["conf"] for r in recs])
print("nominal confidence levels present:", sorted(set(conf_nom.tolist())))
print("age range:", ages.min(), "-", ages.max())
print("tickers:", sorted({r['ticker'] for r in recs}))

# ---- coverage by staleness bucket (calendar-day ages; NSE has ~5/7 trading days)
buckets = {}
for r in recs:
    a = r["age"]
    # group: 1..14 individually, 15+ into swing bucket
    key = a if a <= 14 else 99
    buckets.setdefault(key, []).append(r["covered"])

print("\nage  n    coverage   gap(0.9-cov)  se")
tab = []
for k in sorted(buckets):
    arr = np.array(buckets[k])
    n, c = len(arr), arr.mean()
    se = float(np.sqrt(max(c * (1 - c), 1e-9) / n))
    gap = 0.90 - c
    tab.append(dict(age=int(k), n=int(n), coverage=float(c), gap=float(gap), se=se))
    print(f"{k:3d} {n:4d}   {c:.3f}      {gap:+.3f}       {se:.3f}")

# ---- fit gap(t) = gamma * t^beta on buckets with enough mass (exclude swing pool)
fit_tab = [b for b in tab if b["age"] <= 14 and b["n"] >= 25]
t_fit = np.array([b["age"] for b in fit_tab], dtype=float)
g_fit = np.array([b["gap"] for b in fit_tab])
w = np.array([1.0 / max(b["se"], 1e-3) for b in fit_tab])


def powerlaw(t, gamma, beta):
    return gamma * np.power(t, beta)


fit = {}
try:
    p0 = (0.01, 1.0)
    popt, pcov = curve_fit(powerlaw, t_fit, g_fit, p0=p0,
                           sigma=1.0 / w, absolute_sigma=False, maxfev=20000)
    perr = np.sqrt(np.diag(pcov))
    fit["gamma"], fit["beta"] = float(popt[0]), float(popt[1])
    fit["gamma_se"], fit["beta_se"] = float(perr[0]), float(perr[1])
    print(f"\npower-law fit: gap(t) = {popt[0]:.4f} * t^{popt[1]:.3f}"
          f"  (se gamma {perr[0]:.4f}, se beta {perr[1]:.3f})")
except Exception as e:  # pragma: no cover
    print("power-law fit failed:", e)

# linear fit for reference (beta fixed at 1)
lin = stats.linregress(t_fit, g_fit)
print(f"linear fit:   gap(t) = {lin.intercept:+.4f} + {lin.slope:.4f} * t"
      f"  (slope se {lin.stderr:.4f}, r={lin.rvalue:.3f}, p={lin.pvalue:.4f})")

# sqrt fit (beta fixed at 0.5)
A = np.sqrt(t_fit)
slope_sqrt = float(np.sum(w * A * g_fit) / np.sum(w * A * A))
resid_sqrt = g_fit - slope_sqrt * A
resid_lin = g_fit - (lin.intercept + lin.slope * t_fit)
print(f"sqrt fit:     gap(t) = {slope_sqrt:.4f} * sqrt(t)"
      f"   SSE sqrt={np.sum(w*resid_sqrt**2):.4f} vs lin={np.sum(w*resid_lin**2):.4f}")

# ---- directional hit-rate by staleness (hit excludes 'flat' predictions)
dir_tab = []
hbuckets = {}
for r in recs:
    if r["hit"] is None:
        continue
    key = r["age"] if r["age"] <= 14 else 99
    hbuckets.setdefault(key, []).append(r["hit"])
print("\nage  n    hit-rate")
for k in sorted(hbuckets):
    arr = np.array(hbuckets[k])
    if len(arr) >= 10:
        dir_tab.append(dict(age=int(k), n=int(len(arr)), hit=float(arr.mean())))
        print(f"{k:3d} {len(arr):4d}   {arr.mean():.3f}")

hit_t = np.array([d["age"] for d in dir_tab if d["age"] <= 14], dtype=float)
hit_v = np.array([d["hit"] for d in dir_tab if d["age"] <= 14])
hit_lin = stats.linregress(hit_t, hit_v)
print(f"hit-rate decay: {hit_lin.intercept:.3f} {hit_lin.slope:+.4f}/day"
      f" (se {hit_lin.stderr:.4f}, p={hit_lin.pvalue:.4f})")

# ---- probabilistic subset (prob_up present + resolved)
pp = [r for r in recs if r["pup"] is not None]
print(f"\nprob_up resolved subset: n={len(pp)}")
prob_sub = {"n": len(pp)}
if len(pp) >= 20:
    p = np.array([r["pup"] for r in pp])
    y = np.array([r["outcome_up"] for r in pp])
    brier = float(np.mean((p - y) ** 2))
    # single-bin ECE given tight clustering
    ece = float(abs(p.mean() - y.mean()))
    prob_sub.update(brier=brier, ece_pooled=ece,
                    p_mean=float(p.mean()), base=float(y.mean()))
    print(f"  brier={brier:.4f}  pooled |p-y| gap={ece:.4f}")

# ---- point-forecast error growth (RMSE of pred vs actual, % of anchor) by age
err_tab = []
ebuckets = {}
for r in recs:
    key = r["age"] if r["age"] <= 14 else 99
    ebuckets.setdefault(key, []).append((r["pred_price"] - r["actual"]) / r["anchor"])
for k in sorted(ebuckets):
    arr = np.array(ebuckets[k])
    if len(arr) >= 25:
        err_tab.append(dict(age=int(k), n=int(len(arr)),
                            rmse_pct=float(np.sqrt(np.mean(arr ** 2)) * 100)))

out = dict(n_resolved=len(recs), coverage_table=tab, fit_powerlaw=fit,
           fit_linear=dict(slope=float(lin.slope), intercept=float(lin.intercept),
                           slope_se=float(lin.stderr),
                           r=float(lin.rvalue), p=float(lin.pvalue)),
           fit_sqrt=dict(slope=slope_sqrt),
           hit_decay=dict(slope=float(hit_lin.slope), intercept=float(hit_lin.intercept),
                          slope_se=float(hit_lin.stderr), p=float(hit_lin.pvalue)),
           dir_table=dir_tab, prob_subset=prob_sub, err_table=err_tab,
           tickers=sorted({r['ticker'] for r in recs}))
(HERE / "results_ledger.json").write_text(json.dumps(out, indent=2))

import csv
with open(HERE / "coverage_by_age.csv", "w", newline="") as f:
    wcsv = csv.DictWriter(f, fieldnames=["age", "n", "coverage", "gap", "se"])
    wcsv.writeheader()
    wcsv.writerows(tab)
print("\nwrote results_ledger.json + coverage_by_age.csv")
