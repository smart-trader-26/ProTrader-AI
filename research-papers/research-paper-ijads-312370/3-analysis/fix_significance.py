"""Recompute the inference block using the same Sharpe definition as every table
(CAGR divided by annualised volatility), so one number never appears in two forms.
"""
import json, os, pickle
import numpy as np
import pandas as pd
from scipy import stats
import pipeline_core as P

OUT = os.path.dirname(os.path.abspath(__file__))
MULTI = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'IEF', 'LQD', 'GLD', 'VNQ']
P.START, P.END = '2007-01-01', '2026-08-27'
HOLDOUT = '2024-01-01'
P._CACHE.clear()
Z = pickle.load(open(os.path.join(OUT, '_final_cache_ext.pkl'), 'rb'))
strat = Z['strat'][0]
idx = strat.index
ew = P.benchmarks(MULTI, idx)['ew']


def gsharpe(x):
    x = pd.Series(np.asarray(x))
    cum = float((1 + x).prod())
    if cum <= 0:
        return float('nan')
    ann = cum ** (252 / len(x)) - 1
    vol = float(x.std() * np.sqrt(252))
    return ann / vol if vol else 0.0


def nw_t(x, lags=10):
    x = np.asarray(x, float); n = len(x); m = x.mean(); e = x - m
    s = (e @ e) / n
    for l in range(1, lags + 1):
        s += 2 * (1 - l / (lags + 1)) * ((e[l:] @ e[:-l]) / n)
    return float(m / np.sqrt(s / n))


rng = np.random.default_rng(42)
x, y = np.asarray(strat), np.asarray(ew)
n, blk = len(x), 21
nb = int(np.ceil(n / blk))
ds, de, dd = [], [], []
for _ in range(2000):
    st = rng.integers(0, n, nb)
    ii = np.concatenate([np.arange(s, s + blk) % n for s in st])[:n]
    a, b = gsharpe(x[ii]), gsharpe(y[ii])
    if np.isnan(a) or np.isnan(b):
        continue
    ds.append(a); de.append(b); dd.append(a - b)
q = lambda a: [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]

ex = (strat - ew).dropna()
t_, p_ = stats.ttest_1samp(ex, 0)
dn, up = ew < 0, ew > 0
tail = ew.nsmallest(int(0.05 * len(ew)))
sig = dict(
    excess_ann=round(float(ex.mean() * 252), 5), t=round(float(t_), 3), p=round(float(p_), 4),
    nw_t=round(nw_t(ex), 3),
    strat_sharpe=round(gsharpe(strat), 4), ew_sharpe=round(gsharpe(ew), 4),
    bootstrap=dict(n_valid=len(dd), strat_ci=q(ds), ew_ci=q(de), diff_ci=q(dd),
                   diff_median=round(float(np.median(dd)), 3),
                   p_diff_le_0=round(float((np.array(dd) <= 0).mean()), 4)),
    capture=dict(downside=round(float(strat[dn].mean() / ew[dn].mean()), 4),
                 upside=round(float(strat[up].mean() / ew[up].mean()), 4)),
    worst_5pct_days=dict(n=len(tail), strategy=round(float(strat[tail.index].mean()), 5),
                         equal_weight=round(float(tail.mean()), 5),
                         cushion=round(float(strat[tail.index].mean() - tail.mean()), 5)))

hi = idx[idx >= HOLDOUT]
xh, yh = np.asarray(strat.reindex(hi)), np.asarray(ew.reindex(hi))
nh, nbh = len(xh), int(np.ceil(len(xh) / blk))
dsh, deh, ddh = [], [], []
rng2 = np.random.default_rng(42)
for _ in range(2000):
    st = rng2.integers(0, nh, nbh)
    ii = np.concatenate([np.arange(s_, s_ + blk) % nh for s_ in st])[:nh]
    a, b = gsharpe(xh[ii]), gsharpe(yh[ii])
    if np.isnan(a) or np.isnan(b):
        continue
    dsh.append(a); deh.append(b); ddh.append(a - b)
exh = (strat.reindex(hi) - ew.reindex(hi)).dropna()
th, ph = stats.ttest_1samp(exh, 0)
sig_h = dict(excess_ann=round(float(exh.mean() * 252), 5), t=round(float(th), 3),
             p=round(float(ph), 4), nw_t=round(nw_t(exh), 3),
             strat_sharpe=round(gsharpe(strat.reindex(hi)), 4),
             ew_sharpe=round(gsharpe(ew.reindex(hi)), 4),
             bootstrap=dict(n_valid=len(ddh), strat_ci=q(dsh), ew_ci=q(deh), diff_ci=q(ddh),
                            diff_median=round(float(np.median(ddh)), 3),
                            p_diff_le_0=round(float((np.array(ddh) <= 0).mean()), 4)))

p = os.path.join(OUT, 'final_tables.json')
R = json.load(open(p))
R['T9_significance'] = sig
R['T10_holdout']['significance'] = sig_h
print('holdout inference', json.dumps(sig_h, indent=1))
json.dump(R, open(p, 'w'), indent=2, default=str)
print(json.dumps(sig, indent=1))
print('\nfinal_tables.json updated')
