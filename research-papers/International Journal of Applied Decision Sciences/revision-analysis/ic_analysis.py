"""Does signal effectiveness actually vary by regime?

This tests the paper's central thesis directly at the signal level, independent of any
portfolio construction: cross-sectional information coefficient of each signal family
against forward returns, split by out-of-sample regime.
"""
import json, os
import numpy as np
import pandas as pd
from scipy import stats
import pipeline_core as P

OUT = os.path.dirname(os.path.abspath(__file__))
RES = {}
H = 21          # forward horizon = rebalance horizon


def nw_t(x, lags=21):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x); m = x.mean(); e = x - m
    s = (e @ e) / n
    for l in range(1, min(lags, n - 1) + 1):
        s += 2 * (1 - l / (lags + 1)) * ((e[l:] @ e[:-l]) / n)
    return float(m / np.sqrt(s / n)) if s > 0 else np.nan


def regime_series(rf, dates, n_regimes=3, seed=42, step=21):
    """expanding-window regime labels on a monthly cadence, forward filled -- no look-ahead"""
    pts = dates[::step]
    lab = {}
    for d in pts:
        lab[d] = P.regime_at(rf, d, n_regimes, seed)
    s = pd.Series(lab).reindex(dates).ffill()
    return s


def run(label, univ, start, end='2026-08-27', holdout='2024-01-01'):
    P.START, P.END = start, end
    P._CACHE.clear()
    D = P.load(univ)
    returns, tech, sent = D['returns'], D['tech'], D['sent']
    rf = D['regime_feat']

    fwd = (returns.shift(-H) / 1.0)
    fwd = (1 + returns).rolling(H).apply(np.prod, raw=True).shift(-H) - 1
    idx = tech.index.intersection(fwd.dropna(how='all').index)
    idx = idx[idx >= returns.index[252]]
    tech, sent, fwd = tech.loc[idx], sent.loc[idx], fwd.loc[idx]
    fused = 0.5 * tech + 0.5 * sent

    reg = regime_series(rf, idx)

    def ic_series(sig):
        out = {}
        for d in idx:
            a, b = sig.loc[d], fwd.loc[d]
            m = a.notna() & b.notna()
            if m.sum() >= 5:
                out[d] = stats.spearmanr(a[m], b[m]).correlation
        return pd.Series(out)

    ic_t, ic_s, ic_f = ic_series(tech), ic_series(sent), ic_series(fused)

    print("\n" + "=" * 78)
    print(f"{label}   {idx[0].date()} -> {idx[-1].date()}   horizon {H}d")
    print("=" * 78)
    print(f"{'signal':10s} {'mean IC':>9s} {'NW t':>7s} {'nonoverlap t':>13s} {'hit rate':>9s}")
    for nm, s in [('technical', ic_t), ('sentiment', ic_s), ('fused', ic_f)]:
        no = s.iloc[::H].dropna()
        t_no = stats.ttest_1samp(no, 0).statistic if len(no) > 3 else np.nan
        print(f"{nm:10s} {s.mean():+9.4f} {nw_t(s):7.2f} {t_no:13.2f} {(s > 0).mean():9.3f}")

    print("\nregime-conditional IC  (the paper's core claim: effectiveness varies by regime)")
    print(f"{'regime':>7s} {'n':>6s} {'tech IC':>9s} {'sent IC':>9s} {'tech-sent':>10s} "
          f"{'paper expects':>16s}")
    expect = {0: 'technical higher', 1: 'balanced', 2: 'sentiment higher'}
    rows = []
    for g in [0, 1, 2]:
        m = reg.reindex(ic_t.index) == g
        if m.sum() < 30:
            continue
        a, b = ic_t[m], ic_s[m]
        rows.append(dict(regime=int(g), n=int(m.sum()), tech_ic=float(a.mean()),
                         sent_ic=float(b.mean()), diff=float(a.mean() - b.mean()),
                         tech_nw_t=nw_t(a), sent_nw_t=nw_t(b)))
        print(f"{g:7d} {int(m.sum()):6d} {a.mean():+9.4f} {b.mean():+9.4f} "
              f"{a.mean() - b.mean():+10.4f} {expect[g]:>16s}")

    # does the tech-minus-sent edge actually differ between calm and stressed?
    verdict = None
    if len(rows) >= 2:
        r0 = [r for r in rows if r['regime'] == 0]
        r2 = [r for r in rows if r['regime'] == 2]
        if r0 and r2:
            m0 = reg.reindex(ic_t.index) == 0
            m2 = reg.reindex(ic_t.index) == 2
            d0 = (ic_t - ic_s)[m0].dropna()
            d2 = (ic_t - ic_s)[m2].dropna()
            tt = stats.ttest_ind(d0, d2, equal_var=False)
            verdict = dict(calm_diff=float(d0.mean()), stress_diff=float(d2.mean()),
                           t=float(tt.statistic), p=float(tt.pvalue),
                           direction_as_paper_predicts=bool(d0.mean() > d2.mean()))
            print(f"\n  (tech-sent) edge in calm {d0.mean():+.4f} vs stressed {d2.mean():+.4f}"
                  f"   t {tt.statistic:.2f}  p {tt.pvalue:.4f}")
            print(f"  direction the paper predicts (tech better in calm, sentiment better in "
                  f"stress): {'YES' if verdict['direction_as_paper_predicts'] else 'NO'}")

    def ov(a, b, c):
        return {'technical': dict(ic=float(a.mean()), nw_t=nw_t(a)),
                'sentiment': dict(ic=float(b.mean()), nw_t=nw_t(b)),
                'fused': dict(ic=float(c.mean()), nw_t=nw_t(c))}

    splits = {}
    for tag, m in [('in_sample', ic_t.index < holdout), ('holdout', ic_t.index >= holdout)]:
        if m.sum() < 60:
            continue
        splits[tag] = dict(ov(ic_t[m], ic_s[m], ic_f[m]),
                           n=int(m.sum()),
                           period=[str(ic_t.index[m][0].date()), str(ic_t.index[m][-1].date())])
    if 'holdout' in splits:
        print('\n  IC by era (does the in-sample sign persist?)')
        for tag in ['in_sample', 'holdout']:
            d = splits[tag]
            print(f"    {tag:<11s} n={d['n']:<5d} fused IC {d['fused']['ic']:+.4f} "
                  f"(t {d['fused']['nw_t']:+.2f})   {d['period'][0]} -> {d['period'][1]}")

    RES[label] = dict(overall=ov(ic_t, ic_s, ic_f),
                      by_regime=rows, regime_contrast=verdict, splits=splits,
                      period=[str(idx[0].date()), str(idx[-1].date())])


run('MULTI-ASSET 2008-2026', P.MULTI if hasattr(P, 'MULTI') else
    ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'IEF', 'LQD', 'GLD', 'VNQ'], '2007-01-01')
run('MEGACAP 2019-2026', P.MEGA, '2018-01-01')
run('SECTORS 2008-2026', ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB'],
    '2007-01-01')

json.dump(RES, open(os.path.join(OUT, 'ic_results.json'), 'w'), indent=2, default=str)
print("\nsaved ic_results.json")
