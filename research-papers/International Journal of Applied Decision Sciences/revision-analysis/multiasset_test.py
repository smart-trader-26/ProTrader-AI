"""Two untried levers: a multi-asset universe (somewhere to rotate to besides cash)
and an extended 2007-2024 history covering 2008, 2011, 2020 and 2022."""
import json, os
import numpy as np
import pandas as pd
from scipy import stats
import pipeline_core as P

OUT = os.path.dirname(os.path.abspath(__file__))
RES = {}

MULTI = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'IEF', 'LQD', 'GLD', 'VNQ']
SECT9 = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB']

EPISODES = {
    'GFC 2008': ('2008-09-01', '2009-03-09'),
    'Euro crisis 2011': ('2011-07-01', '2011-10-03'),
    'COVID 2020': ('2020-02-19', '2020-03-23'),
    '2022 bear': ('2022-01-03', '2022-10-12'),
}


def sh(x):
    return float(x.mean() / x.std() * np.sqrt(252)) if x.std() else 0.0


def mdd(x):
    c = (1 + x).cumprod()
    return float((c / c.cummax() - 1).min())


def nw_t(x, lags=10):
    x = np.asarray(x, float); n = len(x); m = x.mean(); e = x - m
    s = (e @ e) / n
    for l in range(1, lags + 1):
        s += 2 * (1 - l / (lags + 1)) * ((e[l:] @ e[:-l]) / n)
    return float(m / np.sqrt(s / n))


def study(label, univ, start):
    P.START = start
    P._CACHE.clear()
    print("\n" + "=" * 78)
    print(f"{label}   universe n={len(univ)}   from {start}")
    print("=" * 78)
    strat, W, RG = P.backtest(univ)
    no_reg, _, _ = P.backtest(univ, use_regime=False)
    idx = strat.index
    ew = P.benchmarks(univ, idx)['ew']
    ewm = ew * (strat.std() / ew.std())
    no_reg = no_reg.reindex(idx).dropna()

    tbl = pd.DataFrame([P.stats_of(strat, 'regime strategy'),
                        P.stats_of(ew, 'equal-weight'),
                        P.stats_of(ewm, 'equal-weight @ strat vol'),
                        P.stats_of(no_reg, 'no-regime ablation')]).set_index('name')
    print(tbl.round(4).to_string())
    print(f"period {idx[0].date()} -> {idx[-1].date()}  ({len(idx)} days)")

    ex = strat - ew
    t_, p_ = stats.ttest_1samp(ex, 0)
    print(f"\nvs equal-weight: excess {ex.mean() * 252:+.2%}/yr  Sharpe {sh(strat):.3f} vs {sh(ew):.3f}"
          f"  t {t_:.3f} (p {p_:.4f})  NW t {nw_t(ex):.3f}")
    print(f"Sharpe beats EW: {sh(strat) > sh(ew)}   maxDD beats EW: {mdd(strat) > mdd(ew)}"
          f"   maxDD beats vol-matched EW: {mdd(strat) > mdd(ewm)}")

    print("\ncrisis episodes:")
    ep = {}
    for nm, (a, b) in EPISODES.items():
        m = (idx >= a) & (idx <= b)
        if m.sum() < 5:
            continue
        cs, ce = float((1 + strat[m]).prod() - 1), float((1 + ew[m]).prod() - 1)
        ep[nm] = dict(strategy=cs, equal_weight=ce, excess=cs - ce)
        print(f"  {nm:18s} strat {cs:+7.2%}  EW {ce:+7.2%}  excess {cs - ce:+7.2%}  "
              f"{'WIN' if cs > ce else 'loss'}")

    print("\nyear-by-year:")
    yrs = []
    for y in sorted(set(idx.year)):
        m = idx.year == y
        if m.sum() < 60:
            continue
        s_, e_ = strat[m], ew[m]
        yrs.append(dict(year=int(y), strat=float((1 + s_).prod() - 1),
                        ew=float((1 + e_).prod() - 1), s_sh=sh(s_), e_sh=sh(e_),
                        win=bool(sh(s_) > sh(e_)), dd_win=bool(mdd(s_) > mdd(e_))))
    yt = pd.DataFrame(yrs)
    print(yt.round(4).to_string(index=False))
    print(f"  Sharpe wins {yt['win'].sum()}/{len(yt)} years | drawdown wins {yt['dd_win'].sum()}/{len(yt)}")

    RES[label] = dict(main=tbl.round(6).to_dict('index'), episodes=ep,
                      years=yt.round(6).to_dict('records'),
                      sig=dict(excess_ann=float(ex.mean() * 252), t=float(t_), p=float(p_),
                               nw_t=nw_t(ex), strat_sharpe=sh(strat), ew_sharpe=sh(ew),
                               ewm_sharpe=sh(ewm), noreg_sharpe=sh(no_reg),
                               strat_dd=mdd(strat), ew_dd=mdd(ew), ewm_dd=mdd(ewm)),
                      period=[str(idx[0].date()), str(idx[-1].date())])
    return strat, ew


study('MULTI-ASSET 2007-2024', MULTI, '2007-01-01')
study('MULTI-ASSET 2018-2024 (paper window)', MULTI, '2018-01-01')
study('SECTORS 2007-2024', SECT9, '2007-01-01')

# ------------------------------------------------ specification sweep on the winner set
print("\n" + "=" * 78)
print("SPECIFICATION SWEEP -- multi-asset, extended history")
print("=" * 78)
P.START = '2007-01-01'
P._CACHE.clear()
runs = []
for seed in [0, 7, 42]:
    for lb in [126, 252]:
        for rb in [10, 21, 42]:
            s_, _, _ = P.backtest(MULTI, dict(seed=seed, cov_lookback=lb, rebal=rb))
            e_ = P.benchmarks(MULTI, s_.index)['ew']
            em = e_ * (s_.std() / e_.std())
            runs.append(dict(seed=seed, cov_lb=lb, rebal=rb, strat_sharpe=round(sh(s_), 4),
                             ew_sharpe=round(sh(e_), 4), strat_dd=round(mdd(s_), 4),
                             ew_dd=round(mdd(e_), 4),
                             sharpe_win=bool(sh(s_) > sh(e_)), dd_win=bool(mdd(s_) > mdd(e_)),
                             dd_win_matched=bool(mdd(s_) > mdd(em))))
            print(f"  seed {seed:<3} cov {lb:<4} rebal {rb:<3} -> Sharpe {sh(s_):.3f} vs {sh(e_):.3f}"
                  f" | maxDD {mdd(s_):+.3f} vs {mdd(e_):+.3f}"
                  f"  {'S-WIN' if sh(s_) > sh(e_) else '     '} {'DD-WIN' if mdd(s_) > mdd(e_) else ''}")
D = pd.DataFrame(runs)
print(f"\n  Sharpe beats equal-weight     : {D['sharpe_win'].sum()} of {len(D)}")
print(f"  Drawdown beats equal-weight   : {D['dd_win'].sum()} of {len(D)}")
print(f"  Drawdown beats vol-matched EW : {D['dd_win_matched'].sum()} of {len(D)}")
RES['sweep'] = dict(runs=D.to_dict('records'), n=len(D),
                    sharpe_wins=int(D['sharpe_win'].sum()), dd_wins=int(D['dd_win'].sum()),
                    dd_wins_matched=int(D['dd_win_matched'].sum()))

json.dump(RES, open(os.path.join(OUT, 'multiasset_results.json'), 'w'), indent=2, default=str)
print("\nsaved multiasset_results.json")
