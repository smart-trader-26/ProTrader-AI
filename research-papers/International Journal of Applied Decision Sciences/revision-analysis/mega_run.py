"""Secondary universe: the ten mega-capitalisation US equities of the submitted study,
2018-2026, under the same corrected specification as the primary results.

Reported as a negative result: the framework's drawdown behaviour survives, its
risk-adjusted return does not.
"""
import json, os
import numpy as np
import pandas as pd
from scipy import stats
import pipeline_core as P

OUT = os.path.dirname(os.path.abspath(__file__))
MEGA = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'XOM']
EPISODES = {'COVID 2020': ('2020-02-19', '2020-03-23'), '2022 bear': ('2022-01-03', '2022-10-12')}
P.START, P.END = '2018-01-01', '2026-08-27'
P._CACHE.clear()


def sh(x):
    return float(x.mean() / x.std() * np.sqrt(252)) if x.std() else 0.0


def mdd(x):
    c = (1 + x).cumprod()
    return float((c / c.cummax() - 1).min())


strat, W, RG = P.backtest(MEGA)
idx = strat.index
ew = P.benchmarks(MEGA, idx)['ew']
ewm = ew * (strat.std() / ew.std())
inv = float(W.sum(axis=1).mean())
noreg, Wn, _ = P.backtest(MEGA, dict(flat_mult=round(inv, 3)), use_regime=False)
noreg = noreg.reindex(idx).dropna()

rows = [P.stats_of(strat, 'Regime-conditional strategy'), P.stats_of(ew, 'Equal-weight'),
        P.stats_of(ewm, 'Equal-weight at strategy volatility'),
        P.stats_of(noreg, 'No regime timing, same average exposure')]
print(pd.DataFrame(rows).set_index('name').round(4).to_string())
print('period %s -> %s (%d days)  mean exposure %.3f' % (idx[0].date(), idx[-1].date(), len(idx), inv))

ex = (strat - ew).dropna()
t_, p_ = stats.ttest_1samp(ex, 0)
R = dict(main={r['name']: {k: round(v, 4) for k, v in r.items() if k != 'name'} for r in rows},
         mean_exposure=round(inv, 4), period=[str(idx[0].date()), str(idx[-1].date()), len(idx)],
         excess_ann=round(float(ex.mean() * 252), 5), t=round(float(t_), 3), p=round(float(p_), 4),
         concentration=dict(
             mean_top_weight=round(float((W.div(W.sum(axis=1), axis=0)).max(axis=1).mean()), 4),
             mean_effective_assets=round(float(
                 (1.0 / (W.div(W.sum(axis=1), axis=0) ** 2).sum(axis=1)).mean()), 2)),
         episodes={}, years=[])
for nm, (a, b) in EPISODES.items():
    m = (idx >= a) & (idx <= b)
    cs, ce = float((1 + strat[m]).prod() - 1), float((1 + ew[m]).prod() - 1)
    R['episodes'][nm] = dict(strategy=round(cs, 4), equal_weight=round(ce, 4),
                             excess=round(cs - ce, 4))
for y in sorted(set(idx.year)):
    m = idx.year == y
    if m.sum() < 60:
        continue
    R['years'].append(dict(year=int(y), strat_sharpe=round(sh(strat[m]), 3),
                           ew_sharpe=round(sh(ew[m]), 3), strat_dd=round(mdd(strat[m]), 3),
                           ew_dd=round(mdd(ew[m]), 3),
                           sharpe_win=bool(sh(strat[m]) > sh(ew[m])),
                           dd_win=bool(mdd(strat[m]) > mdd(ew[m]))))
R['dd_wins'] = sum(r['dd_win'] for r in R['years'])
R['sharpe_wins'] = sum(r['sharpe_win'] for r in R['years'])
print('\nepisodes', json.dumps(R['episodes'], indent=1))
print('years: Sharpe %d/%d, DD %d/%d' % (R['sharpe_wins'], len(R['years']),
                                         R['dd_wins'], len(R['years'])))
print('concentration', R['concentration'])
json.dump(R, open(os.path.join(OUT, 'mega_results.json'), 'w'), indent=2)
print('\nsaved mega_results.json')
