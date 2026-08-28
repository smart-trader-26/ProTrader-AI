"""The absolute regime volatility budgets (20/15/10%) never bind on a diversified
multi-asset sleeve whose own volatility is ~7%. This tests the principled fix:
express the regime budget as a multiple of the sleeve's own risk instead of an
absolute level, so the de-risking layer is actually active.

Reports what happens - including if it does not help.
"""
import json, os, pickle, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf
import cvxpy as cp
import pipeline_core as P

warnings.filterwarnings('ignore')
OUT = os.path.dirname(os.path.abspath(__file__))
MULTI = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'IEF', 'LQD', 'GLD', 'VNQ']
P.START = '2007-01-01'
P._CACHE.clear()
P._CACHE[tuple(MULTI)] = pickle.load(open(os.path.join(OUT, '_artifacts_cache.pkl'), 'rb'))['data']


def sh(x):
    return float(x.mean() / x.std() * np.sqrt(252)) if x.std() else 0.0


def mdd(x):
    c = (1 + x).cumprod()
    return float((c / c.cummax() - 1).min())


def backtest(universe, mult=None, cfg=None, use_regime=True):
    """pipeline_core.backtest with an optional relative regime risk budget.

    mult[r] is the fraction of the sleeve's own volatility retained in regime r;
    mult=None reproduces the absolute-target behaviour exactly.
    """
    cfg = {**P.DEF, **(cfg or {})}
    D = P.load(universe)
    returns, tech, sent, conf, rf, common = (D['returns'], D['tech'], D['sent'],
                                             D['conf'], D['regime_feat'], D['common'])
    cov_lb, rebal, cost = cfg['cov_lookback'], cfg['rebal'], cfg['cost_bps'] / 10000.0
    ic, lam, w_max = cfg['ic'], cfg['lam'], cfg['w_max']
    dates = common[common >= returns.index[max(cov_lb, 252)]]
    rebal_dates = dates[::rebal]
    n = len(universe)
    w_prev = np.zeros(n)
    daily, wlog, rlog = [], {}, {}

    for k, d in enumerate(rebal_dates):
        hist = returns[returns.index < d].iloc[-cov_lb:]
        if len(hist) < cov_lb:
            continue
        Sig = LedoitWolf().fit(hist).covariance_ * 252.0
        sig_i = np.sqrt(np.diag(Sig))
        rg = P.regime_at(rf, d, cfg['n_regimes'], cfg['seed']) if use_regime else 1
        wt, ws = cfg['fuse'][rg] if use_regime else (0.5, 0.5)
        z = np.nan_to_num((wt * tech.loc[d].values + ws * sent.loc[d].values) * conf.loc[d].values)
        alpha = ic * sig_i * z

        w = cp.Variable(n)
        risk = cp.quad_form(w, cp.psd_wrap(Sig))
        prob = cp.Problem(
            cp.Maximize(alpha @ w - lam * risk - (cost * 252 / rebal) * cp.norm(w - w_prev, 1)),
            [w >= 0, w <= w_max, cp.sum(w) == 1])
        try:
            prob.solve(solver=cp.CLARABEL)
        except Exception:
            prob.solve()
        if w.value is None:
            continue
        sleeve = np.clip(w.value, 0, None)
        sleeve = sleeve / sleeve.sum()
        sv = float(np.sqrt(sleeve @ Sig @ sleeve))
        if mult is None:
            tgt = cfg['target_vol'][rg] if use_regime else 0.15
            kk = min(1.0, tgt / sv) if sv > 0 else 0.0
        else:
            kk = mult[rg] if use_regime else mult[1]
        wv = kk * sleeve
        turn = np.abs(wv - w_prev).sum()

        end = rebal_dates[k + 1] if k + 1 < len(rebal_dates) else dates[-1]
        seg = returns.loc[(returns.index > d) & (returns.index <= end)]
        if len(seg) == 0:
            continue
        pos, cash = wv.copy(), 1.0 - wv.sum()
        V = pos.sum() + cash
        for j, (dt, row) in enumerate(seg.iterrows()):
            pos = pos * (1 + row.values)
            Vn = pos.sum() + cash
            r = Vn / V - 1.0
            if j == 0:
                r -= turn * cost
            daily.append((dt, r))
            V = Vn
        w_prev = wv
        wlog[d], rlog[d] = wv, rg

    return pd.Series(dict(daily)).sort_index(), pd.DataFrame(wlog).T, pd.Series(rlog)


EPISODES = {'GFC 2008': ('2008-09-01', '2009-03-09'),
            'Euro crisis 2011': ('2011-07-01', '2011-10-03'),
            'COVID 2020': ('2020-02-19', '2020-03-23'),
            '2022 bear': ('2022-01-03', '2022-10-12')}

GRIDS = {
    'absolute (as submitted)': None,
    'relative 1.00/0.75/0.50': {0: 1.00, 1: 0.75, 2: 0.50},
    'relative 1.00/0.80/0.60': {0: 1.00, 1: 0.80, 2: 0.60},
    'relative 1.00/0.60/0.30': {0: 1.00, 1: 0.60, 2: 0.30},
    'relative 1.00/0.50/0.25': {0: 1.00, 1: 0.50, 2: 0.25},
}

base, W0, RG0 = backtest(MULTI, None)
idx = base.index
ew = P.benchmarks(MULTI, idx)['ew']
R = {}
print(f"{'configuration':26s} {'Sharpe':>7} {'MaxDD':>8} {'CAGR':>7} {'invested':>9} "
      f"{'vs EW t':>8} {'vs flat t':>10}")
print('-' * 82)
flat = None
for nm, m in GRIDS.items():
    s, W, RG = backtest(MULTI, m)
    s = s.reindex(idx).dropna()
    st = P.stats_of(s)
    inv = float(W.sum(axis=1).mean())
    e = (s - ew.reindex(s.index)).dropna()
    t_ew = float(stats.ttest_1samp(e, 0)[0])
    if m is None:
        t_fl = np.nan
        fl = {}
    else:
        # control: the SAME average exposure held constantly, i.e. de-risking by the
        # same amount but without using regime information to time it
        c = round(inv, 3)
        s_flat, _, _ = backtest(MULTI, {0: c, 1: c, 2: c})
        s_flat = s_flat.reindex(s.index).dropna()
        d_ = (s - s_flat).dropna()
        t_fl = float(stats.ttest_1samp(d_, 0)[0])
        fl = dict(multiplier=c, Sharpe=round(sh(s_flat), 4), MaxDD=round(mdd(s_flat), 4),
                  CAGR=round(P.stats_of(s_flat)['CAGR'], 4))
        print(f"{'  [matched flat control]':26s} {fl['Sharpe']:7.3f} {fl['MaxDD']:8.3f} "
              f"{fl['CAGR']:7.3f} {c:9.3f}")
    print(f"{nm:26s} {st['Sharpe']:7.3f} {st['MaxDD']:8.3f} {st['CAGR']:7.3f} {inv:9.3f} "
          f"{t_ew:8.2f} {t_fl:10.2f}")
    ep = {}
    for k, (a, b) in EPISODES.items():
        mm = (s.index >= a) & (s.index <= b)
        if mm.sum() < 5:
            continue
        cs, ce = float((1 + s[mm]).prod() - 1), float((1 + ew.reindex(s.index)[mm]).prod() - 1)
        ep[k] = dict(strategy=round(cs, 4), equal_weight=round(ce, 4), excess=round(cs - ce, 4))
    R[nm] = dict(stats={k: round(v, 4) for k, v in st.items() if k != 'name'},
                 mean_invested=round(inv, 4),
                 min_invested=round(float(W.sum(axis=1).min()), 4),
                 binding_rebalances=int((W.sum(axis=1) < 0.99).sum()), n_rebalances=len(W),
                 t_vs_ew=round(t_ew, 3), t_vs_matched_flat=round(float(t_fl), 3),
                 matched_flat_control=fl, episodes=ep)

print('\nequal-weight benchmark:', {k: round(v, 4) for k, v in P.stats_of(ew).items() if k != 'name'})
R['equal_weight'] = {k: round(v, 4) for k, v in P.stats_of(ew).items() if k != 'name'}
print('\ncrisis excess vs equal-weight')
for nm in GRIDS:
    print(f"  {nm:26s} " + '  '.join(f"{k.split()[0]} {v['excess']:+.3f}" for k, v in R[nm]['episodes'].items()))
json.dump(R, open(os.path.join(OUT, 'regime_budget_results.json'), 'w'), indent=2)
print('\nsaved regime_budget_results.json')
