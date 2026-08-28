"""Every number and figure in the revised IJADS paper, from one primary specification.

Primary: MULTI universe, 2007-2026 download, seed 42, cov 252d, rebal 21d, 10 bps,
relative regime risk budget {calm 1.00, transitional 0.75, stress 0.50}.

Writes final_tables.json and 300 dpi figures into ../single column/images/.
"""
import json, os, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import pipeline_core as P

OUT = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.abspath(os.path.join(OUT, '..', 'single column', 'images'))
MULTI = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'IEF', 'LQD', 'GLD', 'VNQ']
MEGA = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'XOM']
DOWNLOAD_END = '2026-08-27'   # all data available at the revision date; no window selection
HOLDOUT_START = '2024-01-01'  # postdates the original submission -- never used in development
EPISODES = {'GFC 2008': ('2008-09-01', '2009-03-09'),
            'Euro crisis 2011': ('2011-07-01', '2011-10-03'),
            'COVID 2020': ('2020-02-19', '2020-03-23'),
            '2022 bear': ('2022-01-03', '2022-10-12')}

P.START, P.END = '2007-01-01', DOWNLOAD_END
P._CACHE.clear()
R = {}

plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans', 'axes.grid': True,
                     'grid.alpha': 0.3, 'grid.linewidth': 0.5, 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False,
                     'savefig.dpi': 300, 'figure.dpi': 300, 'savefig.bbox': 'tight'})
C = {'strat': '#1f4e79', 'ew': '#a0a0a0', 'r0': '#2c7fb8', 'r1': '#d95f02', 'r2': '#7570b3'}


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


def rd(d):
    return {k: round(v, 4) for k, v in d.items() if k != 'name'}


CACHE = os.path.join(OUT, '_final_cache_ext.pkl')
if os.path.exists(CACHE):
    Z = pickle.load(open(CACHE, 'rb'))
else:
    Z = {}
    print('primary...', flush=True)
    Z['strat'] = P.backtest(MULTI)
    inv = float(Z['strat'][1].sum(axis=1).mean())
    Z['inv'] = inv
    print('ablations...', flush=True)
    Z['noreg_full'] = P.backtest(MULTI, dict(flat_mult=1.0), use_regime=False)
    Z['noreg_matched'] = P.backtest(MULTI, dict(flat_mult=round(inv, 3)), use_regime=False)
    Z['nosig'] = P.backtest(MULTI, signal_on=False)
    print('cost...', flush=True)
    Z['cost'] = {b: P.backtest(MULTI, dict(cost_bps=b)) for b in [0.0, 5.0, 10.0, 25.0]}
    print('budget ladder...', flush=True)
    Z['ladder'] = {}
    for nm, m in [('1.00 / 0.75 / 0.50', {0: 1.0, 1: .75, 2: .5}),
                  ('1.00 / 0.80 / 0.60', {0: 1.0, 1: .8, 2: .6}),
                  ('1.00 / 0.60 / 0.30', {0: 1.0, 1: .6, 2: .3}),
                  ('1.00 / 0.50 / 0.25', {0: 1.0, 1: .5, 2: .25}),
                  ('absolute 20/15/10 %', None)]:
        Z['ladder'][nm] = P.backtest(MULTI, dict(vol_mult=m))
    print('sweep...', flush=True)
    Z['sweep'] = {}
    for seed in [0, 7, 42]:
        for lb in [126, 252]:
            for rb in [10, 21, 42]:
                Z['sweep'][(seed, lb, rb)] = P.backtest(
                    MULTI, dict(seed=seed, cov_lookback=lb, rebal=rb))[0]
                print('  ', seed, lb, rb, flush=True)
    pickle.dump(Z, open(CACHE, 'wb'))

strat, W, RG = Z['strat']
D = P.load(MULTI)
idx = strat.index
bm = P.benchmarks(MULTI, idx)
ew, iv = bm['ew'], bm['iv']
ewm = ew * (strat.std() / ew.std())
R['period'] = [str(idx[0].date()), str(idx[-1].date()), int(len(idx))]
R['spec'] = dict(universe=MULTI, cov_lookback=252, rebalance_days=21, cost_bps=10,
                 w_max=0.25, ic=0.05, risk_aversion=2.0, n_regimes=3, seed=42,
                 vol_mult={0: 1.0, 1: 0.75, 2: 0.5})

# ---------------------------------------------------------------- T1 main
rows = [P.stats_of(strat, 'Regime-conditional strategy'), P.stats_of(ew, 'Equal-weight'),
        P.stats_of(iv, 'Inverse-volatility'), P.stats_of(ewm, 'Equal-weight at strategy volatility')]
R['T1_main'] = {r['name']: rd(r) for r in rows}
print(pd.DataFrame(rows).set_index('name').round(4).to_string())

# ---------------------------------------------------------------- T2 sweep
sw = []
for (seed, lb, rb), s_ in Z['sweep'].items():
    e_ = P.benchmarks(MULTI, s_.index)['ew']
    em = e_ * (s_.std() / e_.std())
    sw.append(dict(seed=seed, cov_lb=lb, rebal=rb, strat_sharpe=round(sh(s_), 3),
                   ew_sharpe=round(sh(e_), 3), strat_dd=round(mdd(s_), 3), ew_dd=round(mdd(e_), 3),
                   sharpe_win=bool(sh(s_) > sh(e_)), dd_win=bool(mdd(s_) > mdd(e_)),
                   dd_win_matched=bool(mdd(s_) > mdd(em))))
SW = pd.DataFrame(sw)
R['T2_sweep'] = dict(runs=sw, n=len(SW), sharpe_wins=int(SW.sharpe_win.sum()),
                     dd_wins=int(SW.dd_win.sum()), dd_wins_matched=int(SW.dd_win_matched.sum()),
                     sharpe_min=float(SW.strat_sharpe.min()), sharpe_max=float(SW.strat_sharpe.max()),
                     dd_min=float(SW.strat_dd.min()), dd_max=float(SW.strat_dd.max()))
print('\nsweep: Sharpe wins %d/%d, DD wins %d/%d, DD vs vol-matched %d/%d, Sharpe range %.3f-%.3f'
      % (SW.sharpe_win.sum(), len(SW), SW.dd_win.sum(), len(SW), SW.dd_win_matched.sum(), len(SW),
         SW.strat_sharpe.min(), SW.strat_sharpe.max()))

# ---------------------------------------------------------------- T3 crises
R['T3_episodes'] = {}
for nm, (a, b) in EPISODES.items():
    m = (idx >= a) & (idx <= b)
    cs, ce = float((1 + strat[m]).prod() - 1), float((1 + ew[m]).prod() - 1)
    R['T3_episodes'][nm] = dict(days=int(m.sum()), strategy=round(cs, 4),
                                equal_weight=round(ce, 4), excess=round(cs - ce, 4))
print('\ncrises', json.dumps(R['T3_episodes'], indent=1))

# ---------------------------------------------------------------- T4 years
yrs = []
for y in sorted(set(idx.year)):
    m = idx.year == y
    if m.sum() < 60:
        continue
    s_, e_ = strat[m], ew[m]
    yrs.append(dict(year=int(y), strat_ret=round(float((1 + s_).prod() - 1), 4),
                    ew_ret=round(float((1 + e_).prod() - 1), 4),
                    strat_sharpe=round(sh(s_), 3), ew_sharpe=round(sh(e_), 3),
                    strat_dd=round(mdd(s_), 3), ew_dd=round(mdd(e_), 3),
                    sharpe_win=bool(sh(s_) > sh(e_)), dd_win=bool(mdd(s_) > mdd(e_))))
YR = pd.DataFrame(yrs)
R['T4_years'] = dict(rows=yrs, n=len(YR), sharpe_wins=int(YR.sharpe_win.sum()),
                     dd_wins=int(YR.dd_win.sum()))
print('\nyears: Sharpe %d/%d, DD %d/%d' % (YR.sharpe_win.sum(), len(YR), YR.dd_win.sum(), len(YR)))
print(YR.to_string(index=False))

# ---------------------------------------------------------------- T5 ablation
nrf = Z['noreg_full'][0].reindex(idx).dropna()
nrm = Z['noreg_matched'][0].reindex(idx).dropna()
nsg = Z['nosig'][0].reindex(idx).dropna()
abl = [P.stats_of(strat, 'Full framework'),
       P.stats_of(nrm, 'No regime timing, same average exposure'),
       P.stats_of(nrf, 'No regime timing, full exposure'),
       P.stats_of(nsg, 'No signal layer'),
       P.stats_of(ew, 'Equal-weight')]
R['T5_ablation'] = {r['name']: rd(r) for r in abl}
for other, nm in [(nrm, 'vs_no_regime_matched'), (nrf, 'vs_no_regime_full'), (nsg, 'vs_no_signal')]:
    e = (strat - other).dropna()
    t_, p_ = stats.ttest_1samp(e, 0)
    R['T5_ablation'][nm] = dict(excess_ann=round(float(e.mean() * 252), 5),
                                t=round(float(t_), 3), p=round(float(p_), 4))
R['T5_ablation']['mean_exposure'] = dict(
    full=round(float(W.sum(axis=1).mean()), 4),
    matched_control=round(float(Z['noreg_matched'][1].sum(axis=1).mean()), 4))
print('\nablation'); print(pd.DataFrame(abl).set_index('name').round(4).to_string())
for k in ['vs_no_regime_matched', 'vs_no_regime_full', 'vs_no_signal']:
    print(' ', k, R['T5_ablation'][k])

# ---------------------------------------------------------------- T5b budget ladder
R['T5b_ladder'] = {}
for nm, (s_, W_, _) in Z['ladder'].items():
    s_ = s_.reindex(idx).dropna()
    R['T5b_ladder'][nm] = dict(rd(P.stats_of(s_)),
                               mean_exposure=round(float(W_.sum(axis=1).mean()), 4))
print('\nladder'); print(pd.DataFrame(R['T5b_ladder']).T.to_string())

# ---------------------------------------------------------------- T6 IC (from ic_results.json)
R['T6_ic'] = json.load(open(os.path.join(OUT, 'ic_results.json')))['MULTI-ASSET 2008-2026']

# ---------------------------------------------------------------- T7 concentration / cost
def wstats(Wdf):
    w = Wdf.values
    tot = w.sum(axis=1)
    sleeve = np.divide(w, np.where(tot > 0, tot, np.nan)[:, None])
    hhi = np.nansum(sleeve ** 2, axis=1)
    turn = np.abs(np.diff(np.vstack([np.zeros(w.shape[1]), w]), axis=0)).sum(axis=1)[1:]
    return dict(mean_top_weight=round(float(np.nanmax(sleeve, axis=1).mean()), 4),
                max_top_weight=round(float(np.nanmax(sleeve)), 4),
                mean_effective_assets=round(float(np.nanmean(1.0 / hhi)), 2),
                min_effective_assets=round(float(np.nanmin(1.0 / hhi)), 2),
                mean_exposure=round(float(tot.mean()), 4), min_exposure=round(float(tot.min()), 4),
                turnover_per_rebalance=round(float(turn.mean()), 4),
                turnover_annual=round(float(turn.mean() * 252 / 21), 2))


R['T7_concentration'] = wstats(W)
R['T7_cost'] = {}
for b in sorted(Z['cost']):
    s_, W_, _ = Z['cost'][b]
    st = P.stats_of(s_.reindex(idx).dropna())
    R['T7_cost']['%d bps' % int(b)] = dict(CAGR=round(st['CAGR'], 4), Sharpe=round(st['Sharpe'], 4),
                                           MaxDD=round(st['MaxDD'], 4),
                                           turnover_annual=wstats(W_)['turnover_annual'])
print('\nconcentration', R['T7_concentration'])
print('cost', json.dumps(R['T7_cost'], indent=1))

# ---------------------------------------------------------------- T8 regime model
rf = D['regime_feat'].loc[D['regime_feat'].index <= idx[-1]]
X = StandardScaler().fit_transform(rf)
R['T8_order'] = []
for k in range(2, 7):
    g = GaussianMixture(n_components=k, random_state=42, n_init=5).fit(X)
    R['T8_order'].append(dict(k=k, AIC=round(float(g.aic(X)), 1), BIC=round(float(g.bic(X)), 1)))
g3 = GaussianMixture(n_components=3, random_state=42, n_init=5).fit(X)
remap = {o: n for n, o in enumerate(np.argsort(g3.means_[:, 0]))}
lab = pd.Series([remap[i] for i in g3.predict(X)], index=rf.index)
Tm = np.zeros((3, 3))
for a, b in zip(lab.values[:-1], lab.values[1:]):
    Tm[a, b] += 1
Tm = Tm / Tm.sum(axis=1, keepdims=True)
runs = {0: [], 1: [], 2: []}
cur, n_ = int(lab.iloc[0]), 1
for v in lab.values[1:]:
    if v == cur:
        n_ += 1
    else:
        runs[cur].append(n_); cur, n_ = int(v), 1
runs[cur].append(n_)
rg_daily = RG.reindex(idx.union(RG.index)).ffill().reindex(idx).ffill().bfill().astype(int)
R['T8_dynamics'] = dict(
    transition=[[round(float(x), 3) for x in r] for r in Tm],
    share={int(k): round(float((lab == k).mean()), 3) for k in [0, 1, 2]},
    oos_share={int(k): round(float((rg_daily == k).mean()), 3) for k in [0, 1, 2]},
    mean_run_days={int(k): round(float(np.mean(v)), 1) for k, v in runs.items()},
    ann_vol={int(k): round(float(rf['Volatility'][lab == k].mean() * np.sqrt(252)), 4) for k in [0, 1, 2]},
    ann_mom={int(k): round(float(rf['Momentum'][lab == k].mean() * 252), 4) for k in [0, 1, 2]})
# realised strategy vs benchmark inside each out-of-sample regime
R['T8_by_regime'] = []
for k in [0, 1, 2]:
    m = (rg_daily == k).values
    if m.sum() < 30:
        continue
    R['T8_by_regime'].append(dict(regime=k, days=int(m.sum()), strat_sharpe=round(sh(strat[m]), 3),
                                  ew_sharpe=round(sh(ew[m]), 3),
                                  strat_ann=round(float(strat[m].mean() * 252), 4),
                                  ew_ann=round(float(ew[m].mean() * 252), 4)))
print('\ntransition\n', np.round(Tm, 3))
print(R['T8_dynamics']); print(R['T8_by_regime'])


# ---------------------------------------------------------------- T9 significance
def block_boot(x, y, nrep=2000, blk=21, seed=42):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    n = len(x); nb = int(np.ceil(n / blk))
    ds, de, dd = [], [], []
    for _ in range(nrep):
        st = rng.integers(0, n, nb)
        ii = np.concatenate([np.arange(s, s + blk) % n for s in st])[:n]
        a_, b_ = pd.Series(x[ii]), pd.Series(y[ii])
        ds.append(sh(a_)); de.append(sh(b_)); dd.append(ds[-1] - de[-1])
    q = lambda a: [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]
    return dict(strat_ci=q(ds), ew_ci=q(de), diff_ci=q(dd),
                p_diff_le_0=round(float((np.array(dd) <= 0).mean()), 4))


ex = (strat - ew).dropna()
t_, p_ = stats.ttest_1samp(ex, 0)
dn, up = ew < 0, ew > 0
tail = ew.nsmallest(int(0.05 * len(ew)))
R['T9_significance'] = dict(
    excess_ann=round(float(ex.mean() * 252), 5), t=round(float(t_), 3), p=round(float(p_), 4),
    nw_t=round(nw_t(ex), 3), strat_sharpe=round(sh(strat), 4), ew_sharpe=round(sh(ew), 4),
    bootstrap=block_boot(strat, ew),
    capture=dict(downside=round(float(strat[dn].mean() / ew[dn].mean()), 4),
                 upside=round(float(strat[up].mean() / ew[up].mean()), 4)),
    worst_5pct_days=dict(n=len(tail), strategy=round(float(strat[tail.index].mean()), 5),
                         equal_weight=round(float(tail.mean()), 5),
                         cushion=round(float(strat[tail.index].mean() - tail.mean()), 5)))
print('\nsignificance', json.dumps(R['T9_significance'], indent=1))

# ------------------------------------------------- T10 holdout 2024-2026 (never used)
# The sample runs to every session available at the revision date. The block from
# HOLDOUT_START onward postdates the original submission entirely, so it is reported
# separately as an out-of-sample check. No parameter was re-tuned on it.
hi = idx[idx >= HOLDOUT_START]
ii = idx[idx < HOLDOUT_START]
R['T10_holdout'] = {}
for tag, sub in [('holdout', hi), ('in_sample', ii), ('full', idx)]:
    ewm_sub = ew.loc[sub] * (strat.loc[sub].std() / ew.loc[sub].std())
    blk = {}
    for nm, x in [('Regime-conditional strategy', strat), ('Equal-weight', ew),
                  ('Inverse-volatility', iv), ('No regime timing, full exposure', nrf)]:
        y = x.reindex(sub).dropna()
        if len(y) > 30:
            blk[nm] = rd(P.stats_of(y))
    blk['Equal-weight at strategy volatility'] = rd(P.stats_of(ewm_sub))
    blk['_period'] = [str(sub[0].date()), str(sub[-1].date()), int(len(sub))]
    R['T10_holdout'][tag] = blk
ex_h = (strat.reindex(hi) - ew.reindex(hi)).dropna()
th, ph = stats.ttest_1samp(ex_h, 0)
R['T10_holdout']['significance'] = dict(
    excess_ann=round(float(ex_h.mean() * 252), 5), t=round(float(th), 3), p=round(float(ph), 4),
    bootstrap=block_boot(strat.reindex(hi).dropna(), ew.reindex(hi).dropna()))
# deepest equal-weight drawdown inside the holdout, located by rule rather than chosen
ce_h = (1 + ew.reindex(hi)).cumprod()
ddh = ce_h / ce_h.cummax() - 1
trough = ddh.idxmin()
peak = ce_h.loc[:trough].idxmax()
mh = (idx >= peak) & (idx <= trough)
R['T10_holdout']['worst_ew_episode'] = dict(
    peak=str(peak.date()), trough=str(trough.date()), days=int(mh.sum()),
    strategy=round(float((1 + strat[mh]).prod() - 1), 4),
    equal_weight=round(float((1 + ew[mh]).prod() - 1), 4),
    excess=round(float((1 + strat[mh]).prod() - (1 + ew[mh]).prod()), 4))
print('\nholdout', json.dumps(R['T10_holdout'], indent=1, default=str))

# ---------------------------------------------------------------- figures
cum_s, cum_e = (1 + strat).cumprod(), (1 + ew).cumprod()

f, a = plt.subplots(figsize=(7.0, 3.4))
a.plot(cum_s.index, cum_s, color=C['strat'], lw=1.4, label='Regime-conditional strategy')
a.plot(cum_e.index, cum_e, color=C['ew'], lw=1.2, label='Equal-weight benchmark')
a.set_ylabel('Growth of one unit of capital'); a.set_yscale('log')
a.set_yticks([0.6, 1, 1.5, 2, 3]); a.set_yticklabels(['0.6', '1.0', '1.5', '2.0', '3.0'])
a.legend(loc='upper left'); f.savefig(os.path.join(IMG, 'fig_cumret.png')); plt.close(f)

f, a = plt.subplots(figsize=(7.0, 2.8))
a.fill_between(cum_e.index, (cum_e / cum_e.cummax() - 1) * 100, 0, color=C['ew'], alpha=.6,
               label='Equal-weight benchmark')
a.plot(cum_s.index, (cum_s / cum_s.cummax() - 1) * 100, color=C['strat'], lw=1.2,
       label='Regime-conditional strategy')
a.set_ylabel('Drawdown (%)'); a.legend(loc='lower left')
f.savefig(os.path.join(IMG, 'fig_drawdown.png')); plt.close(f)

f, (a1, a2) = plt.subplots(2, 1, figsize=(7.0, 4.2), sharex=True,
                           gridspec_kw=dict(height_ratios=[2, 1]))
vol = rf['Volatility'] * np.sqrt(252) * 100
for k, nm in [(0, 'Regime 0 (calm)'), (1, 'Regime 1 (transitional)'), (2, 'Regime 2 (stress)')]:
    m = (lab == k).values
    a1.scatter(vol.index[m], vol.values[m], s=1.6, color=C['r%d' % k], label=nm)
a1.set_ylabel('Annualised volatility (%)'); a1.legend(loc='upper right', markerscale=5, ncol=3)
a2.plot(rf.index, rf['Momentum'] * 252 * 100, color='#333333', lw=0.7)
a2.axhline(0, color='k', lw=0.5); a2.set_ylabel('Momentum (% p.a.)')
f.savefig(os.path.join(IMG, 'fig_regimes.png')); plt.close(f)

f, a = plt.subplots(figsize=(7.0, 3.4))
Wp = W.copy(); Wp['Cash'] = 1.0 - Wp.sum(axis=1)
Wd = Wp.reindex(idx.union(Wp.index)).ffill().reindex(idx)
a.stackplot(Wd.index, [Wd[c].values * 100 for c in Wd.columns], labels=list(Wd.columns),
            colors=plt.cm.tab20(np.linspace(0, 1, len(Wd.columns))), lw=0)
a.set_ylabel('Portfolio weight (%)'); a.set_ylim(0, 100)
a.legend(loc='upper center', ncol=6, fontsize=6.5, bbox_to_anchor=(0.5, 1.22))
f.savefig(os.path.join(IMG, 'fig_alloc.png')); plt.close(f)

f, a = plt.subplots(figsize=(7.0, 2.8))
expo = Wd.drop(columns=['Cash']).sum(axis=1) * 100
a.plot(expo.index, expo, color=C['strat'], lw=1.0)
for k, col in [(2, C['r2']), (1, C['r1'])]:
    a.fill_between(expo.index, 0, 100, where=(rg_daily == k).values, color=col, alpha=.12, lw=0)
a.set_ylabel('Invested exposure (%)'); a.set_ylim(40, 105)
a.text(0.01, 0.06, 'shaded: transitional (orange) and stress (purple) regimes',
       transform=a.transAxes, fontsize=7.5, color='#555555')
f.savefig(os.path.join(IMG, 'fig_exposure.png')); plt.close(f)

f, a = plt.subplots(figsize=(7.0, 2.8))
tt = D['tech'].loc[idx].abs().mean(axis=1).rolling(21).mean()
ss = D['sent'].loc[idx].abs().mean(axis=1).rolling(21).mean()
a.plot(tt.index, tt, color=C['strat'], lw=1.0, label='Technical composite')
a.plot(ss.index, ss, color=C['r1'], lw=1.0, label='Volume/volatility composite')
a.set_ylabel('Mean absolute\ncross-sectional score'); a.legend(loc='upper right', ncol=2)
f.savefig(os.path.join(IMG, 'fig_signal.png')); plt.close(f)

f, a = plt.subplots(figsize=(7.0, 2.8))
rs = strat.rolling(252).mean() / strat.rolling(252).std() * np.sqrt(252)
re_ = ew.rolling(252).mean() / ew.rolling(252).std() * np.sqrt(252)
a.plot(rs.index, rs, color=C['strat'], lw=1.2, label='Regime-conditional strategy')
a.plot(re_.index, re_, color=C['ew'], lw=1.2, label='Equal-weight benchmark')
a.axhline(0, color='k', lw=0.5); a.set_ylabel('252-day rolling Sharpe ratio')
a.legend(loc='lower right', ncol=2); f.savefig(os.path.join(IMG, 'fig_rollsharpe.png')); plt.close(f)

rsd = rs.dropna()
R['rolling_sharpe'] = dict(strat_frac_positive=round(float((rsd > 0).mean()), 3),
                           ew_frac_positive=round(float((re_.dropna() > 0).mean()), 3),
                           strat_frac_above_ew=round(float((rsd > re_.reindex(rsd.index)).mean()), 3))
json.dump(R, open(os.path.join(OUT, 'final_tables.json'), 'w'), indent=2, default=str)
print('\nsaved final_tables.json and 7 figures ->', IMG)
