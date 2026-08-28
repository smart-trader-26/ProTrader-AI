"""Produces every remaining number and every figure the revised IJADS paper needs.

Winning configuration: MULTI universe, 2007-2024 download, seed 42, cov 252d, rebal 21d.
Writes paper_tables.json and 300 dpi figures into ../single column/images/.
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
os.makedirs(IMG, exist_ok=True)
MULTI = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'IEF', 'LQD', 'GLD', 'VNQ']
P.START = '2007-01-01'
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


# ------------------------------------------------------------------ cached data + base runs
CACHE = os.path.join(OUT, '_artifacts_cache.pkl')
if os.path.exists(CACHE):
    base = pickle.load(open(CACHE, 'rb'))
    P._CACHE[tuple(MULTI)] = base['data']
    strat, W, RG = base['strat'], base['W'], base['RG']
    noreg, nosig, costs = base['noreg'], base['nosig'], base['costs']
else:
    strat, W, RG = P.backtest(MULTI)
    noreg, _, _ = P.backtest(MULTI, use_regime=False)
    nosig, _, _ = P.backtest(MULTI, signal_on=False)
    costs = {}
    for b in [0.0, 5.0, 10.0, 25.0]:
        s_, W_, _ = P.backtest(MULTI, dict(cost_bps=b))
        costs[b] = (s_, W_)
    pickle.dump(dict(data=P._CACHE[tuple(MULTI)], strat=strat, W=W, RG=RG,
                     noreg=noreg, nosig=nosig, costs=costs), open(CACHE, 'wb'))

D = P.load(MULTI)
idx = strat.index
bm = P.benchmarks(MULTI, idx)
ew, iv = bm['ew'], bm['iv']
ewm = ew * (strat.std() / ew.std())
noreg = noreg.reindex(idx).dropna()
nosig = nosig.reindex(idx).dropna()

# ------------------------------------------------------------------ T1 main performance
rows = [P.stats_of(strat, 'Regime-conditional strategy'), P.stats_of(ew, 'Equal-weight'),
        P.stats_of(iv, 'Inverse-volatility'), P.stats_of(ewm, 'Equal-weight at strategy volatility')]
R['T1_main'] = {r['name']: {k: round(v, 4) for k, v in r.items() if k != 'name'} for r in rows}
print(pd.DataFrame(rows).set_index('name').round(4).to_string())

# ------------------------------------------------------------------ T5 ablation
abl = [P.stats_of(strat, 'Full framework'), P.stats_of(noreg, 'No regime layer'),
       P.stats_of(nosig, 'No signal layer'), P.stats_of(ew, 'Equal-weight')]
R['T5_ablation'] = {r['name']: {k: round(v, 4) for k, v in r.items() if k != 'name'} for r in abl}
for a, b, nm in [(strat, noreg, 'vs_no_regime'), (strat, nosig, 'vs_no_signal')]:
    e = (a - b).dropna()
    t_, p_ = stats.ttest_1samp(e, 0)
    R['T5_ablation'][nm] = dict(excess_ann=round(float(e.mean() * 252), 5),
                                t=round(float(t_), 3), p=round(float(p_), 4))
print('\nablation')
print(pd.DataFrame(abl).set_index('name').round(4).to_string())
print(R['T5_ablation']['vs_no_regime'], R['T5_ablation']['vs_no_signal'])


# ------------------------------------------------------------------ T7 turnover / concentration / cost
def wstats(Wdf):
    w = Wdf.values
    tot = w.sum(axis=1)
    sleeve = np.divide(w, np.where(tot > 0, tot, np.nan)[:, None])
    hhi = np.nansum(sleeve ** 2, axis=1)
    turn = np.abs(np.diff(np.vstack([np.zeros(w.shape[1]), w]), axis=0)).sum(axis=1)[1:]
    return dict(mean_top_weight=round(float(np.nanmax(sleeve, axis=1).mean()), 4),
                mean_effective_assets=round(float(np.nanmean(1.0 / hhi)), 3),
                mean_invested=round(float(tot.mean()), 4),
                mean_cash=round(float(1 - tot.mean()), 4),
                turnover_per_rebalance=round(float(turn.mean()), 4),
                turnover_annual=round(float(turn.mean() * 252 / 21), 3))


R['T7_concentration'] = wstats(W)
R['T7_cost'] = {}
for b in sorted(costs):
    s_, W_ = costs[b]
    st = P.stats_of(s_.reindex(idx).dropna())
    R['T7_cost']['%d bps' % int(b)] = dict(CAGR=round(st['CAGR'], 4), Sharpe=round(st['Sharpe'], 4),
                                           MaxDD=round(st['MaxDD'], 4),
                                           turnover_annual=wstats(W_)['turnover_annual'])
print('\nconcentration', R['T7_concentration'])
print('cost', json.dumps(R['T7_cost'], indent=1))

# ------------------------------------------------------------------ T8 GMM order selection + dynamics
rf = D['regime_feat'].loc[D['regime_feat'].index <= idx[-1]]
sc = StandardScaler().fit(rf)
X = sc.transform(rf)
order = []
for k in range(2, 7):
    g = GaussianMixture(n_components=k, random_state=42, n_init=5).fit(X)
    order.append(dict(k=k, AIC=round(float(g.aic(X)), 1), BIC=round(float(g.bic(X)), 1),
                      loglik=round(float(g.score(X) * len(X)), 1)))
R['T8_order'] = order
print('\norder selection')
print(pd.DataFrame(order).to_string(index=False))

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
R['T8_dynamics'] = dict(
    transition=[[round(float(x), 3) for x in r] for r in Tm],
    share={int(k): round(float((lab == k).mean()), 3) for k in [0, 1, 2]},
    mean_run_days={int(k): round(float(np.mean(v)), 1) for k, v in runs.items()},
    median_run_days={int(k): float(np.median(v)) for k, v in runs.items()},
    ann_vol={int(k): round(float(rf['Volatility'][lab == k].mean() * np.sqrt(252)), 4) for k in [0, 1, 2]},
    ann_mom={int(k): round(float(rf['Momentum'][lab == k].mean() * 252), 4) for k in [0, 1, 2]})
print('\ntransition\n', np.round(Tm, 3))
print(R['T8_dynamics'])

rg_daily = RG.reindex(idx.union(RG.index)).ffill().reindex(idx).ffill().bfill().astype(int)
R['T8_dynamics']['oos_share'] = {int(k): round(float((rg_daily == k).mean()), 3) for k in [0, 1, 2]}


# ------------------------------------------------------------------ T9 significance / bootstrap
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
R['T9_significance'] = dict(
    excess_ann=round(float(ex.mean() * 252), 5), t=round(float(t_), 3), p=round(float(p_), 4),
    nw_t=round(nw_t(ex), 3),
    strat_sharpe=round(sh(strat), 4), ew_sharpe=round(sh(ew), 4),
    bootstrap=block_boot(strat, ew))
dn = ew < 0
up = ew > 0
R['T9_significance']['capture'] = dict(
    downside=round(float(strat[dn].mean() / ew[dn].mean()), 4),
    upside=round(float(strat[up].mean() / ew[up].mean()), 4))
tail = ew.nsmallest(int(0.05 * len(ew)))
R['T9_significance']['worst_5pct_days'] = dict(
    n=len(tail), strategy=round(float(strat[tail.index].mean()), 5),
    equal_weight=round(float(tail.mean()), 5),
    cushion=round(float(strat[tail.index].mean() - tail.mean()), 5))
print('\nsignificance', json.dumps(R['T9_significance'], indent=1))

# ------------------------------------------------------------------ figures
cum_s, cum_e = (1 + strat).cumprod(), (1 + ew).cumprod()

f, a = plt.subplots(figsize=(7.0, 3.4))
a.plot(cum_s.index, cum_s, color=C['strat'], lw=1.4, label='Regime-conditional strategy')
a.plot(cum_e.index, cum_e, color=C['ew'], lw=1.2, label='Equal-weight benchmark')
a.set_ylabel('Growth of one unit of capital')
a.set_yscale('log')
a.set_yticks([0.6, 1, 1.5, 2, 3])
a.set_yticklabels(['0.6', '1.0', '1.5', '2.0', '3.0'])
a.legend(loc='upper left')
f.savefig(os.path.join(IMG, 'fig_cumret.png')); plt.close(f)

f, a = plt.subplots(figsize=(7.0, 2.8))
a.fill_between(cum_e.index, (cum_e / cum_e.cummax() - 1) * 100, 0, color=C['ew'], alpha=.6,
               label='Equal-weight benchmark')
a.plot(cum_s.index, (cum_s / cum_s.cummax() - 1) * 100, color=C['strat'], lw=1.2,
       label='Regime-conditional strategy')
a.set_ylabel('Drawdown (%)')
a.legend(loc='lower left')
f.savefig(os.path.join(IMG, 'fig_drawdown.png')); plt.close(f)

f, (a1, a2) = plt.subplots(2, 1, figsize=(7.0, 4.2), sharex=True,
                           gridspec_kw=dict(height_ratios=[2, 1]))
vol = rf['Volatility'] * np.sqrt(252) * 100
for k, nm in [(0, 'Regime 0 (calm)'), (1, 'Regime 1 (transitional)'), (2, 'Regime 2 (stress)')]:
    m = (lab == k).values
    a1.scatter(vol.index[m], vol.values[m], s=1.6, color=C['r%d' % k], label=nm)
a1.set_ylabel('Annualised volatility (%)')
a1.legend(loc='upper right', markerscale=5, ncol=3)
a2.plot(rf.index, rf['Momentum'] * 252 * 100, color='#333333', lw=0.7)
a2.axhline(0, color='k', lw=0.5)
a2.set_ylabel('Momentum (% p.a.)')
f.savefig(os.path.join(IMG, 'fig_regimes.png')); plt.close(f)

f, a = plt.subplots(figsize=(7.0, 3.4))
Wp = W.copy()
Wp['Cash'] = 1.0 - Wp.sum(axis=1)
Wd = Wp.reindex(idx.union(Wp.index)).ffill().reindex(idx)
a.stackplot(Wd.index, [Wd[c].values * 100 for c in Wd.columns], labels=list(Wd.columns),
            colors=plt.cm.tab20(np.linspace(0, 1, len(Wd.columns))), lw=0)
a.set_ylabel('Portfolio weight (%)')
a.set_ylim(0, 100)
a.legend(loc='upper center', ncol=6, fontsize=6.5, bbox_to_anchor=(0.5, 1.22))
f.savefig(os.path.join(IMG, 'fig_alloc.png')); plt.close(f)

f, a = plt.subplots(figsize=(7.0, 2.8))
tt = D['tech'].loc[idx].abs().mean(axis=1).rolling(21).mean()
ss = D['sent'].loc[idx].abs().mean(axis=1).rolling(21).mean()
a.plot(tt.index, tt, color=C['strat'], lw=1.0, label='Technical composite')
a.plot(ss.index, ss, color=C['r1'], lw=1.0, label='Volume/volatility composite')
a.set_ylabel('Mean absolute\ncross-sectional score')
a.legend(loc='upper right', ncol=2)
f.savefig(os.path.join(IMG, 'fig_signal.png')); plt.close(f)

f, a = plt.subplots(figsize=(7.0, 2.8))
rs = strat.rolling(252).mean() / strat.rolling(252).std() * np.sqrt(252)
re_ = ew.rolling(252).mean() / ew.rolling(252).std() * np.sqrt(252)
a.plot(rs.index, rs, color=C['strat'], lw=1.2, label='Regime-conditional strategy')
a.plot(re_.index, re_, color=C['ew'], lw=1.2, label='Equal-weight benchmark')
a.axhline(0, color='k', lw=0.5)
a.set_ylabel('252-day rolling Sharpe ratio')
a.legend(loc='lower right', ncol=2)
f.savefig(os.path.join(IMG, 'fig_rollsharpe.png')); plt.close(f)

rsd = rs.dropna()
R['rolling_sharpe'] = dict(
    strat_frac_positive=round(float((rsd > 0).mean()), 3),
    ew_frac_positive=round(float((re_.dropna() > 0).mean()), 3),
    strat_frac_above_ew=round(float((rsd > re_.reindex(rsd.index)).mean()), 3))
R['period'] = [str(idx[0].date()), str(idx[-1].date()), int(len(idx))]
json.dump(R, open(os.path.join(OUT, 'paper_tables.json'), 'w'), indent=2, default=str)
print('\nsaved paper_tables.json and 6 figures at 300 dpi ->', IMG)
