"""Lever 1: regime-conditional covariance, which section 3.2.3 of the paper claims but the
code never implemented. At each rebalance the GMM is fit on past data only, that same model
labels all prior history, and the covariance is estimated from same-regime days.
"""
import json, os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
import cvxpy as cp
import pipeline_core as P

OUT = os.path.dirname(os.path.abspath(__file__))
MULTI = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'IEF', 'LQD', 'GLD', 'VNQ']
SECT9 = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB']
RES = {}


def sh(x):
    return float(x.mean() / x.std() * np.sqrt(252)) if x.std() else 0.0


def mdd(x):
    c = (1 + x).cumprod()
    return float((c / c.cummax() - 1).min())


def backtest(universe, cfg=None, regime_cov=False, min_regime_days=120, blend=0.5):
    cfg = {**P.DEF, **(cfg or {})}
    D = P.load(universe)
    returns, tech, sent, conf, rf, common = (D['returns'], D['tech'], D['sent'],
                                             D['conf'], D['regime_feat'], D['common'])
    cov_lb, rebal = cfg['cov_lookback'], cfg['rebal']
    cost = cfg['cost_bps'] / 10000.0
    icf, lam, w_max = cfg['ic'], cfg['lam'], cfg['w_max']
    dates = common[common >= returns.index[max(cov_lb, 252)]]
    rebal_dates = dates[::rebal]
    n = len(universe)
    w_prev = np.zeros(n)
    daily, rlog = [], {}

    for k, d in enumerate(rebal_dates):
        hist_all = returns[returns.index < d]
        if len(hist_all) < cov_lb:
            continue
        rfh = rf[rf.index < d]
        if len(rfh) < 252:
            rg, labels = 1, None
        else:
            sc = StandardScaler().fit(rfh)
            g = GaussianMixture(n_components=cfg['n_regimes'], random_state=cfg['seed'],
                                n_init=3).fit(sc.transform(rfh))
            remap = {o: nw for nw, o in enumerate(np.argsort(g.means_[:, 0]))}
            labels = pd.Series(g.predict(sc.transform(rfh)), index=rfh.index).map(remap)
            rg = int(labels.iloc[-1])

        # ---- covariance
        base = LedoitWolf().fit(hist_all.iloc[-cov_lb:]).covariance_ * 252.0
        if regime_cov and labels is not None:
            same = labels[labels == rg].index
            sub = hist_all.reindex(same).dropna()
            if len(sub) >= min_regime_days:
                Sr = LedoitWolf().fit(sub).covariance_ * 252.0
                Sig = blend * Sr + (1 - blend) * base       # shrink toward the trailing estimate
            else:
                Sig = base
        else:
            Sig = base
        sig_i = np.sqrt(np.diag(Sig))

        wt, ws = cfg['fuse'][rg]
        tgt = cfg['target_vol'][rg]
        z = np.nan_to_num((wt * tech.loc[d].values + ws * sent.loc[d].values) * conf.loc[d].values)
        alpha = icf * sig_i * z

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
        wv = (min(1.0, tgt / sv) if sv > 0 else 0.0) * sleeve
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
        rlog[d] = rg
    return pd.Series(dict(daily)).sort_index(), pd.Series(rlog)


for label, univ, start in [('MULTI-ASSET 2008-2023', MULTI, '2007-01-01'),
                           ('SECTORS 2008-2023', SECT9, '2007-01-01')]:
    P.START = start
    P._CACHE.clear()
    print("\n" + "=" * 78)
    print(label)
    print("=" * 78)
    base_s, _ = backtest(univ, regime_cov=False)
    rc_s, _ = backtest(univ, regime_cov=True)
    idx = base_s.index.intersection(rc_s.index)
    base_s, rc_s = base_s.loc[idx], rc_s.loc[idx]
    ew = P.benchmarks(univ, idx)['ew']

    tbl = pd.DataFrame([P.stats_of(rc_s, 'regime-conditional covariance'),
                        P.stats_of(base_s, 'single trailing covariance'),
                        P.stats_of(ew, 'equal-weight')]).set_index('name')
    print(tbl.round(4).to_string())
    dif = rc_s - base_s
    t_, p_ = stats.ttest_1samp(dif, 0)
    print(f"  regime-cov minus single-cov: {dif.mean() * 252:+.3%}/yr  t {t_:.3f} (p {p_:.4f})")
    print(f"  Sharpe {sh(rc_s):.3f} vs {sh(base_s):.3f}   maxDD {mdd(rc_s):+.4f} vs {mdd(base_s):+.4f}")
    print(f"  helps? {'YES' if sh(rc_s) > sh(base_s) else 'no'}")

    # sweep over the blend weight and the minimum-sample threshold
    print("  sweep:")
    sw = []
    for bl in [0.25, 0.5, 0.75, 1.0]:
        s_, _ = backtest(univ, regime_cov=True, blend=bl)
        s_ = s_.reindex(idx).dropna()
        sw.append(dict(blend=bl, sharpe=round(sh(s_), 4), mdd=round(mdd(s_), 4),
                       beats_single=bool(sh(s_) > sh(base_s))))
        print(f"    blend {bl:<5} Sharpe {sh(s_):.3f}  maxDD {mdd(s_):+.4f}  "
              f"{'beats single-cov' if sh(s_) > sh(base_s) else ''}")
    RES[label] = dict(main=tbl.round(6).to_dict('index'), sweep=sw,
                      diff_ann=float(dif.mean() * 252), t=float(t_), p=float(p_),
                      sharpe_rc=sh(rc_s), sharpe_base=sh(base_s),
                      mdd_rc=mdd(rc_s), mdd_base=mdd(base_s))

json.dump(RES, open(os.path.join(OUT, 'regime_cov_results.json'), 'w'), indent=2, default=str)
print("\nsaved regime_cov_results.json")
