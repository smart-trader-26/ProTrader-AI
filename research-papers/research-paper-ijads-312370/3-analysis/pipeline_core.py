"""Corrected IJADS pipeline, parameterised so several universes/configs can be run.

Same nine fixes as fixed_pipeline.py; refactored into functions so the defence analysis
can sweep universes, seeds, windows and rebalance frequencies.
"""
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import ta
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
import cvxpy as cp

warnings.filterwarnings('ignore')

MEGA = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'XOM']
SECT = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE']
START, END = '2018-01-01', '2024-01-01'

DEF = dict(cov_lookback=252, rebal=21, ic=0.05, lam=2.0, w_max=0.25, cost_bps=10.0,
           n_regimes=3, seed=42,
           target_vol={0: 0.20, 1: 0.15, 2: 0.10},
           # relative regime risk budget: fraction of the sleeve's own volatility retained.
           # None falls back to the absolute target_vol levels above, which do not bind on a
           # diversified multi-asset sleeve (1 of 192 rebalances) and leave the layer inert.
           vol_mult={0: 1.00, 1: 0.75, 2: 0.50},
           fuse={0: (0.7, 0.3), 1: (0.5, 0.5), 2: (0.3, 0.7)})

TECH_DIR = ['rsi', 'macd', 'stoch', 'mfi', 'obv_flow']
SENT_SIGN = {'mom5': +1, 'vol_ratio': +1, 'stress': -1, 'vol_of_vol': -1}
_CACHE = {}


def xs_z(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)


def load(universe):
    key = tuple(universe)
    if key in _CACHE:
        return _CACHE[key]
    md = yf.download(list(universe), start=START, end=END, auto_adjust=True,
                     progress=False, threads=False)
    returns = md['Close'].pct_change().dropna()
    raw = {}
    for t in universe:
        high, low, close, vt = md['High'][t], md['Low'][t], md['Close'][t], md['Volume'][t]
        bb = ta.volatility.BollingerBands(close)
        raw[t] = pd.DataFrame({
            'rsi': ta.momentum.RSIIndicator(close).rsi() - 50.0,
            'macd': ta.trend.MACD(close).macd() / close,
            'stoch': ta.momentum.StochasticOscillator(high, low, close).stoch() - 50.0,
            'mfi': ta.volume.MFIIndicator(high, low, close, vt).money_flow_index() - 50.0,
            'obv_flow': ta.volume.OnBalanceVolumeIndicator(close, vt).on_balance_volume()
                        .diff(20) / vt.rolling(20).mean(),
            'mom5': close.pct_change(5),
            'vol_ratio': vt / vt.rolling(20).mean(),
            'stress': close.pct_change().rolling(20).std() * np.sqrt(252),
            'vol_of_vol': vt.pct_change().rolling(20).std(),
        }, index=close.index)
    panel = {f: xs_z(pd.DataFrame({t: raw[t][f] for t in universe}))
             for f in raw[universe[0]].columns}
    tech = (sum(panel[f] for f in TECH_DIR) / len(TECH_DIR)).dropna()
    sent = (sum(SENT_SIGN[f] * panel[f] for f in SENT_SIGN) / len(SENT_SIGN)).dropna()
    common = tech.index.intersection(sent.index).intersection(returns.index)
    tech, sent = tech.loc[common], sent.loc[common]
    agree = pd.DataFrame({t: tech[t].rolling(63).corr(sent[t]) for t in universe})
    conf = (0.5 + 0.5 * agree).clip(0.1, 1.0).fillna(0.5)
    rf = pd.DataFrame({'Volatility': returns.rolling(20).std().mean(axis=1),
                       'Momentum': returns.rolling(10).mean().mean(axis=1)}).dropna()
    out = dict(returns=returns, tech=tech, sent=sent, conf=conf, regime_feat=rf, common=common)
    _CACHE[key] = out
    return out


def regime_at(rf, date, n_regimes, seed, min_obs=252):
    hist = rf[rf.index < date]
    if len(hist) < min_obs:
        return 1
    sc = StandardScaler().fit(hist)
    g = GaussianMixture(n_components=n_regimes, random_state=seed, n_init=3).fit(sc.transform(hist))
    remap = {old: new for new, old in enumerate(np.argsort(g.means_[:, 0]))}
    cur = rf.loc[rf.index <= date].iloc[-1:]
    return int(remap[int(g.predict(sc.transform(cur))[0])])


def backtest(universe, cfg=None, use_regime=True, signal_on=True):
    cfg = {**DEF, **(cfg or {})}
    D = load(universe)
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
        rg = regime_at(rf, d, cfg['n_regimes'], cfg['seed']) if use_regime else 1
        wt, ws = cfg['fuse'][rg] if use_regime else (0.5, 0.5)
        mult = cfg.get('vol_mult')
        tgt = (cfg['target_vol'][rg] if use_regime else 0.15) if mult is None else None

        if signal_on:
            z = np.nan_to_num((wt * tech.loc[d].values + ws * sent.loc[d].values)
                              * conf.loc[d].values)
            alpha = ic * sig_i * z
        else:
            alpha = np.zeros(n)

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
            k_scale = (min(1.0, tgt / sv) if sv > 0 else 0.0)
        elif use_regime:
            k_scale = mult[rg]
        else:
            k_scale = cfg.get('flat_mult', 1.0)
        wv = k_scale * sleeve
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

    s = pd.Series(dict(daily)).sort_index()
    return s, pd.DataFrame(wlog).T, pd.Series(rlog)


def stats_of(x, name=""):
    cum = (1 + x).cumprod()
    ann = cum.iloc[-1] ** (252 / len(x)) - 1
    vol = x.std() * np.sqrt(252)
    dd = (cum / cum.cummax() - 1).min()
    return dict(name=name, CAGR=float(ann), Vol=float(vol),
                Sharpe=float(ann / vol) if vol else 0.0, MaxDD=float(dd),
                Calmar=float(ann / abs(dd)) if dd else 0.0)


def benchmarks(universe, idx):
    r = load(universe)['returns'].loc[idx]
    ew = r.mean(axis=1)
    ivw = 1 / load(universe)['returns'].rolling(60).std().loc[idx]
    ivw = ivw.div(ivw.sum(axis=1), axis=0)
    return dict(ew=ew, iv=(r * ivw).sum(axis=1))
MULTI = ['SPY','QQQ','IWM','EFA','EEM','TLT','IEF','LQD','GLD','VNQ']
