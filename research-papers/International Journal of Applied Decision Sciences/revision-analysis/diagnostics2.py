"""Follow-ups: permutation-safe leakage check, net-of-cost P&L, window sensitivity."""
import json, os, warnings
import numpy as np
import pandas as pd
import yfinance as yf
import ta
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
import cvxpy as cp

warnings.filterwarnings('ignore')
OUT = os.path.dirname(os.path.abspath(__file__))
R = {}
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'XOM']

md = yf.download(TICKERS, start='2018-01-01', end='2024-01-01', auto_adjust=True,
                 progress=False, threads=False)
returns = md['Close'].pct_change().dropna()
volume = md['Volume']

# rebuild features exactly as the notebook does
tech = {}
for t in TICKERS:
    high, low, close, vt = md['High'][t], md['Low'][t], md['Close'][t], md['Volume'][t]
    bb = ta.volatility.BollingerBands(close)
    tech[t] = pd.DataFrame({
        f'{t}_RSI': ta.momentum.RSIIndicator(close).rsi(),
        f'{t}_MACD': ta.trend.MACD(close).macd(),
        f'{t}_Stoch': ta.momentum.StochasticOscillator(high, low, close).stoch(),
        f'{t}_BB_Width': (bb.bollinger_hband() - bb.bollinger_lband()) / close,
        f'{t}_ATR': ta.volatility.AverageTrueRange(high, low, close).average_true_range(),
        f'{t}_MFI': ta.volume.MFIIndicator(high, low, close, vt).money_flow_index(),
        f'{t}_OBV': ta.volume.OnBalanceVolumeIndicator(close, vt).on_balance_volume()},
        index=close.index)
technical = pd.concat(tech.values(), axis=1).dropna()

sent = {}
for t in TICKERS:
    rt, vt = returns[t], volume[t]
    sent[t] = pd.DataFrame({
        f'{t}_Price_Momentum': rt.rolling(5).mean(),
        f'{t}_Volatility_Regime': rt.rolling(20).std(),
        f'{t}_Volume_Ratio': vt / vt.rolling(20).mean(),
        f'{t}_Volume_Volatility': vt.pct_change().rolling(20).std(),
        f'{t}_Market_Stress': rt.rolling(20).std() * np.sqrt(252)}, index=rt.index)
sentiment = pd.concat(sent.values(), axis=1).dropna()

vol20 = returns.rolling(20).std().dropna()
mom10 = returns.rolling(10).mean().dropna()
ix = vol20.index.intersection(mom10.index)
feat = pd.DataFrame({'Volatility': vol20.loc[ix].mean(axis=1),
                     'Momentum': mom10.loc[ix].mean(axis=1)}).dropna()

# ---------------------------------------------------------------- D1 leakage, permutation-safe
def fit_labels(train_feat, all_feat, seed=42):
    sc = StandardScaler().fit(train_feat)
    g = GaussianMixture(n_components=3, random_state=seed).fit(sc.transform(train_feat))
    order = np.argsort(g.means_[:, 0])                      # sort components by mean volatility
    remap = {old: new for new, old in enumerate(order)}
    return pd.Series(g.predict(sc.transform(all_feat)), index=all_feat.index).map(remap)

full_lab = fit_labels(feat, feat)
split = feat.index[int(len(feat) * 0.5)]
oos_lab = fit_labels(feat[feat.index < split], feat)
test = feat.index >= split
agree_all = float((full_lab == oos_lab).mean())
agree_oos = float((full_lab[test] == oos_lab[test]).mean())
print(f"D1  vol-sorted labels: full-sample vs train-only agree {agree_all:.1%} overall, "
      f"{agree_oos:.1%} on the out-of-sample half")
R['D1'] = dict(agree_all=agree_all, agree_oos=agree_oos)

# ---------------------------------------------------------------- shared backtest
def run(lookback=63, cost_bps=0.0, regime_labels=None, seed=42):
    if regime_labels is None:
        sc = StandardScaler()
        g = GaussianMixture(n_components=3, random_state=seed)
        lab = pd.Series(g.fit_predict(sc.fit_transform(feat)), index=feat.index)
    else:
        lab = regime_labels
    rd = pd.DataFrame({'Regime': lab})
    idx = technical.index.intersection(sentiment.index).intersection(rd.index)
    te, se, rg = technical.loc[idx], sentiment.loc[idx], rd.loc[idx]
    rw = {0: (.3, .7), 1: (.5, .5), 2: (.7, .3)}
    fused, conf = {}, {}
    for t in TICKERS:
        ts = te[[c for c in te.columns if c.startswith(t)]].mean(axis=1)
        ss = se[[c for c in se.columns if c.startswith(t)]].mean(axis=1)
        f = pd.Series(index=idx, dtype=float)
        c = pd.Series(index=idx, dtype=float)
        for k, (a, b) in rw.items():
            m = rg['Regime'] == k
            f[m] = a * ts[m] + b * ss[m]
            cc = ts[m].corr(ss[m])
            c[m] = 0.5 + 0.5 * cc if not np.isnan(cc) else 0.5
        fused[t], conf[t] = f, c
    F, C = pd.DataFrame(fused), pd.DataFrame(conf)

    ii = returns.index.intersection(F.index)
    ra = returns.loc[ii]
    W = pd.DataFrame(index=ii, columns=returns.columns, dtype=float)
    P = pd.Series(index=ii, dtype=float)
    prev = np.zeros(len(TICKERS))
    for i in range(lookback, len(ii)):
        d = ii[i]
        r = rg['Regime'].iloc[i]
        rt = 0.10 if r == 0 else (0.15 if r == 1 else 0.20)
        tn = 0.05 if r == 0 else (0.10 if r == 1 else 0.15)
        er = (F.iloc[i] * C.iloc[i]).values
        cov = LedoitWolf().fit(ra.iloc[i - lookback:i]).covariance_
        w = cp.Variable(len(TICKERS))
        risk = cp.quad_form(w, cp.psd_wrap(cov))
        prob = cp.Problem(cp.Maximize(er @ w - 0.5 * risk),
                          [cp.sum(w) == 1, w >= 0, risk <= rt ** 2, cp.norm(w, 1) <= 1 + tn])
        prob.solve()
        if prob.status == 'optimal':
            wv = w.value
            W.loc[d] = wv
            if i < len(ii) - 1:
                gross = ra.iloc[i + 1] @ wv
                P.loc[d] = gross - np.abs(wv - prev).sum() * cost_bps / 10000.0
                prev = wv
    W, P = W.dropna(), P.dropna()
    bm = returns.mean(axis=1).loc[P.index]
    def st(x):
        cum = (1 + x).cumprod()
        ann = cum.iloc[-1] ** (252 / len(x)) - 1
        v = x.std() * np.sqrt(252)
        dd = (cum / cum.cummax() - 1).min()
        return dict(ann=float(ann), vol=float(v), sharpe=float(ann / v), mdd=float(dd))
    return st(P), st(bm), W, P

# ---------------------------------------------------------------- D2 transaction costs
print("\nD2  net-of-cost performance (the paper claims a cost-aware optimiser):")
R['D2'] = {}
for bps in [0, 5, 10, 25]:
    p, b, _, _ = run(63, bps)
    print(f"    {bps:2d} bps -> strategy CAGR {p['ann']:+.2%}  Sharpe {p['sharpe']:.3f}  maxDD {p['mdd']:.2%}"
          f"   |  equal-weight CAGR {b['ann']:+.2%}  Sharpe {b['sharpe']:.3f}  maxDD {b['mdd']:.2%}")
    R['D2'][bps] = dict(strategy=p, benchmark=b)

# ---------------------------------------------------------------- D3 window sensitivity
print("\nD3  sensitivity to the covariance lookback window (reviewer 2, point 3):")
R['D3'] = {}
for lb in [21, 63, 126, 252]:
    p, b, _, _ = run(lb, 0)
    print(f"    lookback {lb:3d}d -> strategy Sharpe {p['sharpe']:.3f}  CAGR {p['ann']:+.2%}  maxDD {p['mdd']:.2%}"
          f"   |  benchmark Sharpe {b['sharpe']:.3f}")
    R['D3'][lb] = dict(strategy=p, benchmark=b)

# ---------------------------------------------------------------- D4 seed sensitivity of the GMM
print("\nD4  sensitivity to the GMM random seed:")
R['D4'] = {}
for sd in [0, 7, 42, 123]:
    p, b, _, _ = run(63, 0, seed=sd)
    print(f"    seed {sd:3d} -> strategy Sharpe {p['sharpe']:.3f}  CAGR {p['ann']:+.2%}  maxDD {p['mdd']:.2%}")
    R['D4'][sd] = dict(strategy=p)

json.dump(R, open(os.path.join(OUT, 'diagnostics2.json'), 'w'), indent=2)
print("\nsaved diagnostics2.json")
