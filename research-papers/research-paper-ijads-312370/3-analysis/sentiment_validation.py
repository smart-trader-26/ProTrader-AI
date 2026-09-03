"""External validation of the market-derived sentiment composite.

The second composite in the framework is built from turnover, short-horizon return and
two dispersion measures. This script asks whether that construction behaves like a
recognised measure of investor sentiment, by correlating its market-level aggregate
against four independent published series, and whether the identified market states
separate on it.

Nothing here feeds the backtest. It is a construct-validity check, run on the same
sample as the main analysis, and it writes sentiment_results.json.
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
ETF = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'IEF', 'LQD', 'GLD', 'VNQ']
START, END = '2007-01-01', '2026-08-27'
EVAL_START = '2008-01-07'
SENT_SIGN = {'mom5': +1, 'vol_ratio': +1, 'stress': -1, 'vol_of_vol': -1}

FRED = [
    ('UMCSENT', 'Michigan consumer sentiment', 'survey', 'M', +1),
    ('STLFSI4', 'St. Louis Fed financial stress index', 'market', 'W', -1),
    ('NFCI', 'Chicago Fed national financial conditions index', 'market', 'W', -1),
]


def zs(s):
    return (s - s.mean()) / s.std()


def build_composite(md, expanding=False):
    """Market-level aggregate of the sentiment composite."""
    agg = {}
    for f in SENT_SIGN:
        per = {}
        for t in ETF:
            close, vt = md['Close'][t], md['Volume'][t]
            if f == 'mom5':
                per[t] = close.pct_change(5)
            elif f == 'vol_ratio':
                per[t] = vt / vt.rolling(20).mean()
            elif f == 'stress':
                per[t] = close.pct_change().rolling(20).std() * np.sqrt(252)
            else:
                per[t] = vt.pct_change().rolling(20).std()
        agg[f] = pd.DataFrame(per).mean(axis=1)
    if expanding:
        z = {f: (s - s.expanding(252).mean()) / s.expanding(252).std()
             for f, s in agg.items()}
    else:
        z = {f: zs(s) for f, s in agg.items()}
    S = sum(SENT_SIGN[f] * z[f] for f in SENT_SIGN) / len(SENT_SIGN)
    return S.dropna()


def hac_corr(a, b, lags):
    """Correlation plus its Newey-West t-statistic (slope of standardised regression)."""
    j = pd.concat([a.rename('a'), b.rename('b')], axis=1).dropna()
    if len(j) < 30:
        return None
    x, y = zs(j['a']), zs(j['b'])
    m = sm.OLS(y.values, sm.add_constant(x.values)).fit(
        cov_type='HAC', cov_kwds={'maxlags': lags})
    return dict(n=int(len(j)), r=float(j['a'].corr(j['b'])),
                t=float(m.tvalues[1]), p=float(m.pvalues[1]))


def main():
    print('downloading the ETF panel ...')
    md = yf.download(ETF, start=START, end=END, auto_adjust=True,
                     progress=False, threads=False)

    S = build_composite(md)
    S = S[S.index >= EVAL_START]
    S_exp = build_composite(md, expanding=True)
    S_exp = S_exp[S_exp.index >= EVAL_START]
    print('composite: %d sessions, %s -> %s'
          % (len(S), S.index[0].date(), S.index[-1].date()))

    out = {'sample': {'n': int(len(S)),
                      'start': str(S.index[0].date()),
                      'end': str(S.index[-1].date())},
           'benchmarks': [], 'stability': [], 'regimes': {}}

    # ---------------- 1. convergent validity ----------------
    series = []
    vix = yf.download('^VIX', start=START, end=END, auto_adjust=True,
                      progress=False, threads=False)['Close']
    vix = vix.iloc[:, 0] if isinstance(vix, pd.DataFrame) else vix
    series.append(('Cboe volatility index (VIX)', 'market', 'D', -1, vix.dropna()))

    for sid, name, kind, freq, sign in FRED:
        df = pd.read_csv('https://fred.stlouisfed.org/graph/fredgraph.csv?id=' + sid)
        df.columns = ['date', 'val']
        v = pd.to_numeric(df['val'], errors='coerce')
        s = pd.Series(v.values, index=pd.to_datetime(df['date'])).dropna()
        series.append((name, kind, freq, sign, s))

    LAGS = {'D': 21, 'W': 8, 'M': 6}
    print()
    print('%-48s %5s %6s %7s %7s %7s' % ('benchmark', 'n', 'r', 't', 'r(chg)', 't(chg)'))
    print('-' * 84)
    for name, kind, freq, sign, s in series:
        s = s[(s.index >= S.index[0]) & (s.index <= S.index[-1])]
        a = S if freq == 'D' else S.resample(freq).mean()
        b = s if freq == 'D' else s.resample(freq).mean()
        lvl = hac_corr(a, b, LAGS[freq])
        chg = hac_corr(a.diff().dropna(), b.diff().dropna(), LAGS[freq])
        aex = S_exp if freq == 'D' else S_exp.resample(freq).mean()
        rex = hac_corr(aex, b, LAGS[freq])
        rec = dict(name=name, kind=kind, freq=freq, expected_sign=sign,
                   level=lvl, change=chg, r_expanding=(rex or {}).get('r'))
        rec['sign_ok'] = bool(np.sign(lvl['r']) == sign)
        out['benchmarks'].append(rec)
        print('%-48s %5d %6.3f %7.2f %7.3f %7.2f'
              % (name[:48], lvl['n'], lvl['r'], lvl['t'], chg['r'], chg['t']))

    # ---------------- 2. sub-period stability ----------------
    print()
    print('sub-period stability (correlation with each benchmark)')
    cuts = [('2008-01-07', '2013-12-31'), ('2014-01-01', '2019-12-31'),
            ('2020-01-01', '2023-12-31'), ('2024-01-01', '2026-08-26')]
    for lo, hi in cuts:
        row = {'from': lo, 'to': hi, 'r': {}}
        Ss = S[(S.index >= lo) & (S.index <= hi)]
        for name, kind, freq, sign, s in series:
            a = Ss if freq == 'D' else Ss.resample(freq).mean()
            b = s if freq == 'D' else s.resample(freq).mean()
            j = pd.concat([a.rename('a'), b.rename('b')], axis=1).dropna()
            row['r'][name] = float(j['a'].corr(j['b'])) if len(j) >= 20 else None
        out['stability'].append(row)
        vals = ['%s %+.3f' % (n.split()[0][:9], v) for n, v in row['r'].items() if v is not None]
        print('  %s -> %s : %s' % (lo, hi, '  '.join(vals)))

    # ---------------- 3. do the identified states separate on sentiment? ----------
    cache = os.path.join(HERE, '_final_cache_ext.pkl')
    if os.path.exists(cache):
        import pickle
        rg = pickle.load(open(cache, 'rb'))['strat'][2]
        j = pd.concat([S.rename('S'), rg.rename('rg')], axis=1).dropna()
        from scipy import stats as st
        groups = [j.loc[j['rg'] == k, 'S'].values for k in sorted(j['rg'].unique())]
        F, pF = st.f_oneway(*groups)
        H, pH = st.kruskal(*groups)
        out['regimes'] = dict(
            n=int(len(j)), F=float(F), p_F=float(pF), H=float(H), p_H=float(pH),
            by_regime=[dict(regime=int(k), n=int(len(g)), mean=float(g.mean()),
                            sd=float(g.std()))
                       for k, g in zip(sorted(j['rg'].unique()), groups)])
        print()
        print('sentiment by identified market state (n=%d rebalance dates)' % len(j))
        for r in out['regimes']['by_regime']:
            print('  regime %d  n=%3d  mean %+.3f  sd %.3f'
                  % (r['regime'], r['n'], r['mean'], r['sd']))
        print('  ANOVA F=%.2f p=%.3g   Kruskal-Wallis H=%.2f p=%.3g' % (F, pF, H, pH))

    with open(os.path.join(HERE, 'sentiment_results.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print('\nwrote sentiment_results.json')


if __name__ == '__main__':
    main()
