"""Out-of-sample IC test: does the fused signal keep its (negative) sign after 2023?"""
import numpy as np, pandas as pd, warnings
from scipy import stats
import pipeline_core as P
warnings.filterwarnings('ignore')

P.START, P.END = '2007-01-01', '2026-08-27'
P._CACHE.clear()
D = P.load(P.MULTI)
tech, sent, ret = D['tech'], D['sent'], D['returns']
H = 21
fwd = (1 + ret).rolling(H).apply(np.prod, raw=True).shift(-H) - 1
idx = tech.index.intersection(fwd.dropna(how='all').index)
tech, sent, fwd = tech.loc[idx], sent.loc[idx], fwd.loc[idx]
fused = 0.5 * tech + 0.5 * sent

def ic_series(sig):
    out = {}
    for d in sig.index:
        a, b = sig.loc[d], fwd.loc[d]
        m = a.notna() & b.notna()
        if m.sum() >= 5:
            out[d] = stats.spearmanr(a[m], b[m]).correlation
    return pd.Series(out).dropna()

def nw_t(x, lags=10):
    x = np.asarray(x); n = len(x); mu = x.mean(); e = x - mu
    g0 = (e @ e) / n; s = g0
    for L in range(1, lags + 1):
        g = (e[L:] @ e[:-L]) / n
        s += 2 * (1 - L / (lags + 1)) * g
    return mu / np.sqrt(s / n)

periods = {
    'IN-SAMPLE 2008-2023': ('2008-01-01', '2023-12-31'),
    'HOLDOUT  2024-2026': ('2024-01-01', '2026-12-31'),
}
print(f"data span: {idx[0].date()} -> {idx[-1].date()}   horizon {H}d")
print()
print(f"{'period':<22}{'signal':<12}{'IC':>9}{'NW t':>9}{'n':>7}")
print('-' * 60)
for label, (a, b) in periods.items():
    for nm, sig in [('technical', tech), ('sentiment', sent), ('fused', fused)]:
        s = ic_series(sig.loc[a:b])
        if len(s) == 0:
            continue
        print(f"{label:<22}{nm:<12}{s.mean():>9.4f}{nw_t(s.values):>9.2f}{len(s):>7}")
    print()
