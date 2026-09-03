"""Does the Sharpe gap become significant on the extended sample?"""
import numpy as np, pandas as pd, warnings
from scipy import stats
import pipeline_core as P
warnings.filterwarnings('ignore')
P.START, P.END = '2007-01-01', '2026-08-27'
P._CACHE.clear()
s, _, _ = P.backtest(P.MULTI)
b = P.benchmarks(P.MULTI, s.index); ew = b['ew']

def sharpe(x):
    c = (1+x).cumprod(); ann = c.iloc[-1]**(252/len(x))-1
    return ann/(x.std()*np.sqrt(252))

def boot(a, e, n=2000, L=21, seed=42):
    rng = np.random.default_rng(seed); T = len(a); out = []
    for _ in range(n):
        i = []
        while len(i) < T:
            st = rng.integers(0, T); ln = rng.geometric(1/L)
            i.extend(((st+np.arange(ln)) % T).tolist())
        i = np.array(i[:T])
        out.append(sharpe(pd.Series(a[i])) - sharpe(pd.Series(e[i])))
    return np.array(out)

for tag, idx in [('EXTENDED 2008-2026', s.index),
                 ('ORIGINAL 2008-2023', s.index[s.index <= '2023-12-31'])]:
    a, e = s.loc[idx].values, ew.loc[idx].values
    d = a - e
    t, p = stats.ttest_1samp(d, 0)
    bs = boot(a, e)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"\n{tag}   n={len(a)}")
    print(f"  Sharpe  strat {sharpe(pd.Series(a)):.3f}  ew {sharpe(pd.Series(e)):.3f} "
          f" diff {sharpe(pd.Series(a))-sharpe(pd.Series(e)):+.3f}")
    print(f"  excess return  {d.mean()*252*100:+.2f}%/yr   paired t={t:.3f}  p={p:.3f}")
    print(f"  bootstrap Sharpe diff  median {np.median(bs):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]"
          f"   {'SIGNIFICANT' if lo > 0 else 'not significant'}")
