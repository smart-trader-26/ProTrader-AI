"""Does extending the sample to Aug 2026 help or hurt? Full rerun + 2024-26 holdout."""
import numpy as np, pandas as pd, json, warnings
import pipeline_core as P
warnings.filterwarnings('ignore')

P.START, P.END = '2007-01-01', '2026-08-27'
P._CACHE.clear()

s, w, rg = P.backtest(P.MULTI)                       # primary spec, extended data
sn, _, _ = P.backtest(P.MULTI, use_regime=False, cfg={'flat_mult': 1.0})
b = P.benchmarks(P.MULTI, s.index)
ew, iv = b['ew'], b['iv']
ewm = ew * (s.std() / ew.std())                      # equal weight at strategy vol

def blk(idx, tag):
    rows = []
    for nm, x in [('Strategy', s), ('Equal-weight', ew), ('Inverse-vol', iv),
                  ('Vol-matched EW', ewm), ('No-regime (full exp.)', sn)]:
        y = x.loc[idx].dropna()
        if len(y) < 30:
            continue
        st = P.stats_of(y, nm)
        rows.append((nm, st))
    print(f"\n=== {tag}  ({idx[0].date()} -> {idx[-1].date()}, {len(idx)} sessions) ===")
    print(f"{'':<24}{'CAGR':>9}{'Vol':>9}{'Sharpe':>9}{'MaxDD':>10}{'Calmar':>9}")
    for nm, st in rows:
        print(f"{nm:<24}{st['CAGR']*100:>8.2f}%{st['Vol']*100:>8.2f}%"
              f"{st['Sharpe']:>9.3f}{st['MaxDD']*100:>9.2f}%{st['Calmar']:>9.3f}")
    return {nm: st for nm, st in rows}

out = {}
out['full_2008_2026'] = blk(s.index, 'EXTENDED FULL SAMPLE 2008-2026')
old = s.index[s.index <= '2023-12-31']
new = s.index[s.index >= '2024-01-01']
out['orig_2008_2023'] = blk(old, 'ORIGINAL WINDOW 2008-2023 (control)')
out['holdout_2024_2026'] = blk(new, 'HOLDOUT 2024-2026 (never used)')

print("\n=== per-year ===")
yr = pd.DataFrame({'s': s, 'ew': ew}).dropna()
for y, g in yr.groupby(yr.index.year):
    cs = (1+g['s']).prod()-1; ce = (1+g['ew']).prod()-1
    ds = ((1+g['s']).cumprod()/(1+g['s']).cumprod().cummax()-1).min()
    de = ((1+g['ew']).cumprod()/(1+g['ew']).cumprod().cummax()-1).min()
    print(f"{y}  strat {cs*100:>7.2f}%  ew {ce*100:>7.2f}%   dd {ds*100:>7.2f}% vs {de*100:>7.2f}%"
          f"   {'WIN' if ds>de else '   '}")
json.dump({k: {n: v for n, v in d.items()} for k, d in out.items()},
          open('extend_results.json', 'w'), indent=1)
print("\nwrote extend_results.json")
