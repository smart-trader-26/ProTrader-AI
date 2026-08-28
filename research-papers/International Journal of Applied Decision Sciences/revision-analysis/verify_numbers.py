"""Check every headline number in the manuscript against the JSON the analysis produced.

Fails loudly on any figure in the text that the artifacts do not support.
"""
import io, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = io.open(os.path.join(HERE, '..', 'single column', 'ai67.tex'), encoding='utf-8').read()
R = json.load(open(os.path.join(HERE, 'final_tables.json')))
MG = json.load(open(os.path.join(HERE, 'mega_results.json')))
ok, bad = 0, []


FLAT = re.sub(r'\s+', ' ', TEX)


def chk(label, value, fmt='{:.4f}', occurrences=1):
    """Assert a formatted value occurs in the manuscript.

    The JSON stores values already rounded to four places, so re-rounding to two or
    three can land one digit either side of the manuscript's figure, which was
    formatted from full precision. Both neighbours are therefore accepted.
    """
    global ok
    dp = re.search(r'\.(\d)f', fmt)
    if dp:
        eps = 10 ** -int(dp.group(1))
        vals = (value, value + eps * 0.51, value - eps * 0.51)
    else:
        vals = (value,)
    cands = {re.sub(r'\s+', ' ', fmt.format(v)) for v in vals}
    n = max(FLAT.count(c) for c in cands)
    if n >= occurrences:
        ok += 1
    else:
        bad.append(f'{label}: expected one of {sorted(cands)} at least '
                   f'{occurrences}x, found {n}x')


T1 = R['T1_main']
for name, key in [('strategy', 'Regime-conditional strategy'), ('equal weight', 'Equal-weight'),
                  ('inverse vol', 'Inverse-volatility'),
                  ('vol-matched EW', 'Equal-weight at strategy volatility')]:
    d = T1[key]
    chk(f'T1 {name} CAGR', d['CAGR'] * 100, '{:.2f}\\%')
    chk(f'T1 {name} Vol', d['Vol'] * 100, '{:.2f}\\%')
    chk(f'T1 {name} Sharpe', d['Sharpe'], '{:.3f}')
    chk(f'T1 {name} MaxDD', abs(d['MaxDD']) * 100, '$-{:.2f}$\\%')
    chk(f'T1 {name} Calmar', d['Calmar'], '{:.3f}')

S = R['T2_sweep']
chk('sweep sharpe wins', S['sharpe_wins'], '{:d} of ' + str(S['n']))
chk('sweep sharpe min', S['sharpe_min'], '{:.3f}')
chk('sweep sharpe max', S['sharpe_max'], '{:.3f}')
for r in S['runs']:
    chk(f"sweep cell {r['seed']}/{r['cov_lb']}/{r['rebal']}", r['strat_sharpe'], '{:.3f}')

for nm, d in R['T3_episodes'].items():
    chk(f'episode {nm} strat', abs(d['strategy']) * 100,
        '$-{:.2f}$\\%' if d['strategy'] < 0 else '$+{:.2f}$\\%')
    chk(f'episode {nm} ew', abs(d['equal_weight']) * 100,
        '$-{:.2f}$\\%' if d['equal_weight'] < 0 else '$+{:.2f}$\\%')
    chk(f'episode {nm} excess', d['excess'] * 100, '$+{:.2f}$ pp')
    chk(f'episode {nm} days', d['days'], '{:d}')

for r in R['T4_years']['rows']:
    chk(f"year {r['year']} strat sharpe", abs(r['strat_sharpe']),
        '$-{:.2f}$' if r['strat_sharpe'] < 0 else '{:.2f}')
_ny = R['T4_years']['n']
chk('year sharpe wins', R['T4_years']['sharpe_wins'], '{:d} of ' + str(_ny))
chk('year dd wins', R['T4_years']['dd_wins'], '{:d} of ' + str(_ny))

A = R['T5_ablation']
for key in ['Full framework', 'No regime timing, same average exposure',
            'No regime timing, full exposure', 'No signal layer']:
    chk(f'ablation {key} Sharpe', A[key]['Sharpe'], '{:.3f}')
    chk(f'ablation {key} MaxDD', abs(A[key]['MaxDD']) * 100, '$-{:.2f}$\\%')
chk('ablation p vs matched', A['vs_no_regime_matched']['p'], '{:.3f}')
chk('ablation p vs full', A['vs_no_regime_full']['p'], '{:.3f}')
chk('ablation p vs nosignal', A['vs_no_signal']['p'], '{:.3f}')

for nm, d in R['T5b_ladder'].items():
    chk(f'ladder {nm} Sharpe', d['Sharpe'], '{:.3f}')
    chk(f'ladder {nm} MaxDD', abs(d['MaxDD']) * 100, '$-{:.2f}$\\%')

IC = R['T6_ic']
for k, d in IC['overall'].items():
    chk(f'IC overall {k}', abs(d['ic']), '$-{:.3f}$' if d['ic'] < 0 else '$+{:.3f}$')
    chk(f'IC overall {k} t', abs(d['nw_t']), '$-{:.2f}$' if d['nw_t'] < 0 else '$+{:.2f}$')
chk('IC contrast p', IC['regime_contrast']['p'], '{:.3f}')

C = R['T7_concentration']
chk('turnover annual', C['turnover_annual'], '{:.2f}$\\times$')
chk('effective assets', C['mean_effective_assets'], '{:.2f}')
chk('min effective assets', C['min_effective_assets'], '{:.2f}')
for k, d in R['T7_cost'].items():
    chk(f'cost {k} Sharpe', d['Sharpe'], '{:.3f}')
    chk(f'cost {k} CAGR', d['CAGR'] * 100, '{:.2f}\\%')

for o in R['T8_order']:
    chk(f"AIC k={o['k']}", o['AIC'], '{:.0f}')
    chk(f"BIC k={o['k']}", o['BIC'], '{:.0f}')
D8 = R['T8_dynamics']
for k in ['0', '1', '2']:
    chk(f'regime {k} vol', D8['ann_vol'][k] * 100, '{:.2f}\\%')
for b in R['T8_by_regime']:
    chk(f"regime {b['regime']} strat sharpe", b['strat_sharpe'], '{:.3f}')
    chk(f"regime {b['regime']} ew sharpe", b['ew_sharpe'], '{:.3f}')

S9 = R['T9_significance']
chk('excess ann', abs(S9['excess_ann']) * 100, '$-{:.2f}$\\%')
chk('t stat', abs(S9['t']), '$-{:.3f}$')
chk('p value', S9['p'], '{:.3f}')
chk('NW t', abs(S9['nw_t']), '$-{:.3f}$')
chk('bootstrap p', S9['bootstrap']['p_diff_le_0'], '{:.3f}')
chk('cushion', S9['worst_5pct_days']['cushion'] * 100, '{:.2f} percentage points')

for key in ['Regime-conditional strategy', 'Equal-weight',
            'Equal-weight at strategy volatility',
            'No regime timing, same average exposure']:
    d = MG['main'][key]
    chk(f'mega {key} CAGR', d['CAGR'] * 100, '{:.2f}\\%')
    chk(f'mega {key} Sharpe', d['Sharpe'], '{:.3f}')
    chk(f'mega {key} MaxDD', abs(d['MaxDD']) * 100, '$-{:.2f}$\\%')
for nm, d in MG['episodes'].items():
    chk(f'mega episode {nm}', d['excess'] * 100, '{:.2f} percentage points')

H = R.get('T10_holdout', {})
for tag in ['holdout', 'in_sample', 'full']:
    for key, d in H.get(tag, {}).items():
        if key.startswith('_'):
            continue
        chk(f'{tag} {key} Sharpe', d['Sharpe'], '{:.3f}')
        chk(f'{tag} {key} MaxDD', abs(d['MaxDD']) * 100, '$-{:.2f}$\%')
if 'worst_ew_episode' in H:
    e = H['worst_ew_episode']
    chk('holdout worst episode strat', abs(e['strategy']) * 100, '$-{:.2f}$\%')
    chk('holdout worst episode ew', abs(e['equal_weight']) * 100, '$-{:.2f}$\%')

IS = R['T6_ic'].get('splits', {})
for tag, d in IS.items():
    chk(f'IC split {tag} fused', abs(d['fused']['ic']),
        '$-{:.3f}$' if d['fused']['ic'] < 0 else '$+{:.3f}$')

print(f'{ok} checks passed, {len(bad)} failed')
for b in bad:
    print('  FAIL', b)
sys.exit(1 if bad else 0)
