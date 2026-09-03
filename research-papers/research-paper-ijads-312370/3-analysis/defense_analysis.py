"""What of the paper's narrative survives on real numbers?

Every claim the paper makes in section 4 is turned into a test here. Nothing is tuned:
the configuration is the one already fixed in pipeline_core.DEF. Wins and losses both reported.
"""
import json, os
import numpy as np
import pandas as pd
from scipy import stats
import pipeline_core as P

OUT = os.path.dirname(os.path.abspath(__file__))
RES = {}
pd.set_option('display.width', 200)


def sh(x):
    return x.mean() / x.std() * np.sqrt(252) if x.std() else 0.0


def mdd(x):
    c = (1 + x).cumprod()
    return float((c / c.cummax() - 1).min())


def nw_t(x, lags=10):
    x = np.asarray(x, float); n = len(x); m = x.mean(); e = x - m
    s = (e @ e) / n
    for l in range(1, lags + 1):
        s += 2 * (1 - l / (lags + 1)) * ((e[l:] @ e[:-l]) / n)
    return float(m / np.sqrt(s / n))


for UNAME, UNIV in [('MEGACAP', P.MEGA), ('SECTOR', P.SECT)]:
    print("\n" + "=" * 78)
    print(f"UNIVERSE: {UNAME}")
    print("=" * 78)
    U = {}

    strat, W, RG = P.backtest(UNIV)
    no_reg, _, _ = P.backtest(UNIV, use_regime=False)
    idx = strat.index
    B = P.benchmarks(UNIV, idx)
    ew = B['ew']
    no_reg = no_reg.reindex(idx).dropna()
    common_nr = strat.index.intersection(no_reg.index)
    ew_matched = ew * (strat.std() / ew.std())

    # ---------------------------------------------------------- C1 drawdown control
    print("\nC1  CLAIM: 'lower drawdowns / defensive capability'")
    c1 = dict(strategy=mdd(strat), equal_weight=mdd(ew), ew_vol_matched=mdd(ew_matched),
              no_regime_ablation=mdd(no_reg))
    for k, v in c1.items():
        print(f"    maxDD {k:22s} {v:+.4f}")
    print(f"    -> vs equal-weight        : {'HOLDS' if c1['strategy'] > c1['equal_weight'] else 'FAILS'}")
    print(f"    -> vs vol-matched EW      : {'HOLDS' if c1['strategy'] > c1['ew_vol_matched'] else 'FAILS'}")
    print(f"    -> regime vs no-regime    : {'HOLDS' if c1['strategy'] > c1['no_regime_ablation'] else 'FAILS'}")
    U['C1_drawdown'] = c1

    # ---------------------------------------------------------- C2 downside capture
    print("\nC2  CLAIM: 'low downside capture, competitive upside'")
    dn, up = ew < 0, ew > 0
    dcap = float(strat[dn].mean() / ew[dn].mean())
    ucap = float(strat[up].mean() / ew[up].mean())
    print(f"    downside capture {dcap:.4f}   upside capture {ucap:.4f}   up/down ratio {ucap / dcap:.4f}")
    print(f"    -> {'HOLDS (captures less down than up)' if ucap > dcap else 'FAILS (captures more down than up)'}")
    U['C2_capture'] = dict(down=dcap, up=ucap, ratio=float(ucap / dcap))

    # ---------------------------------------------------------- C3 named crises
    print("\nC3  CLAIM: 'outperformed during the 2020 crash and the 2022 inflation shock'")
    episodes = {'COVID crash 2020-02-19..2020-03-23': ('2020-02-19', '2020-03-23'),
                '2022 bear 2022-01-03..2022-10-12': ('2022-01-03', '2022-10-12'),
                '2018 Q4 selloff': ('2018-10-01', '2018-12-24'),
                '2023 regional banks': ('2023-03-01', '2023-03-31')}
    ep = {}
    for nm, (a, b) in episodes.items():
        m = (idx >= a) & (idx <= b)
        if m.sum() < 5:
            continue
        s_, e_ = strat[m], ew[m]
        cs, ce = float((1 + s_).prod() - 1), float((1 + e_).prod() - 1)
        ep[nm] = dict(days=int(m.sum()), strategy=cs, equal_weight=ce, excess=cs - ce)
        print(f"    {nm:38s} strat {cs:+7.2%}  EW {ce:+7.2%}  excess {cs - ce:+7.2%}  "
              f"{'WIN' if cs > ce else 'loss'}")
    U['C3_episodes'] = ep

    # ---------------------------------------------------------- C4 regime-conditional
    print("\nC4  CLAIM: 'performance attribution by regime'  (out-of-sample regime labels)")
    rgd = RG.reindex(idx).ffill()
    rows = []
    for g in sorted(rgd.dropna().unique()):
        m = rgd == g
        if m.sum() < 20:
            continue
        s_, e_ = strat[m], ew[m]
        rows.append(dict(regime=int(g), days=int(m.sum()), strat_sharpe=sh(s_),
                         ew_sharpe=sh(e_), excess_ann=float((s_.mean() - e_.mean()) * 252),
                         win=bool(sh(s_) > sh(e_))))
    rt = pd.DataFrame(rows)
    print(rt.round(4).to_string(index=False))
    U['C4_regime'] = rt.round(6).to_dict('records')

    # ---------------------------------------------------------- C5 calendar years
    print("\nC5  year-by-year (this is the honest 'sometimes it wins' table)")
    yr = []
    for y in sorted(set(idx.year)):
        m = idx.year == y
        if m.sum() < 60:
            continue
        s_, e_ = strat[m], ew[m]
        yr.append(dict(year=int(y), strat_ret=float((1 + s_).prod() - 1),
                       ew_ret=float((1 + e_).prod() - 1), strat_sharpe=sh(s_), ew_sharpe=sh(e_),
                       strat_dd=mdd(s_), ew_dd=mdd(e_),
                       sharpe_win=bool(sh(s_) > sh(e_)), dd_win=bool(mdd(s_) > mdd(e_))))
    yt = pd.DataFrame(yr)
    print(yt.round(4).to_string(index=False))
    print(f"    Sharpe wins {yt['sharpe_win'].sum()} of {len(yt)} years | "
          f"drawdown wins {yt['dd_win'].sum()} of {len(yt)} years")
    U['C5_years'] = yt.round(6).to_dict('records')

    # ---------------------------------------------------------- C6 rolling Sharpe stability
    print("\nC6  CLAIM: 'rolling Sharpe mostly positive and smoother than the benchmark'")
    rs = strat.rolling(63).mean() / strat.rolling(63).std() * np.sqrt(252)
    re_ = ew.rolling(63).mean() / ew.rolling(63).std() * np.sqrt(252)
    rs, re_ = rs.dropna(), re_.dropna()
    c6 = dict(strat_frac_positive=float((rs > 0).mean()), ew_frac_positive=float((re_ > 0).mean()),
              strat_sd_of_rolling=float(rs.std()), ew_sd_of_rolling=float(re_.std()))
    print(f"    fraction positive: strategy {c6['strat_frac_positive']:.3f}  EW {c6['ew_frac_positive']:.3f}")
    print(f"    s.d. of rolling Sharpe: strategy {c6['strat_sd_of_rolling']:.3f}  EW {c6['ew_sd_of_rolling']:.3f}"
          f"   -> smoother: {'HOLDS' if c6['strat_sd_of_rolling'] < c6['ew_sd_of_rolling'] else 'FAILS'}")
    U['C6_rolling'] = c6

    # ---------------------------------------------------------- C7 worst market days
    print("\nC7  behaviour in the worst 5% of market days")
    q = ew.quantile(0.05)
    m = ew <= q
    c7 = dict(n=int(m.sum()), strat_mean=float(strat[m].mean()), ew_mean=float(ew[m].mean()),
              cushion=float(strat[m].mean() - ew[m].mean()))
    print(f"    n={c7['n']}  strategy {c7['strat_mean']:+.4%}  EW {c7['ew_mean']:+.4%}  "
          f"cushion {c7['cushion']:+.4%} per day")
    U['C7_tail'] = c7

    # ---------------------------------------------------------- C8 full-sample significance
    print("\nC8  full-sample comparison (reported whether it helps or not)")
    ex = strat - ew
    t_, p_ = stats.ttest_1samp(ex, 0)
    exn = (strat - no_reg.reindex(strat.index)).dropna()
    tn_, pn_ = stats.ttest_1samp(exn, 0)
    c8 = dict(strat_sharpe=sh(strat), ew_sharpe=sh(ew), ewm_sharpe=sh(ew_matched),
              noreg_sharpe=sh(no_reg), excess_vs_ew_ann=float(ex.mean() * 252),
              t_vs_ew=float(t_), p_vs_ew=float(p_), nw_t_vs_ew=nw_t(ex),
              excess_vs_noreg_ann=float(exn.mean() * 252), t_vs_noreg=float(tn_),
              p_vs_noreg=float(pn_), nw_t_vs_noreg=nw_t(exn))
    print(f"    Sharpe  strategy {c8['strat_sharpe']:.3f} | EW {c8['ew_sharpe']:.3f} | "
          f"EW vol-matched {c8['ewm_sharpe']:.3f} | no-regime ablation {c8['noreg_sharpe']:.3f}")
    print(f"    vs EW          : excess {c8['excess_vs_ew_ann']:+.2%}/yr  t {t_:.3f} (p {p_:.4f})  NW t {c8['nw_t_vs_ew']:.3f}")
    print(f"    vs no-regime   : excess {c8['excess_vs_noreg_ann']:+.2%}/yr  t {tn_:.3f} (p {pn_:.4f})  NW t {c8['nw_t_vs_noreg']:.3f}")
    U['C8_significance'] = c8

    RES[UNAME] = U

# ================================================================ C9 real distribution
print("\n" + "=" * 78)
print("C9  DISTRIBUTION OF OUTCOMES ACROSS SPECIFICATIONS  (the honest 'N of M' table)")
print("=" * 78)
runs = []
for UNAME, UNIV in [('MEGACAP', P.MEGA), ('SECTOR', P.SECT)]:
    for seed in [0, 7, 42]:
        for lb in [126, 252]:
            for rb in [10, 21, 42]:
                cfg = dict(seed=seed, cov_lookback=lb, rebal=rb)
                s_, _, _ = P.backtest(UNIV, cfg)
                e_ = P.benchmarks(UNIV, s_.index)['ew']
                em = e_ * (s_.std() / e_.std())
                runs.append(dict(universe=UNAME, seed=seed, cov_lb=lb, rebal=rb,
                                 strat_sharpe=round(sh(s_), 4), ew_sharpe=round(sh(e_), 4),
                                 strat_dd=round(mdd(s_), 4), ew_dd=round(mdd(e_), 4),
                                 ewm_sharpe=round(sh(em), 4), ewm_dd=round(mdd(em), 4),
                                 sharpe_win=bool(sh(s_) > sh(e_)),
                                 dd_win=bool(mdd(s_) > mdd(e_)),
                                 dd_win_matched=bool(mdd(s_) > mdd(em))))
                print(f"  {UNAME:8s} seed {seed:<3} cov {lb:<4} rebal {rb:<3} -> "
                      f"Sharpe {sh(s_):.3f} vs EW {sh(e_):.3f} | maxDD {mdd(s_):+.3f} vs {mdd(e_):+.3f}"
                      f" | {'S-WIN' if sh(s_) > sh(e_) else '     '} {'DD-WIN' if mdd(s_) > mdd(e_) else ''}")
D = pd.DataFrame(runs)
print("\n  --- tally over all %d specifications ---" % len(D))
print(f"  Sharpe beats equal-weight        : {D['sharpe_win'].sum()} of {len(D)}")
print(f"  Drawdown beats equal-weight      : {D['dd_win'].sum()} of {len(D)}")
print(f"  Drawdown beats vol-matched EW    : {D['dd_win_matched'].sum()} of {len(D)}")
for u in D['universe'].unique():
    d = D[D['universe'] == u]
    print(f"    {u:8s} Sharpe {d['sharpe_win'].sum()}/{len(d)}   DD {d['dd_win'].sum()}/{len(d)}")
RES['C9_specification_runs'] = D.to_dict('records')
RES['C9_tally'] = dict(n=len(D), sharpe_wins=int(D['sharpe_win'].sum()),
                       dd_wins=int(D['dd_win'].sum()),
                       dd_wins_matched=int(D['dd_win_matched'].sum()))

json.dump(RES, open(os.path.join(OUT, 'defense_results.json'), 'w'), indent=2, default=str)
print("\nsaved defense_results.json")
