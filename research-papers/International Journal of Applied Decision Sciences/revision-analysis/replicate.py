"""Faithful re-run of code.ipynb (IJADS-312370) + honest diagnostics.

Part A reproduces the notebook cell-for-cell (same tickers, dates, seed, logic).
Part B adds the diagnostics the notebook never computed.
"""
import json, os, warnings
import numpy as np
import pandas as pd
import yfinance as yf
import ta
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
import cvxpy as cp
from scipy import stats

warnings.filterwarnings('ignore')
OUT = os.path.dirname(os.path.abspath(__file__))
R = {}

# ---------------------------------------------------------------- Cell 2
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'XOM']
START_DATE, END_DATE = '2018-01-01', '2024-01-01'

print("Fetching market data...", flush=True)
market_data = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True,
                          progress=False, threads=False)
returns = market_data['Close'].pct_change().dropna()
volume = market_data['Volume']
print("Data shape:", returns.shape)
print("Date range:", returns.index[0], "to", returns.index[-1])
R['data_shape'] = list(returns.shape)
R['date_range'] = [str(returns.index[0].date()), str(returns.index[-1].date())]

# ---------------------------------------------------------------- Cell 3
def calculate_technical_indicators(data, tickers):
    tech = {}
    for t in tickers:
        high, low, close = data['High'][t], data['Low'][t], data['Close'][t]
        vol_t = data['Volume'][t]
        rsi = ta.momentum.RSIIndicator(close).rsi()
        macd = ta.trend.MACD(close).macd()
        stoch = ta.momentum.StochasticOscillator(high, low, close).stoch()
        bb = ta.volatility.BollingerBands(close)
        atr = ta.volatility.AverageTrueRange(high, low, close).average_true_range()
        mfi = ta.volume.MFIIndicator(high, low, close, vol_t).money_flow_index()
        obv = ta.volume.OnBalanceVolumeIndicator(close, vol_t).on_balance_volume()
        tech[t] = pd.DataFrame({
            f'{t}_RSI': rsi, f'{t}_MACD': macd, f'{t}_Stoch': stoch,
            f'{t}_BB_Width': (bb.bollinger_hband() - bb.bollinger_lband()) / close,
            f'{t}_ATR': atr, f'{t}_MFI': mfi, f'{t}_OBV': obv}, index=close.index)
    return pd.concat([tech[t] for t in tickers], axis=1).dropna()

technical_indicators = calculate_technical_indicators(market_data, TICKERS)
print("Technical indicators shape:", technical_indicators.shape)
R['tech_shape'] = list(technical_indicators.shape)

# ---------------------------------------------------------------- Cell 4
def calculate_market_sentiment_proxies(returns, volume, tickers, window=20):
    sent = {}
    for t in tickers:
        rt, vt = returns[t], volume[t]
        sent[t] = pd.DataFrame({
            f'{t}_Price_Momentum': rt.rolling(5).mean(),
            f'{t}_Volatility_Regime': rt.rolling(window).std(),
            f'{t}_Volume_Ratio': vt / vt.rolling(window).mean(),
            f'{t}_Volume_Volatility': vt.pct_change().rolling(window).std(),
            f'{t}_Market_Stress': rt.rolling(window).std() * np.sqrt(252)}, index=rt.index)
    return pd.concat([sent[t] for t in tickers], axis=1).dropna()

sentiment_indicators = calculate_market_sentiment_proxies(returns, volume, TICKERS)
print("Sentiment indicators shape:", sentiment_indicators.shape)
R['sent_shape'] = list(sentiment_indicators.shape)

# ---------------------------------------------------------------- Cell 5
def detect_market_regimes(returns, n_regimes=3):
    volatility = returns.rolling(20).std().dropna()
    momentum = returns.rolling(10).mean().dropna()
    idx = volatility.index.intersection(momentum.index)
    features = pd.DataFrame({'Volatility': volatility.loc[idx].mean(axis=1),
                             'Momentum': momentum.loc[idx].mean(axis=1)}).dropna()
    scaler = StandardScaler()
    fs = scaler.fit_transform(features)
    gmm = GaussianMixture(n_components=n_regimes, random_state=42)
    regimes = gmm.fit_predict(fs)
    rdf = pd.DataFrame({'Regime': regimes, 'Volatility': features['Volatility'],
                        'Momentum': features['Momentum']}, index=features.index)
    stats_ = rdf.groupby('Regime')[['Volatility', 'Momentum']].mean()
    print("Regime Statistics:\n", stats_)
    return rdf, gmm, scaler, features, stats_

regime_data, gmm_model, scaler, regime_features, regime_stats = detect_market_regimes(returns)
R['regime_stats'] = regime_stats.round(6).to_dict()
R['regime_counts'] = {int(k): int(v) for k, v in regime_data['Regime'].value_counts().sort_index().items()}

# ---------------------------------------------------------------- Cell 6
def calculate_signal_confidence(tech, sent, regime, tickers):
    idx = tech.index.intersection(sent.index).intersection(regime.index)
    tech, sent, regime = tech.loc[idx], sent.loc[idx], regime.loc[idx]
    fused, conf = {}, {}
    rw = {0: {'tech': .3, 'sent': .7}, 1: {'tech': .5, 'sent': .5}, 2: {'tech': .7, 'sent': .3}}
    for t in tickers:
        ts = tech[[c for c in tech.columns if c.startswith(t)]].mean(axis=1)
        ss = sent[[c for c in sent.columns if c.startswith(t)]].mean(axis=1)
        f = pd.Series(index=idx, dtype=float)
        c = pd.Series(index=idx, dtype=float)
        for rg, w in rw.items():
            m = regime['Regime'] == rg
            f[m] = w['tech'] * ts[m] + w['sent'] * ss[m]
            sc = ts[m].corr(ss[m])
            c[m] = 0.5 + 0.5 * sc if not np.isnan(sc) else 0.5
        fused[t], conf[t] = f, c
    return pd.DataFrame(fused), pd.DataFrame(conf)

fused_signals, confidence_scores = calculate_signal_confidence(
    technical_indicators, sentiment_indicators, regime_data, TICKERS)
print("Fused signals shape:", fused_signals.shape)

# ---------------------------------------------------------------- Cell 7
def regime_dependent_portfolio_optimization(returns, fused, conf, regime, lookback=63):
    idx = returns.index.intersection(fused.index)
    ra, sa, ca, rga = returns.loc[idx], fused.loc[idx], conf.loc[idx], regime.loc[idx]
    W = pd.DataFrame(index=idx, columns=returns.columns, dtype=float)
    P = pd.Series(index=idx, dtype=float)
    n_fallback = 0
    for i in range(lookback, len(idx)):
        d = idx[i]
        lb = ra.iloc[i - lookback:i]
        cs, cc = sa.iloc[i], ca.iloc[i]
        rg = rga['Regime'].iloc[i]
        try:
            if rg == 0:
                risk_target, turn = 0.10, 0.05
            elif rg == 1:
                risk_target, turn = 0.15, 0.10
            else:
                risk_target, turn = 0.20, 0.15
            er = cs * cc
            cov = LedoitWolf().fit(lb).covariance_
            w = cp.Variable(len(TICKERS))
            pr = er.values @ w
            prisk = cp.quad_form(w, cp.psd_wrap(cov))
            cons = [cp.sum(w) == 1, w >= 0, prisk <= risk_target ** 2,
                    cp.norm(w, 1) <= 1 + turn]
            prob = cp.Problem(cp.Maximize(pr - 0.5 * prisk), cons)
            prob.solve()
            if prob.status == 'optimal':
                W.loc[d] = w.value
                if i < len(idx) - 1:
                    P.loc[d] = ra.iloc[i + 1] @ w.value
        except Exception:
            n_fallback += 1
            W.loc[d] = np.ones(len(TICKERS)) / len(TICKERS)
            if i < len(idx) - 1:
                P.loc[d] = ra.iloc[i + 1].mean()
    print("solver fallbacks:", n_fallback)
    R['solver_fallbacks'] = n_fallback
    return W.dropna(), P.dropna()

print("Optimising (this takes a few minutes)...", flush=True)
portfolio_weights, portfolio_returns = regime_dependent_portfolio_optimization(
    returns, fused_signals, confidence_scores, regime_data)
print("Portfolio weights shape:", portfolio_weights.shape)
R['weights_shape'] = list(portfolio_weights.shape)

# ---------------------------------------------------------------- Cell 8
def analyze(pr):
    cum = (1 + pr).cumprod()
    total = cum.iloc[-1] - 1
    ann = (1 + total) ** (252 / len(pr)) - 1
    vol = pr.std() * np.sqrt(252)
    sharpe = ann / vol if vol > 0 else 0
    dd = (cum - cum.expanding().max()) / cum.expanding().max()
    mdd = dd.min()
    return dict(total_return=total, annual_return=ann, annual_vol=vol, sharpe=sharpe,
                max_dd=mdd, calmar=ann / abs(mdd) if mdd else 0,
                win_rate=(pr > 0).mean()), cum, dd

benchmark_returns = returns.mean(axis=1)
benchmark_aligned = benchmark_returns.loc[portfolio_returns.index]
perf, cum, dd = analyze(portfolio_returns)
bperf, bcum, bdd = analyze(benchmark_aligned)
print("\n=== PORTFOLIO PERFORMANCE (cell 8 definition) ===")
for k, v in perf.items():
    print(f"  {k:18s} {v: .6f}")
print("=== BENCHMARK, same definition ===")
for k, v in bperf.items():
    print(f"  {k:18s} {v: .6f}")
R['cell8_portfolio'] = {k: float(v) for k, v in perf.items()}
R['cell8_benchmark'] = {k: float(v) for k, v in bperf.items()}

# ---------------------------------------------------------------- Cell 10
pa = portfolio_returns.mean() * 252
ba = benchmark_aligned.mean() * 252
pv = portfolio_returns.std() * np.sqrt(252)
bv = benchmark_aligned.std() * np.sqrt(252)
ex = portfolio_returns - benchmark_aligned
beta = portfolio_returns.cov(benchmark_aligned) / benchmark_aligned.var()
alpha = ex.mean() * 252
te = ex.std() * np.sqrt(252)
ir = alpha / te if te > 0 else 0
t_stat, p_value = stats.ttest_ind(portfolio_returns, benchmark_aligned)
print("\n=== STATISTICAL ANALYSIS (cell 10 definition) ===")
print(f"  portfolio ann ret {pa:.6f}  vol {pv:.6f}  sharpe {pa/pv:.6f}")
print(f"  benchmark ann ret {ba:.6f}  vol {bv:.6f}  sharpe {ba/bv:.6f}")
print(f"  beta {beta:.6f}  alpha {alpha:.6f}  IR {ir:.6f}  t {t_stat:.6f}  p {p_value:.6f}")
R['cell10'] = dict(port_ann=float(pa), bench_ann=float(ba), port_vol=float(pv),
                   bench_vol=float(bv), port_sharpe=float(pa / pv), bench_sharpe=float(ba / bv),
                   beta=float(beta), alpha=float(alpha), ir=float(ir),
                   t_stat=float(t_stat), p_value=float(p_value))

# ---------------------------------------------------------------- Cell 11
rows = []
idx = portfolio_returns.index.intersection(regime_data.index)
for rg in [0, 1, 2]:
    m = regime_data.loc[idx, 'Regime'] == rg
    p_ = portfolio_returns.loc[idx][m]
    b_ = benchmark_aligned.loc[idx][m]
    if len(p_) == 0:
        continue
    prt, brt = p_.mean() * 252, b_.mean() * 252
    pvv, bvv = p_.std() * np.sqrt(252), b_.std() * np.sqrt(252)
    rows.append(dict(Regime=rg, Days=len(p_), Portfolio_Return=prt, Benchmark_Return=brt,
                     Excess=prt - brt, Portfolio_Sharpe=prt / pvv if pvv else 0,
                     Benchmark_Sharpe=brt / bvv if bvv else 0,
                     Outperformance_Rate=(p_ > b_).mean()))
attr = pd.DataFrame(rows)
print("\n=== REGIME ATTRIBUTION (cell 11 definition) ===")
print(attr.to_string(index=False))
R['cell11'] = attr.round(6).to_dict('records')

# ================================================================ PART B
print("\n\n############ PART B: DIAGNOSTICS THE NOTEBOOK NEVER RAN ############")

t_p, p_p = stats.ttest_1samp(ex, 0.0)


def nw_t(x, lags=10):
    x = np.asarray(x, float)
    n = len(x)
    m = x.mean()
    e = x - m
    s = (e @ e) / n
    for l in range(1, lags + 1):
        gl = (e[l:] @ e[:-l]) / n
        s += 2 * (1 - l / (lags + 1)) * gl
    return m / np.sqrt(s / n)


print(f"\nB1  paired t on daily excess return : t={t_p:.4f}  p={p_p:.4f}")
print(f"B1  Newey-West(10) t on excess      : t={nw_t(ex):.4f}")
print(f"B1  notebook ttest_ind (unpaired)   : t={t_stat:.4f}  p={p_value:.4f}  <- wrong test for paired series")
R['B1'] = dict(paired_t=float(t_p), paired_p=float(p_p), nw_t=float(nw_t(ex)))

rng = np.random.default_rng(42)


def sharpe(x):
    return x.mean() / x.std() * np.sqrt(252)


pv_, bv_ = portfolio_returns.values, benchmark_aligned.values
n = len(pv_)
block = 21
diffs = []
for _ in range(5000):
    ids = []
    while len(ids) < n:
        s = int(rng.integers(0, n))
        ids.extend(range(s, min(s + block, n)))
    ids = np.array(ids[:n])
    diffs.append(sharpe(pv_[ids]) - sharpe(bv_[ids]))
diffs = np.array(diffs)
lo, hi = np.percentile(diffs, [2.5, 97.5])
obs = sharpe(pv_) - sharpe(bv_)
print(f"\nB2  Sharpe(port) {sharpe(pv_):.4f}   Sharpe(bench) {sharpe(bv_):.4f}")
print(f"B2  Sharpe difference {obs:+.4f}   95% block-bootstrap CI [{lo:+.4f}, {hi:+.4f}]")
print(f"B2  P(strategy Sharpe > benchmark Sharpe) = {(diffs > 0).mean():.3f}")
R['B2'] = dict(sharpe_port=float(sharpe(pv_)), sharpe_bench=float(sharpe(bv_)),
               diff=float(obs), ci=[float(lo), float(hi)], p_gt=float((diffs > 0).mean()))

W = portfolio_weights.astype(float)
top1 = W.max(axis=1)
hhi = (W ** 2).sum(axis=1)
neff = 1 / hhi
turnover = W.diff().abs().sum(axis=1).dropna()
print(f"\nB3  mean largest single weight          : {top1.mean():.4f}   median {top1.median():.4f}")
print(f"B3  share of days with top weight > 0.99: {(top1 > 0.99).mean():.4f}")
print(f"B3  mean effective no. of assets (1/HHI): {neff.mean():.3f}   (10 = fully diversified)")
print(f"B3  mean daily turnover                 : {turnover.mean():.4f}  ({turnover.mean() * 252:.1f}x per year)")
print(f"B3  cost at 10 bps round trip           : {turnover.mean() * 252 * 0.0010 * 100:.2f}% of capital per year")
R['B3'] = dict(top1_mean=float(top1.mean()), top1_median=float(top1.median()),
               frac_corner=float((top1 > 0.99).mean()), neff=float(neff.mean()),
               daily_turnover=float(turnover.mean()), ann_turnover=float(turnover.mean() * 252))

cov_last = LedoitWolf().fit(returns.iloc[-63:]).covariance_
w_eq = np.ones(10) / 10
daily_var = w_eq @ cov_last @ w_eq
print(f"\nB4  typical daily portfolio variance : {daily_var:.6e}  (daily vol {np.sqrt(daily_var):.4f})")
print(f"B4  tightest budget in code          : risk_target**2 = {0.10 ** 2:.4f}")
print(f"B4  constraint binding?              : {'YES' if daily_var > 0.01 else 'NO - covariance is DAILY, target is ANNUAL, so it is inert'}")
R['B4'] = dict(daily_var=float(daily_var), tightest_budget=0.01, binds=bool(daily_var > 0.01))

one = fused_signals.abs().mean()
print("\nB5  mean |fused signal| per ticker (this is the alpha vector fed to the optimiser):")
print(one.round(1).to_string())
obv_scale = technical_indicators[[c for c in technical_indicators.columns if c.endswith('_OBV')]].abs().mean()
rsi_scale = technical_indicators[[c for c in technical_indicators.columns if c.endswith('_RSI')]].abs().mean()
print(f"B5  mean |OBV| {obv_scale.mean():.3e}   vs   mean |RSI| {rsi_scale.mean():.2f}")
R['B5'] = dict(mean_abs_signal={k: float(v) for k, v in one.items()},
               obv_scale=float(obv_scale.mean()), rsi_scale=float(rsi_scale.mean()))

fs = StandardScaler().fit_transform(regime_features)
print("\nB6  GMM model selection on the regime feature matrix:")
sel = {}
for k in range(1, 7):
    g = GaussianMixture(n_components=k, random_state=42, n_init=5).fit(fs)
    sel[k] = dict(aic=float(g.aic(fs)), bic=float(g.bic(fs)))
    print(f"    k={k}  AIC {g.aic(fs):10.2f}   BIC {g.bic(fs):10.2f}")
R['B6_model_selection'] = sel

rg = regime_data['Regime'].values
trans = pd.crosstab(pd.Series(rg[:-1], name='from'), pd.Series(rg[1:], name='to'), normalize='index')
print("\nB6  regime transition matrix (row-normalised):")
print(trans.round(4).to_string())
runs = []
cur, ln = rg[0], 1
for x in rg[1:]:
    if x == cur:
        ln += 1
    else:
        runs.append((cur, ln))
        cur, ln = x, 1
runs.append((cur, ln))
rl = pd.DataFrame(runs, columns=['regime', 'len'])
print("\nB6  run lengths (days) per regime:")
print(rl.groupby('regime')['len'].agg(['count', 'mean', 'max']).round(2).to_string())
R['B6_persistence'] = dict(diag=[float(trans.iloc[i, i]) for i in range(len(trans))],
                           mean_run={int(k): float(v) for k, v in rl.groupby('regime')['len'].mean().items()})

split = regime_features.index[int(len(regime_features) * 0.5)]
train_feat = regime_features[regime_features.index < split]
sc_oos = StandardScaler().fit(train_feat)
g_oos = GaussianMixture(n_components=3, random_state=42).fit(sc_oos.transform(train_feat))
lab_oos = pd.Series(g_oos.predict(sc_oos.transform(regime_features)), index=regime_features.index)
agree = (lab_oos.loc[regime_data.index] == regime_data['Regime']).mean()
print(f"\nB7  full-sample GMM labels vs train-only GMM labels agree on {agree:.1%} of days")
R['B7_label_agreement'] = float(agree)

json.dump(R, open(os.path.join(OUT, 'results.json'), 'w'), indent=2, default=str)
portfolio_returns.to_csv(os.path.join(OUT, 'portfolio_returns.csv'))
benchmark_aligned.to_csv(os.path.join(OUT, 'benchmark_returns.csv'))
portfolio_weights.to_csv(os.path.join(OUT, 'portfolio_weights.csv'))
regime_data.to_csv(os.path.join(OUT, 'regime_data.csv'))
print("\nSaved -> results.json + csvs in", OUT)
