"""IJADS-312370 -- corrected pipeline.

Fixes applied (each one is a genuine specification error in code.ipynb, not a tuning knob):

 F1  Cross-sectional z-scoring of every raw indicator before fusion.
     Was: raw levels averaged together, so On-Balance Volume (1e8-2e9) swamped RSI (~50)
     and the optimiser always went 100% into one name.
 F2  Directional vs risk indicators separated. ATR and Bollinger width measure volatility,
     not direction, so they no longer sit inside a directional alpha. OBV enters as a
     20-day flow scaled by average volume, not as a non-stationary level.
 F3  Units reconciled. Covariance is annualised (252x) so the regime risk budget
     (10/15/20% vol) is comparable with it. Was: daily covariance vs annual target,
     so the constraint could never bind.
 F4  Expected returns on a return scale: alpha_i = IC * sigma_i * z_i (Grinold), so the
     linear and quadratic terms of the objective are commensurate.
 F5  Transaction costs actually in the objective AND charged to P&L. The paper's Eq. (1)
     always had the term; the code never did.
 F6  Position limit w_max and a cash allocation (sum w <= 1) give the risk budgeting
     something to do, and let the strategy genuinely de-risk in stress.
 F7  True 63/21 walk-forward: re-optimise every 21 trading days, hold in between --
     the protocol the paper already describes.
 F8  No look-ahead in regime detection. The GMM and its scaler are refit on an expanding
     window of past data only at each rebalance, and components are sorted by volatility
     so regime 0 = calm, 2 = stressed (also fixes the mislabelling in the paper's Fig. 4).
 F9  The degenerate confidence score (one constant per regime over the whole sample)
     is replaced by a trailing 63-day agreement between the two signal families.
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

TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'XOM']
START, END = '2018-01-01', '2024-01-01'

# ---- defaults fixed BEFORE seeing any result; the grid at the end reports every cell
DEF = dict(cov_lookback=252, rebal=21, ic=0.05, lam=2.0, w_max=0.25,
           cost_bps=10.0, n_regimes=3,
           target_vol={0: 0.20, 1: 0.15, 2: 0.10},          # calm / normal / stressed
           fuse={0: (0.7, 0.3), 1: (0.5, 0.5), 2: (0.3, 0.7)})  # (technical, sentiment)

# ================================================================= data
print("downloading...", flush=True)
md = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False, threads=False)
spy = yf.download('SPY', start=START, end=END, auto_adjust=True, progress=False,
                  threads=False)['Close'].squeeze()
returns = md['Close'].pct_change().dropna()
volume = md['Volume']
print("returns", returns.shape)


def xs_z(df):
    """cross-sectional z-score, same date only -- no time leakage"""
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)


# ================================================================= F1/F2 signals
raw = {}
for t in TICKERS:
    high, low, close, vt = md['High'][t], md['Low'][t], md['Close'][t], md['Volume'][t]
    bb = ta.volatility.BollingerBands(close)
    raw[t] = pd.DataFrame({
        'rsi': ta.momentum.RSIIndicator(close).rsi() - 50.0,
        'macd': ta.trend.MACD(close).macd() / close,
        'stoch': ta.momentum.StochasticOscillator(high, low, close).stoch() - 50.0,
        'mfi': ta.volume.MFIIndicator(high, low, close, vt).money_flow_index() - 50.0,
        'obv_flow': ta.volume.OnBalanceVolumeIndicator(close, vt).on_balance_volume()
                    .diff(20) / vt.rolling(20).mean(),
        'atr': ta.volatility.AverageTrueRange(high, low, close).average_true_range() / close,
        'bbw': (bb.bollinger_hband() - bb.bollinger_lband()) / close,
        'mom5': close.pct_change(5),
        'vol_ratio': vt / vt.rolling(20).mean(),
        'stress': close.pct_change().rolling(20).std() * np.sqrt(252),
        'vol_of_vol': vt.pct_change().rolling(20).std(),
    }, index=close.index)

panel = {f: xs_z(pd.DataFrame({t: raw[t][f] for t in TICKERS})) for f in raw[TICKERS[0]].columns}

TECH_DIR = ['rsi', 'macd', 'stoch', 'mfi', 'obv_flow']          # directional
SENT_DIR = ['mom5', 'vol_ratio', 'stress', 'vol_of_vol']        # proxy "sentiment"
SENT_SIGN = {'mom5': +1, 'vol_ratio': +1, 'stress': -1, 'vol_of_vol': -1}

tech_sig = sum(panel[f] for f in TECH_DIR) / len(TECH_DIR)
sent_sig = sum(SENT_SIGN[f] * panel[f] for f in SENT_DIR) / len(SENT_DIR)
tech_sig, sent_sig = tech_sig.dropna(), sent_sig.dropna()
common = tech_sig.index.intersection(sent_sig.index).intersection(returns.index)
tech_sig, sent_sig = tech_sig.loc[common], sent_sig.loc[common]

# F9 trailing agreement between the two families, per asset
agree = pd.DataFrame({t: tech_sig[t].rolling(63).corr(sent_sig[t]) for t in TICKERS})
confidence = (0.5 + 0.5 * agree).clip(0.1, 1.0).fillna(0.5)

# regime features (market-level), same construction as the original
vol20 = returns.rolling(20).std().mean(axis=1)
mom10 = returns.rolling(10).mean().mean(axis=1)
regime_feat = pd.DataFrame({'Volatility': vol20, 'Momentum': mom10}).dropna()


# ================================================================= F8 leak-free regimes
def regime_at(date, n_regimes, min_obs=252):
    hist = regime_feat[regime_feat.index < date]
    if len(hist) < min_obs:
        return 1, None
    sc = StandardScaler().fit(hist)
    g = GaussianMixture(n_components=n_regimes, random_state=42, n_init=3).fit(sc.transform(hist))
    order = np.argsort(g.means_[:, 0])                 # ascending mean volatility
    remap = {old: new for new, old in enumerate(order)}
    cur = regime_feat.loc[regime_feat.index <= date].iloc[-1:]
    return int(remap[int(g.predict(sc.transform(cur))[0])]), g


# ================================================================= backtest
def backtest(cfg, use_regime=True, signal_on=True):
    cov_lb, rebal = cfg['cov_lookback'], cfg['rebal']
    ic, lam, w_max = cfg['ic'], cfg['lam'], cfg['w_max']
    cost = cfg['cost_bps'] / 10000.0
    dates = common[common >= returns.index[max(cov_lb, 252)]]
    rebal_dates = dates[::rebal]

    n = len(TICKERS)
    w_prev = np.zeros(n)
    daily, wlog, rlog, cashlog = [], {}, {}, {}
    for k, d in enumerate(rebal_dates):
        hist = returns[returns.index < d].iloc[-cov_lb:]
        if len(hist) < cov_lb:
            continue
        Sig = LedoitWolf().fit(hist).covariance_ * 252.0            # F3 annualised
        sig_i = np.sqrt(np.diag(Sig))

        rg, _ = regime_at(d, cfg['n_regimes']) if use_regime else (1, None)
        wt, ws = cfg['fuse'][rg] if use_regime else (0.5, 0.5)
        tgt = cfg['target_vol'][rg] if use_regime else 0.15

        if signal_on:
            z = wt * tech_sig.loc[d].values + ws * sent_sig.loc[d].values
            z = z * confidence.loc[d].values
            z = np.nan_to_num(z)
            alpha = ic * sig_i * z                                  # F4 Grinold scaling
        else:
            alpha = np.zeros(n)

        # step 1: relative allocation inside the risky sleeve, fully invested + position cap
        w = cp.Variable(n)
        risk = cp.quad_form(w, cp.psd_wrap(Sig))
        obj = cp.Maximize(alpha @ w - lam * risk
                          - (cost * 252 / rebal) * cp.norm(w - w_prev, 1))   # F5
        cons = [w >= 0, w <= w_max, cp.sum(w) == 1]                          # F6
        prob = cp.Problem(obj, cons)
        try:
            prob.solve(solver=cp.CLARABEL)
        except Exception:
            prob.solve()
        if w.value is None:
            continue
        sleeve = np.clip(w.value, 0, None)
        sleeve = sleeve / sleeve.sum()

        # step 2: regime risk budget -- scale the sleeve to hit the target vol, never lever up
        sleeve_vol = float(np.sqrt(sleeve @ Sig @ sleeve))
        k_scale = min(1.0, tgt / sleeve_vol) if sleeve_vol > 0 else 0.0
        wv = k_scale * sleeve
        turn = np.abs(wv - w_prev).sum()

        end = rebal_dates[k + 1] if k + 1 < len(rebal_dates) else dates[-1]
        seg = returns.loc[(returns.index > d) & (returns.index <= end)]
        if len(seg) == 0:
            continue
        # buy-and-hold inside the period: risky positions drift with prices, cash is idle
        pos = wv.copy()
        cash = 1.0 - wv.sum()
        V = pos.sum() + cash
        for j, (dt, row) in enumerate(seg.iterrows()):
            pos = pos * (1 + row.values)
            V_new = pos.sum() + cash
            r = V_new / V - 1.0
            if j == 0:
                r -= turn * cost                                     # F5 charged to P&L
            daily.append((dt, r))
            V = V_new
        w_prev = wv
        wlog[d], rlog[d], cashlog[d] = wv, rg, 1 - wv.sum()

    s = pd.Series(dict(daily)).sort_index()
    return s, pd.DataFrame(wlog).T, pd.Series(rlog), pd.Series(cashlog)


def stats_of(x, name=""):
    cum = (1 + x).cumprod()
    ann = cum.iloc[-1] ** (252 / len(x)) - 1
    vol = x.std() * np.sqrt(252)
    sh = ann / vol if vol else 0
    dd = (cum / cum.cummax() - 1).min()
    return dict(name=name, CAGR=float(ann), Vol=float(vol), Sharpe=float(sh),
                MaxDD=float(dd), Calmar=float(ann / abs(dd)) if dd else 0.0,
                WinRate=float((x > 0).mean()))


# ================================================================= run
print("\nrunning corrected pipeline...", flush=True)
strat, W, RG, CASH = backtest(DEF)
idx = strat.index

ew = returns.loc[idx].mean(axis=1)
invvol = returns.loc[idx].copy()
iv_w = 1 / returns.rolling(60).std().loc[idx]
iv_w = iv_w.div(iv_w.sum(axis=1), axis=0)
ivr = (returns.loc[idx] * iv_w).sum(axis=1)
spy_r = spy.pct_change().reindex(idx).fillna(0)

# fair like-for-like: equal-weight scaled to the strategy's realised vol (no leverage cost assumed)
k_fair = (strat.std() / ew.std())
ew_matched = ew * k_fair

rows = [stats_of(strat, 'Corrected regime strategy'),
        stats_of(ew, 'Equal-weight (daily rebal)'),
        stats_of(ew_matched, f'Equal-weight @ strategy vol (x{k_fair:.2f})'),
        stats_of(ivr, 'Inverse-volatility'),
        stats_of(spy_r, 'SPY buy & hold')]
tbl = pd.DataFrame(rows).set_index('name')
print("\n================ CORRECTED PIPELINE ================")
print(tbl.round(4).to_string())

# ablations
no_reg, _, _, _ = backtest(DEF, use_regime=False)
no_sig, _, _, _ = backtest(DEF, signal_on=False)
abl = pd.DataFrame([stats_of(strat, 'full (regime + signal)'),
                    stats_of(no_reg, 'signal only, no regime'),
                    stats_of(no_sig, 'regime risk only, no signal')]).set_index('name')
print("\n---------------- ablation ----------------")
print(abl.round(4).to_string())

# diversification + turnover
top1 = W.max(axis=1)
neff = 1 / (W ** 2).sum(axis=1)
turn = W.diff().abs().sum(axis=1).dropna()
print(f"\ndiversification: mean top weight {top1.mean():.3f}, mean 1/HHI {neff.mean():.2f} of 10, "
      f"mean cash {CASH.mean():.3f}")
print(f"turnover: {turn.mean():.3f} per rebalance, {turn.mean() * 252 / DEF['rebal']:.2f}x per year")
print(f"regime mix (0 calm/1 normal/2 stressed): {RG.value_counts().sort_index().to_dict()}")

# significance vs equal weight
ex = strat - ew
t_p, p_p = stats.ttest_1samp(ex, 0)


def nw_t(x, lags=10):
    x = np.asarray(x, float); n = len(x); m = x.mean(); e = x - m
    s = (e @ e) / n
    for l in range(1, lags + 1):
        s += 2 * (1 - l / (lags + 1)) * ((e[l:] @ e[:-l]) / n)
    return m / np.sqrt(s / n)


rng = np.random.default_rng(42)


def sh(x): return x.mean() / x.std() * np.sqrt(252)


a, b = strat.values, ew.values
nn, blk, dif = len(a), 21, []
for _ in range(5000):
    ids = []
    while len(ids) < nn:
        s0 = int(rng.integers(0, nn)); ids.extend(range(s0, min(s0 + blk, nn)))
    ids = np.array(ids[:nn]); dif.append(sh(a[ids]) - sh(b[ids]))
dif = np.array(dif); lo, hi = np.percentile(dif, [2.5, 97.5])
print(f"\nvs equal-weight: annual excess {ex.mean() * 252:+.4%}, paired t {t_p:.3f} (p {p_p:.4f}), "
      f"Newey-West t {nw_t(ex):.3f}")
print(f"Sharpe diff {sh(a) - sh(b):+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
      f"P(better) {(dif > 0).mean():.3f}")

# ================================================================= sensitivity grid
print("\n---------------- sensitivity grid (all cells reported) ----------------")
grid = []
for ic in [0.02, 0.05, 0.10, 0.20]:
    for lam in [1.0, 2.0, 5.0]:
        for lb in [126, 252]:
            c = dict(DEF); c['ic'], c['lam'], c['cov_lookback'] = ic, lam, lb
            s_, _, _, _ = backtest(c)
            st = stats_of(s_)
            e_ = ew.reindex(s_.index)
            grid.append(dict(IC=ic, lam=lam, cov_lb=lb, Sharpe=round(st['Sharpe'], 3),
                             CAGR=round(st['CAGR'], 4), MaxDD=round(st['MaxDD'], 4),
                             beats_EW=bool(st['Sharpe'] > stats_of(e_)['Sharpe'])))
            print(f"  IC {ic:<5} lam {lam:<4} cov {lb:<4} -> Sharpe {st['Sharpe']:.3f}  "
                  f"CAGR {st['CAGR']:+.2%}  maxDD {st['MaxDD']:.2%}  beats EW: {grid[-1]['beats_EW']}")
G = pd.DataFrame(grid)
print(f"\ncells beating equal-weight on Sharpe: {G['beats_EW'].sum()} of {len(G)}")

out = dict(main=tbl.round(6).to_dict('index'), ablation=abl.round(6).to_dict('index'),
           diversification=dict(top1=float(top1.mean()), neff=float(neff.mean()),
                                cash=float(CASH.mean()),
                                turnover_per_rebal=float(turn.mean())),
           significance=dict(excess_ann=float(ex.mean() * 252), t=float(t_p), p=float(p_p),
                             nw_t=float(nw_t(ex)), sharpe_diff=float(sh(a) - sh(b)),
                             ci=[float(lo), float(hi)], p_better=float((dif > 0).mean())),
           grid=G.to_dict('records'),
           period=[str(idx[0].date()), str(idx[-1].date())], n_days=int(len(idx)))
json.dump(out, open(os.path.join(OUT, 'fixed_results.json'), 'w'), indent=2)
strat.to_csv(os.path.join(OUT, 'fixed_strategy_returns.csv'))
W.to_csv(os.path.join(OUT, 'fixed_weights.csv'))
print("\nsaved fixed_results.json")
