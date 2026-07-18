"""Simulation validation of the recalibration cadence law.

Law (general drift exponent beta > 0):
    gap(t) = gamma * t^beta   (L2 calibration gap, t = days since recalibration)
    excess Brier rate r(t) = rho * gap(t)^2          (rho = forecasts/day)
    A(T) = c/T + rho*gamma^2*T^{2beta} / (2beta+1)   (renewal-reward average)
    T*   = [ (2beta+1) c / (2 beta rho gamma^2) ]^{1/(2beta+1)}
    A*   = (2beta+1)/(2beta) * c / T*
    A(xT*)/A* = (2beta/(2beta+1)) / x + x^{2beta} / (2beta+1)

Experiments
  E1  Monte Carlo: drifting-shift forecaster over Bernoulli outcomes; empirical
      argmin of measured average loss vs theoretical T* across a (c, gamma) grid.
  E2  Scaling exponents: log-log slope of empirical T-hat vs c (expect 1/(2b+1))
      and vs gamma (expect -2/(2b+1)), for beta in {0.5, 1.0}.
  E3  Mechanism robustness: temperature drift (gap varies with q, REL > ECE_1^2):
      law applied with the L2 drift rate stays accurate; with the L1 rate it
      underestimates loss (lower bound), quantified.
  E4  Plug-in estimation: estimate gamma from a finite monitoring window with
      binomial noise, compute plug-in T-hat, measure regret via penalty curve.

Outputs: results_sim.json + CSVs for figures.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RNG = np.random.default_rng(7)

EPS = 1e-4


def t_star(c, gamma, beta=1.0, rho=1.0):
    return ((2 * beta + 1) * c / (2 * beta * rho * gamma ** 2)) ** (1.0 / (2 * beta + 1))


def a_theory(T, c, gamma, beta=1.0, rho=1.0):
    return c / T + rho * gamma ** 2 * T ** (2 * beta) / (2 * beta + 1)


def penalty(x, beta=1.0):
    return (2 * beta / (2 * beta + 1)) / x + x ** (2 * beta) / (2 * beta + 1)


def sample_q(n):
    """True event probabilities: logit-normal centred at 0.55 (drifting market)."""
    return 1 / (1 + np.exp(-(RNG.normal(0.2, 0.9, n))))


def run_cycle_loss(T, c, gamma, beta, n_days, mech="shift", rho=1):
    """Average daily excess loss (vs perfectly calibrated forecaster) + amortized cost.

    mech='shift': p = clip(q + gap(t)); uniform additive gap -> REL = ECE^2.
    mech='temp' : logit(p) = (1 + k*t^beta) * logit(q); gap varies with q.
    """
    n_days = int(n_days)
    q = sample_q(n_days * rho).reshape(n_days, rho)
    y = (RNG.random((n_days, rho)) < q).astype(float)
    age = (np.arange(n_days) % T + 1).astype(float)  # days since recalibration
    gap_t = gamma * age ** beta
    if mech == "shift":
        p = np.clip(q + gap_t[:, None], EPS, 1 - EPS)
    elif mech == "temp":
        # temperature drift: same *average L1* gap magnitude per day by construction
        # (k chosen per-day so mean |p-q| = gap_t)
        lq = np.log(q / (1 - q))
        p = np.empty_like(q)
        for i in range(n_days):
            # scale factor solving mean|sigmoid((1+k)lq) - q| = gap target, bisection
            target = gap_t[i]
            lo_k, hi_k = 0.0, 20.0
            for _ in range(40):
                mid = 0.5 * (lo_k + hi_k)
                pm = 1 / (1 + np.exp(-(1 + mid) * lq[i]))
                if np.mean(np.abs(pm - q[i])) < target:
                    lo_k = mid
                else:
                    hi_k = mid
            p[i] = 1 / (1 + np.exp(-(1 + 0.5 * (lo_k + hi_k)) * lq[i]))
        p = np.clip(p, EPS, 1 - EPS)
    else:
        raise ValueError(mech)
    excess = np.mean(np.sum((p - y) ** 2 - (q - y) ** 2, axis=1))  # per day
    return excess + c / T


def empirical_argmin(c, gamma, beta, n_days, mech="shift", rho=1, grid=None):
    if grid is None:
        ts = t_star(c, gamma, beta, rho)
        grid = np.unique(np.clip(np.round(ts * np.geomspace(0.25, 4.0, 25)), 1, None)).astype(int)
    losses = np.array([run_cycle_loss(int(T), c, gamma, beta, n_days, mech, rho) for T in grid])
    return grid, losses, int(grid[np.argmin(losses)])


results = {}

# ---------------- E1: argmin agreement on a grid ----------------
print("E1: empirical argmin vs theoretical T*")
e1 = []
for beta in (0.5, 1.0):
    for c in (0.002, 0.01, 0.05):
        for gamma in (0.004, 0.008, 0.016):
            ts = t_star(c, gamma, beta)
            grid, losses, that = empirical_argmin(c, gamma, beta, n_days=400_000)
            # interpolated minimum (parabola around argmin) for sub-grid resolution
            e1.append(dict(beta=beta, c=c, gamma=gamma, t_star=float(ts), t_hat=that,
                           ratio=float(that / ts)))
            print(f"  beta={beta} c={c:<6} gamma={gamma:<6} T*={ts:7.2f}  T^={that:4d}  ratio={that/ts:.3f}")
results["E1"] = e1
ratios = np.array([r["ratio"] for r in e1])
results["E1_summary"] = dict(mean_ratio=float(ratios.mean()), max_abs_log=float(np.max(np.abs(np.log(ratios)))))
print(f"  mean ratio {ratios.mean():.3f}, worst |log ratio| {np.max(np.abs(np.log(ratios))):.3f}")

# ---------------- E2: scaling exponents ----------------
print("\nE2: scaling exponents")
e2 = {}
for beta in (0.5, 1.0):
    expect_c = 1.0 / (2 * beta + 1)
    expect_g = -2.0 / (2 * beta + 1)
    cs = np.geomspace(0.001, 0.1, 7)
    th = [empirical_argmin(c, 0.008, beta, n_days=300_000)[2] for c in cs]
    slope_c = float(np.polyfit(np.log(cs), np.log(th), 1)[0])
    gs = np.geomspace(0.002, 0.03, 7)
    th_g = [empirical_argmin(0.01, g, beta, n_days=300_000)[2] for g in gs]
    slope_g = float(np.polyfit(np.log(gs), np.log(th_g), 1)[0])
    e2[str(beta)] = dict(slope_c=slope_c, expect_c=expect_c,
                         slope_g=slope_g, expect_g=expect_g,
                         cs=cs.tolist(), t_hat_c=[int(t) for t in th],
                         gs=gs.tolist(), t_hat_g=[int(t) for t in th_g])
    print(f"  beta={beta}: slope vs c = {slope_c:+.3f} (theory {expect_c:+.3f}); "
          f"slope vs gamma = {slope_g:+.3f} (theory {expect_g:+.3f})")
results["E2"] = e2

# ---------------- E3: mechanism robustness (temperature drift) ----------------
print("\nE3: temperature-drift mechanism (REL > ECE_L1^2)")
e3 = []
beta = 1.0
for c, gamma in ((0.01, 0.008), (0.02, 0.012)):
    ts_l1 = t_star(c, gamma, beta)
    grid = np.unique(np.clip(np.round(ts_l1 * np.geomspace(0.3, 3.0, 21)), 1, None)).astype(int)
    losses = np.array([run_cycle_loss(int(T), c, gamma, beta, 60_000, "temp") for T in grid])
    that = int(grid[np.argmin(losses)])
    # effective L2 drift rate measured from the mechanism at age grid
    ages = np.arange(1, 30, dtype=float)
    q = sample_q(20_000)
    lq = np.log(q / (1 - q))
    l2 = []
    for t in ages:
        target = gamma * t
        lo_k, hi_k = 0.0, 20.0
        for _ in range(40):
            mid = 0.5 * (lo_k + hi_k)
            pm = 1 / (1 + np.exp(-(1 + mid) * lq))
            if np.mean(np.abs(pm - q)) < target:
                lo_k = mid
            else:
                hi_k = mid
        pm = 1 / (1 + np.exp(-(1 + 0.5 * (lo_k + hi_k)) * lq))
        l2.append(np.sqrt(np.mean((pm - q) ** 2)))
    gamma2 = float(np.polyfit(ages, l2, 1)[0])
    ts_l2 = t_star(c, gamma2, beta)
    e3.append(dict(c=c, gamma_l1=gamma, gamma_l2=gamma2, t_hat=that,
                   t_star_l1=float(ts_l1), t_star_l2=float(ts_l2),
                   ratio_l1=float(that / ts_l1), ratio_l2=float(that / ts_l2)))
    print(f"  c={c} g1={gamma}: T^={that}, T*(L1)={ts_l1:.1f} (ratio {that/ts_l1:.2f}), "
          f"gamma2={gamma2:.4f}, T*(L2)={ts_l2:.1f} (ratio {that/ts_l2:.2f})")
results["E3"] = e3

# ---------------- E4: plug-in estimation with finite monitoring window ----------------
print("\nE4: plug-in T* from noisy gamma estimates (m outcomes/day, w-day window)")
e4 = []
beta, c, gamma = 1.0, 0.01, 0.008
ts_true = t_star(c, gamma, beta)
for m in (50, 200, 1000):
    regrets = []
    for rep in range(400):
        w = 15  # monitoring window of w days after a recalibration
        ages = np.arange(1, w + 1, dtype=float)
        # measured |gap| per age from m Bernoulli outcomes with true shift gamma*age
        q = sample_q(w * m).reshape(w, m)
        p = np.clip(q + (gamma * ages)[:, None], EPS, 1 - EPS)
        y = (RNG.random((w, m)) < q).astype(float)
        ghat = np.abs(p.mean(axis=1) - y.mean(axis=1))  # noisy ECE per age
        g_est = float(np.polyfit(ages, ghat, 1)[0])
        g_est = max(g_est, 1e-4)
        t_plug = t_star(c, g_est, beta)
        regrets.append(penalty(t_plug / ts_true, beta))
    regrets = np.array(regrets)
    e4.append(dict(m=m, mean_regret=float(regrets.mean()),
                   p90_regret=float(np.quantile(regrets, 0.9))))
    print(f"  m={m:5d}: mean excess loss vs oracle {100*(regrets.mean()-1):.1f}%, "
          f"p90 {100*(np.quantile(regrets,0.9)-1):.1f}%")
results["E4"] = e4

# ---------------- penalty curve data for the figure ----------------
xs = np.geomspace(0.25, 4, 60)
results["penalty_curve"] = dict(
    x=xs.tolist(),
    beta_1=[float(penalty(x, 1.0)) for x in xs],
    beta_05=[float(penalty(x, 0.5)) for x in xs],
)

# ---------------- A(T) curve (theory vs MC) for figure ----------------
c0, g0 = 0.01, 0.008
grid = np.arange(1, 41)
mc = [run_cycle_loss(int(T), c0, g0, 1.0, 400_000) for T in grid]
results["AT_curve"] = dict(T=grid.tolist(), mc=[float(v) for v in mc],
                           theory=[float(a_theory(T, c0, g0)) for T in grid],
                           c=c0, gamma=g0, t_star=float(t_star(c0, g0)))

(HERE / "results_sim.json").write_text(json.dumps(results, indent=2))
print("\nwrote results_sim.json")
