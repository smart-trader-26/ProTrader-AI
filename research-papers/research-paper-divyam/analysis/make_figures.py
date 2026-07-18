"""Paper figures for the recalibration cadence law. Outputs PDFs to ../."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
OUT = HERE.parent

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 200,
    "axes.linewidth": 0.6,
})

ledger = json.loads((HERE / "results_ledger.json").read_text())
sim = json.loads((HERE / "results_sim.json").read_text())

# ---------------- Fig 1: live ledger calibration drift ----------------
tab = [b for b in ledger["coverage_table"] if b["age"] <= 12 and b["n"] >= 25]
t = np.array([b["age"] for b in tab], float)
g = np.array([b["gap"] for b in tab])
se = np.array([b["se"] for b in tab])
lin = ledger["fit_linear"]
sq = ledger["fit_sqrt"]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.3))
ax.errorbar(t, 100 * g, yerr=100 * 1.96 * se, fmt="o", ms=3.5, lw=0.8,
            capsize=2, color="#1f5fa8", label="measured gap (95% CI)")
tt = np.linspace(1, 12, 100)
ax.plot(tt, 100 * (lin["intercept"] + lin["slope"] * tt), "-", color="#c44e52",
        lw=1.2, label=f"affine: $E_0+\\gamma t$ ($\\hat\\gamma$={100*lin['slope']:.2f} pp/d)")
ax.plot(tt, 100 * sq["slope"] * np.sqrt(tt), "--", color="#55a868",
        lw=1.2, label=f"diffusive: $\\gamma\\sqrt{{t}}$ ($\\hat\\gamma$={100*sq['slope']:.2f} pp)")
ax.axhline(0, color="k", lw=0.5, alpha=0.4)
ax.set_xlabel("forecast age $t$ = days since model fit")
ax.set_ylabel("coverage gap  $90\\% - \\mathrm{cov}(t)$  [pp]")
ax.set_title("(a) Interval calibration decay — live ledger, 564 forecasts")
ax.legend(loc="lower right", frameon=False)

dt = np.array([d["age"] for d in ledger["dir_table"] if d["age"] <= 12], float)
dh = np.array([d["hit"] for d in ledger["dir_table"] if d["age"] <= 12])
dn = np.array([d["n"] for d in ledger["dir_table"] if d["age"] <= 12], float)
dse = np.sqrt(dh * (1 - dh) / dn)
hd = ledger["hit_decay"]
ax2.errorbar(dt, 100 * dh, yerr=100 * 1.96 * dse, fmt="s", ms=3.5, lw=0.8,
             capsize=2, color="#1f5fa8", label="hit-rate (95% CI)")
ax2.plot(tt, 100 * (hd["intercept"] + hd["slope"] * tt), "-", color="#c44e52", lw=1.2,
         label=f"fit: {100*hd['slope']:+.2f} pp/day")
ax2.axhline(50, color="k", lw=0.6, ls=":", alpha=0.6)
ax2.set_xlabel("forecast age $t$ [days]")
ax2.set_ylabel("directional hit-rate [%]")
ax2.set_title("(b) Directional skill decay with staleness")
ax2.legend(loc="lower left", frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig_ledger.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------- Fig 2: A(T) theory vs Monte Carlo ----------------
at = sim["AT_curve"]
fig, ax = plt.subplots(figsize=(3.45, 2.2))
ax.plot(at["T"], at["mc"], "o", ms=3, color="#1f5fa8", label="Monte Carlo (400k days)")
ax.plot(at["T"], at["theory"], "-", color="#c44e52", lw=1.2,
        label="theory  $c/T+\\gamma^2T^2/3$")
ax.axvline(at["t_star"], color="#55a868", lw=1.0, ls="--",
           label=f"$T^*$ = {at['t_star']:.1f} d")
ax.set_xlabel("recalibration interval $T$ [days]")
ax.set_ylabel("average loss rate $A(T)$")
ax.set_title(f"$c$={at['c']}, $\\gamma$={at['gamma']}, $\\beta$=1")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig_AT.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------- Fig 3: scaling laws ----------------
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.3))
colors = {"0.5": "#55a868", "1.0": "#1f5fa8"}
for b, d in sim["E2"].items():
    cs, th = np.array(d["cs"]), np.array(d["t_hat_c"], float)
    ax.loglog(cs, th, "o", ms=3.5, color=colors[b])
    fitl = np.poly1d(np.polyfit(np.log(cs), np.log(th), 1))
    ax.loglog(cs, np.exp(fitl(np.log(cs))), "-", lw=1.0, color=colors[b],
              label=f"$\\beta$={b}: slope {d['slope_c']:.2f} (theory {d['expect_c']:.2f})")
ax.set_xlabel("recalibration cost $c$")
ax.set_ylabel("empirical optimum $\\hat T$ [days]")
ax.set_title("(a) $\\hat T \\propto c^{1/(2\\beta+1)}$")
ax.legend(frameon=False)
for b, d in sim["E2"].items():
    gs, th = np.array(d["gs"]), np.array(d["t_hat_g"], float)
    ax2.loglog(gs, th, "s", ms=3.5, color=colors[b])
    fitl = np.poly1d(np.polyfit(np.log(gs), np.log(th), 1))
    ax2.loglog(gs, np.exp(fitl(np.log(gs))), "-", lw=1.0, color=colors[b],
               label=f"$\\beta$={b}: slope {d['slope_g']:.2f} (theory {d['expect_g']:.2f})")
ax2.set_xlabel("drift rate $\\gamma$")
ax2.set_ylabel("empirical optimum $\\hat T$ [days]")
ax2.set_title("(b) $\\hat T \\propto \\gamma^{-2/(2\\beta+1)}$")
ax2.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig_scaling.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------- Fig 4: penalty curve + E1 ratios ----------------
pc = sim["penalty_curve"]
fig, ax = plt.subplots(figsize=(3.45, 2.2))
ax.plot(pc["x"], pc["beta_1"], "-", color="#1f5fa8", lw=1.2,
        label="$\\beta$=1:  $\\frac{2}{3x}+\\frac{x^2}{3}$")
ax.plot(pc["x"], pc["beta_05"], "--", color="#55a868", lw=1.2,
        label="$\\beta$=0.5:  $\\frac{1}{2x}+\\frac{x}{2}$")
e1x = np.array([r["ratio"] for r in sim["E1"]])
ax.plot(e1x, [1.0] * len(e1x), "|", color="#c44e52", ms=8,
        label="E1 empirical $\\hat T/T^*$")
ax.axvline(1, color="k", lw=0.5, ls=":")
for x0 in (2 ** (1 / 3),):
    ax.annotate("2$\\times$ cost error\n$\\to$ +5.8% loss", xy=(x0, (2/(3*x0) + x0**2/3)),
                xytext=(1.8, 1.10), fontsize=6.5,
                arrowprops=dict(arrowstyle="->", lw=0.6))
ax.set_xscale("log")
ax.set_xlabel("cadence misspecification $x = T/T^*$")
ax.set_ylabel("loss penalty $A(xT^*)/A(T^*)$")
ax.set_ylim(0.95, 2.0)
ax.set_xlim(0.25, 4)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "fig_penalty.pdf", bbox_inches="tight")
plt.close(fig)

print("wrote fig_ledger.pdf, fig_AT.pdf, fig_scaling.pdf, fig_penalty.pdf")
