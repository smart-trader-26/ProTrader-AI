"""
Step 5 - figures.

Print conventions used throughout: a four-hue categorical palette validated for
colour-vision deficiency and for contrast against the page, identity carried by
line style and marker as well as by hue so the figures survive greyscale
printing, recessive grids, and no dual axes anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.use("Agg")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"

# Validated categorical palette (CVD separation, chroma, contrast all pass).
C = ["#0072B2", "#D55E00", "#009E73", "#7B3294"]
INK = "#1a1a1a"
INK2 = "#555555"
GRID = "#d9d9d9"
BASE = "#666666"          # reference lines only, never a series identity
STYLES = ["-", "--", "-.", ":"]
MARKS = ["o", "s", "^", "D"]

VARIANT_LABEL = {
    "price_plus_A": "Price + gated signal $A$",
    "price_only": "Price only",
    "price_plus_relfilt": "Price + filtered polarity",
    "price_plus_polarity": "Price + mean polarity",
    "A_only": "Text only",
}
AGG_LABEL = {
    "A_mig": "Gated $A$",
    "pol_mean": "Mean polarity",
    "pol_relf": "Filtered polarity",
    "pol_cnt": "Count-weighted",
    "add_comb": "Additive combiner",
    "A_nu": r"$s\nu$ only",
    "A_mu": r"$s\mu$ only",
}


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.edgecolor": INK2,
        "axes.linewidth": 0.7,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.9,
        "axes.axisbelow": True,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
    })


def despine(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# ---------------------------------------------------------------------------
# Figure 1 - what the three axes look like, and the veto in action
# ---------------------------------------------------------------------------
def fig_axes_and_veto() -> None:
    ev = pd.read_parquet(ROOT / "cache" / "events.parquet")
    pan = pd.read_parquet(ROOT / "cache" / "mig_panel.parquet")
    pan = pan[pan["has_scored_news"] == 1]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))

    # The model returns scores on a coarse grid rather than a continuum, so the
    # honest form for the marginals is a discrete bar chart: a smooth histogram
    # would turn that grid into comb artefacts and hide the real property.
    grid = np.round(np.arange(0, 1.0001, 0.1), 1)

    def share_on_grid(v: pd.Series) -> np.ndarray:
        snapped = np.round(np.clip(v.to_numpy(), 0, 1) * 10) / 10
        counts = pd.Series(snapped).value_counts(normalize=True)
        return np.array([counts.get(g, 0.0) for g in grid])

    ax = axes[0]
    w = 0.042
    for i, (col, lab) in enumerate([("nu", "Novelty $\\nu$"), ("mu", "Materiality $\\mu$")]):
        ax.bar(grid + (i - 0.5) * w, share_on_grid(ev[col]), width=w,
               color=C[i], edgecolor="white", linewidth=0.5, label=lab)
    ax.set_xlabel("Score")
    ax.set_ylabel("Share of events")
    ax.set_xlim(-0.07, 1.03)
    ax.set_title("(a) Novelty and materiality", loc="left")
    ax.legend(frameon=False, loc="upper right")
    despine(ax)

    # (b) the joint mass on the same discrete grid.  Almost all events sit in the
    # low-novelty, low-materiality corner, which is exactly the mass the
    # multiplicative form is designed to suppress; a power-law colour norm keeps
    # the sparse but important high-high cells visible.
    ax = axes[1]
    edges = np.round(np.arange(-0.05, 1.06, 0.1), 2)
    h, xe, ye = np.histogram2d(ev["nu"], ev["mu"], bins=[edges, edges])
    h = h / h.sum()
    im = ax.pcolormesh(xe, ye, h.T, cmap="Blues", shading="flat",
                       norm=mpl.colors.PowerNorm(gamma=0.4, vmin=0, vmax=h.max()))
    ax.set_xlabel("Novelty $\\nu$")
    ax.set_ylabel("Materiality $\\mu$")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("(b) Joint mass of the two gates", loc="left")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Share of events", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)

    # (c) the veto: session-level |A| against |mean polarity|
    ax = axes[2]
    x = pan["pol_mean"].abs().to_numpy()
    y = pan["A_mig"].abs().to_numpy()
    ax.scatter(x, y, s=2.5, alpha=0.18, color=C[0], edgecolors="none", rasterized=True)
    lim = float(np.nanpercentile(x, 99.5))
    ax.plot([0, lim], [0, lim], color=BASE, lw=1.0, ls="--", label="No gating ($|A|=|\\bar{s}|$)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("$|\\bar{s}|$, mean polarity")
    ax.set_ylabel("$|A|$, gated signal")
    ax.set_title("(c) The veto in action", loc="left")
    ax.legend(frameon=False, loc="upper left")
    despine(ax)

    fig.tight_layout()
    fig.savefig(FIGS / "fig1_axes_veto.pdf")
    plt.close(fig)
    print("wrote fig1_axes_veto.pdf")


# ---------------------------------------------------------------------------
# Figure 2 - predictive content of each aggregator, by horizon
# ---------------------------------------------------------------------------
def fig_univariate() -> None:
    uni = pd.read_csv(RESULTS / "univariate.csv")
    horizons = sorted(uni["horizon"].unique())
    order = ["A_mig", "pol_relf", "pol_mean", "pol_cnt", "add_comb", "A_nu", "A_mu"]
    order = [o for o in order if o in set(uni["aggregator"])]

    fig, axes = plt.subplots(1, len(horizons), figsize=(7.2, 2.6), sharey=True)
    if len(horizons) == 1:
        axes = [axes]
    for k, h in enumerate(horizons):
        ax = axes[k]
        sub = uni[uni["horizon"] == h].set_index("aggregator")
        ys = np.arange(len(order))
        coefs = np.array([sub.loc[o, "coef_bps"] for o in order])
        ts = np.array([sub.loc[o, "t_2way"] for o in order])
        errs = np.abs(coefs / np.where(np.abs(ts) < 1e-9, np.nan, ts)) * 1.96
        # errorbar takes a single ecolor, so each row is drawn on its own; the
        # gated aggregate is the one series that carries a hue, the alternatives
        # stay neutral so the comparison reads at a glance
        for i, o in enumerate(order):
            col = C[0] if o == "A_mig" else BASE
            ax.errorbar(coefs[i], ys[i], xerr=errs[i], fmt="none", ecolor=col,
                        elinewidth=1.1, capsize=2.0, capthick=1.0)
            ax.plot(coefs[i], ys[i], MARKS[0] if o == "A_mig" else "o",
                    ms=5 if o == "A_mig" else 3.6, color=col,
                    markeredgecolor="white", markeredgewidth=0.6, zorder=3)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_yticks(ys)
        if k == 0:
            ax.set_yticklabels([AGG_LABEL.get(o, o) for o in order])
        ax.set_title(f"$H={h}$ session" + ("s" if h > 1 else ""), loc="left")
        ax.set_xlabel("Basis points per 1 SD")
        ax.invert_yaxis()
        despine(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_univariate.pdf")
    plt.close(fig)
    print("wrote fig2_univariate.pdf")


# ---------------------------------------------------------------------------
# Figure 3 - risk-coverage curves: the threshold-free comparison
# ---------------------------------------------------------------------------
def fig_risk_coverage(tag: str = "newsrows") -> None:
    summ = json.loads((RESULTS / f"gate_summary_{tag}.json").read_text())
    horizons = sorted(summ.keys(), key=int)
    show = ["price_plus_A", "price_only", "price_plus_relfilt", "price_plus_polarity"]

    fig, axes = plt.subplots(1, len(horizons), figsize=(7.2, 2.5), sharey=False)
    if len(horizons) == 1:
        axes = [axes]
    for k, h in enumerate(horizons):
        ax = axes[k]
        curves = summ[h]["curves"]
        pooled = summ[h]["pooled"]
        for i, name in enumerate([s for s in show if s in curves]):
            cov = np.array(curves[name]["coverage"])
            risk = np.array(curves[name]["risk"])
            m = cov >= 0.02
            ax.plot(cov[m], risk[m], STYLES[i], color=C[i], lw=1.5,
                    label=f"{VARIANT_LABEL[name]} (AURC {pooled[name]['aurc']:.3f})")
        if "price_only" in pooled:
            ax.axhline(1 - pooled["price_only"]["base_rate"], color=BASE, lw=1.0,
                       ls=(0, (1, 2)), label="Always-up error rate")
        ax.set_xlabel("Coverage")
        if k == 0:
            ax.set_ylabel("Selective risk (error rate)")
        ax.set_title(f"$H={h}$ session" + ("s" if int(h) > 1 else ""), loc="left")
        ax.legend(frameon=False, loc="lower right", fontsize=6.2)
        despine(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_risk_coverage.pdf")
    plt.close(fig)
    print("wrote fig3_risk_coverage.pdf")


# ---------------------------------------------------------------------------
# Figure 4 - the exponent surface, with the unit-exponent product marked
# ---------------------------------------------------------------------------
def fig_exponents(h: int = 5) -> None:
    path = RESULTS / f"exponent_surface_h{h}.csv"
    if not path.exists():
        print(f"skip fig4: {path.name} not found")
        return
    s = pd.read_csv(path).dropna(subset=["ic"])
    piv = s.pivot(index="beta", columns="alpha", values="ic")
    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    im = ax.pcolormesh(piv.columns.values, piv.index.values, piv.values,
                       cmap="Blues", shading="auto")

    best = s.loc[s["ic"].idxmax()]
    unit = s[(s["alpha"] == 1.0) & (s["beta"] == 1.0)].iloc[0]
    pure = s[(s["alpha"] == 0.0) & (s["beta"] == 0.0)].iloc[0]

    # Three points carry the argument: the empirical optimum, the unit-exponent
    # product the theory's special case assumes, and pure polarity (no gating at
    # all).  Each is labelled with its own IC because the whole surface spans a
    # trivially narrow range -- that flatness *is* the result, and an unlabelled
    # colour scale would dress a 0.002 spread up as structure.
    for (row, mark, col, lab) in [
        (best, "*", C[1], "Best fit"),
        (unit, "o", C[3], "Unit exponents"),
        (pure, "s", C[2], "No gating"),
    ]:
        ax.plot(row["alpha"], row["beta"], mark, ms=12 if mark == "*" else 7,
                color=col, markeredgecolor="white", markeredgewidth=0.8, zorder=4,
                label=f"{lab} ($\\alpha={row['alpha']:.2f}$, $\\beta={row['beta']:.2f}$): "
                      f"IC {row['ic']:.4f}")
    ax.set_xlabel(r"Novelty exponent $\alpha$")
    ax.set_ylabel(r"Materiality exponent $\beta$")
    ax.set_title(
        f"Rank IC over the exponent grid, $H={h}$\n"
        f"(whole surface spans {s['ic'].min():.4f}--{s['ic'].max():.4f})",
        loc="left", fontsize=8)
    ax.grid(False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28),
              fontsize=6.6, handletextpad=0.4)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Rank IC", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_exponents.pdf")
    plt.close(fig)
    print("wrote fig4_exponents.pdf")


# ---------------------------------------------------------------------------
# Figure 5 - deployment evidence from the live ledger
# ---------------------------------------------------------------------------
def fig_deployment() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))

    # (a) interval coverage by horizon against the nominal 90%
    ax = axes[0]
    labels = ["5-day", "10-day", "20-day", "All"]
    emp = [100.0, 69.2, 86.8, 70.7]
    n = [8, 575, 38, 621]
    xs = np.arange(len(labels))
    ax.bar(xs, emp, width=0.55, color=C[0], edgecolor="white", linewidth=0.8)
    ax.axhline(90, color=C[1], lw=1.4, ls="--", label="Nominal 90\\%")
    for x, v, nn in zip(xs, emp, n):
        ax.text(x, v + 2.5, f"{v:.1f}", ha="center", fontsize=7, color=INK)
        ax.text(x, 4, f"$n={nn}$", ha="center", fontsize=6.4, color="white")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Empirical coverage (\\%)")
    ax.set_title("(a) Interval coverage, live ledger", loc="left")
    ax.legend(frameon=False, loc="lower right")
    despine(ax)

    # (b) walk-forward conviction gate against the always-up base, by test year
    ax = axes[1]
    years = ["2022", "2023", "2024", "2025", "2026"]
    prec = [61.6, 68.3, 53.3, 69.0, 54.5]
    base = [54.0, 65.3, 55.7, 58.2, 47.1]
    xs = np.arange(len(years))
    w = 0.36
    ax.bar(xs - w / 2, prec, width=w, color=C[0], edgecolor="white",
           linewidth=0.8, label="Fired-bucket precision")
    ax.bar(xs + w / 2, base, width=w, color=BASE, edgecolor="white",
           linewidth=0.8, label="Always-up base rate")
    ax.set_xticks(xs)
    ax.set_xticklabels(years)
    ax.set_ylim(40, 76)
    ax.set_ylabel("Up-rate (\\%)")
    ax.set_title("(b) Conviction gate by test year", loc="left")
    ax.legend(frameon=False, loc="upper left", ncol=1)
    despine(ax)

    fig.tight_layout()
    fig.savefig(FIGS / "fig5_deployment.pdf")
    plt.close(fig)
    print("wrote fig5_deployment.pdf")


def fig_convergent_validity() -> None:
    """Do the two axes track their own external criteria, or each other?"""
    path = RESULTS / "convergent_validity.json"
    if not path.exists():
        print("skip fig6: convergent_validity.json not found")
        return
    cv = json.loads(path.read_text())
    prof = pd.DataFrame(cv["decile_profile"])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))

    # (a) the two axes plotted against mechanical novelty: if they measured
    # different things these lines would separate; instead they move in lockstep
    ax = axes[0]
    x = prof["decile"].to_numpy() + 1
    for i, (col, lab) in enumerate([("nu", "Novelty $\\nu$"),
                                    ("mu", "Materiality $\\mu$")]):
        ax.plot(x, prof[col], STYLES[i], marker=MARKS[i], ms=4.5, lw=1.6,
                color=C[i], markeredgecolor="white", markeredgewidth=0.5, label=lab)
    ax.set_xlabel("Decile of mechanical novelty (1 = stalest)")
    ax.set_ylabel("Mean score")
    ax.set_xticks(x)
    ax.set_title("(a) Both axes, one profile", loc="left")
    ax.legend(frameon=False, loc="lower left")
    despine(ax)

    # (b) each axis against each criterion, standardised, with 95% intervals
    ax = axes[1]
    st, ar = cv["staleness_regression"], cv["absret_regression"]
    # Both criteria are regressed on standardised axes with a standardised
    # dependent variable, so every coefficient is in the same unit -- standard
    # deviations of criterion per standard deviation of axis -- and the four can
    # legitimately share one axis.
    items = [
        ("Novelty", "Staleness", st["nu_beta"], st["nu_t"], C[0]),
        ("Materiality", "Staleness", st["mu_beta"], st["mu_t"], C[1]),
        ("Novelty", "$|$return$|$", ar["nu_beta_z"], ar["nu_t_z"], C[0]),
        ("Materiality", "$|$return$|$", ar["mu_beta_z"], ar["mu_t_z"], C[1]),
    ]
    ys = np.arange(len(items))
    for i, (axis, crit, b, t, col) in enumerate(items):
        err = abs(b / t) * 1.96 if t else 0
        ax.errorbar(b, ys[i], xerr=err, fmt="none", ecolor=col, elinewidth=1.2,
                    capsize=2.5, capthick=1.0)
        ax.plot(b, ys[i], MARKS[0] if axis == "Novelty" else MARKS[1], ms=5.5,
                color=col, markeredgecolor="white", markeredgewidth=0.6, zorder=3)
    ax.axvline(0, color=INK2, lw=0.9)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{a}\n$\\rightarrow$ {c}" for a, c, _, _, _ in items],
                       fontsize=6.6)
    ax.set_xlabel("SD of criterion per SD of axis")
    ax.set_title("(b) Each axis against its criterion", loc="left")
    ax.invert_yaxis()
    despine(ax)

    fig.tight_layout()
    fig.savefig(FIGS / "fig6_convergent_validity.pdf")
    plt.close(fig)
    print("wrote fig6_convergent_validity.pdf")


def main() -> None:
    style()
    FIGS.mkdir(exist_ok=True)
    fig_deployment()
    fig_convergent_validity()
    if (ROOT / "cache" / "events.parquet").exists():
        fig_axes_and_veto()
    if (RESULTS / "univariate.csv").exists():
        fig_univariate()
    for tag in ("newsrows",):
        if (RESULTS / f"gate_summary_{tag}.json").exists():
            fig_risk_coverage(tag)
    fig_exponents(5)


if __name__ == "__main__":
    main()
