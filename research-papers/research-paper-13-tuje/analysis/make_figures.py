"""
Generate all data figures for the TUJE paper from the real, already-computed
Study 1 / Study 2 result files. No synthetic numbers -- every point plotted
here is read from study1_daily.csv, study1_summary.json, or
study2_fused_events.csv produced by the two analysis scripts.

Palette (validated with the dataviz skill's validate_palette.js, all-pass on
the adjacent-pair list used in each figure): technical=#2a78d6 (blue),
volatility=#008300 (green), sentiment=#4a3aa7 (violet), static fusion=#52514e
(neutral secondary ink), dynamic fusion=#e34948 (red, WARN-band vs green ->
mitigated here with hatching + direct end labels).
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RES = HERE / "results"
FIGDIR = HERE.parent / "figures"
FIGDIR.mkdir(exist_ok=True)

C_TECH = "#2a78d6"
C_VOL = "#008300"
C_SENT = "#4a3aa7"
C_STATIC = "#52514e"
C_DYNAMIC = "#e34948"
GRID = "#e1e0d9"
MUTED = "#898781"
INK = "#0b0b0b"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})


def fig2_weight_collapse():
    daily = pd.read_csv(RES / "study1_daily.csv", parse_dates=["date"])
    q = daily.set_index("date").resample("QE").agg(
        w_tech=("w_tech", "mean"), w_tech_scaled=("w_tech_scaled", "mean"),
        w_tech_min=("w_tech_scaled", "min"), w_tech_max=("w_tech_scaled", "max"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(6.6, 3.1), dpi=400)
    ax.axhline(0.5, color=MUTED, lw=1.0, ls=":", zorder=1)
    ax.fill_between(q["date"], q["w_tech_min"], q["w_tech_max"], color=C_TECH, alpha=0.12, lw=0, zorder=2)
    ax.plot(q["date"], q["w_tech_scaled"], color=C_TECH, lw=2.0, marker="o", ms=3.5,
            label="Scale-normalized $w_{tech}$ (quarterly mean, band = intra-quarter range)", zorder=4)
    ax.plot(q["date"], q["w_tech"], color=C_STATIC, lw=1.4, ls="--",
            label="Raw formula $w_i=\\exp(-\\sigma_i^2)/\\sum_j\\exp(-\\sigma_j^2)$ (quarterly mean)", zorder=3)

    ax.set_ylim(0.44, 0.56)
    ax.set_ylabel("Technical-expert weight $w_{tech}$")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.set_xlabel("Quarter (2018Q1–2026Q3, 2,088 real NSE trading days, 44 tickers)")
    ax.legend(loc="upper left", frameon=False, fontsize=7.3, handlelength=2.2)
    ax.text(0.985, 0.04, "static equal weight (0.50)", transform=ax.transAxes, ha="right",
            fontsize=7, color=MUTED, style="italic")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig2_weight_collapse.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig2_weight_collapse.png")


def fig3_peryear_accuracy():
    with open(RES / "study1_summary.json") as f:
        s = json.load(f)
    py = s["per_year_accuracy"]
    years = sorted(py.keys())
    tech = [py[y]["acc_tech"] * 100 for y in years]
    vol = [py[y]["acc_vol"] * 100 for y in years]
    static = [py[y]["acc_static"] * 100 for y in years]
    dyn = [py[y]["acc_dynamic_scaled"] * 100 for y in years]

    x = np.arange(len(years))
    w = 0.2
    fig, ax = plt.subplots(figsize=(6.6, 3.3), dpi=400)
    ax.axhline(50.0, color=MUTED, lw=1.0, ls=":", zorder=1)
    ax.bar(x - 1.5 * w, tech, width=w, color=C_TECH, label="Technical alone", zorder=3)
    ax.bar(x - 0.5 * w, vol, width=w, color=C_VOL, label="Volatility alone", zorder=3)
    ax.bar(x + 0.5 * w, static, width=w, color=C_STATIC, label="Static equal-weight fusion", zorder=3)
    ax.bar(x + 1.5 * w, dyn, width=w, color=C_DYNAMIC, hatch="////", edgecolor="white", lw=0.4,
           label="Scaled dynamic fusion", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=8)
    ax.set_ylabel("Directional accuracy (%)")
    ax.set_ylim(47, 53.5)
    ax.text(x[0] - 1.5 * w, 50.3, "chance = 50%", fontsize=6.5, color=MUTED, style="italic")
    ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig3_peryear_accuracy.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig3_peryear_accuracy.png")


def fig4_sentiment_scatter():
    df = pd.read_csv(RES / "study2_fused_events.csv")
    from scipy import stats as sstats
    slope, intercept, r, p, se = sstats.linregress(df["sent_mean"], df["true_return"])

    fig, ax = plt.subplots(figsize=(6.6, 3.1), dpi=400)
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
    ax.scatter(df["sent_mean"], df["true_return"] * 100, s=22, color=C_SENT, alpha=0.75,
               edgecolor="white", lw=0.4, zorder=3)
    xs = np.linspace(df["sent_mean"].min(), df["sent_mean"].max(), 50)
    ax.plot(xs, (intercept + slope * xs) * 100, color=C_STATIC, lw=1.6, ls="--", zorder=2,
            label=f"OLS fit: $r$={r:.3f}, $p$={p:.2f}, $n$={len(df)} (in-sample)")
    ax.set_xlabel("Real FinBERT signed sentiment (headline-level, aggregated per ticker-day)")
    ax.set_ylabel("Realized next-day return (%)")
    ax.legend(loc="upper left", frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig4_sentiment_scatter.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig4_sentiment_scatter.png")


def fig5_study2_weights():
    df = pd.read_csv(RES / "study2_fused_events.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    idx = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(6.6, 3.1), dpi=400)
    ax.axhline(1 / 3, color=MUTED, lw=1.0, ls=":", zorder=1)
    ax.plot(idx, df["w_tech"], color=C_TECH, lw=1.5, marker="o", ms=2.6, label="$w_{technical}$", zorder=3)
    ax.plot(idx, df["w_sent"], color=C_SENT, lw=1.5, marker="s", ms=2.6, label="$w_{sentiment}$", zorder=3)
    ax.plot(idx, df["w_vol"], color=C_VOL, lw=1.5, marker="^", ms=2.6, label="$w_{volatility}$", zorder=3)
    ax.set_ylim(0.15, 0.50)
    ax.set_ylabel("Fusion weight")
    ax.set_xlabel(f"Real ticker-day events with $\\geq$1 FinBERT-scored headline, in date order "
                  f"({df['date'].min().date()} – {df['date'].max().date()}, $n$={len(df)})")
    ax.text(idx[-1], 1 / 3 + 0.012, "equal weight (1/3)", fontsize=6.5, color=MUTED,
            style="italic", ha="right")
    ax.legend(loc="upper right", frameon=False, fontsize=7.5, ncol=3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig5_study2_weights.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig5_study2_weights.png")


def fig6_study3_horizon_gate():
    with open(RES / "study3_summary.json") as f:
        s = json.load(f)
    groups = ["ungated", "gated_top20pct_conviction"]
    labels = ["All predictions\n(n={:,})".format(s["n_rows_total"]),
              f"Top {int(s['gate_coverage_pct'])}% conviction\n(n={s['n_rows_gated']:,})"]
    strategies = [("acc_tech", "Technical alone", C_TECH, None),
                  ("acc_vol", "Volatility alone", C_VOL, None),
                  ("acc_static", "Static equal-weight fusion", C_STATIC, None),
                  ("acc_dynamic", "Scaled dynamic fusion", C_DYNAMIC, "////")]

    x = np.arange(len(groups))
    w = 0.19
    fig, ax = plt.subplots(figsize=(6.6, 3.3), dpi=400)
    ax.axhline(50.0, color=MUTED, lw=1.0, ls=":", zorder=1)
    for i, (key, label, color, hatch) in enumerate(strategies):
        vals = [s[g][key] * 100 for g in groups]
        bars = ax.bar(x + (i - 1.5) * w, vals, width=w, color=color, label=label,
                       hatch=hatch, edgecolor="white" if hatch else color, lw=0.4, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}", ha="center",
                    va="bottom", fontsize=6.3, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Directional accuracy (%), 20-day horizon")
    ax.set_ylim(48, 60.5)
    ax.text(x[0] - 1.5 * w, 50.4, "chance = 50%", fontsize=6.5, color=MUTED, style="italic")
    ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.30))
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig6_study3_horizon_gate.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig6_study3_horizon_gate.png")


def fig7_study4_comparison():
    """Forest plot of every leak-free dynamic-vs-static comparison. The 3-source
    cells are read from the corrected runs (study5d = both leaks fixed at H=1;
    study6 part A = lag-fixed panel at H=20), NOT from the superseded
    study4_summary_* files."""
    with open(RES / "study1_summary.json") as f:
        s1 = json.load(f)
    with open(RES / "study3_summary.json") as f:
        s3 = json.load(f)
    with open(RES / "study5d_doublefixed_results.json") as f:
        s5d = {r["label"]: r for r in json.load(f)}
    with open(RES / "study6_corrected_final.json") as f:
        s6 = json.load(f)

    c3_h1 = s5d["H1_3src_usdinr_crude"]
    c3_h20 = s6["part_a_h20_lagfixed"]

    conditions = [
        ("2 sources\nnext-day", s1["scaled_vs_static_acc_diff_mean"] * 100,
         [x * 100 for x in s1["scaled_vs_static_acc_diff_ci95"]]),
        ("3 sources\nnext-day", c3_h1["mean_diff_pp"], c3_h1["ci95_pp"]),
        ("2 sources\n20d (all)", s3["ungated"]["dynamic_minus_static_mean"] * 100,
         [x * 100 for x in s3["ungated"]["dynamic_minus_static_ci95"]]),
        ("3 sources\n20d (all)", c3_h20["mean_diff_pp"], c3_h20["ci95_pp"]),
        ("2 sources\n20d (gated)", s3["gated_top20pct_conviction"]["dynamic_minus_static_mean"] * 100,
         [x * 100 for x in s3["gated_top20pct_conviction"]["dynamic_minus_static_ci95"]]),
        ("3 sources\n20d (gated)", c3_h20["gated_top20pct"]["mean_diff_pp"],
         c3_h20["gated_top20pct"]["ci95_pp"]),
    ]

    x = np.arange(len(conditions))
    means = [c[1] for c in conditions]
    los = [c[1] - c[2][0] for c in conditions]
    his = [c[2][1] - c[1] for c in conditions]
    colors = [C_STATIC, C_DYNAMIC, C_STATIC, C_DYNAMIC, C_STATIC, C_DYNAMIC]

    fig, ax = plt.subplots(figsize=(6.6, 3.3), dpi=400)
    ax.axhline(0, color=MUTED, lw=1.0, ls=":", zorder=1)
    ax.bar(x, means, yerr=[los, his], capsize=3, color=colors, width=0.55,
           error_kw=dict(lw=1.1, ecolor=INK), zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in conditions], fontsize=7.3)
    ax.set_ylabel("Dynamic $-$ static accuracy (pp), 95% CI")
    ax.text(0.02, 0.95, "gray = 2 real sources (technical+volatility)\nred = 3 real sources (+ macro: USD/INR, crude), leak-free protocol",
            transform=ax.transAxes, fontsize=6.5, color=MUTED, va="top", style="italic")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig7_study4_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig7_study4_comparison.png")


def fig8_leak_anatomy():
    """The 2x2 leak factorial: identical data, identical fusion math, four
    evaluation protocols differing only in which look-ahead defect is present.
    All numbers read from the saved result files of the runs that produced
    them -- nothing hardcoded."""
    with open(RES / "study6b_leak_anatomy.json") as f:
        s6b = json.load(f)
    with open(RES / "study5c_precise_results.json") as f:
        s5c = {r["label"]: r for r in json.load(f)}
    with open(RES / "study5d_doublefixed_results.json") as f:
        s5d = {r["label"]: r for r in json.load(f)}

    cells = [
        ("Both leaks\n(as first run)", s6b["T_present_O_present_verified_from_csv"], C_STATIC),
        ("Ordering leak\nonly", s5c["usdinr_crude"], C_STATIC),
        ("Timezone leak\nonly", s6b["T_present_O_fixed"], C_STATIC),
        ("Leak-free\n(valid result)", s5d["H1_3src_usdinr_crude"], C_TECH),
    ]

    x = np.arange(len(cells))
    means = [c[1]["mean_diff_pp"] for c in cells]
    los = [c[1]["mean_diff_pp"] - c[1]["ci95_pp"][0] for c in cells]
    his = [c[1]["ci95_pp"][1] - c[1]["mean_diff_pp"] for c in cells]
    colors = [c[2] for c in cells]

    fig, ax = plt.subplots(figsize=(6.6, 3.3), dpi=400)
    ax.axhline(0, color=MUTED, lw=1.0, ls=":", zorder=1)
    bars = ax.bar(x, means, yerr=[los, his], capsize=3, color=colors, width=0.55,
                  error_kw=dict(lw=1.1, ecolor=INK), zorder=3)
    for b, c in zip(bars, cells):
        p = c[1]["wilcoxon_p"]
        ptxt = f"$p_W$={p:.4f}" if p < 0.01 else f"$p_W$={p:.2f}"
        ax.text(b.get_x() + b.get_width() / 2, c[1]["ci95_pp"][1] + 0.008,
                f"{c[1]['mean_diff_pp']:+.3f} pp\n{ptxt}", ha="center", va="bottom",
                fontsize=6.8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in cells], fontsize=7.5)
    ax.set_ylabel("Dynamic $-$ static accuracy (pp), 95% CI")
    ax.set_ylim(-0.06, 0.23)
    ax.text(0.985, 0.95, "identical data, identical fusion rule:\nonly the evaluation protocol differs",
            transform=ax.transAxes, fontsize=6.5, color=MUTED, va="top", ha="right", style="italic")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig8_leak_anatomy.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig8_leak_anatomy.png")


def fig9_tuned_eta():
    """Selection-window vs test-window dynamic-minus-static difference across
    the pre-specified learning-rate grid (3-source config). Shows the
    validation-chosen k*=20 reversing sign out of sample."""
    with open(RES / "study6_corrected_final.json") as f:
        s6 = json.load(f)
    grid = s6["part_b_tuned_eta"]["3src_usdinr_crude"]["grid"]
    k_star = s6["part_b_tuned_eta"]["3src_usdinr_crude"]["k_star"]

    ks = [g["k"] for g in grid]
    sel = [g["sel_mean_diff_pp"] for g in grid]
    tst = [g["test_mean_diff_pp"] for g in grid]

    x = np.arange(len(ks))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.6, 3.1), dpi=400)
    ax.axhline(0, color=MUTED, lw=1.0, ls=":", zorder=1)
    ax.bar(x - w / 2, sel, width=w, color=C_TECH,
           label="Selection window (2018–2019)", zorder=3)
    ax.bar(x + w / 2, tst, width=w, color=C_DYNAMIC, hatch="////", edgecolor="white", lw=0.4,
           label="Held-out test window (2020–2026)", zorder=3)
    ki = ks.index(k_star)
    ax.annotate(f"validation picks $k^*$={k_star}\n… which reverses sign out of sample",
                xy=(ki - w / 2, sel[ki]), xytext=(ki - 2.9, 0.16),
                fontsize=6.8, color=INK,
                arrowprops=dict(arrowstyle="->", lw=0.9, color=INK))
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks], fontsize=8)
    ax.set_xlabel("Learning-rate multiplier $k$ in $w_i \\propto \\exp(-k\\,\\sigma_i^2/\\tau_t)$ (pre-specified grid)")
    ax.set_ylabel("Dynamic $-$ static accuracy (pp)")
    ax.legend(loc="lower left", frameon=False, fontsize=7.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig9_tuned_eta.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig9_tuned_eta.png")


if __name__ == "__main__":
    fig2_weight_collapse()
    fig3_peryear_accuracy()
    fig4_sentiment_scatter()
    fig5_study2_weights()
    fig6_study3_horizon_gate()
    fig7_study4_comparison()
    fig8_leak_anatomy()
    fig9_tuned_eta()
