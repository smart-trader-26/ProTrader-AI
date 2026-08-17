"""Step 7: figures for the paper.

All figures are drawn with distinguishable line styles and markers so that
they remain readable when printed in grayscale, and with 8 pt minimum type
as required by the journal template.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.dpi": 400, "savefig.dpi": 400, "axes.linewidth": 0.6,
    "grid.linewidth": 0.4, "lines.linewidth": 1.1,
})

W1 = 3.35   # one column, inches
W2 = 6.90   # two columns


def fig_block_diagram(out: str) -> None:
    """Single-column signal-flow diagram of the identification and the loop."""
    fig, ax = plt.subplots(figsize=(W1, 2.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(cx, cy, w, h, label, fs=6.6):
        ax.add_patch(plt.Rectangle((cx - w / 2, cy - h / 2), w, h,
                                   fill=False, lw=0.8))
        ax.text(cx, cy, label, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2, style="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=0.7,
                                    linestyle=style, color="0.1"))

    box(0.20, 0.90, 0.34, 0.13, "headline stream")
    box(0.20, 0.68, 0.34, 0.13, "scorers\n$m_1\\ldots m_4$")
    box(0.20, 0.44, 0.34, 0.15, "session\naggregation $x(t)$")
    box(0.72, 0.68, 0.40, 0.15, "cross-scorer\ncovariance $R_{SS}$")
    box(0.72, 0.44, 0.40, 0.15, "kernel $\\tilde{g}$\n(deconvolution)")
    box(0.72, 0.21, 0.40, 0.13, "matched filter $w$")
    box(0.20, 0.21, 0.34, 0.13, "forecast $\\hat{r}$")

    arrow(0.20, 0.835, 0.20, 0.75)
    arrow(0.20, 0.615, 0.20, 0.52)
    arrow(0.37, 0.68, 0.52, 0.68)
    arrow(0.37, 0.44, 0.52, 0.485)
    arrow(0.72, 0.605, 0.72, 0.52)
    arrow(0.72, 0.365, 0.72, 0.275)
    arrow(0.52, 0.21, 0.37, 0.21)

    # feedback path
    ax.plot([0.20, 0.20, 0.95, 0.95], [0.145, 0.06, 0.06, 0.21],
            lw=0.7, color="0.1", ls="--")
    ax.annotate("", xy=(0.92, 0.21), xytext=(0.95, 0.21),
                arrowprops=dict(arrowstyle="->", lw=0.7, ls="--",
                                color="0.1"))
    ax.text(0.50, 0.015, "$e(t)$, loop gain $\\mu$", ha="center",
            fontsize=6.6)
    fig.tight_layout(pad=0.1)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_kernel(kern: dict, out: str) -> None:
    K = kern["K"]
    k = np.arange(K)
    fig, ax = plt.subplots(figsize=(W1, 2.35))
    lo = np.array(kern["boot_corrected_lo"])
    hi = np.array(kern["boot_corrected_hi"])
    ax.fill_between(k, lo, hi, color="0.85", label="95% CI (corrected)")
    ax.plot(k, kern["lag_profile"], "s--", ms=3, color="0.45",
            label="lag profile")
    ax.plot(k, kern["deconvolved"], "^-.", ms=3, color="0.25",
            label="deconvolved")
    ax.plot(k, kern["corrected"], "o-", ms=3.2, color="0.0",
            label="noise corrected")
    ax.axhline(0, lw=0.5, color="0.6")
    ax.set_xlabel("lag $k$ (sessions)")
    ax.set_ylabel("response $g(k)$")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout(pad=0.2)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_reliability(kern: dict, out: str) -> None:
    st = pd.DataFrame(kern["reliability_by_news"])
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    lab = [f"{int(r.n_lo)}" if r.n_lo == r.n_hi else
           (f"{int(r.n_lo)}+" if r.n_hi > 1000 else
            f"{int(r.n_lo)}-{int(r.n_hi)}") for r in st.itertuples()]
    x = np.arange(len(st))
    ax.plot(x, st["rel_index"], "o-", ms=3.5, color="0.0")
    ax.set_xticks(x)
    ax.set_xticklabels(lab)
    ax.set_xlabel("headlines per session")
    ax.set_ylabel("reliability index $\\hat{\\lambda}$")
    ax.grid(alpha=0.3)
    fig.tight_layout(pad=0.2)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_loop(loop: dict, sweep: pd.DataFrame, out: str) -> None:
    """Error against loop gain.

    Beyond the stability bound the error diverges by many orders of
    magnitude, so the vertical axis is clipped to the region where the loop
    is usable; otherwise the informative part of the curve is a flat line.
    """
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    d = sweep.dropna(subset=["mse"]).sort_values("mu")
    y = d["mse"] * 1e4
    ref = float(loop["mse_open"] * 1e4)
    lo = min(float(y.min()), ref)
    # the simplified trace bound is necessary, not sufficient: a gain just
    # inside it can still diverge, so cap the axis at a small multiple of
    # the open-loop error rather than trusting the stability flag
    stable = d[d["mu"] < loop["mu_max"]]
    hi = float((stable["mse"] * 1e4).max()) if len(stable) else float(y.max())
    hi = max(min(hi, 3.0 * ref), ref)
    pad = 0.12 * (hi - lo) if hi > lo else 0.1 * max(abs(hi), 1e-9)
    ax.semilogx(d["mu"], y, "o-", ms=3, color="0.0", label="closed loop")
    ax.axhline(ref, ls="--", lw=0.9, color="0.45", label="open loop")
    ax.axvline(loop["mu_star_pred"], ls=":", lw=1.0, color="0.0",
               label="$\\mu^{*}$ predicted")
    ax.axvline(loop["mu_max"], ls="-.", lw=0.9, color="0.55",
               label="stability bound")
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("loop gain $\\mu$")
    ax.set_ylabel("out-of-sample MSE ($\\times 10^{-4}$)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False, fontsize=7.2, loc="best")
    fig.tight_layout(pad=0.2)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_filters(cmp_: pd.DataFrame, out: str, horizon: int = 1) -> None:
    d = cmp_[cmp_["horizon"] == horizon].copy()
    order = ["latest", "uniform-2", "uniform-3", "uniform-5", "uniform-10",
             "cwin-2", "cwin-3", "cwin-5", "cwin-10",
             "exp-1", "exp-2", "exp-3", "exp-5", "lagprofile", "corrected",
             "wiener"]
    d["o"] = d["filter"].apply(lambda s: order.index(s) if s in order else 99)
    d = d.sort_values("o")
    fig, ax = plt.subplots(figsize=(W1, 2.5))
    x = np.arange(len(d))
    col, hatch = [], []
    for f in d["filter"]:
        if f in ("corrected", "wiener"):
            col.append("0.2")
            hatch.append("")
        elif f == "lagprofile":
            col.append("0.55")
            hatch.append("")
        else:
            col.append("0.85")
            hatch.append("")
    ax.barh(x, d["ic"] * 100, color=col, edgecolor="0.1", linewidth=0.5)
    ax.set_yticks(x)
    ax.set_yticklabels(d["filter"], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("out-of-sample IC ($\\times 10^{-2}$)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout(pad=0.2)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--figdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.figdir, exist_ok=True)
    J = lambda n: json.load(open(os.path.join(args.outdir, n)))  # noqa: E731

    fig_block_diagram(os.path.join(args.figdir, "fig1_block.png"))
    print("fig1 ok")
    kern = J("kernel.json")
    fig_kernel(kern, os.path.join(args.figdir, "fig2_kernel.png"))
    print("fig2 ok")
    if os.path.exists(os.path.join(args.outdir, "kernel_lrv_innov.json")):
        fig_kernel(J("kernel_lrv_innov.json"),
                   os.path.join(args.figdir, "fig2b_kernel_vol.png"))
        print("fig2b ok")
    if kern.get("reliability_by_news"):
        fig_reliability(kern, os.path.join(args.figdir, "fig3_reliability.png"))
        print("fig3 ok")
    p = os.path.join(args.outdir, "filter_comparison_lrv_innov.csv")
    if not os.path.exists(p):
        p = os.path.join(args.outdir, "filter_comparison.csv")
    if os.path.exists(p):
        fig_filters(pd.read_csv(p), os.path.join(args.figdir,
                                                 "fig4_filters.png"))
        print("fig4 ok")
    for tag in ("_lrv_innov", ""):
        p = os.path.join(args.outdir, f"closed_loop{tag}.json")
        s = os.path.join(args.outdir, f"loop_gain_sweep{tag}.csv")
        if os.path.exists(p) and os.path.exists(s):
            fig_loop(J(f"closed_loop{tag}.json"), pd.read_csv(s),
                     os.path.join(args.figdir, "fig5_loop.png"))
            print(f"fig5 ok ({tag or 'ret'})")
            break


if __name__ == "__main__":
    main()
