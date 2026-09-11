"""Generate the data-driven figures for the pre-synopsis report.

Every number plotted here is transcribed from the candidate's own five papers;
no value is simulated. The source paper of each series is given in the
docstring of the corresponding function.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linestyle": ":",
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})

C = ["#1f3b63", "#4a7fb5", "#9dbdd9", "#c8552b", "#e0a45a", "#6b6b6b"]


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", p)


def fig_attention():
    """Cross-modal attention weights by market regime (conference paper C1, MSFN)."""
    regimes = ["Overall\naverage", "Earnings\nseason", "High\nvolatility"]
    data = {
        "Financial metrics": [35, 42, 28],
        "News sentiment": [22, 28, 25],
        "Social media": [18, 15, 28],
        "Macroeconomic": [25, 15, 19],
    }
    fig, ax = plt.subplots(figsize=(5.4, 2.45))
    x = np.arange(len(regimes))
    w = 0.2
    for i, (k, v) in enumerate(data.items()):
        b = ax.bar(x + (i - 1.5) * w, v, w, label=k, color=C[i], edgecolor="white", linewidth=0.5)
        ax.bar_label(b, fmt="%d", padding=1.5, fontsize=6.8)
    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.set_ylabel("Attention weight (%)")
    ax.set_ylim(0, 50)
    ax.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.30))
    save(fig, "f_attention.pdf")


def fig_walkforward():
    """Expanding-window protocol and fold-wise stability (C1, Table V)."""
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.45), gridspec_kw={"width_ratios": [1.05, 1]})
    fig.subplots_adjust(wspace=0.40)

    ax = axes[0]
    for i in range(5):
        y = 4 - i
        ax.broken_barh([(0, 4 + i)], (y - 0.28, 0.56), facecolors=C[1], edgecolor="white")
        ax.broken_barh([(4 + i, 1)], (y - 0.28, 0.56), facecolors=C[3], edgecolor="white")
        ax.text(-0.35, y, "F%d" % (i + 1), va="center", ha="right", fontsize=7.5)
    ax.set_xlim(-0.5, 11.8)
    ax.set_ylim(-0.75, 4.9)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_title("(a) Expanding-window walk-forward protocol", fontsize=8.5)
    ax.plot([], [], color=C[1], lw=6, label="training window")
    ax.plot([], [], color=C[3], lw=6, label="evaluation fold")
    ax.legend(frameon=False, loc="center left", fontsize=7.2, bbox_to_anchor=(0.78, 0.60),
              handlelength=1.4, labelspacing=0.9)
    ax.annotate("", xy=(9.3, -0.45), xytext=(0, -0.45),
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#444"))
    ax.text(4.6, -0.72, "chronological time", ha="center", fontsize=7.2, color="#444")

    ax = axes[1]
    folds = np.arange(1, 6)
    da = [80.0, 80.0, 92.0, 68.0, 72.0]
    mae = [0.0012, 0.0014, 0.0024, 0.0016, 0.0015]
    b = ax.bar(folds, da, 0.55, color=C[1], edgecolor="white", linewidth=0.5,
               label="Directional accuracy")
    ax.bar_label(b, fmt="%.0f", padding=1.5, fontsize=7)
    ax.axhline(78.4, color=C[3], lw=1.1, ls="--", label="mean 78.4%")
    ax.axhline(50, color="#888", lw=0.9, ls=":", label="chance level")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Walk-forward fold")
    ax.set_ylabel("Directional accuracy (%)")
    ax2 = ax.twinx()
    ax2.plot(folds, mae, "o-", color=C[5], ms=3.5, lw=1.1, label="normalised MAE")
    ax2.set_ylabel("Normalised MAE")
    ax2.set_ylim(0, 0.004)
    ax2.grid(False)
    ax2.spines["top"].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=6.8, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.26))
    ax.set_title("(b) Fold-wise stability over the test period", fontsize=8.5)
    save(fig, "f_walkforward.pdf")


def fig_models():
    """Single-name directional classification with LLM sentiment features (journal J1)."""
    models = ["Random\nForest", "SVM", "Logistic\nRegression", "XGBoost +\nSentiment"]
    acc = [0.4982, 0.4246, 0.4702, 0.4772]
    pre = [0.4345, 0.4246, 0.4257, 0.4103]
    rec = [0.6033, 1.0000, 0.7107, 0.5289]
    f1 = [0.5052, 0.5961, 0.5325, 0.4621]
    fig, ax = plt.subplots(figsize=(5.9, 2.45))
    x = np.arange(len(models))
    w = 0.2
    for i, (v, k) in enumerate(zip([acc, pre, rec, f1],
                                   ["Accuracy", "Precision", "Recall", "F1-score"])):
        b = ax.bar(x + (i - 1.5) * w, v, w, label=k, color=C[i], edgecolor="white", linewidth=0.5)
        ax.bar_label(b, fmt="%.2f", padding=1.5, fontsize=6.5)
    ax.axhline(0.5, color="#888", lw=0.9, ls=":")
    ax.text(-0.44, 0.515, "chance", fontsize=7, color="#666", ha="left")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.14)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.19))
    save(fig, "f_models.pdf")


def fig_sentiment_diag():
    """External validity and regime separation of the sentiment composite (journal J3)."""
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.6), gridspec_kw={"width_ratios": [1.3, 1]})
    fig.subplots_adjust(wspace=0.44)

    ax = axes[0]
    series = ["Michigan\nconsumer\nsentiment", "Cboe\nvolatility\nindex",
              "St. Louis Fed\nfinancial\nstress", "Chicago Fed\nfinancial\nconditions"]
    lev = [0.262, -0.541, -0.606, -0.513]
    chg = [0.240, -0.112, -0.254, -0.123]
    x = np.arange(len(series))
    w = 0.36
    ax.bar(x - w / 2, lev, w, label="levels", color=C[0], edgecolor="white", linewidth=0.5)
    ax.bar(x + w / 2, chg, w, label="first differences", color=C[2], edgecolor="white", linewidth=0.5)
    for xi, v in zip(x - w / 2, lev):
        ax.text(xi, v + (0.03 if v > 0 else -0.075), "%+.2f" % v, ha="center", fontsize=6.6)
    for xi, v in zip(x + w / 2, chg):
        ax.text(xi, v + (0.03 if v > 0 else -0.075), "%+.2f" % v, ha="center", fontsize=6.6)
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(series, fontsize=6.8)
    ax.set_ylabel("Correlation")
    ax.set_ylim(-0.80, 0.48)
    ax.legend(frameon=False, fontsize=7.2, loc="lower left", ncol=2)
    ax.set_title("(a) Validation against four external series", fontsize=8.5)
    ax.text(0.0, 0.37, "expected $+$", ha="center", fontsize=6.6, color="#555")
    ax.text(2.05, 0.37, "expected $-$", ha="center", fontsize=6.6, color="#555")

    ax = axes[1]
    states = ["Calm\n$(n=160)$", "Transitional\n$(n=56)$", "Stress\n$(n=7)$"]
    mean = [0.073, -0.140, -0.831]
    sd = [0.407, 0.382, 0.735]
    ax.bar(states, mean, 0.55, yerr=sd, capsize=3, color=[C[2], C[4], C[3]],
           edgecolor="white", linewidth=0.5, error_kw=dict(lw=0.8, ecolor="#555"))
    for xi, m in enumerate(mean):
        ax.text(xi + 0.33, m, "%+.3f" % m, ha="left", va="center", fontsize=7)
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_ylabel("Sentiment composite")
    ax.set_xlim(-0.6, 2.95)
    ax.set_ylim(-1.90, 0.78)
    ax.tick_params(axis="x", labelsize=7.2)
    ax.set_title("(b) Composite by identified market state", fontsize=8.5)
    ax.text(1.15, -1.76, "ANOVA $F=19.34$,  $p=1.8\\times10^{-8}$", ha="center",
            fontsize=7, color="#333")
    save(fig, "f_sentiment_diag.pdf")


def fig_ablation():
    """Matched-exposure ablation and stress-episode behaviour (journal J3)."""
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.8), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.subplots_adjust(wspace=0.66)

    ax = axes[0]
    cfg = ["Full\nframework", "No regime\ntiming\n(matched)", "No regime\ntiming\n(full)",
           "No signal\nlayer", "Equal-weight\nbenchmark"]
    sharpe = [0.878, 0.795, 0.923, 0.908, 0.628]
    dd = [18.98, 22.26, 22.53, 18.87, 36.08]
    x = np.arange(len(cfg))
    w = 0.38
    b1 = ax.bar(x - w / 2, sharpe, w, color=C[0], edgecolor="white", linewidth=0.5,
                label="Sharpe ratio")
    ax.bar_label(b1, fmt="%.3f", fontsize=6.3, padding=1.5)
    ax.set_ylabel("Sharpe ratio")
    ax.set_ylim(0, 1.26)
    ax.set_xticks(x)
    ax.set_xticklabels(cfg, fontsize=6.3)
    ax2 = ax.twinx()
    b2 = ax2.bar(x + w / 2, dd, w, color=C[3], edgecolor="white", linewidth=0.5,
                 label="Maximum drawdown")
    ax2.bar_label(b2, fmt="%.1f", fontsize=6.3, padding=1.5)
    ax2.set_ylabel("Maximum drawdown (%)")
    ax2.set_ylim(0, 50)
    ax2.grid(False)
    ax2.spines["top"].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7, loc="upper center",
              bbox_to_anchor=(0.5, 1.21), ncol=2)
    ax.set_title("(a) Component ablation at matched exposure", fontsize=8.5, y=1.21)

    ax = axes[1]
    ep = ["Global financial\ncrisis 2008-09", "Euro area\ncrisis 2011", "COVID-19\ncrash 2020",
          "Inflation bear\nmarket 2022", "Holdout drawdown\n2025"]
    adv = [22.74, 10.65, 14.99, 7.73, 5.42]
    col = [C[0], C[0], C[0], C[0], C[4]]
    pos = np.arange(len(ep))[::-1]
    b = ax.barh(pos, adv, 0.55, color=col, edgecolor="white", linewidth=0.5)
    ax.bar_label(b, fmt="+%.2f pp", fontsize=6.8, padding=2)
    ax.set_yticks(pos)
    ax.set_yticklabels(ep, fontsize=6.6)
    ax.set_xlabel("Advantage over equal-weight\nbenchmark (percentage points)")
    ax.set_xlim(0, 31)
    ax.set_title("(b) Behaviour in stress episodes", fontsize=8.5, y=1.21)
    save(fig, "f_ablation.pdf")


if __name__ == "__main__":
    fig_attention()
    fig_walkforward()
    fig_models()
    fig_sentiment_diag()
    fig_ablation()
    print("done")
