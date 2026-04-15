"""
figures_code.py — Generate all figures for the ProTrader AI research paper (part2.tex).

Run this script from the docs/ directory:
    cd docs
    python figures_code.py

All output PNGs are saved into docs/figures/ (created automatically).
Dependencies: matplotlib, numpy, networkx (pip install matplotlib numpy networkx)

Each section below contains:
  1. Python code to generate the figure
  2. Equivalent Mermaid diagram code (as a comment) for flow-based figures

Output files:
  fig_system_architecture.png
  fig_feature_framework.png
  fig_hybrid_model.png
  fig_dynamic_fusion.png
  fig_sentiment_pipeline.png
  fig_pattern_analysis.png
  fig_hurst_regimes.png
  fig_dashboard_tabs.png
  fig_feature_importance.png
  fig_ablation_study.png
  fig_equity_curve.png
  fig_forecast_fan.png
"""

import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.gridspec import GridSpec

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# ── Colour palette (matches Streamlit dashboard) ─────────────────────────────
C_BLUE   = "#00d4ff"   # primary
C_GREEN  = "#00ff88"   # bullish
C_RED    = "#ff4444"   # bearish
C_ORANGE = "#ffaa00"   # neutral/institutional
C_PURPLE = "#a29bfe"   # volatility
C_DARK   = "#1a1a2e"   # background
C_GREY   = "#888888"
C_NAVY   = "#16213e"

def save(name):
    path = os.path.join(OUT, name)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")


# =============================================================================
# FIGURE 1 — System Architecture
# =============================================================================
# Mermaid equivalent:
# ```mermaid
# flowchart TD
#   subgraph DataSources["Data Sources"]
#     YF[Yahoo Finance OHLCV]
#     NSE[NSE FII/DII]
#     VIX[India VIX]
#     RSS[RSS Feeds]
#     NAPI[NewsAPI]
#     RED[Reddit]
#     GT[Google Trends]
#   end
#   subgraph Features["Feature Engineering (27)"]
#     TC[Technical Core x5]
#     TE[Technical Enhanced x9]
#     TA[Technical Advanced x4]
#     SF[Sentiment x3]
#     IF[Institutional x4]
#     VF[Volatility x2]
#   end
#   subgraph Models["Hybrid Engine"]
#     XGB[XGBoost n=150]
#     LGB[LightGBM n=200]
#     CAT[CatBoost n=300]
#     GRU[LSTM-GRU Seq=60]
#     META[Ridge Meta-Stacker]
#     ISO[Isotonic Calibration]
#   end
#   subgraph Fusion["Dynamic Fusion"]
#     TE_EX[Technical Expert GRU-128]
#     SE_EX[Sentiment Expert Dense-64]
#     VE_EX[Volatility Expert MLP-32]
#     BAY[Bayesian Weight w=exp-sigma2]
#   end
#   subgraph Output["Output & UI"]
#     PRED[Price Forecast 200 paths]
#     RISK[Risk Management ATR/Kelly]
#     PAT[Pattern Analysis 12 types]
#     UI[8-Tab Dashboard]
#   end
#   DataSources --> Features --> Models --> META --> ISO --> Fusion --> Output
# ```

def fig_system_architecture():
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16); ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_facecolor("white")

    def box(x, y, w, h, label, color, fontsize=8, textcolor="white"):
        r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                           facecolor=color, edgecolor="white", linewidth=1.2)
        ax.add_patch(r)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=fontsize, color=textcolor, fontweight="bold",
                wrap=True, multialignment="center")

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))

    # Layer labels
    layers = [
        (0.3,  "Data Sources"),
        (3.2,  "Feature Eng."),
        (6.0,  "Hybrid Models"),
        (9.5,  "Fusion"),
        (12.8, "Output / UI"),
    ]
    for lx, ltxt in layers:
        ax.text(lx + 1.2, 8.6, ltxt, ha="center", va="bottom", fontsize=9,
                color="#333", fontweight="bold")

    # --- Data Sources (col 0) ---
    srcs = [("Yahoo Finance\nOHLCV", "#1565c0"),
            ("NSE FII/DII", "#1565c0"),
            ("India VIX", "#1565c0"),
            ("RSS Feeds\n(30%)", "#004d40"),
            ("NewsAPI\n(25%)", "#004d40"),
            ("Reddit\n(25%)", "#004d40"),
            ("Google\nTrends (20%)", "#004d40")]
    for i, (lbl, col) in enumerate(srcs):
        box(0.1, 7.6 - i * 1.05, 2.4, 0.85, lbl, col, fontsize=7)

    # --- Feature groups (col 1) ---
    feats = [("Technical Core (×5)\nLog_Ret, Vol_5D, RSI, VR, MA_Div", "#37474f"),
             ("Technical Enhanced (×9)\nMACD, BB, ATR, OBV, Ret_kD", "#37474f"),
             ("Technical Advanced (×4)\nCMF, Williams%R, RSI_Div", "#37474f"),
             ("Sentiment (×3)\nBase, Multi, Confidence", "#1b5e20"),
             ("Institutional (×4)\nFII/DII Net+5D Avg", "#e65100"),
             ("Volatility (×2)\nVIX_Norm, VIX_Change", "#880e4f")]
    for i, (lbl, col) in enumerate(feats):
        box(2.9, 7.6 - i * 1.21, 2.6, 1.05, lbl, col, fontsize=6.5)

    # --- Hybrid Models (col 2) ---
    mods = [("XGBoost\nn_est=150, depth=4", "#1a237e"),
            ("LightGBM\nn_est=200, leaves=31", "#1a237e"),
            ("CatBoost\niter=300, depth=6", "#1a237e"),
            ("LSTM-GRU\nseq=60, units=64/32", "#4a148c"),
            ("Ridge\nMeta-Stacker", "#b71c1c"),
            ("Isotonic\nCalibration", "#b71c1c")]
    for i, (lbl, col) in enumerate(mods):
        box(5.8, 7.6 - i * 1.21, 2.6, 1.05, lbl, col, fontsize=6.5)

    # --- Fusion (col 3) ---
    exps = [("Technical Expert\nGRU(128→64→32)", "#006064"),
            ("Sentiment Expert\nDense(64→32→16)", "#006064"),
            ("Volatility Expert\nMLP(32→16→8)", "#006064"),
            ("Bayesian Weights\nw=exp(−σ²)/Σ", "#bf360c")]
    for i, (lbl, col) in enumerate(exps):
        box(9.2, 7.5 - i * 1.5, 2.7, 1.3, lbl, col, fontsize=7)

    # --- Output (col 4) ---
    outs = [("Price Forecast\n200 Monte Carlo paths\nP5/P25/P75/P95", "#1b5e20"),
            ("Risk Management\nATR Stop-Loss\nFibonacci + Kelly", "#4a148c"),
            ("Pattern Analysis\n12 archetypes\nHybrid Classical+Vision", "#e65100"),
            ("8-Tab Dashboard\nStreamlit + Plotly\nDeepSeek+Gemini AI", "#1565c0")]
    for i, (lbl, col) in enumerate(outs):
        box(12.6, 7.5 - i * 1.8, 3.2, 1.5, lbl, col, fontsize=6.5)

    # Arrows between columns
    for y in [4.2, 5.0, 6.0, 7.0]:
        arrow(2.5,  y, 2.9,  y)
        arrow(5.5,  y, 5.8,  y)
        arrow(8.4,  y, 9.2,  y)
        arrow(11.9, y, 12.6, y)

    ax.set_title("ProTrader AI — System Architecture", fontsize=13, fontweight="bold", pad=10)
    save("fig_system_architecture.png")

fig_system_architecture()


# =============================================================================
# FIGURE 2 — Feature Framework (radial / grouped bar)
# =============================================================================

def fig_feature_framework():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis("off")

    groups = {
        "Technical Core\n(5 features)": {
            "color": "#1565c0",
            "items": ["Log_Ret", "Volatility_5D", "RSI_Norm", "Vol_Ratio", "MA_Div"]
        },
        "Technical Enhanced\n(9 features)": {
            "color": "#0288d1",
            "items": ["MACD_Norm", "MACD_Hist", "BB_PctB", "ATR_Norm",
                      "OBV_Slope", "Ret_2D", "Ret_5D", "Ret_10D", "Ret_20D"]
        },
        "Technical Advanced\n(4 features)": {
            "color": "#006064",
            "items": ["CMF_20", "Williams_R", "RSI_Bear_Div", "RSI_Bull_Div"]
        },
        "Sentiment\n(3 features)": {
            "color": "#2e7d32",
            "items": ["Sentiment", "Multi_Sentiment", "Confidence"]
        },
        "Institutional\n(4 features)": {
            "color": "#e65100",
            "items": ["FII_Net_Norm", "DII_Net_Norm", "FII_5D_Avg", "DII_5D_Avg"]
        },
        "Volatility\n(2 features)": {
            "color": "#880e4f",
            "items": ["VIX_Norm", "VIX_Change"]
        },
    }

    col_w = 14 / len(groups)
    for gi, (gname, gdata) in enumerate(groups.items()):
        gx = gi * col_w + 0.3
        # Header box
        hbox = FancyBboxPatch((gx, 5.5), col_w - 0.6, 1.1,
                              boxstyle="round,pad=0.05",
                              facecolor=gdata["color"], edgecolor="white", lw=1.5)
        ax.add_patch(hbox)
        ax.text(gx + (col_w - 0.6) / 2, 6.05, gname,
                ha="center", va="center", fontsize=7.5, color="white",
                fontweight="bold", multialignment="center")
        # Item boxes
        for fi, item in enumerate(gdata["items"]):
            fy = 4.9 - fi * 0.58
            ibox = FancyBboxPatch((gx, fy), col_w - 0.6, 0.48,
                                  boxstyle="round,pad=0.03",
                                  facecolor=gdata["color"] + "44",
                                  edgecolor=gdata["color"], lw=0.8)
            ax.add_patch(ibox)
            ax.text(gx + (col_w - 0.6) / 2, fy + 0.24, item,
                    ha="center", va="center", fontsize=6.5, color="#111")

    ax.set_xlim(0, 14); ax.set_ylim(0, 7.5)
    ax.set_title("ProTrader AI — 27-Feature Framework", fontsize=13,
                 fontweight="bold", pad=10)
    # Legend: total count
    ax.text(7, 0.2, "Total: 5 + 9 + 4 + 3 + 4 + 2 = 27 stationary features",
            ha="center", fontsize=9, style="italic", color="#444")
    save("fig_feature_framework.png")

fig_feature_framework()


# =============================================================================
# FIGURE 3 — Hybrid Model Pipeline
# =============================================================================

def fig_hybrid_model():
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.axis("off"); ax.set_xlim(0, 15); ax.set_ylim(0, 7)

    def rbox(x, y, w, h, txt, fc, fs=8):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                           facecolor=fc, edgecolor="#ccc", lw=1.2)
        ax.add_patch(p)
        ax.text(x + w/2, y + h/2, txt, ha="center", va="center",
                fontsize=fs, color="white", fontweight="bold",
                multialignment="center")

    def arr(x0, y0, x1, y1, col="#555"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.5))

    # Input
    rbox(0.2, 3.0, 1.8, 1.0, "27 Features\n(scaled)", "#37474f")

    # Tree models
    rbox(2.4, 5.2, 2.2, 0.8, "XGBoost\nn=150, d=4", "#1a237e")
    rbox(2.4, 4.0, 2.2, 0.8, "LightGBM\nn=200, d=5", "#0d47a1")
    rbox(2.4, 2.8, 2.2, 0.8, "CatBoost\niter=300, d=6", "#1565c0")
    rbox(2.4, 1.6, 2.2, 0.8, "ARIMA(2,0,2)", "#263238")

    # Neural branch
    rbox(2.4, 0.2, 2.2, 0.9, "LSTM(64→32)\n║\nGRU(64→32)", "#4a148c")

    arr(2.0, 3.5, 2.4, 5.6)
    arr(2.0, 3.5, 2.4, 4.4)
    arr(2.0, 3.5, 2.4, 3.2)
    arr(2.0, 3.5, 2.4, 2.0)
    arr(2.0, 3.5, 2.4, 0.65)

    # OOF predictions
    rbox(5.2, 3.5, 2.2, 0.9, "Out-of-Fold\nPredictions\n(5-fold expand.)", "#4e342e")
    arr(4.6, 5.6, 5.2, 3.9)
    arr(4.6, 4.4, 5.2, 3.9)
    arr(4.6, 3.2, 5.2, 3.9)

    # GRU dense head
    rbox(5.2, 0.2, 2.2, 0.9, "Dense(32→16→1)\nDropout(0.2)", "#6a1b9a")
    arr(4.6, 0.65, 5.2, 0.65)

    # Stacker
    rbox(8.0, 3.2, 2.2, 1.1, "Ridge\nMeta-Stacker\n(α=1.0)", "#b71c1c")
    arr(7.4, 3.9, 8.0, 3.75)
    arr(7.4, 0.65, 8.0, 3.3)

    # Dynamic weight
    rbox(8.0, 1.5, 2.2, 0.9, "Dynamic\nWeight Adjust.\nw_xgb ≤ 0.65", "#bf360c")
    arr(9.1, 3.2, 9.1, 2.4)

    # Isotonic
    rbox(11.0, 2.8, 2.2, 1.0, "Isotonic\nCalibration\nP(up|ŷ)", "#880e4f")
    arr(10.2, 2.75, 11.0, 3.3)
    arr(10.2, 1.95, 11.0, 3.3)

    # Output
    rbox(13.6, 2.6, 1.2, 1.4, "ŷ\nreturn\n+\nprob", "#1b5e20", fs=7)
    arr(13.2, 3.3, 13.6, 3.3)

    ax.set_title("Hybrid Prediction Engine Pipeline", fontsize=13,
                 fontweight="bold", pad=10)
    save("fig_hybrid_model.png")

fig_hybrid_model()


# =============================================================================
# FIGURE 4 — Dynamic Fusion Framework
# =============================================================================

def fig_dynamic_fusion():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis("off"); ax.set_xlim(0, 14); ax.set_ylim(0, 7)

    def rbox(x, y, w, h, txt, fc, fs=8):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                           facecolor=fc, edgecolor="#ccc", lw=1.2)
        ax.add_patch(p)
        ax.text(x + w/2, y + h/2, txt, ha="center", va="center",
                fontsize=fs, color="white", fontweight="bold",
                multialignment="center")

    def arr(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="#444", lw=1.5))

    # Inputs
    rbox(0.2, 5.6, 2.0, 0.8, "30-Day\nTechnical\nFeatures", "#1565c0")
    rbox(0.2, 3.6, 2.0, 0.8, "8 Sentiment\nFeatures", "#2e7d32")
    rbox(0.2, 1.6, 2.0, 0.8, "6 Volatility\nFeatures\n(VIX)", "#880e4f")

    # Experts
    rbox(3.0, 5.3, 2.8, 1.4,
         "Technical Expert\nGRU(128) → GRU(64)\n→ GRU(32) → Dense(1)", "#006064")
    rbox(3.0, 3.3, 2.8, 1.4,
         "Sentiment Expert\nDense(64) → Dense(32)\n→ Dense(16) → Dense(1)", "#006064")
    rbox(3.0, 1.3, 2.8, 1.4,
         "Volatility Expert\nMLP(32) → MLP(16)\n→ MLP(8) → Dense(1)", "#006064")

    arr(2.2, 6.0, 3.0, 6.0)
    arr(2.2, 4.0, 3.0, 4.0)
    arr(2.2, 2.0, 3.0, 2.0)

    # Uncertainty
    rbox(6.6, 5.3, 2.2, 1.4,
         "σ²_tech\nRolling MSE\n(N=15 days)", "#bf360c")
    rbox(6.6, 3.3, 2.2, 1.4,
         "σ²_sent\nRolling MSE\n(N=15 days)", "#bf360c")
    rbox(6.6, 1.3, 2.2, 1.4,
         "σ²_vol\nRolling MSE\n(N=15 days)", "#bf360c")

    arr(5.8, 6.0, 6.6, 6.0)
    arr(5.8, 4.0, 6.6, 4.0)
    arr(5.8, 2.0, 6.6, 2.0)

    # Bayesian weights
    rbox(9.5, 3.3, 2.5, 1.4,
         "Bayesian Weights\nwᵢ = exp(−σᵢ²)\n  / Σ exp(−σⱼ²)\n\nΣwᵢ = 1.0", "#e65100")
    arr(8.8, 6.0, 9.5, 4.0)
    arr(8.8, 4.0, 9.5, 4.0)
    arr(8.8, 2.0, 9.5, 4.0)

    # Fusion output
    rbox(12.3, 3.3, 1.4, 1.4,
         "ŷ_fusion\n= Σwᵢ·ŷᵢ", "#1b5e20")
    arr(12.0, 4.0, 12.3, 4.0)

    ax.set_title("Bayesian Dynamic Fusion Framework", fontsize=13,
                 fontweight="bold", pad=10)
    # Annotation for weight evolution
    ax.text(7.0, 0.3, "← Weights evolve daily; displayed in Tab 2 weight-evolution chart →",
            ha="center", fontsize=8, style="italic", color="#555")
    save("fig_dynamic_fusion.png")

fig_dynamic_fusion()


# =============================================================================
# FIGURE 5 — Sentiment Pipeline
# =============================================================================

def fig_sentiment_pipeline():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off"); ax.set_xlim(0, 14); ax.set_ylim(0, 6)

    def rbox(x, y, w, h, txt, fc, fs=8):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                           facecolor=fc, edgecolor="#ccc", lw=1.2)
        ax.add_patch(p)
        ax.text(x + w/2, y + h/2, txt, ha="center", va="center",
                fontsize=fs, color="white", fontweight="bold",
                multialignment="center")

    def arr(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="#444", lw=1.5))

    srcs = [("RSS Feeds\n(Moneycontrol,\nET, Mint, BS)\nWeight: 30%", "#1565c0", 4.8),
            ("NewsAPI\n(Global financial\nnews aggregation)\nWeight: 25%", "#00695c", 3.3),
            ("Reddit\n(r/IndianStockMarket\ndalal, investments)\nWeight: 25%", "#6a1b9a", 1.8),
            ("Google Trends\n(retail search\ninterest proxy)\nWeight: 20%", "#e65100", 0.3)]

    for lbl, col, yy in srcs:
        rbox(0.2, yy, 2.4, 1.2, lbl, col, fs=7)

    rbox(3.3, 2.4, 2.0, 1.2,
         "Temporal\nDecay\nw=e^{−0.3d}", "#37474f")
    rbox(6.0, 2.4, 2.0, 1.2,
         "Event-Type\nWeight\n(1×..2×)", "#4e342e")
    rbox(8.7, 2.4, 2.5, 1.2,
         "DistilRoBERTa\n-Financial\ns∈[−1,+1]", "#880e4f")
    rbox(11.7, 4.0, 2.0, 0.9,
         "Disagreement\nDetector\n(std>0.2)", "#bf360c")
    rbox(11.7, 2.4, 2.0, 1.2,
         "Combined\nSentiment\nScore + Label\n+ Confidence", "#1b5e20")

    for yy in [5.4, 3.9, 2.4, 0.9]:
        arr(2.6, yy + 0.6, 3.3, 3.0)
    arr(5.3, 3.0, 6.0, 3.0)
    arr(8.0, 3.0, 8.7, 3.0)
    arr(11.2, 3.0, 11.7, 3.0)
    ax.annotate("", xy=(12.7, 4.0), xytext=(12.7, 3.6),
                arrowprops=dict(arrowstyle="->", color="#bf360c", lw=1.2))

    ax.set_title("Multi-Source Sentiment Aggregation Pipeline", fontsize=13,
                 fontweight="bold", pad=10)
    save("fig_sentiment_pipeline.png")

fig_sentiment_pipeline()


# =============================================================================
# FIGURE 6 — Pattern Analysis Overview
# =============================================================================

def fig_pattern_analysis():
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 3, figure=fig, wspace=0.35)

    # Subplot 1: Multi-timeframe diagram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 5); ax1.set_ylim(0, 7); ax1.axis("off")
    ax1.set_title("Multi-Timeframe Scanning", fontweight="bold", fontsize=9)
    for i, (order, label, col) in enumerate([(3, "Order 3\n(Minor swings)", "#1565c0"),
                                              (5, "Order 5\n(Medium swings)", "#e65100"),
                                              (7, "Order 7\n(Major swings)", "#880e4f")]):
        p = FancyBboxPatch((0.3, 5.2 - i * 1.8), 4.4, 1.5,
                           boxstyle="round,pad=0.1", facecolor=col + "33",
                           edgecolor=col, lw=1.5)
        ax1.add_patch(p)
        ax1.text(2.5, 6.0 - i * 1.8, f"scipy.argrelextrema\norder={order}\n{label}",
                 ha="center", va="center", fontsize=8, color="#111",
                 multialignment="center")
    ax1.text(2.5, 0.3, "↓  Merge & deduplicate peaks/troughs",
             ha="center", fontsize=7.5, style="italic", color="#555")

    # Subplot 2: 12 patterns table
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    ax2.set_title("12 Pattern Archetypes", fontweight="bold", fontsize=9)
    patterns = [("Double Top",           "Bearish", "#ff4444"),
                ("Double Bottom",        "Bullish", "#00ff88"),
                ("Head & Shoulders",     "Bearish", "#ff4444"),
                ("Inverse H&S",          "Bullish", "#00ff88"),
                ("Ascending Triangle",   "Bullish", "#00ff88"),
                ("Descending Triangle",  "Bearish", "#ff4444"),
                ("Symmetric Triangle",   "Neutral", "#ffaa00"),
                ("Ascending Channel",    "Bullish", "#00ff88"),
                ("Descending Channel",   "Bearish", "#ff4444"),
                ("Bull Flag",            "Bullish", "#00ff88"),
                ("Bear Flag",            "Bearish", "#ff4444"),
                ("BB Squeeze",           "Neutral", "#ffaa00")]
    for i, (pname, bias, col) in enumerate(patterns):
        yy = 6.5 - i * 0.52
        ax2.text(0.1, yy, pname, fontsize=7.5, va="center", color="#222")
        ax2.text(3.8, yy, bias, fontsize=7.5, va="center", color=col,
                 fontweight="bold", ha="right")

    # Subplot 3: Hybrid consensus
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_xlim(0, 5); ax3.set_ylim(0, 7); ax3.axis("off")
    ax3.set_title("Hybrid Consensus", fontweight="bold", fontsize=9)
    for yy, lbl, col in [(5.5, "Classical\nscipy Detection", "#1565c0"),
                          (3.5, "Roboflow\nVision API\n(conf ≥ 0.30)", "#4a148c"),
                          (1.5, "Hybrid\nConsensus\nLayer", "#1b5e20")]:
        p = FancyBboxPatch((0.3, yy), 4.4, 1.2, boxstyle="round,pad=0.1",
                           facecolor=col + "33", edgecolor=col, lw=1.5)
        ax3.add_patch(p)
        ax3.text(2.5, yy + 0.6, lbl, ha="center", va="center", fontsize=8,
                 color="#111", multialignment="center")
    ax3.annotate("", xy=(2.5, 3.5), xytext=(2.5, 5.5),
                 arrowprops=dict(arrowstyle="->", color="#444", lw=1.5))
    ax3.annotate("", xy=(2.5, 1.5), xytext=(2.5, 3.5),
                 arrowprops=dict(arrowstyle="->", color="#444", lw=1.5))
    ax3.text(2.5, 0.4, "Precision: 64.1% → 71.3% → 78.4%",
             ha="center", fontsize=7.5, style="italic", color="#555")

    fig.suptitle("Chart Pattern Recognition System", fontsize=13,
                 fontweight="bold", y=0.98)
    save("fig_pattern_analysis.png")

fig_pattern_analysis()


# =============================================================================
# FIGURE 7 — Hurst Regime Detection
# =============================================================================

def fig_hurst_regimes():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: R/S curve example
    ax = axes[0]
    lags = np.arange(2, 21)
    H_trend = 0.65; H_rw = 0.50; H_mr = 0.35
    np.random.seed(42)
    for H, label, col in [(H_trend, f"Trending (H={H_trend})", "#00ff88"),
                           (H_rw,    f"Random Walk (H={H_rw})", "#ffaa00"),
                           (H_mr,    f"Mean-Rev. (H={H_mr})", "#ff4444")]:
        rs = lags ** H * np.exp(np.random.normal(0, 0.02, len(lags)))
        ax.plot(np.log(lags), np.log(rs), "-o", color=col, label=label,
                linewidth=2, markersize=4)
    ax.set_xlabel("log(lag)", fontsize=10)
    ax.set_ylabel("log(R/S)", fontsize=10)
    ax.set_title("R/S Analysis — Hurst Estimation\n(slope = H)", fontsize=10)
    ax.legend(fontsize=8)
    ax.axvline(x=np.log(10), color="#ccc", linestyle="--", alpha=0.5)
    ax.set_facecolor("#f9f9f9")
    ax.grid(alpha=0.3)

    # Right: Decision tree
    ax2 = axes[1]
    ax2.set_xlim(0, 10); ax2.set_ylim(0, 10); ax2.axis("off")
    ax2.set_title("Regime Classification Logic", fontsize=10, fontweight="bold")

    def dbox(x, y, w, h, txt, col, fs=8):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                           facecolor=col, edgecolor="#ccc", lw=1.2)
        ax2.add_patch(p)
        ax2.text(x + w/2, y + h/2, txt, ha="center", va="center",
                 fontsize=fs, color="white", fontweight="bold",
                 multialignment="center")

    dbox(3.5, 8.5, 3.0, 1.0, "Recent 120 closes\n→ compute H (R/S)", "#37474f")
    dbox(3.5, 6.8, 3.0, 1.0, "Vol > 85th pct?", "#37474f")
    dbox(7.5, 6.8, 2.0, 1.0, "HIGH\nVOLATILITY", "#880e4f")
    dbox(3.5, 5.1, 3.0, 1.0, "|slope|>0.002\nR²>0.3 and H≥0.45?", "#37474f")
    dbox(7.5, 5.1, 2.0, 1.0, "TRENDING", "#1b5e20")
    dbox(3.5, 3.4, 3.0, 1.0, "H<0.45 or\nVol<30th+flat?", "#37474f")
    dbox(7.5, 3.4, 2.0, 1.0, "MEAN\nREVERTING", "#1565c0")
    dbox(3.5, 1.7, 3.0, 1.0, "Otherwise", "#37474f")
    dbox(7.5, 1.7, 2.0, 1.0, "NORMAL", "#4e342e")

    for y in [8.5, 6.8, 5.1, 3.4, 1.7]:
        ax2.annotate("", xy=(3.5, y - 0.2) if y > 1.7 else (3.5, y),
                     xytext=(5.0, y if y == 8.5 else y + 1.0),
                     arrowprops=dict(arrowstyle="->", color="#444", lw=1.2))
    for y in [6.8, 5.1, 3.4, 1.7]:
        ax2.annotate("", xy=(7.5, y + 0.5), xytext=(6.5, y + 0.5),
                     arrowprops=dict(arrowstyle="->", color="#ff4444", lw=1.2))
        ax2.text(7.0, y + 0.55, "Yes", ha="center", fontsize=7, color="#cc0000")

    fig.suptitle("Hurst Exponent and Market Regime Detection", fontsize=12,
                 fontweight="bold")
    plt.tight_layout()
    save("fig_hurst_regimes.png")

fig_hurst_regimes()


# =============================================================================
# FIGURE 8 — Dashboard Tabs Overview
# =============================================================================

def fig_dashboard_tabs():
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis("off"); ax.set_xlim(0, 16); ax.set_ylim(0, 6)

    tabs = [
        ("Tab 1\n📊 Dashboard",
         "• Current price metrics\n• Candlestick/line chart\n• AI forecast fan (P5-P95)\n"
         "• Directional accuracy\n• SHAP importance\n• Verdict card (BULLISH/BEARISH)\n"
         "• DeepSeek+Gemini analysis",
         "#1565c0"),
        ("Tab 2\n🔬 Dynamic Fusion",
         "• Live Bayesian weights\n  (Technical/Sentiment/Vol)\n"
         "• 15-day weight evolution\n• Stacked-area chart",
         "#006064"),
        ("Tab 3\n📈 Technicals & Risk",
         "• Fibonacci retracement\n• ATR stop-loss\n• Trade Setup Calculator\n"
         "• Entry/Stop/Target/RR",
         "#37474f"),
        ("Tab 4\n🏛️ Fundamentals",
         "• Forward P/E\n• PEG Ratio\n• Price/Book\n• Debt/Equity\n• ROE\n"
         "• Profit Margins",
         "#4e342e"),
        ("Tab 5\n💼 FII/DII",
         "• NSE institutional data\n• Daily net buy/sell bars\n"
         "• 20-day cumulative lines\n• FII/DII trend flags",
         "#e65100"),
        ("Tab 6\n📰 Sentiment",
         "• Combined score+label\n• Event pie chart\n• Article table\n"
         "• Per-source breakdown\n• Disagreement warning",
         "#2e7d32"),
        ("Tab 7\n🛠️ Backtest",
         "• After-cost performance\n• Benchmark comparison\n"
         "• Equity curve\n• Monte Carlo (1000 paths)\n• Pipeline timings",
         "#880e4f"),
        ("Tab 8\n📐 Patterns",
         "• 12 pattern archetypes\n• Confidence scores\n• Target prices\n"
         "• Volume confirmation\n• Support/resistance\n• Hurst regime",
         "#4a148c"),
    ]

    tw = 16 / len(tabs)
    for i, (title, body, col) in enumerate(tabs):
        x = i * tw + 0.1
        # Header
        hp = FancyBboxPatch((x, 4.5), tw - 0.2, 1.3, boxstyle="round,pad=0.05",
                            facecolor=col, edgecolor="white", lw=1.5)
        ax.add_patch(hp)
        ax.text(x + (tw - 0.2)/2, 5.15, title, ha="center", va="center",
                fontsize=6.5, color="white", fontweight="bold",
                multialignment="center")
        # Body
        bp = FancyBboxPatch((x, 0.2), tw - 0.2, 4.2, boxstyle="round,pad=0.05",
                            facecolor=col + "22", edgecolor=col, lw=1)
        ax.add_patch(bp)
        ax.text(x + (tw - 0.2)/2, 2.3, body, ha="center", va="center",
                fontsize=5.8, color="#222", multialignment="left",
                fontfamily="monospace")

    ax.set_title("ProTrader AI — Eight-Tab Interactive Dashboard", fontsize=13,
                 fontweight="bold", pad=8)
    save("fig_dashboard_tabs.png")

fig_dashboard_tabs()


# =============================================================================
# FIGURE 9 — SHAP Feature Importance (Table 4 values)
# =============================================================================

def fig_feature_importance():
    features = ["Log_Ret", "Volatility_5D", "RSI_Norm", "Multi_Sentiment",
                "VIX_Norm", "FII_Net_Norm", "DII_Net_Norm", "Vol_Ratio",
                "CMF_20", "MA_Div"]
    shap_vals = [0.187, 0.142, 0.098, 0.089, 0.082, 0.076, 0.071, 0.065, 0.058, 0.054]
    cats      = ["Technical"]*3 + ["Sentiment"] + ["Volatility"] + \
                ["Institutional"]*2 + ["Technical"]*2 + ["Technical"]
    cat_cols  = {"Technical": "#1565c0", "Sentiment": "#2e7d32",
                 "Volatility": "#880e4f", "Institutional": "#e65100"}
    colors = [cat_cols[c] for c in cats]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(features[::-1], shap_vals[::-1], color=colors[::-1], edgecolor="white")
    for bar, val in zip(bars, shap_vals[::-1]):
        ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title("SHAP Feature Importance — XGBoost Model\n"
                 "(Mean absolute SHAP across 10 NSE stocks, test set)", fontsize=11)
    legend_patches = [mpatches.Patch(color=v, label=k) for k, v in cat_cols.items()]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
    ax.axvline(x=0.05, color="#ccc", linestyle="--", alpha=0.5,
               label="0.05 threshold")
    ax.set_facecolor("#fafafa"); ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    save("fig_feature_importance.png")

fig_feature_importance()


# =============================================================================
# FIGURE 10 — Ablation Study (Table 5 values)
# =============================================================================

def fig_ablation_study():
    configs = ["Full Model", "− Institutional", "− Sentiment",
               "− Dynamic Fusion", "− LSTM-GRU", "− VIX Features",
               "− CMF/Divergences", "Tech Core Only"]
    acc = [55.8, 53.9, 54.1, 54.2, 54.3, 54.6, 55.1, 52.4]
    cols = ["#1b5e20"] + ["#e65100"] * 6 + ["#b71c1c"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(configs, acc, color=cols, edgecolor="white", width=0.6)
    ax.axhline(y=55.8, color="#1b5e20", linestyle="--", lw=1.5,
               label="Full model (55.8%)")
    ax.axhline(y=50.0, color="#888", linestyle=":", lw=1.2,
               label="Random baseline (50.0%)")
    for bar, val in zip(bars, acc):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.1,
                f"{val:.1f}%", ha="center", fontsize=8.5, fontweight="bold")
    ax.set_ylim(50, 57.5)
    ax.set_ylabel("Direction Accuracy (%)", fontsize=11)
    ax.set_title("Ablation Study — Direction Accuracy by Configuration\n"
                 "(10 NSE stocks, test set, p-values in Table 5)", fontsize=11)
    ax.legend(fontsize=9)
    plt.xticks(rotation=25, ha="right", fontsize=9)
    ax.set_facecolor("#fafafa"); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save("fig_ablation_study.png")

fig_ablation_study()


# =============================================================================
# FIGURE 11 — Equity Curves (Table 6 values)
# =============================================================================

def fig_equity_curve():
    np.random.seed(42)
    n = 250
    t = np.arange(n)

    def gen_equity(total_r, vol, seed=0):
        """Simulate equity curve with a given total return and volatility."""
        np.random.seed(seed)
        daily = total_r / n
        rets = np.random.normal(daily, vol, n)
        equity = 100000 * np.cumprod(1 + rets)
        return equity

    curves = {
        "NIFTY Buy-and-Hold":    (0.243, 0.012, "#888888", "--"),
        "MA Crossover (20/50)":  (0.216, 0.011, "#aaaaaa", ":"),
        "5-Model Ensemble":      (0.317, 0.010, "#00d4ff", "-"),
        "Dynamic Fusion":        (0.342, 0.009, "#00ff88", "-"),
        "Sentiment-Only":        (0.184, 0.013, "#ff9999", "--"),
    }

    fig, ax = plt.subplots(figsize=(11, 6))
    for label, (tr, vol, col, ls) in curves.items():
        eq = gen_equity(tr, vol, seed=hash(label) % 100)
        lw = 2.5 if "Dynamic" in label else (2.0 if "Ensemble" in label else 1.2)
        ax.plot(t, eq, color=col, linestyle=ls, linewidth=lw, label=label)

    # Monte Carlo band for Dynamic Fusion
    np.random.seed(99)
    paths = []
    for _ in range(1000):
        rets = np.random.normal(0.342/n, 0.009, n)
        paths.append(100000 * np.cumprod(1 + rets))
    paths = np.array(paths)
    ax.fill_between(t, np.percentile(paths, 5, axis=0),
                    np.percentile(paths, 95, axis=0),
                    alpha=0.12, color="#00ff88", label="P5–P95 (MC, Dynamic Fusion)")

    ax.set_xlabel("Trading Days", fontsize=11)
    ax.set_ylabel("Portfolio Value (₹)", fontsize=11)
    ax.set_title("Strategy Equity Curves — ₹100,000 Initial Capital\n"
                 "(Indicative simulation; see Table 6 for exact after-cost figures)",
                 fontsize=11)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}k"))
    ax.legend(fontsize=9, loc="upper left")
    ax.set_facecolor("#f9f9f9"); ax.grid(alpha=0.3)
    plt.tight_layout()
    save("fig_equity_curve.png")

fig_equity_curve()


# =============================================================================
# FIGURE 12 — Probabilistic Forecast Fan Chart
# =============================================================================

def fig_forecast_fan():
    np.random.seed(7)
    hist_days = 30
    fcast_days = 10
    n_paths = 200
    start_price = 2800.0
    daily_vol = 0.015
    bull_prob = 0.62  # example directional probability

    # Historical prices
    hist = start_price * np.cumprod(1 + np.random.normal(0, daily_vol, hist_days))

    # Monte Carlo forecast
    drift = (2 * bull_prob - 1) * 0.0008
    paths = np.zeros((n_paths, fcast_days))
    for i in range(n_paths):
        r = np.random.normal(drift, daily_vol, fcast_days)
        paths[i] = hist[-1] * np.cumprod(1 + r)

    t_hist = np.arange(hist_days)
    t_fcast = np.arange(hist_days - 1, hist_days + fcast_days)
    anchor = hist[-1]
    p5  = np.concatenate([[anchor], np.percentile(paths, 5,  axis=0)])
    p25 = np.concatenate([[anchor], np.percentile(paths, 25, axis=0)])
    p50 = np.concatenate([[anchor], np.percentile(paths, 50, axis=0)])
    p75 = np.concatenate([[anchor], np.percentile(paths, 75, axis=0)])
    p95 = np.concatenate([[anchor], np.percentile(paths, 95, axis=0)])

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(t_hist, hist, color="#00d4ff", lw=2, label="Historical (last 30 days)")
    ax.fill_between(t_fcast, p5, p95, alpha=0.10, color="#00d4ff",
                    label="P5–P95 outer band")
    ax.fill_between(t_fcast, p25, p75, alpha=0.22, color="#00d4ff",
                    label="P25–P75 inner band")
    ax.plot(t_fcast, p5,  color="#ff6b6b", lw=1, linestyle=":",
            label="P5  (Pessimistic)")
    ax.plot(t_fcast, p95, color="#7bed9f", lw=1, linestyle=":",
            label="P95 (Optimistic)")
    ax.plot(t_fcast, p50, color="#ff9800", lw=2.5, label="Median forecast")
    ax.axvline(x=hist_days - 1, color="#aaa", linestyle="--", lw=1)
    ax.text(hist_days - 0.5, ax.get_ylim()[1] * 0.97, "Today",
            fontsize=8, color="#888")
    ax.set_xlabel("Days", fontsize=11)
    ax.set_ylabel("Price (₹)", fontsize=11)
    ax.set_title(f"Probabilistic 10-Day Forecast Fan (200 Monte Carlo Paths)\n"
                 f"Bullish Probability: {bull_prob*100:.0f}% | "
                 f"Drift = {drift*100:+.3f}%/day",
                 fontsize=11)
    ax.legend(fontsize=8.5, ncol=2)
    ax.set_facecolor("#f9f9f9"); ax.grid(alpha=0.3)
    plt.tight_layout()
    save("fig_forecast_fan.png")

fig_forecast_fan()


# =============================================================================
# FIGURE 13 — Dynamic Expert Weight Evolution (illustrative)
# =============================================================================

def fig_weight_evolution():
    np.random.seed(3)
    days = 15
    # Simulate realistic weight series that sum to 1
    base = np.array([0.45, 0.28, 0.27])
    noise = np.random.dirichlet(np.ones(3) * 8, days)
    weights = 0.3 * base + 0.7 * noise

    fig, ax = plt.subplots(figsize=(10, 5))
    d = np.arange(days)
    ax.stackplot(d, weights[:, 0], weights[:, 1], weights[:, 2],
                 labels=["Technical Expert", "Sentiment Expert", "Volatility Expert"],
                 colors=["#00d4ff", "#00ff88", "#ff9800"], alpha=0.85)
    ax.set_xlabel("Trading Day (relative to today)", fontsize=11)
    ax.set_ylabel("Expert Weight (sums to 1.0)", fontsize=11)
    ax.set_title("Dynamic Fusion — Expert Weight Evolution (Last 15 Days)\n"
                 "Bayesian weights: wᵢ = exp(−σᵢ²) / Σ exp(−σⱼ²)",
                 fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_facecolor("#f9f9f9"); ax.grid(alpha=0.2)
    plt.tight_layout()
    save("fig_weight_evolution.png")

fig_weight_evolution()


print("\n✅ All figures generated in:", os.path.abspath(OUT))
print("   Place the PNG files next to part2.tex (or in figures/ with \\graphicspath{{figures/}})")
print("   Then compile: pdflatex part2.tex  (run twice)")
