"""Redraw the framework diagram at 300 dpi, matching the pipeline as actually specified.

The submitted diagram showed a news/text ingestion path that the implementation does not
contain; this version shows the four stages that exist. Geometry is computed top-down so
that boxes size themselves to their contents and nothing overlaps.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

IMG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   '..', 'single column', 'images'))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'savefig.dpi': 300,
                     'figure.dpi': 300, 'savefig.bbox': 'tight',
                     'mathtext.fontset': 'dejavusans'})

NAVY, TEAL, RUST, INK, GREY = '#1f4e79', '#2c7fb8', '#c1571a', '#1a1a1a', '#7a8794'
W = 100.0                      # drawing width
TITLE_H, LINE_H, PAD = 8.8, 4.4, 3.2
GAP_BOX, GAP_STAGE = 5.0, 9.5
LABEL_H = 5.5

fig, ax = plt.subplots(figsize=(7.1, 8.6))
ax.set_xlim(-2, W + 2)
ax.axis('off')
y = 0.0                        # cursor, grows downward as negative
boxes = []                     # (x, ytop, w, h)


def box(x, w, title, lines, edge, title_size=8.8, body_size=7.5):
    """Draw a box at the cursor; return its height."""
    h = PAD + TITLE_H + len(lines) * LINE_H + PAD
    ax.add_patch(FancyBboxPatch((x, y - h), w, h,
                                boxstyle='round,pad=0,rounding_size=1.4',
                                lw=1.15, edgecolor=edge, facecolor='white', zorder=2))
    ax.text(x + w / 2, y - PAD - TITLE_H * 0.72, title, ha='center', va='center',
            fontsize=title_size, fontweight='bold', color=edge, zorder=3)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y - PAD - TITLE_H - (i + 0.55) * LINE_H, ln,
                ha='center', va='center', fontsize=body_size, color=INK, zorder=3)
    boxes.append((x, y, w, h))
    return h


def stage(label, color):
    global y
    ax.text(0, y - LABEL_H * 0.55, label, ha='left', va='center', fontsize=7.6,
            fontweight='bold', color=color, zorder=3)
    ax.plot([0, W], [y - LABEL_H, y - LABEL_H], color=color, lw=0.9, alpha=0.45, zorder=1)
    y -= LABEL_H + 3.2


def arrow(x, y1, y2, color=GREY):
    ax.add_patch(FancyArrowPatch((x, y1), (x, y2), arrowstyle='-|>', lw=1.15,
                                 color=color, zorder=4, mutation_scale=10))


# ---------------------------------------------------------------- stage 1
stage('STAGE 1     FEATURE CONSTRUCTION', NAVY)
h = box(0, W, 'Daily OHLCV for ten instruments',
        ['Yahoo Finance, adjusted for splits and dividends, 2007–2024',
         'nine features per asset from causal rolling windows, then cross-sectionally',
         'standardised each session'], NAVY)
y -= h
arrow(25, y, y - GAP_BOX)
arrow(75, y, y - GAP_BOX)
y -= GAP_BOX
hl = box(0, 47, 'Technical composite',
         ['RSI, MACD, stochastic,', 'MFI, OBV flow'], TEAL)
y_save = y
hr = box(53, 47, 'Market-state composite',
         ['return, relative volume, realised', 'volatility, volume instability',
          'no text or language model is used'], TEAL)
y -= max(hl, hr)

# ---------------------------------------------------------------- stage 2
y -= GAP_STAGE + 2.5
stage('STAGE 2     REGIME IDENTIFICATION AND SIGNAL FUSION', RUST)
hl = box(0, 47, 'Gaussian mixture, $K=3$',
         ['expanding window, refit at every rebalance,', 'components sorted by fitted volatility',
          'calm / transitional / stress'], RUST)
y_l = y
hr = box(53, 47, 'Regime-conditional fusion',
         [r'$z = c\,(w^{T}_{r}T + w^{M}_{r}M)$',
          r'$\alpha = \mathrm{IC}\cdot\sigma\cdot z$',
          'places the linear term on a return scale'], RUST)
y -= max(hl, hr)
ax.add_patch(FancyArrowPatch((47, y_l - max(hl, hr) / 2), (53, y_l - max(hl, hr) / 2),
                             arrowstyle='-|>', lw=1.15, color=RUST, zorder=4,
                             mutation_scale=10))

# ---------------------------------------------------------------- stage 3
y -= GAP_STAGE
stage('STAGE 3     TWO-STEP PORTFOLIO CONSTRUCTION', NAVY)
h = box(0, W, 'Step 1     relative allocation, fully invested',
        [r'$\max_{w}\;\; \alpha^{\top}w \;-\; \lambda\,w^{\top}\Sigma w \;-\; \kappa\,\|w-w_{t-1}\|_{1}$',
         r'subject to    $w \geq 0$,    $w \leq 0.25$,    $\mathbf{1}^{\top}w = 1$',
         'Ledoit–Wolf shrinkage on 252 sessions; solved with CVXPY and Clarabel'], NAVY)
y -= h
arrow(50, y, y - GAP_BOX, NAVY)
y -= GAP_BOX
h = box(0, W, 'Step 2     regime risk budget',
        [r'$w^{\ast} = k_{r}\,\tilde{w}$,      $k = \{1.00,\; 0.75,\; 0.50\}$ as a fraction of the sleeve’s own risk',
         'residual held in cash; one-sided, so the portfolio is never levered',
         'relative, not absolute: an absolute target binds in 1 rebalance out of 192'], NAVY)
y -= h

# ---------------------------------------------------------------- stage 4
y -= GAP_STAGE
stage('STAGE 4     WALK-FORWARD EVALUATION', TEAL)
hl = box(0, 47, 'Walk-forward backtest',
         ['21-session rebalance, drift between', '10 bps one-way cost on turnover'], TEAL)
hr = box(53, 47, 'Benchmarks',
         ['equal weight, inverse volatility,', 'equal weight at strategy volatility'], TEAL)
y -= max(hl, hr)
arrow(25, y, y - GAP_BOX, TEAL)
arrow(75, y, y - GAP_BOX, TEAL)
y -= GAP_BOX
h = box(0, W, 'Component ablation at matched average exposure',
        ['regime layer removed; signal layer removed; risk-budget ladder;',
         '18-cell specification sweep; block-bootstrap inference'], RUST)
y -= h

ax.set_ylim(y - 2, 3)
fig.savefig(os.path.join(IMG, 'fig_architecture.png'))
print('wrote', os.path.join(IMG, 'fig_architecture.png'))
