"""
Generate: Bayesian Dynamic Fusion Framework
Three expert rows (Technical, Sentiment, Volatility) feeding into
Bayesian Weights and final fusion output.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ═══════════════════════════════════════════════════════════
# Global style
# ═══════════════════════════════════════════════════════════
BG = '#ffffff'
TEXT = '#1a1a1a'

plt.rcParams.update({
    'font.family':      'sans-serif',
    'font.sans-serif':  ['Arial', 'Helvetica', 'DejaVu Sans'],
    'text.color':       TEXT,
})

fig, ax = plt.subplots(figsize=(24, 11), facecolor=BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 24)
ax.set_ylim(1.0, 12.5)
ax.axis('off')

# ── Title ──
ax.text(12, 12.0, 'Bayesian Dynamic Fusion Framework',
        fontsize=32, fontweight='bold', color=TEXT, ha='center', va='center')

# ═══════════════════════════════════════════════════════════
# Colours
# ═══════════════════════════════════════════════════════════
C_INPUT_TECH = '#2b5ea7'     # blue
C_INPUT_SENT = '#3a7d44'     # green
C_INPUT_VOL  = '#7b2d5f'     # purple
C_EXPERT     = '#1a6b6d'     # teal
C_MSE        = '#b94a2c'     # burnt orange
C_BAYES      = '#d97b1a'     # orange
C_FUSION     = '#1a5c2e'     # dark green

# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════
PAD = 0.22   # must match boxstyle pad
EDGE_OFF = PAD + 0.06   # offset edges outward past the visual rounded corner

def draw_box(cx, cy, w, h, text, fc, fs=18, fw='normal', tc='white', ec='#666'):
    p = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                        boxstyle=f'round,pad={PAD}',
                        fc=fc, ec=ec, lw=2.0, zorder=3)
    ax.add_patch(p)
    ax.text(cx, cy, text, fontsize=fs, color=tc,
            ha='center', va='center', fontweight=fw,
            zorder=4, linespacing=1.35)
    # Edge coords offset outward past the visual padding
    return {
        'left':   (cx - w/2 - EDGE_OFF, cy),
        'right':  (cx + w/2 + EDGE_OFF, cy),
        'top':    (cx, cy + h/2 + EDGE_OFF),
        'bottom': (cx, cy - h/2 - EDGE_OFF),
    }


def arrow(start, end, color='#555', lw=2.5):
    a = FancyArrowPatch(start, end,
                         arrowstyle='-|>',
                         mutation_scale=28,
                         color=color, lw=lw, zorder=5,
                         connectionstyle='arc3,rad=0',
                         shrinkA=0, shrinkB=0)
    ax.add_patch(a)


# ═══════════════════════════════════════════════════════════
# Layout — three rows
# ═══════════════════════════════════════════════════════════
# Columns:  Input(1.8)  Expert(5.5)  MSE(10.2)  Bayes(15.5)  Fusion(21)
# Rows:     top=10.0    mid=6.5     bottom=3.0

Y_TOP = 10.0
Y_MID = 6.5
Y_BOT = 3.0

# Box dimensions
INPUT_W, INPUT_H = 2.4, 1.6
EXPERT_W, EXPERT_H = 4.8, 1.8
MSE_W, MSE_H = 3.4, 1.6
BAYES_W, BAYES_H = 4.0, 2.8
FUSION_W, FUSION_H = 2.8, 1.6

# Column x-centres
X_INPUT  = 1.8
X_EXPERT = 6.2
X_MSE    = 11.0
X_BAYES  = 16.0
X_FUSION = 21.0

# ═══════════════════════════════════════════════════════════
# Row 1 — Technical
# ═══════════════════════════════════════════════════════════
inp1 = draw_box(X_INPUT, Y_TOP, INPUT_W, INPUT_H,
                '30-Day\nTechnical\nFeatures',
                C_INPUT_TECH, fs=19, fw='bold')

exp1 = draw_box(X_EXPERT, Y_TOP, EXPERT_W, EXPERT_H,
                'Technical Expert\nGRU(128) → GRU(64)\n→ GRU(32) → Dense(1)',
                C_EXPERT, fs=18, fw='bold')

mse1 = draw_box(X_MSE, Y_TOP, MSE_W, MSE_H,
                'σ²_tech\nRolling MSE\n(N=15 days)',
                C_MSE, fs=18, fw='bold')

# ═══════════════════════════════════════════════════════════
# Row 2 — Sentiment
# ═══════════════════════════════════════════════════════════
inp2 = draw_box(X_INPUT, Y_MID, INPUT_W, INPUT_H,
                '8 Sentiment\nFeatures',
                C_INPUT_SENT, fs=19, fw='bold')

exp2 = draw_box(X_EXPERT, Y_MID, EXPERT_W, EXPERT_H,
                'Sentiment Expert\nDense(64) → Dense(32)\n→ Dense(16) → Dense(1)',
                C_EXPERT, fs=18, fw='bold')

mse2 = draw_box(X_MSE, Y_MID, MSE_W, MSE_H,
                'σ²_sent\nRolling MSE\n(N=15 days)',
                C_MSE, fs=18, fw='bold')

# ═══════════════════════════════════════════════════════════
# Row 3 — Volatility
# ═══════════════════════════════════════════════════════════
inp3 = draw_box(X_INPUT, Y_BOT, INPUT_W, INPUT_H,
                '6 Volatility\nFeatures\n(VIX)',
                C_INPUT_VOL, fs=19, fw='bold')

exp3 = draw_box(X_EXPERT, Y_BOT, EXPERT_W, EXPERT_H,
                'Volatility Expert\nMLP(32) → MLP(16)\n→ MLP(8) → Dense(1)',
                C_EXPERT, fs=18, fw='bold')

mse3 = draw_box(X_MSE, Y_BOT, MSE_W, MSE_H,
                'σ²_vol\nRolling MSE\n(N=15 days)',
                C_MSE, fs=18, fw='bold')

# ═══════════════════════════════════════════════════════════
# Bayesian Weights box (tall, centred at Y_MID)
# ═══════════════════════════════════════════════════════════
bayes = draw_box(X_BAYES, Y_MID, BAYES_W, BAYES_H,
                 'Bayesian Weights\nwᵢ = exp(−σᵢ²)\n/ Σ exp(−σⱼ²)\n\nΣwᵢ = 1.0',
                 C_BAYES, fs=18, fw='bold', tc='white')

# ═══════════════════════════════════════════════════════════
# Fusion output box
# ═══════════════════════════════════════════════════════════
fusion = draw_box(X_FUSION, Y_MID, FUSION_W, FUSION_H,
                  'ŷ_fusion\n= Σwᵢ·ŷᵢ',
                  C_FUSION, fs=20, fw='bold')

# ═══════════════════════════════════════════════════════════
# Arrows — edge to edge
# ═══════════════════════════════════════════════════════════
ARROW_COLOR = '#444444'

# Input → Expert (horizontal, same row)
arrow(inp1['right'], exp1['left'], ARROW_COLOR)
arrow(inp2['right'], exp2['left'], ARROW_COLOR)
arrow(inp3['right'], exp3['left'], ARROW_COLOR)

# Expert → MSE (horizontal, same row)
arrow(exp1['right'], mse1['left'], ARROW_COLOR)
arrow(exp2['right'], mse2['left'], ARROW_COLOR)
arrow(exp3['right'], mse3['left'], ARROW_COLOR)

# MSE → Bayesian Weights (angled arrows from each row to the centre box)
# Top MSE → Bayes top-left area
arrow(mse1['right'], (X_BAYES - BAYES_W/2, Y_MID + BAYES_H/2 - 0.3), ARROW_COLOR)
# Mid MSE → Bayes left centre
arrow(mse2['right'], bayes['left'], ARROW_COLOR)
# Bot MSE → Bayes bottom-left area
arrow(mse3['right'], (X_BAYES - BAYES_W/2, Y_MID - BAYES_H/2 + 0.3), ARROW_COLOR)

# Bayesian Weights → Fusion
arrow(bayes['right'], fusion['left'], ARROW_COLOR)


# ═══════════════════════════════════════════════════════════
#  Save
# ═══════════════════════════════════════════════════════════
out = r'c:\Users\divya\Desktop\finance\research-paper-1\figures\fig_dynamic_fusion.png'
plt.savefig(out, dpi=200, bbox_inches='tight',
            facecolor=fig.get_facecolor(), edgecolor='none')
print(f'Saved → {out}')
plt.show()
