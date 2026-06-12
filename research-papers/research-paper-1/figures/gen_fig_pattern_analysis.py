"""
Generate: Chart Pattern Recognition System
Three panels: Multi-Timeframe Scanning | 12 Pattern Archetypes | Hybrid Consensus
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

BG = '#ffffff'
TEXT = '#1a1a1a'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'text.color': TEXT,
})

fig = plt.figure(figsize=(28, 14), facecolor=BG)
fig.suptitle('Chart Pattern Recognition System',
             fontsize=34, fontweight='bold', color=TEXT, y=0.97)

# ═══════════════════════════════════════════════════════════
#  PANEL 1 — Multi-Timeframe Scanning  (left)
# ═══════════════════════════════════════════════════════════
ax1 = fig.add_axes([0.02, 0.08, 0.28, 0.82])
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')

ax1.text(5, 9.6, 'Multi-Timeframe Scanning',
         fontsize=27, fontweight='bold', ha='center', color=TEXT)

# Three separate boxes with spacing
box_data = [
    (5, 8.1, 7.0, 2.0, '#2b7bba', '#cce5ff',
     'scipy.argrelextrema\norder=3\nOrder 3\n(Minor swings)'),
    (5, 5.3, 7.0, 2.0, '#e07020', '#fde0cc',
     'scipy.argrelextrema\norder=5\nOrder 5\n(Medium swings)'),
    (5, 2.5, 7.0, 2.0, '#b03080', '#f5e6f0',
     'scipy.argrelextrema\norder=7\nOrder 7\n(Major swings)'),
]

for cx, cy, w, h, ec, fc, txt in box_data:
    p = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                        boxstyle='round,pad=0.20',
                        fc=fc, ec=ec, lw=3.0, zorder=3)
    ax1.add_patch(p)
    ax1.text(cx, cy, txt, fontsize=27, color=TEXT,
             ha='center', va='center', fontweight='bold',
             zorder=4, linespacing=1.3)

ax1.text(5, 0.4, '↓  Merge & deduplicate peaks/troughs',
         fontsize=25, color='#555', ha='center', fontweight='bold')

# ═══════════════════════════════════════════════════════════
#  PANEL 2 — 12 Pattern Archetypes  (centre)
# ═══════════════════════════════════════════════════════════
ax2 = fig.add_axes([0.32, 0.08, 0.32, 0.82])
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 14)
ax2.axis('off')

ax2.text(5, 13.4, '12 Pattern Archetypes',
         fontsize=27, fontweight='bold', ha='center', color=TEXT)

patterns = [
    ('Double Top',           'Bearish',  '#e74c3c'),
    ('Double Bottom',        'Bullish',  '#27ae60'),
    ('Head & Shoulders',     'Bearish',  '#e74c3c'),
    ('Inverse H&S',          'Bullish',  '#27ae60'),
    ('Ascending Triangle',   'Bullish',  '#27ae60'),
    ('Descending Triangle',  'Bearish',  '#e74c3c'),
    ('Symmetric Triangle',   'Neutral',  '#d4a017'),
    ('Ascending Channel',    'Bullish',  '#27ae60'),
    ('Descending Channel',   'Bearish',  '#e74c3c'),
    ('Bull Flag',            'Bullish',  '#27ae60'),
    ('Bear Flag',            'Bearish',  '#e74c3c'),
    ('BB Squeeze',           'Neutral',  '#d4a017'),
]

y_start = 12.4
y_step = 0.92
for i, (name, label, color) in enumerate(patterns):
    y = y_start - i * y_step
    ax2.text(1.5, y, name, fontsize=27, color=TEXT,
             ha='left', va='center')
    ax2.text(7.5, y, label, fontsize=27, color=color,
             ha='left', va='center', fontweight='bold')
    # subtle separator line
    if i < len(patterns) - 1:
        ax2.plot([1.5, 9.0], [y - y_step/2, y - y_step/2],
                 color='#e0e0e0', lw=0.8, zorder=1)


# ═══════════════════════════════════════════════════════════
#  PANEL 3 — Hybrid Consensus  (right)
# ═══════════════════════════════════════════════════════════
ax3 = fig.add_axes([0.68, 0.08, 0.30, 0.82])
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.axis('off')

ax3.text(5, 9.6, 'Hybrid Consensus',
         fontsize=27, fontweight='bold', ha='center', color=TEXT)

# Outer container box (blue border)
outer = FancyBboxPatch((0.8, 1.2), 8.4, 7.8,
                        boxstyle='round,pad=0.3',
                        fc='#eef4fb', ec='#2b7bba', lw=3.0, zorder=1)
ax3.add_patch(outer)

# Helper for Hybrid Consensus boxes
HC_PAD = 0.20

def hc_box(ax, cx, cy, w, h, text, fc, ec, fs=20):
    p = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                        boxstyle=f'round,pad={HC_PAD}',
                        fc=fc, ec=ec, lw=2.5, zorder=3)
    ax.add_patch(p)
    ax.text(cx, cy, text, fontsize=fs, color=TEXT,
            ha='center', va='center', fontweight='bold',
            zorder=4, linespacing=1.3)
    off = HC_PAD + 0.08
    return {'top': (cx, cy + h/2 + off), 'bottom': (cx, cy - h/2 - off)}


b1 = hc_box(ax3, 5, 7.8, 5.5, 1.8,
             'Classical\nscipy Detection',
             '#d5c8e8', '#6a3d9a', fs=27)

b2 = hc_box(ax3, 5, 5.0, 5.5, 1.8,
             'Roboflow\nVision API\n(conf ≥ 0.30)',
             '#d5c8e8', '#6a3d9a', fs=27)

b3 = hc_box(ax3, 5, 2.2, 5.5, 1.8,
             'Hybrid\nConsensus\nLayer',
             '#d4edda', '#1a5c2e', fs=27)

# Arrows between boxes — start/end outside the visual box edge
for top_box, bot_box in [(b1, b2), (b2, b3)]:
    a = FancyArrowPatch(
        top_box['bottom'], bot_box['top'],
        arrowstyle='-|>', mutation_scale=28,
        color='#444', lw=2.8, zorder=5,
        shrinkA=0, shrinkB=0)
    ax3.add_patch(a)

ax3.text(5, 0.35, 'Precision: 64.1% → 71.3% → 78.4%',
         fontsize=25, color='#555', ha='center', fontweight='bold')


# ═══════════════════════════════════════════════════════════
#  Save
# ═══════════════════════════════════════════════════════════
out = r'c:\Users\divya\Desktop\finance\research-paper-1\figures\fig_pattern_analysis.png'
plt.savefig(out, dpi=200, bbox_inches='tight',
            facecolor=fig.get_facecolor(), edgecolor='none')
print(f'Saved → {out}')
plt.show()
