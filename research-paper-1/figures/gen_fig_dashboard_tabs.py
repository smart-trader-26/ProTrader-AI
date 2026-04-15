"""
Generate: ProTrader AI — Eight-Tab Interactive Dashboard
Eight tab cards in a row, each with a coloured header and bullet-point body.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import textwrap

BG = '#ffffff'
TEXT = '#1a1a1a'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'text.color': TEXT,
})

fig, ax = plt.subplots(figsize=(42, 14), facecolor=BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 42)
ax.set_ylim(0, 14)
ax.axis('off')

# ── Title ──
ax.text(21, 12.4, 'ProTrader AI — Eight-Tab Interactive Dashboard',
        fontsize=39, fontweight='bold', color=TEXT, ha='center', va='center')

# ═══════════════════════════════════════════════════════════
# Tab data
# ═══════════════════════════════════════════════════════════
tabs = [
    ('Tab 1', 'Dashboard', '#2b5ea7',  '#dbe8f7', [
        'Current price metrics',
        'Candlestick/line chart',
        'AI forecast fan (P5-P95)',
        'Directional accuracy',
        'SHAP importance',
        'Verdict card\n(Bullish/Bearish)',
        'DeepSeek+Gemini\nanalysis',
    ]),
    ('Tab 2', 'Dynamic Fusion', '#1a6b6d', '#d6eded', [
        'Live Bayesian weights',
        '(Technical / Sentiment\n / Vol)',
        '15-day weight evolution',
        'Stacked-area chart',
    ]),
    ('Tab 3', 'Technicals & Risk', '#3d5a6e', '#d8e2e8', [
        'Fibonacci retracement',
        'ATR stop-loss',
        'Trade Setup Calculator',
        'Entry/Stop/Target/RR',
    ]),
    ('Tab 4', 'Fundamentals', '#4a3728', '#f5e8dc', [
        'Forward P/E',
        'PEG Ratio',
        'Price/Book',
        'Debt/Equity',
        'ROE',
        'Profit Margins',
    ]),
    ('Tab 5', 'FII/DII', '#d35400', '#fde8d8', [
        'NSE institutional data',
        'Daily net buy/sell bars',
        '20-day cumulative lines',
        'FII/DII trend flags',
    ]),
    ('Tab 6', 'Sentiment', '#1a5c2e', '#dff0e3', [
        'Combined score+label',
        'Event pie chart',
        'Article table',
        'Per-source breakdown',
        'Disagreement warning',
    ]),
    ('Tab 7', 'Backtest', '#2d6a2e', '#dff0e3', [
        'After-cost performance',
        'Benchmark comparison',
        'Equity curve',
        'Monte Carlo\n(1000 paths)',
        'Pipeline timings',
    ]),
    ('Tab 8', 'Patterns', '#5b2d78', '#e8daf0', [
        '12 pattern archetypes',
        'Confidence scores',
        'Target prices',
        'Volume confirmation',
        'Support/resistance',
        'Hurst regime',
    ]),
]

N = len(tabs)
TAB_W = 4.6          # wider cards to fit text
GAP = 0.35           # gap between tabs
HEADER_H = 2.0       # header box height
BODY_H = 7.5        # body box height
PAD = 0.15           # rounded corner pad

total_w = N * TAB_W + (N - 1) * GAP
x_start = (42 - total_w) / 2

for i, (line1, line2, hdr_color, body_bg, bullets) in enumerate(tabs):
    x = x_start + i * (TAB_W + GAP)
    y_hdr_top = 11.5
    y_body_top = y_hdr_top - HEADER_H - 0.15

    # ── Header box ──
    hdr = FancyBboxPatch((x, y_hdr_top - HEADER_H), TAB_W, HEADER_H,
                          boxstyle=f'round,pad={PAD}',
                          fc=hdr_color, ec=hdr_color, lw=2, zorder=3)
    ax.add_patch(hdr)

    ax.text(x + TAB_W/2, y_hdr_top - HEADER_H/2 + 0.25, line1,
            fontsize=24, color='white', ha='center', va='center',
            fontweight='bold', zorder=4)
    ax.text(x + TAB_W/2, y_hdr_top - HEADER_H/2 - 0.35, line2,
            fontsize=26, color='white', ha='center', va='center',
            fontweight='bold', zorder=4)

    # ── Body box ──
    body = FancyBboxPatch((x, y_body_top - BODY_H), TAB_W, BODY_H,
                           boxstyle=f'round,pad={PAD}',
                           fc=body_bg, ec='#bbb', lw=1.5, zorder=2)
    ax.add_patch(body)

    # ── Bullet points — dynamic spacing for multi-line items ──
    bullet_x = x + 0.40
    cur_y = y_body_top - 0.65
    LINE_H = 0.88          # spacing for a single-line bullet
    EXTRA_LINE = 0.55      # extra drop per additional \n line

    for j, bullet in enumerate(bullets):
        ax.text(bullet_x, cur_y, f'•  {bullet}',
                fontsize=20, color='#2a2a2a', ha='left', va='top',
                zorder=4, clip_on=True)
        n_lines = bullet.count('\n') + 1
        cur_y -= LINE_H + (n_lines - 1) * EXTRA_LINE


# ═══════════════════════════════════════════════════════════
#  Save
# ═══════════════════════════════════════════════════════════
out = r'c:\Users\divya\Desktop\finance\research-paper-1\figures\fig_dashboard_tabs.png'
plt.savefig(out, dpi=200, bbox_inches='tight',
            facecolor=fig.get_facecolor(), edgecolor='none')
plt.close(fig)
print(f'Saved → {out}')
