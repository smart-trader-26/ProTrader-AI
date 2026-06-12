"""
Generate: Hurst Exponent and Market Regime Detection
  Left  — R/S Analysis line chart
  Right — Regime Classification Logic flowchart
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

np.random.seed(42)

# ═══════════════════════════════════════════════════════════
# Global style  —  WHITE background
# ═══════════════════════════════════════════════════════════
BG = '#ffffff'
TEXT_COLOR = '#1a1a1a'

plt.rcParams.update({
    'font.family':      'sans-serif',
    'font.sans-serif':  ['Arial', 'Helvetica', 'DejaVu Sans'],
    'text.color':       TEXT_COLOR,
    'axes.labelcolor':  TEXT_COLOR,
    'xtick.color':      TEXT_COLOR,
    'ytick.color':      TEXT_COLOR,
})

fig = plt.figure(figsize=(22, 11), facecolor=BG)
fig.suptitle('Hurst Exponent and Market Regime Detection',
             fontsize=32, fontweight='bold', color=TEXT_COLOR, y=0.98)

# ═══════════════════════════════════════════════════════════
#  LEFT — R/S Analysis Line Chart
# ═══════════════════════════════════════════════════════════
ax1 = fig.add_axes([0.06, 0.10, 0.46, 0.76])
ax1.set_facecolor('#f8f9fa')

# ── Border around the chart area ──
for spine in ax1.spines.values():
    spine.set_color('#333')
    spine.set_linewidth(1.5)
    spine.set_visible(True)

lags = np.linspace(0.6, 3.15, 20)

# Simulated log(R/S) = H * log(lag) + noise
noise_t = np.random.normal(0, 0.018, len(lags))
noise_r = np.random.normal(0, 0.018, len(lags))
noise_m = np.random.normal(0, 0.014, len(lags))

rs_trend = 0.65 * lags + noise_t
rs_rw    = 0.50 * lags + noise_r
rs_mr    = 0.35 * lags + noise_m

# Anchor starting values to match the original figure
rs_trend = rs_trend - rs_trend[0] + 0.47
rs_rw    = rs_rw    - rs_rw[0]    + 0.33
rs_mr    = rs_mr    - rs_mr[0]    + 0.22

ax1.plot(lags, rs_trend, 'o-', color='#27ae60', lw=2.8, ms=7,
         label='Trending (H=0.65)', zorder=3)
ax1.plot(lags, rs_rw,    'o-', color='#e67e22', lw=2.8, ms=7,
         label='Random Walk (H=0.5)', zorder=3)
ax1.plot(lags, rs_mr,    'o-', color='#e74c3c', lw=2.8, ms=7,
         label='Mean-Rev. (H=0.35)', zorder=3)

# Vertical dashed reference lines
ax1.axvline(2.2, color='#999', ls='--', lw=1.2, alpha=0.6)
ax1.axvline(2.4, color='#999', ls='--', lw=1.2, alpha=0.6)

ax1.set_xlabel('log(lag)', fontsize=20, color=TEXT_COLOR)
ax1.set_ylabel('log(R/S)', fontsize=20, color=TEXT_COLOR)
ax1.set_title('R/S Analysis — Hurst Estimation\n(slope = H)',
              fontsize=22, pad=14, color=TEXT_COLOR)

ax1.legend(loc='upper left', fontsize=16,
           facecolor='white', edgecolor='#ccc',
           framealpha=0.95, labelcolor=TEXT_COLOR)

ax1.tick_params(labelsize=15)
ax1.set_xlim(0.5, 3.25)
ax1.set_ylim(0.10, 2.10)
ax1.grid(True, alpha=0.2, color='#bbb')


# ═══════════════════════════════════════════════════════════
#  RIGHT — Regime Classification Flowchart
# ═══════════════════════════════════════════════════════════
ax2 = fig.add_axes([0.54, 0.03, 0.45, 0.90])
ax2.set_facecolor(BG)
ax2.set_xlim(0, 12)
ax2.set_ylim(0, 13)
ax2.axis('off')

ax2.text(6, 12.5, 'Regime Classification Logic',
         fontsize=26, fontweight='bold', color=TEXT_COLOR, ha='center')

# ── Palette ──
C_START    = '#2c3e50'
C_DECISION = '#d4a017'
C_RED      = '#c0392b'
C_GREEN    = '#1e8449'
C_BLUE     = '#2471a3'

# ── Helpers ──
def draw_box(ax, cx, cy, w, h, text, fc, fs=18, fw='normal', tc='white', ec='#777'):
    p = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                        boxstyle='round,pad=0.20',
                        fc=fc, ec=ec, lw=2.0, zorder=3)
    ax.add_patch(p)
    ax.text(cx, cy, text, fontsize=fs, color=tc,
            ha='center', va='center', fontweight=fw,
            zorder=4, linespacing=1.3)


def draw_arrow(ax, x1, y1, x2, y2, color='#555555', lw=2.2):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                         arrowstyle='-|>',
                         mutation_scale=24,
                         color=color, lw=lw, zorder=2,
                         connectionstyle='arc3,rad=0',
                         shrinkA=3, shrinkB=3)
    ax.add_patch(a)


# ── Layout constants ──
XD   = 3.5          # x-centre of decision column
XR   = 10.0         # x-centre of result column  (pushed further right)
BW_D = 5.0          # decision-box width
BW_R = 3.6          # result-box  width
BH   = 1.10         # box height
GAP  = 2.20         # vertical gap between box centres (more spacing)

TOP  = 10.6
ys = [TOP,                        # start
      TOP - GAP,                  # decision 1
      TOP - 2*GAP,               # decision 2
      TOP - 3*GAP,               # decision 3
      TOP - 4*GAP]               # otherwise

# ── Draw nodes ──
draw_box(ax2, XD, ys[0], BW_D, BH,
         'Recent 120 closes\n→ compute H (R/S)',
         C_START, fs=20, fw='bold')

draw_box(ax2, XD, ys[1], BW_D, BH,
         'Vol > 85th pct?',
         C_DECISION, fs=20, tc='#1a1a1a')
draw_box(ax2, XR, ys[1], BW_R, BH,
         'HIGH\nVOLATILITY',
         C_RED, fs=21, fw='bold')

draw_box(ax2, XD, ys[2], BW_D, BH,
         '|slope|>0.002\nR²>0.3 and H≥0.45?',
         C_DECISION, fs=19, tc='#1a1a1a')
draw_box(ax2, XR, ys[2], BW_R, BH,
         'TRENDING',
         C_GREEN, fs=21, fw='bold')

draw_box(ax2, XD, ys[3], BW_D, BH,
         'H<0.45 or\nVol<30th+flat?',
         C_DECISION, fs=20, tc='#1a1a1a')
draw_box(ax2, XR, ys[3], BW_R, BH,
         'MEAN\nREVERTING',
         C_BLUE, fs=21, fw='bold')

draw_box(ax2, XD, ys[4], BW_D, BH,
         'Otherwise',
         C_START, fs=20)
draw_box(ax2, XR, ys[4], BW_R, BH,
         'NORMAL',
         C_BLUE, fs=21, fw='bold')

# ── Vertical arrow: Start → first decision (no label) ──
draw_arrow(ax2,
           XD, ys[0] - BH/2 - 0.08,
           XD, ys[1] + BH/2 + 0.08,
           color='#555555', lw=2.2)

# ── Vertical arrows (down) with "No" labels — only between decisions ──
for i in range(1, 4):
    y_top    = ys[i]   - BH/2 - 0.08
    y_bottom = ys[i+1] + BH/2 + 0.08
    draw_arrow(ax2, XD, y_top, XD, y_bottom, color='#555555', lw=2.2)
    # "No" label centred between the two boxes
    y_mid = (y_top + y_bottom) / 2
    ax2.text(XD + 0.45, y_mid, 'No',
             fontsize=16, color='#555555',
             ha='left', va='center', fontstyle='italic', fontweight='bold')

# ── Horizontal arrows (right — "Yes") + labels ──
for i in range(1, 5):
    x_start = XD + BW_D/2 + 0.12
    x_end   = XR - BW_R/2 - 0.12
    y       = ys[i]
    draw_arrow(ax2, x_start, y, x_end, y, color='#c0392b', lw=2.4)
    # "Yes" label above the midpoint of the arrow
    ax2.text((x_start + x_end) / 2, y + 0.35, 'Yes',
             fontsize=16, color='#c0392b',
             ha='center', va='center', fontweight='bold')


# ═══════════════════════════════════════════════════════════
#  Save
# ═══════════════════════════════════════════════════════════
out = r'c:\Users\divya\Desktop\finance\research-paper-1\figures\fig_hurst_regimes.png'
plt.savefig(out, dpi=200, bbox_inches='tight',
            facecolor=fig.get_facecolor(), edgecolor='none')
print(f'Saved → {out}')
plt.show()
