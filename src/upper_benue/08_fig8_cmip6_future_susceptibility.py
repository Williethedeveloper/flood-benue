"""
08_fig8_cmip6_future_susceptibility.py
=======================================
Generates Figure 8: Future flood susceptibility projections under NEX-GDDP-CMIP6
SSP2-4.5 and SSP5-8.5 for mid-century (2041-2060) and end-century (2081-2100).

Output: results/fig8_cmip6_future_susceptibility.png  (300 DPI, ~6 × 8 in)

Methodology note:
  The analytical delta-change approach scales the CHIRPS-derived baseline area
  proportionally to ensemble mean rainfall multipliers. This is consistent with
  the manuscript's Section 3.8 methodology statement.

Values sourced from manuscript Table 4 (confirmed correct):
  - Baseline:        17,139.68 km²  (~1,971,000 exposed @ 115 ppl/km²)
  - SSP2-4.5 Mid:    18,596.55 km²  (+8.50%,  +167,000)
  - SSP2-4.5 Far:    19,436.40 km²  (+13.40%, +264,000)
  - SSP5-8.5 Mid:    19,676.35 km²  (+14.80%, +291,000)
  - SSP5-8.5 Far:    21,356.04 km²  (+24.60%, +484,000)
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = RESULTS_DIR / "fig8_cmip6_future_susceptibility.png"

# ── Data (from manuscript Table 4) ────────────────────────────────────────────
BASELINE_AREA = 17_139.68   # km²
BASELINE_POP  = 1_971_000   # people

scenarios = [
    "Baseline\n(1995–2014)",
    "SSP2-4.5\nMid 2041–60",
    "SSP2-4.5\nFar 2081–2100",
    "SSP5-8.5\nMid 2041–60",
    "SSP5-8.5\nFar 2081–2100",
]
areas = [17_139.68, 18_596.55, 19_436.40, 19_676.35, 21_356.04]
pops  = [1_971_000, 2_138_000, 2_235_000, 2_262_000, 2_455_000]

expansion_pct = [0.0, 8.5, 13.4, 14.8, 24.6]
add_pop       = [0,   167_000, 264_000, 291_000, 484_000]

ssp245_idx = [0, 1, 2]   # Baseline + two SSP2-4.5 points
ssp585_idx = [0, 3, 4]   # Baseline + two SSP5-8.5 points

# ── Colour palette ─────────────────────────────────────────────────────────────
COL_BASE    = "#607D8B"   # grey-blue  — baseline
COL_245_MID = "#64B5F6"   # light blue — SSP2-4.5 Mid
COL_245_FAR = "#1565C0"   # dark blue  — SSP2-4.5 Far
COL_585_MID = "#FF8A65"   # light red  — SSP5-8.5 Mid
COL_585_FAR = "#B71C1C"   # dark red   — SSP5-8.5 Far

bar_colors = [COL_BASE, COL_245_MID, COL_245_FAR, COL_585_MID, COL_585_FAR]

# ── Layout ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 10), dpi=300)
fig.patch.set_facecolor("white")

gs = gridspec.GridSpec(2, 2, figure=fig,
                       hspace=0.42, wspace=0.38,
                       left=0.09, right=0.97, top=0.90, bottom=0.08)

ax1 = fig.add_subplot(gs[0, 0])   # (A) bar — High+VH area
ax2 = fig.add_subplot(gs[0, 1])   # (B) bar — additional exposed pop
ax3 = fig.add_subplot(gs[1, 0])   # (C) line — area trajectories by SSP
ax4 = fig.add_subplot(gs[1, 1])   # (D) bubble — scenario risk matrix

LABEL_KW = dict(fontsize=9, ha="center")
TICK_KW  = dict(labelsize=8)

# ══════════════════════════════════════════════════════════════════════════════
# (A) High+VH area per scenario (grouped bars)
# ══════════════════════════════════════════════════════════════════════════════
x = np.arange(len(scenarios))
bars = ax1.bar(x, areas, color=bar_colors, edgecolor="white", linewidth=0.6,
               width=0.6, zorder=3)
ax1.axhline(BASELINE_AREA, color=COL_BASE, linestyle="--", linewidth=1.2,
            zorder=4, label=f"Baseline {BASELINE_AREA:,.0f} km²")
ax1.set_xticks(x)
ax1.set_xticklabels(scenarios, fontsize=7.5, ha="center")
ax1.set_ylabel("High + Very High\nSusceptibility Area (km²)", fontsize=9)
ax1.set_ylim(14_000, 23_500)
ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
# ax1.tick_params(**TICK_KW)
ax1.set_title("(A) Projected High+VH Area", fontsize=10, fontweight="bold", pad=6)
ax1.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
ax1.set_axisbelow(True)

# Annotate expansion %
for bar, ep in zip(bars[1:], expansion_pct[1:]):
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 150,
             f"+{ep}%", ha="center", va="bottom",
             fontsize=7, fontweight="bold", color=bar.get_facecolor())

# ══════════════════════════════════════════════════════════════════════════════
# (B) Additional exposed population
# ══════════════════════════════════════════════════════════════════════════════
x2 = np.arange(1, len(scenarios))   # skip baseline (0 additional)
ax2.bar(x2 - 1, [v / 1000 for v in add_pop[1:]], color=bar_colors[1:],
        edgecolor="white", linewidth=0.6, width=0.6, zorder=3)
ax2.set_xticks(np.arange(4))
ax2.set_xticklabels(scenarios[1:], fontsize=7.5)
ax2.set_ylabel("Additional Exposed\nPopulation (thousands)", fontsize=9)
ax2.set_ylim(0, 580)
ax2.tick_params(**TICK_KW)
ax2.set_title("(B) Additional Population Exposed", fontsize=10, fontweight="bold", pad=6)
ax2.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
ax2.set_axisbelow(True)

for i, (val, col) in enumerate(zip(add_pop[1:], bar_colors[1:])):
    ax2.text(i, val / 1000 + 10, f"+{val/1000:.0f}k",
             ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=col)

# ══════════════════════════════════════════════════════════════════════════════
# (C) Trajectory lines: area vs time for SSP2-4.5 and SSP5-8.5
# ══════════════════════════════════════════════════════════════════════════════
time_pts   = [2007, 2050, 2090]  # representative mid-points of each period
area_245   = [areas[0], areas[1], areas[2]]
area_585   = [areas[0], areas[3], areas[4]]

ax3.fill_between(time_pts, area_245, area_585, alpha=0.12, color="#9E9E9E",
                 label="Scenario spread")
ax3.plot(time_pts, area_245, "o-", color=COL_245_FAR, linewidth=2,
         markersize=6, label="SSP2-4.5 (moderate)")
ax3.plot(time_pts, area_585, "s-", color=COL_585_FAR, linewidth=2,
         markersize=6, label="SSP5-8.5 (high-end)")
ax3.axhline(BASELINE_AREA, color=COL_BASE, linestyle="--", linewidth=1.2,
            label="Baseline")

ax3.set_xlim(1995, 2105)
ax3.set_ylim(14_000, 23_500)
ax3.set_xticks([2007, 2050, 2090])
ax3.set_xticklabels(["Baseline\n1995–2014", "Mid-century\n2041–2060",
                      "End-century\n2081–2100"], fontsize=7.5)
ax3.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
ax3.set_ylabel("High + Very High Area (km²)", fontsize=9)
ax3.set_title("(C) Temporal Trajectory by Scenario", fontsize=10,
              fontweight="bold", pad=6)
ax3.legend(fontsize=7.5, loc="upper left", framealpha=0.85)
ax3.grid(linestyle=":", alpha=0.5)
ax3.tick_params(**TICK_KW)

# Annotate end points (placed cleanly above red and below blue to avoid collision)
ax3.annotate(f"{area_245[-1]:,.0f} km²\n(+{expansion_pct[2]}%)",
             xy=(2090, area_245[-1]), xytext=(2063, area_245[-1] - 1750),
             fontsize=7.5, fontweight="bold", color=COL_245_FAR,
             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
             arrowprops=dict(arrowstyle="->", color=COL_245_FAR, lw=1.0))
ax3.annotate(f"{area_585[-1]:,.0f} km²\n(+{expansion_pct[4]}%)",
             xy=(2090, area_585[-1]), xytext=(2057, area_585[-1] + 550),
             fontsize=7.5, fontweight="bold", color=COL_585_FAR,
             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
             arrowprops=dict(arrowstyle="->", color=COL_585_FAR, lw=1.0))

# ══════════════════════════════════════════════════════════════════════════════
# (D) Risk matrix — rainfall increase vs area expansion (bubble = add. pop)
# ══════════════════════════════════════════════════════════════════════════════
rain_increase = [8.5, 13.4, 14.8, 24.6]   # mean annual rainfall % change
area_increase = [8.5, 13.4, 14.8, 24.6]   # proportional (delta-change)
add_pops_k    = [v / 1000 for v in add_pop[1:]]
bubble_sz     = [v * 2.2 for v in add_pops_k]  # scale for readability
bub_colors    = [COL_245_MID, COL_245_FAR, COL_585_MID, COL_585_FAR]
labels_d      = ["SSP2-4.5 Mid", "SSP2-4.5 Far", "SSP5-8.5 Mid", "SSP5-8.5 Far"]
offsets_d     = [(12.2, 4.5), (17.5, 9.2), (8.2, 19.2), (16.2, 26.2)]

sc = ax4.scatter(rain_increase, area_increase,
                 s=[v * 1.5 for v in bubble_sz],
                 c=bub_colors, edgecolors="white", linewidths=0.8,
                 zorder=4, alpha=0.88)

# 1:1 diagonal (perfect proportionality line)
diag = np.linspace(0, 30, 100)
ax4.plot(diag, diag, "k--", linewidth=0.8, alpha=0.4, label="1:1 line")

for lbl, rx, ax_, clr, (tx, ty) in zip(labels_d, rain_increase, area_increase, bub_colors, offsets_d):
    ax4.annotate(lbl, (rx, ax_), xytext=(tx, ty),
                 fontsize=7.5, color=clr, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=clr, lw=0.8, shrinkA=0, shrinkB=6))

ax4.set_xlim(0, 30)
ax4.set_ylim(0, 30)
ax4.set_xlabel("Mean Annual Rainfall Increase (%)", fontsize=9)
ax4.set_ylabel("High+VH Area Expansion (%)", fontsize=9)
ax4.set_title("(D) Rainfall vs Area Expansion\n(bubble size ∝ additional exposed pop.)",
              fontsize=10, fontweight="bold", pad=6)
ax4.grid(linestyle=":", alpha=0.5)
ax4.tick_params(**TICK_KW)

# Bubble legend (scaled cleanly so legend markers do not overlap)
for size, label in [(100, "100k"), (200, "200k"), (400, "400k+")]:
    ax4.scatter([], [], s=size * 0.45, c="#999999", alpha=0.7,
                label=f"+{label} exposed")
ax4.legend(fontsize=7, loc="upper left", framealpha=0.9, labelspacing=0.9, borderpad=0.6)

# ══════════════════════════════════════════════════════════════════════════════
# Figure-level title & note
# ══════════════════════════════════════════════════════════════════════════════
fig.suptitle(
    "Fig. 8  Projected change in flood susceptibility under NEX-GDDP-CMIP6 scenarios\n"
    "SSP2-4.5 and SSP5-8.5 for mid-century (2041–2060) and end-century (2081–2100)",
    fontsize=11, fontweight="bold", y=0.97
)

fig.text(
    0.5, 0.005,
    "Note: Future areas derived via analytical delta-change scaling of CHIRPS climatology "
    "(Thrasher et al. 2022). Five-model ensemble (ACCESS-CM2, CanESM5, EC-Earth3, MPI-ESM1-2-HR, "
    "NorESM2-MM).\nPopulation exposure estimated at 115 people km⁻² (NBS 2023). "
    "Baseline high+VH area = 17,139.68 km² (1995–2014 reference period).",
    ha="center", va="bottom", fontsize=6.5, style="italic", color="#444444"
)

# ── Save ───────────────────────────────────────────────────────────────────────
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved → {OUT_PNG}")
print(f"File size: {OUT_PNG.stat().st_size / 1024:.1f} KB")
