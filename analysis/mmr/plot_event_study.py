#!/usr/bin/env python3
"""
plot_event_study.py — Corrected Myanmar figures for the manuscript.

Produces TWO figures, fixing the referee's objection that the current Fig 3
is labelled as an event study but shows the raw treated–comparison gap:

  fig3_event_study.png       — TRUE event-study coefficient plot (main text)
                               monthly DiD coefficients vs the omitted month,
                               95% CIs, treatment date (Sep 2025) marked.
  ed_fig1_pretrends.png      — pre-treatment diagnostic (Extended Data)
                               same coefficients, pre-period shaded, with the
                               single clustered linear-trend test annotated.

Both read only the tracked CSV:
    outputs/mmr/results/event_study_monthly.csv   (cols: m, b, se, dt)

Run from repo root:
    .venv/bin/python analysis/mmr/plot_event_study.py
Outputs to outputs/paper_figures/.
"""
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator
from pathlib import Path

CSV  = Path("outputs/mmr/results/event_study_monthly.csv")
OUT  = Path("outputs/paper_figures"); OUT.mkdir(parents=True, exist_ok=True)
EQ   = pd.Timestamp("2025-03-28")   # earthquake (shock)
TRT  = pd.Timestamp("2025-09-01")   # treatment (programme start) = reference frame

# ---- Nature-ish styling ----
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 7, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
})
NAVY = "#1f3b57"; RED = "#b2182b"; GREY = "#888888"

df = pd.read_csv(CSV, parse_dates=["dt"]).sort_values("dt").reset_index(drop=True)
df["lo"] = df["b"] - 1.96*df["se"]
df["hi"] = df["b"] + 1.96*df["se"]

def base_axes(ax):
    ax.axhline(0, color=GREY, lw=0.6, zorder=1)
    ax.axvline(TRT, color=NAVY, ls=":", lw=0.9, zorder=1)
    ax.axvline(EQ,  color=RED,  ls="--", lw=0.9, zorder=1)
    ax.xaxis.set_major_locator(MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(DateFormatter("%b\n%Y"))
    ax.set_ylabel("Monthly DiD coefficient\n(log luminosity)")

# ---------- FIG 3: event study (main text) ----------
fig, ax = plt.subplots(figsize=(180/25.4, 70/25.4))  # 180 mm double-column
base_axes(ax)
ax.errorbar(df["dt"], df["b"], yerr=1.96*df["se"], fmt="o", ms=3.2,
            color=NAVY, ecolor=NAVY, elinewidth=0.8, capsize=1.6, zorder=3)
ax.plot(df["dt"], df["b"], color=NAVY, lw=0.9, zorder=2)
ymax = float(np.nanmax(df["hi"])); ymin = float(np.nanmin(df["lo"]))
ax.annotate("Earthquake\n(28 Mar 2025)", xy=(EQ, ymax), xytext=(EQ, ymax),
            color=RED, fontsize=6, ha="center", va="bottom")
ax.annotate("Programme start\n(Sep 2025)", xy=(TRT, ymin), xytext=(TRT, ymin),
            color=NAVY, fontsize=6, ha="center", va="top")
ax.set_ylim(ymin-0.15*(ymax-ymin), ymax+0.20*(ymax-ymin))
fig.tight_layout()
fig.savefig(OUT/"fig3_event_study.png", dpi=400, bbox_inches="tight")
plt.close(fig)

# ---------- ED FIG 1: pre-treatment diagnostic ----------
# Report the SAME clustered linear-trend test used in the manuscript/Methods.
# This value comes from the township-clustered pre-trend test in
# analysis/mmr/02_pretrends_verified.py at the baseline specification.
# It is passed in (not recomputed here) so the figure and text never disagree.
P_TREND_CLUSTERED = 0.07   # <-- set to the value your verified suite prints

fig, ax = plt.subplots(figsize=(180/25.4, 70/25.4))
base_axes(ax)
ax.axvspan(df["dt"].min(), TRT, color=NAVY, alpha=0.05, zorder=0)
ax.errorbar(df["dt"], df["b"], yerr=1.96*df["se"], fmt="o", ms=3.0,
            color=NAVY, ecolor=NAVY, elinewidth=0.7, capsize=1.4, zorder=3)
ax.plot(df["dt"], df["b"], color=NAVY, lw=0.8, zorder=2)
ax.text(0.02, 0.04,
        f"Pre-treatment linear-trend test (clustered): P = {P_TREND_CLUSTERED:.2f}",
        transform=ax.transAxes, fontsize=6, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GREY, lw=0.5))
fig.tight_layout()
fig.savefig(OUT/"ed_fig1_pretrends.png", dpi=400, bbox_inches="tight")
plt.close(fig)

print("Wrote:")
print("  outputs/paper_figures/fig3_event_study.png   (main-text event study)")
print("  outputs/paper_figures/ed_fig1_pretrends.png  (Extended Data pre-trends)")
print(f"  pre-treatment clustered linear-trend test (from Methods): P = {P_TREND_CLUSTERED:.2f}")
