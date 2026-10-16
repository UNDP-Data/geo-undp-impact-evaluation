#!/usr/bin/env python3
"""Extended Data Fig. 3: Gaza estimates by donut band. Run after gaza_buffer_sweep.py."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = pd.read_csv("gaza_buffer_sweep.csv")
R["inner"] = R["band"].str.split("-").str[0].astype(int)
R = R.sort_values("inner")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, oc, title, log in [(axes[0], "viirs", "Nighttime lights, log(VIIRS+1)", True),
                           (axes[1], "sar_vh", "Radar backscatter, SAR VH (dB)", False)]:
    x = np.arange(len(R))
    for key, lab, mk in [("s1", "Original specification", "s"), ("s3", "Site FE", "^"),
                         ("s4", "Site FE + brightness x month", "o"), ("s5", "Site FE, conflict-period baseline", "D")]:
        b = R[f"{oc}_{key}_b"].values
        y = 100 * (np.exp(b) - 1) if log else b
        ax.plot(x, y, marker=mk, ms=5, lw=1.2, label=lab)
    ok = (R[[f"{oc}_pre_joint_p", f"{oc}_pre_slope_p"]].min(axis=1) > 0.10).values
    for xi, good in zip(x, ok):
        if good: ax.axvspan(xi - 0.4, xi + 0.4, color="#d9ead3", alpha=0.6, zorder=0)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(R["band"], rotation=45, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10); ax.set_xlabel("Donut band, inner-outer radius (m)", fontsize=9)
    ax.set_ylabel("Treated x post effect (%)" if log else "Treated x post effect (dB)", fontsize=9)
axes[0].legend(fontsize=8, loc="upper left")
fig.text(0.5, -0.02, "Shaded bands: parallel pre-trends not rejected (joint and slope tests, P > 0.10, conflict-period baseline).",
         ha="center", fontsize=8)
fig.tight_layout(); fig.savefig("ED_fig3_gaza_band_sweep.png", dpi=200, bbox_inches="tight")
print("Saved ED_fig3_gaza_band_sweep.png")
