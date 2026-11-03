#!/usr/bin/env python3
"""Extended Data Fig. 2 (regenerated): Myanmar luminosity effect by donut band,
three estimators, from the reanalysis results files part3_myanmar_<band>_results.csv."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
BANDS = [("0.5_1.5km", "0.5-1.5"), ("1_3km", "1-3"), ("2_5km", "2-5"),
         ("3_8km", "3-8"), ("5_15km", "5-15"), ("8_20km", "8-20")]
EST = [("PPML raw y", "PPML (raw radiance)", "o"),
       ("OLS log(y+1)", "OLS log(VIIRS+1)", "s"),
       ("OLS y/pre-mean", "OLS levels / pre-event mean", "^")]
rows = []
for lab, name in BANDS:
    try:
        r = pd.read_csv(f"part3_myanmar_{lab}_results.csv")
    except FileNotFoundError:
        print(f"missing part3_myanmar_{lab}_results.csv"); continue
    for key, _, _ in EST:
        m = r[r.estimator.str.startswith(key)]
        if len(m):
            rows.append((name, key, *m.iloc[0][["implied_pct", "ci_lo_pct", "ci_hi_pct"]]))
D = pd.DataFrame(rows, columns=["band", "est", "pct", "lo", "hi"])
fig, ax = plt.subplots(figsize=(8, 4.2))
order = [n for _, n in BANDS if n in set(D.band)]
x = np.arange(len(order))
for k, (key, label, mk) in enumerate(EST):
    d = D[D.est == key].set_index("band").reindex(order)
    off = (k - 1) * 0.22
    ax.errorbar(x + off, d.pct, yerr=[d.pct - d.lo, d.hi - d.pct], fmt=mk, ms=5,
                capsize=3, lw=1, label=label)
ax.axhline(0, color="grey", lw=0.8)
ax.axvspan(order.index("3-8") - 0.45, order.index("3-8") + 0.45, color="#eeeeee", zorder=0)
ax.set_xticks(x); ax.set_xticklabels([f"{b} km" for b in order])
ax.set_xlabel("Donut band (inner-outer radius)"); ax.set_ylabel("Effect on luminosity (%), 95% CI")
ax.legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig("EDfig2_myanmar_bands.png", dpi=200)
print("Saved EDfig2_myanmar_bands.png (shaded: baseline 3-8 km)")
