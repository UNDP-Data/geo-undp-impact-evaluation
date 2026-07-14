#!/usr/bin/env python3
"""
analysis/mmr/06_convergence_window.py
======================================
Formalizes the convergence finding: Myanmar's treatment effect is concentrated
in the active recovery window (Sep 2025 - Jan 2026) and disappears as control
townships catch up (Feb - May 2026). Produces the tracked table and figure
behind the paper's convergence subsection.

For each buffer spec (log_nightlights + controls):
  - full window (through 2026-05)
  - early window (through 2026-01)
  - late-post only (pre-period + Feb-May 2026)

Output: outputs/mmr/results/convergence_window.csv
        outputs/mmr/figures/convergence_window.png

Usage: .venv/bin/python analysis/mmr/06_convergence_window.py
"""
import importlib.util, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("d", HERE / "01_did_buffer_fixed.py")
d = importlib.util.module_from_spec(spec); sys.modules["d"] = d; spec.loader.exec_module(d)

RESULTS_DIR = Path("outputs/mmr/results"); RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = Path("outputs/mmr/figures"); FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SPECS = [(1000,3000,"1-3km"),(2000,5000,"2-5km"),(3000,8000,"3-8km"),
         (5000,15000,"5-15km"),(8000,20000,"8-20km")]
CTRL = ["rainfall","conflict_events"]

def main():
    panel = d.load_panel_data(); gdf, id_col = d.load_boundaries(); sites = d.load_undp_sites()
    dist = d.calculate_min_distance_to_sites(gdf, sites, id_col)
    date, _ = d.get_infrastructure_treatment_date()

    rows = []
    for inner, outer, label in SPECS:
        df = d.create_buffer_treatment(panel, dist, inner, outer, date)
        windows = {
            "full (Sep25-May26)": df[df.period_start <= "2026-05-31"],
            "early (Sep25-Jan26)": df[df.period_start <= "2026-01-31"],
            "late only (Feb-May26)": df[(df.period_start < "2025-09-01") | (df.period_start >= "2026-02-01")],
        }
        for wname, wdf in windows.items():
            r = d.run_twfe_regression(wdf, "log_nightlights", label, controls=CTRL)
            if r:
                rows.append({"spec": label, "window": wname,
                             "coef": float(r["Coeff"]), "se": float(r["SE"]),
                             "pval": float(r["P_value"]),
                             "pct_effect": 100*(np.exp(float(r["Coeff"]))-1),
                             "n_treated": r["N_Treated"], "n_control": r["N_Control"]})
                print(f"{label:8s} {wname:24s} b={r['Coeff']:+.4f} p={r['P_value']:.4f}")

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS_DIR / "convergence_window.csv", index=False)
    print(f"\n✓ Saved {RESULTS_DIR/'convergence_window.csv'}")

    # Figure: coefficient per window per spec
    fig, ax = plt.subplots(figsize=(10, 5.5))
    winkeys = list(dict.fromkeys(res["window"]))
    x = np.arange(len(SPECS)); w = 0.26
    colors = ["#1f4e79", "#2e86ab", "#b0b0b0"]
    for i, wk in enumerate(winkeys):
        sub = res[res.window == wk].set_index("spec").reindex([s[2] for s in SPECS])
        ax.bar(x + (i-1)*w, sub["coef"], w, yerr=1.96*sub["se"], capsize=3,
               label=wk, color=colors[i % 3])
    ax.axhline(0, color="grey", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([s[2] for s in SPECS])
    ax.set_ylabel("DiD coefficient (log nightlights, with controls)")
    ax.set_title("Myanmar: recovery acceleration and convergence by estimation window")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "convergence_window.png", dpi=150, facecolor="white")
    print(f"✓ Saved {FIGURES_DIR/'convergence_window.png'}")

if __name__ == "__main__":
    main()
