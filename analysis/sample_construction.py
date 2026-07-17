#!/usr/bin/env python3
"""
sample_construction.py  —  Freeze-and-report the analysis samples for both pilots.

Addresses referee points #3 (Myanmar sample counts must match everywhere) and
#4 (Gaza 454 vs 367 must be an explicit exclusion chain).

Emits, for each pilot, a sample-construction funnel:
   digitised/available geocodes -> eligible units -> treated -> comparison
   -> final unit-month observations
and writes a single frozen JSON that every table/figure should cite.

Run from repo root with the project venv:
    .venv/bin/python analysis/sample_construction.py
Outputs:
    outputs/sample_construction.json
    prints a human-readable funnel to stdout
"""
import json, csv
from pathlib import Path
from collections import defaultdict

OUT = Path("outputs/sample_construction.json")
report = {}

# ---------------- GAZA ----------------
# The estimation panel is the tracked CS baseline panel (site-month long form).
gaza_panel = Path("outputs/pse/site_panel_baseline.csv")
gaza = {}
if gaza_panel.exists():
    groups = defaultdict(set); has_start = set(); by_type = defaultdict(set)
    months = set(); obs = 0
    for row in csv.DictReader(open(gaza_panel)):
        sid = row["site_id"]; grp = row["group"]
        groups[grp].add(sid); months.add(row["date"]); obs += 1
        if row.get("project_start"): has_start.add(sid)
        if grp == "Treated": by_type[row.get("intervention_type", "?")].add(sid)
    gaza = {
        "digitised_sites_total": 454,            # from beneficiary_estimates.json (population base)
        "treated_in_estimation_panel": len(groups.get("Treated", set())),
        "comparison_points_donut": len(groups.get("Control", set())),
        "total_units": sum(len(s) for s in groups.values()),
        "n_months": len(months),
        "unit_month_observations": obs,
        "exclusion_reason": ("Sites retained if they have a usable project start date and "
                             ">=80% satellite coverage across the study window; matched 1:1 "
                             "with donut-ring comparison points (500 m-1.5 km)."),
        "treated_by_type": {t: len(s) for t, s in sorted(by_type.items(), key=lambda x:-len(x[1]))},
    }
report["gaza"] = gaza

# ---------------- MYANMAR ----------------
# Freeze from the DiD results table (already the frozen estimation output).
mmr_res = Path("outputs/mmr/results/did_buffer_results.csv")
mmr = {"buffers": []}
if mmr_res.exists():
    for r in csv.DictReader(open(mmr_res)):
        if r["Outcome"] == "log_nightlights" and "controls" in r["Buffer_Spec"]:
            mmr["buffers"].append({
                "spec": r["Buffer_Spec"].replace(" + controls", ""),
                "n_treated": int(r["N_Treated"]),
                "n_control": int(r["N_Control"]),
                "n_obs": int(r["N_Obs"]),
                "pct_effect": r["Pct_Effect"],
                "p": round(float(r["P_value"]), 3),
            })
    mmr["note"] = ("Treated = townships with >=1 UNDP early-recovery project within the inner "
                   "radius; comparison = townships in the outer ring. Counts are the frozen "
                   "estimation samples; the same numbers must appear in every table and figure.")
report["myanmar"] = mmr

OUT.write_text(json.dumps(report, indent=2))

# ---------------- human-readable ----------------
print("="*64)
print(" SAMPLE CONSTRUCTION — FROZEN (cite these everywhere)")
print("="*64)
print("\nGAZA")
if gaza:
    print(f"  Digitised UNDP sites (population base) .......... {gaza['digitised_sites_total']}")
    print(f"  -> Treated sites in estimation panel ........... {gaza['treated_in_estimation_panel']}")
    print(f"     (usable project date + >=80% coverage)")
    print(f"  -> Donut comparison points (500 m-1.5 km) ...... {gaza['comparison_points_donut']}")
    print(f"  = Total units .................................. {gaza['total_units']}")
    print(f"  x {gaza['n_months']} months = {gaza['unit_month_observations']:,} unit-month obs")
    print(f"  Treated by type: " + ", ".join(f"{t} {n}" for t,n in gaza['treated_by_type'].items()))
print("\nMYANMAR (per buffer specification)")
print(f"  {'spec':14} {'treated':>7} {'control':>7} {'obs':>6} {'effect':>8} {'p':>6}")
for b in mmr.get("buffers", []):
    print(f"  {b['spec']:14} {b['n_treated']:>7} {b['n_control']:>7} {b['n_obs']:>6} {b['pct_effect']:>8} {b['p']:>6}")
print(f"\n  Wrote {OUT}")
