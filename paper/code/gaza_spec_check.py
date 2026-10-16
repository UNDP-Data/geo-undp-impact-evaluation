#!/usr/bin/env python3
"""
Gaza specification check. Reproduces the paper's exact TWFE formula on the
exported estimation panel, then adds the terms a difference-in-differences
normally requires, one at a time, to see where the +17.4% comes from.

Paper formula (analysis/pse/01_did_comprehensive.py, line 460):
    value ~ treated:post + site_conflict_events + C(district) + C(date_str)
In patsy, `treated:post` includes ONLY the interaction, not the `treated`
main effect. If treated sites are persistently brighter or darker than their
donut controls, the interaction absorbs that level gap.

Usage (repo root, venv active):
    python3 gaza_spec_check.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import statsmodels.formula.api as smf

P = "outputs/pse/estimation_panel_500_1500.csv"
try:
    d = pd.read_csv(P, parse_dates=["period_start", "treat_date"])
except FileNotFoundError:
    sys.exit(f"{P} not found; run gaza_export.py first")

d["y"] = np.log1p(d["nightlights"].clip(lower=0))
d["post"] = (d["treat_date"].notna() & (d["period_start"] >= d["treat_date"])).astype(int)
d["month"] = d["period_start"].dt.strftime("%Y-%m")
d["district"] = d["district"].astype(str)
d["site_conflict_events"] = d["site_conflict_events"].fillna(0)

ctrl_with_date = d.loc[d.treated == 0, "treat_date"].notna().mean()
print(f"Sites: {d.unit_id.nunique()} ({d.loc[d.treated==1,'unit_id'].nunique()} treated) | "
      f"months: {d.month.nunique()} | controls carrying a matched start date: {100*ctrl_with_date:.0f}%")

gap_pre = (d[(d.period_start < "2023-10-01")].groupby("treated")["y"].mean())
print(f"Pre-war mean log(VIIRS+1): treated {gap_pre.get(1, np.nan):.3f} vs control "
      f"{gap_pre.get(0, np.nan):.3f} (gap {gap_pre.get(1, np.nan)-gap_pre.get(0, np.nan):+.3f})\n")

specs = [
    ("1. Paper, exact", "y ~ treated:post + site_conflict_events + C(district) + C(month)"),
    ("2. + treated main effect", "y ~ treated + treated:post + site_conflict_events + C(district) + C(month)"),
    ("3. + treated and post mains", "y ~ treated * post + site_conflict_events + C(district) + C(month)"),
    ("4. Site FE (within-site)", "y ~ treated:post + site_conflict_events + C(unit_id) + C(month)"),
]
print(f"{'specification':30s} {'beta':>8s} {'se':>7s} {'p':>7s} {'implied %':>10s}")
for name, f in specs:
    m = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["unit_id"]})
    term = next(t for t in m.params.index if t.replace(" ", "") in ("treated:post",))
    b, s, p = m.params[term], m.bse[term], m.pvalues[term]
    print(f"{name:30s} {b:8.3f} {s:7.3f} {p:7.3f} {100*(np.exp(b)-1):9.1f}%")

print("\nReading: if spec 1 is near +17% and specs 2-4 fall toward zero, the published "
      "Gaza estimate reflects a level difference between treated and control sites, "
      "not a change after programming began.")
