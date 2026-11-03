#!/usr/bin/env python3
"""
Gaza: every outcome, the paper's exact specification, and the corrections.

For each outcome (VIIRS, SAR VH, BSI, NDBI, NDVI) and for all sites and the
debris-management subset, estimates treated x post under:

  1  Paper, exact        district FE + month FE + conflict      (no treated main effect)
  2  + treated main      as 1, plus the treated main effect
  3  Site FE             site FE + month FE + conflict
  4  Site FE + bright    site FE + month FE x pre-conflict brightness tercile + conflict
                         -> lets bright and dim places follow different paths through
                            the conflict's collapse in lights ("controls for greater
                            luminosity in nearby areas")
  5  Site FE, conflict   as 3, sample from Oct 2023 on: baseline is conflict-period,
     baseline            not pre-conflict (close in spirit to Callaway-Sant'Anna)

Fixed effects are absorbed by alternating projections (exact FWL on convergence),
SEs clustered by site. Spec 1 on VIIRS should reproduce the published +17.4%.

Usage (repo root, venv active):
    python3 gaza_export.py --inner 500 --outer 1500 --indicator all   # once
    python3 gaza_spec_check_all.py
"""
import sys, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
warnings.filterwarnings("ignore")

P = "outputs/pse/estimation_panel_250_500_all.csv"
CONFLICT_START = pd.Timestamp("2023-10-01")
try:
    D = pd.read_csv(P, parse_dates=["period_start", "treat_date"])
except FileNotFoundError:
    sys.exit(f"{P} not found. Run: python3 gaza_export.py --inner 500 --outer 1500 --indicator all")

OUTCOMES = [c for c in ["viirs", "sar_vh", "bsi", "ndbi", "ndvi"] if c in D.columns]
D["post"] = (D["treat_date"].notna() & (D["period_start"] >= D["treat_date"])).astype(int)
D["tp"] = D["treated"] * D["post"]
D["month"] = D["period_start"].dt.strftime("%Y-%m")
D["district"] = D["district"].astype(str)
D["site_conflict_events"] = D["site_conflict_events"].fillna(0)

# pre-conflict brightness tercile per site (the luminosity control)
pre_b = D[D.period_start < CONFLICT_START].groupby("unit_id")["viirs"].mean()
D["bright_t"] = D["unit_id"].map(pd.qcut(pre_b.rank(method="first"), 3, labels=False)).fillna(-1).astype(int)
D["tercile_month"] = D["bright_t"].astype(str) + "_" + D["month"]


def demean(df, cols, fes, tol=1e-9, maxit=1000):
    X = df[cols].astype(float).copy()
    for _ in range(maxit):
        old = X.values.copy()
        for f in fes:
            X = X - X.groupby(df[f]).transform("mean")
        if np.max(np.abs(X.values - old)) < tol:
            break
    return X


def est(df, ycol, regs, fes):
    df = df.dropna(subset=[ycol] + regs)
    Z = demean(df, [ycol] + regs, fes)
    m = sm.OLS(Z[ycol].values, Z[regs].values).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(df["unit_id"])[0]})
    i = regs.index("tp")
    return m.params[i], m.bse[i], m.pvalues[i]


SPECS = [
    ("1 Paper, exact", ["tp", "site_conflict_events"], ["district", "month"], None),
    ("2 + treated main effect", ["tp", "treated", "site_conflict_events"], ["district", "month"], None),
    ("3 Site FE", ["tp", "site_conflict_events"], ["unit_id", "month"], None),
    ("4 Site FE + brightness x month", ["tp", "site_conflict_events"], ["unit_id", "tercile_month"], None),
    ("5 Site FE, conflict-period base", ["tp", "site_conflict_events"], ["unit_id", "month"], "conflict"),
]


def run(sample, label):
    n_t = sample.loc[sample.treated == 1, "unit_id"].nunique()
    n_c = sample.loc[sample.treated == 0, "unit_id"].nunique()
    print("\n" + "=" * 78)
    print(f"{label}: {n_t} treated sites, {n_c} controls")
    print("=" * 78)
    for oc in OUTCOMES:
        s = sample.copy()
        is_log = oc == "viirs"
        s["y"] = np.log1p(s[oc].clip(lower=0)) if is_log else s[oc]
        sd = s.loc[s.period_start < CONFLICT_START, "y"].std()
        unit = "% (log VIIRS+1)" if is_log else f"raw units; effect in pre-conflict SDs"
        print(f"\n  {oc.upper()}  [{unit}]")
        for name, regs, fes, restr in SPECS:
            ss = s[s.period_start >= CONFLICT_START] if restr == "conflict" else s
            try:
                b, se, p = est(ss, "y", regs, fes)
            except Exception as e:
                print(f"    {name:34s} failed: {e}"); continue
            star = "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""
            eff = f"{100*(np.exp(b)-1):+7.1f}%" if is_log else f"{b/sd:+7.2f} SD"
            print(f"    {name:34s} beta={b:+8.4f} se={se:7.4f} p={p:6.3f}{star:3s} {eff}")


print(f"Panel: {D.unit_id.nunique()} sites, {D.month.nunique()} months "
      f"({D.period_start.min().date()} to {D.period_start.max().date()}); outcomes: {OUTCOMES}")
gap = D[D.period_start < CONFLICT_START].groupby("treated")["viirs"].mean()
print(f"Pre-conflict mean VIIRS: treated {gap.get(1):.2f} vs control {gap.get(0):.2f} nW")

run(D, "ALL SITES")
deb = D["intervention_type"].astype(str).str.contains("debris", case=False, na=False)
deb_ids = set(D.loc[deb & (D.treated == 1), "unit_id"])
if deb_ids:
    run(D[D.unit_id.isin(deb_ids) | (D.treated == 0)], "DEBRIS-MANAGEMENT SITES vs all controls")
else:
    print("\n(no treated sites with 'debris' in intervention_type; types found: "
          f"{sorted(D.intervention_type.dropna().astype(str).unique())[:10]})")

print("\nReading guide:")
print("  Spec 1 on VIIRS should reproduce the published +17.4%.")
print("  Specs 4 and 5 are the fair tests: they compare treated and control sites that")
print("  started from similar brightness / the same conflict-period baseline.")
print("  If 4 and 5 are positive and significant, a Gaza effect survives honestly.")
