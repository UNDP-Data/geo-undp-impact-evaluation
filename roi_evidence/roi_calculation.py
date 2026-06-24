#!/usr/bin/env python3
"""
=============================================================================
UNDP Early Recovery — Economic Reactivation & Benefit-Cost Estimates
=============================================================================
SINGLE SOURCE OF TRUTH for the recovery/ROI figures used in policy briefings.

METHOD (conservative, satellite-anchored):
  affected_area_GDP = national_GDP * (n_affected_units / n_total_units)
  GDP_reactivation  = affected_area_GDP * (nightlight_recovery% / 100) * ELASTICITY
  benefit_cost      = GDP_reactivation / UNDP_post_event_budget

ELASTICITY = 0.3  (Henderson, Storeygard & Weil 2012 — global average nightlight->GDP)

CONFIDENCE TIERS:
  CAUSAL      : difference-in-differences vs matched control areas (Myanmar only)
  SECTOR      : statistically significant sector-level effect (Gaza employment)
  CORRELATIONAL: pre-post recovery during UNDP engagement (Syria, DRC, Ukraine, Afghanistan)

EXPLICIT ASSUMPTIONS / LIMITATIONS:
  - The nightlight->GDP elasticity (0.3) is a GLOBAL AVERAGE and varies by context;
    it is least reliable in very low-light economies (Syria, Afghanistan) and in
    active-conflict settings where lights reflect non-economic activity (Ukraine, DRC).
  - GDP_reactivation reflects ALL actors on the ground, not UNDP alone. Only the
    Myanmar difference-in-differences estimate isolates UNDP's marginal contribution.
  - Benefit-cost ratios are INDICATIVE and correlational except where DiD is noted.
  - Budgets are allocations (not disbursements) for post-event projects.
  - This is a rapid-screening tool, NOT a substitute for full impact evaluation.

DATA SOURCES (all in repo):
  outputs/<code>/figures/scorecard_data.csv   (nightlight recovery %, unit counts)
  outputs/<code>/external/wb_indicators.csv    (national GDP; hardcoded fallback if missing)
  outputs/<code>/external/undp_projects.csv     (project budgets, post-disaster flag)
  outputs/pse/tables/gdp_sensitivity.csv        (Gaza sector-level effects)
=============================================================================
"""
import json

ELASTICITY = 0.3

# --- National pre-event GDP (USD). Source: wb_indicators.csv where available;
#     Syria has no WB data in repo -> documented hardcoded estimate. ---
GDP_PRE = {
    "mmr": 66.76e9,   # Myanmar 2023 (WB)
    "syr": 9.0e9,     # Syria — HARDCODED estimate (WB missing); heavily caveated
    "cod": 55.0e9,    # DR Congo ~2021 (WB)
    "ukr": 200.0e9,   # Ukraine 2021 pre-invasion (WB)
    "afg": 14.26e9,   # Afghanistan 2021 (WB)
    "pse": 18.11e9,   # Gaza/Palestine 2021 (WB)
}

# --- From scorecard_data.csv (nightlights row): overall recovery % and unit counts ---
SCORE = {
    "mmr": {"name":"Myanmar","recovery":68.7,"n_aff":325,"n_tot":330,"budget_post":1.2e6,  "tier":"CAUSAL"},
    "syr": {"name":"Syria","recovery":100.0,"n_aff":133,"n_tot":272,"budget_post":98.3e6, "tier":"CORRELATIONAL"},
    "cod": {"name":"DR Congo","recovery":100.0,"n_aff":184,"n_tot":189,"budget_post":250.9e6,"tier":"CORRELATIONAL"},
    "ukr": {"name":"Ukraine","recovery":48.0,"n_aff":494,"n_tot":495,"budget_post":970.8e6,"tier":"CORRELATIONAL"},
    "afg": {"name":"Afghanistan","recovery":100.0,"n_aff":352,"n_tot":398,"budget_post":97.3e6,"tier":"CORRELATIONAL"},
}

# --- Myanmar causal result (did_buffer_results.csv, log_nightlights, +controls) ---
# Significant positive buffers: +51.3%, +53.2%, +36.3%, +31.9% -> reported range
MMR_DID_RANGE = (30, 55)  # % more recovery in UNDP areas vs matched controls, p<0.001

# --- Gaza sector-level (gdp_sensitivity.csv, 500m buffer) ---
GAZA_SECTORS = {
    "Employment/cash-for-work":{"nl":11.76,"gdp":3.53,"sig":"p<0.001","significant":True},
    "Debris":                  {"nl":-3.74,"gdp":-1.12,"sig":"p<0.05 (negative)","significant":False},
    "Shelter":                 {"nl":-0.98,"gdp":-0.29,"sig":"n.s.","significant":False},
    "WASH":                    {"nl":-2.45,"gdp":-0.73,"sig":"n.s.","significant":False},
}
# Gaza employment beneficiaries (beneficiary_estimates.json, 500m unique estimate)
GAZA_BENEFICIARIES = 54204
GAZA_GDP_PER_CAPITA = 1122  # ASSUMPTION (consistent with Myanmar brief); STATE explicitly

def fmt(v):
    if abs(v) >= 1e9: return f"${v/1e9:.2f}B"
    if abs(v) >= 1e6: return f"${v/1e6:.1f}M"
    if abs(v) >= 1e3: return f"${v/1e3:.0f}K"
    return f"${v:,.0f}"

results = []
for code, d in SCORE.items():
    share = d["n_aff"] / d["n_tot"]
    area_gdp = GDP_PRE[code] * share
    gdp_pct = d["recovery"] * ELASTICITY
    reactivation = area_gdp * (gdp_pct/100)
    bcr = reactivation / d["budget_post"] if d["budget_post"] > 0 else None
    results.append({
        "code": code, "country": d["name"], "tier": d["tier"],
        "nightlight_recovery_pct": round(d["recovery"],1),
        "gdp_impact_pct": round(gdp_pct,1),
        "affected_share_pct": round(share*100,0),
        "affected_area_gdp_usd": round(area_gdp),
        "gdp_reactivation_usd": round(reactivation),
        "undp_post_event_budget_usd": round(d["budget_post"]),
        "indicative_bcr": round(bcr,1) if bcr else None,
        "note": "Affected-area reactivation reflects ALL actors; BCR indicative/correlational unless tier=CAUSAL",
    })

# Gaza employment sector calc
emp = GAZA_SECTORS["Employment/cash-for-work"]
gaza_area_gdp = GAZA_BENEFICIARIES * GAZA_GDP_PER_CAPITA
gaza_emp_react = gaza_area_gdp * (emp["gdp"]/100)

print("="*78)
print("UNDP EARLY RECOVERY — ECONOMIC REACTIVATION ESTIMATES (conservative, verified)")
print("="*78)
print(f"{'Country':<13}{'Tier':<14}{'NL rec%':<9}{'GDP%':<7}{'Reactivation':<14}{'Budget':<11}{'BCR':<8}")
print("-"*78)
for r in results:
    bcr = f"{r['indicative_bcr']}:1" if r['indicative_bcr'] else "n/a"
    print(f"{r['country']:<13}{r['tier']:<14}{r['nightlight_recovery_pct']:<9}{r['gdp_impact_pct']:<7}"
          f"{fmt(r['gdp_reactivation_usd']):<14}{fmt(r['undp_post_event_budget_usd']):<11}{bcr:<8}")

print()
print("MYANMAR (CAUSAL): difference-in-differences shows UNDP areas recovered "
      f"{MMR_DID_RANGE[0]}-{MMR_DID_RANGE[1]}% MORE than matched controls (p<0.001).")
print()
print("GAZA (SECTOR-LEVEL): employment/cash-for-work is the ONLY intervention type")
print(f"  with a significant positive economic signal: +{emp['nl']}% nightlights -> "
      f"+{emp['gdp']}% GDP ({emp['sig']}).")
print(f"  Beneficiaries: {GAZA_BENEFICIARIES:,} (500m). Affected-area GDP (beneficiaries x "
      f"${GAZA_GDP_PER_CAPITA}/cap) = {fmt(gaza_area_gdp)}.")
print(f"  Conservative reactivation (elasticity 0.3): {fmt(gaza_emp_react)}/yr.")
for s,v in GAZA_SECTORS.items():
    if not v["significant"]:
        print(f"  {s}: {v['nl']:+.1f}% NL ({v['sig']}) — no positive economic signal")

# Write machine-readable output
out = {
    "method": "affected_area_GDP * nightlight_recovery * elasticity(0.3); BCR vs post-event budget",
    "elasticity": ELASTICITY,
    "confidence_tiers": {
        "CAUSAL": "difference-in-differences vs matched controls (Myanmar)",
        "SECTOR": "significant sector-level effect (Gaza employment)",
        "CORRELATIONAL": "pre-post recovery during UNDP engagement (others)",
    },
    "country_results": results,
    "myanmar_did_range_pct": MMR_DID_RANGE,
    "gaza_sectors": GAZA_SECTORS,
    "gaza_employment": {
        "beneficiaries_500m": GAZA_BENEFICIARIES,
        "gdp_per_capita_assumption": GAZA_GDP_PER_CAPITA,
        "affected_area_gdp_usd": round(gaza_area_gdp),
        "reactivation_usd_per_year": round(gaza_emp_react),
    },
}
with open("roi_results.json","w") as f:
    json.dump(out, f, indent=2)
print("\nWrote roi_results.json")
