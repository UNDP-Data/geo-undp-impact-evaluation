#!/usr/bin/env python3
"""
=============================================================================
ATTRIBUTED ROI — Myanmar & Gaza (the two cases with causal/sector identification)
=============================================================================
This script reproduces every ROI/economic-activity number in the key-messages
document (key_messages.md). Run it to regenerate the figures.

METHOD (differs from correlational countries in TWO ways):
  1. Uses the CAUSAL/SECTOR marginal effect (DiD or significant sector estimate),
     not raw area-wide recovery -> the gain is UNDP-attributable.
  2. Sizes the economy to BENEFICIARIES REACHED x GDP/capita (local),
     not whole-country GDP -> numerator and denominator matched in scope.

  GDP_gain = beneficiaries x GDP_per_capita x (marginal_NL_effect% x ELASTICITY)
  ROI      = GDP_gain / UNDP_budget

ELASTICITY = 0.3  (Henderson, Storeygard & Weil 2012, global average)

VERIFIED INPUTS (traceable to repo files):
  - Myanmar DiD effect: outputs/mmr/results/did_buffer_results.csv
       significant positive buffers: +51.3%, +53.2%, +36.3%, +31.9% -> band 30-55%
  - Myanmar budget: outputs/mmr/external/undp_projects.csv
       $1,215,388 = single Earthquake Recovery project (is_post_disaster=True)
       $16.8M     = SUM of all active UNDP projects in Myanmar (chosen denominator;
                    the area-wide recovery reflects UNDP's full ER presence, not one project)
  - Gaza employment effect: outputs/pse/tables/gdp_sensitivity.csv
       "Other" (=employment/cash-for-work): +11.8% NL -> +3.53% GDP, p<0.001
  - Gaza beneficiaries: outputs/pse/tables/beneficiary_estimates.json
       54,204 unique people in 500m buffers (conservative satellite floor)

ASSUMPTIONS TO CONFIRM BEFORE EXTERNAL CITATION:
  - Myanmar beneficiaries = 3,500,000 (UNDP-reported; NOT in repo -> confirm)
  - GDP per capita = $1,122 (Myanmar/Gaza; STATED assumption)
  - Gaza employment-sector budget = $12.5M (approximate; confirm against project list)
=============================================================================
"""
import json

EL = 0.3

# ---------- MYANMAR (CAUSAL) ----------
MMR_DID = (30, 55)                 # % more recovery than matched controls (p<0.001), VERIFIED
MMR_BENEFICIARIES = 3_500_000      # ASSUMPTION — confirm against UNDP records
MMR_GDP_PC = 1122                  # ASSUMPTION
MMR_BUDGET = 16.8e6                # total active UNDP early-recovery portfolio (chosen denominator)
MMR_BUDGET_EQ_ONLY = 1_215_388     # single earthquake-recovery project (for reference)

mmr_area_gdp = MMR_BENEFICIARIES * MMR_GDP_PC
mmr_gain_low  = mmr_area_gdp * (MMR_DID[0]*EL)/100
mmr_gain_high = mmr_area_gdp * (MMR_DID[1]*EL)/100
mmr_roi_low   = mmr_gain_low  / MMR_BUDGET
mmr_roi_high  = mmr_gain_high / MMR_BUDGET

# ---------- GAZA (SECTOR-LEVEL) ----------
GAZA_EMP_GDP_PCT = 3.53            # employment/"Other" GDP impact, p<0.001, VERIFIED
GAZA_BENEFICIARIES = 54_204        # satellite floor (500m), VERIFIED
GAZA_GDP_PC = 1122                 # ASSUMPTION
GAZA_EMP_BUDGET = 12.5e6           # ASSUMPTION — confirm

gaza_area_gdp = GAZA_BENEFICIARIES * GAZA_GDP_PC
gaza_gain = gaza_area_gdp * GAZA_EMP_GDP_PCT/100
gaza_roi  = gaza_gain / GAZA_EMP_BUDGET

def m(v): return f"${v/1e6:.0f}M"

print("="*70)
print("MYANMAR (causal DiD)")
print(f"  Effect: +{MMR_DID[0]}-{MMR_DID[1]}% more recovery than controls (p<0.001)")
print(f"  Beneficiary-area GDP: {MMR_BENEFICIARIES:,} x ${MMR_GDP_PC} = ${mmr_area_gdp/1e9:.2f}B")
print(f"  GDP gain: {m(mmr_gain_low)}-{m(mmr_gain_high)}")
print(f"  Budget (total active ER portfolio): {m(MMR_BUDGET)}")
print(f"  ROI: {mmr_roi_low:.0f}-{mmr_roi_high:.0f} per $1   -> document says '20-40 per $1'")
print(f"  [ref] earthquake-only project budget: ${MMR_BUDGET_EQ_ONLY:,}")
print()
print("GAZA (sector-level, employment/cash-for-work)")
print(f"  Effect: +11.8% NL -> +{GAZA_EMP_GDP_PCT}% GDP (p<0.001; only significant sector)")
print(f"  Beneficiary-area GDP: {GAZA_BENEFICIARIES:,} x ${GAZA_GDP_PC} = ${gaza_area_gdp/1e6:.1f}M")
print(f"  GDP gain: ${gaza_gain/1e6:.2f}M/yr")
print(f"  Budget (employment sector, approx): {m(GAZA_EMP_BUDGET)}")
print(f"  ROI: {gaza_roi:.2f} per $1 per year (satellite floor)")
print(f"       -> rises toward break-even (~1:1) with UNDP's full reported caseload")

out = {
  "myanmar": {"did_pct":MMR_DID,"beneficiaries":MMR_BENEFICIARIES,"gdp_per_capita":MMR_GDP_PC,
              "budget_total_active":MMR_BUDGET,"budget_earthquake_only":MMR_BUDGET_EQ_ONLY,
              "gdp_gain_usd":[round(mmr_gain_low),round(mmr_gain_high)],
              "roi_per_dollar":[round(mmr_roi_low),round(mmr_roi_high)]},
  "gaza": {"emp_gdp_pct":GAZA_EMP_GDP_PCT,"beneficiaries_satellite":GAZA_BENEFICIARIES,
           "gdp_per_capita":GAZA_GDP_PC,"emp_budget_usd":GAZA_EMP_BUDGET,
           "gdp_gain_usd_per_year":round(gaza_gain),"roi_per_dollar_per_year":round(gaza_roi,2)},
  "elasticity": EL,
}
with open("roi_attributed.json","w") as f: json.dump(out,f,indent=2)
print("\nWrote roi_attributed.json")
