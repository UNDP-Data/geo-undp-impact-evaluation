# How the Key-Messages Numbers Were Calculated

This file documents every figure in `key_messages.md` and links it to its source
file and the script that reproduces it. Run `python3 roi_attributed.py` to regenerate
the Myanmar/Gaza ROI figures; run `python3 roi_calculation.py` for the full country table.

## Myanmar — "USD 20–40 per USD 1 invested"

| Input | Value | Source |
|---|---|---|
| Causal effect (DiD) | +30–55% more recovery vs controls, p<0.001 | `outputs/mmr/results/did_buffer_results.csv` (significant positive buffers: +51.3%, +53.2%, +36.3%, +31.9%) |
| Beneficiaries | 3,500,000 | UNDP-reported — **NOT in repo, confirm** |
| GDP per capita | USD 1,122 | Assumption (stated) |
| Budget (denominator) | USD 16.8M | `outputs/mmr/external/undp_projects.csv` — SUM of all active projects |
| Elasticity | 0.3 | Henderson, Storeygard & Weil (2012) |

**Formula:** `GDP_gain = 3,500,000 × $1,122 × (effect% × 0.3)`
- Low: 3.5M × $1,122 × (30 × 0.3)% = $3.93B × 9.0% = **$353M**
- High: 3.5M × $1,122 × (55 × 0.3)% = $3.93B × 16.5% = **$648M**

**ROI = GDP_gain / $16.8M = 21:1 to 39:1** -> reported as "20–40 per $1".

*Note on the denominator:* the dashboard shows the single Earthquake Recovery project
at **$1,215,388**. We do NOT use that as the denominator because the area-wide recovery
reflects UNDP's full early-recovery presence, not one project. We use the total active
portfolio ($16.8M), which is the conservative (larger) denominator. Using $1.2M instead
would raise ROI to ~290–540:1, which would overstate attribution.

## Gaza — "employment is the only intervention with a measurable return"

| Input | Value | Source |
|---|---|---|
| Employment effect | +11.8% NL -> +3.53% GDP, p<0.001 | `outputs/pse/tables/gdp_sensitivity.csv` ("Other" = employment/cash-for-work) |
| Beneficiaries (floor) | 54,204 (500m buffers) | `outputs/pse/tables/beneficiary_estimates.json` |
| GDP per capita | USD 1,122 | Assumption (stated) |
| Employment budget | ~USD 12.5M | Approximate — **confirm against project list** |
| Other sectors | Debris -3.7% (neg, p<0.05); Shelter/WASH/Social Cohesion n.s. | `outputs/pse/tables/gdp_sensitivity.csv` |

**Formula:** `GDP_gain = 54,204 × $1,122 × 3.53% = $2.15M/yr`
**ROI = $2.15M / $12.5M = 0.17 per $1 per year** (satellite-verified floor).
Rises toward break-even (~1:1) when UNDP's full reported employment caseload replaces
the 54,204 satellite-buffer floor.

## Recovery percentages (all countries)

Source: `outputs/<code>/figures/scorecard_data.csv`, nightlights row, `overall_recovery_pct`.

| Country | Recovery % | Post-event budget | Budget source |
|---|---|---|---|
| Myanmar | 68.7% | $16.8M active / $1.2M earthquake | `undp_projects.csv` |
| Gaza | 14.7% | $112.2M | `undp_projects.csv` |
| Syria | 100% | $98.3M | `undp_projects.csv` |
| Afghanistan | 100% | $97.3M | `undp_projects.csv` |
| DR Congo | 100% | $250.9M | `undp_projects.csv` |
| Ukraine | 48.0% | $970.8M | `undp_projects.csv` |

## Confidence tiers
- **CAUSAL** — Myanmar only (difference-in-differences vs matched controls).
- **SECTOR** — Gaza employment (significant sector-level effect).
- **CORRELATIONAL** — Syria, Afghanistan, DR Congo, Ukraine (recovery alongside investment).

## Scripts
- `roi_attributed.py` -> Myanmar & Gaza ROI (reproduces the headline numbers) -> `roi_attributed.json`
- `roi_calculation.py` -> full six-country table -> `roi_results.json`
