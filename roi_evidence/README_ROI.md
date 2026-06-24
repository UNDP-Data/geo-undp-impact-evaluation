# Early Recovery — Economic Reactivation & ROI Evidence

This folder is the **single source of truth** for the recovery and ROI figures
used in the policy briefing deck (`recovery_evidence.qmd`).

## Files
- `roi_calculation.py` — documented calculation. Run it to (re)generate `roi_results.json`.
- `roi_results.json` — machine-readable results (all figures, tiers, assumptions).
- `recovery_evidence.qmd` — Quarto reveal.js policy deck.
- `README_ROI.md` — this file.

## Method (conservative, satellite-anchored)
```
affected_area_GDP = national_GDP * (n_affected_units / n_total_units)
GDP_reactivation  = affected_area_GDP * (nightlight_recovery% / 100) * 0.3
benefit_cost      = GDP_reactivation / UNDP_post_event_budget
```
Elasticity 0.3 = Henderson, Storeygard & Weil (2012), global average.

## Confidence tiers
- **CAUSAL** — difference-in-differences vs matched controls. *Myanmar only.*
- **SECTOR** — statistically significant sector-level effect. *Gaza employment.*
- **CORRELATIONAL** — pre-post recovery during UNDP engagement. *Syria, DRC, Ukraine, Afghanistan.*

## Honest-use rules (important)
1. Only **Myanmar** supports a causal claim (UNDP areas recovered 30–55% more, p<0.001).
2. **Gaza**: employment/cash-for-work is the **only** intervention with a significant
   positive economic signal (+11.8% nightlights → +3.5% GDP, p<0.001). Debris is negative;
   shelter/WASH not significant.
3. Per-country **dollar reactivation reflects ALL actors**, not UNDP alone. The raw
   benefit-cost ratios (e.g. Myanmar 11,292:1) are **NOT** defensible as UNDP returns and
   must never be presented as such. They appear in `roi_results.json` for transparency only.
4. The "$1.13 per $1" sector-multiplier figure from earlier slide decks could **not** be
   reproduced from repo data and should **not** be cited until rebuilt with disclosed
   assumptions (ILO multipliers, DALY values, etc.).

## Data sources (in repo)
- `outputs/<code>/figures/scorecard_data.csv` — recovery %, unit counts
- `outputs/<code>/external/wb_indicators.csv` — national GDP (hardcoded fallback for Syria)
- `outputs/<code>/external/undp_projects.csv` — budgets, post-disaster flag
- `outputs/pse/tables/gdp_sensitivity.csv` — Gaza sector-level effects
- `outputs/pse/tables/beneficiary_estimates.json` — Gaza beneficiaries

## Render the deck
```
quarto render recovery_evidence.qmd --to revealjs   # interactive HTML
# or to PDF (if decktape installed):
quarto render recovery_evidence.qmd --to revealjs && \
  decktape reveal recovery_evidence.html recovery_evidence.pdf
```
