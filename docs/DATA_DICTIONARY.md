# Data Dictionary & Known Gotchas
**Purpose:** permanent record of naming conventions, label meanings, and recurring points of confusion in this repository, so they are documented once instead of re-discovered every session. Referenced by the paper ("From Space to Impact") and by `PROJECT_STATE` notes.

*Last updated: 2026-07-13. Update this file whenever a new convention or gotcha is settled.*

---

## 1. Gaza intervention categories — the "Other" label (THE BIG ONE)

The Gaza site digitisation (`analysis/pse/00_digitize_all_maps.py`, local machine) assigns each of the **454** geocoded UNDP sites to one of five categories. The category values as they appear in the output CSVs are:

| CSV label | Meaning |
|---|---|
| `Debris` | Debris management |
| `Shelter` | Shelter |
| `WASH` | Water, sanitation & hygiene |
| `Social Cohesion` | Social cohesion |
| `Other` | **Residual class: employment-related interventions (cash-for-work and allied income-injection activities) PLUS sites not classifiable into the four named sectors.** It is NOT a clean "Employment" sector. |

**Why this matters:** `Other` is the category with the headline +11.8% luminosity effect (Table 4 of the paper). The paper labels it **"Employment-related & other"** and explains the residual nature in Methods. Do **not** relabel it as pure "Employment / cash-for-work" — that was a mislabel caught and corrected on 2026-07-13.

Files where the `Other` label appears: `outputs/pse/tables/gdp_sensitivity.csv`, `did_by_intervention_type.csv`, `pretrends_tests.csv` (and any other per-sector output).

**TODO (local machine):** export per-category site counts from the digitisation output and record them here, so the size of each class is documented.

## 2. Gaza site count: 454, not 364

`outputs/pse/tables/beneficiary_estimates.json` → `direct_population.n_sites = 454`. This is the number underlying the population (54,204 unique / 77,434 with overlap) and valuation figures. The slide deck's **364 is stale** — do not propagate it.

## 3. Output-folder naming asymmetry (intentional to leave as-is)

- Myanmar results live in `outputs/mmr/results/`
- Gaza results live in `outputs/pse/tables/`

Yes, this is inconsistent. **Do not rename either folder** — pipeline and analysis scripts reference these paths, and renaming has previously broken things. The asymmetry is documented here instead.

## 4. Myanmar buffer-spec names vs. km radii

The DiD table (`outputs/mmr/results/did_buffer_results.csv`) uses km labels: `0.5km-1.5km`, `1.0km-3.0km`, `2.0km-5.0km`, `3.0km-8.0km`, `5.0km-15.0km`, `8.0km-20.0km` (each with a `+ controls` variant — **paper Table 3 uses the `+ controls` rows**).

The pre-trends figures (`outputs/mmr/figures/pretrends/`, `pretrends_calendar/`) use five names: `tight`, `moderate`, `baseline`, `wide`, `very_wide`.

**TODO (local machine):** transcribe the name→radius mapping from the spec definitions at the top of `analysis/mmr/02_pretrends_verified.py` into the table below:

| Figure name | km radii (inner–outer) |
|---|---|
| tight | ? |
| moderate | ? |
| baseline | ? (paper's "primary specification") |
| wide | ? |
| very_wide | ? |

## 5. `config/*.yaml` `intervention_start` is NOT the DiD treatment date

`config/myanmar.yaml` has `intervention_start: '2025-09-01'` — this matches the start date of the debris project (01004748) and drives dashboard visuals. The causal analysis convention (established earlier in the project) is that the **DiD post-period begins with actual UNDP arrival (May 2025)**, and Gaza uses **staggered per-site project start dates**. Confirm against the local `analysis/` scripts before citing any single treatment date.

## 6. Which file backs which headline number

| Claim | Ground-truth file |
|---|---|
| Myanmar 31.9–53.2% (Table 3) | `outputs/mmr/results/did_buffer_results.csv` (`log_nightlights`, `+ controls` rows) |
| Gaza overall +17.6% & sector rows (Table 4) | `outputs/pse/tables/gdp_sensitivity.csv` (log/staggered/conflict-controlled spec; log coef 0.1623 in `beneficiary_estimates.json`) |
| Gaza level-based donut sensitivity (negative viirs coefs — expected, targeting) | `outputs/pse/tables/did_conflict_controlled.csv`, `sensitivity_donut_specs.csv` |
| Gaza population & per-person gains | `outputs/pse/tables/beneficiary_estimates.json` (WorldPop) |
| Gaza SAR pixel DiD (+1.19, p<0.001, 500 m) | `outputs/pse/tables/pixel_did_results_500m.csv` (`sar_vh` row) |
| Gaza pre-trends (SAR VH parallel all sectors p=0.51–0.97; viirs violated) | `outputs/pse/tables/pretrends_tests.csv` |
| Myanmar cost base $16.8M / earthquake project $1.2M | `outputs/mmr/external/undp_projects.csv` (total 16,783,198; project 01004383 = 1,215,388) |

**Rule of the project:** repo CSVs are ground truth. Slide decks and the ROI summary doc have historically disagreed with the repo and with each other.

## 7. Git LFS

All `*.png`, `*.pdf`, `*.jpg` are Git LFS objects (`.gitattributes`). Cloud/sandbox environments can clone the repo and read all CSVs/JSON/YAML but generally **cannot download the image binaries** (LFS CDN blocked). Figure inspection must happen on a local machine.

## 8. Panel windows (as of 2026-07-13)

- Myanmar: 2023-10 → 2026-01 (28 months; `config/myanmar.yaml` `study_period_end: 2026-01-30`)
- Gaza: 2023-01 → 2025-12 (36 months)

VIIRS monthly composites publish with ~1–2-month lag; both panels are extendable to ~2026-05.

## 9. Known-missing from the public repo (as of 2026-07-13)

- `analysis/` (all Myanmar and Gaza analysis scripts — local only)
- ROI workbook: `roi_evidence/`, `CALCULATIONS.md`, `roi_attributed.json` (local only)
- Myanmar robustness **tables**: placebo, balance, pre-trends CSVs (figures exist; tables don't)
- Goodman-Bacon decomposition and dose-response-by-epicentre outputs (claimed in draft; no file anywhere — regenerate from `analysis/mmr/old/` or cut)
