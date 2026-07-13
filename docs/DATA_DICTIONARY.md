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

**How the label arises (traced 2026-07-13):** `analysis/pse/00_digitize_all_maps.py` assigns `intervention_type` from the source map's *filename* (`guess_category()`: wash→WASH, debris→Debris, shelter→Shelter, health→Health, employment→Employment, education→Education, anything else→**Other**). Sites from maps whose filenames didn't match a keyword fell into Other — which in practice contains the employment-related maps plus miscellaneous ones. Downstream, `CATEGORY_MAP` in `01_did_comprehensive.py` maps site types to project categories for start-date matching (Other → Health & Emergency / Governance & Rule of Law).

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

**RESOLVED (2026-07-13)** from `BUFFER_SPECS` in `analysis/mmr/02_pretrends_verified.py`:

| Figure name | km radii (inner–outer) |
|---|---|
| tight | 1–3 km |
| moderate | 2–5 km |
| baseline | 3–8 km |
| wide | 5–15 km |
| very_wide | 8–20 km |

The 0.5–1.5 km spec is excluded from the pre-trend suite (single treated township). **Gotcha: the file named "baseline" is 3–8 km, NOT 2–5 km.** Gaza's donut specs use a different naming (`did_conflict_controlled.csv`): very_tight 200–600 m, tight 300–800 m, moderate 400 m–1 km, etc.

## 5. Treatment dates — RESOLVED (2026-07-13, from the scripts)

**Myanmar:** single COMMON treatment date, **2025-09-01**, from `get_infrastructure_treatment_date()` in `analysis/mmr/01_did_buffer_fixed.py` — deliberately the first post-earthquake *infrastructure* project, "NOT the Environment project (April 2025)" per the script's own docstring. `config/myanmar.yaml` `intervention_start: 2025-09-01` is consistent. Because timing is common, Goodman-Bacon decomposition is *inapplicable* to Myanmar (nothing staggered to decompose) — the paper now says so explicitly.

**Gaza:** staggered per-site treatment; each site's `project_start` comes from matching its `intervention_type` through `CATEGORY_MAP` (in `analysis/pse/01_did_comprehensive.py`) to the earliest post-Oct-7-2023 project of the corresponding `ml_category` in `undp_projects.csv`.

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

- ~~`analysis/`~~ **PUSHED 2026-07-13 (commit 7098f5e)** along with `roi_attributed.json` and `roi_evidence/` (ACLED password scrubbed from 4 archived scripts first — rotate the ACLED credential).
- Site coordinates and rasters remain in gitignored `data/` (correct — size/licensing).
- Myanmar robustness tables ON THE CURRENT BUFFER SPEC: still missing. Archived tables from the earlier township-level spec are now tracked under `analysis/mmr/old_myanmar_analysis/outputs/` (placebo p=0.88/0.25, IPW balance, event study, dose-response +0.067/project p<0.001, damage controls +1.44 p=0.006).
- Goodman-Bacon: RESOLVED as inapplicable to Myanmar (common treatment timing); relevant only to Gaza's staggered design, where a Callaway–Sant'Anna robustness run remains the recommended referee-proofing.
- Gaza GDP per capita ($1,122) is an ASSUMPTION per `CALCULATIONS.md` (identical to Myanmar's WDI figure) — source a Gaza-Strip-specific pre-conflict value (PCBS) and recompute per-person gains. Gaza employment budget ($12.5M) is approximate — confirm against project list.
