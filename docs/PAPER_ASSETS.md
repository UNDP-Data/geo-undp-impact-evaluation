# Paper Assets Manifest — "From Space to Impact"
Maps every table and figure in the manuscript to its exact source file in this repository. If a number in the paper and a number in these files disagree, **the file wins** (then fix the paper).

*Last updated: 2026-07-13 (manuscript v9).*

## Tables

| Paper element | Source file | Notes |
|---|---|---|
| Table 1 (data sources) | — | descriptive; cross-check `config/*.yaml` |
| Table 2 (Myanmar buffer specs) | `outputs/mmr/results/did_buffer_results.csv` | N treated/control per spec |
| Table 3 (Myanmar DiD) | `outputs/mmr/results/did_buffer_results.csv` | `log_nightlights`, **`+ controls`** rows only |
| Table 3 alt-sensor note / Supp S1b | same file | `rvi`, `vh`, `vv`, `ndvi`, `conflict_events` outcomes |
| Table 4 (Gaza sectors) | `outputs/pse/tables/gdp_sensitivity.csv` | 500 m rows shown; "Other" = Employment-related & other (see DATA_DICTIONARY §1) |
| Table 4 population footnote | `outputs/pse/tables/beneficiary_estimates.json` | n_sites 454; 54,204 unique / 77,434 total |
| Table 5 (Myanmar valuation) | `outputs/mmr/external/undp_projects.csv` (cost base); `roi_attributed.json` (LOCAL ONLY — grid still qualitative in paper) | portfolio $16,783,198; earthquake project $1,215,388 |
| Supp Table S0 (robustness asymmetry) | `outputs/pse/tables/` (54 files) vs `outputs/mmr/results/` (1 file) + `outputs/mmr/figures/pretrends*/` (62 figs) | |

## Figures (all Git LFS — inspect locally)

| Paper figure | File(s) |
|---|---|
| Fig 1a Myanmar setting | `outputs/mmr/figures/raster_maps/viirs_three_panel_with_undp.png` (alt: `map_conflict_events.png`) |
| Fig 1b Gaza setting | `outputs/pse/figures/conflict_map_three_panel_with_undp.png` |
| Fig 2a Myanmar NL raster | `outputs/mmr/figures/map_nightlights.png` |
| Fig 2b Myanmar NL time series | `outputs/mmr/figures/ts_nightlights.png` |
| Fig 3a Myanmar DiD coefficients | `outputs/mmr/figures/coef_plot_log_nightlights_buffer.png` |
| Fig 3b Myanmar pre-trends (primary spec) | `outputs/mmr/figures/pretrends/pretrends_log_nightlights_baseline.png` (calendar variant in `pretrends_calendar/`) |
| Fig 4a Gaza NL raster | `outputs/pse/figures/map_nightlights.png` |
| Fig 4b Gaza results summary | `outputs/pse/figures/did_results_summary.png` |
| Supp S1 pre-trend suite | `outputs/mmr/figures/pretrends/pretrends_summary.png`; `outputs/mmr/figures/pretrends_calendar/pretrends_summary_heatmap.png`; `outputs/pse/figures/pretrends_staggered/`, `pretrends_calendar/` |
| Supp S3 synthetic control | `outputs/pse/figures/synthetic_control/` |

## Workflow reminders

1. Repo CSVs > decks > memory. Always re-verify after any pipeline rerun.
2. Figures must be **visually confirmed by a human locally** before embedding — automated visual matching has failed twice in this project.
3. This manifest and `DATA_DICTIONARY.md` are additive documentation. Do not delete or rename pipeline-referenced folders (`outputs/mmr/results/` vs `outputs/pse/tables/` asymmetry is intentional-to-keep).
