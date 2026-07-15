# Paper Assets Manifest — "From Space to Impact"
Maps every table and figure in the manuscript to its exact source file in this repository. If a number in the paper and a number in these files disagree, **the file wins** (then fix the paper).

*Last updated: 2026-07-14 (manuscript v15, figure-complete).*

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

## Figures (final v15 structure; standardized PNGs in `outputs/paper_figures/`, all Git LFS)

| Paper figure | File | Producing script |
|---|---|---|
| Fig 1a MMR corridor triptych (Apr 2024/25/26) | `fig1a_mmr_triptych.png` | `analysis/mmr/07_paper_figures.py` |
| Fig 1b MMR study-design map | `fig1c_mmr_design.png` | same |
| Fig 2 MMR treated-vs-comparison series | `fig2b_mmr_ts.png` | same |
| Fig 3 MMR monthly event study | `fig3a_mmr_event_study.png` (+ `outputs/mmr/results/event_study_monthly.csv`) | same |
| Fig 4a PSE triptych (Sep/Nov 2023, Apr 2026) | `fig4a_pse_triptych.png` | `analysis/pse/06_paper_figures.py` |
| Fig 4b PSE TWFE across buffers (PASS/FAIL colours) | `fig4b_pse_results.png` | `analysis/pse/01_did_comprehensive.py` |
| Fig 4c PSE Callaway–Sant'Anna event study | `fig4c_pse_cs_event_study.png` | `analysis/pse/06_paper_figures.py` |
| Fig 4d PSE donut design map | `fig4d_pse_design.png` | same |
| Supp S1a MMR pre-trends gap (canonical test annotated) | `fig3b_mmr_pretrends.png` | `analysis/mmr/07_paper_figures.py` |
| Supp: PSE CS event study, SAR VH | `supp_pse_cs_event_study_sar_vh.png` | `analysis/pse/06_paper_figures.py` |
| Supp: MMR full-window coefficient plot | `outputs/mmr/figures/coef_plot_log_nightlights_buffer.png` | `analysis/mmr/01_did_buffer_fixed.py` |

**Testing convention (settled 2026-07-14):** the pre-trend test shown in figures is the CANONICAL one — identical to `02_pretrends_verified.py` (unclustered OLS, linear month×treated interaction, pre-earthquake months, pass = p>0.10). The township-clustered variant (baseline p=0.07) is reported as a robustness sentence in Methods only, never as a figure verdict. Do not reintroduce mixed test implementations across figures and text.

## Workflow reminders

1. Repo CSVs > decks > memory. Always re-verify after any pipeline rerun.
2. Figures must be **visually confirmed by a human locally** before embedding — automated visual matching has failed twice in this project.
3. This manifest and `DATA_DICTIONARY.md` are additive documentation. Do not delete or rename pipeline-referenced folders (`outputs/mmr/results/` vs `outputs/pse/tables/` asymmetry is intentional-to-keep).
