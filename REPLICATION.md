# Replication Manifest — "From Space to Impact"
Every number and figure in the manuscript traces to the scripts and outputs below.
Run order matters. All commands from the repo root with the project venv.

## 0. Data acquisition (once, or when refreshing)
| Step | Script | Output |
|---|---|---|
| District/township panels (GEE) | `run_pipeline.py --stage gee` per `config/{myanmar,gaza}.yaml` | `data/{mmr,pse}/gee_panel_*_centroid.csv` |
| Conflict events | `run_pipeline.py --stage acled` (needs myACLED creds; scope=authenticated) | `data/*/external/acled_conflict.csv` |
| Panel assembly | `run_pipeline.py --stage preprocess` | `outputs/{mmr,pse}/panel_data.csv` |
| Site-level rasters (Gaza + Myanmar YoY) | `analysis/pse/synthetic_controls_other/raster_export.py` | `data/{pse,mmr}/rasters/*.tif` |
| WorldPop raster (Myanmar) | manual download, see header of `analysis/mmr/05_population_worldpop.py` | `data/mmr/population/mmr_ppp_2020_UNadj_constrained.tif` |

## 1. Causal estimates
| Paper element | Script | Tracked output |
|---|---|---|
| Table 3 (Myanmar DiD) | `analysis/mmr/01_did_buffer_fixed.py` | `outputs/mmr/results/did_buffer_results.csv` |
| Myanmar pre-trends (30/30) | `analysis/mmr/02_pretrends_verified.py` | `outputs/mmr/figures/pretrends*/` |
| Convergence windows (Fig 3a basis) | `analysis/mmr/06_convergence_window.py` | `outputs/mmr/results/convergence_window.csv` |
| Monthly event study | `analysis/mmr/07_paper_figures.py` (fig3a) | `outputs/mmr/results/event_study_monthly.csv` |
| Table 4 (Gaza sectors) | `analysis/pse/01_did_comprehensive.py` | `outputs/pse/tables/did_comprehensive_unified.csv` |
| Callaway–Sant'Anna robustness | `analysis/pse/05_callaway_santanna.py` | `outputs/pse/tables/cs_did_results.csv`, `cs_did_event_study.csv`, `outputs/pse/site_panel_baseline.csv` |
| WorldPop populations + BCR grid | `analysis/mmr/05_population_worldpop.py` | `outputs/mmr/results/population_worldpop.csv`, `population_estimates.json` |

## 2. Paper figures (standardized style: `analysis/paper_style.py`)
| Figure | Script | File in `outputs/paper_figures/` |
|---|---|---|
| Fig 1a MMR triptych | `analysis/mmr/07_paper_figures.py` | `fig1a_mmr_triptych.png` |
| Fig 1c MMR design map | same | `fig1c_mmr_design.png` |
| Fig 2b MMR series | same | `fig2b_mmr_ts.png` |
| Fig 3a MMR event study | same | `fig3a_mmr_event_study.png` |
| Fig 3b MMR pre-trends | same | `fig3b_mmr_pretrends.png` |
| Fig 4a PSE triptych | `analysis/pse/06_paper_figures.py` | `fig4a_pse_triptych.png` |
| Fig 4c PSE CS event study | same | `fig4c_pse_cs_event_study.png` |
| Fig 4d PSE design map | same | `fig4d_pse_design.png` |
| Fig 4b PSE sector summary | `analysis/pse/01_did_comprehensive.py` | copy of `outputs/pse/figures/did_results_summary.png` |

## Conventions
- Repo CSVs are ground truth over decks and memory.
- Category meanings, buffer-name mappings, and known gotchas: `docs/DATA_DICTIONARY.md`.
- Paper table/figure → file mapping: `docs/PAPER_ASSETS.md`.
- Additive-only policy: no deletions/renames of pipeline-referenced paths.
