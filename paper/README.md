# From Space to Impact: replication package

Everything in the manuscript is produced by `bash paper/reproduce.sh`, run from the repository root with the project environment (`paper/requirements.txt`). Myanmar inputs are frozen in `paper/inputs/myanmar/` (checksums in `SHA256SUMS.txt`). Gaza estimation panels are rebuilt from the rasters in `data/pse/rasters/` by `code/gaza_export.py`, which calls the functions in `analysis/pse/01_did_comprehensive.py`; their checksums are written to `outputs/logs/gaza_panel_SHA256SUMS.txt`. The code and outputs of the submitted version are tagged `paper-v1`.

| Manuscript item | Script | Output |
|---|---|---|
| Table 1, Myanmar estimates by band (PPML, OLS log, levels) and joint pre-test P | `code/part3_diagnostics.py` | `outputs/logs/myanmar_*km.txt` (section C for pre-tests), `outputs/tables/part3_myanmar_*km_results.csv` |
| Myanmar monsoon-excluded estimate; binned pre-trend test | `code/part3_diagnostics.py --ev-window 12 --drop-cal-months 6,7` | `outputs/logs/myanmar_3_8km_binned.txt` |
| Myanmar pre-earthquake anchored event study, pre-trend slope | `code/part3_diagnostics.py --ref-month -7 --pretrend-end -7` | `outputs/logs/myanmar_3_8km_prequake.txt`, `outputs/tables/part3_myanmar_3_8km_prequake_eventstudy.csv` |
| Fig. 2 bottom panel, Myanmar estimates by band | `code/make_myanmar_band_figure.py` | `outputs/figures/EDfig2_myanmar_bands.png` (formerly Extended Data Fig. 2, now shown only in Fig. 2) |
| Gaza original specification and its correction (17.4%, -2.0%, -4.7%) | `code/gaza_spec_check.py` | `outputs/logs/gaza_spec_check_original_500_1500.txt` |
| Gaza corrected specifications, 500-1,500 m; brightness gap 20.5 vs 13.4 | `code/gaza_spec_check_all.py` | `outputs/logs/gaza_spec_check_500_1500.txt` |
| Gaza 250-500 m pooled estimates, including the debris-site estimates quoted in the text; gap 20.5 vs 17.5 | `code/gaza_spec_check_all.py --panel ..._250_500_all.csv` | `outputs/logs/gaza_spec_check_250_500.txt` |
| Table 2, Gaza band robustness (estimates, SEs, P, pre-trends) | `code/gaza_buffer_sweep.py` | `outputs/tables/gaza_buffer_sweep.csv` |
| Extended Data Fig. 2, band sweep | `code/make_sweep_figure.py` | `outputs/figures/ED_fig3_gaza_band_sweep.png` |
| Fig. 4, Gaza event study, all sites, 250-500 m | `code/gaza_event_study.py` | `outputs/figures/fig4_gaza_event_study_250_500_all.png` |
| Extended Data Fig. 1, debris-management sites | `code/gaza_event_study.py --debris` | `outputs/figures/EDfig1_gaza_event_study_250_500_debris.png` |
| Extended Data Table 2 and in-text valuation figures | `code/valuation.py` | `outputs/tables/valuation_EDtable2.csv`, `outputs/logs/valuation.txt` |
| Valuation population input (3,005,825 in the 14 treated townships) | `analysis/mmr/05_population_worldpop.py` | `outputs/logs/worldpop_myanmar.txt` |
| Extended Data Table 5, Gaza estimator comparison | `code/part3_diagnostics.py` (Gaza panel) | `outputs/tables/part3_gaza_500_1500_results.csv` |
| Fig. 1 top (radiance maps) and bottom (3-8 km design map); Fig. 2 top (treated vs comparison series); Fig. 3 top (Gaza radiance maps) | dashboard pipeline (`run_pipeline.py --stage visuals`) and `analysis/` | `figures_in_manuscript/image1.png`, `image2.png`, `image3.png`, `image5.png` (exact files as embedded) |
| Fig. 3 bottom, UNDP sites and conflict-event density | dashboard pipeline (`run_pipeline.py --stage visuals`) | `figures_in_manuscript/conflict_map_three_panel_with_undp.png` (exact file as embedded) |
| Extended Data Table 3, data sources | none (Earth Engine collection IDs and extraction scales listed in Methods) | not applicable |
| Extended Data Tables 1 and 4, earlier specification | `analysis/mmr/01_did_buffer_fixed.py` (legacy) | reported for comparison only |

`figures_in_manuscript/image7.png` and `image8.png` duplicate the Fig. 4 and Extended Data Fig. 1 files in `outputs/figures/`, and `image9.png` is a superseded band plot; none is embedded in the manuscript.

Scripts elsewhere in the repository (including earlier versions of the above and exploratory analyses) were not used for the reported results.
