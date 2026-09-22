# From Space to Impact: replication package

Everything in the manuscript is produced by `bash paper/reproduce.sh`, run from the repository root with the project environment (`paper/requirements.txt`). Myanmar inputs are frozen in `paper/inputs/myanmar/` (checksums in `SHA256SUMS.txt`). Gaza estimation panels are rebuilt from the rasters in `data/pse/rasters/` by `code/gaza_export.py`, which calls the functions in `analysis/pse/01_did_comprehensive.py`; their checksums are written to `outputs/logs/gaza_panel_SHA256SUMS.txt`.

| Manuscript item | Script | Output |
|---|---|---|
| Table 1, Myanmar estimates by band (PPML, OLS log, levels) | `code/part3_diagnostics.py` | `outputs/logs/myanmar_*km.txt`, `outputs/tables/part3_myanmar_*km_results.csv` |
| Myanmar monsoon-excluded estimate; binned pre-trend test | `code/part3_diagnostics.py --ev-window 12 --drop-cal-months 6,7` | `outputs/logs/myanmar_3_8km_binned.txt` |
| Myanmar pre-earthquake anchored event study, pre-trend slope | `code/part3_diagnostics.py --ref-month -7 --pretrend-end -7` | `outputs/logs/myanmar_3_8km_prequake.txt`, `outputs/tables/part3_myanmar_3_8km_prequake_eventstudy.csv` |
| Extended Data Fig. 2, Myanmar bands | `code/make_myanmar_band_figure.py` | `outputs/figures/EDfig2_myanmar_bands.png` |
| Gaza original specification and its correction (17.4%, -2.0%, -4.7%) | `code/gaza_spec_check.py` | `outputs/logs/gaza_spec_check_original_500_1500.txt` |
| Gaza corrected specifications, 500-1,500 m; brightness gap 20.5 vs 13.4 | `code/gaza_spec_check_all.py` | `outputs/logs/gaza_spec_check_500_1500.txt` |
| Gaza 250-500 m pooled estimates incl. debris sites; gap 20.5 vs 17.5 | `code/gaza_spec_check_all.py --panel ..._250_500_all.csv` | `outputs/logs/gaza_spec_check_250_500.txt` |
| Table 2, Gaza band robustness (estimates, SEs, P, pre-trends) | `code/gaza_buffer_sweep.py` | `outputs/tables/gaza_buffer_sweep.csv` |
| Extended Data Fig. 3, band sweep | `code/make_sweep_figure.py` | `outputs/figures/ED_fig3_gaza_band_sweep.png` |
| Fig. 4, Gaza event study, all sites, 250-500 m | `code/gaza_event_study.py` | `outputs/figures/fig4_gaza_event_study_250_500_all.png` |
| Extended Data Fig. 1, debris-management sites | `code/gaza_event_study.py --debris` | `outputs/figures/EDfig1_gaza_event_study_250_500_debris.png` |
| Extended Data Table 5, Gaza estimator comparison | `code/part3_diagnostics.py` (Gaza panel) | `outputs/tables/part3_gaza_500_1500_results.csv` |
| Extended Data Table 2, valuation | computed from Table 1 (formula in Methods) | `code/valuation.py` (to add) |
| Figs. 1 to 3 (maps, time series, design maps) | dashboard pipeline (`run_pipeline.py --stage visuals`) and `analysis/` | `figures_in_manuscript/` (exact files as embedded) |
| Extended Data Tables 1 and 4, earlier specification | `analysis/mmr/01_did_buffer_fixed.py` (legacy) | reported for comparison only |

Scripts elsewhere in the repository (including earlier versions of the above and exploratory analyses) were not used for the reported results.
