#!/usr/bin/env bash
# Build the paper/ replication folder for "From Space to Impact".
# Run from the repo root:   bash build_paper_folder.sh
# Then reproduce everything: bash paper/reproduce.sh
set -euo pipefail
[ -f analysis/pse/01_did_comprehensive.py ] || { echo "Run from the repo root (data_analysis)."; exit 1; }

P=paper
mkdir -p $P/code $P/inputs/myanmar $P/outputs/{tables,logs,figures} $P/figures_in_manuscript

# ---- 1. final code only (the scripts that produced the paper) ----
for f in part3_diagnostics.py gaza_export.py gaza_spec_check.py gaza_spec_check_all.py \
         gaza_buffer_sweep.py gaza_event_study.py make_sweep_figure.py make_myanmar_band_figure.py; do
  [ -f "$f" ] || { echo "missing $f in repo root"; exit 1; }
  cp "$f" $P/code/
done
# let gaza_spec_check_all take --panel (it had a hard-coded path)
python3 - << 'EOF'
p = "paper/code/gaza_spec_check_all.py"; s = open(p).read()
old = 'P = "outputs/pse/estimation_panel_500_1500_all.csv"'
if old in s:
    s = s.replace(old, 'import argparse as _ap\n_a = _ap.ArgumentParser(); _a.add_argument("--panel", default="outputs/pse/estimation_panel_500_1500_all.csv")\nP = _a.parse_args().panel')
    open(p, "w").write(s); print("gaza_spec_check_all.py: --panel added")
EOF

# ---- 2. frozen Myanmar inputs (+ checksums) ----
cp outputs/mmr/panel_data.csv data/mmr/external/undp_sites.csv data/mmr/boundaries/mmr_admin3.geojson $P/inputs/myanmar/
( cd $P/inputs/myanmar && shasum -a 256 * > SHA256SUMS.txt )

# ---- 3. figures exactly as embedded in the manuscript ----
DOCX=$(ls -t ~/Downloads/From_Space_to_Impact_V6*.docx 2>/dev/null | head -1 || true)
if [ -n "${DOCX:-}" ]; then
  unzip -j -o -q "$DOCX" 'word/media/*' -d $P/figures_in_manuscript/ && echo "figures extracted from $DOCX"
else
  echo "NOTE: manuscript .docx not found in Downloads; copy it there and rerun to extract its figures"
fi

# ---- 4. environment ----
pip freeze | grep -iE "^(numpy|pandas|statsmodels|scipy|geopandas|shapely|pyproj|pyogrio|matplotlib|rasterio)==" > $P/requirements.txt || true

# ---- 5. one-command reproduction ----
cat > $P/reproduce.sh << 'EOF'
#!/usr/bin/env bash
# Reproduce every table and figure in the paper. Run from the repo root: bash paper/reproduce.sh
# Gaza estimation panels are rebuilt from rasters only if missing (slow, ~3 min per band).
set -euo pipefail
cd "$(dirname "$0")/.."
P=paper; C=$P/code; O=$P/outputs; mkdir -p $O/tables $O/logs $O/figures
MMR="--panel $P/inputs/myanmar/panel_data.csv --event-date 2025-09-01 --sites $P/inputs/myanmar/undp_sites.csv --boundaries $P/inputs/myanmar/mmr_admin3.geojson --id-column shapeID"

echo "== Myanmar: Table 1 (six bands) =="
for pair in 0.5:1.5 1:3 2:5 3:8 5:15 8:20; do in=${pair%%:*}; out=${pair##*:}
  python3 $C/part3_diagnostics.py $MMR --inner $in --outer $out --label myanmar_${in}_${out}km > $O/logs/myanmar_${in}_${out}km.txt 2>&1
  grep -E "^Units:" $O/logs/myanmar_${in}_${out}km.txt | sed "s/^/  ${in}-${out} km  /"
done
echo "== Myanmar: pre-earthquake anchored and monsoon-excluded event studies =="
python3 $C/part3_diagnostics.py $MMR --inner 3 --outer 8 --ref-month -7 --pretrend-end -7 --label myanmar_3_8km_prequake > $O/logs/myanmar_3_8km_prequake.txt 2>&1
python3 $C/part3_diagnostics.py $MMR --inner 3 --outer 8 --ref-month -7 --pretrend-end -7 --ev-window 12 --drop-cal-months 6,7 --label myanmar_3_8km_binned > $O/logs/myanmar_3_8km_binned.txt 2>&1
mv part3_myanmar_*.csv $O/tables/
( cd $O/tables && python3 ../../code/make_myanmar_band_figure.py > /dev/null && mv EDfig2_myanmar_bands.png ../figures/ )

echo "== Gaza: estimation panels =="
BANDS="100-200 100-250 200-300 250-500 300-600 500-1000 250-750 500-1500 750-2000 1000-3000 1500-4000"
for b in $BANDS; do i=${b%-*}; o=${b#*-}
  [ -f outputs/pse/estimation_panel_${i}_${o}_all.csv ] || python3 $C/gaza_export.py --inner $i --outer $o --indicator all
done
[ -f outputs/pse/estimation_panel_500_1500.csv ] || python3 $C/gaza_export.py --inner 500 --outer 1500 --indicator viirs
shasum -a 256 outputs/pse/estimation_panel_*.csv > $O/logs/gaza_panel_SHA256SUMS.txt

echo "== Gaza: original vs corrected specifications (text, Table 2 debris) =="
python3 $C/gaza_spec_check.py > $O/logs/gaza_spec_check_original_500_1500.txt 2>&1
python3 $C/gaza_spec_check_all.py --panel outputs/pse/estimation_panel_500_1500_all.csv > $O/logs/gaza_spec_check_500_1500.txt 2>&1
python3 $C/gaza_spec_check_all.py --panel outputs/pse/estimation_panel_250_500_all.csv > $O/logs/gaza_spec_check_250_500.txt 2>&1

echo "== Gaza: band sweep (Table 2, Extended Data Fig. 3) =="
python3 $C/gaza_buffer_sweep.py --skip-export --bands ${BANDS// /,} > $O/logs/gaza_buffer_sweep.txt 2>&1
mv gaza_buffer_sweep.csv $O/tables/
( cd $O/tables && python3 ../../code/make_sweep_figure.py > /dev/null && mv ED_fig3_gaza_band_sweep.png ../figures/ )

echo "== Gaza: event studies, 250-500 m (Fig. 4, Extended Data Fig. 1) =="
python3 $C/gaza_event_study.py --panel outputs/pse/estimation_panel_250_500_all.csv > $O/logs/gaza_event_study_250_500_all.txt 2>&1
mv gaza_event_study_all.csv $O/tables/gaza_event_study_250_500_all.csv; mv gaza_event_study_all.png $O/figures/fig4_gaza_event_study_250_500_all.png
python3 $C/gaza_event_study.py --panel outputs/pse/estimation_panel_250_500_all.csv --debris > $O/logs/gaza_event_study_250_500_debris.txt 2>&1
mv gaza_event_study_debris.csv $O/tables/gaza_event_study_250_500_debris.csv; mv gaza_event_study_debris.png $O/figures/EDfig1_gaza_event_study_250_500_debris.png

echo "== Gaza: estimator comparison, 500-1,500 m (Extended Data Table 5) =="
python3 $C/part3_diagnostics.py --panel outputs/pse/estimation_panel_500_1500.csv --event-date 2023-10-07 --treat-col treated --treat-date-col treat_date --ev-window 12 --label gaza_500_1500 > $O/logs/gaza_500_1500_estimators.txt 2>&1
mv part3_gaza_500_1500_*.csv $O/tables/

echo; echo "Done. Check: 3-8 km should read 14 treated, 18 control; PPML +19.4% in $O/logs/myanmar_3_8km.txt"
EOF
chmod +x $P/reproduce.sh

# ---- 6. map of every table and figure ----
cat > $P/README.md << 'EOF'
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
EOF

echo "paper/ folder built. Next: bash paper/reproduce.sh"
