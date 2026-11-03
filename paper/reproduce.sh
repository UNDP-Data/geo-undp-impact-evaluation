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

echo "== Myanmar: valuation (Extended Data Table 2 and text figures) =="
[ -f $O/logs/worldpop_myanmar.txt ] || python3 analysis/mmr/05_population_worldpop.py > $O/logs/worldpop_myanmar.txt 2>&1
python3 $C/valuation.py > $O/logs/valuation.txt 2>&1

echo "== Gaza: estimation panels =="
BANDS="100-200 100-250 200-300 250-500 300-600 500-1000 250-750 500-1500 750-2000 1000-3000 1500-4000"
for b in $BANDS; do i=${b%-*}; o=${b#*-}
  [ -f outputs/pse/estimation_panel_${i}_${o}_all.csv ] || python3 $C/gaza_export.py --inner $i --outer $o --indicator all
done
[ -f outputs/pse/estimation_panel_500_1500.csv ] || python3 $C/gaza_export.py --inner 500 --outer 1500 --indicator viirs
shasum -a 256 outputs/pse/estimation_panel_*.csv > $O/logs/gaza_panel_SHA256SUMS.txt

echo "== Gaza: original vs corrected specifications (text, including debris-site estimates) =="
python3 $C/gaza_spec_check.py > $O/logs/gaza_spec_check_original_500_1500.txt 2>&1
python3 $C/gaza_spec_check_all.py --panel outputs/pse/estimation_panel_500_1500_all.csv > $O/logs/gaza_spec_check_500_1500.txt 2>&1
python3 $C/gaza_spec_check_all.py --panel outputs/pse/estimation_panel_250_500_all.csv > $O/logs/gaza_spec_check_250_500.txt 2>&1

echo "== Gaza: band sweep (Table 2, Extended Data Fig. 2) =="
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

echo; echo "Done. Check: 3-8 km should read 14 treated, 18 control; PPML +19.4% in $O/logs/myanmar_3_8km.txt; valuation in $O/logs/valuation.txt"
