#!/bin/bash
# setup_showcase.sh — Create showcase folders with curated impact analysis files
# Run once, then add/remove files as needed. The app reads from showcase/ only.

set -e

# ── Myanmar ──
mkdir -p outputs/mmr/showcase
# Buffer zone coefficient plots
cp outputs/mmr/figures/coef_plot_log_nightlights_buffer.png outputs/mmr/showcase/ 2>/dev/null || true
cp outputs/mmr/figures/coef_plot_ndvi_buffer.png outputs/mmr/showcase/ 2>/dev/null || true
cp outputs/mmr/figures/coef_plot_vh_buffer.png outputs/mmr/showcase/ 2>/dev/null || true
# DiD regression table
cp outputs/mmr/results/did_buffer_results.csv outputs/mmr/showcase/ 2>/dev/null || true
# Pretrends (baseline specification)
for f in outputs/mmr/figures/pretrends/*_baseline.png; do
    cp "$f" outputs/mmr/showcase/ 2>/dev/null || true
done
# UNDP location map
cp outputs/mmr/figures/raster_maps/viirs_three_panel_with_undp.png outputs/mmr/showcase/ 2>/dev/null || true
echo "✓ Myanmar showcase: $(ls outputs/mmr/showcase/ | wc -l) files"

# ── Gaza ──
mkdir -p outputs/pse/showcase
# DiD summary
cp outputs/pse/figures/did_results_summary.png outputs/pse/showcase/ 2>/dev/null || true
# Debris three-panel maps (clean versions)
cp outputs/pse/figures/undp_maps/viirs_debris_three_panel_clean.png outputs/pse/showcase/ 2>/dev/null || true
cp outputs/pse/figures/undp_maps/ndvi_debris_three_panel_clean.png outputs/pse/showcase/ 2>/dev/null || true
cp outputs/pse/figures/undp_maps/ndbi_debris_three_panel_clean.png outputs/pse/showcase/ 2>/dev/null || true
cp outputs/pse/figures/undp_maps/sar_vh_debris_three_panel_clean.png outputs/pse/showcase/ 2>/dev/null || true
cp outputs/pse/figures/undp_maps/bsi_debris_three_panel_clean.png outputs/pse/showcase/ 2>/dev/null || true
# Pretrends (baseline from calendar)
for f in outputs/pse/figures/pretrends_calendar/*_baseline.png; do
    cp "$f" outputs/pse/showcase/ 2>/dev/null || true
done
# DiD table (comprehensive unified, will be filtered in app to 500m-1.5km)
cp outputs/pse/tables/did_comprehensive_unified.csv outputs/pse/showcase/ 2>/dev/null || true
echo "✓ Gaza showcase: $(ls outputs/pse/showcase/ | wc -l) files"

# ── Afghanistan ──
mkdir -p outputs/afg/showcase
echo "✓ Afghanistan showcase: (empty — add files as needed)"

# ── Ukraine ──
mkdir -p outputs/ukr/showcase
echo "✓ Ukraine showcase: (empty — add files as needed)"

echo ""
echo "Done! Add or remove files from outputs/{code}/showcase/ to control what appears in the Impact Measurement tab."
