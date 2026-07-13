#!/usr/bin/env python3
"""
analysis/pse/11_pretrends_calendar_time.py
==========================================
Pre-Trends Plots Using Calendar Time (Jan 2023 to Latest).

This script creates easy-to-understand pre-trends plots showing:
- X-axis: Calendar dates (Jan 2023 onwards)
- Treated vs Control group means over time
- Vertical line at Oct 7, 2023 (conflict start)
- One plot per outcome × buffer specification

Usage:
    python analysis/pse/11_pretrends_calendar_time.py
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import warnings
from datetime import datetime
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings('ignore')

try:
    from scipy.spatial import cKDTree
    import rasterio
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path("data")
RASTER_DIR = DATA_DIR / "pse" / "rasters"

SITES_CSV = DATA_DIR / "pse" / "external" / "undp_sites_digitized.csv"
BOUNDARIES_FILE = DATA_DIR / "pse" / "boundaries" / "gaza_admin2.geojson"

OUTPUT_DIR = Path("outputs/pse")
FIGURES_DIR = OUTPUT_DIR / "figures" / "pretrends_calendar"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Buffer specifications (same as DiD regressions)
DONUT_SPECS = [
    (200, 600, "200m-600m", "very_tight"),
    (300, 800, "300m-800m", "tight"),
    (400, 1000, "400m-1km", "moderate"),
    (500, 1500, "500m-1.5km", "baseline"),
    (600, 1500, "600m-1.5km", "conservative"),
    (750, 2000, "750m-2km", "wide"),
]

DEG_PER_M = 1 / 111000
CONFLICT_START = pd.Timestamp("2023-10-07")

# Outcomes
OUTCOMES = {
    'viirs': 'VIIRS Nightlights',
    'log_viirs': 'Log Nightlights',
    'sar_vh': 'SAR VH (Structures)',
    'bsi': 'BSI (Bare Soil)',
    'ndvi': 'NDVI (Vegetation)',
    'ndbi': 'NDBI (Built-up)',
}

# =============================================================================
# DATA LOADING
# =============================================================================

def load_sites():
    """Load UNDP intervention sites."""
    sites = pd.read_csv(SITES_CSV)
    sites['group'] = 'Treated'
    return sites


def load_boundaries():
    """Load Gaza boundaries."""
    return gpd.read_file(BOUNDARIES_FILE).to_crs("EPSG:4326")


def find_rasters():
    """Find all raster files."""
    rasters = {}
    
    for f in RASTER_DIR.glob("*.tif"):
        parts = f.stem.split('_')
        
        if 'sar' in f.name.lower():
            ind = 'sar_vh' if 'vh' in f.name.lower() else 'sar_vv' if 'vv' in f.name.lower() else None
        else:
            ind = parts[0].lower()
        
        if ind:
            for part in parts:
                try:
                    if len(part) == 6 and part.isdigit():
                        d = datetime.strptime(part, "%Y%m")
                        if ind not in rasters:
                            rasters[ind] = {}
                        rasters[ind][d] = f
                        break
                except:
                    continue
    
    return rasters


def generate_donut_controls(boundaries, treated_gdf, inner_m, outer_m, n_controls=None, seed=42):
    """Generate control points in donut zone."""
    from shapely.geometry import Point
    
    np.random.seed(seed)
    
    if n_controls is None:
        n_controls = len(treated_gdf)
    
    inner_deg = inner_m * DEG_PER_M
    outer_deg = outer_m * DEG_PER_M
    
    treated_coords = np.array([[p.x, p.y] for p in treated_gdf.geometry])
    tree = cKDTree(treated_coords)
    
    control_points = []
    minx, miny, maxx, maxy = boundaries.total_bounds
    
    attempts = 0
    max_attempts = 500000
    
    while len(control_points) < n_controls and attempts < max_attempts:
        attempts += 1
        
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        
        pnt = Point(x, y)
        if not boundaries.contains(pnt).any():
            continue
        
        dist, _ = tree.query([x, y])
        
        if dist < inner_deg or dist > outer_deg:
            continue
        
        control_points.append({
            'site_id': f'CTRL_{len(control_points):04d}',
            'intervention_type': 'Control',
            'latitude': y,
            'longitude': x,
            'group': 'Control',
        })
    
    return pd.DataFrame(control_points)


# =============================================================================
# TIME SERIES EXTRACTION
# =============================================================================

def extract_time_series(sites, rasters):
    """Extract raster values for all sites over time."""
    records = []
    
    for indicator, dates in rasters.items():
        for date, raster_path in sorted(dates.items()):
            try:
                with rasterio.open(raster_path) as src:
                    band = src.read(1)
                    
                    for _, site in sites.iterrows():
                        try:
                            r, c = src.index(site['longitude'], site['latitude'])
                            val = band[r, c]
                            
                            if np.isnan(val) or val == src.nodata:
                                continue
                            
                            records.append({
                                'site_id': site['site_id'],
                                'group': site['group'],
                                'date': pd.Timestamp(date),
                                'indicator': indicator,
                                'value': float(val),
                            })
                            
                            # Add log for viirs
                            if indicator == 'viirs':
                                records.append({
                                    'site_id': site['site_id'],
                                    'group': site['group'],
                                    'date': pd.Timestamp(date),
                                    'indicator': 'log_viirs',
                                    'value': np.log(max(0, val) + 1),
                                })
                        except:
                            continue
            except:
                continue
    
    return pd.DataFrame(records)


# =============================================================================
# PRE-TRENDS TESTING
# =============================================================================

def test_parallel_trends(df, indicator):
    """Test parallel trends in pre-period using calendar time."""
    pre_data = df[(df['date'] < CONFLICT_START) & (df['indicator'] == indicator)].copy()
    
    if len(pre_data) < 50:
        return None, None, None
    
    pre_data['month_idx'] = pre_data['date'].dt.year * 12 + pre_data['date'].dt.month
    pre_data['is_treated'] = (pre_data['group'] == 'Treated').astype(int)
    
    try:
        model = smf.ols('value ~ month_idx * is_treated', data=pre_data).fit()
        
        if 'month_idx:is_treated' in model.pvalues:
            pval = model.pvalues['month_idx:is_treated']
            coef = model.params['month_idx:is_treated']
            passed = pval > 0.10
            return passed, pval, coef
    except:
        pass
    
    return None, None, None


def compute_group_means(df, indicator):
    """Compute monthly means by group."""
    subset = df[df['indicator'] == indicator].copy()
    
    if subset.empty:
        return None, None
    
    # Aggregate by month and group
    subset['month'] = subset['date'].dt.to_period('M').dt.to_timestamp()
    
    treated = subset[subset['group'] == 'Treated'].groupby('month')['value'].agg(['mean', 'std', 'count']).reset_index()
    treated['se'] = treated['std'] / np.sqrt(treated['count'])
    treated['group'] = 'Treated'
    
    control = subset[subset['group'] == 'Control'].groupby('month')['value'].agg(['mean', 'std', 'count']).reset_index()
    control['se'] = control['std'] / np.sqrt(control['count'])
    control['group'] = 'Control'
    
    return treated, control


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_pretrends_calendar(df, indicator, spec_label, passed, pval, output_path):
    """Create pre-trends plot with calendar time."""
    treated, control = compute_group_means(df, indicator)
    
    if treated is None or control is None or len(treated) < 3:
        return
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    outcome_name = OUTCOMES.get(indicator, indicator.upper())
    
    # Background shading
    min_date = min(treated['month'].min(), control['month'].min())
    max_date = max(treated['month'].max(), control['month'].max())
    
    ax.axvspan(min_date, CONFLICT_START, alpha=0.1, color='#3498db', label='_')
    ax.axvspan(CONFLICT_START, max_date, alpha=0.1, color='#e74c3c', label='_')
    
    # Conflict start line
    ax.axvline(CONFLICT_START, color='#c0392b', linewidth=2.5, linestyle='--', label='Oct 7, 2023')
    
    # Treated
    ax.fill_between(treated['month'],
                    treated['mean'] - 1.96*treated['se'],
                    treated['mean'] + 1.96*treated['se'],
                    alpha=0.2, color='#e74c3c')
    ax.plot(treated['month'], treated['mean'], 'o-', color='#e74c3c',
            linewidth=2, markersize=6, label='Treated (UNDP sites)')
    
    # Control
    ax.fill_between(control['month'],
                    control['mean'] - 1.96*control['se'],
                    control['mean'] + 1.96*control['se'],
                    alpha=0.2, color='#3498db')
    ax.plot(control['month'], control['mean'], 's--', color='#3498db',
            linewidth=2, markersize=6, label=f'Control ({spec_label})')
    
    # Pre-trends result annotation
    if passed is not None:
        if passed:
            status = f"✓ Parallel Trends HOLD (p={pval:.3f})"
            color = '#27ae60'
        else:
            status = f"⚠ Parallel Trends FAIL (p={pval:.3f})"
            color = '#e74c3c'
        
        ax.annotate(status, xy=(0.02, 0.98), xycoords='axes fraction',
                   fontsize=12, ha='left', va='top', fontweight='bold', color=color,
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                            edgecolor=color, alpha=0.95))
    
    # Formatting
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel(outcome_name, fontsize=12, fontweight='bold')
    ax.set_title(f'Pre-Trends Analysis: {outcome_name}\nControl Group: {spec_label}',
                fontsize=14, fontweight='bold')
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()
    
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add text annotations for pre/post periods
    ax.text(0.15, 0.02, 'Pre-Conflict', transform=ax.transAxes, fontsize=10,
           color='#3498db', fontweight='bold', ha='center')
    ax.text(0.75, 0.02, 'Post-Conflict', transform=ax.transAxes, fontsize=10,
           color='#e74c3c', fontweight='bold', ha='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_summary_grid(all_results, output_dir):
    """Create summary grid of all pre-trends plots."""
    # Create a heatmap-style summary
    fig, ax = plt.subplots(figsize=(14, 8))
    
    specs = [s[2] for s in DONUT_SPECS]
    outcomes = list(OUTCOMES.keys())
    
    # Build matrix
    matrix = np.full((len(outcomes), len(specs)), np.nan)
    annotations = np.empty((len(outcomes), len(specs)), dtype=object)
    
    for i, outcome in enumerate(outcomes):
        for j, spec_label in enumerate(specs):
            key = (spec_label, outcome)
            if key in all_results:
                passed, pval = all_results[key]
                if passed is not None:
                    matrix[i, j] = 1 if passed else 0
                    annotations[i, j] = f"p={pval:.2f}" if pval is not None else "N/A"
                else:
                    annotations[i, j] = "N/A"
            else:
                annotations[i, j] = "N/A"
    
    # Plot
    import seaborn as sns
    cmap = sns.diverging_palette(10, 130, as_cmap=True)
    
    sns.heatmap(matrix, ax=ax, annot=annotations, fmt='', cmap=cmap,
                xticklabels=specs,
                yticklabels=[OUTCOMES[o] for o in outcomes],
                cbar_kws={'label': 'Pre-Trends Test (1=PASS, 0=FAIL)'},
                linewidths=1, linecolor='white',
                vmin=0, vmax=1, center=0.5)
    
    ax.set_title('Pre-Trends Test Results Summary\n(Green = PASS [p>0.10], Red = FAIL [p<0.10])',
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Donut Control Specification', fontsize=12, fontweight='bold')
    ax.set_ylabel('Outcome', fontsize=12, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "pretrends_summary_heatmap.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: pretrends_summary_heatmap.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print(" PRE-TRENDS ANALYSIS (Calendar Time)")
    print(" Jan 2023 to Latest Available Date")
    print("=" * 70)
    
    if not HAS_DEPS:
        print("\n⚠ Missing dependencies")
        return
    
    # Load data
    print("\n--- Loading Data ---")
    
    sites = load_sites()
    print(f"  Treated sites: {len(sites)}")
    
    boundaries = load_boundaries()
    
    treated_gdf = gpd.GeoDataFrame(
        sites,
        geometry=gpd.points_from_xy(sites.longitude, sites.latitude),
        crs="EPSG:4326"
    )
    
    rasters = find_rasters()
    print(f"  Raster indicators: {list(rasters.keys())}")
    
    # Store all results for summary
    all_results = {}
    
    # Process each buffer specification
    for inner_m, outer_m, spec_label, spec_key in DONUT_SPECS:
        print(f"\n{'='*70}")
        print(f" SPECIFICATION: {spec_label}")
        print("=" * 70)
        
        # Generate controls
        controls = generate_donut_controls(boundaries, treated_gdf, inner_m, outer_m)
        
        if len(controls) < len(sites) * 0.5:
            print(f"  ⚠ Insufficient controls ({len(controls)})")
            continue
        
        print(f"  Controls: {len(controls)}")
        
        # Combine sites
        all_sites = pd.concat([sites, controls], ignore_index=True)
        
        # Extract time series
        print("  Extracting time series...")
        ts_df = extract_time_series(all_sites, rasters)
        print(f"  Observations: {len(ts_df):,}")
        
        # Test and plot each outcome
        print("\n  Pre-trends results:")
        print(f"  {'Outcome':<25} {'Result':>12} {'p-value':>10}")
        print("  " + "-" * 50)
        
        for indicator in OUTCOMES.keys():
            passed, pval, coef = test_parallel_trends(ts_df, indicator)
            
            # Store result
            all_results[(spec_label, indicator)] = (passed, pval)
            
            # Print
            status = "PASS ✓" if passed else "FAIL ⚠" if passed is not None else "N/A"
            pval_str = f"{pval:.4f}" if pval is not None else "N/A"
            print(f"  {OUTCOMES[indicator]:<25} {status:>12} {pval_str:>10}")
            
            # Create plot
            plot_pretrends_calendar(ts_df, indicator, spec_label, passed, pval,
                                   FIGURES_DIR / f"pretrends_{indicator}_{spec_key}.png")
        
        print(f"\n  ✓ Saved plots to {FIGURES_DIR}")
    
    # Create summary heatmap
    print("\n--- Creating Summary ---")
    create_summary_grid(all_results, FIGURES_DIR)
    
    # Print overall summary
    print("\n" + "=" * 70)
    print(" SUMMARY: Pre-Trends Pass Rates")
    print("=" * 70)
    
    print("\n  By Outcome:")
    for indicator in OUTCOMES.keys():
        passed_count = sum(1 for (s, i), (p, _) in all_results.items() 
                         if i == indicator and p == True)
        total = sum(1 for (s, i), (p, _) in all_results.items() 
                   if i == indicator and p is not None)
        pct = passed_count / total * 100 if total > 0 else 0
        status = "★ CREDIBLE" if pct >= 50 else ""
        print(f"    {OUTCOMES[indicator]:<25}: {passed_count}/{total} ({pct:.0f}%) {status}")
    
    print("\n  By Specification:")
    for inner_m, outer_m, spec_label, spec_key in DONUT_SPECS:
        passed_count = sum(1 for (s, i), (p, _) in all_results.items() 
                         if s == spec_label and p == True)
        total = sum(1 for (s, i), (p, _) in all_results.items() 
                   if s == spec_label and p is not None)
        pct = passed_count / total * 100 if total > 0 else 0
        print(f"    {spec_label:<20}: {passed_count}/{total} ({pct:.0f}%)")
    
    print(f"""
    
  OUTPUT FILES:
  ─────────────
  Individual plots:
    {FIGURES_DIR}/pretrends_[indicator]_[spec].png
  
  Summary:
    {FIGURES_DIR}/pretrends_summary_heatmap.png
    """)
    
    print("=" * 70)
    print(" ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
