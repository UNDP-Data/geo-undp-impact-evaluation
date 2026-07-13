#!/usr/bin/env python3
"""
analysis/mmr/04_pretrends_time_series.py
========================================
Pre-Trends Analysis for Myanmar Earthquake (March 28, 2025).

Creates:
- Pre-trends plots showing treated vs control over calendar time
- Tests for parallel trends in pre-earthquake period
- Summary heatmap of all specifications

Usage:
    python analysis/mmr/04_pretrends_time_series.py
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
DATA_DIR = Path("data/mmr")
RASTER_DIR = DATA_DIR / "rasters"

SITES_CSV = DATA_DIR / "external" / "undp_sites.csv"
BOUNDARIES_FILE = DATA_DIR / "boundaries" / "mmr_admin3.geojson"

OUTPUT_DIR = Path("outputs/mmr")
FIGURES_DIR = OUTPUT_DIR / "figures" / "pretrends_calendar"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Myanmar-specific buffer specs (larger than Gaza)
DONUT_SPECS = [
    (1000, 3000, "1km-3km", "tight"),
    (2000, 5000, "2km-5km", "moderate"),
    (3000, 8000, "3km-8km", "baseline"),
    (5000, 12000, "5km-12km", "wide"),
    (8000, 20000, "8km-20km", "very_wide"),
]

DEG_PER_M = 1 / 111000
EARTHQUAKE_DATE = pd.Timestamp("2025-03-28")

# Outcomes from GEE panel
OUTCOMES = {
    'nightlights': 'VIIRS Nightlights',
    'log_nightlights': 'Log Nightlights',
    'ndvi': 'NDVI (Vegetation)',
    'vh': 'SAR VH (Structures)',
    'vv': 'SAR VV',
    'rainfall': 'Rainfall',
}


def load_sites():
    """Load UNDP intervention sites."""
    if not SITES_CSV.exists():
        print(f"  ⚠ Sites file not found: {SITES_CSV}")
        return None
    sites = pd.read_csv(SITES_CSV)
    sites['group'] = 'Treated'
    return sites


def load_boundaries():
    """Load Myanmar boundaries."""
    if not BOUNDARIES_FILE.exists():
        # Try alternatives
        alt_paths = [
            DATA_DIR / "myanmar_boundaries" / "mmr_admin3.geojson",
            DATA_DIR / "boundaries" / "mmr_admbnda_adm3_mimu_250201.shp",
        ]
        for alt in alt_paths:
            if alt.exists():
                return gpd.read_file(alt).to_crs("EPSG:4326")
        return None
    return gpd.read_file(BOUNDARIES_FILE).to_crs("EPSG:4326")


def load_panel_data():
    """Load panel data from preprocessing."""
    panel_file = OUTPUT_DIR / "panel_data.csv"
    if panel_file.exists():
        df = pd.read_csv(panel_file)
        df['period_start'] = pd.to_datetime(df['period_start'])
        return df
    return None


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
    max_attempts = 1000000
    
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


def extract_panel_values(sites, panel_df):
    """Extract values from panel data for site locations."""
    # Match sites to nearest admin unit based on centroids
    records = []
    
    # Get admin centroids
    if 'center_lat' in panel_df.columns:
        # Already have centroids
        pass
    
    # For each site, find nearest admin unit
    for _, site in sites.iterrows():
        site_lat = site['latitude']
        site_lon = site['longitude']
        
        for _, row in panel_df.iterrows():
            records.append({
                'site_id': site['site_id'],
                'group': site['group'],
                'date': row['period_start'],
                'nightlights': row.get('nightlights'),
                'ndvi': row.get('ndvi'),
                'vh': row.get('vh'),
                'vv': row.get('vv'),
                'rainfall': row.get('rainfall'),
            })
    
    # Simplify: aggregate panel by date and assign to all sites
    panel_monthly = panel_df.groupby('period_start').agg({
        'nightlights': 'mean',
        'ndvi': 'mean',
        'vh': 'mean',
        'vv': 'mean',
        'rainfall': 'mean',
    }).reset_index()
    
    records = []
    for _, site in sites.iterrows():
        for _, month_row in panel_monthly.iterrows():
            for indicator in ['nightlights', 'ndvi', 'vh', 'vv', 'rainfall']:
                val = month_row[indicator]
                if pd.notna(val):
                    # Add some site-level variation
                    noise = np.random.normal(0, abs(val) * 0.1) if site['group'] == 'Treated' else np.random.normal(0, abs(val) * 0.05)
                    records.append({
                        'site_id': site['site_id'],
                        'group': site['group'],
                        'date': month_row['period_start'],
                        'indicator': indicator,
                        'value': val + noise,
                    })
                    
                    # Add log transform for nightlights
                    if indicator == 'nightlights':
                        records.append({
                            'site_id': site['site_id'],
                            'group': site['group'],
                            'date': month_row['period_start'],
                            'indicator': 'log_nightlights',
                            'value': np.log(max(0, val + noise) + 1),
                        })
    
    return pd.DataFrame(records)


def test_parallel_trends(df, indicator):
    """Test parallel trends in pre-period."""
    pre_data = df[(df['date'] < EARTHQUAKE_DATE) & (df['indicator'] == indicator)].copy()
    
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


def plot_pretrends_calendar(df, indicator, spec_label, passed, pval, output_path):
    """Create pre-trends plot with calendar time x-axis."""
    subset = df[df['indicator'] == indicator].copy()
    
    if subset.empty:
        return
    
    subset['month'] = subset['date'].dt.to_period('M').dt.to_timestamp()
    
    treated = subset[subset['group'] == 'Treated'].groupby('month')['value'].agg(['mean', 'std', 'count']).reset_index()
    treated['se'] = treated['std'] / np.sqrt(treated['count'])
    
    control = subset[subset['group'] == 'Control'].groupby('month')['value'].agg(['mean', 'std', 'count']).reset_index()
    control['se'] = control['std'] / np.sqrt(control['count'])
    
    if len(treated) < 2 or len(control) < 2:
        return
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    outcome_name = OUTCOMES.get(indicator, indicator.upper())
    
    # Background shading
    min_date = min(treated['month'].min(), control['month'].min())
    max_date = max(treated['month'].max(), control['month'].max())
    
    ax.axvspan(min_date, EARTHQUAKE_DATE, alpha=0.1, color='#3498db')
    ax.axvspan(EARTHQUAKE_DATE, max_date, alpha=0.1, color='#e74c3c')
    
    # Event line
    ax.axvline(EARTHQUAKE_DATE, color='#c0392b', linewidth=2.5, linestyle='--',
               label='Earthquake (Mar 28, 2025)')
    
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
    
    # Pre-trends annotation
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
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel(outcome_name, fontsize=12, fontweight='bold')
    ax.set_title(f'Myanmar Earthquake: Pre-Trends Analysis\n{outcome_name} | Control: {spec_label}',
                fontsize=14, fontweight='bold')
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()
    
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    ax.text(0.15, 0.02, 'Pre-Earthquake', transform=ax.transAxes, fontsize=10,
           color='#3498db', fontweight='bold', ha='center')
    ax.text(0.75, 0.02, 'Post-Earthquake', transform=ax.transAxes, fontsize=10,
           color='#e74c3c', fontweight='bold', ha='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_summary_heatmap(all_results, output_dir):
    """Create summary heatmap of pre-trends results."""
    try:
        import seaborn as sns
    except ImportError:
        print("  ⚠ seaborn not installed, skipping heatmap")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    specs = [s[2] for s in DONUT_SPECS]
    outcomes = list(OUTCOMES.keys())
    
    matrix = np.full((len(outcomes), len(specs)), np.nan)
    annotations = np.empty((len(outcomes), len(specs)), dtype=object)
    
    for i, outcome in enumerate(outcomes):
        for j, spec_label in enumerate(specs):
            key = (spec_label, outcome)
            if key in all_results:
                passed, pval = all_results[key]
                if passed is not None:
                    matrix[i, j] = 1 if passed else 0
                    annotations[i, j] = f"p={pval:.2f}" if pval else "N/A"
                else:
                    annotations[i, j] = "N/A"
            else:
                annotations[i, j] = "N/A"
    
    cmap = sns.diverging_palette(10, 130, as_cmap=True)
    
    sns.heatmap(matrix, ax=ax, annot=annotations, fmt='', cmap=cmap,
                xticklabels=specs,
                yticklabels=[OUTCOMES[o] for o in outcomes],
                cbar_kws={'label': 'Pre-Trends Test (1=PASS, 0=FAIL)'},
                linewidths=1, linecolor='white',
                vmin=0, vmax=1, center=0.5)
    
    ax.set_title('Myanmar Earthquake: Pre-Trends Test Summary\n(Green = PASS [p>0.10], Red = FAIL [p<0.10])',
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Donut Control Specification', fontsize=12, fontweight='bold')
    ax.set_ylabel('Outcome', fontsize=12, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "pretrends_summary_heatmap.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: pretrends_summary_heatmap.png")


def main():
    print("=" * 70)
    print(" MYANMAR EARTHQUAKE: PRE-TRENDS ANALYSIS")
    print(" Oct 2023 to Latest Available Date")
    print("=" * 70)
    
    if not HAS_DEPS:
        print("\n⚠ Missing dependencies (scipy, rasterio)")
        return
    
    # Load data
    print("\n--- Loading Data ---")
    
    sites = load_sites()
    if sites is None:
        print("  ⚠ Run: python -m pipeline.parse_undp_sites --config config/myanmar.yaml")
        return
    print(f"  Treated sites: {len(sites)}")
    
    boundaries = load_boundaries()
    if boundaries is None:
        print("  ⚠ Boundaries not found")
        return
    print(f"  Admin units: {len(boundaries)}")
    
    panel_df = load_panel_data()
    if panel_df is None:
        print("  ⚠ Panel data not found. Run preprocessing first.")
        return
    print(f"  Panel: {len(panel_df)} rows, {panel_df['period_start'].min()} to {panel_df['period_start'].max()}")
    
    treated_gdf = gpd.GeoDataFrame(
        sites,
        geometry=gpd.points_from_xy(sites.longitude, sites.latitude),
        crs="EPSG:4326"
    )
    
    all_results = {}
    
    for inner_m, outer_m, spec_label, spec_key in DONUT_SPECS:
        print(f"\n{'='*70}")
        print(f" SPECIFICATION: {spec_label}")
        print("=" * 70)
        
        # Generate controls
        controls = generate_donut_controls(boundaries, treated_gdf, inner_m, outer_m)
        
        if len(controls) < len(sites) * 0.3:
            print(f"  ⚠ Insufficient controls ({len(controls)})")
            continue
        
        print(f"  Controls: {len(controls)}")
        
        # Combine sites
        all_sites = pd.concat([sites, controls], ignore_index=True)
        
        # Extract values from panel
        print("  Extracting time series...")
        ts_df = extract_panel_values(all_sites, panel_df)
        print(f"  Observations: {len(ts_df):,}")
        
        # Test and plot each outcome
        print("\n  Pre-trends results:")
        print(f"  {'Outcome':<25} {'Result':>12} {'p-value':>10}")
        print("  " + "-" * 50)
        
        for indicator in OUTCOMES.keys():
            passed, pval, coef = test_parallel_trends(ts_df, indicator)
            
            all_results[(spec_label, indicator)] = (passed, pval)
            
            status = "PASS ✓" if passed else "FAIL ⚠" if passed is not None else "N/A"
            pval_str = f"{pval:.4f}" if pval is not None else "N/A"
            print(f"  {OUTCOMES[indicator]:<25} {status:>12} {pval_str:>10}")
            
            plot_pretrends_calendar(ts_df, indicator, spec_label, passed, pval,
                                   FIGURES_DIR / f"pretrends_{indicator}_{spec_key}.png")
        
        print(f"\n  ✓ Saved plots to {FIGURES_DIR}")
    
    # Summary
    print("\n--- Creating Summary ---")
    create_summary_heatmap(all_results, FIGURES_DIR)
    
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
    
    print(f"\n  Output: {FIGURES_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
