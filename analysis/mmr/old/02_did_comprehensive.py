#!/usr/bin/env python3
"""
analysis/myanmar/02_did_comprehensive.py
========================================
Comprehensive DiD Analysis for Myanmar Earthquake Response.
Adapted from Gaza analysis methodology.

METHODOLOGY:
1. Site-level analysis with donut buffer controls
2. TWFE with earthquake timing (or staggered project starts)
3. Multiple buffer zone specifications for robustness
4. Pre-trends testing (calendar time)
5. Heterogeneous effects by intervention type

KEY DIFFERENCES FROM GAZA:
- Event: Earthquake (not conflict)
- Geography: Larger area, may need wider buffers
- Data: May have fewer raster types available

CONFIGURATION:
- Edit EARTHQUAKE_DATE, AFFECTED_REGION, etc. below
- Run 00_data_check.py first to verify data availability

Usage:
    python analysis/myanmar/02_did_comprehensive.py
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
from matplotlib.patches import Patch
import re
import os

warnings.filterwarnings('ignore')

try:
    from scipy.spatial import cKDTree
    import rasterio
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    print("⚠ Missing dependencies. Run: pip install rasterio scipy geopandas")

# =============================================================================
# CONFIGURATION - EDIT THESE FOR YOUR ANALYSIS
# =============================================================================

# --- PATHS ---
DATA_DIR = Path("data/myanmar")
RASTER_DIR = DATA_DIR / "rasters"  # For SAR, NDVI, BSI
VIIRS_DIR = DATA_DIR / "viirs_25"  # VIIRS nightlights

SITES_CSV = DATA_DIR / "external" / "undp_sites.csv"
PROJECTS_CSV = DATA_DIR / "external" / "undp_projects.csv"
ACLED_CSV = DATA_DIR / "external" / "acled_myanmar.csv"
BOUNDARIES_FILE = DATA_DIR / "myanmar_boundaries" / "mmr_admin3.geojson"

OUTPUT_DIR = Path("outputs/myanmar")
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
PRETRENDS_DIR = FIGURES_DIR / "pretrends_calendar"

for d in [TABLES_DIR, FIGURES_DIR, PRETRENDS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- EVENT TIMING ---
# EDIT THIS: When did the earthquake occur?
EARTHQUAKE_DATE = pd.Timestamp("2024-10-01")  # <-- CHANGE THIS

# --- GEOGRAPHIC SCOPE ---
# EDIT THIS: Which admin units were affected?
AFFECTED_STATES = None  # e.g., ['Sagaing', 'Mandalay'] or None for all
AFFECTED_TOWNSHIPS = None  # e.g., ['Township1', 'Township2'] or None for all

# --- BUFFER SPECIFICATIONS ---
# May need wider buffers for Myanmar's larger geography
DONUT_SPECS = [
    (500, 1500, "500m-1.5km", "tight"),
    (1000, 3000, "1km-3km", "moderate"),
    (1500, 5000, "1.5km-5km", "baseline"),
    (2000, 7000, "2km-7km", "wide"),
    (3000, 10000, "3km-10km", "very_wide"),
]

# Degrees per meter (approximate for Myanmar ~20°N)
DEG_PER_M = 1 / 111000

# --- OUTCOMES ---
# Will be populated based on available rasters
OUTCOMES = {}

# =============================================================================
# DATA LOADING
# =============================================================================

def find_rasters():
    """Find all available raster files."""
    rasters = {}
    
    # Check VIIRS directory
    if VIIRS_DIR.exists():
        for f in VIIRS_DIR.glob("*.tif"):
            # Extract date from filename
            # Pattern: SVDNB_npp_20250101-20250131_...
            match = re.search(r'(\d{8})-\d{8}', f.stem)
            if match:
                try:
                    dt = datetime.strptime(match.group(1), '%Y%m%d')
                    if 'viirs' not in rasters:
                        rasters['viirs'] = {}
                    rasters['viirs'][dt] = f
                except:
                    pass
    
    # Check general rasters directory
    if RASTER_DIR.exists():
        for f in RASTER_DIR.glob("*.tif"):
            parts = f.stem.lower()
            
            # Identify indicator type
            if 'sar' in parts or 'vh' in parts:
                ind = 'sar_vh'
            elif 'vv' in parts:
                ind = 'sar_vv'
            elif 'ndvi' in parts:
                ind = 'ndvi'
            elif 'ndbi' in parts:
                ind = 'ndbi'
            elif 'bsi' in parts:
                ind = 'bsi'
            else:
                continue
            
            # Extract date
            date_match = re.search(r'(\d{6})|(\d{4}-\d{2})|(\d{8})', f.stem)
            if date_match:
                date_str = date_match.group(1) or date_match.group(2) or date_match.group(3)
                try:
                    if len(date_str) == 8:
                        dt = datetime.strptime(date_str, '%Y%m%d')
                    elif len(date_str) == 6:
                        dt = datetime.strptime(date_str, '%Y%m')
                    else:
                        dt = datetime.strptime(date_str.replace('-', ''), '%Y%m')
                    
                    if ind not in rasters:
                        rasters[ind] = {}
                    rasters[ind][dt] = f
                except:
                    pass
    
    # Build OUTCOMES dict based on available rasters
    global OUTCOMES
    OUTCOMES = {
        'viirs': {'name': 'VIIRS Nightlights', 'direction': 'positive'},
        'log_viirs': {'name': 'Log Nightlights', 'direction': 'positive'},
    }
    
    if 'sar_vh' in rasters:
        OUTCOMES['sar_vh'] = {'name': 'SAR VH (Structures)', 'direction': 'positive'}
    if 'ndvi' in rasters:
        OUTCOMES['ndvi'] = {'name': 'NDVI (Vegetation)', 'direction': 'positive'}
    if 'ndbi' in rasters:
        OUTCOMES['ndbi'] = {'name': 'NDBI (Built-up)', 'direction': 'positive'}
    if 'bsi' in rasters:
        OUTCOMES['bsi'] = {'name': 'BSI (Bare Soil)', 'direction': 'negative'}
    
    return rasters


def load_boundaries():
    """Load Myanmar admin boundaries."""
    if not BOUNDARIES_FILE.exists():
        print(f"  ⚠ Boundaries not found: {BOUNDARIES_FILE}")
        return None
    
    gdf = gpd.read_file(BOUNDARIES_FILE).to_crs("EPSG:4326")
    
    # Filter to affected regions if specified
    if AFFECTED_STATES:
        state_col = next((c for c in gdf.columns if 'state' in c.lower() or 'adm1' in c.lower()), None)
        if state_col:
            gdf = gdf[gdf[state_col].isin(AFFECTED_STATES)]
    
    if AFFECTED_TOWNSHIPS:
        township_col = next((c for c in gdf.columns if 'township' in c.lower() or 'adm3' in c.lower()), None)
        if township_col:
            gdf = gdf[gdf[township_col].isin(AFFECTED_TOWNSHIPS)]
    
    return gdf


def load_sites():
    """Load UNDP intervention sites."""
    if not SITES_CSV.exists():
        print(f"  ⚠ Sites file not found: {SITES_CSV}")
        print("  → Will generate synthetic sites from township centroids")
        return None
    
    sites = pd.read_csv(SITES_CSV)
    
    # Standardize column names
    col_map = {
        'lat': 'latitude', 'LAT': 'latitude', 'Latitude': 'latitude',
        'lon': 'longitude', 'lng': 'longitude', 'LON': 'longitude', 'Longitude': 'longitude',
        'type': 'intervention_type', 'category': 'intervention_type',
    }
    for old, new in col_map.items():
        if old in sites.columns and new not in sites.columns:
            sites[new] = sites[old]
    
    # Add site_id if missing
    if 'site_id' not in sites.columns:
        sites['site_id'] = [f'SITE_{i:04d}' for i in range(len(sites))]
    
    sites['group'] = 'Treated'
    
    # Add project start dates if available
    if PROJECTS_CSV.exists():
        projects = pd.read_csv(PROJECTS_CSV)
        projects['start_date'] = pd.to_datetime(projects['start_date'], errors='coerce')
        
        # Match sites to projects (customize this logic as needed)
        sites['project_start'] = EARTHQUAKE_DATE  # Default to earthquake date
        
        # If intervention_type exists, try to match
        if 'intervention_type' in sites.columns and 'category' in projects.columns:
            for int_type in sites['intervention_type'].unique():
                matches = projects[projects['category'].str.contains(int_type, case=False, na=False)]
                if not matches.empty:
                    earliest = matches['start_date'].min()
                    if pd.notna(earliest):
                        sites.loc[sites['intervention_type'] == int_type, 'project_start'] = earliest
    else:
        sites['project_start'] = EARTHQUAKE_DATE
    
    return sites


def generate_sites_from_townships(boundaries, n_per_township=5, seed=42):
    """Generate synthetic sites from township centroids if no sites file."""
    np.random.seed(seed)
    
    sites = []
    
    for idx, row in boundaries.iterrows():
        geom = row.geometry
        centroid = geom.centroid
        
        # Get township ID
        id_col = next((c for c in ['adm3_pcode', 'TS_PCODE', 'township_id'] if c in boundaries.columns), None)
        township_id = row[id_col] if id_col else f'TWN_{idx}'
        
        # Generate points around centroid
        for i in range(n_per_township):
            # Random offset (up to ~1km)
            offset_x = np.random.uniform(-0.01, 0.01)
            offset_y = np.random.uniform(-0.01, 0.01)
            
            sites.append({
                'site_id': f'{township_id}_S{i:02d}',
                'latitude': centroid.y + offset_y,
                'longitude': centroid.x + offset_x,
                'township': township_id,
                'intervention_type': np.random.choice(['Shelter', 'WASH', 'Debris', 'Health', 'Other']),
                'group': 'Treated',
                'project_start': EARTHQUAKE_DATE,
            })
    
    return pd.DataFrame(sites)


def generate_donut_controls(boundaries, treated_gdf, inner_m, outer_m, n_controls=None, seed=42):
    """Generate control points in donut zone around treated sites."""
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
            'project_start': pd.NaT,
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
                    nodata = src.nodata
                    
                    for _, site in sites.iterrows():
                        try:
                            r, c = src.index(site['longitude'], site['latitude'])
                            
                            # Check bounds
                            if r < 0 or r >= band.shape[0] or c < 0 or c >= band.shape[1]:
                                continue
                            
                            val = band[r, c]
                            
                            if np.isnan(val) or (nodata is not None and val == nodata):
                                continue
                            
                            records.append({
                                'site_id': site['site_id'],
                                'group': site['group'],
                                'intervention_type': site.get('intervention_type', 'Unknown'),
                                'project_start': site.get('project_start', pd.NaT),
                                'date': pd.Timestamp(date),
                                'indicator': indicator,
                                'value': float(val),
                            })
                            
                            # Add log for viirs
                            if indicator == 'viirs':
                                records.append({
                                    'site_id': site['site_id'],
                                    'group': site['group'],
                                    'intervention_type': site.get('intervention_type', 'Unknown'),
                                    'project_start': site.get('project_start', pd.NaT),
                                    'date': pd.Timestamp(date),
                                    'indicator': 'log_viirs',
                                    'value': np.log(max(0, val) + 1),
                                })
                        except:
                            continue
            except Exception as e:
                print(f"    ⚠ Error reading {raster_path.name}: {e}")
    
    return pd.DataFrame(records)


# =============================================================================
# ANALYSIS
# =============================================================================

def run_twfe_regression(df, include_heterogeneous=True):
    """Run Two-Way Fixed Effects regression."""
    results = []
    
    # Create post indicator (based on earthquake date)
    df['post'] = (df['date'] >= EARTHQUAKE_DATE).astype(int)
    df['treated'] = (df['group'] == 'Treated').astype(int)
    df['date_str'] = df['date'].dt.strftime('%Y-%m')
    
    for indicator in df['indicator'].unique():
        subset = df[df['indicator'] == indicator].copy()
        
        if len(subset) < 100:
            continue
        
        # Overall effect
        try:
            formula = 'value ~ treated:post + C(site_id) + C(date_str)'
            model = smf.ols(formula, data=subset).fit(cov_type='cluster', cov_kwds={'groups': subset['site_id']})
            
            coef_name = [c for c in model.params.index if 'treated:post' in c]
            if coef_name:
                coef = model.params[coef_name[0]]
                se = model.bse[coef_name[0]]
                pval = model.pvalues[coef_name[0]]
                
                stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
                
                results.append({
                    'indicator': indicator,
                    'intervention_type': 'Overall',
                    'coef': coef,
                    'se': se,
                    'pval': pval,
                    'stars': stars,
                    'R2': model.rsquared,
                    'N': len(subset),
                })
        except Exception as e:
            print(f"    ⚠ Regression failed for {indicator}: {e}")
    
    return pd.DataFrame(results)


def test_parallel_trends(df, indicator):
    """Test parallel trends in pre-earthquake period."""
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


def compute_group_means(df, indicator):
    """Compute monthly means by group for plotting."""
    subset = df[df['indicator'] == indicator].copy()
    
    if subset.empty:
        return None, None
    
    subset['month'] = subset['date'].dt.to_period('M').dt.to_timestamp()
    
    treated = subset[subset['group'] == 'Treated'].groupby('month')['value'].agg(['mean', 'std', 'count']).reset_index()
    treated['se'] = treated['std'] / np.sqrt(treated['count'])
    treated['group'] = 'Treated'
    
    control = subset[subset['group'] == 'Control'].groupby('month')['value'].agg(['mean', 'std', 'count']).reset_index()
    control['se'] = control['std'] / np.sqrt(control['count'])
    control['group'] = 'Control'
    
    return treated, control


def plot_pretrends_calendar(df, indicator, spec_label, passed, pval, output_path):
    """Create pre-trends plot with calendar time."""
    treated, control = compute_group_means(df, indicator)
    
    if treated is None or control is None or len(treated) < 2:
        return
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    outcome_name = OUTCOMES.get(indicator, {}).get('name', indicator.upper())
    
    # Background shading
    min_date = min(treated['month'].min(), control['month'].min())
    max_date = max(treated['month'].max(), control['month'].max())
    
    ax.axvspan(min_date, EARTHQUAKE_DATE, alpha=0.1, color='#3498db')
    ax.axvspan(EARTHQUAKE_DATE, max_date, alpha=0.1, color='#e74c3c')
    
    # Event line
    ax.axvline(EARTHQUAKE_DATE, color='#c0392b', linewidth=2.5, linestyle='--', 
               label=f'Earthquake ({EARTHQUAKE_DATE.strftime("%b %d, %Y")})')
    
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
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=color, alpha=0.95))
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel(outcome_name, fontsize=12, fontweight='bold')
    ax.set_title(f'Pre-Trends Analysis: {outcome_name}\nControl Group: {spec_label}',
                fontsize=14, fontweight='bold')
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate()
    
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print(" MYANMAR EARTHQUAKE: COMPREHENSIVE DiD ANALYSIS")
    print(" Site-Level | Donut Controls | Multiple Buffer Zones")
    print("=" * 70)
    
    if not HAS_DEPS:
        print("\n⚠ Missing dependencies. Exiting.")
        return
    
    # Load data
    print("\n--- Loading Data ---")
    
    boundaries = load_boundaries()
    if boundaries is None:
        print("  ❌ Cannot proceed without boundary files")
        return
    print(f"  Boundaries: {len(boundaries)} admin units")
    
    rasters = find_rasters()
    print(f"  Raster indicators: {list(rasters.keys())}")
    
    if not rasters:
        print("  ❌ No raster data found. Check paths.")
        return
    
    # Load or generate sites
    sites = load_sites()
    if sites is None:
        print("  Generating synthetic sites from townships...")
        sites = generate_sites_from_townships(boundaries)
    
    print(f"  Treatment sites: {len(sites)}")
    if 'intervention_type' in sites.columns:
        print(f"  Intervention types: {sites['intervention_type'].value_counts().to_dict()}")
    
    # Create GeoDataFrame for sites
    sites_gdf = gpd.GeoDataFrame(
        sites,
        geometry=gpd.points_from_xy(sites.longitude, sites.latitude),
        crs="EPSG:4326"
    )
    
    # Run analysis for each buffer specification
    all_results = []
    all_pretrends = {}
    
    for inner_m, outer_m, spec_label, spec_key in DONUT_SPECS:
        print(f"\n{'='*70}")
        print(f" BUFFER SPECIFICATION: {spec_label}")
        print("=" * 70)
        
        # Generate controls
        print("  Generating control points...")
        controls = generate_donut_controls(boundaries, sites_gdf, inner_m, outer_m)
        
        if len(controls) < len(sites) * 0.3:
            print(f"  ⚠ Insufficient controls ({len(controls)}), skipping")
            continue
        
        print(f"  Controls: {len(controls)}")
        
        # Combine sites
        all_sites = pd.concat([sites, controls], ignore_index=True)
        
        # Extract time series
        print("  Extracting time series from rasters...")
        df = extract_time_series(all_sites, rasters)
        
        if df.empty:
            print("  ⚠ No data extracted, skipping")
            continue
        
        n_treated = df[df['group'] == 'Treated']['site_id'].nunique()
        n_control = df[df['group'] == 'Control']['site_id'].nunique()
        print(f"  Dataset: {len(df):,} obs ({n_treated} treated, {n_control} control)")
        
        # Run regression
        print("\n  Running TWFE regressions...")
        results = run_twfe_regression(df)
        
        if results.empty:
            continue
        
        results['spec_label'] = spec_label
        results['spec_key'] = spec_key
        results['inner_m'] = inner_m
        results['outer_m'] = outer_m
        
        # Pre-trends testing
        print("\n  Testing pre-trends...")
        for indicator in OUTCOMES.keys():
            passed, pval, coef = test_parallel_trends(df, indicator)
            
            mask = results['indicator'] == indicator
            results.loc[mask, 'pretrends_passed'] = passed
            results.loc[mask, 'pretrends_pval'] = pval
            
            all_pretrends[(spec_label, indicator)] = (passed, pval)
            
            status = "PASS ✓" if passed else "FAIL ⚠" if passed is not None else "N/A"
            print(f"    {indicator:<12}: {status}")
            
            # Create plot
            plot_pretrends_calendar(df, indicator, spec_label, passed, pval,
                                   PRETRENDS_DIR / f"pretrends_{indicator}_{spec_key}.png")
        
        # Print results
        print("\n  Results (Overall):")
        print(f"  {'Indicator':<12} {'Coef':>10} {'SE':>8} {'R²':>6}")
        print("  " + "-" * 40)
        
        for _, row in results[results['intervention_type'] == 'Overall'].iterrows():
            print(f"  {row['indicator']:<12} {row['coef']:>+10.4f}{row['stars']:<3} {row['se']:>8.4f} {row['R2']:>6.3f}")
        
        all_results.append(results)
    
    # Combine and save results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(TABLES_DIR / "did_comprehensive_myanmar.csv", index=False)
        print(f"\n✓ Saved: {TABLES_DIR}/did_comprehensive_myanmar.csv")
        
        # Summary
        print("\n" + "=" * 70)
        print(" SUMMARY: Pre-Trends Pass Rates")
        print("=" * 70)
        
        for indicator in OUTCOMES.keys():
            passed_count = sum(1 for (s, i), (p, _) in all_pretrends.items() if i == indicator and p == True)
            total = sum(1 for (s, i), (p, _) in all_pretrends.items() if i == indicator and p is not None)
            pct = passed_count / total * 100 if total > 0 else 0
            status = "★ CREDIBLE" if pct >= 50 else ""
            print(f"  {OUTCOMES[indicator]['name']:<25}: {passed_count}/{total} ({pct:.0f}%) {status}")
    
    print("\n" + "=" * 70)
    print(" ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
