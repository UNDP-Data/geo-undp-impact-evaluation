#!/usr/bin/env python3
"""
analysis/pse/02_did_pretrends_unified.py
========================================
Unified DiD and Pre-Trends Analysis with Consistent Buffer Zones.

Features:
1. Consistent buffer specifications across DiD and pre-trends
2. Conflict events as an additional outcome variable
3. All satellite indicators + conflict in one analysis
4. Pre-trends tests for all outcomes including conflict

Buffer Zones (unified):
- 200m-600m (very_tight)
- 300m-800m (tight)
- 400m-1km (moderate)
- 500m-1.5km (baseline)
- 600m-1.5km (conservative)
- 750m-2km (wide)

Usage:
    python analysis/pse/02_did_pretrends_unified.py
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import statsmodels.formula.api as smf
import warnings
from datetime import datetime
from collections import defaultdict

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

# ACLED paths to try
ACLED_PATHS = [
    DATA_DIR / "pse" / "external" / "acled_conflict.csv",  # Your actual file
    DATA_DIR / "pse" / "conflict" / "acled_pse.csv",
    DATA_DIR / "pse" / "acled_pse.csv",
    DATA_DIR / "pse" / "external" / "acled_pse.csv",
    DATA_DIR / "pse" / "conflict" / "acled_gaza.csv",
    DATA_DIR / "pse" / "acled.csv",
]

OUTPUT_DIR = Path("outputs/pse")
FIGURES_DIR = OUTPUT_DIR / "figures" / "pretrends_unified"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

CONFLICT_START = pd.Timestamp("2023-10-07")
DEG_PER_M = 1 / 111000

# =============================================================================
# UNIFIED BUFFER SPECIFICATIONS
# =============================================================================
DONUT_SPECS = [
    (200, 600, "200m-600m", "very_tight"),
    (300, 800, "300m-800m", "tight"),
    (400, 1000, "400m-1km", "moderate"),
    (500, 1500, "500m-1.5km", "baseline"),
    (600, 1500, "600m-1.5km", "conservative"),
    (750, 2000, "750m-2km", "wide"),
]

# Conflict buffer for exposure calculation
CONFLICT_BUFFER_M = 500

# All outcomes including conflict
OUTCOME_LABELS = {
    'sar_vh': 'SAR VH (Structures)',
    'bsi': 'BSI (Bare Soil)',
    'viirs': 'VIIRS (Nightlights)',
    'ndbi': 'NDBI (Built-up)',
    'ndvi': 'NDVI (Vegetation)',
    'conflict_events': 'Conflict Events',
}

# =============================================================================
# DATA LOADING
# =============================================================================

def load_acled():
    """Load ACLED conflict data."""
    for path in ACLED_PATHS:
        if path.exists():
            print(f"  Found ACLED: {path}")
            df = pd.read_csv(path)
            
            # Standardize columns
            col_map = {
                'event_date': 'date', 'EVENT_DATE': 'date', 'Date': 'date',
                'latitude': 'lat', 'LATITUDE': 'lat', 'Latitude': 'lat',
                'longitude': 'lon', 'LONGITUDE': 'lon', 'Longitude': 'lon',
                'fatalities': 'fatalities', 'FATALITIES': 'fatalities',
            }
            for old, new in col_map.items():
                if old in df.columns and new not in df.columns:
                    df[new] = df[old]
            
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            return df
    
    print("  ⚠ No ACLED file found")
    return None


def load_treated_sites():
    """Load UNDP intervention sites."""
    sites = pd.read_csv(SITES_CSV)
    sites['group'] = 'Treated'
    return sites


def generate_donut_controls(gdf_boundary, gdf_treated, inner_m, outer_m, n_controls=None, seed=42):
    """Generate control points in donut zone."""
    from shapely.geometry import Point
    
    np.random.seed(seed)
    
    if n_controls is None:
        n_controls = len(gdf_treated)
    
    inner_deg = inner_m * DEG_PER_M
    outer_deg = outer_m * DEG_PER_M
    
    treated_coords = np.array([[p.x, p.y] for p in gdf_treated.geometry])
    tree = cKDTree(treated_coords)
    
    control_points = []
    minx, miny, maxx, maxy = gdf_boundary.total_bounds
    
    attempts = 0
    max_attempts = 500000
    
    while len(control_points) < n_controls and attempts < max_attempts:
        attempts += 1
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        
        pnt = Point(x, y)
        if not gdf_boundary.contains(pnt).any():
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


# =============================================================================
# TIME SERIES EXTRACTION
# =============================================================================

def extract_satellite_time_series(sites, rasters):
    """Extract satellite raster values for all sites over time."""
    records = []
    
    for indicator, dates in rasters.items():
        for date, path in sorted(dates.items()):
            try:
                with rasterio.open(path) as src:
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
                                'intervention_type': site.get('intervention_type', 'Unknown'),
                                'date': date,
                                'indicator': indicator,
                                'value': float(val),
                            })
                        except:
                            continue
            except:
                continue
    
    return pd.DataFrame(records)


def extract_conflict_time_series(sites, acled_df, buffer_m=CONFLICT_BUFFER_M):
    """Extract monthly conflict events for each site."""
    if acled_df is None or acled_df.empty:
        return pd.DataFrame()
    
    buffer_deg = buffer_m * DEG_PER_M
    
    # Create monthly data
    acled_df['month'] = acled_df['date'].dt.to_period('M')
    all_months = sorted(acled_df['month'].unique())
    
    records = []
    
    for _, site in sites.iterrows():
        site_lon, site_lat = site['longitude'], site['latitude']
        
        for month in all_months:
            month_events = acled_df[acled_df['month'] == month]
            
            if len(month_events) == 0:
                n_events = 0
            else:
                # Calculate distances
                distances = np.sqrt(
                    (month_events['lon'].values - site_lon)**2 +
                    (month_events['lat'].values - site_lat)**2
                )
                n_events = np.sum(distances <= buffer_deg)
            
            records.append({
                'site_id': site['site_id'],
                'group': site['group'],
                'intervention_type': site.get('intervention_type', 'Unknown'),
                'date': month.to_timestamp(),
                'indicator': 'conflict_events',
                'value': n_events,
            })
    
    return pd.DataFrame(records)


# =============================================================================
# PRE-TRENDS ANALYSIS
# =============================================================================

def test_parallel_trends(df, indicator):
    """Test parallel pre-trends using interaction term."""
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
    """Compute mean values by group and date."""
    subset = df[df['indicator'] == indicator].copy()
    
    if subset.empty:
        return None, None
    
    treated = subset[subset['group'] == 'Treated'].groupby('date')['value'].agg(['mean', 'std', 'count']).reset_index()
    treated['se'] = treated['std'] / np.sqrt(treated['count'])
    treated['group'] = 'Treated'
    
    control = subset[subset['group'] == 'Control'].groupby('date')['value'].agg(['mean', 'std', 'count']).reset_index()
    control['se'] = control['std'] / np.sqrt(control['count'])
    control['group'] = 'Control'
    
    return treated, control


# =============================================================================
# DID REGRESSION
# =============================================================================

def run_did(df, indicator):
    """Run difference-in-differences regression."""
    subset = df[df['indicator'] == indicator].copy()
    
    if subset.empty or len(subset) < 100:
        return None
    
    subset['post'] = (subset['date'] >= CONFLICT_START).astype(int)
    subset['treated'] = (subset['group'] == 'Treated').astype(int)
    subset['post_treated'] = subset['post'] * subset['treated']
    
    try:
        model = smf.ols('value ~ post * treated', data=subset).fit(cov_type='HC3')
        
        coef = model.params.get('post:treated', np.nan)
        se = model.bse.get('post:treated', np.nan)
        pval = model.pvalues.get('post:treated', np.nan)
        
        # Stars
        if pval < 0.01:
            stars = '***'
        elif pval < 0.05:
            stars = '**'
        elif pval < 0.10:
            stars = '*'
        else:
            stars = ''
        
        return {
            'coef': coef,
            'se': se,
            'pval': pval,
            'stars': stars,
            'ci_lower': coef - 1.96 * se,
            'ci_upper': coef + 1.96 * se,
        }
    except:
        return None


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_pretrends_single(df, indicator, spec_label, passed, pval, output_path):
    """Create pre-trends plot for one specification × outcome."""
    treated, control = compute_group_means(df, indicator)
    
    if treated is None or control is None:
        return
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    outcome_label = OUTCOME_LABELS.get(indicator, indicator.upper())
    
    # Background shading
    ax.axvspan(treated['date'].min(), CONFLICT_START, alpha=0.1, color='#3498db')
    ax.axvspan(CONFLICT_START, treated['date'].max(), alpha=0.1, color='#e74c3c')
    
    # Conflict line
    ax.axvline(CONFLICT_START, color='#c0392b', linewidth=2.5, linestyle='--', label='Oct 7, 2023')
    
    # Treated
    ax.fill_between(treated['date'],
                    treated['mean'] - 1.96*treated['se'],
                    treated['mean'] + 1.96*treated['se'],
                    alpha=0.2, color='#e74c3c')
    ax.plot(treated['date'], treated['mean'], 'o-', color='#e74c3c',
            linewidth=2, markersize=5, label='Treated (UNDP sites)')
    
    # Control
    ax.fill_between(control['date'],
                    control['mean'] - 1.96*control['se'],
                    control['mean'] + 1.96*control['se'],
                    alpha=0.2, color='#3498db')
    ax.plot(control['date'], control['mean'], 's--', color='#3498db',
            linewidth=2, markersize=5, label=f'Control ({spec_label})')
    
    # Pre-trends result
    if passed is not None:
        if passed:
            status = f"✓ Parallel trends HOLD (p={pval:.3f})"
            color = '#27ae60'
        else:
            status = f"⚠ Parallel trends VIOLATED (p={pval:.3f})"
            color = '#e74c3c'
        
        ax.annotate(status, xy=(0.02, 0.98), xycoords='axes fraction',
                   fontsize=12, ha='left', va='top', fontweight='bold', color=color,
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                            edgecolor=color, alpha=0.95))
    
    ax.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax.set_ylabel(outcome_label, fontsize=11, fontweight='bold')
    ax.set_title(f'Pre-Trends: {outcome_label}\nControl Group: {spec_label}',
                fontsize=13, fontweight='bold')
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()
    
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_pretrends_heatmap(results_df, output_path):
    """Create heatmap of pre-trends test results."""
    # Pivot to matrix
    specs = [s[2] for s in DONUT_SPECS]
    outcomes = list(OUTCOME_LABELS.keys())
    
    matrix = np.full((len(outcomes), len(specs)), np.nan)
    pval_matrix = np.empty((len(outcomes), len(specs)), dtype=object)
    
    for i, outcome in enumerate(outcomes):
        for j, spec_label in enumerate(specs):
            row = results_df[(results_df['indicator'] == outcome) & 
                            (results_df['spec_label'] == spec_label)]
            if not row.empty:
                passed = row['pretrends_passed'].values[0]
                pval = row['pretrends_pval'].values[0]
                if pd.notna(passed):
                    matrix[i, j] = 1 if passed else 0
                    pval_matrix[i, j] = f"{pval:.2f}" if pd.notna(pval) else "N/A"
                else:
                    pval_matrix[i, j] = "N/A"
            else:
                pval_matrix[i, j] = "N/A"
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    cmap = sns.diverging_palette(10, 130, as_cmap=True)
    
    sns.heatmap(matrix, ax=ax, annot=pval_matrix, fmt='', cmap=cmap,
                xticklabels=specs,
                yticklabels=[OUTCOME_LABELS[o] for o in outcomes],
                cbar_kws={'label': 'Parallel Trends (1=Pass, 0=Fail)'},
                linewidths=1, linecolor='white',
                vmin=0, vmax=1, center=0.5)
    
    ax.set_title('Pre-Trends Test Results: All Outcomes × All Specifications\n'
                 '(Green=Pass [p>0.10], Red=Fail [p<0.10])',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Donut Control Specification', fontsize=12, fontweight='bold')
    ax.set_ylabel('Outcome', fontsize=12, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_did_results(results_df, output_path):
    """Create bar chart of DiD results across specifications."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    outcomes = list(OUTCOME_LABELS.keys())
    
    for i, outcome in enumerate(outcomes):
        ax = axes[i]
        
        subset = results_df[results_df['indicator'] == outcome].sort_values('inner_m')
        
        if subset.empty:
            ax.set_visible(False)
            continue
        
        x = np.arange(len(subset))
        colors = ['#27ae60' if p else '#e74c3c' for p in subset['pretrends_passed']]
        
        bars = ax.bar(x, subset['did_coef'], yerr=1.96*subset['did_se'],
                     color=colors, edgecolor='black', linewidth=1.5, capsize=5, alpha=0.8)
        
        ax.axhline(0, color='black', linewidth=1)
        
        # Add significance stars
        for j, (_, row) in enumerate(subset.iterrows()):
            if pd.notna(row['did_stars']) and row['did_stars']:
                ax.annotate(row['did_stars'], xy=(j, row['did_coef']),
                           xytext=(0, 10 if row['did_coef'] >= 0 else -15),
                           textcoords='offset points', ha='center', fontsize=10, fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels(subset['spec_label'], rotation=45, ha='right')
        ax.set_ylabel('DiD Coefficient', fontsize=11)
        ax.set_title(OUTCOME_LABELS[outcome], fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Legend for first plot
        if i == 0:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#27ae60', edgecolor='black', label='Pre-trends PASS'),
                Patch(facecolor='#e74c3c', edgecolor='black', label='Pre-trends FAIL'),
            ]
            ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    fig.suptitle('DiD Results Across Specifications\n(Color = Pre-trends test result)',
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print(" UNIFIED DID + PRE-TRENDS ANALYSIS")
    print(" (Consistent Buffer Zones + Conflict as Outcome)")
    print("=" * 70)
    
    if not HAS_DEPS:
        print("\n⚠ Missing dependencies")
        return
    
    # Load data
    print("\n--- Loading Data ---")
    
    treated_sites = load_treated_sites()
    print(f"  Treated sites: {len(treated_sites)}")
    
    acled_df = load_acled()
    if acled_df is not None:
        print(f"  ACLED events: {len(acled_df)}")
    
    boundaries = gpd.read_file(BOUNDARIES_FILE)
    treated_gdf = gpd.GeoDataFrame(
        treated_sites,
        geometry=gpd.points_from_xy(treated_sites.longitude, treated_sites.latitude),
        crs="EPSG:4326"
    )
    
    rasters = find_rasters()
    print(f"  Raster indicators: {list(rasters.keys())}")
    
    # Store all results
    all_results = []
    all_pretrends_data = {}
    
    # Process each specification
    for inner, outer, spec_label, spec_key in DONUT_SPECS:
        print(f"\n{'='*70}")
        print(f" SPECIFICATION: {spec_label}")
        print("=" * 70)
        
        # Generate controls
        controls = generate_donut_controls(boundaries, treated_gdf, inner, outer)
        
        if len(controls) < len(treated_sites) * 0.5:
            print(f"  ⚠ Insufficient controls ({len(controls)}), skipping")
            continue
        
        print(f"  Controls generated: {len(controls)}")
        
        # Combine sites
        all_sites = pd.concat([treated_sites, controls], ignore_index=True)
        
        # Extract satellite time series
        print("  Extracting satellite data...")
        satellite_df = extract_satellite_time_series(all_sites, rasters)
        print(f"    Satellite observations: {len(satellite_df):,}")
        
        # Extract conflict time series
        print("  Extracting conflict data...")
        conflict_df = extract_conflict_time_series(all_sites, acled_df)
        print(f"    Conflict observations: {len(conflict_df):,}")
        
        # Combine all data
        panel_df = pd.concat([satellite_df, conflict_df], ignore_index=True)
        
        # Process each indicator
        print("\n  Results:")
        print(f"  {'Indicator':<20} {'DiD Coef':>12} {'Pre-trends':>12}")
        print("  " + "-" * 50)
        
        for indicator in OUTCOME_LABELS.keys():
            # Pre-trends test
            passed, pval, trend_coef = test_parallel_trends(panel_df, indicator)
            
            # DiD regression
            did_result = run_did(panel_df, indicator)
            
            # Store results
            row = {
                'spec_label': spec_label,
                'spec_key': spec_key,
                'inner_m': inner,
                'outer_m': outer,
                'indicator': indicator,
                'pretrends_passed': passed,
                'pretrends_pval': pval,
                'pretrends_coef': trend_coef,
            }
            
            if did_result:
                row.update({
                    'did_coef': did_result['coef'],
                    'did_se': did_result['se'],
                    'did_pval': did_result['pval'],
                    'did_stars': did_result['stars'],
                    'did_ci_lower': did_result['ci_lower'],
                    'did_ci_upper': did_result['ci_upper'],
                })
            
            all_results.append(row)
            
            # Store pre-trends data for plotting
            treated_agg, control_agg = compute_group_means(panel_df, indicator)
            all_pretrends_data[(spec_key, indicator)] = {
                'treated': treated_agg,
                'control': control_agg,
                'passed': passed,
                'pval': pval,
            }
            
            # Print summary
            did_coef = did_result['coef'] if did_result else np.nan
            did_stars = did_result['stars'] if did_result else ''
            pt_status = "PASS ✓" if passed else "FAIL ⚠" if passed is not None else "N/A"
            
            print(f"  {indicator:<20} {did_coef:>+10.4f}{did_stars:<3} {pt_status:>12}")
            
            # Create individual pre-trends plot
            plot_pretrends_single(panel_df, indicator, spec_label, passed, pval,
                                 FIGURES_DIR / f"pretrends_{indicator}_{spec_key}.png")
    
    # Create results dataframe
    results_df = pd.DataFrame(all_results)
    
    # Save to CSV
    results_df.to_csv(TABLES_DIR / "did_pretrends_unified.csv", index=False)
    print(f"\n✓ Saved: did_pretrends_unified.csv")
    
    # Create summary visualizations
    print(f"\n{'='*70}")
    print(" CREATING SUMMARY VISUALIZATIONS")
    print("=" * 70)
    
    # Heatmap
    print("  Creating pre-trends heatmap...")
    plot_pretrends_heatmap(results_df, FIGURES_DIR / "pretrends_heatmap_all.png")
    print("  ✓ Saved: pretrends_heatmap_all.png")
    
    # DiD results bar chart
    print("  Creating DiD results chart...")
    plot_did_results(results_df, FIGURES_DIR / "did_results_all.png")
    print("  ✓ Saved: did_results_all.png")
    
    # Summary tables
    print("\n" + "=" * 70)
    print(" FINAL SUMMARY")
    print("=" * 70)
    
    print("\n  PRE-TRENDS PASS RATES BY OUTCOME:")
    for indicator in OUTCOME_LABELS.keys():
        ind_results = results_df[results_df['indicator'] == indicator]
        passed = ind_results['pretrends_passed'].sum()
        total = ind_results['pretrends_passed'].notna().sum()
        pct = passed / total * 100 if total > 0 else 0
        status = "★ CREDIBLE" if pct >= 70 else ""
        print(f"    {OUTCOME_LABELS[indicator]:<25}: {passed}/{total} ({pct:.0f}%) {status}")
    
    print("\n  PRE-TRENDS PASS RATES BY SPECIFICATION:")
    for spec_label in [s[2] for s in DONUT_SPECS]:
        spec_results = results_df[results_df['spec_label'] == spec_label]
        passed = spec_results['pretrends_passed'].sum()
        total = spec_results['pretrends_passed'].notna().sum()
        pct = passed / total * 100 if total > 0 else 0
        print(f"    {spec_label:<20}: {passed}/{total} ({pct:.0f}%)")
    
    print(f"""
    
  OUTPUT FILES:
  ─────────────
  Tables:
    - {TABLES_DIR}/did_pretrends_unified.csv
  
  Figures:
    - {FIGURES_DIR}/pretrends_heatmap_all.png
    - {FIGURES_DIR}/did_results_all.png
    - {FIGURES_DIR}/pretrends_[indicator]_[spec].png (individual plots)
    """)
    
    print("=" * 70)
    print(" ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
