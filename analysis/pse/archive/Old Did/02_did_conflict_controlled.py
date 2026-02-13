#!/usr/bin/env python3
"""
analysis/pse/02_did_conflict_controlled.py
==========================================
DiD Analysis with Conflict Controls + Conflict Pre-Trends Validation.

This script:
1. Calculates conflict exposure for each site (treatment and control)
2. Tests whether treatment sites have more conflict than controls (pre-trends for conflict)
3. Runs DiD WITH conflict controls to isolate true treatment effects
4. Compares results with and without conflict controls

Key hypothesis:
- UNDP targeted heavily damaged areas → more conflict at treatment sites
- Without conflict controls: VIIRS is negative (selection effect)
- WITH conflict controls: VIIRS should be positive or less negative (true effect)

Usage:
    python analysis/pse/02_did_conflict_controlled.py
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
ACLED_FILE = DATA_DIR / "pse" / "conflict" / "acled_pse.csv"
BOUNDARIES_FILE = DATA_DIR / "pse" / "boundaries" / "gaza_admin2.geojson"

OUTPUT_DIR = Path("outputs/pse")
FIGURES_DIR = OUTPUT_DIR / "figures" / "conflict_analysis"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

CONFLICT_START = pd.Timestamp("2023-10-07")
DEG_PER_M = 1 / 111000

# Donut specifications to test
DONUT_SPECS = [
    (200, 600, "200m-600m", "very_tight"),
    (300, 800, "300m-800m", "tight"),
    (400, 1000, "400m-1km", "moderate"),
    (500, 1500, "500m-1.5km", "baseline"),
    (600, 1500, "600m-1.5km", "conservative"),
    (750, 2000, "750m-2km", "wide"),
]

# Conflict buffer for exposure calculation (meters)
CONFLICT_BUFFER_M = 500

OUTCOME_LABELS = {
    'sar_vh': 'SAR VH (Structures)',
    'bsi': 'BSI (Bare Soil)',
    'viirs': 'VIIRS (Nightlights)',
    'ndbi': 'NDBI (Built-up)',
    'ndvi': 'NDVI (Vegetation)',
}

# =============================================================================
# DATA LOADING
# =============================================================================

def load_acled_data():
    """Load ACLED conflict events."""
    if not ACLED_FILE.exists():
        # Try alternative paths
        alt_paths = [
            DATA_DIR / "pse" / "acled_pse.csv",
            DATA_DIR / "pse" / "external" / "acled_pse.csv",
            DATA_DIR / "pse" / "conflict" / "acled_gaza.csv",
        ]
        for p in alt_paths:
            if p.exists():
                return pd.read_csv(p)
        return None
    
    return pd.read_csv(ACLED_FILE)


def load_treated_sites():
    """Load UNDP intervention sites."""
    sites = pd.read_csv(SITES_CSV)
    sites['group'] = 'Treated'
    return sites


def generate_donut_controls(gdf_boundary, gdf_treated, inner_m, outer_m, n_controls=None):
    """Generate control points in the donut zone."""
    from shapely.geometry import Point
    
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


# =============================================================================
# CONFLICT EXPOSURE CALCULATION
# =============================================================================

def calculate_conflict_exposure(sites_df, acled_df, buffer_m=CONFLICT_BUFFER_M):
    """
    Calculate conflict exposure for each site by month.
    
    Returns DataFrame with:
    - site_id
    - month
    - conflict_events (count within buffer)
    - conflict_fatalities (sum within buffer)
    """
    if acled_df is None or acled_df.empty:
        print("  ⚠ No ACLED data available")
        return None
    
    # Parse ACLED dates
    date_cols = ['event_date', 'date', 'EVENT_DATE', 'Date']
    date_col = None
    for col in date_cols:
        if col in acled_df.columns:
            date_col = col
            break
    
    if date_col is None:
        print(f"  ⚠ No date column found in ACLED. Columns: {list(acled_df.columns)}")
        return None
    
    acled_df['date'] = pd.to_datetime(acled_df[date_col], errors='coerce')
    acled_df = acled_df.dropna(subset=['date'])
    
    # Get coordinates
    lat_cols = ['latitude', 'lat', 'LATITUDE', 'Latitude']
    lon_cols = ['longitude', 'lon', 'LONGITUDE', 'Longitude']
    
    lat_col = next((c for c in lat_cols if c in acled_df.columns), None)
    lon_col = next((c for c in lon_cols if c in acled_df.columns), None)
    
    if lat_col is None or lon_col is None:
        print(f"  ⚠ No coordinate columns found in ACLED")
        return None
    
    # Get fatalities column
    fat_cols = ['fatalities', 'FATALITIES', 'Fatalities', 'deaths']
    fat_col = next((c for c in fat_cols if c in acled_df.columns), None)
    
    # Create monthly periods
    acled_df['month'] = acled_df['date'].dt.to_period('M')
    
    # Build KDTree for conflict events
    buffer_deg = buffer_m * DEG_PER_M
    
    # Group ACLED by month
    monthly_groups = acled_df.groupby('month')
    
    records = []
    
    for _, site in sites_df.iterrows():
        site_lon, site_lat = site['longitude'], site['latitude']
        
        for month, month_events in monthly_groups:
            # Calculate distance to all events in this month
            event_coords = month_events[[lon_col, lat_col]].values
            
            if len(event_coords) == 0:
                continue
            
            # Distance in degrees (approximate)
            distances = np.sqrt(
                (event_coords[:, 0] - site_lon)**2 + 
                (event_coords[:, 1] - site_lat)**2
            )
            
            # Count events within buffer
            within_buffer = distances <= buffer_deg
            n_events = np.sum(within_buffer)
            
            # Sum fatalities within buffer
            if fat_col and n_events > 0:
                fatalities = month_events.loc[within_buffer, fat_col].sum()
            else:
                fatalities = 0
            
            records.append({
                'site_id': site['site_id'],
                'group': site['group'],
                'month': month,
                'date': month.to_timestamp(),
                'conflict_events': n_events,
                'conflict_fatalities': fatalities,
            })
    
    return pd.DataFrame(records)


def calculate_cumulative_conflict(conflict_df):
    """Calculate cumulative conflict exposure up to each month."""
    if conflict_df is None:
        return None
    
    conflict_df = conflict_df.sort_values(['site_id', 'date'])
    
    # Cumulative sum by site
    conflict_df['cum_events'] = conflict_df.groupby('site_id')['conflict_events'].cumsum()
    conflict_df['cum_fatalities'] = conflict_df.groupby('site_id')['conflict_fatalities'].cumsum()
    
    return conflict_df


# =============================================================================
# PRE-TRENDS ANALYSIS FOR CONFLICT
# =============================================================================

def test_conflict_pretrends(conflict_df):
    """
    Test whether treatment and control groups have parallel trends in CONFLICT exposure.
    
    If treatment sites have systematically MORE conflict, this explains the negative VIIRS.
    """
    if conflict_df is None:
        return None, None, None
    
    # Pre-conflict period only
    pre_data = conflict_df[conflict_df['date'] < CONFLICT_START].copy()
    
    if len(pre_data) < 50:
        return None, None, None
    
    pre_data['month_idx'] = pre_data['date'].dt.year * 12 + pre_data['date'].dt.month
    pre_data['is_treated'] = (pre_data['group'] == 'Treated').astype(int)
    
    try:
        # Test for differential trends
        model = smf.ols('conflict_events ~ month_idx * is_treated', data=pre_data).fit()
        
        # Also test level difference
        level_model = smf.ols('conflict_events ~ is_treated', data=pre_data).fit()
        
        results = {
            'trend_interaction_coef': model.params.get('month_idx:is_treated', np.nan),
            'trend_interaction_pval': model.pvalues.get('month_idx:is_treated', np.nan),
            'level_diff_coef': level_model.params.get('is_treated', np.nan),
            'level_diff_pval': level_model.pvalues.get('is_treated', np.nan),
        }
        
        return results
        
    except Exception as e:
        print(f"    Error in conflict pre-trends test: {e}")
        return None


def compute_conflict_means(conflict_df):
    """Compute mean conflict by group and date."""
    if conflict_df is None:
        return None, None
    
    treated = conflict_df[conflict_df['group'] == 'Treated'].groupby('date').agg({
        'conflict_events': ['mean', 'std', 'count']
    }).reset_index()
    treated.columns = ['date', 'mean', 'std', 'count']
    treated['se'] = treated['std'] / np.sqrt(treated['count'])
    treated['group'] = 'Treated'
    
    control = conflict_df[conflict_df['group'] == 'Control'].groupby('date').agg({
        'conflict_events': ['mean', 'std', 'count']
    }).reset_index()
    control.columns = ['date', 'mean', 'std', 'count']
    control['se'] = control['std'] / np.sqrt(control['count'])
    control['group'] = 'Control'
    
    return treated, control


# =============================================================================
# DID WITH CONFLICT CONTROLS
# =============================================================================

def extract_time_series(sites, rasters):
    """Extract raster values for all sites over time."""
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
                                'type': site.get('intervention_type', 'Unknown'),
                                'date': date,
                                'indicator': indicator,
                                'value': float(val),
                            })
                        except:
                            continue
            except:
                continue
    
    return pd.DataFrame(records)


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


def run_did_with_conflict_control(panel_df, conflict_df, indicator):
    """
    Run DiD with and without conflict controls.
    
    Models:
    1. Basic: value ~ post × treated
    2. With conflict: value ~ post × treated + conflict_events
    3. With cumulative conflict: value ~ post × treated + cum_events
    """
    if panel_df is None or panel_df.empty:
        return None
    
    # Filter to indicator
    df = panel_df[panel_df['indicator'] == indicator].copy()
    
    if df.empty:
        return None
    
    # Create DiD variables
    df['post'] = (df['date'] >= CONFLICT_START).astype(int)
    df['treated'] = (df['group'] == 'Treated').astype(int)
    df['post_treated'] = df['post'] * df['treated']
    
    # Merge conflict data
    if conflict_df is not None:
        df['month'] = df['date'].dt.to_period('M')
        conflict_monthly = conflict_df.copy()
        conflict_monthly['month'] = conflict_monthly['date'].dt.to_period('M')
        
        # Aggregate conflict to site-month level
        conflict_agg = conflict_monthly.groupby(['site_id', 'month']).agg({
            'conflict_events': 'sum',
            'conflict_fatalities': 'sum',
            'cum_events': 'last',
            'cum_fatalities': 'last',
        }).reset_index()
        
        df = df.merge(conflict_agg, on=['site_id', 'month'], how='left')
        df['conflict_events'] = df['conflict_events'].fillna(0)
        df['cum_events'] = df['cum_events'].fillna(0)
        df['log_conflict'] = np.log1p(df['conflict_events'])
        df['log_cum_conflict'] = np.log1p(df['cum_events'])
    
    results = {}
    
    # Model 1: Basic DiD (no conflict control)
    try:
        model1 = smf.ols('value ~ post * treated', data=df).fit(cov_type='HC3')
        results['basic'] = {
            'coef': model1.params.get('post:treated', np.nan),
            'se': model1.bse.get('post:treated', np.nan),
            'pval': model1.pvalues.get('post:treated', np.nan),
        }
    except:
        results['basic'] = None
    
    # Model 2: With conflict events control
    if conflict_df is not None and 'conflict_events' in df.columns:
        try:
            model2 = smf.ols('value ~ post * treated + conflict_events', data=df).fit(cov_type='HC3')
            results['with_conflict'] = {
                'coef': model2.params.get('post:treated', np.nan),
                'se': model2.bse.get('post:treated', np.nan),
                'pval': model2.pvalues.get('post:treated', np.nan),
                'conflict_coef': model2.params.get('conflict_events', np.nan),
                'conflict_pval': model2.pvalues.get('conflict_events', np.nan),
            }
        except:
            results['with_conflict'] = None
        
        # Model 3: With cumulative conflict
        try:
            model3 = smf.ols('value ~ post * treated + cum_events', data=df).fit(cov_type='HC3')
            results['with_cum_conflict'] = {
                'coef': model3.params.get('post:treated', np.nan),
                'se': model3.bse.get('post:treated', np.nan),
                'pval': model3.pvalues.get('post:treated', np.nan),
                'conflict_coef': model3.params.get('cum_events', np.nan),
                'conflict_pval': model3.pvalues.get('cum_events', np.nan),
            }
        except:
            results['with_cum_conflict'] = None
        
        # Model 4: With log conflict (to handle outliers)
        try:
            model4 = smf.ols('value ~ post * treated + log_cum_conflict', data=df).fit(cov_type='HC3')
            results['with_log_conflict'] = {
                'coef': model4.params.get('post:treated', np.nan),
                'se': model4.bse.get('post:treated', np.nan),
                'pval': model4.pvalues.get('post:treated', np.nan),
                'conflict_coef': model4.params.get('log_cum_conflict', np.nan),
                'conflict_pval': model4.pvalues.get('log_cum_conflict', np.nan),
            }
        except:
            results['with_log_conflict'] = None
    
    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_conflict_pretrends(conflict_df, spec_label, output_path):
    """
    Create pre-trends plot for conflict exposure.
    Shows whether treatment sites have more conflict than controls.
    """
    treated, control = compute_conflict_means(conflict_df)
    
    if treated is None or control is None:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Panel 1: Time series
    ax1 = axes[0]
    
    # Background shading
    ax1.axvspan(treated['date'].min(), CONFLICT_START, alpha=0.1, color='#3498db')
    ax1.axvspan(CONFLICT_START, treated['date'].max(), alpha=0.1, color='#e74c3c')
    ax1.axvline(CONFLICT_START, color='#c0392b', linewidth=2.5, linestyle='--', label='Conflict Start')
    
    # Treated
    ax1.fill_between(treated['date'],
                     treated['mean'] - 1.96*treated['se'],
                     treated['mean'] + 1.96*treated['se'],
                     alpha=0.2, color='#e74c3c')
    ax1.plot(treated['date'], treated['mean'], 'o-', color='#e74c3c',
             linewidth=2, markersize=5, label='Treated (UNDP sites)')
    
    # Control
    ax1.fill_between(control['date'],
                     control['mean'] - 1.96*control['se'],
                     control['mean'] + 1.96*control['se'],
                     alpha=0.2, color='#3498db')
    ax1.plot(control['date'], control['mean'], 's--', color='#3498db',
             linewidth=2, markersize=5, label=f'Control ({spec_label})')
    
    ax1.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Conflict Events (within 500m)', fontsize=11, fontweight='bold')
    ax1.set_title(f'Conflict Exposure: Treatment vs Control\nControl Group: {spec_label}',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    # Panel 2: Pre-conflict comparison (bar chart)
    ax2 = axes[1]
    
    pre_treated = treated[treated['date'] < CONFLICT_START]['mean'].mean()
    pre_control = control[control['date'] < CONFLICT_START]['mean'].mean()
    
    post_treated = treated[treated['date'] >= CONFLICT_START]['mean'].mean()
    post_control = control[control['date'] >= CONFLICT_START]['mean'].mean()
    
    x = np.arange(2)
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, [pre_treated, post_treated], width, 
                    label='Treated', color='#e74c3c', edgecolor='black')
    bars2 = ax2.bar(x + width/2, [pre_control, post_control], width,
                    label='Control', color='#3498db', edgecolor='black')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Pre-Conflict', 'Post-Conflict'])
    ax2.set_ylabel('Avg Conflict Events/Month', fontsize=11, fontweight='bold')
    ax2.set_title('Conflict Exposure by Period', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', fontsize=10, fontweight='bold')
    
    # Add difference annotation
    diff_pre = pre_treated - pre_control
    diff_post = post_treated - post_control
    
    ax2.annotate(f'Diff: +{diff_pre:.2f}' if diff_pre > 0 else f'Diff: {diff_pre:.2f}',
                xy=(0, max(pre_treated, pre_control) + 0.5),
                ha='center', fontsize=10, color='#8e44ad', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_conflict_pretrends_comparison(all_conflict_results, output_path):
    """
    Multi-panel comparison of conflict pre-trends across all donut specs.
    """
    n_specs = len(DONUT_SPECS)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (inner, outer, spec_label, spec_key) in enumerate(DONUT_SPECS):
        if i >= len(axes):
            break
        
        ax = axes[i]
        
        if spec_key not in all_conflict_results:
            ax.set_visible(False)
            continue
        
        result = all_conflict_results[spec_key]
        treated = result['treated']
        control = result['control']
        
        if treated is None or control is None:
            ax.set_visible(False)
            continue
        
        # Pre-conflict only
        pre_treat = treated[treated['date'] < CONFLICT_START]
        pre_ctrl = control[control['date'] < CONFLICT_START]
        
        if pre_treat.empty or pre_ctrl.empty:
            ax.set_visible(False)
            continue
        
        # Plot
        ax.fill_between(pre_treat['date'],
                       pre_treat['mean'] - 1.96*pre_treat['se'],
                       pre_treat['mean'] + 1.96*pre_treat['se'],
                       alpha=0.25, color='#e74c3c')
        ax.plot(pre_treat['date'], pre_treat['mean'], 'o-', color='#e74c3c',
               linewidth=2, markersize=4, label='Treated')
        
        ax.fill_between(pre_ctrl['date'],
                       pre_ctrl['mean'] - 1.96*pre_ctrl['se'],
                       pre_ctrl['mean'] + 1.96*pre_ctrl['se'],
                       alpha=0.25, color='#3498db')
        ax.plot(pre_ctrl['date'], pre_ctrl['mean'], 's--', color='#3498db',
               linewidth=2, markersize=4, label='Control')
        
        # Level difference test
        if 'test_results' in result and result['test_results']:
            test = result['test_results']
            level_diff = test.get('level_diff_coef', 0)
            level_pval = test.get('level_diff_pval', 1)
            
            if level_diff > 0:
                status = f"Treated +{level_diff:.2f} events (p={level_pval:.3f})"
                color = '#e74c3c'
            else:
                status = f"Treated {level_diff:.2f} events (p={level_pval:.3f})"
                color = '#3498db'
            
            ax.annotate(status, xy=(0.05, 0.95), xycoords='axes fraction',
                       fontsize=9, ha='left', va='top', fontweight='bold', color=color,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor=color, alpha=0.9))
        
        ax.set_title(spec_label, fontsize=12, fontweight='bold')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        ax.grid(True, alpha=0.3, linestyle='--')
        
        if i == 0:
            ax.legend(fontsize=9, loc='lower right')
        
        if i % 3 == 0:
            ax.set_ylabel('Conflict Events', fontsize=10)
    
    # Hide unused
    for j in range(len(DONUT_SPECS), len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle('Conflict Exposure: Pre-Trends Comparison Across Donut Specifications\n'
                 '(Do UNDP sites have MORE conflict than controls?)',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_did_with_without_conflict(results_df, output_path):
    """
    Compare DiD results with and without conflict controls.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Panel 1: VIIRS specifically
    ax1 = axes[0]
    
    viirs = results_df[results_df['indicator'] == 'viirs']
    
    if not viirs.empty:
        specs = viirs['spec_label'].unique()
        x = np.arange(len(specs))
        width = 0.25
        
        basic = viirs['basic_coef'].values
        with_conf = viirs['conflict_coef'].values if 'conflict_coef' in viirs.columns else basic
        with_cum = viirs['cum_conflict_coef'].values if 'cum_conflict_coef' in viirs.columns else basic
        
        bars1 = ax1.bar(x - width, basic, width, label='No Control', color='#e74c3c', edgecolor='black')
        bars2 = ax1.bar(x, with_conf, width, label='+ Conflict Events', color='#f39c12', edgecolor='black')
        bars3 = ax1.bar(x + width, with_cum, width, label='+ Cumulative Conflict', color='#27ae60', edgecolor='black')
        
        ax1.axhline(0, color='black', linewidth=1)
        ax1.set_xticks(x)
        ax1.set_xticklabels(specs, rotation=45, ha='right')
        ax1.set_ylabel('DiD Coefficient (VIIRS)', fontsize=11, fontweight='bold')
        ax1.set_title('VIIRS Effect: With vs Without Conflict Controls',
                     fontsize=13, fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3, axis='y')
    
    # Panel 2: All indicators comparison (baseline spec)
    ax2 = axes[1]
    
    baseline = results_df[results_df['spec_key'] == 'baseline']
    
    if not baseline.empty:
        indicators = baseline['indicator'].unique()
        x = np.arange(len(indicators))
        width = 0.35
        
        basic = baseline['basic_coef'].values
        with_conf = baseline['cum_conflict_coef'].values if 'cum_conflict_coef' in baseline.columns else basic
        
        bars1 = ax2.bar(x - width/2, basic, width, label='No Control', color='#e74c3c', edgecolor='black')
        bars2 = ax2.bar(x + width/2, with_conf, width, label='+ Conflict', color='#27ae60', edgecolor='black')
        
        ax2.axhline(0, color='black', linewidth=1)
        ax2.set_xticks(x)
        ax2.set_xticklabels([OUTCOME_LABELS.get(i, i) for i in indicators], rotation=45, ha='right')
        ax2.set_ylabel('DiD Coefficient', fontsize=11, fontweight='bold')
        ax2.set_title('All Outcomes: Baseline Spec (500m-1.5km)\nWith vs Without Conflict Control',
                     fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print(" DID WITH CONFLICT CONTROLS + CONFLICT PRE-TRENDS VALIDATION")
    print("=" * 70)
    
    if not HAS_DEPS:
        print("\n⚠ Missing dependencies. Exiting.")
        return
    
    # Load data
    print("\n--- Loading Data ---")
    
    treated_sites = load_treated_sites()
    print(f"  Treated sites: {len(treated_sites)}")
    
    acled_df = load_acled_data()
    if acled_df is not None:
        print(f"  ACLED events: {len(acled_df)}")
    else:
        print("  ⚠ ACLED data not found - conflict analysis will be skipped")
    
    boundaries = gpd.read_file(BOUNDARIES_FILE)
    treated_gdf = gpd.GeoDataFrame(
        treated_sites,
        geometry=gpd.points_from_xy(treated_sites.longitude, treated_sites.latitude),
        crs="EPSG:4326"
    )
    
    rasters = find_rasters()
    print(f"  Rasters: {list(rasters.keys())}")
    
    # Store results
    all_conflict_results = {}
    all_did_results = []
    
    # Analyze each specification
    for inner, outer, spec_label, spec_key in DONUT_SPECS:
        print(f"\n{'='*70}")
        print(f" SPECIFICATION: {spec_label}")
        print("=" * 70)
        
        # Generate controls
        np.random.seed(42)  # For reproducibility
        controls = generate_donut_controls(boundaries, treated_gdf, inner, outer,
                                          n_controls=len(treated_sites))
        
        if len(controls) < len(treated_sites) * 0.5:
            print(f"  ⚠ Insufficient controls ({len(controls)}), skipping")
            continue
        
        print(f"  Controls generated: {len(controls)}")
        
        # Combine sites
        all_sites = pd.concat([treated_sites, controls], ignore_index=True)
        
        # Calculate conflict exposure
        print("\n  Calculating conflict exposure...")
        conflict_df = calculate_conflict_exposure(all_sites, acled_df)
        
        if conflict_df is not None:
            conflict_df = calculate_cumulative_conflict(conflict_df)
            
            # Test pre-trends for conflict
            print("  Testing conflict pre-trends...")
            test_results = test_conflict_pretrends(conflict_df)
            
            if test_results:
                level_diff = test_results['level_diff_coef']
                level_pval = test_results['level_diff_pval']
                
                if level_diff > 0:
                    print(f"    → Treated sites have +{level_diff:.2f} MORE conflict events (p={level_pval:.3f})")
                else:
                    print(f"    → Treated sites have {level_diff:.2f} conflict events (p={level_pval:.3f})")
            
            # Store for plotting
            treated_agg, control_agg = compute_conflict_means(conflict_df)
            all_conflict_results[spec_key] = {
                'treated': treated_agg,
                'control': control_agg,
                'test_results': test_results,
            }
            
            # Create individual plot
            plot_conflict_pretrends(conflict_df, spec_label, 
                                   FIGURES_DIR / f"conflict_pretrends_{spec_key}.png")
            print(f"  ✓ Saved: conflict_pretrends_{spec_key}.png")
        
        # Extract satellite data
        print("\n  Extracting satellite data...")
        panel_df = extract_time_series(all_sites, rasters)
        
        if panel_df.empty:
            print("    ⚠ No satellite data extracted")
            continue
        
        print(f"    Observations: {len(panel_df):,}")
        
        # Run DiD with and without conflict controls
        print("\n  Running DiD models...")
        
        for indicator in panel_df['indicator'].unique():
            results = run_did_with_conflict_control(panel_df, conflict_df, indicator)
            
            if results is None:
                continue
            
            row = {
                'spec_label': spec_label,
                'spec_key': spec_key,
                'inner_m': inner,
                'outer_m': outer,
                'indicator': indicator,
            }
            
            # Basic DiD
            if results.get('basic'):
                row['basic_coef'] = results['basic']['coef']
                row['basic_se'] = results['basic']['se']
                row['basic_pval'] = results['basic']['pval']
            
            # With conflict control
            if results.get('with_conflict'):
                row['conflict_coef'] = results['with_conflict']['coef']
                row['conflict_se'] = results['with_conflict']['se']
                row['conflict_pval'] = results['with_conflict']['pval']
                row['conflict_control_coef'] = results['with_conflict']['conflict_coef']
            
            # With cumulative conflict
            if results.get('with_cum_conflict'):
                row['cum_conflict_coef'] = results['with_cum_conflict']['coef']
                row['cum_conflict_se'] = results['with_cum_conflict']['se']
                row['cum_conflict_pval'] = results['with_cum_conflict']['pval']
                row['cum_conflict_control_coef'] = results['with_cum_conflict']['conflict_coef']
            
            # With log conflict
            if results.get('with_log_conflict'):
                row['log_conflict_coef'] = results['with_log_conflict']['coef']
                row['log_conflict_se'] = results['with_log_conflict']['se']
                row['log_conflict_pval'] = results['with_log_conflict']['pval']
            
            all_did_results.append(row)
            
            # Print comparison
            basic = results.get('basic', {}).get('coef', np.nan)
            controlled = results.get('with_cum_conflict', {}).get('coef', np.nan)
            
            if not np.isnan(basic) and not np.isnan(controlled):
                change = controlled - basic
                direction = "↑" if change > 0 else "↓"
                print(f"    {indicator:<10}: {basic:+.3f} → {controlled:+.3f} ({direction} {abs(change):.3f})")
    
    # Create summary visualizations
    print(f"\n{'='*70}")
    print(" CREATING SUMMARY VISUALIZATIONS")
    print("=" * 70)
    
    # Conflict pre-trends comparison
    if all_conflict_results:
        print("\n  Creating conflict pre-trends comparison...")
        plot_conflict_pretrends_comparison(all_conflict_results, 
                                          FIGURES_DIR / "conflict_pretrends_comparison.png")
        print("  ✓ Saved: conflict_pretrends_comparison.png")
    
    # DiD with/without conflict comparison
    if all_did_results:
        results_df = pd.DataFrame(all_did_results)
        results_df.to_csv(TABLES_DIR / "did_conflict_controlled.csv", index=False)
        print(f"\n  ✓ Saved: did_conflict_controlled.csv")
        
        print("\n  Creating DiD comparison plot...")
        plot_did_with_without_conflict(results_df, FIGURES_DIR / "did_with_without_conflict.png")
        print("  ✓ Saved: did_with_without_conflict.png")
    
    # Final summary
    print(f"\n{'='*70}")
    print(" SUMMARY: CONFLICT ANALYSIS RESULTS")
    print("=" * 70)
    
    if all_conflict_results:
        print("\n  CONFLICT PRE-TRENDS (Do treated sites have more conflict?):")
        for spec_key, result in all_conflict_results.items():
            if result.get('test_results'):
                test = result['test_results']
                diff = test.get('level_diff_coef', 0)
                pval = test.get('level_diff_pval', 1)
                status = "MORE conflict" if diff > 0 else "LESS conflict"
                sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
                print(f"    {spec_key:<15}: Treated have {status} ({diff:+.2f} events, p={pval:.3f}) {sig}")
    
    if all_did_results:
        results_df = pd.DataFrame(all_did_results)
        
        print("\n  VIIRS EFFECT (with vs without conflict control):")
        viirs = results_df[results_df['indicator'] == 'viirs']
        for _, row in viirs.iterrows():
            basic = row.get('basic_coef', np.nan)
            controlled = row.get('cum_conflict_coef', np.nan)
            if not np.isnan(basic) and not np.isnan(controlled):
                change = controlled - basic
                print(f"    {row['spec_label']:<20}: {basic:+.2f} → {controlled:+.2f} (change: {change:+.2f})")
    
    print(f"""
    
  INTERPRETATION:
  ───────────────
  If treated sites have SIGNIFICANTLY MORE conflict than controls,
  this confirms UNDP targeted heavily damaged areas.
  
  When we CONTROL for conflict:
  - VIIRS effect should become LESS NEGATIVE (or positive)
  - This reveals the TRUE intervention effect
  - The negative raw effect was due to site selection, not failure
    """)
    
    print("=" * 70)
    print(" ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
