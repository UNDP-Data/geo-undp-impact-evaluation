#!/usr/bin/env python3
"""
analysis/pse/02_did_comprehensive_unified.py
============================================
Comprehensive DiD Analysis with CORRECT Methodology.

METHODOLOGY (from original validated script):
1. STAGGERED TREATMENT: post=1 when date >= site's project_start
2. LOG TRANSFORMATION: log(viirs + 1) for nightlights
3. CONFLICT CONTROL: site_conflict_events within buffer
4. TWFE: District + Time fixed effects
5. HETEROGENEOUS EFFECTS: By intervention type

NEW FEATURES:
- Multiple buffer zone specifications for robustness
- Systematic pre-trends testing
- Comprehensive output tables and figures

Buffer Zones:
- 200m-600m (very_tight)
- 300m-800m (tight)
- 400m-1km (moderate)
- 500m-1.5km (baseline) - original specification
- 600m-1.5km (conservative)
- 750m-2km (wide)

Usage:
    python analysis/pse/02_did_comprehensive_unified.py
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
import seaborn as sns

warnings.filterwarnings('ignore')

try:
    from scipy.spatial import cKDTree
    import rasterio
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    print("⚠ Missing dependencies. Run: pip install rasterio scipy")

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path("data")
RASTER_DIR = DATA_DIR / "pse" / "rasters"

SITES_CSV = DATA_DIR / "pse" / "external" / "undp_sites_digitized.csv"
PROJECTS_CSV = DATA_DIR / "pse" / "external" / "undp_projects.csv"
ACLED_CSV = DATA_DIR / "pse" / "external" / "acled_conflict.csv"
PANEL_CSV = Path("outputs/pse/panel_data.csv")
BOUNDARIES_FILE = DATA_DIR / "pse" / "boundaries" / "gaza_admin2.geojson"

OUTPUT_DIR = Path("outputs/pse")
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
PRETRENDS_DIR = FIGURES_DIR / "pretrends_staggered"

for d in [TABLES_DIR, FIGURES_DIR, PRETRENDS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# BUFFER ZONE SPECIFICATIONS
# =============================================================================
DONUT_SPECS = [
    (200, 600, "200m-600m", "very_tight"),
    (300, 800, "300m-800m", "tight"),
    (400, 1000, "400m-1km", "moderate"),
    (500, 1500, "500m-1.5km", "baseline"),
    (600, 1500, "600m-1.5km", "conservative"),
    (750, 2000, "750m-2km", "wide"),
]

# Conflict buffer for local exposure (meters)
CONFLICT_BUFFER_M = 500

# Degrees per meter (approximate for Gaza ~31.5°N)
DEG_PER_M = 1 / 111000

# Conflict start date (for filtering projects)
CONFLICT_START = pd.Timestamp("2023-10-07")

# Category mapping for site types to project categories
CATEGORY_MAP = {
    'WASH': ['Water & Sanitation (WASH)'],
    'Debris': ['Debris & Solid Waste', 'Environment'],
    'Shelter': ['Infrastructure & Housing', 'Social Inclusion'],
    'Employment': ['Livelihoods & Economy'],
    'Social Cohesion': ['Governance & Rule of Law', 'Social Inclusion'],
    'Other': ['Health & Emergency', 'Governance & Rule of Law'],
    'Health': ['Health & Emergency']
}

# District-level controls
DISTRICT_CONTROLS = ['rainfall', 'price_bread', 'conflict_events']

# Outcomes to analyze
OUTCOMES = {
    'viirs': {'name': 'VIIRS Nightlights', 'direction': 'positive', 'log': False},
    'log_viirs': {'name': 'Log Nightlights', 'direction': 'positive', 'log': True},
    'ndvi': {'name': 'NDVI (Vegetation)', 'direction': 'positive', 'log': False},
    'ndbi': {'name': 'NDBI (Built-up)', 'direction': 'positive', 'log': False},
    'bsi': {'name': 'BSI (Bare Soil)', 'direction': 'negative', 'log': False},
    'sar_vh': {'name': 'SAR VH (Structures)', 'direction': 'positive', 'log': False},
}

# =============================================================================
# DATA LOADING
# =============================================================================

def load_acled():
    """Load ACLED conflict data."""
    if not ACLED_CSV.exists():
        print("  ⚠ ACLED file not found")
        return None
    
    df = pd.read_csv(ACLED_CSV)
    
    # Standardize columns
    col_map = {
        'event_date': 'date', 'EVENT_DATE': 'date', 'Date': 'date',
        'latitude': 'lat', 'LATITUDE': 'lat', 'Latitude': 'lat',
        'longitude': 'lon', 'LONGITUDE': 'lon', 'Longitude': 'lon',
    }
    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    
    # Filter to Gaza
    if 'admin1' in df.columns:
        df = df[df['admin1'].str.contains('Gaza', case=False, na=False)]
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['year_month'] = df['date'].dt.to_period('M')
    
    return df.dropna(subset=['lat', 'lon'])


def load_sites_with_projects():
    """Load UNDP sites matched to crisis-response projects."""
    if not SITES_CSV.exists():
        print(f"  ⚠ Sites CSV not found: {SITES_CSV}")
        return None
    
    sites = pd.read_csv(SITES_CSV)
    
    # Load and match projects
    if PROJECTS_CSV.exists():
        projects = pd.read_csv(PROJECTS_CSV)
        projects['start_date'] = pd.to_datetime(projects['start_date'], errors='coerce')
        
        # Filter for post-conflict projects
        projects_crisis = projects[projects['start_date'] >= CONFLICT_START]
        print(f"  Projects after Oct 7, 2023: {len(projects_crisis)}")
        
        # Assign start dates based on intervention type
        sites['project_start'] = pd.NaT
        for idx, site in sites.iterrows():
            site_type = site.get('intervention_type', '').strip()
            cats = CATEGORY_MAP.get(site_type, [])
            matches = projects_crisis[projects_crisis['ml_category'].isin(cats)]
            if not matches.empty:
                best = matches.sort_values('start_date').iloc[0]
                sites.at[idx, 'project_start'] = best['start_date']
        
        # Keep only sites with matched projects
        sites = sites.dropna(subset=['project_start'])
    else:
        # If no projects file, use a default start date
        print("  ⚠ Projects CSV not found, using Nov 15, 2023 as default start")
        sites['project_start'] = pd.Timestamp("2023-11-15")
    
    if len(sites) == 0:
        print("  ⚠ No sites matched to crisis-response projects")
        return None
    
    sites['group'] = 'Treated'
    return sites


def load_district_panel():
    """Load district-level panel controls."""
    if not PANEL_CSV.exists():
        return None
    
    df = pd.read_csv(PANEL_CSV)
    df['period_start'] = pd.to_datetime(df['period_start'])
    df['unit_name_clean'] = df['unit_name'].str.lower().str.strip()
    cols = ['unit_name_clean', 'period_start'] + [c for c in DISTRICT_CONTROLS if c in df.columns]
    return df[cols].drop_duplicates(subset=['unit_name_clean', 'period_start'])


def find_rasters():
    """Find all raster files."""
    rasters = {}
    
    if not RASTER_DIR.exists():
        return rasters
    
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
# DONUT CONTROL GENERATION
# =============================================================================

def generate_donut_controls(gdf_boundary, gdf_treated, inner_m, outer_m, n_controls=None, seed=42):
    """Generate control points in the donut zone around treatment sites."""
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
            'project_start': pd.NaT,
        })
    
    return pd.DataFrame(control_points)


# =============================================================================
# DATASET BUILDING (STAGGERED TREATMENT)
# =============================================================================

def build_analysis_dataset(sites, acled, rasters, inner_m, outer_m, panel_controls=None):
    """
    Build analysis dataset with STAGGERED treatment timing.
    
    Key: post=1 when date >= site's project_start (not Oct 7)
    """
    # Load boundaries and create GeoDataFrames
    boundaries = gpd.read_file(BOUNDARIES_FILE).to_crs("EPSG:4326")
    
    treated_gdf = gpd.GeoDataFrame(
        sites[sites['group'] == 'Treated'],
        geometry=gpd.points_from_xy(
            sites[sites['group'] == 'Treated'].longitude,
            sites[sites['group'] == 'Treated'].latitude
        ),
        crs="EPSG:4326"
    )
    
    # Generate donut controls
    controls = generate_donut_controls(boundaries, treated_gdf, inner_m, outer_m)
    
    if len(controls) < len(treated_gdf) * 0.5:
        print(f"    ⚠ Only {len(controls)} controls generated")
    
    # Combine treated and controls
    all_sites = pd.concat([sites, controls], ignore_index=True)
    
    # Add district names
    all_sites_gdf = gpd.GeoDataFrame(
        all_sites,
        geometry=gpd.points_from_xy(all_sites.longitude, all_sites.latitude),
        crs="EPSG:4326"
    )
    
    name_col = next((c for c in boundaries.columns if 'name' in c.lower()), None)
    if name_col:
        joined = gpd.sjoin(all_sites_gdf, boundaries[[name_col, 'geometry']], how="left", predicate="within")
        all_sites['district'] = joined[name_col].fillna('Unknown').values
    else:
        all_sites['district'] = 'Gaza'
    
    # Build ACLED spatial trees by month
    acled_trees = {}
    if acled is not None:
        for period, group in acled.groupby('year_month'):
            ts = period.to_timestamp()
            coords = group[['lon', 'lat']].values
            if len(coords) > 0:
                acled_trees[ts] = cKDTree(coords)
    
    # Extract raster values
    data_records = []
    conflict_buffer_deg = CONFLICT_BUFFER_M * DEG_PER_M
    
    for indicator, dates in rasters.items():
        if indicator not in OUTCOMES and indicator != 'viirs':
            continue
        
        for date, raster_path in dates.items():
            month_key = pd.Timestamp(year=date.year, month=date.month, day=1)
            acled_tree = acled_trees.get(month_key)
            
            # District controls
            if panel_controls is not None:
                monthly_controls = panel_controls[panel_controls['period_start'] == date]
            else:
                monthly_controls = pd.DataFrame()
            
            try:
                with rasterio.open(raster_path) as src:
                    band = src.read(1)
                    
                    for _, site in all_sites.iterrows():
                        try:
                            r, c = src.index(site['longitude'], site['latitude'])
                            val = band[r, c]
                            
                            if np.isnan(val) or val == src.nodata:
                                continue
                            
                            # Conflict events within buffer
                            site_events = 0
                            if acled_tree:
                                site_events = len(acled_tree.query_ball_point(
                                    [site['longitude'], site['latitude']], conflict_buffer_deg))
                            
                            # STAGGERED TREATMENT: post based on site's project_start
                            if pd.notna(site.get('project_start')):
                                post = 1 if date >= site['project_start'] else 0
                                # Relative month for event study
                                diff = (date.year - site['project_start'].year) * 12 + \
                                       (date.month - site['project_start'].month)
                                rel_month = max(-6, min(12, diff))  # Cap at -6 to +12
                            else:
                                post = 0  # Controls are never "post"
                                rel_month = np.nan
                            
                            # District controls
                            d_name = str(site.get('district', 'Gaza')).lower().strip()
                            d_row = monthly_controls[monthly_controls['unit_name_clean'] == d_name] if not monthly_controls.empty else pd.DataFrame()
                            
                            record = {
                                'site_id': site['site_id'],
                                'intervention_type': site['intervention_type'],
                                'group': site['group'],
                                'district': site.get('district', 'Gaza'),
                                'date': date,
                                'indicator': indicator,
                                'value': float(val),
                                'post': post,
                                'rel_month': rel_month,
                                'site_conflict_events': site_events,
                                'project_start': site.get('project_start', pd.NaT),
                            }
                            
                            # Add district controls
                            if not d_row.empty:
                                for ctrl in DISTRICT_CONTROLS:
                                    if ctrl in d_row.columns:
                                        record[f'dist_{ctrl}'] = d_row.iloc[0][ctrl]
                            
                            data_records.append(record)
                            
                            # Add log transformation for VIIRS
                            if indicator == 'viirs':
                                record_log = record.copy()
                                record_log['indicator'] = 'log_viirs'
                                record_log['value'] = np.log(max(0, val) + 1)
                                data_records.append(record_log)
                        
                        except:
                            continue
            except Exception as e:
                continue
    
    return pd.DataFrame(data_records)


# =============================================================================
# TWFE REGRESSION
# =============================================================================


def run_twfe_regression(df):
    """
    Run Two-Way Fixed Effects regression.
    
    Model: value ~ treated:post + site_conflict_events + C(district) + C(date_str)
    """
    results = []
    
    df['date_str'] = df['date'].dt.strftime('%Y-%m')
    df['treated'] = (df['group'] == 'Treated').astype(int)
    
    # Fill missing controls
    fill_cols = [c for c in df.columns if 'dist_' in c]
    if fill_cols:
        df[fill_cols] = df[fill_cols].fillna(0)
    
    for indicator in df['indicator'].unique():
        subset = df[df['indicator'] == indicator].copy()
        
        if len(subset) < 100:
            continue
            
        # CALCULATE CONTROL MEAN (for percentage translation)
        # Use mean of control group across the whole period as baseline reference
        ctrl_mean = subset[subset['group'] == 'Control']['value'].mean()
        
        # Main TWFE model
        formula = "value ~ treated:post + site_conflict_events + C(district) + C(date_str)"
        
        try:
            model = smf.ols(formula, data=subset).fit(cov_type='HC1')
            
            main_term = 'treated:post'
            if main_term in model.params:
                coef = model.params[main_term]
                se = model.bse[main_term]
                pval = model.pvalues[main_term]
                ci = model.conf_int().loc[main_term]
                stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
                
                # Calculate percentage change
                # For log-linear models: 100 * (exp(beta) - 1)
                # For linear models: 100 * (beta / mean)
                if indicator.startswith('log_'):
                    pct_change = (np.exp(coef) - 1) * 100
                else:
                    pct_change = (coef / ctrl_mean) * 100 if ctrl_mean and not np.isnan(ctrl_mean) else 0

                results.append({
                    'indicator': indicator,
                    'intervention_type': 'Overall',
                    'coef': coef,
                    'control_mean': ctrl_mean,
                    'pct_change': pct_change,
                    'se': se,
                    'ci_lower': ci[0],
                    'ci_upper': ci[1],
                    'pval': pval,
                    'stars': stars,
                    'R2': model.rsquared,
                    'N': int(model.nobs),
                })
        except Exception as e:
            continue
        
        # Heterogeneous effects by intervention type (treated only)
        treated_only = subset[subset['group'] == 'Treated'].copy()
        
        if len(treated_only) > 100:
            formula_het = "value ~ post:C(intervention_type) + site_conflict_events + C(district) + C(date_str)"
            
            try:
                model_het = smf.ols(formula_het, data=treated_only).fit(cov_type='HC1')
                
                for param in model_het.params.index:
                    if 'post:' in param and 'intervention_type' in param:
                        # Improved name cleaning
                        int_type = param
                        if "[T." in param:
                            int_type = param.split("[T.")[1].split("]")[0]
                        elif "intervention_type" in param:
                            int_type = param.split("intervention_type")[1].replace("[", "").replace("]", "").replace(":", "").replace("post", "")
                        
                        coef = model_het.params[param]
                        se = model_het.bse[param]
                        pval = model_het.pvalues[param]
                        ci = model_het.conf_int().loc[param]
                        stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
                        
                        if indicator.startswith('log_'):
                            pct_change = (np.exp(coef) - 1) * 100
                        else:
                            pct_change = (coef / ctrl_mean) * 100 if ctrl_mean and not np.isnan(ctrl_mean) else 0

                        results.append({
                            'indicator': indicator,
                            'intervention_type': int_type,
                            'coef': coef,
                            'control_mean': ctrl_mean,
                            'pct_change': pct_change,
                            'se': se,
                            'ci_lower': ci[0],
                            'ci_upper': ci[1],
                            'pval': pval,
                            'stars': stars,
                            'R2': model_het.rsquared,
                            'N': int(model_het.nobs),
                        })
            except:
                continue
    
    return pd.DataFrame(results)

# =============================================================================
# PRE-TRENDS TESTING (STAGGERED)
# =============================================================================

def test_pretrends_staggered(df, indicator):
    """
    Test pre-trends using relative time to treatment.
    
    For staggered DiD, we test whether treatment and control
    had parallel trends in the pre-intervention period.
    """
    subset = df[df['indicator'] == indicator].copy()
    
    # Pre-period: rel_month < 0 for treated, all months for control
    subset['pre_period'] = ((subset['rel_month'] < 0) | (subset['group'] == 'Control')).astype(int)
    
    pre_data = subset[
        ((subset['group'] == 'Treated') & (subset['rel_month'] < 0)) |
        (subset['group'] == 'Control')
    ].copy()
    
    if len(pre_data) < 100:
        return None, None, None
    
    pre_data['month_idx'] = pre_data['date'].dt.year * 12 + pre_data['date'].dt.month
    pre_data['is_treated'] = (pre_data['group'] == 'Treated').astype(int)
    
    try:
        # Test differential pre-trends
        model = smf.ols('value ~ month_idx * is_treated', data=pre_data).fit()
        
        interaction_term = 'month_idx:is_treated'
        if interaction_term in model.pvalues:
            pval = model.pvalues[interaction_term]
            coef = model.params[interaction_term]
            passed = pval > 0.10
            return passed, pval, coef
    except:
        pass
    
    return None, None, None


def compute_group_means_staggered(df, indicator):
    """Compute means by group using relative time for treated."""
    subset = df[df['indicator'] == indicator].copy()
    
    # For treated: use relative month
    treated = subset[subset['group'] == 'Treated'].copy()
    treated_agg = treated.groupby('rel_month')['value'].agg(['mean', 'std', 'count']).reset_index()
    treated_agg['se'] = treated_agg['std'] / np.sqrt(treated_agg['count'])
    treated_agg['group'] = 'Treated'
    treated_agg = treated_agg.rename(columns={'rel_month': 'time'})
    
    # For control: use calendar time (centered on median treated start)
    control = subset[subset['group'] == 'Control'].copy()
    control['month_idx'] = control['date'].dt.year * 12 + control['date'].dt.month
    
    # Center control time on median treatment start
    treated_starts = df[df['group'] == 'Treated']['project_start'].dropna()
    if not treated_starts.empty:
        median_start = treated_starts.median()
        median_month = median_start.year * 12 + median_start.month
        control['time'] = control['month_idx'] - median_month
    else:
        control['time'] = control['month_idx'] - control['month_idx'].median()
    
    control_agg = control.groupby('time')['value'].agg(['mean', 'std', 'count']).reset_index()
    control_agg['se'] = control_agg['std'] / np.sqrt(control_agg['count'])
    control_agg['group'] = 'Control'
    
    return treated_agg, control_agg


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_pretrends_staggered(df, indicator, spec_label, passed, pval, output_path):
    """Plot pre-trends with relative time (staggered design)."""
    treated_agg, control_agg = compute_group_means_staggered(df, indicator)
    
    if treated_agg is None or len(treated_agg) < 3:
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    outcome_name = OUTCOMES.get(indicator, {}).get('name', indicator.upper())
    
    # Background shading
    ax.axvspan(-12, 0, alpha=0.1, color='#3498db')
    ax.axvspan(0, 12, alpha=0.1, color='#27ae60')
    
    # Treatment line
    ax.axvline(0, color='#27ae60', linewidth=2.5, linestyle='--', label='Project Start')
    
    # Treated trajectory
    t = treated_agg[(treated_agg['time'] >= -6) & (treated_agg['time'] <= 12)]
    ax.fill_between(t['time'], t['mean'] - 1.96*t['se'], t['mean'] + 1.96*t['se'],
                    alpha=0.2, color='#e74c3c')
    ax.plot(t['time'], t['mean'], 'o-', color='#e74c3c', linewidth=2, markersize=6,
            label='Treated (UNDP sites)')
    
    # Control trajectory
    c = control_agg[(control_agg['time'] >= -6) & (control_agg['time'] <= 12)]
    ax.fill_between(c['time'], c['mean'] - 1.96*c['se'], c['mean'] + 1.96*c['se'],
                    alpha=0.2, color='#3498db')
    ax.plot(c['time'], c['mean'], 's--', color='#3498db', linewidth=2, markersize=6,
            label=f'Control ({spec_label})')
    
    # Pre-trends result
    if passed is not None:
        if passed:
            status = f"✓ Parallel trends HOLD (p={pval:.3f})"
            color = '#27ae60'
        else:
            status = f"⚠ Parallel trends VIOLATED (p={pval:.3f})"
            color = '#e74c3c'
        
        ax.annotate(status, xy=(0.02, 0.98), xycoords='axes fraction',
                   fontsize=11, ha='left', va='top', fontweight='bold', color=color,
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                            edgecolor=color, alpha=0.95))
    
    ax.set_xlabel('Months Relative to Project Start', fontsize=11, fontweight='bold')
    ax.set_ylabel(outcome_name, fontsize=11, fontweight='bold')
    ax.set_title(f'Staggered DiD Pre-Trends: {outcome_name}\nControl: {spec_label}',
                fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_results_summary(all_results, output_path):
    """Create summary plot of results across specifications."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    indicators = ['log_viirs', 'bsi', 'sar_vh', 'ndvi', 'ndbi', 'viirs']
    
    for i, indicator in enumerate(indicators):
        ax = axes[i]
        
        ind_results = all_results[(all_results['indicator'] == indicator) & 
                                  (all_results['intervention_type'] == 'Overall')]
        
        if ind_results.empty:
            ax.set_visible(False)
            continue
        
        ind_results = ind_results.sort_values('inner_m')
        
        x = np.arange(len(ind_results))
        colors = ['#27ae60' if p else '#e74c3c' for p in ind_results['pretrends_passed']]
        
        bars = ax.bar(x, ind_results['coef'], yerr=1.96*ind_results['se'],
                     color=colors, edgecolor='black', capsize=5, alpha=0.8)
        
        ax.axhline(0, color='black', linewidth=1)
        
        # Add stars
        for j, (_, row) in enumerate(ind_results.iterrows()):
            if row['stars']:
                y_pos = row['coef'] + 1.96*row['se'] if row['coef'] > 0 else row['coef'] - 1.96*row['se']
                ax.annotate(row['stars'], xy=(j, y_pos), ha='center', fontsize=10, fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels(ind_results['spec_label'], rotation=45, ha='right')
        ax.set_ylabel('DiD Coefficient', fontsize=10)
        ax.set_title(OUTCOMES.get(indicator, {}).get('name', indicator), fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Legend
    legend_elements = [
        Patch(facecolor='#27ae60', edgecolor='black', label='Pre-trends PASS'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Pre-trends FAIL'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=11, 
               bbox_to_anchor=(0.5, 0.02))
    
    fig.suptitle('TWFE DiD Results Across Buffer Specifications\n(Staggered Treatment, Conflict-Controlled)',
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_intervention_type_table(all_results, output_path):
    """Create formatted table by intervention type."""
    # Filter to baseline spec and intervention types
    baseline = all_results[(all_results['spec_key'] == 'baseline') & 
                           (all_results['intervention_type'] != 'Overall')]
    
    if baseline.empty:
        return
    
    # Pivot
    pivot = baseline.pivot_table(
        index='intervention_type',
        columns='indicator',
        values='coef',
        aggfunc='first'
    )
    
    # Add stars
    for indicator in pivot.columns:
        for int_type in pivot.index:
            row = baseline[(baseline['indicator'] == indicator) & 
                          (baseline['intervention_type'] == int_type)]
            if not row.empty:
                coef = row['coef'].values[0]
                stars = row['stars'].values[0]
                pivot.loc[int_type, indicator] = f"{coef:.4f} {stars}"
    
    pivot.to_csv(output_path)
    print(f"  ✓ Saved: {output_path.name}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print(" COMPREHENSIVE TWFE DiD ANALYSIS")
    print(" Staggered Treatment | Conflict Controls | Multiple Buffer Zones")
    print("=" * 70)
    
    if not HAS_DEPS:
        print("\n⚠ Missing dependencies. Exiting.")
        return
    
    # Load data
    print("\n--- Loading Data ---")
    
    sites = load_sites_with_projects()
    if sites is None:
        print("  ❌ Failed to load sites. Exiting.")
        return
    
    print(f"  Treated sites: {len(sites)}")
    print(f"  Intervention types: {sites['intervention_type'].value_counts().to_dict()}")
    
    acled = load_acled()
    if acled is not None:
        print(f"  ACLED events: {len(acled)}")
    
    panel_controls = load_district_panel()
    
    rasters = find_rasters()
    print(f"  Raster indicators: {list(rasters.keys())}")
    
    # Run analysis for each buffer specification
    all_results = []
    
    for inner_m, outer_m, spec_label, spec_key in DONUT_SPECS:
        print(f"\n{'='*70}")
        print(f" BUFFER SPECIFICATION: {spec_label}")
        print("=" * 70)
        
        # Build dataset
        print("  Building dataset...")
        df = build_analysis_dataset(sites, acled, rasters, inner_m, outer_m, panel_controls)
        
        if df.empty:
            print("  ⚠ Empty dataset, skipping")
            continue
        
        n_treated = df[df['group'] == 'Treated']['site_id'].nunique()
        n_control = df[df['group'] == 'Control']['site_id'].nunique()
        print(f"  Dataset: {len(df):,} obs ({n_treated} treated, {n_control} control)")
        
        # Run TWFE regression
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
        for indicator in df['indicator'].unique():
            passed, pval, coef = test_pretrends_staggered(df, indicator)
            
            mask = results['indicator'] == indicator
            results.loc[mask, 'pretrends_passed'] = passed
            results.loc[mask, 'pretrends_pval'] = pval
            
            status = "PASS ✓" if passed else "FAIL ⚠" if passed is not None else "N/A"
            print(f"    {indicator:<12}: {status}")
            
            # Create pre-trends plot
            plot_pretrends_staggered(df, indicator, spec_label, passed, pval,
                                    PRETRENDS_DIR / f"pretrends_{indicator}_{spec_key}.png")
        
        # Print results
        print("\n  Results (Overall):")
        print(f"  {'Indicator':<12} {'Coef':>10} {'SE':>8} {'R²':>6}")
        print("  " + "-" * 40)
        
        for _, row in results[results['intervention_type'] == 'Overall'].iterrows():
            print(f"  {row['indicator']:<12} {row['coef']:>+10.4f}{row['stars']:<3} {row['se']:>8.4f} {row['R2']:>6.3f}")
        
        all_results.append(results)
    
    # Combine all results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(TABLES_DIR / "did_comprehensive_unified.csv", index=False)
        print(f"\n✓ Saved: {TABLES_DIR}/did_comprehensive_unified.csv")
        
        # Summary visualizations
        print("\n--- Creating Summary Visualizations ---")
        plot_results_summary(combined, FIGURES_DIR / "did_results_summary.png")
        print(f"  ✓ Saved: did_results_summary.png")
        
        create_intervention_type_table(combined, TABLES_DIR / "did_by_intervention_type.csv")
        
        # Final summary
        print("\n" + "=" * 70)
        print(" SUMMARY: Baseline Specification (500m-1.5km)")
        print("=" * 70)
        
        baseline = combined[(combined['spec_key'] == 'baseline') & 
                           (combined['intervention_type'] == 'Overall')]
        
        for _, row in baseline.iterrows():
            pt_status = "✓" if row.get('pretrends_passed', False) else "⚠"
            print(f"  {row['indicator']:<12}: {row['coef']:>+8.4f}{row['stars']:<3} (R²={row['R2']:.2f}) {pt_status}")
    
    print("\n" + "=" * 70)
    print(" ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
