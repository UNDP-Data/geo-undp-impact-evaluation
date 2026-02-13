#!/usr/bin/env python3
"""
analysis/pse/02_did_comprehensive.py
====================================
Comprehensive DiD Analysis with Donut Control Group.

Key Features:
1. DONUT CONTROL GROUP: Controls between 500m-1.5km from treatment sites
   (close enough to be comparable, far enough to avoid spillovers)
2. Multiple buffer zones for conflict control (250m, 500m, 1000m)
3. ALL raster outcomes including SAR VH/VV
4. TWFE models with heterogeneous treatment effects
5. Event study plots with improved visualization
6. Pre-trends validation
7. GDP elasticity sensitivity analysis

Usage:
    python analysis/pse/02_did_comprehensive.py
"""

import argparse
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
import re

try:
    from scipy.spatial import cKDTree
    import rasterio
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    print("⚠ Missing dependencies. Run: pip install rasterio scipy --break-system-packages")

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path("data")
RASTER_DIR = DATA_DIR / "pse" / "rasters"

# Inputs
SITES_CSV = DATA_DIR / "pse" / "external" / "undp_sites_digitized.csv"
PROJECTS_CSV = DATA_DIR / "pse" / "external" / "undp_projects.csv"
ACLED_CSV = DATA_DIR / "pse" / "external" / "acled_conflict.csv"
PANEL_CSV = Path("outputs/pse/panel_data.csv")
BOUNDARIES_FILE = DATA_DIR / "pse" / "boundaries" / "gaza_admin2.geojson"

OUTPUT_DIR = Path("outputs/pse")
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
EVENT_STUDY_DIR = FIGURES_DIR / "event_studies"
PRETRENDS_DIR = FIGURES_DIR / "pretrends"

for d in [TABLES_DIR, EVENT_STUDY_DIR, PRETRENDS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DONUT CONTROL GROUP PARAMETERS
# =============================================================================
# Controls must be:
#   - At least INNER_BUFFER meters from any treatment site (avoid spillovers)
#   - At most OUTER_BUFFER meters from nearest treatment site (ensure comparability)
INNER_BUFFER_M = 500    # Minimum distance from treatment (spillover buffer)
OUTER_BUFFER_M = 1500   # Maximum distance from treatment (comparability)

# Convert to degrees (approximate for Gaza latitude ~31.5°N)
# 1 degree latitude ≈ 111 km, 1 degree longitude ≈ 95 km at this latitude
DEG_PER_M = 1 / 100000  # Rough approximation: 100m ≈ 0.001 degrees
INNER_BUFFER_DEG = INNER_BUFFER_M * DEG_PER_M
OUTER_BUFFER_DEG = OUTER_BUFFER_M * DEG_PER_M

# Buffer zones for conflict control (in meters)
BUFFER_ZONES = [250, 500, 1000]

# Conflict start date
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

# District controls from panel data
DISTRICT_CONTROLS = ['rainfall', 'price_bread', 'conflict_events']

# GDP Elasticity values for sensitivity analysis
GDP_ELASTICITIES = {
    'low': 0.2,
    'medium': 0.3,
    'high': 0.5
}

# =============================================================================
# COMPREHENSIVE OUTCOMES LIST
# =============================================================================
ALL_OUTCOMES = {
    'viirs': {'name': 'VIIRS Nightlights', 'direction': 'positive'},
    'log_viirs': {'name': 'Log VIIRS', 'direction': 'positive'},
    'ndvi': {'name': 'NDVI', 'direction': 'positive'},
    'ndbi': {'name': 'NDBI', 'direction': 'positive'},
    'bsi': {'name': 'BSI', 'direction': 'negative'},
    'sar_vh': {'name': 'SAR VH', 'direction': 'positive'},
    'sar_vv': {'name': 'SAR VV', 'direction': 'positive'},
}

# =============================================================================
# DONUT CONTROL GROUP GENERATION
# =============================================================================

def generate_donut_controls(gdf_boundary, gdf_treated, n_controls=None, 
                            inner_buffer=INNER_BUFFER_DEG, outer_buffer=OUTER_BUFFER_DEG):
    """
    Generate control points in the "donut" zone around treatment sites.
    
    Controls are:
    - At least inner_buffer from ANY treatment site (avoid spillovers)
    - At most outer_buffer from the NEAREST treatment site (ensure comparability)
    
    This creates controls that are similar to treated areas but not contaminated.
    """
    if n_controls is None:
        n_controls = len(gdf_treated)
    
    print(f"  Generating {n_controls} donut controls ({int(inner_buffer/DEG_PER_M)}m - {int(outer_buffer/DEG_PER_M)}m from treatment)...")
    
    # Build KD-tree for fast nearest-neighbor queries
    treated_coords = np.array([[p.x, p.y] for p in gdf_treated.geometry])
    tree = cKDTree(treated_coords)
    
    control_points = []
    minx, miny, maxx, maxy = gdf_boundary.total_bounds
    
    attempts = 0
    max_attempts = 200000
    
    while len(control_points) < n_controls and attempts < max_attempts:
        attempts += 1
        
        # Random point
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        
        # Check if inside Gaza
        from shapely.geometry import Point
        pnt = Point(x, y)
        if not gdf_boundary.contains(pnt).any():
            continue
        
        # Find distance to nearest treatment site
        dist, _ = tree.query([x, y])
        
        # Check donut condition: inner_buffer < dist < outer_buffer
        if dist < inner_buffer or dist > outer_buffer:
            continue
        
        control_points.append({
            'site_id': f'CTRL_{len(control_points):04d}',
            'intervention_type': 'Control',
            'latitude': y,
            'longitude': x,
            'group': 'Control',
            'project_start': pd.NaT,
            'dist_to_nearest_treated': dist * 100000  # Convert to meters approx
        })
    
    print(f"    Created {len(control_points)} valid donut controls (after {attempts} attempts)")
    
    if len(control_points) < n_controls:
        print(f"    ⚠ Only found {len(control_points)}/{n_controls} controls in donut zone")
    
    return pd.DataFrame(control_points)


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_acled():
    """Load ACLED conflict data."""
    if not ACLED_CSV.exists():
        print("  ⚠ ACLED file not found")
        return None
    df = pd.read_csv(ACLED_CSV)
    if 'admin1' in df.columns:
        df = df[df['admin1'].str.contains('Gaza', case=False, na=False)]
    df['event_date'] = pd.to_datetime(df['event_date'])
    df['year_month'] = df['event_date'].dt.to_period('M')
    return df.dropna(subset=['latitude', 'longitude'])


def load_and_prep_sites():
    """Load and prepare UNDP intervention sites with donut controls."""
    if not SITES_CSV.exists():
        print(f"  ⚠ Sites CSV not found: {SITES_CSV}")
        return None
    if not BOUNDARIES_FILE.exists():
        print(f"  ⚠ Boundaries file not found: {BOUNDARIES_FILE}")
        return None

    sites = pd.read_csv(SITES_CSV)
    
    # Load projects and match
    if PROJECTS_CSV.exists():
        projects = pd.read_csv(PROJECTS_CSV)
        projects['start_date'] = pd.to_datetime(projects['start_date'], errors='coerce')
        
        # Filter for post-conflict projects
        projects_crisis = projects[projects['start_date'] >= CONFLICT_START]
        print(f"  Projects after Oct 7, 2023: {len(projects_crisis)}")
        
        # Assign start dates
        sites['project_start'] = pd.NaT
        for idx, site in sites.iterrows():
            site_type = site.get('intervention_type', '').strip()
            cats = CATEGORY_MAP.get(site_type, [])
            matches = projects_crisis[projects_crisis['ml_category'].isin(cats)]
            if not matches.empty:
                best = matches.sort_values('start_date').iloc[0]
                sites.at[idx, 'project_start'] = best['start_date']
        
        sites = sites.dropna(subset=['project_start'])
    
    if len(sites) == 0:
        print("  ⚠ No sites matched to crisis-response projects")
        return None
    
    sites['group'] = 'Treated'
    print(f"  Treated sites: {len(sites)}")

    # Create GeoDataFrames
    sites_gdf = gpd.GeoDataFrame(
        sites, 
        geometry=gpd.points_from_xy(sites.longitude, sites.latitude),
        crs="EPSG:4326"
    )
    boundaries = gpd.read_file(BOUNDARIES_FILE).to_crs("EPSG:4326")
    
    # Add district names
    name_col = next((c for c in boundaries.columns if 'name' in c.lower()), None)
    if name_col:
        sites_joined = gpd.sjoin(sites_gdf, boundaries[[name_col, 'geometry']], how="left", predicate="within")
        sites_joined = sites_joined.rename(columns={name_col: 'district_name'})
        sites_joined['district_name'] = sites_joined['district_name'].fillna('Unknown')
    else:
        sites_joined = sites_gdf.copy()
        sites_joined['district_name'] = 'Gaza'
    
    # Generate DONUT control group
    controls_df = generate_donut_controls(boundaries, sites_joined, n_controls=len(sites))
    
    # Add district to controls
    if not controls_df.empty:
        controls_gdf = gpd.GeoDataFrame(
            controls_df,
            geometry=gpd.points_from_xy(controls_df.longitude, controls_df.latitude),
            crs="EPSG:4326"
        )
        if name_col:
            controls_joined = gpd.sjoin(controls_gdf, boundaries[[name_col, 'geometry']], how="left", predicate="within")
            controls_joined = controls_joined.rename(columns={name_col: 'district_name'})
            controls_joined['district_name'] = controls_joined['district_name'].fillna('Unknown')
        else:
            controls_joined = controls_gdf.copy()
            controls_joined['district_name'] = 'Gaza'
        
        # Combine treated and controls
        combined = pd.concat([
            sites_joined.drop(columns=['geometry', 'index_right'], errors='ignore'),
            controls_joined.drop(columns=['geometry', 'index_right'], errors='ignore')
        ], ignore_index=True)
    else:
        combined = sites_joined.drop(columns=['geometry', 'index_right'], errors='ignore')
    
    print(f"  Total sites (treated + control): {len(combined)}")
    return combined


def load_district_panel():
    """Load district-level panel controls."""
    if not PANEL_CSV.exists():
        print("  ⚠ Panel CSV not found")
        return None
    df = pd.read_csv(PANEL_CSV)
    df['period_start'] = pd.to_datetime(df['period_start'])
    df['unit_name_clean'] = df['unit_name'].str.lower().str.strip()
    cols = ['unit_name_clean', 'period_start'] + [c for c in DISTRICT_CONTROLS if c in df.columns]
    return df[cols].drop_duplicates(subset=['unit_name_clean', 'period_start'])


def find_all_rasters():
    """Find ALL raster files including SAR."""
    rasters = {}
    
    if not RASTER_DIR.exists():
        print(f"  ⚠ Raster directory not found: {RASTER_DIR}")
        return rasters
    
    for f in RASTER_DIR.glob("*.tif"):
        parts = f.stem.split('_')
        
        # Handle SAR files
        if 'sar' in f.name.lower():
            if 'vh' in f.name.lower():
                ind = 'sar_vh'
            elif 'vv' in f.name.lower():
                ind = 'sar_vv'
            else:
                continue
        else:
            ind = parts[0].lower()
        
        # Find date
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
# DATASET BUILDING
# =============================================================================

def build_analysis_dataset(sites, panel_controls, acled, rasters, buffer):
    """Build comprehensive analysis dataset for a given conflict buffer size."""
    data_records = []
    
    # Build ACLED spatial trees by month
    acled_trees = {}
    if acled is not None:
        for period, group in acled.groupby('year_month'):
            ts = period.to_timestamp()
            coords = group[['longitude', 'latitude']].values
            if len(coords) > 0:
                acled_trees[ts] = cKDTree(coords)

    # Process each indicator
    for indicator, dates in rasters.items():
        print(f"    Processing {indicator} ({len(dates)} dates)...")
        
        for date, raster_path in dates.items():
            month_key = pd.Timestamp(year=date.year, month=date.month, day=1)
            tree = acled_trees.get(month_key)
            
            # Get district controls
            if panel_controls is not None:
                monthly_controls = panel_controls[panel_controls['period_start'] == date]
            else:
                monthly_controls = pd.DataFrame()

            try:
                with rasterio.open(raster_path) as src:
                    band = src.read(1)
                    
                    for _, site in sites.iterrows():
                        try:
                            r, c = src.index(site['longitude'], site['latitude'])
                            val = band[r, c]
                            
                            if np.isnan(val) or val == src.nodata:
                                continue
                            
                            # Conflict events within buffer
                            site_events = 0
                            if tree:
                                deg_radius = buffer / 100000.0
                                site_events = len(tree.query_ball_point(
                                    [site['longitude'], site['latitude']], deg_radius))
                            
                            # District controls
                            d_name = str(site.get('district_name', 'Gaza')).lower().strip()
                            d_row = monthly_controls[monthly_controls['unit_name_clean'] == d_name]
                            
                            # Calculate relative month for treated sites
                            if pd.notna(site.get('project_start')):
                                diff = (date.year - site['project_start'].year) * 12 + \
                                       (date.month - site['project_start'].month)
                                if diff < -6:
                                    rel_month = -7
                                elif diff > 12:
                                    rel_month = 13
                                else:
                                    rel_month = diff
                                post = 1 if date >= site['project_start'] else 0
                            else:
                                rel_month = np.nan
                                post = 0
                            
                            record = {
                                'site_id': site['site_id'],
                                'type': site['intervention_type'],
                                'group': site.get('group', 'Treated'),
                                'district': site.get('district_name', 'Gaza'),
                                'date': date,
                                'indicator': indicator,
                                'value': float(val),
                                'post': post,
                                'rel_month': rel_month,
                                'site_conflict_events': site_events,
                                'buffer': buffer
                            }
                            
                            # Add district controls
                            if not d_row.empty:
                                for ctrl in DISTRICT_CONTROLS:
                                    if ctrl in d_row.columns:
                                        record[f"dist_{ctrl}"] = d_row.iloc[0][ctrl]
                            
                            data_records.append(record)
                            
                            # Add log transformation for VIIRS
                            if indicator == 'viirs':
                                record_log = record.copy()
                                record_log['indicator'] = 'log_viirs'
                                v_safe = max(0, val)
                                record_log['value'] = np.log(v_safe + 1)
                                data_records.append(record_log)

                        except Exception as e:
                            continue
                            
            except Exception as e:
                print(f"      ⚠ Error reading {raster_path.name}: {e}")
                continue

    return pd.DataFrame(data_records)


# =============================================================================
# REGRESSION ANALYSIS
# =============================================================================

def run_twfe_regression(df, buffer):
    """Run Two-Way Fixed Effects regression with treated vs donut controls."""
    results = []
    df['date_str'] = df['date'].dt.strftime('%Y-%m')
    
    # Create treated indicator (1 for treated sites, 0 for controls)
    df['treated'] = (df['group'] == 'Treated').astype(int)
    
    # Fill missing controls
    fill_cols = [c for c in df.columns if 'dist_' in c]
    if fill_cols:
        df[fill_cols] = df[fill_cols].fillna(0)
    
    for indicator in df['indicator'].unique():
        subset = df[df['indicator'] == indicator].copy()
        if len(subset) < 100:
            print(f"    Skipping {indicator} (n={len(subset)})")
            continue
        
        print(f"\n  --- {indicator.upper()} (Buffer={buffer}m) ---")
        
        # TWFE: Value ~ Post×Treated + Post×Type (for treated only) + Controls + FE
        # This compares treated vs control AND allows heterogeneity by intervention type
        
        # Simple specification: treated × post interaction
        formula = (
            "value ~ treated:post + "
            "site_conflict_events + "
            "C(district) + C(date_str)"
        )
        
        try:
            model = smf.ols(formula, data=subset).fit(cov_type='HC1')
            
            print(f"    R²: {model.rsquared:.3f} | N: {int(model.nobs)}")
            
            # Main treatment effect
            main_term = 'treated:post'
            if main_term in model.params:
                coef = model.params[main_term]
                se = model.bse[main_term]
                pval = model.pvalues[main_term]
                ci = model.conf_int().loc[main_term]
                stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
                
                print(f"    Treatment Effect: {coef:>8.4f} {stars} (SE: {se:.4f})")
                
                results.append({
                    'buffer': buffer,
                    'indicator': indicator,
                    'intervention': 'Overall (Treated vs Control)',
                    'coef': coef,
                    'se': se,
                    'ci_lower': ci[0],
                    'ci_upper': ci[1],
                    'pval': pval,
                    'stars': stars,
                    'R2': model.rsquared,
                    'N': int(model.nobs)
                })
            
            # Also run heterogeneous model for treated sites only
            treated_only = subset[subset['group'] == 'Treated'].copy()
            if len(treated_only) > 50:
                formula_het = (
                    "value ~ post:C(type) + "
                    "site_conflict_events + "
                    "C(district) + C(date_str)"
                )
                
                try:
                    model_het = smf.ols(formula_het, data=treated_only).fit(cov_type='HC1')
                    
                    interactions = [x for x in model_het.params.index if ':' in x and 'post' in x]
                    
                    for param in interactions:
                        try:
                            raw_content = param.split("[")[1].split("]")[0]
                            clean_name = raw_content.replace("T.", "")
                        except:
                            clean_name = param
                        
                        coef = model_het.params[param]
                        se = model_het.bse[param]
                        pval = model_het.pvalues[param]
                        ci = model_het.conf_int().loc[param]
                        stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
                        
                        if stars:
                            print(f"      {clean_name:<20}: {coef:>8.4f} {stars}")
                        
                        results.append({
                            'buffer': buffer,
                            'indicator': indicator,
                            'intervention': clean_name,
                            'coef': coef,
                            'se': se,
                            'ci_lower': ci[0],
                            'ci_upper': ci[1],
                            'pval': pval,
                            'stars': stars,
                            'R2': model_het.rsquared,
                            'N': int(model_het.nobs)
                        })
                except:
                    pass
                    
        except Exception as e:
            print(f"    ⚠ Model failed: {e}")

    return pd.DataFrame(results)


def run_event_study(df, indicator, int_type, buffer):
    """Run event study regression."""
    if int_type == 'Control':
        return None, None
    
    subset = df[(df['indicator'] == indicator) & (df['type'] == int_type)].copy()
    subset = subset.dropna(subset=['rel_month'])
    
    if len(subset) < 50:
        return None, None
    
    formula = (
        "value ~ C(rel_month, Treatment(reference=-1)) + "
        "site_conflict_events + "
        "C(district) + C(date)"
    )
    
    try:
        model = smf.ols(formula, data=subset).fit(cov_type='HC1')
        
        coef_data = []
        for t in range(-7, 14):
            if t == -1:
                coef_data.append({'rel_month': t, 'coef': 0, 'ci_lower': 0, 'ci_upper': 0})
                continue
            
            name = f"C(rel_month, Treatment(reference=-1))[T.{t}]"
            if name in model.params:
                ci = model.conf_int().loc[name]
                coef_data.append({
                    'rel_month': t,
                    'coef': model.params[name],
                    'ci_lower': ci[0],
                    'ci_upper': ci[1]
                })
        
        return model, pd.DataFrame(coef_data)
        
    except Exception as e:
        return None, None


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_event_study_improved(coef_df, indicator, int_type, buffer, output_path):
    """Create improved event study plot."""
    if coef_df is None or len(coef_df) < 3:
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    outcome_info = ALL_OUTCOMES.get(indicator, {'name': indicator.upper()})
    
    # Shaded regions
    ax.axvspan(-7.5, -0.5, alpha=0.1, color='blue', label='Pre-treatment')
    ax.axvspan(-0.5, 13.5, alpha=0.1, color='green', label='Post-treatment')
    
    # Zero and treatment lines
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.axvline(-0.5, color='red', linestyle='--', linewidth=2, label='Intervention Start')
    
    # CI fill and line
    ax.fill_between(coef_df['rel_month'], coef_df['ci_lower'], coef_df['ci_upper'],
                    alpha=0.3, color='#2c3e50')
    ax.plot(coef_df['rel_month'], coef_df['coef'], 'o-', 
            color='#2c3e50', linewidth=2, markersize=8, label='Point Estimate (95% CI)')
    
    # Highlight significant
    for _, row in coef_df.iterrows():
        if row['ci_lower'] > 0 or row['ci_upper'] < 0:
            ax.plot(row['rel_month'], row['coef'], 'o', color='#e74c3c', markersize=10, zorder=5)
    
    ax.set_xlabel('Months Relative to Intervention Start', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Effect on {outcome_info["name"]}', fontsize=12, fontweight='bold')
    ax.set_title(f'Event Study: {outcome_info["name"]} — {int_type}\n'
                 f'Conflict Buffer: {buffer}m | Donut Controls: {INNER_BUFFER_M}m-{OUTER_BUFFER_M}m',
                 fontsize=14, fontweight='bold')
    
    xticks = list(range(-7, 14))
    xlabels = ['<-6' if x == -7 else '>12' if x == 13 else str(x) for x in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_xlim(-7.5, 13.5)
    
    ax.legend(loc='upper left', frameon=True, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_buffer_comparison(results_df, output_path):
    """Create comparison plot across buffer zones."""
    if results_df.empty:
        return
    
    # Filter for overall treatment effects
    overall = results_df[results_df['intervention'] == 'Overall (Treated vs Control)'].copy()
    
    if overall.empty:
        print("  No overall effects to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Treatment effects by indicator and buffer
    ax1 = axes[0]
    
    indicators = overall['indicator'].unique()
    x = np.arange(len(indicators))
    width = 0.25
    
    for i, buffer in enumerate(BUFFER_ZONES):
        buf_data = overall[overall['buffer'] == buffer]
        coefs = [buf_data[buf_data['indicator'] == ind]['coef'].values[0] 
                 if ind in buf_data['indicator'].values else 0 
                 for ind in indicators]
        errors = [buf_data[buf_data['indicator'] == ind]['se'].values[0] 
                  if ind in buf_data['indicator'].values else 0 
                  for ind in indicators]
        
        bars = ax1.bar(x + i*width, coefs, width, label=f'{buffer}m', yerr=errors, capsize=3)
        
        # Add significance stars
        for j, (ind, coef) in enumerate(zip(indicators, coefs)):
            row = buf_data[buf_data['indicator'] == ind]
            if not row.empty and row['pval'].values[0] < 0.1:
                stars = row['stars'].values[0]
                ax1.annotate(stars, (x[j] + i*width, coef + errors[j] + 0.01),
                           ha='center', fontsize=8)
    
    ax1.set_ylabel('Treatment Effect (Treated vs Donut Control)', fontsize=11)
    ax1.set_xlabel('Outcome', fontsize=11)
    ax1.set_title(f'Overall Treatment Effects by Buffer Zone\n'
                  f'Donut Controls: {INNER_BUFFER_M}m-{OUTER_BUFFER_M}m from treatment',
                  fontsize=12, fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(indicators, rotation=45, ha='right')
    ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax1.legend(title='Conflict Buffer')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Significance heatmap
    ax2 = axes[1]
    
    # Include heterogeneous effects
    het_results = results_df[results_df['intervention'] != 'Overall (Treated vs Control)']
    
    if not het_results.empty:
        interventions = het_results['intervention'].unique()
        indicators_het = het_results['indicator'].unique()
        
        matrix = np.zeros((len(indicators_het), len(interventions)))
        for i, ind in enumerate(indicators_het):
            for j, int_type in enumerate(interventions):
                subset = het_results[(het_results['indicator'] == ind) & 
                                    (het_results['intervention'] == int_type)]
                if not subset.empty:
                    sig_count = (subset['pval'] < 0.1).sum()
                    matrix[i, j] = sig_count
        
        sns.heatmap(matrix, ax=ax2, annot=True, fmt='.0f',
                    xticklabels=interventions, yticklabels=indicators_het,
                    cmap='YlGn', cbar_kws={'label': 'Significant across buffers (of 3)'})
        ax2.set_title('Robustness: Significance by Intervention Type', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Intervention Type')
        ax2.set_ylabel('Outcome')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")


def create_gdp_sensitivity_table(results_df, output_path):
    """Create GDP impact sensitivity analysis table."""
    viirs_results = results_df[results_df['indicator'] == 'log_viirs'].copy()
    
    if viirs_results.empty:
        print("  No log_viirs results for GDP sensitivity")
        return
    
    sensitivity_rows = []
    
    for _, row in viirs_results.iterrows():
        pct_change = (np.exp(row['coef']) - 1) * 100
        
        for elast_name, elast_val in GDP_ELASTICITIES.items():
            gdp_impact = pct_change * elast_val
            
            sensitivity_rows.append({
                'buffer': row['buffer'],
                'intervention': row['intervention'],
                'nightlight_pct_change': pct_change,
                'elasticity': elast_val,
                'elasticity_scenario': elast_name,
                'gdp_pct_impact': gdp_impact,
                'pval': row['pval'],
                'stars': row['stars']
            })
    
    sens_df = pd.DataFrame(sensitivity_rows)
    sens_df.to_csv(output_path, index=False)
    print(f"  ✓ Saved: {output_path.name}")
    
    # Print summary
    print("\n  GDP Impact Sensitivity Analysis (log_viirs → GDP):")
    print("  " + "="*60)
    for elast_name, elast_val in GDP_ELASTICITIES.items():
        subset = sens_df[sens_df['elasticity_scenario'] == elast_name]
        avg_impact = subset['gdp_pct_impact'].mean()
        print(f"    Elasticity {elast_val} ({elast_name}): Avg GDP impact = {avg_impact:.2f}%")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 70)
    print(" COMPREHENSIVE DiD ANALYSIS WITH DONUT CONTROLS")
    print(f" Treatment-Control Buffer: {INNER_BUFFER_M}m - {OUTER_BUFFER_M}m")
    print(" Conflict Buffers: " + ", ".join([f"{b}m" for b in BUFFER_ZONES]))
    print("=" * 70)
    
    if not HAS_DEPS:
        print("\n⚠ Missing dependencies. Exiting.")
        return
    
    # Load data
    print("\n--- Loading Data ---")
    sites = load_and_prep_sites()
    panel_controls = load_district_panel()
    acled = load_acled()
    rasters = find_all_rasters()
    
    if sites is None:
        print("\n❌ Failed to load sites. Exiting.")
        return
    
    print(f"\n  Rasters found:")
    for ind, dates in rasters.items():
        print(f"    {ind}: {len(dates)} time periods")
    
    # Run analysis for each buffer zone
    all_results = []
    
    for buffer in BUFFER_ZONES:
        print(f"\n{'='*70}")
        print(f" CONFLICT BUFFER ZONE: {buffer}m")
        print("=" * 70)
        
        print(f"\n  Building dataset...")
        df = build_analysis_dataset(sites, panel_controls, acled, rasters, buffer)
        
        if df.empty:
            print(f"  ⚠ Empty dataset for buffer {buffer}m")
            continue
        
        n_treated = len(df[df['group'] == 'Treated']['site_id'].unique())
        n_control = len(df[df['group'] == 'Control']['site_id'].unique())
        print(f"  Dataset: {len(df):,} observations ({n_treated} treated, {n_control} control sites)")
        print(f"  Indicators: {df['indicator'].unique().tolist()}")
        
        # Run TWFE regression
        print(f"\n  Running TWFE regressions...")
        results = run_twfe_regression(df, buffer)
        all_results.append(results)
        
        # Event studies for treated sites
        print(f"\n  Generating event study plots...")
        for indicator in df['indicator'].unique():
            for int_type in df[df['group'] == 'Treated']['type'].unique():
                model, coef_df = run_event_study(df, indicator, int_type, buffer)
                if coef_df is not None and len(coef_df) >= 3:
                    clean_type = int_type.lower().replace(' ', '_')
                    out_path = EVENT_STUDY_DIR / f"event_{indicator}_{clean_type}_{buffer}m.png"
                    plot_event_study_improved(coef_df, indicator, int_type, buffer, out_path)
    
    # Combine and save results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_results.to_csv(TABLES_DIR / "did_comprehensive_results.csv", index=False)
        print(f"\n✓ Saved: {TABLES_DIR}/did_comprehensive_results.csv")
        
        # Summary visualizations
        print("\n--- Creating Summary Visualizations ---")
        plot_buffer_comparison(combined_results, FIGURES_DIR / "buffer_comparison.png")
        create_gdp_sensitivity_table(combined_results, TABLES_DIR / "gdp_sensitivity.csv")
        
        # Summary
        print("\n" + "=" * 70)
        print(" SUMMARY: Overall Treatment Effects (Treated vs Donut Control)")
        print("=" * 70)
        
        overall = combined_results[combined_results['intervention'] == 'Overall (Treated vs Control)']
        if not overall.empty:
            for indicator in overall['indicator'].unique():
                ind_results = overall[overall['indicator'] == indicator]
                avg_coef = ind_results['coef'].mean()
                sig_buffers = ind_results[ind_results['pval'] < 0.1]['buffer'].tolist()
                sig_str = f"sig at {sig_buffers}" if sig_buffers else "not sig"
                print(f"  {indicator:<12}: avg effect = {avg_coef:>8.4f} | {sig_str}")
    
    print("\n" + "=" * 70)
    print(" ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
