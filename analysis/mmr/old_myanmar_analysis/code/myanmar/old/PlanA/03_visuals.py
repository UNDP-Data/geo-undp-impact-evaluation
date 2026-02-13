#!/usr/bin/env python3
"""
03_visuals.py
===================
FINAL VISUALS (Control Group: 25km Default)
1. Maps: Full Country Context.
2. Nightlights: Shows PIXEL-LEVEL variation (Raster).
3. Time Series: Fixed Legend (UNDP/Control) & Grid.
4. Conflict (ACLED) & Rice Prices (WFP).
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import numpy as np
import rasterio
from rasterio.plot import show
from rasterio.mask import mask
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
import glob
import warnings
import os
import re
from difflib import get_close_matches

# --- CONFIGURATION ---
DATA_DIR = "data/myanmar"
EXTERNAL_DIR = "data/myanmar/external"
OUTPUT_DIR = "outputs/myanmar/plots"
VIIRS_DIR = "data/myanmar/viirs_25"
BOUNDARIES_FILE = "data/myanmar/myanmar_boundaries/mmr_admin3.geojson"
PROJECTS_GEOJSON = "data/myanmar/undp/undp_interventions.geojson"
PROJECT_FILE = "data/myanmar/undp/project_data.xlsx"
LOCAL_NL_FILE = "outputs/myanmar/local_nightlights_panel.csv"
REGRESSION_RESULTS = "outputs/myanmar/regression_results_main.csv"

EARTHQUAKE_DATE = pd.Timestamp("2025-03-28")
UNDP_START_DATE = pd.Timestamp("2025-05-01")
MYANMAR_EXTENT = [92.0, 9.5, 101.5, 28.5] 
SPATIAL_RADIUS_DEG = 0.25 # UPDATED: 25km (approx 0.25 deg)
EXCLUDE_MONTHS = ['2025-03']

# --- COLOR CONFIGURATION ---
# --- UPDATED COLOR CONFIGURATION ---
UNDP_COLOR = 'blue'
CONTROL_COLOR = '#A23B72'
OTHER_COLOR = '#e0e0e0'

# Specific Sector Colors (for Time Series logic)
AGRI_COLOR = '#228B22'    # Forest Green
SOLAR_COLOR = '#FFD700'   # Gold
SHELTER_COLOR = '#FF4500' # Orange Red
CASH_COLOR = '#1E90FF'    # Dodger Blue
HEALTH_COLOR = '#32CD32'  # Lime Green
WASH_COLOR = '#00CED1'    # Dark Turquoise

# Map Dots Configuration
SECTOR_MAP = {
    'Solar Energy': {'color': SOLAR_COLOR, 'keywords': ['Solar']},
    'Agriculture': {'color': AGRI_COLOR, 'keywords': ['Agricultural', 'Livestock']},
    'Shelter/Repair': {'color': SHELTER_COLOR, 'keywords': ['Construction', 'Tool kits']},
    'Cash/Finance': {'color': CASH_COLOR, 'keywords': ['Cash', 'Grants', 'Microfinance', 'Voucher']},
    'Health': {'color': HEALTH_COLOR, 'keywords': ['Medicines', 'Medical']},
    'WASH': {'color': WASH_COLOR, 'keywords': ['Water', 'Sanitation']}
}

def apply_treatment_to_panel(df, treatment_df):
    """Applies treatment using clean name matching."""
    if 'clean_name' not in df.columns:
        df['clean_name'] = df['township_name'].apply(clean_name)
    
    treatment_map = dict(zip(
        treatment_df['clean_name'], 
        treatment_df['undp_treated']
    ))
    df['treated'] = df['clean_name'].map(treatment_map).fillna(0).astype(int)
    
    return df

def clean_name(name):
    if pd.isna(name): return ""
    return re.sub(r'[^a-z0-9]', '', str(name).lower())

def calculate_sar_indices(df):
    if 'vh' not in df.columns or 'vv' not in df.columns:
        return df

    vh_lin = 10 ** (df['vh'].clip(lower=-30) / 10.0)
    vv_lin = 10 ** (df['vv'].clip(lower=-30) / 10.0)

    # Existing index
    df['rvi'] = (4 * vh_lin) / (vv_lin + vh_lin)

    # NEW: radar-based NDBI analogue
    df['r_ndbi'] = (vv_lin - vh_lin) / (vv_lin + vh_lin)

    # Other proxies
    df['flood_proxy_vv'] = df['vv']
    df['structure_proxy_vh'] = df['vh']

    return df

def apply_treatment_map(df, target_set):
    if 'name_clean' not in df.columns:
        df['name_clean'] = df['township_name'].apply(clean_name)
    
    return df['name_clean'].isin(target_set).astype(int)

# --- DATA PROCESSING ---
def process_acled_data(boundaries_path):
    acled_path = f"{EXTERNAL_DIR}/acled_myanmar.csv"
    if not Path(acled_path).exists(): return None, None
    try:
        df_acled = pd.read_csv(acled_path)
        if df_acled.empty: return None, None
        
        # Clean Coords
        df_acled = df_acled.dropna(subset=['latitude', 'longitude'])
        
        gdf_acled = gpd.GeoDataFrame(
            df_acled, geometry=gpd.points_from_xy(df_acled.longitude, df_acled.latitude), crs="EPSG:4326"
        )
        
        gdf_bounds = gpd.read_file(boundaries_path)
        if gdf_bounds.crs != "EPSG:4326": gdf_bounds = gdf_bounds.to_crs("EPSG:4326")
        
        # Robust ID Column
        id_col = next((c for c in gdf_bounds.columns if 'pcode' in c.lower()), None)
        if not id_col: id_col = 'geometry' # Fallback
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            joined = gpd.sjoin(gdf_acled, gdf_bounds[[id_col, 'geometry']], how='inner', predicate='intersects')
        
        joined['event_date'] = pd.to_datetime(joined['event_date'])
        joined['period_start'] = joined['event_date'].dt.to_period('M').dt.to_timestamp()
        
        monthly_stats = joined.groupby([id_col, 'period_start']).agg({
            'event_id_cnty': 'count',
            'fatalities': 'sum'
        }).rename(columns={'event_id_cnty': 'conflict_events'}).reset_index()
        
        return monthly_stats, id_col
        
    except: return None, None

def load_and_process_wfp(wfp_path, gee_townships_gdf):
    if not Path(wfp_path).exists(): return None
    print("  > Processing WFP Data (State-Blocked Matching)...")
    
    try:
        # 1. Load WFP Data
        df = pd.read_csv(wfp_path, on_bad_lines='skip', low_memory=False)
        if df.iloc[0].astype(str).str.contains('#').any(): df = df.iloc[1:]
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Identify Price Column
        price_col = next((c for c in df.columns if c in ['price', 'value']), None)
        if not price_col: price_col = next((c for c in df.columns if 'price' in c and 'flag' not in c), None)
        
        req_cols = ['date', 'admin1', 'market', 'commodity', price_col]
        if not all(c in df.columns for c in req_cols): return None

        # Filter for Rice
        df = df[df['commodity'].str.contains('Rice', case=False, na=False)].copy()
        df['price'] = pd.to_numeric(df[price_col], errors='coerce')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['price', 'date'])

        # 2. Reference Data
        gee_ref = gee_townships_gdf[['township_name', 'state_name']].drop_duplicates()
        gee_ref['t_clean'] = gee_ref['township_name'].apply(clean_name)
        gee_ref['s_clean'] = gee_ref['state_name'].apply(clean_name)
        gee_states = gee_ref['s_clean'].unique()
        
        matched_map = {}
        unique_markets = df[['market', 'admin1']].drop_duplicates()
        
        for idx, row in unique_markets.iterrows():
            market_raw = str(row['market'])
            state_raw = str(row['admin1'])
            m_clean = clean_name(market_raw)
            s_clean = clean_name(state_raw)
            
            # Match State
            state_matches = get_close_matches(s_clean, gee_states, n=1, cutoff=0.7)
            if state_matches:
                candidates = gee_ref[gee_ref['s_clean'] == state_matches[0]]['t_clean'].tolist()
            else:
                candidates = gee_ref['t_clean'].tolist()

            # Match Township
            matches = get_close_matches(m_clean, candidates, n=1, cutoff=0.6)
            if matches: matched_map[market_raw] = matches[0]

        df['township_clean'] = df['market'].map(matched_map)
        print(f"    -> Successfully matched {df['township_clean'].nunique()} markets.")
        
        df = df.dropna(subset=['township_clean'])
        df['period_start'] = df['date'].dt.to_period('M').dt.to_timestamp()
        
        # Aggregate
        panel = df.groupby(['township_clean', 'period_start'])['price'].mean().reset_index()
        panel = panel.rename(columns={'price': 'price_rice', 'township_clean': 'name_clean'})
        return panel
        
    except Exception as e:
        print(f"  ! WFP Error: {e}")
        return None

def load_clean_project_data():
    """Load project data with intervention types for visualizations."""
    project_path = "data/myanmar/undp/project_data.xlsx"
    if not Path(project_path).exists():
        print("  ! Project data file not found, using treatment file instead")
        treatment_path = "outputs/myanmar/treatment_undp_final.csv"
        if Path(treatment_path).exists():
            return pd.read_csv(treatment_path)
        return pd.DataFrame()
    
    # Load and process project data
    df = pd.read_excel(project_path)
    
    # Find township column (flexible naming)
    township_col = None
    for col in df.columns:
        if any(x in col.lower() for x in ['township', 'town', 'location', 'admin3']):
            township_col = col
            break
    
    if township_col is None:
        print(f"  ! Could not find township column in {project_path}")
        print(f"  ! Available columns: {list(df.columns)}")
        return pd.DataFrame()
    
    # Find type/category column (flexible naming)
    type_col = None
    for col in df.columns:
        if any(x in col.lower() for x in ['type', 'category', 'sector', 'intervention']):
            type_col = col
            break
    
    if type_col is None:
        print(f"  ! Could not find intervention type column in {project_path}")
        print(f"  ! Available columns: {list(df.columns)}")
        return pd.DataFrame()
    
    # Create clean names
    df['clean_name'] = df[township_col].apply(clean_name)
    
    # Mark as UNDP treated
    df['undp_treated'] = 1
    
    # Intervention type indicators (matching script 01 logic)
    df['is_solar'] = df[type_col].str.contains('Solar', case=False, na=False).astype(int)
    df['is_shelter'] = df[type_col].str.contains('Shelter', case=False, na=False).astype(int)
    df['is_agri'] = df[type_col].str.contains('Agri', case=False, na=False).astype(int)
    df['is_cash'] = df[type_col].str.contains('Cash', case=False, na=False).astype(int)
    df['is_health'] = df[type_col].str.contains('Health', case=False, na=False).astype(int)
    df['is_wash'] = df[type_col].str.contains('Wash|Water|Sanitation', case=False, na=False).astype(int)
    df['is_training'] = df[type_col].str.contains('Training', case=False, na=False).astype(int)
    
    return df

def normalize_names(series):
    return series.astype(str).str.lower().str.strip().str.replace(r'[^\w\s]', '', regex=True)

def load_undp_projects():
    if not Path(PROJECTS_GEOJSON).exists(): return None
    gdf = gpd.read_file(PROJECTS_GEOJSON)
    cat_col = next((c for c in gdf.columns if 'type' in c.lower() or 'sector' in c.lower()), None)
    if cat_col: gdf = gdf.rename(columns={cat_col: 'category'})
    else: gdf['category'] = 'Other'
    return gdf

# Unused function removed

def get_spatial_subset(df, radius_deg, treatment_col='treated'):
    if not Path(BOUNDARIES_FILE).exists(): return df
    gdf = gpd.read_file(BOUNDARIES_FILE)
    id_col = next((c for c in gdf.columns if 'pcode' in c.lower()), None)
    treated_ids = df[df[treatment_col] == 1]['township_id'].unique()
    treated_geoms = gdf[gdf[id_col].isin(treated_ids)]
    if treated_geoms.empty: return df
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        treated_buffer = treated_geoms.geometry.buffer(radius_deg).unary_union
    neighbors = gdf[gdf.geometry.intersects(treated_buffer)]
    valid_ids = np.concatenate([treated_ids, neighbors[id_col].unique()])
    return df[df['township_id'].isin(valid_ids)].copy()

def plot_study_design_map(df, output_path):
    print("  > Drawing Study Design Map...")
    if not Path(BOUNDARIES_FILE).exists(): return
    gdf = gpd.read_file(BOUNDARIES_FILE)
    id_col = next((c for c in gdf.columns if 'pcode' in c.lower()), None)
    
    treated_ids = df[df['treated'] == 1]['township_id'].unique()
    treated_gdf = gdf[gdf[id_col].isin(treated_ids)]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        treated_buffer = treated_gdf.geometry.buffer(SPATIAL_RADIUS_DEG).unary_union
    
    neighbor_mask = gdf.geometry.intersects(treated_buffer) & (~gdf[id_col].isin(treated_ids))
    neighbor_gdf = gdf[neighbor_mask]
    other_gdf = gdf[~gdf[id_col].isin(treated_ids) & ~gdf[id_col].isin(neighbor_gdf[id_col])]
    
    fig, ax = plt.subplots(figsize=(10, 12), facecolor='white')
    other_gdf.plot(ax=ax, facecolor=OTHER_COLOR, edgecolor='white', linewidth=0.1)
    neighbor_gdf.plot(ax=ax, facecolor=CONTROL_COLOR, alpha=0.6, edgecolor='white', linewidth=0.3)
    treated_gdf.plot(ax=ax, facecolor=UNDP_COLOR, alpha=0.9, edgecolor='white', linewidth=0.3)
    
    # UPDATED LABELS for 25km
    legend_elements = [Patch(facecolor=UNDP_COLOR, label=f'Treated (N={len(treated_gdf)})'),
                       Patch(facecolor=CONTROL_COLOR, label=f'Control 25km (N={len(neighbor_gdf)})'),
                       Patch(facecolor=OTHER_COLOR, label='Excluded')]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=12, frameon=False)
    ax.set_title("Study Design: 25km Spatial Control Group", fontsize=16, fontweight='bold')
    ax.set_axis_off()
    ax.set_xlim(MYANMAR_EXTENT[0], MYANMAR_EXTENT[2]); ax.set_ylim(MYANMAR_EXTENT[1], MYANMAR_EXTENT[3])
    plt.savefig(output_path, dpi=300, bbox_inches='tight'); plt.close()

def draw_project_dots(ax, gdf):
    if gdf is None or gdf.empty: return

    # 1. Plot "Other" (Base Layer) - anything that doesn't match specific sectors
    # We plot all points as 'Other' first to ensure coverage
    ax.scatter(gdf.geometry.x, gdf.geometry.y, c='#FFFFFF', s=15, 
               edgecolors='#999999', linewidth=0.2, zorder=5, label='Other')

    # 2. Plot Specific Sectors (Layered on top)
    legend_handles = []
    
    # Iterate through our defined sectors
    for label, props in SECTOR_MAP.items():
        # Create a regex pattern from the keywords (e.g., "Solar|Energy|Electr")
        pattern = "|".join(props['keywords'])
        
        # Filter the GeoDataFrame for matches
        subset = gdf[gdf['category'].astype(str).str.contains(pattern, case=False, na=False)]
        
        if not subset.empty:
            ax.scatter(subset.geometry.x, subset.geometry.y, 
                       c=props['color'], s=20, edgecolors='black', linewidth=0.5, 
                       zorder=10)
            
            # Add to legend
            legend_handles.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor=props['color'], 
                       markersize=8, markeredgecolor='black', label=label)
            )

    # 3. Create Custom Legend
    if legend_handles:
        leg = ax.legend(handles=legend_handles, loc='upper right', fontsize=8, 
                        title="UNDP Projects", facecolor='white', edgecolor='gray', 
                        framealpha=0.9)
        leg.get_title().set_weight('bold')
        
def plot_3panel_map(df, gdf, outcome, title, output_path, undp_gdf=None, mode='vector', valid_ids=None):
    print(f"  > Drawing Map: {title} ({mode})...")
    if outcome not in df.columns: return
    dates = ['2025-02', '2025-04', '2025-09']
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), facecolor='white') 
    plt.subplots_adjust(wspace=0.05)

    if outcome == 'nightlights':
        colors = ["#000000", "#0a0a0a", "#443300", "#ffaa00", "#ffff00", "#ffffff"]
        cmap = LinearSegmentedColormap.from_list("electric_gold", list(zip([0,0.15,0.3,0.55,0.8,1], colors)))
        norm = LogNorm(vmin=0.05, vmax=35) 
    elif 'conflict' in outcome:
        cmap = plt.cm.Reds
        vals = df[outcome].dropna()
        norm = Normalize(vmin=0, vmax=vals.quantile(0.99) if not vals.empty else 1)
    elif 'price' in outcome:
        cmap = plt.cm.YlOrRd
        vals = df[outcome].dropna()
        norm = Normalize(vmin=vals.min() if not vals.empty else 0, vmax=vals.max() if not vals.empty else 1)
    else:
        vals = df[outcome].dropna()
        if vals.empty: return
        vmin, vmax = vals.quantile([0.02, 0.98])
        norm = Normalize(vmin=vmin, vmax=vmax)
        if outcome in ['ndvi', 'rvi']: cmap = plt.cm.Greens
        elif outcome in ['rainfall', 'flood_proxy_vv']: cmap = plt.cm.Blues
        elif outcome in ['ndbi', 'r_ndbi', 'structure_proxy_vh']: cmap = plt.cm.Reds
        else: cmap = plt.cm.viridis

    subset_geom = None
    if valid_ids is not None:
        id_col = next((c for c in gdf.columns if 'pcode' in c.lower()), None)
        subset_gdf = gdf[gdf[id_col].isin(valid_ids)]
        if not subset_gdf.empty: subset_geom = subset_gdf.geometry

    for i, d in enumerate(dates):
        ax = axes[i]
        ax.set_axis_off()
        ax.set_title(d, fontsize=16, fontweight='bold', color='black')
        gdf.plot(ax=ax, facecolor='#f5f5f5', edgecolor='#e0e0e0', linewidth=0.5, zorder=1)
        
        if outcome == 'nightlights' and mode == 'raster':
            tif_search = glob.glob(f"{VIIRS_DIR}/*{d.replace('-','')}*.tif")
            if tif_search and subset_geom is not None:
                try:
                    with rasterio.open(tif_search[0]) as src:
                        out_image, out_transform = mask(src, subset_geom, crop=True, nodata=0)
                        out_image = out_image[0]
                        show(out_image, transform=out_transform, ax=ax, cmap=cmap, norm=norm, zorder=2)
                except: pass
        else:
            mdata = df[df['period_start'].astype(str).str.startswith(d)].copy()
            gdf['norm_name'] = normalize_names(gdf.get('adm3_name', gdf.get('township_name')))
            mdata['norm_name'] = normalize_names(mdata['township_name'])
            merged = gdf.merge(mdata, on='norm_name', how='left')
            if valid_ids is not None:
                id_col = next((c for c in merged.columns if 'pcode' in c.lower()), None)
                merged = merged[merged[id_col].isin(valid_ids)]
            if not merged.dropna(subset=[outcome]).empty:
                merged.plot(column=outcome, ax=ax, cmap=cmap, norm=norm, zorder=2)
        if i == 2 and undp_gdf is not None:
            draw_project_dots(ax, undp_gdf)
        ax.set_xlim(MYANMAR_EXTENT[0], MYANMAR_EXTENT[2])
        ax.set_ylim(MYANMAR_EXTENT[1], MYANMAR_EXTENT[3])

    cbar_ax = fig.add_axes([0.3, 0.05, 0.4, 0.02])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, orientation='horizontal')
    cb.set_label(f"{title}", fontsize=12, color='black')
    plt.suptitle(f"2025 Myanmar Earthquake: {title}", fontsize=22, fontweight='bold', y=0.95, color='black')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white'); plt.close()

def plot_time_series(df, col, title, ylabel, output_path, treated_col='treated', group_label='UNDP', color='blue'):
    """
    Plots time series trends with a specific visual logic:
    - Control: Always Pink/Magenta (Background reference).
    - Treated: Continuous line, gray before UNDP arrival (May 2025), colored after.
    - Includes standard error bands
    """
    
    # 1. Prepare Data - Mean and Standard Error
    grouped = df.groupby(['period_start', treated_col])[col].agg(['mean', 'sem']).unstack()
    
    # Apply rolling window to smooth
    daily_mean = grouped['mean'].rolling(window=2, min_periods=1).mean() 
    daily_sem = grouped['sem'].rolling(window=2, min_periods=1).mean()
    
    # Separate Pre-UNDP and Post-UNDP datasets for the Treated group
    pre_undp_mask = daily_mean.index < UNDP_START_DATE
    post_undp_mask = daily_mean.index >= UNDP_START_DATE
    
    fig, ax = plt.subplots(figsize=(10, 6))

    # 2. Plot CONTROL Group (Background) with confidence band
    if 0 in daily_mean.columns:
        ax.plot(daily_mean.index, daily_mean[0], color=CONTROL_COLOR, linestyle='--', 
                linewidth=1.5, alpha=0.6, label='Control (Non-Project)')
        
        # Add confidence band for control
        if 0 in daily_sem.columns:
            ax.fill_between(daily_mean.index, 
                           daily_mean[0] - 1.96 * daily_sem[0],
                           daily_mean[0] + 1.96 * daily_sem[0],
                           color=CONTROL_COLOR, alpha=0.2)

    # 3. Plot TREATED Group (Continuous with color change) with confidence bands
    if 1 in daily_mean.columns:
        # A. Pre-UNDP Period -> Neutral Gray (no label to avoid cluttering legend)
        pre_undp_inclusive = daily_mean.index <= UNDP_START_DATE
        ax.plot(daily_mean.index[pre_undp_inclusive], daily_mean.loc[pre_undp_inclusive, 1], 
                color='gray', linestyle='-', linewidth=2, alpha=0.7)
        
        # Add confidence band for pre-UNDP period
        if 1 in daily_sem.columns:
            ax.fill_between(daily_mean.index[pre_undp_inclusive],
                           daily_mean.loc[pre_undp_inclusive, 1] - 1.96 * daily_sem.loc[pre_undp_inclusive, 1],
                           daily_mean.loc[pre_undp_inclusive, 1] + 1.96 * daily_sem.loc[pre_undp_inclusive, 1],
                           color='gray', alpha=0.2)
        
        # B. Post-UNDP Period -> Active Color with label
        active_label = f"{group_label}" if group_label != 'UNDP' else "UNDP"
        
        ax.plot(daily_mean.index[post_undp_mask], daily_mean.loc[post_undp_mask, 1], 
                color=color, linestyle='-', linewidth=3, marker='o', markersize=4,
                label=active_label)
        
        # Add confidence band for post-UNDP period
        if 1 in daily_sem.columns:
            ax.fill_between(daily_mean.index[post_undp_mask],
                           daily_mean.loc[post_undp_mask, 1] - 1.96 * daily_sem.loc[post_undp_mask, 1],
                           daily_mean.loc[post_undp_mask, 1] + 1.96 * daily_sem.loc[post_undp_mask, 1],
                           color=color, alpha=0.2)

    # 4. Formatting
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel('date', fontsize=11)
    
    # Vertical Line for Earthquake
    ax.axvline(EARTHQUAKE_DATE, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='Earthquake')

    # Grid & Legend
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='best', fontsize=10, frameon=True, facecolor='white', framealpha=0.9)
    
    # Date Formatting on X-Axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  > Saved Time Series: {output_path}")
    
def get_regression_stats(outcome, sample="Spatial 25km"):
    """
    Returns formatted coefficient string with significance stars from regression results.
    Handles various possible column name formats.
    """
    if not Path(REGRESSION_RESULTS).exists():
        print(f"    ! Regression results file not found: {REGRESSION_RESULTS}")
        return None
    
    try:
        df_res = pd.read_csv(REGRESSION_RESULTS)
        print(f"    > Loaded regression results, columns: {list(df_res.columns)}")
        
        # 1. Filter for the outcome and sample
        # Try different possible column name variations
        outcome_col = next((c for c in df_res.columns if c.lower() in ['outcome', 'dependent', 'y']), 'Outcome')
        sample_col = next((c for c in df_res.columns if c.lower() in ['sample', 'subsample', 'specification']), 'Sample')
        controls_col = next((c for c in df_res.columns if c.lower() in ['controls', 'control', 'covariates']), 'Controls')
        
        # Select models that include "rainfall" (The full model)
        row = df_res[
            (df_res[outcome_col] == outcome) & 
            (df_res[sample_col] == sample) & 
            (df_res[controls_col].astype(str).str.contains("rainfall", case=False, na=False))
        ]
        
        # 2. Fallback: If no specific control model found, take whatever is available
        if row.empty: 
            row = df_res[(df_res[outcome_col] == outcome) & (df_res[sample_col] == sample)]
        
        if row.empty:
            print(f"    ! No regression results found for outcome='{outcome}', sample='{sample}'")
            return None
        
        # 3. Select the last row to ensure the numbers match the regression table
        row = row.drop_duplicates(keep='last').iloc[-1]
        
        # Get coefficient - try different column names
        coeff_col = next((c for c in df_res.columns if c.lower() in ['coeff', 'coefficient', 'estimate', 'beta']), 'Coeff')
        coeff = row[coeff_col]
        
        # Add significance stars based on p-value if available
        stars = ""
        pval_col = next((c for c in df_res.columns if c.lower() in ['p-value', 'p_value', 'pval', 'p', 'pvalue']), None)
        
        if pval_col and pval_col in row.index:
            pval = row[pval_col]
            if pd.notna(pval):
                if pval < 0.01:
                    stars = "***"
                elif pval < 0.05:
                    stars = "**"
                elif pval < 0.1:
                    stars = "*"
        
        coeff_str = f"{coeff:.4f}{stars}"
        print(f"    > Stats found for {outcome} ({sample}): {coeff_str}")
        return coeff_str
            
    except Exception as e: 
        print(f"    ! Error reading regression stats: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_did_annotated(df, outcome, title, ylabel, output_path, treatment_col, treat_label, treat_color):
    if outcome not in df.columns or df[treatment_col].sum() == 0: return
    print(f"  > DiD Plot: {title}")
    
    # Get Data using updated Radius (25km)
    df_subset = get_spatial_subset(df, SPATIAL_RADIUS_DEG, treatment_col)
    pre = df_subset[(df_subset['post']==0) & (pd.to_datetime(df_subset['period_start']).dt.year==2025)]
    post = df_subset[df_subset['post']==1]
    
    # Calculate Means
    y_tp = pre[pre[treatment_col]==1][outcome].mean()
    y_t_post = post[post[treatment_col]==1][outcome].mean()
    y_cp = pre[pre[treatment_col]==0][outcome].mean()
    y_c_post = post[post[treatment_col]==0][outcome].mean()
    
    if pd.isna(y_tp): return
    y_cf = y_tp + (y_c_post - y_cp)
    
    # Calculate simple DiD effect
    simple_did = (y_t_post - y_tp) - (y_c_post - y_cp)
    
    # Try to get regression stats (with controls)
    coef_str = get_regression_stats(outcome)
    
    # Plot
    fig, ax = plt.subplots(figsize=(9, 7), facecolor='white')
    ax.plot([0, 1], [y_tp, y_t_post], 'o-', color=treat_color, lw=3, markersize=10, label=treat_label)
    ax.plot([0, 1], [y_cp, y_c_post], 's-', color=CONTROL_COLOR, lw=3, markersize=10, label='Control')
    ax.plot([0, 1], [y_tp, y_cf], '--', color='#7FB2F0', lw=2, label='Counterfactual')
    ax.annotate('', xy=(1, y_t_post), xytext=(1, y_cf), arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Add Text Box - Always show effect, prefer regression if available
    if coef_str is not None:
        # Use regression coefficient with controls
        text_str = (f"Adjusted DiD Effect\n(Spatial 25km + Controls)\nCoef: {coef_str}")
    else:
        # Fallback to simple DiD from the plot
        text_str = (f"DiD Effect\n(Simple Difference)\nCoef: {simple_did:.4f}")
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='red')
    ax.text(0.5, 0.85, text_str, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='center', bbox=props, color='red', fontweight='bold')

    ax.set_xticks([0, 1]); ax.set_xticklabels(['Pre', 'Post'])
    ax.set_title(title, fontweight='bold')
    plt.savefig(output_path, dpi=300, bbox_inches='tight'); plt.close()

def compute_simple_event_study(df, outcome, treatment_col='treated', window=6):
    """
    Simple ES: mean(treated - control) by months_since_eq with delta-method SE.
    Used as fallback when pre-computed ES CSV is missing.
    """
    if outcome not in df.columns or treatment_col not in df.columns:
        return pd.DataFrame()

    df_mod = df.dropna(subset=[outcome]).copy()
    df_mod['months_since_eq'] = (
        (df_mod['period_start'].dt.year - EARTHQUAKE_DATE.year) * 12 +
        (df_mod['period_start'].dt.month - EARTHQUAKE_DATE.month)
    )

    df_mod = df_mod[(df_mod['months_since_eq'] >= -window) &
                    (df_mod['months_since_eq'] <= window)]

    rows = []
    for t in range(-window, window + 1):
        # keep -1 (pre period) so we can show it explicitly
        subset = df_mod[df_mod['months_since_eq'] == t]
        if subset.empty:
            continue

        g_t = subset[subset[treatment_col] == 1][outcome]
        g_c = subset[subset[treatment_col] == 0][outcome]
        if g_t.empty or g_c.empty:
            continue

        diff = g_t.mean() - g_c.mean()
        # delta-method SE: Var(T)/n_T + Var(C)/n_C
        se = np.sqrt(
            g_t.var(ddof=1) / len(g_t) +
            g_c.var(ddof=1) / len(g_c)
        )
        rows.append({'t': t, 'coef': diff, 'err': se})

    return pd.DataFrame(rows)

def plot_event_study_generic(df, outcome, title, output_path):
    csv_path = Path("outputs/myanmar/robustness") / f"es_coeffs_{outcome}.csv"

    if csv_path.exists():
        print(f"  > Event Study (from CSV): {title} ({csv_path.name})")
        es_df = pd.read_csv(csv_path)
    else:
        print(f"  ! Event Study CSV not found for '{outcome}' at {csv_path}; "
              f"computing simple ES from panel data instead.")
        es_df = compute_simple_event_study(df, outcome, treatment_col='treated', window=6)

    if es_df.empty or not {'t', 'coef', 'err'}.issubset(es_df.columns):
        print(f"  ! Event Study: no valid data for '{outcome}' – skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    ax.errorbar(
        es_df['t'], es_df['coef'], yerr=1.96 * es_df['err'],
        fmt='o-', linewidth=2, markeredgewidth=0.5,
        color=UNDP_COLOR, ecolor=UNDP_COLOR,
        elinewidth=2, capsize=4, markersize=8
    )
    ax.axhline(0, color='black', alpha=0.5)
    ax.axvline(-1, color='red', linestyle='--')

    ax.set_title(f"Dynamic Effect: {title}", fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_google_trends_ts(output_path):
    gt_path = f"{EXTERNAL_DIR}/google_trends_mm.csv"
    manual_anxiety_path = f"{EXTERNAL_DIR}/google_trends_anxiety.csv"
    
    print("  > Generating Separate Google Trends Plots")
    
    # --- 1. LOAD DATA ---
    df = pd.DataFrame()
    
    # Load Main API Data
    if Path(gt_path).exists():
        try:
            df = pd.read_csv(gt_path)
            df['date'] = pd.to_datetime(df['date'])
        except Exception as e: print(f"    ! Error reading GT API data: {e}")
    
    # Load Manual Anxiety Data
    if Path(manual_anxiety_path).exists():
        try:
            # Skip header rows standard in manual downloads
            df_man = pd.read_csv(manual_anxiety_path, skiprows=2)
            df_man.columns = ['date', 'score']
            df_man['keyword'] = 'Anxiety' # Assign English keyword directly
            df_man['date'] = pd.to_datetime(df_man['date'])
            df = pd.concat([df, df_man], ignore_index=True)
            print("    -> Integrated manual Anxiety data.")
        except Exception as e: print(f"    ! Error reading manual anxiety file: {e}")

    if df.empty: return

    # --- 2. DEFINE FOCUS KEYWORDS ---
    # Strictly the English terms you collected
    focus_keywords = ['Earthquake', 'Donation', 'Passport', 'Anxiety']
    
    # --- 3. GENERATE PLOTS ---
    for kw in focus_keywords:
        # Filter data (No translation map needed anymore)
        df_kw = df[df['keyword'] == kw].copy()
        
        if df_kw.empty: 
            print(f"    ! Warning: No data found for '{kw}'")
            continue
        
        # Create individual plot
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
        
        # Plot Line
        sns.lineplot(data=df_kw, x='date', y='score', color='#1f77b4', linewidth=2.5, ax=ax)
        
        # Add Earthquake Line
        ax.axvline(EARTHQUAKE_DATE, color='red', linestyle='--', lw=1.5, label='Earthquake Date')
        
        # Formatting
        ax.set_ylabel("Search Interest (0-100)")
        ax.set_title(f"Public Sentiment: {kw}", fontweight='bold', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        
        # Save with unique filename
        clean_kw = re.sub(r'[^a-zA-Z0-9]', '', kw).lower()
        fname = output_path.replace("google_trends.png", f"google_trends_{clean_kw}.png")
        
        # Fallback if filename format is different in main()
        if fname == output_path: 
            fname = output_path.replace(".png", f"_{clean_kw}.png")
        
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    -> Saved plot: {fname}")

def load_and_clean_data():
    print("Loading Data for Visuals...")
    
    # 1. Load GEE Panel Files
    files = glob.glob(f"{DATA_DIR}/gee_panel_myanmar_PART*.csv")
    if not files: return pd.DataFrame()

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    
    # Rename Columns
    rename_map = {'admin3_id': 'township_id', 'admin3_nam': 'township_name', 
                  'NDVI': 'ndvi', 'NDBI': 'ndbi', 'RAIN_TOTAL': 'rainfall', 
                  'NL_AVG_RAD': 'nightlights_gee', # Store GEE as backup
                  'VH': 'vh', 'VV': 'vv'}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    
    # Fallback renaming
    if 'nightlights_gee' not in df.columns and 'nightlights' in df.columns:
        df = df.rename(columns={'nightlights': 'nightlights_gee'})
        
    df['period_start'] = pd.to_datetime(df['period_start'])
    
    # Create clean_name immediately
    df['clean_name'] = df['township_name'].apply(clean_name)
    df['name_clean'] = df['clean_name'] 

    # --- FIX: MERGE LOCAL DATA (Primary Source) ---
    if 'LOCAL_NL_FILE' in globals() and Path(LOCAL_NL_FILE).exists():
        print("    -> Merging Local Nightlights (April-Sept Data)...")
        df_local = pd.read_csv(LOCAL_NL_FILE)
        df_local['period_start'] = pd.to_datetime(df_local['period_start'])
        df_local['name_clean'] = df_local['township_name'].apply(clean_name)
        
        # Outer join to keep new dates
        df = pd.merge(df, df_local[['name_clean', 'period_start', 'nightlights']], 
                      on=['name_clean', 'period_start'], 
                      how='outer', 
                      suffixes=('_gee', '')) 
        
        # PRIORITIZE Local Data. If missing, use GEE.
        if 'nightlights_gee' in df.columns:
            df['nightlights'] = df['nightlights'].combine_first(df['nightlights_gee'])
            
        # CRITICAL FIX: Backfill 'township_name' for the new rows
        # The new rows have 'name_clean' but 'township_name' is NaN.
        # We fill NaN with the clean name so the plotter has something to use.
        df['township_name'] = df['township_name'].fillna(df['name_clean'].str.title())
            
    else:
        df['nightlights'] = df['nightlights_gee']

    # Sort
    df = df.sort_values(['name_clean', 'period_start'])

# 2. Map Treatment (Robust Version)
    proj_df = load_clean_project_data()
    
    # Default treatment columns to 0
    for col in ['treated', 'treated_agri', 'treated_solar', 'treated_shelter']:
        df[col] = 0
    
    if not proj_df.empty:
        # 1. Base Treatment Map (Safe check for column existence)
        if 'undp_treated' in proj_df.columns:
            treated_base = proj_df[proj_df['undp_treated'] == 1]
        else:
            # Fallback: assume all rows in this file are treated if column missing
            treated_base = proj_df
            
        treated_set = set(treated_base['clean_name'])
        df['treated'] = apply_treatment_map(df, treated_set)

        # 2. Sector Maps (Safe check for column existence)
        # Helper function to get sector sets safely
        def get_sector_set(dframe, col_name):
            if col_name in dframe.columns:
                return set(dframe[dframe[col_name] == 1]['clean_name'])
            return set() # Return empty set if column missing

        # Apply specific sectors safely
        df['treated_agri'] = apply_treatment_map(df, get_sector_set(treated_base, 'is_agri'))
        df['treated_solar'] = apply_treatment_map(df, get_sector_set(treated_base, 'is_solar'))
        df['treated_shelter'] = apply_treatment_map(df, get_sector_set(treated_base, 'is_shelter'))
        
    # 3. Merge ACLED Conflict Data
    acled_stats, acled_id_col = process_acled_data(BOUNDARIES_FILE)
    if acled_stats is not None and acled_id_col:
        print("    -> Merging ACLED Conflict Data...")
        df = pd.merge(df, acled_stats, 
                     left_on=['township_id', 'period_start'],
                     right_on=[acled_id_col, 'period_start'],
                     how='left')
        # Fill missing conflict events with 0
        df['conflict_events'] = df['conflict_events'].fillna(0)
        if 'fatalities' in df.columns:
            df['fatalities'] = df['fatalities'].fillna(0)

    # 4. Merge WFP
    if Path(BOUNDARIES_FILE).exists():
        gdf_bounds = gpd.read_file(BOUNDARIES_FILE)
        rename_geo = {'adm3_name': 'township_name', 'adm1_name': 'state_name'}
        gdf_bounds = gdf_bounds.rename(columns=rename_geo)
        
        wfp_df = load_and_process_wfp(f"{EXTERNAL_DIR}/wfp_prices.csv", gdf_bounds)
        if wfp_df is not None:
            df = pd.merge(df, wfp_df, on=['name_clean', 'period_start'], how='left')
            # Interpolate Prices by Clean Name (More robust)
            df['price_rice'] = df.groupby('name_clean')['price_rice'].transform(
                lambda x: x.interpolate(method='linear').ffill().bfill()
            )

    # 5. Clean & Transform
    df['ym_str'] = df['period_start'].dt.strftime('%Y-%m')
    df = df[~df['ym_str'].isin(EXCLUDE_MONTHS)].copy()
    
    # Recalculate Indices (NDVI/RVI might be NaN for new months, that's okay, NL is the focus)
    df = calculate_sar_indices(df)
    df['post'] = (df['period_start'] > EARTHQUAKE_DATE).astype(int)
    
    return df

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    df = load_and_clean_data()
    boundaries = gpd.read_file(BOUNDARIES_FILE)
    undp_gdf = load_undp_projects()
    
    # Calculate Subset for Maps (now 25km)
    subset_df = get_spatial_subset(df, SPATIAL_RADIUS_DEG, 'treated')
    valid_ids_25km = subset_df['township_id'].unique()

    plot_study_design_map(df, f"{OUTPUT_DIR}/map_study_design.png")

    indicators = [
        ('nightlights', 'Nightlights', 'Radiance', 'raster', 'nl'),
        ('ndvi', 'Vegetation (NDVI)', 'NDVI', 'vector', 'ndvi'),
        ('rvi', 'Radar Vegetation (RVI)', 'RVI', 'vector', 'rvi'),
        ('ndbi', 'Construction (NDBI)', 'NDBI', 'vector', 'ndbi'),
        ('r_ndbi', 'Radar Construction (R-NDBI)', 'R-NDBI', 'vector', 'rndbi'),
        ('rainfall', 'Rainfall', 'mm', 'vector', 'rain'),
        ('flood_proxy_vv', 'Flood Proxy (VV)', 'dB', 'vector', 'flood'),
        ('structure_proxy_vh', 'Structure Proxy (VH)', 'dB', 'vector', 'vh'),
        ('conflict_events', 'Conflict Intensity', 'Events/Month', 'vector', 'conflict'),
        ('price_rice', 'Rice Market Price', 'MMK/kg', 'vector', 'price'),
    ]

    # LOOP OVER ALL OUTCOMES (incl. r_ndbi, rainfall, flood) + EVENT STUDY
    for col, title, ylabel, mode, suffix in indicators:
        print(f"\nProcessing {title}...")

        if col not in df.columns:
            print(f"  ! Skipping {title}: Column '{col}' not found.")
            continue

        # 1. Clean map
        plot_3panel_map(
            df, boundaries, col, title,
            f"{OUTPUT_DIR}/map_{suffix}_clean.png",
            mode=mode, valid_ids=valid_ids_25km
        )

        # 2. Map with project locations
        plot_3panel_map(
            df, boundaries, col, title,
            f"{OUTPUT_DIR}/map_{suffix}_projects.png",
            undp_gdf, mode=mode, valid_ids=valid_ids_25km
        )

        # 3. Time series
        plot_time_series(
            df, col, f"{title} Trends", ylabel,
            f"{OUTPUT_DIR}/ts_{suffix}.png",
            'treated', 'UNDP', UNDP_COLOR
        )

        # 4. DiD
        plot_did_annotated(
            df, col, title, ylabel,
            f"{OUTPUT_DIR}/did_{suffix}.png",
            'treated', 'UNDP', UNDP_COLOR
        )

        # 5. Event Study
        plot_event_study_generic(
            df, col, title,
            f"{OUTPUT_DIR}/es_{suffix}.png"
            )
    
    # SECTOR-SPECIFIC PLOTS (run once, outside the loop)
    print("\n-- SECTORS --")
    plot_time_series(
        df, 'rvi', 'Agri Impact (RVI)', 'RVI',
        f"{OUTPUT_DIR}/ts_rvi_agri.png",
        'treated_agri', 'Agri', AGRI_COLOR
    )
    plot_did_annotated(
        df, 'rvi', 'Agri Impact (RVI)', 'RVI',
        f"{OUTPUT_DIR}/did_rvi_agri.png",
        'treated_agri', 'Agri', AGRI_COLOR
    )

    plot_time_series(
        df, 'nightlights', 'Solar Impact (Lights)', 'Radiance',
        f"{OUTPUT_DIR}/ts_nl_solar.png",
        'treated_solar', 'Solar', SOLAR_COLOR
    )
    plot_did_annotated(
        df, 'nightlights', 'Solar Impact (Lights)', 'Radiance',
        f"{OUTPUT_DIR}/did_nl_solar.png",
        'treated_solar', 'Solar', SOLAR_COLOR
    )

    plot_time_series(
        df, 'structure_proxy_vh', 'Shelter Impact (VH)', 'dB',
        f"{OUTPUT_DIR}/ts_vh_shelter.png",
        'treated_shelter', 'Shelter', SHELTER_COLOR
    )
    plot_did_annotated(
        df, 'structure_proxy_vh', 'Shelter Impact (VH)', 'dB',
        f"{OUTPUT_DIR}/did_vh_shelter.png",
        'treated_shelter', 'Shelter', SHELTER_COLOR
    )

    # Google Trends
    plot_google_trends_ts(f"{OUTPUT_DIR}/ts_google_trends.png")
    print("\nVisuals Complete.")

if __name__ == "__main__":
    main()
