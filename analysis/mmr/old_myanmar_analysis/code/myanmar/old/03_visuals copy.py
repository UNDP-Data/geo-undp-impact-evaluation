#!/usr/bin/env python3
"""
03_visuals.py
===================
FINAL VISUALS - UPDATED
1. Maps: Full Country Context, Data ONLY in 5km Areas.
2. Nightlights: Shows PIXEL-LEVEL variation (Raster).
3. Time Series: Fixed Legend (UNDP/Control) & Grid.
4. New Indicators: Conflict (ACLED) & Rice Prices (WFP).
   - FIX: Robust column renaming (handles NL_AVG_RAD and RAIN_TOTAL).
   - FIX: Prevents KeyError 'nightlights_gee'.
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
REGRESSION_RESULTS = "outputs/myanmar/regression_results_sar_extended.csv"

EARTHQUAKE_DATE = pd.Timestamp("2025-03-28")
MYANMAR_EXTENT = [92.0, 9.5, 101.5, 28.5] 
SPATIAL_RADIUS_DEG = 0.045 # 5km
EXCLUDE_MONTHS = ['2025-03']

# Colors
UNDP_COLOR = 'blue' 
AGRI_COLOR = '#228B22' 
SOLAR_COLOR = '#FFD700' 
SHELTER_COLOR = '#FF4500' 
CONTROL_COLOR = '#A23B72' 
OTHER_COLOR = '#e0e0e0'   
PROJ_COLORS = {
    'Solar Energy': '#FFD700', 'Medical/Health': '#00FF00', 
    'Construction/Repair': '#FF4500', 'Shelter': '#1E90FF', 
    'Agricultural Support': '#228B22', 'Other': '#FFFFFF'
}

# Manual Map for WFP Matching
WFP_MANUAL_FIX = {
    'yangon': ['tamwe', 'botahtaung', 'pabedan', 'kyauktada', 'latha', 'lanmadaw', 'ahlone', 'dagon'],
    'mandalay': ['chanayethazan', 'chanmyathazi', 'mahaaungmyay', 'pyigyitagon', 'aungmyaythazan'],
    'naypyitaw': ['zabuthiri', 'zayarthiri', 'pobbathiri', 'ottarathiri'],
    'sittwe': ['sittwe'],
    'maungdaw': ['maungdaw'],
    'myitkyina': ['myitkyina'],
    'lashio': ['lashio'],
    'waimaw': ['waingmaw']
}

def clean_name(name):
    if pd.isna(name): return ""
    return re.sub(r'[^a-z0-9]', '', str(name).lower())

def calculate_sar_indices(df):
    if 'vh' not in df.columns or 'vv' not in df.columns: return df
    vh_lin = 10 ** (df['vh'].clip(lower=-30) / 10.0)
    vv_lin = 10 ** (df['vv'].clip(lower=-30) / 10.0)
    df['rvi'] = (4 * vh_lin) / (vv_lin + vh_lin)
    df['r_ndbi'] = (vv_lin - vh_lin) / (vv_lin + vh_lin)
    df['flood_proxy_vv'] = df['vv']
    df['structure_proxy_vh'] = df['vh']
    return df

def get_treated_townships():
    if not Path(PROJECT_FILE).exists(): return set()
    df = pd.read_excel(PROJECT_FILE)
    col = next((c for c in df.columns if "Township" in c), None)
    return set(df[col].apply(clean_name).unique())

def _get_category_townships(keywords):
    if Path(PROJECTS_GEOJSON).exists():
        try:
            gdf = gpd.read_file(PROJECTS_GEOJSON)
            cat_col = next((c for c in gdf.columns if 'type' in c.lower() or 'sector' in c.lower() or 'cluster' in c.lower()), None)
            name_col = next((c for c in gdf.columns if 'township' in c.lower() or 'name' in c.lower()), None)
            if cat_col and name_col:
                regex = "|".join(keywords)
                sub_gdf = gdf[gdf[cat_col].astype(str).str.contains(regex, case=False, na=False)]
                return set(sub_gdf[name_col].apply(clean_name).unique())
        except: pass
    return set()

def get_agri_townships(): return _get_category_townships(['Agri', 'Farm', 'Live', 'Food'])
def get_solar_townships(): return _get_category_townships(['Solar', 'Energy', 'Renewable', 'Electr'])
def get_shelter_townships(): return _get_category_townships(['Shelter', 'Housing', 'Construct', 'Repair', 'Infra'])

def apply_treatment_map(df, target_set):
    treatment_map = {}
    for name in df['township_name'].unique():
        cname = clean_name(name)
        treatment_map[name] = 1 if cname in target_set else 0
    return df['township_name'].map(treatment_map)

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
        id_col = next((c for c in gdf_bounds.columns if 'pcode' in c.lower()), None)
        
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

        # 2. Reference Data (GEE Townships)
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
            
            # Match State First
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

def load_and_clean_data():
    print("Loading Data for Visuals...")
    
    # 1. Load Panel Files
    files = glob.glob(f"{DATA_DIR}/gee_panel_myanmar_PART*.csv")
    if not files:
        print(f"  ! ERROR: No files found in {DATA_DIR}")
        return pd.DataFrame()

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    
    # Rename GEE Columns
    rename_map = {'admin3_id': 'township_id', 'admin3_nam': 'township_name', 
                  'NDVI': 'ndvi', 'NDBI': 'ndbi', 'RAIN_TOTAL': 'rainfall', 
                  'NL_AVG_RAD': 'nightlights', 'VH': 'vh', 'VV': 'vv'}
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})
    
    # Standardize 'nightlights' column name if inconsistent
    if 'nightlights' not in df.columns and 'NL_AVG_RAD' in df.columns:
        df['nightlights'] = df['NL_AVG_RAD']
        
    df['period_start'] = pd.to_datetime(df['period_start'])
    
    # Features
    df['treated'] = apply_treatment_map(df, get_treated_townships())
    df['treated_agri'] = apply_treatment_map(df, get_agri_townships())
    df['treated_solar'] = apply_treatment_map(df, get_solar_townships())
    df['treated_shelter'] = apply_treatment_map(df, get_shelter_townships())
    df['name_clean'] = df['township_name'].apply(clean_name)

    # Local NL Merge (If exists)
    if Path(LOCAL_NL_FILE).exists():
        df_local = pd.read_csv(LOCAL_NL_FILE)
        df_local['period_start'] = pd.to_datetime(df_local['period_start'])
        df_local['name_clean'] = df_local['township_name'].apply(clean_name)
        df = pd.merge(df, df_local[['name_clean', 'period_start', 'nightlights']], 
                      on=['name_clean', 'period_start'], how='left', suffixes=('', '_local'))
        df['nightlights'] = df['nightlights'].combine_first(df.get('nightlights_gee', df.get('nightlights')))

    # ACLED Merge
    acled_df, id_col_bounds = process_acled_data(BOUNDARIES_FILE)
    if acled_df is not None:
        df = pd.merge(df, acled_df, left_on=['township_id', 'period_start'], right_on=[id_col_bounds, 'period_start'], how='left')
        df['conflict_events'] = df['conflict_events'].fillna(0)

    # 2. WFP Merge (With Boundary Fix)
    if Path(BOUNDARIES_FILE).exists():
        gdf_bounds = gpd.read_file(BOUNDARIES_FILE)
        
        # --- FIX: Explicit Column Mapping ---
        rename_geo = {'adm3_name': 'township_name', 'adm1_name': 'state_name'}
        gdf_bounds = gdf_bounds.rename(columns=rename_geo)
        
        # Fallback
        if 'township_name' not in gdf_bounds.columns:
             t_col = next((c for c in gdf_bounds.columns if c.lower() in ['admin3_nam', 'ts_name']), None)
             if t_col: gdf_bounds = gdf_bounds.rename(columns={t_col: 'township_name'})
        if 'state_name' not in gdf_bounds.columns:
             s_col = next((c for c in gdf_bounds.columns if c.lower() in ['admin1_nam', 'st_name']), None)
             if s_col: gdf_bounds = gdf_bounds.rename(columns={s_col: 'state_name'})

        # Call the new processing function
        wfp_df = load_and_process_wfp(f"{EXTERNAL_DIR}/wfp_prices.csv", gdf_bounds)
        
        if wfp_df is not None:
            df = pd.merge(df, wfp_df, on=['name_clean', 'period_start'], how='left')
            # Interpolate (Time-only, no spatial fill)
            df['price_rice'] = df.groupby('township_id')['price_rice'].transform(lambda x: x.interpolate(method='linear').ffill().bfill())

    df['ym_str'] = df['period_start'].dt.strftime('%Y-%m')
    df = df[~df['ym_str'].isin(EXCLUDE_MONTHS)].copy()
    
    df = calculate_sar_indices(df)
    df['post'] = (df['period_start'] > EARTHQUAKE_DATE).astype(int)
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

def get_agri_projects_gdf(full_gdf):
    if full_gdf is None: return None
    return full_gdf[full_gdf['category'].astype(str).str.contains('Agri|Farm|Live|Food', case=False, na=False)].copy()

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
    
    legend_elements = [Patch(facecolor=UNDP_COLOR, label=f'Treated (N={len(treated_gdf)})'),
                       Patch(facecolor=CONTROL_COLOR, label=f'Control 5km (N={len(neighbor_gdf)})'),
                       Patch(facecolor=OTHER_COLOR, label='Excluded')]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=12, frameon=False)
    ax.set_title("Study Design: 5km Spatial Control Group", fontsize=16, fontweight='bold')
    ax.set_axis_off()
    ax.set_xlim(MYANMAR_EXTENT[0], MYANMAR_EXTENT[2]); ax.set_ylim(MYANMAR_EXTENT[1], MYANMAR_EXTENT[3])
    plt.savefig(output_path, dpi=300, bbox_inches='tight'); plt.close()

def draw_project_dots(ax, gdf):
    if gdf is None: return
    specific_keys = [k for k in PROJ_COLORS.keys() if k != 'Other']
    other_mask = ~gdf['category'].apply(lambda x: any(k.lower() in str(x).lower() for k in specific_keys))
    other_subset = gdf[other_mask]
    if not other_subset.empty:
        ax.scatter(other_subset.geometry.x, other_subset.geometry.y, c=PROJ_COLORS['Other'], s=8, edgecolors='#999999', linewidth=0.2, zorder=5)
    for cat in specific_keys:
        subset = gdf[gdf['category'].astype(str).str.contains(cat, case=False, na=False)]
        if not subset.empty:
            ax.scatter(subset.geometry.x, subset.geometry.y, c=PROJ_COLORS[cat], s=15, edgecolors='black', linewidth=0.5, label=cat, zorder=10)
    handles = [Line2D([0], [0], marker='o', color='k', markerfacecolor=c, markersize=8, markeredgecolor='white', markeredgewidth=0.5, label=l) 
               for l, c in PROJ_COLORS.items() if l in gdf['category'].unique() or l == 'Other']
    leg = ax.legend(handles=handles, loc='upper right', fontsize=8, title="UNDP Projects", facecolor='black', edgecolor='gray', labelcolor='white', framealpha=0.9)
    leg.get_title().set_color('white')

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

def plot_time_series(df, outcome, title, ylabel, output_path, treatment_col, treat_label, treat_color):
    if outcome not in df.columns or df[treatment_col].sum() == 0: return
    print(f"  > Time Series: {title}")
    df_subset = get_spatial_subset(df, SPATIAL_RADIUS_DEG, treatment_col)
    df_subset['date'] = pd.to_datetime(df_subset['period_start'])
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    sns.lineplot(data=df_subset, x='date', y=outcome, hue=treatment_col, palette={1: treat_color, 0: CONTROL_COLOR}, ax=ax, linewidth=3, errorbar='se', legend=False)
    ax.axvline(EARTHQUAKE_DATE, color='red', linestyle='--', lw=1.5)
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.5)
    legend_elements = [
        Line2D([0], [0], color=treat_color, lw=3, label=treat_label),
        Line2D([0], [0], color=CONTROL_COLOR, lw=3, label='Control (Neighbors)'),
        Line2D([0], [0], color='red', linestyle='--', lw=1.5, label='Earthquake')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.savefig(output_path, dpi=300, bbox_inches='tight'); plt.close()

def get_regression_stats(outcome, sample="Spatial 5km"):
    if not Path(REGRESSION_RESULTS).exists(): return None, None
    try:
        df_res = pd.read_csv(REGRESSION_RESULTS)
        row = df_res[(df_res['Outcome'] == outcome) & (df_res['Sample'] == sample) & (df_res['Controls'].str.contains("price", case=False, na=False))]
        if row.empty: row = df_res[(df_res['Outcome'] == outcome) & (df_res['Sample'] == sample)]
        if not row.empty: return row.iloc[0]['Raw_Coef'], row.iloc[0]['Raw_Pval']
    except: pass
    return None, None

def plot_did_annotated(df, outcome, title, ylabel, output_path, treatment_col, treat_label, treat_color):
    if outcome not in df.columns or df[treatment_col].sum() == 0: return
    print(f"  > DiD Plot: {title}")
    df_subset = get_spatial_subset(df, SPATIAL_RADIUS_DEG, treatment_col)
    pre = df_subset[(df_subset['post']==0) & (pd.to_datetime(df_subset['period_start']).dt.year==2025)]
    post = df_subset[df_subset['post']==1]
    y_tp = pre[pre[treatment_col]==1][outcome].mean()
    y_t_post = post[post[treatment_col]==1][outcome].mean()
    y_cp = pre[pre[treatment_col]==0][outcome].mean()
    y_c_post = post[post[treatment_col]==0][outcome].mean()
    if pd.isna(y_tp): return
    y_cf = y_tp + (y_c_post - y_cp)
    coef, pval = get_regression_stats(outcome)
    fig, ax = plt.subplots(figsize=(9, 7), facecolor='white')
    ax.plot([0, 1], [y_tp, y_t_post], 'o-', color=treat_color, lw=3, markersize=10, label=treat_label)
    ax.plot([0, 1], [y_cp, y_c_post], 's-', color=CONTROL_COLOR, lw=3, markersize=10, label='Control')
    ax.plot([0, 1], [y_tp, y_cf], '--', color='#7FB2F0', lw=2, label='Counterfactual')
    ax.annotate('', xy=(1, y_t_post), xytext=(1, y_cf), arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.grid(True, linestyle='--', alpha=0.5)
    if coef is not None:
        signif = "*" if pval < 0.1 else ""
        text_str = (f"Adjusted DiD Effect\n(Spatial 5km + Controls)\nCoef: {coef:.3f}{signif}\n(p={pval:.3f})")
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='red')
        ax.text(0.5, 0.85, text_str, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='center', bbox=props, color='red', fontweight='bold')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Pre', 'Post'])
    ax.set_title(title, fontweight='bold')
    plt.savefig(output_path, dpi=300, bbox_inches='tight'); plt.close()

def plot_interaction_coeff(outcome, title, output_path, color, suffix=""):
    csv_path = f"outputs/myanmar/robustness/es_coeffs_interaction_{outcome}{suffix}.csv"
    if not Path(csv_path).exists(): return
    print(f"  > Interaction Plot: {title}")
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    ax.errorbar(df['t'], df['coef'], yerr=1.96*df['err'], fmt='o', color=color, ecolor=color, elinewidth=2, capsize=4, markersize=8)
    ax.axhline(0, color='black', alpha=0.5); ax.axvline(-1, color='red', linestyle='--')
    ax.set_title(f"Additional Effect: {title}", fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(output_path, dpi=300, bbox_inches='tight'); plt.close()

def plot_event_study_generic(outcome, title, output_path):
    csv_path = f"outputs/myanmar/robustness/es_coeffs_{outcome}.csv"
    if not Path(csv_path).exists(): return
    print(f"  > Event Study: {title}")
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    ax.errorbar(df['t'], df['coef'], yerr=1.96*df['err'], fmt='o', color=UNDP_COLOR, ecolor=UNDP_COLOR, elinewidth=2, capsize=4, markersize=8)
    ax.axhline(0, color='black', alpha=0.5); ax.axvline(-1, color='red', linestyle='--')
    ax.set_title(f"Dynamic Effect: {title}", fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(output_path, dpi=300, bbox_inches='tight'); plt.close()

def plot_google_trends_ts(output_path):
    gt_path = f"{EXTERNAL_DIR}/google_trends_mm.csv"
    if not Path(gt_path).exists(): return
    print("  > Google Trends Time Series...")
    df = pd.read_csv(gt_path)
    df['date'] = pd.to_datetime(df['date'])
    trans_map = {"ရွှေဈေး": "Gold Price", "ဒေါ်လာဈေး": "Dollar Price", "ပတ်စပို့": "Passport", "ဆန်ဈေး": "Rice Price"}
    df['English Keyword'] = df['keyword'].map(trans_map)
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
    sns.lineplot(data=df, x='date', y='score', hue='English Keyword', ax=ax, linewidth=2.5)
    ax.axvline(EARTHQUAKE_DATE, color='red', linestyle='--', lw=1.5, label='Earthquake')
    ax.set_title("Google Search Trends (Economic Anxiety)", fontweight='bold', fontsize=14)
    ax.set_ylabel("Search Interest (0-100)")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.savefig(output_path, dpi=300, bbox_inches='tight'); plt.close()

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    df = load_and_clean_data()
    boundaries = gpd.read_file(BOUNDARIES_FILE)
    undp_gdf = load_undp_projects()
    agri_gdf = get_agri_projects_gdf(undp_gdf)
    
    subset_df = get_spatial_subset(df, SPATIAL_RADIUS_DEG, 'treated')
    valid_ids_5km = subset_df['township_id'].unique()

    plot_study_design_map(df, f"{OUTPUT_DIR}/map_study_design.png")

    indicators = [
        ('nightlights', 'Nightlights', 'Radiance', 'raster', 'nl'),
        ('ndvi', 'Vegetation (NDVI)', 'NDVI', 'vector', 'ndvi'),
        ('rvi', 'Radar Vegetation (RVI)', 'RVI', 'vector', 'rvi'),
        ('ndbi', 'Construction (NDBI)', 'NDBI', 'vector', 'ndbi'),
        ('structure_proxy_vh', 'Structure Proxy (VH)', 'dB', 'vector', 'vh'),
        ('conflict_events', 'Conflict Intensity', 'Events/Month', 'vector', 'conflict'),
        ('price_rice', 'Rice Market Price', 'MMK/kg', 'vector', 'price')
    ]

    for col, title, ylabel, mode, suffix in indicators:
        print(f"\nProcessing {title}...")
        plot_3panel_map(df, boundaries, col, title, f"{OUTPUT_DIR}/map_{suffix}_clean.png", mode=mode, valid_ids=valid_ids_5km)
        plot_3panel_map(df, boundaries, col, title, f"{OUTPUT_DIR}/map_{suffix}_projects.png", undp_gdf, mode=mode, valid_ids=valid_ids_5km)
        plot_time_series(df, col, f"{title} Trends", ylabel, f"{OUTPUT_DIR}/ts_{suffix}.png", 'treated', 'UNDP', UNDP_COLOR)
        plot_did_annotated(df, col, title, ylabel, f"{OUTPUT_DIR}/did_{suffix}.png", 'treated', 'UNDP', UNDP_COLOR)
        plot_event_study_generic(col, title, f"{OUTPUT_DIR}/es_{suffix}.png")

    print("\n-- SECTORS --")
    plot_time_series(df, 'rvi', 'Agri Impact (RVI)', 'RVI', f"{OUTPUT_DIR}/ts_rvi_agri.png", 'treated_agri', 'Agri', AGRI_COLOR)
    plot_did_annotated(df, 'rvi', 'Agri Impact (RVI)', 'RVI', f"{OUTPUT_DIR}/did_rvi_agri.png", 'treated_agri', 'Agri', AGRI_COLOR)
    plot_interaction_coeff('rvi', 'Agri Projects (RVI)', f"{OUTPUT_DIR}/coef_interaction_rvi.png", AGRI_COLOR)

    plot_time_series(df, 'nightlights', 'Solar Impact (Lights)', 'Radiance', f"{OUTPUT_DIR}/ts_nl_solar.png", 'treated_solar', 'Solar', SOLAR_COLOR)
    plot_did_annotated(df, 'nightlights', 'Solar Impact (Lights)', 'Radiance', f"{OUTPUT_DIR}/did_nl_solar.png", 'treated_solar', 'Solar', SOLAR_COLOR)
    plot_interaction_coeff('nightlights', 'Solar Projects (Lights)', f"{OUTPUT_DIR}/coef_interaction_nl_solar.png", SOLAR_COLOR, suffix="")

    plot_time_series(df, 'structure_proxy_vh', 'Shelter Impact (VH)', 'dB', f"{OUTPUT_DIR}/ts_vh_shelter.png", 'treated_shelter', 'Shelter', SHELTER_COLOR)
    plot_did_annotated(df, 'structure_proxy_vh', 'Shelter Impact (VH)', 'dB', f"{OUTPUT_DIR}/did_vh_shelter.png", 'treated_shelter', 'Shelter', SHELTER_COLOR)
    plot_interaction_coeff('structure_proxy_vh', 'Shelter Projects (VH)', f"{OUTPUT_DIR}/coef_interaction_vh_shelter.png", SHELTER_COLOR, suffix="")

    plot_google_trends_ts(f"{OUTPUT_DIR}/ts_google_trends.png")
    print("\nVisuals Complete.")

if __name__ == "__main__":
    main()
