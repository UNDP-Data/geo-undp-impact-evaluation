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
EXCLUDE_MONTHS = []  # Keep March in analysis
MAX_DATE = pd.Timestamp("2025-09-30")  # Exclude data after September 2025 due to NAs

# --- COLOR CONFIGURATION ---
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
        df_acled = df_acled.dropna(subset=['latitude', 'longitude'])
        gdf_acled = gpd.GeoDataFrame(df_acled, geometry=gpd.points_from_xy(df_acled.longitude, df_acled.latitude), crs="EPSG:4326")
        gdf_bounds = gpd.read_file(boundaries_path)
        if gdf_bounds.crs != "EPSG:4326": gdf_bounds = gdf_bounds.to_crs("EPSG:4326")
        id_col = next((c for c in gdf_bounds.columns if 'pcode' in c.lower()), None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            joined = gpd.sjoin(gdf_acled, gdf_bounds[[id_col, 'geometry']], how='inner', predicate='intersects')
        joined['event_date'] = pd.to_datetime(joined['event_date'])
        joined['period_start'] = joined['event_date'].dt.to_period('M').dt.to_timestamp()
        return joined.groupby([id_col, 'period_start']).agg({'event_id_cnty':'count', 'fatalities':'sum'}).rename(columns={'event_id_cnty':'conflict_events'}).reset_index(), id_col
    except: return None, None

def load_and_process_wfp(wfp_path, gee_townships_gdf):
    if not Path(wfp_path).exists(): return None
    print("  > Processing WFP Data (State-Blocked Matching)...")
    try:
        df = pd.read_csv(wfp_path, on_bad_lines='skip', low_memory=False)
        if df.iloc[0].astype(str).str.contains('#').any(): df = df.iloc[1:]
        df.columns = [c.lower().strip() for c in df.columns]
        price_col = next((c for c in df.columns if c in ['price', 'value']), None)
        if not price_col: price_col = next((c for c in df.columns if 'price' in c and 'flag' not in c), None)
        req_cols = ['date', 'admin1', 'market', 'commodity', price_col]
        if not all(c in df.columns for c in req_cols): return None

        df = df[df['commodity'].str.contains('Rice', case=False, na=False)].copy()
        df['price'] = pd.to_numeric(df[price_col], errors='coerce')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['price', 'date'])

        gee_ref = gee_townships_gdf[['township_name', 'state_name']].drop_duplicates()
        gee_ref['t_clean'] = gee_ref['township_name'].apply(clean_name)
        gee_ref['s_clean'] = gee_ref['state_name'].apply(clean_name)
        gee_states = gee_ref['s_clean'].unique()
        matched_map = {}
        unique_markets = df[['market', 'admin1']].drop_duplicates()
        for idx, row in unique_markets.iterrows():
            m_clean = clean_name(str(row['market']))
            s_clean = clean_name(str(row['admin1']))
            state_matches = get_close_matches(s_clean, gee_states, n=1, cutoff=0.7)
            if state_matches:
                candidates = gee_ref[gee_ref['s_clean'] == state_matches[0]]['t_clean'].tolist()
            else:
                candidates = gee_ref['t_clean'].tolist()
            matches = get_close_matches(m_clean, candidates, n=1, cutoff=0.6)
            if matches: matched_map[str(row['market'])] = matches[0]

        df['township_clean'] = df['market'].map(matched_map)
        df = df.dropna(subset=['township_clean'])
        df['period_start'] = df['date'].dt.to_period('M').dt.to_timestamp()
        return df.groupby(['township_clean', 'period_start'])['price'].mean().reset_index().rename(columns={'price': 'price_rice', 'township_clean': 'name_clean'})
    except: return None

def load_clean_project_data():
    treatment_path = "outputs/myanmar/treatment_undp_final.csv"
    if not Path(treatment_path).exists(): return pd.DataFrame()
    return pd.read_csv(treatment_path)

def normalize_names(series):
    return series.astype(str).str.lower().str.strip().str.replace(r'[^\w\s]', '', regex=True)

def load_undp_projects():
    if not Path(PROJECTS_GEOJSON).exists(): return None
    gdf = gpd.read_file(PROJECTS_GEOJSON)
    cat_col = next((c for c in gdf.columns if 'type' in c.lower() or 'sector' in c.lower()), None)
    if cat_col: gdf = gdf.rename(columns={cat_col: 'category'})
    else: gdf['category'] = 'Other'
    return gdf

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
                       Patch(facecolor=CONTROL_COLOR, label=f'Control 25km (N={len(neighbor_gdf)})'),
                       Patch(facecolor=OTHER_COLOR, label='Excluded')]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=12, frameon=False)
    ax.set_title("Study Design: 25km Spatial Control Group", fontsize=16, fontweight='bold')
    ax.set_axis_off()
    ax.set_xlim(MYANMAR_EXTENT[0], MYANMAR_EXTENT[2]); ax.set_ylim(MYANMAR_EXTENT[1], MYANMAR_EXTENT[3])
    plt.savefig(output_path, dpi=300, bbox_inches='tight'); plt.close()

def draw_project_dots(ax, gdf):
    if gdf is None or gdf.empty: return
    ax.scatter(gdf.geometry.x, gdf.geometry.y, c='#FFFFFF', s=15, edgecolors='#999999', linewidth=0.2, zorder=5, label='Other')
    legend_handles = []
    for label, props in SECTOR_MAP.items():
        pattern = "|".join(props['keywords'])
        subset = gdf[gdf['category'].astype(str).str.contains(pattern, case=False, na=False)]
        if not subset.empty:
            ax.scatter(subset.geometry.x, subset.geometry.y, c=props['color'], s=20, edgecolors='black', linewidth=0.5, zorder=10)
            legend_handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=props['color'], markersize=8, markeredgecolor='black', label=label))
    if legend_handles:
        leg = ax.legend(handles=legend_handles, loc='upper right', fontsize=8, title="UNDP Projects", facecolor='white', edgecolor='gray', framealpha=0.9)
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

def plot_time_series(df, col, title, ylabel, output_path, treated_col='treated', group_label='UNDP', color='blue', valid_ids=None):
    """
    Plots time series trends with clean SE bands.
    Treatment line changes color at UNDP_START_DATE to highlight intervention period.
    """
    
    # Filter to 25km spatial subset if valid_ids provided
    if valid_ids is not None:
        df = df[df['township_id'].isin(valid_ids)].copy()
    
    if col not in df.columns or df[treated_col].sum() == 0: 
        return
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['period_start'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    
    # Plot full treatment line in gray first (for continuous SE bands)
    df_treated = df[df[treated_col] == 1].copy()
    df_control = df[df[treated_col] == 0].copy()
    
    # Plot control group
    if not df_control.empty:
        sns.lineplot(data=df_control, x='date', y=col, 
                     color=CONTROL_COLOR, linewidth=2.5, 
                     errorbar='se', ax=ax, label='Control (Neighbors)')
    
    # Plot treated group in gray (entire timeline for smooth SE)
    if not df_treated.empty:
        sns.lineplot(data=df_treated, x='date', y=col, 
                     color='gray', linewidth=2.5, alpha=0.7,
                     errorbar='se', ax=ax, legend=False)
        
        # Overlay post-UNDP portion in blue (without separate SE bands to avoid overlap)
        df_treated_post = df_treated[df_treated['date'] >= UNDP_START_DATE]
        if not df_treated_post.empty:
            post_grouped = df_treated_post.groupby('date')[col].mean()
            ax.plot(post_grouped.index, post_grouped.values, 
                    color=color, linewidth=3, marker='o', markersize=4, 
                    zorder=5, label=f'Treatment ({group_label})')
    
    # Earthquake line
    ax.axvline(EARTHQUAKE_DATE, color='red', linestyle='--', lw=1.5, alpha=0.8, label='Earthquake', zorder=3)
    
    # Formatting
    ax.set_title(title, fontweight='bold', fontsize=14, pad=15)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel('Date', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Legend
    ax.legend(loc='best', fontsize=10, frameon=True, facecolor='white', framealpha=0.9)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  > Saved Time Series: {output_path}")
    
def get_regression_stats(outcome, sample="Spatial 25km"):
    if not Path(REGRESSION_RESULTS).exists(): return None
    try:
        df_res = pd.read_csv(REGRESSION_RESULTS)
        outcome_col = next((c for c in df_res.columns if c.lower() in ['outcome', 'dependent', 'y']), 'Outcome')
        sample_col = next((c for c in df_res.columns if c.lower() in ['sample', 'subsample', 'specification']), 'Sample')
        controls_col = next((c for c in df_res.columns if c.lower() in ['controls', 'control', 'covariates']), 'Controls')
        
        row = df_res[(df_res[outcome_col] == outcome) & (df_res[sample_col] == sample) & (df_res[controls_col].astype(str).str.contains("rainfall", case=False, na=False))]
        if row.empty: row = df_res[(df_res[outcome_col] == outcome) & (df_res[sample_col] == sample)]
        if row.empty: return None
        
        row = row.drop_duplicates(keep='last').iloc[-1]
        coeff_col = next((c for c in df_res.columns if c.lower() in ['coeff', 'coefficient', 'estimate', 'beta']), 'Coeff')
        coeff = row[coeff_col]
        
        # --- CRITICAL FIX: Handle string coefficient properly ---
        # The coeff is likely already a string with stars (e.g. "0.45***")
        # We just return it directly instead of trying to format it
        return str(coeff)
    except: return None

def plot_did_annotated(df, outcome, title, ylabel, output_path, treatment_col, treat_label, treat_color, valid_ids=None):
    if outcome not in df.columns or df[treatment_col].sum() == 0: return
    print(f"  > DiD Plot: {title}")
    
    # Use provided valid_ids if available, otherwise compute spatial subset
    if valid_ids is not None:
        df_subset = df[df['township_id'].isin(valid_ids)].copy()
    else:
        df_subset = get_spatial_subset(df, SPATIAL_RADIUS_DEG, treatment_col)
    
    pre = df_subset[(df_subset['post']==0) & (pd.to_datetime(df_subset['period_start']).dt.year==2025)]
    post = df_subset[df_subset['post']==1]
    
    y_tp = pre[pre[treatment_col]==1][outcome].mean()
    y_t_post = post[post[treatment_col]==1][outcome].mean()
    y_cp = pre[pre[treatment_col]==0][outcome].mean()
    y_c_post = post[post[treatment_col]==0][outcome].mean()
    
    if pd.isna(y_tp): return
    y_cf = y_tp + (y_c_post - y_cp)
    simple_did = (y_t_post - y_tp) - (y_c_post - y_cp)
    coef_str = get_regression_stats(outcome)
    
    fig, ax = plt.subplots(figsize=(9, 7), facecolor='white')
    ax.plot([0, 1], [y_tp, y_t_post], 'o-', color=treat_color, lw=3, markersize=10, label=treat_label)
    ax.plot([0, 1], [y_cp, y_c_post], 's-', color=CONTROL_COLOR, lw=3, markersize=10, label='Control')
    ax.plot([0, 1], [y_tp, y_cf], '--', color='#7FB2F0', lw=2, label='Counterfactual')
    ax.annotate('', xy=(1, y_t_post), xytext=(1, y_cf), arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.grid(True, linestyle='--', alpha=0.5)
    
    if coef_str is not None:
        text_str = (f"Adjusted DiD Effect\n(Spatial 25km + Controls)\nCoef: {coef_str}")
    else:
        text_str = (f"DiD Effect\n(Simple Difference)\nCoef: {simple_did:.4f}")
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='red')
    ax.text(0.5, 0.85, text_str, transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='center', bbox=props, color='red', fontweight='bold')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Pre', 'Post'])
    ax.set_title(title, fontweight='bold')
    plt.savefig(output_path, dpi=300, bbox_inches='tight'); plt.close()

def plot_event_study_generic(df, outcome, title, output_path, valid_ids=None):
    csv_path = Path("outputs/myanmar/robustness") / f"es_coeffs_{outcome}.csv"
    if csv_path.exists():
        es_df = pd.read_csv(csv_path)
    else:
        # Simple fallback ES computation
        if outcome not in df.columns or 'treated' not in df.columns: return
        
        # Filter to 25km spatial subset if valid_ids provided
        df_mod = df.copy()
        if valid_ids is not None:
            df_mod = df_mod[df_mod['township_id'].isin(valid_ids)].copy()
        
        df_mod = df_mod.dropna(subset=[outcome]).copy()
        df_mod['months_since_eq'] = ((df_mod['period_start'].dt.year - EARTHQUAKE_DATE.year) * 12 + (df_mod['period_start'].dt.month - EARTHQUAKE_DATE.month))
        df_mod = df_mod[(df_mod['months_since_eq'] >= -6) & (df_mod['months_since_eq'] <= 6)]
        
        rows = []
        for t in range(-6, 7):
            subset = df_mod[df_mod['months_since_eq'] == t]
            if subset.empty: continue
            g_t = subset[subset['treated'] == 1][outcome]
            g_c = subset[subset['treated'] == 0][outcome]
            if g_t.empty or g_c.empty: continue
            diff = g_t.mean() - g_c.mean()
            se = np.sqrt(g_t.var(ddof=1)/len(g_t) + g_c.var(ddof=1)/len(g_c))
            rows.append({'t': t, 'coef': diff, 'err': se})
        es_df = pd.DataFrame(rows)

    if es_df.empty or not {'t', 'coef', 'err'}.issubset(es_df.columns): return

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    ax.errorbar(es_df['t'], es_df['coef'], yerr=1.96 * es_df['err'], fmt='o-', linewidth=2, color=UNDP_COLOR, ecolor=UNDP_COLOR, capsize=4)
    ax.axhline(0, color='black', alpha=0.5); ax.axvline(-1, color='red', linestyle='--')
    ax.set_title(f"Dynamic Effect: {title}", fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(output_path, dpi=300, bbox_inches='tight'); plt.close()

def plot_google_trends_ts(output_path):
    gt_path = f"{EXTERNAL_DIR}/google_trends_mm.csv"
    if not Path(gt_path).exists(): return
    df = pd.read_csv(gt_path)
    df['date'] = pd.to_datetime(df['date'])
    focus_keywords = ['Earthquake', 'Donation', 'Passport', 'Anxiety']
    
    for kw in focus_keywords:
        df_kw = df[df['keyword'] == kw].copy()
        if df_kw.empty: continue
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
        sns.lineplot(data=df_kw, x='date', y='score', color='#1f77b4', linewidth=2.5, ax=ax)
        ax.axvline(EARTHQUAKE_DATE, color='red', linestyle='--', lw=1.5, label='Earthquake Date')
        ax.set_title(f"Public Sentiment: {kw}", fontweight='bold', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)
        clean_kw = re.sub(r'[^a-zA-Z0-9]', '', kw).lower()
        fname = output_path.replace("google_trends.png", f"google_trends_{clean_kw}.png")
        plt.savefig(fname, dpi=300, bbox_inches='tight'); plt.close()

def load_and_clean_data():
    print("Loading Data for Visuals...")
    
    files = glob.glob(f"{DATA_DIR}/gee_panel_myanmar_PART*.csv")
    if not files: return pd.DataFrame()

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    
    rename_map = {'admin3_id': 'township_id', 'admin3_nam': 'township_name', 
                  'NDVI': 'ndvi', 'NDBI': 'ndbi', 'RAIN_TOTAL': 'rainfall', 
                  'NL_AVG_RAD': 'nightlights_gee', 'VH': 'vh', 'VV': 'vv'}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    
    if 'nightlights_gee' not in df.columns and 'nightlights' in df.columns:
        df = df.rename(columns={'nightlights': 'nightlights_gee'})
        
    df['period_start'] = pd.to_datetime(df['period_start'])
    df['clean_name'] = df['township_name'].apply(clean_name)
    df['name_clean'] = df['clean_name'] 

    if 'LOCAL_NL_FILE' in globals() and Path(LOCAL_NL_FILE).exists():
        print("    -> Merging Local Nightlights (April-Sept Data)...")
        df_local = pd.read_csv(LOCAL_NL_FILE)
        df_local['period_start'] = pd.to_datetime(df_local['period_start'])
        df_local['name_clean'] = df_local['township_name'].apply(clean_name)
        
        df = pd.merge(df, df_local[['name_clean', 'period_start', 'nightlights']], 
                      on=['name_clean', 'period_start'], how='outer', suffixes=('_gee', '')) 
        
        if 'nightlights_gee' in df.columns:
            df['nightlights'] = df['nightlights'].combine_first(df['nightlights_gee'])
            
        df['township_name'] = df['township_name'].fillna(df['name_clean'].str.title())
        df = df.sort_values(['name_clean', 'period_start'])
        df['township_id'] = df.groupby('name_clean')['township_id'].ffill().bfill()
            
    else:
        df['nightlights'] = df['nightlights_gee']

    df = df.sort_values(['name_clean', 'period_start'])

    proj_df = load_clean_project_data()
    for col in ['treated', 'treated_agri', 'treated_solar', 'treated_shelter']:
        df[col] = 0
    
    if not proj_df.empty:
        if 'undp_treated' in proj_df.columns:
            treated_base = proj_df[proj_df['undp_treated'] == 1]
        else:
            treated_base = proj_df
            
        treated_set = set(treated_base['clean_name'])
        df['treated'] = apply_treatment_map(df, treated_set)

        def get_sector_set(dframe, col_name):
            if col_name in dframe.columns:
                return set(dframe[dframe[col_name] == 1]['clean_name'])
            return set()

        df['treated_agri'] = apply_treatment_map(df, get_sector_set(treated_base, 'is_agri'))
        df['treated_solar'] = apply_treatment_map(df, get_sector_set(treated_base, 'is_solar'))
        df['treated_shelter'] = apply_treatment_map(df, get_sector_set(treated_base, 'is_shelter'))
        
    acled_stats, acled_id_col = process_acled_data(BOUNDARIES_FILE)
    if acled_stats is not None and acled_id_col:
        print("    -> Merging ACLED Conflict Data...")
        df = pd.merge(df, acled_stats, left_on=['township_id', 'period_start'], right_on=[acled_id_col, 'period_start'], how='left')
        df['conflict_events'] = df['conflict_events'].fillna(0)

    if Path(BOUNDARIES_FILE).exists():
        gdf_bounds = gpd.read_file(BOUNDARIES_FILE)
        rename_geo = {'adm3_name': 'township_name', 'adm1_name': 'state_name'}
        gdf_bounds = gdf_bounds.rename(columns=rename_geo)
        wfp_df = load_and_process_wfp(f"{EXTERNAL_DIR}/wfp_prices.csv", gdf_bounds)
        if wfp_df is not None:
            df = pd.merge(df, wfp_df, on=['name_clean', 'period_start'], how='left')
            df['price_rice'] = df.groupby('name_clean')['price_rice'].transform(
                lambda x: x.interpolate(method='linear').ffill().bfill()
            )

    df['ym_str'] = df['period_start'].dt.strftime('%Y-%m')
    df = df[~df['ym_str'].isin(EXCLUDE_MONTHS)].copy()
    df = df[df['period_start'] <= MAX_DATE].copy()  # Exclude data after September 2025
    
    df = calculate_sar_indices(df)
    df['post'] = (df['period_start'] > EARTHQUAKE_DATE).astype(int)
    
    # --- UPDATE 1: CALCULATE LOG NIGHTLIGHTS ---
    if 'nightlights' in df.columns:
        df['log_nightlights'] = np.log1p(df['nightlights'].clip(lower=0))
    
    return df

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    df = load_and_clean_data()
    boundaries = gpd.read_file(BOUNDARIES_FILE)
    undp_gdf = load_undp_projects()
    
    subset_df = get_spatial_subset(df, SPATIAL_RADIUS_DEG, 'treated')
    valid_ids_25km = subset_df['township_id'].unique()

    plot_study_design_map(df, f"{OUTPUT_DIR}/map_study_design.png")

    indicators = [
        ('nightlights', 'Nightlights', 'Radiance', 'raster', 'nl'),
        # --- UPDATE 2: ADD LOG NIGHTLIGHTS ---
        ('log_nightlights', 'Log Nightlights', 'Log Radiance', 'vector', 'log_nl'), 
        ('ndvi', 'Vegetation (NDVI)', 'NDVI', 'vector', 'ndvi'),
        ('rvi', 'Radar Vegetation (RVI)', 'RVI', 'vector', 'rvi'),
        ('ndbi', 'Construction (NDBI)', 'NDBI', 'vector', 'ndbi'),
        ('r_ndbi', 'Radar Construction (R-NDBI)', 'R-NDBI', 'vector', 'rndbi'),
        ('rainfall', 'Rainfall', 'mm', 'vector', 'rain'),
        ('structure_proxy_vh', 'Structure Proxy (VH)', 'dB', 'vector', 'vh'),
        ('conflict_events', 'Conflict Intensity', 'Events/Month', 'vector', 'conflict'),
        ('price_rice', 'Rice Market Price', 'MMK/kg', 'vector', 'price'),
    ]

    for col, title, ylabel, mode, suffix in indicators:
        print(f"\nProcessing {title}...")
        if col not in df.columns: continue
        plot_3panel_map(df, boundaries, col, title, f"{OUTPUT_DIR}/map_{suffix}_clean.png", mode=mode, valid_ids=valid_ids_25km)
        plot_3panel_map(df, boundaries, col, title, f"{OUTPUT_DIR}/map_{suffix}_projects.png", undp_gdf, mode=mode, valid_ids=valid_ids_25km)
        plot_time_series(df, col, f"{title} Trends", ylabel, f"{OUTPUT_DIR}/ts_{suffix}.png", 'treated', 'UNDP', UNDP_COLOR, valid_ids=valid_ids_25km)
        plot_did_annotated(df, col, title, ylabel, f"{OUTPUT_DIR}/did_{suffix}.png", 'treated', 'UNDP', UNDP_COLOR, valid_ids=valid_ids_25km)
        plot_event_study_generic(df, col, title, f"{OUTPUT_DIR}/es_{suffix}.png", valid_ids=valid_ids_25km)
    
    print("\n-- SECTORS --")
    plot_time_series(df, 'nightlights', 'Solar Impact (Lights)', 'Radiance', f"{OUTPUT_DIR}/ts_nl_solar.png", 'treated_solar', 'Solar', SOLAR_COLOR, valid_ids=valid_ids_25km)
    plot_did_annotated(df, 'nightlights', 'Solar Impact (Lights)', 'Radiance', f"{OUTPUT_DIR}/did_nl_solar.png", 'treated_solar', 'Solar', SOLAR_COLOR, valid_ids=valid_ids_25km)
    plot_google_trends_ts(f"{OUTPUT_DIR}/ts_google_trends.png")
    print("\nVisuals Complete.")

if __name__ == "__main__":
    main()
