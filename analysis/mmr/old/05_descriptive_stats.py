#!/usr/bin/env python3
"""
05_descriptive_stats.py
=======================
Generates a Descriptive Statistics / Balance Table.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
import glob
import re
from difflib import get_close_matches
import warnings
import os

# --- CONFIGURATION ---
DATA_DIR = "data/myanmar"
EXTERNAL_DIR = "data/myanmar/external"  # <--- ADD THIS LINE
OUTPUT_DIR = "outputs/myanmar/tables_formatted"
BOUNDARIES_FILE = "data/myanmar/myanmar_boundaries/mmr_admin3.geojson"
PROJECT_FILE = "data/myanmar/undp/project_data.xlsx"
PROJECTS_GEOJSON = "data/myanmar/undp/undp_interventions.geojson"
LOCAL_NL_FILE = "outputs/myanmar/local_nightlights_panel.csv"

EARTHQUAKE_DATE = pd.Timestamp("2025-03-28")
SPATIAL_RADIUS_DEG = 0.25 
EXCLUDE_MONTHS = ['2025-03']
UNDP_START_DATE = pd.Timestamp("2025-05-01")

MANUAL_MAP = {
    'mahaaungmyay': 'mandalay', 'aungmyaythazan': 'mandalay', 'chanayethazan': 'mandalay',
    'chanmyathazi': 'mandalay', 'pyigyitagon': 'mandalay', 'poke ba thi ri': 'pobbathiri',
    'za bu thi ri': 'zabuthiri', 'oke ta ra thi ri': 'ottarathiri', 'det khi na thi ri': 'dekkhinathiri',
    'zay yar thi ri': 'zeyarthiri', 'pyinoolwin': 'pyinoolwin', 'meiktila': 'meikthila',
    'taunggyi': 'taunggyi', 'nawnghkio': 'nawnghkio', 'yamethin': 'yamethin', 'patheingyi': 'patheingyi',
    'amarapura': 'amarapura', 'tada-u': 'tada-u', 'sintgaing': 'sintgaing', 'kyaukse': 'kyaukse',
    'myittha': 'myittha', 'thazi': 'thazi', 'wundwin': 'wundwin', 'mahlaing': 'mahlaing'
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

def load_clean_project_data():
    treatment_path = "outputs/myanmar/treatment_undp_final.csv"
    if not Path(treatment_path).exists():
        raise FileNotFoundError("Run 01_main_analysis_FINAL.py first!")
    return pd.read_csv(treatment_path)

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

def process_acled_data(boundaries_path):
    """Process ACLED conflict data and spatially join to townships."""
    acled_path = f"{EXTERNAL_DIR}/acled_myanmar.csv"
    if not Path(acled_path).exists(): 
        print("  ! ACLED data not found")
        return None, None
    try:
        df_acled = pd.read_csv(acled_path)
        if df_acled.empty: return None, None
        
        # Clean coordinates
        df_acled = df_acled.dropna(subset=['latitude', 'longitude'])
        
        gdf_acled = gpd.GeoDataFrame(
            df_acled, geometry=gpd.points_from_xy(df_acled.longitude, df_acled.latitude), crs="EPSG:4326"
        )
        
        gdf_bounds = gpd.read_file(boundaries_path)
        if gdf_bounds.crs != "EPSG:4326": 
            gdf_bounds = gdf_bounds.to_crs("EPSG:4326")
        
        # Robust ID Column
        id_col = next((c for c in gdf_bounds.columns if 'pcode' in c.lower()), None)
        if not id_col: id_col = 'geometry'
        
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
        
    except Exception as e:
        print(f"  ! ACLED processing error: {e}")
        return None, None

def get_treated_townships():
    if not Path(PROJECT_FILE).exists():
        search = glob.glob("data/myanmar/undp/Myanmar*.xlsx")
        if search: PROJECT_FILE_ACTUAL = search[0]
        else: return set()
    else: PROJECT_FILE_ACTUAL = PROJECT_FILE
    df = pd.read_excel(PROJECT_FILE_ACTUAL)
    col = next((c for c in df.columns if "Township" in c), None)
    if not col: return set()
    return set(df[col].apply(clean_name).unique())

def get_agri_townships():
    if Path(PROJECTS_GEOJSON).exists():
        try:
            gdf = gpd.read_file(PROJECTS_GEOJSON)
            # FIX: Updated to match the specific dataset column "E2a... assistance... type"
            cat_col = next((c for c in gdf.columns if 'assistance' in c.lower() and 'type' in c.lower()), None)
            
            name_col = next((c for c in gdf.columns if 'township' in c.lower() or 'name' in c.lower()), None)
            if cat_col and name_col:
                agri_gdf = gdf[gdf[cat_col].astype(str).str.contains('Agri|Farm|Live|Food', case=False, na=False)]
                return set(agri_gdf[name_col].apply(clean_name).unique())
        except: pass
    return set()

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

def load_data():
    print("Loading Data for Stats...")
    
    # 1. Load Panel Files
    files = glob.glob(f"{DATA_DIR}/gee_panel_myanmar_PART*.csv")
    if not files:
        print("  ! Error: No panel files found.")
        return pd.DataFrame()

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    
    # Rename GEE Columns
    rename_map = {'admin3_id': 'township_id', 'admin3_nam': 'township_name', 
                  'NDVI': 'ndvi', 'NDBI': 'ndbi', 'RAIN_TOTAL': 'rainfall', 
                  'NL_AVG_RAD': 'nightlights', 'VH': 'vh', 'VV': 'vv'}
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})
    
    # Fix standard column name for nightlights if missing
    if 'nightlights' not in df.columns and 'nightlights_gee' in df.columns:
        df['nightlights'] = df['nightlights_gee']

    df['period_start'] = pd.to_datetime(df['period_start'])

    # --- LOAD OFFICIAL TREATMENT ---
    # Load official treatment definitions from 01_main_analysis_FINAL.py
    treatment_df = load_clean_project_data()

    # Create clean names for matching
    if 'clean_name' not in df.columns:
        df['clean_name'] = df['township_name'].apply(clean_name)

    # Map treatment (binary 0/1)
    treatment_map = dict(zip(treatment_df['clean_name'], treatment_df['undp_treated']))
    df['treated'] = df['clean_name'].map(treatment_map).fillna(0).astype(int)

    # Map agriculture-specific treatment (for heterogeneity)
    if 'has_agri' in treatment_df.columns:
        agri_map = dict(zip(treatment_df['clean_name'], treatment_df['has_agri']))
        df['treated_agri'] = df['clean_name'].map(agri_map).fillna(0).astype(int)
    else:
        df['treated_agri'] = 0
    # --- END TREATMENT LOADING ---
    
    df['name_clean'] = df['township_name'].apply(clean_name)

    # Local NL Merge - FIXED SUFFIX ORDER
    if Path(LOCAL_NL_FILE).exists():
        print("    -> Merging Local Nightlights...")
        df_local = pd.read_csv(LOCAL_NL_FILE)
        df_local['period_start'] = pd.to_datetime(df_local['period_start'])
        df_local['name_clean'] = df_local['township_name'].apply(clean_name)
        
        # CRITICAL FIX: Correct suffix order
        df = pd.merge(df, df_local[['name_clean', 'period_start', 'nightlights']], 
                      on=['name_clean', 'period_start'], 
                      how='outer',  # Keep all dates including April-Sept
                      suffixes=('_gee', ''))  # ✅ CORRECT ORDER
        
        # Combine: Prioritize local data
        if 'nightlights_gee' in df.columns:
            df['nightlights'] = df['nightlights'].combine_first(df['nightlights_gee'])
        
        # Backfill township_name for new rows
        df['township_name'] = df['township_name'].fillna(df['name_clean'].str.title())

    # ACLED Conflict Data Integration
    acled_stats, acled_id_col = process_acled_data(BOUNDARIES_FILE)
    if acled_stats is not None and acled_id_col:
        print("    -> Merging ACLED Conflict Data...")
        df = pd.merge(df, acled_stats, 
                     left_on=['township_id', 'period_start'],
                     right_on=[acled_id_col, 'period_start'],
                     how='left')
        df['conflict_events'] = df['conflict_events'].fillna(0)
        if 'fatalities' in df.columns:
            df['fatalities'] = df['fatalities'].fillna(0)

    # 2. WFP Merge (With Boundary Fix)
    if Path(BOUNDARIES_FILE).exists():
        gdf_bounds = gpd.read_file(BOUNDARIES_FILE)
        rename_geo = {'adm3_name': 'township_name', 'adm1_name': 'state_name'}
        gdf_bounds = gdf_bounds.rename(columns=rename_geo)
        
        # Fallback
        if 'township_name' not in gdf_bounds.columns:
             t_col = next((c for c in gdf_bounds.columns if c.lower() in ['admin3_nam', 'ts_name']), None)
             if t_col: gdf_bounds = gdf_bounds.rename(columns={t_col: 'township_name'})
        if 'state_name' not in gdf_bounds.columns:
             s_col = next((c for c in gdf_bounds.columns if c.lower() in ['admin1_nam', 'st_name']), None)
             if s_col: gdf_bounds = gdf_bounds.rename(columns={s_col: 'state_name'})

        wfp_df = load_and_process_wfp(f"{EXTERNAL_DIR}/wfp_prices.csv", gdf_bounds)
        if wfp_df is not None:
            df = pd.merge(df, wfp_df, on=['name_clean', 'period_start'], how='left')
            # Interpolate
            df['price_rice'] = df.groupby('township_id')['price_rice'].transform(lambda x: x.interpolate(method='linear').ffill().bfill())

    df['ym_str'] = df['period_start'].dt.strftime('%Y-%m')
    df = df[~df['ym_str'].isin(EXCLUDE_MONTHS)].copy()
    df = calculate_sar_indices(df)
    return df

def get_spatial_subset(df, radius_deg, treatment_col='treated'):
    if not Path(BOUNDARIES_FILE).exists(): return df
    gdf = gpd.read_file(BOUNDARIES_FILE)
    id_col = next((c for c in gdf.columns if 'pcode' in c.lower()), None)
    
    treated_ids = df[df[treatment_col] == 1]['township_id'].unique()
    treated_geoms = gdf[gdf[id_col].isin(treated_ids)]

    if treated_geoms.empty:
        print("    ! Warning: No treated units found for this category. Returning empty structure.")
        # Return empty DF but WITH the 'group' column to prevent KeyError downstream
        empty_df = pd.DataFrame(columns=list(df.columns) + ['group'])
        return empty_df

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        treated_buffer = treated_geoms.geometry.buffer(radius_deg).unary_union
        potential_neighbors = gdf[gdf.geometry.intersects(treated_buffer)]
    
    neighbor_ids = potential_neighbors[~potential_neighbors[id_col].isin(treated_ids)][id_col].unique()
    
    df_treated = df[df['township_id'].isin(treated_ids)].copy()
    df_treated['group'] = 'Treated'
    
    df_control = df[df['township_id'].isin(neighbor_ids)].copy()
    df_control['group'] = 'Control'
    
    return pd.concat([df_treated, df_control])

def generate_row(df_sub, variable, label, period_label):
    t_vals = df_sub[df_sub['group']=='Treated'][variable].dropna()
    t_mean = t_vals.mean()
    t_sd = t_vals.std()
    t_n = len(t_vals)
    
    c_vals = df_sub[df_sub['group']=='Control'][variable].dropna()
    c_mean = c_vals.mean()
    c_sd = c_vals.std()
    c_n = len(c_vals)
    
    return {
        'Variable': label,
        'Period': period_label,
        'Treated Mean': f"{t_mean:.3f}" if t_n > 0 else "-",
        'Treated SD': f"({t_sd:.3f})" if t_n > 0 else "-",
        'Control Mean': f"{c_mean:.3f}" if c_n > 0 else "-",
        'Control SD': f"({c_sd:.3f})" if c_n > 0 else "-",
        'Diff': f"{(t_mean - c_mean):.3f}" if (t_n > 0 and c_n > 0) else "-",
        'Obs (Treated)': f"{t_n}",
        'Obs (Control)': f"{c_n}"
    }

def process_panel(df_subset, vars_map, title):
    rows = []
    rows.append({'Variable': title, 'Period': '', 'Treated Mean': '', 'Treated SD': '', 'Control Mean': '', 'Control SD': '', 'Diff': '', 'Obs (Treated)': '', 'Obs (Control)': ''})
    
    df_pre = df_subset[df_subset['period_start'] < UNDP_START_DATE]
    df_post = df_subset[df_subset['period_start'] >= UNDP_START_DATE]

    for var, label in vars_map:
        if var in df_subset.columns:
            rows.append(generate_row(df_pre, var, label, "Pre-Quake"))
            rows.append(generate_row(df_post, var, "", "Post-Quake"))
            
    return rows

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    df = load_data()

    vars_map = [
        ('nightlights', 'Nightlights (Radiance)'),
        ('ndvi', 'Vegetation (NDVI)'),
        ('rvi', 'Radar Vegetation (RVI)'),
        ('ndbi', 'Construction (NDBI)'),
        ('r_ndbi', 'Radar Structure (R-NDBI)'),
        ('rainfall', 'Rainfall (mm)'),
        ('price_rice', 'Rice Price (MMK/kg)'), # <--- ADDED THIS
        ('flood_proxy_vv', 'Flood Proxy (VV dB)'),
        ('structure_proxy_vh', 'Structure Proxy (VH dB)')
    ]
    
    final_rows = []
    
    print("Processing Panel A...")
    df_std = get_spatial_subset(df, SPATIAL_RADIUS_DEG, 'treated')
    final_rows.extend(process_panel(df_std, vars_map, "PANEL A: ALL UNDP PROJECTS (25km)"))
    
    final_rows.append({'Variable': '', 'Period': '', 'Treated Mean': '', 'Treated SD': '', 'Control Mean': '', 'Control SD': '', 'Diff': '', 'Obs (Treated)': '', 'Obs (Control)': ''})
    
    print("Processing Panel B...")
    df_agri = get_spatial_subset(df, SPATIAL_RADIUS_DEG, 'treated_agri')
    final_rows.extend(process_panel(df_agri, vars_map, "PANEL B: AGRICULTURE PROJECTS ONLY"))

    res_df = pd.DataFrame(final_rows)
    cols = ['Variable', 'Period', 'Treated Mean', 'Treated SD', 'Control Mean', 'Control SD', 'Diff', 'Obs (Treated)', 'Obs (Control)']
    res_df = res_df[cols]
    
    out_file = f"{OUTPUT_DIR}/table_descriptive_stats_split.csv"
    res_df.to_csv(out_file, index=False)
    print(f"\nTime-Split Descriptive Stats saved to: {out_file}")

if __name__ == "__main__":
    main()
