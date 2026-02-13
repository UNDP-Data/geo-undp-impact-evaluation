#!/usr/bin/env python3
"""
05_descriptive_stats.py
=======================
Generates a Descriptive Statistics / Balance Table.
Sample: Spatial 5km Boundary (Treated vs Control).
Split: Pre-Earthquake vs Post-Earthquake.
Groups: 
  1. Standard Treatment (All UNDP)
  2. Agricultural Treatment (Agri Only)
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
OUTPUT_DIR = "outputs/myanmar/tables_formatted"
BOUNDARIES_FILE = "data/myanmar/myanmar_boundaries/mmr_admin3.geojson"
PROJECT_FILE = "data/myanmar/undp/project_data.xlsx"
PROJECTS_GEOJSON = "data/myanmar/undp/undp_interventions.geojson"
LOCAL_NL_FILE = "outputs/myanmar/local_nightlights_panel.csv"

EARTHQUAKE_DATE = pd.Timestamp("2025-03-28")
SPATIAL_RADIUS_DEG = 0.045 # 5km
EXCLUDE_MONTHS = ['2025-03']

MANUAL_MAP = {
    'mahaaungmyay': 'mandalay', 'aungmyaythazan': 'mandalay', 'chanayethazan': 'mandalay',
    'chanmyathazi': 'mandalay', 'pyigyitagon': 'mandalay', 'poke ba thi ri': 'pobbathiri',
    'za bu thi ri': 'zabuthiri', 'oke ta ra thi ri': 'ottarathiri', 'det khi na thi ri': 'dekkhinathiri',
    'zay yar thi ri': 'zeyarthiri', 'pyinoolwin': 'pyinoolwin', 'meiktila': 'meikthila',
    'taunggyi': 'taunggyi', 'nawnghkio': 'nawnghkio', 'yamethin': 'yamethin', 'patheingyi': 'patheingyi',
    'amarapura': 'amarapura', 'tada-u': 'tada-u', 'sintgaing': 'sintgaing', 'kyaukse': 'kyaukse',
    'myittha': 'myittha', 'thazi': 'thazi', 'wundwin': 'wundwin', 'mahlaing': 'mahlaing'
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
            # Robust column detection
            cat_col = None
            candidates = ['intervention_type', 'cluster', 'sector', 'type', 'category']
            for cand in candidates:
                if cand in gdf.columns: cat_col = cand; break
            if not cat_col:
                for col in gdf.columns:
                    if 'type' in col.lower() or 'sector' in col.lower(): cat_col = col; break
            
            name_col = next((c for c in gdf.columns if 'township' in c.lower() or 'name' in c.lower()), None)
            
            if cat_col and name_col:
                agri_gdf = gdf[gdf[cat_col].astype(str).str.contains('Agri|Farm|Live|Food', case=False, na=False)]
                return set(agri_gdf[name_col].apply(clean_name).unique())
        except: pass
    return set()

def apply_treatment_map(df, target_set):
    treatment_map = {}
    for name in df['township_name'].unique():
        cname = clean_name(name)
        is_treated = 0
        if cname in target_set: is_treated = 1
        if is_treated == 0:
             for proj_name, gee_equiv in MANUAL_MAP.items():
                 if clean_name(gee_equiv) in cname and clean_name(proj_name) in target_set:
                     is_treated = 1; break
        if is_treated == 0 and target_set:
             matches = get_close_matches(cname, list(target_set), n=1, cutoff=0.85)
             if matches: is_treated = 1
        treatment_map[name] = is_treated
    return df['township_name'].map(treatment_map)

def load_data():
    print("Loading Data...")
    part_files = sorted(glob.glob(os.path.join(DATA_DIR, "gee_panel_myanmar_PART*.csv")))
    if not part_files: raise FileNotFoundError("No GEE PART files found.")
    df = pd.concat([pd.read_csv(f) for f in part_files], ignore_index=True)
    
    rename_map = {'admin3_id': 'township_id', 'admin3_nam': 'township_name', 
                  'NDVI': 'ndvi', 'NDBI': 'ndbi', 'RAIN_MEAN': 'rainfall', 
                  'NL_AVG_RAD': 'nightlights_gee', 'VH': 'vh', 'VV': 'vv'}
    cols = df.columns
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in cols})
    df['period_start'] = pd.to_datetime(df['period_start'])
    
    df['treated'] = apply_treatment_map(df, get_treated_townships())
    df['treated_agri'] = apply_treatment_map(df, get_agri_townships())

    if Path(LOCAL_NL_FILE).exists():
        df_local = pd.read_csv(LOCAL_NL_FILE)
        df_local['period_start'] = pd.to_datetime(df_local['period_start'])
        df_local['name_clean'] = df_local['township_name'].apply(clean_name)
        df['name_clean'] = df['township_name'].apply(clean_name)
        df = pd.merge(df, df_local[['name_clean', 'period_start', 'nightlights']], 
                      on=['name_clean', 'period_start'], how='left', suffixes=('', '_local'))
        df['nightlights'] = df['nightlights'].combine_first(df['nightlights_gee'])
    else:
        df['nightlights'] = df['nightlights_gee']

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
    if treated_geoms.empty: return df

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        treated_buffer = treated_geoms.geometry.buffer(radius_deg).unary_union
        potential_neighbors = gdf[gdf.geometry.intersects(treated_buffer)]
    
    # Strict separation
    neighbor_ids = potential_neighbors[~potential_neighbors[id_col].isin(treated_ids)][id_col].unique()
    
    # Subset dataframe
    df_treated = df[df['township_id'].isin(treated_ids)].copy()
    df_treated['group'] = 'Treated'
    
    df_control = df[df['township_id'].isin(neighbor_ids)].copy()
    df_control['group'] = 'Control'
    
    return pd.concat([df_treated, df_control])

def generate_row(df_sub, variable, label, period_label):
    # Treated
    t_vals = df_sub[df_sub['group']=='Treated'][variable].dropna()
    t_mean = t_vals.mean()
    t_sd = t_vals.std()
    t_n = len(t_vals)
    
    # Control
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
        'N (T/C)': f"{t_n}/{c_n}"
    }

def process_panel(df_subset, vars_map, title):
    rows = []
    rows.append({'Variable': title, 'Period': '', 'Treated Mean': '', 'Treated SD': '', 'Control Mean': '', 'Control SD': '', 'Diff': '', 'N (T/C)': ''})
    
    # Split Time
    df_pre = df_subset[df_subset['period_start'] < EARTHQUAKE_DATE]
    df_post = df_subset[df_subset['period_start'] > EARTHQUAKE_DATE]
    
    for var, label in vars_map:
        if var in df_subset.columns:
            # Pre Row
            rows.append(generate_row(df_pre, var, label, "Pre-Quake"))
            # Post Row
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
        ('flood_proxy_vv', 'Flood Proxy (VV dB)'),
        ('structure_proxy_vh', 'Structure Proxy (VH dB)')
    ]
    
    final_rows = []
    
    # Panel A: All Projects
    print("Processing Panel A...")
    df_std = get_spatial_subset(df, SPATIAL_RADIUS_DEG, 'treated')
    final_rows.extend(process_panel(df_std, vars_map, "PANEL A: ALL UNDP PROJECTS"))
    
    # Spacer
    final_rows.append({'Variable': '', 'Period': '', 'Treated Mean': '', 'Treated SD': '', 'Control Mean': '', 'Control SD': '', 'Diff': '', 'N (T/C)': ''})
    
    # Panel B: Agri Projects
    print("Processing Panel B...")
    df_agri = get_spatial_subset(df, SPATIAL_RADIUS_DEG, 'treated_agri')
    final_rows.extend(process_panel(df_agri, vars_map, "PANEL B: AGRICULTURE PROJECTS ONLY"))

    # Output
    res_df = pd.DataFrame(final_rows)
    cols = ['Variable', 'Period', 'Treated Mean', 'Treated SD', 'Control Mean', 'Control SD', 'Diff', 'N (T/C)']
    res_df = res_df[cols]
    
    out_file = f"{OUTPUT_DIR}/table_descriptive_stats_split.csv"
    res_df.to_csv(out_file, index=False)
    print(f"\nTime-Split Descriptive Stats saved to: {out_file}")
    print(res_df.head(10))

if __name__ == "__main__":
    main()
