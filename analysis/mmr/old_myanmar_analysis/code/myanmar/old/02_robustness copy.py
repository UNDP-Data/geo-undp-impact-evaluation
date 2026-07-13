#!/usr/bin/env python3
"""
02_robustness.py
================
Robustness: Event Study (TWFE)
UPDATED: Includes safety checks for Rank Deficient matrices (e.g., missing NDVI data).
"""

import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
import geopandas as gpd
from pathlib import Path
import warnings
import os
import glob
import re
from difflib import get_close_matches

# --- CONFIGURATION ---
DATA_DIR = "data/myanmar"
OUTPUT_DIR = "outputs/myanmar/robustness"
BOUNDARIES_FILE = "data/myanmar/myanmar_boundaries/mmr_admin3.geojson"
PROJECT_FILE = "data/myanmar/undp/project_data.xlsx"
PROJECTS_GEOJSON = "data/myanmar/undp/undp_interventions.geojson"
LOCAL_NL_FILE = "outputs/myanmar/local_nightlights_panel.csv"
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
            cat_col = next((c for c in gdf.columns if 'type' in c.lower() or 'sector' in c.lower() or 'cluster' in c.lower()), None)
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

def calculate_sar_indices(df):
    if 'vh' not in df.columns or 'vv' not in df.columns: return df
    vh_lin = 10 ** (df['vh'].clip(lower=-30) / 10.0)
    vv_lin = 10 ** (df['vv'].clip(lower=-30) / 10.0)
    df['rvi'] = (4 * vh_lin) / (vv_lin + vh_lin)
    df['r_ndbi'] = (vv_lin - vh_lin) / (vv_lin + vh_lin)
    df['flood_proxy_vv'] = df['vv'] 
    df['structure_proxy_vh'] = df['vh']
    return df

def load_and_clean_data():
    print("Loading Data (Parts)...")
    part_files = sorted(glob.glob(os.path.join(DATA_DIR, "gee_panel_myanmar_PART*.csv")))
    if not part_files: raise FileNotFoundError("No GEE PART files found.")
    df = pd.concat([pd.read_csv(f) for f in part_files], ignore_index=True)
    
    rename_map = {
        'admin3_id': 'township_id', 'admin3_nam': 'township_name', 
        'NDVI': 'ndvi', 'NDBI': 'ndbi', 
        'RAIN_MEAN': 'rainfall', 'NL_AVG_RAD': 'nightlights_gee',
        'VH': 'vh', 'VV': 'vv' 
    }
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
    for col in ['nightlights', 'ndvi', 'ndbi', 'rainfall', 'rvi']:
        if col in df.columns: 
            df[f'log_{col}'] = np.log1p(df[col].clip(lower=0))
            
    return df

def get_spatial_subset(df, radius_deg, treatment_col):
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
    neighbor_ids = neighbors[id_col].unique()
    valid_ids = np.concatenate([treated_ids, neighbor_ids])
    return df[df['township_id'].isin(valid_ids)].copy()

def run_event_study(df, outcome, output_dir, treatment_col='treated', suffix='', controls=[]):
    """ Standard Event Study: Y ~ Treated*Month + FEs """
    print(f"--- Event Study: {outcome.upper()} {suffix} ---")
    if outcome not in df.columns: return

    df_subset = get_spatial_subset(df, SPATIAL_RADIUS_DEG, treatment_col)
    
    # Drop rows where outcome is NaN (Crucial for NDVI/NDBI)
    df_subset = df_subset.dropna(subset=[outcome])
    
    df_subset['period_start'] = pd.to_datetime(df_subset['period_start'])
    df_subset['rel_month'] = ((df_subset['period_start'].dt.year - 2025) * 12 + (df_subset['period_start'].dt.month - 3))
    
    periods = sorted(df_subset['rel_month'].unique())
    reference = -1
    
    interaction_cols = []
    for t in periods:
        if t == reference: continue
        label = f"T_x_m{abs(t)}{'pre' if t < 0 else 'post'}"
        
        # FIX: Check for variance. If column is all 0, do not add it.
        col_data = ((df_subset['rel_month'] == t) & (df_subset[treatment_col] == 1)).astype(int)
        if col_data.sum() > 0:
            df_subset[label] = col_data
            interaction_cols.append(label)
        else:
            print(f"    ! Skipping {label}: No valid treated observations (rank safety).")
    
    if df_subset.duplicated(subset=['township_id', 'period_start']).any():
        df_subset = df_subset.drop_duplicates(subset=['township_id', 'period_start'])
    
    if df_subset[treatment_col].sum() == 0:
        print(f"  ! No treated units for {suffix}. Skipping.")
        return

    model_data = df_subset.set_index(['township_id', 'period_start'])
    formula = f"{outcome} ~ " + " + ".join(interaction_cols) + " + EntityEffects + TimeEffects"
    if controls: formula += " + " + " + ".join(controls)

    try:
        mod = PanelOLS.from_formula(formula, data=model_data)
        res = mod.fit(cov_type='clustered', cluster_entity=True)
        
        results = [{'t': reference, 'coef': 0.0, 'err': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0}]
        for t in periods:
            if t == reference: continue
            label = f"T_x_m{abs(t)}{'pre' if t < 0 else 'post'}"
            if label in res.params:
                coef = res.params[label]
                se = res.std_errors[label]
                results.append({'t': t, 'coef': coef, 'err': se, 'ci_lower': coef - 1.96*se, 'ci_upper': coef + 1.96*se})
        
        res_df = pd.DataFrame(results).sort_values('t')
        res_df.to_csv(f"{output_dir}/es_coeffs_{outcome}{suffix}.csv", index=False)
        print(f"  -> Saved {outcome}{suffix} coefficients.")
    except Exception as e:
        print(f"  ! Failed: {e}")

def run_interaction_event_study(df, outcome, output_dir, controls=[]):
    """
    Interaction Event Study: Y ~ Treated*Time + Agri*Time + FEs.
    Extracts the 'Agri*Time' coefficients to visualize the 'Agri Premium' over time.
    """
    print(f"--- Interaction Event Study: {outcome.upper()} ---")
    
    # 1. Setup Data: Use standard treated subset (all projects)
    df_subset = get_spatial_subset(df, SPATIAL_RADIUS_DEG, 'treated') 
    
    # Drop rows where outcome is NaN
    df_subset = df_subset.dropna(subset=[outcome])
    
    df_subset['rel_month'] = ((df_subset['period_start'].dt.year - 2025) * 12 + (df_subset['period_start'].dt.month - 3))
    
    # 2. Create Terms
    periods = sorted(df_subset['rel_month'].unique())
    reference = -1
    
    base_terms = []
    agri_terms = []
    
    for t in periods:
        if t == reference: continue
        label_t = f"T_x_m{abs(t)}{'pre' if t < 0 else 'post'}"
        label_a = f"Agri_x_m{abs(t)}{'pre' if t < 0 else 'post'}"
        
        # Base Effect (Treated * Month)
        col_t = ((df_subset['rel_month'] == t) & (df_subset['treated'] == 1)).astype(int)
        if col_t.sum() > 0:
            df_subset[label_t] = col_t
            base_terms.append(label_t)

        # Interaction Effect (Agri * Month) - treated_agri is subset of treated
        col_a = ((df_subset['rel_month'] == t) & (df_subset['treated_agri'] == 1)).astype(int)
        if col_a.sum() > 0:
            df_subset[label_a] = col_a
            agri_terms.append(label_a)
        else:
            print(f"    ! Skipping {label_a}: No valid Agri observations (rank safety).")

    if df_subset.duplicated(subset=['township_id', 'period_start']).any():
        df_subset = df_subset.drop_duplicates(subset=['township_id', 'period_start'])

    # 3. Run Model
    model_data = df_subset.set_index(['township_id', 'period_start'])
    
    # Check if we have enough terms to run
    if not agri_terms:
        print("    ! Not enough Agri variation to run interaction model.")
        return

    formula = f"{outcome} ~ " + " + ".join(base_terms) + " + " + " + ".join(agri_terms) + " + EntityEffects + TimeEffects"
    if controls: formula += " + " + " + ".join(controls)
    
    try:
        mod = PanelOLS.from_formula(formula, data=model_data)
        res = mod.fit(cov_type='clustered', cluster_entity=True)
        
        # 4. Extract ONLY Agri Terms for the plot
        results = [{'t': reference, 'coef': 0.0, 'err': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0}]
        for t in periods:
            if t == reference: continue
            label_a = f"Agri_x_m{abs(t)}{'pre' if t < 0 else 'post'}"
            
            if label_a in res.params:
                coef = res.params[label_a]
                se = res.std_errors[label_a]
                results.append({'t': t, 'coef': coef, 'err': se, 'ci_lower': coef - 1.96*se, 'ci_upper': coef + 1.96*se})
        
        res_df = pd.DataFrame(results).sort_values('t')
        # Save with specific suffix
        res_df.to_csv(f"{output_dir}/es_coeffs_interaction_{outcome}.csv", index=False)
        print(f"  -> Saved Interaction coefficients for {outcome}.")
        
    except Exception as e:
        print(f"  ! Failed Interaction ES: {e}")

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    df = load_and_clean_data()
    controls = ['rainfall']
    
    # 1. Standard Analysis (Treated = Any)
    for out in ['log_nightlights', 'nightlights', 'ndvi', 'rvi', 'ndbi', 'r_ndbi']:
        run_event_study(df, out, OUTPUT_DIR, treatment_col='treated', controls=controls)
    
    # 2. Agricultural Variation (Treated = Agri, Control = Rest)
    for out in ['ndvi', 'rvi']:
        run_event_study(df, out, OUTPUT_DIR, treatment_col='treated_agri', suffix='_agri', controls=controls)

    # 3. Interaction Analysis (Agri Premium)
    print("\n--- Running Interaction Analyses ---")
    for out in ['rvi', 'ndvi']:
        run_interaction_event_study(df, out, OUTPUT_DIR, controls=controls)

if __name__ == "__main__":
    main()
