#!/usr/bin/env python3
"""
01_main_analysis.py
===================
Main Analysis: TWFE (Two-Way Fixed Effects)
UPDATED: 
- Includes 'Donut' Placebo.
- Adds Solar Interaction for Nightlights.
- Adds Shelter Interaction for Construction/Debris.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import re
from difflib import get_close_matches
import geopandas as gpd
import warnings
from linearmodels.panel import PanelOLS

# --- CONFIGURATION ---
DATA_DIR = "data/myanmar"
OUTPUT_DIR = "outputs/myanmar"
PROJECT_FILE = "data/myanmar/undp/project_data.xlsx"
PROJECTS_GEOJSON = "data/myanmar/undp/undp_interventions.geojson"
BOUNDARIES_FILE = "data/myanmar/myanmar_boundaries/mmr_admin3.geojson"
LOCAL_NL_FILE = "outputs/myanmar/local_nightlights_panel.csv"
EARTHQUAKE_DATE = pd.Timestamp("2025-03-28")
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

def _get_category_townships(keywords):
    """Generic helper to find townships for specific project categories."""
    if Path(PROJECTS_GEOJSON).exists():
        try:
            gdf = gpd.read_file(PROJECTS_GEOJSON)
            cat_col = next((c for c in gdf.columns if 'type' in c.lower() or 'sector' in c.lower() or 'cluster' in c.lower()), None)
            name_col = next((c for c in gdf.columns if 'township' in c.lower() or 'name' in c.lower()), None)
            if cat_col and name_col:
                # Join keywords with | for regex OR
                regex = "|".join(keywords)
                sub_gdf = gdf[gdf[cat_col].astype(str).str.contains(regex, case=False, na=False)]
                return set(sub_gdf[name_col].apply(clean_name).unique())
        except: pass
    return set()

def get_agri_townships():
    return _get_category_townships(['Agri', 'Farm', 'Live', 'Food'])

def get_solar_townships():
    return _get_category_townships(['Solar', 'Energy', 'Renewable', 'Electr'])

def get_shelter_townships():
    return _get_category_townships(['Shelter', 'Housing', 'Construct', 'Repair', 'Infra'])

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

def load_and_clean_data():
    print("Loading Data...")
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
    
    # 1. Base Treatment (Any Project)
    treated_all = get_treated_townships()
    df['treated'] = apply_treatment_map(df, treated_all)

    # 2. Specific Categories
    df['is_agri_project'] = apply_treatment_map(df, get_agri_townships())
    df['is_solar_project'] = apply_treatment_map(df, get_solar_townships())
    df['is_shelter_project'] = apply_treatment_map(df, get_shelter_townships())

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
    df = df[(df['period_start'] >= '2024-10-01') & (df['period_start'] <= '2025-09-30')]
    df = df[~df['ym_str'].isin(EXCLUDE_MONTHS)].copy()
    
    df['post'] = (df['period_start'] > EARTHQUAKE_DATE).astype(int)
    
    # --- INTERACTION TERMS ---
    df['treated_x_post'] = df['treated'] * df['post']
    
    # Specific Interactions
    df['agri_x_post'] = df['is_agri_project'] * df['post']
    df['solar_x_post'] = df['is_solar_project'] * df['post']
    df['shelter_x_post'] = df['is_shelter_project'] * df['post']
    
    df = calculate_sar_indices(df)

    for col in ['nightlights', 'ndvi', 'ndbi', 'rainfall', 'rvi']:
        if col in df.columns: 
            df[f'log_{col}'] = np.log1p(df[col].clip(lower=0))

    return df

def get_spatial_subset(df, radius_deg, min_radius_deg=0):
    if not Path(BOUNDARIES_FILE).exists(): return df
    gdf = gpd.read_file(BOUNDARIES_FILE)
    id_col = next((c for c in gdf.columns if 'pcode' in c.lower()), None)
    
    treated_ids = df[df['treated'] == 1]['township_id'].unique()
    treated_geoms = gdf[gdf[id_col].isin(treated_ids)]
    if treated_geoms.empty: return df

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        outer_buffer = treated_geoms.geometry.buffer(radius_deg).unary_union
        potential_neighbors = gdf[gdf.geometry.intersects(outer_buffer)]
        
        if min_radius_deg > 0:
            inner_buffer = treated_geoms.geometry.buffer(min_radius_deg).unary_union
            valid_neighbors = potential_neighbors[
                (~potential_neighbors.geometry.intersects(inner_buffer)) & 
                (~potential_neighbors[id_col].isin(treated_ids))
            ]
            neighbor_ids = valid_neighbors[id_col].unique()
        else:
            neighbor_ids = potential_neighbors[id_col].unique()

    valid_ids = np.concatenate([treated_ids, neighbor_ids])
    return df[df['township_id'].isin(valid_ids)].copy()

def run_regression(df, outcome, sample_name, controls, interaction_var=None):
    model_data = df.copy()
    model_data = model_data.dropna(subset=[outcome])
    if model_data.duplicated(subset=['township_id', 'period_start']).any():
        model_data = model_data.drop_duplicates(subset=['township_id', 'period_start'])
    model_data = model_data.set_index(['township_id', 'period_start'])
    
    # Base Formula
    formula = f"{outcome} ~ treated_x_post + EntityEffects + TimeEffects"
    
    # Interaction Formula
    if interaction_var:
        formula = f"{outcome} ~ treated_x_post + {interaction_var} + EntityEffects + TimeEffects"
        
    if controls: formula += " + " + " + ".join(controls)
    
    try:
        mod = PanelOLS.from_formula(formula, data=model_data)
        res = mod.fit(cov_type='clustered', cluster_entity=True)
        
        coef = res.params['treated_x_post']
        pval = res.pvalues['treated_x_post']
        stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        
        row = {
            'Outcome': outcome,
            'Sample': sample_name,
            'Controls': " + ".join(controls) if controls else "None",
            'Coeff': f"{coef:.4f}{stars}",
            'SE': f"({res.std_errors['treated_x_post']:.4f})",
            'Obs': int(res.nobs),
            'R2': f"{res.rsquared:.3f}",
            'Raw_Coef': coef, 
            'Raw_Pval': pval
        }

        rows = [row]
        if interaction_var:
            # Check if interaction term exists in results (it might be dropped if colinear)
            if interaction_var in res.params:
                i_coef = res.params[interaction_var]
                i_pval = res.pvalues[interaction_var]
                i_stars = "***" if i_pval < 0.01 else "**" if i_pval < 0.05 else "*" if i_pval < 0.1 else ""
                
                row_int = row.copy()
                # Label format: "Sample Name (Tag Interaction)"
                tag = interaction_var.replace('_x_post', '').title()
                row_int['Sample'] = f"{sample_name} ({tag} Interaction)"
                row_int['Coeff'] = f"{i_coef:.4f}{i_stars}"
                row_int['SE'] = f"({res.std_errors[interaction_var]:.4f})"
                row_int['Raw_Coef'] = i_coef
                row_int['Raw_Pval'] = i_pval
                rows.append(row_int)

        return rows
    except Exception as e:
        print(f"Failed {outcome} {sample_name}: {e}")
        return []

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    df_full = load_and_clean_data()
    
    subsets = {
        'Spatial 5km': get_spatial_subset(df_full, 0.045),
        'Spatial 15km': get_spatial_subset(df_full, 0.14),
        'Spatial 25km': get_spatial_subset(df_full, 0.25),
        'Spatial 50km': get_spatial_subset(df_full, 0.5),
        'Donut (15-50km)': get_spatial_subset(df_full, 0.5, min_radius_deg=0.14)
    }

    table_rows = []
    
    # 1. Standard Analysis (ALL Outcomes)
    outcomes_std = ['nightlights', 'log_nightlights', 'ndvi', 'rvi', 'ndbi', 'r_ndbi', 'structure_proxy_vh', 'rainfall', 'flood_proxy_vv']
    print("\n--- RUNNING STANDARD ANALYSIS ---")
    for outcome in outcomes_std:
        print(f"Standard: {outcome}")
        for sample_name, df_sub in subsets.items():
            table_rows.extend(run_regression(df_sub, outcome, sample_name, [], interaction_var=None))
            if 'rainfall' not in outcome:
                table_rows.extend(run_regression(df_sub, outcome, sample_name, ['rainfall'], interaction_var=None))

    # 2. Agri Interaction (Vegetation)
    outcomes_agri = ['ndvi', 'rvi']
    print("\n--- RUNNING AGRI-INTERACTION ---")
    for outcome in outcomes_agri:
        print(f"Agri: {outcome}")
        for sample_name, df_sub in subsets.items():
            table_rows.extend(run_regression(df_sub, outcome, sample_name, ['rainfall'], interaction_var='agri_x_post'))

    # 3. Solar Interaction (Nightlights)
    outcomes_solar = ['nightlights', 'log_nightlights']
    print("\n--- RUNNING SOLAR-INTERACTION ---")
    for outcome in outcomes_solar:
        print(f"Solar: {outcome}")
        for sample_name, df_sub in subsets.items():
            table_rows.extend(run_regression(df_sub, outcome, sample_name, ['rainfall'], interaction_var='solar_x_post'))

    # 4. Shelter Interaction (Construction/Debris)
    outcomes_shelter = ['ndbi', 'r_ndbi', 'structure_proxy_vh']
    print("\n--- RUNNING SHELTER-INTERACTION ---")
    for outcome in outcomes_shelter:
        print(f"Shelter: {outcome}")
        for sample_name, df_sub in subsets.items():
            table_rows.extend(run_regression(df_sub, outcome, sample_name, ['rainfall'], interaction_var='shelter_x_post'))

    summary_df = pd.DataFrame(table_rows)
    summary_df.to_csv(f"{OUTPUT_DIR}/regression_results_sar_extended.csv", index=False)
    print(f"\nSaved Full Results to: {OUTPUT_DIR}/regression_results_sar_extended.csv")

if __name__ == "__main__":
    main()
