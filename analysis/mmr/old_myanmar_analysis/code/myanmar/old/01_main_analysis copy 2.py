#!/usr/bin/env python3
"""
01_main_analysis.py
===================
Main Analysis: TWFE with External Data Integration
UPDATED: 
- Integrates ACLED (Conflict) and WFP (Prices) as Local Controls.
- Smart Control Selection: Avoids "Bad Controls" (e.g., controlling for the outcome).
- Fuzzy Matching for WFP township names.
"""

import os
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
import glob
from linearmodels.panel import PanelOLS
import re
import warnings
from difflib import get_close_matches

# --- CONFIGURATION ---
DATA_DIR = "data/myanmar"
EXTERNAL_DIR = "data/myanmar/external"
OUTPUT_DIR = "outputs/myanmar"
BOUNDARIES_FILE = "data/myanmar/myanmar_boundaries/mmr_admin3.geojson"
PROJECT_FILE = "data/myanmar/undp/project_data.xlsx"
PROJECTS_GEOJSON = "data/myanmar/undp/undp_interventions.geojson"
EARTHQUAKE_DATE = pd.Timestamp("2025-03-28")
EXCLUDE_MONTHS = ['2025-03']

def clean_name(name):
    if pd.isna(name): return ""
    return re.sub(r'[^a-z0-9]', '', str(name).lower())

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
        treatment_map[name] = 1 if cname in target_set else 0
    return df['township_name'].map(treatment_map)

# --- ACLED PROCESSING ---
def process_acled_data(boundaries_path):
    acled_path = f"{EXTERNAL_DIR}/acled_myanmar.csv"
    if not Path(acled_path).exists(): return None, None
    
    print("  > Processing ACLED Conflict Data...")
    try:
        df_acled = pd.read_csv(acled_path)
        if df_acled.empty: return None, None
        
        gdf_acled = gpd.GeoDataFrame(
            df_acled, geometry=gpd.points_from_xy(df_acled.longitude, df_acled.latitude), crs="EPSG:4326"
        )
        gdf_bounds = gpd.read_file(boundaries_path)
        id_col = next((c for c in gdf_bounds.columns if 'pcode' in c.lower()), None)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            joined = gpd.sjoin(gdf_acled, gdf_bounds[[id_col, 'geometry']], how='inner', predicate='within')
        
        joined['event_date'] = pd.to_datetime(joined['event_date'])
        joined['period_start'] = joined['event_date'].dt.to_period('M').dt.to_timestamp()
        
        monthly_stats = joined.groupby([id_col, 'period_start']).agg({
            'event_id_cnty': 'count',
            'fatalities': 'sum'
        }).rename(columns={'event_id_cnty': 'conflict_events'}).reset_index()
        
        return monthly_stats, id_col
    except Exception as e:
        print(f"    ! ACLED Error: {e}")
        return None, None

# --- WFP PRICE PROCESSING (WITH FUZZY MATCH) ---
def process_wfp_data(geo_names):
    wfp_path = f"{EXTERNAL_DIR}/wfp_prices.csv"
    if not Path(wfp_path).exists(): return None
    
    print("  > Processing WFP Price Data (Fuzzy Match)...")
    try:
        df = pd.read_csv(wfp_path, on_bad_lines='skip', low_memory=False)
        df.columns = [c.lower().strip() for c in df.columns]
        
        comm_col = next((c for c in df.columns if 'commodity' in c or 'cmname' in c), None)
        price_col = next((c for c in df.columns if ('price' in c or 'mp_price' in c) and 'unit' not in c), None)
        date_col = next((c for c in df.columns if 'date' in c), None)
        admin3_col = next((c for c in df.columns if 'township' in c or 'admin3' in c or 'admin2' in c or 'market' in c), None)
        
        if not (comm_col and price_col and date_col and admin3_col): return None

        df_rice = df[df[comm_col].astype(str).str.contains('Rice', case=False, na=False)].copy()
        if df_rice.empty: return None

        df_rice['period_start'] = pd.to_datetime(df_rice[date_col], errors='coerce').dt.to_period('M').dt.to_timestamp()
        df_rice[price_col] = pd.to_numeric(df_rice[price_col], errors='coerce')
        df_rice['wfp_clean'] = df_rice[admin3_col].apply(clean_name)
        
        # Fuzzy Match
        official_names = list(geo_names)
        name_map = {}
        for wfp_name in df_rice['wfp_clean'].unique():
            if not wfp_name: continue
            if wfp_name in official_names:
                name_map[wfp_name] = wfp_name
            else:
                matches = get_close_matches(wfp_name, official_names, n=1, cutoff=0.7)
                if matches: name_map[wfp_name] = matches[0]
        
        df_rice['name_clean'] = df_rice['wfp_clean'].map(name_map)
        
        price_panel = df_rice.groupby(['name_clean', 'period_start'])[price_col].mean().reset_index()
        price_panel = price_panel.rename(columns={price_col: 'price_rice'})
        return price_panel
    except: return None

def load_and_clean_data():
    print("Loading Data...")
    part_files = sorted(glob.glob(os.path.join(DATA_DIR, "gee_panel_myanmar_PART*.csv")))
    df = pd.concat([pd.read_csv(f) for f in part_files], ignore_index=True)
    
    rename_map = {'admin3_id': 'township_id', 'admin3_nam': 'township_name', 
                  'NDVI': 'ndvi', 'NDBI': 'ndbi', 'RAIN_TOTAL': 'rainfall', 
                  'NL_AVG_RAD': 'nightlights', 'VH': 'vh', 'VV': 'vv'}
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})
    df['period_start'] = pd.to_datetime(df['period_start'])
    
    df['treated'] = apply_treatment_map(df, get_treated_townships())
    df['name_clean'] = df['township_name'].apply(clean_name)
    
    # Merge ACLED
    acled_df, id_col_bounds = process_acled_data(BOUNDARIES_FILE)
    if acled_df is not None:
        df = pd.merge(df, acled_df, left_on=['township_id', 'period_start'], right_on=[id_col_bounds, 'period_start'], how='left')
        df['conflict_events'] = df['conflict_events'].fillna(0)
        df['log_conflict'] = np.log1p(df['conflict_events'])

    # Merge WFP
    unique_townships = set(df['name_clean'].unique())
    wfp_df = process_wfp_data(unique_townships)
    if wfp_df is not None:
        df = pd.merge(df, wfp_df, on=['name_clean', 'period_start'], how='left')
        df['log_price_rice'] = np.log1p(df['price_rice'])

    # Final Setup
    df['ym_str'] = df['period_start'].dt.strftime('%Y-%m')
    df = df[~df['ym_str'].isin(EXCLUDE_MONTHS)].copy()
    df['post'] = (df['period_start'] > EARTHQUAKE_DATE).astype(int)
    df['treated_x_post'] = df['treated'] * df['post']
    
    # Interactions
    df['is_agri'] = apply_treatment_map(df, get_agri_townships())
    df['is_solar'] = apply_treatment_map(df, get_solar_townships())
    df['is_shelter'] = apply_treatment_map(df, get_shelter_townships())
    
    df['agri_x_post'] = df['is_agri'] * df['post']
    df['solar_x_post'] = df['is_solar'] * df['post']
    df['shelter_x_post'] = df['is_shelter'] * df['post']
    
    df = calculate_sar_indices(df)
    for col in ['nightlights', 'ndvi', 'ndbi', 'rainfall', 'rvi']:
        if col in df.columns: df[f'log_{col}'] = np.log1p(df[col].clip(lower=0))

    return df

def get_spatial_subset(df, radius_deg, min_radius_deg=0):
    if not Path(BOUNDARIES_FILE).exists(): return df
    gdf = gpd.read_file(BOUNDARIES_FILE)
    id_col = next((c for c in gdf.columns if 'pcode' in c.lower()), None)
    
    treated_ids = df[df['treated'] == 1]['township_id'].unique()
    treated_geoms = gdf[gdf[id_col].isin(treated_ids)]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        outer = treated_geoms.geometry.buffer(radius_deg).unary_union
        potential = gdf[gdf.geometry.intersects(outer)]
        
        if min_radius_deg > 0:
            inner = treated_geoms.geometry.buffer(min_radius_deg).unary_union
            valid = potential[(~potential.geometry.intersects(inner)) & (~potential[id_col].isin(treated_ids))]
            neighbor_ids = valid[id_col].unique()
        else:
            neighbor_ids = potential[id_col].unique()

    valid_ids = np.concatenate([treated_ids, neighbor_ids])
    return df[df['township_id'].isin(valid_ids)].copy()

def run_regression(df, outcome, sample_name, controls, interaction_var=None):
    model_data = df.copy()
    cols = [outcome] + controls
    if interaction_var: cols.append(interaction_var)
    
    missing = [c for c in cols if c not in model_data.columns]
    if missing: return []

    model_data = model_data.dropna(subset=cols).set_index(['township_id', 'period_start'])
    if model_data.empty: return []
    
    formula = f"{outcome} ~ treated_x_post + EntityEffects + TimeEffects"
    if interaction_var: formula = f"{outcome} ~ treated_x_post + {interaction_var} + EntityEffects + TimeEffects"
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
        
        if interaction_var and interaction_var in res.params:
            i_coef = res.params[interaction_var]
            i_pval = res.pvalues[interaction_var]
            i_stars = "***" if i_pval < 0.01 else "**" if i_pval < 0.05 else "*" if i_pval < 0.1 else ""
            i_row = row.copy()
            tag = interaction_var.replace('_x_post', '').title()
            i_row['Sample'] = f"{sample_name} ({tag} Interaction)"
            i_row['Coeff'] = f"{i_coef:.4f}{i_stars}"
            i_row['SE'] = f"({res.std_errors[interaction_var]:.4f})"
            i_row['Raw_Coef'] = i_coef
            i_row['Raw_Pval'] = i_pval
            rows.append(i_row)
            
        return rows
    except: return []

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
    
    # Available Variables
    has_conflict = 'log_conflict' in df_full.columns
    has_price = 'log_price_rice' in df_full.columns
    
    # 1. Outcomes List
    outcomes = ['nightlights', 'log_nightlights', 'ndvi', 'rvi', 'ndbi', 'r_ndbi', 'structure_proxy_vh', 'flood_proxy_vv']
    
    # Add New Outcomes if available
    if has_conflict: outcomes.append('log_conflict')
    if has_price: outcomes.append('log_price_rice')

    print("\n--- RUNNING ANALYSIS ---")
    for outcome in outcomes:
        print(f"  > {outcome}")
        
        # SMART CONTROLS SELECTION
        # Base: Rain
        current_controls = ['rainfall']
        
        # If Outcome is Nightlights/Infrastructure -> Control for Conflict + Price
        if outcome in ['nightlights', 'log_nightlights', 'ndbi', 'r_ndbi', 'structure_proxy_vh']:
            if has_conflict: current_controls.append('log_conflict')
            if has_price: current_controls.append('log_price_rice')
            
        # If Outcome is Conflict -> Control for Price (Grievance)
        elif outcome == 'log_conflict':
            if has_price: current_controls.append('log_price_rice')
            
        # If Outcome is Price -> Control for Conflict (Supply Shock)
        elif outcome == 'log_price_rice':
            if has_conflict: current_controls.append('log_conflict')

        for name, df_sub in subsets.items():
            # A. Full Controls (Smart)
            table_rows.extend(run_regression(df_sub, outcome, name, current_controls))
            # B. No Controls (Robustness)
            table_rows.extend(run_regression(df_sub, outcome, name, []))

    # 2. INTERACTIONS (With full controls)
    print("  > Interactions")
    full_controls = ['rainfall']
    if has_conflict: full_controls.append('log_conflict')
    
    for name, df_sub in subsets.items():
        table_rows.extend(run_regression(df_sub, 'rvi', name, full_controls, 'agri_x_post'))
        table_rows.extend(run_regression(df_sub, 'nightlights', name, full_controls, 'solar_x_post'))
        table_rows.extend(run_regression(df_sub, 'structure_proxy_vh', name, full_controls, 'shelter_x_post'))

    pd.DataFrame(table_rows).to_csv(f"{OUTPUT_DIR}/regression_results_sar_extended.csv", index=False)
    print(f"\nSaved Results to: {OUTPUT_DIR}/regression_results_sar_extended.csv")

if __name__ == "__main__":
    main()
