#!/usr/bin/env python3
"""
01_main_analysis_FINAL.py
==========================
FINAL TREATMENT DEFINITION:

Primary: UNDP Project List (Intent-to-Treat)
- Treatment = Township in UNDP project records
- Evaluates: "Did UNDP-targeted areas show better recovery?"

Robustness: Validated Delivery (Treatment-on-Treated)
- Treatment = UNDP township AND E1="Yes" (survey confirms delivery)
- Evaluates: "Did UNDP assistance where delivered show better recovery?"

Heterogeneity: By Intervention Type (within UNDP townships)
- Solar, Shelter, Cash, Agriculture, etc.
- Evaluates: "Which UNDP interventions were most effective?"
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

# --- CONFIGURATION ---
DATA_DIR = "data/myanmar"
EXTERNAL_DIR = "data/myanmar/external"
OUTPUT_DIR = "outputs/myanmar"
BOUNDARIES_FILE = "data/myanmar/myanmar_boundaries/mmr_admin3.geojson"
PROJECT_FILE = "data/myanmar/undp/project_data.xlsx"
LOCAL_NL_FILE = "outputs/myanmar/local_nightlights_panel.csv"
EARTHQUAKE_DATE = pd.Timestamp("2025-03-28")
UNDP_START_DATE = pd.Timestamp("2025-05-01")
EXCLUDE_MONTHS = ['2025-03']

def clean_name(name):
    if pd.isna(name): return ""
    return re.sub(r'[^a-z0-9]', '', str(name).lower())

def load_undp_project_data():
    """
    PRIMARY TREATMENT: Load UNDP project townships.
    
    Returns:
    - undp_townships: Set of township names in UNDP project list
    - e2_data: DataFrame with E2 intervention details (for heterogeneity)
    """
    
    file_path = PROJECT_FILE
    if not Path(file_path).exists():
        fallback = "data/myanmar/undp/project_data.xlsx - Dataset_cleaned_to_be_shared.csv"
        if Path(fallback).exists(): file_path = fallback
        else: 
            print("  ! Project file not found")
            return None, None
    
    try:
        if str(file_path).endswith('.csv'): df_proj = pd.read_csv(file_path)
        else: df_proj = pd.read_excel(file_path)
    except Exception as e:
        print(f"  ! Error reading project file: {e}")
        return None, None
    
    print("\n" + "="*70)
    print("TREATMENT DEFINITION: UNDP PROJECT LIST")
    print("="*70)
    
    # Get township column
    town_col = next((c for c in df_proj.columns if 'Township' in c and 'State' not in c), None)
    if not town_col: return None, None
    df_proj['clean_name'] = df_proj[town_col].apply(clean_name)
    
    # Get E1 column (for validation)
    e1_col = next((c for c in df_proj.columns if c.startswith('E1.') and 'Has your community' in c), None)
    if e1_col:
        df_proj['e1_response'] = df_proj[e1_col].astype(str).str.lower().str.strip()
    
    # Get E2 columns (for heterogeneity)
    e2a_cols = [c for c in df_proj.columns if c.startswith('E2a.') and not c.endswith('*')]
    e2b_cols = [c for c in df_proj.columns if c.startswith('E2b.') and not c.endswith('*')]
    e2c_cols = [c for c in df_proj.columns if c.startswith('E2c.') and not c.endswith('*')]
    e2_all_cols = e2a_cols + e2b_cols + e2c_cols
    
    # Convert E2 to numeric
    for col in e2_all_cols:
        df_proj[col] = pd.to_numeric(df_proj[col], errors='coerce')
    
    # Define intervention categories
    intervention_cols = {
        'solar': [c for c in e2_all_cols if 'solar' in c.lower() or 'Solar home systems' in c],
        'shelter': [c for c in e2_all_cols if 'shelter' in c.lower() or 'house repair' in c.lower() or 'construction materials' in c.lower() or 'Tool kits for house repair' in c or 'Construction materials' in c or 'safe space' in c.lower()],
        'agri': [c for c in e2_all_cols if 'agricultural' in c.lower() or 'livestock' in c.lower() or 'Agricultural inputs' in c or 'Livestock inputs' in c],
        'cash': [c for c in e2_all_cols if 'cash' in c.lower() or 'grant' in c.lower() or 'voucher' in c.lower() or 'microfinance' in c.lower() or 'Cash-for-work' in c or 'Start-up grants' in c or 'Microfinance' in c or 'Vouchers' in c],
        'health': [c for c in e2_all_cols if 'health' in c.lower() or 'medicine' in c.lower() or 'medical' in c.lower() or 'Medicines' in c or 'Medical' in c or 'Psychological support' in c],
        'wash': [c for c in e2_all_cols if 'water' in c.lower() or 'sanitation' in c.lower() or 'hygiene' in c.lower() or 'Material kits for household/community water supply' in c],
        'training': [c for c in e2_all_cols if 'training' in c.lower() or 'capacity' in c.lower() or 'awareness' in c.lower() or 'education' in c.lower() or 'TVET' in c or 'Carpenter Training' in c or 'business' in c.lower()]
    }
    
    # Aggregate by township
    results = []
    for township in df_proj['clean_name'].unique():
        town_data = df_proj[df_proj['clean_name'] == township]
        
        row = {
            'clean_name': township,
            'undp_treated': 1  # In UNDP project list
        }
        
        # E1 validation (for robustness)
        if e1_col:
            has_yes = (town_data['e1_response'] == 'yes').any()
            row['e1_yes'] = 1 if has_yes else 0
        else:
            row['e1_yes'] = np.nan
        
        # E2 intervention types (for heterogeneity)
        for intervention, cols in intervention_cols.items():
            has_intervention = any(
                town_data[col].fillna(0).sum() > 0 
                for col in cols if col in town_data.columns
            )
            row[f'has_{intervention}'] = 1 if has_intervention else 0
        
        # Treatment intensity (count of intervention types)
        row['treatment_intensity'] = sum(row[f'has_{k}'] for k in intervention_cols.keys())
        
        results.append(row)
    
    df_treatment = pd.DataFrame(results)
    
    # UNDP townships (all in project list)
    undp_townships = set(df_treatment['clean_name'].unique())
    
    print(f"\n  UNDP Treatment Group:")
    print(f"    - UNDP-targeted townships: {len(undp_townships)}")
    
    if 'e1_yes' in df_treatment.columns and not df_treatment['e1_yes'].isna().all():
        validated = (df_treatment['e1_yes'] == 1).sum()
        print(f"    - With E1='Yes' validation: {validated} ({validated/len(undp_townships)*100:.1f}%)")
    
    print(f"\n  Intervention Types (within UNDP townships):")
    for intervention in intervention_cols.keys():
        col = f'has_{intervention}'
        if col in df_treatment.columns:
            count = (df_treatment[col] == 1).sum()
            print(f"    - {intervention.capitalize()}: {count} townships")
    
    print("="*70 + "\n")
    
    return undp_townships, df_treatment

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

def get_spatial_subset(df, radius_deg, min_radius_deg=0):
    if not Path(BOUNDARIES_FILE).exists(): return df
    gdf = gpd.read_file(BOUNDARIES_FILE)
    id_col = next((c for c in gdf.columns if 'pcode' in c.lower()), None)
    treated_ids = df[df['treated'] == 1]['township_id'].unique()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        treated_geoms = gdf[gdf[id_col].isin(treated_ids)]
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
    valid_controls = [c for c in controls if c != outcome]
    cols = [outcome] + valid_controls
    if interaction_var: cols.append(interaction_var)
    
    if any(c not in model_data.columns for c in cols): return []
    model_data = model_data.dropna(subset=cols).set_index(['township_id', 'period_start'])
    if model_data.empty: return []
    
    final_controls = [c for c in valid_controls if model_data[c].std() > 1e-6]
    
    formula = f"{outcome} ~ treated_x_post + EntityEffects + TimeEffects"
    if interaction_var: formula = f"{outcome} ~ treated_x_post + {interaction_var} + EntityEffects + TimeEffects"
    if final_controls: formula += " + " + " + ".join(final_controls)
    
    try:
        mod = PanelOLS.from_formula(formula, data=model_data, check_rank=False)
        res = mod.fit(cov_type='clustered', cluster_entity=True)
        
        row = {
            'Outcome': outcome,
            'Sample': sample_name,
            'Controls': " + ".join(final_controls) if final_controls else "None",
            'Coeff': f"{res.params['treated_x_post']:.4f}{'***' if res.pvalues['treated_x_post']<0.01 else '**' if res.pvalues['treated_x_post']<0.05 else '*' if res.pvalues['treated_x_post']<0.1 else ''}",
            'SE': f"({res.std_errors['treated_x_post']:.4f})",
            'P-value': res.pvalues['treated_x_post'],
            'Obs': int(res.nobs),
            'R2': f"{res.rsquared:.3f}"
        }
        return [row]
    except: return []

def load_and_clean_data(undp_townships, treatment_df, treatment_type='main'):
    """
    Load panel data and apply treatment.
    
    treatment_type:
    - 'main': UNDP townships (Intent-to-Treat)
    - 'validated': UNDP townships AND E1=Yes (Treatment-on-Treated)
    - intervention name: For heterogeneity (e.g., 'solar', 'shelter')
    """
    
    print(f"Loading Data (treatment: {treatment_type})...")
    
    files = glob.glob(f"{DATA_DIR}/gee_panel_myanmar_PART*.csv")
    if not files: return pd.DataFrame()
    
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    
    rename_map = {'admin3_id': 'township_id', 'admin3_nam': 'township_name', 
                  'NDVI': 'ndvi', 'NDBI': 'ndbi', 'RAIN_TOTAL': 'rainfall', 
                  'NL_AVG_RAD': 'nightlights_gee', 'VH': 'vh', 'VV': 'vv'}
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})
    df['period_start'] = pd.to_datetime(df['period_start'])
    df['clean_name'] = df['township_name'].apply(clean_name)
    df['name_clean'] = df['clean_name']
    
    # Merge local nightlights
    if Path(LOCAL_NL_FILE).exists():
        print("    -> Merging Local Nightlights...")
        df_local = pd.read_csv(LOCAL_NL_FILE)
        df_local['period_start'] = pd.to_datetime(df_local['period_start'])
        df_local['name_clean'] = df_local['township_name'].apply(clean_name)
        df = pd.merge(df, df_local[['name_clean', 'period_start', 'nightlights']], 
                      on=['name_clean', 'period_start'], how='outer', suffixes=('_gee', ''))
        if 'nightlights_gee' in df.columns:
            df['nightlights'] = df['nightlights'].combine_first(df['nightlights_gee'])
        df['township_name'] = df['township_name'].fillna(df['name_clean'].str.title())
        df = df.sort_values(['name_clean', 'period_start'])
    
    # Merge treatment data
    df = pd.merge(df, treatment_df, on='clean_name', how='left')
    
    # Define treatment based on type
    if treatment_type == 'main':
        # Primary: UNDP project list
        df['treated'] = df['undp_treated'].fillna(0).astype(int)
    elif treatment_type == 'validated':
        # Robustness: UNDP AND E1=Yes
        df['treated'] = ((df['undp_treated'] == 1) & (df['e1_yes'] == 1)).astype(int)
    elif treatment_type in ['solar', 'shelter', 'agri', 'cash', 'health', 'wash', 'training']:
        # Heterogeneity: Specific intervention type
        col = f'has_{treatment_type}'
        if col in df.columns:
            df['treated'] = df[col].fillna(0).astype(int)
        else:
            df['treated'] = 0
    else:
        df['treated'] = df['undp_treated'].fillna(0).astype(int)
    
    # ACLED conflict data
    acled_stats, acled_id_col = process_acled_data(BOUNDARIES_FILE)
    if acled_stats is not None and acled_id_col:
        print("    -> Merging ACLED Conflict Data...")
        df = pd.merge(df, acled_stats, left_on=['township_id', 'period_start'],
                     right_on=[acled_id_col, 'period_start'], how='left')
        df['conflict_events'] = df['conflict_events'].fillna(0)
    
    # Clean & transform
    df['ym_str'] = df['period_start'].dt.strftime('%Y-%m')
    df = df[~df['ym_str'].isin(EXCLUDE_MONTHS)].copy()
    df = calculate_sar_indices(df)
    
    # Create post variable
    print(f"    -> Using UNDP start date: {UNDP_START_DATE}")
    df['post'] = (df['period_start'] >= UNDP_START_DATE).astype(int)
    df['treated_x_post'] = df['treated'] * df['post']
    
    # Logs
    for col in ['nightlights', 'ndvi', 'ndbi', 'rainfall', 'price_rice', 'conflict_events']:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col].fillna(0).clip(lower=0))
    
    return df

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load UNDP treatment
    undp_townships, treatment_df = load_undp_project_data()
    if undp_townships is None or treatment_df is None:
        print("  ! Error: Could not load UNDP treatment data")
        return
    
    # Save treatment definitions
    treatment_df.to_csv(f"{OUTPUT_DIR}/treatment_undp_final.csv", index=False)
    
    # Outcomes to analyze
    outcomes = ['nightlights', 'log_nightlights', 'ndvi', 'rvi', 'ndbi', 'r_ndbi', 
                'structure_proxy_vh', 'flood_proxy_vv', 'log_conflict', 'log_price_rice']
    base_controls = ['rainfall']
    
    # ========================================
    # 1. MAIN ANALYSIS: UNDP Project List (ITT)
    # ========================================
    print("\n" + "="*70)
    print("1. MAIN ANALYSIS: UNDP Project List (Intent-to-Treat)")
    print("="*70)
    
    df_main = load_and_clean_data(undp_townships, treatment_df, treatment_type='main')
    
    subsets_main = {
        'National (All)': df_main,
        'Spatial 25km': get_spatial_subset(df_main, 0.25, 0),
        'Spatial 50km': get_spatial_subset(df_main, 0.50, 0),
        'Donut (15-50km)': get_spatial_subset(df_main, 0.50, 0.15),
    }
    
    table_main = []
    print("\n--- Running Main Regressions ---")
    for outcome in outcomes:
        if outcome not in df_main.columns: continue
        print(f"  > {outcome}")
        for name, df_sub in subsets_main.items():
            table_main.extend(run_regression(df_sub, outcome, name, []))
            table_main.extend(run_regression(df_sub, outcome, name, base_controls))
    
    pd.DataFrame(table_main).to_csv(f"{OUTPUT_DIR}/regression_results_main.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR}/regression_results_main.csv")
    
    # ========================================
    # 2. ROBUSTNESS: Validated Delivery (TOT)
    # ========================================
    if 'e1_yes' in treatment_df.columns and not treatment_df['e1_yes'].isna().all():
        print("\n" + "="*70)
        print("2. ROBUSTNESS CHECK: Validated Delivery (Treatment-on-Treated)")
        print("="*70)
        
        df_validated = load_and_clean_data(undp_townships, treatment_df, treatment_type='validated')
        
        subsets_validated = {
            'National (All)': df_validated,
            'Spatial 25km': get_spatial_subset(df_validated, 0.25, 0),
            'Spatial 50km': get_spatial_subset(df_validated, 0.50, 0),
            'Donut (15-50km)': get_spatial_subset(df_validated, 0.50, 0.15),
        }
        
        table_validated = []
        print("\n--- Running Validated Regressions ---")
        for outcome in outcomes:
            if outcome not in df_validated.columns: continue
            print(f"  > {outcome}")
            for name, df_sub in subsets_validated.items():
                table_validated.extend(run_regression(df_sub, outcome, name, []))
                table_validated.extend(run_regression(df_sub, outcome, name, base_controls))
        
        pd.DataFrame(table_validated).to_csv(f"{OUTPUT_DIR}/regression_results_validated.csv", index=False)
        print(f"\nSaved: {OUTPUT_DIR}/regression_results_validated.csv")
    
    # ========================================
    # 3. HETEROGENEITY: By Intervention Type
    # ========================================
    print("\n" + "="*70)
    print("3. HETEROGENEITY ANALYSIS: By Intervention Type (UNDP townships only)")
    print("="*70)
    
    interventions = ['solar', 'shelter', 'agri', 'cash', 'health', 'wash', 'training']
    table_heterogeneity = []
    
    for intervention in interventions:
        col = f'has_{intervention}'
        if col not in treatment_df.columns:
            continue
        
        # Check if any townships have this intervention
        if (treatment_df[col] == 1).sum() == 0:
            continue
        
        print(f"\n--- {intervention.capitalize()} Intervention ---")
        
        df_intervention = load_and_clean_data(undp_townships, treatment_df, treatment_type=intervention)
        
        # Only analyze Spatial 25km (main model)
        df_25km = get_spatial_subset(df_intervention, 0.25, 0)
        
        for outcome in outcomes:
            if outcome not in df_25km.columns: continue
            results = run_regression(df_25km, outcome, f'Spatial 25km ({intervention.capitalize()})', base_controls)
            if results:
                table_heterogeneity.extend(results)
                print(f"  > {outcome}: {results[0]['Coeff']}")
    
    if table_heterogeneity:
        pd.DataFrame(table_heterogeneity).to_csv(f"{OUTPUT_DIR}/regression_results_heterogeneity.csv", index=False)
        print(f"\nSaved: {OUTPUT_DIR}/regression_results_heterogeneity.csv")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nOutput files:")
    print(f"  - treatment_undp_final.csv (treatment definitions)")
    print(f"  - regression_results_main.csv (primary results)")
    if Path(f"{OUTPUT_DIR}/regression_results_validated.csv").exists():
        print(f"  - regression_results_validated.csv (robustness check)")
    if Path(f"{OUTPUT_DIR}/regression_results_heterogeneity.csv").exists():
        print(f"  - regression_results_heterogeneity.csv (by intervention type)")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
