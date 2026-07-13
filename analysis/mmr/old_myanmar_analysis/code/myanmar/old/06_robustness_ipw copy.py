#!/usr/bin/env python3
"""
06_robustness_ipw.py
====================
Robustness: Inverse Probability Weighted DiD (IPW-DiD)
Methodology: Abadie (2005) / Doubly Robust Estimation.
1. Collapse data to cross-section to calculate Propensity Scores (Prob of Treatment).
2. Generate Weights (ATT): Treated = 1, Control = P / (1-P).
3. Generate Balance Table (Unweighted vs Weighted).
4. Run Weighted TWFE Regression.
"""

import pandas as pd
import numpy as np
import geopandas as gpd  # <--- Was missing
from linearmodels.panel import PanelOLS
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import glob
import os
from difflib import get_close_matches # <--- Was missing (needed for WFP matching)
import warnings

# --- CONFIGURATION ---
DATA_DIR = "data/myanmar"
EXTERNAL_DIR = "data/myanmar/external"  # <--- Added
OUTPUT_DIR = "outputs/myanmar/robustness"
BOUNDARIES_FILE = "data/myanmar/myanmar_boundaries/mmr_admin3.geojson"  # <--- Added
PROJECT_FILE = "data/myanmar/undp/project_data.xlsx"
LOCAL_NL_FILE = "outputs/myanmar/local_nightlights_panel.csv"
EARTHQUAKE_DATE = pd.Timestamp("2025-03-28")
UNDP_START_DATE = pd.Timestamp("2025-05-01")
EXCLUDE_MONTHS = ['2025-03']
SPATIAL_RADIUS_DEG = 0.25  # 25km Radius

def apply_treatment_to_panel(df, treatment_df):
    """Applies treatment using clean name matching."""
    if 'clean_name' not in df.columns:
        df['clean_name'] = df['township_name'].apply(clean_name)
        df['name_clean'] = df['clean_name'] 
    
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

def get_treated_townships():
    if not Path(PROJECT_FILE).exists(): return set()
    df = pd.read_excel(PROJECT_FILE)
    col = next((c for c in df.columns if "Township" in c), None)
    return set(df[col].apply(clean_name).unique())

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
    print("Loading Data for IPW...")
    files = glob.glob(f"{DATA_DIR}/gee_panel_myanmar_PART*.csv")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    
    rename_map = {'admin3_id': 'township_id', 'admin3_nam': 'township_name', 
                  'NDVI': 'ndvi', 'NDBI': 'ndbi', 'RAIN_TOTAL': 'rainfall', 
                  'NL_AVG_RAD': 'nightlights_gee', 'VH': 'vh', 'VV': 'vv'}
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})
    df['period_start'] = pd.to_datetime(df['period_start'])
    
    # --- EVERYTHING BELOW WAS UNINDENTED. FIXED INDENTATION: ---
    
    # Load official treatment definitions
    treatment_df = load_clean_project_data()

    # Create clean names for matching
    if 'clean_name' not in df.columns:
        df['clean_name'] = df['township_name'].apply(clean_name)

    df['name_clean'] = df['clean_name']

    # Map treatment (binary 0/1)
    treatment_map = dict(zip(treatment_df['clean_name'], treatment_df['undp_treated']))
    df['treated'] = df['clean_name'].map(treatment_map).fillna(0).astype(int)
    
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
                      suffixes=('_gee', ''))  # CORRECT ORDER
        
        # Combine: Prioritize local data
        if 'nightlights_gee' in df.columns:
            df['nightlights'] = df['nightlights'].combine_first(df['nightlights_gee'])
            
        # Backfill township_name for new rows
        df['township_name'] = df['township_name'].fillna(df['name_clean'].str.title())
    else:
        df['nightlights'] = df.get('nightlights_gee', df.get('nightlights'))

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

    # WFP Merge (Robust)
    if Path(BOUNDARIES_FILE).exists():
        gdf_bounds = gpd.read_file(BOUNDARIES_FILE)
        # Fix Columns
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

    df = df[~df['period_start'].dt.strftime('%Y-%m').isin(EXCLUDE_MONTHS)]
    df['post'] = (df['period_start'] >= UNDP_START_DATE).astype(int)
    
    # Clip and log
    for col in ['nightlights', 'ndvi', 'ndbi', 'rainfall', 'price_rice']:
        if col in df.columns:
            df[col] = df[col].fillna(0) 
            if col == 'nightlights':
                df['log_nightlights'] = np.log1p(df[col].clip(lower=0))
            if col == 'price_rice':
                df['log_price_rice'] = np.log1p(df[col])
    
    df['treated_x_post'] = df['treated'] * df['post']

    if 'vh' in df.columns and 'vv' in df.columns:
        vh_lin = 10 ** (df['vh'].clip(lower=-30) / 10.0)
        vv_lin = 10 ** (df['vv'].clip(lower=-30) / 10.0)
        df['rvi'] = (4 * vh_lin) / (vv_lin + vh_lin)
        df['ndbi'] = df.get('ndbi', 0)
    
    return df

def calculate_ipw_weights(df):
    """ Calculates Weights based on Pre-Earthquake Characteristics. """
    print("Calculating Propensity Scores...")
    
    # 1. Collapse to Cross-Section (Baseline Averages)
    # CRITICAL: We DO NOT include price here to avoid dropping 85% of the sample.
    pre_data = df[df['post'] == 0].groupby('township_id').agg({
        'treated': 'max',
        'log_nightlights': 'mean',
        'rainfall': 'mean',
        'ndvi': 'mean',
        'ndbi': 'mean'
    }).dropna()
    
    # 2. Logit Model
    vars_to_use = ['log_nightlights', 'rainfall', 'ndvi', 'ndbi'] 
    
    X = sm.add_constant(pre_data[vars_to_use])
    y = pre_data['treated']
    
    try:
        logit = sm.Logit(y, X).fit(disp=0)
        pre_data['pscore'] = logit.predict(X)
        
        # 3. Common Support Trimming
        pre_data = pre_data[(pre_data['pscore'] > 0.01) & (pre_data['pscore'] < 0.99)]
        
        # 4. ATT Weights
        pre_data['ipw'] = np.where(
            pre_data['treated'] == 1,
            1.0,
            pre_data['pscore'] / (1.0 - pre_data['pscore'])
        )
        return pre_data
    except Exception as e:
        print(f"  ! Error in IPW Calculation: {e}")
        return pd.DataFrame()

def generate_balance_table(pre_data):
    if pre_data.empty: return pd.DataFrame()
    print("Generating Balance Table...")
    
    covariates = ['log_nightlights', 'rainfall', 'ndvi', 'ndbi'] 
    rows = []
    
    for var in covariates:
        if var not in pre_data.columns: continue

        # --- Unweighted ---
        t_mean = pre_data[pre_data['treated']==1][var].mean()
        c_mean = pre_data[pre_data['treated']==0][var].mean()
        pool_sd = np.sqrt((pre_data[pre_data['treated']==1][var].var() + pre_data[pre_data['treated']==0][var].var()) / 2)
        smd_raw = (t_mean - c_mean) / pool_sd if pool_sd > 0 else 0
        
        # --- Weighted ---
        t_w = pre_data[pre_data['treated']==1]['ipw']
        c_w = pre_data[pre_data['treated']==0]['ipw']
        
        t_mean_w = np.average(pre_data[pre_data['treated']==1][var], weights=t_w)
        c_mean_w = np.average(pre_data[pre_data['treated']==0][var], weights=c_w)
        
        smd_w = (t_mean_w - c_mean_w) / pool_sd if pool_sd > 0 else 0
        
        rows.append({
            'Variable': var,
            'Treated Mean': f"{t_mean:.3f}",
            'Control (Raw)': f"{c_mean:.3f}",
            'SMD (Raw)': f"{smd_raw:.3f}",
            'Control (Weighted)': f"{c_mean_w:.3f}",
            'SMD (Weighted)': f"{smd_w:.3f}",
            'Status': "Balanced" if abs(smd_w) < 0.2 else "Imbalanced"
        })
        
    bal_df = pd.DataFrame(rows)
    bal_df.to_csv(f"{OUTPUT_DIR}/table_ipw_balance.csv", index=False)
    print("  > Balance table saved.")
    return bal_df

def run_ipw_regression(df, weights_df, outcome):
    # Merge weights (inner join keeps only trimmed common support)
    model_data = df.merge(weights_df[['ipw']], on='township_id', how='inner')
    model_data = model_data.set_index(['township_id', 'period_start'])
    
    print(f"\nRunning IPW-DiD for {outcome}...")
    
    mod = PanelOLS.from_formula(
        f"{outcome} ~ treated_x_post + EntityEffects + TimeEffects",
        data=model_data,
        weights=model_data['ipw']
    )
    res = mod.fit(cov_type='clustered', cluster_entity=True)
    
    with open(f"{OUTPUT_DIR}/ipw_did_results.txt", "a") as f:
        f.write(f"\n\n--- {outcome.upper()} (IPW-DiD) ---\n")
        f.write(f"Observations: {res.nobs}\n")
        f.write(f"R-Squared: {res.rsquared:.4f}\n")
        f.write(f"{res.summary.tables[1]}\n")

def get_spatial_subset(df, radius_deg, treatment_col='treated'):
    """ Filters to Treated units + Control units within radius_deg """
    if not Path(BOUNDARIES_FILE).exists(): return df
    
    gdf = gpd.read_file(BOUNDARIES_FILE)
    id_col = next((c for c in gdf.columns if 'pcode' in c.lower()), None)
    
    # 1. Identify Treated Geometries
    treated_ids = df[df[treatment_col] == 1]['township_id'].unique()
    treated_geoms = gdf[gdf[id_col].isin(treated_ids)]
    
    if treated_geoms.empty: return df
    
    # 2. Create Buffer
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Union of all treated buffers
        treated_buffer = treated_geoms.geometry.buffer(radius_deg).unary_union
    
    # 3. Find Valid Neighbors (Intersect buffer but NOT treated)
    potential_controls = gdf[gdf.geometry.intersects(treated_buffer)]
    control_ids = potential_controls[~potential_controls[id_col].isin(treated_ids)][id_col].unique()
    
    # 4. Filter Data
    valid_ids = np.concatenate([treated_ids, control_ids])
    return df[df['township_id'].isin(valid_ids)].copy()

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    df = load_data()
    
    # --- Apply 25km Filter ---
    print(f"  > Filtering to Spatial Control Group ({int(SPATIAL_RADIUS_DEG*100)}km approx)...")
    df = get_spatial_subset(df, SPATIAL_RADIUS_DEG)
    print(f"    -> Sample reduced to {df['township_id'].nunique()} townships.")
    # ------------------------------
    
    # 2. Weights & Balance (Cross-Sectional)
    pre_data_w_weights = calculate_ipw_weights(df)
    generate_balance_table(pre_data_w_weights)
    
    # 3. Regressions
    if Path(f"{OUTPUT_DIR}/ipw_did_results.txt").exists():
        os.remove(f"{OUTPUT_DIR}/ipw_did_results.txt")
        
    for outcome in ['log_nightlights', 'nightlights', 'rvi', 'ndbi']:
        try:
            run_ipw_regression(df, pre_data_w_weights, outcome)
        except Exception as e:
            print(f"Failed {outcome}: {e}")

    print(f"\nRobustness Complete. Results saved to {OUTPUT_DIR}/ipw_did_results.txt")
    
if __name__ == "__main__":
    main()
