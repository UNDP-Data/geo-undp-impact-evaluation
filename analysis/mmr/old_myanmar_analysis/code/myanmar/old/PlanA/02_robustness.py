#!/usr/bin/env python3
"""
02_robustness.py
================
Robustness: Advanced Sensitivity Analysis
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
EXTERNAL_DIR = "data/myanmar/external"
OUTPUT_DIR = "outputs/myanmar/robustness"
BOUNDARIES_FILE = "data/myanmar/myanmar_boundaries/mmr_admin3.geojson"
PROJECT_FILE = "data/myanmar/undp/project_data.xlsx"
LOCAL_NL_FILE = "outputs/myanmar/local_nightlights_panel.csv"
EARTHQUAKE_DATE = pd.Timestamp("2025-03-28")
UNDP_START_DATE = pd.Timestamp("2025-05-01")
EPICENTER = (22.01, 95.92)

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
        panel = df.groupby(['township_clean', 'period_start'])['price'].mean().reset_index().rename(columns={'price': 'price_rice', 'township_clean': 'name_clean'})
        return panel
        
    except Exception as e:
        print(f"  ! WFP Error: {e}")
        return None

# --- SPATIAL & LOADING ---
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((phi2-phi1)/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(np.radians(lon2-lon1)/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))

def add_spatial_features(df):
    gdf = gpd.read_file(BOUNDARIES_FILE)
    id_col = next((c for c in gdf.columns if 'pcode' in c.lower()), None)
    
    gdf_metric = gdf.to_crs(epsg=32647)
    gdf_metric['centroid'] = gdf_metric.geometry.centroid
    centroids_geo = gdf_metric['centroid'].to_crs(epsg=4326)
    gdf['lat'] = centroids_geo.y
    gdf['lon'] = centroids_geo.x
    
    gdf[id_col] = gdf[id_col].astype(str)
    df['township_id'] = df['township_id'].astype(str)

    coords = gdf[[id_col, 'lat', 'lon']].rename(columns={id_col: 'township_id'})
    df = pd.merge(df, coords, on='township_id', how='left')
    df['dist_epicenter'] = calculate_distance(df['lat'], df['lon'], EPICENTER[0], EPICENTER[1])
    
    treated_ids = df[df['treated']==1]['township_id'].unique()
    treated_geoms = gdf[gdf[id_col].isin(treated_ids)]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spillover_zone = treated_geoms.geometry.buffer(0.14).unary_union
    
    potential = gdf[gdf.geometry.intersects(spillover_zone)]
    spillover_ids = set(potential[id_col]) - set(treated_ids)
    df['is_spillover'] = df['township_id'].apply(lambda x: 1 if x in spillover_ids else 0)
    return df

def load_and_clean_data():
    print("Loading Data for Robustness...")
    
    # 1. Load Panel Files
    files = glob.glob(f"{DATA_DIR}/gee_panel_myanmar_PART*.csv")
    if not files:
        return pd.DataFrame()

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    
    # Rename GEE Columns
    rename_map = {
        'admin3_id': 'township_id',
        'admin3_nam': 'township_name', 
        'NDVI': 'ndvi',
        'NDBI': 'ndbi',
        'RAIN_TOTAL': 'rainfall', 
        'NL_AVG_RAD': 'nightlights_gee',
        'VH': 'vh',
        'VV': 'vv'
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df['period_start'] = pd.to_datetime(df['period_start'])
    
    # Load treatment
    treatment_df = load_clean_project_data()
    df['name_clean'] = df['township_name'].apply(clean_name)
    treatment_map = dict(zip(treatment_df['clean_name'], treatment_df['undp_treated']))
    df['treated'] = df['name_clean'].map(treatment_map).fillna(0).astype(int)
    
    if 'treatment_intensity' in treatment_df.columns:
        intensity_map = dict(zip(treatment_df['clean_name'], treatment_df['treatment_intensity']))
        df['project_count'] = df['name_clean'].map(intensity_map).fillna(0)
    else:
        df['project_count'] = df['treated']
    
    # Local NL Merge
    if Path(LOCAL_NL_FILE).exists():
        print("    -> Merging Local Nightlights...")
        df_local = pd.read_csv(LOCAL_NL_FILE)
        df_local['period_start'] = pd.to_datetime(df_local['period_start'])
        df_local['name_clean'] = df_local['township_name'].apply(clean_name)
        
        df = pd.merge(df, df_local[['name_clean', 'period_start', 'nightlights']], 
                      on=['name_clean', 'period_start'], 
                      how='outer',
                      suffixes=('_gee', ''))
        
        if 'nightlights_gee' in df.columns:
            df['nightlights'] = df['nightlights'].combine_first(df['nightlights_gee'])
        
        df['township_name'] = df['township_name'].fillna(df['name_clean'].str.title())
        df = df.sort_values(['name_clean', 'period_start'])
    else:
        df['nightlights'] = df.get('nightlights_gee', 0)
    
    # Merge WFP and ACLED
    if Path(BOUNDARIES_FILE).exists():
        gdf_bounds = gpd.read_file(BOUNDARIES_FILE)
        rename_geo = {'adm3_name': 'township_name', 'adm1_name': 'state_name'}
        gdf_bounds = gdf_bounds.rename(columns=rename_geo)
        
        if 'township_name' not in gdf_bounds.columns:
            t_col = next((c for c in gdf_bounds.columns if c.lower() in ['admin3_nam', 'ts_name']), None)
            if t_col:
                gdf_bounds = gdf_bounds.rename(columns={t_col: 'township_name'})
        
        if 'state_name' not in gdf_bounds.columns:
            s_col = next((c for c in gdf_bounds.columns if c.lower() in ['admin1_nam', 'st_name']), None)
            if s_col:
                gdf_bounds = gdf_bounds.rename(columns={s_col: 'state_name'})

        wfp_df = load_and_process_wfp(f"{EXTERNAL_DIR}/wfp_prices.csv", gdf_bounds)
        if wfp_df is not None and not wfp_df.empty:
            df = pd.merge(df, wfp_df, on=['name_clean', 'period_start'], how='left')
            df['price_rice'] = df.groupby('township_id')['price_rice'].transform(
                lambda x: x.interpolate(method='linear').ffill().bfill()
            )
            df['log_price_rice'] = np.log1p(df['price_rice'])
    
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
    
    df = add_spatial_features(df)
    
    df['ym_str'] = df['period_start'].dt.strftime('%Y-%m')
    EXCLUDE_MONTHS = ['2025-03']
    df = df[~df['ym_str'].isin(EXCLUDE_MONTHS)].copy()
    df['post'] = (df['period_start'] >= UNDP_START_DATE).astype(int)
    
    df['treated_x_post'] = df['treated'] * df['post']
    df['intensity_x_post'] = df['project_count'] * df['post']
    df['spillover_x_post'] = df['is_spillover'] * df['post']
    df['dist_x_post'] = df['dist_epicenter'] * df['post']
    
    df = calculate_sar_indices(df)
    for col in ['nightlights', 'ndvi', 'ndbi', 'rainfall', 'rvi',
                'structure_proxy_vh', 'flood_proxy_vv', 'r_ndbi']:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col].clip(lower=0))
    
    return df



def run_robustness_regression(df, outcome, spec_name, formula_str):
    if outcome not in df.columns: return None
    
    # Strict Drop: Only run if we have data for Outcome AND Rain
    model_data = df.dropna(subset=[outcome, 'rainfall']).set_index(['township_id', 'period_start'])
    
    if model_data.empty: 
        # DEBUG PRINT
        # print(f"  ! Skipped {outcome} ({spec_name}): No data after dropna.")
        return None

    try:
        mod = PanelOLS.from_formula(f"{outcome} ~ {formula_str} + rainfall + EntityEffects + TimeEffects", data=model_data, check_rank=False)
        res = mod.fit(cov_type='clustered', cluster_entity=True)
        main_var = formula_str.split(" + ")[0]
        
        return {
            'Outcome': outcome, 'Specification': spec_name,
            'Coeff': f"{res.params[main_var]:.4f}{'***' if res.pvalues[main_var]<0.01 else '**' if res.pvalues[main_var]<0.05 else '*' if res.pvalues[main_var]<0.1 else ''}",
            'SE': f"({res.std_errors[main_var]:.4f})", 'Obs': int(res.nobs), 'R2': f"{res.rsquared:.3f}"
        }
    except: return None

def run_event_study(df, outcome):
    """ Runs Dynamic TWFE: outcome ~ treated x leads + treated x lags """
    if outcome not in df.columns: return pd.DataFrame()
    
    df_mod = df.dropna(subset=[outcome]).copy()
    # Months relative to Earthquake
    df_mod['months_since_eq'] = ((df_mod['period_start'].dt.year - EARTHQUAKE_DATE.year) * 12 + 
                                 (df_mod['period_start'].dt.month - EARTHQUAKE_DATE.month))
    
    # Window: -6 to +6
    WINDOW = 6
    df_mod = df_mod[(df_mod['months_since_eq'] >= -WINDOW) & (df_mod['months_since_eq'] <= WINDOW)]
    
    formula_parts = []
    periods = [t for t in range(-WINDOW, WINDOW+1) if t != -1] # Drop -1 as reference
    
    for t in periods:
        # Create Dummy: Treated * (Month == t)
        col = f"rel_{t}".replace("-", "m")
        df_mod[col] = (df_mod['treated'] * (df_mod['months_since_eq'] == t)).astype(int)
        formula_parts.append(col)
        
    formula = f"{outcome} ~ " + " + ".join(formula_parts) + " + EntityEffects + TimeEffects"
    model_data = df_mod.set_index(['township_id', 'period_start'])
    
    try:
        mod = PanelOLS.from_formula(formula, data=model_data, drop_absorbed=True)
        res = mod.fit(cov_type='clustered', cluster_entity=True)
        
        coefs = []
        for t in periods:
            c_name = f"rel_{t}".replace("-", "m")
            if c_name in res.params:
                coefs.append({'outcome': outcome, 't': t, 'coef': res.params[c_name], 'err': res.std_errors[c_name]})
        # Add reference period manually
        coefs.append({'outcome': outcome, 't': -1, 'coef': 0, 'err': 0})
        return pd.DataFrame(coefs)
    except: 
        return pd.DataFrame()

def get_spatial_subset(df, radius_deg, treatment_col='treated'):
    """Filters data to Treated units + Control units within radius_deg."""
    if not Path(BOUNDARIES_FILE).exists(): return df
    
    gdf = gpd.read_file(BOUNDARIES_FILE)
    id_col = next((c for c in gdf.columns if 'pcode' in c.lower()), None)
    
    # 1. Identify Treated Geometries
    treated_ids = df[df[treatment_col] == 1]['township_id'].unique()
    treated_geoms = gdf[gdf[id_col].isin(treated_ids)]
    
    if treated_geoms.empty: return df
    
    # 2. Create Buffer (25km approx 0.25 deg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        treated_buffer = treated_geoms.geometry.buffer(radius_deg).unary_union
    
    # 3. Find Valid Controls (Intersect buffer but NOT treated)
    # This excludes far-away regions that introduce noise
    potential_controls = gdf[gdf.geometry.intersects(treated_buffer)]
    control_ids = potential_controls[~potential_controls[id_col].isin(treated_ids)][id_col].unique()
    
    # 4. Filter DataFrame
    valid_ids = np.concatenate([treated_ids, control_ids])
    return df[df['township_id'].isin(valid_ids)].copy()

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    df = load_and_clean_data()
    
    # --- UPDATE: Enforce 25km Spatial Control Group ---
    # This aligns robustness checks with the best-performing main model.
    print(f"  > Filtering dataset to 25km Spatial Radius...")
    df = get_spatial_subset(df, 0.25) 
    print(f"    -> Sample reduced to {df['township_id'].nunique()} townships (Treated + 25km Neighbors).")
    
    # 2. Outcomes for robustness + event study
    outcomes = [
        'nightlights',
        'log_nightlights',
        'ndvi',
        'rvi',
        'ndbi',
        'r_ndbi',             
        'rainfall',           
        'structure_proxy_vh',
        'flood_proxy_vv'
    ]

    # Optional: include rice outcomes if present
    if 'price_rice' in df.columns:
        outcomes.append('price_rice')          # for visuals ES plots
    if 'log_price_rice' in df.columns:
        outcomes.append('log_price_rice')      # if you want log version in robustness tables
    
    # 3. Run Sensitivity Analysis
    results = []
    print("\n--- RUNNING SENSITIVITY CHECKS ---")
    for out in outcomes:
        print(f"  > Sensitivity: {out}")
        res = run_robustness_regression(df, out, "Control: Dist to Epicenter", "treated_x_post + dist_x_post")
        if res:
            results.append(res)
        res = run_robustness_regression(df, out, "Intensity (Project Count)", "intensity_x_post")
        if res:
            results.append(res)
        res_spill = run_robustness_regression(df, out, "Spillover Effects", "treated_x_post + spillover_x_post")
        if res_spill:
            results.append(res_spill)

    if results:
        pd.DataFrame(results).to_csv(f"{OUTPUT_DIR}/robustness_sensitivity.csv", index=False)
        print("Sensitivity Results Saved.")

    # 4. Run Event Studies
    print("\n--- RUNNING EVENT STUDIES ---")
    for out in outcomes:
        print(f"  > Event Study: {out}")
        es_df = run_event_study(df, out)

    if es_df.empty:
        print(f"    ! No dynamic coefficients estimated for {out} (no file saved).")
    else:
        out_path = Path(OUTPUT_DIR) / f"es_coeffs_{out}.csv"
        es_df.to_csv(out_path, index=False)
        print(f"    -> Saved ES file: {out_path}")

    print("\nRobustness Complete.")

if __name__ == "__main__":
    main()
