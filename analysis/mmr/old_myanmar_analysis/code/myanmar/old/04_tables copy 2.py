#!/usr/bin/env python3
"""
04_tables.py
===================
Generates publication-ready CSV tables.
UPDATED: 
- Handling for Rice Price outcome.
- NEW: Generates Interaction/Heterogeneity Tables.
- NEW: Formats and centralizes IPW Balance Table.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import shutil

# --- CONFIGURATION ---
INPUT_REGRESSION = "outputs/myanmar/regression_results_sar_extended.csv" 
INPUT_SENSITIVITY = "outputs/myanmar/robustness_sensitivity.csv"
INPUT_IPW_BALANCE = "outputs/myanmar/table_ipw_balance.csv" # From Script 06
OUTPUT_DIR = "outputs/myanmar/tables_formatted"

SAMPLE_ORDER = ['Spatial 5km', 'Spatial 15km', 'Spatial 25km', 'Spatial 50km', 'Donut (15-50km)', 'National (All)']

def format_wide_table(df, outcome_name, title_label, filter_query=None):
    """ Standard Main Results Table """
    sub = df[df['Outcome'] == outcome_name].copy()
    if sub.empty: return None

    # Filter out interactions for the main table
    if filter_query:
        sub = sub[sub['Sample'].str.contains(filter_query, regex=False)]
    else:
        sub = sub[~sub['Sample'].str.contains("Interaction", case=False, na=False)]

    def get_spec_type(ctl):
        s = str(ctl).lower()
        if 'none' in s or s == '' or pd.isna(ctl): return 'Raw'
        if 'conflict' in s or 'price' in s: return 'Full'
        return 'Rain'

    sub['Control_Type'] = sub['Controls'].apply(get_spec_type)
    sub['SampleSort'] = sub['Sample'].apply(lambda x: SAMPLE_ORDER.index(x) if x in SAMPLE_ORDER else 99)
    sub['ControlSort'] = sub['Control_Type'].map({'Raw':0, 'Rain':1, 'Full':2})
    sub = sub.sort_values(['SampleSort', 'ControlSort'])
    
    sub['Header'] = sub['Sample'] + "\n(" + sub['Control_Type'] + ")"
    
    table_rows = []
    headers = sub['Header'].tolist()
    
    table_rows.append(["Specification"] + headers)
    table_rows.append(["Treated x Post"] + sub['Coeff'].astype(str).tolist())
    table_rows.append([""] + sub['SE'].astype(str).tolist())
    
    rain_row = ["Yes" if 'rainfall' in str(c).lower() else "No" for c in sub['Controls']]
    conf_row = ["Yes" if 'conflict' in str(c).lower() else "No" for c in sub['Controls']]
    price_row = ["Yes" if 'price' in str(c).lower() else "No" for c in sub['Controls']]
    
    table_rows.append(["Rainfall Control"] + rain_row)
    table_rows.append(["Conflict Control"] + conf_row)
    table_rows.append(["Price Control"] + price_row)
    table_rows.append(["Township FE"] + ["Yes"] * len(headers))
    table_rows.append(["Time FE"] + ["Yes"] * len(headers))
    table_rows.append(["Observations"] + [f"{int(x):,}" for x in sub['Obs']])
    table_rows.append(["R-Squared"] + [f"{float(x):.3f}" for x in sub['R2']])
    
    final_df = pd.DataFrame(table_rows)
    title_row = pd.DataFrame([[title_label.upper()] + [""] * len(headers)])
    final_df = pd.concat([title_row, final_df], ignore_index=True)
    return final_df

def format_interaction_table(df):
    """ Creates a table specifically for Heterogeneity (Interactions) """
    sub = df[df['Sample'].str.contains("Interaction", case=False, na=False)].copy()
    if sub.empty: return
    
    # Pivot so columns are the interaction types
    # We assume 'Outcome' tells us the dependent var
    # And 'Sample' contains the interaction info
    
    # Clean Sample Name to get Interaction Type
    def clean_int_name(s):
        if "Agri" in s: return "Agri Interaction"
        if "Solar" in s: return "Solar Interaction"
        if "Shelter" in s: return "Shelter Interaction"
        return s
    
    sub['Int_Type'] = sub['Sample'].apply(clean_int_name)
    
    # Select best model (Full controls)
    sub = sub[sub['Controls'].str.contains("price|rainfall", regex=True, case=False)]
    
    # Create simple list table
    out_df = sub[['Outcome', 'Int_Type', 'Coeff', 'SE', 'Obs', 'R2']].copy()
    out_df.columns = ['Outcome Variable', 'Interaction Term', 'Coefficient', 'SE', 'Obs', 'R2']
    
    title_row = pd.DataFrame([["TABLE 9: HETEROGENEITY (INTERACTIONS)"] + [""] * 5], columns=out_df.columns)
    final_df = pd.concat([title_row, out_df], ignore_index=True)
    
    final_df.to_csv(f"{OUTPUT_DIR}/table_heterogeneity.csv", index=False, header=True)

def format_sensitivity_table():
    if not Path(INPUT_SENSITIVITY).exists(): return
    df = pd.read_csv(INPUT_SENSITIVITY)
    
    outcomes = list(df['Outcome'].unique())
    df_pivot = df.pivot(index='Specification', columns='Outcome', values='Coeff')
    df_se = df.pivot(index='Specification', columns='Outcome', values='SE')
    
    # Define preferred order
    specs = ["Control: Dist to Epicenter", "Intensity (Project Count)", "Spillover Effects", "Spillover (Neighbor Effect)"]
    
    table_rows = []
    table_rows.append(["Model Specification"] + outcomes)
    
    for spec in specs:
        if spec not in df_pivot.index: continue
        coefs = [df_pivot.loc[spec, out] if out in df_pivot.columns and pd.notna(df_pivot.loc[spec, out]) else "-" for out in outcomes]
        ses = [df_se.loc[spec, out] if out in df_se.columns and pd.notna(df_se.loc[spec, out]) else "-" for out in outcomes]
        
        table_rows.append([spec] + coefs)
        table_rows.append([""] + ses)
        
    final_df = pd.DataFrame(table_rows)
    title_row = pd.DataFrame([["TABLE A1: SENSITIVITY & SPILLOVERS"] + [""] * len(outcomes)])
    final_df = pd.concat([title_row, final_df], ignore_index=True)
    final_df.to_csv(f"{OUTPUT_DIR}/table_sensitivity.csv", index=False, header=False)

def process_ipw_table():
    if not Path(INPUT_IPW_BALANCE).exists(): return
    
    df = pd.read_csv(INPUT_IPW_BALANCE)
    # Just add a nice title header and save to formatted folder
    title_row = pd.DataFrame([["TABLE A2: IPW BALANCE CHECK"] + [""] * (len(df.columns)-1)], columns=df.columns)
    final_df = pd.concat([title_row, df], ignore_index=True)
    
    final_df.to_csv(f"{OUTPUT_DIR}/table_ipw_balance_formatted.csv", index=False)

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    if Path(INPUT_REGRESSION).exists():
        df = pd.read_csv(INPUT_REGRESSION)
        
        # 1. Main Tables
        tasks = [
            ('nightlights', 'Table 1: Nightlights'),
            ('log_nightlights', 'Table 2: Log Nightlights'),
            ('ndvi', 'Table 3: Vegetation (NDVI)'),
            ('rvi', 'Table 4: Radar Vegetation (RVI)'),
            ('ndbi', 'Table 5: Built-Up (NDBI)'),
            ('structure_proxy_vh', 'Table 6: Structure VH'),
            ('log_conflict', 'Table 7: Conflict'),
            ('log_price_rice', 'Table 8: Prices')
        ]
        
        print("Generating Main Tables...")
        for var, title in tasks:
            t = format_wide_table(df, var, title) 
            if t is not None: 
                t.to_csv(f"{OUTPUT_DIR}/table_{var}.csv", index=False, header=False)
        
        # 2. Interaction Table
        print("Generating Interaction Table...")
        format_interaction_table(df)

    # 3. Robustness Tables
    print("Generating Robustness Tables...")
    format_sensitivity_table()
    process_ipw_table()
    
    print(f"\nAll Tables saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
