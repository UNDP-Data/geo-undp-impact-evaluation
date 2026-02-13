#!/usr/bin/env python3
"""
04_tables.py
===================
Generates publication-ready CSV tables.
UPDATED: Handles Interaction Terms and Deduplication.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob

# --- CONFIGURATION ---
INPUT_REGRESSION = "outputs/myanmar/regression_results_sar_extended.csv" 
INPUT_ES_DIR = "outputs/myanmar/robustness"
OUTPUT_DIR = "outputs/myanmar/tables_formatted"

SAMPLE_ORDER = [
    'Spatial 5km', 
    'Spatial 15km', 
    'Spatial 25km', 
    'Spatial 50km', 
    'Donut (15-50km)'
]

def format_wide_table(df, outcome_name, title_label, filter_query=None, coeff_label="Treated × Post"):
    """
    Generates a wide-format table.
    - df: Full results dataframe
    - outcome_name: 'nightlights', 'ndvi', etc.
    - filter_query: String to filter Sample names (e.g., 'Agri Interaction'). 
                    If None, it EXCLUDES interaction rows (Standard Table).
    """
    # 1. Filter for Outcome
    sub = df[df['Outcome'] == outcome_name].copy()
    if sub.empty: return None

    # 2. Handle Interactions vs Standard
    if filter_query:
        # Keep ONLY rows matching the query (e.g., Interaction Terms)
        sub = sub[sub['Sample'].str.contains(filter_query, regex=False)]
        # Clean up sample names for display (remove the tag)
        sub['Sample'] = sub['Sample'].str.replace(filter_query, "", regex=False).str.strip()
    else:
        # Standard Table: EXCLUDE interaction rows
        sub = sub[~sub['Sample'].str.contains("Interaction", case=False, na=False)]
        # CRITICAL: Drop duplicates. 
        # The interaction model produces a base row "Spatial 50km" identical to the standard model.
        # We keep='first' to retain the Standard Model's ATE result.
        sub = sub.drop_duplicates(subset=['Sample', 'Controls'], keep='first')

    # 3. Sorting
    sub['SampleSort'] = sub['Sample'].apply(lambda x: SAMPLE_ORDER.index(x) if x in SAMPLE_ORDER else 99)
    sub['ControlSort'] = sub['Controls'].apply(lambda x: 0 if 'None' in str(x) or pd.isna(x) else 1)
    sub = sub.sort_values(['SampleSort', 'ControlSort'])
    
    if sub.empty: return None

    # 4. Construct Table
    table_rows = []
    headers = sub['Sample'].tolist()
    
    table_rows.append(["Specification"] + headers)
    table_rows.append([coeff_label] + sub['Coeff'].astype(str).tolist())
    table_rows.append([""] + sub['SE'].astype(str).tolist())
    
    rain_flags = ["Yes" if 'rainfall' in str(c).lower() else "No" for c in sub['Controls']]
    table_rows.append(["Rainfall Control"] + rain_flags)
    table_rows.append(["Township FE"] + ["Yes"] * len(headers))
    table_rows.append(["Time FE"] + ["Yes"] * len(headers))
    
    table_rows.append(["Observations"] + [f"{int(x):,}" for x in sub['Obs']])
    table_rows.append(["R-Squared"] + [f"{float(x):.3f}" for x in sub['R2']])
    
    table_rows.append([""] * (len(headers) + 1))
    note = "Note: * p<0.1, ** p<0.05, *** p<0.01. Clustered SE."
    table_rows.append([note] + [""] * len(headers))

    final_df = pd.DataFrame(table_rows)
    title_row = pd.DataFrame([[title_label.upper()] + [""] * len(headers)])
    final_df = pd.concat([title_row, final_df], ignore_index=True)
    
    return final_df

def format_event_study_table():
    print("  > Generating Event Study Summary Table...")
    es_files = glob.glob(f"{INPUT_ES_DIR}/es_coeffs_*.csv")
    if not es_files: return

    summary_rows = []
    target_vars = {
        'log_nightlights': 'Log Nightlights (Main)',
        'rvi': 'Radar Vegetation (RVI)',
        'rvi_agri': 'Agri Projects (RVI)', 
        'r_ndbi': 'Radar Structure (R-NDBI)',
        'flood_proxy_vv': 'Flood Proxy (VV)'
    }

    for filename in es_files:
        var_code = Path(filename).stem.replace('es_coeffs_', '')
        if var_code not in target_vars: continue
        
        df = pd.read_csv(filename)
        def get_stat(t_val):
            row = df[df['t'] == t_val]
            if row.empty: return "-"
            val = row.iloc[0]['coef']
            err = row.iloc[0]['err']
            z = abs(val / err) if err > 0 else 0
            stars = "***" if z > 2.576 else "**" if z > 1.96 else "*" if z > 1.645 else ""
            return f"{val:.3f}{stars}"

        pre_rows = df[(df['t'] >= -5) & (df['t'] <= -2)]
        pre_sig = any((abs(r['coef'] / r['err']) > 1.96) for _, r in pre_rows.iterrows())
        pre_status = "Fail" if pre_sig else "Pass"
        
        summary_rows.append({
            'Outcome': target_vars[var_code],
            'Pre-Trend': pre_status,
            'Shock (t=3)': get_stat(3),
            'End (t=6)': get_stat(6)
        })
    
    if not summary_rows: return
    sdf = pd.DataFrame(summary_rows)
    
    final_df = pd.concat([
        pd.DataFrame([["TABLE 4: DYNAMIC ROBUSTNESS", "", "", ""]], columns=sdf.columns),
        sdf
    ], ignore_index=True)
    
    out_path = f"{OUTPUT_DIR}/table_event_study_summary.csv"
    final_df.to_csv(out_path, index=False, header=False)

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    if not Path(INPUT_REGRESSION).exists():
        print(f"CRITICAL: Missing {INPUT_REGRESSION}. Run 01_main_analysis.py first.")
        return

    df = pd.read_csv(INPUT_REGRESSION)
    
    # 1. Standard Tables (Filters OUT interactions)
    tasks = [
        ('log_nightlights', 'Table 1: Economic Activity (Lights)'),
        ('nightlights', 'Table 2: Economic Activity (Levels)'),
        ('rvi', 'Table 3: Crop Recovery (Radar Vegetation)'),
        ('r_ndbi', 'Table 4: Reconstruction (Radar Structure)'),
        ('flood_proxy_vv', 'Table 5: Flood Impact (Inverse VV)')
    ]
    
    for var, title in tasks:
        t = format_wide_table(df, var, title, filter_query=None) 
        if t is not None: 
            t.to_csv(f"{OUTPUT_DIR}/table_{var}.csv", index=False, header=False)

    # 2. Agri Interaction Tables (Filters FOR interactions)
    # This table shows the "Agri Premium" (Agri x Post)
    t_agri = format_wide_table(df, 'rvi', 'Table A1: Heterogeneity - Agri Projects (RVI)', 
                               filter_query='(Agri Interaction)', 
                               coeff_label="Agri × Post")
    if t_agri is not None:
        t_agri.to_csv(f"{OUTPUT_DIR}/table_agri_heterogeneity.csv", index=False, header=False)

    format_event_study_table()
    print(f"\nTables saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
