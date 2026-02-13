#!/usr/bin/env python3
"""
04_tables.py
===================
Generates publication-ready CSV tables with enhanced sensitivity table.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
INPUT_REGRESSION = "outputs/myanmar/regression_results_main.csv"
INPUT_SENSITIVITY = "outputs/myanmar/robustness/robustness_sensitivity.csv"
INPUT_IPW_BALANCE = "outputs/myanmar/table_ipw_balance.csv"
OUTPUT_DIR = "outputs/myanmar/tables_formatted"

# DEFINING THE COLUMNS TO SHOW (In Order)
# 25km is first because it's your new Default/Preferred model.
SAMPLE_ORDER = ['Spatial 25km', 'Spatial 50km', 'Donut (15-50km)']

def format_wide_table(df, outcome_name, title_label):
    """ Standard Main Results Table (Multi-Column) """
    sub = df[df['Outcome'] == outcome_name].copy()
    if sub.empty: return None

    # Filter out interactions for the main table
    sub = sub[~sub['Sample'].str.contains("Interaction", case=False, na=False)]
    
    # Filter to ONLY keep the samples defined in SAMPLE_ORDER
    sub = sub[sub['Sample'].isin(SAMPLE_ORDER)]
    
    if sub.empty: return None

    def get_spec_type(ctl):
        s = str(ctl).lower()
        if 'none' in s or s == '' or pd.isna(ctl): return 'Raw'
        if 'conflict' in s or 'price' in s: return 'Full'
        return 'Rain'

    sub['Control_Type'] = sub['Controls'].apply(get_spec_type)
    
    # Sort by Sample Order (25km -> 50km -> Donut) then by Control Type
    sub['SampleSort'] = sub['Sample'].apply(lambda x: SAMPLE_ORDER.index(x) if x in SAMPLE_ORDER else 99)
    sub['ControlSort'] = sub['Control_Type'].map({'Raw':0, 'Rain':1, 'Full':2})
    
    # Drop duplicates if Base/Full controls are identical
    sub = sub.drop_duplicates(subset=['Sample', 'Control_Type'], keep='last')
    
    sub = sub.sort_values(['SampleSort', 'ControlSort'])
    
    # Create Header Label
    sub['Header'] = sub['Sample'] + "\n(" + sub['Control_Type'] + ")"
    
    # Build Table
    table_rows = []
    headers = sub['Header'].tolist()
    
    table_rows.append(["Specification"] + headers)
    table_rows.append(["Treated x Post"] + sub['Coeff'].astype(str).tolist())
    table_rows.append([""] + sub['SE'].astype(str).tolist())
    
    rain_row = ["Yes" if 'rainfall' in str(c).lower() else "No" for c in sub['Controls']]
    conf_row = ["Yes" if 'conflict' in str(c).lower() else "No" for c in sub['Controls']]
    
    table_rows.append(["Rainfall Control"] + rain_row)
    table_rows.append(["Conflict Control"] + conf_row)
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
    # Filter for Interactions
    sub = df[df['Sample'].str.contains("Interaction", case=False, na=False)].copy()
    
    # Keep Interactions relevant to our Main Model (25km)
    target_sample = "Spatial 25km"
    sub = sub[sub['Sample'].str.contains(target_sample, case=False, na=False)]
    
    if sub.empty: return
    
    def clean_int_name(s):
        if "Agri" in s: return "Agri Interaction"
        if "Solar" in s: return "Solar Interaction"
        if "Shelter" in s: return "Shelter Interaction"
        return s
    
    sub['Int_Type'] = sub['Sample'].apply(clean_int_name)
    
    # Select best model (with rainfall controls)
    sub = sub[sub['Controls'].str.contains("rainfall", regex=True, case=False)]
    
    out_df = sub[['Outcome', 'Int_Type', 'Coeff', 'SE', 'Obs', 'R2']].copy()
    out_df.columns = ['Outcome Variable', 'Interaction Term', 'Coefficient', 'SE', 'Obs', 'R2']
    
    title_row = pd.DataFrame([[f"TABLE 9: HETEROGENEITY ({target_sample.upper()})"] + [""] * 5], columns=out_df.columns)
    final_df = pd.concat([title_row, out_df], ignore_index=True)
    
    final_df.to_csv(f"{OUTPUT_DIR}/table_heterogeneity.csv", index=False, header=True)
    print("    âœ“ Saved: table_heterogeneity.csv")

def format_sensitivity_table_enhanced(df_main, df_robustness):
    """
    Enhanced sensitivity table that includes:
    1. Regular DiD (TWFE) coefficients from main results
    2. Robustness checks
    3. Statistical significance indicators
    4. N obs and RÂ²
    """
    print("  > Processing: Enhanced Sensitivity Analysis")
    
    # Get main DiD results (Spatial 25km with rainfall controls as baseline)
    baseline = df_main[
        (df_main['Sample'] == 'Spatial 25km') & 
        (df_main['Controls'].str.contains('rainfall', case=False, na=False)) &
        (~df_main['Sample'].str.contains('Interaction', case=False, na=False))
    ].copy()
    
    if baseline.empty:
        print("    ! No baseline results found for Spatial 25km")
        return
    
    # Get outcomes from robustness
    outcomes = list(df_robustness['Outcome'].unique())
    
    # Create main results row
    main_coefs = []
    main_ses = []
    main_obs = []
    main_r2 = []
    
    for outcome in outcomes:
        row = baseline[baseline['Outcome'] == outcome]
        if not row.empty:
            main_coefs.append(row['Coeff'].iloc[0])
            main_ses.append(row['SE'].iloc[0])
            main_obs.append(int(row['Obs'].iloc[0]))
            main_r2.append(f"{float(row['R2'].iloc[0]):.3f}")
        else:
            main_coefs.append("-")
            main_ses.append("-")
            main_obs.append("-")
            main_r2.append("-")
    
    # Get robustness results
    df_pivot = df_robustness.pivot(index='Specification', columns='Outcome', values='Coeff')
    df_se = df_robustness.pivot(index='Specification', columns='Outcome', values='SE')
    df_obs = df_robustness.pivot(index='Specification', columns='Outcome', values='Obs')
    df_r2 = df_robustness.pivot(index='Specification', columns='Outcome', values='R2')
    
    specs = [
        "Control: Dist to Epicenter", 
        "Intensity (Project Count)", 
        "Spillover Effects"
    ]
    
    # Build table
    table_rows = []
    table_rows.append(["Model Specification"] + outcomes)
    
    # Add main DiD results first
    table_rows.append(["Baseline DiD (TWFE)"] + [str(c) for c in main_coefs])
    table_rows.append([""] + [str(s) for s in main_ses])
    table_rows.append(["  Observations"] + [str(o) for o in main_obs])
    table_rows.append(["  R-squared"] + main_r2)
    
    # Add separator
    table_rows.append([""] * (len(outcomes) + 1))
    
    # Add robustness checks
    for spec in specs:
        if spec not in df_pivot.index: 
            continue
            
        coefs = []
        ses = []
        obs_list = []
        r2_list = []
        
        for out in outcomes:
            if out in df_pivot.columns and pd.notna(df_pivot.loc[spec, out]):
                coefs.append(df_pivot.loc[spec, out])
                ses.append(df_se.loc[spec, out])
                obs_list.append(f"{int(df_obs.loc[spec, out]):,}" if pd.notna(df_obs.loc[spec, out]) else "-")
                r2_list.append(f"{float(df_r2.loc[spec, out]):.3f}" if pd.notna(df_r2.loc[spec, out]) else "-")
            else:
                coefs.append("-")
                ses.append("-")
                obs_list.append("-")
                r2_list.append("-")
        
        table_rows.append([spec] + coefs)
        table_rows.append([""] + ses)
        table_rows.append(["  Observations"] + obs_list)
        table_rows.append(["  R-squared"] + r2_list)
        
    final_df = pd.DataFrame(table_rows)
    title_row = pd.DataFrame([["TABLE A1: SENSITIVITY & SPILLOVERS (SPATIAL 25KM BASELINE)"] + [""] * len(outcomes)])
    final_df = pd.concat([title_row, final_df], ignore_index=True)
    final_df.to_csv(f"{OUTPUT_DIR}/table_sensitivity_enhanced.csv", index=False, header=False)
    print(f"    âœ“ Saved: table_sensitivity_enhanced.csv")

def format_sensitivity_table():
    """Original sensitivity table format for backwards compatibility"""
    if not Path(INPUT_SENSITIVITY).exists(): 
        print(f"    ! Sensitivity file not found: {INPUT_SENSITIVITY}")
        return
    
    df = pd.read_csv(INPUT_SENSITIVITY)
    
    outcomes = list(df['Outcome'].unique())
    df_pivot = df.pivot(index='Specification', columns='Outcome', values='Coeff')
    df_se = df.pivot(index='Specification', columns='Outcome', values='SE')
    
    specs = ["Control: Dist to Epicenter", "Intensity (Project Count)", "Spillover Effects"]
    
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
    print(f"    âœ“ Saved: table_sensitivity.csv")

def process_ipw_table():
    if not Path(INPUT_IPW_BALANCE).exists():
        print(f"    ! IPW file not found: {INPUT_IPW_BALANCE}")
        return
    
    df = pd.read_csv(INPUT_IPW_BALANCE)
    title_row = pd.DataFrame([["TABLE A2: IPW BALANCE CHECK"] + [""] * (len(df.columns)-1)], columns=df.columns)
    final_df = pd.concat([title_row, df], ignore_index=True)
    final_df.to_csv(f"{OUTPUT_DIR}/table_ipw_balance_formatted.csv", index=False)
    print(f"    âœ“ Saved: table_ipw_balance_formatted.csv")

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("GENERATING TABLES")
    print("="*70)
    
    if not Path(INPUT_REGRESSION).exists():
        print(f"\nâŒ ERROR: Regression file not found: {INPUT_REGRESSION}")
        print("   â†’ Run 01_main_analysis.py first!")
        return
    
    print(f"\nâœ“ Found regression file: {INPUT_REGRESSION}")
    df = pd.read_csv(INPUT_REGRESSION)
    
    print(f"  - Total rows: {len(df)}")
    print(f"  - Outcomes: {sorted(df['Outcome'].unique())}")
    print(f"  - Samples: {sorted(df['Sample'].unique())}")
    
    # Check if spatial samples exist
    has_spatial = any('Spatial' in str(s) for s in df['Sample'].unique())
    if not has_spatial:
        print("\nâš ï¸  WARNING: No spatial samples found (Spatial 25km, Spatial 50km, etc.)")
        print("   â†’ Re-run 01_main_analysis.py with spatial subsets enabled")
    
    # 1. Main Tables (25km, 50km, Donut)
    print("\n" + "-"*70)
    print("1. MAIN TABLES (25km / 50km / Donut)")
    print("-"*70)
    
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
    
    tables_created = 0
    for var, title in tasks:
        t = format_wide_table(df, var, title) 
        if t is not None: 
            t.to_csv(f"{OUTPUT_DIR}/table_{var}.csv", index=False, header=False)
            print(f"    âœ“ Saved: table_{var}.csv")
            tables_created += 1
        else:
            print(f"    âœ— Skipped: table_{var}.csv (no data)")
    
    print(f"\n  â†’ Created {tables_created}/{len(tasks)} main tables")
    
    # 2. Interaction Table
    print("\n" + "-"*70)
    print("2. HETEROGENEITY TABLE")
    print("-"*70)
    format_interaction_table(df)

    # 3. Robustness Tables
    print("\n" + "-"*70)
    print("3. ROBUSTNESS TABLES")
    print("-"*70)
    
    # Create both versions of sensitivity table
    format_sensitivity_table()  # Original version
    
    # Enhanced version with baseline DiD
    if Path(INPUT_SENSITIVITY).exists():
        df_robustness = pd.read_csv(INPUT_SENSITIVITY)
        format_sensitivity_table_enhanced(df, df_robustness)
    
    process_ipw_table()
    
    print("\n" + "="*70)
    print(f"COMPLETE - Tables saved to: {OUTPUT_DIR}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
