#!/usr/bin/env python3
"""
analysis/pse/check_data_availability.py
=======================================
Check which outcomes are available in each data source.
"""

import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("outputs/pse")
DATA_DIR = Path("data")

PANEL_FILE = OUTPUT_DIR / "panel_data.csv"
DONOR_POOL = DATA_DIR / "synthetic_control" / "donor_pool_combined.csv"

# All outcomes we want
OUTCOMES = [
    # Satellite
    'nightlights', 'ndvi', 'ndbi', 'bsi', 'vh', 'vv', 'rvi',
    # Prices
    'price_bread', 'price_wheat', 'price_rice', 'price_milk',
    'price_flour', 'price_sugar', 'price_oil', 'price_chicken',
    # Google Trends
    'gt_water', 'gt_food', 'gt_hospital', 'gt_help', 'gt_electricity',
    'gt_aid', 'gt_shelter', 'gt_medicine', 'gt_bread', 'gt_flour',
    'gt_passport', 'gt_gaza', 'gt_war',
    # Population
    'population', 'pop_density', 'worldpop',
    # Conflict
    'conflict_events', 'conflict_fatalities', 'acled_events', 'acled_fatalities'
]

def main():
    print("=" * 70)
    print("DATA AVAILABILITY CHECK")
    print("=" * 70)
    
    # Load panel
    if PANEL_FILE.exists():
        panel = pd.read_csv(PANEL_FILE)
        panel_cols = set(panel.columns)
        print(f"\nPanel data: {len(panel)} rows, {len(panel_cols)} columns")
        
        # Check Gaza vs WB
        panel['unit_id'] = panel['unit_id'].astype(str)
        gaza = panel[panel['unit_id'].str.startswith('PS02')]
        wb = panel[panel['unit_id'].str.startswith('PS01')]
        print(f"  Gaza: {len(gaza)} obs")
        print(f"  West Bank: {len(wb)} obs")
    else:
        print(f"\n❌ Panel not found: {PANEL_FILE}")
        panel_cols = set()
        gaza = None
        wb = None
    
    # Load donor pool
    if DONOR_POOL.exists():
        donor = pd.read_csv(DONOR_POOL)
        donor_cols = set(donor.columns)
        print(f"\nDonor pool: {len(donor)} rows, {len(donor_cols)} columns")
        if 'country' in donor.columns:
            print(f"  Countries: {donor['country'].unique().tolist()}")
    else:
        print(f"\n❌ Donor pool not found: {DONOR_POOL}")
        donor_cols = set()
        donor = None
    
    # Check each outcome
    print("\n" + "=" * 70)
    print("OUTCOME AVAILABILITY")
    print("=" * 70)
    print(f"\n{'Outcome':<25} {'Panel':<10} {'Gaza':<10} {'WB':<10} {'Donors':<10} {'SC Ready':<10}")
    print("-" * 75)
    
    available_for_sc = []
    
    for outcome in OUTCOMES:
        in_panel = outcome in panel_cols
        
        if in_panel and gaza is not None:
            gaza_n = gaza[outcome].notna().sum() if outcome in gaza.columns else 0
            wb_n = wb[outcome].notna().sum() if outcome in wb.columns else 0
        else:
            gaza_n = 0
            wb_n = 0
        
        in_donor = outcome in donor_cols if donor_cols else False
        if in_donor and donor is not None:
            donor_n = donor[outcome].notna().sum()
        else:
            donor_n = 0
        
        # SC ready if in all three with sufficient obs
        sc_ready = gaza_n > 20 and wb_n > 20 and donor_n > 50
        
        status_panel = "✓" if in_panel else "✗"
        status_gaza = f"{gaza_n}" if gaza_n > 0 else "✗"
        status_wb = f"{wb_n}" if wb_n > 0 else "✗"
        status_donor = f"{donor_n}" if donor_n > 0 else "✗"
        status_sc = "✓ YES" if sc_ready else "✗ No"
        
        print(f"{outcome:<25} {status_panel:<10} {status_gaza:<10} {status_wb:<10} {status_donor:<10} {status_sc:<10}")
        
        if sc_ready:
            available_for_sc.append(outcome)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nOutcomes ready for SC: {len(available_for_sc)}")
    for o in available_for_sc:
        print(f"  - {o}")
    
    # Check what columns exist in panel that we might be missing
    print("\n" + "=" * 70)
    print("ALL PANEL COLUMNS")
    print("=" * 70)
    if panel_cols:
        # Group by prefix
        gt_cols = [c for c in panel_cols if c.startswith('gt_')]
        price_cols = [c for c in panel_cols if c.startswith('price_')]
        other_cols = [c for c in panel_cols if not c.startswith('gt_') and not c.startswith('price_')]
        
        print(f"\nGoogle Trends ({len(gt_cols)}):")
        for c in sorted(gt_cols)[:20]:
            print(f"  {c}")
        if len(gt_cols) > 20:
            print(f"  ... and {len(gt_cols)-20} more")
        
        print(f"\nPrices ({len(price_cols)}):")
        for c in sorted(price_cols):
            print(f"  {c}")
        
        print(f"\nOther key columns:")
        key_cols = ['nightlights', 'ndvi', 'ndbi', 'bsi', 'vh', 'vv', 'rvi', 
                   'conflict_events', 'conflict_fatalities', 'population', 'rainfall']
        for c in key_cols:
            if c in panel_cols:
                print(f"  {c} ✓")
            else:
                print(f"  {c} ✗")
    
    # Check donor pool columns
    print("\n" + "=" * 70)
    print("DONOR POOL COLUMNS")
    print("=" * 70)
    if donor_cols:
        print(f"\nAvailable in donors:")
        for c in sorted(donor_cols):
            if c not in ['admin_id', 'country', 'period_start', 'adm2_pcode', 'adm1_pcode']:
                print(f"  {c}")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    # Missing from donors
    in_panel_not_donor = []
    for outcome in OUTCOMES:
        if outcome in panel_cols and outcome not in donor_cols:
            in_panel_not_donor.append(outcome)
    
    if in_panel_not_donor:
        print(f"\nOutcomes in panel but NOT in donor pool:")
        for o in in_panel_not_donor:
            print(f"  - {o}")
        print("\n  → These cannot use SC method. Use ITS or DiD instead.")
    
    # WorldPop check
    print("\n" + "=" * 70)
    print("WORLDPOP DATA")
    print("=" * 70)
    worldpop_cols = [c for c in panel_cols if 'pop' in c.lower() or 'world' in c.lower()]
    if worldpop_cols:
        print(f"  Found: {worldpop_cols}")
    else:
        print("  ✗ No population data found in panel")
        print("  → Need to download from WorldPop if required")


if __name__ == "__main__":
    main()
