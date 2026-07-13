#!/usr/bin/env python3
"""
Convert UNDP Myanmar Data to GeoJSON with Intervention Types
=============================================================

Converts the UNDP Early Recovery Assessment to GeoJSON with colored markers
for different intervention types (solar panels, construction, medical, etc.)

Usage:
    python convert_undp_with_interventions.py \\
        --input Myanmar_Early_Recovery_Needs_Assessment_-_Dataset_cleaned_anonymous_Final.xlsx \\
        --output undp_interventions.geojson
"""

import argparse
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
import sys


def load_and_classify_interventions(excel_file):
    """Load UNDP data and classify intervention types."""
    print(f"\nLoading UNDP data from: {excel_file}")
    
    df = pd.read_excel(excel_file)
    print(f"✓ Loaded {len(df)} rows")
    
    # Get coordinates
    lat_col = '_L6. GPS_latitude'
    lon_col = '_L6. GPS_longitude'
    
    # Remove rows with missing coordinates
    df_clean = df[[lat_col, lon_col]].dropna()
    print(f"  {len(df_clean)} rows have valid coordinates")
    
    if len(df_clean) == 0:
        print("✗ No valid coordinates found!")
        return None
    
    # Merge back with full data
    df_clean = df.loc[df_clean.index].copy()
    
    # Classify intervention types based on E2a columns
    df_clean['intervention_type'] = 'Other'
    df_clean['intervention_color'] = '#FFFFFF'  # White default
    
    # Solar home systems
    solar_col = 'E2a. If E1= YES, what type of assistance (non-food items and cash) for early recovery was received or provided?/Solar home systems'
    if solar_col in df_clean.columns:
        solar_mask = df_clean[solar_col] == 1.0
        df_clean.loc[solar_mask, 'intervention_type'] = 'Solar Energy'
        df_clean.loc[solar_mask, 'intervention_color'] = '#FFD700'  # Gold
        print(f"  {solar_mask.sum()} solar energy interventions")
    
    # Construction/house repair
    construction_col = 'E2a. If E1= YES, what type of assistance (non-food items and cash) for early recovery was received or provided?/Construction materials for house repair (CGI sheets, Cement, etc.)'
    tool_col = 'E2a. If E1= YES, what type of assistance (non-food items and cash) for early recovery was received or provided?/Tool kits for house repair'
    
    if construction_col in df_clean.columns and tool_col in df_clean.columns:
        construction_mask = (df_clean[construction_col] == 1.0) | (df_clean[tool_col] == 1.0)
        # Only set if not already solar
        construction_mask = construction_mask & (df_clean['intervention_type'] == 'Other')
        df_clean.loc[construction_mask, 'intervention_type'] = 'Construction/Repair'
        df_clean.loc[construction_mask, 'intervention_color'] = '#FF6347'  # Tomato red
        print(f"  {construction_mask.sum()} construction/repair interventions")
    
    # Medical assistance
    medical_col = 'E2a. If E1= YES, what type of assistance (non-food items and cash) for early recovery was received or provided?/Medicines and medical supplies (including assistive devices)'
    medical_protection_col = 'E2b. If E1= YES, what type of protection for early recovery was received or provided?/Medical assistance'
    
    if medical_col in df_clean.columns:
        medical_mask = (df_clean[medical_col] == 1.0)
        if medical_protection_col in df_clean.columns:
            medical_mask = medical_mask | (df_clean[medical_protection_col] == 1.0)
        # Only set if not already classified
        medical_mask = medical_mask & (df_clean['intervention_type'] == 'Other')
        df_clean.loc[medical_mask, 'intervention_type'] = 'Medical/Health'
        df_clean.loc[medical_mask, 'intervention_color'] = '#32CD32'  # Lime green
        print(f"  {medical_mask.sum()} medical/health interventions")
    
    # Shelter
    shelter_col = 'E2b. If E1= YES, what type of protection for early recovery was received or provided?/Shelter or safe spaces/center'
    if shelter_col in df_clean.columns:
        shelter_mask = (df_clean[shelter_col] == 1.0) & (df_clean['intervention_type'] == 'Other')
        df_clean.loc[shelter_mask, 'intervention_type'] = 'Shelter'
        df_clean.loc[shelter_mask, 'intervention_color'] = '#1E90FF'  # Dodger blue
        print(f"  {shelter_mask.sum()} shelter interventions")
    
    # Cash-for-work
    cash_col = 'E2a. If E1= YES, what type of assistance (non-food items and cash) for early recovery was received or provided?/Cash-for-work programs'
    if cash_col in df_clean.columns:
        cash_mask = (df_clean[cash_col] == 1.0) & (df_clean['intervention_type'] == 'Other')
        df_clean.loc[cash_mask, 'intervention_type'] = 'Cash-for-Work'
        df_clean.loc[cash_mask, 'intervention_color'] = '#FF1493'  # Deep pink
        print(f"  {cash_mask.sum()} cash-for-work interventions")
    
    # Agricultural support
    agri_col = 'E2a. If E1= YES, what type of assistance (non-food items and cash) for early recovery was received or provided?/Agricultural inputs (seeds, fertilizer, water pumps…)'
    if agri_col in df_clean.columns:
        agri_mask = (df_clean[agri_col] == 1.0) & (df_clean['intervention_type'] == 'Other')
        df_clean.loc[agri_mask, 'intervention_type'] = 'Agricultural Support'
        df_clean.loc[agri_mask, 'intervention_color'] = '#228B22'  # Forest green
        print(f"  {agri_mask.sum()} agricultural support interventions")
    
    # Count remaining 'Other'
    other_count = (df_clean['intervention_type'] == 'Other').sum()
    print(f"  {other_count} other/general assistance interventions")
    
    # Add location info
    df_clean['township'] = df['L3. Township'] if 'L3. Township' in df.columns else 'Unknown'
    df_clean['state'] = df['L1. State/ Region'] if 'L1. State/ Region' in df.columns else 'Unknown'
    
    return df_clean[[lat_col, lon_col, 'intervention_type', 'intervention_color', 'township', 'state']]


def create_geojson(df, output_file):
    """Convert dataframe to GeoJSON."""
    print(f"\nCreating GeoJSON...")
    
    lat_col = '_L6. GPS_latitude'
    lon_col = '_L6. GPS_longitude'
    
    # Create point geometries
    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    
    # Save to file
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_file, driver='GeoJSON')
    
    print(f"✓ Saved GeoJSON: {output_file}")
    print(f"  {len(gdf)} intervention locations")
    
    # Show summary by type
    print("\nIntervention type summary:")
    for itype, count in gdf['intervention_type'].value_counts().items():
        color = gdf[gdf['intervention_type'] == itype]['intervention_color'].iloc[0]
        print(f"  {itype:20s}: {count:3d} locations (color: {color})")
    
    return gdf


def main():
    parser = argparse.ArgumentParser(
        description="Convert UNDP data to GeoJSON with intervention type classification"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input Excel file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="undp_interventions.geojson",
        help="Output GeoJSON file"
    )
    
    args = parser.parse_args()
    
    # Check input exists
    input_file = Path(args.input)
    if not input_file.exists():
        print(f"\n✗ Input file not found: {input_file}")
        return 1
    
    # Load and classify
    df = load_and_classify_interventions(input_file)
    if df is None or len(df) == 0:
        return 1
    
    # Create GeoJSON
    gdf = create_geojson(df, args.output)
    
    print("\n" + "="*80)
    print("CONVERSION COMPLETE")
    print("="*80)
    print(f"\nUse with visualization:")
    print(f"  python visualize_2025_earthquake_nightlights_FIXED.py \\")
    print(f"    --viirs-dir data/viirs_25 \\")
    print(f"    --boundaries-dir data/boundaries/processed \\")
    print(f"    --undp-support {args.output} \\")
    print(f"    --output-dir outputs/earthquake_2025_analysis")
    print("="*80 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
