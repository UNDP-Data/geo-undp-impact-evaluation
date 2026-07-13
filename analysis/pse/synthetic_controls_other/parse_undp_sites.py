#!/usr/bin/env python3
"""
pipeline/parse_undp_sites.py
============================
Parse UNDP intervention site coordinates from project_data.xlsx.

Creates standardized undp_sites.csv for use in site-level DiD analysis.

Usage:
    python -m pipeline.parse_undp_sites --config config/myanmar.yaml
    python pipeline/parse_undp_sites.py --config config/myanmar.yaml
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import yaml
import argparse


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_excel_sites(filepath: Path, lat_col: str, lon_col: str) -> pd.DataFrame:
    """Parse UNDP sites from Excel file."""
    print(f"\n--- Parsing: {filepath.name} ---")
    
    if not filepath.exists():
        print(f"  ✗ File not found: {filepath}")
        return None
    
    # Try reading Excel
    try:
        df = pd.read_excel(filepath)
    except Exception as e:
        print(f"  ✗ Error reading Excel: {e}")
        return None
    
    print(f"  Loaded {len(df)} rows")
    print(f"  Columns: {list(df.columns)[:8]}...")
    
    # Find coordinate columns
    cols_map = {c: c for c in df.columns}
    
    # Try exact match first
    actual_lat_col = None
    actual_lon_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if lat_col.lower() in col_lower or 'latitude' in col_lower:
            if actual_lat_col is None:
                actual_lat_col = col
        if lon_col.lower() in col_lower or 'longitude' in col_lower:
            if actual_lon_col is None:
                actual_lon_col = col
    
    if actual_lat_col is None or actual_lon_col is None:
        # Try combined GPS column
        gps_col = None
        for col in df.columns:
            if 'gps' in col.lower() and 'lat' not in col.lower() and 'lon' not in col.lower():
                gps_col = col
                break
        
        if gps_col:
            print(f"  Using combined GPS column: {gps_col}")
            sites = []
            for idx, row in df.iterrows():
                try:
                    gps_str = str(row[gps_col])
                    parts = gps_str.split()
                    if len(parts) >= 2:
                        lat = float(parts[0])
                        lon = float(parts[1])
                        if 9 <= lat <= 29 and 92 <= lon <= 102:  # Myanmar bounds
                            sites.append({
                                'site_id': f'UNDP_{idx:04d}',
                                'latitude': lat,
                                'longitude': lon,
                            })
                except:
                    continue
            
            if sites:
                return pd.DataFrame(sites)
        
        print(f"  ✗ Could not find coordinate columns")
        print(f"    Looking for: {lat_col}, {lon_col}")
        print(f"    Available: {list(df.columns)}")
        return None
    
    print(f"  Using columns: {actual_lat_col}, {actual_lon_col}")
    
    # Extract coordinates
    sites = []
    for idx, row in df.iterrows():
        try:
            lat = float(row[actual_lat_col])
            lon = float(row[actual_lon_col])
            
            # Validate for Myanmar
            if 9 <= lat <= 29 and 92 <= lon <= 102:
                site = {
                    'site_id': f'UNDP_{idx:04d}',
                    'latitude': lat,
                    'longitude': lon,
                }
                
                # Try to get intervention type
                for potential_col in ['type', 'category', 'sector', 'activity', 'intervention_type']:
                    for col in df.columns:
                        if potential_col.lower() in col.lower():
                            site['intervention_type'] = str(row[col])
                            break
                    if 'intervention_type' in site:
                        break
                
                if 'intervention_type' not in site:
                    site['intervention_type'] = 'Unknown'
                
                # Try to get location name
                for potential_col in ['township', 'location', 'village', 'admin', 'name']:
                    for col in df.columns:
                        if potential_col.lower() in col.lower():
                            val = row[col]
                            if pd.notna(val):
                                site['location_name'] = str(val)
                                break
                    if 'location_name' in site:
                        break
                
                sites.append(site)
        except (ValueError, TypeError):
            continue
    
    if sites:
        print(f"  ✓ Extracted {len(sites)} valid sites")
        return pd.DataFrame(sites)
    else:
        print("  ✗ No valid coordinates found")
        return None


def parse_geojson_sites(filepath: Path) -> pd.DataFrame:
    """Parse UNDP sites from GeoJSON file."""
    print(f"\n--- Parsing: {filepath.name} ---")
    
    if not filepath.exists():
        print(f"  ✗ File not found")
        return None
    
    try:
        gdf = gpd.read_file(filepath)
        print(f"  Loaded {len(gdf)} features")
        
        if len(gdf) == 0:
            return None
        
        sites = []
        for idx, row in gdf.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            
            if geom.geom_type == 'Point':
                lon, lat = geom.x, geom.y
            else:
                centroid = geom.centroid
                lon, lat = centroid.x, centroid.y
            
            # Validate for Myanmar
            if 9 <= lat <= 29 and 92 <= lon <= 102:
                site = {
                    'site_id': f'UNDP_GJ_{idx:04d}',
                    'latitude': lat,
                    'longitude': lon,
                    'intervention_type': 'Unknown',
                }
                
                # Get properties
                for col in gdf.columns:
                    if col == 'geometry':
                        continue
                    col_lower = col.lower()
                    if 'type' in col_lower or 'category' in col_lower:
                        site['intervention_type'] = str(row[col])
                    elif 'name' in col_lower or 'township' in col_lower:
                        site['location_name'] = str(row[col])
                
                sites.append(site)
        
        if sites:
            print(f"  ✓ Extracted {len(sites)} valid sites")
            return pd.DataFrame(sites)
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    return None


def main():
    parser = argparse.ArgumentParser(description='Parse UNDP site coordinates')
    parser.add_argument('--config', '-c', required=True, help='Path to country config YAML')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    country_code = config.get('country_code', 'mmr').lower()
    country_name = config.get('country_name', 'Unknown')
    
    print("=" * 60)
    print(f"UNDP SITE PARSER: {country_name}")
    print("=" * 60)
    
    # Paths
    data_dir = Path(f"data/{country_code}")
    output_dir = data_dir / "external"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    undp_config = config.get('undp_sites', {})
    excel_path = Path(undp_config.get('file_path', f'data/{country_code}/undp/project_data.xlsx'))
    geojson_path = Path(undp_config.get('geojson_path', f'data/{country_code}/undp/undp_interventions.geojson'))
    lat_col = undp_config.get('lat_column', '_L6. GPS_latitude')
    lon_col = undp_config.get('lon_column', '_L6. GPS_longitude')
    
    # Parse from both sources
    excel_sites = parse_excel_sites(excel_path, lat_col, lon_col)
    geojson_sites = parse_geojson_sites(geojson_path)
    
    # Combine
    dfs = []
    if excel_sites is not None and len(excel_sites) > 0:
        dfs.append(excel_sites)
    if geojson_sites is not None and len(geojson_sites) > 0:
        dfs.append(geojson_sites)
    
    if not dfs:
        print("\n✗ No sites found from any source")
        return
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Remove duplicates (within ~100m)
    combined['lat_round'] = (combined['latitude'] * 1000).round()
    combined['lon_round'] = (combined['longitude'] * 1000).round()
    before = len(combined)
    combined = combined.drop_duplicates(subset=['lat_round', 'lon_round'])
    combined = combined.drop(columns=['lat_round', 'lon_round'])
    
    if before > len(combined):
        print(f"\n  Removed {before - len(combined)} duplicates")
    
    # Re-index
    combined = combined.reset_index(drop=True)
    combined['site_id'] = [f'UNDP_{i:04d}' for i in range(len(combined))]
    combined['group'] = 'Treated'
    
    # Save
    output_csv = output_dir / "undp_sites.csv"
    combined.to_csv(output_csv, index=False)
    
    # Also save as GeoJSON
    output_geojson = output_dir / "undp_sites.geojson"
    sites_gdf = gpd.GeoDataFrame(
        combined,
        geometry=gpd.points_from_xy(combined.longitude, combined.latitude),
        crs="EPSG:4326"
    )
    sites_gdf.to_file(output_geojson, driver='GeoJSON')
    
    print(f"\n{'=' * 60}")
    print(f"✓ SUCCESS: {len(combined)} sites extracted")
    print(f"{'=' * 60}")
    print(f"  CSV:     {output_csv}")
    print(f"  GeoJSON: {output_geojson}")
    
    print(f"\n  Coordinate ranges:")
    print(f"    Latitude:  {combined['latitude'].min():.4f} to {combined['latitude'].max():.4f}")
    print(f"    Longitude: {combined['longitude'].min():.4f} to {combined['longitude'].max():.4f}")
    
    if 'intervention_type' in combined.columns:
        print(f"\n  By intervention type:")
        for t, n in combined['intervention_type'].value_counts().head(10).items():
            print(f"    {t}: {n}")
    
    print(f"\n  Next: make run COUNTRY=myanmar")


if __name__ == "__main__":
    main()
