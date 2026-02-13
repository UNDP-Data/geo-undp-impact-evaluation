#!/usr/bin/env python3
"""
Simple Myanmar Boundaries Downloader
====================================

Downloads Myanmar administrative boundaries and identifies Cyclone Mocha areas.

Requirements:
    pip install requests geopandas matplotlib

Usage:
    python download_myanmar_simple.py
"""

import os
import requests
import zipfile
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt


def download_myanmar_boundaries():
    """Download Myanmar boundaries from working sources."""
    
    output_dir = Path("data/boundaries")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = output_dir / "myanmar_boundaries.zip"
    extract_dir = output_dir / "myanmar_boundaries"
    
    # Check if already have files
    if extract_dir.exists():
        shp_files = list(extract_dir.rglob("*.shp"))
        if shp_files:
            print(f"✓ Myanmar boundaries already exist ({len(shp_files)} shapefiles)")
            return extract_dir
    
    print("="*80)
    print("  DOWNLOADING MYANMAR BOUNDARIES")
    print("="*80)
    
    # Try multiple sources
    sources = [
        {
            'name': 'MIMU Geonode (Admin 1)',
            'url': 'https://geonode.themimu.info/geoserver/geonode/mmr_polbnda2_adm1_mimu_250k/ows?service=WFS&version=1.0.0&request=GetFeature&typeName=geonode:mmr_polbnda2_adm1_mimu_250k&outputFormat=SHAPE-ZIP'
        },
        {
            'name': 'MIMU Geonode (Admin 2)',
            'url': 'https://geonode.themimu.info/geoserver/geonode/mmr_polbnda2_adm2_mimu_250k/ows?service=WFS&version=1.0.0&request=GetFeature&typeName=geonode:mmr_polbnda2_adm2_mimu_250k&outputFormat=SHAPE-ZIP'
        }
    ]
    
    for source in sources:
        print(f"\nTrying: {source['name']}")
        print(f"URL: {source['url'][:80]}...")
        
        try:
            print("Downloading...")
            response = requests.get(source['url'], timeout=120)
            response.raise_for_status()
            
            # Save
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            size_mb = zip_path.stat().st_size / (1024**2)
            print(f"✓ Downloaded: {size_mb:.1f} MB")
            
            # Extract
            print("Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Check for shapefiles
            shp_files = list(extract_dir.rglob("*.shp"))
            if shp_files:
                print(f"✓ Found {len(shp_files)} shapefiles")
                for shp in shp_files:
                    print(f"  - {shp.name}")
                return extract_dir
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            continue
    
    # All failed
    print("\n" + "="*80)
    print("AUTOMATIC DOWNLOAD FAILED")
    print("="*80)
    print("\nPlease download manually:")
    print("\n1. Visit: https://geonode.themimu.info/")
    print("2. Search: 'Myanmar Administrative Boundaries'")
    print("3. Download: Admin Level 1 and 2 boundaries")
    print(f"4. Extract to: {extract_dir.absolute()}")
    print("\n5. Re-run this script")
    print("="*80)
    
    return None


def identify_mocha_areas(boundaries_dir):
    """Mark Cyclone Mocha affected areas."""
    
    print("\n" + "="*80)
    print("  IDENTIFYING CYCLONE MOCHA AFFECTED AREAS")
    print("="*80)
    
    boundaries_dir = Path(boundaries_dir)
    
    # Find admin level 1 shapefile
    shp_files = list(boundaries_dir.rglob("*adm1*.shp"))
    
    if not shp_files:
        print("\n⚠ No admin level 1 shapefile found")
        return None
    
    shp_file = shp_files[0]
    print(f"\nLoading: {shp_file.name}")
    
    # Load
    gdf = gpd.read_file(shp_file)
    print(f"✓ Loaded {len(gdf)} regions")
    
    # Find name column
    name_cols = [c for c in gdf.columns if 'name' in c.lower() or 'st' in c.lower()]
    if name_cols:
        name_col = name_cols[0]
    else:
        name_col = gdf.columns[1]  # Usually second column
    
    print(f"\nUsing name column: {name_col}")
    print("\nRegions found:")
    for name in gdf[name_col].head(10):
        print(f"  - {name}")
    
    # Mark affected areas
    gdf['mocha_impact'] = 'none'
    
    # Primary: Rakhine
    rakhine_mask = gdf[name_col].str.contains('Rakhine', case=False, na=False)
    gdf.loc[rakhine_mask, 'mocha_impact'] = 'primary'
    
    # Secondary: Chin, Sagaing, Magway
    for state in ['Chin', 'Sagaing', 'Magway']:
        mask = gdf[name_col].str.contains(state, case=False, na=False)
        gdf.loc[mask, 'mocha_impact'] = 'secondary'
    
    print(f"\n✓ Marked affected areas:")
    print(f"  Primary (Rakhine): {(gdf['mocha_impact'] == 'primary').sum()}")
    print(f"  Secondary: {(gdf['mocha_impact'] == 'secondary').sum()}")
    
    # Save processed
    processed_dir = boundaries_dir.parent / "processed"
    processed_dir.mkdir(exist_ok=True)
    
    output_file = processed_dir / "myanmar_admin1_mocha.geojson"
    gdf.to_file(output_file, driver='GeoJSON')
    print(f"\n✓ Saved: {output_file}")
    
    # Create map
    print("\nCreating map...")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot
    gdf.plot(ax=ax, color='lightgray', edgecolor='black', linewidth=0.5, alpha=0.3)
    
    # Highlight affected areas
    gdf[gdf['mocha_impact'] == 'primary'].plot(ax=ax, color='red', 
                                                 edgecolor='black', linewidth=1, 
                                                 alpha=0.7, label='Primary Impact (Rakhine)')
    gdf[gdf['mocha_impact'] == 'secondary'].plot(ax=ax, color='orange',
                                                   edgecolor='black', linewidth=1,
                                                   alpha=0.5, label='Secondary Impact')
    
    ax.set_title('Cyclone Mocha Affected Areas - Myanmar\nMay 14, 2023',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.2)
    
    # Save map
    maps_dir = boundaries_dir.parent / "maps"
    maps_dir.mkdir(exist_ok=True)
    map_file = maps_dir / "cyclone_mocha_affected_areas.png"
    
    plt.savefig(map_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved map: {map_file}")
    plt.close()
    
    return gdf


def main():
    print("="*80)
    print("  MYANMAR BOUNDARIES SETUP")
    print("  Cyclone Mocha Analysis")
    print("="*80)
    
    # Download
    boundaries_dir = download_myanmar_boundaries()
    
    if boundaries_dir is None:
        print("\n✗ Cannot continue without boundaries")
        return
    
    # Process
    gdf = identify_mocha_areas(boundaries_dir)
    
    if gdf is not None:
        print("\n" + "="*80)
        print("SUCCESS!")
        print("="*80)
        print("\n✓ Myanmar boundaries ready")
        print("✓ Affected areas identified")
        print("✓ Map created")
        print("\nNext: Download VIIRS data (if not done)")
        print("  python download_viirs_simple.py")
        print("\nThen: Visualize nightlights")
        print("  python 3_visualize_mocha_nightlights.py \\")
        print("    --viirs-dir ./data/viirs \\")
        print("    --boundaries-dir ./data/boundaries/processed \\")
        print("    --output-dir ./outputs/mocha_analysis")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
