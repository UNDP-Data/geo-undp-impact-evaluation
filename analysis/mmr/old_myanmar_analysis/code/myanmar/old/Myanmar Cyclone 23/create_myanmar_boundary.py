#!/usr/bin/env python3
"""
Create Myanmar Boundary - No Download Needed
============================================

Creates a simple boundary for Myanmar from coordinates.
Good enough for Cyclone Mocha analysis.

Requirements:
    pip install geopandas shapely

Usage:
    python create_myanmar_boundary.py
"""

import geopandas as gpd
from shapely.geometry import Polygon, Point
from pathlib import Path
import matplotlib.pyplot as plt


def create_myanmar_boundary():
    """
    Create Myanmar boundary from approximate coordinates.
    Includes Rakhine State marking.
    """
    
    print("="*80)
    print("  CREATING MYANMAR BOUNDARY")
    print("="*80)
    
    # Myanmar approximate boundary (simplified)
    myanmar_coords = [
        (92.2, 9.8),    # Southwest
        (92.2, 28.5),   # Northwest
        (97.9, 28.2),   # North
        (101.2, 21.5),  # Northeast
        (100.6, 16.0),  # East
        (98.9, 10.8),   # Southeast
        (94.6, 9.8),    # South
        (92.2, 9.8)     # Close polygon
    ]
    
    # Rakhine State approximate boundary
    rakhine_coords = [
        (92.2, 20.0),
        (92.2, 21.5),
        (94.0, 21.5),
        (94.5, 20.0),
        (92.2, 20.0)
    ]
    
    # Create geometries
    myanmar_geom = Polygon(myanmar_coords)
    rakhine_geom = Polygon(rakhine_coords)
    
    # Create GeoDataFrames
    myanmar_gdf = gpd.GeoDataFrame({
        'name': ['Myanmar'],
        'admin_level': [0],
        'mocha_impact': ['country']
    }, geometry=[myanmar_geom], crs='EPSG:4326')
    
    rakhine_gdf = gpd.GeoDataFrame({
        'name': ['Rakhine'],
        'admin_level': [1],
        'mocha_impact': ['primary']
    }, geometry=[rakhine_geom], crs='EPSG:4326')
    
    # Create output directories
    output_dir = Path('data/boundaries/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    maps_dir = Path('data/boundaries/maps')
    maps_dir.mkdir(parents=True, exist_ok=True)
    
    # Save files
    myanmar_file = output_dir / 'myanmar_admin0.geojson'
    rakhine_file = output_dir / 'myanmar_admin1_mocha.geojson'
    
    myanmar_gdf.to_file(myanmar_file, driver='GeoJSON')
    rakhine_gdf.to_file(rakhine_file, driver='GeoJSON')
    
    print(f"\n✓ Created Myanmar boundary: {myanmar_file}")
    print(f"✓ Created Rakhine boundary: {rakhine_file}")
    
    # Create visualization
    print("\nCreating map...")
    
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Plot Myanmar
    myanmar_gdf.boundary.plot(ax=ax, color='black', linewidth=2)
    myanmar_gdf.plot(ax=ax, color='lightgray', alpha=0.3)
    
    # Plot Rakhine (Mocha impact area)
    rakhine_gdf.plot(ax=ax, color='red', alpha=0.5, 
                     edgecolor='darkred', linewidth=2,
                     label='Primary Impact (Rakhine)')
    
    # Mark Cyclone Mocha landfall location (approximate)
    mocha_location = Point(92.9, 20.7)
    ax.plot(92.9, 20.7, 'r*', markersize=20, label='Cyclone Mocha Landfall\n(May 14, 2023)')
    
    # Styling
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Myanmar - Cyclone Mocha Affected Area\nRakhine State', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Save map
    map_file = maps_dir / 'myanmar_mocha_boundary.png'
    plt.savefig(map_file, dpi=300, bbox_inches='tight')
    print(f"✓ Created map: {map_file}")
    
    plt.close()
    
    # Print summary
    print("\n" + "="*80)
    print("SUCCESS!")
    print("="*80)
    print("\nFiles created:")
    print(f"  1. {myanmar_file}")
    print(f"  2. {rakhine_file}")
    print(f"  3. {map_file}")
    
    print("\nBoundaries:")
    print(f"  Myanmar bbox: {myanmar_gdf.total_bounds}")
    print(f"  Rakhine bbox: {rakhine_gdf.total_bounds}")
    
    print("\n✓ Ready for VIIRS analysis!")
    print("\nNext: Once VIIRS files are downloaded, run:")
    print("  python 3_visualize_mocha_nightlights.py \\")
    print("    --viirs-dir ./data/viirs \\")
    print("    --boundaries-dir ./data/boundaries/processed \\")
    print("    --output-dir ./outputs/mocha_analysis")
    
    print("\n" + "="*80)
    
    return myanmar_file, rakhine_file


if __name__ == "__main__":
    try:
        create_myanmar_boundary()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
