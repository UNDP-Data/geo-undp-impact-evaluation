#!/usr/bin/env python3
"""
Download Myanmar Administrative Boundaries and Identify Cyclone Mocha Affected Areas
=====================================================================================

Downloads Myanmar shapefiles from HDX (Humanitarian Data Exchange) and identifies
areas affected by Cyclone Mocha.

Cyclone Mocha Affected Areas:
- PRIMARY: Rakhine State (Sittwe, Rathedaung, Pauktaw, Myebon, Maungdaw townships)
- SECONDARY: Chin, Sagaing, Magway regions

Requirements:
    pip install geopandas requests pandas matplotlib

Usage:
    python 2_get_myanmar_boundaries.py --output-dir ./data/boundaries
"""

import os
import argparse
import requests
import zipfile
from pathlib import Path
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO


# HDX URLs for Myanmar administrative boundaries
HDX_MYANMAR_ADMIN = {
    'admin0': 'https://data.humdata.org/dataset/cod-ab-mmr',  # Country boundary
    'admin1': 'https://data.humdata.org/dataset/cod-ab-mmr',  # States/Regions
    'admin2': 'https://data.humdata.org/dataset/cod-ab-mmr',  # Townships
}

# Direct download link (may need to be updated)
MYANMAR_BOUNDARIES_URL = "https://data.humdata.org/dataset/2e83d360-0a82-45cf-bde6-223f8264ba42/resource/96973c70-eaa4-4a85-a98c-f5c5c3d8e084/download/mmr_adm_mimu_250k_shp.zip"

# Cyclone Mocha affected areas
MOCHA_AFFECTED_AREAS = {
    'primary': {
        'state': 'Rakhine',
        'townships': [
            'Sittwe',
            'Rathedaung',  
            'Pauktaw',
            'Myebon',
            'Maungdaw',
            'Kyaukpyu',
            'Ponnagyun',
            'Ramree'
        ]
    },
    'secondary': {
        'states': ['Chin', 'Sagaing', 'Magway']
    }
}


def download_myanmar_boundaries(output_dir):
    """
    Download Myanmar administrative boundaries from HDX.
    
    Parameters:
        output_dir: Path to save downloaded files
    
    Returns:
        Path to extracted directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = output_dir / "myanmar_boundaries.zip"
    extract_dir = output_dir / "myanmar_boundaries"
    
    # Check if already downloaded
    if extract_dir.exists() and any(extract_dir.glob("*.shp")):
        print(f"✓ Myanmar boundaries already exist in {extract_dir}")
        return extract_dir
    
    print("\n" + "="*80)
    print("DOWNLOADING MYANMAR ADMINISTRATIVE BOUNDARIES")
    print("="*80)
    print(f"\nSource: Humanitarian Data Exchange (HDX)")
    print(f"URL: {MYANMAR_BOUNDARIES_URL}")
    print(f"\nDownloading...")
    
    try:
        response = requests.get(MYANMAR_BOUNDARIES_URL, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end='')
        
        print(f"\n✓ Downloaded: {zip_path.name} ({zip_path.stat().st_size / (1024**2):.1f} MB)")
        
        # Extract
        print(f"\nExtracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        print(f"✓ Extracted to: {extract_dir}")
        
        # List contents
        shapefiles = list(extract_dir.rglob("*.shp"))
        print(f"\nFound {len(shapefiles)} shapefiles:")
        for shp in shapefiles:
            print(f"  - {shp.name}")
        
        return extract_dir
    
    except Exception as e:
        print(f"\n✗ Error downloading Myanmar boundaries: {e}")
        print("\nMANUAL DOWNLOAD INSTRUCTIONS:")
        print("="*80)
        print("1. Visit: https://data.humdata.org/dataset/cod-ab-mmr")
        print("2. Download: Myanmar - Subnational Administrative Boundaries")
        print("3. Extract to: {output_dir}/myanmar_boundaries/")
        print("4. Re-run this script")
        return None


def load_myanmar_admin_levels(boundaries_dir):
    """
    Load Myanmar administrative boundaries at different levels.
    
    Parameters:
        boundaries_dir: Path to extracted boundaries directory
    
    Returns:
        dict with admin0, admin1, admin2 GeoDataFrames
    """
    boundaries_dir = Path(boundaries_dir)
    
    print("\n" + "="*80)
    print("LOADING ADMINISTRATIVE BOUNDARIES")
    print("="*80)
    
    admin_data = {}
    
    # Try to find shapefiles by level
    # Common patterns: mmr_admbnda_adm0_*.shp, mmr_admbnda_adm1_*.shp, etc.
    
    for level in [0, 1, 2, 3]:
        pattern = f"*adm{level}*.shp"
        shapefiles = list(boundaries_dir.rglob(pattern))
        
        if shapefiles:
            shp_file = shapefiles[0]  # Take first match
            print(f"\nAdmin Level {level}:")
            print(f"  Loading: {shp_file.name}")
            
            try:
                gdf = gpd.read_file(shp_file)
                admin_data[f'admin{level}'] = gdf
                
                print(f"  ✓ Loaded {len(gdf)} features")
                print(f"  Columns: {', '.join(gdf.columns[:10])}")
                
                # Show some example names
                name_cols = [col for col in gdf.columns if 'name' in col.lower()]
                if name_cols:
                    name_col = name_cols[0]
                    examples = gdf[name_col].dropna().unique()[:5]
                    print(f"  Examples: {', '.join(examples)}")
            
            except Exception as e:
                print(f"  ✗ Error loading: {e}")
    
    if not admin_data:
        print("\n⚠ No shapefiles found. Check the directory structure.")
    
    return admin_data


def identify_affected_areas(admin_data):
    """
    Identify and mark Cyclone Mocha affected areas.
    
    Parameters:
        admin_data: dict with admin level GeoDataFrames
    
    Returns:
        GeoDataFrame with affected areas marked
    """
    print("\n" + "="*80)
    print("IDENTIFYING CYCLONE MOCHA AFFECTED AREAS")
    print("="*80)
    
    if 'admin1' not in admin_data:
        print("⚠ Admin level 1 (states) not found")
        return None
    
    admin1 = admin_data['admin1'].copy()
    
    # Find name column
    name_col = None
    for col in admin1.columns:
        if 'name' in col.lower() and 'adm1' in col.lower():
            name_col = col
            break
    
    if not name_col:
        # Try generic name columns
        for col in ['ST', 'ST_RG', 'ADM1_EN', 'NAME_1', 'admin1Name']:
            if col in admin1.columns:
                name_col = col
                break
    
    if not name_col:
        print(f"⚠ Could not identify state/region name column")
        print(f"Available columns: {', '.join(admin1.columns)}")
        # Use first column that's not geometry
        name_col = [c for c in admin1.columns if c != 'geometry'][0]
    
    print(f"\nUsing name column: {name_col}")
    print(f"\nAvailable states/regions:")
    for i, name in enumerate(admin1[name_col].unique(), 1):
        print(f"  {i}. {name}")
    
    # Mark affected areas
    admin1['mocha_impact'] = 'none'
    
    # Primary impact - Rakhine
    rakhine_mask = admin1[name_col].str.contains('Rakhine', case=False, na=False)
    admin1.loc[rakhine_mask, 'mocha_impact'] = 'primary'
    n_primary = rakhine_mask.sum()
    
    # Secondary impact - Chin, Sagaing, Magway
    secondary_states = ['Chin', 'Sagaing', 'Magway']
    for state in secondary_states:
        mask = admin1[name_col].str.contains(state, case=False, na=False)
        admin1.loc[mask, 'mocha_impact'] = 'secondary'
    
    n_secondary = (admin1['mocha_impact'] == 'secondary').sum()
    
    print(f"\n✓ Identified affected areas:")
    print(f"  Primary impact (Rakhine): {n_primary} state(s)")
    print(f"  Secondary impact: {n_secondary} state(s)")
    
    # Print affected area details
    affected = admin1[admin1['mocha_impact'] != 'none']
    print(f"\nAffected states:")
    for _, row in affected.iterrows():
        impact = row['mocha_impact']
        name = row[name_col]
        print(f"  - {name} ({impact.upper()})")
    
    return admin1


def identify_affected_townships(admin_data):
    """
    Identify affected townships (admin level 2).
    
    Parameters:
        admin_data: dict with admin level GeoDataFrames
    
    Returns:
        GeoDataFrame with affected townships marked
    """
    if 'admin2' not in admin_data:
        print("⚠ Admin level 2 (townships) not found")
        return None
    
    admin2 = admin_data['admin2'].copy()
    
    # Find name columns
    state_col = None
    township_col = None
    
    for col in admin2.columns:
        if 'adm1' in col.lower() and 'name' in col.lower():
            state_col = col
        if 'adm2' in col.lower() and 'name' in col.lower():
            township_col = col
    
    if not state_col or not township_col:
        print("⚠ Could not identify state/township columns")
        print(f"Available columns: {', '.join(admin2.columns)}")
        return admin2
    
    print(f"\nUsing columns: {state_col} (state), {township_col} (township)")
    
    # Mark affected townships
    admin2['mocha_impact'] = 'none'
    
    # Primary impact - Rakhine townships
    rakhine_mask = admin2[state_col].str.contains('Rakhine', case=False, na=False)
    
    # Specific townships
    primary_townships = MOCHA_AFFECTED_AREAS['primary']['townships']
    for township in primary_townships:
        mask = rakhine_mask & admin2[township_col].str.contains(township, case=False, na=False)
        admin2.loc[mask, 'mocha_impact'] = 'primary'
    
    # Rest of Rakhine as moderate
    remaining_rakhine = rakhine_mask & (admin2['mocha_impact'] == 'none')
    admin2.loc[remaining_rakhine, 'mocha_impact'] = 'moderate'
    
    # Secondary states
    for state in MOCHA_AFFECTED_AREAS['secondary']['states']:
        mask = admin2[state_col].str.contains(state, case=False, na=False)
        admin2.loc[mask & (admin2['mocha_impact'] == 'none'), 'mocha_impact'] = 'secondary'
    
    print(f"\n✓ Classified townships:")
    print(f"  Primary impact: {(admin2['mocha_impact'] == 'primary').sum()}")
    print(f"  Moderate impact: {(admin2['mocha_impact'] == 'moderate').sum()}")
    print(f"  Secondary impact: {(admin2['mocha_impact'] == 'secondary').sum()}")
    print(f"  No impact: {(admin2['mocha_impact'] == 'none').sum()}")
    
    return admin2


def create_affected_areas_map(admin1, admin2, output_dir):
    """
    Create map showing Cyclone Mocha affected areas.
    
    Parameters:
        admin1: GeoDataFrame with states
        admin2: GeoDataFrame with townships (optional)
        output_dir: Where to save the map
    """
    print("\n" + "="*80)
    print("CREATING AFFECTED AREAS MAP")
    print("="*80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot base map
    admin1.plot(ax=ax, color='lightgray', edgecolor='black', linewidth=0.5, alpha=0.3)
    
    # Plot affected areas by impact level
    impact_colors = {
        'primary': '#d62728',    # Red
        'moderate': '#ff7f0e',   # Orange
        'secondary': '#ffbb78',  # Light orange
        'none': 'lightgray'
    }
    
    for impact, color in impact_colors.items():
        if impact == 'none':
            continue
        mask = admin1['mocha_impact'] == impact
        if mask.any():
            admin1[mask].plot(ax=ax, color=color, edgecolor='black', 
                            linewidth=0.8, alpha=0.7, label=impact.upper())
    
    # Add township boundaries if available
    if admin2 is not None:
        admin2.boundary.plot(ax=ax, edgecolor='gray', linewidth=0.3, alpha=0.5)
    
    # Styling
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_title('Cyclone Mocha Affected Areas - Myanmar\nMay 14, 2023', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Legend
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9, title='Impact Level')
    
    # Grid
    ax.grid(True, alpha=0.2, linestyle='--')
    
    # Add text annotation for Rakhine
    rakhine = admin1[admin1['mocha_impact'] == 'primary']
    if not rakhine.empty:
        centroid = rakhine.geometry.centroid.iloc[0]
        ax.annotate('Rakhine State\n(Primary Impact)', 
                   xy=(centroid.x, centroid.y),
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    output_file = output_dir / 'cyclone_mocha_affected_areas.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved map: {output_file}")
    
    plt.close()


def save_boundaries(admin_data, output_dir):
    """
    Save processed boundaries as GeoJSON for easy use.
    
    Parameters:
        admin_data: dict with admin level GeoDataFrames
        output_dir: Where to save files
    """
    print("\n" + "="*80)
    print("SAVING PROCESSED BOUNDARIES")
    print("="*80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for level, gdf in admin_data.items():
        output_file = output_dir / f'myanmar_{level}_mocha.geojson'
        gdf.to_file(output_file, driver='GeoJSON')
        print(f"✓ Saved: {output_file.name} ({len(gdf)} features)")


def create_rakhine_bbox():
    """
    Create bounding box for Rakhine State (primary affected area).
    
    Returns:
        tuple: (lon_min, lat_min, lon_max, lat_max)
    """
    # Approximate Rakhine State bounding box
    rakhine_bbox = (92.0, 19.0, 94.5, 21.5)
    
    print("\n" + "="*80)
    print("RAKHINE STATE BOUNDING BOX")
    print("="*80)
    print(f"\nBounding box for nightlight clipping:")
    print(f"  Longitude: {rakhine_bbox[0]}° to {rakhine_bbox[2]}°")
    print(f"  Latitude: {rakhine_bbox[1]}° to {rakhine_bbox[3]}°")
    print(f"\nUse this to crop VIIRS data for faster processing")
    
    return rakhine_bbox


def main():
    parser = argparse.ArgumentParser(
        description="Download Myanmar boundaries and identify Cyclone Mocha affected areas"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/boundaries",
        help="Output directory (default: ./data/boundaries)"
    )
    
    parser.add_argument(
        "--create-map",
        action="store_true",
        default=True,
        help="Create affected areas map"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("  MYANMAR ADMINISTRATIVE BOUNDARIES")
    print("  Cyclone Mocha Affected Areas Identification")
    print("="*80)
    print(f"\nCyclone Mocha: May 14, 2023")
    print(f"Primary Impact: Rakhine State")
    print(f"Secondary Impact: Chin, Sagaing, Magway")
    
    # Download boundaries
    boundaries_dir = download_myanmar_boundaries(args.output_dir)
    
    if boundaries_dir is None:
        return
    
    # Load admin levels
    admin_data = load_myanmar_admin_levels(boundaries_dir)
    
    if not admin_data:
        print("\n✗ Could not load administrative boundaries")
        return
    
    # Identify affected areas
    if 'admin1' in admin_data:
        admin_data['admin1'] = identify_affected_areas(admin_data)
    
    if 'admin2' in admin_data:
        admin_data['admin2'] = identify_affected_townships(admin_data)
    
    # Create map
    if args.create_map and 'admin1' in admin_data:
        admin2 = admin_data.get('admin2', None)
        create_affected_areas_map(
            admin_data['admin1'], 
            admin2,
            Path(args.output_dir) / 'maps'
        )
    
    # Save processed boundaries
    save_boundaries(admin_data, Path(args.output_dir) / 'processed')
    
    # Create Rakhine bbox
    rakhine_bbox = create_rakhine_bbox()
    
    # Save bbox to file
    bbox_file = Path(args.output_dir) / 'rakhine_bbox.txt'
    with open(bbox_file, 'w') as f:
        f.write(f"{rakhine_bbox[0]},{rakhine_bbox[1]},{rakhine_bbox[2]},{rakhine_bbox[3]}\n")
    print(f"\n✓ Saved bounding box to: {bbox_file}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nProcessed administrative boundaries:")
    for level, gdf in admin_data.items():
        print(f"  {level}: {len(gdf)} features")
    
    print(f"\nOutput directory: {Path(args.output_dir).absolute()}")
    print(f"\nNext step: Run script 3 to visualize nightlights")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
