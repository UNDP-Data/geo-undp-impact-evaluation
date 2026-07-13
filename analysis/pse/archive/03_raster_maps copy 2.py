#!/usr/bin/env python3
"""
analysis/pse/09_raster_maps_with_undp.py
========================================
Create raster maps showing UNDP intervention locations overlaid.

Creates side-by-side comparisons:
- Left: Pre-intervention period (without UNDP markers)
- Right: Post-intervention period (with UNDP markers showing where projects are)

This helps visualize whether areas near UNDP interventions show different
patterns compared to areas far from interventions.

Usage:
    python analysis/pse/09_raster_maps_with_undp.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import glob
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import rasterio
    from rasterio.plot import show
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("⚠ rasterio not installed. Run: pip install rasterio --break-system-packages")

try:
    import geopandas as gpd
    from shapely.geometry import Point, box
    HAS_GEO = True
except ImportError:
    HAS_GEO = False

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.lines as mlines

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path("data")
RASTER_DIR = DATA_DIR / "pse" / "rasters"
# Updated to point to the external digitized CSV
UNDP_SITES_CSV = DATA_DIR / "pse" / "external" / "undp_sites_digitized.csv"

OUTPUT_DIR = Path("outputs/pse")
FIGURES_DIR = OUTPUT_DIR / "figures" / "undp_maps"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Key dates
CONFLICT_START = pd.Timestamp("2023-10-07")
DEBRIS_START = pd.Timestamp("2024-10-01")  # When UNDP debris removal started

# Color maps for different indicators
COLORMAPS = {
    'ndvi': 'RdYlGn',      # Red (low veg) to Green (high veg)
    'ndbi': 'RdBu_r',      # Blue (low built-up) to Red (high built-up)
    'bsi': 'YlOrBr',       # Yellow to Brown (bare soil)
    'viirs': 'magma',      # Dark to bright (nightlights)
    'sar_vh': 'viridis',   # SAR backscatter
    'sar_vv': 'viridis',
}

# =============================================================================
# DATA LOADING (Dynamic from CSV)
# =============================================================================

def load_undp_sites():
    """
    Load UNDP sites from the digitized CSV file and organize them by type.
    Maps CSV categories to script categories:
      - WASH -> wash
      - SWM, Debris -> debris
      - Shelter -> shelter
      - Employment -> employment
      - Social Cohesion -> social
    """
    if not UNDP_SITES_CSV.exists():
        print(f"⚠ CSV not found: {UNDP_SITES_CSV}")
        return {}

    print(f"Loading sites from: {UNDP_SITES_CSV}")
    df = pd.read_csv(UNDP_SITES_CSV)
    
    # Category mapping (CSV Type -> Script Category)
    category_map = {
        'WASH': 'wash',
        'SWM': 'debris',       # Solid Waste Management -> debris group
        'Debris': 'debris',
        'Shelter': 'shelter',
        'Employment': 'employment',
        'Social Cohesion': 'social'
    }

    all_sites = {}

    # Iterate through the dataframe
    for _, row in df.iterrows():
        csv_type = row['intervention_type']
        
        # Skip if type not in our map
        if csv_type not in category_map:
            continue
            
        script_cat = category_map[csv_type]
        
        # Initialize list for this category if needed
        if script_cat not in all_sites:
            all_sites[script_cat] = []
            
        # Create site dictionary matching what the plot functions expect
        site_data = {
            'lat': row['latitude'],
            'lon': row['longitude'],
            'type': row['subtype'].strip(),  # Use exact subtype from CSV (e.g. 'Distribution Point')
            'desc': row['location_description']
        }
        
        all_sites[script_cat].append(site_data)
        
    # Summary of loaded data
    for cat, sites in all_sites.items():
        print(f"  - {cat.title()}: {len(sites)} sites")
        
    return all_sites


def get_marker_styles():
    """
    Define marker styles for specific subtypes found in the CSV.
    Returns nested dict: category -> subtype -> (marker, color, size)
    """
    return {
        'debris': {
            # Debris & Roads
            'Cleared Road': ('o', 'lime', 6),
            'Crusher Site': ('^', 'yellow', 9),
            # SWM (Solid Waste)
            'Dumping Site': ('v', 'saddlebrown', 7),
            'Landfill Site': ('s', 'black', 8),
            'Waste Collection Point': ('P', 'forestgreen', 6),
            'Waste Container Site': ('X', 'olive', 6)
        },
        'wash': {
            'Distribution Point': ('s', 'cyan', 5),
            'Filling Point': ('p', 'teal', 7),
            'Wastewater Hotspot': ('D', 'blue', 7),
            'Rehabilitated Water Well': ('*', 'navy', 9)
        },
        'shelter': {
            'Distributed RHU': ('p', 'red', 8),
            'Metal Tent': ('^', 'orange', 6),
            'Wooden Tent': ('v', 'peru', 6),
            'Installed Latrines': ('s', 'pink', 6),
            'Assessed Site': ('x', 'gray', 5)
        },
        'employment': {
            'Health Sector': ('P', 'red', 7),
            'Education Sector': ('s', 'blue', 6),
            'Public Sector': ('D', 'green', 6),
            'Productive Sector': ('*', 'purple', 8)
        },
        'social': {
            'Activity': ('o', 'magenta', 7)
        }
    }


def find_raster_files():
    """Find all raster files grouped by indicator and date."""
    rasters = {}
    
    if not RASTER_DIR.exists():
        print(f"⚠ Raster directory not found: {RASTER_DIR}")
        return rasters
    
    for f in sorted(RASTER_DIR.glob("*.tif")):
        name = f.stem
        
        # Parse indicator and date from filename like "ndbi_pse_202411"
        parts = name.split('_')
        if len(parts) >= 3:
            indicator = parts[0]
            if indicator == 'sar':
                # Handle sar_vh_pse_YYYYMM
                if len(parts) >= 4:
                    indicator = f"{parts[0]}_{parts[1]}"
                    date_str = parts[-1]
                else:
                    date_str = parts[-1]
            else:
                date_str = parts[-1]
            
            # Parse date
            if len(date_str) == 6:
                try:
                    date = datetime(int(date_str[:4]), int(date_str[4:6]), 1)
                    
                    if indicator not in rasters:
                        rasters[indicator] = {}
                    rasters[indicator][date] = f
                except:
                    pass
    
    return rasters


def load_raster(filepath):
    """Load a raster file and return data + metadata."""
    with rasterio.open(filepath) as src:
        data = src.read(1)
        bounds = src.bounds
        transform = src.transform
        crs = src.crs
    
    return {
        'data': data,
        'bounds': bounds,
        'transform': transform,
        'crs': crs,
        'extent': [bounds.left, bounds.right, bounds.bottom, bounds.top],
    }


def create_legend_handles(intervention_type):
    """Create legend handles for intervention markers (for use outside plot)."""
    styles = get_marker_styles()
    
    legend_handles = []
    if intervention_type in styles:
        # Sort so legend is consistent
        for site_type in sorted(styles[intervention_type].keys()):
            marker, color, size = styles[intervention_type][site_type]
            handle = mlines.Line2D([], [], marker=marker, color=color, 
                                  markersize=size, linestyle='None',
                                  markeredgecolor='black', markeredgewidth=0.5,
                                  label=site_type)
            legend_handles.append(handle)
    
    return legend_handles


def plot_intervention_markers(ax, sites, intervention_type, show_legend=True):
    """Add intervention site markers to a plot."""
    styles = get_marker_styles()
    
    legend_handles = []
    used_subtypes = set()

    # Get style map for this intervention type
    type_styles = styles.get(intervention_type, {})
    
    for site in sites:
        subtype = site['type']
        
        # Default style if subtype matches (exact match from CSV)
        if subtype in type_styles:
            marker, color, size = type_styles[subtype]
            ax.scatter(site['lon'], site['lat'], marker=marker, c=color, 
                      s=size**2, edgecolors='black', linewidths=0.5, zorder=10, alpha=0.8)
            used_subtypes.add(subtype)
        else:
            # Fallback for unknown subtypes
            ax.scatter(site['lon'], site['lat'], marker='.', c='white', 
                      s=20, edgecolors='black', zorder=10)

    # Create legend handles only for types that exist in the data/style map
    if show_legend:
        for site_type in sorted(type_styles.keys()):
            # Optional: only show legend items if they appear in the data? 
            # For now, we show all defined styles for clarity or just used ones.
            # Let's show all defined styles to be safe, or just used ones if preferred.
            # Using all defined for completeness.
            marker, color, size = type_styles[site_type]
            handle = mlines.Line2D([], [], marker=marker, color=color, 
                                  markersize=size, linestyle='None',
                                  markeredgecolor='black', markeredgewidth=0.5,
                                  label=site_type)
            legend_handles.append(handle)
    
    return legend_handles


def create_three_panel_map(indicator, rasters_dict, intervention_type, sites, output_path):
    """
    Create three-panel comparison:
    - Left: Pre-conflict (Sep 2023) - NO markers
    - Middle: Post-conflict (Sep 2024) - NO markers  
    - Right: Post-UNDP intervention (Sep 2025) - WITH UNDP markers
    
    Gaza is zoomed in, colorbar below, legend outside.
    """
    
    if indicator not in rasters_dict:
        print(f"  ⚠ No rasters found for {indicator}")
        return
    
    available_dates = sorted(rasters_dict[indicator].keys())
    
    # Target dates - all September for consistency
    targets = [
        datetime(2023, 9, 1),   # Pre-conflict (before Oct 7, 2023)
        datetime(2024, 9, 1),   # Post-conflict, pre-UNDP intervention
        datetime(2025, 9, 1),   # Post-UNDP intervention
    ]
    
    selected = []
    for target in targets:
        closest = min(available_dates, key=lambda d: abs((d - target).days))
        selected.append((closest, rasters_dict[indicator][closest]))
    
    # Load rasters
    raster_data = [load_raster(f) for _, f in selected]
    
    # Gaza extent (zoomed in)
    gaza_extent = [34.20, 34.60, 31.20, 31.60]  # [left, right, bottom, top]
    
    # Try to load actual shapefile/geojson if available
    gaza_gdf = None
    boundary_paths = [
        Path("data/pse/boundaries/gaza_admin2.geojson"),  # Best - Gaza specific
        Path("data/pse/boundaries/pse_admin1.geojson"),   # Fallback - all Palestine admin1
        Path("data/pse/boundaries/pse_admin0.geojson"),   # Fallback - country level
    ]
    
    for boundary_path in boundary_paths:
        if boundary_path.exists() and HAS_GEO:
            try:
                gdf = gpd.read_file(boundary_path)
                # For gaza_admin2, use all rows (it's already Gaza only)
                if 'gaza' in boundary_path.name.lower():
                    gaza_gdf = gdf
                else:
                    # Filter to Gaza if it's a broader file
                    for col in ['admin1Name', 'NAME_1', 'name', 'ADM1_EN']:
                        if col in gdf.columns:
                            mask = gdf[col].str.contains('Gaza', case=False, na=False)
                            if mask.any():
                                gaza_gdf = gdf[mask]
                                break
                
                if gaza_gdf is not None and len(gaza_gdf) > 0:
                    # print(f"    Using boundary: {boundary_path.name}")
                    break
            except Exception as e:
                pass
    
    # Create figure with more space between panels
    fig, axes = plt.subplots(1, 3, figsize=(20, 9))
    plt.subplots_adjust(bottom=0.18, wspace=0.25, left=0.06, right=0.94)
    
    # Select colormap - special handling for nightlights
    if indicator == 'viirs':
        cmap = LinearSegmentedColormap.from_list('nightlights', 
            ['black', '#1a0a00', '#3d1c02', '#6b3a0a', '#a86412', '#d9a521', '#f0d040', '#ffec70', '#ffff99'])
    else:
        cmap = COLORMAPS.get(indicator, 'viridis')
    
    # Compute common color scale from all three rasters
    all_vals = []
    for r in raster_data:
        all_vals.extend(r['data'].flatten())
    all_vals = np.array(all_vals)
    all_vals = all_vals[~np.isnan(all_vals)]
    
    if indicator == 'viirs':
        vmin = 0
        vmax = np.percentile(all_vals, 95)
    else:
        vmin = np.percentile(all_vals, 2)
        vmax = np.percentile(all_vals, 98)
    
    titles = [
        ('Pre-Conflict', 'blue'),
        ('Post-Conflict', 'red'),
        ('Post-UNDP Intervention', 'green'),
    ]
    
    indicator_display = 'Nightlights (VIIRS)' if indicator == 'viirs' else indicator.upper()
    
    for i, (ax, (date, _), data, (title, color)) in enumerate(zip(axes, selected, raster_data, titles)):
        im = ax.imshow(data['data'], extent=data['extent'], 
                       cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')
        
        # Zoom to Gaza
        ax.set_xlim(gaza_extent[0], gaza_extent[1])
        ax.set_ylim(gaza_extent[2], gaza_extent[3])
        
        # Add Gaza outline
        if gaza_gdf is not None and len(gaza_gdf) > 0:
            # Use actual shapefile
            gaza_gdf.boundary.plot(ax=ax, color='white', linewidth=1.5, alpha=0.9)
        else:
            # Use simplified rectangle outline for Gaza Strip
            rect_lons = [34.22, 34.22, 34.46, 34.56, 34.56, 34.46, 34.22]
            rect_lats = [31.22, 31.35, 31.60, 31.55, 31.30, 31.22, 31.22]
            ax.plot(rect_lons, rect_lats, 'w-', linewidth=1.5, alpha=0.9)
        
        # Only add markers to the LAST panel (post-intervention)
        if i == 2:
            plot_intervention_markers(ax, sites, intervention_type, show_legend=False)
        
        ax.set_title(f"{date.strftime('%Y-%m')}\n({title})", 
                    fontsize=14, fontweight='bold', color=color)
        ax.set_xlabel('Longitude', fontsize=11)
        if i == 0:
            ax.set_ylabel('Latitude', fontsize=11)
        else:
            ax.set_yticklabels([])
        
        ax.set_xticks([34.25, 34.35, 34.45, 34.55])
    
    # Colorbar and legend
    cbar_ax = fig.add_axes([0.06, 0.08, 0.55, 0.025])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(f'{indicator_display} Index', fontsize=12)
    
    legend_handles = create_legend_handles(intervention_type)
    if legend_handles:
        fig.legend(handles=legend_handles, loc='lower right', 
                  bbox_to_anchor=(0.94, 0.06), fontsize=10,
                  title=f'{intervention_type.title()} Sites', 
                  framealpha=0.95, ncol=1)
    
    fig.suptitle(f'Gaza Strip: {indicator_display} Over Time with UNDP {intervention_type.title()} Sites',
                 fontsize=16, fontweight='bold', y=0.96)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {output_path.name}")


def main():
    print("=" * 70)
    print("RASTER MAPS WITH UNDP INTERVENTION LOCATIONS (DIGITIZED DATA)")
    print("=" * 70)
    
    if not HAS_RASTERIO:
        print("\n⚠ rasterio required. Install: pip install rasterio --break-system-packages")
        return
    
    # 1. Load UNDP Sites from CSV
    print("\n--- Loading Intervention Data ---")
    all_interventions = load_undp_sites()
    
    if not all_interventions:
        print("⚠ No intervention data loaded. Checking file existence...")
        if not UNDP_SITES_CSV.exists():
            print(f"  File missing: {UNDP_SITES_CSV}")
        return

    # 2. Find raster files
    print("\n--- Finding Raster Files ---")
    rasters = find_raster_files()
    
    for indicator, dates in rasters.items():
        print(f"  {indicator}: {len(dates)} files ({min(dates).strftime('%Y-%m')} to {max(dates).strftime('%Y-%m')})")
    
    if not rasters:
        print("  ⚠ No raster files found!")
        return
    
    # Indicators to map
    indicators_to_map = ['viirs', 'ndvi', 'ndbi', 'sar_vh', 'bsi']
    
    print("\n--- Creating Three-Panel Maps ---")
    
    # Create maps for EACH indicator × EACH intervention type found in CSV
    total_maps = 0
    for indicator in indicators_to_map:
        if indicator not in rasters:
            print(f"\n  ⚠ {indicator} not found in rasters, skipping...")
            continue
            
        print(f"\n  Processing {indicator.upper()}...")
        
        for intervention_type, sites in all_interventions.items():
            if not sites:
                continue
                
            create_three_panel_map(
                indicator, rasters, intervention_type, sites,
                FIGURES_DIR / f'{indicator}_{intervention_type}_three_panel.png'
            )
            total_maps += 1
    
    print("\n" + "=" * 70)
    print("✓ MAPS COMPLETE")
    print(f"  Total maps created: {total_maps}")
    print(f"  Output directory: {FIGURES_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
