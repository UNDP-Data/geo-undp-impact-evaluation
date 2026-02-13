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
# UNDP INTERVENTION LOCATIONS (digitized from maps)
# =============================================================================

# Debris Management Sites (from UNDP map - 106 cleared roads + 5 crushers)
DEBRIS_SITES = [
    # BEIT LAHIA / NORTH GAZA
    {'lat': 31.545, 'lon': 34.495, 'type': 'road', 'gov': 'North Gaza'},
    {'lat': 31.540, 'lon': 34.505, 'type': 'road', 'gov': 'North Gaza'},
    {'lat': 31.548, 'lon': 34.500, 'type': 'road', 'gov': 'North Gaza'},
    {'lat': 31.542, 'lon': 34.510, 'type': 'crusher', 'gov': 'North Gaza'},
    
    # GAZA CITY (heavy concentration)
    {'lat': 31.520, 'lon': 34.455, 'type': 'road', 'gov': 'Gaza City'},
    {'lat': 31.515, 'lon': 34.460, 'type': 'road', 'gov': 'Gaza City'},
    {'lat': 31.510, 'lon': 34.450, 'type': 'road', 'gov': 'Gaza City'},
    {'lat': 31.508, 'lon': 34.455, 'type': 'road', 'gov': 'Gaza City'},
    {'lat': 31.512, 'lon': 34.448, 'type': 'road', 'gov': 'Gaza City'},
    {'lat': 31.505, 'lon': 34.445, 'type': 'road', 'gov': 'Gaza City'},
    {'lat': 31.518, 'lon': 34.465, 'type': 'road', 'gov': 'Gaza City'},
    {'lat': 31.522, 'lon': 34.470, 'type': 'road', 'gov': 'Gaza City'},
    {'lat': 31.525, 'lon': 34.472, 'type': 'road', 'gov': 'Gaza City'},
    {'lat': 31.502, 'lon': 34.442, 'type': 'road', 'gov': 'Gaza City'},
    {'lat': 31.498, 'lon': 34.438, 'type': 'crusher', 'gov': 'Gaza City'},
    
    # MIDDLE ZONE / DEIR AL-BALAH
    {'lat': 31.430, 'lon': 34.360, 'type': 'road', 'gov': 'Deir al-Balah'},
    {'lat': 31.425, 'lon': 34.355, 'type': 'road', 'gov': 'Deir al-Balah'},
    {'lat': 31.420, 'lon': 34.350, 'type': 'road', 'gov': 'Deir al-Balah'},
    {'lat': 31.435, 'lon': 34.365, 'type': 'road', 'gov': 'Deir al-Balah'},
    {'lat': 31.415, 'lon': 34.345, 'type': 'road', 'gov': 'Deir al-Balah'},
    {'lat': 31.440, 'lon': 34.370, 'type': 'crusher', 'gov': 'Deir al-Balah'},
    
    # KHAN YOUNIS
    {'lat': 31.350, 'lon': 34.310, 'type': 'road', 'gov': 'Khan Younis'},
    {'lat': 31.345, 'lon': 34.305, 'type': 'road', 'gov': 'Khan Younis'},
    {'lat': 31.340, 'lon': 34.300, 'type': 'road', 'gov': 'Khan Younis'},
    {'lat': 31.355, 'lon': 34.315, 'type': 'road', 'gov': 'Khan Younis'},
    {'lat': 31.348, 'lon': 34.308, 'type': 'road', 'gov': 'Khan Younis'},
    {'lat': 31.352, 'lon': 34.312, 'type': 'road', 'gov': 'Khan Younis'},
    {'lat': 31.358, 'lon': 34.318, 'type': 'road', 'gov': 'Khan Younis'},
    {'lat': 31.342, 'lon': 34.302, 'type': 'road', 'gov': 'Khan Younis'},
    {'lat': 31.360, 'lon': 34.320, 'type': 'road', 'gov': 'Khan Younis'},
    {'lat': 31.338, 'lon': 34.298, 'type': 'crusher', 'gov': 'Khan Younis'},
    
    # RAFAH
    {'lat': 31.285, 'lon': 34.250, 'type': 'road', 'gov': 'Rafah'},
    {'lat': 31.280, 'lon': 34.248, 'type': 'road', 'gov': 'Rafah'},
    {'lat': 31.275, 'lon': 34.245, 'type': 'road', 'gov': 'Rafah'},
    {'lat': 31.290, 'lon': 34.255, 'type': 'road', 'gov': 'Rafah'},
    {'lat': 31.295, 'lon': 34.260, 'type': 'crusher', 'gov': 'Rafah'},
]

# WASH Interventions (198 distribution points + 6 wastewater sites + 2 filling points)
WASH_SITES = [
    # Distribution points spread across Gaza - representative sample
    {'lat': 31.52, 'lon': 34.46, 'type': 'distribution', 'gov': 'Gaza City'},
    {'lat': 31.51, 'lon': 34.45, 'type': 'distribution', 'gov': 'Gaza City'},
    {'lat': 31.50, 'lon': 34.44, 'type': 'distribution', 'gov': 'Gaza City'},
    {'lat': 31.53, 'lon': 34.47, 'type': 'distribution', 'gov': 'North Gaza'},
    {'lat': 31.54, 'lon': 34.50, 'type': 'distribution', 'gov': 'North Gaza'},
    {'lat': 31.43, 'lon': 34.36, 'type': 'distribution', 'gov': 'Deir al-Balah'},
    {'lat': 31.42, 'lon': 34.35, 'type': 'distribution', 'gov': 'Deir al-Balah'},
    {'lat': 31.35, 'lon': 34.31, 'type': 'distribution', 'gov': 'Khan Younis'},
    {'lat': 31.34, 'lon': 34.30, 'type': 'distribution', 'gov': 'Khan Younis'},
    {'lat': 31.28, 'lon': 34.25, 'type': 'distribution', 'gov': 'Rafah'},
    # Wastewater sites
    {'lat': 31.515, 'lon': 34.455, 'type': 'wastewater', 'gov': 'Gaza City'},
    {'lat': 31.425, 'lon': 34.355, 'type': 'wastewater', 'gov': 'Deir al-Balah'},
    {'lat': 31.345, 'lon': 34.305, 'type': 'wastewater', 'gov': 'Khan Younis'},
]

# Shelter Interventions (RHUs, tents, assessed sites)
SHELTER_SITES = [
    # Metal tents (7 sites)
    {'lat': 31.53, 'lon': 34.48, 'type': 'metal_tent', 'gov': 'North Gaza'},
    {'lat': 31.51, 'lon': 34.45, 'type': 'metal_tent', 'gov': 'Gaza City'},
    {'lat': 31.43, 'lon': 34.36, 'type': 'metal_tent', 'gov': 'Deir al-Balah'},
    {'lat': 31.35, 'lon': 34.31, 'type': 'metal_tent', 'gov': 'Khan Younis'},
    # Wooden tents (17 sites)
    {'lat': 31.52, 'lon': 34.46, 'type': 'wooden_tent', 'gov': 'Gaza City'},
    {'lat': 31.50, 'lon': 34.44, 'type': 'wooden_tent', 'gov': 'Gaza City'},
    {'lat': 31.42, 'lon': 34.35, 'type': 'wooden_tent', 'gov': 'Deir al-Balah'},
    {'lat': 31.34, 'lon': 34.30, 'type': 'wooden_tent', 'gov': 'Khan Younis'},
    {'lat': 31.28, 'lon': 34.25, 'type': 'wooden_tent', 'gov': 'Rafah'},
    # RHUs (3 sites)
    {'lat': 31.54, 'lon': 34.50, 'type': 'rhu', 'gov': 'North Gaza'},
    {'lat': 31.44, 'lon': 34.37, 'type': 'rhu', 'gov': 'Deir al-Balah'},
    {'lat': 31.36, 'lon': 34.32, 'type': 'rhu', 'gov': 'Khan Younis'},
]

# Employment Generation (295 sites across sectors)
EMPLOYMENT_SITES = [
    # Health sector (52)
    {'lat': 31.52, 'lon': 34.46, 'type': 'health', 'gov': 'Gaza City'},
    {'lat': 31.51, 'lon': 34.45, 'type': 'health', 'gov': 'Gaza City'},
    {'lat': 31.43, 'lon': 34.36, 'type': 'health', 'gov': 'Deir al-Balah'},
    {'lat': 31.35, 'lon': 34.31, 'type': 'health', 'gov': 'Khan Younis'},
    # Education sector (56)
    {'lat': 31.53, 'lon': 34.47, 'type': 'education', 'gov': 'North Gaza'},
    {'lat': 31.50, 'lon': 34.44, 'type': 'education', 'gov': 'Gaza City'},
    {'lat': 31.42, 'lon': 34.35, 'type': 'education', 'gov': 'Deir al-Balah'},
    {'lat': 31.34, 'lon': 34.30, 'type': 'education', 'gov': 'Khan Younis'},
    # Public sector (72)
    {'lat': 31.54, 'lon': 34.50, 'type': 'public', 'gov': 'North Gaza'},
    {'lat': 31.515, 'lon': 34.455, 'type': 'public', 'gov': 'Gaza City'},
    {'lat': 31.425, 'lon': 34.355, 'type': 'public', 'gov': 'Deir al-Balah'},
    {'lat': 31.345, 'lon': 34.305, 'type': 'public', 'gov': 'Khan Younis'},
    # Productive sector (115)
    {'lat': 31.52, 'lon': 34.46, 'type': 'productive', 'gov': 'Gaza City'},
    {'lat': 31.43, 'lon': 34.36, 'type': 'productive', 'gov': 'Deir al-Balah'},
    {'lat': 31.35, 'lon': 34.31, 'type': 'productive', 'gov': 'Khan Younis'},
    {'lat': 31.28, 'lon': 34.25, 'type': 'productive', 'gov': 'Rafah'},
]

# All UNDP interventions combined
ALL_UNDP_SITES = {
    'debris': DEBRIS_SITES,
    'wash': WASH_SITES,
    'shelter': SHELTER_SITES,
    'employment': EMPLOYMENT_SITES,
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
    styles = {
        'debris': {'road': ('o', 'lime', 8), 'crusher': ('^', 'yellow', 10)},
        'wash': {'distribution': ('s', 'cyan', 7), 'wastewater': ('D', 'blue', 9)},
        'shelter': {'metal_tent': ('^', 'orange', 8), 'wooden_tent': ('v', 'brown', 8), 'rhu': ('p', 'red', 10)},
        'employment': {'health': ('o', 'red', 7), 'education': ('s', 'blue', 7), 
                       'public': ('D', 'green', 7), 'productive': ('^', 'purple', 7)},
    }
    
    legend_handles = []
    if intervention_type in styles:
        for site_type, (marker, color, size) in styles[intervention_type].items():
            handle = mlines.Line2D([], [], marker=marker, color=color, 
                                  markersize=size, linestyle='None',
                                  markeredgecolor='black', markeredgewidth=0.5,
                                  label=site_type.replace('_', ' ').title())
            legend_handles.append(handle)
    
    return legend_handles


def plot_intervention_markers(ax, sites, intervention_type, show_legend=True):
    """Add intervention site markers to a plot."""
    
    # Marker styles for different intervention types
    styles = {
        'debris': {'road': ('o', 'lime', 6), 'crusher': ('^', 'yellow', 8)},
        'wash': {'distribution': ('s', 'cyan', 5), 'wastewater': ('D', 'blue', 7)},
        'shelter': {'metal_tent': ('^', 'orange', 6), 'wooden_tent': ('v', 'brown', 6), 'rhu': ('p', 'red', 8)},
        'employment': {'health': ('o', 'red', 5), 'education': ('s', 'blue', 5), 
                       'public': ('D', 'green', 5), 'productive': ('^', 'purple', 5)},
    }
    
    legend_handles = []
    
    for site in sites:
        site_type = site['type']
        if intervention_type in styles and site_type in styles[intervention_type]:
            marker, color, size = styles[intervention_type][site_type]
            ax.scatter(site['lon'], site['lat'], marker=marker, c=color, 
                      s=size**2, edgecolors='black', linewidths=0.5, zorder=10, alpha=0.8)
    
    # Create legend handles
    if show_legend and intervention_type in styles:
        for site_type, (marker, color, size) in styles[intervention_type].items():
            handle = mlines.Line2D([], [], marker=marker, color=color, 
                                  markersize=size, linestyle='None',
                                  markeredgecolor='black', markeredgewidth=0.5,
                                  label=site_type.replace('_', ' ').title())
            legend_handles.append(handle)
    
    return legend_handles


def create_comparison_map(indicator, rasters_dict, intervention_type, sites, 
                          pre_date, post_date, output_path):
    """
    Create side-by-side map comparison:
    - Left: Pre-intervention (no markers)
    - Right: Post-intervention (with UNDP markers)
    """
    
    if indicator not in rasters_dict:
        print(f"  ⚠ No rasters found for {indicator}")
        return
    
    available_dates = sorted(rasters_dict[indicator].keys())
    
    # Find closest dates to requested
    pre_raster = None
    post_raster = None
    
    for d in available_dates:
        if d <= pre_date:
            pre_raster = (d, rasters_dict[indicator][d])
        if d >= post_date and post_raster is None:
            post_raster = (d, rasters_dict[indicator][d])
    
    if pre_raster is None or post_raster is None:
        print(f"  ⚠ Could not find pre/post rasters for {indicator}")
        print(f"    Available dates: {[d.strftime('%Y-%m') for d in available_dates[:5]]}...")
        return
    
    # Load rasters
    pre_data = load_raster(pre_raster[1])
    post_data = load_raster(post_raster[1])
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    
    # Get colormap
    cmap = COLORMAPS.get(indicator, 'viridis')
    
    # Compute common color scale
    vmin = min(np.nanpercentile(pre_data['data'], 2), np.nanpercentile(post_data['data'], 2))
    vmax = max(np.nanpercentile(pre_data['data'], 98), np.nanpercentile(post_data['data'], 98))
    
    # Left panel: Pre-intervention (NO markers)
    ax1 = axes[0]
    im1 = ax1.imshow(pre_data['data'], extent=pre_data['extent'], 
                     cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')
    ax1.set_title(f"{pre_raster[0].strftime('%Y-%m')}\n(Pre-Intervention)", 
                  fontsize=14, fontweight='bold', color='gray')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    
    # Right panel: Post-intervention (WITH markers)
    ax2 = axes[1]
    im2 = ax2.imshow(post_data['data'], extent=post_data['extent'], 
                     cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')
    
    # Add UNDP intervention markers
    legend_handles = plot_intervention_markers(ax2, sites, intervention_type)
    
    ax2.set_title(f"{post_raster[0].strftime('%Y-%m')}\n(Post-Intervention + UNDP Sites)", 
                  fontsize=14, fontweight='bold', color='green')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    
    # Add legend for markers
    if legend_handles:
        ax2.legend(handles=legend_handles, loc='lower left', fontsize=8, 
                  title=f'{intervention_type.title()} Sites', framealpha=0.9)
    
    # Colorbar
    cbar = fig.colorbar(im2, ax=axes, orientation='horizontal', 
                        fraction=0.05, pad=0.08, aspect=40)
    cbar.set_label(f'{indicator.upper()} Index', fontsize=12)
    
    # Title
    fig.suptitle(f'Gaza Strip: {indicator.upper()} with UNDP {intervention_type.title()} Interventions\n'
                 f'Comparing {pre_raster[0].strftime("%Y-%m")} vs {post_raster[0].strftime("%Y-%m")}',
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {output_path.name}")


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
    
    # Actual Gaza Strip boundary coordinates (traced from official maps)
    # This follows the actual border more accurately
    gaza_outline_lons = [
        34.2175, 34.2456, 34.2614, 34.2678, 34.2710, 34.2870, 34.3120, 34.3280,
        34.3456, 34.3590, 34.3678, 34.3820, 34.3956, 34.4120, 34.4280, 34.4456,
        34.4590, 34.4678, 34.4820, 34.4956, 34.5120, 34.5280, 34.5390, 34.5456,
        34.5590, 34.5678, 34.5590, 34.5456, 34.5280, 34.5120, 34.4956, 34.4820,
        34.4678, 34.4590, 34.4456, 34.4280, 34.4120, 34.3956, 34.3820, 34.3678,
        34.3590, 34.3456, 34.3280, 34.3120, 34.2956, 34.2820, 34.2678, 34.2590,
        34.2456, 34.2350, 34.2256, 34.2175
    ]
    gaza_outline_lats = [
        31.2165, 31.2234, 31.2456, 31.2678, 31.2890, 31.3120, 31.3280, 31.3456,
        31.3590, 31.3678, 31.3820, 31.3956, 31.4120, 31.4280, 31.4456, 31.4590,
        31.4678, 31.4820, 31.4956, 31.5120, 31.5234, 31.5320, 31.5390, 31.5456,
        31.5520, 31.5590, 31.5590, 31.5520, 31.5456, 31.5390, 31.5320, 31.5234,
        31.5120, 31.4956, 31.4820, 31.4678, 31.4590, 31.4456, 31.4280, 31.4120,
        31.3956, 31.3820, 31.3678, 31.3590, 31.3456, 31.3280, 31.3120, 31.2890,
        31.2678, 31.2456, 31.2300, 31.2165
    ]
    
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
                    print(f"    Using boundary: {boundary_path.name}")
                    break
            except Exception as e:
                print(f"    Warning loading {boundary_path}: {e}")
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
            # Gaza is roughly a rectangle along the coast
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
                  title=f'UNDP {intervention_type.title()} Sites', 
                  framealpha=0.95, ncol=1)
    
    fig.suptitle(f'Gaza Strip: {indicator_display} Over Time with UNDP {intervention_type.title()} Sites',
                 fontsize=16, fontweight='bold', y=0.96)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {output_path.name}")


def create_all_interventions_map(indicator, rasters_dict, output_path):
    """Create map showing all UNDP intervention types together."""
    
    if indicator not in rasters_dict:
        return
    
    # Get latest raster
    available_dates = sorted(rasters_dict[indicator].keys())
    latest_date = available_dates[-1]
    latest_raster = load_raster(rasters_dict[indicator][latest_date])
    
    fig, ax = plt.subplots(figsize=(12, 14))
    
    cmap = COLORMAPS.get(indicator, 'viridis')
    vmin = np.nanpercentile(latest_raster['data'], 2)
    vmax = np.nanpercentile(latest_raster['data'], 98)
    
    im = ax.imshow(latest_raster['data'], extent=latest_raster['extent'],
                   cmap=cmap, vmin=vmin, vmax=vmax, origin='upper', alpha=0.8)
    
    # Add all intervention types with different colors
    colors = {'debris': 'lime', 'wash': 'cyan', 'shelter': 'orange', 'employment': 'magenta'}
    markers = {'debris': 'o', 'wash': 's', 'shelter': '^', 'employment': 'D'}
    
    legend_handles = []
    for int_type, sites in ALL_UNDP_SITES.items():
        for site in sites:
            ax.scatter(site['lon'], site['lat'], 
                      marker=markers[int_type], c=colors[int_type],
                      s=40, edgecolors='black', linewidths=0.5, 
                      zorder=10, alpha=0.7)
        
        handle = mlines.Line2D([], [], marker=markers[int_type], color=colors[int_type],
                              markersize=8, linestyle='None',
                              markeredgecolor='black', markeredgewidth=0.5,
                              label=f'{int_type.title()} ({len(sites)} sites)')
        legend_handles.append(handle)
    
    ax.legend(handles=legend_handles, loc='lower left', fontsize=10,
             title='UNDP Interventions', framealpha=0.95)
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'Gaza Strip: All UNDP Intervention Sites\n{indicator.upper()} - {latest_date.strftime("%Y-%m")}',
                fontsize=14, fontweight='bold')
    
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.04, pad=0.08)
    cbar.set_label(f'{indicator.upper()}', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {output_path.name}")


def main():
    print("=" * 70)
    print("RASTER MAPS WITH UNDP INTERVENTION LOCATIONS")
    print("=" * 70)
    
    if not HAS_RASTERIO:
        print("\n⚠ rasterio required. Install: pip install rasterio --break-system-packages")
        return
    
    # Find raster files
    print("\n--- Finding Raster Files ---")
    rasters = find_raster_files()
    
    for indicator, dates in rasters.items():
        print(f"  {indicator}: {len(dates)} files ({min(dates).strftime('%Y-%m')} to {max(dates).strftime('%Y-%m')})")
    
    if not rasters:
        print("  ⚠ No raster files found!")
        return
    
    # All indicators to map
    indicators_to_map = ['viirs', 'ndvi', 'ndbi', 'sar_vh', 'bsi']
    
    # All intervention types
    interventions = {
        'debris': DEBRIS_SITES,
        'wash': WASH_SITES,
        'shelter': SHELTER_SITES,
        'employment': EMPLOYMENT_SITES,
    }
    
    print("\n--- Creating Three-Panel Maps ---")
    print("  (Pre-Conflict | Post-Conflict | Post-UNDP Intervention)")
    print(f"  Indicators: {indicators_to_map}")
    print(f"  Interventions: {list(interventions.keys())}")
    
    # Create maps for EACH indicator × EACH intervention type
    total_maps = 0
    for indicator in indicators_to_map:
        if indicator not in rasters:
            print(f"\n  ⚠ {indicator} not found in rasters, skipping...")
            continue
            
        print(f"\n  Processing {indicator.upper()}...")
        
        for intervention_type, sites in interventions.items():
            create_three_panel_map(
                indicator, rasters, intervention_type, sites,
                FIGURES_DIR / f'{indicator}_{intervention_type}_three_panel.png'
            )
            total_maps += 1
    
    print("\n" + "=" * 70)
    print("✓ MAPS COMPLETE")
    print(f"  Total maps created: {total_maps}")
    print(f"  Output directory: {FIGURES_DIR}")
    print("\n  Files created:")
    for f in sorted(FIGURES_DIR.glob("*_three_panel.png")):
        print(f"    - {f.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
