#!/usr/bin/env python3
"""
Comprehensive Landsat Visualization - Myanmar 2025 Earthquake
==============================================================

Creates spatial visualizations for Landsat indices (NDBI, NDVI, NBR, NDMI):
1. Before/After/Recovery maps WITHOUT UNDP locations
2. Before/After/Recovery maps WITH UNDP locations
3. All four indices displayed separately

Similar to nightlight visualization but using Landsat multispectral data.

Usage:
    python visualize_landsat_comprehensive.py \
        --landsat-dir ./data/landsat_composites \
        --boundaries ./data/boundaries/processed/mmr_admin1.geojson \
        --undp-interventions ./data/undp/undp_interventions.geojson \
        --output-dir ./outputs/landsat_visualizations
"""

import os
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

try:
    import rioxarray as rxr
    import xarray as xr
    from rasterio.features import geometry_mask
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    RASTER_OK = True
except ImportError:
    RASTER_OK = False
    print("⚠ rioxarray not installed. Install with: pip install rioxarray")


# 2025 Earthquake event date
EARTHQUAKE_DATE = datetime(2025, 3, 28)

# Period definitions
PRE_PERIOD = datetime(2025, 2, 1)
POST_PERIOD = datetime(2025, 4, 1)
RECOVERY_PERIOD = datetime(2025, 9, 1)


def find_landsat_files(landsat_dir, start_date=None, end_date=None):
    """Find all Landsat monthly composite files in directory."""
    landsat_dir = Path(landsat_dir)
    
    tif_files = list(landsat_dir.glob("LANDSAT_*.tif"))
    
    print(f"\nFound {len(tif_files)} Landsat composite files")
    
    if len(tif_files) == 0:
        print("\n⚠ No Landsat .tif files found!")
        return []
    
    files_with_dates = []
    
    for f in tif_files:
        # Parse LANDSAT_2025-01.tif format
        parts = f.stem.split('_')
        if len(parts) >= 2:
            date_str = parts[1]  # "2025-01"
            try:
                year, month = date_str.split('-')
                date = datetime(int(year), int(month), 1)
                
                if start_date and date < start_date:
                    continue
                if end_date and date > end_date:
                    continue
                
                files_with_dates.append((date, f))
            except:
                continue
    
    files_with_dates.sort(key=lambda x: x[0])
    
    if len(files_with_dates) > 0:
        print(f"Date range: {files_with_dates[0][0].strftime('%Y-%m')} to {files_with_dates[-1][0].strftime('%Y-%m')}")
    
    return files_with_dates


def load_and_clip_landsat(tif_file, bbox):
    """Load Landsat file and clip to bounding box."""
    if not RASTER_OK:
        raise RuntimeError("rioxarray not installed")
    
    raster = rxr.open_rasterio(tif_file, masked=True)
    
    lon_min, lat_min, lon_max, lat_max = bbox
    raster_clipped = raster.rio.clip_box(
        minx=lon_min, miny=lat_min,
        maxx=lon_max, maxy=lat_max
    )
    
    return raster_clipped


def mask_raster_to_boundaries(raster, boundaries_gdf):
    """
    Mask raster to only show values within boundaries.
    """
    # Handle multi-band or single-band rasters
    if len(raster.shape) == 3:
        # Multi-band raster
        mask = geometry_mask(
            boundaries_gdf.geometry,
            out_shape=(raster.shape[1], raster.shape[2]),
            transform=raster.rio.transform(),
            invert=True
        )
        
        raster_masked = raster.copy()
        for i in range(raster.shape[0]):
            raster_masked.values[i, ~mask] = np.nan
    else:
        # Single-band raster
        mask = geometry_mask(
            boundaries_gdf.geometry,
            out_shape=raster.shape,
            transform=raster.rio.transform(),
            invert=True
        )
        
        raster_masked = raster.copy()
        raster_masked.values[~mask] = np.nan
    
    return raster_masked


def calculate_indices(raster):
    """
    Extract or calculate spectral indices from Landsat composite.
    
    Handles two cases:
    1. 4-band files with pre-calculated indices (NDVI, NDBI, NBR, NDMI)
    2. 6-band files with raw spectral bands (calculates indices)
    """
    indices = {}
    
    try:
        num_bands = raster.shape[0]
        print(f"  Raster shape: {raster.shape}")
        print(f"  Number of bands: {num_bands}")
        
        if num_bands == 4:
            # Pre-calculated indices in order: NDVI, NDBI, NBR, NDMI
            print("  → Detected 4-band composite (pre-calculated indices)")
            
            indices['NDVI'] = raster[0, :, :].values.astype(float)
            indices['NDBI'] = raster[1, :, :].values.astype(float)
            indices['NBR'] = raster[2, :, :].values.astype(float)
            indices['NDMI'] = raster[3, :, :].values.astype(float)
            
            # Print value ranges for verification
            for idx_name, idx_data in indices.items():
                valid_data = idx_data[~np.isnan(idx_data)]
                if len(valid_data) > 0:
                    print(f"  {idx_name}: [{np.min(valid_data):.4f}, {np.max(valid_data):.4f}] ({len(valid_data)} valid pixels)")
            
        elif num_bands >= 6:
            # Raw spectral bands - calculate indices
            print("  → Detected 6+ band composite (raw spectral bands)")
            
            # Assuming bands are in order: B2, B3, B4, B5, B6, B7
            B4 = raster[2, :, :].values.astype(float)  # Red
            B5 = raster[3, :, :].values.astype(float)  # NIR
            B6 = raster[4, :, :].values.astype(float)  # SWIR1
            B7 = raster[5, :, :].values.astype(float)  # SWIR2
            
            print(f"  B4 (Red): [{np.nanmin(B4):.4f}, {np.nanmax(B4):.4f}]")
            print(f"  B5 (NIR): [{np.nanmin(B5):.4f}, {np.nanmax(B5):.4f}]")
            
            # NDVI = (NIR - Red) / (NIR + Red)
            with np.errstate(divide='ignore', invalid='ignore'):
                ndvi = (B5 - B4) / (B5 + B4)
                ndvi[~np.isfinite(ndvi)] = np.nan
            
            # NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
            with np.errstate(divide='ignore', invalid='ignore'):
                ndbi = (B6 - B5) / (B6 + B5)
                ndbi[~np.isfinite(ndbi)] = np.nan
            
            # NBR = (NIR - SWIR2) / (NIR + SWIR2)
            with np.errstate(divide='ignore', invalid='ignore'):
                nbr = (B5 - B7) / (B5 + B7)
                nbr[~np.isfinite(nbr)] = np.nan
            
            # NDMI = (NIR - SWIR1) / (NIR + SWIR1)
            with np.errstate(divide='ignore', invalid='ignore'):
                ndmi = (B5 - B6) / (B5 + B6)
                ndmi[~np.isfinite(ndmi)] = np.nan
            
            indices['NDVI'] = ndvi
            indices['NDBI'] = ndbi
            indices['NBR'] = nbr
            indices['NDMI'] = ndmi
            
            print("  ✓ Calculated all 4 indices")
        else:
            print(f"  ⚠ Unexpected number of bands: {num_bands}")
            print(f"     Expected 4 (pre-calculated) or 6+ (raw bands)")
    
    except Exception as e:
        print(f"  ✗ Error extracting indices: {e}")
        import traceback
        traceback.print_exc()
    
    return indices


def create_index_maps(landsat_files, boundaries_gdf, undp_gdf, output_dir, include_undp=True):
    """
    Create before/after/recovery maps for all Landsat indices.
    
    Parameters:
        include_undp: If True, plot UNDP locations. If False, create clean map.
    """
    print("\n" + "="*80)
    if include_undp:
        print("CREATING LANDSAT MAPS WITH UNDP INTERVENTIONS")
    else:
        print("CREATING LANDSAT MAPS (CLEAN VERSION)")
    print("="*80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find files for three periods
    before_file = None
    after_file = None
    recovery_file = None
    
    for date, file in landsat_files:
        if date == PRE_PERIOD:
            before_file = file
        if date == POST_PERIOD:
            after_file = file
        if date == RECOVERY_PERIOD:
            recovery_file = file
    
    if not before_file or not after_file or not recovery_file:
        print("\n⚠ Could not find February, April, and September 2025 files")
        print("Available months:")
        for date, file in landsat_files:
            print(f"  - {date.strftime('%Y-%m')}: {file.name}")
        return
    
    print(f"\nBefore (Feb):     {before_file.name}")
    print(f"After (Apr):      {after_file.name}")
    print(f"Recovery (Sep):   {recovery_file.name}")
    
    # Get Myanmar bbox
    bounds = boundaries_gdf.total_bounds
    
    if (bounds[2] - bounds[0]) < 5 or (bounds[3] - bounds[1]) < 5:
        print(f"\n⚠ Detected small boundary extent, using full Myanmar coordinates")
        bbox = (92.0, 9.5, 101.5, 28.5)
    else:
        buffer = 0.5
        bbox = (bounds[0] - buffer, bounds[1] - buffer, 
                bounds[2] + buffer, bounds[3] + buffer)
    
    print(f"\nShowing Myanmar region:")
    print(f"  Lon: {bbox[0]:.2f}° to {bbox[2]:.2f}°")
    print(f"  Lat: {bbox[1]:.2f}° to {bbox[3]:.2f}°")
    
    # Load rasters
    print("\nLoading Landsat data...")
    try:
        raster_before = load_and_clip_landsat(before_file, bbox)
        raster_after = load_and_clip_landsat(after_file, bbox)
        raster_recovery = load_and_clip_landsat(recovery_file, bbox)
        
        print(f"✓ Loaded before:   {raster_before.shape}")
        print(f"✓ Loaded after:    {raster_after.shape}")
        print(f"✓ Loaded recovery: {raster_recovery.shape}")
    except Exception as e:
        print(f"\n✗ Error loading rasters: {e}")
        return
    
    # Mask to boundaries
    print("\nMasking to Myanmar boundaries...")
    raster_before = mask_raster_to_boundaries(raster_before, boundaries_gdf)
    raster_after = mask_raster_to_boundaries(raster_after, boundaries_gdf)
    raster_recovery = mask_raster_to_boundaries(raster_recovery, boundaries_gdf)
    print("✓ Masked all rasters")
    
    # Calculate indices
    print("\nExtracting spectral indices...")
    indices_before = calculate_indices(raster_before)
    indices_after = calculate_indices(raster_after)
    indices_recovery = calculate_indices(raster_recovery)
    
    if not indices_before:
        print("✗ Failed to extract indices - check band configuration")
        return
    
    print(f"✓ Successfully extracted {len(indices_before)} indices")
    
    # Report coverage statistics
    print("\nData coverage by period:")
    total_pixels = raster_before.shape[1] * raster_before.shape[2]
    for period_name, indices_data in [('February', indices_before), 
                                       ('April', indices_after), 
                                       ('September', indices_recovery)]:
        # Use NDVI as reference for coverage
        valid_pixels = np.sum(~np.isnan(indices_data['NDVI']))
        coverage_pct = (valid_pixels / total_pixels) * 100
        print(f"  {period_name}: {valid_pixels:,} valid pixels ({coverage_pct:.1f}% coverage)")
    
    # Define index properties
    index_configs = {
        'NDVI': {
            'title': 'Vegetation Cover (NDVI)',
            'cmap': 'RdYlGn',  # Red-Yellow-Green
            'label': 'NDVI Value',
            'vmin': -0.2,
            'vmax': 0.8
        },
        'NDBI': {
            'title': 'Built-up Development (NDBI)',
            'cmap': 'RdYlBu_r',  # Red-Yellow-Blue reversed (red = more built-up)
            'label': 'NDBI Value',
            'vmin': -0.5,
            'vmax': 0.5
        },
        'NBR': {
            'title': 'Burn/Bare Ground Index (NBR)',
            'cmap': 'RdYlGn',  # Red-Yellow-Green
            'label': 'NBR Value',
            'vmin': -0.5,
            'vmax': 0.8
        },
        'NDMI': {
            'title': 'Moisture Index (NDMI)',
            'cmap': 'RdYlBu',  # Red-Yellow-Blue (blue = more moisture)
            'label': 'NDMI Value',
            'vmin': -0.5,
            'vmax': 0.5
        }
    }
    
    # Create maps for each index
    for index_name, config in index_configs.items():
        if index_name not in indices_before:
            continue
        
        print(f"\n📍 Creating map: {config['title']}")
        
        # Create figure with 3 panels
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        
        periods = [
            (indices_before[index_name], 'BEFORE Earthquake\nFebruary 2025', axes[0]),
            (indices_after[index_name], 'AFTER Earthquake\nApril 2025', axes[1]),
            (indices_recovery[index_name], 'RECOVERY\nSeptember 2025', axes[2])
        ]
        
        # Plot each period
        for idx, (data, title, ax) in enumerate(periods):
            # Plot the index values
            im = ax.imshow(
                data,
                cmap=config['cmap'],
                vmin=config['vmin'],
                vmax=config['vmax'],
                extent=[bbox[0], bbox[2], bbox[1], bbox[3]],
                aspect='auto'
            )
            
            # Overlay Myanmar boundaries
            boundaries_gdf.boundary.plot(ax=ax, edgecolor='white', linewidth=0.5, alpha=0.8)
            
            # Add UNDP interventions ONLY on the Recovery panel (idx == 2)
            if include_undp and undp_gdf is not None and idx == 2:
                # Get unique intervention types and colors
                intervention_types = undp_gdf['intervention_type'].unique()
                
                for itype in intervention_types:
                    itype_data = undp_gdf[undp_gdf['intervention_type'] == itype]
                    if len(itype_data) > 0:
                        color = itype_data.iloc[0]['intervention_color']
                        itype_data.plot(
                            ax=ax,
                            color=color,
                            markersize=30,
                            alpha=0.8,
                            edgecolor='black',
                            linewidth=0.5,
                            zorder=10
                        )
            
            # Format
            ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
            ax.set_xlabel('Longitude', fontsize=11)
            ax.set_ylabel('Latitude', fontsize=11)
            ax.set_xlim(bbox[0], bbox[2])
            ax.set_ylim(bbox[1], bbox[3])
            
            # Add note on Recovery panel if interventions are shown
            if include_undp and undp_gdf is not None and idx == 2:
                ax.text(0.5, 1.06, '(UNDP interventions shown)', 
                       transform=ax.transAxes, ha='center', 
                       fontsize=9, style='italic', color='darkred')
        
        # Overall title
        fig.suptitle(
            f'2025 Myanmar Earthquake Impact & Recovery\nMarch 28, 2025',
            fontsize=16,
            fontweight='bold',
            y=0.99
        )
        
        # Add subtitle explaining coverage
        fig.text(0.5, 0.95, 
                'Note: White areas indicate cloud cover or missing satellite observations',
                ha='center', fontsize=10, style='italic', color='gray')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=axes, orientation='horizontal', 
                           fraction=0.046, pad=0.04, aspect=40)
        cbar.set_label(config['label'], fontsize=12, fontweight='bold')
        
        # Add legend if UNDP data included
        if include_undp and undp_gdf is not None:
            legend_elements = []
            for itype in intervention_types:
                itype_data = undp_gdf[undp_gdf['intervention_type'] == itype]
                if len(itype_data) > 0:
                    color = itype_data.iloc[0]['intervention_color']
                    legend_elements.append(
                        Line2D([0], [0], marker='o', color='w', 
                               markerfacecolor=color, markersize=10,
                               markeredgecolor='black', markeredgewidth=0.5,
                               label=itype)
                    )
            
            fig.legend(handles=legend_elements, loc='lower center',
                      bbox_to_anchor=(0.5, -0.05), ncol=len(legend_elements),
                      fontsize=10, frameon=True)
        
        plt.tight_layout()
        
        # Save
        suffix = '_with_interventions' if include_undp else '_clean'
        map_file = output_dir / f'{index_name.lower()}_earthquake_2025{suffix}.png'
        plt.savefig(map_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {map_file.name}")
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Landsat visualization for Myanmar 2025 earthquake"
    )
    
    parser.add_argument(
        "--landsat-dir",
        type=str,
        default="./data/landsat_composites",
        help="Directory with Landsat composite .tif files"
    )
    
    parser.add_argument(
        "--boundaries",
        type=str,
        default="./data/boundaries/processed/mmr_admin1.geojson",
        help="GeoJSON file with Myanmar boundaries"
    )
    
    parser.add_argument(
        "--undp-interventions",
        type=str,
        default=None,
        help="GeoJSON file with UNDP intervention locations (optional)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/landsat_visualizations",
        help="Output directory for visualizations"
    )
    
    args = parser.parse_args()
    
    if not RASTER_OK:
        print("\n✗ Missing required packages")
        print("Install with: pip install rioxarray xarray rasterio")
        return
    
    print("="*80)
    print("  MYANMAR 2025 EARTHQUAKE LANDSAT VISUALIZATION")
    print("  March 28, 2025")
    print("="*80)
    
    # Find Landsat files
    start = datetime(2025, 1, 1)
    end = datetime(2025, 10, 31)
    
    landsat_files = find_landsat_files(args.landsat_dir, start, end)
    
    if not landsat_files:
        print("\n✗ No Landsat files found")
        print(f"Check directory: {args.landsat_dir}")
        return
    
    # Load boundaries
    print("\nLoading boundaries...")
    boundaries_file = Path(args.boundaries)
    
    if not boundaries_file.exists():
        print(f"\n✗ Boundaries file not found: {boundaries_file}")
        return
    
    boundaries_gdf = gpd.read_file(boundaries_file)
    print(f"✓ Loaded {len(boundaries_gdf)} regions from {boundaries_file.name}")
    
    # Load UNDP interventions
    undp_gdf = None
    if args.undp_interventions:
        undp_file = Path(args.undp_interventions)
        if undp_file.exists():
            undp_gdf = gpd.read_file(undp_file)
            print(f"✓ Loaded {len(undp_gdf)} UNDP interventions from {undp_file.name}")
        else:
            print(f"\n⚠ UNDP interventions file not found: {undp_file}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # 1. Clean maps (without UNDP locations)
    print("\n[1/2] Creating clean Landsat index maps...")
    create_index_maps(landsat_files, boundaries_gdf, undp_gdf, 
                     output_dir, include_undp=False)
    
    # 2. Maps with UNDP interventions (if available)
    if undp_gdf is not None:
        print("\n[2/2] Creating maps with UNDP interventions...")
        create_index_maps(landsat_files, boundaries_gdf, undp_gdf, 
                         output_dir, include_undp=True)
    else:
        print("\n[2/2] Skipping UNDP intervention maps (no data)")
    
    print("\n" + "="*80)
    print("ALL VISUALIZATIONS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("\nCLEAN MAPS (no interventions):")
    print("  - ndvi_earthquake_2025_clean.png")
    print("  - ndbi_earthquake_2025_clean.png")
    print("  - nbr_earthquake_2025_clean.png")
    print("  - ndmi_earthquake_2025_clean.png")
    
    if undp_gdf is not None:
        print("\nWITH UNDP INTERVENTIONS:")
        print("  - ndvi_earthquake_2025_with_interventions.png")
        print("  - ndbi_earthquake_2025_with_interventions.png")
        print("  - nbr_earthquake_2025_with_interventions.png")
        print("  - ndmi_earthquake_2025_with_interventions.png")
    
    print("\n" + "="*80)
    print("NOTES:")
    print("="*80)
    print("✓ All indices calculated from Landsat spectral bands")
    print("✓ Values masked to Myanmar boundaries only")
    print("✓ March 2025 excluded (earthquake month)")
    print("✓ UNDP interventions shown only on Recovery panel (September)")
    print("")
    print("⚠ DATA COVERAGE NOTES:")
    print("  - Patchy coverage is due to cloud masking in Landsat imagery")
    print("  - February typically has limited coverage (cloudy season)")
    print("  - April and September have better but still partial coverage")
    print("  - White/blank areas = clouds or no valid satellite observations")
    print("")
    print("INDEX INTERPRETATIONS:")
    print("✓ NDVI: Vegetation cover (green = healthy vegetation)")
    print("✓ NDBI: Built-up areas (red = more built-up)")
    print("✓ NBR: Burn/bare ground (green = healthier)")
    print("✓ NDMI: Moisture content (blue = more moisture)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
