#!/usr/bin/env python3
"""
Comprehensive Landsat Visualization - Myanmar 2025 Earthquake
Multi-Month Composites for Better Coverage
=============================================================

Creates spatial visualizations using multi-month averages to reduce cloud gaps:
- Pre-period: January-February
- Post-immediate: April-June  
- Recovery: July-September

Usage:
    python visualize_landsat_multimonth.py \
        --landsat-dir ./data/landsat_composites \
        --boundaries ./data/myanmar_boundaries/mmr_admin1.geojson \
        --undp-interventions ./data/undp/undp_interventions.geojson \
        --output-dir ./outputs/landsat_visualizations_multimonth
"""

import os
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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


def find_landsat_files(landsat_dir):
    """Find all Landsat monthly composite files in directory."""
    landsat_dir = Path(landsat_dir)
    
    tif_files = list(landsat_dir.glob("LANDSAT_*.tif"))
    
    print(f"\nFound {len(tif_files)} Landsat composite files")
    
    if len(tif_files) == 0:
        print("\n⚠ No Landsat .tif files found!")
        return {}
    
    files_by_month = {}
    
    for f in tif_files:
        # Parse LANDSAT_2025-01.tif format
        parts = f.stem.split('_')
        if len(parts) >= 2:
            date_str = parts[1]  # "2025-01"
            try:
                year, month = date_str.split('-')
                month_num = int(month)
                files_by_month[month_num] = f
            except:
                continue
    
    return files_by_month


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
    """Mask raster to only show values within boundaries."""
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


def extract_indices(raster):
    """
    Extract spectral indices from 4-band Landsat composite.
    Assumes bands: NDVI, NDBI, NBR, NDMI
    """
    indices = {}
    
    try:
        if raster.shape[0] == 4:
            indices['NDVI'] = raster[0, :, :].values.astype(float)
            indices['NDBI'] = raster[1, :, :].values.astype(float)
            indices['NBR'] = raster[2, :, :].values.astype(float)
            indices['NDMI'] = raster[3, :, :].values.astype(float)
        else:
            print(f"  ⚠ Unexpected number of bands: {raster.shape[0]}")
    
    except Exception as e:
        print(f"  ✗ Error extracting indices: {e}")
    
    return indices


def create_multimonth_composite(files_by_month, months, bbox, boundaries_gdf):
    """
    Create composite by averaging multiple months.
    
    Parameters:
        files_by_month: dict mapping month number to file path
        months: list of month numbers to average
        bbox: bounding box
        boundaries_gdf: boundaries for masking
    """
    print(f"  Compositing months: {months}")
    
    all_indices = {idx: [] for idx in ['NDVI', 'NDBI', 'NBR', 'NDMI']}
    
    for month in months:
        if month not in files_by_month:
            print(f"    ⚠ Month {month} not found, skipping")
            continue
        
        file = files_by_month[month]
        print(f"    Loading {file.name}...")
        
        try:
            raster = load_and_clip_landsat(file, bbox)
            raster = mask_raster_to_boundaries(raster, boundaries_gdf)
            indices = extract_indices(raster)
            
            for idx_name, idx_data in indices.items():
                all_indices[idx_name].append(idx_data)
        
        except Exception as e:
            print(f"    ✗ Error loading {file.name}: {e}")
            continue
    
    # Average across months (ignoring NaNs)
    composite_indices = {}
    for idx_name, idx_list in all_indices.items():
        if len(idx_list) > 0:
            # Stack and compute nanmean
            stacked = np.stack(idx_list, axis=0)
            composite_indices[idx_name] = np.nanmean(stacked, axis=0)
            
            # Report coverage
            valid_pixels = np.sum(~np.isnan(composite_indices[idx_name]))
            print(f"    {idx_name}: {valid_pixels:,} valid pixels")
        else:
            composite_indices[idx_name] = None
    
    return composite_indices


def create_index_maps(files_by_month, boundaries_gdf, undp_gdf, output_dir, include_undp=True):
    """
    Create before/after/recovery maps using multi-month composites.
    """
    print("\n" + "="*80)
    if include_undp:
        print("CREATING LANDSAT MAPS WITH UNDP INTERVENTIONS")
    else:
        print("CREATING LANDSAT MAPS (CLEAN VERSION)")
    print("="*80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get Myanmar bbox
    bounds = boundaries_gdf.total_bounds
    
    if (bounds[2] - bounds[0]) < 5 or (bounds[3] - bounds[1]) < 5:
        bbox = (92.0, 9.5, 101.5, 28.5)
    else:
        buffer = 0.5
        bbox = (bounds[0] - buffer, bounds[1] - buffer, 
                bounds[2] + buffer, bounds[3] + buffer)
    
    print(f"\nBounding box: [{bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f}]")
    
    # Define multi-month periods
    print("\n" + "-"*80)
    print("PRE-PERIOD: January-February 2025")
    print("-"*80)
    indices_pre = create_multimonth_composite(files_by_month, [1, 2], bbox, boundaries_gdf)
    
    print("\n" + "-"*80)
    print("POST-IMMEDIATE: April-June 2025")
    print("-"*80)
    indices_post = create_multimonth_composite(files_by_month, [4, 5, 6], bbox, boundaries_gdf)
    
    print("\n" + "-"*80)
    print("RECOVERY: July-September 2025")
    print("-"*80)
    indices_recovery = create_multimonth_composite(files_by_month, [7, 8, 9], bbox, boundaries_gdf)
    
    if not indices_pre or not indices_post or not indices_recovery:
        print("\n✗ Failed to create composites")
        return
    
    print("\n✓ All composites created successfully")
    
    # Define index properties
    index_configs = {
        'NDVI': {
            'title': 'Vegetation Cover (NDVI)',
            'cmap': 'RdYlGn',
            'label': 'NDVI Value',
            'vmin': -0.2,
            'vmax': 0.8
        },
        'NDBI': {
            'title': 'Built-up Development (NDBI)',
            'cmap': 'RdYlBu_r',
            'label': 'NDBI Value',
            'vmin': -0.5,
            'vmax': 0.5
        },
        'NBR': {
            'title': 'Burn/Bare Ground Index (NBR)',
            'cmap': 'RdYlGn',
            'label': 'NBR Value',
            'vmin': -0.5,
            'vmax': 0.8
        },
        'NDMI': {
            'title': 'Moisture Index (NDMI)',
            'cmap': 'RdYlBu',
            'label': 'NDMI Value',
            'vmin': -0.5,
            'vmax': 0.5
        }
    }
    
    # Create maps for each index
    print("\n" + "="*80)
    print("CREATING MAPS")
    print("="*80)
    
    for index_name, config in index_configs.items():
        if indices_pre[index_name] is None:
            print(f"\n⚠️  {index_name} data missing, skipping")
            continue
        
        print(f"\n📍 Creating map: {config['title']}")
        
        # Create figure with 3 panels
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        
        periods = [
            (indices_pre[index_name], 'BEFORE Earthquake\nJanuary-February 2025', axes[0], 0),
            (indices_post[index_name], 'AFTER Earthquake\nApril-June 2025', axes[1], 1),
            (indices_recovery[index_name], 'RECOVERY\nJuly-September 2025', axes[2], 2)
        ]
        
        # Plot each period
        for idx, (data, title, ax, panel_idx) in enumerate(periods):
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
            
            # Add UNDP interventions ONLY on Recovery panel (panel_idx == 2)
            if include_undp and undp_gdf is not None and panel_idx == 2:
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
        
        # Overall title
        fig.suptitle(
            f'2025 Myanmar Earthquake Impact & Recovery\nMarch 28, 2025',
            fontsize=16,
            fontweight='bold',
            y=0.98
        )
        
        # Add colorbar BELOW the maps
        # Create axis for colorbar below the plots
        cbar_ax = fig.add_axes([0.15, 0.08, 0.70, 0.02])  # [left, bottom, width, height]
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(config['label'], fontsize=12, fontweight='bold')
        
        # Add legend VERTICALLY on the right side if UNDP data included
        if include_undp and undp_gdf is not None:
            legend_elements = []
            intervention_types = undp_gdf['intervention_type'].unique()
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
            
            # Position legend on the right side, vertically
            fig.legend(handles=legend_elements, 
                      loc='center right',
                      bbox_to_anchor=(1.0, 0.5),
                      fontsize=10, 
                      frameon=True,
                      title='UNDP Interventions',
                      title_fontsize=11)
        
        plt.tight_layout(rect=[0, 0.12, 0.88, 0.96])  # Leave space on right for legend
        
        # Save
        suffix = '_with_interventions' if include_undp else '_clean'
        map_file = output_dir / f'{index_name.lower()}_earthquake_2025_multimonth{suffix}.png'
        plt.savefig(map_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {map_file.name}")
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Landsat visualization with multi-month composites"
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
        default="./data/myanmar_boundaries/mmr_admin1.geojson",
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
        default="./outputs/landsat_visualizations_multimonth",
        help="Output directory for visualizations"
    )
    
    args = parser.parse_args()
    
    if not RASTER_OK:
        print("\n✗ Missing required packages")
        print("Install with: pip install rioxarray xarray rasterio")
        return
    
    print("="*80)
    print("  MYANMAR 2025 EARTHQUAKE LANDSAT VISUALIZATION")
    print("  Multi-Month Composites for Better Coverage")
    print("  March 28, 2025")
    print("="*80)
    
    # Find Landsat files
    files_by_month = find_landsat_files(args.landsat_dir)
    
    if not files_by_month:
        print("\n✗ No Landsat files found")
        print(f"Check directory: {args.landsat_dir}")
        return
    
    print(f"Available months: {sorted(files_by_month.keys())}")
    
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
    
    # 1. Clean maps
    print("\n[1/2] Creating clean Landsat index maps...")
    create_index_maps(files_by_month, boundaries_gdf, undp_gdf, 
                     output_dir, include_undp=False)
    
    # 2. Maps with UNDP interventions
    if undp_gdf is not None:
        print("\n[2/2] Creating maps with UNDP interventions...")
        create_index_maps(files_by_month, boundaries_gdf, undp_gdf, 
                         output_dir, include_undp=True)
    else:
        print("\n[2/2] Skipping UNDP intervention maps (no data)")
    
    print("\n" + "="*80)
    print("ALL VISUALIZATIONS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("\nCLEAN MAPS:")
    print("  - ndvi_earthquake_2025_multimonth_clean.png")
    print("  - ndbi_earthquake_2025_multimonth_clean.png")
    print("  - nbr_earthquake_2025_multimonth_clean.png")
    print("  - ndmi_earthquake_2025_multimonth_clean.png")
    
    if undp_gdf is not None:
        print("\nWITH UNDP INTERVENTIONS:")
        print("  - ndvi_earthquake_2025_multimonth_with_interventions.png")
        print("  - ndbi_earthquake_2025_multimonth_with_interventions.png")
        print("  - nbr_earthquake_2025_multimonth_with_interventions.png")
        print("  - ndmi_earthquake_2025_multimonth_with_interventions.png")
    
    print("\n" + "="*80)
    print("COMPOSITE PERIODS:")
    print("="*80)
    print("✓ Pre-period: January-February (before earthquake)")
    print("✓ Post-immediate: April-June (immediate aftermath)")
    print("✓ Recovery: July-September (long-term recovery)")
    print("✓ March excluded (earthquake month)")
    print("✓ Multi-month averaging reduces cloud gaps")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
