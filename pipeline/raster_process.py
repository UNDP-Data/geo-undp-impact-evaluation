#!/usr/bin/env python3
"""
pipeline/raster_process.py
==========================
Process local raster files (GeoTIFFs) for:
1. Zonal statistics extraction (admin-level panel data)
2. Pixel-level visualizations (maps)

Usage:
    python -m pipeline.raster_process --config config/gaza.yaml
    python -m pipeline.raster_process --config config/myanmar.yaml --viz-only
    
Expects raster files in:
    data/{country_code}/rasters/viirs_{country_code}_YYYYMM.tif
    data/{country_code}/rasters/ndvi_{country_code}_YYYYMM.tif
    data/{country_code}/rasters/sar_{country_code}_YYYYMM.tif
"""

import os
import glob
import re
import yaml
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm, Normalize, LinearSegmentedColormap
from matplotlib.patches import Patch
import rasterio
from rasterio.mask import mask
from rasterstats import zonal_stats
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# Custom colormaps
NIGHTLIGHT_CMAP = LinearSegmentedColormap.from_list(
    'nightlights', ['#000000', '#1a1a00', '#4d4d00', '#808000', '#b3b300', '#ffff00', '#ffff66']
)

NDVI_CMAP = LinearSegmentedColormap.from_list(
    'vegetation', ['#8B4513', '#D2691E', '#F0E68C', '#ADFF2F', '#228B22', '#006400']
)


class RasterProcessor:
    """Process raster files for impact evaluation."""
    
    def __init__(self, config_path: str):
        """Initialize with config."""
        self.config = load_config(config_path)
        
        self.country_name = self.config.get('country_name', 'Unknown')
        self.country_code = self.config.get('country_code', 'XXX').lower()
        
        # Directories
        self.raster_dir = Path(f"data/{self.country_code}/rasters")
        self.output_dir = Path(f"outputs/{self.country_code}")
        self.figures_dir = self.output_dir / "figures" / "rasters"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Load boundaries
        boundary_path = self.config['boundaries']['file_path']
        self.gdf = gpd.read_file(boundary_path)
        self.id_col = self.config['boundaries']['id_column']
        self.name_col = self.config['boundaries']['name_column']
        
        # Key dates
        self.disaster_date = pd.Timestamp(self.config['events']['disaster_date'])
        self.disaster_type = self.config['events'].get('disaster_type', 'disaster')
        
        # Detect Gaza bounds for zooming (if Palestine)
        self.gaza_bounds = self._detect_gaza_bounds()
        
        print(f"\n{'='*60}")
        print(f"RASTER PROCESSOR: {self.country_name}")
        print(f"{'='*60}")
        print(f"Raster directory: {self.raster_dir}")
        print(f"Boundaries: {len(self.gdf)} admin units")
        if self.gaza_bounds:
            print(f"Gaza zoom enabled: {self.gaza_bounds}")
    
    def _detect_gaza_bounds(self):
        """Detect Gaza-specific bounds for zooming."""
        # Check if this is Palestine
        if self.country_code.lower() not in ['pse', 'ps']:
            return None
        
        # Try to identify Gaza districts
        gaza_gdf = None
        
        if 'adm1_pcode' in self.gdf.columns:
            # Gaza = PS02, West Bank = PS01
            gaza_gdf = self.gdf[self.gdf['adm1_pcode'].str.startswith('PS02', na=False)]
        elif 'adm1_name' in self.gdf.columns:
            # Try by name
            gaza_gdf = self.gdf[self.gdf['adm1_name'].str.lower().str.contains('gaza', na=False)]
        
        if gaza_gdf is not None and len(gaza_gdf) > 0:
            bounds = gaza_gdf.total_bounds  # [minx, miny, maxx, maxy]
            # Small buffer for tight Gaza zoom
            buffer = 0.03  # ~3km - tight zoom on Gaza
            return [bounds[0] - buffer, bounds[1] - buffer, 
                    bounds[2] + buffer, bounds[3] + buffer]
        
        return None
    
    def find_rasters(self) -> dict:
        """Find available raster files."""
        rasters = {
            'viirs': [],
            'ndvi': [],
            'ndbi': [],
            'bsi': [],
            'sar': [],
            'sar_vh': []
        }
        
        if not self.raster_dir.exists():
            print(f"\n⚠ Raster directory not found: {self.raster_dir}")
            print(f"  Run: python -m pipeline.raster_export --config {self.config}")
            return rasters
        
        for tif in sorted(self.raster_dir.glob("*.tif")):
            name = tif.stem.lower()
            if 'viirs' in name or 'nightlight' in name or 'dnb' in name:
                rasters['viirs'].append(tif)
            elif 'ndbi' in name or 'builtup' in name or 'built_up' in name:
                rasters['ndbi'].append(tif)
            elif 'bsi' in name or 'baresoil' in name or 'bare_soil' in name:
                rasters['bsi'].append(tif)
            elif 'ndvi' in name or 'vegetation' in name:
                rasters['ndvi'].append(tif)
            elif 'sar_vh' in name:
                rasters['sar_vh'].append(tif)
            elif 'sar' in name or 'sentinel' in name:
                rasters['sar'].append(tif)
        
        print(f"\nFound rasters:")
        for dtype, files in rasters.items():
            if files:
                print(f"  {dtype}: {len(files)} files")
        
        return rasters
    
    def parse_date_from_filename(self, filename: str) -> datetime:
        """Extract date from filename."""
        name = Path(filename).stem
        
        # Try YYYYMM format
        match = re.search(r'(\d{6})(?!\d)', name)
        if match:
            date_str = match.group(1)
            return datetime.strptime(date_str, '%Y%m')
        
        # Try YYYY-MM format
        match = re.search(r'(\d{4})-(\d{2})', name)
        if match:
            return datetime(int(match.group(1)), int(match.group(2)), 1)
        
        # Try YYYYMMDD format
        match = re.search(r'(\d{8})', name)
        if match:
            return datetime.strptime(match.group(1)[:6], '%Y%m')
        
        return None
    
    def extract_zonal_stats(self, raster_path: Path, stat: str = 'mean') -> pd.DataFrame:
        """Extract zonal statistics for each admin unit."""
        stats = zonal_stats(
            self.gdf, 
            str(raster_path), 
            stats=[stat],
            all_touched=True
        )
        
        results = []
        for idx, s in enumerate(stats):
            results.append({
                'unit_id': self.gdf.iloc[idx][self.id_col],
                'unit_name': self.gdf.iloc[idx][self.name_col],
                'value': s[stat] if s and s[stat] is not None else np.nan
            })
        
        return pd.DataFrame(results)
    
    def process_zonal_stats(self, rasters: dict) -> pd.DataFrame:
        """Process all rasters and create panel dataset."""
        print("\n--- Extracting Zonal Statistics ---")
        
        all_data = []
        
        for dtype, files in rasters.items():
            if not files:
                continue
            
            for raster_path in files:
                date = self.parse_date_from_filename(str(raster_path))
                if date is None:
                    print(f"  ⚠ Could not parse date: {raster_path.name}")
                    continue
                
                print(f"  Processing {dtype}: {raster_path.name}...", end=" ")
                
                try:
                    df = self.extract_zonal_stats(raster_path)
                    df['period_start'] = date
                    df['dataset'] = dtype
                    all_data.append(df)
                    print(f"✓")
                except Exception as e:
                    print(f"✗ {e}")
        
        if not all_data:
            print("  No data extracted")
            return pd.DataFrame()
        
        # Combine and pivot
        combined = pd.concat(all_data, ignore_index=True)
        
        # Pivot to wide format
        panel = combined.pivot_table(
            index=['unit_id', 'unit_name', 'period_start'],
            columns='dataset',
            values='value'
        ).reset_index()
        
        # Rename columns
        panel.columns.name = None
        col_map = {
            'viirs': 'nightlights_raster', 
            'ndvi': 'ndvi_raster', 
            'ndbi': 'ndbi_raster',
            'bsi': 'bsi_raster',
            'sar': 'sar_raster',
            'sar_vh': 'sar_vh_raster'
        }
        panel = panel.rename(columns=col_map)
        
        # Save
        output_path = self.output_dir / "raster_panel.csv"
        panel.to_csv(output_path, index=False)
        print(f"\n✓ Saved: {output_path}")
        print(f"  Rows: {len(panel)}")
        
        return panel
    
    def create_raster_map(self, raster_path: Path, title: str, output_name: str,
                          cmap=None, log_scale: bool = False, vmin: float = None, 
                          vmax: float = None, clip_to_bounds: bool = True,
                          mask_zeros: bool = True):
        """Create a pixel-level raster map."""
        
        with rasterio.open(raster_path) as src:
            # Read data
            data = src.read(1)
            transform = src.transform
            bounds = src.bounds
            
            # Mask nodata
            nodata = src.nodata
            if nodata is not None:
                data = np.ma.masked_equal(data, nodata)
            data = np.ma.masked_invalid(data)
            if mask_zeros:
                data = np.ma.masked_less_equal(data, 0)  # Mask zeros for log scale
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Determine color scale
        valid_data = data.compressed()
        if len(valid_data) == 0:
            print(f"  ⚠ No valid data in {raster_path.name}")
            plt.close()
            return
        
        if vmin is None:
            vmin = np.percentile(valid_data, 2)
        if vmax is None:
            vmax = np.percentile(valid_data, 98)
        
        # Plot raster
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        
        if log_scale and vmin > 0:
            norm = LogNorm(vmin=max(vmin, 0.01), vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
        
        im = ax.imshow(data, extent=extent, cmap=cmap or 'viridis', 
                       norm=norm, interpolation='nearest', origin='upper')
        
        # Overlay boundaries
        self.gdf.boundary.plot(ax=ax, color='white', linewidth=0.5, alpha=0.7)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(title, fontsize=11)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        # Set bounds - use Gaza zoom if available
        if clip_to_bounds:
            if self.gaza_bounds:
                ax.set_xlim(self.gaza_bounds[0], self.gaza_bounds[2])
                ax.set_ylim(self.gaza_bounds[1], self.gaza_bounds[3])
            else:
                gdf_bounds = self.gdf.total_bounds
                ax.set_xlim(gdf_bounds[0] - 0.05, gdf_bounds[2] + 0.05)
                ax.set_ylim(gdf_bounds[1] - 0.05, gdf_bounds[3] + 0.05)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / output_name, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  ✓ {output_name}")
    
    def create_comparison_maps(self, rasters: dict, dataset: str = 'viirs', zoom_to_gaza: bool = True):
        """Create before/after comparison maps for a dataset."""
        files = rasters.get(dataset, [])
        if len(files) < 2:
            print(f"  ⚠ Need at least 2 {dataset} files for comparison")
            return
        
        # Sort by date
        file_dates = [(f, self.parse_date_from_filename(str(f))) for f in files]
        file_dates = [(f, d) for f, d in file_dates if d is not None]
        file_dates.sort(key=lambda x: x[1])
        
        # Select 3 time points for comparison:
        # 1. Pre-disaster (prefer Sep 2023 for seasonality, else latest pre-disaster)
        # 2. Post-disaster Year 1 (prefer Sep 2024)
        # 3. Post-disaster Year 2 or latest (prefer Sep 2025 or Oct 2025)
        
        selected = []
        
        # Target months for same-season comparison
        target_months = [
            (2023, 9, 'pre'),      # Pre-disaster baseline
            (2024, 9, 'post'),     # 1 year post
            (2025, 9, 'latest'),   # 2 years post (or Oct 2025)
            (2025, 10, 'latest'),  # Fallback latest
        ]
        
        # Find best matches for each target
        for target_year, target_month, period in target_months:
            if len(selected) >= 3:
                break
            
            # First try exact match
            exact_match = next(
                ((f, d) for f, d in file_dates 
                 if d.year == target_year and d.month == target_month),
                None
            )
            
            if exact_match:
                f, d = exact_match
                # Verify period is correct
                actual_period = 'pre' if d < self.disaster_date else ('post' if len(selected) < 2 else 'latest')
                if not any(d == s[1] for s in selected):  # Avoid duplicates
                    selected.append((f, d, actual_period))
        
        # Fallback: if we don't have 3 files, use original logic
        if len(selected) < 2:
            selected = []
            # Get latest pre-disaster
            pre_files = [(f, d) for f, d in file_dates if d < self.disaster_date]
            if pre_files:
                f, d = pre_files[-1]  # Latest pre-disaster
                selected.append((f, d, 'pre'))
            
            # Get first post-disaster
            post_files = [(f, d) for f, d in file_dates if d >= self.disaster_date]
            if post_files:
                f, d = post_files[0]  # First post-disaster
                selected.append((f, d, 'post'))
            
            # Get latest
            if file_dates:
                f, d = file_dates[-1]
                if not selected or d > selected[-1][1]:
                    selected.append((f, d, 'latest'))
        
        selected = selected[:3]  # Max 3
        
        if len(selected) < 2:
            return
        
        print(f"\n--- {dataset.upper()} Comparison Maps ---")
        print(f"    Dates: {[s[1].strftime('%Y-%m') for s in selected]}")
        
        # Dataset-specific settings
        dataset_configs = {
            'viirs': {
                'cmap': NIGHTLIGHT_CMAP,
                'log_scale': True,
                'label': 'Nightlights',
                'mask_zeros': True,
                'vmin_pct': 2, 'vmax_pct': 98
            },
            'ndvi': {
                'cmap': NDVI_CMAP,
                'log_scale': False,
                'label': 'Vegetation (NDVI)',
                'mask_zeros': False,
                'vmin_pct': 2, 'vmax_pct': 98
            },
            'ndbi': {
                'cmap': 'RdYlBu_r',  # Red = high built-up, Blue = low
                'log_scale': False,
                'label': 'Built-up Index (NDBI)',
                'mask_zeros': False,
                'vmin_pct': 2, 'vmax_pct': 98
            },
            'bsi': {
                'cmap': 'YlOrBr',  # Brown = bare soil/debris
                'log_scale': False,
                'label': 'Bare Soil Index (BSI)',
                'mask_zeros': False,
                'vmin_pct': 2, 'vmax_pct': 98
            },
            'sar': {
                'cmap': 'viridis',
                'log_scale': False,
                'label': 'SAR Backscatter (dB)',
                'mask_zeros': False,
                'vmin_pct': 2, 'vmax_pct': 98
            },
            'sar_vh': {
                'cmap': 'inferno',  # Good for SAR
                'log_scale': False,
                'label': 'SAR VH Backscatter (dB)',
                'mask_zeros': False,
                'vmin_pct': 2, 'vmax_pct': 98
            },
        }
        
        cfg = dataset_configs.get(dataset, {
            'cmap': 'viridis', 'log_scale': False, 
            'label': dataset.upper(), 'mask_zeros': False,
            'vmin_pct': 2, 'vmax_pct': 98
        })
        
        # Determine common color scale
        all_data = []
        for f, d, period in selected:
            with rasterio.open(f) as src:
                data = src.read(1)
                # Don't filter zeros for indices that can be negative
                if cfg['mask_zeros']:
                    data = data[data > 0]
                else:
                    data = data[~np.isnan(data)]
                all_data.extend(data.flatten())
        
        if not all_data:
            return
        
        vmin = np.percentile(all_data, cfg['vmin_pct'])
        vmax = np.percentile(all_data, cfg['vmax_pct'])
        
        # Create multi-panel figure
        n_panels = len(selected)
        fig, axes = plt.subplots(1, n_panels, figsize=(5*n_panels + 2, 8))
        if n_panels == 1:
            axes = [axes]
        
        for ax, (raster_path, date, period) in zip(axes, selected):
            with rasterio.open(raster_path) as src:
                data = src.read(1)
                bounds = src.bounds
                
                # Masking depends on dataset
                if cfg['mask_zeros']:
                    data = np.ma.masked_less_equal(data, 0)
                data = np.ma.masked_invalid(data)
            
            extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
            
            if cfg['log_scale'] and vmin > 0:
                norm = LogNorm(vmin=max(vmin, 0.01), vmax=vmax)
            else:
                norm = Normalize(vmin=vmin, vmax=vmax)
            
            im = ax.imshow(data, extent=extent, cmap=cfg['cmap'], norm=norm, 
                          interpolation='nearest', origin='upper')
            
            # Overlay boundaries
            self.gdf.boundary.plot(ax=ax, color='white', linewidth=0.5, alpha=0.7)
            
            # Title with period info
            date_str = date.strftime('%Y-%m')
            if period == 'pre':
                title_color = 'green'
                period_label = f"Pre-{self.disaster_type.title()}"
            elif period == 'post':
                title_color = 'red'
                period_label = f"Post-{self.disaster_type.title()}"
            else:
                title_color = 'blue'
                period_label = "Latest"
            
            ax.set_title(f"{date_str}\n({period_label})", fontsize=12, 
                        fontweight='bold', color=title_color)
            ax.axis('off')
            
            # Set zoom bounds based on zoom_mode parameter
            if zoom_to_gaza and self.gaza_bounds:
                ax.set_xlim(self.gaza_bounds[0], self.gaza_bounds[2])
                ax.set_ylim(self.gaza_bounds[1], self.gaza_bounds[3])
            else:
                gdf_bounds = self.gdf.total_bounds
                ax.set_xlim(gdf_bounds[0] - 0.02, gdf_bounds[2] + 0.02)
                ax.set_ylim(gdf_bounds[1] - 0.02, gdf_bounds[3] + 0.02)
        
        # Colorbar
        cbar = fig.colorbar(im, ax=axes, orientation='horizontal', 
                           fraction=0.05, pad=0.08, aspect=40)
        cbar.set_label(cfg['label'], fontsize=12)
        
        # Title - indicate if zoomed
        short_name = self.country_name.split(' - ')[-1]
        title_suffix = " (Gaza)" if zoom_to_gaza and self.gaza_bounds else " (Full Region)"
        fig.suptitle(f"{short_name}: {cfg['label']}{title_suffix}", fontsize=14, fontweight='bold', y=0.95)
        
        # Filename - indicate zoom level
        suffix = "_gaza" if zoom_to_gaza and self.gaza_bounds else "_full"
        plt.savefig(self.figures_dir / f'raster_{dataset}_comparison{suffix}.png', 
                   dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  ✓ raster_{dataset}_comparison{suffix}.png")
    
    def create_time_series_plots(self, rasters: dict):
        """Create time series plots from raster data showing temporal evolution."""
        import matplotlib.patches as mpatches
        from matplotlib.lines import Line2D
        
        print("\n--- Creating Time Series Plots ---")
        
        # Load UNDP projects if available
        undp_df = self._load_undp_projects()
        
        # Dataset configurations for time series
        ts_configs = {
            'viirs': {
                'name': 'Nighttime Lights (VIIRS)',
                'line_color': 'darkblue',
                'fill_color': 'steelblue',
                'unit': 'nW/cm²/sr'
            },
            'ndvi': {
                'name': 'Vegetation Index (NDVI)',
                'line_color': 'forestgreen',
                'fill_color': 'mediumseagreen',
                'unit': 'Index'
            },
            'ndbi': {
                'name': 'Built-up Index (NDBI)',
                'line_color': 'saddlebrown',
                'fill_color': 'sienna',
                'unit': 'Index'
            },
            'bsi': {
                'name': 'Bare Soil Index (BSI)',
                'line_color': 'darkorange',
                'fill_color': 'orange',
                'unit': 'Index'
            },
            'sar': {
                'name': 'SAR Backscatter (VV)',
                'line_color': 'purple',
                'fill_color': 'mediumpurple',
                'unit': 'dB'
            },
            'sar_vh': {
                'name': 'SAR VH Backscatter',
                'line_color': 'indigo',
                'fill_color': 'slateblue',
                'unit': 'dB'
            },
        }
        
        # Identify Gaza and West Bank
        if 'adm2_pcode' in self.gdf.columns:
            id_col = 'adm2_pcode'
        elif 'adm1_pcode' in self.gdf.columns:
            id_col = 'adm1_pcode'
        else:
            id_col = self.id_col
        
        gdf_gaza = self.gdf[self.gdf[id_col].str.startswith('PS02', na=False)]
        gdf_westbank = self.gdf[self.gdf[id_col].str.startswith('PS01', na=False)]
        
        for dataset, files in rasters.items():
            if not files or dataset not in ts_configs:
                continue
            
            cfg = ts_configs[dataset]
            
            # Extract time series data
            ts_data = []
            for raster_path in sorted(files):
                date = self.parse_date_from_filename(str(raster_path))
                if date is None:
                    continue
                
                try:
                    # Gaza stats
                    gaza_stats = zonal_stats(gdf_gaza, str(raster_path), stats=['mean'], nodata=np.nan)
                    gaza_means = [s['mean'] for s in gaza_stats if s['mean'] is not None]
                    gaza_mean = np.mean(gaza_means) if gaza_means else np.nan
                    
                    # West Bank stats
                    wb_mean = np.nan
                    if len(gdf_westbank) > 0:
                        wb_stats = zonal_stats(gdf_westbank, str(raster_path), stats=['mean'], nodata=np.nan)
                        wb_means = [s['mean'] for s in wb_stats if s['mean'] is not None]
                        wb_mean = np.mean(wb_means) if wb_means else np.nan
                    
                    ts_data.append({
                        'date': date,
                        'gaza_mean': gaza_mean,
                        'westbank_mean': wb_mean
                    })
                except Exception as e:
                    print(f"    Warning: Could not process {raster_path.name}: {e}")
            
            if len(ts_data) < 2:
                print(f"  ✗ ts_raster_{dataset}.png (insufficient data)")
                continue
            
            # Create DataFrame
            df = pd.DataFrame(ts_data).sort_values('date')
            
            # Create plot
            self._create_ts_plot(df, dataset, cfg, undp_df)
            print(f"  ✓ ts_raster_{dataset}.png")
        
        # Save time series data
        self._save_raster_timeseries(rasters)
    
    def _load_undp_projects(self):
        """Load UNDP projects for timeline markers."""
        undp_paths = [
            Path(f"data/{self.country_code}/external/undp_projects.csv"),
            Path(f"data/{self.country_code}/external/pal_undp_projects_ml_classified.csv"),
        ]
        
        for path in undp_paths:
            if path.exists():
                try:
                    df = pd.read_csv(path)
                    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
                    if 'is_likely_umbrella' in df.columns:
                        df = df[df['is_likely_umbrella'] != True]
                    return df
                except:
                    pass
        return None
    
    def _create_ts_plot(self, df, dataset, cfg, undp_df):
        """Create a single time series plot with UNDP markers."""
        
        # Category colors for UNDP
        CATEGORY_COLORS = {
            'Education': '#3498DB',
            'Debris & Solid Waste': '#795548',
            'Water & Sanitation (WASH)': '#00BCD4',
            'Energy & Power': '#FFC107',
            'Health & Emergency': '#F44336',
            'Infrastructure & Housing': '#9C27B0',
            'Livelihoods & Economy': '#4CAF50',
            'Governance & Rule of Law': '#607D8B',
            'Social Inclusion': '#E91E63',
            'Environment': '#8BC34A',
            'Other / Unspecified': '#9E9E9E'
        }
        
        # Create figure
        fig = plt.figure(figsize=(16, 9))
        ax_main = fig.add_axes([0.08, 0.22, 0.85, 0.68])
        ax_strip = fig.add_axes([0.08, 0.12, 0.85, 0.06])
        
        # Plot Gaza
        ax_main.fill_between(df['date'], df['gaza_mean'], alpha=0.3, color=cfg['fill_color'])
        ax_main.plot(df['date'], df['gaza_mean'], color=cfg['line_color'], 
                     linewidth=2.5, marker='o', markersize=5, label='Gaza Strip')
        
        # Plot West Bank if available
        if df['westbank_mean'].notna().any():
            ax_main.plot(df['date'], df['westbank_mean'], color='gray', 
                        linewidth=2, linestyle='--', alpha=0.7, label='West Bank (Control)')
        
        # Pre/post means
        pre_data = df[df['date'] < self.disaster_date]['gaza_mean']
        post_data = df[df['date'] >= self.disaster_date]['gaza_mean']
        
        pre_mean = pre_data.mean() if len(pre_data) > 0 else np.nan
        post_mean = post_data.mean() if len(post_data) > 0 else np.nan
        
        if not pd.isna(pre_mean):
            ax_main.axhline(y=pre_mean, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
        if not pd.isna(post_mean):
            ax_main.axhline(y=post_mean, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
        
        # Conflict line
        ax_main.axvline(x=self.disaster_date, color='red', linestyle='--', linewidth=2.5)
        
        # Shade periods
        xmin, xmax = df['date'].min(), df['date'].max()
        if xmin < self.disaster_date:
            ax_main.axvspan(xmin, min(self.disaster_date, xmax), alpha=0.1, color='green')
        if xmax > self.disaster_date:
            ax_main.axvspan(max(self.disaster_date, xmin), xmax, alpha=0.08, color='red')
        
        # Conflict label
        y_top = ax_main.get_ylim()[1]
        ax_main.text(self.disaster_date, y_top * 0.95, f'Conflict\n{self.disaster_date.strftime("%b %d, %Y")}', 
                     ha='center', va='top', fontsize=9, fontweight='bold', color='red',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='red'))
        
        # Title with % change
        if pre_mean != 0 and not pd.isna(pre_mean) and not pd.isna(post_mean):
            pct_change = ((post_mean - pre_mean) / abs(pre_mean)) * 100
            title_suffix = f" ({pct_change:+.1f}% After Conflict)"
        else:
            title_suffix = ""
        
        ax_main.set_title(f'Gaza: {cfg["name"]} (Raster){title_suffix}', fontsize=14, fontweight='bold', pad=20)
        ax_main.set_ylabel(f'{cfg["name"]} ({cfg["unit"]})', fontsize=11)
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax_main.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax_main.set_xticklabels([])
        ax_main.grid(True, alpha=0.3)
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=cfg['line_color'], linewidth=2.5, marker='o', label='Gaza Strip'),
        ]
        if df['westbank_mean'].notna().any():
            legend_elements.append(Line2D([0], [0], color='gray', linewidth=2, linestyle='--', label='West Bank'))
        if not pd.isna(pre_mean):
            legend_elements.append(Line2D([0], [0], color='green', linestyle=':', linewidth=1.5, label=f'Pre-Conflict Mean: {pre_mean:.2f}'))
        if not pd.isna(post_mean):
            legend_elements.append(Line2D([0], [0], color='red', linestyle=':', linewidth=1.5, label=f'Post-Conflict Mean: {post_mean:.2f}'))
        
        ax_main.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        # === UNDP Strip ===
        ax_strip.set_xlim(ax_main.get_xlim())
        ax_strip.set_ylim(0, 1)
        
        if xmin < self.disaster_date:
            ax_strip.axvspan(xmin, min(self.disaster_date, xmax), alpha=0.15, color='green')
        if xmax > self.disaster_date:
            ax_strip.axvspan(max(self.disaster_date, xmin), xmax, alpha=0.1, color='red')
        
        ax_strip.axvline(x=self.disaster_date, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # UNDP markers
        categories_shown = set()
        if undp_df is not None:
            undp_in_range = undp_df[(undp_df['start_date'] >= xmin) & (undp_df['start_date'] <= xmax)]
            
            if len(undp_in_range) > 0:
                from collections import defaultdict
                monthly_markers = defaultdict(list)
                for _, row in undp_in_range.iterrows():
                    month_key = row['start_date'].to_period('M').to_timestamp()
                    monthly_markers[month_key].append(row['ml_category'])
                
                for month, categories in sorted(monthly_markers.items()):
                    for i, category in enumerate(categories[:3]):
                        color = CATEGORY_COLORS.get(category, '#9E9E9E')
                        y_pos = 0.5 + (i - len(categories)/2 + 0.5) * 0.25
                        y_pos = max(0.15, min(0.85, y_pos))
                        ax_strip.scatter([month], [y_pos], color=color, s=80, 
                                        marker='s', edgecolor='black', linewidth=0.5, zorder=5)
                        categories_shown.add(category)
        
        ax_strip.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax_strip.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax_strip.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax_strip.set_yticks([])
        ax_strip.set_ylabel('UNDP', fontsize=9, rotation=0, ha='right', va='center')
        ax_strip.spines['top'].set_visible(False)
        ax_strip.spines['right'].set_visible(False)
        ax_strip.spines['left'].set_visible(False)
        
        # Category legend
        if categories_shown:
            import matplotlib.patches as mpatches
            legend_patches = [mpatches.Patch(color=CATEGORY_COLORS.get(cat, '#9E9E9E'), label=cat[:20]) 
                              for cat in sorted(categories_shown)]
            fig.legend(handles=legend_patches, loc='lower center', 
                       bbox_to_anchor=(0.5, -0.02), ncol=min(5, len(legend_patches)), 
                       fontsize=7, frameon=False)
        
        plt.savefig(self.figures_dir / f'ts_raster_{dataset}.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _save_raster_timeseries(self, rasters: dict):
        """Save raster time series to CSV."""
        
        # Identify Gaza and West Bank
        if 'adm2_pcode' in self.gdf.columns:
            id_col = 'adm2_pcode'
        elif 'adm1_pcode' in self.gdf.columns:
            id_col = 'adm1_pcode'
        else:
            id_col = self.id_col
        
        gdf_gaza = self.gdf[self.gdf[id_col].str.startswith('PS02', na=False)]
        gdf_westbank = self.gdf[self.gdf[id_col].str.startswith('PS01', na=False)]
        
        all_data = []
        
        for dataset, files in rasters.items():
            if not files:
                continue
            
            for raster_path in sorted(files):
                date = self.parse_date_from_filename(str(raster_path))
                if date is None:
                    continue
                
                try:
                    gaza_stats = zonal_stats(gdf_gaza, str(raster_path), stats=['mean'], nodata=np.nan)
                    gaza_means = [s['mean'] for s in gaza_stats if s['mean'] is not None]
                    gaza_mean = np.mean(gaza_means) if gaza_means else np.nan
                    
                    wb_mean = np.nan
                    if len(gdf_westbank) > 0:
                        wb_stats = zonal_stats(gdf_westbank, str(raster_path), stats=['mean'], nodata=np.nan)
                        wb_means = [s['mean'] for s in wb_stats if s['mean'] is not None]
                        wb_mean = np.mean(wb_means) if wb_means else np.nan
                    
                    all_data.append({
                        'dataset': dataset,
                        'date': date,
                        'gaza_mean': gaza_mean,
                        'westbank_mean': wb_mean
                    })
                except:
                    pass
        
        if all_data:
            df = pd.DataFrame(all_data)
            df.to_csv(self.output_dir / 'raster_timeseries.csv', index=False)
            print(f"\n  Saved: raster_timeseries.csv")

    def run(self, extract_stats: bool = True, create_viz: bool = True):
        """Run full raster processing pipeline."""
        rasters = self.find_rasters()
        
        if not any(rasters.values()):
            print("\n⚠ No raster files found. Export them first:")
            print(f"   python -m pipeline.raster_export --config <your_config.yaml>")
            return
        
        if extract_stats:
            self.process_zonal_stats(rasters)
        
        if create_viz:
            print("\n--- Creating Visualizations ---")
            
            # Create comparison maps for all available datasets
            # Generate BOTH zoomed (Gaza) and full region versions
            for dataset in ['viirs', 'ndvi', 'ndbi', 'bsi', 'sar', 'sar_vh']:
                if rasters.get(dataset):
                    # Gaza-zoomed version
                    self.create_comparison_maps(rasters, dataset, zoom_to_gaza=True)
                    # Full region version (if we have Gaza bounds, also make full)
                    if self.gaza_bounds:
                        self.create_comparison_maps(rasters, dataset, zoom_to_gaza=False)
            
            # Individual maps for each file
            for dtype, files in rasters.items():
                for raster_path in files:
                    date = self.parse_date_from_filename(str(raster_path))
                    if date is None:
                        continue
                    
                    date_str = date.strftime('%Y%m')
                    
                    if dtype == 'viirs':
                        self.create_raster_map(
                            raster_path,
                            f"Nightlights - {date.strftime('%B %Y')}",
                            f"raster_viirs_{date_str}.png",
                            cmap=NIGHTLIGHT_CMAP,
                            log_scale=True
                        )
                    elif dtype == 'ndvi':
                        self.create_raster_map(
                            raster_path,
                            f"Vegetation (NDVI) - {date.strftime('%B %Y')}",
                            f"raster_ndvi_{date_str}.png",
                            cmap=NDVI_CMAP,
                            log_scale=False,
                            mask_zeros=False
                        )
                    elif dtype == 'ndbi':
                        self.create_raster_map(
                            raster_path,
                            f"Built-up Index (NDBI) - {date.strftime('%B %Y')}",
                            f"raster_ndbi_{date_str}.png",
                            cmap='RdYlBu_r',
                            log_scale=False,
                            mask_zeros=False
                        )
                    elif dtype == 'bsi':
                        self.create_raster_map(
                            raster_path,
                            f"Bare Soil Index (BSI) - {date.strftime('%B %Y')}",
                            f"raster_bsi_{date_str}.png",
                            cmap='YlOrBr',
                            log_scale=False,
                            mask_zeros=False
                        )
                    elif dtype == 'sar_vh':
                        self.create_raster_map(
                            raster_path,
                            f"SAR VH Backscatter - {date.strftime('%B %Y')}",
                            f"raster_sar_vh_{date_str}.png",
                            cmap='inferno',
                            log_scale=False,
                            mask_zeros=False
                        )
            
            # Create time series plots from raster data
            self.create_time_series_plots(rasters)
        
        print(f"\n{'='*60}")
        print(f"✓ Raster processing complete")
        print(f"  Figures: {self.figures_dir}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Process local raster files')
    parser.add_argument('--config', '-c', required=True, help='Path to country config YAML')
    parser.add_argument('--viz-only', action='store_true', help='Only create visualizations (skip stats)')
    parser.add_argument('--stats-only', action='store_true', help='Only extract statistics (skip viz)')
    
    args = parser.parse_args()
    
    processor = RasterProcessor(args.config)
    processor.run(
        extract_stats=not args.viz_only,
        create_viz=not args.stats_only
    )


if __name__ == "__main__":
    main()
