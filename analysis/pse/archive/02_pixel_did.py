#!/usr/bin/env python3
"""
analysis/pse/08_pixel_did_rasters.py
====================================
Pixel-Level Difference-in-Differences Using Existing Raster Data

This script uses the rasters already downloaded via your pipeline to run
a proper spatial DiD comparing areas near UNDP interventions vs far areas.

METHODOLOGY:
============
1. Load existing rasters (VH, VV, NDVI from Sentinel-1/2)
2. Create treatment zones around debris/intervention sites
3. Extract pixel values for each zone
4. Compare: 
   - TREATMENT: Pixels within buffer of intervention sites
   - CONTROL: Pixels far from intervention sites (same governorate)
5. Run DiD: Y_it = α_i + γ_t + β(Near × Post) + ε_it

DATA REQUIREMENTS:
==================
1. Raster files in data/pse/rasters/ or data/sentinel/
2. debris_sites.csv with coordinates
3. Gaza boundaries shapefile

Usage:
    python analysis/pse/08_pixel_did_rasters.py
    python analysis/pse/08_pixel_did_rasters.py --buffer 500  # meters
"""

import argparse
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
    from rasterio.mask import mask
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("⚠ rasterio not installed. Run: pip install rasterio --break-system-packages")

try:
    import geopandas as gpd
    from shapely.geometry import Point, box
    from shapely.ops import unary_union
    HAS_GEO = True
except ImportError:
    HAS_GEO = False
    print("⚠ geopandas not installed")

try:
    from rasterstats import zonal_stats
    HAS_RASTERSTATS = True
except ImportError:
    HAS_RASTERSTATS = False
    print("⚠ rasterstats not installed. Run: pip install rasterstats --break-system-packages")

try:
    import statsmodels.api as sm
    HAS_SM = True
except ImportError:
    HAS_SM = False

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs/pse")
FIGURES_DIR = OUTPUT_DIR / "figures" / "pixel_did"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Key dates
CONFLICT_START = pd.Timestamp("2023-10-07")
DEBRIS_START = pd.Timestamp("2024-10-01")

# Buffer distances (meters) - will test multiple
BUFFER_DISTANCES = [250, 500, 1000]

# Possible raster locations (check all)
RASTER_DIRS = [
    DATA_DIR / "pse" / "rasters",  # Your actual location
]

# Gaza bounding box
GAZA_BBOX = box(34.2, 31.2, 34.6, 31.6)

# Path to digitized UNDP sites
UNDP_SITES_FILE = DATA_DIR / "pse" / "external" / "undp_sites_digitized.csv"

# =============================================================================
# LOAD UNDP INTERVENTION SITES FROM CSV
# =============================================================================
def load_undp_sites():
    """Load digitized UNDP intervention coordinates from CSV."""
    
    # Check multiple possible paths
    possible_paths = [
        UNDP_SITES_FILE,
        Path("data/pse/external/undp_sites_digitized.csv"),
        Path("../data/pse/external/undp_sites_digitized.csv"),
    ]
    
    for path in possible_paths:
        if path.exists():
            df = pd.read_csv(path)
            print(f"  Loaded {len(df)} UNDP sites from {path}")
            return df
    
    print("  ⚠ UNDP sites CSV not found, using fallback hardcoded sites")
    return None


def get_debris_sites():
    """Get debris/infrastructure sites for DiD analysis."""
    
    # Try to load from digitized CSV first
    undp_df = load_undp_sites()
    
    if undp_df is not None:
        # Filter for debris and infrastructure interventions
        debris_types = ['Debris', 'SWM', 'Shelter']
        debris_df = undp_df[undp_df['intervention_type'].isin(debris_types)].copy()
        
        # Convert to list of tuples for compatibility
        sites = []
        for _, row in debris_df.iterrows():
            # Assign governorate based on latitude
            lat = row['latitude']
            if lat >= 31.50:
                gov = 'North Gaza'
            elif lat >= 31.45:
                gov = 'Gaza City'
            elif lat >= 31.35:
                gov = 'Deir al-Balah'
            elif lat >= 31.28:
                gov = 'Khan Younis'
            else:
                gov = 'Rafah'
            
            site_type = 'crusher' if 'Crusher' in row['subtype'] else 'road'
            sites.append((row['latitude'], row['longitude'], site_type, gov))
        
        print(f"  Using {len(sites)} digitized debris/infrastructure sites")
        return sites
    
    # Fallback to hardcoded if CSV not available
    return DEBRIS_SITES_FALLBACK


# Fallback hardcoded sites (in case CSV not available)
DEBRIS_SITES_FALLBACK = [
    # BEIT LAHIA / NORTH
    (31.545, 34.495, 'road', 'North Gaza'),
    (31.540, 34.505, 'road', 'North Gaza'),
    (31.548, 34.500, 'road', 'North Gaza'),
    (31.542, 34.510, 'crusher', 'North Gaza'),
    
    # GAZA CITY (heavy concentration)
    (31.520, 34.455, 'road', 'Gaza City'),
    (31.515, 34.460, 'road', 'Gaza City'),
    (31.510, 34.450, 'road', 'Gaza City'),
    (31.508, 34.455, 'road', 'Gaza City'),
    (31.512, 34.448, 'road', 'Gaza City'),
    (31.505, 34.445, 'road', 'Gaza City'),
    (31.518, 34.465, 'road', 'Gaza City'),
    (31.522, 34.470, 'road', 'Gaza City'),
    (31.525, 34.472, 'road', 'Gaza City'),
    (31.502, 34.442, 'road', 'Gaza City'),
    (31.498, 34.438, 'crusher', 'Gaza City'),
    
    # MIDDLE ZONE / DEIR AL-BALAH
    (31.430, 34.360, 'road', 'Deir al-Balah'),
    (31.425, 34.355, 'road', 'Deir al-Balah'),
    (31.420, 34.350, 'road', 'Deir al-Balah'),
    (31.435, 34.365, 'road', 'Deir al-Balah'),
    (31.415, 34.345, 'road', 'Deir al-Balah'),
    (31.440, 34.370, 'crusher', 'Deir al-Balah'),
    
    # KHAN YOUNIS (significant cluster)
    (31.350, 34.310, 'road', 'Khan Younis'),
    (31.345, 34.305, 'road', 'Khan Younis'),
    (31.340, 34.300, 'road', 'Khan Younis'),
    (31.355, 34.315, 'road', 'Khan Younis'),
    (31.348, 34.308, 'road', 'Khan Younis'),
    (31.352, 34.312, 'road', 'Khan Younis'),
    (31.358, 34.318, 'road', 'Khan Younis'),
    (31.342, 34.302, 'road', 'Khan Younis'),
    (31.360, 34.320, 'road', 'Khan Younis'),
    (31.338, 34.298, 'crusher', 'Khan Younis'),
    
    # RAFAH
    (31.285, 34.250, 'road', 'Rafah'),
    (31.280, 34.248, 'road', 'Rafah'),
    (31.275, 34.245, 'road', 'Rafah'),
    (31.290, 34.255, 'road', 'Rafah'),
    (31.295, 34.260, 'crusher', 'Rafah'),
]


def find_raster_files():
    """Find all available raster files."""
    rasters = {
        'sar_vh': [],      # SAR VH backscatter
        'sar_vv': [],      # SAR VV backscatter  
        'ndvi': [],        # Vegetation index
        'ndbi': [],        # Built-up index (you have this!)
        'bsi': [],         # Bare soil index (you have this!)
        'viirs': [],       # Nightlights
    }
    
    # Your exact file patterns: indicator_pse_YYYYMM.tif
    patterns = {
        'sar_vh': ['sar_vh_pse_*.tif'],
        'sar_vv': ['sar_vv_pse_*.tif', 'sar_pse_*.tif'],  # sar_pse might be VV
        'ndvi': ['ndvi_pse_*.tif'],
        'ndbi': ['ndbi_pse_*.tif'],
        'bsi': ['bsi_pse_*.tif'],
        'viirs': ['viirs_pse_*.tif'],
    }
    
    for raster_dir in RASTER_DIRS:
        if raster_dir.exists():
            print(f"  Checking {raster_dir}...")
            for indicator, pats in patterns.items():
                for pat in pats:
                    files = list(raster_dir.glob(pat))
                    rasters[indicator].extend(files)
    
    # Remove duplicates
    for k in rasters:
        rasters[k] = sorted(list(set(rasters[k])))
    
    return rasters


def parse_date_from_filename(filename):
    """Extract date from raster filename like ndbi_pse_202411.tif."""
    name = Path(filename).stem
    
    # Your pattern: indicator_pse_YYYYMM
    match = re.search(r'_(\d{6})$', name)  # Match YYYYMM at end
    if match:
        yyyymm = match.group(1)
        year = int(yyyymm[:4])
        month = int(yyyymm[4:6])
        return datetime(year, month, 1)
    
    # Fallback patterns
    patterns = [
        r'(\d{4})(\d{2})(\d{2})',  # YYYYMMDD
        r'(\d{4})-(\d{2})-(\d{2})',  # YYYY-MM-DD
        r'(\d{4})_(\d{2})_(\d{2})',  # YYYY_MM_DD
    ]
    
    for pat in patterns:
        match = re.search(pat, name)
        if match:
            groups = match.groups()
            if len(groups) == 3:
                return datetime(int(groups[0]), int(groups[1]), int(groups[2]))
    
    return None


def create_treatment_zones(debris_sites_df, buffer_m=500):
    """
    Create treatment and control zones from debris site locations.
    
    Returns GeoDataFrame with:
    - Treatment zones: buffer_m meters around each site
    - Control zones: Rest of Gaza (excluding treatment buffers)
    """
    # Create points from debris sites
    geometry = [Point(lon, lat) for lat, lon, _, _ in 
                zip(debris_sites_df['lat'], debris_sites_df['lon'], 
                    debris_sites_df['type'], debris_sites_df['governorate'])]
    
    sites_gdf = gpd.GeoDataFrame(debris_sites_df, geometry=geometry, crs="EPSG:4326")
    
    # Convert to meters for buffering (UTM zone 36N for Gaza)
    sites_utm = sites_gdf.to_crs("EPSG:32636")
    
    # Create buffers
    sites_utm['buffer'] = sites_utm.geometry.buffer(buffer_m)
    
    # Merge all treatment buffers
    treatment_zone = unary_union(sites_utm['buffer'])
    
    # Convert back to WGS84
    treatment_gdf = gpd.GeoDataFrame(
        {'zone': ['treatment']}, 
        geometry=[treatment_zone], 
        crs="EPSG:32636"
    ).to_crs("EPSG:4326")
    
    # Create Gaza boundary
    gaza_gdf = gpd.GeoDataFrame(
        {'zone': ['gaza']},
        geometry=[GAZA_BBOX],
        crs="EPSG:4326"
    )
    
    # Control zone = Gaza minus treatment
    control_zone = gaza_gdf.geometry[0].difference(treatment_gdf.geometry[0])
    control_gdf = gpd.GeoDataFrame(
        {'zone': ['control']},
        geometry=[control_zone],
        crs="EPSG:4326"
    )
    
    return treatment_gdf, control_gdf, sites_gdf


def extract_zonal_stats_from_raster(raster_path, treatment_gdf, control_gdf):
    """Extract mean values for treatment and control zones from a raster."""
    
    if not HAS_RASTERSTATS:
        return None, None
    
    try:
        # Treatment zone stats
        treatment_stats = zonal_stats(
            treatment_gdf, 
            str(raster_path), 
            stats=['mean', 'std', 'count'],
            nodata=-9999
        )
        
        # Control zone stats
        control_stats = zonal_stats(
            control_gdf, 
            str(raster_path), 
            stats=['mean', 'std', 'count'],
            nodata=-9999
        )
        
        treatment_mean = treatment_stats[0]['mean'] if treatment_stats and treatment_stats[0] else np.nan
        control_mean = control_stats[0]['mean'] if control_stats and control_stats[0] else np.nan
        
        return treatment_mean, control_mean
        
    except Exception as e:
        print(f"    Error extracting stats: {e}")
        return None, None


def build_pixel_panel(raster_files, treatment_gdf, control_gdf):
    """
    Build panel dataset from raster time series.
    
    Returns DataFrame with:
    - period_start: date
    - treatment_mean: mean value in treatment zone
    - control_mean: mean value in control zone
    - outcome: indicator name
    """
    results = []
    
    for indicator, files in raster_files.items():
        if not files:
            continue
            
        print(f"\n  Processing {indicator}: {len(files)} files...")
        
        for f in sorted(files):
            date = parse_date_from_filename(f)
            if date is None:
                continue
            
            treatment_val, control_val = extract_zonal_stats_from_raster(
                f, treatment_gdf, control_gdf
            )
            
            if treatment_val is not None and control_val is not None:
                results.append({
                    'period_start': pd.Timestamp(date),
                    'outcome': indicator,
                    'treatment_mean': treatment_val,
                    'control_mean': control_val,
                    'file': str(f),
                })
                print(f"    {date.strftime('%Y-%m')}: T={treatment_val:.4f}, C={control_val:.4f}")
    
    return pd.DataFrame(results)


def run_pixel_did(panel_df, outcome):
    """
    Run DiD regression on zone-level panel.
    
    Y_zt = α_z + γ_t + β(Treatment_z × Post_t) + ε_zt
    """
    # Reshape to long format (zone × time)
    df = panel_df[panel_df['outcome'] == outcome].copy()
    
    if len(df) < 5:
        return None
    
    # Create long format
    long_data = []
    for _, row in df.iterrows():
        # Treatment zone observation
        long_data.append({
            'period_start': row['period_start'],
            'zone': 'treatment',
            'value': row['treatment_mean'],
            'treated': 1,
        })
        # Control zone observation
        long_data.append({
            'period_start': row['period_start'],
            'zone': 'control', 
            'value': row['control_mean'],
            'treated': 0,
        })
    
    long_df = pd.DataFrame(long_data)
    long_df['post'] = (long_df['period_start'] >= DEBRIS_START).astype(int)
    long_df['treated_post'] = long_df['treated'] * long_df['post']
    
    # Drop NaN
    long_df = long_df.dropna(subset=['value'])
    
    if len(long_df) < 10:
        return None
    
    # Regression
    y = long_df['value'].astype(float).values
    X = long_df[['treated_post', 'treated', 'post']].astype(float)
    X = sm.add_constant(X)
    
    try:
        model = sm.OLS(y, X).fit(cov_type='HC1')
        
        return {
            'outcome': outcome,
            'did_coef': model.params['treated_post'],
            'did_se': model.bse['treated_post'],
            'did_pval': model.pvalues['treated_post'],
            'n_obs': len(long_df),
            'r2': model.rsquared,
            'pre_treatment_mean': long_df[(long_df['treated']==1) & (long_df['post']==0)]['value'].mean(),
            'post_treatment_mean': long_df[(long_df['treated']==1) & (long_df['post']==1)]['value'].mean(),
            'pre_control_mean': long_df[(long_df['treated']==0) & (long_df['post']==0)]['value'].mean(),
            'post_control_mean': long_df[(long_df['treated']==0) & (long_df['post']==1)]['value'].mean(),
        }
    except Exception as e:
        print(f"    Regression error: {e}")
        return None


def create_pixel_did_plot(panel_df, outcome, result, buffer_m, output_path):
    """Create visualization of pixel-level DiD results."""
    
    df = panel_df[panel_df['outcome'] == outcome].copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # =========================================================================
    # Panel 1: Treatment vs Control Time Series
    # =========================================================================
    ax1 = axes[0, 0]
    
    df_sorted = df.sort_values('period_start')
    
    ax1.plot(df_sorted['period_start'], df_sorted['treatment_mean'], 
             'b-o', linewidth=2, markersize=6, label=f'Treatment (<{buffer_m}m from site)')
    ax1.plot(df_sorted['period_start'], df_sorted['control_mean'], 
             'r--s', linewidth=2, markersize=6, label=f'Control (>{buffer_m}m from site)')
    
    ax1.axvline(x=DEBRIS_START, color='green', linewidth=2.5, linestyle='--',
                label='Debris Removal Start')
    ax1.axvline(x=CONFLICT_START, color='black', linewidth=2, label='Oct 7, 2023')
    
    # Shading
    if len(df_sorted) > 0:
        max_date = df_sorted['period_start'].max()
        ax1.axvspan(CONFLICT_START, DEBRIS_START, alpha=0.1, color='orange')
        ax1.axvspan(DEBRIS_START, max_date, alpha=0.1, color='green')
    
    ax1.set_ylabel(outcome.upper(), fontsize=11)
    ax1.set_title('Treatment vs Control Zones Over Time', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # =========================================================================
    # Panel 2: Gap (Treatment - Control)
    # =========================================================================
    ax2 = axes[0, 1]
    
    df_sorted['gap'] = df_sorted['treatment_mean'] - df_sorted['control_mean']
    
    colors = ['green' if d >= DEBRIS_START else 'gray' for d in df_sorted['period_start']]
    ax2.bar(df_sorted['period_start'], df_sorted['gap'], color=colors, alpha=0.7, width=20)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.axvline(x=DEBRIS_START, color='green', linewidth=2.5, linestyle='--')
    
    # Average gap pre/post
    pre_gap = df_sorted[df_sorted['period_start'] < DEBRIS_START]['gap'].mean()
    post_gap = df_sorted[df_sorted['period_start'] >= DEBRIS_START]['gap'].mean()
    
    ax2.axhline(y=pre_gap, color='gray', linestyle=':', linewidth=2, label=f'Pre avg: {pre_gap:.4f}')
    ax2.axhline(y=post_gap, color='green', linestyle=':', linewidth=2, label=f'Post avg: {post_gap:.4f}')
    
    ax2.set_ylabel(f'Gap ({outcome.upper()})', fontsize=11)
    ax2.set_title('Treatment - Control Gap', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # =========================================================================
    # Panel 3: DiD Visualization
    # =========================================================================
    ax3 = axes[1, 0]
    
    if result:
        x = [0, 1]
        ax3.plot(x, [result['pre_treatment_mean'], result['post_treatment_mean']], 
                 'b-o', linewidth=3, markersize=12, label='Treatment Zone')
        ax3.plot(x, [result['pre_control_mean'], result['post_control_mean']], 
                 'r--s', linewidth=3, markersize=12, label='Control Zone')
        
        # Counterfactual
        cf = result['pre_treatment_mean'] + (result['post_control_mean'] - result['pre_control_mean'])
        ax3.plot([1], [cf], 'b^', markersize=15, markerfacecolor='none',
                 markeredgewidth=2, label='Counterfactual')
        
        # DiD arrow
        ax3.annotate('', xy=(1.05, result['post_treatment_mean']), xytext=(1.05, cf),
                    arrowprops=dict(arrowstyle='<->', color='green', lw=2))
        ax3.annotate(f'DiD={result["did_coef"]:.4f}', xy=(1.1, (result['post_treatment_mean'] + cf)/2),
                    fontsize=11, fontweight='bold', color='green')
    
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(['Pre-Intervention\n(Oct 2023 - Sep 2024)', 'Post-Intervention\n(Oct 2024+)'])
    ax3.set_ylabel(outcome.upper(), fontsize=11)
    ax3.set_title('Difference-in-Differences', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # =========================================================================
    # Panel 4: Results Table
    # =========================================================================
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    if result:
        sig = '***' if result['did_pval'] < 0.01 else '**' if result['did_pval'] < 0.05 else '*' if result['did_pval'] < 0.1 else ''
        interpretation = "✓ POSITIVE EFFECT" if result['did_coef'] > 0 else "⚠ NEGATIVE EFFECT"
        interp_color = '#90EE90' if result['did_coef'] > 0 else '#FFB6C1'
        
        table_data = [
            ['PIXEL-LEVEL DiD', ''],
            ['Outcome', outcome.upper()],
            ['Buffer distance', f'{buffer_m}m'],
            ['', ''],
            ['DiD COEFFICIENT', ''],
            ['β (Near × Post)', f"{result['did_coef']:.6f}{sig}"],
            ['SE (robust)', f"{result['did_se']:.6f}"],
            ['p-value', f"{result['did_pval']:.4f}"],
            ['', ''],
            ['MODEL FIT', ''],
            ['R²', f"{result['r2']:.3f}"],
            ['N observations', f"{result['n_obs']}"],
            ['', ''],
            ['INTERPRETATION', interpretation],
        ]
        
        table = ax4.table(cellText=table_data, loc='center', cellLoc='left',
                         colWidths=[0.5, 0.45])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.7)
        
        for i, row in enumerate(table_data):
            if 'INTERPRETATION' in row[0]:
                table[(i, 0)].set_facecolor(interp_color)
                table[(i, 1)].set_facecolor(interp_color)
            elif row[0] in ['PIXEL-LEVEL DiD', 'DiD COEFFICIENT', 'MODEL FIT']:
                table[(i, 0)].set_facecolor('#E0E0E0')
                table[(i, 1)].set_facecolor('#E0E0E0')
    
    # Title
    sig_str = f"{result['did_coef']:.4f}" if result else "N/A"
    fig.suptitle(
        f'Pixel-Level DiD: Debris Removal Effect on {outcome.upper()}\n'
        f'Buffer: {buffer_m}m | DiD: {sig_str}',
        fontsize=13, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def use_existing_panel_data():
    """
    Alternative: Use the existing panel_data.csv and run zone-level analysis
    based on treatment intensity rather than pixel extraction.
    """
    panel_path = OUTPUT_DIR / "panel_data.csv"
    
    if not panel_path.exists():
        return None
    
    print("\n--- Using Existing Panel Data ---")
    
    panel = pd.read_csv(panel_path)
    panel['period_start'] = pd.to_datetime(panel['period_start'])
    
    # Filter to Gaza
    gaza = panel[panel['unit_id'].str.startswith('PS02')].copy()
    
    # Aggregate to monthly Gaza-wide
    numeric_cols = gaza.select_dtypes(include=[np.number]).columns.tolist()
    monthly = gaza.groupby('period_start')[numeric_cols].mean().reset_index()
    
    return monthly


def main():
    parser = argparse.ArgumentParser(description='Pixel-level DiD using existing rasters')
    parser.add_argument('--buffer', type=int, default=500, help='Buffer distance in meters')
    parser.add_argument('--use-panel', action='store_true', help='Use existing panel data instead of rasters')
    args = parser.parse_args()
    
    print("=" * 70)
    print("PIXEL-LEVEL DIFFERENCE-IN-DIFFERENCES")
    print("=" * 70)
    print(f"""
    Method: Compare areas NEAR debris removal sites vs FAR from sites
    Buffer: {args.buffer}m
    Treatment: Pixels within {args.buffer}m of debris clearing
    Control: Pixels >{args.buffer}m from any site
    """)
    
    # Load debris sites from digitized CSV or fallback
    debris_sites = get_debris_sites()
    debris_df = pd.DataFrame(debris_sites, columns=['lat', 'lon', 'type', 'governorate'])
    print(f"\n  Debris sites: {len(debris_df)} locations")
    print(f"    - Roads cleared: {len(debris_df[debris_df['type']=='road'])}")
    print(f"    - Crusher sites: {len(debris_df[debris_df['type']=='crusher'])}")
    
    # Check for required packages
    if not HAS_GEO or not HAS_RASTERSTATS:
        print("\n⚠ Missing required packages!")
        print("  Run: pip install geopandas rasterstats rasterio --break-system-packages")
        
        # Fallback to panel data
        print("\n--- Falling back to existing panel data ---")
        monthly = use_existing_panel_data()
        if monthly is not None:
            print(f"  Loaded {len(monthly)} monthly observations")
            # Save for reference
            monthly.to_csv(TABLES_DIR / 'gaza_monthly_aggregates.csv', index=False)
            print(f"  Saved: {TABLES_DIR}/gaza_monthly_aggregates.csv")
        return
    
    # Create treatment zones
    print(f"\n--- Creating Treatment Zones ({args.buffer}m buffer) ---")
    treatment_gdf, control_gdf, sites_gdf = create_treatment_zones(debris_df, args.buffer)
    
    treatment_area = treatment_gdf.to_crs("EPSG:32636").area[0] / 1e6  # km²
    control_area = control_gdf.to_crs("EPSG:32636").area[0] / 1e6
    
    print(f"  Treatment zone area: {treatment_area:.2f} km²")
    print(f"  Control zone area: {control_area:.2f} km²")
    
    # Find raster files
    print("\n--- Searching for Raster Files ---")
    raster_files = find_raster_files()
    
    total_files = sum(len(f) for f in raster_files.values())
    print(f"\n  Found {total_files} raster files:")
    for ind, files in raster_files.items():
        print(f"    {ind}: {len(files)} files")
    
    if total_files == 0:
        print("\n⚠ No raster files found!")
        print("  Expected locations:")
        for d in RASTER_DIRS:
            print(f"    - {d}")
        print("\n  Falling back to existing panel data...")
        
        monthly = use_existing_panel_data()
        if monthly is not None:
            print(f"  Using panel with {len(monthly)} observations")
        return
    
    # Build pixel panel
    print("\n--- Extracting Zonal Statistics ---")
    panel_df = build_pixel_panel(raster_files, treatment_gdf, control_gdf)
    
    if len(panel_df) == 0:
        print("\n⚠ Could not extract data from rasters")
        return
    
    print(f"\n  Built panel: {len(panel_df)} observations")
    panel_df.to_csv(TABLES_DIR / f'pixel_panel_{args.buffer}m.csv', index=False)
    
    # Run DiD for each outcome
    print("\n--- Running DiD Regressions ---")
    results = []
    
    for outcome in panel_df['outcome'].unique():
        print(f"\n  {outcome}...")
        result = run_pixel_did(panel_df, outcome)
        
        if result:
            sig = '***' if result['did_pval'] < 0.01 else '**' if result['did_pval'] < 0.05 else '*' if result['did_pval'] < 0.1 else ''
            print(f"    DiD: {result['did_coef']:.6f}{sig} (p={result['did_pval']:.4f})")
            
            # Create plot
            create_pixel_did_plot(
                panel_df, outcome, result, args.buffer,
                FIGURES_DIR / f'pixel_did_{outcome}_{args.buffer}m.png'
            )
            print(f"    ✓ pixel_did_{outcome}_{args.buffer}m.png")
            
            results.append(result)
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(TABLES_DIR / f'pixel_did_results_{args.buffer}m.csv', index=False)
        print(f"\n✓ Results: {TABLES_DIR}/pixel_did_results_{args.buffer}m.csv")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
    Buffer distance: {args.buffer}m
    Treatment area: {treatment_area:.2f} km²
    Control area: {control_area:.2f} km²
    
    DiD estimates effect of being NEAR a debris removal site
    vs being FAR from any site, before vs after intervention.
    
    Positive coefficient = areas near sites improved MORE than
    areas far from sites after intervention started.
    """)
    
    print(f"\n{'='*70}")
    print("✓ PIXEL-LEVEL DiD COMPLETE")
    print(f"  Figures: {FIGURES_DIR}/pixel_did_*.png")
    print(f"  Tables: {TABLES_DIR}/pixel_did_results_*.csv")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
