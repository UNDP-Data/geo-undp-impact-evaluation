#!/usr/bin/env python3
"""
analysis/pse/07_spatial_did_debris.py
=====================================
Spatial Difference-in-Differences Analysis for Debris Removal

METHODOLOGY:
============
Using the exact locations of UNDP debris removal interventions, we can run
a within-Gaza DiD comparing:
- TREATED: Pixels near debris clearing sites
- CONTROL: Pixels in same governorate but far from any site

This is stronger than cross-country SC because:
1. Controls for Gaza-wide shocks (conflict, weather, etc.)
2. Geographic proximity = better counterfactual
3. Can estimate precise effect sizes

DATA REQUIREMENTS:
==================
1. Debris site locations (from UNDP map → digitized to CSV)
2. Raster time series (VH, VV, NDVI at pixel level)
3. Gaza governorate boundaries

Usage:
    python analysis/pse/07_spatial_did_debris.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    import geopandas as gpd
    from shapely.geometry import Point
    from shapely.ops import unary_union
    HAS_GEO = True
except ImportError:
    HAS_GEO = False
    print("⚠ geopandas not installed. Run: pip install geopandas --break-system-packages")

try:
    import statsmodels.api as sm
    from linearmodels.panel import PanelOLS
    HAS_PANEL = True
except ImportError:
    HAS_PANEL = False

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs/pse")
FIGURES_DIR = OUTPUT_DIR / "figures" / "spatial_did"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Debris removal start date
DEBRIS_START = pd.Timestamp("2024-10-01")
CONFLICT_START = pd.Timestamp("2023-10-07")

# Treatment buffer distances (meters)
BUFFER_DISTANCES = [100, 250, 500]  # Test multiple radii

# Gaza bounding box (approximate)
GAZA_BOUNDS = {
    'min_lon': 34.2,
    'max_lon': 34.6,
    'min_lat': 31.2,
    'max_lat': 31.6,
}

# Governorate approximate centroids (for fallback if no shapefile)
GOVERNORATES = {
    'North Gaza': {'lat': 31.55, 'lon': 34.50},
    'Gaza City': {'lat': 31.50, 'lon': 34.45},
    'Deir al-Balah': {'lat': 31.42, 'lon': 34.35},
    'Khan Younis': {'lat': 31.35, 'lon': 34.30},
    'Rafah': {'lat': 31.28, 'lon': 34.25},
}

# =============================================================================
# DEBRIS SITE LOCATIONS (Digitized from UNDP Map)
# =============================================================================
# These are approximate coordinates extracted from the map image
# In production, these should come from UNDP's actual GIS data

DEBRIS_SITES = [
    # Format: (lat, lon, type, area)
    # NORTH GAZA / BEIT LAHIA / BEIT HANOUN (fewer sites - Coordination Area)
    (31.545, 34.495, 'road', 'North Gaza'),
    (31.540, 34.505, 'road', 'North Gaza'),
    (31.535, 34.515, 'road', 'North Gaza'),
    (31.550, 34.520, 'road', 'North Gaza'),
    
    # GAZA CITY (heavy concentration)
    (31.520, 34.455, 'road', 'Gaza City'),
    (31.515, 34.460, 'road', 'Gaza City'),
    (31.510, 34.450, 'road', 'Gaza City'),
    (31.508, 34.455, 'road', 'Gaza City'),
    (31.505, 34.448, 'road', 'Gaza City'),
    (31.512, 34.465, 'road', 'Gaza City'),
    (31.518, 34.470, 'road', 'Gaza City'),
    (31.522, 34.475, 'road', 'Gaza City'),
    (31.498, 34.445, 'road', 'Gaza City'),
    (31.502, 34.452, 'road', 'Gaza City'),
    (31.495, 34.440, 'crusher', 'Gaza City'),  # Crusher site
    
    # MIDDLE ZONE / DEIR AL-BALAH
    (31.430, 34.360, 'road', 'Deir al-Balah'),
    (31.425, 34.355, 'road', 'Deir al-Balah'),
    (31.420, 34.350, 'road', 'Deir al-Balah'),
    (31.415, 34.345, 'road', 'Deir al-Balah'),
    (31.435, 34.365, 'road', 'Deir al-Balah'),
    (31.440, 34.370, 'crusher', 'Deir al-Balah'),  # Crusher site
    
    # KHAN YOUNIS (significant cluster)
    (31.350, 34.310, 'road', 'Khan Younis'),
    (31.345, 34.305, 'road', 'Khan Younis'),
    (31.340, 34.300, 'road', 'Khan Younis'),
    (31.355, 34.315, 'road', 'Khan Younis'),
    (31.348, 34.308, 'road', 'Khan Younis'),
    (31.352, 34.312, 'road', 'Khan Younis'),
    (31.358, 34.318, 'road', 'Khan Younis'),
    (31.342, 34.302, 'road', 'Khan Younis'),
    (31.338, 34.298, 'crusher', 'Khan Younis'),  # Crusher site
    (31.360, 34.320, 'road', 'Khan Younis'),
    (31.365, 34.325, 'road', 'Khan Younis'),
    
    # RAFAH
    (31.285, 34.250, 'road', 'Rafah'),
    (31.280, 34.248, 'road', 'Rafah'),
    (31.275, 34.245, 'road', 'Rafah'),
    (31.290, 34.255, 'road', 'Rafah'),
    (31.295, 34.260, 'crusher', 'Rafah'),  # Crusher site
]


def create_debris_sites_csv():
    """Create CSV file with debris site locations."""
    df = pd.DataFrame(DEBRIS_SITES, columns=['lat', 'lon', 'type', 'governorate'])
    output_path = DATA_DIR / "pse" / "external" / "debris_sites.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Created debris sites CSV: {output_path}")
    print(f"  Total sites: {len(df)} ({len(df[df['type']=='road'])} roads, {len(df[df['type']=='crusher'])} crushers)")
    return df


def load_debris_sites():
    """Load debris site locations."""
    csv_path = DATA_DIR / "pse" / "external" / "debris_sites.csv"
    
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df)} debris sites from CSV")
    else:
        print("  Creating debris sites from embedded coordinates...")
        df = create_debris_sites_csv()
    
    return df


def load_panel_data():
    """Load the existing panel data with district-level observations."""
    panel_file = OUTPUT_DIR / "panel_data.csv"
    
    if not panel_file.exists():
        print(f"  ⚠ Panel data not found: {panel_file}")
        return None
    
    df = pd.read_csv(panel_file)
    df['period_start'] = pd.to_datetime(df['period_start'])
    df['unit_id'] = df['unit_id'].astype(str)
    
    # Filter to Gaza only
    gaza = df[df['unit_id'].str.startswith('PS02')].copy()
    
    print(f"  Loaded panel: {len(gaza)} obs, {gaza['unit_id'].nunique()} districts")
    
    return gaza


def assign_treatment_status(panel_df, debris_sites, district_mapping=None):
    """
    Assign treatment status to districts based on debris site presence.
    
    For district-level analysis:
    - Treated: Districts with debris sites
    - Control: Districts without debris sites (or fewer sites)
    """
    # Count sites per governorate
    sites_per_gov = debris_sites.groupby('governorate').size().to_dict()
    
    # Map unit_id to governorate
    # Gaza districts: PS0201 (North), PS0202 (Gaza), PS0203 (Deir al-Balah), 
    #                 PS0204 (Khan Younis), PS0205 (Rafah)
    unit_to_gov = {
        'PS0201': 'North Gaza',
        'PS0202': 'Gaza City', 
        'PS0203': 'Deir al-Balah',
        'PS0204': 'Khan Younis',
        'PS0205': 'Rafah',
    }
    
    panel_df = panel_df.copy()
    panel_df['governorate'] = panel_df['unit_id'].map(unit_to_gov)
    panel_df['n_debris_sites'] = panel_df['governorate'].map(sites_per_gov).fillna(0)
    
    # Treatment: any district with debris sites (or above median)
    median_sites = panel_df['n_debris_sites'].median()
    panel_df['treated'] = (panel_df['n_debris_sites'] > 0).astype(int)
    panel_df['high_treatment'] = (panel_df['n_debris_sites'] > median_sites).astype(int)
    
    # Post period
    panel_df['post'] = (panel_df['period_start'] >= DEBRIS_START).astype(int)
    
    # DiD interaction
    panel_df['treated_post'] = panel_df['treated'] * panel_df['post']
    panel_df['high_treatment_post'] = panel_df['high_treatment'] * panel_df['post']
    
    print(f"\n  Treatment assignment:")
    for gov in unit_to_gov.values():
        n_sites = sites_per_gov.get(gov, 0)
        treated = "TREATED" if n_sites > 0 else "CONTROL"
        print(f"    {gov}: {n_sites} sites → {treated}")
    
    return panel_df


def run_district_did(panel_df, outcome, treatment_var='treated_post'):
    """
    Run district-level DiD regression.
    
    Model: Y_dt = α_d + γ_t + β(Treated_d × Post_t) + ε_dt
    
    NOTE: If all districts are treated, this uses INTENSITY (n_debris_sites)
    instead of binary treatment.
    """
    df = panel_df.dropna(subset=[outcome]).copy()
    
    if len(df) < 10:
        return None
    
    # Ensure outcome is numeric
    df[outcome] = pd.to_numeric(df[outcome], errors='coerce')
    df = df.dropna(subset=[outcome])
    
    if len(df) < 10:
        return None
    
    # Check if we have variation in treatment
    n_treated = df['treated'].sum()
    n_control = len(df) - n_treated
    
    # Simple OLS with fixed effects
    y = df[outcome].astype(float).values
    
    # If ALL districts treated, use intensity-based approach
    if n_control == 0:
        print(f"    ⚠ All districts treated - using intensity approach")
        # Use continuous treatment intensity
        df['intensity_post'] = df['n_debris_sites'].astype(float) * df['post'].astype(float)
        treatment_var = 'intensity_post'
        X_vars = ['intensity_post', 'post']
    else:
        # Ensure numeric
        df[treatment_var] = pd.to_numeric(df[treatment_var], errors='coerce').fillna(0)
        df['treated'] = pd.to_numeric(df['treated'], errors='coerce').fillna(0)
        df['post'] = pd.to_numeric(df['post'], errors='coerce').fillna(0)
        X_vars = [treatment_var, 'treated', 'post']
    
    X = df[X_vars].astype(float).copy()
    
    # Add district dummies (excluding one for identification)
    district_dummies = pd.get_dummies(df['unit_id'], prefix='d', drop_first=True)
    # Ensure all columns are numeric
    for col in district_dummies.columns:
        district_dummies[col] = district_dummies[col].astype(float)
    X = pd.concat([X.reset_index(drop=True), district_dummies.reset_index(drop=True)], axis=1)
    
    # Add time dummies (excluding one) - convert period to string first
    df['time_str'] = df['period_start'].astype(str)
    time_dummies = pd.get_dummies(df['time_str'], prefix='t', drop_first=True)
    for col in time_dummies.columns:
        time_dummies[col] = time_dummies[col].astype(float)
    X = pd.concat([X.reset_index(drop=True), time_dummies.reset_index(drop=True)], axis=1)
    
    X = sm.add_constant(X)
    
    # Ensure all X columns are numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    try:
        # Use HC1 robust SEs instead of cluster (may fail with few clusters)
        model = sm.OLS(y, X.astype(float)).fit(cov_type='HC1')
        
        # Extract DiD coefficient
        did_coef = model.params[treatment_var]
        did_se = model.bse[treatment_var]
        did_pval = model.pvalues[treatment_var]
        
        return {
            'outcome': outcome,
            'treatment_var': treatment_var,
            'did_coef': did_coef,
            'did_se': did_se,
            'did_pval': did_pval,
            'n_obs': len(df),
            'r2': model.rsquared,
            'model': model,
            'intensity_based': n_control == 0,
        }
    except Exception as e:
        print(f"    Regression error: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_continuous_treatment_did(panel_df, outcome):
    """
    Run DiD with continuous treatment intensity (number of sites).
    
    Model: Y_dt = α_d + γ_t + β(N_sites_d × Post_t) + ε_dt
    
    This is the PREFERRED approach when all districts have some treatment.
    """
    df = panel_df.dropna(subset=[outcome]).copy()
    
    if len(df) < 10:
        return None
    
    # Ensure numeric
    df[outcome] = pd.to_numeric(df[outcome], errors='coerce')
    df = df.dropna(subset=[outcome])
    
    # Create interaction
    df['intensity_post'] = df['n_debris_sites'].astype(float) * df['post'].astype(float)
    
    y = df[outcome].astype(float).values
    X = df[['intensity_post', 'n_debris_sites', 'post']].astype(float).copy()
    
    # District FE
    district_dummies = pd.get_dummies(df['unit_id'], prefix='d', drop_first=True)
    for col in district_dummies.columns:
        district_dummies[col] = district_dummies[col].astype(float)
    X = pd.concat([X.reset_index(drop=True), district_dummies.reset_index(drop=True)], axis=1)
    
    # Time FE
    df['time_str'] = df['period_start'].astype(str)
    time_dummies = pd.get_dummies(df['time_str'], prefix='t', drop_first=True)
    for col in time_dummies.columns:
        time_dummies[col] = time_dummies[col].astype(float)
    X = pd.concat([X.reset_index(drop=True), time_dummies.reset_index(drop=True)], axis=1)
    
    X = sm.add_constant(X)
    
    # Ensure all numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    try:
        model = sm.OLS(y, X.astype(float)).fit(cov_type='HC1')
        
        return {
            'outcome': outcome,
            'treatment_var': 'intensity_post',
            'did_coef': model.params['intensity_post'],
            'did_se': model.bse['intensity_post'],
            'did_pval': model.pvalues['intensity_post'],
            'n_obs': len(df),
            'r2': model.rsquared,
            'interpretation': 'Effect per additional debris site',
        }
    except Exception as e:
        print(f"    Intensity regression error: {e}")
        return None


def create_did_plot(panel_df, outcome, result, debris_sites, output_path):
    """Create visualization of spatial DiD results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # =========================================================================
    # Panel 1: Time series by treatment intensity
    # =========================================================================
    ax1 = axes[0, 0]
    
    # Ensure period_start is datetime
    panel_df = panel_df.copy()
    panel_df['period_start'] = pd.to_datetime(panel_df['period_start'])
    
    # Group by high vs low intensity
    median_sites = panel_df['n_debris_sites'].median()
    high_mask = panel_df['n_debris_sites'] > median_sites
    low_mask = panel_df['n_debris_sites'] <= median_sites
    
    high_intensity = panel_df[high_mask].groupby('period_start')[outcome].mean()
    low_intensity = panel_df[low_mask].groupby('period_start')[outcome].mean()
    
    # Get max date for axvspan - handle empty case
    all_dates = panel_df['period_start'].dropna()
    if len(all_dates) > 0:
        max_date = all_dates.max()
    else:
        max_date = pd.Timestamp('2025-12-01')
    
    if len(high_intensity) > 0:
        ax1.plot(high_intensity.index, high_intensity.values, 'b-', linewidth=2.5, marker='o',
                 markersize=5, label=f'High intensity (>{median_sites:.0f} sites)')
    if len(low_intensity) > 0:
        ax1.plot(low_intensity.index, low_intensity.values, 'r--', linewidth=2, marker='s',
                 markersize=5, label=f'Low intensity (≤{median_sites:.0f} sites)')
    
    ax1.axvline(x=DEBRIS_START, color='green', linewidth=2.5, linestyle='--',
                label='Debris Removal Start')
    ax1.axvline(x=CONFLICT_START, color='black', linewidth=2, label='Oct 7, 2023')
    
    ax1.axvspan(CONFLICT_START, DEBRIS_START, alpha=0.1, color='orange', label='Pre-treatment')
    ax1.axvspan(DEBRIS_START, max_date, alpha=0.1, color='green', label='Post-treatment')
    
    ax1.set_ylabel(outcome, fontsize=11)
    ax1.set_title('Treatment Intensity: High vs Low Districts', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # =========================================================================
    # Panel 2: Treatment intensity map
    # =========================================================================
    ax2 = axes[0, 1]
    
    # Simple scatter of debris sites
    sites_df = debris_sites.copy()
    
    # Color by governorate
    gov_colors = {
        'North Gaza': 'red',
        'Gaza City': 'blue',
        'Deir al-Balah': 'green',
        'Khan Younis': 'orange',
        'Rafah': 'purple',
    }
    
    for gov, color in gov_colors.items():
        gov_sites = sites_df[sites_df['governorate'] == gov]
        marker = 'o' if gov_sites['type'].iloc[0] == 'road' else '^' if len(gov_sites) > 0 else 'o'
        ax2.scatter(gov_sites['lon'], gov_sites['lat'], c=color, s=50, alpha=0.7,
                   label=f"{gov} ({len(gov_sites)} sites)", edgecolors='black')
    
    ax2.set_xlabel('Longitude', fontsize=11)
    ax2.set_ylabel('Latitude', fontsize=11)
    ax2.set_title('Debris Site Locations by Governorate', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # =========================================================================
    # Panel 3: Intensity effect visualization
    # =========================================================================
    ax3 = axes[1, 0]
    
    # Pre-post comparison by intensity
    median_sites = panel_df['n_debris_sites'].median()
    pre_high = panel_df[(panel_df['n_debris_sites'] > median_sites) & (panel_df['post']==0)][outcome].mean()
    post_high = panel_df[(panel_df['n_debris_sites'] > median_sites) & (panel_df['post']==1)][outcome].mean()
    pre_low = panel_df[(panel_df['n_debris_sites'] <= median_sites) & (panel_df['post']==0)][outcome].mean()
    post_low = panel_df[(panel_df['n_debris_sites'] <= median_sites) & (panel_df['post']==1)][outcome].mean()
    
    # Handle NaN values
    if np.isnan(pre_high) or np.isnan(post_high) or np.isnan(pre_low) or np.isnan(post_low):
        ax3.text(0.5, 0.5, 'Insufficient data\nfor comparison', ha='center', va='center',
                transform=ax3.transAxes, fontsize=12)
    else:
        x = [0, 1]
        ax3.plot(x, [pre_high, post_high], 'b-o', linewidth=3, markersize=12, label='High intensity')
        ax3.plot(x, [pre_low, post_low], 'r--s', linewidth=3, markersize=12, label='Low intensity')
        
        # Counterfactual for high intensity (what if it had low intensity trend)
        counterfactual = pre_high + (post_low - pre_low)
        ax3.plot([1], [counterfactual], 'b^', markersize=15, markerfacecolor='none',
                 markeredgewidth=2, label='Counterfactual')
        
        # DiD arrow
        ax3.annotate('', xy=(1.05, post_high), xytext=(1.05, counterfactual),
                    arrowprops=dict(arrowstyle='<->', color='green', lw=2))
        ax3.annotate(f'β = {result["did_coef"]:.4f}', xy=(1.1, (post_high + counterfactual)/2),
                    fontsize=11, fontweight='bold', color='green')
    
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(['Pre-Treatment\n(Oct 2023 - Sep 2024)', 'Post-Treatment\n(Oct 2024+)'])
    ax3.set_ylabel(outcome, fontsize=11)
    ax3.set_title('Effect by Treatment Intensity', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # =========================================================================
    # Panel 4: Results table
    # =========================================================================
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    sig = '***' if result['did_pval'] < 0.01 else '**' if result['did_pval'] < 0.05 else '*' if result['did_pval'] < 0.1 else ''
    
    interpretation = "✓ POSITIVE EFFECT" if result['did_coef'] > 0 else "⚠ NEGATIVE EFFECT"
    interp_color = '#90EE90' if result['did_coef'] > 0 else '#FFB6C1'
    
    # Check if intensity-based
    is_intensity = result.get('intensity_based', True)  # Default True since all districts treated
    
    table_data = [
        ['INTENSITY-BASED DiD', ''],
        ['Outcome', outcome],
        ['Treatment', 'N debris sites × Post'],
        ['Note', 'All districts treated'],
        ['', ''],
        ['INTENSITY COEFFICIENT', ''],
        ['β (per site × Post)', f"{result['did_coef']:.6f}{sig}"],
        ['SE (robust)', f"{result['did_se']:.6f}"],
        ['p-value', f"{result['did_pval']:.4f}"],
        ['', ''],
        ['MODEL FIT', ''],
        ['R²', f"{result['r2']:.3f}"],
        ['N observations', f"{result['n_obs']}"],
        ['Fixed Effects', 'District + Time'],
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
        elif row[0] in ['INTENSITY-BASED DiD', 'INTENSITY COEFFICIENT', 'MODEL FIT']:
            table[(i, 0)].set_facecolor('#E0E0E0')
            table[(i, 1)].set_facecolor('#E0E0E0')
    
    # =========================================================================
    # Title
    # =========================================================================
    fig.suptitle(
        f'Spatial DiD: Debris Removal Effect on {outcome}\n'
        f'DiD Coefficient: {result["did_coef"]:.4f}{sig} (p={result["did_pval"]:.3f})',
        fontsize=13, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    print("\n" + "=" * 70)
    print("SPATIAL DIFFERENCE-IN-DIFFERENCES: DEBRIS REMOVAL")
    print("=" * 70)
    print("""
    Comparing districts WITH debris removal vs WITHOUT debris removal
    Using within-Gaza variation for cleaner identification
    """)
    
    # Load debris sites
    print("\n--- Loading Debris Site Locations ---")
    debris_sites = load_debris_sites()
    
    # Load panel data
    print("\n--- Loading Panel Data ---")
    panel_df = load_panel_data()
    
    if panel_df is None:
        print("⚠ Cannot proceed without panel data")
        return
    
    # Assign treatment status
    print("\n--- Assigning Treatment Status ---")
    panel_df = assign_treatment_status(panel_df, debris_sites)
    
    # Outcomes to test
    outcomes = ['vh', 'vv', 'ndvi', 'nightlights', 'rvi']
    outcomes = [o for o in outcomes if o in panel_df.columns]
    
    print(f"\n--- Running DiD Analysis ---")
    print(f"  Outcomes: {outcomes}")
    
    results = []
    
    for outcome in outcomes:
        print(f"\n  {outcome}...")
        
        # Binary treatment DiD
        result = run_district_did(panel_df, outcome, 'treated_post')
        
        if result:
            sig = '***' if result['did_pval'] < 0.01 else '**' if result['did_pval'] < 0.05 else '*' if result['did_pval'] < 0.1 else ''
            print(f"    DiD (binary): β={result['did_coef']:.4f}{sig} (p={result['did_pval']:.3f})")
            
            # Create plot
            create_did_plot(panel_df, outcome, result, debris_sites,
                          FIGURES_DIR / f'spatial_did_{outcome}.png')
            print(f"    ✓ spatial_did_{outcome}.png")
            
            results.append(result)
        
        # Continuous treatment intensity
        result_intensity = run_continuous_treatment_did(panel_df, outcome)
        
        if result_intensity:
            sig = '***' if result_intensity['did_pval'] < 0.01 else '**' if result_intensity['did_pval'] < 0.05 else '*' if result_intensity['did_pval'] < 0.1 else ''
            print(f"    DiD (intensity): β={result_intensity['did_coef']:.4f}{sig} per site")
            
            results.append(result_intensity)
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(TABLES_DIR / 'spatial_did_results.csv', index=False)
        print(f"\n✓ Results saved: {TABLES_DIR}/spatial_did_results.csv")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("""
    INTERPRETATION:
    ───────────────
    Positive DiD coefficient = Debris removal improved outcomes
    
    Key assumption: Parallel trends in pre-treatment period
    → Check top-left panel of each plot
    
    LIMITATIONS:
    ────────────
    1. Only 5 districts → limited statistical power
    2. All districts may have SOME debris clearance (spillovers)
    3. Treatment timing may vary within districts
    
    NEXT STEPS FOR STRONGER IDENTIFICATION:
    ───────────────────────────────────────
    1. Get exact GPS coordinates from UNDP GIS team
    2. Extract pixel-level raster data (not district means)
    3. Run pixel-level DiD with 100m/250m/500m buffers
    4. This would give thousands of observations vs 5 districts
    """)
    
    print(f"\n{'='*70}")
    print("✓ SPATIAL DiD ANALYSIS COMPLETE")
    print(f"  Figures: {FIGURES_DIR}/spatial_did_*.png")
    print(f"  Results: {TABLES_DIR}/spatial_did_results.csv")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
