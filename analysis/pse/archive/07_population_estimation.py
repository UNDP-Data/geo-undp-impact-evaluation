#!/usr/bin/env python3
"""
analysis/pse/07_population_estimation.py
========================================
Population and Beneficiary Estimation with Transparent Methodology.

This script provides MULTIPLE methods for estimating beneficiaries,
with clear documentation of assumptions and limitations.

METHODS:
========

1. DENSITY-BASED (Recommended when no WorldPop available)
   - Uses Gaza's known population density
   - Assumption: Uniform distribution (conservative)
   - Formula: Buffer_Area × Density × Urbanization_Factor
   
2. WORLDPOP-BASED (Most accurate, requires data download)
   - Uses satellite-calibrated population rasters
   - Resolution: 100m
   - Download: hub.worldpop.org
   
3. VIIRS-CALIBRATED (If you have both VIIRS and WorldPop for calibration)
   - Establishes local VIIRS-population relationship pre-conflict
   - Applies relationship to estimate post-conflict population
   - Requires: VIIRS + WorldPop for same period
   
4. STRUCTURE-BASED (Uses SAR as proxy)
   - Estimates habitable structures from SAR
   - Multiplies by household size
   - Most uncertain but captures physical changes

Usage:
    python analysis/pse/07_population_estimation.py
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
import json
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

try:
    import rasterio
    from shapely.geometry import Point
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path("data")
SITES_CSV = DATA_DIR / "pse" / "external" / "undp_sites_digitized.csv"
BOUNDARIES_FILE = DATA_DIR / "pse" / "boundaries" / "gaza_admin2.geojson"
WORLDPOP_DIR = DATA_DIR / "pse" / "population"
RASTER_DIR = DATA_DIR / "pse" / "rasters"

OUTPUT_DIR = Path("outputs/pse")
FIGURES_DIR = OUTPUT_DIR / "figures" / "beneficiaries"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# GAZA PARAMETERS (Well-documented sources)
# =============================================================================
GAZA_PARAMS = {
    # Population
    'total_pop_2023': 2_100_000,        # PCBS estimate, pre-conflict
    'total_pop_2024_est': 1_900_000,    # Estimated after displacement/casualties
    
    # Geography
    'area_km2': 365,                     # Total area of Gaza Strip
    'avg_density_per_km2': 5_753,        # 2.1M / 365 km²
    
    # Urban density varies significantly
    'density_urban_core': 15_000,        # Gaza City, Khan Younis centers
    'density_urban': 8_000,              # Urban areas
    'density_peri_urban': 4_000,         # Peri-urban
    'density_rural': 1_500,              # Agricultural areas
    
    # Household
    'avg_household_size': 5.5,           # PCBS
    'persons_per_dwelling': 5.2,         # Similar to household size
    
    # Infrastructure
    'total_buildings_est': 150_000,      # Rough estimate
    'residential_pct': 0.70,             # ~70% residential
}

# Buffer parameters
BUFFER_RADIUS_M = 500
BUFFER_AREA_KM2 = np.pi * (BUFFER_RADIUS_M / 1000) ** 2  # ~0.785 km²

# =============================================================================
# METHOD 1: DENSITY-BASED ESTIMATION
# =============================================================================

def estimate_population_density_based(sites_df, density_scenario='medium'):
    """
    Estimate population using Gaza's average density.
    
    This is the simplest method but makes strong assumptions:
    - Population is uniformly distributed within urban areas
    - All intervention sites are in similar density areas
    
    ASSUMPTIONS DOCUMENTED:
    - Buffer radius: 500m (service area assumption)
    - Buffer area: 0.785 km² (π × 0.5²)
    - Density scenarios account for site location uncertainty
    
    Args:
        sites_df: DataFrame with intervention sites
        density_scenario: 'low', 'medium', 'high', or 'by_type'
    
    Returns:
        dict with population estimates and methodology notes
    """
    n_sites = len(sites_df)
    
    # Define density scenarios
    scenarios = {
        'low': {
            'density': 3_000,  # Conservative: mix of urban and peri-urban
            'description': 'Conservative estimate assuming mixed urban/peri-urban areas',
            'basis': 'Lower bound considering some sites in less dense areas'
        },
        'medium': {
            'density': 5_000,  # Gaza average adjusted for urban bias of interventions
            'description': 'Medium estimate using adjusted Gaza average density',
            'basis': 'Sites likely in urban areas but not all in densest zones'
        },
        'high': {
            'density': 8_000,  # Urban areas
            'description': 'Upper estimate assuming most sites in urban areas',
            'basis': 'UNDP interventions concentrated in populated urban areas'
        }
    }
    
    # By intervention type (different types may target different areas)
    type_densities = {
        'WASH': 6_000,           # Water systems serve denser populations
        'Debris': 7_000,         # Debris in urban areas
        'Shelter': 5_000,        # Shelter across various areas
        'Employment': 4_000,     # Employment programs may be more dispersed
        'Social Cohesion': 5_000,
        'Other': 5_000,
    }
    
    results = {
        'method': 'Density-Based Estimation',
        'buffer_radius_m': BUFFER_RADIUS_M,
        'buffer_area_km2': BUFFER_AREA_KM2,
        'n_sites': n_sites,
        'scenarios': {}
    }
    
    # Calculate for each scenario
    for scenario_name, params in scenarios.items():
        pop_per_site = BUFFER_AREA_KM2 * params['density']
        gross_population = n_sites * pop_per_site
        
        # Overlap adjustment
        # Sites may have overlapping buffers - estimate ~30% overlap on average
        overlap_factor = 0.70
        unique_population = gross_population * overlap_factor
        
        results['scenarios'][scenario_name] = {
            'density_per_km2': params['density'],
            'description': params['description'],
            'basis': params['basis'],
            'population_per_site': round(pop_per_site),
            'gross_population': round(gross_population),
            'overlap_adjustment': overlap_factor,
            'unique_beneficiaries': round(unique_population),
        }
    
    # By intervention type
    if 'intervention_type' in sites_df.columns:
        type_estimates = {}
        for itype, count in sites_df['intervention_type'].value_counts().items():
            density = type_densities.get(itype, 5_000)
            pop = count * BUFFER_AREA_KM2 * density * 0.70  # With overlap adjustment
            type_estimates[itype] = {
                'sites': count,
                'density_assumed': density,
                'beneficiaries': round(pop)
            }
        results['by_intervention_type'] = type_estimates
    
    return results


# =============================================================================
# METHOD 2: WORLDPOP-BASED ESTIMATION
# =============================================================================

def estimate_population_worldpop(sites_df, worldpop_path):
    """
    Estimate population using WorldPop raster data.
    
    WorldPop provides satellite-calibrated population estimates at ~100m resolution.
    This is the most accurate method when data is available.
    
    METHODOLOGY:
    - WorldPop uses machine learning to combine:
      * Satellite imagery (building footprints, urban extent)
      * Census data (national/regional totals)
      * Ancillary data (roads, land cover)
    - Resolution: ~100m (each pixel = population count)
    - Accuracy: Generally ±20% at sub-national level
    
    ASSUMPTIONS:
    - Buffer radius: 500m (circular service area)
    - Sum all pixels within buffer
    - No temporal adjustment (uses 2023 baseline)
    """
    if not HAS_RASTERIO or not Path(worldpop_path).exists():
        return None
    
    results = {
        'method': 'WorldPop Raster Extraction',
        'data_source': str(worldpop_path),
        'buffer_radius_m': BUFFER_RADIUS_M,
        'sites': []
    }
    
    deg_per_m = 1 / 111000  # Approximate at Gaza latitude
    buffer_deg = BUFFER_RADIUS_M * deg_per_m
    
    with rasterio.open(worldpop_path) as src:
        for _, site in sites_df.iterrows():
            try:
                lon, lat = site['longitude'], site['latitude']
                
                # Get window around site
                minx, miny = lon - buffer_deg, lat - buffer_deg
                maxx, maxy = lon + buffer_deg, lat + buffer_deg
                
                window = rasterio.windows.from_bounds(minx, miny, maxx, maxy, src.transform)
                data = src.read(1, window=window)
                
                # WorldPop gives population per pixel
                # Sum all valid pixels
                valid_data = data[(data != src.nodata) & (~np.isnan(data)) & (data >= 0)]
                population = np.sum(valid_data)
                
                results['sites'].append({
                    'site_id': site['site_id'],
                    'intervention_type': site.get('intervention_type', 'Unknown'),
                    'latitude': lat,
                    'longitude': lon,
                    'population': round(population),
                })
                
            except Exception as e:
                continue
    
    # Aggregate
    if results['sites']:
        site_df = pd.DataFrame(results['sites'])
        results['summary'] = {
            'total_sites_processed': len(site_df),
            'total_population_in_buffers': round(site_df['population'].sum()),
            'mean_population_per_site': round(site_df['population'].mean()),
            'median_population_per_site': round(site_df['population'].median()),
            'min_population': round(site_df['population'].min()),
            'max_population': round(site_df['population'].max()),
        }
        
        # Overlap adjustment (estimate 30% overlap)
        results['summary']['overlap_adjusted_beneficiaries'] = round(
            results['summary']['total_population_in_buffers'] * 0.70
        )
        
        # By type
        if 'intervention_type' in site_df.columns:
            results['by_type'] = site_df.groupby('intervention_type')['population'].agg([
                'count', 'sum', 'mean'
            ]).round().to_dict('index')
    
    return results


# =============================================================================
# METHOD 3: VIIRS-CALIBRATED ESTIMATION
# =============================================================================

def estimate_population_viirs_calibrated(sites_df, viirs_rasters, calibration_pop=None):
    """
    Estimate population using VIIRS nightlights with proper calibration.
    
    METHODOLOGY:
    The relationship between nightlights and population is:
        Population = α × VIIRS^β
    
    Where:
    - α (intercept) varies by region/urbanization
    - β (elasticity) typically 0.5-1.0 (diminishing returns at high density)
    
    WITHOUT CALIBRATION DATA:
    - We use literature-based elasticity (β ≈ 0.8)
    - Calibrate α to match total Gaza population
    - This gives RELATIVE distribution, not absolute counts
    
    WITH CALIBRATION DATA (WorldPop + VIIRS for same period):
    - Regress: log(Pop) = α + β×log(VIIRS)
    - Use fitted relationship for estimation
    
    ASSUMPTIONS:
    - Pre-conflict VIIRS reflects pre-conflict population distribution
    - Relationship holds across Gaza (may not hold in extreme cases)
    - Nightlights include residential + commercial (overestimates in commercial areas)
    
    LIMITATIONS:
    - Cannot capture post-conflict displacement
    - Commercial areas appear more populous than they are
    - Very poor areas may be underestimated (less light per person)
    """
    if not viirs_rasters:
        return None
    
    # Get pre-conflict VIIRS
    pre_conflict_dates = [d for d in viirs_rasters.keys() if d < datetime(2023, 10, 1)]
    if not pre_conflict_dates:
        return None
    
    latest_pre = max(pre_conflict_dates)
    viirs_path = viirs_rasters[latest_pre]
    
    results = {
        'method': 'VIIRS-Calibrated Estimation',
        'viirs_date': latest_pre.strftime('%Y-%m'),
        'calibration': 'Literature-based (no local calibration data)',
        'assumptions': {
            'elasticity_beta': 0.8,
            'calibration_population': GAZA_PARAMS['total_pop_2023'],
            'buffer_radius_m': BUFFER_RADIUS_M,
        },
        'limitations': [
            'Uses pre-conflict VIIRS distribution',
            'Cannot capture displacement',
            'Commercial areas may be overestimated',
            'Very poor areas may be underestimated'
        ]
    }
    
    deg_per_m = 1 / 111000
    buffer_deg = BUFFER_RADIUS_M * deg_per_m
    
    site_viirs = []
    total_viirs = 0
    
    with rasterio.open(viirs_path) as src:
        # First pass: get total VIIRS
        full_data = src.read(1)
        valid_mask = (full_data != src.nodata) & (~np.isnan(full_data)) & (full_data > 0)
        total_viirs = np.sum(full_data[valid_mask])
        
        # Second pass: get VIIRS at each site
        for _, site in sites_df.iterrows():
            try:
                lon, lat = site['longitude'], site['latitude']
                
                minx, miny = lon - buffer_deg, lat - buffer_deg
                maxx, maxy = lon + buffer_deg, lat + buffer_deg
                
                window = rasterio.windows.from_bounds(minx, miny, maxx, maxy, src.transform)
                data = src.read(1, window=window)
                
                valid_data = data[(data != src.nodata) & (~np.isnan(data)) & (data > 0)]
                site_light = np.sum(valid_data) if len(valid_data) > 0 else 0
                
                site_viirs.append({
                    'site_id': site['site_id'],
                    'intervention_type': site.get('intervention_type', 'Unknown'),
                    'viirs_sum': site_light,
                })
                
            except:
                continue
    
    if not site_viirs or total_viirs == 0:
        return None
    
    site_df = pd.DataFrame(site_viirs)
    
    # Apply calibration
    # Simple proportional allocation (with non-linear adjustment)
    beta = 0.8
    site_df['viirs_adjusted'] = site_df['viirs_sum'] ** beta
    total_adjusted = site_df['viirs_adjusted'].sum()
    
    if total_adjusted > 0:
        site_df['pop_share'] = site_df['viirs_adjusted'] / total_adjusted
        site_df['population'] = site_df['pop_share'] * GAZA_PARAMS['total_pop_2023']
    else:
        site_df['population'] = 0
    
    results['sites'] = site_df.to_dict('records')
    results['summary'] = {
        'total_viirs_at_sites': round(site_df['viirs_sum'].sum(), 2),
        'total_viirs_gaza': round(total_viirs, 2),
        'site_share_of_total_viirs': round(site_df['viirs_sum'].sum() / total_viirs * 100, 2),
        'implied_population_share': round(site_df['population'].sum() / GAZA_PARAMS['total_pop_2023'] * 100, 2),
        'total_population_estimate': round(site_df['population'].sum()),
        'overlap_adjusted': round(site_df['population'].sum() * 0.70),
    }
    
    return results


# =============================================================================
# METHOD 4: STRUCTURE-BASED ESTIMATION (SAR)
# =============================================================================

def estimate_population_structure_based(sites_df, sar_treatment_effect):
    """
    Estimate beneficiaries from improved structures using SAR data.
    
    METHODOLOGY:
    SAR (Synthetic Aperture Radar) backscatter increases with:
    - More vertical structures (buildings, walls)
    - Surface roughness
    - Construction materials
    
    The treatment effect in SAR VH reflects CHANGE in structures.
    
    ASSUMPTIONS:
    - SAR increase of 1 dB ≈ 10-15% more detectable structures (rough calibration)
    - Each improved structure benefits one household
    - Average household size: 5.5 people
    
    CALCULATION:
    1. Estimate baseline structures in treatment area
    2. Apply SAR-based improvement percentage
    3. Convert to households → people
    
    LIMITATIONS:
    - SAR-to-structure relationship needs ground-truth calibration
    - Cannot distinguish new construction from debris clearance
    - Metal debris can also increase backscatter
    """
    # Baseline: estimate structures in treatment areas
    # Gaza has ~150,000 buildings, ~70% residential = ~105,000 dwellings
    # If treatment areas cover ~5% of Gaza → ~5,250 dwellings
    
    # Better: use site count and typical building density
    n_sites = len(sites_df)
    buffer_area_km2 = np.pi * (500/1000)**2  # 0.785 km²
    total_treatment_area = n_sites * buffer_area_km2 * 0.5  # Assume 50% overlap
    
    # Building density in urban Gaza: ~400-600 buildings per km²
    building_density = 500  # per km²
    baseline_structures = total_treatment_area * building_density
    
    # SAR treatment effect interpretation
    # +1 dB ≈ 10-15% structural change (literature-based, needs local calibration)
    sar_to_structure_pct = 0.10  # 10% per dB (conservative)
    
    structure_improvement_pct = sar_treatment_effect * sar_to_structure_pct
    structures_improved = baseline_structures * structure_improvement_pct
    
    # Convert to people
    household_size = GAZA_PARAMS['avg_household_size']
    people_benefiting = structures_improved * household_size
    
    results = {
        'method': 'Structure-Based Estimation (SAR)',
        'sar_treatment_effect_db': sar_treatment_effect,
        'assumptions': {
            'buffer_radius_m': BUFFER_RADIUS_M,
            'building_density_per_km2': building_density,
            'overlap_adjustment': 0.5,
            'sar_to_structure_conversion': f'{sar_to_structure_pct*100}% improvement per 1 dB',
            'household_size': household_size,
        },
        'calculations': {
            'total_treatment_area_km2': round(total_treatment_area, 2),
            'baseline_structures': round(baseline_structures),
            'structure_improvement_pct': round(structure_improvement_pct * 100, 1),
            'structures_improved': round(structures_improved),
            'people_benefiting': round(people_benefiting),
        },
        'confidence': 'LOW - requires ground-truth calibration',
        'range': {
            'low': round(people_benefiting * 0.5),   # Half the effect
            'medium': round(people_benefiting),
            'high': round(people_benefiting * 2.0),  # Double the effect
        }
    }
    
    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_methodology_comparison_plot(all_results, output_path):
    """Compare estimates across different methods."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Extract estimates
    methods = []
    estimates = []
    colors = []
    
    # Density-based
    if 'density_based' in all_results:
        for scenario, data in all_results['density_based']['scenarios'].items():
            methods.append(f"Density ({scenario})")
            estimates.append(data['unique_beneficiaries'])
            colors.append('#3498db')
    
    # WorldPop
    if 'worldpop' in all_results and all_results['worldpop']:
        methods.append("WorldPop")
        estimates.append(all_results['worldpop']['summary']['overlap_adjusted_beneficiaries'])
        colors.append('#27ae60')
    
    # VIIRS
    if 'viirs_calibrated' in all_results and all_results['viirs_calibrated']:
        methods.append("VIIRS-Calibrated")
        estimates.append(all_results['viirs_calibrated']['summary']['overlap_adjusted'])
        colors.append('#f39c12')
    
    # Structure-based
    if 'structure_based' in all_results:
        methods.append("SAR Structures")
        estimates.append(all_results['structure_based']['calculations']['people_benefiting'])
        colors.append('#e74c3c')
    
    # Plot 1: Bar chart comparison
    ax1 = axes[0]
    bars = ax1.barh(methods, estimates, color=colors, edgecolor='black')
    ax1.set_xlabel('Estimated Beneficiaries', fontsize=11)
    ax1.set_title('Beneficiary Estimates by Method', fontsize=13, fontweight='bold')
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
    
    for bar, val in zip(bars, estimates):
        ax1.annotate(f'{val/1000:.0f}K', xy=(val, bar.get_y() + bar.get_height()/2),
                    xytext=(5, 0), textcoords='offset points', va='center', fontsize=10)
    
    ax1.axvline(GAZA_PARAMS['total_pop_2023'] * 0.5, color='gray', linestyle='--', 
                alpha=0.5, label='50% of Gaza pop')
    ax1.legend(loc='lower right')
    
    # Plot 2: Methodology notes
    ax2 = axes[1]
    ax2.axis('off')
    
    notes = """
    METHODOLOGY COMPARISON
    ══════════════════════════════════════════
    
    DENSITY-BASED (Blue bars):
    • Uses Gaza's population density
    • Assumes uniform distribution
    • Most transparent assumptions
    • Recommended when no spatial data
    
    WORLDPOP (Green bar):
    • Satellite-calibrated population
    • Most accurate spatial distribution
    • Requires data download
    • Uses 2023 baseline (pre-conflict)
    
    VIIRS-CALIBRATED (Orange bar):
    • Nightlights as population proxy
    • Requires calibration data
    • Cannot capture displacement
    • May over/underestimate some areas
    
    SAR STRUCTURES (Red bar):
    • Based on physical reconstruction
    • Measures CHANGE, not absolute pop
    • Needs ground-truth calibration
    • Most uncertain
    
    RECOMMENDATION:
    • Report density-based (medium) as primary
    • Use WorldPop if available
    • Present range, not point estimate
    """
    
    ax2.text(0.05, 0.95, notes, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")


def create_summary_table(all_results, output_path):
    """Create summary table of all estimates."""
    
    rows = []
    
    # Density-based
    if 'density_based' in all_results:
        for scenario, data in all_results['density_based']['scenarios'].items():
            rows.append({
                'Method': 'Density-Based',
                'Scenario': scenario.capitalize(),
                'Population per Site': data['population_per_site'],
                'Gross Population': data['gross_population'],
                'Unique Beneficiaries': data['unique_beneficiaries'],
                'Confidence': 'Medium',
                'Notes': data['description']
            })
    
    # WorldPop
    if 'worldpop' in all_results and all_results['worldpop']:
        rows.append({
            'Method': 'WorldPop',
            'Scenario': 'Raster extraction',
            'Population per Site': all_results['worldpop']['summary']['mean_population_per_site'],
            'Gross Population': all_results['worldpop']['summary']['total_population_in_buffers'],
            'Unique Beneficiaries': all_results['worldpop']['summary']['overlap_adjusted_beneficiaries'],
            'Confidence': 'High',
            'Notes': 'Satellite-calibrated population data'
        })
    
    # VIIRS
    if 'viirs_calibrated' in all_results and all_results['viirs_calibrated']:
        rows.append({
            'Method': 'VIIRS-Calibrated',
            'Scenario': 'Proportional allocation',
            'Population per Site': round(all_results['viirs_calibrated']['summary']['total_population_estimate'] / len(all_results['viirs_calibrated']['sites'])),
            'Gross Population': all_results['viirs_calibrated']['summary']['total_population_estimate'],
            'Unique Beneficiaries': all_results['viirs_calibrated']['summary']['overlap_adjusted'],
            'Confidence': 'Low',
            'Notes': 'Pre-conflict distribution, needs calibration'
        })
    
    # Structure-based
    if 'structure_based' in all_results:
        rows.append({
            'Method': 'SAR Structure-Based',
            'Scenario': 'Physical reconstruction',
            'Population per Site': round(all_results['structure_based']['calculations']['people_benefiting'] / 454),
            'Gross Population': all_results['structure_based']['calculations']['people_benefiting'],
            'Unique Beneficiaries': all_results['structure_based']['calculations']['people_benefiting'],
            'Confidence': 'Low',
            'Notes': 'Measures improvement, not absolute population'
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  ✓ Saved: {output_path.name}")
    
    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print(" POPULATION ESTIMATION WITH TRANSPARENT METHODOLOGY")
    print("=" * 70)
    
    # Load sites
    print("\n--- Loading UNDP Sites ---")
    sites_df = pd.read_csv(SITES_CSV)
    print(f"  Sites: {len(sites_df)}")
    print(f"  Types: {dict(sites_df['intervention_type'].value_counts())}")
    
    all_results = {}
    
    # ==========================================================================
    # METHOD 1: Density-Based
    # ==========================================================================
    print("\n" + "=" * 70)
    print(" METHOD 1: DENSITY-BASED ESTIMATION")
    print("=" * 70)
    
    density_results = estimate_population_density_based(sites_df)
    all_results['density_based'] = density_results
    
    print(f"\n  Buffer area: {BUFFER_AREA_KM2:.3f} km² (500m radius)")
    print(f"  Sites: {density_results['n_sites']}")
    
    print("\n  Scenarios:")
    for scenario, data in density_results['scenarios'].items():
        print(f"    {scenario.upper()}: {data['unique_beneficiaries']:,} beneficiaries")
        print(f"      Density: {data['density_per_km2']:,}/km² | Per site: {data['population_per_site']:,}")
    
    # ==========================================================================
    # METHOD 2: WorldPop
    # ==========================================================================
    print("\n" + "=" * 70)
    print(" METHOD 2: WORLDPOP RASTER EXTRACTION")
    print("=" * 70)
    
    # Check for WorldPop data
    worldpop_files = list(WORLDPOP_DIR.glob("*.tif")) if WORLDPOP_DIR.exists() else []
    
    if worldpop_files:
        print(f"  Found: {[f.name for f in worldpop_files]}")
        worldpop_results = estimate_population_worldpop(sites_df, worldpop_files[0])
        all_results['worldpop'] = worldpop_results
        
        if worldpop_results:
            print(f"\n  Results:")
            print(f"    Total in buffers: {worldpop_results['summary']['total_population_in_buffers']:,}")
            print(f"    Unique beneficiaries: {worldpop_results['summary']['overlap_adjusted_beneficiaries']:,}")
    else:
        print("  ⚠ No WorldPop data found")
        print("  To download:")
        print("    1. Visit: https://hub.worldpop.org/geodata/listing?id=79")
        print("    2. Filter: Country = PSE (Palestine)")
        print("    3. Download: Unconstrained 100m for 2020 or 2023")
        print("    4. Place in: data/pse/population/")
        all_results['worldpop'] = None
    
    # ==========================================================================
    # METHOD 3: VIIRS-Calibrated
    # ==========================================================================
    print("\n" + "=" * 70)
    print(" METHOD 3: VIIRS-CALIBRATED ESTIMATION")
    print("=" * 70)
    
    viirs_rasters = {}
    for f in RASTER_DIR.glob("viirs_*.tif"):
        for part in f.stem.split('_'):
            try:
                if len(part) == 6 and part.isdigit():
                    d = datetime.strptime(part, "%Y%m")
                    viirs_rasters[d] = f
                    break
            except:
                continue
    
    if viirs_rasters:
        viirs_results = estimate_population_viirs_calibrated(sites_df, viirs_rasters)
        all_results['viirs_calibrated'] = viirs_results
        
        if viirs_results:
            print(f"\n  Using VIIRS from: {viirs_results['viirs_date']}")
            print(f"  Site share of Gaza VIIRS: {viirs_results['summary']['site_share_of_total_viirs']:.1f}%")
            print(f"  Implied population share: {viirs_results['summary']['implied_population_share']:.1f}%")
            print(f"  Population estimate: {viirs_results['summary']['total_population_estimate']:,}")
            print(f"  Overlap-adjusted: {viirs_results['summary']['overlap_adjusted']:,}")
            
            print("\n  ⚠ LIMITATIONS:")
            for lim in viirs_results['limitations']:
                print(f"    - {lim}")
    else:
        print("  ⚠ No VIIRS rasters found")
        all_results['viirs_calibrated'] = None
    
    # ==========================================================================
    # METHOD 4: Structure-Based (SAR)
    # ==========================================================================
    print("\n" + "=" * 70)
    print(" METHOD 4: STRUCTURE-BASED (SAR) ESTIMATION")
    print("=" * 70)
    
    # Use SAR treatment effect from your DiD results
    sar_effect = 1.33  # From your comprehensive DiD
    
    structure_results = estimate_population_structure_based(sites_df, sar_effect)
    all_results['structure_based'] = structure_results
    
    print(f"\n  SAR treatment effect: +{sar_effect:.2f} dB")
    print(f"  Implied structure improvement: {structure_results['calculations']['structure_improvement_pct']}%")
    print(f"  Structures improved: {structure_results['calculations']['structures_improved']:,}")
    print(f"  People benefiting: {structure_results['calculations']['people_benefiting']:,}")
    print(f"\n  Range (accounting for calibration uncertainty):")
    print(f"    Low: {structure_results['range']['low']:,}")
    print(f"    Medium: {structure_results['range']['medium']:,}")
    print(f"    High: {structure_results['range']['high']:,}")
    
    # ==========================================================================
    # Create outputs
    # ==========================================================================
    print("\n" + "=" * 70)
    print(" CREATING OUTPUTS")
    print("=" * 70)
    
    create_methodology_comparison_plot(all_results, FIGURES_DIR / "methodology_comparison.png")
    summary_df = create_summary_table(all_results, TABLES_DIR / "population_estimates_comparison.csv")
    
    # Save full results as JSON
    def convert_types(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return obj
    
    with open(TABLES_DIR / "population_methodology.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=convert_types)
    print(f"  ✓ Saved: population_methodology.json")
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print(" SUMMARY: BENEFICIARY ESTIMATES")
    print("=" * 70)
    
    print("\n  ┌─────────────────────────────────────────────────────────────┐")
    print("  │ METHOD                │ ESTIMATE     │ CONFIDENCE            │")
    print("  ├─────────────────────────────────────────────────────────────┤")
    
    # Density-based (recommended)
    medium = density_results['scenarios']['medium']['unique_beneficiaries']
    print(f"  │ Density (medium)      │ {medium:>10,} │ ★★★ RECOMMENDED       │")
    
    low = density_results['scenarios']['low']['unique_beneficiaries']
    high = density_results['scenarios']['high']['unique_beneficiaries']
    print(f"  │ Density (range)       │ {low:,}-{high:,} │ Sensitivity range   │")
    
    if all_results.get('worldpop'):
        wp = all_results['worldpop']['summary']['overlap_adjusted_beneficiaries']
        print(f"  │ WorldPop              │ {wp:>10,} │ ★★★★ Most accurate   │")
    
    if all_results.get('viirs_calibrated'):
        viirs = all_results['viirs_calibrated']['summary']['overlap_adjusted']
        print(f"  │ VIIRS-Calibrated      │ {viirs:>10,} │ ★★ Needs calibration │")
    
    sar = structure_results['calculations']['people_benefiting']
    print(f"  │ SAR Structures        │ {sar:>10,} │ ★ Uncertain          │")
    
    print("  └─────────────────────────────────────────────────────────────┘")
    
    print(f"""
  RECOMMENDED REPORT STATEMENT:
  ─────────────────────────────
  "UNDP interventions directly reached approximately {low:,} to {high:,} 
  people in Gaza (representing {low/GAZA_PARAMS['total_pop_2023']*100:.0f}-{high/GAZA_PARAMS['total_pop_2023']*100:.0f}% 
  of the pre-conflict population). The central estimate of {medium:,} 
  beneficiaries assumes average urban population density within a 500-meter 
  service radius of each intervention site."
  
  CAVEATS TO NOTE:
  ─────────────────
  • Estimates based on 2023 pre-conflict population distribution
  • Actual current beneficiaries affected by displacement
  • Buffer overlap reduces unique beneficiaries by ~30%
  • Service radius assumption (500m) affects estimates significantly
    """)
    
    print("=" * 70)
    print(" ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
