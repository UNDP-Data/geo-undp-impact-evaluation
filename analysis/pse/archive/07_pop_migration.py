#!/usr/bin/env python3
"""
analysis/pse/07_beneficiary_analysis.py
=======================================
Population and Beneficiary Impact Analysis.

This script:
1. Estimates population at UNDP intervention sites using multiple methods
2. Integrates displacement data from humanitarian sources
3. Translates DiD findings into "people helped" estimates
4. Creates cost-effectiveness metrics

Data Sources:
- WorldPop 2023 (baseline population)
- OCHA displacement estimates (via HDX API)
- VIIRS nightlights as population proxy
- SAR/NDBI as habitable structure proxy
- Google Trends (optional)

Usage:
    python analysis/pse/07_beneficiary_analysis.py
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import requests
import warnings
from datetime import datetime
import json

warnings.filterwarnings('ignore')

try:
    import rasterio
    from rasterio.mask import mask
    from shapely.geometry import Point, box
    from scipy.spatial import cKDTree
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

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

# Gaza parameters
GAZA_TOTAL_POP_2023 = 2_100_000  # Approximate pre-conflict population
GAZA_TOTAL_POP_2024 = 1_900_000  # Estimated after displacement (rough)
GAZA_AVG_HOUSEHOLD_SIZE = 5.5    # Average household size in Gaza
GAZA_AREA_KM2 = 365              # Total area of Gaza Strip

# Buffer for population extraction (meters)
POPULATION_BUFFER_M = 500

# Conversion factor (approximate for Gaza latitude)
DEG_PER_M = 1 / 100000

# =============================================================================
# HUMANITARIAN DATA SOURCES
# =============================================================================

def fetch_ocha_displacement_data():
    """
    Fetch displacement data from OCHA's Humanitarian Data Exchange (HDX).
    
    Note: This requires checking HDX for current Gaza datasets.
    URL: https://data.humdata.org/dataset?q=gaza+displacement
    """
    print("\n  Checking OCHA/HDX for displacement data...")
    
    # HDX API endpoint
    hdx_api = "https://data.humdata.org/api/3/action/package_search"
    
    try:
        # Search for Gaza displacement datasets
        params = {
            'q': 'gaza displacement population',
            'rows': 10,
            'sort': 'metadata_modified desc'
        }
        
        response = requests.get(hdx_api, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            datasets = data.get('result', {}).get('results', [])
            
            print(f"    Found {len(datasets)} relevant datasets on HDX:")
            
            hdx_sources = []
            for ds in datasets[:5]:
                title = ds.get('title', 'Unknown')
                modified = ds.get('metadata_modified', '')[:10]
                url = f"https://data.humdata.org/dataset/{ds.get('name', '')}"
                print(f"      - {title[:60]}... ({modified})")
                hdx_sources.append({
                    'title': title,
                    'url': url,
                    'modified': modified
                })
            
            return hdx_sources
        else:
            print(f"    ⚠ HDX API returned status {response.status_code}")
            
    except Exception as e:
        print(f"    ⚠ Could not fetch HDX data: {e}")
    
    return []


def fetch_unrwa_shelter_data():
    """
    Fetch UNRWA shelter population data.
    
    UNRWA publishes daily situation reports with shelter populations.
    These are typically in PDF format and need manual extraction.
    """
    print("\n  UNRWA Shelter Data Sources:")
    print("    - Daily situation reports: https://www.unrwa.org/resources/reports/gaza-situation-report")
    print("    - Data portal: https://www.unrwa.org/what-we-do/relief-and-social-services")
    
    # Known shelter population estimates (would need manual updates)
    shelter_data = {
        'source': 'UNRWA Situation Reports',
        'note': 'Requires manual extraction from PDF reports',
        'estimates': {
            '2024-01': {'sheltered_population': 1_400_000, 'num_shelters': 150},
            '2024-06': {'sheltered_population': 1_100_000, 'num_shelters': 100},
            '2024-12': {'sheltered_population': 900_000, 'num_shelters': 80},
        }
    }
    
    return shelter_data


def get_displacement_estimates():
    """
    Compile displacement estimates from multiple sources.
    """
    print("\n--- Gathering Displacement Estimates ---")
    
    # Known estimates from humanitarian reports
    displacement_data = {
        'pre_conflict_pop': GAZA_TOTAL_POP_2023,
        'estimates': [
            {
                'date': '2023-10-07',
                'source': 'Pre-conflict baseline',
                'total_pop': 2_100_000,
                'idps': 0,
                'idp_pct': 0
            },
            {
                'date': '2023-12-01',
                'source': 'OCHA Situation Report',
                'total_pop': 2_100_000,
                'idps': 1_900_000,
                'idp_pct': 90.5
            },
            {
                'date': '2024-03-01',
                'source': 'OCHA/UNRWA estimates',
                'total_pop': 2_000_000,
                'idps': 1_700_000,
                'idp_pct': 85.0
            },
            {
                'date': '2024-06-01',
                'source': 'OCHA Flash Appeal',
                'total_pop': 1_900_000,
                'idps': 1_500_000,
                'idp_pct': 78.9
            },
            {
                'date': '2024-12-01',
                'source': 'Estimated (ongoing conflict)',
                'total_pop': 1_800_000,
                'idps': 1_200_000,
                'idp_pct': 66.7
            },
        ]
    }
    
    # Try to fetch additional data from HDX
    hdx_sources = fetch_ocha_displacement_data()
    displacement_data['hdx_sources'] = hdx_sources
    
    # UNRWA data
    unrwa_data = fetch_unrwa_shelter_data()
    displacement_data['unrwa'] = unrwa_data
    
    return displacement_data


# =============================================================================
# POPULATION AT INTERVENTION SITES
# =============================================================================

def load_worldpop_data():
    """Load available WorldPop rasters."""
    worldpop_files = {}
    
    if WORLDPOP_DIR.exists():
        for f in WORLDPOP_DIR.glob("*.tif"):
            # Try to extract year from filename
            for year in ['2020', '2021', '2022', '2023', '2024']:
                if year in f.name:
                    worldpop_files[int(year)] = f
                    break
    
    return worldpop_files


def estimate_population_at_sites(sites_df, worldpop_path, buffer_m=POPULATION_BUFFER_M):
    """
    Extract population within buffer of each intervention site.
    """
    if worldpop_path is None or not Path(worldpop_path).exists():
        return None
    
    buffer_deg = buffer_m * DEG_PER_M
    
    results = []
    
    with rasterio.open(worldpop_path) as src:
        for _, site in sites_df.iterrows():
            try:
                # Create buffer around site
                lon, lat = site['longitude'], site['latitude']
                
                # Simple bounding box approach
                minx, miny = lon - buffer_deg, lat - buffer_deg
                maxx, maxy = lon + buffer_deg, lat + buffer_deg
                
                # Get window
                window = rasterio.windows.from_bounds(minx, miny, maxx, maxy, src.transform)
                
                # Read data
                data = src.read(1, window=window)
                
                # Sum population (WorldPop gives people per pixel)
                pop = np.nansum(data[data != src.nodata])
                
                results.append({
                    'site_id': site['site_id'],
                    'intervention_type': site['intervention_type'],
                    'latitude': lat,
                    'longitude': lon,
                    'population_in_buffer': pop,
                    'buffer_m': buffer_m
                })
                
            except Exception as e:
                continue
    
    return pd.DataFrame(results)


def estimate_population_from_viirs(sites_df, viirs_rasters, reference_pop=GAZA_TOTAL_POP_2023):
    """
    Use VIIRS nightlights as a proxy for population distribution.
    
    Method: Population ∝ Nightlight intensity
    Calibrated to total Gaza population.
    """
    print("\n  Using VIIRS as population proxy...")
    
    # Get most recent pre-conflict VIIRS
    pre_conflict_dates = [d for d in viirs_rasters.keys() if d < datetime(2023, 10, 1)]
    
    if not pre_conflict_dates:
        print("    ⚠ No pre-conflict VIIRS data")
        return None
    
    latest_pre = max(pre_conflict_dates)
    viirs_path = viirs_rasters[latest_pre]
    
    print(f"    Using VIIRS from {latest_pre.strftime('%Y-%m')}")
    
    buffer_deg = POPULATION_BUFFER_M * DEG_PER_M
    
    results = []
    total_viirs = 0
    site_viirs = []
    
    with rasterio.open(viirs_path) as src:
        # First pass: get total VIIRS for Gaza
        full_data = src.read(1)
        full_data[full_data == src.nodata] = 0
        total_viirs = np.sum(full_data[full_data > 0])
        
        # Second pass: get VIIRS at each site
        for _, site in sites_df.iterrows():
            try:
                lon, lat = site['longitude'], site['latitude']
                
                minx, miny = lon - buffer_deg, lat - buffer_deg
                maxx, maxy = lon + buffer_deg, lat + buffer_deg
                
                window = rasterio.windows.from_bounds(minx, miny, maxx, maxy, src.transform)
                data = src.read(1, window=window)
                data[data == src.nodata] = 0
                
                site_light = np.sum(data[data > 0])
                
                # Population estimate = (site_light / total_light) * total_pop
                # But we need to account for overlap between sites
                pop_share = site_light / total_viirs if total_viirs > 0 else 0
                pop_estimate = pop_share * reference_pop
                
                results.append({
                    'site_id': site['site_id'],
                    'intervention_type': site['intervention_type'],
                    'viirs_sum': site_light,
                    'viirs_pop_share': pop_share,
                    'population_estimate': pop_estimate,
                    'method': 'viirs_proxy'
                })
                
            except Exception as e:
                continue
    
    return pd.DataFrame(results)


def estimate_structures_from_sar(sites_df, sar_rasters):
    """
    Use SAR backscatter to estimate number of structures.
    
    Higher SAR VH = more vertical structures.
    This is a relative measure, not absolute count.
    """
    print("\n  Estimating structures from SAR...")
    
    # Get latest SAR
    if not sar_rasters:
        return None
    
    latest_date = max(sar_rasters.keys())
    sar_path = sar_rasters[latest_date]
    
    buffer_deg = POPULATION_BUFFER_M * DEG_PER_M
    
    results = []
    
    with rasterio.open(sar_path) as src:
        for _, site in sites_df.iterrows():
            try:
                lon, lat = site['longitude'], site['latitude']
                
                minx, miny = lon - buffer_deg, lat - buffer_deg
                maxx, maxy = lon + buffer_deg, lat + buffer_deg
                
                window = rasterio.windows.from_bounds(minx, miny, maxx, maxy, src.transform)
                data = src.read(1, window=window)
                data = data[data != src.nodata]
                
                if len(data) > 0:
                    results.append({
                        'site_id': site['site_id'],
                        'intervention_type': site['intervention_type'],
                        'sar_mean': np.mean(data),
                        'sar_max': np.max(data),
                        'sar_std': np.std(data),
                    })
                
            except:
                continue
    
    return pd.DataFrame(results)


# =============================================================================
# BENEFICIARY CALCULATIONS
# =============================================================================

def calculate_beneficiaries(did_results, population_data, sites_df):
    """
    Translate DiD findings into beneficiary estimates.
    
    Methods:
    1. Direct population in treatment areas
    2. Economic benefit (VIIRS → GDP → income)
    3. Infrastructure benefit (SAR → structures → households)
    """
    print("\n--- Calculating Beneficiary Estimates ---")
    
    results = {}
    
    # Load DiD results
    if isinstance(did_results, str) or isinstance(did_results, Path):
        did_df = pd.read_csv(did_results)
    else:
        did_df = did_results
    
    # =========================================================================
    # METHOD 1: Direct Population Count
    # =========================================================================
    if population_data is not None and not population_data.empty:
        total_pop = population_data['population_in_buffer'].sum()
        
        # Adjust for overlap (sites may have overlapping buffers)
        n_sites = len(sites_df)
        avg_pop_per_site = total_pop / n_sites
        
        # Estimate unique beneficiaries (rough adjustment for overlap)
        # Assume ~30% overlap on average in densely clustered areas
        overlap_factor = 0.7
        unique_beneficiaries = total_pop * overlap_factor
        
        results['direct_population'] = {
            'total_in_buffers': total_pop,
            'unique_estimate': unique_beneficiaries,
            'avg_per_site': avg_pop_per_site,
            'n_sites': n_sites,
            'buffer_m': POPULATION_BUFFER_M,
            'method': 'WorldPop extraction'
        }
        
        print(f"\n  Direct Population Method:")
        print(f"    Total in {POPULATION_BUFFER_M}m buffers: {total_pop:,.0f}")
        print(f"    Estimated unique beneficiaries: {unique_beneficiaries:,.0f}")
        print(f"    Average per site: {avg_pop_per_site:,.0f}")
    
    # =========================================================================
    # METHOD 2: Economic Benefit (VIIRS → GDP)
    # =========================================================================
    viirs_results = did_df[did_df['indicator'] == 'log_viirs']
    
    if not viirs_results.empty:
        # Get overall treatment effect
        overall = viirs_results[viirs_results['intervention'] == 'Overall (Treated vs Control)']
        
        if not overall.empty:
            coef = overall['coef'].values[0]
            pct_change = (np.exp(coef) - 1) * 100
            
            # GDP impact using different elasticities
            elasticities = {'low': 0.2, 'medium': 0.3, 'high': 0.5}
            
            results['economic_benefit'] = {
                'viirs_coef': coef,
                'viirs_pct_change': pct_change,
                'gdp_impacts': {}
            }
            
            print(f"\n  Economic Benefit Method (VIIRS):")
            print(f"    Log VIIRS increase: {coef:.4f} ({pct_change:.1f}%)")
            
            for name, elast in elasticities.items():
                gdp_impact_pct = pct_change * elast
                
                # Translate to per-capita income impact
                # Gaza GDP per capita ~$1,000-1,500 (pre-conflict)
                gdp_per_capita = 1200  # USD
                income_gain_per_person = gdp_per_capita * (gdp_impact_pct / 100)
                
                results['economic_benefit']['gdp_impacts'][name] = {
                    'elasticity': elast,
                    'gdp_impact_pct': gdp_impact_pct,
                    'income_gain_per_person': income_gain_per_person
                }
                
                print(f"    Elasticity {elast}: GDP +{gdp_impact_pct:.1f}% → ${income_gain_per_person:.0f}/person/year")
    
    # =========================================================================
    # METHOD 3: Infrastructure Benefit (SAR → Structures)
    # =========================================================================
    sar_results = did_df[did_df['indicator'] == 'sar_vh']
    
    if not sar_results.empty:
        overall = sar_results[sar_results['intervention'] == 'Overall (Treated vs Control)']
        
        if not overall.empty:
            sar_coef = overall['coef'].values[0]
            
            # SAR VH increase of ~1 dB corresponds roughly to structural changes
            # This is a rough calibration - would need ground truth for precision
            # Assume 1 dB ≈ 10% more detectable structures
            structure_increase_pct = sar_coef * 10
            
            # Estimate households benefiting
            # If there are ~500,000 structures in Gaza (rough estimate)
            # And treatment covers ~10% of area
            estimated_structures_treated = 50000  # Rough estimate
            structures_improved = estimated_structures_treated * (structure_increase_pct / 100)
            households_benefiting = structures_improved
            people_benefiting = households_benefiting * GAZA_AVG_HOUSEHOLD_SIZE
            
            results['infrastructure_benefit'] = {
                'sar_coef': sar_coef,
                'structure_increase_pct': structure_increase_pct,
                'structures_improved_estimate': structures_improved,
                'households_benefiting': households_benefiting,
                'people_benefiting': people_benefiting,
                'method': 'SAR backscatter proxy'
            }
            
            print(f"\n  Infrastructure Benefit Method (SAR):")
            print(f"    SAR VH increase: {sar_coef:.2f} dB")
            print(f"    Estimated structure improvement: {structure_increase_pct:.0f}%")
            print(f"    Households benefiting: ~{households_benefiting:,.0f}")
            print(f"    People benefiting: ~{people_benefiting:,.0f}")
    
    return results


def calculate_cost_effectiveness(beneficiary_estimates, total_undp_budget=None):
    """
    Calculate cost-effectiveness metrics.
    
    Requires knowing total UNDP budget for Gaza interventions.
    """
    if total_undp_budget is None:
        # Placeholder - would need actual budget data
        total_undp_budget = 50_000_000  # $50M as example
    
    results = {
        'total_budget': total_undp_budget,
        'metrics': {}
    }
    
    # Cost per beneficiary (direct population)
    if 'direct_population' in beneficiary_estimates:
        beneficiaries = beneficiary_estimates['direct_population']['unique_estimate']
        cost_per_beneficiary = total_undp_budget / beneficiaries
        
        results['metrics']['cost_per_beneficiary'] = {
            'value': cost_per_beneficiary,
            'unit': 'USD per person',
            'beneficiaries': beneficiaries
        }
    
    # Cost per household with improved infrastructure
    if 'infrastructure_benefit' in beneficiary_estimates:
        households = beneficiary_estimates['infrastructure_benefit']['households_benefiting']
        if households > 0:
            cost_per_household = total_undp_budget / households
            
            results['metrics']['cost_per_household_infrastructure'] = {
                'value': cost_per_household,
                'unit': 'USD per household',
                'households': households
            }
    
    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_beneficiary_summary_plot(beneficiary_estimates, sites_df, output_path):
    """Create visual summary of beneficiary analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Population by intervention type
    ax1 = axes[0, 0]
    if 'direct_population' in beneficiary_estimates:
        # Load population data for this
        pop_by_type = sites_df.groupby('intervention_type').size()
        avg_pop = beneficiary_estimates['direct_population']['avg_per_site']
        
        estimated_pop = pop_by_type * avg_pop
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(estimated_pop)))
        bars = ax1.barh(estimated_pop.index, estimated_pop.values, color=colors, edgecolor='black')
        
        ax1.set_xlabel('Estimated Beneficiaries', fontsize=11)
        ax1.set_title('Beneficiaries by Intervention Type\n(Population within 500m buffer)', 
                     fontsize=12, fontweight='bold')
        ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
        
        # Add value labels
        for bar, val in zip(bars, estimated_pop.values):
            ax1.annotate(f'{val/1000:.0f}K', xy=(val, bar.get_y() + bar.get_height()/2),
                        xytext=(5, 0), textcoords='offset points', va='center', fontsize=10)
    
    # Plot 2: Economic impact scenarios
    ax2 = axes[0, 1]
    if 'economic_benefit' in beneficiary_estimates:
        gdp_impacts = beneficiary_estimates['economic_benefit']['gdp_impacts']
        
        scenarios = list(gdp_impacts.keys())
        impacts = [gdp_impacts[s]['gdp_impact_pct'] for s in scenarios]
        
        colors = ['#3498db', '#27ae60', '#e74c3c']
        bars = ax2.bar(scenarios, impacts, color=colors, edgecolor='black')
        
        ax2.set_ylabel('GDP Impact (%)', fontsize=11)
        ax2.set_title('Economic Impact by Elasticity Scenario\n(Nightlights → GDP)', 
                     fontsize=12, fontweight='bold')
        ax2.set_ylim(0, max(impacts) * 1.3)
        
        for bar, val in zip(bars, impacts):
            ax2.annotate(f'+{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                        xytext=(0, 5), textcoords='offset points', ha='center', fontsize=12, fontweight='bold')
    
    # Plot 3: Site distribution map (simplified)
    ax3 = axes[1, 0]
    if 'intervention_type' in sites_df.columns:
        types = sites_df['intervention_type'].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(types)))
        
        for i, t in enumerate(types):
            subset = sites_df[sites_df['intervention_type'] == t]
            ax3.scatter(subset['longitude'], subset['latitude'], 
                       c=[colors[i]], label=t, alpha=0.7, s=50, edgecolor='white')
        
        ax3.set_xlabel('Longitude', fontsize=11)
        ax3.set_ylabel('Latitude', fontsize=11)
        ax3.set_title('UNDP Intervention Sites by Type', fontsize=12, fontweight='bold')
        ax3.legend(loc='upper left', fontsize=9)
        ax3.set_aspect('equal')
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = "BENEFICIARY ANALYSIS SUMMARY\n" + "="*40 + "\n\n"
    
    if 'direct_population' in beneficiary_estimates:
        dp = beneficiary_estimates['direct_population']
        summary_text += f"DIRECT POPULATION METHOD:\n"
        summary_text += f"  • Sites analyzed: {dp['n_sites']}\n"
        summary_text += f"  • Buffer radius: {dp['buffer_m']}m\n"
        summary_text += f"  • Total in buffers: {dp['total_in_buffers']:,.0f}\n"
        summary_text += f"  • Unique beneficiaries: ~{dp['unique_estimate']:,.0f}\n\n"
    
    if 'economic_benefit' in beneficiary_estimates:
        eb = beneficiary_estimates['economic_benefit']
        summary_text += f"ECONOMIC BENEFIT:\n"
        summary_text += f"  • Nightlight increase: {eb['viirs_pct_change']:.1f}%\n"
        for name, data in eb['gdp_impacts'].items():
            summary_text += f"  • GDP impact ({name}): +{data['gdp_impact_pct']:.1f}%\n"
        summary_text += "\n"
    
    if 'infrastructure_benefit' in beneficiary_estimates:
        ib = beneficiary_estimates['infrastructure_benefit']
        summary_text += f"INFRASTRUCTURE BENEFIT (SAR):\n"
        summary_text += f"  • SAR increase: +{ib['sar_coef']:.2f} dB\n"
        summary_text += f"  • Households benefiting: ~{ib['households_benefiting']:,.0f}\n"
        summary_text += f"  • People benefiting: ~{ib['people_benefiting']:,.0f}\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")


def create_displacement_timeline(displacement_data, output_path):
    """Create visualization of displacement over time."""
    
    estimates = displacement_data['estimates']
    df = pd.DataFrame(estimates)
    df['date'] = pd.to_datetime(df['date'])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: IDP count over time
    ax1 = axes[0]
    ax1.fill_between(df['date'], 0, df['idps'], alpha=0.3, color='#e74c3c')
    ax1.plot(df['date'], df['idps'], 'o-', color='#e74c3c', linewidth=2, markersize=8)
    
    ax1.axvline(pd.Timestamp('2023-10-07'), color='black', linestyle='--', 
                linewidth=2, label='Conflict Start')
    
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('Internally Displaced Persons', fontsize=11)
    ax1.set_title('Gaza Displacement Over Time', fontsize=13, fontweight='bold')
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: IDP percentage
    ax2 = axes[1]
    ax2.bar(df['date'], df['idp_pct'], width=20, color='#e74c3c', edgecolor='black', alpha=0.8)
    
    ax2.axhline(50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
    
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('% Population Displaced', fontsize=11)
    ax2.set_title('Displacement Rate', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print(" BENEFICIARY AND POPULATION IMPACT ANALYSIS")
    print("=" * 70)
    
    if not HAS_DEPS:
        print("\n⚠ Missing dependencies. Exiting.")
        return
    
    # Load sites
    print("\n--- Loading UNDP Sites ---")
    sites_df = pd.read_csv(SITES_CSV)
    print(f"  Sites: {len(sites_df)}")
    print(f"  Types: {sites_df['intervention_type'].value_counts().to_dict()}")
    
    # Gather displacement data
    displacement_data = get_displacement_estimates()
    
    # Load WorldPop data
    print("\n--- Loading Population Data ---")
    worldpop_files = load_worldpop_data()
    
    if worldpop_files:
        print(f"  WorldPop files found: {list(worldpop_files.keys())}")
        
        # Use most recent year available
        latest_year = max(worldpop_files.keys())
        population_data = estimate_population_at_sites(sites_df, worldpop_files[latest_year])
        
        if population_data is not None:
            print(f"  Population extracted for {len(population_data)} sites")
            print(f"  Total in buffers: {population_data['population_in_buffer'].sum():,.0f}")
    else:
        print("  ⚠ No WorldPop files found. Using VIIRS proxy instead.")
        population_data = None
    
    # Load VIIRS for proxy estimation
    print("\n--- Loading VIIRS for Population Proxy ---")
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
        viirs_pop = estimate_population_from_viirs(sites_df, viirs_rasters)
        
        if viirs_pop is not None and population_data is None:
            # Use VIIRS as primary population estimate
            population_data = viirs_pop.rename(columns={'population_estimate': 'population_in_buffer'})
    
    # Load SAR for structure estimation
    print("\n--- Loading SAR for Structure Proxy ---")
    sar_rasters = {}
    for f in RASTER_DIR.glob("*sar*vh*.tif"):
        for part in f.stem.split('_'):
            try:
                if len(part) == 6 and part.isdigit():
                    d = datetime.strptime(part, "%Y%m")
                    sar_rasters[d] = f
                    break
            except:
                continue
    
    if sar_rasters:
        sar_structures = estimate_structures_from_sar(sites_df, sar_rasters)
    
    # Calculate beneficiaries using DiD results
    print("\n--- Calculating Beneficiary Estimates ---")
    did_results_path = TABLES_DIR / "did_comprehensive_results.csv"
    
    if did_results_path.exists():
        beneficiary_estimates = calculate_beneficiaries(
            did_results_path, 
            population_data, 
            sites_df
        )
        
        # Cost-effectiveness (if budget known)
        # cost_effectiveness = calculate_cost_effectiveness(beneficiary_estimates, total_undp_budget=50_000_000)
    else:
        print("  ⚠ DiD results not found. Run 02_did_comprehensive.py first.")
        beneficiary_estimates = {}
    
    # Create visualizations
    print("\n--- Creating Visualizations ---")
    
    if beneficiary_estimates:
        create_beneficiary_summary_plot(beneficiary_estimates, sites_df, 
                                        FIGURES_DIR / "beneficiary_summary.png")
    
    create_displacement_timeline(displacement_data, FIGURES_DIR / "displacement_timeline.png")
    
    # Save results
    print("\n--- Saving Results ---")
    
    if population_data is not None:
        population_data.to_csv(TABLES_DIR / "population_at_sites.csv", index=False)
        print(f"  ✓ Saved: population_at_sites.csv")
    
    # Save beneficiary estimates as JSON
    import json
    
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return obj
    
    if beneficiary_estimates:
        serializable = json.loads(
            json.dumps(beneficiary_estimates, default=convert_to_serializable)
        )
        with open(TABLES_DIR / "beneficiary_estimates.json", 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"  ✓ Saved: beneficiary_estimates.json")
    
    # Summary
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    
    print("""
  DATA SOURCES FOR POPULATION/DISPLACEMENT:
  ─────────────────────────────────────────
  1. WorldPop: Baseline population (2020-2023)
  2. VIIRS: Nightlights as population proxy
  3. SAR: Structure detection for infrastructure
  4. OCHA/HDX: Displacement tracking (requires manual download)
  5. UNRWA: Shelter population data
  
  BENEFICIARY CALCULATION METHODS:
  ─────────────────────────────────
  1. Direct: Population within 500m of intervention sites
  2. Economic: VIIRS → GDP impact per person
  3. Infrastructure: SAR → households with improved structures
  
  LIMITATIONS:
  ────────────
  • Population data is 2023 baseline (pre-conflict)
  • Displacement makes current population uncertain
  • Economic impact assumes standard elasticities
  • Infrastructure proxy needs ground-truth calibration
    """)
    
    print("=" * 70)
    print(" ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
