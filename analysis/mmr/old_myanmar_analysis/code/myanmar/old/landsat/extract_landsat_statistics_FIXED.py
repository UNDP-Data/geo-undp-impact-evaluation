#!/usr/bin/env python3
"""
Step 2: Extract Landsat Statistics from Composites - CORRECTED VERSION

FIXES:
- Proper treatment assignment that doesn't mark all townships as treated
- Better handling of the spatial join
"""

import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as rxr
from rasterio.features import geometry_mask


def find_landsat_composites(landsat_dir: Path):
    """Find all Landsat composite files."""
    tif_files = list(landsat_dir.glob("LANDSAT_*.tif"))
    
    print(f"\n📁 Found {len(tif_files)} Landsat composite files")
    
    if len(tif_files) == 0:
        print("\n⚠️ No LANDSAT_*.tif files found!")
        print(f"   Check directory: {landsat_dir}")
        return []
    
    files_with_dates = []
    
    for f in tif_files:
        parts = f.stem.split('_')
        if len(parts) >= 2:
            yyyymm = parts[1]
            try:
                date = datetime.strptime(yyyymm, "%Y-%m")
                files_with_dates.append((date, f))
            except:
                print(f"   ⚠️ Could not parse date from: {f.name}")
    
    files_with_dates.sort(key=lambda x: x[0])
    
    if len(files_with_dates) > 0:
        print(f"   Date range: {files_with_dates[0][0].strftime('%Y-%m')} to {files_with_dates[-1][0].strftime('%Y-%m')}")
    
    return files_with_dates


def load_townships(boundaries_file: Path) -> gpd.GeoDataFrame:
    """Load township boundaries."""
    print(f"\n📁 Loading boundaries from: {boundaries_file}")
    gdf = gpd.read_file(boundaries_file)
    
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    else:
        gdf = gdf.to_crs(epsg=4326)
    
    # Find name column
    name_cols = ["NAME_3", "ADM3_EN", "TS", "TS_NAME", "adm3_name", "Township", "township", "NAME"]
    name_col = None
    for c in name_cols:
        if c in gdf.columns:
            name_col = c
            break
    
    if name_col is None:
        gdf["township_name"] = [f"Township_{i}" for i in range(len(gdf))]
    else:
        gdf["township_name"] = gdf[name_col].astype(str)
    
    print(f"   ✓ Loaded {len(gdf)} townships")
    
    return gdf[["township_name", "geometry"]].copy()


def merge_with_undp_FIXED(gdf: gpd.GeoDataFrame, undp_file: Path) -> gpd.GeoDataFrame:
    """
    CORRECTED: Merge with UNDP support data.
    
    CRITICAL FIX: Properly handle spatial join to avoid marking all townships as treated.
    """
    if undp_file is None or not undp_file.exists():
        print("   ⚠️ No UNDP support file. All townships marked as control.")
        gdf["treated"] = 0
        gdf["n_interventions"] = 0
        return gdf
    
    print(f"\n📁 Loading UNDP support from: {undp_file}")
    undp_gdf = gpd.read_file(undp_file)
    
    if undp_gdf.crs is None:
        undp_gdf.set_crs(epsg=4326, inplace=True)
    else:
        undp_gdf = undp_gdf.to_crs(epsg=4326)
    
    print(f"   ✓ Loaded {len(undp_gdf)} UNDP intervention points")
    
    # CRITICAL FIX: Do spatial join and count properly
    # Use sjoin to find which townships contain UNDP interventions
    joined = gpd.sjoin(undp_gdf, gdf[["geometry", "township_name"]], 
                      how="left", predicate="within")
    
    # Count interventions per township
    intervention_counts = joined.groupby('township_name').size().reset_index(name='n_interventions')
    
    # Merge back to townships
    gdf = gdf.merge(intervention_counts, on='township_name', how='left')
    gdf['n_interventions'] = gdf['n_interventions'].fillna(0).astype(int)
    
    # Define treatment: any township with interventions = treated
    gdf['treated'] = (gdf['n_interventions'] > 0).astype(int)
    
    print(f"   ✓ Treatment assignment:")
    print(f"      Treated: {gdf['treated'].sum()}")
    print(f"      Control: {(gdf['treated']==0).sum()}")
    
    # Show top treated townships
    if gdf['treated'].sum() > 0:
        print(f"\n   Top treated townships:")
        top_treated = gdf[gdf['treated'] == 1].nlargest(10, 'n_interventions')[['township_name', 'n_interventions']]
        for _, row in top_treated.iterrows():
            print(f"      {row['township_name']:20s}: {row['n_interventions']:3d} interventions")
    
    return gdf


def load_landsat_composite(tif_file: Path):
    """Load a Landsat composite."""
    raster = rxr.open_rasterio(tif_file, masked=True)
    return raster


def compute_indices_from_raster(raster):
    """
    Compute spectral indices from 4-band raster.
    Bands: 0=red, 1=nir, 2=swir1, 3=swir2
    """
    red = raster[0, :, :].values
    nir = raster[1, :, :].values
    swir1 = raster[2, :, :].values
    swir2 = raster[3, :, :].values
    
    # NDVI
    ndvi = np.where((nir + red) != 0, (nir - red) / (nir + red), np.nan)
    
    # NBR
    nbr = np.where((nir + swir2) != 0, (nir - swir2) / (nir + swir2), np.nan)
    
    # NDMI
    ndmi = np.where((nir + swir1) != 0, (nir - swir1) / (nir + swir1), np.nan)
    
    # NDBI (Normalized Difference Built-up Index) - BETTER than simple threshold
    ndbi = np.where((swir1 + nir) != 0, (swir1 - nir) / (swir1 + nir), np.nan)
    
    # Built-up approximation (keep original for comparison)
    builtup_simple = np.where((ndvi < 0.2) & (swir1 > 0.1), 1.0, 0.0)
    
    return {
        'red': red,
        'nir': nir,
        'swir1': swir1,
        'swir2': swir2,
        'ndvi': ndvi,
        'nbr': nbr,
        'ndmi': ndmi,
        'ndbi': ndbi,  # ADDED: Better built-up index
        'builtup': builtup_simple  # Keep original
    }


def calculate_mean_index(index_array, raster, gdf, index_name):
    """Calculate mean value for areas in GeoDataFrame."""
    try:
        mask = geometry_mask(
            gdf.geometry,
            out_shape=(raster.shape[1], raster.shape[2]),
            transform=raster.rio.transform(),
            invert=True
        )
        
        values = index_array[mask]
        values = values[~np.isnan(values)]
        
        if len(values) > 0:
            return {
                f'{index_name}_mean': float(np.mean(values)),
                f'{index_name}_std': float(np.std(values)),
                'n_valid_pixels': int(len(values))
            }
        else:
            return {
                f'{index_name}_mean': np.nan,
                f'{index_name}_std': np.nan,
                'n_valid_pixels': 0
            }
    
    except Exception as e:
        print(f"      Error calculating {index_name}: {e}")
        return {
            f'{index_name}_mean': np.nan,
            f'{index_name}_std': np.nan,
            'n_valid_pixels': 0
        }


def extract_statistics_for_month(raster, townships_gdf, date):
    """Extract statistics for all townships for one month."""
    # Compute indices
    indices = compute_indices_from_raster(raster)
    
    # Extract for each township
    results = []
    
    for idx, row in townships_gdf.iterrows():
        township = row['township_name']
        treated = row.get('treated', 0)
        n_interventions = row.get('n_interventions', 0)
        geom = row['geometry']
        
        # Create single-township GeoDataFrame
        township_gdf = gpd.GeoDataFrame([row], geometry=[geom], crs=townships_gdf.crs)
        
        # Calculate statistics for each index
        stats = {
            'township_name': township, 
            'treated': treated, 
            'n_interventions': n_interventions,
            'date': date
        }
        
        for index_name, index_array in indices.items():
            index_stats = calculate_mean_index(index_array, raster, township_gdf, index_name)
            stats.update(index_stats)
        
        results.append(stats)
    
    return results


def extract_all_statistics(
    landsat_dir: Path,
    boundaries_file: Path,
    undp_file: Path,
    output_dir: Path,
):
    """Main extraction function with FIXED treatment assignment."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("EXTRACTING LANDSAT STATISTICS - CORRECTED VERSION")
    print("=" * 80)
    
    # Find composite files
    landsat_files = find_landsat_composites(landsat_dir)
    if not landsat_files:
        return
    
    # Load boundaries
    townships_gdf = load_townships(boundaries_file)
    
    # CRITICAL: Use FIXED treatment assignment
    townships_gdf = merge_with_undp_FIXED(townships_gdf, undp_file)
    
    print("\n" + "=" * 80)
    print(f"PROCESSING {len(landsat_files)} MONTHS")
    print("=" * 80)
    
    all_results = []
    
    for date, file in landsat_files:
        ym = date.strftime("%Y-%m")
        print(f"\n   📅 {ym}: {file.name}")
        
        try:
            # Load composite
            print(f"      Loading composite...", end=" ")
            raster = load_landsat_composite(file)
            print(f"✓ {raster.shape}")
            
            # Extract statistics for all townships
            print(f"      Extracting {len(townships_gdf)} townships...", end=" ")
            month_results = extract_statistics_for_month(raster, townships_gdf, date)
            print(f"✓ Done")
            
            all_results.extend(month_results)
            
        except Exception as e:
            print(f"      ✗ Error: {e}")
            continue
    
    if not all_results:
        print("\n⚠️ No data extracted")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Add temporal columns
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # Reorder columns
    cols = ['township_name', 'treated', 'n_interventions', 'year', 'month', 'date'] + \
           [c for c in df.columns if c not in ['township_name', 'treated', 'n_interventions', 'year', 'month', 'date']]
    df = df[cols]
    
    # Save
    output_csv = output_dir / 'landsat_township_monthly_FIXED.csv'
    df.to_csv(output_csv, index=False)
    
    print("\n" + "=" * 80)
    print("✅ EXTRACTION COMPLETE!")
    print("=" * 80)
    print(f"\n📊 Results:")
    print(f"   Total observations: {len(df)}")
    print(f"   Townships: {df['township_name'].nunique()}")
    print(f"   Months: {df['date'].nunique()}")
    print(f"   Treated observations: {(df['treated']==1).sum()}")
    print(f"   Control observations: {(df['treated']==0).sum()}")
    print(f"   Treated townships: {df[df['treated']==1]['township_name'].nunique()}")
    print(f"   Control townships: {df[df['treated']==0]['township_name'].nunique()}")
    
    print(f"\n💾 Saved: {output_csv}")
    
    # Show data quality
    print("\n📈 Data Quality:")
    for var in ['ndvi_mean', 'ndbi_mean', 'builtup_mean']:
        if var in df.columns:
            valid = df[var].notna().sum()
            pct = 100 * valid / len(df)
            print(f"  {var:15s}: {valid:4d} / {len(df)} ({pct:5.1f}%) valid")
    
    print("\n📈 Sample data:")
    print(df[['township_name', 'year', 'month', 'treated', 'n_interventions', 'ndvi_mean', 'ndbi_mean']].head(10))
    
    print("\n" + "=" * 80)
    print("CRITICAL FIX APPLIED:")
    print("  ✓ Treatment assignment now works correctly")
    print("  ✓ Added NDBI (better built-up index)")
    print("  ✓ Added n_interventions column (for dose-response)")
    print("=" * 80)
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Extract township statistics from Landsat composites - FIXED VERSION"
    )
    parser.add_argument("--landsat-dir", type=str, required=True)
    parser.add_argument("--boundaries-file", type=str, required=True)
    parser.add_argument("--undp-support", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)

    args = parser.parse_args()

    extract_all_statistics(
        landsat_dir=Path(args.landsat_dir),
        boundaries_file=Path(args.boundaries_file),
        undp_file=Path(args.undp_support) if args.undp_support else None,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
