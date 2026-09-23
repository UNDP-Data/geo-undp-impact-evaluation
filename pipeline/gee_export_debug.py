#!/usr/bin/env python3
"""
pipeline/gee_export_debug.py
============================
Debugged GEE export using centroids (more reliable than polygons).

This version:
1. Uses admin unit centroids instead of complex polygons
2. Buffers centroids to create sample areas
3. Has detailed error reporting
4. Tests connectivity before full export

Usage:
    python -m pipeline.gee_export_debug --config config/myanmar.yaml
    python -m pipeline.gee_export_debug --config config/myanmar.yaml --test
"""

import ee
import os
import json
import pandas as pd
import geopandas as gpd
from pathlib import Path
from datetime import datetime
import time
import argparse
import yaml

def load_config(config_path):
    """Load YAML config."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class GEEExporterDebug:
    """GEE Exporter using centroids for reliability."""
    
    def __init__(self, config_path: str):
        """Initialize with config."""
        self.config = load_config(config_path)
        self.country_name = self.config.get('country_name', 'Unknown')
        self.country_code = self.config.get('country_code', 'XXX').lower()
        self.study_start = self.config.get('study_period_start')
        self.study_end = self.config.get('study_period_end')
        
        boundary_config = self.config.get('boundaries', {})
        boundary_path = Path(boundary_config.get('file_path', ''))
        self.id_col = boundary_config.get('id_column', 'adm3_pcode')
        self.name_col = boundary_config.get('name_column', 'adm3_name')
        
        # Initialize GEE
        print("Initializing GEE...")
        _gee_project = os.environ.get("GEE_PROJECT", "") or None
        _sa_json = os.environ.get("GEE_SERVICE_ACCOUNT_JSON", "")
        if not _sa_json:
            for _cand in ["/app/gee_service_account.json", "gee_service_account.json",
                          os.path.join(os.path.expanduser("~"), ".undp_dashboard", "gee_service_account.json")]:
                if os.path.exists(_cand):
                    _sa_json = _cand
                    break
        try:
            if _sa_json and os.path.exists(_sa_json):
                with open(_sa_json) as _f:
                    _sa_email = json.load(_f).get("client_email")
                _creds = ee.ServiceAccountCredentials(_sa_email, _sa_json)
                if _gee_project:
                    ee.Initialize(_creds, project=_gee_project)
                else:
                    ee.Initialize(_creds)
                print("✓ GEE Initialized (service account)")
            else:
                if _gee_project:
                    ee.Initialize(project=_gee_project)
                else:
                    ee.Initialize()
                print("✓ GEE Initialized")
        except Exception as _e:
            print("GEE init failed (" + str(_e) + "); trying interactive auth...")
            ee.Authenticate()
            if _gee_project:
                ee.Initialize(project=_gee_project)
            else:
                ee.Initialize()
            print("✓ GEE Authenticated")
        
        # Load boundaries
        print(f"\nLoading boundaries: {boundary_path}")
        self.gdf = gpd.read_file(boundary_path)
        if self.gdf.crs != "EPSG:4326":
            self.gdf = self.gdf.to_crs("EPSG:4326")
        
        print(f"  Admin units: {len(self.gdf)}")
        print(f"  ID column: {self.id_col}")
        print(f"  Name column: {self.name_col}")
        
        # Use centroids - check if we have center_lat/lon columns
        if 'center_lat' in self.gdf.columns and 'center_lon' in self.gdf.columns:
            print("  ✓ Using pre-computed centroids (center_lat, center_lon)")
            self.gdf['centroid_lat'] = self.gdf['center_lat']
            self.gdf['centroid_lon'] = self.gdf['center_lon']
        else:
            print("  Computing centroids from geometry...")
            centroids = self.gdf.geometry.centroid
            self.gdf['centroid_lat'] = centroids.y
            self.gdf['centroid_lon'] = centroids.x
        
        # Output directory
        self.output_dir = Path(f"data/{self.country_code}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"GEE EXPORT (DEBUG): {self.country_name}")
        print(f"{'='*60}")
        print(f"Study Period: {self.study_start} to {self.study_end}")
        print(f"Output: {self.output_dir}")
    
    def test_connection(self):
        """Test GEE connectivity with simple queries."""
        print("\n--- Testing GEE Connection ---")
        
        # Get a test point (first admin unit centroid)
        test_lat = self.gdf['centroid_lat'].iloc[0]
        test_lon = self.gdf['centroid_lon'].iloc[0]
        test_name = self.gdf[self.name_col].iloc[0]
        
        print(f"  Test location: {test_name} ({test_lat:.4f}, {test_lon:.4f})")
        
        # Create point with 5km buffer
        point = ee.Geometry.Point([test_lon, test_lat])
        buffer = point.buffer(5000)  # 5km buffer
        
        # Test VIIRS monthly
        print("\n  Testing VIIRS monthly...")
        try:
            viirs = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG") \
                .filterDate('2024-01-01', '2024-02-01') \
                .select('avg_rad')
            
            count = viirs.size().getInfo()
            print(f"    Images found: {count}")
            
            if count > 0:
                stats = viirs.first().reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=buffer,
                    scale=500
                ).getInfo()
                print(f"    Mean nightlight: {stats.get('avg_rad', 'N/A')}")
                
                if stats.get('avg_rad') is not None:
                    print("    ✓ VIIRS working!")
                else:
                    print("    ⚠ VIIRS returned None (might be ocean/cloud)")
            else:
                print("    ⚠ No VIIRS images found for this date")
                
        except Exception as e:
            print(f"    ✗ VIIRS error: {e}")
        
        # Test MODIS NDVI
        print("\n  Testing MODIS NDVI...")
        try:
            modis = ee.ImageCollection("MODIS/061/MOD13A1") \
                .filterDate('2024-01-01', '2024-02-01') \
                .select('NDVI')
            
            count = modis.size().getInfo()
            print(f"    Images found: {count}")
            
            if count > 0:
                stats = modis.first().reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=buffer,
                    scale=500
                ).getInfo()
                ndvi_raw = stats.get('NDVI')
                if ndvi_raw is not None:
                    print(f"    Mean NDVI: {ndvi_raw * 0.0001:.4f}")
                    print("    ✓ MODIS working!")
                else:
                    print("    ⚠ MODIS returned None")
                    
        except Exception as e:
            print(f"    ✗ MODIS error: {e}")
        
        # Test Sentinel-1 SAR
        print("\n  Testing Sentinel-1 SAR...")
        try:
            s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
                .filterDate('2024-01-01', '2024-02-01') \
                .filterBounds(buffer) \
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
                .filter(ee.Filter.eq('instrumentMode', 'IW'))
            
            count = s1.size().getInfo()
            print(f"    Images found: {count}")
            
            if count > 0:
                stats = s1.first().reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=buffer,
                    scale=100
                ).getInfo()
                print(f"    VV: {stats.get('VV', 'N/A')}, VH: {stats.get('VH', 'N/A')}")
                print("    ✓ SAR working!")
                
        except Exception as e:
            print(f"    ✗ SAR error: {e}")
        
        print("\n  ✓ Connection test complete")
        return True
    
    def export_centroids(self, test_mode=False):
        """Export using centroid-based sampling."""
        
        print("\n--- Exporting Data (Centroid Method) ---")
        
        # Create batches
        start = pd.Timestamp(self.study_start)
        end = pd.Timestamp(self.study_end)
        
        months = pd.date_range(start, end, freq='MS')
        print(f"  Months to process: {len(months)}")
        
        if test_mode:
            months = months[:2]  # Just first 2 months for testing
            print(f"  TEST MODE: Processing only {len(months)} months")
        
        all_results = []
        
        for month_start in months:
            month_end = month_start + pd.DateOffset(months=1)
            month_str = month_start.strftime('%Y-%m')
            
            print(f"\n  > {month_str}...", end=" ", flush=True)
            
            month_results = self._process_month(
                month_start.strftime('%Y-%m-%d'),
                month_end.strftime('%Y-%m-%d')
            )
            
            if month_results:
                for r in month_results:
                    r['period_start'] = month_start.strftime('%Y-%m-01')
                    r['period'] = month_str
                all_results.extend(month_results)
                print(f"✓ ({len(month_results)} units)")
            else:
                print("no data")
        
        # Save results
        if all_results:
            df = pd.DataFrame(all_results)
            
            # Save full file
            output_file = self.output_dir / f"gee_panel_{self.country_code}_centroid.csv"
            df.to_csv(output_file, index=False)
            
            print(f"\n{'='*60}")
            print(f"✓ EXPORT COMPLETE")
            print(f"{'='*60}")
            print(f"  Rows: {len(df):,}")
            print(f"  File: {output_file}")
            print(f"  Columns: {list(df.columns)}")
            
            # Summary stats
            for col in ['nightlights', 'ndvi', 'vv', 'vh']:
                if col in df.columns:
                    valid = df[col].notna().sum()
                    print(f"  {col}: {valid}/{len(df)} valid ({100*valid/len(df):.0f}%)")
            
            return df
        else:
            print("\n  ✗ No data exported")
            return None
    
    def _process_month(self, start_date, end_date):
        """Process a single month using centroids."""
        
        results = []
        
        # Create list of centroid geometries with buffers (2km)
        # Process in chunks to avoid memory issues
        chunk_size = 50
        n_chunks = (len(self.gdf) + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min((chunk_idx + 1) * chunk_size, len(self.gdf))
            chunk_gdf = self.gdf.iloc[chunk_start:chunk_end]
            
            # Build FeatureCollection from centroids
            features = []
            for idx, row in chunk_gdf.iterrows():
                point = ee.Geometry.Point([row['centroid_lon'], row['centroid_lat']])
                buffer = point.buffer(2000)  # 2km buffer
                feat = ee.Feature(buffer, {
                    'admin_id': str(row[self.id_col]),
                    'admin_name': str(row[self.name_col])
                })
                features.append(feat)
            
            fc = ee.FeatureCollection(features)
            
            # Get stats for this chunk
            chunk_results = self._get_stats_for_fc(fc, start_date, end_date)
            results.extend(chunk_results)
        
        return results if results else None
    
    def _get_stats_for_fc(self, fc, start_date, end_date):
        """Get satellite stats for a FeatureCollection."""
        
        results = []
        
        # Initialize with admin IDs
        fc_info = fc.getInfo()
        for feat in fc_info['features']:
            props = feat['properties']
            results.append({
                'admin_id': props['admin_id'],
                'admin_name': props['admin_name']
            })
        
        has_data = False
        
        # VIIRS
        try:
            viirs = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG") \
                .filterDate(start_date, end_date) \
                .select('avg_rad')
            
            if viirs.size().getInfo() > 0:
                viirs_mean = viirs.mean()
                viirs_stats = viirs_mean.reduceRegions(
                    collection=fc,
                    reducer=ee.Reducer.mean(),
                    scale=500
                ).getInfo()
                
                for feat in viirs_stats['features']:
                    props = feat['properties']
                    admin_id = props['admin_id']
                    result = next((r for r in results if r['admin_id'] == admin_id), None)
                    if result and props.get('mean') is not None:
                        result['nightlights'] = props['mean']
                        has_data = True
        except:
            pass
        
        # MODIS NDVI
        try:
            modis = ee.ImageCollection("MODIS/061/MOD13A1") \
                .filterDate(start_date, end_date) \
                .select('NDVI')
            
            if modis.size().getInfo() > 0:
                modis_mean = modis.mean().multiply(0.0001)
                modis_stats = modis_mean.reduceRegions(
                    collection=fc,
                    reducer=ee.Reducer.mean(),
                    scale=500
                ).getInfo()
                
                for feat in modis_stats['features']:
                    props = feat['properties']
                    admin_id = props['admin_id']
                    result = next((r for r in results if r['admin_id'] == admin_id), None)
                    if result and props.get('mean') is not None:
                        result['ndvi'] = props['mean']
                        has_data = True
        except:
            pass
        
        # Sentinel-1 SAR
        try:
            # Get bounding box for the FC
            bounds = fc.geometry().bounds()
            
            s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
                .filterDate(start_date, end_date) \
                .filterBounds(bounds) \
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
                .filter(ee.Filter.eq('instrumentMode', 'IW')) \
                .select(['VV', 'VH'])
            
            if s1.size().getInfo() > 0:
                s1_mean = s1.mean()
                s1_stats = s1_mean.reduceRegions(
                    collection=fc,
                    reducer=ee.Reducer.mean(),
                    scale=100
                ).getInfo()
                
                for feat in s1_stats['features']:
                    props = feat['properties']
                    admin_id = props['admin_id']
                    result = next((r for r in results if r['admin_id'] == admin_id), None)
                    if result:
                        if props.get('VV') is not None:
                            result['vv'] = props['VV']
                            has_data = True
                        if props.get('VH') is not None:
                            result['vh'] = props['VH']
                            has_data = True
        except:
            pass
        
        # CHIRPS rainfall
        try:
            chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/PENTAD") \
                .filterDate(start_date, end_date) \
                .select('precipitation')
            
            if chirps.size().getInfo() > 0:
                chirps_sum = chirps.sum()
                chirps_stats = chirps_sum.reduceRegions(
                    collection=fc,
                    reducer=ee.Reducer.mean(),
                    scale=5000
                ).getInfo()
                
                for feat in chirps_stats['features']:
                    props = feat['properties']
                    admin_id = props['admin_id']
                    result = next((r for r in results if r['admin_id'] == admin_id), None)
                    if result and props.get('mean') is not None:
                        result['rainfall'] = props['mean']
                        has_data = True
        except:
            pass
        
        return results if has_data else []


def main():
    parser = argparse.ArgumentParser(description='Debug GEE Export')
    parser.add_argument('--config', '-c', required=True, help='Config path')
    parser.add_argument('--test', '-t', action='store_true', help='Test mode (2 months only)')
    parser.add_argument('--test-connection', action='store_true', help='Only test GEE connection')
    
    args = parser.parse_args()
    
    exporter = GEEExporterDebug(args.config)
    
    if args.test_connection:
        exporter.test_connection()
    else:
        exporter.test_connection()
        exporter.export_centroids(test_mode=args.test)


if __name__ == "__main__":
    main()
