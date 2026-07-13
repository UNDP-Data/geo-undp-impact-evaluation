#!/usr/bin/env python3
"""
00_data_acquisition.py
======================
Unified Data Downloader:
1. Triggers Google Earth Engine (GEE) exports.
2. Downloads ACLED Data (using OAuth Token method).
3. Downloads Meta & WFP Data (HDX) - FIXED RENAMING.
"""

import ee
import geopandas as gpd
import pandas as pd
import os
import math
import requests
import io
import glob
import shutil
from hdx.api.configuration import Configuration
from hdx.data.dataset import Dataset
from shapely.geometry import Polygon, MultiPolygon

# --- CONFIGURATION ---
# 1. GEE Settings
BOUNDARIES_FILE = "data/myanmar/myanmar_boundaries/mmr_admin3.geojson"
START_DATE = '2024-10-01'
N_MONTHS = 13
SCALE = 500 

# 2. ACLED Settings (NEW OAUTH METHOD)
ACLED_EMAIL = "hernando.gruesohurtado@spi.ox.ac.uk"
ACLED_PASSWORD = "your_password"
TARGET_COUNTRY = "Myanmar"
ACLED_START_DATE = "2024-01-01"

# 3. Output
EXTERNAL_DIR = "data/myanmar/external"

# --- PART 1: GEE EXPORT ---
def trigger_gee_exports():
    print("\n--- 1. Triggering GEE Satellite Exports ---")
    try:
        ee.Initialize()
    except:
        print("Please authenticate GEE: 'earthengine authenticate'")
        return

    gdf = gpd.read_file(BOUNDARIES_FILE)
    print(f"Loaded {len(gdf)} townships.")
    
    cols = gdf.columns.tolist()
    id_col = next((c for c in ['adm3_pcode', 'admin3_id', 'TS_PCODE'] if c in cols), None)
    name_col = next((c for c in ['adm3_name', 'admin3_nam', 'TS_NAME'] if c in cols), None)
    
    if not id_col: 
        print("Error: No ID column found in boundaries.")
        return

    def mask_l8(img):
        qa = img.select('QA_PIXEL')
        mask = qa.bitwiseAnd(1<<3).eq(0).And(qa.bitwiseAnd(1<<4).eq(0))
        return img.updateMask(mask)

    def get_combined_image(start, end, region):
        l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(region).filterDate(start, end).map(mask_l8)
        ndvi = l8.map(lambda i: i.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')).median()
        ndbi = l8.map(lambda i: i.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI')).median()
        nl = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG').filterBounds(region).filterDate(start, end).select('avg_rad').mean()
        rn = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').filterBounds(region).filterDate(start, end).select('precipitation').sum().rename('RAIN_TOTAL')
        s1 = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(region).filterDate(start, end)\
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))\
            .filter(ee.Filter.eq('instrumentMode', 'IW')).select(['VH', 'VV']).mean()
        return ee.Image([ndvi, ndbi, nl, rn, s1]).rename(['NDVI', 'NDBI', 'NL_AVG_RAD', 'RAIN_TOTAL', 'VH', 'VV'])

    start_dt = ee.Date(START_DATE)
    periods = []
    for i in range(N_MONTHS):
        m_start = start_dt.advance(i, 'month')
        m_end = m_start.advance(1, 'month').advance(-1, 'day')
        periods.append({'start': m_start, 'end': m_end, 'name': m_start.format('YYYY-MM')})

    def convert_to_ee_feature(row):
        geom = row.geometry
        if not geom.is_valid: return None
        coords = [list(x) for x in geom.exterior.coords] if geom.geom_type == 'Polygon' else \
                 [[list(x) for x in p.exterior.coords] for p in geom.geoms]
        ee_geom = ee.Geometry.Polygon(coords) if geom.geom_type == 'Polygon' else ee.Geometry.MultiPolygon(coords)
        return ee.Feature(ee_geom, {'admin3_id': str(row[id_col]), 'admin3_nam': str(row[name_col])})

    gdf.geometry = gdf.geometry.simplify(0.01)
    chunk_size = math.ceil(len(gdf) / 4)
    
    for i in range(4):
        chunk = gdf.iloc[i*chunk_size : (i+1)*chunk_size]
        if chunk.empty: continue
        features = [f for f in [convert_to_ee_feature(r) for _, r in chunk.iterrows()] if f]
        fc = ee.FeatureCollection(features)
        
        def process_period(p):
            p = ee.Dictionary(p)
            img = get_combined_image(p.get('start'), p.get('end'), fc.geometry())
            return fc.map(lambda ft: ft.set(img.reduceRegion(ee.Reducer.mean(), ft.geometry(), SCALE, tileScale=4, bestEffort=True))\
                          .set('period', p.get('name')).set('period_start', p.get('start')))
        
        flat_fc = ee.FeatureCollection(ee.List(periods).map(process_period)).flatten()
        task_name = f"gee_panel_myanmar_PART{i+1}"
        task = ee.batch.Export.table.toDrive(
            collection=flat_fc, description=task_name, folder='EarthEngine',
            fileFormat='CSV', selectors=['admin3_id', 'admin3_nam', 'period', 'NDVI', 'NDBI', 'NL_AVG_RAD', 'RAIN_TOTAL', 'VH', 'VV']
        )
        task.start()
        print(f"  -> Submitted Batch {i+1}")

# --- PART 2: ACLED CONFLICT DATA ---
def get_acled_token(email, password):
    auth_url = "https://acleddata.com/oauth/token"
    payload = {
        'username': email,
        'password': password,
        'grant_type': 'password',
        'client_id': 'acled'
    }
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    
    try:
        r = requests.post(auth_url, data=payload, headers=headers)
        if r.status_code == 200:
            return r.json()['access_token']
        else:
            print(f"  ! ACLED Auth Failed: {r.status_code} {r.text}")
            return None
    except Exception as e:
        print(f"  ! ACLED Auth Error: {e}")
        return None

def download_acled():
    print("\n--- 2. Downloading ACLED Conflict Data ---")
    
    out_path = f"{EXTERNAL_DIR}/acled_myanmar.csv"
    if os.path.exists(out_path):
        print(f"  -> Data already exists at {out_path}. Skipping download.")
        return

    if "YourRealPassword" in ACLED_PASSWORD:
        print("  ! Skipping ACLED: Please update Email/Password in script.")
        return

    token = get_acled_token(ACLED_EMAIL, ACLED_PASSWORD)
    if not token: return

    print("  > Authentication successful. Fetching events...")
    base_url = "https://acleddata.com/api/acled/read"
    params = {
        '_format': 'csv',
        'country': TARGET_COUNTRY,
        'event_date': f"{ACLED_START_DATE}|2025-12-31",
        'event_date_where': 'BETWEEN',
        'limit': 15000 
    }
    headers = {"Authorization": f"Bearer {token}"}

    try:
        r = requests.get(base_url, params=params, headers=headers)
        if r.status_code == 200:
            df = pd.read_csv(io.StringIO(r.content.decode('utf-8')))
            df.to_csv(out_path, index=False)
            print(f"  -> Success! Saved {len(df)} events to {out_path}")
        else:
            print(f"  ! Data Request Failed: {r.status_code} {r.text}")
    except Exception as e:
        print(f"  ! Download Error: {e}")

# --- PART 3: HDX (Mobility & Prices) ---
def download_hdx():
    print("\n--- 3. Downloading HDX Data (Mobility & Prices) ---")
    Configuration.create(hdx_site="prod", user_agent="My_Pipeline", hdx_read_only=True)
    
    def download_dataset_by_id(dataset_id, keyword, final_name):
        # 1. Check if already exists
        target_path = os.path.join(EXTERNAL_DIR, final_name)
        if os.path.exists(target_path):
            print(f"  -> {final_name} already exists. Skipping.")
            return

        try:
            dataset = Dataset.read_from_hdx(dataset_id)
            if not dataset:
                print(f"  ! Dataset '{dataset_id}' not found.")
                return

            resources = dataset.get_resources()
            found_resource = None
            
            # Find Resource
            for r in resources:
                if keyword.lower() in r['name'].lower():
                    found_resource = r
                    break
            
            if found_resource:
                print(f"  -> Found resource: {found_resource['name']}")
                
                # FIX: Download to folder, then rename manually
                url, downloaded_path = found_resource.download(folder=EXTERNAL_DIR)
                
                # Rename to standard name
                if os.path.exists(downloaded_path):
                    shutil.move(downloaded_path, target_path)
                    print(f"     Downloaded and renamed to {final_name}")
                else:
                    print(f"     ! Error: Download reported success but file missing at {downloaded_path}")

            else:
                print(f"  ! No resource matching '{keyword}' in {dataset_id}")

        except Exception as e:
            print(f"  ! HDX Error for {dataset_id}: {e}")

    # 1. WFP Food Prices
    download_dataset_by_id("wfp-food-prices-for-myanmar", "price", "wfp_prices.csv")

    # 2. Meta Mobility
    download_dataset_by_id("movement-range-maps", "movement", "meta_mobility.csv")

def main():
    os.makedirs(EXTERNAL_DIR, exist_ok=True)
    
    # Run the sequence
    trigger_gee_exports()
    download_acled()
    download_hdx()
    
    print("\nData Acquisition Complete.")

if __name__ == "__main__":
    main()
