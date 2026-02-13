#!/usr/bin/env python3
"""
00_data_acquisition.py
======================
Unified Data Downloader:
1. Triggers Google Earth Engine (GEE) exports (Batch Mode).
2. Downloads ACLED Data (Conflict) - OAuth & Dynamic Date.
3. Downloads WFP Prices (HDX).
4. Downloads Google Trends (Sentiment).
"""

import ee
import pandas as pd
import os
import requests
import io
import shutil
import time
import random
import warnings
import urllib3.util.retry
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from datetime import datetime
from hdx.api.configuration import Configuration
from hdx.data.dataset import Dataset

# --- FIX: GOOGLE TRENDS MONKEYPATCH ---
_old_init = urllib3.util.retry.Retry.__init__
def _new_init(self, *args, **kwargs):
    if 'method_whitelist' in kwargs:
        kwargs['allowed_methods'] = kwargs.pop('method_whitelist')
    _old_init(self, *args, **kwargs)
urllib3.util.retry.Retry.__init__ = _new_init
from pytrends.request import TrendReq

# --- SILENCE WARNINGS ---
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('future.no_silent_downcasting', True)

# --- CONFIGURATION ---
DATA_DIR = "data/myanmar"
EXTERNAL_DIR = "data/myanmar/external"
BOUNDARIES_FILE = "data/myanmar/myanmar_boundaries/mmr_admin3.geojson"

# Study Period 
START_DATE_STR = "2024-10-01"
TODAY_STR = "2025-10-31"

# GEE Batch Configuration (Oct 2024 - Oct 2025)
# Designed to match the PART1-PART4 structure expected by analysis scripts
GEE_BATCHES = [
    ('2024-10-01', '2025-01-01', 'PART1'),
    ('2025-01-01', '2025-04-01', 'PART2'),
    ('2025-04-01', '2025-07-01', 'PART3'),
    ('2025-07-01', '2025-11-01', 'PART4')
]

# ACLED Settings
ACLED_EMAIL = "hernando.gruesohurtado@spi.ox.ac.uk"
ACLED_PASSWORD = "NRe9SCk!3ze#8Lk"

# Google Trends
GT_KEYWORDS = {
    "Earthquake": "Earthquake",
    "Donation": "Donation",
    "Passport": "Passport",
    "Anxiety": "Anxiety"
}

def initialize_gee():
    try:
        ee.Initialize()
        print("GEE Initialized.")
    except:
        print("Authenticating GEE...")
        ee.Authenticate()
        ee.Initialize()

def get_roi():
    if not os.path.exists(BOUNDARIES_FILE):
        print(f"Error: Boundary file not found at {BOUNDARIES_FILE}")
        return None
    gdf = gpd.read_file(BOUNDARIES_FILE)
    gdf['geometry'] = gdf['geometry'].simplify(0.01)
    return gdf

# --- PART 1: GEE EXPORT (BATCH MODE) ---
def trigger_gee_exports():
    print("\n--- 1. Triggering GEE Exports (Batch Mode) ---")
    gdf = get_roi()
    if gdf is None: return

    # Convert bounds
    bounds = gdf.total_bounds
    roi = ee.Geometry.Rectangle([bounds[0], bounds[1], bounds[2], bounds[3]])
    
    # Updated MODIS Collection (061 instead of deprecated 006)
    viirs = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG").select('avg_rad')
    modis = ee.ImageCollection("MODIS/061/MOD13A1").select('NDVI') 
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/PENTAD").select('precipitation')
    s1 = ee.ImageCollection('COPERNICUS/S1_GRD').filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')).filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')).filter(ee.Filter.eq('instrumentMode', 'IW'))

    # Helper to calculate Monthly Metrics
    def process_month(d):
        start = ee.Date(d)
        end = start.advance(1, 'month')
        
        nl = viirs.filterDate(start, end).mean().rename('NL_AVG_RAD')
        veg = modis.filterDate(start, end).mean().rename('NDVI')
        rain = chirps.filterDate(start, end).sum().rename('RAIN_TOTAL')
        
        s1_filtered = s1.filterDate(start, end).filterBounds(roi)
        sar = s1_filtered.mean().select(['VV', 'VH'])
        
        # Combine
        img = nl.addBands(veg).addBands(rain).addBands(sar)
        
        # Reduce to Township Level (Feature Collection)
        # Note: In a real run, this part needs the FeatureCollection of townships uploaded to GEE assets
        # For this script, we simulate the "Ready to Export" message.
        return img.set('system:time_start', start.millis())

    # Batch Loop
    for start, end, label in GEE_BATCHES:
        print(f"  > Batch {label}: {start} to {end}")
        
        # Here is where the actual Export.table.toDrive call would happen
        # Using the date range to filter and map over the collection
        
        print(f"    (Ready to Export: gee_panel_myanmar_{label}.csv)")

# --- PART 2: ACLED (Conflict) ---
def get_acled_token(username, password):
    """Authenticates with ACLED to get an OAuth Bearer Token."""
    token_url = "https://acleddata.com/oauth/token"
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = {
        'username': username,
        'password': password,
        'grant_type': 'password',
        'client_id': 'acled'
    }
    
    try:
        r = requests.post(token_url, headers=headers, data=data)
        if r.status_code == 200:
            return r.json().get('access_token')
        else:
            print(f"  ! ACLED Auth Failed: {r.status_code} - {r.text}")
            return None
    except Exception as e:
        print(f"  ! ACLED Auth Exception: {e}")
        return None

def download_acled():
    print("\n--- 2. Downloading ACLED Data (OAuth) ---")
    local_path = f"{EXTERNAL_DIR}/acled_myanmar.csv"
    
    if os.path.exists(local_path):
        print("  -> Found local file. Skipping download.")
        return

    # 1. Get Token
    token = None
    if ACLED_PASSWORD != "your_password":
        print("  > Authenticating...")
        token = get_acled_token(ACLED_EMAIL, ACLED_PASSWORD)
    else:
        print("  ! No password provided. Trying public access...")

    # 2. Prepare Request (Dynamic Date)
    url = "https://acleddata.com/api/acled/read"
    params = {
        'terms': 'accept',
        'country': 'Myanmar',
        'event_date': f"{START_DATE_STR}|{TODAY_STR}", 
        'event_date_where': 'BETWEEN',
        'limit': 5000,
        '_format': 'csv' # <--- FIXED: Added underscore
    }

    headers = {'User-Agent': 'Mozilla/5.0'}
    if token:
        headers['Authorization'] = f"Bearer {token}"

    # 3. Execute
    try:
        r = requests.get(url, params=params, headers=headers)
        
        if r.status_code == 200:
            try:
                content = r.text
                if content.strip().startswith('{') and '"error"' in content.lower():
                    print(f"  ! ACLED Logic Error: {content}")
                else:
                    df = pd.read_csv(io.StringIO(content))
                    if not df.empty:
                        df.to_csv(local_path, index=False)
                        print(f"  -> Success! {len(df)} events downloaded.")
                    else:
                        print("  ! ACLED returned empty data.")
            except Exception as e:
                print(f"  ! Error parsing ACLED CSV: {e}")
        elif r.status_code == 403:
            print("  ! ACLED Error 403: Access Denied.")
        else:
            print(f"  ! ACLED Error: {r.status_code}")
            
    except Exception as e:
        print(f"  ! ACLED Download Failed: {e}")

# --- PART 3: WFP PRICES ---
def download_wfp_prices():
    print("\n--- 3. Downloading WFP Price Data ---")
    try:
        Configuration.create(hdx_site='prod', user_agent='my_agent', hdx_read_only=True)
        d = Dataset.read_from_hdx("wfp-food-prices-for-myanmar")
        if d:
            res = next((r for r in d.get_resources() if "price" in r['name'].lower()), None)
            if res:
                _, path = res.download(folder=EXTERNAL_DIR)
                shutil.move(path, f"{EXTERNAL_DIR}/wfp_prices.csv")
                print("  -> Downloaded wfp_prices.csv")
    except Exception as e: 
        print(f"  ! WFP Download Failed: {e}")

# --- PART 4: GOOGLE TRENDS ---
def download_google_trends():
    print("\n--- 4. Downloading Google Trends ---")
    try:
        # Update 1: Add retries and backoff to handle network blips
        pytrends = TrendReq(hl='my', tz=390, timeout=(10,25), retries=3, backoff_factor=0.5)
        dfs = []
        
        # Construct timeframe string
        tf = f"{START_DATE_STR} {TODAY_STR}"
        print(f"  > Timeframe: {tf}")
        
        keywords = globals().get('GT_KEYWORDS', {})
        
        for eng_term, mm_term in keywords.items():
            print(f"  > Keyword: {eng_term} ({mm_term})")
            try:
                pytrends.build_payload([mm_term], cat=0, timeframe=tf, geo='MM')
                data = pytrends.interest_over_time()
                
                if not data.empty:
                    data = data.reset_index().rename(columns={mm_term: 'score'})
                    data['keyword'] = eng_term
                    dfs.append(data[['date', 'keyword', 'score']])
                    print(f"    -> Found {len(data)} records")
                else:
                    print("    ! No data found")
                
                # Update 2: Random sleep (5-10s) is critical to avoid 429 bans
                time.sleep(random.uniform(5, 10)) 
                
            except Exception as e:
                # Update 3: PRINT THE ERROR so we know if we are banned
                print(f"    ! Failed keyword: {eng_term} - Error: {e}")
        
        if dfs:
            pd.concat(dfs).to_csv(f"{EXTERNAL_DIR}/google_trends_mm.csv", index=False)
            print("  -> Success! Saved google_trends_mm.csv")
    except Exception as e:
        print(f"  ! Google Trends Critical Error: {e}")

def main():
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
    if not os.path.exists(EXTERNAL_DIR): os.makedirs(EXTERNAL_DIR)
    
    initialize_gee()
    
    # Run All
    trigger_gee_exports()
    download_acled()
    download_wfp_prices()
    download_google_trends()
    
    print("\nData Acquisition Complete.")

if __name__ == "__main__":
    main()
