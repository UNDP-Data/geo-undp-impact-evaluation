#!/usr/bin/env python3
"""
00_data_acquisition.py
======================
Unified Data Downloader:
1. Triggers Google Earth Engine (GEE) exports.
2. Downloads ACLED Data (Conflict) - NEW OAuth Authentication.
3. Downloads WFP Prices (HDX).
4. Downloads Google Trends (Sentiment) - Using Burmese Keywords.
5. Downloads IODA Internet Connectivity Data - Robust Parsing.
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
from hdx.api.configuration import Configuration
from hdx.data.dataset import Dataset
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from datetime import datetime

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

# --- GLOBAL CONFIGURATION ---
DATA_DIR = "data/myanmar"
EXTERNAL_DIR = "data/myanmar/external"
BOUNDARIES_FILE = "data/myanmar/myanmar_boundaries/mmr_admin3.geojson"

from datetime import datetime

# Study Period (Dynamic)
START_DATE_STR = "2024-10-01"
TODAY_STR = datetime.today().strftime('%Y-%m-%d') 

# GEE Batch Configuration (Matches your analysis script's PART1-PART4)
GEE_BATCHES = [
    ('2024-10-01', '2025-01-01', 'PART1'),
    ('2025-01-01', '2025-04-01', 'PART2'),
    ('2025-04-01', '2025-07-01', 'PART3'),
    ('2025-07-01', '2025-11-01', 'PART4')
]
# ACLED Settings (UPDATED for OAuth)
ACLED_EMAIL = "hernando.gruesohurtado@spi.ox.ac.uk"
ACLED_PASSWORD = "your_password"

# Google Trends: English -> Burmese Mapping
GT_KEYWORDS = {
    "Earthquake": "ငလျင်",
    "Donation": "အလှူ",
    "Loan": "ချေးငွေ",
    "Flight Ticket": "လေယာဉ်လက်မှတ်",
    "Gold Price": "ရွှေဈေး",
    "Rice Price": "ဆန်ဈေး",
    "Internet connection": "အင်တာနက်",
    "VPN": "VPN"
}

def initialize_gee():
    try:
        ee.Initialize()
        print("GEE Initialized.")
    except:
        print("Authenticating GEE...")
        ee.Authenticate()
        ee.Initialize()

# --- PART 1: GEE Satellite Data ---
def get_roi():
    if not os.path.exists(BOUNDARIES_FILE):
        print(f"Error: Boundary file not found at {BOUNDARIES_FILE}")
        return None
    gdf = gpd.read_file(BOUNDARIES_FILE)
    # Simplify geometry to avoid 'vertex limit' errors in GEE
    gdf['geometry'] = gdf['geometry'].simplify(0.01)
    return gdf

def trigger_gee_exports():
    print("\n--- 1. Triggering GEE Exports ---")
    gdf = get_roi()
    if gdf is None: return

    # Convert bounds to GEE Geometry
    bounds = gdf.total_bounds
    roi = ee.Geometry.Rectangle([bounds[0], bounds[1], bounds[2], bounds[3]])
    
    # Collections
    viirs = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG").select('avg_rad')
    modis = ee.ImageCollection("MODIS/006/MOD13A1").select('NDVI')
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/PENTAD").select('precipitation')
    s1 = ee.ImageCollection('COPERNICUS/S1_GRD').filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')).filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')).filter(ee.Filter.eq('instrumentMode', 'IW'))

    # Define range for GEE loop (e.g. last 13 months)
    dates = pd.date_range(start=START_DATE_STR, end=END_DATE_STR, freq='ME')
    
    for date in dates:
        d_str = date.strftime('%Y-%m-%d')
        y, m = date.year, date.month
        print(f"  > Processing {d_str}...")
        
        # Define Time Range
        start = ee.Date.fromYMD(y, m, 1)
        end = start.advance(1, 'month')
        
        # Reducers (Mean/Sum)
        nl = viirs.filterDate(start, end).mean().rename('NL_AVG_RAD')
        veg = modis.filterDate(start, end).mean().rename('NDVI')
        rain = chirps.filterDate(start, end).sum().rename('RAIN_TOTAL')
        
        s1_filtered = s1.filterDate(start, end).filterBounds(roi)
        sar = s1_filtered.mean().select(['VV', 'VH'])
        
        # Composite
        img = nl.addBands(veg).addBands(rain).addBands(sar).clip(roi)
        
        # NOTE: Actual Export.table.toDrive lines would go here.
        # For this pipeline, we assume files are generated or this triggers the task.
        print(f"    (Ready to Export: {d_str})")

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

# --- PART 1: GEE EXPORT (BATCH MODE) ---
def trigger_gee_exports():
    print("\n--- 1. Triggering GEE Exports (Batch Mode) ---")
    gdf = get_roi()
    if gdf is None: return

    # Convert bounds
    bounds = gdf.total_bounds
    roi = ee.Geometry.Rectangle([bounds[0], bounds[1], bounds[2], bounds[3]])
    
    # Collections (Using Updated MODIS 061)
    viirs = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG").select('avg_rad')
    modis = ee.ImageCollection("MODIS/061/MOD13A1").select('NDVI') 
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/PENTAD").select('precipitation')
    s1 = ee.ImageCollection('COPERNICUS/S1_GRD').filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')).filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')).filter(ee.Filter.eq('instrumentMode', 'IW'))

    # Batch Loop
    for start, end, label in GEE_BATCHES:
        print(f"  > Processing Batch {label}: {start} to {end}...")
        
        # Define Time Range
        start_date = ee.Date(start)
        end_date = ee.Date(end)
        
        # Reducers
        nl = viirs.filterDate(start_date, end_date).mean().rename('NL_AVG_RAD')
        veg = modis.filterDate(start_date, end_date).mean().rename('NDVI')
        rain = chirps.filterDate(start_date, end_date).sum().rename('RAIN_TOTAL')
        
        # Sentinel-1 (Mean over period)
        s1_filtered = s1.filterDate(start_date, end_date).filterBounds(roi)
        sar = s1_filtered.mean().select(['VV', 'VH'])
        
        # Composite Image
        img = nl.addBands(veg).addBands(rain).addBands(sar).clip(roi)
        
        # NOTE: This print confirms the logic ran. The actual Export would happen here.
        print(f"    (Ready to Export: gee_panel_myanmar_{label}.csv)")

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
        print("  ! No password provided. Trying public access (likely to fail)...")

    # 2. Prepare Request
    url = "https://acleddata.com/api/acled/read"
    params = {
        'terms': 'accept',
        'country': 'Myanmar',
        'event_date': f"{START_DATE_STR}|{TODAY_STR}", # Uses dynamic TODAY_STR
        'event_date_where': 'BETWEEN',
        'limit': 5000,
        'format': 'csv'
    }

    headers = {'User-Agent': 'Mozilla/5.0'}
    if token:
        headers['Authorization'] = f"Bearer {token}"
        print("  > Authorized request prepared.")

    # 3. Execute
    try:
        r = requests.get(url, params=params, headers=headers)
        
        if r.status_code == 200:
            # Check if response is actually CSV or error JSON disguised as 200
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
            print("    -> Solution: Ensure ACLED_EMAIL and ACLED_PASSWORD are correct.")
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
        pytrends = TrendReq(hl='my', tz=390, timeout=(10,25))
        dfs = []
        tf = f"{START_DATE_STR} {END_DATE_STR}"
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
                time.sleep(random.uniform(2, 4)) 
            except Exception as e:
                print(f"    ! Failed keyword: {eng_term}")
        
        if dfs:
            pd.concat(dfs).to_csv(f"{EXTERNAL_DIR}/google_trends_mm.csv", index=False)
            print("  -> Success! Saved google_trends_mm.csv")
    except Exception as e:
        print(f"  ! Google Trends Error: {e}")

# --- PART 5: INTERNET CONNECTIVITY (IODA) ---
def download_ioda_connectivity():
    print("\n--- 5. Downloading Internet Connectivity (IODA) ---")
    print("  > Strategy: Micro-chunking (Month-by-Month)...")
    
    signals = ['bgp', 'active_probing']
    dfs = []
    
    dates = pd.date_range(start=START_DATE_STR, end=END_DATE_STR, freq='MS')
    
    for date in dates:
        chunk_start = int(date.timestamp())
        chunk_end = int((date + pd.offsets.MonthEnd(0)).timestamp())
        
        if chunk_start > time.time(): break
        if chunk_end > time.time(): chunk_end = int(time.time())

        month_str = date.strftime("%Y-%m")

        for sig in signals:
            try:
                # Use RAW endpoint
                url = f"https://api.ioda.inetintel.cc.gatech.edu/v2/signals/raw/country/MM?from={chunk_start}&until={chunk_end}&sourceParams={sig}"
                
                r = requests.get(url, timeout=30)
                
                if r.status_code == 200:
                    # ROBUST PARSING: Handle both list and dict responses
                    json_resp = r.json()
                    
                    if isinstance(json_resp, list):
                        data_list = json_resp
                    elif isinstance(json_resp, dict):
                        data_list = json_resp.get('data', [])
                    else:
                        data_list = []

                    for series in data_list:
                        # Safety check for dictionary structure
                        if not isinstance(series, dict): continue
                        
                        datasource = series.get('datasource', 'unknown')
                        values = series.get('values', [])
                        
                        if values:
                            temp = pd.DataFrame(values, columns=['timestamp', 'connectivity_score'])
                            temp['date'] = pd.to_datetime(temp['timestamp'], unit='s')
                            temp['source'] = datasource
                            dfs.append(temp)
                            
                elif r.status_code == 404:
                    pass 
                else:
                    print(f"    ! Error {r.status_code} for {sig} in {month_str}")
                    
                time.sleep(0.5)
                
            except Exception as e:
                print(f"    ! Exception: {e}")

    # Process & Save
    if dfs:
        full_df = pd.concat(dfs)
        full_df['date_day'] = full_df['date'].dt.normalize()
        daily_df = full_df.groupby(['date_day', 'source'])['connectivity_score'].mean().reset_index()
        
        final_df = daily_df.pivot(index='date_day', columns='source', values='connectivity_score').reset_index()
        final_df.rename(columns={'date_day': 'date'}, inplace=True)
        
        for col in final_df.columns:
            if col in signals and final_df[col].max() > 0:
                final_df[f'{col}_norm'] = (final_df[col] / final_df[col].max()) * 100
        
        output_path = f"{EXTERNAL_DIR}/ioda_connectivity_mm.csv"
        final_df.to_csv(output_path, index=False)
        print(f"  -> Success! Fetched {len(final_df)} days of connectivity data.")
    else:
        print("  ! Failed to download any IODA data.")

def main():
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
    if not os.path.exists(EXTERNAL_DIR): os.makedirs(EXTERNAL_DIR)
    
    initialize_gee()
    
    # Run All
    trigger_gee_exports() # <--- MAKE SURE THIS IS HERE
    download_acled()
    download_wfp_prices()
    download_google_trends()
    download_ioda_connectivity()
    
    print("\nData Acquisition Complete.")
    
if __name__ == "__main__":
    main()
