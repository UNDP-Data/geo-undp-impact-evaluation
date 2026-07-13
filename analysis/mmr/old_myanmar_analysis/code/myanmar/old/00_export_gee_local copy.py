#!/usr/bin/env python3
"""
00_export_gee_local.py
======================
Exports GEE data using YOUR LOCAL 'mmr_admin3.geojson' boundaries.
FIXED: Auto-detects correct column names (adm3_pcode vs admin3_id) to prevent 'Unknown'.
"""

import ee
import geopandas as gpd
import pandas as pd
import json
import os
import math
from shapely.geometry import Polygon, MultiPolygon

# --- CONFIGURATION ---
BOUNDARIES_FILE = "data/myanmar/myanmar_boundaries/mmr_admin3.geojson"
START_DATE = '2024-10-01'
N_MONTHS = 13
SCALE = 500 

# Initialize Earth Engine
try:
    ee.Initialize()
except:
    print("Please authenticate first: 'earthengine authenticate'")
    exit(1)

def get_periods():
    periods = []
    start_dt = ee.Date(START_DATE)
    for i in range(N_MONTHS):
        m_start = start_dt.advance(i, 'month')
        m_end = m_start.advance(1, 'month').advance(-1, 'day')
        periods.append({
            'start': m_start, 'end': m_end, 'name': m_start.format('YYYY-MM')
        })
    return periods

# Image Functions
def mask_l8(img):
    qa = img.select('QA_PIXEL')
    mask = qa.bitwiseAnd(1<<3).eq(0).And(qa.bitwiseAnd(1<<4).eq(0))
    return img.updateMask(mask)

def add_indices(img):
    ndvi = img.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
    ndbi = img.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI')
    return img.addBands([ndvi, ndbi])

def get_combined_image(start, end, region):
    # Landsat
    l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(region).filterDate(start, end).map(mask_l8).map(add_indices).select(['NDVI', 'NDBI'])
    img_l8 = ee.Algorithms.If(l8.size().eq(0), ee.Image.constant([0,0]).rename(['NDVI','NDBI']), l8.median().clip(region))
    
    # Nightlights
    nl = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG').filterBounds(region).filterDate(start, end).select('avg_rad')
    img_nl = ee.Algorithms.If(nl.size().eq(0), ee.Image.constant(0).rename('NL_AVG_RAD'), nl.mean().rename('NL_AVG_RAD').clip(region))
    
    # Rain
    rn = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').filterBounds(region).filterDate(start, end).select('precipitation')
    img_rn = ee.Algorithms.If(rn.size().eq(0), ee.Image.constant([0,0]).rename(['RAIN_TOTAL','RAIN_MEAN']), 
                              rn.sum().rename('RAIN_TOTAL').addBands(rn.mean().rename('RAIN_MEAN')).clip(region))
    
    return ee.Image(img_l8).addBands(img_nl).addBands(img_rn)

def convert_to_ee_feature(row, id_col, name_col):
    """Converts a GeoPandas row to an EE Feature using correct columns."""
    geom = row.geometry
    if isinstance(geom, Polygon):
        coords = [list(x) for x in geom.exterior.coords]
        ee_geom = ee.Geometry.Polygon(coords)
    elif isinstance(geom, MultiPolygon):
        coords = []
        for part in geom.geoms:
            coords.append([list(x) for x in part.exterior.coords])
        ee_geom = ee.Geometry.MultiPolygon(coords)
    else:
        return None
        
    # Properties - Use the detected columns
    props = {
        'admin3_id': str(row[id_col]), 
        'admin3_nam': str(row[name_col])
    }
    return ee.Feature(ee_geom, props)

def export_batch(gdf_chunk, batch_id, id_col, name_col):
    print(f"  > Preparing Batch {batch_id} ({len(gdf_chunk)} townships)...")
    
    features = []
    for _, row in gdf_chunk.iterrows():
        ft = convert_to_ee_feature(row, id_col, name_col)
        if ft: features.append(ft)
    
    fc_chunk = ee.FeatureCollection(features)
    periods = get_periods()
    
    # Server-side mapping
    def process_period(p_obj):
        p = ee.Dictionary(p_obj)
        name = p.get('name')
        start = p.get('start')
        end = p.get('end')
        
        img = get_combined_image(start, end, fc_chunk.geometry())
        
        def reduce_ft(ft):
            stats = img.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=ft.geometry(),
                scale=SCALE, maxPixels=1e13, tileScale=4, bestEffort=True
            )
            return ft.set(stats).set('period', name).set('period_start', start)
            
        return fc_chunk.map(reduce_ft)

    # Execute
    nested = ee.List(periods).map(process_period)
    flat_fc = ee.FeatureCollection(nested).flatten()
    
    task_name = f"gee_panel_myanmar_PART{batch_id}"
    task = ee.batch.Export.table.toDrive(
        collection=flat_fc,
        description=task_name,
        folder='EarthEngine',
        fileNamePrefix=task_name,
        fileFormat='CSV',
        selectors=['admin3_id', 'admin3_nam', 'period', 'period_start', 'NDVI', 'NDBI', 'NL_AVG_RAD', 'RAIN_MEAN', 'RAIN_TOTAL']
    )
    task.start()
    print(f"    -> Task {task_name} started! ID: {task.id}")

def main():
    print("Loading Local Boundaries...")
    gdf = gpd.read_file(BOUNDARIES_FILE)
    print(f"Total Townships: {len(gdf)}")
    
    # --- COLUMN DETECTION ---
    print("Detecting columns...")
    cols = gdf.columns.tolist()
    
    # ID Column
    id_candidates = ['adm3_pcode', 'admin3_pcode', 'TS_PCODE', 'township_pcode', 'admin3_id', 'id']
    id_col = next((c for c in id_candidates if c in cols), None)
    
    # Name Column
    name_candidates = ['adm3_name', 'admin3_name', 'admin3_nam', 'township_name', 'TS_NAME', 'Name']
    name_col = next((c for c in name_candidates if c in cols), None)
    
    if not id_col or not name_col:
        print(f"!!! ERROR: Could not identify ID or Name columns. Available: {cols}")
        return

    print(f"  > Using ID Column: '{id_col}'")
    print(f"  > Using Name Column: '{name_col}'")
    
    print("Simplifying geometry...")
    gdf.geometry = gdf.geometry.simplify(0.01) 
    
    # Split into 4 chunks
    chunk_size = math.ceil(len(gdf) / 4)
    
    for i in range(4):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk = gdf.iloc[start:end]
        if not chunk.empty:
            export_batch(chunk, i + 1, id_col, name_col)
            
    print("\nAll tasks submitted! Check https://code.earthengine.google.com/tasks")

if __name__ == "__main__":
    main()
