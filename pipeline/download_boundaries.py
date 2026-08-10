#!/usr/bin/env python3
"""
pipeline/download_boundaries.py
================================
Download administrative boundaries from HDX or geoBoundaries.

Usage:
    python -m pipeline.download_boundaries --config config/afghanistan.yaml
    python -m pipeline.download_boundaries --config config/ukraine.yaml
    python -m pipeline.download_boundaries --config config/myanmar.yaml
"""

import requests
import zipfile
import geopandas as gpd
from pathlib import Path
import yaml
import argparse
import io
import warnings
warnings.filterwarnings('ignore')


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def download_geoboundaries(country_code: str, admin_level: int, output_dir: Path) -> Path:
    """Download from geoBoundaries (open data)."""
    
    url = f"https://github.com/wmgeolab/geoBoundaries/raw/main/releaseData/gbOpen/{country_code.upper()}/ADM{admin_level}/geoBoundaries-{country_code.upper()}-ADM{admin_level}.geojson"
    
    print(f"  Trying geoBoundaries: {url[:60]}...")
    response = requests.get(url, timeout=60)
    
    if response.status_code == 200:
        output_file = output_dir / f"{country_code.lower()}_admin{admin_level}.geojson"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(response.text)
        return output_file
    
    return None


def download_hdx(country_code: str, admin_level: int, output_dir: Path) -> Path:
    """Download from HDX CKAN API."""
    
    dataset_name = f"cod-ab-{country_code.lower()}"
    api_url = f"https://data.humdata.org/api/3/action/package_show?id={dataset_name}"
    
    print(f"  Trying HDX: {dataset_name}...")
    response = requests.get(api_url, timeout=60)
    
    if response.status_code != 200:
        return None
    
    data = response.json()
    if not data.get('success'):
        return None
    
    resources = data['result']['resources']
    
    # Find geojson or shapefile for this admin level
    resource_url = None
    for r in resources:
        name = r.get('name', '').lower()
        fmt = r.get('format', '').lower()
        
        if f'adm{admin_level}' in name:
            if 'geojson' in fmt or 'json' in fmt:
                resource_url = r.get('url')
                break
            elif 'shp' in fmt or 'zip' in fmt:
                resource_url = r.get('url')
    
    if not resource_url:
        return None
    
    print(f"  Downloading: {resource_url[:60]}...")
    response = requests.get(resource_url, timeout=120)
    
    if response.status_code != 200:
        return None
    
    # Handle different file types
    if resource_url.endswith('.geojson') or resource_url.endswith('.json'):
        output_file = output_dir / f"{country_code.lower()}_admin{admin_level}.geojson"
        with open(output_file, 'wb') as f:
            f.write(response.content)
        return output_file
    
    elif resource_url.endswith('.zip'):
        zip_data = io.BytesIO(response.content)
        with zipfile.ZipFile(zip_data, 'r') as z:
            z.extractall(output_dir)
            
            # Find and convert shapefile
            shp_files = [f for f in z.namelist() if f.endswith('.shp')]
            adm_shp = [f for f in shp_files if f'adm{admin_level}' in f.lower()]
            shp_file = adm_shp[0] if adm_shp else (shp_files[0] if shp_files else None)
            
            if shp_file:
                gdf = gpd.read_file(output_dir / shp_file)
                output_file = output_dir / f"{country_code.lower()}_admin{admin_level}.geojson"
                gdf.to_file(output_file, driver='GeoJSON')
                return output_file
    
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', required=True)
    parser.add_argument('--admin-level', '-a', type=int, default=None)
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    country_name = config.get('country_name', 'Unknown')
    country_code = config.get('country_code', 'XXX').upper()
    admin_level = args.admin_level or config['boundaries'].get('admin_level', 2)
    
    output_dir = Path(f"data/{country_code.lower()}/boundaries")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"DOWNLOAD BOUNDARIES: {country_name}")
    print(f"{'='*70}")
    print(f"Country: {country_code}")
    print(f"Admin level: {admin_level}")
    print(f"Output: {output_dir}")
    
    # Try requested admin level; if it doesn't exist for this country
    # (e.g. Venezuela/Colombia have no ADM3+ in geoBoundaries), fall back
    # level by level and update the config so downstream stages match.
    output_file = None
    used_level = admin_level
    for lvl in range(admin_level, 0, -1):
        output_file = download_geoboundaries(country_code, lvl, output_dir)
        if output_file is None:
            output_file = download_hdx(country_code, lvl, output_dir)
        if output_file is not None:
            used_level = lvl
            break
        print(f"  ⚠ ADM{lvl} not available for {country_code} — trying ADM{lvl-1}")

    if output_file is None:
        print(f"\n❌ Download failed at every admin level (ADM{admin_level}..ADM1)")
        print(f"   Try manual download from:")
        print(f"   https://data.humdata.org/dataset/cod-ab-{country_code.lower()}")
        return 1

    if used_level != admin_level:
        print(f"\n  ⚠ Requested ADM{admin_level} does not exist for {country_code}.")
        print(f"  ✓ Using ADM{used_level} instead — updating config to match.")
        try:
            with open(args.config) as _f:
                _cfg_full = yaml.safe_load(_f)
            _cfg_full['boundaries']['admin_level'] = used_level
            _cfg_full['boundaries']['file_path'] = (
                f"data/{country_code.lower()}/boundaries/"
                f"{country_code.lower()}_admin{used_level}.geojson")
            with open(args.config, 'w') as _f:
                yaml.dump(_cfg_full, _f, default_flow_style=False, sort_keys=False)
        except Exception as _e:
            print(f"  ⚠ Could not update config automatically ({_e}) — "
                  f"set admin_level: {used_level} manually.")
    
    # Validate
    gdf = gpd.read_file(output_file)
    print(f"\n✓ Downloaded: {output_file.name}")
    print(f"  Units: {len(gdf)}")
    print(f"  Columns: {list(gdf.columns)[:8]}...")
    
    # Check expected columns
    id_col = config['boundaries'].get('id_column', 'ADM2_PCODE')
    if id_col not in gdf.columns:
        pcode_cols = [c for c in gdf.columns if 'pcode' in c.lower() or 'code' in c.lower()]
        print(f"\n⚠️ Expected '{id_col}' not found")
        print(f"   Available: {pcode_cols}")
        if pcode_cols:
            print(f"   → Update config: id_column: \"{pcode_cols[0]}\"")
    
    print(f"\n{'='*70}")
    print(f"✓ COMPLETE")
    print(f"{'='*70}")
    return 0


if __name__ == "__main__":
    exit(main())
