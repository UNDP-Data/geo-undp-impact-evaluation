#!/usr/bin/env python3
"""
WFP food prices — pipeline stage.

Downloads the public WFP food-prices dataset for the configured country from
HDX (data.humdata.org) and saves it as data/{code}/external/wfp_prices.csv —
the exact file and format pipeline/preprocessing.py already expects
(standard WFP CSV with the '#' hashtag header row).

Resilience: WFP does not publish prices for every country. If no dataset is
found, this stage says so plainly and exits 0 — it never breaks the pipeline.

Usage:
    python -m pipeline.wfp_download --config config/ven.yaml
"""
import argparse
import sys
from pathlib import Path

import requests
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def find_wfp_dataset(country_name):
    """Search HDX for the WFP food-prices dataset of this country."""
    api = "https://data.humdata.org/api/3/action/package_search"
    r = requests.get(api, params={"q": f"wfp food prices {country_name}",
                                  "rows": 10}, timeout=60)
    r.raise_for_status()
    results = r.json().get("result", {}).get("results", [])
    cname = country_name.lower()
    for ds in results:
        name = ds.get("name", "")
        title = ds.get("title", "").lower()
        if name.startswith("wfp-food-prices") and cname.split()[0] in (name + " " + title):
            return ds
    # fallback: any wfp-food-prices hit
    for ds in results:
        if ds.get("name", "").startswith("wfp-food-prices"):
            return ds
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    code = cfg.get("country_code", "xxx").lower()
    name = cfg.get("acled_country_name") or cfg.get("country_name", code)

    out_dir = Path(f"data/{code}/external")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "wfp_prices.csv"

    print("=" * 60)
    print(f"WFP FOOD PRICES: {name}")
    print("=" * 60)

    ds = find_wfp_dataset(name)
    if ds is None:
        print(f"  ⚠ No WFP food-prices dataset on HDX for '{name}' — "
              f"WFP does not publish prices for every country. Skipping.")
        return 0
    print(f"  Dataset: {ds.get('title', ds['name'])}")

    csvs = [r for r in ds.get("resources", [])
            if r.get("format", "").lower() == "csv"]
    if not csvs:
        print("  ⚠ Dataset has no CSV resource — skipping.")
        return 0

    url = csvs[0].get("url")
    print(f"  Downloading: {url[:70]}...")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    out_file.write_bytes(r.content)

    n_rows = max(0, r.text.count("\n") - 2)  # minus header + hashtag row
    print(f"✓ Saved: {out_file} (~{n_rows} rows)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"  ⚠ WFP stage failed ({e}) — continuing without prices")
        sys.exit(0)
