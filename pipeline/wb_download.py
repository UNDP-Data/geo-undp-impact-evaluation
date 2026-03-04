"""Download World Bank GDP and population data for a country."""
import argparse
import yaml
import pandas as pd
from pathlib import Path

def download_wb_data(config_path):
    """Download GDP and population from World Bank API."""
    try:
        import wbgapi as wb
    except ImportError:
        print("  pip install wbgapi")
        return None

    config = yaml.safe_load(open(config_path))
    code = config["country_code"].lower()
    iso3 = config["country_code"].upper()
    event_date = pd.Timestamp(config["events"]["disaster_date"])
    event_year = event_date.year

    # Download 5 years before event to 1 year after
    start_yr = event_year - 5
    end_yr = min(event_year + 1, 2023)  # WB data usually lags

    indicators = {
        "NY.GDP.MKTP.CD": "gdp_current_usd",
        "NY.GDP.PCAP.CD": "gdp_per_capita_usd",
        "SP.POP.TOTL": "population",
    }

    print(f"  Downloading WB data for {iso3} ({start_yr}-{end_yr})...")
    rows = []
    for ind_code, ind_name in indicators.items():
        try:
            df = wb.data.DataFrame(ind_code, iso3, time=range(start_yr, end_yr + 1))
            for col in df.columns:
                year = int(col.replace("YR", ""))
                val = df[col].iloc[0]
                if pd.notna(val):
                    rows.append({"indicator": ind_name, "year": year, "value": float(val)})
        except Exception as e:
            print(f"  Warning: {ind_name} failed: {e}")

    if not rows:
        print("  No data retrieved")
        return None

    out_dir = Path(f"data/{code}/external")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wb_indicators.csv"
    result = pd.DataFrame(rows)
    result.to_csv(out_path, index=False)
    print(f"  Saved {len(result)} rows to {out_path}")

    # Print summary
    for ind_name in indicators.values():
        subset = result[result["indicator"] == ind_name].sort_values("year")
        if not subset.empty:
            latest = subset.iloc[-1]
            if ind_name == "gdp_current_usd":
                print(f"  GDP: ${latest['value']:,.0f} ({int(latest['year'])})")
            elif ind_name == "gdp_per_capita_usd":
                print(f"  GDP/capita: ${latest['value']:,.0f} ({int(latest['year'])})")
            elif ind_name == "population":
                print(f"  Population: {latest['value']:,.0f} ({int(latest['year'])})")

    return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    download_wb_data(args.config)
