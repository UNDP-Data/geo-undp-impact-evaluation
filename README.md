# 🌍 UNDP Impact Evaluation Pipeline

Satellite-based causal inference for measuring UNDP post-disaster recovery interventions.

Built by [AImpact Lab](https://aimpactlab.com).

## What it does

This pipeline automates the data collection, processing, and visualization for impact evaluations of UNDP disaster recovery programs. It uses satellite remote sensing (nightlights, vegetation, radar) combined with conflict data and Google Trends to measure recovery trajectories at the subnational level.

**Currently configured for:**
- 🇲🇲 Myanmar (M7.7 Earthquake, March 2025)
- 🇵🇸 Gaza Strip (Conflict, October 2023)
- 🇦🇫 Afghanistan (Herat Earthquake, October 2023)
- 🇺🇦 Ukraine (Russian Invasion, February 2022)

## Quick Start

### 1. Clone and setup

```bash
git clone https://github.com/YOUR_ORG/undp-impact-evaluation.git
cd undp-impact-evaluation

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp .env.example .env
# Edit .env with your ACLED credentials
# Authenticate GEE: earthengine authenticate
```

### 3. Run the pipeline

**Option A — Command line:**
```bash
make run COUNTRY=myanmar
```

**Option B — Web interface:**
```bash
streamlit run app.py
```

## Pipeline Stages

| Stage | Command | What it does |
|-------|---------|-------------|
| Boundaries | `make boundaries` | Download admin boundaries from geoBoundaries |
| GEE | `make gee` | Export VIIRS nightlights, NDVI, SAR, rainfall |
| External | `make external` | Download ACLED conflict + Google Trends |
| UNDP | `make undp` | Scrape UNDP project data + ML classification |
| Preprocess | `make preprocess` | Build township-month panel dataset |
| Visuals | `make visuals` | Time series, choropleths, UNDP timeline |

Run everything: `make run COUNTRY=myanmar`

Check status: `make status COUNTRY=myanmar`

## Adding a New Country

1. Create `config/newcountry.yaml`:

```yaml
country_name: "New Country"
country_code: "NCC"

study_period_start: "2023-01-01"
study_period_end: "2025-12-31"

events:
  disaster_type: "earthquake"
  disaster_date: "2024-06-15"
  intervention_start: "2024-07-01"

boundaries:
  file_path: "data/ncc/boundaries/ncc_admin2.geojson"
  id_column: "shapeID"
  name_column: "shapeName"
  admin_level: 2

acled_country_name: "New Country"
wfp_commodity: "Rice"
```

2. Run: `make run COUNTRY=newcountry`

Or use the Streamlit interface to create configs visually.

## Data Sources

| Source | What | Resolution |
|--------|------|-----------|
| VIIRS DNB | Nighttime lights (economic activity proxy) | 500m monthly |
| MODIS | NDVI vegetation index | 250m bi-monthly |
| Sentinel-1 | SAR radar (structural changes) | 10m monthly |
| CHIRPS | Rainfall | 5km daily |
| ACLED | Armed conflict events | Event-level |
| UNDP API | Project data + ML categorization | Project-level |
| Google Trends | Search interest indicators | Weekly |

## Analysis Scripts

Country-specific causal inference scripts live in `analysis/`:

```
analysis/
├── pse/   # Gaza: DiD + buffer sensitivity + pre-trends
└── mmr/   # Myanmar: Buffer-based DiD + raster maps
```

These use the panel data built by the pipeline to run difference-in-differences regressions with multiple buffer zone specifications.

## Requirements

- Python 3.10+
- Google Earth Engine account (free)
- ACLED API key (free academic access)
