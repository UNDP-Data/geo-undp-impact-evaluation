# 🌍 UNDP Early Recovery Impact Dashboard

Satellite-based monitoring and causal inference for measuring UNDP post-disaster recovery interventions.

Built by [**AImpact Lab**](https://www.aimpactlab.com) for UNDP Crisis Bureau / C3RT.

---

## What it does

This system automates the full pipeline from satellite data collection to interactive dashboard for monitoring early recovery. It combines satellite remote sensing (nightlights, vegetation, radar), conflict data, economic indicators, and UNDP project data to measure recovery trajectories at the subnational level.

### Key Features

- **Recovery Scorecard** — Unit-level recovery tracking across 4+ satellite indicators
- **Economic Reactivation Estimates** — Nightlight-to-GDP conversion using Henderson et al. (2012) elasticity
- **AI Co-pilot** — Natural language assistant for interpreting recovery data (OpenAI GPT-4o)
- **140+ Countries** — Pre-configured with sensible defaults for rapid deployment
- **Causal Impact Analysis** — DiD, Synthetic Control, and event studies for Myanmar and Gaza
- **Interactive Dashboard** — Streamlit web application with real-time pipeline execution

### Currently configured with full data for:

- 🇲🇲 **Myanmar** — M7.7 Earthquake, March 2025
- 🇵🇸 **Gaza Strip** — Conflict, October 2023
- 🇦🇫 **Afghanistan** — Herat Earthquake, October 2023
- 🇺🇦 **Ukraine** — Russian Invasion, February 2022
- 🇨🇩 **DR Congo** — M23 Conflict, March 2022
- 🇸🇩 **Sudan** — Civil War, April 2023
- 🇸🇴 **Somalia** — Drought, January 2023
- 🇸🇾 **Syria** — Earthquake, February 2023
- 🇯🇲 **Jamaica** — Hurricane Melissa, October 2025

---

## Quick Start

### 1. Clone and setup
```bash
git clone https://github.com/hgrueso/undp-impact-evaluation.git
cd undp-impact-evaluation

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure credentials
```bash
cp .env.example .env
# Edit .env with your credentials:
#   ACLED_EMAIL=your@email.com
#   ACLED_KEY=your_acled_key
#   GEE_PROJECT=your_gee_project_id
#   OPENAI_API_KEY=sk-your-openai-key (optional, for AI Co-pilot)

# Authenticate Google Earth Engine:
earthengine authenticate
```

### 3. Run the dashboard
```bash
streamlit run app.py
```

Or run the pipeline from the command line:
```bash
make run COUNTRY=myanmar
```

---

## Pipeline Architecture

| Stage | Command | What it does |
|-------|---------|--------------|
| Boundaries | `make boundaries` | Download admin boundaries from geoBoundaries |
| Satellite | `make gee` | Export VIIRS nightlights, NDVI, SAR, rainfall via Google Earth Engine |
| Conflict | `make acled` | Download ACLED conflict events and fatalities |
| Economic | `make wb` | Download GDP, population from World Bank WDI |
| Food Prices | `make wfp` | Download WFP market food prices |
| Trends | `make gt` | Download Google Trends search indices |
| UNDP Projects | `make undp` | Scrape UNDP project data and budgets |
| Preprocessing | `make preprocess` | Build admin-unit × month panel dataset |
| Visualisation | `make visuals` | Time series, choropleth maps, raster comparisons |
| Scorecard | automatic | Recovery scorecard with unit-level filtering |

Run everything: `make run COUNTRY=myanmar`
Check status: `make status COUNTRY=myanmar`

---

## Data Sources

| Source | Indicator | Coverage | Use |
|--------|-----------|----------|-----|
| **NASA VIIRS** | Nighttime Lights | Monthly, 500m, 2012-present | Economic activity proxy |
| **ESA Sentinel-1** | Radar Backscatter (VV) | 12-day, 10m, 2014-present | Surface change, debris |
| **NASA MODIS** | Vegetation Index (NDVI) | 16-day, 250m, 2000-present | Agricultural health |
| **ACLED** | Conflict Events & Fatalities | Daily, geocoded | Security context |
| **World Bank WDI** | GDP, GDP/capita, Population | Annual | Economic baseline |
| **WFP** | Food Prices | Monthly, market-level | Food security |
| **UNDP API** | Project budgets & timelines | Per project | Intervention tracking |
| **Google Trends** | Search interest indices | Weekly/monthly | Population sentiment |
| **CHIRPS** | Rainfall | Monthly, 5km | Environmental control |

---

## Recovery Scorecard Methodology

For each indicator and admin unit:

- **Baseline:** Mean of all available pre-event months (NDVI uses year-over-year seasonal adjustment)
- **Trough:** Single worst post-event month
- **Affected units:** Only units with >10% decline are scored; others are "Stable"
- **Recovery:** (Current − Trough) / (Baseline − Trough) × 100, capped at 100%
- **Since Project:** Recovery from selected UNDP project start date (never exceeds Overall)

## Economic Reactivation Model

Converts nightlight recovery to USD using Henderson, Storeygard & Weil (2012) elasticity of 0.3:

1. Pre-shock national GDP from World Bank WDI
2. Affected area GDP = national GDP × share of affected admin units
3. GDP reactivation = affected GDP × NL recovery × 0.3
4. Project ROI = proportional share based on budget

**Note:** Correlational estimates for rapid screening, not causal impact.

---

## Causal Impact Analysis

For countries with geocoded UNDP project locations (Myanmar and Gaza), the pipeline includes formal causal inference:

- **Difference-in-Differences (DiD)** — UNDP-treated vs control areas
- **Two-Way Fixed Effects** — Unit and time fixed effects
- **Event Study Design** — Pre-trend validation and dynamic effects
- **Synthetic Control** — Counterfactual construction
- **Placebo Tests** — Artificial treatment date validation
- **Dose-Response** — Effect variation by proximity to intervention

Analysis scripts: `analysis/mmr/` (Myanmar) and `analysis/pse/` (Gaza).

---

## Adding a New Country

### Option A: Dashboard (recommended)

1. Open the dashboard → **Setup & Run** tab
2. Select country from the 140+ dropdown
3. Set event date and type
4. Click **Save Configuration** then **Run Full Pipeline**

### Option B: Command line

1. Create `config/newcountry.yaml`:
```yaml
country_name: "New Country"
country_code: "NCC"
study_period_start: "2023-01-01"
study_period_end: "2025-12-31"
events:
  disaster_type: "earthquake"
  disaster_date: "2024-06-15"
  intervention_start: "2024-06-15"
boundaries:
  file_path: "data/ncc/boundaries/ncc_admin2.geojson"
  id_column: "shapeID"
  name_column: "shapeName"
  admin_level: 2
acled_country_name: "New Country"
wfp_commodity: "Rice"
```

2. Run: `make run COUNTRY=newcountry`

---

## Deployment

### Local
```bash
streamlit run app.py
```

### Server (Azure / any Linux)
```bash
git clone https://github.com/hgrueso/undp-impact-evaluation.git
cd undp-impact-evaluation
pip install -r requirements.txt
# Configure .env with credentials
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Docker
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
```

---

## Requirements

- Python 3.10+
- Google Earth Engine account (free)
- ACLED API key (free academic access)
- OpenAI API key (optional, for AI Co-pilot)

---

## Contact

**AImpact Lab** — Tech-enabled impact evaluation consultancy

- Website: [www.aimpactlab.com](https://www.aimpactlab.com)
- Email: [info@aimpactlab.com](mailto:info@aimpactlab.com)

**Hernando Grueso, PhD** — Lead Data Scientist & Founder
([hernando@aimpactlab.com](mailto:hernando@aimpactlab.com))
