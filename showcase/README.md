# Showcase — Pre-computed Results

This folder contains curated outputs from the UNDP Early Recovery Impact pipeline.
These results are committed to the repository so they are immediately available
without needing to re-run the pipeline.

## Contents

### Myanmar (mmr/) — M7.7 Earthquake, March 2025
- **Causal impact estimates** (DiD with buffer zones)
- Coefficient plots showing treatment effects on nightlights, NDVI, and radar
- Time series and spatial maps
- Recovery scorecard data

### Gaza (pse/) — Conflict, October 2023
- **Causal impact estimates** (DiD, pre-trends validation)
- Conflict hotspot maps with/without UNDP presence
- Time series and spatial maps
- Recovery scorecard data

### Jamaica (jam/) — Hurricane Melissa, October 2025
- Nightlight and vegetation maps showing hurricane impact
- Time series and recovery scorecard

### Afghanistan (afg/) — Herat Earthquake, October 2023
- Nightlight and vegetation time series
- Recovery scorecard

### Ukraine (ukr/) — Russian Invasion, February 2022
- Nightlight and conflict maps
- Recovery scorecard

### DR Congo (cod/) — M23 Conflict, March 2022
- Nightlight and conflict maps
- Recovery scorecard

## Reproducing Results

To regenerate these results from scratch:
```bash
make run COUNTRY=myanmar   # or any country code
python pipeline/recovery_scorecard.py --config config/myanmar.yaml
```

For causal impact analysis:
```bash
cd analysis/mmr
python 01_did_buffer_fixed.py
python 02_pretrends_verified.py
```

## Note

The `data/` and `outputs/` directories are in `.gitignore` to keep the repository
lightweight. This `showcase/` folder provides the key results for review.
To run the full pipeline, configure your credentials in `.env` and run `make run`.
