# =============================================================================
# IMPACT EVALUATION PIPELINE - Makefile
# =============================================================================
# Usage:
#   make run COUNTRY=myanmar         # Full pipeline (skips existing data)
#   make run-fresh COUNTRY=myanmar   # Force re-download everything
#   make visuals COUNTRY=myanmar     # Just regenerate figures
#   make status COUNTRY=myanmar      # Check what exists
# =============================================================================

COUNTRY ?= myanmar
CONFIG = config/$(COUNTRY).yaml

ifeq ($(COUNTRY),gaza)
    CODE = pse
else ifeq ($(COUNTRY),myanmar)
    CODE = mmr
else ifeq ($(COUNTRY),afghanistan)
    CODE = afg
else ifeq ($(COUNTRY),ukraine)
    CODE = ukr
else
    CODE = $(COUNTRY)
endif

DATA_DIR = data/$(CODE)
OUTPUT_DIR = outputs/$(CODE)
FIGURES_DIR = $(OUTPUT_DIR)/figures
ANALYSIS_DIR = analysis/$(CODE)

.PHONY: all run run-fresh data gee acled wfp gt undp preprocess visuals analysis clean status help boundaries

# =============================================================================
# MAIN TARGETS
# =============================================================================

## Run full pipeline (skips existing data)
run: data preprocess visuals trends
	@echo ""
	@echo "============================================================"
	@echo "✓ Pipeline complete for $(COUNTRY)"
	@echo "============================================================"
	@echo "  Panel data: $(OUTPUT_DIR)/panel_data.csv"
	@echo "  Figures:    $(FIGURES_DIR)/"
	@echo ""

## Force re-download and re-run everything
run-fresh: clean
	$(MAKE) run COUNTRY=$(COUNTRY)

## Run everything including analysis
all: run analysis
	@echo "✓ Full pipeline with analysis complete for $(COUNTRY)"

# =============================================================================
# STAGE 0: BOUNDARIES
# =============================================================================

boundaries:
	@echo ""
	@echo "--- Admin Boundaries ---"
	@BOUNDARY_FILE=$$(python -c "import yaml; c=yaml.safe_load(open('$(CONFIG)')); print(c['boundaries']['file_path'])" 2>/dev/null); \
	if [ -f "$$BOUNDARY_FILE" ]; then \
		UNITS=$$(python -c "import geopandas as gpd; print(len(gpd.read_file('$$BOUNDARY_FILE')))" 2>/dev/null); \
		echo "  ✓ Boundaries exist: $$BOUNDARY_FILE ($$UNITS units) — skipping"; \
	else \
		python -m pipeline.download_boundaries --config $(CONFIG); \
	fi

# =============================================================================
# STAGE 1: DATA ACQUISITION (with skip-if-exists)
# =============================================================================

data: boundaries gee external undp
	@echo "✓ Data acquisition complete"

## GEE satellite data (skip if CSV exists)
gee:
	@echo ""
	@echo "--- GEE Satellite Data ---"
	@GEE_FILE=$(DATA_DIR)/gee_panel_$(CODE)_centroid.csv; \
	if [ -f "$$GEE_FILE" ]; then \
		ROWS=$$(python -c "import pandas as pd; print(len(pd.read_csv('$$GEE_FILE')))" 2>/dev/null); \
		echo "  ✓ GEE data exists: $$GEE_FILE ($$ROWS rows) — skipping"; \
		echo "  (use 'make run-fresh' to re-download)"; \
	else \
		python -m pipeline.gee_export_debug --config $(CONFIG); \
	fi

external: acled wfp gt
	@echo "✓ External data complete"

## ACLED conflict data (skip if CSV exists)
acled:
	@echo ""
	@echo "--- ACLED Conflict Data ---"
	@if [ -f "$(DATA_DIR)/external/acled_conflict.csv" ]; then \
		ROWS=$$(python -c "import pandas as pd; print(len(pd.read_csv('$(DATA_DIR)/external/acled_conflict.csv')))" 2>/dev/null); \
		echo "  ✓ ACLED data exists ($$ROWS events) — skipping"; \
	elif [ -n "$$ACLED_EMAIL" ] && [ -n "$$ACLED_KEY" ]; then \
		python -m pipeline.acled_download --config $(CONFIG) \
			--email "$$ACLED_EMAIL" --password "$$ACLED_KEY"; \
	elif [ -n "$$ACLED_EMAIL" ] && [ -n "$$ACLED_PASSWORD" ]; then \
		python -m pipeline.acled_download --config $(CONFIG) \
			--email "$$ACLED_EMAIL" --password "$$ACLED_PASSWORD"; \
	else \
		echo "  ⚠ ACLED credentials not set. Set ACLED_EMAIL and ACLED_KEY"; \
	fi

## WFP food prices (skip if CSV exists)
wfp:
	@echo ""
	@echo "--- WFP Food Prices ---"
	@if [ -f "$(DATA_DIR)/external/wfp_prices.csv" ]; then \
		echo "  ✓ WFP data exists — skipping"; \
	else \
		python -m pipeline.wfp_download --config $(CONFIG) 2>/dev/null || \
			echo "  ⚠ WFP download not available or failed"; \
	fi

## Google Trends (skip if CSV exists)
gt:
	@echo ""
	@echo "--- Google Trends ---"
	@if [ -f "$(DATA_DIR)/external/google_trends.csv" ]; then \
		ROWS=$$(python -c "import pandas as pd; print(len(pd.read_csv('$(DATA_DIR)/external/google_trends.csv')))" 2>/dev/null); \
		echo "  ✓ Google Trends data exists ($$ROWS records) — skipping"; \
	elif [ -f "pipeline/google_trends.py" ]; then \
		python -m pipeline.google_trends --config $(CONFIG) 2>/dev/null || \
			echo "  ⚠ Google Trends download failed"; \
	else \
		echo "  ⚠ Google Trends module not found"; \
	fi

## UNDP projects (skip if CSV exists with >0 rows)
undp:
	@echo ""
	@echo "--- UNDP Projects ---"
	@if [ -f "$(DATA_DIR)/external/undp_projects.csv" ]; then \
		ROWS=$$(python -c "import pandas as pd; print(len(pd.read_csv('$(DATA_DIR)/external/undp_projects.csv')))" 2>/dev/null); \
		if [ "$$ROWS" -gt 0 ] 2>/dev/null; then \
			echo "  ✓ UNDP projects exist ($$ROWS projects) — skipping"; \
		else \
			python -m pipeline.undp_projects --config $(CONFIG); \
		fi; \
	else \
		python -m pipeline.undp_projects --config $(CONFIG); \
	fi

# =============================================================================
# STAGE 2: PREPROCESSING
# =============================================================================

preprocess:
	@echo ""
	@echo "--- Preprocessing ---"
	@python -m pipeline.orchestrator --config $(CONFIG) --stage preprocess

# =============================================================================
# STAGE 3: VISUALIZATIONS
# =============================================================================

visuals:
	@echo ""
	@echo "--- Visualizations ---"
	@python -m pipeline.visualizations --config $(CONFIG)
	@echo ""
	@echo "--- UNDP Timeline ---"
	@if [ -f "$(DATA_DIR)/external/undp_projects.csv" ]; then \
		python -m pipeline.undp_timeline --config $(CONFIG) 2>/dev/null || \
			echo "  ⚠ UNDP timeline generation failed"; \
	else \
		echo "  ⚠ No UNDP project data for timeline"; \
	fi
	@echo "✓ Figures saved to $(FIGURES_DIR)/"

trends:
	@echo ""
	@echo "--- Google Trends Plots ---"
	@if [ -f "$(DATA_DIR)/external/google_trends.csv" ]; then \
		python -m pipeline.google_trends_plots --config $(CONFIG); \
	else \
		echo "  ⚠ Google Trends data not found"; \
	fi

# =============================================================================
# STAGE 4: ANALYSIS
# =============================================================================

analysis: did pretrends
	@echo ""
	@echo "✓ Analysis complete for $(COUNTRY)"

did:
	@echo ""
	@echo "--- DiD Analysis ---"
	@if [ -f "$(ANALYSIS_DIR)/02_did_buffer_fixed.py" ]; then \
		python $(ANALYSIS_DIR)/02_did_buffer_fixed.py; \
	elif [ -f "$(ANALYSIS_DIR)/02_did_analysis.py" ]; then \
		python $(ANALYSIS_DIR)/02_did_analysis.py; \
	else \
		echo "  ⚠ No DiD script found in $(ANALYSIS_DIR)/"; \
	fi

pretrends:
	@echo ""
	@echo "--- Pre-Trends Analysis ---"
	@if [ -f "$(ANALYSIS_DIR)/04_pretrends_verified.py" ]; then \
		python $(ANALYSIS_DIR)/04_pretrends_verified.py; \
	elif [ -f "$(ANALYSIS_DIR)/04_pretrends_time_series.py" ]; then \
		python $(ANALYSIS_DIR)/04_pretrends_time_series.py; \
	else \
		echo "  ⚠ No pre-trends script found in $(ANALYSIS_DIR)/"; \
	fi

# =============================================================================
# UTILITIES
# =============================================================================

status:
	@echo ""
	@echo "============================================================"
	@echo "DATA STATUS: $(COUNTRY) ($(CODE))"
	@echo "============================================================"
	@echo ""
	@echo "--- Boundaries ---"
	@BOUNDARY_FILE=$$(python -c "import yaml; c=yaml.safe_load(open('$(CONFIG)')); print(c['boundaries']['file_path'])" 2>/dev/null); \
	if [ -f "$$BOUNDARY_FILE" ]; then \
		echo "  ✓ $$BOUNDARY_FILE"; \
	else \
		echo "  ✗ Not found"; \
	fi
	@echo ""
	@echo "--- GEE Satellite ---"
	@if [ -f "$(DATA_DIR)/gee_panel_$(CODE)_centroid.csv" ]; then \
		python -c "import pandas as pd; df=pd.read_csv('$(DATA_DIR)/gee_panel_$(CODE)_centroid.csv'); \
			print(f'  ✓ {len(df):,} rows, {df[\"admin_id\"].nunique()} units')"; \
	else \
		echo "  ✗ Not found"; \
	fi
	@echo ""
	@echo "--- ACLED ---"
	@if [ -f "$(DATA_DIR)/external/acled_conflict.csv" ]; then \
		python -c "import pandas as pd; print(f'  ✓ {len(pd.read_csv(\"$(DATA_DIR)/external/acled_conflict.csv\")):,} events')"; \
	else \
		echo "  ✗ Not found"; \
	fi
	@echo ""
	@echo "--- UNDP Projects ---"
	@if [ -f "$(DATA_DIR)/external/undp_projects.csv" ]; then \
		python -c "import pandas as pd; df=pd.read_csv('$(DATA_DIR)/external/undp_projects.csv'); \
			print(f'  ✓ {len(df)} projects')"; \
	else \
		echo "  ✗ Not found"; \
	fi
	@echo ""
	@echo "--- Google Trends ---"
	@if [ -f "$(DATA_DIR)/external/google_trends.csv" ]; then \
		python -c "import pandas as pd; df=pd.read_csv('$(DATA_DIR)/external/google_trends.csv'); \
			print(f'  ✓ {len(df)} records')"; \
	else \
		echo "  ✗ Not found"; \
	fi
	@echo ""
	@echo "--- Panel Data ---"
	@if [ -f "$(OUTPUT_DIR)/panel_data.csv" ]; then \
		python -c "import pandas as pd; df=pd.read_csv('$(OUTPUT_DIR)/panel_data.csv'); \
			print(f'  ✓ {len(df):,} rows, {df.columns.tolist()[:6]}...')"; \
	else \
		echo "  ✗ Not found"; \
	fi
	@echo ""
	@echo "--- Figures ---"
	@echo "  Total: $$(ls $(FIGURES_DIR)/*.png 2>/dev/null | wc -l | tr -d ' ') files"
	@echo ""

clean:
	@echo "Cleaning outputs for $(COUNTRY)..."
	rm -rf $(OUTPUT_DIR)/*
	@echo "✓ Cleaned $(OUTPUT_DIR)/"

help:
	@echo ""
	@echo "IMPACT EVALUATION PIPELINE"
	@echo "=========================="
	@echo ""
	@echo "Usage: make <target> COUNTRY=<country>"
	@echo ""
	@echo "Countries: myanmar, afghanistan, ukraine, gaza"
	@echo ""
	@echo "Main targets:"
	@echo "  run          Full pipeline (skips existing data)"
	@echo "  run-fresh    Force re-download everything"
	@echo "  visuals      Regenerate figures only"
	@echo "  status       Check what data exists"
	@echo ""
	@echo "Data targets:"
	@echo "  boundaries   Download admin boundaries"
	@echo "  gee          Satellite data (GEE)"
	@echo "  acled        Conflict data"
	@echo "  undp         UNDP projects"
	@echo "  gt           Google Trends"
	@echo ""
	@echo "Analysis:"
	@echo "  analysis     Run DiD + pre-trends"
	@echo ""
	@echo "Examples:"
	@echo "  make run COUNTRY=myanmar"
	@echo "  make visuals COUNTRY=gaza"
	@echo "  make status COUNTRY=afghanistan"
	@echo "  for c in myanmar gaza afghanistan ukraine; do make run COUNTRY=\$$c; done"
	@echo ""
