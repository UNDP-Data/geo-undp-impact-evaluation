#!/usr/bin/env Rscript

cat("=== Myanmar Monthly DiD script starting ===\n")

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(lubridate)
  library(fixest)
  library(modelsummary)
})

# --------------------------------------------------
# 0. Paths & I/O
# --------------------------------------------------
out_dir <- "outputs/myanmar"
plots_dir <- file.path(out_dir, "plots")
tables_dir <- file.path(out_dir, "tables")

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(plots_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(tables_dir, showWarnings = FALSE, recursive = TRUE)

cat("Outputs will be written to:", normalizePath(out_dir), "\n\n")

panel_path <- file.path(
  "data", "myanmar", "gee_panel_landsat_viirs_chirps_townships_monthly.csv"
)
cat("Loading panel from:", normalizePath(panel_path), "\n")

dt <- fread(panel_path)
cat("Loaded rows:", nrow(dt), "\n")
cat("Columns:", paste(names(dt), collapse = ", "), "\n\n")

# --------------------------------------------------
# 1. Clean & construct monthly variables
# --------------------------------------------------

# keep only needed columns, guard against missing
needed <- c(
  "NDVI", "NL_AVG_RAD", "RAIN_MEAN", "RAIN_TOTAL",
  "admin3_id", "admin3_nam", "period", "period_start",
  "period_end", "undp_treat"
)
present <- intersect(needed, names(dt))
dt <- dt[, ..present]

dt <- dt %>%
  mutate(
    period_start = as.Date(period_start),
    period_end   = as.Date(period_end),
    year         = year(period_start),
    month        = month(period_start),
    ym           = sprintf("%04d-%02d", year, month),
    township_id  = as.character(admin3_id),
    township     = admin3_nam,
    treated      = as.integer(undp_treat == 1)
  ) %>%
  filter(!is.na(township_id))

# Earthquake on 2025-03-28; first full "post" month is 2025-04
quake_date <- as.Date("2025-03-28")

dt <- dt %>%
  mutate(
    ym_date    = as.Date(paste0(ym, "-01")),
    post_quake = ym_date >= as.Date("2025-04-01"),
    post_quake_num = as.integer(post_quake),
    # event-time in months relative to quake month (2025-03)
    event_time = 12 * (year(ym_date) - 2025L) + (month(ym_date) - 3L)
  )

cat("Sample of cleaned data:\n")
print(head(as.data.table(dt)[,
  .(NDVI, NL_AVG_RAD, RAIN_MEAN, RAIN_TOTAL,
    admin3_id, township_id, township,
    period, period_start, period_end,
    treated, year, month, ym, event_time, post_quake)
]))
cat("\n")

# --------------------------------------------------
# 2. TWFE DiD on NDVI (monthly FE, rainfall controls)
# --------------------------------------------------

cat("=== TWFE DiD on NDVI (with rainfall controls, monthly FE) ===\n")

dt_ndvi <- dt[!is.na(NDVI)]

m_ndvi_twfe <- feols(
  NDVI ~ RAIN_MEAN + RAIN_TOTAL + treated:post_quake_num |
    township_id + ym,
  data    = dt_ndvi,
  cluster = ~ township_id
)

print(summary(m_ndvi_twfe))
cat("\n")

# --------------------------------------------------
# 3. TWFE DiD on Nightlights (if post-quake NL exists)
# --------------------------------------------------

cat("=== TWFE DiD on Nightlights (econ recovery) ===\n")

dt_nl <- dt[!is.na(NL_AVG_RAD)]

# Check if we have any truly post-quake months with NL data
has_post_nl <- any(dt_nl$post_quake, na.rm = TRUE)

if (!has_post_nl) {
  cat("⚠ No post-quake nightlights observations yet in the panel.\n")
  cat("  Nightlights DiD cannot be estimated; only pre-trends are available.\n\n")
  m_nl_twfe <- NULL
} else {
  m_nl_twfe <- feols(
    NL_AVG_RAD ~ RAIN_MEAN + RAIN_TOTAL + treated:post_quake_num |
      township_id + ym,
    data    = dt_nl,
    cluster = ~ township_id
  )
  print(summary(m_nl_twfe))
  cat("\n")
}

# --------------------------------------------------
# 4. (Optional) Event-study on NDVI using monthly event_time
# --------------------------------------------------

cat("=== Event-study on NDVI (monthly event time) ===\n")

# restrict to a reasonable window, e.g. [-6, +6] months around quake
dt_es <- dt_ndvi %>%
  filter(event_time >= -6, event_time <= 6)

# Use -1 (one month before quake = 2025-02) as ref category if available
ref_et <- ifelse(any(dt_es$event_time == -1L), -1L, min(dt_es$event_time))

m_es_ndvi <- feols(
  NDVI ~ RAIN_MEAN + RAIN_TOTAL +
    i(event_time, treated, ref = ref_et) |
    township_id + ym,
  data    = dt_es,
  cluster = ~ township_id
)

print(summary(m_es_ndvi))
cat("\n")

# --------------------------------------------------
# 5. AE-style regression table (LaTeX + HTML)
# --------------------------------------------------

cat("=== Building AE-style regression tables ===\n")

models <- list(
  "NDVI, TWFE (township & month FE)" = m_ndvi_twfe
)

if (!is.null(m_nl_twfe)) {
  models[["Nightlights, TWFE (township & month FE)"]] <- m_nl_twfe
}

tex_path  <- file.path(tables_dir, "myanmar_did_regressions_ae.tex")
html_path <- file.path(tables_dir, "myanmar_did_regressions_ae.html")

modelsummary(
  models,
  gof_omit = "IC|Log|Adj|Within",
  fmt = 3,
  stars = TRUE,
  output = c("latex" = tex_path)
)

modelsummary(
  models,
  gof_omit = "IC|Log|Adj|Within",
  fmt = 3,
  stars = TRUE,
  output = c("html" = html_path)
)

cat("LaTeX table written to:", normalizePath(tex_path), "\n")
cat("HTML table written to:", normalizePath(html_path), "\n")

cat("✅ R monthly DiD script finished.\n")
