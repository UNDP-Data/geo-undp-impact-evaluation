# Methodology addendum — Near-real-time nighttime lights (Black Marble)

*Append this section to the repository README (or link it as METHODOLOGY.md).*

## Two nighttime-light products, kept separate by design

| | VIIRS Monthly Composite | Black Marble Daily (VNP46A2) |
|---|---|---|
| Producer | NASA/NOAA | NASA |
| Cadence | Monthly | Daily |
| Latency | ~3–4 months | ~1 week |
| Corrections | Stray-light-corrected monthly aggregate | BRDF-, moonlight-, and atmosphere-corrected at observation level; gap-filled |
| Role in dashboard | Established economic-activity proxy | Early post-crisis signal (experimental) |

**Processing applied to Black Marble.** From Google Earth Engine collection
`NASA/VIIRS/002/VNP46A2`, band `Gap_Filled_DNB_BRDF_Corrected_NTL`:

1. Keep only pixels with `Mandatory_Quality_Flag = 0` (NASA's high-quality
   screen; removes cloud-contaminated and otherwise poor retrievals).
2. Average surviving daily images to a monthly mean composite.
3. Reduce to admin-unit means at 500 m scale (`reduceRegions`, mean).
4. Store as a separate column (`bm_nightlights`) merged into the panel by
   unit and month.

No additional cleaning (no outlier trimming, no snow handling, no
cross-calibration to the monthly composite) is applied beyond NASA's own
quality flag.

**Why the two series are never spliced.** The products use different
calibrations: in validation on Venezuela (335 municipalities, Jan–Apr 2026
overlap) they correlate strongly across units (overall r = 0.95; monthly
r = 0.92–0.97) but differ systematically in level. Splicing them would
manufacture an artificial break at the splice point — exactly where a crisis
event typically sits. They are therefore presented side by side as distinct
indicators, and any causal analysis uses one product consistently across its
full study window.

**Known limitations of the daily product.** Month-to-month country-wide
levels are noisier than the monthly composite, particularly over dim rural
areas and in the wet season (fewer valid observations, gap-filling
artifacts). Relative spatial contrasts (e.g., affected corridor vs. rest of
country, or same-month-last-year comparisons) are more reliable than
absolute level changes and are the recommended reading.

## A note on radar (Sentinel-1) and seasonality

Sentinel-1 is the radar mission of the EU **Copernicus** programme. Radar
sees *through clouds* — acquisition is weather-independent — but backscatter
is not season-independent: it responds to surface moisture, standing water,
and vegetation structure, all of which follow the wet/dry cycle. Seasonal
variability in VV/VH series is therefore expected physics, not a data error.
The dashboard addresses it three ways: recovery scoring benchmarks seasonal
indicators against the *same calendar month of prior years*; time-series
charts display a rainfall reference band; and choropleth comparisons use a
same-month-last-year baseline panel where data allows.
