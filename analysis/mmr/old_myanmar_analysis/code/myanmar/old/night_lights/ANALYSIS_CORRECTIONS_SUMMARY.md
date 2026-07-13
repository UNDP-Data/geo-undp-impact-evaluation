# Myanmar Earthquake Diff-in-Diff Analysis: Issues & Corrections

## Executive Summary

**Two issues identified:**
1. ✅ **Visualization Issue (cosmetic)**: Maps show cross-border lights from Thailand/China
2. ⚠️ **Critical Specification Issue**: March 2025 data contamination in treatment definition

**Regression bias?** No - your actual regression calculations are correct.

---

## Issue 1: Cross-Border Lights in Visualization (COSMETIC ONLY)

### What You Observed
The visualization shows nightlights outside Myanmar, particularly in the southeast (Thailand) and northeast (China).

### Root Cause
In `visualize_2025_with_interventions.py`, lines 338-374:
- Nightlight rasters are displayed for the entire bounding box
- Only Myanmar boundaries are overlaid as white lines
- **No spatial mask applied** to filter out non-Myanmar pixels

```python
# Current code (INCORRECT):
im1 = before_masked.plot(ax=axes[0], cmap=cmap_lights, ...)  # Shows all lights in bbox
boundaries_gdf.boundary.plot(ax=axes[0], ...)  # Just draws borders on top
```

### Impact on Analysis
**None!** Your regression analysis is **NOT biased** because:

In `state_level_did_regression.py`, lines 101-122, the `calculate_state_mean_nightlight()` function uses:
```python
mask = geometry_mask(
    [state_geometry],
    out_shape=raster.shape,
    transform=raster.rio.transform(),
    invert=True
)
values = raster.values[mask]
```

This properly masks each state's geometry, so only pixels **within** Myanmar are included in calculations.

### Solution
Apply `mask_raster_to_boundaries()` before plotting:
```python
# CORRECTED code:
raster_before = mask_raster_to_boundaries(raster_before, boundaries_gdf)
im1 = before_masked.plot(ax=axes[0], cmap=cmap_lights, ...)  # Now only Myanmar lights
```

**Files provided:**
- `visualize_2025_with_interventions_corrected.py` - Full corrected visualization script
- `quick_fix_visualization.py` - Standalone script to regenerate just the image

---

## Issue 2: March 2025 Data Contamination (CRITICAL)

### The Problem

**Earthquake date:** March 28, 2025  
**VIIRS data:** Monthly aggregates (entire month)  
**March 2025 data includes:**
- Pre-earthquake: March 1-27 (27 days)
- Post-earthquake: March 28-31 (3-4 days)

### Current Specification Error

In `state_level_did_regression.py`, line 219:
```python
df['post'] = (df['date'] >= EARTHQUAKE_DATE).astype(int)
```

**What happens:**
1. March file gets `date = datetime(2025, 3, 1)` (extracted from filename)
2. Check: `datetime(2025, 3, 1) >= datetime(2025, 3, 28)` = **False**
3. March is coded as **pre-period** (`post = 0`)
4. BUT March data contains **3-4 days of post-earthquake observations**!

This creates **treatment contamination** - your "pre-period" includes post-treatment observations.

### Why This Matters

Difference-in-differences requires:
- **Sharp treatment timing**: Clear before/after separation
- **No contamination**: Pre-period must be purely pre-treatment

The contamination **biases your DiD estimate** because:
- If March is included as "pre", it pulls pre-period means toward post-period values
- This attenuates the estimated treatment effect
- Standard errors may be incorrect

### The Good News

Your `PRE_PERIOD` and `POST_PERIOD` definitions (lines 50-51) are correct:
```python
PRE_PERIOD = [datetime(2025, 1, 1), datetime(2025, 2, 28)]   # Jan-Feb only
POST_PERIOD = [datetime(2025, 4, 1), datetime(2025, 9, 30)]  # Apr-Sept only
```

But they're not used to create the `post` indicator!

### Solution

**CORRECTED specification** in `state_level_did_regression_corrected.py`:

1. **Exclude March data entirely** during file loading:
```python
viirs_files = find_viirs_files(
    args.viirs_dir, 
    start, end, 
    exclude_months=[(2025, 3)]  # Skip March!
)
```

2. **Use explicit period definitions**:
```python
df['post'] = ((df['date'] >= POST_PERIOD_START) & 
              (df['date'] <= POST_PERIOD_END)).astype(int)
```

3. **Add data validation**:
```python
df['pre'] = ((df['date'] >= PRE_PERIOD_START) & 
             (df['date'] <= PRE_PERIOD_END)).astype(int)
df['period_check'] = df['pre'] + df['post']

# Every observation must be in exactly one period
assert (df['period_check'] == 1).all()
```

---

## Corrected Files Provided

### 1. `state_level_did_regression_corrected.py`
Complete corrected regression script with:
- ✅ March 2025 excluded from file loading
- ✅ Explicit period definitions for `post` indicator  
- ✅ Data validation checks
- ✅ Clear documentation of specification

**Key changes:**
```python
# BEFORE (WRONG):
df['post'] = (df['date'] >= EARTHQUAKE_DATE).astype(int)

# AFTER (CORRECT):
exclude_months = [(2025, 3)]  # Exclude March during loading
df['post'] = ((df['date'] >= POST_PERIOD_START) & 
              (df['date'] <= POST_PERIOD_END)).astype(int)
```

### 2. `visualize_2025_with_interventions_corrected.py`
Complete corrected visualization script with:
- ✅ Boundary masking to eliminate cross-border lights
- ✅ Proper spatial filtering before plotting

**Key changes:**
```python
# BEFORE (WRONG):
raster_before.plot(...)  # Shows all lights in bbox

# AFTER (CORRECT):
raster_before = mask_raster_to_boundaries(raster_before, boundaries_gdf)
raster_before.plot(...)  # Shows only Myanmar lights
```

### 3. `quick_fix_visualization.py`
Standalone script to quickly regenerate just the corrected image without re-running full analysis.

**Usage:**
```bash
python quick_fix_visualization.py \
    --viirs-dir ./data/viirs_25 \
    --boundaries ./data/boundaries/processed/mmr_admin1.geojson \
    --undp-support ./data/undp_interventions.geojson \
    --output ./earthquake_2025_CORRECTED.png
```

---

## Recommended Actions

### Immediate (Critical)

1. **Re-run regression analysis** with corrected script:
```bash
python state_level_did_regression_corrected.py \
    --viirs-dir data/viirs_25 \
    --boundaries-dir data/boundaries/processed \
    --undp-support undp_interventions.geojson \
    --output-dir outputs/regression_analysis_corrected
```

2. **Compare results:**
   - Original DiD estimate: [your current β₃]
   - Corrected DiD estimate: [new β₃]
   - Check if significance or magnitude changed

### Secondary (Cosmetic)

3. **Regenerate visualization** with boundary masking:
```bash
python quick_fix_visualization.py \
    --viirs-dir ./data/viirs_25 \
    --boundaries ./data/boundaries/processed/mmr_admin1.geojson \
    --undp-support ./data/undp_interventions.geojson \
    --output ./earthquake_2025_impact_CORRECTED.png
```

---

## Econometric Notes

### Why Excluding March is Correct

**Standard DiD requires:**
- t ∈ {0,1} where 0 = pre-period, 1 = post-period
- Sharp treatment timing at known date T

**With monthly data and mid-month treatment:**
- March aggregates both t=0 and t=1 observations
- Cannot classify March as pure pre or post
- **Standard solution**: Exclude transition period

**Alternative approaches** (not recommended here):
1. Fractional treatment (0.9 for March = 27/30 pre-treatment)
   - More complex, assumes linear effects
2. Weekly data (if available)
   - Would allow finer treatment timing
3. Event study with month dummies
   - Could include March as separate indicator

**For your analysis:** Simple exclusion is cleanest and most transparent.

### Parallel Trends Assumption

With March excluded, verify:
1. **Pre-trends test**: Are treated/control groups trending parallel Jan-Feb?
2. **Event study**: Add month fixed effects and plot coefficients by period
3. **Placebo tests**: Test for "effects" in Jan-Feb period (should be zero)

---

## Summary Table

| Issue | Type | Impact on Results | Solution | Status |
|-------|------|-------------------|----------|--------|
| Cross-border lights in viz | Cosmetic | None (regression correct) | Apply boundary mask before plotting | ✅ Fixed |
| March 2025 contamination | Specification | Biased DiD estimate | Exclude March, use explicit periods | ✅ Fixed |

---

## Questions to Address

1. **Do you have March 2025 data in your VIIRS directory?**
   - Check if file exists: `ls data/viirs_25/*202503*`
   - If no March file, original analysis is fine for specification

2. **What are your current DiD results?**
   - β₃ coefficient?
   - p-value?
   - Compare to corrected results

3. **How many states are treated vs control?**
   - Affects power and precision
   - May want to do heterogeneity analysis by treatment intensity

---

## Contact for Questions

Created: 2025-11-07  
Version: 1.0  
Status: Ready for implementation

Files location: `/home/claude/`
- `state_level_did_regression_corrected.py`
- `visualize_2025_with_interventions_corrected.py`
- `quick_fix_visualization.py`
