# Quick Reference: Myanmar Earthquake DiD Analysis Corrections

## 🚨 Critical Issues Found

| Issue | Type | Impact | Status |
|-------|------|--------|--------|
| Cross-border lights in viz | Cosmetic | None on regression | ✅ Fixed |
| March 2025 contamination | **Specification** | **Biased DiD estimate** | ✅ Fixed |

---

## 📋 Corrected Files Provided

### 1. Core Analysis Scripts

| File | Purpose | Key Fix |
|------|---------|---------|
| `state_level_did_regression_corrected.py` | Main DiD analysis | Excludes March 2025 data |
| `visualize_2025_with_interventions_corrected.py` | Full visualization | Masks to Myanmar boundaries |
| `quick_fix_visualization.py` | Fast viz regeneration | Boundary masking only |
| `comprehensive_robustness_checks_CORRECTED.py` | All robustness tests | March validation + all checks |

### 2. Documentation

| File | Contents |
|------|----------|
| `ANALYSIS_CORRECTIONS_SUMMARY.md` | Complete explanation of issues |
| `ROBUSTNESS_CHECKS_CORRECTIONS.md` | Robustness script changes |
| This file | Quick reference guide |

---

## 🔧 How to Run Corrected Analysis

### Step 1: Corrected Regression (REQUIRED)

```bash
python state_level_did_regression_corrected.py \
    --viirs-dir data/viirs_25 \
    --boundaries-dir data/boundaries/processed \
    --undp-support undp_interventions.geojson \
    --output-dir outputs/regression_corrected
```

**Output:** `panel_data_corrected.csv` ← **You need this for Step 2!**

### Step 2: Robustness Checks (OPTIONAL)

```bash
python comprehensive_robustness_checks_CORRECTED.py \
    --panel-data outputs/regression_corrected/panel_data_corrected.csv \
    --boundaries-dir data/boundaries/processed \
    --undp-support undp_interventions.geojson \
    --output-dir outputs/robustness_corrected
```

### Step 3: Fix Visualization (OPTIONAL)

```bash
python quick_fix_visualization.py \
    --viirs-dir data/viirs_25 \
    --boundaries data/boundaries/processed/mmr_admin1.geojson \
    --undp-support undp_interventions.geojson \
    --output earthquake_2025_CORRECTED.png
```

---

## ❓ FAQ

### Q1: Do I have March 2025 data?

Check:
```bash
ls data/viirs_25/*202503*
```

- **If files found:** Your original analysis is biased - rerun with corrected scripts
- **If no files found:** Original analysis specification is fine, but still use corrected scripts for consistency

### Q2: What changes in my results?

**If you had March data:**
- DiD coefficient may change (direction depends on data)
- Standard errors will change
- Significance may change
- More observations excluded (March removed)

**If you didn't have March data:**
- Results identical to original
- Better documentation and validation

### Q3: Which scripts should I use?

| Task | Use This Script |
|------|----------------|
| Main DiD regression | `state_level_did_regression_corrected.py` |
| Full robustness checks | `comprehensive_robustness_checks_CORRECTED.py` |
| Just fix the map | `quick_fix_visualization.py` |
| Full visualizations | `visualize_2025_with_interventions_corrected.py` |

### Q4: Can I use my old panel data?

**NO!** The robustness script **validates** your data and will **stop** if March is detected:

```
⚠️ CRITICAL WARNING: Found X observations from March 2025!
❌ STOPPING: Use panel data from corrected regression script!
```

You **must** use `panel_data_corrected.csv` from Step 1.

---

## 📊 What's Different?

### Regression Script

**Before:**
```python
df['post'] = (df['date'] >= EARTHQUAKE_DATE).astype(int)
# March gets post=0 but contains post-earthquake days!
```

**After:**
```python
# Exclude March during file loading
exclude_months = [(2025, 3)]
viirs_files = find_viirs_files(..., exclude_months=exclude_months)

# Use explicit period definitions
df['post'] = ((df['date'] >= POST_PERIOD_START) & 
              (df['date'] <= POST_PERIOD_END)).astype(int)
```

### Visualization Script

**Before:**
```python
# Shows all lights in bounding box (including Thailand, China)
raster_before.plot(ax=axes[0], ...)
```

**After:**
```python
# Mask to Myanmar boundaries first
raster_before = mask_raster_to_boundaries(raster_before, boundaries_gdf)
raster_before.plot(ax=axes[0], ...)  # Only Myanmar lights
```

### Robustness Script

**Before:**
```python
# Assumed March was excluded, didn't validate
df['relative_month'] = df['month_int'] - 3
```

**After:**
```python
# Validates March exclusion
if (df['date'].dt.month == 3).any():
    raise ValueError("March detected!")

# Corrects relative_month for March gap
# Jan=-2, Feb=-1, [March excluded], Apr=0, May=1, ...
```

---

## ✅ Validation Checklist

Before submitting your analysis, verify:

- [ ] Used `state_level_did_regression_corrected.py` for main analysis
- [ ] No March 2025 data in results (check `panel_data_corrected.csv`)
- [ ] Robustness checks run successfully without March warnings
- [ ] Event study plot shows March exclusion zone
- [ ] Visualization shows only Myanmar (no Thailand/China lights)
- [ ] All documentation mentions March exclusion

---

## 📈 Expected Results

### Period Definitions (Corrected)

```
Pre-period:   Jan-Feb 2025  (2 months)
[EXCLUDED]:   Mar 2025      (contaminated)
Post-period:  Apr-Sep 2025  (6 months)
```

### Data Validation Output

```
✓ VALIDATION PASSED: No March 2025 data detected
✓ Relative month calculation:
  -2 (Jan):  X observations
  -1 (Feb):  X observations
   0 (Apr):  X observations
   1 (May):  X observations
   ...
```

---

## 🔍 Comparison: Before vs After

| Aspect | Original | Corrected |
|--------|----------|-----------|
| **March treatment** | Ambiguous | Excluded |
| **Period coding** | ≥ earthquake date | Explicit dates |
| **Visualization** | Shows neighbors | Myanmar only |
| **Validation** | None | Comprehensive |
| **Error handling** | Silent | Stops with message |

---

## 📞 If You Need Help

1. **Check error messages** - They tell you exactly what's wrong
2. **Review documentation** - ANALYSIS_CORRECTIONS_SUMMARY.md explains everything
3. **Verify inputs** - Make sure you're using corrected panel data
4. **Compare outputs** - Look at before/after results

---

## 🎯 Bottom Line

### What You Must Do:

1. ✅ Rerun analysis with `state_level_did_regression_corrected.py`
2. ✅ Use `panel_data_corrected.csv` for all downstream analyses
3. ✅ Update visualizations to show Myanmar only

### What's Optional:

- 📊 Run comprehensive robustness checks
- 📈 Compare old vs new results
- 📝 Document specification change in paper

### Why This Matters:

**Scientific validity** - Proper treatment definition  
**Publication quality** - Addresses reviewer concerns  
**Transparency** - Clear documentation  
**Reproducibility** - Explicit validation

---

**Last Updated:** 2025-11-07  
**Version:** 2.0  
**Files Location:** `/mnt/user-data/outputs/`

All corrected scripts are ready to use! 🚀
