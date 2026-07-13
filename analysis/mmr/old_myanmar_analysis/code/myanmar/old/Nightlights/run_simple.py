#!/usr/bin/env python3
"""
Simple Cyclone Mocha Analysis - Master Script
==============================================

Runs the complete analysis pipeline with simple, working scripts.

Usage:
    python run_simple.py
"""

import subprocess
import sys
from pathlib import Path


def run_script(script_name, description):
    """Run a Python script."""
    print("\n" + "="*80)
    print(f"  {description}")
    print("="*80)
    
    try:
        result = subprocess.run([sys.executable, script_name], check=True)
        print(f"\n✓ {description} completed")
        return True
    except subprocess.CalledProcessError:
        print(f"\n✗ {description} failed")
        return False
    except FileNotFoundError:
        print(f"\n✗ Script not found: {script_name}")
        print("Make sure all scripts are in the current directory")
        return False


def check_viirs_files():
    """Check if we have VIIRS files."""
    viirs_dir = Path("data/viirs")
    if not viirs_dir.exists():
        return 0
    
    tif_files = list(viirs_dir.glob("*.tif"))
    return len(tif_files)


def check_boundaries():
    """Check if we have boundary files."""
    boundaries_dir = Path("data/boundaries/myanmar_boundaries")
    if not boundaries_dir.exists():
        return False
    
    shp_files = list(boundaries_dir.rglob("*.shp"))
    return len(shp_files) > 0


def main():
    print("="*80)
    print("  CYCLONE MOCHA ANALYSIS")
    print("  Simple Automated Pipeline")
    print("="*80)
    
    print("\nThis will run:")
    print("  1. Download Myanmar boundaries (2 min)")
    print("  2. Download VIIRS nightlights (2-4 hours, ~70 GB)")
    print("  3. Visualize nightlights (10-20 min)")
    print("  4. Budget analysis (after you fill in data)")
    
    input("\nPress Enter to continue (Ctrl+C to cancel)...")
    
    # Create directories
    print("\nCreating directories...")
    Path("data/viirs").mkdir(parents=True, exist_ok=True)
    Path("data/boundaries").mkdir(parents=True, exist_ok=True)
    Path("outputs/mocha_analysis").mkdir(parents=True, exist_ok=True)
    Path("outputs/budget_analysis").mkdir(parents=True, exist_ok=True)
    print("✓ Directories ready")
    
    # Step 1: Myanmar boundaries
    print("\n" + "="*80)
    print("STEP 1: MYANMAR BOUNDARIES")
    print("="*80)
    
    if check_boundaries():
        print("\n✓ Boundaries already exist, skipping")
    else:
        if not run_script("download_myanmar_simple.py", "Download Myanmar Boundaries"):
            print("\n⚠ Boundaries download failed")
            print("You can continue if you download them manually")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return
    
    # Step 2: VIIRS download
    print("\n" + "="*80)
    print("STEP 2: VIIRS NIGHTLIGHTS")
    print("="*80)
    
    viirs_count = check_viirs_files()
    print(f"\nCurrent VIIRS files: {viirs_count}")
    
    if viirs_count >= 40:
        print("✓ Sufficient VIIRS files exist, skipping download")
    else:
        print("\n⚠ This step downloads ~70 GB and takes 2-4 hours")
        print("You can test with just 2023 data first (see instructions)")
        
        response = input("\nDownload VIIRS data now? (y/n): ")
        
        if response.lower() == 'y':
            run_script("download_viirs_simple.py", "Download VIIRS Nightlights")
        else:
            print("\n⊘ Skipping VIIRS download")
            print("\nTo download later:")
            print("  python download_viirs_simple.py")
            print("\nOr test with 2023 only (edit the script start/end dates)")
    
    # Check if we can continue
    viirs_count = check_viirs_files()
    
    if viirs_count < 3:
        print("\n✗ Need at least 3 VIIRS files to continue")
        print("Download VIIRS data first, then re-run this script")
        return
    
    print(f"\n✓ Found {viirs_count} VIIRS files")
    
    # Step 3: Visualization
    print("\n" + "="*80)
    print("STEP 3: VISUALIZE NIGHTLIGHTS")
    print("="*80)
    
    if not run_script("3_visualize_mocha_nightlights.py --viirs-dir ./data/viirs --boundaries-dir ./data/boundaries/processed --output-dir ./outputs/mocha_analysis", 
                     "Visualize Nightlights"):
        
        # Try with subprocess directly
        print("\nTrying alternative method...")
        try:
            subprocess.run([
                sys.executable,
                "3_visualize_mocha_nightlights.py",
                "--viirs-dir", "./data/viirs",
                "--boundaries-dir", "./data/boundaries/processed",
                "--output-dir", "./outputs/mocha_analysis"
            ], check=True)
            print("✓ Visualization completed")
        except:
            print("✗ Visualization failed")
            print("\nRun manually:")
            print("  python 3_visualize_mocha_nightlights.py \\")
            print("    --viirs-dir ./data/viirs \\")
            print("    --boundaries-dir ./data/boundaries/processed \\")
            print("    --output-dir ./outputs/mocha_analysis")
            return
    
    # Step 4: Budget analysis
    print("\n" + "="*80)
    print("STEP 4: BUDGET ANALYSIS")
    print("="*80)
    
    budget_file = Path("outputs/mocha_analysis/budget_disbursement_quarterly.csv")
    
    if not budget_file.exists():
        print("\n⚠ Budget data not found")
        print("\nYou need to:")
        print("  1. Find: outputs/mocha_analysis/budget_template.csv")
        print("  2. Copy to: budget_disbursement_quarterly.csv")
        print("  3. Fill in actual UNDP disbursement data")
        print("  4. Re-run this script")
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE (Through Step 3)")
        print("="*80)
        print("\nOutputs:")
        print("  outputs/mocha_analysis/mocha_before_after_comparison.png")
        print("  outputs/mocha_analysis/mocha_nightlight_time_series.png")
        print("  outputs/mocha_analysis/nightlight_time_series.csv")
        
        print("\nNext: Fill in budget data, then run Step 4")
    else:
        # Run budget analysis
        try:
            subprocess.run([
                sys.executable,
                "4_budget_nightlight_correlation.py",
                "--nightlights", "./outputs/mocha_analysis/nightlights_quarterly_for_budget.csv",
                "--budget", "./outputs/mocha_analysis/budget_disbursement_quarterly.csv",
                "--output-dir", "./outputs/budget_analysis"
            ], check=True)
            
            print("\n" + "="*80)
            print("COMPLETE PIPELINE SUCCESS!")
            print("="*80)
            print("\nAll outputs ready in:")
            print("  outputs/mocha_analysis/")
            print("  outputs/budget_analysis/")
        
        except:
            print("\n✗ Budget analysis failed")
            print("Check the budget data format")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Cancelled by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
