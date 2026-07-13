#!/usr/bin/env python3
"""
Complete Cyclone Mocha Analysis Pipeline Runner
===============================================

Runs all scripts in sequence with proper error handling.

Usage:
    python run_pipeline.py
"""

import subprocess
import sys
from pathlib import Path


def run_command(description, command):
    """Run a command and handle errors."""
    print("\n" + "="*80)
    print(f"STEP: {description}")
    print("="*80)
    print(f"Command: {' '.join(command)}\n")
    
    try:
        result = subprocess.run(command, check=True)
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with error code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n✗ Command not found: {command[0]}")
        print("Make sure Python is in your PATH")
        return False


def check_files_exist(file_patterns, directory):
    """Check if files matching patterns exist."""
    directory = Path(directory)
    if not directory.exists():
        return False, 0
    
    count = 0
    for pattern in file_patterns:
        count += len(list(directory.glob(pattern)))
    
    return count > 0, count


def main():
    print("="*80)
    print("  CYCLONE MOCHA ANALYSIS PIPELINE")
    print("  Automated Execution")
    print("="*80)
    print("\nThis script will run all 4 steps of the analysis:")
    print("  1. Download VIIRS nightlights (automated)")
    print("  2. Get Myanmar boundaries (automated)")
    print("  3. Visualize nightlights")
    print("  4. Budget correlation analysis")
    
    print("\n" + "="*80)
    print("SETUP")
    print("="*80)
    
    # Create directories
    print("\nCreating directories...")
    dirs = [
        "data/viirs",
        "data/boundaries",
        "outputs/mocha_analysis",
        "outputs/budget_analysis"
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {d}")
    
    # Step 1: Download VIIRS
    print("\n" + "="*80)
    print("STEP 1: DOWNLOAD VIIRS NIGHTLIGHTS")
    print("="*80)
    print("\nThis will download ~46 files (~70 GB total)")
    print("It may take 2-4 hours depending on your connection")
    
    response = input("\nDo you want to download VIIRS data now? (y/n): ").lower()
    
    if response == 'y':
        success = run_command(
            "Download VIIRS nightlights",
            [
                sys.executable,
                "1_download_viirs_auto.py",
                "--output-dir", "./data/viirs",
                "--start", "2021-01",
                "--end", "2024-10",
                "--extract",
                "--skip-existing"
            ]
        )
        
        if not success:
            print("\n⚠ VIIRS download had issues. You can:")
            print("  1. Re-run this script to continue")
            print("  2. Download manually from: https://eogdata.mines.edu/nighttime_light/monthly/v10/")
            print("  3. Skip to Step 2 if you already have the files")
    else:
        print("\n⊘ Skipping VIIRS download")
        print("Make sure you have VIIRS .tif files in data/viirs/ before continuing")
    
    # Check if we have VIIRS files
    has_viirs, viirs_count = check_files_exist(["*.tif"], "data/viirs")
    print(f"\nVIIRS files found: {viirs_count}")
    
    # Step 2: Myanmar boundaries
    if not run_command(
        "Download Myanmar boundaries",
        [
            sys.executable,
            "2_get_myanmar_boundaries.py",
            "--output-dir", "./data/boundaries",
            "--create-map"
        ]
    ):
        print("\n⚠ Myanmar boundaries download failed")
        print("You can download manually from: https://data.humdata.org/dataset/cod-ab-mmr")
        response = input("\nContinue anyway? (y/n): ").lower()
        if response != 'y':
            print("\nExiting. Fix the boundaries issue and re-run.")
            return
    
    # Check if we have boundaries
    has_boundaries, _ = check_files_exist(["*.shp"], "data/boundaries/myanmar_boundaries")
    
    if not has_boundaries:
        print("\n✗ No boundary files found. Cannot continue.")
        print("Please download Myanmar boundaries manually and re-run.")
        return
    
    # Step 3: Visualize nightlights
    if not has_viirs:
        print("\n✗ No VIIRS files found. Skipping visualization.")
        print("Download VIIRS data first, then re-run this script.")
        return
    
    if not run_command(
        "Visualize nightlights",
        [
            sys.executable,
            "3_visualize_mocha_nightlights.py",
            "--viirs-dir", "./data/viirs",
            "--boundaries-dir", "./data/boundaries/processed",
            "--output-dir", "./outputs/mocha_analysis",
            "--start-date", "2021-01",
            "--end-date", "2024-10"
        ]
    ):
        print("\n⚠ Visualization failed")
        print("Check the errors above. Common issues:")
        print("  - Missing VIIRS files")
        print("  - Missing boundaries")
        print("  - Insufficient memory (try fewer months)")
        return
    
    # Step 4: Budget correlation
    print("\n" + "="*80)
    print("STEP 4: BUDGET CORRELATION")
    print("="*80)
    
    budget_file = Path("outputs/mocha_analysis/budget_disbursement_quarterly.csv")
    
    if not budget_file.exists():
        print("\n⚠ Budget file not found")
        print("You need to fill in UNDP budget data first.")
        print(f"\nTemplate created at: outputs/mocha_analysis/budget_template.csv")
        print("\nSteps:")
        print("  1. Copy budget_template.csv to budget_disbursement_quarterly.csv")
        print("  2. Fill in actual UNDP disbursement amounts")
        print("  3. Re-run this script")
        
        print("\nAnalysis complete through Step 3!")
        print("\nOutputs available in: outputs/mocha_analysis/")
        return
    
    run_command(
        "Budget correlation analysis",
        [
            sys.executable,
            "4_budget_nightlight_correlation.py",
            "--nightlights", "./outputs/mocha_analysis/nightlights_quarterly_for_budget.csv",
            "--budget", "./outputs/mocha_analysis/budget_disbursement_quarterly.csv",
            "--output-dir", "./outputs/budget_analysis"
        ]
    )
    
    # Final summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    
    print("\nOutputs:")
    print("  1. Maps: outputs/mocha_analysis/")
    print("  2. Time series: outputs/mocha_analysis/")
    print("  3. Budget analysis: outputs/budget_analysis/")
    
    print("\nKey files:")
    print("  - mocha_before_after_comparison.png")
    print("  - mocha_nightlight_time_series.png")
    print("  - budget_nightlight_correlation.png")
    print("  - impact_story.txt")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
