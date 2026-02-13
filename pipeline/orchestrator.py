#!/usr/bin/env python3
"""
pipeline/orchestrator.py
========================
Routes pipeline stages to appropriate modules.

This is a thin wrapper that calls the appropriate module based on --stage.

Usage:
    python -m pipeline.orchestrator --config config/myanmar.yaml --stage preprocess
    python -m pipeline.orchestrator --config config/myanmar.yaml --stage visuals
    python -m pipeline.orchestrator --config config/myanmar.yaml --stage data
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Pipeline orchestrator')
    parser.add_argument('--config', '-c', required=True, help='Path to country config YAML')
    parser.add_argument('--stage', '-s', required=True, 
                       choices=['data', 'preprocess', 'visuals', 'analysis'],
                       help='Pipeline stage to run')
    
    args = parser.parse_args()
    
    config_path = args.config
    
    if args.stage == 'data':
        # Data acquisition - run GEE export
        print("Running GEE data acquisition...")
        try:
            # Try debug version first (more reliable)
            subprocess.run([
                sys.executable, '-m', 'pipeline.gee_export_debug',
                '--config', config_path
            ], check=True)
        except subprocess.CalledProcessError:
            # Fall back to original
            subprocess.run([
                sys.executable, '-m', 'pipeline.gee_export',
                '--config', config_path
            ], check=True)
    
    elif args.stage == 'preprocess':
        # Preprocessing
        print("Running preprocessing...")
        subprocess.run([
            sys.executable, '-m', 'pipeline.preprocessing',
            '--config', config_path
        ], check=True)
    
    elif args.stage == 'visuals':
        # Visualizations
        print("Running visualizations...")
        subprocess.run([
            sys.executable, '-m', 'pipeline.visualizations',
            '--config', config_path
        ], check=True)
    
    elif args.stage == 'analysis':
        # Analysis - this would call analysis scripts
        print("Analysis stage - run analysis scripts directly")
        print(f"  e.g., python analysis/mmr/02_did_comprehensive.py")


if __name__ == "__main__":
    main()
