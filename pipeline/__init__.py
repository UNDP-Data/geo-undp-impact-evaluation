"""
Impact Evaluation Pipeline
==========================
A modular pipeline for satellite-based impact evaluation.

Usage:
    make run COUNTRY=gaza
    make run COUNTRY=myanmar
"""

__version__ = "2.0.0"

# Only import modules that exist - let individual scripts handle their own imports
# This prevents import errors when running individual modules

__all__ = [
    'gee_export',
    'preprocessing', 
    'visualizations',
    'acled_download',
]
