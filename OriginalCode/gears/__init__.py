"""
Gears Package

This package provides functionality for gear design and analysis, including:
- Rack cutter modeling (standard and complex geometries)
- Gear generation using envelope theory
- Ease-off micro-geometry modifications
- Gear surface analysis and visualization

Main modules:
- core: Rack cutter and gear generation classes
- ease_off: Micro-geometry modification tools
- data_structs: Data structures for gear parameters

Examples:
- examples/gear_generation_example.py: Rack cutter and gear generation demo
- examples/ease_off_example.py: Micro-geometry modification demo
"""

# Main classes
from .main.core import rackCutter, Gear
from .main.ease_off import crowning_RZE
from .main.data_structs import rackData

# Version info
__version__ = "0.1.0"
__author__ = "Your Name"

# Package exports
__all__ = [
    # Core gear functionality
    "rackCutter",
    "Gear", 
    
    # Ease-off functionality
    "crowning_RZE",
    
    # Data structures
    "rackData",
]