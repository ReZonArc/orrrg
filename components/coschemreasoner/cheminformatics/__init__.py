"""
OpenCog Cheminformatics Framework

This package provides cheminformatics capabilities with specialized support
for cosmetic chemistry applications through the OpenCog AtomSpace framework.

Modules:
    types: Atom type definitions for cosmetic chemistry
"""

__version__ = "1.0.0"
__author__ = "OpenCog Cheminformatics Team"

# Import main classes for convenience
try:
    from .types.cosmetic_atoms import *
except ImportError:
    # Fallback if OpenCog is not available
    pass