"""
MATLAB2C++ CLI Tools

This module provides command-line interfaces for MATLAB to C++ conversion.
"""

from .general_converter import main as general_converter_main

__all__ = [
    'general_converter_main'
]
