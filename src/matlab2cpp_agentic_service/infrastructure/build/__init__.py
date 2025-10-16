"""
Build System Support for MATLAB to C++ Conversion.

This package provides support for different build systems:
- CMake: Professional cross-platform build system
- gcc: Direct compiler invocation (default)

Also includes smart helper detection to reduce file bloat.
"""

from .cmake_generator import CMakeGenerator, generate_cmake_file
from .helper_detector import HelperDetector, detect_needed_helpers

__all__ = ['CMakeGenerator', 'generate_cmake_file', 'HelperDetector', 'detect_needed_helpers']

