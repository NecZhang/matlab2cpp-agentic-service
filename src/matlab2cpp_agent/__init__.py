"""
MATLAB2C++ Agent

A comprehensive agentic system for converting MATLAB projects to C++ with
intelligent analysis, planning, and iterative optimization.
"""

from .services import (
    MATLAB2CPPService,
    convert_matlab_project,
    convert_matlab_script,
    ConversionRequest,
    ConversionPlan,
    ConversionResult,
    ConversionStatus
)

__version__ = "1.0.0"
__author__ = "MATLAB2C++ Agent Team"

__all__ = [
    'MATLAB2CPPService',
    'convert_matlab_project',
    'convert_matlab_script',
    'ConversionRequest',
    'ConversionPlan',
    'ConversionResult',
    'ConversionStatus'
]
