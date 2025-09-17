"""
MATLAB2C++ Agent

A comprehensive agentic system for converting MATLAB projects to C++ with
intelligent analysis, planning, and iterative optimization.
"""

from .services import (
    MATLAB2CPPOrchestrator,
    ConversionRequest,
    ConversionResult,
    ConversionStatus
)

__version__ = "1.0.0"
__author__ = "MATLAB2C++ Agent Team"

__all__ = [
    'MATLAB2CPPOrchestrator',
    'ConversionRequest',
    'ConversionResult',
    'ConversionStatus'
]
