"""
MATLAB2C++ Agentic Service

A comprehensive agentic service for converting MATLAB projects to C++ with
intelligent analysis, planning, and iterative optimization.
"""

from .services import (
    MATLAB2CPPOrchestrator,
    ConversionRequest,
    ConversionResult,
    ConversionStatus
)

__version__ = "1.0.0"
__author__ = "Nec Z"

__all__ = [
    'MATLAB2CPPOrchestrator',
    'ConversionRequest',
    'ConversionResult',
    'ConversionStatus'
]
