"""
MATLAB2C++ Services

This module provides high-level services for MATLAB to C++ conversion.
"""

from .matlab2cpp_service import (
    MATLAB2CPPService,
    convert_matlab_project,
    convert_matlab_script,
    ConversionRequest,
    ConversionPlan,
    ConversionResult,
    ConversionStatus
)

from .matlab2cpp_orchestrator import (
    MATLAB2CPPOrchestrator,
    ConversionRequest as OrchestratorConversionRequest,
    ConversionResult as OrchestratorConversionResult,
    ConversionStatus as OrchestratorConversionStatus
)

__all__ = [
    'MATLAB2CPPService',
    'MATLAB2CPPOrchestrator',
    'convert_matlab_project',
    'convert_matlab_script',
    'ConversionRequest',
    'ConversionPlan',
    'ConversionResult',
    'ConversionStatus',
    'OrchestratorConversionRequest',
    'OrchestratorConversionResult',
    'OrchestratorConversionStatus'
]
