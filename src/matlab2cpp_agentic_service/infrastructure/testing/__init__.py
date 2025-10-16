"""
C++ Testing Framework for MATLAB to C++ Conversion Service

This module provides Docker-based C++ compilation and testing capabilities
to validate generated C++ code quality and correctness.
"""

from .docker_manager import DockerTestingManager
from .compilation_manager import CPPCompilationManager
from .runtime_executor import CPPRuntimeExecutor
from .quality_assessor import CPPQualityAssessor
from .types import (
    CompilationResult,
    ExecutionResult,
    TestingResult,
    QualityMetrics
)

__all__ = [
    'DockerTestingManager',
    'CPPCompilationManager', 
    'CPPRuntimeExecutor',
    'CPPQualityAssessor',
    'CompilationResult',
    'ExecutionResult',
    'TestingResult',
    'QualityMetrics'
]








