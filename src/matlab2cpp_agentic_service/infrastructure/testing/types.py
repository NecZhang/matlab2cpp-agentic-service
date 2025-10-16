"""
Type definitions for C++ testing framework.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum


class TestStatus(Enum):
    """Status of a test operation."""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class CompilationResult:
    """Result of C++ compilation."""
    success: bool
    output: str
    errors: List[str]
    warnings: List[str]
    binary_path: Optional[str] = None
    compilation_time: float = 0.0
    memory_usage: float = 0.0


@dataclass
class ExecutionResult:
    """Result of C++ execution."""
    success: bool
    output: str
    errors: List[str]
    execution_time: float = 0.0
    memory_usage: float = 0.0
    return_code: int = 0
    stdout: str = ""
    stderr: str = ""


@dataclass
class QualityMetrics:
    """Quality metrics for generated C++ code."""
    compilation_success: bool
    compilation_score: float  # 0.0 to 1.0
    runtime_correctness: float  # 0.0 to 1.0
    performance_score: float  # 0.0 to 1.0
    memory_efficiency: float  # 0.0 to 1.0
    code_quality: float  # 0.0 to 1.0
    code_simplicity: float  # 0.0 to 1.0 - NEW! Rewards minimal files/no bloat
    overall_score: float  # 0.0 to 1.0


@dataclass
class TestingResult:
    """Complete result of C++ testing."""
    project_name: str
    compilation_result: CompilationResult
    execution_results: List[ExecutionResult]
    quality_metrics: QualityMetrics
    test_duration: float
    status: TestStatus
    recommendations: List[str]


@dataclass
class TestCase:
    """Individual test case for C++ execution."""
    name: str
    inputs: Dict[str, Any]
    expected_outputs: Optional[Dict[str, Any]] = None
    timeout: int = 60  # seconds
    description: str = ""


@dataclass
class ProjectFiles:
    """Structure for project files."""
    source_files: Dict[str, str]  # filename -> content
    header_files: Dict[str, str]  # filename -> content
    additional_files: Dict[str, str]  # filename -> content (CMakeLists.txt, etc.)
    project_name: str
    dependencies: List[str]








