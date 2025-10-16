"""
Runtime executor for C++ testing framework.
"""

import json
import time
from typing import Dict, List, Any, Optional
from ...utils.logger import get_logger
from .types import ExecutionResult, TestCase
from .docker_manager import DockerTestingManager

logger = get_logger(__name__)


class CPPRuntimeExecutor:
    """Executes C++ binaries and manages runtime testing."""
    
    def __init__(self, docker_manager: Optional[DockerTestingManager] = None):
        """Initialize runtime executor."""
        self.docker_manager = docker_manager or DockerTestingManager()
        self.logger = logger
    
    def execute_binary(self,
                      binary_path: str,
                      test_inputs: Dict[str, Any],
                      timeout: int = 60) -> ExecutionResult:
        """Execute C++ binary with test inputs."""
        
        self.logger.info(f"Executing binary: {binary_path}")
        
        # Run execution in Docker
        execution_result = self.docker_manager.run_execution_test(
            binary_path, test_inputs, timeout
        )
        
        self.logger.info(f"Execution {'successful' if execution_result.success else 'failed'}")
        return execution_result
    
    def run_test_cases(self,
                      binary_path: str,
                      test_cases: List[TestCase]) -> List[ExecutionResult]:
        """Run multiple test cases against a binary."""
        
        results = []
        
        for test_case in test_cases:
            self.logger.info(f"Running test case: {test_case.name}")
            
            result = self.execute_binary(
                binary_path,
                test_case.inputs,
                test_case.timeout
            )
            
            # Add test case information to result
            result.test_case_name = test_case.name
            result.test_case_description = test_case.description
            
            results.append(result)
        
        return results
    
    def generate_test_cases(self, 
                           project_name: str,
                           matlab_analysis: Dict[str, Any]) -> List[TestCase]:
        """Generate test cases based on MATLAB analysis."""
        
        test_cases = []
        
        # Generate basic test cases based on project type
        if 'skeleton_vessel' in project_name.lower():
            test_cases.extend(self._generate_skeleton_vessel_test_cases())
        elif 'arma_filter' in project_name.lower():
            test_cases.extend(self._generate_arma_filter_test_cases())
        else:
            # Generate generic test cases
            test_cases.extend(self._generate_generic_test_cases())
        
        return test_cases
    
    def _generate_skeleton_vessel_test_cases(self) -> List[TestCase]:
        """Generate test cases for skeleton vessel project."""
        
        return [
            TestCase(
                name="basic_image_test",
                inputs={
                    "image_width": 100,
                    "image_height": 100,
                    "source_point": [50, 50],
                    "start_point": [10, 10]
                },
                description="Basic skeleton vessel test with small image",
                timeout=30
            ),
            TestCase(
                name="edge_case_test",
                inputs={
                    "image_width": 10,
                    "image_height": 10,
                    "source_point": [5, 5],
                    "start_point": [0, 0]
                },
                description="Edge case test with minimal image size",
                timeout=30
            )
        ]
    
    def _generate_arma_filter_test_cases(self) -> List[TestCase]:
        """Generate test cases for ARMA filter project."""
        
        return [
            TestCase(
                name="basic_filter_test",
                inputs={
                    "p": 2,
                    "iterations": 5,
                    "data_size": 100
                },
                description="Basic ARMA filter test",
                timeout=30
            ),
            TestCase(
                name="high_iteration_test",
                inputs={
                    "p": 3,
                    "iterations": 20,
                    "data_size": 50
                },
                description="High iteration count test",
                timeout=60
            )
        ]
    
    def _generate_generic_test_cases(self) -> List[TestCase]:
        """Generate generic test cases."""
        
        return [
            TestCase(
                name="default_test",
                inputs={},
                description="Default test case",
                timeout=30
            )
        ]
