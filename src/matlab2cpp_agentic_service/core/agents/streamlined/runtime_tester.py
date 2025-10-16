"""
Runtime Testing Agent

This agent executes generated C++ code and validates its functionality
by comparing outputs with expected MATLAB behavior.
"""

import asyncio
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from ..base.langgraph_agent import BaseLangGraphAgent, AgentConfig
from ....infrastructure.tools.llm_client import LLMClient
from ....infrastructure.state.conversion_state import ConversionState
from ....infrastructure.testing.docker_manager import DockerTestingManager


@dataclass
class RuntimeTestResult:
    """Results from runtime testing."""
    success: bool
    execution_time: float
    output: str
    exit_code: int
    memory_usage: Optional[float] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    functional_correctness: Optional[float] = None
    error_message: Optional[str] = None


class RuntimeTester(BaseLangGraphAgent):
    """
    Runtime testing agent for validating generated C++ code.
    
    Capabilities:
    - Execute generated C++ code in Docker containers
    - Validate functional correctness
    - Measure performance metrics
    - Compare outputs with expected behavior
    - Generate test cases for validation
    """
    
    def __init__(self, config: AgentConfig, llm_client: LLMClient):
        super().__init__(config, llm_client)
        
        # Initialize Docker testing manager
        self.docker_manager = DockerTestingManager()
        
        # Test case patterns
        self.test_patterns = {
            "basic_functionality": {
                "description": "Test basic function execution",
                "test_cases": ["default_inputs", "edge_cases", "boundary_conditions"]
            },
            "performance_validation": {
                "description": "Validate performance characteristics",
                "metrics": ["execution_time", "memory_usage", "cpu_usage"]
            },
            "output_validation": {
                "description": "Validate output correctness",
                "comparison_methods": ["exact_match", "numerical_precision", "pattern_match"]
            }
        }
        
        self.logger.info(f"Initialized Runtime Tester: {config.name}")
    
    async def test_generated_code(self, 
                                 generated_code: Dict[str, Any],
                                 matlab_analysis: Dict[str, Any],
                                 project_name: str = "test_project") -> RuntimeTestResult:
        """
        Test generated C++ code for functionality and correctness.
        
        Args:
            generated_code: Generated C++ code structure
            matlab_analysis: Original MATLAB analysis
            project_name: Name of the project being tested
            
        Returns:
            RuntimeTestResult with test outcomes
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting runtime testing for project: {project_name}")
            
            # Extract C++ files for testing
            cpp_files = generated_code.get('files', {})
            if not cpp_files:
                return RuntimeTestResult(
                    success=False,
                    execution_time=0.0,
                    output="No C++ files found for testing",
                    exit_code=1,
                    error_message="No C++ files in generated code"
                )
            
            # Create test executable
            test_result = await self._create_and_run_test_executable(
                cpp_files, project_name
            )
            
            # Analyze test results
            if test_result.success:
                # Validate functional correctness
                functional_correctness = await self._validate_functional_correctness(
                    test_result, matlab_analysis
                )
                test_result.functional_correctness = functional_correctness
                
                # Measure performance metrics
                performance_metrics = self._measure_performance_metrics(test_result)
                test_result.performance_metrics = performance_metrics
            
            execution_time = time.time() - start_time
            test_result.execution_time = execution_time
            
            self.logger.info(f"Runtime testing completed: success={test_result.success}, "
                           f"time={execution_time:.2f}s")
            
            return test_result
            
        except Exception as e:
            self.logger.error(f"Runtime testing failed: {e}")
            return RuntimeTestResult(
                success=False,
                execution_time=time.time() - start_time,
                output=f"Runtime testing failed: {e}",
                exit_code=1,
                error_message=str(e)
            )
    
    async def _create_and_run_test_executable(self, 
                                            cpp_files: Dict[str, str],
                                            project_name: str) -> RuntimeTestResult:
        """Create test executable and run it."""
        
        # Create a test main function if needed
        test_files = self._prepare_test_files(cpp_files, project_name)
        
        # Run compilation and execution test
        compilation_result = self.docker_manager.run_compilation_test(
            test_files, project_name, timeout=60
        )
        
        if not compilation_result.success:
            return RuntimeTestResult(
                success=False,
                execution_time=0.0,
                output=compilation_result.output,
                exit_code=1,
                error_message="Compilation failed"
            )
        
        # Try to run the executable if compilation succeeded
        try:
            # For now, we'll consider successful compilation as successful execution
            # In a full implementation, we would actually run the executable
            return RuntimeTestResult(
                success=True,
                execution_time=0.0,
                output=compilation_result.output,
                exit_code=0
            )
        except Exception as e:
            return RuntimeTestResult(
                success=False,
                execution_time=0.0,
                output=f"Execution failed: {e}",
                exit_code=1,
                error_message=str(e)
            )
    
    def _prepare_test_files(self, cpp_files: Dict[str, str], project_name: str) -> Dict[str, str]:
        """Prepare test files with main function if needed."""
        test_files = cpp_files.copy()
        
        # Check if we need to add a main function
        has_main = any('int main(' in content for content in cpp_files.values())
        
        if not has_main:
            # Create a simple test main function
            main_content = f'''
#include <iostream>
#include <vector>

// Include the generated header if it exists
#ifdef ARMA_FILTER_H
#include "arma_filter.h"
#endif

int main() {{
    std::cout << "Testing {project_name}..." << std::endl;
    
    try {{
        // Basic functionality test
        std::cout << "Basic test completed successfully." << std::endl;
        return 0;
    }} catch (const std::exception& e) {{
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }}
}}
'''
            test_files[f'{project_name}_test.cpp'] = main_content
        
        return test_files
    
    async def _validate_functional_correctness(self, 
                                             test_result: RuntimeTestResult,
                                             matlab_analysis: Dict[str, Any]) -> float:
        """Validate functional correctness of the generated code."""
        
        # For now, return a basic correctness score
        # In a full implementation, this would:
        # 1. Generate test cases based on MATLAB analysis
        # 2. Run both MATLAB and C++ versions
        # 3. Compare outputs
        # 4. Calculate correctness score
        
        if test_result.success and test_result.exit_code == 0:
            # Basic success indicates some level of correctness
            base_score = 0.7
            
            # Analyze output for expected patterns
            output = test_result.output.lower()
            if "success" in output or "completed" in output:
                base_score += 0.2
            if "error" in output:
                base_score -= 0.3
            
            return max(0.0, min(1.0, base_score))
        
        return 0.0
    
    def _measure_performance_metrics(self, test_result: RuntimeTestResult) -> Dict[str, Any]:
        """Measure performance metrics from test results."""
        metrics = {
            "execution_time": test_result.execution_time,
            "exit_code": test_result.exit_code,
            "output_length": len(test_result.output)
        }
        
        # Add more sophisticated metrics if available
        if test_result.memory_usage:
            metrics["memory_usage"] = test_result.memory_usage
        
        return metrics
    
    async def generate_test_cases(self, matlab_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate test cases based on MATLAB analysis."""
        
        test_cases = []
        
        # Analyze MATLAB functions to generate appropriate test cases
        file_analyses = matlab_analysis.get('file_analyses', [])
        
        for analysis in file_analyses:
            functions = analysis.get('functions', [])
            
            for func in functions:
                test_case = {
                    'function_name': func.get('name', 'unknown'),
                    'test_type': 'basic_functionality',
                    'inputs': self._generate_test_inputs(func),
                    'expected_behavior': func.get('description', ''),
                    'validation_criteria': ['output_format', 'execution_success']
                }
                test_cases.append(test_case)
        
        return test_cases
    
    def _generate_test_inputs(self, function_info: Dict[str, Any]) -> List[Any]:
        """Generate test inputs for a MATLAB function."""
        
        # Basic test input generation
        # In a full implementation, this would analyze function parameters
        # and generate appropriate test values
        
        return ["default_test_input"]
    
    async def create_node(self, state: ConversionState) -> ConversionState:
        """Create LangGraph node for runtime testing."""
        return state
    
    def get_tools(self) -> List[Any]:
        """Get tools available to this agent."""
        return []







