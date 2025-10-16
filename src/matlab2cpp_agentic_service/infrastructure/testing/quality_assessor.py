"""
Quality assessor for C++ testing framework.
"""

import time
from typing import Dict, List, Any, Optional
from ...utils.logger import get_logger
from .types import (
    CompilationResult, ExecutionResult, QualityMetrics, 
    TestingResult, TestStatus, TestCase
)
from .compilation_manager import CPPCompilationManager
from .runtime_executor import CPPRuntimeExecutor

logger = get_logger(__name__)


class CPPQualityAssessor:
    """Assesses quality of generated C++ code."""
    
    def __init__(self):
        """Initialize quality assessor."""
        self.compilation_manager = CPPCompilationManager()
        self.runtime_executor = CPPRuntimeExecutor(self.compilation_manager.docker_manager)
        self.logger = logger
    
    def assess_project(self,
                      project_files: Dict[str, str],
                      project_name: str,
                      matlab_analysis: Optional[Dict[str, Any]] = None) -> TestingResult:
        """Assess complete project quality."""
        
        start_time = time.time()
        
        self.logger.info(f"Starting quality assessment for project: {project_name}")
        
        # Step 1: Compilation testing
        compilation_result = self.compilation_manager.compile_project(
            project_files, project_name
        )
        
        execution_results = []
        if compilation_result.success:
            # Step 2: Runtime testing
            test_cases = self.runtime_executor.generate_test_cases(project_name, matlab_analysis or {})
            execution_results = self.runtime_executor.run_test_cases(
                compilation_result.binary_path, test_cases
            )
        
        # Step 3: Quality metrics calculation (now includes code_simplicity!)
        quality_metrics = self._calculate_quality_metrics(
            compilation_result, execution_results, project_files
        )
        
        # Step 4: Generate recommendations
        recommendations = self._generate_recommendations(
            compilation_result, execution_results, quality_metrics
        )
        
        test_duration = time.time() - start_time
        
        # Determine overall status
        status = TestStatus.SUCCESS if compilation_result.success and all(r.success for r in execution_results) else TestStatus.FAILURE
        
        return TestingResult(
            project_name=project_name,
            compilation_result=compilation_result,
            execution_results=execution_results,
            quality_metrics=quality_metrics,
            test_duration=test_duration,
            status=status,
            recommendations=recommendations
        )
    
    def _calculate_code_simplicity(self, project_files: Dict[str, str]) -> float:
        """
        Calculate code simplicity score based on file count and structure.
        
        Rewards:
        - Fewer files (less bloat)
        - No unnecessary helper libraries
        - Clean project structure
        
        Args:
            project_files: Dictionary of filename -> content
        
        Returns:
            Float 0.0-1.0, where 1.0 is perfectly minimal
        """
        # Count C++ source/header files
        cpp_files = [f for f in project_files.keys() if f.endswith(('.cpp', '.h'))]
        total_files = len(cpp_files)
        
        if total_files == 0:
            return 0.0
        
        # Scoring based on file count
        # Ideal: 2-4 files for simple projects, up to 20 for complex
        if total_files <= 4:
            file_score = 1.0  # Perfect! Minimal project
        elif total_files <= 6:
            file_score = 0.95  # Very good
        elif total_files <= 10:
            file_score = 0.85  # Good
        elif total_files <= 15:
            file_score = 0.70  # Acceptable
        elif total_files <= 20:
            file_score = 0.60  # Getting bloated
        else:
            # Penalty for excessive files
            file_score = max(0.40, 0.60 - (total_files - 20) * 0.02)
        
        self.logger.debug(f"Code simplicity: {total_files} files → score {file_score:.2f}")
        return file_score
    
    def _calculate_quality_metrics(self,
                                  compilation_result: CompilationResult,
                                  execution_results: List[ExecutionResult],
                                  project_files: Dict[str, str] = None) -> QualityMetrics:
        """Calculate quality metrics from test results."""
        
        # Compilation score
        compilation_score = 1.0 if compilation_result.success else 0.0
        
        # Reduce score ONLY for OUR code warnings (not library/system warnings)
        if compilation_result.warnings:
            # Filter to only count warnings from our generated code
            our_code_warnings = [
                w for w in compilation_result.warnings
                if not any(lib_path in str(w) for lib_path in [
                    '/usr/include/eigen3',  # Eigen library
                    '/usr/include/opencv',  # OpenCV library
                    '/usr/lib/',            # System libraries
                    'Eigen/src/',          # Eigen internals
                    'note:'                # Compiler notes (not warnings)
                ])
            ]
            if our_code_warnings:
                compilation_score -= min(0.2, len(our_code_warnings) * 0.05)
                self.logger.debug(f"Reduced compilation score for {len(our_code_warnings)} code warnings (ignoring {len(compilation_result.warnings) - len(our_code_warnings)} library warnings)")
        
        # Runtime correctness score
        if execution_results:
            success_count = sum(1 for r in execution_results if r.success)
            runtime_correctness = success_count / len(execution_results)
        else:
            runtime_correctness = 0.0 if compilation_result.success else 0.0
        
        # Performance score (based on execution time)
        if execution_results:
            avg_execution_time = sum(r.execution_time for r in execution_results) / len(execution_results)
            # Assume good performance if execution time < 1 second
            performance_score = min(1.0, 1.0 / max(avg_execution_time, 0.1))
        else:
            performance_score = 0.0
        
        # Memory efficiency score (placeholder - would need actual memory measurement)
        memory_efficiency = 1.0 if compilation_result.success else 0.0
        
        # Code quality score (based on compilation warnings and errors from OUR code)
        code_quality = 1.0 if compilation_result.success else 0.0
        if compilation_result.warnings:
            # Only penalize for OUR code quality issues, not library warnings
            our_code_warnings = [
                w for w in compilation_result.warnings
                if not any(lib_path in str(w) for lib_path in [
                    '/usr/include/eigen3', '/usr/include/opencv', '/usr/lib/', 'Eigen/src/', 'note:'
                ])
            ]
            # Reduce for common code quality issues in OUR code
            for warning in our_code_warnings:
                if any(keyword in warning.lower() for keyword in ['unused', 'deprecated', 'shadow']):
                    code_quality -= 0.1
        
        # Code simplicity score (NEW! Rewards minimal file count)
        if project_files:
            code_simplicity = self._calculate_code_simplicity(project_files)
        else:
            # Default: assume reasonable project structure when files not available
            code_simplicity = 0.8
            self.logger.debug("No project_files provided, using default code_simplicity=0.8")
        
        # Overall score (weighted average) - UPDATED to include code_simplicity!
        overall_score = (
            compilation_score * 0.25 +      # Reduced from 0.3
            runtime_correctness * 0.25 +    # Reduced from 0.3
            performance_score * 0.15 +      # Reduced from 0.2
            memory_efficiency * 0.10 +      # Same
            code_quality * 0.10 +           # Same
            code_simplicity * 0.15          # NEW! 15% weight
        )
        
        return QualityMetrics(
            compilation_success=compilation_result.success,
            compilation_score=max(0.0, min(1.0, compilation_score)),
            runtime_correctness=max(0.0, min(1.0, runtime_correctness)),
            performance_score=max(0.0, min(1.0, performance_score)),
            memory_efficiency=max(0.0, min(1.0, memory_efficiency)),
            code_quality=max(0.0, min(1.0, code_quality)),
            code_simplicity=max(0.0, min(1.0, code_simplicity)),
            overall_score=max(0.0, min(1.0, overall_score))
        )
    
    def _generate_recommendations(self,
                                 compilation_result: CompilationResult,
                                 execution_results: List[ExecutionResult],
                                 quality_metrics: QualityMetrics) -> List[str]:
        """Generate improvement recommendations."""
        
        recommendations = []
        
        # Compilation recommendations
        if not compilation_result.success:
            recommendations.append("Fix compilation errors before proceeding with testing")
        
        if compilation_result.warnings:
            recommendations.append(f"Address {len(compilation_result.warnings)} compilation warnings")
            
            # Specific warning recommendations
            for warning in compilation_result.warnings:
                if 'unused' in warning.lower():
                    recommendations.append("Remove unused variables or functions")
                elif 'deprecated' in warning.lower():
                    recommendations.append("Update deprecated function calls")
                elif 'shadow' in warning.lower():
                    recommendations.append("Fix variable name shadowing")
        
        # Runtime recommendations
        failed_tests = [r for r in execution_results if not r.success]
        if failed_tests:
            recommendations.append(f"Fix {len(failed_tests)} failing test cases")
        
        # Performance recommendations
        if quality_metrics.performance_score < 0.5:
            recommendations.append("Optimize code performance - execution time is high")
        
        # Overall recommendations
        if quality_metrics.overall_score < 0.7:
            recommendations.append("Overall code quality needs improvement")
        elif quality_metrics.overall_score > 0.9:
            recommendations.append("Excellent code quality! Consider adding more test cases")
        
        return recommendations
    
    def assess_compilation_only(self,
                               project_files: Dict[str, str],
                               project_name: str) -> Dict[str, Any]:
        """Assess only compilation quality (faster than full assessment)."""
        
        compilation_result = self.compilation_manager.compile_project(
            project_files, project_name
        )
        
        return {
            'compilation_success': compilation_result.success,
            'compilation_score': 1.0 if compilation_result.success else 0.0,
            'errors': compilation_result.errors,
            'warnings': compilation_result.warnings,
            'compilation_time': compilation_result.compilation_time,
            'binary_path': compilation_result.binary_path
        }
