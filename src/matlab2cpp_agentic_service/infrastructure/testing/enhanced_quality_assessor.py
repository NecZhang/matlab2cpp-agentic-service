"""
Enhanced quality assessor that integrates C++ testing with existing LLM-based assessment.
"""

from typing import Dict, Any, Optional
from ...utils.logger import get_logger
from ..state.conversion_state import ConversionState
from .quality_assessor import CPPQualityAssessor
from .types import TestingResult

logger = get_logger(__name__)


class EnhancedQualityAssessor:
    """
    Enhanced quality assessor that combines LLM-based assessment with real C++ testing.
    
    This class integrates seamlessly with the existing quality assessment system
    by adding real C++ compilation and testing capabilities.
    """
    
    def __init__(self, config):
        """Initialize enhanced quality assessor."""
        self.config = config
        self.logger = logger
        
        # Initialize C++ testing components only if enabled
        self.cpp_tester = None
        if config.cpp_testing.enable_cpp_testing:
            try:
                self.cpp_tester = CPPQualityAssessor()
                self.logger.info("C++ testing framework initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize C++ testing framework: {e}")
                self.logger.warning("Continuing with LLM-only assessment")
    
    def assess_code_quality(self, state: ConversionState) -> ConversionState:
        """
        Enhanced quality assessment that combines LLM and C++ testing.
        
        This method extends the existing LLM-based assessment with real C++ testing
        when enabled, providing more accurate quality scores.
        """
        
        # Get existing quality scores from LLM assessment
        existing_scores = state.get("quality_scores", {})
        
        # Add C++ testing results if enabled and available
        if self.cpp_tester and self.config.cpp_testing.enable_cpp_testing:
            cpp_results = self._run_cpp_testing(state)
            
            if cpp_results:
                # Integrate C++ testing results with existing scores
                enhanced_scores = self._integrate_cpp_results(existing_scores, cpp_results)
                state["quality_scores"] = enhanced_scores
                
                # Add C++ testing metadata
                state["cpp_testing_results"] = cpp_results
                
                self.logger.info(f"C++ testing completed with overall score: {enhanced_scores.get('overall', 0.0):.2f}")
            else:
                self.logger.warning("C++ testing failed, using LLM-only results")
        else:
            self.logger.debug("C++ testing disabled, using LLM-only assessment")
        
        return state
    
    def _run_cpp_testing(self, state: ConversionState) -> Optional[Dict[str, Any]]:
        """Run C++ testing on generated code."""
        
        try:
            generated_code = state.get("generated_code", {})
            project_name = state.get("project_name", "test_project")
            
            if not generated_code:
                self.logger.warning("No generated code found for C++ testing")
                return None
            
            # Convert generated code to project files format
            project_files = self._convert_to_project_files(generated_code)
            
            if not project_files:
                self.logger.warning("No valid C++ files found for testing")
                return None
            
            # Run C++ testing
            self.logger.info(f"Running C++ testing for project: {project_name}")
            
            testing_result = self.cpp_tester.assess_project(
                project_files=project_files,
                project_name=project_name,
                matlab_analysis=state.get("analysis_results", {})
            )
            
            return self._convert_testing_result_to_dict(testing_result)
            
        except Exception as e:
            self.logger.error(f"Error running C++ testing: {e}")
            return None
    
    def _convert_to_project_files(self, generated_code: Dict[str, Any]) -> Dict[str, str]:
        """Convert generated code to project files format."""
        
        project_files = {}
        
        # Extract header files
        if "header" in generated_code and generated_code["header"]:
            project_files["main.h"] = generated_code["header"]
        
        # Extract implementation files
        if "implementation" in generated_code and generated_code["implementation"]:
            project_files["main.cpp"] = generated_code["implementation"]
        
        # For multi-file projects, extract individual files
        if "files" in generated_code:
            for file_info in generated_code["files"]:
                if "filename" in file_info and "content" in file_info:
                    filename = file_info["filename"]
                    content = file_info["content"]
                    project_files[filename] = content
        
        return project_files
    
    def _convert_testing_result_to_dict(self, testing_result: TestingResult) -> Dict[str, Any]:
        """Convert TestingResult to dictionary for state integration."""
        
        return {
            "compilation_success": testing_result.compilation_result.success,
            "compilation_score": testing_result.quality_metrics.compilation_score,
            "runtime_correctness": testing_result.quality_metrics.runtime_correctness,
            "performance_score": testing_result.quality_metrics.performance_score,
            "memory_efficiency": testing_result.quality_metrics.memory_efficiency,
            "code_quality": testing_result.quality_metrics.code_quality,
            "overall_score": testing_result.quality_metrics.overall_score,
            "test_duration": testing_result.test_duration,
            "status": testing_result.status.value,
            "recommendations": testing_result.recommendations,
            "compilation_errors": testing_result.compilation_result.errors,
            "compilation_warnings": testing_result.compilation_result.warnings,
            "execution_results": [
                {
                    "success": r.success,
                    "execution_time": r.execution_time,
                    "errors": r.errors
                }
                for r in testing_result.execution_results
            ]
        }
    
    def _integrate_cpp_results(self, 
                              existing_scores: Dict[str, Any], 
                              cpp_results: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate C++ testing results with existing LLM scores."""
        
        enhanced_scores = existing_scores.copy()
        
        # Override with real C++ testing results where available
        if "compilation_success" in cpp_results:
            enhanced_scores["compilation_success"] = cpp_results["compilation_success"]
            enhanced_scores["compilation_score"] = cpp_results["compilation_score"]
        
        if "runtime_correctness" in cpp_results:
            enhanced_scores["runtime_correctness"] = cpp_results["runtime_correctness"]
        
        if "performance_score" in cpp_results:
            enhanced_scores["performance_score"] = cpp_results["performance_score"]
        
        if "memory_efficiency" in cpp_results:
            enhanced_scores["memory_efficiency"] = cpp_results["memory_efficiency"]
        
        # Recalculate overall score with real data
        if "overall_score" in cpp_results:
            # Use weighted average: 70% C++ testing, 30% LLM assessment
            llm_overall = existing_scores.get("overall", 0.0)
            cpp_overall = cpp_results["overall_score"]
            enhanced_scores["overall"] = (cpp_overall * 0.7) + (llm_overall * 0.3)
        
        # Add C++-specific quality indicators
        enhanced_scores["cpp_tested"] = True
        enhanced_scores["cpp_test_duration"] = cpp_results.get("test_duration", 0.0)
        
        return enhanced_scores
    
    def assess_compilation_only(self, generated_code: Dict[str, Any], project_name: str) -> Dict[str, Any]:
        """Quick compilation-only assessment (faster than full testing)."""
        
        if not self.cpp_tester:
            return {"compilation_success": False, "error": "C++ testing not available"}
        
        try:
            project_files = self._convert_to_project_files(generated_code)
            
            if not project_files:
                return {"compilation_success": False, "error": "No valid C++ files found"}
            
            compilation_result = self.cpp_tester.assess_compilation_only(
                project_files, project_name
            )
            
            return compilation_result
            
        except Exception as e:
            self.logger.error(f"Error in compilation-only assessment: {e}")
            return {"compilation_success": False, "error": str(e)}
