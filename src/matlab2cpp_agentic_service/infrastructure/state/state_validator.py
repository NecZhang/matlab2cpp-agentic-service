"""
State Validator for LangGraph Conversion Workflows

This module provides validation and integrity checking for conversion states
in the LangGraph-based MATLAB2C++ conversion service.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import time
from loguru import logger

from .conversion_state import ConversionState, ConversionRequest, ConversionResult, ConversionStatus


@dataclass
class ValidationRule:
    """Validation rule definition."""
    name: str
    description: str
    check_function: callable
    severity: str = "error"  # "error", "warning", "info"
    auto_fix: bool = False
    fix_function: Optional[callable] = None


@dataclass
class ValidationResult:
    """Result of state validation."""
    valid: bool
    errors: List[str]
    warnings: List[str]
    info: List[str]
    fixes_applied: List[str]
    recommendations: List[str]
    validation_time: float


class StateValidator:
    """
    Validator for conversion states.
    
    This class provides comprehensive validation of conversion states
    including data integrity, workflow consistency, and performance checks.
    """
    
    def __init__(self):
        """Initialize the state validator."""
        self.rules: List[ValidationRule] = []
        self.logger = logger.bind(name="state_validator")
        
        # Register default validation rules
        self._register_default_rules()
    
    def _register_default_rules(self):
        """Register default validation rules."""
        
        # Required fields validation
        self.add_rule(ValidationRule(
            name="required_fields",
            description="Check for required fields in state",
            check_function=self._check_required_fields,
            severity="error"
        ))
        
        # Request validation
        self.add_rule(ValidationRule(
            name="request_validity",
            description="Validate conversion request",
            check_function=self._check_request_validity,
            severity="error"
        ))
        
        # Result validation
        self.add_rule(ValidationRule(
            name="result_validity",
            description="Validate conversion result",
            check_function=self._check_result_validity,
            severity="error"
        ))
        
        # Workflow consistency
        self.add_rule(ValidationRule(
            name="workflow_consistency",
            description="Check workflow step consistency",
            check_function=self._check_workflow_consistency,
            severity="warning"
        ))
        
        # Agent memory validation
        self.add_rule(ValidationRule(
            name="agent_memory_validity",
            description="Validate agent memory structure",
            check_function=self._check_agent_memory_validity,
            severity="warning"
        ))
        
        # Performance validation
        self.add_rule(ValidationRule(
            name="performance_validation",
            description="Check performance metrics",
            check_function=self._check_performance_metrics,
            severity="info"
        ))
        
        # File path validation
        self.add_rule(ValidationRule(
            name="file_paths_validity",
            description="Validate file paths in state",
            check_function=self._check_file_paths_validity,
            severity="warning",
            auto_fix=True,
            fix_function=self._fix_file_paths
        ))
        
        # Memory usage validation
        self.add_rule(ValidationRule(
            name="memory_usage_validation",
            description="Check memory usage",
            check_function=self._check_memory_usage,
            severity="warning"
        ))
        
        # Error context validation
        self.add_rule(ValidationRule(
            name="error_context_validation",
            description="Validate error context",
            check_function=self._check_error_context,
            severity="info"
        ))
    
    def add_rule(self, rule: ValidationRule):
        """Add a custom validation rule."""
        self.rules.append(rule)
        self.logger.debug(f"Added validation rule: {rule.name}")
    
    def validate_state(self, state: ConversionState, auto_fix: bool = False) -> ValidationResult:
        """
        Validate a conversion state.
        
        Args:
            state: Conversion state to validate
            auto_fix: Whether to apply automatic fixes
            
        Returns:
            Validation result
        """
        start_time = time.time()
        
        result = ValidationResult(
            valid=True,
            errors=[],
            warnings=[],
            info=[],
            fixes_applied=[],
            recommendations=[],
            validation_time=0.0
        )
        
        try:
            for rule in self.rules:
                try:
                    rule_result = rule.check_function(state)
                    
                    if isinstance(rule_result, bool):
                        # Simple boolean result
                        if not rule_result:
                            if rule.severity == "error":
                                result.errors.append(f"{rule.name}: {rule.description}")
                                result.valid = False
                            elif rule.severity == "warning":
                                result.warnings.append(f"{rule.name}: {rule.description}")
                            else:
                                result.info.append(f"{rule.name}: {rule.description}")
                    
                    elif isinstance(rule_result, dict):
                        # Detailed result
                        if "valid" in rule_result and not rule_result["valid"]:
                            message = rule_result.get("message", rule.description)
                            
                            if rule.severity == "error":
                                result.errors.append(f"{rule.name}: {message}")
                                result.valid = False
                            elif rule.severity == "warning":
                                result.warnings.append(f"{rule.name}: {message}")
                            else:
                                result.info.append(f"{rule.name}: {message}")
                        
                        # Add recommendations
                        if "recommendations" in rule_result:
                            result.recommendations.extend(rule_result["recommendations"])
                        
                        # Apply auto-fix if available
                        if auto_fix and rule.auto_fix and rule.fix_function:
                            try:
                                fix_result = rule.fix_function(state)
                                if fix_result:
                                    result.fixes_applied.append(f"{rule.name}: Applied automatic fix")
                            except Exception as e:
                                result.warnings.append(f"{rule.name}: Auto-fix failed: {e}")
                    
                    elif isinstance(rule_result, list):
                        # List of issues
                        for issue in rule_result:
                            if rule.severity == "error":
                                result.errors.append(f"{rule.name}: {issue}")
                                result.valid = False
                            elif rule.severity == "warning":
                                result.warnings.append(f"{rule.name}: {issue}")
                            else:
                                result.info.append(f"{rule.name}: {issue}")
                
                except Exception as e:
                    error_msg = f"Validation rule '{rule.name}' failed: {e}"
                    result.warnings.append(error_msg)
                    self.logger.error(error_msg, exc_info=True)
            
            # Generate overall recommendations
            result.recommendations.extend(self._generate_recommendations(state, result))
            
        except Exception as e:
            result.valid = False
            result.errors.append(f"Validation process failed: {e}")
            self.logger.error(f"State validation failed: {e}", exc_info=True)
        
        result.validation_time = time.time() - start_time
        
        self.logger.info(f"State validation completed: {len(result.errors)} errors, "
                        f"{len(result.warnings)} warnings, {len(result.info)} info messages")
        
        return result
    
    def _check_required_fields(self, state: ConversionState) -> Dict[str, Any]:
        """Check for required fields in state."""
        required_fields = [
            "request", "result", "workflow_step", "start_time",
            "agent_memory", "processing_times", "total_processing_time"
        ]
        
        missing_fields = []
        for field in required_fields:
            if field not in state:
                missing_fields.append(field)
        
        if missing_fields:
            return {
                "valid": False,
                "message": f"Missing required fields: {', '.join(missing_fields)}"
            }
        
        return {"valid": True}
    
    def _check_request_validity(self, state: ConversionState) -> Dict[str, Any]:
        """Validate conversion request."""
        request = state["request"]
        if not request:
            return {"valid": False, "message": "Request is missing"}
        
        issues = []
        
        # Check request attributes
        if not hasattr(request, 'project_name') or not request.project_name:
            issues.append("Request missing project_name")
        
        if not hasattr(request, 'matlab_path') or not request.matlab_path:
            issues.append("Request missing matlab_path")
        
        if hasattr(request, 'max_optimization_turns'):
            if request.max_optimization_turns < 0:
                issues.append("max_optimization_turns cannot be negative")
        
        if hasattr(request, 'target_quality_score'):
            if not (0.0 <= request.target_quality_score <= 10.0):
                issues.append("target_quality_score must be between 0.0 and 10.0")
        
        if hasattr(request, 'conversion_mode'):
            if request.conversion_mode not in ["faithful", "result-focused"]:
                issues.append("Invalid conversion_mode")
        
        if issues:
            return {"valid": False, "message": "; ".join(issues)}
        
        return {"valid": True}
    
    def _check_result_validity(self, state: ConversionState) -> Dict[str, Any]:
        """Validate conversion result."""
        result = state["result"]
        if not result:
            return {"valid": False, "message": "Result is missing"}
        
        issues = []
        
        # Check result attributes
        if not hasattr(result, 'status'):
            issues.append("Result missing status")
        
        if not hasattr(result, 'generated_files'):
            issues.append("Result missing generated_files")
        
        if not hasattr(result, 'assessment_reports'):
            issues.append("Result missing assessment_reports")
        
        # Check status consistency
        if hasattr(result, 'status'):
            status = result.status
            if hasattr(status, 'value'):
                status_value = status.value
            else:
                status_value = str(status)
            
            if status_value == "completed":
                if not result.generated_files:
                    issues.append("Completed conversion should have generated files")
            
            elif status_value == "failed":
                if not hasattr(result, 'error_message') or not result.error_message:
                    issues.append("Failed conversion should have error message")
        
        if issues:
            return {"valid": False, "message": "; ".join(issues)}
        
        return {"valid": True}
    
    def _check_workflow_consistency(self, state: ConversionState) -> Dict[str, Any]:
        """Check workflow step consistency."""
        workflow_step = state["workflow_step"]
        result_status = state["result"].status
        
        if not workflow_step:
            return {"valid": False, "message": "Missing workflow_step"}
        
        # Map status to expected workflow step
        status_to_step = {
            "pending": "initialization",
            "analyzing": "analyzing",
            "planning": "planning",
            "generating": "generating",
            "assessing": "assessing",
            "optimizing": "optimizing",
            "saving": "saving",
            "completed": "completed",
            "failed": "failed"
        }
        
        if hasattr(result_status, 'value'):
            status_value = result_status.value
        else:
            status_value = str(result_status) if result_status else "pending"
        
        expected_step = status_to_step.get(status_value, "unknown")
        
        if workflow_step != expected_step:
            return {
                "valid": False,
                "message": f"Workflow step '{workflow_step}' inconsistent with status '{status_value}' (expected: '{expected_step}')",
                "recommendations": [f"Consider updating workflow_step to '{expected_step}'"]
            }
        
        return {"valid": True}
    
    def _check_agent_memory_validity(self, state: ConversionState) -> Dict[str, Any]:
        """Validate agent memory structure."""
        agent_memory = state["agent_memory"]
        
        issues = []
        recommendations = []
        
        for agent_name, memory in agent_memory.items():
            if not isinstance(memory, dict):
                issues.append(f"Agent '{agent_name}' memory is not a dictionary")
                continue
            
            # Check required memory fields
            required_fields = ["short_term", "long_term", "context", "performance_history"]
            for field in required_fields:
                if field not in memory:
                    issues.append(f"Agent '{agent_name}' missing memory field: {field}")
            
            # Check performance history
            if "performance_history" in memory:
                perf_history = memory["performance_history"]
                if not isinstance(perf_history, list):
                    issues.append(f"Agent '{agent_name}' performance_history is not a list")
                elif len(perf_history) > 100:
                    recommendations.append(f"Agent '{agent_name}' has large performance history ({len(perf_history)} entries)")
        
        if issues:
            return {"valid": False, "message": "; ".join(issues)}
        
        if recommendations:
            return {"valid": True, "recommendations": recommendations}
        
        return {"valid": True}
    
    def _check_performance_metrics(self, state: ConversionState) -> Dict[str, Any]:
        """Check performance metrics."""
        processing_times = state["processing_times"]
        total_time = state["total_processing_time"]
        
        recommendations = []
        
        # Check individual operation times
        for operation, time_taken in processing_times.items():
            if time_taken > 300:  # More than 5 minutes
                recommendations.append(f"Operation '{operation}' took {time_taken:.1f}s - consider optimization")
        
        # Check total processing time
        if total_time > 1800:  # More than 30 minutes
            recommendations.append(f"Total processing time is {total_time:.1f}s - consider parallel processing")
        
        # Check for very short times (potential issues)
        for operation, time_taken in processing_times.items():
            if time_taken < 0.1:  # Less than 100ms
                recommendations.append(f"Operation '{operation}' completed very quickly ({time_taken:.3f}s) - verify correctness")
        
        if recommendations:
            return {"valid": True, "recommendations": recommendations}
        
        return {"valid": True}
    
    def _check_file_paths_validity(self, state: ConversionState) -> Dict[str, Any]:
        """Validate file paths in state."""
        issues = []
        
        # Check generated files
        result = state["result"]
        generated_files = result.generated_files
        
        for file_path in generated_files:
            path_obj = Path(file_path)
            if not path_obj.exists():
                issues.append(f"Generated file does not exist: {file_path}")
        
        # Check MATLAB path in request
        request = state["request"]
        if request and hasattr(request, 'matlab_path'):
            matlab_path = Path(request.matlab_path)
            if not matlab_path.exists():
                issues.append(f"MATLAB path does not exist: {request.matlab_path}")
        
        if issues:
            return {"valid": False, "message": "; ".join(issues)}
        
        return {"valid": True}
    
    def _fix_file_paths(self, state: ConversionState) -> bool:
        """Fix file path issues."""
        fixed = False
        
        # This is a placeholder - actual fixes would depend on specific issues
        # For now, just return True to indicate no fixes needed
        return fixed
    
    def _check_memory_usage(self, state: ConversionState) -> Dict[str, Any]:
        """Check memory usage."""
        recommendations = []
        
        # Check streaming updates
        streaming_updates = state["streaming_updates"]
        if len(streaming_updates) > 50:
            recommendations.append(f"Large number of streaming updates ({len(streaming_updates)}) - consider cleanup")
        
        # Check operation results
        operation_results = state["operation_results"]
        if len(operation_results) > 100:
            recommendations.append(f"Large number of operation results ({len(operation_results)}) - consider cleanup")
        
        # Check system metrics
        system_metrics = state["system_metrics"]
        if len(system_metrics) > 50:
            recommendations.append(f"Large number of system metrics ({len(system_metrics)}) - consider cleanup")
        
        if recommendations:
            return {"valid": True, "recommendations": recommendations}
        
        return {"valid": True}
    
    def _check_error_context(self, state: ConversionState) -> Dict[str, Any]:
        """Validate error context."""
        error_context = state["error_context"]
        
        if not error_context:
            return {"valid": True}
        
        issues = []
        
        # Check if error context has reasonable structure
        if "last_error" in error_context:
            if not isinstance(error_context["last_error"], str):
                issues.append("last_error should be a string")
        
        if "error_timestamp" in error_context:
            if not isinstance(error_context["error_timestamp"], (int, float)):
                issues.append("error_timestamp should be a number")
        
        if issues:
            return {"valid": False, "message": "; ".join(issues)}
        
        return {"valid": True}
    
    def _generate_recommendations(self, state: ConversionState, validation_result: ValidationResult) -> List[str]:
        """Generate overall recommendations based on validation results."""
        recommendations = []
        
        # Performance recommendations
        total_time = state["total_processing_time"]
        if total_time > 600:  # More than 10 minutes
            recommendations.append("Consider implementing caching for repeated operations")
        
        # Memory recommendations
        if len(validation_result.errors) == 0 and len(validation_result.warnings) > 5:
            recommendations.append("Multiple warnings detected - consider reviewing state management")
        
        # Workflow recommendations
        retry_count = state["retry_count"]
        if retry_count > 1:
            recommendations.append("Multiple retries detected - consider improving error handling")
        
        # Agent recommendations
        agent_count = len(state["agent_memory"])
        if agent_count == 0:
            recommendations.append("No agent memory found - verify agent execution")
        
        return recommendations
    
    def export_validation_report(self, validation_result: ValidationResult, 
                               file_path: Union[str, Path]) -> bool:
        """Export validation report to file."""
        try:
            report = {
                "timestamp": time.time(),
                "validation_result": {
                    "valid": validation_result.valid,
                    "errors": validation_result.errors,
                    "warnings": validation_result.warnings,
                    "info": validation_result.info,
                    "fixes_applied": validation_result.fixes_applied,
                    "recommendations": validation_result.recommendations,
                    "validation_time": validation_result.validation_time
                },
                "summary": {
                    "total_issues": len(validation_result.errors) + len(validation_result.warnings) + len(validation_result.info),
                    "error_count": len(validation_result.errors),
                    "warning_count": len(validation_result.warnings),
                    "info_count": len(validation_result.info),
                    "fixes_count": len(validation_result.fixes_applied)
                }
            }
            
            with open(file_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Exported validation report to: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export validation report: {e}")
            return False
