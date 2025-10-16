"""
Advanced Compilation Log Analyzer

This agent provides intelligent analysis of compilation logs to provide
detailed feedback for code generation improvement.
"""

import re
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from ..base.langgraph_agent import BaseLangGraphAgent, AgentConfig
from ....infrastructure.tools.llm_client import LLMClient
from ....infrastructure.state.conversion_state import ConversionState


@dataclass
class CompilationError:
    """Structured compilation error information."""
    line_number: Optional[int]
    column_number: Optional[int]
    error_type: str
    error_message: str
    file_name: Optional[str]
    severity: str  # 'error', 'warning', 'info'
    suggestion: Optional[str] = None


@dataclass
class CompilationAnalysis:
    """Comprehensive compilation analysis results."""
    success: bool
    total_errors: int
    total_warnings: int
    errors: List[CompilationError]
    warnings: List[CompilationError]
    error_categories: Dict[str, int]
    improvement_suggestions: List[str]
    code_quality_issues: List[str]
    llm_prompt_enhancements: List[str]


class CompilationLogAnalyzer(BaseLangGraphAgent):
    """
    Advanced compilation log analyzer with LLM-powered insights.
    
    Capabilities:
    - Parse and structure compilation errors
    - Categorize error types and severity
    - Generate improvement suggestions
    - Provide LLM prompt enhancements
    - Analyze code quality issues
    """
    
    def __init__(self, config: AgentConfig, llm_client: LLMClient):
        super().__init__(config, llm_client)
        
        # Error pattern definitions
        self.error_patterns = {
            "include_errors": {
                "regex": r"(.*):(\d+):(\d+):\s*(error|warning):\s*(#include expects.*|fatal error: .*\.h: No such file.*)",
                "severity": "error",
                "category": "include",
                "suggestions": [
                    "Fix include statement syntax",
                    "Add missing header files",
                    "Check include path configuration"
                ]
            },
            "syntax_errors": {
                "regex": r"(.*):(\d+):(\d+):\s*(error):\s*(expected.*|missing.*|invalid.*)",
                "severity": "error", 
                "category": "syntax",
                "suggestions": [
                    "Check syntax for missing semicolons, braces, or parentheses",
                    "Verify proper C++ syntax usage",
                    "Review variable declarations and function definitions"
                ]
            },
            "type_errors": {
                "regex": r"(.*):(\d+):(\d+):\s*(error):\s*(cannot convert.*|no matching function.*|invalid conversion)",
                "severity": "error",
                "category": "type",
                "suggestions": [
                    "Check variable type declarations",
                    "Verify function parameter types",
                    "Add explicit type conversions"
                ]
            },
            "linker_errors": {
                "regex": r"(.*):\s*(undefined reference to|multiple definition of|collect2: error)",
                "severity": "error",
                "category": "linker",
                "suggestions": [
                    "Check function definitions and declarations",
                    "Verify library linking",
                    "Ensure proper namespace usage"
                ]
            },
            "compiler_warnings": {
                "regex": r"(.*):(\d+):(\d+):\s*(warning):\s*(.*)",
                "severity": "warning",
                "category": "warning",
                "suggestions": [
                    "Address compiler warnings for better code quality",
                    "Enable additional warning flags for stricter checking"
                ]
            }
        }
    
    async def analyze_compilation_logs(self, 
                                     logs: str, 
                                     generated_code: Dict[str, Any],
                                     project_name: str = "unknown") -> CompilationAnalysis:
        """
        Analyze compilation logs and provide comprehensive feedback.
        
        Args:
            logs: Raw compilation log output
            generated_code: The generated C++ code that was compiled
            project_name: Name of the project being compiled
            
        Returns:
            CompilationAnalysis with detailed insights
        """
        self.logger.info(f"Analyzing compilation logs for project: {project_name}")
        
        # Parse compilation errors and warnings
        errors, warnings = self._parse_compilation_output(logs)
        
        # Categorize errors
        error_categories = self._categorize_errors(errors + warnings)
        
        # Generate improvement suggestions
        improvement_suggestions = self._generate_improvement_suggestions(errors, warnings)
        
        # Analyze code quality issues
        code_quality_issues = self._analyze_code_quality_issues(errors, warnings, generated_code)
        
        # Generate LLM prompt enhancements
        llm_prompt_enhancements = await self._generate_llm_prompt_enhancements(
            errors, warnings, generated_code
        )
        
        # Create comprehensive analysis
        analysis = CompilationAnalysis(
            success=len(errors) == 0,
            total_errors=len(errors),
            total_warnings=len(warnings),
            errors=errors,
            warnings=warnings,
            error_categories=error_categories,
            improvement_suggestions=improvement_suggestions,
            code_quality_issues=code_quality_issues,
            llm_prompt_enhancements=llm_prompt_enhancements
        )
        
        self.logger.info(f"Compilation analysis complete: {len(errors)} errors, {len(warnings)} warnings")
        return analysis
    
    def _parse_compilation_output(self, logs: str) -> Tuple[List[CompilationError], List[CompilationError]]:
        """Parse compilation output into structured errors and warnings (OPTIMIZED)."""
        errors = []
        warnings = []
        lines = logs.split('\n')
        
        total_lines = len(lines)
        self.logger.info(f"Parsing compilation output with {total_lines} lines")
        
        # OPTIMIZATION 1: Pre-filter to only error/warning lines (10-100x faster)
        relevant_lines = []
        for line in lines:
            line_stripped = line.strip()
            # Quick check: only process lines that look like errors/warnings
            if any(keyword in line_stripped for keyword in ['error:', 'warning:', 'note:']):
                relevant_lines.append(line_stripped)
        
        self.logger.info(f"Filtered to {len(relevant_lines)} relevant lines (from {total_lines})")
        
        # OPTIMIZATION 2: Limit processing if too many lines (prevent hang)
        max_lines_to_process = 500
        if len(relevant_lines) > max_lines_to_process:
            self.logger.warning(f"⚠️ Too many error lines ({len(relevant_lines)}), processing first {max_lines_to_process} only")
            relevant_lines = relevant_lines[:max_lines_to_process]
        
        # OPTIMIZATION 3: Only log every 50th line (reduce I/O)
        for idx, line in enumerate(relevant_lines):
            if not line:
                continue
            
            # Only log every 50th line to reduce I/O overhead
            if idx % 50 == 0:
                self.logger.debug(f"Processing line {idx}/{len(relevant_lines)}: {line[:100]}...")
                
            # Try to match against known error patterns
            for pattern_name, pattern_info in self.error_patterns.items():
                regex = pattern_info["regex"]
                match = re.search(regex, line)
                
                if match:
                    groups = match.groups()
                    
                    # Extract error information
                    file_name = groups[0] if len(groups) > 0 else None
                    line_number = int(groups[1]) if len(groups) > 1 and groups[1].isdigit() else None
                    column_number = int(groups[2]) if len(groups) > 2 and groups[2].isdigit() else None
                    severity = groups[3] if len(groups) > 3 else "error"
                    error_message = groups[4] if len(groups) > 4 else line
                    
                    # Clean up file name
                    if file_name and file_name.startswith('./'):
                        file_name = file_name[2:]
                    
                    # Create error object
                    compilation_error = CompilationError(
                        line_number=line_number,
                        column_number=column_number,
                        error_type=pattern_info["category"],
                        error_message=error_message,
                        file_name=file_name,
                        severity=pattern_info["severity"]
                    )
                    
                    if severity == "error":
                        errors.append(compilation_error)
                    elif severity == "warning":
                        warnings.append(compilation_error)
                    
                    break
        
        self.logger.info(f"✅ Parsing complete: {len(errors)} errors, {len(warnings)} warnings extracted")
        return errors, warnings
    
    def _categorize_errors(self, errors: List[CompilationError]) -> Dict[str, int]:
        """Categorize errors by type and count occurrences."""
        categories = {}
        
        for error in errors:
            category = error.error_type
            categories[category] = categories.get(category, 0) + 1
        
        return categories
    
    def _generate_improvement_suggestions(self, 
                                        errors: List[CompilationError],
                                        warnings: List[CompilationError]) -> List[str]:
        """Generate improvement suggestions based on error patterns."""
        suggestions = set()
        
        # Add suggestions from error patterns
        for error in errors + warnings:
            pattern_info = self._get_pattern_info(error.error_type)
            if pattern_info and "suggestions" in pattern_info:
                suggestions.update(pattern_info["suggestions"])
        
        # Add specific suggestions based on error categories
        error_categories = self._categorize_errors(errors + warnings)
        
        if "include" in error_categories:
            suggestions.add("Review and fix all #include statements")
        
        if "syntax" in error_categories:
            suggestions.add("Perform thorough syntax review of generated code")
        
        if "type" in error_categories:
            suggestions.add("Ensure consistent type usage throughout the code")
        
        if "linker" in error_categories:
            suggestions.add("Check function definitions and linking requirements")
        
        return list(suggestions)
    
    def _analyze_code_quality_issues(self, 
                                   errors: List[CompilationError],
                                   warnings: List[CompilationError],
                                   generated_code: Dict[str, Any]) -> List[str]:
        """Analyze code quality issues from compilation feedback."""
        issues = []
        
        # Analyze error patterns for quality issues
        error_categories = self._categorize_errors(errors + warnings)
        
        if "include" in error_categories:
            issues.append("Poor include statement management - may indicate inadequate header organization")
        
        if "syntax" in error_categories:
            issues.append("Syntax errors suggest need for better C++ code structure")
        
        if "type" in error_categories:
            issues.append("Type mismatches indicate insufficient type safety considerations")
        
        if len(warnings) > 5:
            issues.append("High number of warnings suggests need for stricter code quality standards")
        
        # Analyze generated code structure if available
        if generated_code and "files" in generated_code:
            files = generated_code["files"]
            if len(files) > 1 and "include" in error_categories:
                issues.append("Multi-file project may need better header organization and dependencies")
        
        return issues
    
    async def _generate_llm_prompt_enhancements(self,
                                              errors: List[CompilationError],
                                              warnings: List[CompilationError],
                                              generated_code: Dict[str, Any]) -> List[str]:
        """Generate LLM prompt enhancements using AI analysis."""
        if not errors and not warnings:
            return []
        
        # Create context for LLM analysis
        error_summary = self._create_error_summary(errors, warnings)
        
        prompt = f"""
        As an expert C++ code generation assistant, analyze these compilation errors and provide specific prompt enhancements for improving C++ code generation:

        Compilation Errors/Warnings:
        {error_summary}

        Current Generated Code Structure:
        {json.dumps(generated_code.get('files', {}), indent=2) if generated_code else 'No code available'}

        Please provide specific prompt enhancements that would help generate better C++ code that avoids these compilation issues. Focus on:
        1. Specific syntax improvements
        2. Include statement best practices
        3. Type safety considerations
        4. Code structure recommendations

        Return as a JSON array of enhancement strings.
        """
        
        try:
            response = self.llm_client.get_completion(prompt)
            # Parse LLM response to extract enhancements
            enhancements = self._parse_llm_enhancements(response)
            return enhancements
        except Exception as e:
            self.logger.warning(f"Failed to generate LLM prompt enhancements: {e}")
            return self._get_fallback_enhancements(errors, warnings)
    
    def _create_error_summary(self, errors: List[CompilationError], warnings: List[CompilationError]) -> str:
        """Create a summary of errors and warnings for LLM analysis."""
        summary_parts = []
        
        if errors:
            summary_parts.append("ERRORS:")
            for error in errors[:5]:  # Limit to first 5 errors
                summary_parts.append(f"  - {error.error_type}: {error.error_message}")
                if error.file_name and error.line_number:
                    summary_parts.append(f"    Location: {error.file_name}:{error.line_number}")
        
        if warnings:
            summary_parts.append("WARNINGS:")
            for warning in warnings[:3]:  # Limit to first 3 warnings
                summary_parts.append(f"  - {warning.error_message}")
        
        return "\n".join(summary_parts)
    
    def _parse_llm_enhancements(self, llm_response: str) -> List[str]:
        """Parse LLM response to extract enhancement suggestions."""
        try:
            # Try to extract JSON array from response
            json_match = re.search(r'\[(.*?)\]', llm_response, re.DOTALL)
            if json_match:
                json_str = "[" + json_match.group(1) + "]"
                enhancements = json.loads(json_str)
                if isinstance(enhancements, list):
                    return enhancements
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # Fallback: extract bullet points or numbered items
        enhancements = []
        lines = llm_response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith(('- ', '* ', '1. ', '2. ', '3. ', '4. ', '5. ')):
                enhancement = re.sub(r'^[-*]\s*|\d+\.\s*', '', line)
                if enhancement:
                    enhancements.append(enhancement)
        
        return enhancements[:5]  # Limit to 5 enhancements
    
    def _get_fallback_enhancements(self, errors: List[CompilationError], warnings: List[CompilationError]) -> List[str]:
        """Provide fallback enhancements when LLM analysis fails."""
        enhancements = []
        
        error_categories = self._categorize_errors(errors + warnings)
        
        if "include" in error_categories:
            enhancements.append("Use proper C++ include syntax with <> for system headers and \"\" for local headers")
        
        if "syntax" in error_categories:
            enhancements.append("Ensure all statements end with semicolons and braces are properly matched")
        
        if "type" in error_categories:
            enhancements.append("Use explicit type declarations and avoid implicit type conversions")
        
        if "linker" in error_categories:
            enhancements.append("Ensure all functions are properly declared and defined")
        
        return enhancements
    
    def _get_pattern_info(self, error_type: str) -> Optional[Dict]:
        """Get pattern information for an error type."""
        for pattern_info in self.error_patterns.values():
            if pattern_info["category"] == error_type:
                return pattern_info
        return None
    
    async def create_node(self, state: ConversionState) -> ConversionState:
        """Create LangGraph node for compilation log analysis."""
        # This would be called from the workflow
        return state
    
    def get_tools(self) -> List[Any]:
        """Get tools available to this agent."""
        return []
