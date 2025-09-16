"""Assessor Agent for analyzing and improving generated C++ code."""

import re
import ast
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json

from ..utils.logger import get_logger
from ..utils.config import LLMConfig
from ..tools.llm_client import create_llm_client, LLMClient

@dataclass
class CodeIssue:
    """Represents a code quality issue."""
    severity: str  # "critical", "high", "medium", "low"
    category: str  # "algorithmic", "performance", "error_handling", "style", "security"
    line_number: Optional[int]
    description: str
    suggestion: str
    confidence: float  # 0.0 to 1.0

@dataclass
class CodeMetrics:
    """Code quality metrics."""
    algorithmic_accuracy: float
    performance_score: float
    error_handling_score: float
    code_style_score: float
    maintainability_score: float
    overall_score: float

@dataclass
class AssessmentResult:
    """Result of code assessment."""
    metrics: CodeMetrics
    issues: List[CodeIssue]
    suggestions: List[str]
    improved_code: Optional[str] = None
    assessment_summary: str = ""

class AssessorAgent:
    """Agent for assessing and improving generated C++ code quality."""
    
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        """Initialize the Assessor Agent."""
        self.logger = get_logger("assessor_agent")
        self.llm_client = create_llm_client(llm_config) if llm_config else None
        
        # Code quality patterns
        self.quality_patterns = {
            "indexing_issues": [
                r"for\s*\(\s*int\s+\w+\s*=\s*\w+\s*;\s*\w+\s*<\s*\w+\s*\+\s*\w+\s*;\s*\+\+\w+\s*\)",
                r"\[\s*\w+\s*\+\s*\w+\s*\]",  # Array indexing with addition
            ],
            "memory_issues": [
                r"new\s+\w+",  # Raw new usage
                r"delete\s+\w+",  # Raw delete usage
                r"malloc\s*\(",  # C-style memory allocation
                r"free\s*\(",  # C-style memory deallocation
            ],
            "error_handling": [
                r"throw\s+std::",  # Exception throwing
                r"try\s*\{",  # Try blocks
                r"catch\s*\(",  # Catch blocks
                r"assert\s*\(",  # Assertions
            ],
            "performance_issues": [
                r"std::vector.*resize\s*\(",  # Vector resizing in loops
                r"for\s*\(.*std::vector",  # Vector operations in loops
                r"std::cout.*<<",  # Console output in performance-critical code
            ],
            "style_issues": [
                r"using\s+namespace\s+std",  # Using namespace std
                r"#define\s+\w+",  # C-style defines
                r"goto\s+\w+",  # Goto statements
            ]
        }
        
        self.logger.info("Assessor Agent initialized")
    
    def assess_code(self, cpp_code: str, matlab_code: str, 
                   original_function: Optional[str] = None) -> AssessmentResult:
        """Assess the quality of generated C++ code."""
        self.logger.info("Starting C++ code assessment")
        
        # Perform static analysis
        static_issues = self._perform_static_analysis(cpp_code)
        
        # Perform algorithmic analysis
        algorithmic_issues = self._analyze_algorithmic_accuracy(cpp_code, matlab_code)
        
        # Perform LLM-based analysis if available
        llm_issues = []
        if self.llm_client:
            llm_issues = self._perform_llm_analysis(cpp_code, matlab_code, original_function)
        
        # Combine all issues
        all_issues = static_issues + algorithmic_issues + llm_issues
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_issues, cpp_code)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(all_issues, metrics)
        
        # Generate improved code if requested
        improved_code = None
        if self.llm_client and any(issue.severity in ["critical", "high"] for issue in all_issues):
            improved_code = self._generate_improved_code(cpp_code, all_issues, matlab_code)
        
        # Create assessment summary
        summary = self._create_assessment_summary(metrics, all_issues)
        
        return AssessmentResult(
            metrics=metrics,
            issues=all_issues,
            suggestions=suggestions,
            improved_code=improved_code,
            assessment_summary=summary
        )
    
    def _perform_static_analysis(self, cpp_code: str) -> List[CodeIssue]:
        """Perform static analysis on C++ code."""
        issues = []
        lines = cpp_code.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for indexing issues
            if re.search(r"for\s*\(\s*int\s+\w+\s*=\s*\w+\s*;\s*\w+\s*<\s*\w+\s*\+\s*\w+\s*;\s*\+\+\w+\s*\)", line):
                issues.append(CodeIssue(
                    severity="high",
                    category="algorithmic",
                    line_number=i,
                    description="Potential indexing issue in loop condition",
                    suggestion="Verify loop bounds match MATLAB 1-based indexing",
                    confidence=0.8
                ))
            
            # Check for memory issues
            if re.search(r"new\s+\w+", line):
                issues.append(CodeIssue(
                    severity="medium",
                    category="performance",
                    line_number=i,
                    description="Raw pointer allocation detected",
                    suggestion="Consider using smart pointers or RAII containers",
                    confidence=0.9
                ))
            
            # Check for missing error handling
            if re.search(r"Eigen::", line) and not re.search(r"try\s*\{|catch\s*\(|if\s*\(.*\.info\(\)", cpp_code):
                issues.append(CodeIssue(
                    severity="medium",
                    category="error_handling",
                    line_number=i,
                    description="Eigen operation without error checking",
                    suggestion="Add error checking for Eigen operations",
                    confidence=0.7
                ))
            
            # Check for performance issues
            if re.search(r"std::cout.*<<", line) and "main" not in cpp_code.lower():
                issues.append(CodeIssue(
                    severity="low",
                    category="performance",
                    line_number=i,
                    description="Console output in performance-critical code",
                    suggestion="Remove or conditionally compile debug output",
                    confidence=0.8
                ))
        
        return issues
    
    def _analyze_algorithmic_accuracy(self, cpp_code: str, matlab_code: str) -> List[CodeIssue]:
        """Analyze algorithmic accuracy compared to MATLAB code."""
        issues = []
        
        # Check for matrix indexing patterns
        matlab_loops = re.findall(r"for\s+\w+\s*=\s*\d+:\s*size\([^,]+,\s*\d+\)", matlab_code)
        cpp_loops = re.findall(r"for\s*\(\s*int\s+\w+\s*=\s*\d+\s*;\s*\w+\s*<\s*[^;]+;\s*\+\+\w+\s*\)", cpp_code)
        
        if len(matlab_loops) != len(cpp_loops):
            issues.append(CodeIssue(
                severity="high",
                category="algorithmic",
                line_number=None,
                description="Mismatch in loop structure between MATLAB and C++",
                suggestion="Verify all MATLAB loops are correctly converted",
                confidence=0.9
            ))
        
        # Check for matrix operations
        matlab_matrix_ops = re.findall(r"[A-Z]\s*=\s*[A-Z].*[A-Z]", matlab_code)
        cpp_matrix_ops = re.findall(r"Eigen::MatrixXd\s+\w+\s*=\s*[^;]+;", cpp_code)
        
        if len(matlab_matrix_ops) > len(cpp_matrix_ops):
            issues.append(CodeIssue(
                severity="critical",
                category="algorithmic",
                line_number=None,
                description="Missing matrix operations in C++ code",
                suggestion="Ensure all MATLAB matrix operations are implemented",
                confidence=0.8
            ))
        
        # Check for function signatures
        matlab_func = re.search(r"function\s+.*=\s*(\w+)\s*\(([^)]+)\)", matlab_code)
        cpp_func = re.search(r"(\w+)\s*\(([^)]+)\)\s*\{", cpp_code)
        
        if matlab_func and cpp_func:
            matlab_params = [p.strip() for p in matlab_func.group(2).split(',')]
            cpp_params = [p.strip() for p in cpp_func.group(2).split(',')]
            
            if len(matlab_params) != len(cpp_params):
                issues.append(CodeIssue(
                    severity="critical",
                    category="algorithmic",
                    line_number=None,
                    description="Parameter count mismatch between MATLAB and C++",
                    suggestion="Verify function parameters match exactly",
                    confidence=1.0
                ))
        
        return issues
    
    def _perform_llm_analysis(self, cpp_code: str, matlab_code: str, 
                            original_function: Optional[str] = None) -> List[CodeIssue]:
        """Perform LLM-based analysis of the code."""
        if not self.llm_client:
            return []
        
        prompt = f"""/no_think

Analyze the following C++ code generated from MATLAB and identify quality issues:

MATLAB Original:
```matlab
{matlab_code}
```

Generated C++ Code:
```cpp
{cpp_code}
```

Please identify issues in the following categories:
1. Algorithmic accuracy (indexing, matrix operations, mathematical correctness)
2. Performance (memory usage, computational efficiency)
3. Error handling (exception handling, input validation)
4. Code style (C++ best practices, readability)
5. Security (buffer overflows, memory leaks)

For each issue, provide:
- Severity: critical, high, medium, low
- Category: algorithmic, performance, error_handling, style, security
- Description: What's wrong
- Suggestion: How to fix it
- Confidence: 0.0 to 1.0

Format as JSON array of issues.
"""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.invoke(messages)
            # Parse LLM response and convert to CodeIssue objects
            # This is a simplified version - in practice, you'd need robust JSON parsing
            issues = self._parse_llm_response(response)
            return issues
        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}")
            return []
    
    def _parse_llm_response(self, response: str) -> List[CodeIssue]:
        """Parse LLM response into CodeIssue objects."""
        issues = []
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                issues_data = json.loads(json_match.group())
                for issue_data in issues_data:
                    issues.append(CodeIssue(
                        severity=issue_data.get('severity', 'medium'),
                        category=issue_data.get('category', 'style'),
                        line_number=issue_data.get('line_number'),
                        description=issue_data.get('description', ''),
                        suggestion=issue_data.get('suggestion', ''),
                        confidence=issue_data.get('confidence', 0.5)
                    ))
        except Exception as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
        
        return issues
    
    def _calculate_metrics(self, issues: List[CodeIssue], cpp_code: str) -> CodeMetrics:
        """Calculate code quality metrics."""
        # Count issues by category and severity
        category_counts = {}
        severity_counts = {}
        
        for issue in issues:
            category_counts[issue.category] = category_counts.get(issue.category, 0) + 1
            severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1
        
        # Calculate scores (higher is better)
        total_issues = len(issues)
        critical_issues = severity_counts.get('critical', 0)
        high_issues = severity_counts.get('high', 0)
        
        # Algorithmic accuracy score
        algorithmic_issues = category_counts.get('algorithmic', 0)
        algorithmic_accuracy = max(0, 1.0 - (algorithmic_issues * 0.2))
        
        # Performance score
        performance_issues = category_counts.get('performance', 0)
        performance_score = max(0, 1.0 - (performance_issues * 0.15))
        
        # Error handling score
        error_handling_issues = category_counts.get('error_handling', 0)
        error_handling_score = max(0, 1.0 - (error_handling_issues * 0.1))
        
        # Code style score
        style_issues = category_counts.get('style', 0)
        code_style_score = max(0, 1.0 - (style_issues * 0.05))
        
        # Maintainability score (based on overall code quality)
        maintainability_score = max(0, 1.0 - (total_issues * 0.05))
        
        # Overall score
        overall_score = (algorithmic_accuracy * 0.4 + 
                        performance_score * 0.2 + 
                        error_handling_score * 0.2 + 
                        code_style_score * 0.1 + 
                        maintainability_score * 0.1)
        
        return CodeMetrics(
            algorithmic_accuracy=algorithmic_accuracy,
            performance_score=performance_score,
            error_handling_score=error_handling_score,
            code_style_score=code_style_score,
            maintainability_score=maintainability_score,
            overall_score=overall_score
        )
    
    def _generate_suggestions(self, issues: List[CodeIssue], metrics: CodeMetrics) -> List[str]:
        """Generate improvement suggestions based on issues and metrics."""
        suggestions = []
        
        # Critical issues
        critical_issues = [i for i in issues if i.severity == 'critical']
        if critical_issues:
            suggestions.append("🚨 CRITICAL: Fix algorithmic issues immediately - these affect correctness")
            for issue in critical_issues:
                suggestions.append(f"   - {issue.description}: {issue.suggestion}")
        
        # High priority issues
        high_issues = [i for i in issues if i.severity == 'high']
        if high_issues:
            suggestions.append("⚠️ HIGH: Address performance and accuracy issues")
            for issue in high_issues:
                suggestions.append(f"   - {issue.description}: {issue.suggestion}")
        
        # Performance improvements
        if metrics.performance_score < 0.7:
            suggestions.append("⚡ PERFORMANCE: Optimize memory usage and computational efficiency")
            suggestions.append("   - Consider using Eigen's optimized operations")
            suggestions.append("   - Pre-allocate matrices outside loops")
            suggestions.append("   - Remove debug output from performance-critical code")
        
        # Error handling improvements
        if metrics.error_handling_score < 0.7:
            suggestions.append("🛡️ ERROR HANDLING: Add comprehensive error checking")
            suggestions.append("   - Add input validation for all parameters")
            suggestions.append("   - Check Eigen operation success")
            suggestions.append("   - Add exception handling for edge cases")
        
        # Code style improvements
        if metrics.code_style_score < 0.8:
            suggestions.append("📝 CODE STYLE: Improve code readability and maintainability")
            suggestions.append("   - Add comprehensive documentation")
            suggestions.append("   - Use consistent naming conventions")
            suggestions.append("   - Consider using modern C++ features")
        
        return suggestions
    
    def _generate_improved_code(self, cpp_code: str, issues: List[CodeIssue], 
                              matlab_code: str) -> Optional[str]:
        """Generate improved C++ code based on identified issues."""
        if not self.llm_client:
            return None
        
        # Focus on critical and high severity issues
        critical_issues = [i for i in issues if i.severity in ['critical', 'high']]
        
        if not critical_issues:
            return None
        
        issues_text = "\n".join([f"- {issue.description}: {issue.suggestion}" 
                               for issue in critical_issues])
        
        prompt = f"""/no_think

Improve the following C++ code by addressing the identified issues:

MATLAB Original:
```matlab
{matlab_code}
```

Current C++ Code:
```cpp
{cpp_code}
```

Issues to Fix:
{issues_text}

Please provide an improved version that:
1. Fixes all critical and high severity issues
2. Maintains the same functionality as the MATLAB code
3. Follows C++ best practices
4. Includes proper error handling
5. Is optimized for performance

Return only the improved C++ code.
"""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            improved_code = self.llm_client.invoke(messages)
            return improved_code
        except Exception as e:
            self.logger.error(f"Failed to generate improved code: {e}")
            return None
    
    def _create_assessment_summary(self, metrics: CodeMetrics, issues: List[CodeIssue]) -> str:
        """Create a summary of the assessment."""
        total_issues = len(issues)
        critical_count = len([i for i in issues if i.severity == 'critical'])
        high_count = len([i for i in issues if i.severity == 'high'])
        
        summary = f"""
📊 CODE ASSESSMENT SUMMARY
========================

Overall Score: {metrics.overall_score:.1f}/10

📈 Detailed Metrics:
  • Algorithmic Accuracy: {metrics.algorithmic_accuracy:.1f}/10
  • Performance: {metrics.performance_score:.1f}/10
  • Error Handling: {metrics.error_handling_score:.1f}/10
  • Code Style: {metrics.code_style_score:.1f}/10
  • Maintainability: {metrics.maintainability_score:.1f}/10

🚨 Issues Found: {total_issues}
  • Critical: {critical_count}
  • High: {high_count}
  • Medium: {len([i for i in issues if i.severity == 'medium'])}
  • Low: {len([i for i in issues if i.severity == 'low'])}

💡 Recommendation: {'Fix critical issues immediately' if critical_count > 0 else 
                   'Address high priority issues' if high_count > 0 else 
                   'Code quality is good, minor improvements suggested'}
"""
        return summary
    
    def assess_file(self, cpp_file_path: Path, matlab_file_path: Path) -> AssessmentResult:
        """Assess a C++ file against its MATLAB source."""
        try:
            cpp_code = cpp_file_path.read_text()
            matlab_code = matlab_file_path.read_text()
            
            return self.assess_code(cpp_code, matlab_code)
        except Exception as e:
            self.logger.error(f"Failed to assess file {cpp_file_path}: {e}")
            raise
    
    def generate_assessment_report(self, result: AssessmentResult, 
                                 output_path: Path) -> None:
        """Generate a detailed assessment report."""
        report = f"""
# C++ Code Assessment Report

{result.assessment_summary}

## Issues Found

"""
        
        for issue in result.issues:
            report += f"""
### {issue.severity.upper()} - {issue.category.upper()}
- **Line**: {issue.line_number or 'N/A'}
- **Description**: {issue.description}
- **Suggestion**: {issue.suggestion}
- **Confidence**: {issue.confidence:.1f}

"""
        
        if result.suggestions:
            report += """
## Improvement Suggestions

"""
            for suggestion in result.suggestions:
                report += f"{suggestion}\n"
        
        if result.improved_code:
            report += """
## Improved Code

```cpp
""" + result.improved_code + """
```
"""
        
        output_path.write_text(report)
        self.logger.info(f"Assessment report saved to {output_path}")