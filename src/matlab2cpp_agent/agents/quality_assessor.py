#!/usr/bin/env python3
"""
Quality Assessor Agent

This agent is responsible for assessing the quality of generated C++ code
and identifying issues across multiple dimensions. It uses LLM to perform
comprehensive code analysis and provide improvement suggestions.
"""

import re
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
from loguru import logger

from matlab2cpp_agent.tools.llm_client import create_llm_client
from matlab2cpp_agent.utils.config import LLMConfig

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
    assessment_summary: str = ""

class QualityAssessorAgent:
    """Agent responsible for assessing C++ code quality and identifying issues."""
    
    def __init__(self, llm_config: LLMConfig):
        """Initialize the quality assessor agent."""
        self.llm_config = llm_config
        self.llm_client = create_llm_client(llm_config)
        self.logger = logger.bind(name="quality_assessor_agent")
        self.logger.info("Quality Assessor Agent initialized")
    
    def assess_code_quality(self, 
                          cpp_code: str, 
                          matlab_code: str,
                          project_name: str) -> AssessmentResult:
        """
        Assess the quality of generated C++ code.
        
        Args:
            cpp_code: Generated C++ code to assess
            matlab_code: Original MATLAB code for comparison
            project_name: Name of the project
            
        Returns:
            AssessmentResult with quality metrics and issues
        """
        self.logger.info(f"Assessing code quality for project: {project_name}")
        
        try:
            # Perform LLM-based analysis
            issues = self._perform_llm_analysis(cpp_code, matlab_code)
            
            # Calculate quality metrics
            metrics = self._calculate_quality_metrics(issues)
            
            # Generate suggestions
            suggestions = self._generate_suggestions(issues, metrics)
            
            # Create assessment summary
            summary = self._create_assessment_summary(metrics, issues, project_name)
            
            result = AssessmentResult(
                metrics=metrics,
                issues=issues,
                suggestions=suggestions,
                assessment_summary=summary
            )
            
            self.logger.info(f"Assessment complete: {len(issues)} issues, "
                           f"overall score: {metrics.overall_score:.1f}/10")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Assessment failed: {e}")
            # Return basic assessment as fallback
            return self._create_fallback_assessment(e)
    
    def _perform_llm_analysis(self, cpp_code: str, matlab_code: str) -> List[CodeIssue]:
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
            issues = self._parse_llm_response(response)
            return issues
            
        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}")
            return []
    
    def _parse_llm_response(self, response: str) -> List[CodeIssue]:
        """Parse LLM response and convert to CodeIssue objects."""
        issues = []
        
        try:
            # Try to parse as JSON first
            if response.strip().startswith('['):
                issues_data = json.loads(response)
                for issue_data in issues_data:
                    issue = CodeIssue(
                        severity=issue_data.get('severity', 'medium'),
                        category=issue_data.get('category', 'style'),
                        line_number=issue_data.get('line_number'),
                        description=issue_data.get('description', ''),
                        suggestion=issue_data.get('suggestion', ''),
                        confidence=issue_data.get('confidence', 0.5)
                    )
                    issues.append(issue)
            else:
                # Fallback: parse text response
                issues = self._parse_text_response(response)
                
        except json.JSONDecodeError:
            # Fallback: parse text response
            issues = self._parse_text_response(response)
        
        return issues
    
    def _parse_text_response(self, response: str) -> List[CodeIssue]:
        """Parse text response as fallback."""
        issues = []
        
        # Simple text parsing - look for issue patterns
        lines = response.split('\n')
        current_issue = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for severity indicators
            if any(severity in line.lower() for severity in ['critical', 'high', 'medium', 'low']):
                if current_issue:
                    issues.append(current_issue)
                
                # Extract severity and category
                severity = 'medium'
                category = 'style'
                
                if 'critical' in line.lower():
                    severity = 'critical'
                elif 'high' in line.lower():
                    severity = 'high'
                elif 'low' in line.lower():
                    severity = 'low'
                
                if 'algorithmic' in line.lower():
                    category = 'algorithmic'
                elif 'performance' in line.lower():
                    category = 'performance'
                elif 'error' in line.lower():
                    category = 'error_handling'
                elif 'security' in line.lower():
                    category = 'security'
                
                current_issue = CodeIssue(
                    severity=severity,
                    category=category,
                    line_number=None,
                    description=line,
                    suggestion="Review and fix",
                    confidence=0.7
                )
            elif current_issue and line.startswith('-'):
                # This is likely a suggestion
                current_issue.suggestion = line[1:].strip()
        
        if current_issue:
            issues.append(current_issue)
        
        return issues
    
    def _calculate_quality_metrics(self, issues: List[CodeIssue]) -> CodeMetrics:
        """Calculate quality metrics from identified issues."""
        if not issues:
            return CodeMetrics(
                algorithmic_accuracy=10.0,
                performance_score=10.0,
                error_handling_score=10.0,
                code_style_score=10.0,
                maintainability_score=10.0,
                overall_score=10.0
            )
        
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
        algorithmic_accuracy = max(0, 10.0 - (algorithmic_issues * 2.0))
        
        # Performance score
        performance_issues = category_counts.get('performance', 0)
        performance_score = max(0, 10.0 - (performance_issues * 1.5))
        
        # Error handling score
        error_handling_issues = category_counts.get('error_handling', 0)
        error_handling_score = max(0, 10.0 - (error_handling_issues * 1.0))
        
        # Code style score
        style_issues = category_counts.get('style', 0)
        code_style_score = max(0, 10.0 - (style_issues * 0.5))
        
        # Maintainability score (based on overall code quality)
        maintainability_score = max(0, 10.0 - (total_issues * 0.5))
        
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
        critical_issues = [issue for issue in issues if issue.severity == 'critical']
        if critical_issues:
            suggestions.append(f"Fix {len(critical_issues)} critical issues immediately")
        
        # High severity issues
        high_issues = [issue for issue in issues if issue.severity == 'high']
        if high_issues:
            suggestions.append(f"Address {len(high_issues)} high severity issues")
        
        # Category-specific suggestions
        if metrics.algorithmic_accuracy < 5.0:
            suggestions.append("Review algorithmic correctness and mathematical operations")
        
        if metrics.performance_score < 5.0:
            suggestions.append("Optimize performance and memory usage")
        
        if metrics.error_handling_score < 5.0:
            suggestions.append("Add comprehensive error handling and input validation")
        
        if metrics.code_style_score < 5.0:
            suggestions.append("Improve code style and follow C++ best practices")
        
        if not suggestions:
            suggestions.append("Code quality is good, minor improvements recommended")
        
        return suggestions
    
    def _create_assessment_summary(self, metrics: CodeMetrics, issues: List[CodeIssue], project_name: str) -> str:
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

💡 Recommendations:
"""
        
        for suggestion in self._generate_suggestions(issues, metrics):
            summary += f"  • {suggestion}\n"
        
        return summary
    
    def _create_fallback_assessment(self, error: Exception) -> AssessmentResult:
        """Create fallback assessment when analysis fails."""
        return AssessmentResult(
            metrics=CodeMetrics(
                algorithmic_accuracy=0.0,
                performance_score=0.0,
                error_handling_score=0.0,
                code_style_score=0.0,
                maintainability_score=0.0,
                overall_score=0.0
            ),
            issues=[],
            suggestions=[f"Assessment failed: {error}"],
            assessment_summary="Assessment failed due to error"
        )
    
    def generate_assessment_report(self, result: AssessmentResult, output_path: Path) -> None:
        """Generate a detailed assessment report."""
        report = result.assessment_summary
        
        if result.issues:
            report += "\n\n## Detailed Issues\n\n"
            for i, issue in enumerate(result.issues, 1):
                report += f"### Issue {i}: {issue.severity.upper()} - {issue.category.upper()}\n"
                report += f"**Description:** {issue.description}\n\n"
                report += f"**Suggestion:** {issue.suggestion}\n\n"
                if issue.line_number:
                    report += f"**Line:** {issue.line_number}\n\n"
                report += f"**Confidence:** {issue.confidence:.1f}\n\n"
        
        if result.suggestions:
            report += "\n## Improvement Suggestions\n\n"
            for suggestion in result.suggestions:
                report += f"{suggestion}\n"
        
        output_path.write_text(report)
        self.logger.info(f"Assessment report saved to {output_path}")
