# /src/matlab2cpp_agentic_service/agents/quality_assessor.py
"""
Quality Assessor (LLM-first)
============================

This agent reviews generated C++ code and identifies quality issues
across multiple categories:
  - algorithmic correctness (numerical stability, correct eigenvector selection, etc.)
  - performance (explicit inverses, raw pointers, missing solver checks, etc.)
  - error handling (lack of try/catch, possible leaks)
  - style (readability, trailing whitespace, 'using namespace std')
  - maintainability (file length, modularity, const-correctness)
  - security (buffer overflows, unchecked array indices, etc.)

The assessor prioritises an LLM-based analysis: if a `llm_client` is
provided, it constructs a detailed prompt with guidelines and instructs
the model to output issues in JSON format.  If the LLM fails or yields
invalid JSON, the agent falls back to deterministic heuristics to ensure
basic quality checking.  It merges issues, computes per-category scores
(out of 10) based on severity, and returns a summary.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import re
import json

@dataclass
class CodeIssue:
    severity: str
    category: str
    description: str
    suggestion: str
    confidence: float = 0.8

@dataclass
class AssessmentResult:
    issues: List[CodeIssue]
    metrics: Dict[str, float]
    summary: str

class QualityAssessorAgent:
    def __init__(self, llm_client: Optional[Any] = None) -> None:
        """
        Args:
            llm_client: Optional.  If provided, must implement a method
                `get_completion(prompt: str) -> str`.  The LLM will be asked
                to identify quality issues in JSON format.
        """
        self.llm_client = llm_client
        self.penalty = {
            'critical': 3.0,
            'high': 2.0,
            'medium': 1.0,
            'low': 0.5,
        }

    def assess(self, code: str, matlab_code: str = "", conversion_plan: Dict[str, Any] = None, conversion_mode: str = "result-focused") -> AssessmentResult:
        """
        Assess the quality of C++ code, using the LLM if available.  If the
        LLM call fails or returns invalid JSON, fall back to a heuristic
        assessment.

        Args:
            code: The C++ code to assess
            matlab_code: The original MATLAB code for algorithmic fidelity comparison
            conversion_plan: The conversion plan used

        Returns:
            An AssessmentResult with issues, per-category metrics and a summary.
        """
        issues: List[CodeIssue] = []
        
        # Add algorithmic fidelity assessment if MATLAB code is provided
        if matlab_code and conversion_plan:
            fidelity_issues = self._assess_algorithmic_fidelity(code, matlab_code, conversion_plan, conversion_mode)
            issues.extend(fidelity_issues)
        
        # Try LLM analysis first
        if self.llm_client:
            try:
                prompt = self._create_llm_prompt(code)
                response = self.llm_client.get_completion(prompt)
                llm_issues = self._parse_llm_issues(response)
                issues.extend(llm_issues)
            except Exception:
                pass
        # Fallback to heuristics if no valid LLM issues (but preserve algorithmic issues)
        if not any(issue.category != 'algorithmic' for issue in issues):
            heuristic_issues = self._heuristic_analysis(code)
            issues.extend(heuristic_issues)
        metrics = self._compute_metrics(issues)
        summary = self._build_summary(metrics, issues)
        return AssessmentResult(issues, metrics, summary)

    # ----------------------------------------------------------------------
    # LLM-based analysis
    # ----------------------------------------------------------------------
    def _create_llm_prompt(self, code: str) -> str:
        """
        Build a prompt instructing the LLM to review the C++ code and return
        JSON-formatted issues across categories: algorithmic, performance,
        error handling, style, maintainability, security.

        Returns:
            A string prompt for the LLM.
        """
        prompt_lines: List[str] = []
        prompt_lines.append(
            "You are a C++ quality auditor reviewing code translated from MATLAB. "
            "Identify any remaining quality issues across these categories:\n"
            "  - algorithmic: misuse of numerical algorithms, wrong eigenvector selection, unstable operations\n"
            "  - performance: explicit inverses, inefficient memory usage, raw pointers, missing solver success checks\n"
            "  - error_handling: missing try/catch, potential memory leaks, unvalidated inputs\n"
            "  - style: poor formatting, trailing whitespace, 'using namespace std', long lines\n"
            "  - maintainability: overly long files, lack of modularity, missing const qualifiers\n"
            "  - security: buffer overflows, unchecked array indices, unsanitised inputs\n"
        )
        prompt_lines.append(
            "Please review the following C++ code and return a JSON array.  "
            "Each element must have the keys: severity (critical/high/medium/low), "
            "category (algorithmic/performance/error_handling/style/maintainability/security), "
            "description, suggestion, confidence (0–1)."
        )
        prompt_lines.append("\nC++ code:\n")
        prompt_lines.append(code.strip())
        prompt_lines.append("\nRespond only with the JSON array.  Do not include any additional text.")
        return "\n".join(prompt_lines)

    def _parse_llm_issues(self, response: str) -> List[CodeIssue]:
        """
        Parse the JSON response from the LLM into CodeIssue instances.

        Args:
            response: The raw string returned by the LLM.

        Returns:
            A list of CodeIssue objects.  Returns an empty list if parsing fails.
        """
        issues: List[CodeIssue] = []
        try:
            data = json.loads(response.strip())
        except Exception:
            return issues
        if not isinstance(data, list):
            return issues
        for item in data:
            try:
                sev  = str(item['severity']).lower()
                cat  = str(item['category']).lower().replace(' ','_')
                desc = str(item['description']).strip()
                sugg = str(item['suggestion']).strip()
                conf = float(item.get('confidence', 0.8))
                if not (sev and cat and desc and sugg):
                    continue
                # Normalise severity and category
                if sev not in {'critical','high','medium','low'}:
                    sev = 'low'
                valid_categories = {
                    'algorithmic','performance','error_handling','style','maintainability','security'
                }
                if cat not in valid_categories:
                    cat = 'style'
                issues.append(CodeIssue(sev, cat, desc, sugg, conf))
            except Exception:
                continue
        return issues

    # ----------------------------------------------------------------------
    # Heuristic analysis (fallback)
    # ----------------------------------------------------------------------
    def _heuristic_analysis(self, code: str) -> List[CodeIssue]:
        """
        Perform deterministic checks to catch common quality issues if LLM fails.

        Returns:
            A list of CodeIssue objects.
        """
        issues: List[CodeIssue] = []
        code_lower = code.lower()
        # Algorithmic: explicit inverses and generic eigensolver use
        if '.inverse(' in code_lower:
            issues.append(CodeIssue(
                'high','algorithmic',
                "Explicit call to '.inverse()' detected.",
                "Replace with factorisation-based solver (e.g. .ldlt().solve(rhs))."
            ))
        if 'eigen::eigensolver' in code_lower and 'selfadjoint' not in code_lower:
            issues.append(CodeIssue(
                'medium','algorithmic',
                "Generic EigenSolver used without symmetry check.",
                "Use Eigen::SelfAdjointEigenSolver for symmetric matrices."
            ))
        if 'eigenvectors' in code_lower and 'col(0)' not in code_lower:
            issues.append(CodeIssue(
                'medium','algorithmic',
                "Eigenvector selection may be incorrect (missing .col(0)).",
                "Select the eigenvector corresponding to the smallest eigenvalue."
            ))
        # Performance: raw pointers and missing solver checks
        if 'new ' in code_lower or 'delete ' in code_lower:
            issues.append(CodeIssue(
                'medium','performance',
                "Raw pointer allocation detected.",
                "Use smart pointers or containers instead."
            ))
        if 'info(' not in code_lower and ('.solve(' in code_lower or 'ldlt' in code_lower or 'llt' in code_lower):
            issues.append(CodeIssue(
                'low','performance',
                "Missing solver success check.",
                "Call solver.info() and verify success after solving."
            ))
        # Error handling: no try/catch, C-style memory
        if 'try' not in code_lower and 'catch' not in code_lower:
            issues.append(CodeIssue(
                'low','error_handling',
                "No exception handling detected.",
                "Wrap critical operations in try/catch blocks and handle exceptions."
            ))
        if 'malloc(' in code_lower or 'free(' in code_lower:
            issues.append(CodeIssue(
                'high','error_handling',
                "C-style memory management detected.",
                "Use RAII objects (std::vector, smart pointers) instead of malloc/free."
            ))
        # Style: long lines, trailing whitespace, using namespace std
        lines = code.splitlines()
        if any(len(line) > 120 for line in lines):
            issues.append(CodeIssue(
                'low','style',
                "Lines exceed 120 characters.",
                "Break long statements into multiple lines."
            ))
        if any(line.rstrip() != line for line in lines):
            issues.append(CodeIssue(
                'low','style',
                "Trailing whitespace detected.",
                "Remove trailing spaces at the ends of lines."
            ))
        if 'using namespace std' in code_lower:
            issues.append(CodeIssue(
                'low','style',
                "'using namespace std' detected.",
                "Remove it and qualify names with std::."
            ))
        # Style: non-const references
        non_const_refs = 0
        for line in lines:
            if ('(' in line and ')' in line and ('void ' in line or 'auto ' in line or 'double ' in line or 'int ' in line or 'Eigen::' in line)):
                params = line.split('(')[1].split(')')[0]
                if '&' in params and 'const' not in params:
                    non_const_refs += 1
        if non_const_refs > 0:
            issues.append(CodeIssue(
                'low','style',
                f"{non_const_refs} parameter(s) passed by non-const reference.",
                "Mark reference parameters as const if they are not modified."
            ))
        # Maintainability: number of functions, file length
        func_count = sum(
            1 for line in lines
            if (('void ' in line or 'auto ' in line or 'double ' in line or 'int ' in line) and '(' in line and ')' in line)
            or ('::' in line and '(' in line and ')' in line)
        )
        if func_count <= 1:
            issues.append(CodeIssue(
                'medium','maintainability',
                "The code defines very few functions/classes.",
                "Split functionality into multiple functions or classes."
            ))
        if len(lines) > 500:
            issues.append(CodeIssue(
                'medium','maintainability',
                "The file is very long (>500 lines).",
                "Split it into smaller modules or files."
            ))
        return issues

    # ----------------------------------------------------------------------
    # Metrics and summary computation
    # ----------------------------------------------------------------------
    def _compute_metrics(self, issues: List[CodeIssue]) -> Dict[str, float]:
        """
        Compute scores out of 10 for each category based on issue severity.
        """
        cats = ['algorithmic','performance','error_handling','style','maintainability','security']
        scores = {c: 10.0 for c in cats}
        for issue in issues:
            penalty = self.penalty.get(issue.severity, 0.5)
            if issue.category in scores:
                scores[issue.category] = max(0.0, scores[issue.category] - penalty)
        return scores

    def _build_summary(self, metrics: Dict[str, float], issues: List[CodeIssue]) -> str:
        """
        Build a concise summary string.
        """
        summary = []
        summary.append(f"Detected {len(issues)} issue(s)." if issues else "No issues detected.")
        summary.append("Category scores:")
        for cat in ['algorithmic','performance','error_handling','style','maintainability','security']:
            summary.append(f"  - {cat.replace('_',' ').title()}: {metrics[cat]:.1f}/10")
        return "\n".join(summary)
    
    def _assess_algorithmic_fidelity(self, cpp_code: str, matlab_code: str, 
                                   conversion_plan: Dict[str, Any], conversion_mode: str = "result-focused") -> List[CodeIssue]:
        """
        Assess algorithmic fidelity between MATLAB and C++ code.
        This is a general-purpose assessment that looks for common patterns.
        """
        issues: List[CodeIssue] = []
        
        # Check for loop structure preservation
        matlab_loops = self._count_nested_loops(matlab_code)
        cpp_loops = self._count_nested_loops(cpp_code)
        
        if matlab_loops > 0 and cpp_loops == 0:
            issues.append(CodeIssue(
                severity="high",
                category="algorithmic",
                description="MATLAB code contains loops but C++ implementation has no loops",
                suggestion="Implement the loop structure from MATLAB code to preserve algorithmic flow",
                confidence=0.9
            ))
        elif matlab_loops != cpp_loops:
            issues.append(CodeIssue(
                severity="medium",
                category="algorithmic", 
                description=f"Loop count mismatch: MATLAB has {matlab_loops} loops, C++ has {cpp_loops}",
                suggestion="Verify that all MATLAB loops are properly translated to C++",
                confidence=0.7
            ))
        
        # Check for matrix operations preservation
        matlab_matrix_ops = self._count_matrix_operations(matlab_code)
        cpp_matrix_ops = self._count_matrix_operations(cpp_code)
        
        if matlab_matrix_ops > 0 and cpp_matrix_ops == 0:
            issues.append(CodeIssue(
                severity="high",
                category="algorithmic",
                description="MATLAB code contains matrix operations but C++ implementation has none",
                suggestion="Implement matrix operations using Eigen or similar library",
                confidence=0.9
            ))
        
        # Check for function call preservation
        matlab_functions = self._extract_function_calls(matlab_code)
        cpp_functions = self._extract_function_calls(cpp_code)
        
        # Look for missing critical MATLAB functions in C++
        missing_functions = []
        for func in matlab_functions:
            if func not in cpp_functions and func not in ['zeros', 'ones', 'eye', 'size', 'length']:
                missing_functions.append(func)
        
        if missing_functions:
            issues.append(CodeIssue(
                severity="medium",
                category="algorithmic",
                description=f"Missing MATLAB function calls in C++: {', '.join(missing_functions)}",
                suggestion="Ensure all MATLAB function calls are properly translated to C++ equivalents",
                confidence=0.8
            ))
        
        # Check for added operations not present in MATLAB (only in faithful mode)
        if conversion_mode == "faithful":
            cpp_specific_ops = ['fft', 'ifft', 'dct', 'idct', 'imread', 'imwrite']
            added_operations = []
            for op in cpp_specific_ops:
                if op in cpp_code.lower() and op not in matlab_code.lower():
                    added_operations.append(op)
            
            if added_operations:
                issues.append(CodeIssue(
                    severity="medium",
                    category="algorithmic",
                    description=f"C++ code contains operations not present in MATLAB: {', '.join(added_operations)}",
                    suggestion="Remove operations that are not in the original MATLAB code to maintain algorithmic fidelity",
                    confidence=0.8
                ))
        
        return issues
    
    def _count_nested_loops(self, code: str) -> int:
        """Count nested loop structures in code."""
        # Count for/while loops
        for_loops = len(re.findall(r'\bfor\b', code, re.IGNORECASE))
        while_loops = len(re.findall(r'\bwhile\b', code, re.IGNORECASE))
        return for_loops + while_loops
    
    def _count_matrix_operations(self, code: str) -> int:
        """Count matrix operations in code."""
        matrix_ops = ['*', '.*', './', '\\', 'inv', 'eig', 'svd', 'qr', 'chol', 'pinv']
        count = 0
        for op in matrix_ops:
            count += len(re.findall(re.escape(op), code))
        return count
    
    def _extract_function_calls(self, code: str) -> List[str]:
        """Extract function calls from code."""
        # Simple pattern to find function calls
        pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        matches = re.findall(pattern, code)
        return list(set(matches))  # Remove duplicates