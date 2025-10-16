"""
Error Extraction Module for Iterative LLM Error Fixing

Parses compilation output into structured error objects for targeted fixing.
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Optional
from loguru import logger


@dataclass
class CompilationError:
    """Structured representation of a compilation error."""
    file: str                    # e.g., "e1.cpp"
    line: int                    # Line number
    column: int                  # Column number
    error_type: str              # "syntax" | "type" | "semantic" | "unknown"
    severity: str                # "error" | "warning"
    message: str                 # Error message
    context: Optional[str] = None           # Surrounding code lines
    suggested_fix: Optional[str] = None     # Auto-generated suggestion
    
    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.column}: {self.severity}: {self.message}"


class ErrorExtractor:
    """
    Extract and categorize compilation errors from g++ output.
    
    Usage:
        extractor = ErrorExtractor()
        errors = extractor.parse_compilation_output(gcc_output)
        grouped = extractor.group_errors_by_file(errors)
    """
    
    def __init__(self):
        self.logger = logger.bind(name="error_extractor")
        
        # Error categorization patterns
        self.error_patterns = {
            'syntax': [
                r"error: expected .+ before",
                r"error: expected declaration",
                r"error: expected unqualified-id",
                r"error: expected primary-expression",
                r"error: expected '[\}\)]' ",
                r"error: expected ';' before",
            ],
            'type': [
                r"error: no matching function",
                r"error: invalid initialization",
                r"error: .+ does not name a type",
                r"error: conflicting types",
                r"error: cannot convert",
                r"error: no match for",
                r"error: request for member",
            ],
            'semantic': [
                r"error: .+ was not declared",
                r"error: .+ is not a member",
                r"error: assignment of read-only",
                r"error: .+ has no member named",
            ]
        }
    
    def parse_compilation_output(self, output: str) -> List[CompilationError]:
        """
        Parse g++ compilation output into structured error objects.
        
        Args:
            output: Raw g++ compilation output
            
        Returns:
            List of CompilationError objects
            
        Example Input:
            e1.cpp:38:58: error: expected ')' before ';' token
               38 | Eigen::MatrixXd Gx = ...;
                  |                      ^
        """
        errors = []
        lines = output.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Match: filename:line:col: error/warning: message
            match = re.match(r'([^:]+\.(?:cpp|h|hpp|cc)):(\d+):(\d+):\s+(error|warning):\s+(.+)', line)
            if match:
                file, line_num, col, severity, message = match.groups()
                
                # Extract context (next 3 lines usually show code)
                context_lines = []
                for j in range(i+1, min(i+5, len(lines))):
                    if lines[j].strip():
                        # Stop at next error or non-context line
                        if re.match(r'[^:]+:\d+:\d+:', lines[j]):
                            break
                        context_lines.append(lines[j])
                    else:
                        break
                
                # Categorize error type
                error_type = self._categorize_error(message)
                
                # Generate suggested fix
                suggested_fix = self._generate_fix_suggestion(error_type, message)
                
                errors.append(CompilationError(
                    file=file,
                    line=int(line_num),
                    column=int(col),
                    error_type=error_type,
                    severity=severity,
                    message=message,
                    context='\n'.join(context_lines) if context_lines else None,
                    suggested_fix=suggested_fix
                ))
            
            i += 1
        
        self.logger.info(f"Extracted {len(errors)} compilation errors/warnings")
        return errors
    
    def _categorize_error(self, message: str) -> str:
        """
        Categorize error as syntax/type/semantic.
        
        Args:
            message: Error message string
            
        Returns:
            Category: 'syntax', 'type', 'semantic', or 'unknown'
        """
        for category, patterns in self.error_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    return category
        return 'unknown'
    
    def _generate_fix_suggestion(self, error_type: str, message: str) -> Optional[str]:
        """
        Generate automatic fix suggestions based on error type.
        
        Args:
            error_type: Error category
            message: Error message
            
        Returns:
            Fix suggestion string or None
        """
        suggestions = {
            'syntax': "Check for missing semicolons, braces, or parentheses",
            'type': "Verify types match between declaration and usage. Check function signatures.",
            'semantic': "Ensure variable/function is declared before use. Check namespace and includes.",
        }
        
        base_suggestion = suggestions.get(error_type, "Review the error message for specific guidance")
        
        # Specific suggestions based on message content
        if 'Zero' in message and 'no matching function' in message:
            return "MatrixXd::Zero() takes exactly 2 arguments (rows, cols). For 3D, use Tensor."
        elif 'was not declared' in message:
            return "Add variable declaration before first use. Check if it's a typo."
        elif 'does not name a type' in message:
            return "Add missing #include directive for this type."
        elif '.slice()' in message or 'no member named' in message and 'slice' in message:
            return "Eigen doesn't have .slice(). Use .segment() for vectors or .block() for matrices."
        
        return base_suggestion
    
    def group_errors_by_file(self, errors: List[CompilationError]) -> Dict[str, List[CompilationError]]:
        """
        Group errors by filename for targeted fixing.
        
        Args:
            errors: List of compilation errors
            
        Returns:
            Dict mapping filename to list of errors in that file
        """
        grouped = {}
        for error in errors:
            if error.file not in grouped:
                grouped[error.file] = []
            grouped[error.file].append(error)
        
        # Sort errors within each file by line number
        for file_errors in grouped.values():
            file_errors.sort(key=lambda e: e.line)
        
        return grouped
    
    def prioritize_errors(self, errors: List[CompilationError]) -> List[CompilationError]:
        """
        Sort errors by priority (blocking first, then by category).
        
        Priority order:
        1. Errors before warnings
        2. Syntax errors (most fundamental)
        3. Type errors
        4. Semantic errors
        5. Unknown errors
        
        Args:
            errors: List of compilation errors
            
        Returns:
            Sorted list (highest priority first)
        """
        def priority_score(error: CompilationError) -> int:
            score = 0
            
            # Severity (errors first)
            if error.severity == 'error':
                score += 1000
            else:
                score += 100  # warnings
            
            # Error type (syntax most critical)
            if error.error_type == 'syntax':
                score += 30
            elif error.error_type == 'type':
                score += 20
            elif error.error_type == 'semantic':
                score += 10
            else:
                score += 5  # unknown
            
            return -score  # Negative for descending sort
        
        return sorted(errors, key=priority_score)
    
    def get_error_statistics(self, errors: List[CompilationError]) -> Dict[str, int]:
        """
        Get statistics about error distribution.
        
        Args:
            errors: List of compilation errors
            
        Returns:
            Dict with error counts by category
        """
        stats = {
            'total': len(errors),
            'errors': sum(1 for e in errors if e.severity == 'error'),
            'warnings': sum(1 for e in errors if e.severity == 'warning'),
            'syntax': sum(1 for e in errors if e.error_type == 'syntax'),
            'type': sum(1 for e in errors if e.error_type == 'type'),
            'semantic': sum(1 for e in errors if e.error_type == 'semantic'),
            'unknown': sum(1 for e in errors if e.error_type == 'unknown'),
        }
        
        # Files affected
        files = set(e.file for e in errors)
        stats['files_affected'] = len(files)
        
        return stats
    
    def filter_errors_only(self, errors: List[CompilationError]) -> List[CompilationError]:
        """Filter out warnings, keep only errors."""
        return [e for e in errors if e.severity == 'error']
    
    def get_top_error_files(self, errors: List[CompilationError], top_n: int = 5) -> List[str]:
        """
        Get filenames with the most errors.
        
        Args:
            errors: List of compilation errors
            top_n: Number of top files to return
            
        Returns:
            List of filenames sorted by error count (descending)
        """
        grouped = self.group_errors_by_file(errors)
        sorted_files = sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True)
        return [f[0] for f in sorted_files[:top_n]]




