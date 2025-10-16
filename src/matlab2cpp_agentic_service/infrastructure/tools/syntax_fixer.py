"""
Robust C++ Syntax Fixer

Comprehensive post-processing to fix common LLM-generated syntax errors:
- Missing '); ' in function declarations
- Wrong include paths and header names
- Namespace closing braces
- Brace matching
- Missing includes
- Type corrections
"""

import re
from typing import List, Tuple, Set
from loguru import logger


class RobustCppSyntaxFixer:
    """
    Comprehensive C++ syntax validation and auto-fixing.
    Catches and fixes all common LLM generation errors.
    """
    
    def __init__(self):
        self.logger = logger.bind(name="syntax_fixer")
        self.fixes_applied: List[str] = []
    
    def fix_all_syntax_issues(self, header_content: str, 
                              implementation_content: str,
                              filename: str) -> Tuple[str, str]:
        """
        Apply all syntax fixes to header and implementation.
        
        Args:
            header_content: Header file content
            implementation_content: Implementation file content
            filename: Base filename (e.g., 'e1' or 'e1.m')
            
        Returns:
            (fixed_header, fixed_implementation)
        """
        self.fixes_applied = []
        base_name = filename.replace('.m', '').replace('.h', '').replace('.cpp', '')
        
        # Fix header
        header_fixed = self._fix_header_syntax(header_content, base_name)
        
        # Fix implementation
        impl_fixed = self._fix_implementation_syntax(implementation_content, base_name)
        
        # Log all fixes
        if self.fixes_applied:
            self.logger.info(f"🔧 Applied {len(self.fixes_applied)} syntax fixes to {filename}")
            for fix in self.fixes_applied:
                self.logger.debug(f"  - {fix}")
        
        return header_fixed, impl_fixed
    
    # ==================== Header Fixes ====================
    
    def _fix_header_syntax(self, content: str, filename: str) -> str:
        """Comprehensive header syntax fixes."""
        if not content or not content.strip():
            return content
        
        # 1. Fix include statements first
        content = self._fix_include_statements(content, filename)
        
        # 2. PRIORITY 2: Add missing includes comprehensively
        content = self._add_missing_includes_comprehensive(content)
        
        # 3. Fix function declarations (CRITICAL for syntax)
        content = self._fix_function_declarations(content)
        
        # 4. Fix namespace syntax
        content = self._fix_namespace_syntax(content)
        
        # 5. Fix header guards
        content = self._fix_header_guards(content, filename)
        
        # 6. Fix brace matching
        content = self._fix_brace_matching(content, is_header=True)
        
        return content
    
    # ==================== Implementation Fixes ====================
    
    def _fix_implementation_syntax(self, content: str, filename: str) -> str:
        """Comprehensive implementation syntax fixes."""
        if not content or not content.strip():
            return content
        
        # 1. Fix include statements
        content = self._fix_include_statements(content, filename)
        
        # 2. Ensure proper header include
        content = self._ensure_header_include(content, filename)
        
        # 3. PRIORITY 2: Add missing includes comprehensively
        content = self._add_missing_includes_comprehensive(content)
        
        # 4. PRIORITY 2: Fix Eigen API mistakes
        content = self._fix_eigen_api_mistakes(content)
        
        # 5. PRIORITY 2: Fix const correctness
        content = self._fix_const_correctness(content)
        
        # 6. PRIORITY 2: Detect undeclared variables
        content = self._detect_undeclared_variables(content)
        
        # 7. Fix namespace syntax
        content = self._fix_namespace_syntax(content)
        
        # 8. Fix type issues
        content = self._fix_type_issues(content)
        
        # 9. Add missing includes based on usage
        content = self._add_missing_includes(content)
        
        # 10. Fix brace matching
        content = self._fix_brace_matching(content, is_header=False)
        
        return content
    
    # ==================== Specific Fix Methods ====================
    
    def _fix_function_declarations(self, content: str) -> str:
        """
        Fix function declaration syntax errors.
        Handles:
        - Missing '); ' in multi-line declarations
        - Incomplete signatures
        """
        lines = content.split('\n')
        fixed_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Detect function declaration start
            if self._is_function_declaration_start(stripped) and not stripped.endswith(');'):
                # Collect full function signature
                func_start = i
                j = i + 1
                paren_depth = stripped.count('(') - stripped.count(')')
                
                while j < len(lines) and paren_depth > 0:
                    next_line = lines[j]
                    paren_depth += next_line.count('(') - next_line.count(')')
                    j += 1
                
                # Check if signature complete
                signature_lines = lines[i:j]
                last_line = signature_lines[-1] if signature_lines else ''
                
                # If doesn't end with ); add it
                if last_line.strip() and not last_line.strip().endswith((');', '{', '}')):
                    # Check if next line suggests end of declaration
                    if j < len(lines):
                        next_stripped = lines[j].strip()
                        if (next_stripped.startswith(('//','/*', 'int ', 'void ', 
                                                      'std::', 'Eigen::', '}', 
                                                      'namespace', 'class ', 'struct '))):
                            # This was end of declaration, add );
                            signature_lines[-1] = signature_lines[-1].rstrip() + ');'
                            self.fixes_applied.append(f"Added missing '); ' to function declaration at line {func_start+1}")
                
                fixed_lines.extend(signature_lines)
                i = j
            else:
                fixed_lines.append(line)
                i += 1
        
        return '\n'.join(fixed_lines)
    
    def _is_function_declaration_start(self, line: str) -> bool:
        """Check if line starts a function declaration."""
        # Skip comments, preprocessor, namespace
        if line.startswith(('//', '/*', '#', 'namespace', 'using ', '}', '{')):
            return False
        
        # Pattern: return_type function_name(
        # Examples:
        #   void func(
        #   std::vector<int> func(
        #   Eigen::MatrixXd func(
        if '(' in line and not line.strip().startswith('if') and not line.strip().startswith('for'):
            # Check if it looks like a function (not a call)
            # Has type before function name
            parts = line.split('(')[0].strip().split()
            if len(parts) >= 2:  # At least: type name
                return True
        
        return False
    
    def _fix_include_statements(self, content: str, filename: str) -> str:
        """
        Fix include statement issues.
        """
        # Fix Eigen/Tensor path
        if '#include <Eigen/Tensor>' in content:
            content = content.replace(
                '#include <Eigen/Tensor>',
                '#include <unsupported/Eigen/CXX11/Tensor>'
            )
            self.fixes_applied.append("Fixed Eigen/Tensor include path")
        
        # Fix header name case issues (snake_case → camelCase)
        case_fixes = {
            'get_boundary_distance': 'getBoundaryDistance',
            'max_distance_point': 'maxDistancePoint',
            'skeleton_vessel': 'skeleton_vessel',  # Already correct
        }
        
        for wrong, correct in case_fixes.items():
            wrong_inc = f'#include "{wrong}.h"'
            correct_inc = f'#include "{correct}.h"'
            if wrong_inc in content and wrong != correct:
                content = content.replace(wrong_inc, correct_inc)
                self.fixes_applied.append(f"Fixed header name: {wrong}.h → {correct}.h")
        
        # Remove duplicate includes
        lines = content.split('\n')
        seen_includes: Set[str] = set()
        fixed_lines = []
        
        for line in lines:
            if line.strip().startswith('#include'):
                if line not in seen_includes:
                    seen_includes.add(line)
                    fixed_lines.append(line)
                # Skip duplicates
            else:
                fixed_lines.append(line)
        
        if len(fixed_lines) < len(lines):
            self.fixes_applied.append(f"Removed {len(lines) - len(fixed_lines)} duplicate include(s)")
        
        return '\n'.join(fixed_lines)
    
    def _ensure_header_include(self, content: str, filename: str) -> str:
        """Ensure implementation includes its own header first."""
        header_inc = f'#include "{filename}.h"'
        
        if header_inc not in content:
            lines = content.split('\n')
            # Insert after any system includes
            insert_idx = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('#include <'):
                    insert_idx = i + 1
                elif line.strip() and not line.strip().startswith('#'):
                    break
            
            lines.insert(insert_idx, header_inc)
            self.fixes_applied.append(f"Added header include: {header_inc}")
            content = '\n'.join(lines)
        
        return content
    
    def _fix_header_guards(self, content: str, filename: str) -> str:
        """Fix header guard issues."""
        lines = content.split('\n')
        
        # Check if guards exist
        has_ifndef = any('#ifndef' in l for l in lines)
        has_define = any('#define' in l for l in lines)
        has_endif = any('#endif' in l for l in lines)
        
        if not (has_ifndef and has_define):
            # Add guards at top
            guard_name = f"{filename.upper().replace('.', '_')}_H"
            
            # Find first non-comment line
            insert_idx = 0
            for i, line in enumerate(lines):
                if line.strip() and not line.strip().startswith('//'):
                    insert_idx = i
                    break
            
            lines.insert(insert_idx, f'#ifndef {guard_name}')
            lines.insert(insert_idx + 1, f'#define {guard_name}')
            lines.insert(insert_idx + 2, '')
            
            if not has_endif:
                lines.append('')
                lines.append(f'#endif // {guard_name}')
            
            self.fixes_applied.append(f"Added header guards: {guard_name}")
            content = '\n'.join(lines)
        
        return content
    
    def _fix_namespace_syntax(self, content: str) -> str:
        """
        Fix namespace syntax issues.
        Ensures all namespace { have matching }.
        """
        # Find all namespace declarations
        namespace_pattern = r'namespace\s+(\w+)\s*\{'
        namespaces = re.findall(namespace_pattern, content)
        
        if not namespaces:
            return content
        
        lines = content.split('\n')
        
        # Find #endif position
        endif_idx = -1
        for i, line in enumerate(lines):
            if '#endif' in line:
                endif_idx = i
                break
        
        if endif_idx > 0:
            # Count braces before #endif
            before_endif = '\n'.join(lines[:endif_idx])
            open_count = before_endif.count('{')
            close_count = before_endif.count('}')
            
            if open_count > close_count:
                # Add missing closing braces
                missing = open_count - close_count
                closing_lines = []
                for namespace in reversed(namespaces[-missing:]):
                    closing_lines.append(f'}} // namespace {namespace}')
                
                closing_lines.append('')  # Empty line before #endif
                
                lines[endif_idx:endif_idx] = closing_lines
                self.fixes_applied.append(f"Added {missing} missing namespace closing brace(s)")
                content = '\n'.join(lines)
        
        return content
    
    def _fix_brace_matching(self, content: str, is_header: bool = False) -> str:
        """
        Ensure all braces match.
        More lenient for implementation files (may have incomplete functions).
        """
        open_count = content.count('{')
        close_count = content.count('}')
        
        if open_count == close_count:
            return content  # Already balanced
        
        diff = abs(open_count - close_count)
        
        if open_count > close_count:
            # Missing closing braces
            self.logger.warning(f"⚠️  {diff} unclosed brace(s) - attempting fix")
            
            if is_header:
                # For headers, add before #endif
                lines = content.split('\n')
                endif_idx = next((i for i, l in enumerate(lines) if '#endif' in l), len(lines))
                
                for _ in range(diff):
                    lines.insert(endif_idx, '}')
                    endif_idx += 1
                
                content = '\n'.join(lines)
                self.fixes_applied.append(f"Added {diff} missing closing brace(s)")
        elif close_count > open_count:
            # Extra closing braces - just log warning
            self.logger.warning(f"⚠️  {diff} extra closing brace(s) - manual review needed")
        
        return content
    
    def _fix_type_issues(self, content: str) -> str:
        """Fix common type errors."""
        fixes_made = []
        
        # Fix ArrayXXb (doesn't exist in Eigen)
        if 'Eigen::ArrayXXb' in content:
            content = content.replace(
                'Eigen::ArrayXXb',
                'Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>'
            )
            fixes_made.append("Fixed Eigen::ArrayXXb → Array<bool, Dynamic, Dynamic>")
        
        # Fix ArrayXXXb (for 3D)
        if 'Eigen::ArrayXXXb' in content:
            content = content.replace(
                'Eigen::ArrayXXXb',
                'Eigen::Tensor<bool, 3>'
            )
            fixes_made.append("Fixed Eigen::ArrayXXXb → Tensor<bool, 3>")
        
        # Add missing Eigen:: prefix for common types
        type_fixes = [
            (r'\bMatrixXd\b', 'Eigen::MatrixXd'),
            (r'\bVectorXd\b', 'Eigen::VectorXd'),
            (r'\bArrayXd\b', 'Eigen::ArrayXd'),
            (r'\bArrayXXd\b', 'Eigen::ArrayXXd'),
        ]
        
        for pattern, replacement in type_fixes:
            # Only replace if not already prefixed
            old_content = content
            content = re.sub(f'(?<!Eigen::){pattern}', replacement, content)
            if content != old_content:
                fixes_made.append(f"Added Eigen:: prefix to types")
                break  # Only log once
        
        self.fixes_applied.extend(fixes_made)
        return content
    
    def _add_missing_includes(self, content: str) -> str:
        """Add missing includes based on code usage."""
        lines = content.split('\n')
        
        # Find end of include section
        include_section_end = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('#include'):
                include_section_end = i + 1
            elif line.strip() and not line.strip().startswith(('#', '//')):
                break
        
        # Check what includes are needed
        needed_includes = []
        
        # Math functions
        if any(func in content for func in ['std::sqrt', 'std::pow', 'std::floor', 
                                             'std::ceil', 'std::abs', 'std::sin', 
                                             'std::cos']):
            if '#include <cmath>' not in content:
                needed_includes.append('#include <cmath>')
        
        # Vectors
        if 'std::vector' in content:
            if '#include <vector>' not in content:
                needed_includes.append('#include <vector>')
        
        # Algorithms
        if any(func in content for func in ['std::min', 'std::max', 'std::sort',
                                             'std::find', 'std::fill']):
            if '#include <algorithm>' not in content:
                needed_includes.append('#include <algorithm>')
        
        # Exceptions
        if any(exc in content for exc in ['std::invalid_argument', 'std::runtime_error',
                                          'std::logic_error', 'std::out_of_range']):
            if '#include <stdexcept>' not in content:
                needed_includes.append('#include <stdexcept>')
        
        # Memory
        if 'std::shared_ptr' in content or 'std::unique_ptr' in content:
            if '#include <memory>' not in content:
                needed_includes.append('#include <memory>')
        
        # Eigen - check for Eigen types
        if any(t in content for t in ['Eigen::MatrixXd', 'Eigen::VectorXd', 'Eigen::ArrayXd',
                                       'Eigen::MatrixXf', 'Eigen::VectorXf', 'Eigen::ArrayXXd']):
            if '#include <Eigen/Dense>' not in content:
                needed_includes.insert(0, '#include <Eigen/Dense>')  # Add first
        
        # Eigen Tensor
        if 'Eigen::Tensor' in content:
            if '#include <unsupported/Eigen/CXX11/Tensor>' not in content:
                needed_includes.insert(0, '#include <unsupported/Eigen/CXX11/Tensor>')
        
        # Insert needed includes
        for inc in needed_includes:
            lines.insert(include_section_end, inc)
            include_section_end += 1
            self.fixes_applied.append(f"Added missing {inc}")
        
        return '\n'.join(lines)
    
    def _fix_function_declarations(self, content: str) -> str:
        """
        Fix missing '); ' in function declarations.
        
        Simpler, more reliable approach:
        - Look for lines that end a parameter (contain types/identifiers)
        - Check if next line is blank/comment/new function/namespace closing
        - If current line doesn't end with ); and looks like last param, add );
        """
        lines = content.split('\n')
        fixed_lines = []
        
        for i in range(len(lines)):
            line = lines[i]
            stripped = line.strip()
            
            # Skip if already ends with proper syntax
            if stripped.endswith((');', '{', '}', ';', ',')):
                fixed_lines.append(line)
                continue
            
            # Check if next line exists
            if i + 1 >= len(lines):
                fixed_lines.append(line)
                continue
            
            next_line = lines[i + 1].strip()
            
            # Detect if this is likely the last parameter of a function declaration
            # Indicators:
            # 1. Current line has type keywords (const, &, *, Eigen::, std::, int, double, etc.)
            # 2. Next line is: blank, comment, new function, or namespace closing
            
            has_type_keywords = any(keyword in line for keyword in [
                'const ', ' &', '& ', ' *', '* ',
                'Eigen::', 'std::', 'int ', 'double ', 'float ', 'bool ',
                'void ', 'char ', 'size_t', 'uint'
            ])
            
            next_is_end_marker = (
                next_line == '' or  # Blank line
                next_line.startswith(('//','/*', '*/', '}')) or  # Comment or closing
                next_line.startswith(('int ', 'void ', 'bool ', 'double ', 'float ',
                                     'std::', 'Eigen::', 'template', 'class ', 'struct ',
                                     'using ', '#endif'))  # New declaration
            )
            
            # Additional check: line should be indented (inside function)
            is_indented = len(line) - len(line.lstrip()) >= 4
            
            # If all conditions met, this is likely incomplete function declaration
            if has_type_keywords and next_is_end_marker and is_indented:
                # Add );
                fixed_lines.append(line.rstrip() + ');')
                self.fixes_applied.append(f"Added missing '); ' to function declaration")
                self.logger.debug(f"  Fixed line {i+1}: {stripped[:60]}...")
            else:
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _fix_header_guards(self, content: str, filename: str) -> str:
        """Ensure proper header guard structure."""
        lines = content.split('\n')
        
        has_ifndef = any('#ifndef' in l for l in lines)
        has_define = any('#define' in l for l in lines)
        has_endif = any('#endif' in l for l in lines)
        
        if has_ifndef and has_define and has_endif:
            return content  # Guards already present
        
        # Need to add guards
        guard_name = f"{filename.upper().replace('.', '_')}_H"
        
        # Find first content line (skip existing includes)
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith(('//','/*', '#include')):
                insert_idx = i
                break
        
        # Insert guards
        if not has_ifndef:
            lines.insert(insert_idx, f'#ifndef {guard_name}')
            lines.insert(insert_idx + 1, f'#define {guard_name}')
            lines.insert(insert_idx + 2, '')
        
        if not has_endif:
            lines.append('')
            lines.append(f'#endif // {guard_name}')
        
        self.fixes_applied.append(f"Ensured header guards: {guard_name}")
        return '\n'.join(lines)
    
    def _fix_namespace_syntax(self, content: str) -> str:
        """Fix namespace closing brace issues."""
        namespace_pattern = r'namespace\s+(\w+)\s*\{'
        namespaces = re.findall(namespace_pattern, content)
        
        if not namespaces:
            return content
        
        lines = content.split('\n')
        endif_idx = next((i for i, l in enumerate(lines) if '#endif' in l), -1)
        
        if endif_idx > 0:
            before_endif = '\n'.join(lines[:endif_idx])
            open_count = before_endif.count('{')
            close_count = before_endif.count('}')
            
            if open_count > close_count:
                missing = open_count - close_count
                closing_lines = []
                for namespace in reversed(namespaces[-missing:]):
                    closing_lines.append(f'}} // namespace {namespace}')
                closing_lines.append('')
                
                lines[endif_idx:endif_idx] = closing_lines
                self.fixes_applied.append(f"Added {missing} namespace closing brace(s)")
                content = '\n'.join(lines)
        
        return content
    
    def _fix_brace_matching(self, content: str, is_header: bool = False) -> str:
        """Ensure brace balance."""
        open_count = content.count('{')
        close_count = content.count('}')
        
        if open_count == close_count:
            return content
        
        diff = abs(open_count - close_count)
        
        if open_count > close_count:
            # Add missing closing braces before #endif
            lines = content.split('\n')
            endif_idx = next((i for i, l in enumerate(lines) if '#endif' in l), len(lines))
            
            for _ in range(diff):
                lines.insert(endif_idx, '}')
            
            self.fixes_applied.append(f"Added {diff} missing closing brace(s)")
            content = '\n'.join(lines)
        
        return content
    
    def _fix_type_issues(self, content: str) -> str:
        """Fix common type errors."""
        if 'Eigen::ArrayXXb' in content:
            content = content.replace('Eigen::ArrayXXb', 
                                     'Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>')
            self.fixes_applied.append("Fixed ArrayXXb type")
        
        if 'Eigen::ArrayXXXb' in content:
            content = content.replace('Eigen::ArrayXXXb', 'Eigen::Tensor<bool, 3>')
            self.fixes_applied.append("Fixed ArrayXXXb type")
        
        return content
    
    # ==================== PRIORITY 2 ENHANCEMENTS ====================
    
    def _add_missing_includes_comprehensive(self, content: str) -> str:
        """
        PRIORITY 2 FIX: Comprehensive missing include detection.
        Scans for used types and auto-adds required includes.
        """
        lines = content.split('\n')
        include_section_end = 0
        
        # Find where includes should be inserted (after header guards, before namespace)
        for i, line in enumerate(lines):
            if line.strip().startswith('#include'):
                include_section_end = i + 1
            elif line.strip().startswith('namespace') or line.strip().startswith('//'):
                break
        
        if include_section_end == 0:
            # No includes yet, insert after header guard
            for i, line in enumerate(lines):
                if line.strip().startswith('#define'):
                    include_section_end = i + 1
                    break
        
        # Check for missing includes
        missing_includes = []
        
        # Check for Eigen types
        if re.search(r'\bEigen::(MatrixXd|VectorXd|ArrayXXd|ArrayXd)', content):
            if not re.search(r'#include\s+[<"]Eigen/Dense[>"]', content):
                missing_includes.append('#include <Eigen/Dense>')
        
        # Check for Eigen::Tensor
        if re.search(r'\bEigen::Tensor\s*<', content):
            if not re.search(r'#include\s+[<"]unsupported/Eigen/CXX11/Tensor[>"]', content):
                missing_includes.append('#include <unsupported/Eigen/CXX11/Tensor>')
        
        # Check for std::vector
        if re.search(r'\bstd::vector\s*<', content):
            if not re.search(r'#include\s+[<"]vector[>"]', content):
                missing_includes.append('#include <vector>')
        
        # Check for std::pair
        if re.search(r'\bstd::pair\s*<', content):
            if not re.search(r'#include\s+[<"]utility[>"]', content):
                missing_includes.append('#include <utility>')
        
        # Check for std::string
        if re.search(r'\bstd::string\b', content):
            if not re.search(r'#include\s+[<"]string[>"]', content):
                missing_includes.append('#include <string>')
        
        # Check for std::cout, std::endl
        if re.search(r'\bstd::(cout|cerr|endl)\b', content):
            if not re.search(r'#include\s+[<"]iostream[>"]', content):
                missing_includes.append('#include <iostream>')
        
        # Check for std::max, std::min
        if re.search(r'\bstd::(max|min)\s*\(', content):
            if not re.search(r'#include\s+[<"]algorithm[>"]', content):
                missing_includes.append('#include <algorithm>')
        
        # Check for std::sqrt, std::pow
        if re.search(r'\bstd::(sqrt|pow|abs|sin|cos|tan)\s*\(', content):
            if not re.search(r'#include\s+[<"]cmath[>"]', content):
                missing_includes.append('#include <cmath>')
        
        # Check for std::runtime_error, std::exception
        if re.search(r'\bstd::(runtime_error|exception|invalid_argument)\b', content):
            if not re.search(r'#include\s+[<"]stdexcept[>"]', content):
                missing_includes.append('#include <stdexcept>')
        
        # Check for std::shared_ptr, std::unique_ptr
        if re.search(r'\bstd::(shared_ptr|unique_ptr|make_shared|make_unique)\b', content):
            if not re.search(r'#include\s+[<"]memory[>"]', content):
                missing_includes.append('#include <memory>')
        
        # Insert missing includes
        if missing_includes:
            # Insert after last existing include or at include_section_end
            for include in missing_includes:
                lines.insert(include_section_end, include)
                include_section_end += 1
                self.fixes_applied.append(f"Added missing include: {include}")
            
            # Add blank line after includes
            if lines[include_section_end].strip():
                lines.insert(include_section_end, '')
        
        return '\n'.join(lines)
    
    def _fix_eigen_api_mistakes(self, content: str) -> str:
        """
        PRIORITY 2 FIX: Fix common Eigen API mistakes.
        - .slice() doesn't exist → suggest block() or segment()
        - Zero(a,b,c) should be Zero(a,b) for 2D (DISABLED - causes issues)
        - dimension() only for Tensor, not MatrixXd
        """
        # Fix .slice() calls (doesn't exist in Eigen)
        if re.search(r'\.slice\s*\(', content):
            # Replace with .block() for 2D matrices
            # More conservative pattern: only fix clear cases
            content = re.sub(r'(\w+)\.slice\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)', 
                           r'\1.segment(\2, \3)',  # For 1D vectors
                           content)
            self.fixes_applied.append("Fixed .slice() → .segment() (Eigen API correction)")
        
        # NOTE: Zero(a,b,c) fix DISABLED - too aggressive, breaks valid code
        # The LLM should handle this with improved prompts
        
        # Fix .dimension() on MatrixXd (should be .rows()/.cols())
        # NOTE: Also disabled - needs context to know which dimension
        
        return content
    
    def _fix_const_correctness(self, content: str) -> str:
        """
        PRIORITY 2 FIX: Fix const correctness issues.
        - Assignment to const variables
        - const in function parameters when modification is needed
        """
        lines = content.split('\n')
        
        # Fix const variables
        for i, line in enumerate(lines):
            # Detect const variable declarations
            match = re.match(r'\s*const\s+\w+\s+(\w+)\s*=', line)
            if match:
                var_name = match.group(1)
                # Check if this variable is later modified
                for j in range(i+1, len(lines)):
                    if re.search(rf'\b{var_name}\s*[+\-*/]=', lines[j]) or \
                       re.search(rf'\b{var_name}\s*=\s*[^=]', lines[j]):
                        # Variable is modified, remove const
                        lines[i] = lines[i].replace('const ', '', 1)
                        self.fixes_applied.append(f"Removed 'const' from variable '{var_name}' (modified later)")
                        break
        
        # FIX #2: Fix const in function parameters
        content_str = '\n'.join(lines)
        content_str = self._fix_const_function_parameters(content_str)
        
        return content_str
    
    def _fix_const_function_parameters(self, content: str) -> str:
        """
        FIX #2: Remove const from function parameters that are modified in function body.
        """
        lines = content.split('\n')
        
        # Find function definitions
        for i, line in enumerate(lines):
            # Match function signatures with const& parameters
            # Pattern: type func(..., const Type& param, ...)
            func_match = re.match(r'^\s*([\w:]+)\s+([\w]+)\s*\(', line)
            if func_match and '{' not in line[:func_match.end()]:
                # Found potential function start
                # Collect full signature
                sig_start = i
                sig_end = i
                open_parens = line.count('(') - line.count(')')
                
                # Find closing )
                while open_parens > 0 and sig_end < len(lines) - 1:
                    sig_end += 1
                    open_parens += lines[sig_end].count('(') - lines[sig_end].count(')')
                
                # Find function body end
                if sig_end < len(lines) - 1:
                    body_start = sig_end + 1
                    # Skip to opening {
                    while body_start < len(lines) and '{' not in lines[body_start]:
                        body_start += 1
                    
                    if body_start < len(lines):
                        # Find matching }
                        brace_count = 0
                        body_end = body_start
                        for j in range(body_start, len(lines)):
                            brace_count += lines[j].count('{') - lines[j].count('}')
                            if brace_count == 0 and '{' in lines[body_start]:
                                body_end = j
                                break
                        
                        # Extract signature and body
                        signature = '\n'.join(lines[sig_start:sig_end+1])
                        body = '\n'.join(lines[body_start:body_end+1])
                        
                        # Find const& parameters
                        const_param_pattern = r'const\s+([\w:]+)\s*&\s*(\w+)'
                        const_params = re.findall(const_param_pattern, signature)
                        
                        for param_type, param_name in const_params:
                            # Check if parameter is modified in body
                            if re.search(rf'\b{param_name}\s*=\s*[^=]', body):
                                # Remove const from this parameter
                                old_param = f"const {param_type}& {param_name}"
                                new_param = f"{param_type}& {param_name}"
                                content = content.replace(old_param, new_param, 1)
                                self.fixes_applied.append(f"Removed 'const' from parameter '{param_name}' (modified in function)")
        
        return content
    
    def _detect_undeclared_variables(self, content: str) -> str:
        """
        PRIORITY 2 FIX: Detect commonly undeclared variables and add declarations.
        Common patterns: check, stepsize, etc.
        """
        # Common undeclared variables in MATLAB→C++ conversion
        common_undeclared = {
            'check': 'bool check = false;',
            'stepsize': 'int stepsize = 1;',
        }
        
        lines = content.split('\n')
        
        for var_name, declaration in common_undeclared.items():
            # Check if variable is used but not declared
            if re.search(rf'\b{var_name}\b', content):
                # Check if it's declared
                if not re.search(rf'\b(int|bool|double|float|auto)\s+{var_name}\s*[=;]', content):
                    # Find first usage and insert declaration before it
                    for i, line in enumerate(lines):
                        if re.search(rf'\b{var_name}\b', line) and not line.strip().startswith('//'):
                            # Insert declaration at function start
                            # Find previous function start
                            for j in range(i-1, -1, -1):
                                if re.search(r'^\s*(void|int|double|bool|std::|Eigen::)', lines[j]):
                                    # Found function, insert after opening brace
                                    for k in range(j, min(j+10, len(lines))):
                                        if '{' in lines[k]:
                                            lines.insert(k+1, f"    {declaration}  // Auto-declared")
                                            self.fixes_applied.append(f"Added declaration for '{var_name}'")
                                            break
                                    break
                            break
        
        return '\n'.join(lines)
    
    # ==================== GENERALIZED SOLUTIONS ====================
    
    def _fix_eigen_api_hallucinations(self, content: str) -> str:
        """
        GENERAL: Fix common Eigen API hallucinations.
        Based on official Eigen API documentation, not project-specific patterns.
        """
        if not content:
            return content
        
        fixes_made = 0
        
        # Fix 1: .hasNonZero() doesn't exist → use (x != 0).any()
        new_content, n = re.subn(
            r'(\w+)\.hasNonZero\(\)',
            r'(\1 != 0).any()',
            content
        )
        if n > 0:
            fixes_made += n
            self.fixes_applied.append(f"Fixed .hasNonZero() → (x != 0).any() ({n} occurrences)")
            content = new_content
        
        # Fix 2: .dimension() on non-Tensor types (MatrixXd/ArrayXXd)
        # Only fix if declared as Matrix/Array (not Tensor)
        for match in re.finditer(r'(\w+)\.dimension\((\d+)\)', content):
            var_name = match.group(1)
            dim_idx = match.group(2)
            
            # Check if this variable is declared as MatrixXd/ArrayXXd (not Tensor)
            if re.search(rf'(Matrix|Array)Xd\s+{var_name}', content) or \
               re.search(rf'(Matrix|Array)XXd\s+{var_name}', content):
                replacement = f'{var_name}.rows()' if dim_idx == '0' else f'{var_name}.cols()'
                content = content.replace(match.group(0), replacement, 1)
                fixes_made += 1
                self.fixes_applied.append(f"Fixed {var_name}.dimension({dim_idx}) → {replacement}")
        
        # Fix 3: .slice() on Matrix/Array → use .block()
        new_content, n = re.subn(
            r'(\w+)\.slice\(',
            r'\1.block(',
            content
        )
        if n > 0:
            fixes_made += n
            self.fixes_applied.append(f"Fixed .slice() → .block() ({n} occurrences)")
            content = new_content
        
        # Fix 4: .flatten() → .reshaped()
        new_content, n = re.subn(
            r'(\w+)\.flatten\(\)',
            r'\1.reshaped()',
            content
        )
        if n > 0:
            fixes_made += n
            self.fixes_applied.append(f"Fixed .flatten() → .reshaped() ({n} occurrences)")
            content = new_content
        
        return content
    
    def _detect_and_fix_undeclared_variables_generic(self, content: str) -> str:
        """
        GENERAL: Detect undeclared variables using generic algorithms.
        Not tied to specific variable names like 'sxm', 'sxp'.
        """
        if not content:
            return content
        
        all_used_vars = set()
        all_declared_vars = set()
        
        lines = content.split('\n')
        
        # Find all variable uses (conservative heuristic)
        for line in lines:
            # Skip comments, preprocessor, declarations
            if line.strip().startswith('//') or line.strip().startswith('#'):
                continue
            if re.match(r'\s*(int|double|float|bool|auto|size_t|const)\s+', line):
                continue
            
            # Find potential variable uses: word followed by operators
            for match in re.finditer(r'\b([a-z_][a-z0-9_]{1,20})\s*[-+*/=<>]', line):
                var = match.group(1)
                if not self._is_cpp_keyword(var):
                    all_used_vars.add(var)
        
        # Find all declarations
        for line in lines:
            # Pattern: Type varName = ... or Type varName;
            match = re.match(r'\s*(int|double|float|bool|auto|size_t|Eigen::\w+)\s+(\w+)\s*[=;]', line)
            if match:
                all_declared_vars.add(match.group(2))
        
        # Find undeclared
        undeclared = all_used_vars - all_declared_vars
        
        # Filter out false positives
        real_undeclared = set()
        for var in undeclared:
            # Skip if it's namespace/class access
            if '::' in var or var.startswith('std') or var.startswith('Eigen'):
                continue
            # Skip very short names (likely false positives)
            if len(var) < 2:
                continue
            # Skip common patterns that aren't variables
            if var in ['return', 'if', 'else', 'for', 'while', 'break', 'continue']:
                continue
            real_undeclared.add(var)
        
        # Declare variables with safe defaults
        if real_undeclared:
            declarations = []
            for var in sorted(real_undeclared):
                # Safe default: double (works for most numeric operations)
                declarations.append(f"    double {var} = 0.0;  // Auto-declared (was undeclared)")
            
            # Insert after first opening brace of first function
            content = self._insert_declarations_in_function(content, declarations)
            self.fixes_applied.append(f"Auto-declared {len(real_undeclared)} undeclared variables")
        
        return content
    
    def _is_cpp_keyword(self, word: str) -> bool:
        """Check if word is a C++ keyword or common identifier."""
        keywords = {
            'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 'continue',
            'return', 'true', 'false', 'nullptr', 'this', 'new', 'delete', 'sizeof',
            'const', 'static', 'inline', 'virtual', 'override', 'final', 'class',
            'struct', 'namespace', 'using', 'template', 'typename', 'public', 'private',
            'int', 'double', 'float', 'bool', 'char', 'void', 'auto', 'size_t',
            'std', 'Eigen', 'cv', 'Matrix', 'Vector', 'Array', 'Tensor', 'Map'
        }
        return word in keywords
    
    def _insert_declarations_in_function(self, content: str, declarations: List[str]) -> str:
        """Insert variable declarations at the start of the first function."""
        lines = content.split('\n')
        
        # Find first function body (first '{' after a function signature)
        for i, line in enumerate(lines):
            if '{' in line and not line.strip().startswith('//'):
                # Check if this is after a function signature
                if i > 0 and ('(' in lines[i-1] or '(' in line):
                    # Insert declarations after this opening brace
                    indent = '    '
                    for declaration in declarations:
                        lines.insert(i+1, declaration)
                    self.fixes_applied.append(f"Inserted {len(declarations)} variable declarations")
                    break
        
        return '\n'.join(lines)
    
    def _fix_const_across_files(self, header_content: str, impl_content: str) -> tuple[str, str]:
        """
        GENERAL: Remove const from parameters that are modified.
        Scans BOTH header and implementation together.
        """
        if not header_content or not impl_content:
            return header_content, impl_content
        
        # Step 1: Extract all const& parameters from header
        const_param_pattern = r'const\s+([\w:]+)\s*&\s*(\w+)'
        const_params_in_header = re.findall(const_param_pattern, header_content)
        
        # Step 2: For each const parameter, check if modified in implementation
        for param_type, param_name in const_params_in_header:
            # Pattern: param = ... (assignment, not comparison)
            if re.search(rf'\b{param_name}\s*=\s*[^=]', impl_content):
                # Remove const from BOTH header and implementation
                old_pattern = f"const {param_type}& {param_name}"
                new_pattern = f"{param_type}& {param_name}"
                
                header_content = header_content.replace(old_pattern, new_pattern)
                impl_content = impl_content.replace(old_pattern, new_pattern)
                
                self.fixes_applied.append(f"Removed const from parameter '{param_name}' (modified in function)")
        
        return header_content, impl_content

