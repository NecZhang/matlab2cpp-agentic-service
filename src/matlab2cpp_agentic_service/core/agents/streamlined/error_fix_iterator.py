"""
Iterative Error Fixing Module

Implements intelligent error feedback loop with targeted LLM re-generation.
"""

import re
from typing import Dict, List, Optional, Any
from loguru import logger
from matlab2cpp_agentic_service.infrastructure.tools.error_extractor import (
    ErrorExtractor, CompilationError
)


class IterativeErrorFixer:
    """
    Iteratively fix compilation errors using LLM feedback.
    
    Strategy:
    1. Parse compilation errors
    2. Select worst offender files
    3. Generate targeted fix prompts
    4. Ask LLM to fix each file
    5. Recompile and check progress
    6. Repeat until < 3 errors or max iterations
    """
    
    def __init__(self, llm_client, compilation_manager):
        self.llm = llm_client
        self.compilation_manager = compilation_manager
        self.error_extractor = ErrorExtractor()
        self.max_iterations = 3
        self.success_threshold = 3  # Stop if errors < 3
        self.iteration_count = 0
        self.logger = logger.bind(name="error_fix_iterator")
    
    def fix_compilation_errors(self, 
                               generated_files: Dict[str, str],
                               matlab_sources: Dict[str, str],
                               compilation_output: str,
                               project_name: str) -> Dict[str, str]:
        """
        Main entry point for iterative error fixing.
        
        Args:
            generated_files: {'e1.cpp': '...', 'e1.h': '...'}
            matlab_sources: {'e1.m': '...'}
            compilation_output: Raw g++ output from first attempt
            project_name: Name of the project being compiled
            
        Returns:
            Fixed files (Dict[filename, content])
        """
        self.logger.info("=" * 80)
        self.logger.info("🔄 Starting Iterative LLM Error Fixing")
        self.logger.info("=" * 80)
        
        current_files = generated_files.copy()
        previous_error_count = 999
        
        for iteration in range(1, self.max_iterations + 1):
            self.iteration_count = iteration
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"🔄 ITERATION {iteration}/{self.max_iterations}")
            self.logger.info(f"{'='*80}\n")
            
            # Step 1: Parse errors
            errors = self.error_extractor.parse_compilation_output(compilation_output)
            error_list = self.error_extractor.filter_errors_only(errors)  # Ignore warnings
            error_count = len(error_list)
            
            self.logger.info(f"📊 Error Analysis:")
            self.logger.info(f"  - Total errors: {error_count}")
            self.logger.info(f"  - Total warnings: {len(errors) - error_count}")
            
            # Check success threshold
            if error_count < self.success_threshold:
                self.logger.info(f"✅ SUCCESS! Only {error_count} errors remaining (< {self.success_threshold})")
                break
            
            # Check progress
            if error_count >= previous_error_count:
                self.logger.warning(f"⚠️ No progress in iteration {iteration} ({error_count} >= {previous_error_count})")
                self.logger.warning("Stopping to prevent infinite loop")
                break
            
            self.logger.info(f"  - Progress: {previous_error_count} → {error_count} errors")
            previous_error_count = error_count
            
            # Step 2: Group by file
            errors_by_file = self.error_extractor.group_errors_by_file(error_list)
            stats = self.error_extractor.get_error_statistics(error_list)
            
            self.logger.info(f"  - Files affected: {stats['files_affected']}")
            self.logger.info(f"  - Error types: syntax={stats['syntax']}, type={stats['type']}, semantic={stats['semantic']}")
            
            # Step 3: Select files to fix
            priority_files = self._select_files_to_fix(errors_by_file, iteration)
            self.logger.info(f"\n📝 Fixing {len(priority_files)} files this iteration:")
            for file in priority_files:
                file_error_count = len(errors_by_file.get(file, []))
                self.logger.info(f"  - {file}: {file_error_count} errors")
            
            # Step 4: Fix each file
            for filename in priority_files:
                file_errors = errors_by_file.get(filename, [])
                if not file_errors:
                    continue
                
                self.logger.info(f"\n🔧 Fixing {filename} ({len(file_errors)} errors)...")
                
                # Generate targeted fix prompt
                fix_prompt = self._build_error_fix_prompt(
                    filename=filename,
                    current_code=current_files.get(filename, ''),
                    matlab_source=self._find_matlab_source(filename, matlab_sources),
                    errors=file_errors,
                    iteration=iteration
                )
                
                # Ask LLM to fix
                fixed_code = self._request_llm_fix(fix_prompt, filename)
                
                # Update file if fix was successful
                if fixed_code and fixed_code.strip():
                    current_files[filename] = fixed_code
                    self.logger.info(f"✅ Updated {filename} with LLM fixes")
                else:
                    self.logger.warning(f"⚠️ LLM fix failed for {filename}, keeping original")
            
            # Step 5: Re-compile
            self.logger.info(f"\n🔨 Re-compiling after iteration {iteration}...")
            compilation_output = self._recompile(current_files, project_name)
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"🏁 Iterative fixing complete after {self.iteration_count} iterations")
        self.logger.info(f"{'='*80}\n")
        
        return current_files
    
    def _select_files_to_fix(self, 
                             errors_by_file: Dict[str, List[CompilationError]], 
                             iteration: int) -> List[str]:
        """
        Select which files to fix in this iteration.
        
        Strategy:
        - Iteration 1: Fix top 3 files with most errors (worst offenders)
        - Iteration 2: Fix files with blocking errors (max 5)
        - Iteration 3: Fix any remaining files (max 5)
        
        Args:
            errors_by_file: Errors grouped by filename
            iteration: Current iteration number
            
        Returns:
            List of filenames to fix
        """
        if iteration == 1:
            # Fix top 3 files with most errors
            sorted_files = sorted(errors_by_file.items(), 
                                 key=lambda x: len(x[1]), 
                                 reverse=True)
            selected = [f[0] for f in sorted_files[:3]]
            self.logger.info(f"Strategy: Fix top 3 files with most errors")
            return selected
        
        elif iteration == 2:
            # Fix files with blocking errors (error severity, not warnings)
            sorted_files = sorted(errors_by_file.items(), 
                                 key=lambda x: len(x[1]), 
                                 reverse=True)
            selected = [f[0] for f in sorted_files[:5]]
            self.logger.info(f"Strategy: Fix files with blocking errors (max 5)")
            return selected
        
        else:
            # Fix all remaining files
            sorted_files = sorted(errors_by_file.items(), 
                                 key=lambda x: len(x[1]), 
                                 reverse=True)
            selected = [f[0] for f in sorted_files[:5]]
            self.logger.info(f"Strategy: Fix any remaining files (max 5)")
            return selected
    
    def _build_error_fix_prompt(self, 
                                filename: str,
                                current_code: str,
                                matlab_source: str,
                                errors: List[CompilationError],
                                iteration: int) -> str:
        """
        Build targeted error-fixing prompt for LLM.
        
        Args:
            filename: File to fix
            current_code: Current C++ code (with errors)
            matlab_source: Original MATLAB code
            errors: List of errors in this file
            iteration: Current iteration number
            
        Returns:
            Formatted prompt string
        """
        # Build error section
        error_section = ""
        for i, error in enumerate(errors, 1):
            error_section += f"\nERROR #{i} ({error.severity.upper()}):\n"
            error_section += f"  Location: Line {error.line}, Column {error.column}\n"
            error_section += f"  Category: {error.error_type}\n"
            error_section += f"  Message: {error.message}\n"
            
            if error.context:
                error_section += f"  Code Context:\n"
                for ctx_line in error.context.split('\n'):
                    error_section += f"    {ctx_line}\n"
            
            if error.suggested_fix:
                error_section += f"  💡 Suggestion: {error.suggested_fix}\n"
        
        prompt = f"""You are fixing COMPILATION ERRORS in C++ code converted from MATLAB.

FILE TO FIX: {filename}
ITERATION: {iteration}/{self.max_iterations}
ERRORS IN THIS FILE: {len(errors)}

═══════════════════════════════════════════════════════════════
ORIGINAL MATLAB CODE (for reference):
═══════════════════════════════════════════════════════════════
{matlab_source or '(MATLAB source not available)'}

═══════════════════════════════════════════════════════════════
CURRENT C++ CODE (WITH ERRORS):
═══════════════════════════════════════════════════════════════
{current_code}

═══════════════════════════════════════════════════════════════
COMPILATION ERRORS TO FIX:
═══════════════════════════════════════════════════════════════
{error_section}

═══════════════════════════════════════════════════════════════
YOUR TASK:
═══════════════════════════════════════════════════════════════
Fix ONLY the {len(errors)} compilation error(s) listed above.
Do NOT change working code or add new features.

CRITICAL RULES:
1. ✅ Fix the EXACT errors shown (check line numbers match)
2. ✅ Keep all working code UNCHANGED
3. ✅ Maintain overall structure, logic, and function signatures
4. ✅ Declare ALL variables before use
5. ✅ Use correct Eigen API (see cheat sheet below)
6. ❌ DO NOT refactor or add new features
7. ❌ DO NOT change function signatures (will break other files)
8. ❌ DO NOT change namespace or class names

EIGEN API CHEAT SHEET (CRITICAL):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MatrixXd Methods:
    ✅ MatrixXd::Zero(rows, cols)          ← 2 args ONLY
    ✅ MatrixXd::Constant(rows, cols, val) ← Initialize with value
    ✅ .rows(), .cols()                    ← Get dimensions
    ✅ .block(i, j, rows, cols)            ← Extract sub-matrix
    ❌ NO .slice() method!
    ❌ NO 3-argument Zero()!
  
  Tensor Methods:
    ✅ Tensor<double, 3> t(dim0, dim1, dim2);
    ✅ t.setZero()                         ← Initialize to zero
    ✅ .dimension(0), .dimension(1), .dimension(2)
    ✅ t(i, j, k)                          ← Access element
    ❌ NO .rows()/.cols() methods!
  
  Common Mistakes:
    ❌ MatrixXd::Zero(a, b, c)     → Use Tensor<double,3>
    ❌ matrix.slice(start, len)     → Use .segment() or .block()
    ❌ matrix.dimension(0)          → Use .rows() for MatrixXd
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VARIABLE DECLARATION CHECKLIST:
Before using ANY variable, ensure it's declared with proper type:
✅ int stepsize = 1;
✅ bool check = false;
✅ double DistancetoEnd = 0.0;
✅ Eigen::VectorXd gradient(2);

OUTPUT FORMAT:
Provide the COMPLETE fixed file in a SINGLE ```cpp code block.
Do NOT use <think> tags or explanations.
Just provide the corrected, compilable C++ code.

```cpp
// Complete fixed version of {filename}
// (Paste ENTIRE file here with all fixes applied)
```

START YOUR RESPONSE IMMEDIATELY WITH ```cpp
"""
        
        return prompt
    
    def _request_llm_fix(self, prompt: str, filename: str) -> Optional[str]:
        """
        Send fix request to LLM and extract fixed code.
        
        Args:
            prompt: Formatted error-fixing prompt
            filename: File being fixed
            
        Returns:
            Fixed code string or None if failed
        """
        try:
            # Call LLM (using existing infrastructure)
            response = self.llm.invoke(prompt)
            
            # Extract code from response
            fixed_code = self._extract_code_from_response(response.content if hasattr(response, 'content') else str(response))
            
            if not fixed_code:
                self.logger.warning(f"Failed to extract code from LLM response for {filename}")
                return None
            
            return fixed_code
            
        except Exception as e:
            self.logger.error(f"LLM fix request failed for {filename}: {e}")
            return None
    
    def _extract_code_from_response(self, response: str) -> Optional[str]:
        """
        Extract C++ code from LLM response.
        
        Args:
            response: LLM response text
            
        Returns:
            Extracted code or None
        """
        # Look for ```cpp ... ``` blocks
        pattern = r'```(?:cpp|c\+\+)?\s*\n(.*?)\n```'
        matches = re.findall(pattern, response, re.DOTALL)
        
        if matches:
            # Return the first (and ideally only) code block
            code = matches[0].strip()
            if code and len(code) > 50:  # Sanity check
                return code
        
        # Fallback: Try to extract any code-like content
        if '#include' in response or 'namespace' in response:
            # Response might be raw code without code blocks
            return response.strip()
        
        return None
    
    def _find_matlab_source(self, cpp_filename: str, matlab_sources: Dict[str, str]) -> str:
        """
        Find original MATLAB source for a C++ file.
        
        Args:
            cpp_filename: C++ filename (e.g., 'e1.cpp')
            matlab_sources: Dict of MATLAB sources
            
        Returns:
            MATLAB source code or empty string
        """
        # Convert e1.cpp → e1.m or e1
        base_name = cpp_filename.replace('.cpp', '').replace('.h', '').replace('.hpp', '')
        
        # Try with .m extension
        if f"{base_name}.m" in matlab_sources:
            return matlab_sources[f"{base_name}.m"]
        
        # Try without extension
        if base_name in matlab_sources:
            return matlab_sources[base_name]
        
        # Try case variations
        for matlab_file, source in matlab_sources.items():
            if base_name.lower() in matlab_file.lower():
                return source
        
        return ""
    
    def _recompile(self, files: Dict[str, str], project_name: str) -> str:
        """
        Recompile the code and return compilation output.
        
        Args:
            files: Generated C++ files
            project_name: Project name for compilation
            
        Returns:
            Compilation output string
        """
        try:
            # Use existing compilation manager
            result = self.compilation_manager.test_compilation(files, project_name)
            return result.get('output', '') if isinstance(result, dict) else str(result)
        except Exception as e:
            self.logger.error(f"Recompilation failed: {e}")
            return str(e)
    
    def get_fixing_summary(self) -> Dict[str, Any]:
        """
        Get summary of fixing process.
        
        Returns:
            Summary dict with statistics
        """
        return {
            'iterations_performed': self.iteration_count,
            'max_iterations': self.max_iterations,
            'success_threshold': self.success_threshold,
        }

