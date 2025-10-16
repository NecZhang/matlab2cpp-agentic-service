"""
Error Fix Prompt Generator

This agent analyzes compilation errors and generates specific, targeted prompts
to fix the exact issues found in the generated code.
"""

import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ..base.langgraph_agent import BaseLangGraphAgent, AgentConfig
from ....infrastructure.tools.llm_client import LLMClient


@dataclass
class ErrorFix:
    """Represents a specific error fix instruction."""
    error_type: str
    error_pattern: str
    fix_instruction: str
    example_wrong: str
    example_correct: str
    priority: int  # 1=critical, 2=high, 3=medium


class ErrorFixPromptGenerator(BaseLangGraphAgent):
    """
    Generates targeted prompts to fix specific compilation errors.
    
    This agent:
    1. Analyzes compilation errors
    2. Maps errors to specific fix instructions
    3. Generates targeted prompts for code regeneration
    4. Provides examples of correct vs incorrect code
    """
    
    def __init__(self, config: AgentConfig, llm_client: LLMClient):
        super().__init__(config, llm_client)
        
        # Define specific error fixes
        self.error_fixes = self._initialize_error_fixes()
        
        self.logger.info(f"Initialized Error Fix Prompt Generator: {config.name}")
    
    def _initialize_error_fixes(self) -> List[ErrorFix]:
        """Initialize the mapping of errors to specific fixes."""
        return [
            ErrorFix(
                error_type="include_syntax",
                error_pattern=r"(#include expects.*|fatal error: .*\.h: No such file.*|error: .*\.h: No such file.*)",
                fix_instruction="""
                🚨 CRITICAL FIX REQUIRED: Include Statement Errors
                
                The generated code has include statement errors. You MUST fix them as follows:
                
                ✅ CORRECT FORMATS:
                - System headers: #include <iostream>
                - System headers: #include <vector>
                - System headers: #include <string>
                - System headers: #include <Eigen/Dense>
                - Local headers: #include "filename.h"
                
                ❌ WRONG FORMATS (NEVER USE):
                - #include 
                - #include <
                - #include >
                - #include ""
                - #include "main.h" (when the actual header is different)
                - Incomplete includes like "#include <iostream"
                
                🔧 IMMEDIATE ACTIONS: 
                1. Replace ALL malformed #include statements with proper syntax
                2. Use the CORRECT header filename (e.g., "arma_filter.h" not "main.h")
                3. Ensure all required system headers are included
                4. Verify all angle brackets and quotes are properly closed
                5. Add missing headers for any types used (Eigen, STL, etc.)
                
                🎯 VALIDATION: Check that every #include line:
                - Starts with #include
                - Has proper opening bracket/quote
                - Has proper closing bracket/quote
                - References an existing file or standard header
                """,
                example_wrong="#include \"main.h\"",
                example_correct="#include \"arma_filter.h\"",
                priority=1
            ),
            
            ErrorFix(
                error_type="missing_semicolon",
                error_pattern=r"expected ';' before|missing ';'",
                fix_instruction="""
                🚨 CRITICAL FIX REQUIRED: Missing Semicolons
                
                Add semicolons after ALL statements:
                - Variable declarations: int x = 5;
                - Function calls: func();
                - Return statements: return 0;
                - Class declarations: class MyClass {};
                - Expression statements: x++;
                
                🔧 IMMEDIATE ACTIONS:
                1. Scan every line for missing semicolons
                2. Add semicolons after variable declarations
                3. Add semicolons after function calls
                4. Add semicolons after return statements
                5. Verify all statements end properly
                
                🎯 VALIDATION: Every statement must end with semicolon except:
                - Control flow statements (if, for, while, etc.)
                - Function definitions
                - Class/struct definitions
                """,
                example_wrong="int x = 5",
                example_correct="int x = 5;",
                priority=1
            ),
            
            ErrorFix(
                error_type="template_syntax",
                error_pattern=r"expected.*>|std::vector>>|template.*>",
                fix_instruction="""
                🚨 CRITICAL FIX REQUIRED: Template Syntax Errors
                
                Fix malformed template syntax:
                - WRONG: std::vector>>
                - CORRECT: std::vector<std::vector<double>>
                
                ✅ CORRECT TEMPLATE PATTERNS:
                - std::vector<int>
                - std::vector<std::vector<double>>
                - std::map<std::string, int>
                - Eigen::MatrixXd
                - Eigen::VectorXd
                
                ❌ WRONG PATTERNS (NEVER USE):
                - std::vector>>
                - std::vector<
                - std::vector<
                - Missing template parameters
                
                🔧 IMMEDIATE ACTIONS:
                1. Find all >> patterns and replace with proper templates
                2. Ensure all < > brackets are properly paired
                3. Add missing template parameters
                4. Verify template syntax is complete
                """,
                example_wrong="std::vector>>",
                example_correct="std::vector<std::vector<double>>",
                priority=1
            ),
            
            ErrorFix(
                error_type="missing_terminating_quote",
                error_pattern=r"missing terminating.*character",
                fix_instruction="""
                CRITICAL FIX REQUIRED: String Literal Errors
                
                Fix all string literals:
                - Ensure all strings are properly quoted
                - Use double quotes for strings: "hello"
                - Escape quotes inside strings: "He said \"Hello\""
                - Close all string literals properly
                
                ACTION: Fix all malformed string literals.
                """,
                example_wrong="std::cout << hello world;",
                example_correct="std::cout << \"hello world\";",
                priority=1
            ),
            
            ErrorFix(
                error_type="undeclared_identifier",
                error_pattern=r"'.*' does not name a type",
                fix_instruction="""
                HIGH PRIORITY FIX: Undeclared Variables/Types
                
                Declare all variables and types before use:
                - Add proper type declarations: int x;
                - Include necessary headers for types
                - Use correct namespace qualifications: std::cout
                
                ACTION: Add missing declarations and fix type errors.
                """,
                example_wrong="cout << \"hello\";",
                example_correct="std::cout << \"hello\";",
                priority=2
            ),
            
            ErrorFix(
                error_type="function_not_declared",
                error_pattern=r"'.*' was not declared",
                fix_instruction="""
                HIGH PRIORITY FIX: Undeclared Functions
                
                Declare all functions before use:
                - Add function prototypes in headers
                - Include necessary headers for functions
                - Use correct function signatures
                
                ACTION: Add missing function declarations.
                """,
                example_wrong="myFunction();",
                example_correct="void myFunction(); // declaration\nmyFunction(); // usage",
                priority=2
            ),
            
            ErrorFix(
                error_type="expected_identifier",
                error_pattern=r"expected unqualified-id",
                fix_instruction="""
                MEDIUM PRIORITY FIX: Syntax Structure Errors
                
                Fix code structure issues:
                - Ensure proper function/class structure
                - Fix misplaced keywords
                - Correct variable naming
                - Fix brace matching
                
                ACTION: Review and fix code structure issues.
                """,
                example_wrong="for (int i = 0; i < 10; i++ {",
                example_correct="for (int i = 0; i < 10; i++) {",
                priority=3
            )
        ]
    
    def generate_error_fix_prompt(self, compilation_errors: List[str], 
                                 generated_code: Optional[Dict[str, Any]] = None,
                                 matlab_analysis: Optional[Dict[str, Any]] = None,
                                 api_knowledge: str = "") -> Optional[str]:
        """
        Generate a targeted prompt to fix specific compilation errors using LLM-powered analysis.
        
        Args:
            compilation_errors: List of compilation error messages
            generated_code: The generated C++ code that failed
            matlab_analysis: Original MATLAB analysis for context
            api_knowledge: API documentation retrieved from knowledge base
            
        Returns:
            Enhanced prompt with specific error fixes, or None if no errors
        """
        if not compilation_errors:
            return None
        
        # Use LLM-powered analysis if we have the full context
        if generated_code and matlab_analysis:
            self.logger.debug(f"Using LLM-powered error analysis for {len(compilation_errors)} errors")
            # First analyze errors intelligently
            error_analysis = self._analyze_errors_with_llm(compilation_errors, generated_code)
            return self._build_enhanced_llm_error_analysis_prompt(
                compilation_errors, generated_code, matlab_analysis, error_analysis, api_knowledge
            )
        
        # Fallback to pattern-based analysis for backward compatibility
        return self._generate_pattern_based_fix_prompt(compilation_errors)
    
    def _generate_pattern_based_fix_prompt(self, compilation_errors: List[str]) -> Optional[str]:
        """Generate pattern-based error fix prompt (legacy method)."""
        # Analyze errors and find applicable fixes
        applicable_fixes = []
        
        self.logger.debug(f"Analyzing {len(compilation_errors)} compilation errors")
        for error in compilation_errors:
            self.logger.debug(f"Checking error: {error}")
            for fix in self.error_fixes:
                if re.search(fix.error_pattern, error, re.IGNORECASE):
                    self.logger.debug(f"Found matching fix for pattern '{fix.error_pattern}': {fix.error_type}")
                    if fix not in applicable_fixes:
                        applicable_fixes.append(fix)
        
        if not applicable_fixes:
            self.logger.warning("No specific fixes found for compilation errors")
            return None
        
        # Sort by priority (critical first)
        applicable_fixes.sort(key=lambda x: x.priority)
        
        # Generate comprehensive fix prompt
        fix_prompt = self._build_error_fix_prompt(applicable_fixes, compilation_errors)
        
        self.logger.info(f"Generated error fix prompt for {len(applicable_fixes)} error types")
        return fix_prompt
    
    def _build_error_fix_prompt(self, fixes: List[ErrorFix], errors: List[str]) -> str:
        """Build a comprehensive error fix prompt."""
        
        prompt_parts = []
        
        # Header
        prompt_parts.append("🚨 CRITICAL CODE GENERATION FIXES REQUIRED 🚨")
        prompt_parts.append("=" * 50)
        prompt_parts.append("")
        prompt_parts.append("The previous code generation failed with compilation errors.")
        prompt_parts.append("You MUST regenerate the code with these specific fixes:")
        prompt_parts.append("")
        
        # List all errors found
        prompt_parts.append("COMPILATION ERRORS DETECTED:")
        for i, error in enumerate(errors[:10], 1):  # Limit to first 10 errors
            prompt_parts.append(f"{i}. {error}")
        prompt_parts.append("")
        
        # Add specific fixes
        for i, fix in enumerate(fixes, 1):
            prompt_parts.append(f"FIX #{i}: {fix.error_type.upper().replace('_', ' ')}")
            prompt_parts.append("-" * 30)
            prompt_parts.append(fix.fix_instruction)
            
            if fix.example_wrong and fix.example_correct:
                prompt_parts.append("")
                prompt_parts.append("EXAMPLES:")
                prompt_parts.append(f"❌ WRONG: {fix.example_wrong}")
                prompt_parts.append(f"✅ CORRECT: {fix.example_correct}")
            
            prompt_parts.append("")
        
        # Instructions
        prompt_parts.append("REGENERATION INSTRUCTIONS:")
        prompt_parts.append("-" * 25)
        prompt_parts.append("1. Apply ALL the fixes listed above")
        prompt_parts.append("2. Ensure the code compiles without errors")
        prompt_parts.append("3. Use proper C++ syntax throughout")
        prompt_parts.append("4. Include all necessary headers")
        prompt_parts.append("5. Declare all variables and functions properly")
        prompt_parts.append("")
        prompt_parts.append("Generate clean, compilable C++ code that addresses ALL these issues.")
        
        return "\n".join(prompt_parts)
    
    def _build_llm_error_analysis_prompt(self, compilation_errors: List[str], 
                                        generated_code: Dict[str, Any],
                                        matlab_analysis: Dict[str, Any]) -> str:
        """Build comprehensive LLM-powered error analysis prompt."""
        return f"""
CRITICAL ERROR ANALYSIS AND FIXING REQUIRED
==========================================

The generated C++ code failed to compile with these errors:
{chr(10).join(f"- {error}" for error in compilation_errors)}

GENERATED CODE TO ANALYZE AND FIX:
{self._format_code_for_analysis(generated_code)}

ORIGINAL MATLAB CODE CONTEXT:
{self._format_matlab_context(matlab_analysis)}

ANALYSIS AND FIXING TASK:
1. Analyze each compilation error and identify the root cause
2. Understand the type system requirements and operation constraints
3. Compare the generated C++ logic with the original MATLAB functionality
4. Generate corrected code that maintains mathematical correctness
5. Ensure proper C++ idioms and type safety

ERROR ANALYSIS REQUIREMENTS:
- Categorize errors by type (syntax, type mismatch, missing declarations, etc.)
- Identify systematic issues or patterns in the errors
- Determine the most critical errors that must be fixed first
- Understand the mathematical operations being performed

FIXING REQUIREMENTS:
- Fix all compilation errors while preserving the original MATLAB functionality
- Use appropriate C++ types for matrix/vector operations (std::vector<std::vector<double>> for matrices, std::vector<double> for vectors)
- Implement proper element-wise operations where MATLAB uses broadcasting
- Add necessary helper functions for complex operations
- Maintain type safety and avoid implicit conversions
- Use proper C++ idioms and best practices

TYPE MAPPING GUIDANCE:
- MATLAB matrices → std::vector<std::vector<double>>
- MATLAB vectors → std::vector<double>
- MATLAB scalars → double
- MATLAB element-wise operations → C++ loops or helper functions
- MATLAB matrix multiplication → Custom matrix multiplication functions

GENERATE THE CORRECTED CODE:
Provide the complete corrected C++ header and implementation files with:
1. Fixed compilation errors
2. Proper type declarations and operations
3. Helper functions for complex operations
4. Comprehensive comments explaining the fixes applied
5. Maintained mathematical correctness of the original MATLAB algorithm

Focus on general-purpose solutions that would work for similar MATLAB-to-C++ conversions.
"""
    
    def _build_enhanced_llm_error_analysis_prompt(self, compilation_errors: List[str], 
                                                 generated_code: Dict[str, Any],
                                                 matlab_analysis: Dict[str, Any],
                                                 error_analysis: Dict[str, Any],
                                                 api_knowledge: str = "") -> str:
        """Build enhanced LLM-powered error analysis prompt with intelligent categorization."""
        # Build knowledge base section if API docs are available
        kb_section = ""
        if api_knowledge:
            kb_section = f"""
{'=' * 80}
📚 API KNOWLEDGE BASE - RELEVANT DOCUMENTATION FOR YOUR ERRORS
{'=' * 80}
{api_knowledge}
{'=' * 80}

"""
        
        return f"""
CRITICAL ERROR ANALYSIS AND FIXING REQUIRED
==========================================

{kb_section}INTELLIGENT ERROR ANALYSIS:
{self._format_error_analysis(error_analysis)}

COMPILATION ERRORS TO FIX:
{chr(10).join(f"- {error}" for error in compilation_errors)}

GENERATED CODE TO ANALYZE AND FIX:
{self._format_code_for_analysis(generated_code)}

ORIGINAL MATLAB CODE CONTEXT:
{self._format_matlab_context(matlab_analysis)}

ANALYSIS AND FIXING TASK:
1. Review the intelligent error categorization above
2. Follow the systematic fix strategy: {error_analysis.get('fix_strategy', 'Address errors by priority')}
3. Fix critical errors first, then important errors, then minor issues
4. Ensure all fixes maintain the original MATLAB functionality
5. Use proper C++ idioms and type safety

PRIORITIZED FIXING APPROACH:
CRITICAL FIXES (Must be fixed first):
{chr(10).join(f"- {error}" for error in error_analysis.get('fix_priorities', {}).get('critical', []))}

IMPORTANT FIXES (Should be fixed):
{chr(10).join(f"- {error}" for error in error_analysis.get('fix_priorities', {}).get('important', []))}

MINOR ISSUES (Warnings to address):
{chr(10).join(f"- {error}" for error in error_analysis.get('fix_priorities', {}).get('minor', []))}

SYSTEMATIC ISSUES IDENTIFIED:
{chr(10).join(f"- {issue}" for issue in error_analysis.get('systematic_issues', []))}

{self._build_available_functions_context(generated_code)}

{self._build_eigen_api_warnings()}

SPECIFIC FIX SUGGESTIONS:
{self._format_fix_suggestions(error_analysis.get('suggested_fixes', {}))}

MANDATORY FIXING REQUIREMENTS (NO EXCEPTIONS):
- CRITICAL: Fix ALL compilation errors - the code MUST compile successfully
- CRITICAL: Add missing #include statements (especially #include <Eigen/Dense> for Eigen types)
- CRITICAL: Fix function declaration syntax errors
- CRITICAL: Ensure all Eigen types are properly declared (use Eigen::MatrixXd, Eigen::VectorXd, NOT ArrayXXXd)
- MANDATORY: Generate syntactically correct C++ code that compiles without errors
- MANDATORY: Use proper C++ function syntax with correct parameter types
- MANDATORY: Include all necessary headers for the types you use
- MANDATORY: Fix any namespace or scope issues
- MANDATORY: Ensure function definitions match their declarations

FAILURE TO COMPLY WITH THESE REQUIREMENTS WILL RESULT IN COMPILATION FAILURE.
- Use proper C++ idioms and best practices

TYPE MAPPING GUIDANCE:
{self._format_type_mapping_guidance(matlab_analysis)}

GENERATE THE CORRECTED CODE:
Provide the complete corrected C++ header and implementation files with:
1. Fixed compilation errors (following the priority order above)
2. Proper type declarations and operations
3. Helper functions for complex operations
4. Comprehensive comments explaining the fixes applied
5. Maintained mathematical correctness of the original MATLAB algorithm

Focus on general-purpose solutions that would work for similar MATLAB-to-C++ conversions.
"""
    
    def _format_type_mapping_guidance(self, matlab_analysis: Dict[str, Any]) -> str:
        """Format type mapping guidance based on MATLAB analysis."""
        has_3d_arrays = matlab_analysis.get('has_3d_arrays', False)
        
        type_mapping = [
            "- MATLAB 2D matrices → Eigen::MatrixXd",
            "- MATLAB 1D vectors → Eigen::VectorXd",
            "- MATLAB scalars → double",
        ]
        
        if has_3d_arrays:
            type_mapping.extend([
                "- MATLAB 3D arrays → Eigen::Tensor<double, 3>",
                "",
                "CRITICAL 3D ARRAY CORRECTIONS:",
                "- NEVER use Eigen::Array3D (doesn't exist)",
                "- NEVER use Eigen::Array3d<double> (Array3d is 1D with 3 elements, NOT 3D)",
                "- NEVER use Eigen::Array3d as a template (it's a typedef, not a template)",
                "- CORRECT: Eigen::Tensor<double, 3> for 3D tensors",
                "- CORRECT: tensor.dimension(0), tensor.dimension(1), tensor.dimension(2) for sizes",
                "- CORRECT: tensor(i, j, k) for element access",
                "- INCLUDE: #include <unsupported/Eigen/CXX11/Tensor>",
                "",
                "FIX PATTERN:",
                "❌ Eigen::Array3D<double>& data     → ✅ Eigen::Tensor<double, 3>& data",
                "❌ Eigen::Array3d<double>& data     → ✅ Eigen::Tensor<double, 3>& data",
                "❌ data.dimension(0) on Array3d     → ✅ tensor.dimension(0) on Tensor",
                "❌ data[i][j][k]                    → ✅ tensor(i, j, k)",
            ])
        else:
            type_mapping.extend([
                "- MATLAB element-wise operations → Eigen array operations or C++ loops",
                "- MATLAB matrix multiplication → Eigen matrix multiplication",
            ])
        
        return "\n".join(type_mapping)
    
    def _format_error_analysis(self, error_analysis: Dict[str, Any]) -> str:
        """Format error analysis for the prompt."""
        categories = error_analysis.get('error_categories', {})
        return f"""
Error Categories:
- Syntax Errors: {len(categories.get('syntax_errors', []))} found
- Type Errors: {len(categories.get('type_errors', []))} found  
- Logic Errors: {len(categories.get('logic_errors', []))} found
- Declaration Errors: {len(categories.get('declaration_errors', []))} found
- Eigen Errors: {len(categories.get('eigen_errors', []))} found
- Tensor Errors: {len(categories.get('tensor_errors', []))} found
- Missing Includes: {len(categories.get('missing_includes', []))} found

Fix Strategy: {error_analysis.get('fix_strategy', 'Address errors systematically')}

Systematic Issues: {', '.join(error_analysis.get('systematic_issues', []))}
"""
    
    def _format_fix_suggestions(self, suggestions: Dict[str, str]) -> str:
        """Format fix suggestions for the prompt."""
        if not suggestions:
            return "No specific suggestions available"
        
        formatted = []
        for error, suggestion in list(suggestions.items())[:5]:  # Limit to first 5
            formatted.append(f"- {error[:100]}... → {suggestion}")
        
        return "\n".join(formatted)
    
    def _analyze_errors_with_llm(self, compilation_errors: List[str], 
                                generated_code: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to analyze and categorize compilation errors intelligently."""
        analysis_prompt = f"""
COMPILATION ERROR ANALYSIS TASK
===============================

Compilation Errors:
{chr(10).join(compilation_errors)}

Generated Code:
{self._format_code_for_analysis(generated_code)}

ANALYSIS REQUIRED:
1. Categorize each error by type (syntax, type, logic, etc.)
2. Identify the root cause of each error
3. Determine the fix priority (critical, important, minor)
4. Suggest specific fixes for each error
5. Identify any patterns or systematic issues

CRITICAL: Focus on Eigen-specific errors like:
- 'ArrayXXXd' does not name a type (should be ArrayXXd or ArrayXd)
- 'Eigen' does not name a type (missing #include <Eigen/Dense>)
- Member access errors on wrong types
- Template parameter errors

Provide analysis in JSON format:
{{
    "error_categories": {{
        "syntax_errors": ["list of syntax errors"],
        "type_errors": ["list of type-related errors"],
        "logic_errors": ["list of logic errors"],
        "declaration_errors": ["list of missing declaration errors"],
        "eigen_errors": ["list of Eigen library specific errors"]
    }},
    "fix_priorities": {{
        "critical": ["errors that must be fixed first"],
        "important": ["errors that should be fixed"],
        "minor": ["warnings or minor issues"]
    }},
    "suggested_fixes": {{
        "error_text": "specific fix suggestion"
    }},
    "systematic_issues": ["any patterns or root causes identified"],
    "fix_strategy": "overall strategy for fixing these errors"
}}

CRITICAL: Return ONLY valid JSON, no <think> tags, no explanations, no reasoning text.
Start your response immediately with {{ and end with }}. Do not include any text before or after the JSON.
"""
        
        try:
            # Actually call the LLM for error analysis
            self.logger.info(f"Calling LLM for error analysis of {len(compilation_errors)} errors")
            response = self.llm_client.get_completion(analysis_prompt)
            
            # Extract JSON from response (remove <think> tags if present)
            import json
            import re
            
            # Remove <think> tags and their content
            clean_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
            
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', clean_response, flags=re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
            else:
                json_text = clean_response
            
            analysis_result = json.loads(json_text)
            
            # Validate the response structure
            required_keys = ["error_categories", "fix_priorities", "suggested_fixes", "systematic_issues", "fix_strategy"]
            for key in required_keys:
                if key not in analysis_result:
                    self.logger.warning(f"Missing key '{key}' in LLM error analysis response")
                    analysis_result[key] = {}
            
            self.logger.info(f"LLM error analysis completed successfully")
            return analysis_result
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM error analysis JSON: {e}")
            self.logger.error(f"LLM response: {response[:500]}...")
            # Fallback to enhanced pattern-based analysis
            return self._enhanced_pattern_based_analysis(compilation_errors)
        except Exception as e:
            self.logger.error(f"LLM error analysis failed: {e}")
            # Fallback to enhanced pattern-based analysis
            return self._enhanced_pattern_based_analysis(compilation_errors)
    
    def _suggest_fix_for_error(self, error: str) -> str:
        """Suggest a specific fix for an error."""
        if "ambiguating new declaration" in error:
            return "CRITICAL: Fix function signature mismatch between header and implementation - ensure return types match exactly"
        elif "invalid initialization of reference" in error:
            return "Fix type mismatch - ensure function parameters match expected types"
        elif "conversion from" in error:
            return "Fix type conversion - use proper type casting or correct variable types"
        elif "sqrt is not a member of std" in error:
            return "CRITICAL: Add #include <cmath> to access std::sqrt function"
        elif "not declared" in error:
            return "Add missing declaration or include necessary headers"
        elif "comparison of integer expressions of different signedness" in error:
            return "Fix signed/unsigned comparison by using size_t for loop indices"
        elif "unused variable" in error:
            return "Remove unused variable or use it in the code"
        else:
            return "Review and fix the specific syntax or type issue"
    
    def _format_code_for_analysis(self, generated_code: Dict[str, Any]) -> str:
        """Format generated code for analysis."""
        files = generated_code.get('files', {})
        if not files:
            return "No generated files available"
        
        formatted_code = []
        for filename, content in files.items():
            formatted_code.append(f"FILE: {filename}")
            formatted_code.append("=" * (len(filename) + 6))
            formatted_code.append(content)
            formatted_code.append("")
        
        return "\n".join(formatted_code)
    
    def _format_matlab_context(self, matlab_analysis: Dict[str, Any]) -> str:
        """Format MATLAB analysis for context."""
        if not matlab_analysis:
            return "No MATLAB analysis available"
        
        context_parts = []
        
        # Add file analyses
        file_analyses = matlab_analysis.get('file_analyses', [])
        if file_analyses:
            context_parts.append("MATLAB FILES:")
            for file_analysis in file_analyses:
                if isinstance(file_analysis, dict):
                    filename = file_analysis.get('filename', 'unknown')
                    functions = file_analysis.get('functions', [])
                    context_parts.append(f"- {filename}: {len(functions)} functions")
                    for func in functions:
                        if isinstance(func, dict):
                            func_name = func.get('name', 'unknown')
                            context_parts.append(f"  * {func_name}")
        
        # Add function call tree
        function_call_tree = matlab_analysis.get('function_call_tree', {})
        if function_call_tree:
            context_parts.append("\nFUNCTION DEPENDENCIES:")
            for func_name, calls in function_call_tree.items():
                context_parts.append(f"- {func_name} calls: {', '.join(calls) if calls else 'none'}")
        
        return "\n".join(context_parts) if context_parts else "No detailed MATLAB context available"
    
    def create_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create LangGraph node for error fix prompt generation."""
        return state
    
    def _enhanced_pattern_based_analysis(self, compilation_errors: List[str]) -> Dict[str, Any]:
        """Enhanced pattern-based error analysis with Eigen-specific patterns."""
        self.logger.info("Using enhanced pattern-based error analysis as fallback")
        
        # Enhanced error categorization with Eigen-specific and Tensor patterns
        error_categories = {
            "syntax_errors": [],
            "type_errors": [],
            "logic_errors": [],
            "declaration_errors": [],
            "eigen_errors": [],
            "tensor_errors": [],
            "missing_includes": []
        }
        
        fix_priorities = {
            "critical": [],
            "important": [],
            "minor": []
        }
        
        suggested_fixes = {}
        systematic_issues = []
        
        for error in compilation_errors:
            error_lower = error.lower()
            
            # Tensor-specific error patterns (check first for 3D arrays)
            if any(pattern in error_lower for pattern in [
                'array3d',
                'array3d<',
                'arrayxxxd',
                'is not a template',
                "has no member named 'dimension'",
            ]):
                error_categories["tensor_errors"].append(error)
                fix_priorities["critical"].append(error)
                if 'array3d' in error_lower:
                    suggested_fixes[error] = "CRITICAL: Use Eigen::Tensor<double, 3> for 3D arrays, not Array3d (which is a 1D array with 3 elements)"
                    systematic_issues.append("Incorrect 3D array type - Array3d is 1D, use Tensor<double, 3>")
                elif 'is not a template' in error_lower:
                    suggested_fixes[error] = "CRITICAL: Array3d is a typedef, not a template. Use Eigen::Tensor<double, 3> for 3D arrays."
                    systematic_issues.append("Trying to use Array3d as template - use Tensor<double, 3>")
                elif 'dimension' in error_lower:
                    suggested_fixes[error] = "Use .dimension(0), .dimension(1), .dimension(2) on Eigen::Tensor, not on Array3d"
                    systematic_issues.append("Incorrect dimension access - use Tensor, not Array3d")
            
            # Eigen-specific error patterns
            elif "arrayxxxd" in error_lower or "does not name a type" in error_lower:
                if "eigen" in error_lower:
                    error_categories["eigen_errors"].append(error)
                    fix_priorities["critical"].append(error)
                    suggested_fixes[error] = "CRITICAL: Replace 'ArrayXXXd' with correct Eigen type (ArrayXXd for 2D, ArrayXd for 1D). ArrayXXXd is not a valid Eigen type."
                    systematic_issues.append("Invalid Eigen type usage")
            
            elif "eigen" in error_lower and "does not name a type" in error_lower:
                error_categories["missing_includes"].append(error)
                fix_priorities["critical"].append(error)
                suggested_fixes[error] = "CRITICAL: Add #include <Eigen/Dense> to access Eigen types and functions."
                systematic_issues.append("Missing Eigen header includes")
            
            elif "request for member" in error_lower and "non-class type" in error_lower:
                error_categories["type_errors"].append(error)
                fix_priorities["critical"].append(error)
                suggested_fixes[error] = "CRITICAL: Type mismatch - parameter is being treated as wrong type. Check function signature and parameter types."
                systematic_issues.append("Function signature type mismatches")
            
            elif "dimension" in error_lower and "no member named" in error_lower:
                error_categories["eigen_errors"].append(error)
                fix_priorities["important"].append(error)
                suggested_fixes[error] = "Use .rows() or .cols() instead of .dimension() for Eigen arrays. .dimension() is not a valid Eigen method."
                systematic_issues.append("Incorrect Eigen API usage")
            
            elif "cannot convert" in error_lower and "to" in error_lower and "in assignment" in error_lower:
                error_categories["type_errors"].append(error)
                fix_priorities["important"].append(error)
                suggested_fixes[error] = "Type conversion error - ensure compatible types for assignment. Use explicit casting if needed."
                systematic_issues.append("Type conversion issues")
            
            elif "expected primary-expression" in error_lower:
                error_categories["syntax_errors"].append(error)
                fix_priorities["critical"].append(error)
                suggested_fixes[error] = "Syntax error - check for missing semicolons, incorrect template syntax, or malformed expressions."
                systematic_issues.append("Syntax errors in generated code")
            
            elif "invalid initialization" in error_lower or "conversion from" in error_lower:
                error_categories["type_errors"].append(error)
                fix_priorities["important"].append(error)
                suggested_fixes[error] = "Type mismatch - ensure variable types match their initialization or assignment."
                systematic_issues.append("Type initialization issues")
            
            elif "not declared" in error_lower or "ambiguating" in error_lower:
                error_categories["declaration_errors"].append(error)
                fix_priorities["critical"].append(error)
                suggested_fixes[error] = "Declaration error - check function signatures, variable declarations, and namespace usage."
                systematic_issues.append("Declaration and namespace issues")
            
            elif "warning:" in error_lower:
                fix_priorities["minor"].append(error)
                suggested_fixes[error] = "Minor warning - consider fixing for code quality."
            else:
                # Default categorization for unrecognized errors
                error_categories["logic_errors"].append(error)
                fix_priorities["important"].append(error)
                suggested_fixes[error] = "Logic error - review the code logic and ensure proper implementation."
        
        # Remove duplicates from systematic issues
        systematic_issues = list(set(systematic_issues))
        
        # Determine fix strategy based on error types
        if error_categories["eigen_errors"]:
            fix_strategy = "CRITICAL: Fix Eigen type errors FIRST (ArrayXXXd → ArrayXXd, missing includes), then type mismatches, then API usage errors"
        elif error_categories["type_errors"]:
            fix_strategy = "Fix type mismatches and conversion errors FIRST, then syntax errors, then declaration issues"
        elif error_categories["syntax_errors"]:
            fix_strategy = "Fix syntax errors FIRST (critical for compilation), then type issues, then logic problems"
        else:
            fix_strategy = "Address declaration errors first, then type issues, then logic problems"
        
        return {
            "error_categories": error_categories,
            "fix_priorities": fix_priorities,
            "suggested_fixes": suggested_fixes,
            "systematic_issues": systematic_issues,
            "fix_strategy": fix_strategy
        }

    def _build_available_functions_context(self, generated_code: Dict[str, Any]) -> str:
        """FIX #1: Build context showing available functions from generated code."""
        if not generated_code or 'files' not in generated_code:
            return ""
        
        available_functions = []
        files = generated_code.get('files', {})
        
        for filename, content in files.items():
            if filename.endswith('.h') and isinstance(content, str):
                # Extract function signatures
                func_sigs = self._extract_function_signatures_from_header(content, filename)
                available_functions.extend(func_sigs)
        
        if not available_functions:
            return ""
        
        return f"""
{"=" * 70}
🔑 AVAILABLE FUNCTIONS (FIX #1 - DO NOT INVENT NEW NAMES!):
{"=" * 70}

You may ONLY call these functions (already defined in generated headers):
{chr(10).join(f"  ✅ {sig}" for sig in available_functions)}

❌ DO NOT invent new function names like 'pointmin2d', 'rk42d', 'euler2d', 'dilate2d'!
❌ DO NOT add '2d', '3d', or other suffixes to existing function names!
❌ Only use the EXACT function names listed above!

If you need a function that doesn't exist, use the closest match from the list.
{"=" * 70}
"""
    
    def _extract_function_signatures_from_header(self, header_content: str, filename: str) -> List[str]:
        """Extract function signatures from header file (FIX #1)."""
        import re
        signatures = []
        lines = header_content.split('\n')
        
        namespace_name = ""
        for i, line in enumerate(lines):
            # Track namespace
            if 'namespace' in line and '{' in line:
                match = re.search(r'namespace\s+(\w+)', line)
                if match:
                    namespace_name = match.group(1)
            
            # Look for function declarations
            if (not line.strip().startswith('//') and 
                not line.strip().startswith('#') and
                not line.strip().startswith('*') and
                '(' in line):
                # Simple extraction - look for patterns like "Type functionName("
                clean_line = line.strip()
                if ');' in clean_line or (i+1 < len(lines) and ');' in lines[i+1]):
                    # Extract just the function name and basic signature
                    func_match = re.search(r'(\w+)\s+(\w+)\s*\(', clean_line)
                    if func_match and namespace_name:
                        return_type = func_match.group(1)
                        func_name = func_match.group(2)
                        if return_type not in ['if', 'for', 'while']:  # Skip control structures
                            signatures.append(f"{namespace_name}::{func_name}(...)")
        
        return signatures
    
    def _build_eigen_api_warnings(self) -> str:
        """FIX #3: Build Eigen API warnings to prevent common mistakes."""
        return f"""
{"=" * 70}
🚨 EIGEN API WARNINGS (FIX #3 - METHODS THAT DO NOT EXIST!):
{"=" * 70}

The following Eigen methods DO NOT EXIST. Do NOT use them:

❌ MatrixXd::Zero(a, b, c)      → Use Tensor<double,3> instead, or Zero(a,b)
❌ matrix.dimension()            → Use .rows()/.cols() for MatrixXd
❌ matrix.tensor<T>()            → No such method, use TensorMap
❌ Tensor::Zero()                → Use .setZero() instead
❌ tensor.rows()/cols()          → Use .dimension(i) for Tensor
❌ array.hasNonZero()            → Use (array != 0).any()
❌ matrix.slice()                → Use .block() or .segment()
❌ matrix.flatten()              → Use .reshaped() or manual reshaping
❌ Vector::replicate(n, m)       → Wrong syntax, check Eigen docs

CORRECT Eigen API usage:
✅ MatrixXd::Zero(rows, cols)    - For 2D matrices
✅ matrix.rows(), matrix.cols()  - Get dimensions of MatrixXd
✅ (array != 0).any()            - Check for non-zero elements
✅ matrix.block(i, j, rows, cols) - Extract sub-matrix
✅ tensor.dimension(i)           - Get dimension i of Tensor
✅ tensor.setZero()              - Initialize Tensor to zero

{"=" * 70}
"""

    def get_tools(self) -> List[Any]:
        """Get tools available to this agent."""
        return []

