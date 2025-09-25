"""
LangGraph-Native Tools

This module provides LangGraph-compatible tools for the MATLAB2C++ conversion service.
These tools are designed to work seamlessly with LangGraph agents and workflows.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import json
import re
from loguru import logger

from .matlab_parser import MATLABParser


class JSONResponseCleaner:
    """Utility class for cleaning malformed JSON responses from LLMs."""
    
    @staticmethod
    def clean_json_response(response: str) -> str:
        """Clean JSON response to remove invalid control characters and fix common JSON issues."""
        # Step 1: Remove thinking content while preserving JSON structure
        cleaned = JSONResponseCleaner._remove_thinking_content(response)
        
        # Step 2: Extract JSON boundaries precisely
        json_content = JSONResponseCleaner._extract_json_content(cleaned)
        
        # Step 3: Fix common JSON issues without breaking structure
        fixed_json = JSONResponseCleaner._fix_json_issues(json_content)
        
        return fixed_json
    
    @staticmethod
    def _remove_thinking_content(response: str) -> str:
        """Remove thinking content while preserving JSON structure."""
        # Remove <think> tags and content
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'<thinking>.*?</thinking>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'<think[^>]*>.*?</think>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'<thinking[^>]*>.*?</thinking>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove any remaining XML-like tags
        cleaned = re.sub(r'<[^>]*>', '', cleaned)
        
        # Remove any text before first { and after last }
        first_brace = cleaned.find('{')
        last_brace = cleaned.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            cleaned = cleaned[first_brace:last_brace+1]
        
        return cleaned.strip()
    
    @staticmethod
    def _extract_json_content(text: str) -> str:
        """Extract JSON content using precise boundary detection."""
        # Find JSON object boundaries with proper nesting
        start_pos = text.find('{')
        if start_pos == -1:
            return text
        
        # Count braces to find proper closing
        brace_count = 0
        end_pos = start_pos
        for i, char in enumerate(text[start_pos:], start_pos):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_pos = i
                    break
        
        return text[start_pos:end_pos+1] if brace_count == 0 else text
    
    @staticmethod
    def _fix_json_issues(json_str: str) -> str:
        """Fix common JSON issues without breaking structure."""
        # Fix unescaped newlines in string values only
        json_str = JSONResponseCleaner._fix_unescaped_newlines(json_str)
        
        # Fix unescaped quotes in string values only
        json_str = JSONResponseCleaner._fix_unescaped_quotes(json_str)
        
        # Remove invalid control characters
        json_str = ''.join(char for char in json_str if ord(char) >= 32 or char in '\n\t\r')
        
        return json_str
    
    @staticmethod
    def _fix_unescaped_newlines(json_str: str) -> str:
        """Fix unescaped newlines in JSON string values only."""
        def fix_string_value(match):
            string_content = match.group(1)
            # Only escape newlines, not quotes or backslashes
            string_content = string_content.replace('\n', '\\n')
            string_content = string_content.replace('\r', '\\r')
            string_content = string_content.replace('\t', '\\t')
            return f'"{string_content}"'
        
        # Find string values and fix newlines
        return re.sub(r'"([^"]*(?:\\.[^"]*)*)"', fix_string_value, json_str)
    
    @staticmethod
    def _fix_unescaped_quotes(json_str: str) -> str:
        """Fix unescaped quotes in JSON string values only."""
        def fix_string_value(match):
            string_content = match.group(1)
            # Only escape unescaped quotes
            string_content = string_content.replace('\\"', '\\"')  # Keep already escaped quotes
            string_content = string_content.replace('"', '\\"')    # Escape unescaped quotes
            return f'"{string_content}"'
        
        # Find string values and fix quotes
        return re.sub(r'"([^"]*(?:\\.[^"]*)*)"', fix_string_value, json_str)


class RobustJSONParser:
    """Robust JSON parser for LLM-generated responses with multiple fallback strategies."""
    
    @staticmethod
    def parse_llm_json(response: str) -> Dict[str, Any]:
        """Parse LLM-generated JSON with multiple fallback strategies."""
        # Strategy 1: Clean and parse with standard JSON
        try:
            cleaned = JSONResponseCleaner.clean_json_response(response)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Extract and reconstruct JSON
        try:
            extracted = RobustJSONParser._extract_json_fields(response)
            return RobustJSONParser._reconstruct_json(extracted)
        except Exception:
            pass
        
        # Strategy 3: Parse line by line
        try:
            return RobustJSONParser._parse_line_by_line(response)
        except Exception:
            pass
        
        # Strategy 4: Use regex-based extraction
        try:
            return RobustJSONParser._regex_extract(response)
        except Exception:
            pass
        
        # Fallback: Return structured default
        return RobustJSONParser._create_fallback_response(response)
    
    @staticmethod
    def _extract_json_fields(response: str) -> Dict[str, str]:
        """Extract JSON fields using pattern matching."""
        fields = {}
        
        # Extract header field
        header_match = re.search(r'"header":\s*"((?:[^"\\]|\\.)*)"', response, re.DOTALL)
        if header_match:
            fields['header'] = RobustJSONParser._unescape_string(header_match.group(1))
        
        # Extract implementation field
        impl_match = re.search(r'"implementation":\s*"((?:[^"\\]|\\.)*)"', response, re.DOTALL)
        if impl_match:
            fields['implementation'] = RobustJSONParser._unescape_string(impl_match.group(1))
        
        # Extract dependencies field
        deps_match = re.search(r'"dependencies":\s*(\[[^\]]*\])', response)
        if deps_match:
            try:
                fields['dependencies'] = json.loads(deps_match.group(1))
            except:
                fields['dependencies'] = ["Eigen3"]
        
        # Extract compilation_instructions field
        comp_match = re.search(r'"compilation_instructions":\s*"([^"]*)"', response)
        if comp_match:
            fields['compilation_instructions'] = comp_match.group(1)
        
        # Extract usage_example field
        usage_match = re.search(r'"usage_example":\s*"([^"]*)"', response)
        if usage_match:
            fields['usage_example'] = usage_match.group(1)
        
        # Extract notes field
        notes_match = re.search(r'"notes":\s*"([^"]*)"', response)
        if notes_match:
            fields['notes'] = notes_match.group(1)
        
        return fields
    
    @staticmethod
    def _reconstruct_json(fields: Dict[str, str]) -> Dict[str, Any]:
        """Reconstruct valid JSON from extracted fields."""
        result = {}
        for key, value in fields.items():
            result[key] = value
        
        # Add default fields if missing
        if 'dependencies' not in result:
            result['dependencies'] = ['Eigen3']
        if 'compilation_instructions' not in result:
            result['compilation_instructions'] = 'g++ -std=c++17 your_file.cpp'
        if 'usage_example' not in result:
            result['usage_example'] = '// Usage example'
        if 'notes' not in result:
            result['notes'] = 'Generated C++ code'
        
        return result
    
    @staticmethod
    def _parse_line_by_line(response: str) -> Dict[str, Any]:
        """Parse JSON by analyzing line by line for key-value pairs."""
        lines = response.split('\n')
        result = {}
        current_key = None
        current_value = []
        in_string = False
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            
            # Look for key-value pairs
            if ':' in line and not in_string:
                if current_key and current_value:
                    result[current_key] = '\n'.join(current_value).strip('"')
                
                parts = line.split(':', 1)
                if len(parts) == 2:
                    current_key = parts[0].strip().strip('"')
                    value_part = parts[1].strip()
                    
                    if value_part.startswith('"') and value_part.endswith('"'):
                        result[current_key] = value_part.strip('"')
                        current_key = None
                        current_value = []
                    else:
                        current_value = [value_part]
                        in_string = not value_part.endswith('"')
        
        # Add final key-value pair
        if current_key and current_value:
            result[current_key] = '\n'.join(current_value).strip('"')
        
        return RobustJSONParser._reconstruct_json(result)
    
    @staticmethod
    def _regex_extract(response: str) -> Dict[str, Any]:
        """Extract JSON using regex patterns."""
        result = {}
        
        # More aggressive regex patterns
        patterns = {
            'header': r'"header"\s*:\s*"([^"]*(?:\\.[^"]*)*)"',
            'implementation': r'"implementation"\s*:\s*"([^"]*(?:\\.[^"]*)*)"',
            'dependencies': r'"dependencies"\s*:\s*(\[[^\]]*\])',
            'compilation_instructions': r'"compilation_instructions"\s*:\s*"([^"]*)"',
            'usage_example': r'"usage_example"\s*:\s*"([^"]*)"',
            'notes': r'"notes"\s*:\s*"([^"]*)"'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, response, re.DOTALL)
            if match:
                if key == 'dependencies':
                    try:
                        result[key] = json.loads(match.group(1))
                    except:
                        result[key] = ['Eigen3']
                else:
                    result[key] = match.group(1)
        
        return RobustJSONParser._reconstruct_json(result)
    
    @staticmethod
    def _create_fallback_response(response: str) -> Dict[str, Any]:
        """Create fallback response when all parsing strategies fail."""
        # Try to extract any code-like content
        header_content = ""
        impl_content = ""
        
        # Look for C++ code patterns
        cpp_patterns = re.findall(r'#include[^}]*', response, re.DOTALL)
        if cpp_patterns:
            impl_content = '\n'.join(cpp_patterns)
        
        return {
            'header': header_content or '#ifndef GENERATED_H\n#define GENERATED_H\n\n#endif',
            'implementation': impl_content or '// Generated implementation',
            'dependencies': ['iostream'],
            'compilation_instructions': 'g++ -std=c++17 your_file.cpp',
            'usage_example': '// Usage example',
            'notes': 'Fallback generated code'
        }
    
    @staticmethod
    def _unescape_string(s: str) -> str:
        """Unescape string content."""
        return s.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r').replace('\\"', '"').replace('\\\\', '\\')
    
    @staticmethod
    def _fix_unterminated_strings(text: str) -> str:
        """Fix unterminated strings in JSON by finding and closing them."""
        try:
            # Count quotes to detect unterminated strings
            quote_count = text.count('"')
            if quote_count % 2 == 1:  # Odd number of quotes means unterminated string
                # Find the last unclosed quote and add a closing quote
                last_quote_pos = text.rfind('"')
                if last_quote_pos != -1:
                    # Check if this quote is already closed by looking for the next non-whitespace char
                    remaining = text[last_quote_pos + 1:].strip()
                    if remaining and not remaining.startswith((':', ',', '}', ']')):
                        # This quote is likely unterminated, close it
                        text = text[:last_quote_pos + 1] + '"' + text[last_quote_pos + 1:]
            
            return text
        except Exception:
            # If fixing fails, return original text
            return text
from .llm_client import LLMClient
from ..state.conversion_state import ConversionState


@dataclass
class ToolResult:
    """Standardized result format for LangGraph tools."""
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None
    metadata: Dict[str, Any] = None


class MATLABParserTool:
    """LangGraph tool for parsing MATLAB code."""
    
    def __init__(self):
        """Initialize the MATLAB parser tool."""
        self.parser = MATLABParser()
        self.logger = logger.bind(name="matlab_parser_tool")
    
    def __call__(self, matlab_code: str, file_path: Optional[str] = None) -> ToolResult:
        """
        Parse MATLAB code and extract structural information.
        
        Args:
            matlab_code: The MATLAB code to parse
            file_path: Optional file path for context
            
        Returns:
            ToolResult with parsed structure information
        """
        try:
            self.logger.debug(f"Parsing MATLAB code (length: {len(matlab_code)})")
            
            # Parse the MATLAB code
            parsed_result = self.parser.parse_project(matlab_code)
            
            # Extract additional information
            functions = parsed_result.get('functions', [])
            dependencies = parsed_result.get('dependencies', [])
            numerical_calls = parsed_result.get('numerical_calls', [])
            function_calls = parsed_result.get('function_calls', {})
            
            # Analyze complexity
            complexity = self._analyze_complexity(matlab_code)
            
            result_data = {
                'functions': functions,
                'dependencies': dependencies,
                'numerical_calls': numerical_calls,
                'function_calls': function_calls,
                'complexity': complexity,
                'file_path': file_path,
                'code_length': len(matlab_code),
                'line_count': len(matlab_code.splitlines())
            }
            
            self.logger.info(f"Successfully parsed MATLAB code: {len(functions)} functions, {len(dependencies)} dependencies")
            
            return ToolResult(
                success=True,
                data=result_data,
                metadata={'tool': 'matlab_parser', 'file_path': file_path}
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing MATLAB code: {e}")
            return ToolResult(
                success=False,
                data={},
                error=str(e),
                metadata={'tool': 'matlab_parser', 'file_path': file_path}
            )
    
    def _analyze_complexity(self, code: str) -> Dict[str, Any]:
        """Analyze code complexity metrics."""
        lines = code.splitlines()
        total_lines = len(lines)
        non_empty_lines = len([line for line in lines if line.strip()])
        
        # Count control structures
        for_loops = len(re.findall(r'\bfor\b', code, re.IGNORECASE))
        while_loops = len(re.findall(r'\bwhile\b', code, re.IGNORECASE))
        if_statements = len(re.findall(r'\bif\b', code, re.IGNORECASE))
        switch_statements = len(re.findall(r'\bswitch\b', code, re.IGNORECASE))
        
        # Determine complexity level
        total_controls = for_loops + while_loops + if_statements + switch_statements
        if total_lines > 800 or total_controls > 20:
            complexity_level = "High"
        elif total_lines > 300 or total_controls > 5:
            complexity_level = "Medium"
        else:
            complexity_level = "Low"
        
        return {
            'level': complexity_level,
            'total_lines': total_lines,
            'non_empty_lines': non_empty_lines,
            'for_loops': for_loops,
            'while_loops': while_loops,
            'if_statements': if_statements,
            'switch_statements': switch_statements,
            'total_control_structures': total_controls
        }


class LLMAnalysisTool:
    """LangGraph tool for LLM-based code analysis."""
    
    def __init__(self, llm_client: LLMClient):
        """Initialize the LLM analysis tool."""
        self.llm_client = llm_client
        self.logger = logger.bind(name="llm_analysis_tool")
    
    def __call__(self, matlab_code: str, parsed_structure: Dict[str, Any], 
                 analysis_type: str = "algorithmic") -> ToolResult:
        """
        Perform LLM-based analysis of MATLAB code.
        
        Args:
            matlab_code: The MATLAB code to analyze
            parsed_structure: Previously parsed structure information
            analysis_type: Type of analysis ("algorithmic", "performance", "complexity")
            
        Returns:
            ToolResult with LLM analysis
        """
        try:
            self.logger.debug(f"Performing LLM analysis: {analysis_type}")
            
            # Create analysis prompt based on type
            if analysis_type == "algorithmic":
                prompt = self._create_algorithmic_prompt(matlab_code, parsed_structure)
            elif analysis_type == "performance":
                prompt = self._create_performance_prompt(matlab_code, parsed_structure)
            elif analysis_type == "complexity":
                prompt = self._create_complexity_prompt(matlab_code, parsed_structure)
            else:
                prompt = self._create_general_prompt(matlab_code, parsed_structure)
            
            # Get LLM response
            response = self.llm_client.get_completion(prompt)
            
            # Parse response
            analysis_result = self._parse_llm_response(response, analysis_type)
            
            self.logger.info(f"Successfully completed LLM analysis: {analysis_type}")
            
            return ToolResult(
                success=True,
                data=analysis_result,
                metadata={'tool': 'llm_analysis', 'analysis_type': analysis_type}
            )
            
        except Exception as e:
            self.logger.error(f"Error in LLM analysis: {e}")
            return ToolResult(
                success=False,
                data={},
                error=str(e),
                metadata={'tool': 'llm_analysis', 'analysis_type': analysis_type}
            )
    
    def _create_algorithmic_prompt(self, matlab_code: str, parsed_structure: Dict[str, Any]) -> str:
        """Create prompt for algorithmic analysis."""
        functions = parsed_structure.get('functions', [])
        dependencies = parsed_structure.get('dependencies', [])
        numerical_calls = parsed_structure.get('numerical_calls', [])
        
        prompt = f"""
You are a domain expert analyzing MATLAB code for algorithmic understanding. 
Focus on the mathematical and computational aspects of the code.

CRITICAL INSTRUCTIONS:
1. Do NOT include any <think>, <thinking>, or any XML-like tags
2. Do NOT include any reasoning, analysis, or explanation
3. Do NOT include any natural language text outside the JSON
4. Return ONLY valid JSON starting with {{ and ending with }}
5. If you need to think, do it silently without outputting it
6. Your response must be parseable by json.loads() without errors

RESPONSE FORMAT: Start immediately with {{ and end with }}

MATLAB Code:
{matlab_code}

Parsed Structure:
- Functions: {', '.join(functions) if functions else 'None'}
- Dependencies: {', '.join(dependencies) if dependencies else 'None'}
- Numerical Calls: {', '.join(numerical_calls) if numerical_calls else 'None'}

Analyze the code and return a JSON object with:
{{
    "purpose": "Brief description of what the code does",
    "domain": "Domain (e.g., signal processing, linear algebra, optimization)",
    "algorithms": ["List of main algorithms used"],
    "mathematical_operations": ["List of mathematical operations"],
    "data_structures": ["List of data structures used"],
    "complexity_analysis": "Big-O complexity analysis",
    "challenges": ["Potential challenges in C++ conversion"],
    "suggestions": ["Suggestions for C++ implementation"],
    "confidence": 0.0-1.0
}}

CRITICAL: Return ONLY the JSON object, no additional text, no thinking process, no explanations.
"""
        return prompt
    
    def _create_performance_prompt(self, matlab_code: str, parsed_structure: Dict[str, Any]) -> str:
        """Create prompt for performance analysis."""
        prompt = f"""
You are a performance expert analyzing MATLAB code for optimization opportunities.

CRITICAL INSTRUCTIONS:
1. Do NOT include any <think>, <thinking>, or any XML-like tags
2. Do NOT include any reasoning, analysis, or explanation
3. Do NOT include any natural language text outside the JSON
4. Return ONLY valid JSON starting with {{ and ending with }}
5. If you need to think, do it silently without outputting it
6. Your response must be parseable by json.loads() without errors

RESPONSE FORMAT: Start immediately with {{ and end with }}

MATLAB Code:
{matlab_code}

Analyze the performance characteristics and return a JSON object with:
{{
    "bottlenecks": ["Performance bottlenecks identified"],
    "optimization_opportunities": ["Areas for optimization"],
    "memory_usage": "Memory usage analysis",
    "computational_complexity": "Computational complexity analysis",
    "parallelization_potential": "Parallelization opportunities",
    "suggestions": ["Performance optimization suggestions"],
    "confidence": 0.0-1.0
}}

CRITICAL: Return ONLY the JSON object, no additional text, no thinking process, no explanations.
"""
        return prompt
    
    def _create_complexity_prompt(self, matlab_code: str, parsed_structure: Dict[str, Any]) -> str:
        """Create prompt for complexity analysis."""
        prompt = f"""
You are a software engineering expert analyzing MATLAB code complexity.

CRITICAL INSTRUCTIONS:
1. Do NOT include any <think>, <thinking>, or any XML-like tags
2. Do NOT include any reasoning, analysis, or explanation
3. Do NOT include any natural language text outside the JSON
4. Return ONLY valid JSON starting with {{ and ending with }}
5. If you need to think, do it silently without outputting it
6. Your response must be parseable by json.loads() without errors

RESPONSE FORMAT: Start immediately with {{ and end with }}

MATLAB Code:
{matlab_code}

Analyze the code complexity and return a JSON object with:
{{
    "cyclomatic_complexity": "Cyclomatic complexity analysis",
    "maintainability_score": "Maintainability score (1-10)",
    "readability_score": "Readability score (1-10)",
    "modularity": "Modularity assessment",
    "refactoring_suggestions": ["Suggestions for refactoring"],
    "testability": "Testability assessment",
    "confidence": 0.0-1.0
}}

CRITICAL: Return ONLY the JSON object, no additional text, no thinking process, no explanations.
"""
        return prompt
    
    def _create_general_prompt(self, matlab_code: str, parsed_structure: Dict[str, Any]) -> str:
        """Create general analysis prompt."""
        prompt = f"""
You are an expert software engineer analyzing MATLAB code for C++ conversion.

CRITICAL INSTRUCTIONS:
1. Do NOT include any <think>, <thinking>, or any XML-like tags
2. Do NOT include any reasoning, analysis, or explanation
3. Do NOT include any natural language text outside the JSON
4. Return ONLY valid JSON starting with {{ and ending with }}
5. If you need to think, do it silently without outputting it
6. Your response must be parseable by json.loads() without errors

RESPONSE FORMAT: Start immediately with {{ and end with }}

MATLAB Code:
{matlab_code}

Provide a comprehensive analysis and return a JSON object with:
{{
    "overview": "General overview of the code",
    "key_features": ["Key features and functionality"],
    "conversion_difficulties": ["Potential conversion difficulties"],
    "recommendations": ["Recommendations for C++ conversion"],
    "confidence": 0.0-1.0
}}

CRITICAL: Return ONLY the JSON object, no additional text, no thinking process, no explanations.
"""
        return prompt
    
    def _parse_llm_response(self, response: str, analysis_type: str) -> Dict[str, Any]:
        """Parse LLM response into structured data."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed_data = json.loads(json_str)
                parsed_data['analysis_type'] = analysis_type
                parsed_data['raw_response'] = response
                return parsed_data
            else:
                # Fallback: create structured response from raw text
                return {
                    'analysis_type': analysis_type,
                    'raw_response': response,
                    'parsed': False,
                    'error': 'Could not extract JSON from response'
                }
        except Exception as e:
            return {
                'analysis_type': analysis_type,
                'raw_response': response,
                'parsed': False,
                'error': str(e)
            }


class CodeGenerationTool:
    """LangGraph tool for C++ code generation."""
    
    def __init__(self, llm_client: LLMClient):
        """Initialize the code generation tool."""
        self.llm_client = llm_client
        self.logger = logger.bind(name="code_generation_tool")
    
    def __call__(self, matlab_analysis: Dict[str, Any], conversion_plan: Dict[str, Any], 
                 conversion_mode: str = "result-focused") -> ToolResult:
        """
        Generate C++ code based on MATLAB analysis and conversion plan.
        
        Args:
            matlab_analysis: Analysis results from MATLAB code
            conversion_plan: Conversion plan and requirements
            conversion_mode: Conversion mode ("faithful" or "result-focused")
            
        Returns:
            ToolResult with generated C++ code
        """
        try:
            self.logger.debug(f"Generating C++ code with mode: {conversion_mode}")
            
            # Create generation prompt
            prompt = self._create_generation_prompt(matlab_analysis, conversion_plan, conversion_mode)
            
            # Generate code using legacy's direct LLM call approach
            try:
                raw_response = self.llm_client.get_completion(prompt)
                if not raw_response:
                    self.logger.warning("LLM returned empty response")
                    raise ValueError("Empty LLM response")
                
                # Parse using legacy-style code block extraction
                generated_code = self._parse_generated_code(raw_response, conversion_mode)
                
            except Exception as e:
                self.logger.error(f"Code generation failed: {e}")
                generated_code = {
                    'header': '',
                    'implementation': '',
                    'dependencies': ['Eigen3'],
                    'compilation_instructions': 'g++ -std=c++17 -I/path/to/eigen main.cpp',
                    'usage_example': '// Usage example not available',
                    'notes': f'Generation failed: {e}',
                    'conversion_mode': conversion_mode,
                    'raw_response': ''
                }
            self.logger.debug(f"Parsed generated code type: {type(generated_code)}")
            if generated_code is None:
                self.logger.error("Generated code is None!")
            
            # Check if this is a single file or multi-file result
            if 'files' in generated_code:
                # Multi-file project result
                self.logger.info(f"Successfully generated C++ code: {len(generated_code.get('files', {}))} files")
            else:
                # Single file result - check if we have header and/or implementation
                file_count = 0
                if generated_code.get('header'):
                    file_count += 1
                if generated_code.get('implementation'):
                    file_count += 1
                self.logger.info(f"Successfully generated C++ code: {file_count} file(s)")
            
            return ToolResult(
                success=True,
                data=generated_code,
                metadata={'tool': 'code_generation', 'conversion_mode': conversion_mode}
            )
            
        except Exception as e:
            self.logger.error(f"Error in code generation: {e}")
            return ToolResult(
                success=False,
                data={},
                error=str(e),
                metadata={'tool': 'code_generation', 'conversion_mode': conversion_mode}
            )
    
    def _create_generation_prompt(self, matlab_analysis: Dict[str, Any], 
                                 conversion_plan: Dict[str, Any], conversion_mode: str) -> str:
        """Create prompt for C++ code generation with enhanced legacy-inspired guidelines."""
        
        # Extract key information
        functions = matlab_analysis.get('functions', [])
        dependencies = matlab_analysis.get('dependencies', [])
        numerical_calls = matlab_analysis.get('numerical_calls', [])
        algorithms = matlab_analysis.get('algorithms', [])
        challenges = matlab_analysis.get('challenges', [])
        source_code = matlab_analysis.get('source_code', '')
        
        system_prompt = self._create_system_prompt()
        
        # Legacy-inspired comprehensive guidelines
        base_guidelines = (
            "- Use Eigen matrices/arrays (Eigen::MatrixXd, Eigen::ArrayXd) to represent MATLAB matrices and vectors.\n"
            "- Convert MATLAB's 1-based indexing to C++'s 0-based indexing. Adjust loop bounds accordingly.\n"
            "- Avoid calling '.inverse()'; instead use factorisation methods: '.ldlt().solve(rhs)', '.llt().solve(rhs)', or '.partialPivLu().solve(rhs)'.\n"
            "- For eigen decomposition of symmetric matrices, use Eigen::SelfAdjointEigenSolver and select the eigenvector associated with the smallest eigenvalue (column 0). For non-symmetric matrices, use Eigen::EigenSolver.\n"
            "- Map MATLAB built-ins to C++: zeros -> Eigen::MatrixXd::Zero(), eye -> Eigen::MatrixXd::Identity(), size(A,1) -> A.rows(), size(A,2) -> A.cols(), length(x) -> x.size().\n"
            "- Use FFTW or Eigen's FFT module for FFT/filtering, and OpenCV for image operations. Wrap external libraries appropriately.\n"
            "- Write small, single-responsibility functions or classes. Pass inputs by const reference; avoid global variables. Use RAII containers (std::vector) or smart pointers instead of raw pointers.\n"
            "- Wrap heavy computations and file I/O in try/catch blocks. After solving linear systems or eigenproblems, check solver.info() == Success before using results.\n"
            "- Do not use 'using namespace std;' in headers. Qualify names explicitly (std::vector, std::string).\n"
            "- Follow modern C++ style: brace initialisers, range-based for loops, auto where appropriate.\n"
            "- Preserve the EXACT algorithmic structure including nested loops and iteration patterns from MATLAB.\n"
            "- Implement the precise mathematical operations in the correct sequence.\n"
            "- Do NOT add operations that are not present in the original MATLAB code.\n"
            "- Maintain the exact data flow and transformations.\n"
        )
        
        prompt_parts = [system_prompt]
        prompt_parts.append("/no_think")
        prompt_parts.append("You are an expert C++ developer tasked with translating MATLAB functions into modern C++ code using Eigen and other appropriate libraries.")
        
        # Add conversion mode specific instructions
        if conversion_mode == "faithful":
            prompt_parts.append("CONVERSION MODE: FAITHFUL - Prioritize bit-level equivalence to MATLAB code.")
            prompt_parts.append("CRITICAL REQUIREMENTS FOR FAITHFUL MODE:")
            prompt_parts.append("- Preserve the EXACT algorithmic structure including nested loops and iteration patterns")
            prompt_parts.append("- Implement the precise mathematical operations in the correct sequence")
            prompt_parts.append("- Do NOT add operations that are not present in the original MATLAB code")
            prompt_parts.append("- Maintain the exact data flow and transformations")
            prompt_parts.append("- Focus on reproducing the same computational results as MATLAB")
        else:  # result-focused
            prompt_parts.append("CONVERSION MODE: RESULT-FOCUSED - Prioritize working, efficient C++ code.")
            prompt_parts.append("CRITICAL REQUIREMENTS FOR RESULT-FOCUSED MODE:")
            prompt_parts.append("- Focus on producing correct computational results using C++ best practices")
            prompt_parts.append("- Optimize for performance, memory efficiency, and maintainability")
            prompt_parts.append("- Use appropriate C++ libraries and modern C++ features")
            prompt_parts.append("- Ensure the C++ code produces equivalent results to MATLAB (not necessarily bit-identical)")
            prompt_parts.append("- Feel free to restructure algorithms for better C++ performance")
            prompt_parts.append("- Prioritize numerical stability and error handling")
        
        prompt_parts.append("Follow all of the guidelines below to ensure numerical stability, performance, and maintainability:\n" + base_guidelines)
        
        # Summarise MATLAB file
        summary_lines = []
        if functions:
            summary_lines.append("Functions defined: " + ", ".join(functions) + ".")
        if dependencies:
            summary_lines.append("External calls: " + ", ".join(dependencies) + ".")
        if numerical_calls:
            summary_lines.append("Numerical operations: " + ", ".join(numerical_calls) +
                                 ". Map these to appropriate C++ equivalents (Eigen, FFTW, OpenCV, etc.).")
        if summary_lines:
            prompt_parts.append("\nMATLAB file summary:\n" + "\n".join(summary_lines))
        
        # Add actual MATLAB source code - CRITICAL for accurate conversion
        if source_code:
            prompt_parts.append("\nORIGINAL MATLAB SOURCE CODE:")
            prompt_parts.append("```matlab")
            prompt_parts.append(source_code)
            prompt_parts.append("```")
            prompt_parts.append("\nCRITICAL: You MUST translate this exact MATLAB code to C++. Do not make up your own implementation!")
        
        # Incorporate conversion plan with algorithmic mapping
        if conversion_plan:
            libs = conversion_plan.get("dependencies", [])
            if libs:
                prompt_parts.append("Required C++ libraries: " + ", ".join(libs) + ".")
            strategy = conversion_plan.get("conversion_strategy", "")
            if strategy:
                prompt_parts.append("Overall conversion strategy: " + strategy)
            
            # Add algorithmic mapping information
            algorithmic_mapping = conversion_plan.get("algorithmic_mapping", {})
            if algorithmic_mapping:
                prompt_parts.append("\nAlgorithmic mapping:")
                for matlab_op, cpp_equiv in algorithmic_mapping.items():
                    prompt_parts.append(f"  {matlab_op} -> {cpp_equiv}")
            
            # Add data flow preservation information
            data_flow = conversion_plan.get("data_flow_preservation", {})
            if data_flow:
                prompt_parts.append("\nData flow preservation:")
                for key, value in data_flow.items():
                    prompt_parts.append(f"  {key}: {value}")
            
            steps = conversion_plan.get("conversion_steps", [])
            if steps:
                prompt_parts.append("Recommended steps:\n" + "\n".join(f"- {step}" for step in steps))
        
        # Use legacy's direct approach - force code blocks format
        prompt_parts.append("\nIMPORTANT: Generate ONLY the C++ code translation. Do NOT include thinking process or explanations.")
        prompt_parts.append("Return EXACTLY two fenced code blocks:")
        prompt_parts.append("1. First block: ```cpp (header file)")
        prompt_parts.append("2. Second block: ```cpp (implementation file)")
        prompt_parts.append("Do NOT include any text before, between, or after the code blocks. ")
        prompt_parts.append("Do NOT include <think> tags or explanatory text. ")
        prompt_parts.append("Just the two code blocks with proper C++ code.")
        
        return "\n\n".join(prompt_parts)
    
    
    def _parse_generated_code(self, response: str, conversion_mode: str) -> Dict[str, Any]:
        """Parse generated C++ code from LLM response."""
        try:
            # Use legacy's simple code block extraction
            header, implementation = self._extract_code_blocks_legacy(response)
            
            if not header and not implementation:
                self.logger.warning("No code blocks extracted from LLM response, trying JSON fallback")
                self.logger.debug(f"LLM response length: {len(response)}")
                self.logger.debug(f"LLM response preview: {response[:200]}...")
                
                # Try JSON parsing as fallback with robust recovery
                try:
                    # Remove <think> tags first
                    clean_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
                    
                    # Try to parse as JSON
                    json_data = json.loads(clean_response)
                    
                    if 'header' in json_data or 'implementation' in json_data:
                        header = json_data.get('header', '')
                        implementation = json_data.get('implementation', '')
                        self.logger.info("Successfully parsed JSON fallback response")
                    else:
                        raise ValueError("JSON missing required fields")
                        
                except Exception as json_error:
                    self.logger.warning(f"JSON fallback also failed: {json_error}")
                    
                    # Try robust JSON recovery for large responses
                    try:
                        recovered = self._recover_json_from_large_response(clean_response)
                        if recovered:
                            header = recovered.get('header', '')
                            implementation = recovered.get('implementation', '')
                            self.logger.info("Successfully recovered JSON from large response")
                        else:
                            raise ValueError("JSON recovery failed")
                    except Exception as recovery_error:
                        self.logger.warning(f"JSON recovery also failed: {recovery_error}")
                        # Return empty result
                        return {
                            'header': '',
                            'implementation': '',
                            'dependencies': ['Eigen3'],
                            'compilation_instructions': 'g++ -std=c++17 -I/path/to/eigen main.cpp',
                            'usage_example': '// Usage example not available',
                            'notes': f'All parsing methods failed: {json_error}, {recovery_error}',
                            'conversion_mode': conversion_mode,
                            'raw_response': response
                        }
            
            # Generate dependencies and other metadata from the code
            dependencies = self._extract_dependencies_from_code(implementation)
            compilation_instructions = self._generate_compilation_instructions(implementation)
            usage_example = self._generate_usage_example(implementation)
            
            result = {
                'header': header,
                'implementation': implementation,
                'dependencies': dependencies,
                'compilation_instructions': compilation_instructions,
                'usage_example': usage_example,
                'notes': 'Generated using JSON fallback parsing',
                'conversion_mode': conversion_mode,
                'raw_response': response
            }
            
            self.logger.debug("Successfully parsed response using JSON fallback")
            return result
                
        except Exception as e:
            self.logger.warning(f"Error parsing generated code: {e}")
            self.logger.debug(f"Raw response (first 200 chars): {response[:200]}")
            
            # Return minimal fallback
            return {
                'header': '',
                'implementation': '',
                'dependencies': ['Eigen3'],
                'compilation_instructions': 'g++ -std=c++17 -I/path/to/eigen main.cpp',
                'usage_example': '// Usage example not available',
                'notes': f'Parsing failed: {e}',
                'conversion_mode': conversion_mode,
                'raw_response': response
            }
            
            # Try to extract C++ code from JSON-like structure even if parsing failed
            if response:
                # Look for JSON structure and try to extract header and implementation
                # Use a more flexible regex that handles multiline strings
                header_match = re.search(r'"header":\s*"((?:[^"\\]|\\.)*)"', response, re.DOTALL)
                impl_match = re.search(r'"implementation":\s*"((?:[^"\\]|\\.)*)"', response, re.DOTALL)
                
                if header_match:
                    header_content = header_match.group(1)
                    # Unescape the JSON string
                    header_content = header_content.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
                    fallback_header = header_content
                
                if impl_match:
                    impl_content = impl_match.group(1)
                    # Unescape the JSON string
                    impl_content = impl_content.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
                    fallback_implementation = impl_content
                
                # If the above regex didn't work, try a simpler approach
                if not fallback_header and not fallback_implementation:
                    # Look for the actual JSON structure with proper multiline handling
                    lines = response.split('\n')
                    in_header = False
                    in_implementation = False
                    header_content = ""
                    impl_content = ""
                    
                    for i, line in enumerate(lines):
                        line = line.strip()
                        if line.startswith('"header":'):
                            in_header = True
                            in_implementation = False
                            # Extract content after the colon
                            header_start = line.find('"', line.find('"header":'))
                            if header_start != -1:
                                header_content = line[header_start+1:]
                                if header_content.endswith('"'):
                                    header_content = header_content[:-1]
                        elif line.startswith('"implementation":'):
                            in_header = False
                            in_implementation = True
                            # Extract content after the colon
                            impl_start = line.find('"', line.find('"implementation":'))
                            if impl_start != -1:
                                impl_content = line[impl_start+1:]
                                if impl_content.endswith('"'):
                                    impl_content = impl_content[:-1]
                        elif in_header and line:
                            # Continue collecting header content
                            if line.startswith('"') and line.endswith('"'):
                                header_content += line[1:-1]
                            elif line.startswith('"'):
                                header_content += line[1:]
                            elif line.endswith('"'):
                                header_content += line[:-1]
                            else:
                                header_content += line
                        elif in_implementation and line:
                            # Continue collecting implementation content
                            if line.startswith('"') and line.endswith('"'):
                                impl_content += line[1:-1]
                            elif line.startswith('"'):
                                impl_content += line[1:]
                            elif line.endswith('"'):
                                impl_content += line[:-1]
                            else:
                                impl_content += line
                    
                    if header_content:
                        header_content = header_content.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
                        fallback_header = header_content
                    if impl_content:
                        impl_content = impl_content.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
                        fallback_implementation = impl_content
                
                # If JSON extraction failed, try to find C++ code patterns
                if not fallback_header and not fallback_implementation and ('#include' in response or 'namespace' in response):
                    # Try to extract header and implementation from raw response
                    lines = response.split('\n')
                    in_header = False
                    in_implementation = False
                    header_lines = []
                    impl_lines = []
                    
                    for line in lines:
                        if line.strip().startswith('#'):
                            in_header = True
                            in_implementation = False
                        elif line.strip().startswith(('int ', 'void ', 'double ', 'float ', 'bool ', 'class ', 'struct ')):
                            in_implementation = True
                            in_header = False
                        
                        if in_header and not line.strip().startswith('<'):
                            header_lines.append(line)
                        elif in_implementation and not line.strip().startswith('<'):
                            impl_lines.append(line)
                    
                    if header_lines:
                        fallback_header = '\n'.join(header_lines)
                    if impl_lines:
                        fallback_implementation = '\n'.join(impl_lines)
            
            return {
                'header': fallback_header,
                'implementation': fallback_implementation,
                'dependencies': ['Eigen3'],
                'compilation_instructions': 'g++ -std=c++17 -I/path/to/eigen your_file.cpp',
                'usage_example': 'See implementation comments',
                'notes': f'Generated with parsing error: {e}',
                'conversion_mode': conversion_mode,
                'raw_response': response
            }
    
    def _extract_code_blocks_legacy(self, llm_output: str) -> Tuple[str, str]:
        """
        Extract header and implementation code from the LLM output using code block extraction.
        
        The LLM is expected to return two fenced code blocks, one for the
        header (.h) and one for the implementation (.cpp). This method
        separates them by detecting the first two code fences.
        
        Args:
            llm_output: The raw LLM response.
            
        Returns:
            A tuple (header_code, implementation_code). If either block
            cannot be found, it returns an empty string for that block.
        """
        # First, try to extract code blocks from the entire response
        fences = re.findall(r"```(?:\w*\n)?(.*?)```", llm_output, re.DOTALL)
        
        # If no code blocks found, try to extract from thinking tags
        if not fences:
            # Look for code blocks inside <think> tags
            think_sections = re.findall(r"<think>(.*?)</think>", llm_output, re.DOTALL)
            for think_section in think_sections:
                think_fences = re.findall(r"```(?:\w*\n)?(.*?)```", think_section, re.DOTALL)
                fences.extend(think_fences)
        
        if len(fences) >= 2:
            header_code = fences[0].strip()
            implementation_code = fences[1].strip()
        elif len(fences) == 1:
            # Only one block returned; assume it's implementation
            header_code = ""
            implementation_code = fences[0].strip()
        else:
            header_code = ""
            implementation_code = ""
        return header_code, implementation_code

    def _recover_json_from_large_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Robust JSON recovery for large responses that fail standard JSON parsing.
        This handles cases where the LLM returns malformed JSON due to large code content.
        """
        try:
            # Method 1: Try to extract JSON fields using regex patterns
            header_match = re.search(r'"header"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', response, re.DOTALL)
            implementation_match = re.search(r'"implementation"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', response, re.DOTALL)
            
            if header_match and implementation_match:
                header_content = header_match.group(1)
                implementation_content = implementation_match.group(1)
                
                # Unescape JSON string content
                header_content = header_content.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')
                implementation_content = implementation_content.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')
                
                return {
                    'header': header_content,
                    'implementation': implementation_content
                }
            
            # Method 2: Try to find JSON boundaries and extract manually
            json_start = response.find('{')
            json_end = response.rfind('}')
            
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_content = response[json_start:json_end + 1]
                
                # Try to fix common JSON issues
                # Remove trailing commas
                json_content = re.sub(r',(\s*[}\]])', r'\1', json_content)
                
                # Try to parse the cleaned JSON
                try:
                    return json.loads(json_content)
                except:
                    pass
            
            # Method 3: Extract content between quotes for each field
            lines = response.split('\n')
            header_lines = []
            implementation_lines = []
            current_field = None
            in_string = False
            escape_next = False
            
            for line in lines:
                if '"header"' in line:
                    current_field = 'header'
                    # Extract from the line
                    start = line.find('"header"')
                    if start != -1:
                        after_colon = line.find(':', start)
                        if after_colon != -1:
                            after_quote = line.find('"', after_colon)
                            if after_quote != -1:
                                header_lines.append(line[after_quote + 1:])
                                in_string = True
                elif '"implementation"' in line:
                    current_field = 'implementation'
                    # Extract from the line
                    start = line.find('"implementation"')
                    if start != -1:
                        after_colon = line.find(':', start)
                        if after_colon != -1:
                            after_quote = line.find('"', after_colon)
                            if after_quote != -1:
                                implementation_lines.append(line[after_quote + 1:])
                                in_string = True
                elif in_string and current_field:
                    if current_field == 'header':
                        header_lines.append(line)
                    elif current_field == 'implementation':
                        implementation_lines.append(line)
                    
                    # Check if string ends
                    if line.strip().endswith('"') and not line.strip().endswith('\\"'):
                        in_string = False
                        current_field = None
            
            # Clean up the extracted content
            if header_lines or implementation_lines:
                header_content = '\n'.join(header_lines).rstrip('",').replace('\\"', '"').replace('\\n', '\n')
                implementation_content = '\n'.join(implementation_lines).rstrip('",').replace('\\"', '"').replace('\\n', '\n')
                
                return {
                    'header': header_content,
                    'implementation': implementation_content
                }
            
            return None
            
        except Exception as e:
            self.logger.debug(f"JSON recovery failed: {e}")
            return None

    def _parse_code_blocks(self, response: str) -> Dict[str, Any]:
        """Parse code blocks from LLM response."""
        try:
            # Look for code blocks with cpp language specification
            cpp_blocks = re.findall(r"```cpp\s*\n(.*?)\n```", response, re.DOTALL)
            
            if len(cpp_blocks) >= 2:
                # Two or more code blocks - first is header, second is implementation
                header_code = cpp_blocks[0].strip()
                implementation_code = cpp_blocks[1].strip()
                
                return {
                    'header': header_code,
                    'implementation': implementation_code,
                    'dependencies': self._extract_dependencies_from_code(implementation_code),
                    'compilation_instructions': self._generate_compilation_instructions(implementation_code),
                    'usage_example': self._generate_usage_example(implementation_code),
                    'notes': 'Generated from code blocks format'
                }
            elif len(cpp_blocks) == 1:
                # Single code block - assume it's implementation
                implementation_code = cpp_blocks[0].strip()
                
                return {
                    'header': self._generate_header_from_implementation(implementation_code),
                    'implementation': implementation_code,
                    'dependencies': self._extract_dependencies_from_code(implementation_code),
                    'compilation_instructions': self._generate_compilation_instructions(implementation_code),
                    'usage_example': self._generate_usage_example(implementation_code),
                    'notes': 'Generated from single code block format'
                }
            
            return {}
            
        except Exception as e:
            self.logger.debug(f"Code block parsing failed: {e}")
            return {}
    
    def _extract_dependencies_from_code(self, code: str) -> List[str]:
        """Extract dependencies from C++ code."""
        dependencies = ['Eigen3']  # Default
        
        if '#include <iostream>' in code:
            dependencies.append('iostream')
        if '#include <vector>' in code:
            dependencies.append('vector')
        if '#include <chrono>' in code:
            dependencies.append('chrono')
        if 'Eigen::' in code:
            dependencies.append('Eigen3')
        if 'std::' in code:
            dependencies.append('c++17')
            
        return dependencies
    
    def _generate_compilation_instructions(self, code: str) -> str:
        """Generate compilation instructions from code."""
        if 'Eigen::' in code:
            return "g++ -std=c++17 -I/path/to/eigen main.cpp -o main"
        else:
            return "g++ -std=c++17 main.cpp -o main"
    
    def _generate_usage_example(self, code: str) -> str:
        """Generate usage example from code."""
        # Extract function names
        functions = re.findall(r'(\w+)\s*\([^)]*\)\s*{', code)
        if functions:
            return f"// Usage: {functions[0]}();"
        return "// Usage example not available"
    
    def _generate_header_from_implementation(self, implementation: str) -> str:
        """Generate a basic header from implementation."""
        # Extract function declarations
        functions = re.findall(r'(\w+::\w+|\w+)\s*\([^)]*\)\s*{', implementation)
        
        header = '#ifndef MAIN_H\n#define MAIN_H\n\n'
        
        # Add includes
        if 'Eigen::' in implementation:
            header += '#include <Eigen/Dense>\n'
        if 'std::' in implementation:
            header += '#include <iostream>\n'
        
        header += '\n'
        
        # Add function declarations
        for func in functions:
            # Convert function definition to declaration
            func_decl = func.replace('{', ';')
            header += f'{func_decl}\n'
        
        header += '\n#endif\n'
        return header
    
    def _extract_fallback_content(self, response: str) -> Dict[str, Any]:
        """Extract content using fallback methods."""
        fallback_header = ''
        fallback_implementation = response or ''
        
        # Try to extract C++ code from JSON-like structure even if parsing failed
        if response:
            # Look for JSON structure and try to extract header and implementation
            header_match = re.search(r'"header":\s*"((?:[^"\\]|\\.)*)"', response, re.DOTALL)
            impl_match = re.search(r'"implementation":\s*"((?:[^"\\]|\\.)*)"', response, re.DOTALL)
            
            if header_match:
                header_content = header_match.group(1)
                # Unescape common escape sequences
                header_content = header_content.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"')
                fallback_header = header_content
            
            if impl_match:
                impl_content = impl_match.group(1)
                # Unescape common escape sequences
                impl_content = impl_content.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"')
                fallback_implementation = impl_content
        
        return {
            'header': fallback_header,
            'implementation': fallback_implementation,
            'dependencies': ['Eigen3'],  # Default dependency
            'compilation_instructions': 'g++ -std=c++17 -I/path/to/eigen main.cpp',
            'usage_example': '// Usage example not available',
            'notes': 'Parsed with fallback method'
        }
    
    def _create_system_prompt(self) -> str:
        """Create system-level prompt to prevent thinking content."""
        return """
You are a JSON-only response generator. You must follow these rules STRICTLY:

1. NEVER include <think>, <thinking>, or any XML-like tags
2. NEVER include reasoning, analysis, or explanations
3. ALWAYS start your response with { and end with }
4. Your response must be valid JSON that can be parsed by json.loads()
5. If you need to think, do it silently without outputting anything

EXAMPLE OF CORRECT RESPONSE:
{
    "header": "#ifndef MAIN_H\\n#define MAIN_H\\n\\n#include <iostream>\\n\\nnamespace example {\\n    void function();\\n}\\n\\n#endif",
    "implementation": "#include \\"main.h\\"\\n\\nnamespace example {\\n    void function() {\\n        std::cout << \\"Hello\\";\\n    }\\n}",
    "dependencies": ["iostream"],
    "compilation_instructions": "g++ -std=c++17 main.cpp",
    "usage_example": "example::function();",
    "notes": "Simple example"
}

EXAMPLE OF INCORRECT RESPONSE:
<think>I need to generate C++ code</think>
{
    "header": "..."
}

Remember: Your response must be parseable JSON only!
"""


class LLMConfigOptimizer:
    """LLM configuration optimizer to reduce thinking content and improve JSON output."""
    
    @staticmethod
    def get_optimized_config() -> Dict[str, Any]:
        """Get optimized LLM configuration to reduce thinking content."""
        return {
            "temperature": 0.1,  # Lower temperature for more deterministic output
            "top_p": 0.9,       # Focus on most likely tokens
            "max_tokens": 4000,  # Sufficient for JSON responses
            "stop": ["<think>", "<thinking>", "```"],  # Stop at thinking tags
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "system_message": """
You are a JSON-only response generator. Follow these rules STRICTLY:
1. NEVER include <think>, <thinking>, or any XML-like tags
2. NEVER include reasoning, analysis, or explanations  
3. ALWAYS start your response with { and end with }
4. Your response must be valid JSON that can be parsed by json.loads()
5. If you need to think, do it silently without outputting anything
"""
        }


class ResponseValidator:
    """Response validator with retry mechanism for LLM responses."""
    
    @staticmethod
    def validate_and_retry(llm_client, prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        """Validate LLM response and retry if invalid."""
        for attempt in range(max_retries):
            try:
                # Get response from LLM (without optimized config for now)
                response = llm_client.get_completion(prompt)
                
                # Validate response
                if ResponseValidator._is_valid_json(response):
                    parsed = RobustJSONParser.parse_llm_json(response)
                    if ResponseValidator._has_required_fields(parsed):
                        return parsed
                
                # If validation fails, try with enhanced prompt
                if attempt < max_retries - 1:
                    enhanced_prompt = ResponseValidator._add_validation_hints(prompt, attempt + 1)
                    prompt = enhanced_prompt
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    continue
        
        # Final fallback
        return ResponseValidator._create_fallback_response()
    
    @staticmethod
    def _is_valid_json(response: str) -> bool:
        """Check if response is valid JSON."""
        try:
            cleaned = JSONResponseCleaner.clean_json_response(response)
            json.loads(cleaned)
            return True
        except:
            return False
    
    @staticmethod
    def _has_required_fields(parsed: Dict[str, Any]) -> bool:
        """Check if parsed response has required fields."""
        required_fields = ['header', 'implementation']
        return all(field in parsed for field in required_fields)
    
    @staticmethod
    def _add_validation_hints(prompt: str, attempt: int) -> str:
        """Add validation hints to prompt for retry attempts."""
        hints = [
            "CRITICAL: Your previous response was not valid JSON. Fix this immediately.",
            "CRITICAL: Your previous response contained thinking content. Remove all <think> tags.",
            "CRITICAL: Your previous response was malformed. Return ONLY valid JSON."
        ]
        
        hint = hints[min(attempt - 1, len(hints) - 1)]
        return f"{hint}\n\n{prompt}"
    
    @staticmethod
    def _create_fallback_response() -> Dict[str, Any]:
        """Create fallback response when all retries fail."""
        return {
            'header': '#ifndef GENERATED_H\n#define GENERATED_H\n\n#endif',
            'implementation': '// Generated implementation',
            'dependencies': ['iostream'],
            'compilation_instructions': 'g++ -std=c++17 your_file.cpp',
            'usage_example': '// Usage example',
            'notes': 'Fallback generated code'
        }


class QualityAssessmentTool:
    """LangGraph tool for code quality assessment."""
    
    def __init__(self, llm_client: LLMClient):
        """Initialize the quality assessment tool."""
        self.llm_client = llm_client
        self.logger = logger.bind(name="quality_assessment_tool")
    
    def __call__(self, cpp_code: str, matlab_code: str, conversion_plan: Dict[str, Any]) -> ToolResult:
        """
        Assess the quality of generated C++ code.
        
        Args:
            cpp_code: The generated C++ code
            matlab_code: The original MATLAB code
            conversion_plan: The conversion plan used
            
        Returns:
            ToolResult with quality assessment
        """
        try:
            self.logger.debug("Assessing C++ code quality")
            
            # Create assessment prompt
            prompt = self._create_assessment_prompt(cpp_code, matlab_code, conversion_plan)
            
            # Get LLM assessment
            response = self.llm_client.get_completion(prompt)
            
            # Parse assessment
            assessment = self._parse_assessment_response(response)
            
            self.logger.info(f"Quality assessment complete: {assessment.get('overall_score', 0)}/10")
            
            return ToolResult(
                success=True,
                data=assessment,
                metadata={'tool': 'quality_assessment'}
            )
            
        except Exception as e:
            self.logger.error(f"Error in quality assessment: {e}")
            return ToolResult(
                success=False,
                data={},
                error=str(e),
                metadata={'tool': 'quality_assessment'}
            )
    
    def _create_assessment_prompt(self, cpp_code: str, matlab_code: str, 
                                 conversion_plan: Dict[str, Any]) -> str:
        """Create prompt for quality assessment."""
        prompt = f"""
You are a C++ quality expert assessing code converted from MATLAB.

CRITICAL INSTRUCTIONS:
1. Do NOT include any <think>, <thinking>, or any XML-like tags
2. Do NOT include any reasoning, analysis, or explanation
3. Do NOT include any natural language text outside the JSON
4. Return ONLY valid JSON starting with {{ and ending with }}
5. If you need to think, do it silently without outputting it
6. Your response must be parseable by json.loads() without errors

RESPONSE FORMAT: Start immediately with {{ and end with }}

Original MATLAB Code:
{matlab_code}

Generated C++ Code:
{cpp_code}

Conversion Plan:
{json.dumps(conversion_plan, indent=2)}

Assess the C++ code quality across these categories and return a JSON object:
{{
    "algorithmic_correctness": {{
        "score": 0-10,
        "issues": ["List of algorithmic issues"],
        "suggestions": ["Suggestions for improvement"]
    }},
    "performance": {{
        "score": 0-10,
        "issues": ["List of performance issues"],
        "suggestions": ["Performance optimization suggestions"]
    }},
    "error_handling": {{
        "score": 0-10,
        "issues": ["List of error handling issues"],
        "suggestions": ["Error handling improvements"]
    }},
    "code_style": {{
        "score": 0-10,
        "issues": ["List of style issues"],
        "suggestions": ["Style improvements"]
    }},
    "maintainability": {{
        "score": 0-10,
        "issues": ["List of maintainability issues"],
        "suggestions": ["Maintainability improvements"]
    }},
    "overall_score": 0-10,
    "summary": "Overall assessment summary",
    "recommendations": ["Key recommendations for improvement"]
}}

CRITICAL: Return ONLY the JSON object, no additional text, no thinking process, no explanations.
"""
        return prompt
    
    def _parse_assessment_response(self, response: str) -> Dict[str, Any]:
        """Parse quality assessment response."""
        # Define categories outside try block for exception handler access
        categories = ['algorithmic_correctness', 'performance', 'error_handling', 
                     'code_style', 'maintainability']
        
        try:
            # Clean the response first
            cleaned_response = JSONResponseCleaner.clean_json_response(response)
            
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                try:
                    parsed_data = json.loads(json_str)
                except json.JSONDecodeError as e:
                    # Try aggressive cleaning for assessment JSON too
                    lines = json_str.split('\n')
                    json_lines = []
                    for line in lines:
                        line = line.strip()
                        if (line.startswith(('{', '"', '}', '[', ']', ',')) or 
                            line.startswith(('    ', '\t')) or  # Indented lines
                            line == '' or  # Empty lines
                            ':' in line):  # Key-value pairs
                            json_lines.append(line)
                    
                    cleaned_json = '\n'.join(json_lines)
                    try:
                        parsed_data = json.loads(cleaned_json)
                        self.logger.info("Successfully parsed assessment JSON after aggressive cleaning")
                    except json.JSONDecodeError as e2:
                        self.logger.warning(f"Assessment JSON parsing failed even after aggressive cleaning: {e2}")
                        # Fall through to fallback
                        json_match = None
                
                if json_match:  # Only process if we successfully parsed JSON
                    # Ensure all required categories are present
                    for category in categories:
                        if category not in parsed_data:
                            parsed_data[category] = {
                                'score': 5.0,
                                'issues': [],
                                'suggestions': []
                            }
                    
                    # Ensure overall score is present
                    if 'overall_score' not in parsed_data:
                        scores = [parsed_data[cat]['score'] for cat in categories]
                        parsed_data['overall_score'] = sum(scores) / len(scores)
                    
                    parsed_data['raw_response'] = response
                    return parsed_data
            else:
                # Fallback assessment
                return {
                    'algorithmic_correctness': {'score': 5.0, 'issues': [], 'suggestions': []},
                    'performance': {'score': 5.0, 'issues': [], 'suggestions': []},
                    'error_handling': {'score': 5.0, 'issues': [], 'suggestions': []},
                    'code_style': {'score': 5.0, 'issues': [], 'suggestions': []},
                    'maintainability': {'score': 5.0, 'issues': [], 'suggestions': []},
                    'overall_score': 5.0,
                    'summary': 'Assessment completed with fallback parsing',
                    'recommendations': ['Review generated code manually'],
                    'raw_response': response
                }
                
        except Exception as e:
            self.logger.warning(f"Error parsing assessment response: {e}")
            return {
                'algorithmic_correctness': {'score': 5.0, 'issues': [], 'suggestions': []},
                'performance': {'score': 5.0, 'issues': [], 'suggestions': []},
                'error_handling': {'score': 5.0, 'issues': [], 'suggestions': []},
                'code_style': {'score': 5.0, 'issues': [], 'suggestions': []},
                'maintainability': {'score': 5.0, 'issues': [], 'suggestions': []},
                'overall_score': 5.0,
                'summary': f'Assessment failed with error: {e}',
                'recommendations': ['Review generated code manually'],
                'raw_response': response,
                'error': str(e)
            }


class ToolRegistry:
    """Registry for managing LangGraph tools."""
    
    def __init__(self):
        """Initialize the tool registry."""
        self.tools = {}
        self.logger = logger.bind(name="tool_registry")
    
    def register_tool(self, name: str, tool: Any):
        """Register a tool in the registry."""
        self.tools[name] = tool
        self.logger.info(f"Registered tool: {name}")
    
    def get_tool(self, name: str) -> Optional[Any]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all registered tools."""
        return list(self.tools.keys())
    
    def create_tool_set(self, llm_client: LLMClient) -> Dict[str, Any]:
        """Create a complete set of tools for LangGraph agents."""
        tools = {
            'matlab_parser': MATLABParserTool(),
            'llm_analysis': LLMAnalysisTool(llm_client),
            'code_generation': CodeGenerationTool(llm_client),
            'quality_assessment': QualityAssessmentTool(llm_client)
        }
        
        # Register all tools
        for name, tool in tools.items():
            self.register_tool(name, tool)
        
        self.logger.info(f"Created tool set with {len(tools)} tools")
        return tools
