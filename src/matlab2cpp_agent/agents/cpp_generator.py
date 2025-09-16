#!/usr/bin/env python3
"""
C++ Generator Agent

This agent is responsible for generating C++ code from MATLAB analysis
and conversion plans. It uses LLM to create complete C++ implementations
following the specified architecture and requirements.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger

from matlab2cpp_agent.tools.llm_client import create_llm_client
from matlab2cpp_agent.utils.config import LLMConfig
from matlab2cpp_agent.agents.conversion_planner import ConversionPlan

class CppGeneratorAgent:
    """Agent responsible for generating C++ code from MATLAB analysis."""
    
    def __init__(self, llm_config: LLMConfig):
        """Initialize the C++ generator agent."""
        self.llm_config = llm_config
        self.llm_client = create_llm_client(llm_config)
        self.logger = logger.bind(name="cpp_generator_agent")
        self.logger.info("C++ Generator Agent initialized")
    
    def generate_cpp_code(self, 
                         matlab_analysis: Dict[str, Any],
                         conversion_plan: ConversionPlan,
                         project_name: str,
                         cpp_standard: str = "C++17",
                         target_quality_score: float = 7.0) -> Optional[Dict[str, str]]:
        """
        Generate C++ code based on MATLAB analysis and conversion plan.
        
        Args:
            matlab_analysis: Results from MATLAB content analysis
            conversion_plan: Comprehensive conversion plan
            project_name: Name for the C++ project
            cpp_standard: C++ standard to use
            target_quality_score: Target quality score for generation
            
        Returns:
            Dictionary with 'header' and 'implementation' keys, or None if failed
        """
        self.logger.info(f"Generating C++ code for project: {project_name}")
        
        try:
            # Generate C++ code using LLM
            response = self._generate_llm_cpp_code(
                matlab_analysis, conversion_plan, project_name, cpp_standard, target_quality_score
            )
            
            # Parse response to extract header and implementation
            header_content, implementation_content = self._extract_cpp_code(response)
            
            if not implementation_content:
                self.logger.error("Failed to extract C++ code from LLM response")
                return None
            
            result = {
                'header': header_content,
                'implementation': implementation_content
            }
            
            self.logger.info(f"C++ code generated successfully: {len(implementation_content)} chars")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to generate C++ code: {e}")
            return None
    
    def generate_improved_cpp_code(self,
                                 current_code: Dict[str, str],
                                 matlab_analysis: Dict[str, Any],
                                 issues: List[Any],
                                 project_name: str) -> Optional[Dict[str, str]]:
        """
        Generate improved C++ code based on assessment feedback.
        
        Args:
            current_code: Current C++ code with 'header' and 'implementation'
            matlab_analysis: Original MATLAB analysis
            issues: List of issues identified by assessor
            project_name: Name for the C++ project
            
        Returns:
            Dictionary with improved 'header' and 'implementation', or None if failed
        """
        self.logger.info(f"Generating improved C++ code for project: {project_name}")
        
        try:
            # Generate improved code using LLM
            response = self._generate_llm_improved_code(
                current_code, matlab_analysis, issues, project_name
            )
            
            # Parse response to extract improved code
            header_content, implementation_content = self._extract_cpp_code(response)
            
            if not implementation_content:
                self.logger.error("Failed to extract improved C++ code from LLM response")
                return None
            
            result = {
                'header': current_code.get('header', ''),  # Keep original header unless improved
                'implementation': implementation_content
            }
            
            self.logger.info(f"Improved C++ code generated successfully: {len(implementation_content)} chars")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to generate improved C++ code: {e}")
            return None
    
    def _generate_llm_cpp_code(self,
                             matlab_analysis: Dict[str, Any],
                             conversion_plan: ConversionPlan,
                             project_name: str,
                             cpp_standard: str,
                             target_quality_score: float) -> str:
        """Generate C++ code using LLM."""
        
        # Create comprehensive generation prompt
        generation_prompt = f"""/no_think

You are a C++ expert. Convert this MATLAB code to C++ using the provided conversion plan.

MATLAB Project Analysis:
{self._format_matlab_analysis(matlab_analysis)}

Conversion Plan:
{self._format_conversion_plan(conversion_plan)}

Requirements:
- Project Name: {project_name}
- C++ Standard: {cpp_standard}
- Target Quality: {target_quality_score}/10

CRITICAL INSTRUCTIONS:
1. Analyze the MATLAB analysis to understand the ACTUAL algorithm being implemented
2. Implement the REAL mathematical algorithm, not placeholder or toy operations
3. Convert MATLAB operations to equivalent C++ operations:
   - zeros() → Eigen::MatrixXd::Zero() or std::vector initialization
   - eye() → Eigen::MatrixXd::Identity()
   - eig() → Eigen::SelfAdjointEigenSolver or Eigen::EigenSolver
   - inv() → .inverse() or .llt().solve() or .ldlt().solve()
   - squeeze() → proper indexing and dimension handling
   - size() → .rows(), .cols(), .size()
   - length() → .size()
4. Handle data types and dimensions correctly (vectors, matrices, 3D arrays)
5. Convert MATLAB 1-based indexing to C++ 0-based indexing
6. Use appropriate C++ libraries (Eigen, std, etc.) based on the algorithm needs
7. Add comprehensive error handling and input validation
8. Follow C++17 best practices
9. Include timing measurements and performance optimizations
10. Preserve the mathematical correctness and algorithm logic

REQUIRED FORMAT - Copy this exactly:

HEADER_FILE:
```cpp
#ifndef {project_name.upper()}_H
#define {project_name.upper()}_H
// Your header code here
#endif
```

IMPLEMENTATION_FILE:
```cpp
#include "{project_name}.h"
// Your implementation code here
```

Do not include any text before or after these code blocks.
"""
        
        messages = [{"role": "user", "content": generation_prompt}]
        response = self.llm_client.invoke(messages)
        return response
    
    def _generate_llm_improved_code(self,
                                  current_code: Dict[str, str],
                                  matlab_analysis: Dict[str, Any],
                                  issues: List[Any],
                                  project_name: str) -> str:
        """Generate improved C++ code using LLM."""
        
        # Format issues for the prompt
        issues_text = self._format_issues(issues)
        
        improvement_prompt = f"""/no_think

Improve the following C++ code by addressing the identified issues:

MATLAB Original:
```matlab
{self._get_matlab_code_content(matlab_analysis)}
```

Current C++ Code:
```cpp
{current_code.get('implementation', '')}
```

Issues to Fix:
{issues_text}

Please provide an improved version that:
1. Fixes all critical and high severity issues
2. Maintains the same functionality as the MATLAB code
3. Follows C++ best practices
4. Includes proper error handling
5. Is optimized for performance

REQUIRED FORMAT - Copy this exactly:

IMPLEMENTATION_FILE:
```cpp
#include "{project_name}.h"
// Your improved implementation code here
```

Do not include any text before or after this code block.
"""
        
        messages = [{"role": "user", "content": improvement_prompt}]
        response = self.llm_client.invoke(messages)
        return response
    
    def _extract_cpp_code(self, response: str) -> Tuple[str, str]:
        """Extract C++ code from LLM response."""
        # Remove <think> tags if present
        clean_response = response
        if "<think>" in response and "</think>" in response:
            think_end = response.find("</think>")
            if think_end != -1:
                clean_response = response[think_end + 8:].strip()
                self.logger.debug("Removed <think> tags from response")
        
        header_content = ""
        implementation_content = ""
        
        # Look for header and implementation
        header_match = clean_response.find("HEADER_FILE:")
        impl_match = clean_response.find("IMPLEMENTATION_FILE:")
        
        if header_match != -1 and impl_match != -1:
            header_start = clean_response.find("```cpp", header_match) + 6
            header_end = clean_response.find("```", header_start)
            header_content = clean_response[header_start:header_end].strip()
            
            impl_start = clean_response.find("```cpp", impl_match) + 6
            impl_end = clean_response.find("```", impl_start)
            implementation_content = clean_response[impl_start:impl_end].strip()
        else:
            # Fallback: try to extract code blocks
            code_blocks = re.findall(r'```cpp\n(.*?)\n```', clean_response, re.DOTALL)
            if len(code_blocks) >= 2:
                header_content = code_blocks[0].strip()
                implementation_content = code_blocks[1].strip()
            elif len(code_blocks) == 1:
                implementation_content = code_blocks[0].strip()
            else:
                self.logger.warning("No C++ code blocks found in response")
        
        return header_content, implementation_content
    
    def _format_matlab_analysis(self, matlab_analysis: Dict[str, Any]) -> str:
        """Format MATLAB analysis for the prompt."""
        return f"""
- Files: {matlab_analysis.get('files_analyzed', 0)}
- Functions: {matlab_analysis.get('total_functions', 0)}
- Dependencies: {matlab_analysis.get('total_dependencies', 0)}
- MATLAB Packages: {matlab_analysis.get('matlab_packages_used', [])}
- MATLAB Functions: {matlab_analysis.get('matlab_functions_used', [])}
- Complexity: {matlab_analysis.get('complexity_assessment', 'Medium')}
"""
    
    def _format_conversion_plan(self, conversion_plan: ConversionPlan) -> str:
        """Format conversion plan for the prompt."""
        return f"""
- Project Structure: {conversion_plan.project_structure}
- C++ Architecture: {conversion_plan.cpp_architecture}
- Conversion Strategy: {conversion_plan.conversion_strategy}
- Dependencies: {conversion_plan.dependencies}
- Conversion Steps: {conversion_plan.conversion_steps}
"""
    
    def _format_issues(self, issues: List[Any]) -> str:
        """Format issues for the improvement prompt."""
        if not issues:
            return "No specific issues identified."
        
        issues_text = []
        for issue in issues:
            if hasattr(issue, 'description') and hasattr(issue, 'suggestion'):
                issues_text.append(f"- {issue.description}: {issue.suggestion}")
            else:
                issues_text.append(f"- {str(issue)}")
        
        return "\n".join(issues_text)
    
    def _get_matlab_code_content(self, matlab_analysis: Dict[str, Any]) -> str:
        """Extract MATLAB code content for assessment."""
        content_parts = []
        for file_analysis in matlab_analysis.get('file_analyses', []):
            content_parts.append(f"File: {file_analysis.get('file_path', 'Unknown')}")
            if 'parsed_structure' in file_analysis and hasattr(file_analysis['parsed_structure'], 'content'):
                content_parts.append(file_analysis['parsed_structure'].content)
            content_parts.append("")
        return "\n".join(content_parts)