#!/usr/bin/env python3
"""
General MATLAB2C++ Agentic Service

This service provides a comprehensive workflow for converting MATLAB projects to C++:
1. Analyze MATLAB content to understand methods, packages, and functions
2. Create comprehensive C++ conversion plan
3. Generate C++ code with iterative optimization (up to 2 turns)
4. Provide final conversion results with assessment
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
from enum import Enum

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from matlab2cpp_agent.tools.matlab_parser import MATLABParser
from matlab2cpp_agent.agents.matlab_analyzer import MATLABAnalyzerAgent
from matlab2cpp_agent.agents.assessor import AssessorAgent
from matlab2cpp_agent.tools.llm_client import create_llm_client
from matlab2cpp_agent.utils.config import get_config
from matlab2cpp_agent.utils.logger import setup_logger

class ConversionStatus(Enum):
    """Conversion process status."""
    PENDING = "pending"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    GENERATING = "generating"
    ASSESSING = "assessing"
    OPTIMIZING = "optimizing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ConversionRequest:
    """Request for MATLAB to C++ conversion."""
    matlab_path: Union[str, Path]
    project_name: str
    output_dir: Optional[Union[str, Path]] = None
    max_optimization_turns: int = 2
    target_quality_score: float = 7.0
    include_tests: bool = True
    cpp_standard: str = "C++17"
    additional_requirements: Optional[Dict[str, Any]] = None

@dataclass
class ConversionPlan:
    """Comprehensive C++ conversion plan."""
    project_structure: Dict[str, Any]
    matlab_analysis: Dict[str, Any]
    cpp_architecture: Dict[str, Any]
    conversion_strategy: Dict[str, Any]
    dependencies: List[str]
    estimated_complexity: str
    conversion_steps: List[str]

@dataclass
class ConversionResult:
    """Final conversion result."""
    status: ConversionStatus
    project_name: str
    original_score: float
    final_score: float
    improvement_turns: int
    generated_files: List[str]
    assessment_reports: List[str]
    conversion_plan: ConversionPlan
    total_processing_time: float
    error_message: Optional[str] = None

class MATLAB2CPPService:
    """General MATLAB2C++ Agentic Service."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the service."""
        setup_logger()
        self.config = get_config()
        self.llm_client = create_llm_client(self.config.llm)
        self.parser = MATLABParser()
        self.analyzer = MATLABAnalyzerAgent()
        self.assessor = AssessorAgent(self.config.llm)
        
        from loguru import logger
        self.logger = logger
        self.logger.info("MATLAB2C++ Service initialized")
    
    def convert_project(self, request: ConversionRequest) -> ConversionResult:
        """
        Convert MATLAB project to C++ with comprehensive analysis and optimization.
        
        Args:
            request: Conversion request with MATLAB path and requirements
            
        Returns:
            ConversionResult with final conversion status and results
        """
        start_time = time.time()
        self.logger.info(f"Starting conversion for project: {request.project_name}")
        
        try:
            # Step 1: Analyze MATLAB content
            self.logger.info("Step 1: Analyzing MATLAB content...")
            matlab_analysis = self._analyze_matlab_content(request.matlab_path)
            
            # Step 2: Create conversion plan
            self.logger.info("Step 2: Creating conversion plan...")
            conversion_plan = self._create_conversion_plan(matlab_analysis, request)
            
            # Step 3: Generate and optimize C++ code
            self.logger.info("Step 3: Generating and optimizing C++ code...")
            final_result = self._generate_and_optimize_code(
                matlab_analysis, conversion_plan, request
            )
            
            # Step 4: Create final result
            total_time = time.time() - start_time
            result = ConversionResult(
                status=ConversionStatus.COMPLETED,
                project_name=request.project_name,
                original_score=final_result.get('original_score', 0.0),
                final_score=final_result.get('final_score', 0.0),
                improvement_turns=final_result.get('improvement_turns', 0),
                generated_files=final_result.get('generated_files', []),
                assessment_reports=final_result.get('assessment_reports', []),
                conversion_plan=conversion_plan,
                total_processing_time=total_time
            )
            
            self.logger.info(f"Conversion completed successfully in {total_time:.1f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Conversion failed: {e}")
            return ConversionResult(
                status=ConversionStatus.FAILED,
                project_name=request.project_name,
                original_score=0.0,
                final_score=0.0,
                improvement_turns=0,
                generated_files=[],
                assessment_reports=[],
                conversion_plan=None,
                total_processing_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _analyze_matlab_content(self, matlab_path: Union[str, Path]) -> Dict[str, Any]:
        """Analyze MATLAB content to understand methods, packages, and functions."""
        matlab_path = Path(matlab_path)
        
        if matlab_path.is_file():
            # Single MATLAB file
            matlab_files = [matlab_path]
        else:
            # MATLAB project directory
            matlab_files = list(matlab_path.glob("**/*.m"))
        
        if not matlab_files:
            raise ValueError(f"No MATLAB files found in {matlab_path}")
        
        self.logger.info(f"Analyzing {len(matlab_files)} MATLAB files...")
        
        # Parse all MATLAB files
        parsed_files = []
        for matlab_file in matlab_files:
            try:
                parsed = self.parser.parse_file(matlab_file)
                parsed_files.append(parsed)
            except Exception as e:
                self.logger.warning(f"Failed to parse {matlab_file}: {e}")
        
        # Analyze with LLM for content understanding
        analysis_results = []
        for parsed_file in parsed_files:
            try:
                analysis = self.analyzer.analyze_file(parsed_file)
                analysis_results.append({
                    'file_path': str(parsed_file.path),
                    'analysis': analysis,
                    'parsed_structure': parsed_file
                })
            except Exception as e:
                self.logger.warning(f"Failed to analyze {parsed_file.path}: {e}")
        
        # Create comprehensive analysis
        matlab_analysis = {
            'files_analyzed': len(analysis_results),
            'file_analyses': analysis_results,
            'total_functions': sum(len(f['parsed_structure'].functions) for f in analysis_results),
            'total_dependencies': sum(len(f['parsed_structure'].dependencies) for f in analysis_results),
            'matlab_packages_used': self._extract_matlab_packages(analysis_results),
            'matlab_functions_used': self._extract_matlab_functions(analysis_results),
            'complexity_assessment': self._assess_overall_complexity(analysis_results)
        }
        
        self.logger.info(f"Analysis complete: {matlab_analysis['total_functions']} functions, "
                        f"{matlab_analysis['total_dependencies']} dependencies")
        
        return matlab_analysis
    
    def _extract_matlab_packages(self, analysis_results: List[Dict]) -> List[str]:
        """Extract MATLAB packages/toolboxes used."""
        packages = set()
        for result in analysis_results:
            for dep in result['parsed_structure'].dependencies:
                if '.' in dep:
                    package = dep.split('.')[0]
                    packages.add(package)
        return list(packages)
    
    def _extract_matlab_functions(self, analysis_results: List[Dict]) -> List[str]:
        """Extract MATLAB functions used."""
        functions = set()
        for result in analysis_results:
            for dep in result['parsed_structure'].dependencies:
                functions.add(dep)
        return list(functions)
    
    def _assess_overall_complexity(self, analysis_results: List[Dict]) -> str:
        """Assess overall project complexity."""
        complexities = [result['analysis'].complexity for result in analysis_results]
        if 'High' in complexities:
            return 'High'
        elif 'Medium' in complexities:
            return 'Medium'
        else:
            return 'Low'
    
    def _create_conversion_plan(self, matlab_analysis: Dict[str, Any], 
                              request: ConversionRequest) -> ConversionPlan:
        """Create comprehensive C++ conversion plan."""
        
        # Generate conversion plan using LLM
        plan_prompt = f"""/no_think

You are a C++ architecture expert. Create a comprehensive conversion plan for this MATLAB project.

MATLAB Project Analysis:
- Files: {matlab_analysis['files_analyzed']}
- Functions: {matlab_analysis['total_functions']}
- Dependencies: {matlab_analysis['total_dependencies']}
- MATLAB Packages: {matlab_analysis['matlab_packages_used']}
- MATLAB Functions: {matlab_analysis['matlab_functions_used']}
- Complexity: {matlab_analysis['complexity_assessment']}

Project Requirements:
- Name: {request.project_name}
- C++ Standard: {request.cpp_standard}
- Include Tests: {request.include_tests}
- Target Quality: {request.target_quality_score}/10

Create a detailed conversion plan including:
1. Project structure (headers, source files, tests)
2. C++ architecture (classes, namespaces, design patterns)
3. Conversion strategy (data types, algorithms, libraries)
4. Dependencies (Eigen, Boost, etc.)
5. Conversion steps

Return as JSON with keys: project_structure, cpp_architecture, conversion_strategy, dependencies, conversion_steps.
"""
        
        try:
            messages = [{"role": "user", "content": plan_prompt}]
            response = self.llm_client.invoke(messages)
            
            # Parse JSON response
            import json
            plan_data = json.loads(response)
            
            conversion_plan = ConversionPlan(
                project_structure=plan_data.get('project_structure', {}),
                matlab_analysis=matlab_analysis,
                cpp_architecture=plan_data.get('cpp_architecture', {}),
                conversion_strategy=plan_data.get('conversion_strategy', {}),
                dependencies=plan_data.get('dependencies', []),
                estimated_complexity=matlab_analysis['complexity_assessment'],
                conversion_steps=plan_data.get('conversion_steps', [])
            )
            
            self.logger.info("Conversion plan created successfully")
            return conversion_plan
            
        except Exception as e:
            self.logger.warning(f"Failed to create detailed plan with LLM: {e}")
            # Fallback to basic plan
            return self._create_basic_conversion_plan(matlab_analysis, request)
    
    def _create_basic_conversion_plan(self, matlab_analysis: Dict[str, Any], 
                                    request: ConversionRequest) -> ConversionPlan:
        """Create basic conversion plan as fallback."""
        return ConversionPlan(
            project_structure={
                'headers': [f"{request.project_name}.h"],
                'sources': [f"{request.project_name}.cpp"],
                'tests': [f"test_{request.project_name}.cpp"] if request.include_tests else []
            },
            matlab_analysis=matlab_analysis,
            cpp_architecture={
                'main_class': request.project_name.title(),
                'namespace': request.project_name.lower(),
                'design_pattern': 'Strategy'
            },
            conversion_strategy={
                'data_types': 'Eigen matrices and vectors',
                'algorithms': 'Direct port with C++ optimizations',
                'libraries': ['Eigen3']
            },
            dependencies=['Eigen3'],
            estimated_complexity=matlab_analysis['complexity_assessment'],
            conversion_steps=[
                'Parse MATLAB functions',
                'Convert data types',
                'Implement C++ classes',
                'Add error handling',
                'Create tests'
            ]
        )
    
    def _generate_and_optimize_code(self, matlab_analysis: Dict[str, Any], 
                                  conversion_plan: ConversionPlan, 
                                  request: ConversionRequest) -> Dict[str, Any]:
        """Generate C++ code with iterative optimization."""
        
        output_dir = Path(request.output_dir or "output")
        output_dir.mkdir(exist_ok=True)
        
        original_score = 0.0
        final_score = 0.0
        improvement_turns = 0
        generated_files = []
        assessment_reports = []
        
        # Generate initial C++ code
        self.logger.info("Generating initial C++ code...")
        initial_code = self._generate_cpp_code(matlab_analysis, conversion_plan, request)
        
        if not initial_code:
            raise ValueError("Failed to generate initial C++ code")
        
        # Save initial code
        initial_files = self._save_generated_code(initial_code, output_dir, request.project_name, "v1")
        generated_files.extend(initial_files)
        
        # Assess initial code
        self.logger.info("Assessing initial code...")
        initial_assessment = self.assessor.assess_code(
            initial_code.get('implementation', ''),
            self._get_matlab_code_content(matlab_analysis),
            f"{request.project_name}_v1"
        )
        original_score = initial_assessment.metrics.overall_score
        
        # Save initial assessment
        initial_report = output_dir / f"{request.project_name}_v1_assessment_report.md"
        self.assessor.generate_assessment_report(initial_assessment, initial_report)
        assessment_reports.append(str(initial_report))
        
        self.logger.info(f"Initial code score: {original_score:.1f}/10")
        
        # Check if optimization is needed
        current_score = original_score
        current_code = initial_code
        current_assessment = initial_assessment
        
        while (current_score < request.target_quality_score and 
               improvement_turns < request.max_optimization_turns):
            
            improvement_turns += 1
            self.logger.info(f"Optimization turn {improvement_turns}/{request.max_optimization_turns}")
            
            # Generate improved code
            improved_code = self._generate_improved_code(
                current_code, current_assessment, matlab_analysis, conversion_plan, request
            )
            
            if not improved_code:
                self.logger.warning(f"Failed to generate improved code in turn {improvement_turns}")
                break
            
            # Save improved code
            improved_files = self._save_generated_code(
                improved_code, output_dir, request.project_name, f"v{improvement_turns + 1}"
            )
            generated_files.extend(improved_files)
            
            # Assess improved code
            improved_assessment = self.assessor.assess_code(
                improved_code.get('implementation', ''),
                self._get_matlab_code_content(matlab_analysis),
                f"{request.project_name}_v{improvement_turns + 1}"
            )
            
            # Save improved assessment
            improved_report = output_dir / f"{request.project_name}_v{improvement_turns + 1}_assessment_report.md"
            self.assessor.generate_assessment_report(improved_assessment, improved_report)
            assessment_reports.append(str(improved_report))
            
            new_score = improved_assessment.metrics.overall_score
            improvement = new_score - current_score
            
            self.logger.info(f"Turn {improvement_turns} - Score: {new_score:.1f}/10 "
                           f"(Improvement: {improvement:+.1f})")
            
            if improvement <= 0:
                self.logger.info("No improvement detected, stopping optimization")
                break
            
            current_score = new_score
            current_code = improved_code
            current_assessment = improved_assessment
        
        final_score = current_score
        
        return {
            'original_score': original_score,
            'final_score': final_score,
            'improvement_turns': improvement_turns,
            'generated_files': generated_files,
            'assessment_reports': assessment_reports
        }
    
    def _generate_cpp_code(self, matlab_analysis: Dict[str, Any], 
                          conversion_plan: ConversionPlan, 
                          request: ConversionRequest) -> Optional[Dict[str, str]]:
        """Generate C++ code based on analysis and plan."""
        
        # Create comprehensive generation prompt
        generation_prompt = f"""/no_think

You are a C++ expert. Convert this MATLAB project to C++ using the provided conversion plan.

MATLAB Project Analysis:
{json.dumps(matlab_analysis, indent=2)}

Conversion Plan:
{json.dumps(asdict(conversion_plan), indent=2)}

Requirements:
- Project Name: {request.project_name}
- C++ Standard: {request.cpp_standard}
- Include Tests: {request.include_tests}
- Target Quality: {request.target_quality_score}/10

CRITICAL INSTRUCTIONS:
1. Provide ONLY C++ code, NO explanations or reasoning
2. Use the EXACT format below with these exact markers
3. Follow the conversion plan architecture
4. Use appropriate C++ libraries (Eigen, etc.)
5. Convert MATLAB 1-based indexing to C++ 0-based indexing
6. Add comprehensive error handling and input validation
7. Follow C++17 best practices
8. Include timing measurements and performance optimizations

REQUIRED FORMAT - Copy this exactly:

HEADER_FILE:
```cpp
#ifndef {request.project_name.upper()}_H
#define {request.project_name.upper()}_H
// Your header code here
#endif
```

IMPLEMENTATION_FILE:
```cpp
#include "{request.project_name}.h"
// Your implementation code here
```

Do not include any text before or after these code blocks.
"""
        
        try:
            messages = [{"role": "user", "content": generation_prompt}]
            response = self.llm_client.invoke(messages)
            
            # Parse response to extract header and implementation
            header_content, implementation_content = self._extract_cpp_code(response)
            
            if not implementation_content:
                self.logger.error("Failed to extract C++ code from LLM response")
                return None
            
            return {
                'header': header_content,
                'implementation': implementation_content
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate C++ code: {e}")
            return None
    
    def _generate_improved_code(self, current_code: Dict[str, str], 
                              current_assessment, matlab_analysis: Dict[str, Any],
                              conversion_plan: ConversionPlan, 
                              request: ConversionRequest) -> Optional[Dict[str, str]]:
        """Generate improved C++ code based on assessment feedback."""
        
        # Use the assessor's improvement generation
        improved_code = self.assessor.generate_improved_code(
            current_code.get('implementation', ''),
            self._get_matlab_code_content(matlab_analysis),
            current_assessment.issues
        )
        
        if not improved_code:
            return None
        
        return {
            'header': current_code.get('header', ''),
            'implementation': improved_code
        }
    
    def _extract_cpp_code(self, response: str) -> tuple[str, str]:
        """Extract C++ code from LLM response."""
        # Remove <think> tags if present
        clean_response = response
        if "<think>" in response and "</think>" in response:
            think_end = response.find("</think>")
            if think_end != -1:
                clean_response = response[think_end + 8:].strip()
        
        # Look for header and implementation
        header_content = ""
        implementation_content = ""
        
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
            import re
            code_blocks = re.findall(r'```cpp\n(.*?)\n```', clean_response, re.DOTALL)
            if len(code_blocks) >= 2:
                header_content = code_blocks[0].strip()
                implementation_content = code_blocks[1].strip()
            elif len(code_blocks) == 1:
                implementation_content = code_blocks[0].strip()
        
        return header_content, implementation_content
    
    def _save_generated_code(self, code: Dict[str, str], output_dir: Path, 
                           project_name: str, version: str) -> List[str]:
        """Save generated C++ code to files."""
        saved_files = []
        
        if code.get('header'):
            header_file = output_dir / f"{project_name}_{version}.h"
            header_file.write_text(code['header'])
            saved_files.append(str(header_file))
        
        if code.get('implementation'):
            impl_file = output_dir / f"{project_name}_{version}.cpp"
            impl_file.write_text(code['implementation'])
            saved_files.append(str(impl_file))
        
        return saved_files
    
    def _get_matlab_code_content(self, matlab_analysis: Dict[str, Any]) -> str:
        """Extract MATLAB code content for assessment."""
        content_parts = []
        for file_analysis in matlab_analysis['file_analyses']:
            content_parts.append(f"File: {file_analysis['file_path']}")
            content_parts.append(file_analysis['parsed_structure'].content)
            content_parts.append("")
        return "\n".join(content_parts)

# Service API functions
def convert_matlab_project(matlab_path: Union[str, Path], 
                          project_name: str,
                          **kwargs) -> ConversionResult:
    """
    Convert MATLAB project to C++.
    
    Args:
        matlab_path: Path to MATLAB file or project directory
        project_name: Name for the C++ project
        **kwargs: Additional conversion options
        
    Returns:
        ConversionResult with conversion status and results
    """
    request = ConversionRequest(
        matlab_path=matlab_path,
        project_name=project_name,
        **kwargs
    )
    
    service = MATLAB2CPPService()
    return service.convert_project(request)

def convert_matlab_script(script_content: str, 
                         project_name: str,
                         **kwargs) -> ConversionResult:
    """
    Convert MATLAB script content to C++.
    
    Args:
        script_content: MATLAB script content as string
        project_name: Name for the C++ project
        **kwargs: Additional conversion options
        
    Returns:
        ConversionResult with conversion status and results
    """
    # Create temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.m', delete=False) as f:
        f.write(script_content)
        temp_path = f.name
    
    try:
        return convert_matlab_project(temp_path, project_name, **kwargs)
    finally:
        # Clean up temporary file
        os.unlink(temp_path)
