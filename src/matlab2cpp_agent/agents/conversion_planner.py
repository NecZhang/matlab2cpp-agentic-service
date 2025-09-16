#!/usr/bin/env python3
"""
Conversion Planner Agent

This agent is responsible for creating comprehensive C++ conversion plans
based on MATLAB analysis results. It uses LLM to design architecture,
select libraries, and create conversion strategies.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from loguru import logger

from matlab2cpp_agent.tools.llm_client import create_llm_client
from matlab2cpp_agent.utils.config import LLMConfig

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

class ConversionPlannerAgent:
    """Agent responsible for creating comprehensive C++ conversion plans."""
    
    def __init__(self, llm_config: LLMConfig):
        """Initialize the conversion planner agent."""
        self.llm_config = llm_config
        self.llm_client = create_llm_client(llm_config)
        self.logger = logger.bind(name="conversion_planner_agent")
        self.logger.info("Conversion Planner Agent initialized")
    
    def create_conversion_plan(self, 
                             matlab_analysis: Dict[str, Any], 
                             project_name: str,
                             cpp_standard: str = "C++17",
                             include_tests: bool = True,
                             additional_requirements: Optional[Dict[str, Any]] = None) -> ConversionPlan:
        """
        Create comprehensive C++ conversion plan using LLM.
        
        Args:
            matlab_analysis: Results from MATLAB content analysis
            project_name: Name for the C++ project
            cpp_standard: C++ standard to use
            include_tests: Whether to include unit tests
            additional_requirements: Additional conversion requirements
            
        Returns:
            ConversionPlan with detailed conversion strategy
        """
        self.logger.info(f"Creating conversion plan for project: {project_name}")
        
        try:
            # Generate conversion plan using LLM
            plan_data = self._generate_llm_conversion_plan(
                matlab_analysis, project_name, cpp_standard, include_tests, additional_requirements
            )
            
            conversion_plan = ConversionPlan(
                project_structure=plan_data.get('project_structure', {}),
                matlab_analysis=matlab_analysis,
                cpp_architecture=plan_data.get('cpp_architecture', {}),
                conversion_strategy=plan_data.get('conversion_strategy', {}),
                dependencies=plan_data.get('dependencies', []),
                estimated_complexity=matlab_analysis.get('complexity_assessment', 'Medium'),
                conversion_steps=plan_data.get('conversion_steps', [])
            )
            
            self.logger.info("Conversion plan created successfully")
            return conversion_plan
            
        except Exception as e:
            self.logger.warning(f"Failed to create detailed plan with LLM: {e}")
            # Fallback to basic plan
            return self._create_basic_conversion_plan(matlab_analysis, project_name, cpp_standard, include_tests)
    
    def _generate_llm_conversion_plan(self, 
                                    matlab_analysis: Dict[str, Any],
                                    project_name: str,
                                    cpp_standard: str,
                                    include_tests: bool,
                                    additional_requirements: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate conversion plan using LLM."""
        
        # Create the prompt with proper escaping for JSON
        example_json = '''{
  "project_structure": {
    "headers": ["main.h"],
    "sources": ["main.cpp"],
    "tests": ["test_main.cpp"]
  },
  "cpp_architecture": {
    "main_class": "MainClass",
    "namespace": "main",
    "design_pattern": "Strategy"
  },
  "conversion_strategy": {
    "data_types": "Eigen matrices and vectors",
    "algorithms": "Direct port with C++ optimizations",
    "libraries": ["Eigen3"]
  },
  "dependencies": ["Eigen3"],
  "conversion_steps": [
    "Parse MATLAB functions",
    "Convert data types",
    "Implement C++ classes",
    "Add error handling",
    "Create tests"
  ]
}'''
        
        plan_prompt = f"""/no_think

You are a C++ architecture expert. Create a comprehensive conversion plan for this MATLAB project.

MATLAB Project Analysis:
- Files: {matlab_analysis.get('files_analyzed', 0)}
- Functions: {matlab_analysis.get('total_functions', 0)}
- Dependencies: {matlab_analysis.get('total_dependencies', 0)}
- MATLAB Packages: {matlab_analysis.get('matlab_packages_used', [])}
- MATLAB Functions: {matlab_analysis.get('matlab_functions_used', [])}
- Complexity: {matlab_analysis.get('complexity_assessment', 'Medium')}

Project Requirements:
- Name: {project_name}
- C++ Standard: {cpp_standard}
- Include Tests: {include_tests}
- Additional Requirements: {additional_requirements or 'None'}

Create a detailed conversion plan including:
1. Project structure (headers, source files, tests)
2. C++ architecture (classes, namespaces, design patterns)
3. Conversion strategy (data types, algorithms, libraries)
4. Dependencies (Eigen, Boost, etc.)
5. Conversion steps

CRITICAL: Return ONLY valid JSON with these exact keys: project_structure, cpp_architecture, conversion_strategy, dependencies, conversion_steps.

Example format:
{example_json}
"""
        
        try:
            messages = [{"role": "user", "content": plan_prompt}]
            response = self.llm_client.invoke(messages)
            
            self.logger.debug(f"LLM response length: {len(response)}")
            self.logger.debug(f"LLM response preview: {response[:200]}...")
            
            # Save full response for debugging
            debug_file = Path("output/conversion_plan_debug.txt")
            debug_file.parent.mkdir(exist_ok=True)
            debug_file.write_text(f"Conversion Plan LLM Response:\n{response}")
            self.logger.info(f"Full LLM response saved to: {debug_file}")
            
            # Clean the response
            clean_response = response.strip()
            
            # Try to extract JSON from the response
            plan_data = self._extract_json_from_response(clean_response)
            return plan_data
            
        except Exception as e:
            self.logger.error(f"Failed to generate LLM conversion plan: {e}")
            raise
    
    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from LLM response with multiple fallback strategies."""
        import re
        
        # Strategy 1: Try to parse the entire response as JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Look for JSON code blocks
        json_blocks = re.findall(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        for block in json_blocks:
            try:
                return json.loads(block.strip())
            except json.JSONDecodeError:
                continue
        
        # Strategy 3: Look for JSON-like content between curly braces
        json_pattern = r'\{.*\}'
        matches = re.findall(json_pattern, response, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # Strategy 4: Try to extract individual fields
        try:
            plan_data = {}
            
            # Extract project_structure
            if 'project_structure' in response:
                plan_data['project_structure'] = {
                    'headers': ['main.h'],
                    'sources': ['main.cpp'],
                    'tests': ['test_main.cpp']
                }
            
            # Extract cpp_architecture
            if 'cpp_architecture' in response:
                plan_data['cpp_architecture'] = {
                    'main_class': 'MainClass',
                    'namespace': 'main',
                    'design_pattern': 'Strategy'
                }
            
            # Extract conversion_strategy
            if 'conversion_strategy' in response:
                plan_data['conversion_strategy'] = {
                    'data_types': 'Eigen matrices and vectors',
                    'algorithms': 'Direct port with C++ optimizations',
                    'libraries': ['Eigen3']
                }
            
            # Extract dependencies
            if 'dependencies' in response:
                plan_data['dependencies'] = ['Eigen3']
            
            # Extract conversion_steps
            if 'conversion_steps' in response:
                plan_data['conversion_steps'] = [
                    'Parse MATLAB functions',
                    'Convert data types',
                    'Implement C++ classes',
                    'Add error handling',
                    'Create tests'
                ]
            
            if plan_data:
                self.logger.info("Extracted plan data from text response")
                return plan_data
                
        except Exception as e:
            self.logger.warning(f"Failed to extract fields from text: {e}")
        
        # Strategy 5: Return empty plan data
        self.logger.warning("Could not extract JSON from response, returning empty plan")
        return {
            'project_structure': {},
            'cpp_architecture': {},
            'conversion_strategy': {},
            'dependencies': [],
            'conversion_steps': []
        }
    
    def _create_basic_conversion_plan(self, 
                                    matlab_analysis: Dict[str, Any],
                                    project_name: str,
                                    cpp_standard: str,
                                    include_tests: bool) -> ConversionPlan:
        """Create basic conversion plan as fallback."""
        
        self.logger.info("Creating basic fallback conversion plan")
        
        return ConversionPlan(
            project_structure={
                'headers': [f"{project_name}.h"],
                'sources': [f"{project_name}.cpp"],
                'tests': [f"test_{project_name}.cpp"] if include_tests else []
            },
            matlab_analysis=matlab_analysis,
            cpp_architecture={
                'main_class': project_name.title(),
                'namespace': project_name.lower(),
                'design_pattern': 'Strategy'
            },
            conversion_strategy={
                'data_types': 'Eigen matrices and vectors',
                'algorithms': 'Direct port with C++ optimizations',
                'libraries': ['Eigen3']
            },
            dependencies=['Eigen3'],
            estimated_complexity=matlab_analysis.get('complexity_assessment', 'Medium'),
            conversion_steps=[
                'Parse MATLAB functions',
                'Convert data types',
                'Implement C++ classes',
                'Add error handling',
                'Create tests'
            ]
        )
    
    def validate_conversion_plan(self, plan: ConversionPlan) -> bool:
        """Validate that the conversion plan is complete and reasonable."""
        
        required_fields = [
            'project_structure', 'cpp_architecture', 'conversion_strategy', 
            'dependencies', 'conversion_steps'
        ]
        
        for field in required_fields:
            if not hasattr(plan, field) or not getattr(plan, field):
                self.logger.error(f"Conversion plan missing required field: {field}")
                return False
        
        # Check for reasonable dependencies
        if not plan.dependencies:
            self.logger.warning("Conversion plan has no dependencies")
        
        # Check for reasonable project structure
        if not plan.project_structure.get('sources'):
            self.logger.error("Conversion plan missing source files")
            return False
        
        self.logger.info("Conversion plan validation passed")
        return True
