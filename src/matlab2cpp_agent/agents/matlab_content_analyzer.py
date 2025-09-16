#!/usr/bin/env python3
"""
MATLAB Content Analyzer Agent

This agent is responsible for analyzing MATLAB code content to understand
functionality, purpose, and requirements. It uses LLM to provide deep
insights into MATLAB code behavior and characteristics.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass
from loguru import logger

from matlab2cpp_agent.tools.matlab_parser import MATLABParser, MATLABFile
from matlab2cpp_agent.tools.llm_client import create_llm_client
from matlab2cpp_agent.utils.config import LLMConfig

@dataclass
class CodeUnderstanding:
    """Represents understanding of a piece of MATLAB code."""
    purpose: str
    domain: str
    algorithms: List[str]
    data_flow: Dict[str, Any]
    complexity: str
    confidence: float
    challenges: List[str]
    suggestions: List[str]

@dataclass
class FunctionUnderstanding:
    """Represents understanding of a MATLAB function."""
    name: str
    purpose: str
    inputs: List[Dict[str, Any]]
    outputs: List[Dict[str, Any]]
    algorithm_type: str
    complexity: str
    dependencies: List[str]
    cpp_equivalent_strategy: str
    confidence: float

@dataclass
class ProjectUnderstanding:
    """Represents overall understanding of a MATLAB project."""
    main_purpose: str
    domain: str
    key_algorithms: List[str]
    architecture: str
    complexity_level: str
    conversion_challenges: List[str]
    recommendations: List[str]
    confidence: float

class MATLABContentAnalyzerAgent:
    """Agent responsible for analyzing MATLAB code content and functionality."""
    
    def __init__(self, llm_config: LLMConfig):
        """Initialize the MATLAB content analyzer agent."""
        self.llm_config = llm_config
        self.llm_client = create_llm_client(llm_config)
        self.parser = MATLABParser()
        self.logger = logger.bind(name="matlab_content_analyzer_agent")
        self.logger.info("MATLAB Content Analyzer Agent initialized")
    
    def analyze_matlab_content(self, matlab_path: Path) -> Dict[str, Any]:
        """
        Analyze MATLAB content to understand functionality and requirements.
        
        Args:
            matlab_path: Path to MATLAB file or project directory
            
        Returns:
            Dictionary with comprehensive MATLAB analysis results
        """
        self.logger.info(f"Analyzing MATLAB content: {matlab_path}")
        
        # Parse MATLAB files
        matlab_files = self._get_matlab_files(matlab_path)
        if not matlab_files:
            raise ValueError(f"No MATLAB files found in {matlab_path}")
        
        # Analyze each file with LLM
        file_analyses = []
        for matlab_file in matlab_files:
            try:
                parsed_file = self.parser.parse_file(matlab_file)
                analysis = self._analyze_file_content(parsed_file)
                file_analyses.append({
                    'file_path': str(matlab_file),
                    'analysis': analysis,
                    'parsed_structure': parsed_file
                })
            except Exception as e:
                self.logger.warning(f"Failed to analyze {matlab_file}: {e}")
        
        # Create comprehensive analysis
        matlab_analysis = {
            'files_analyzed': len(file_analyses),
            'file_analyses': file_analyses,
            'total_functions': sum(len(f['parsed_structure'].functions) for f in file_analyses),
            'total_dependencies': sum(len(f['parsed_structure'].dependencies) for f in file_analyses),
            'matlab_packages_used': self._extract_matlab_packages(file_analyses),
            'matlab_functions_used': self._extract_matlab_functions(file_analyses),
            'complexity_assessment': self._assess_overall_complexity(file_analyses),
            'project_understanding': self._create_project_understanding(file_analyses)
        }
        
        self.logger.info(f"Analysis complete: {matlab_analysis['total_functions']} functions, "
                        f"{matlab_analysis['total_dependencies']} dependencies")
        
        return matlab_analysis
    
    def _get_matlab_files(self, matlab_path: Path) -> List[Path]:
        """Get list of MATLAB files to analyze."""
        if matlab_path.is_file():
            return [matlab_path]
        else:
            return list(matlab_path.glob("**/*.m"))
    
    def _analyze_file_content(self, matlab_file: MATLABFile) -> CodeUnderstanding:
        """Analyze MATLAB file content using LLM."""
        self.logger.debug(f"Analyzing file content: {matlab_file.path.name}")
        
        # Create analysis prompt
        analysis_prompt = self._create_analysis_prompt(matlab_file)
        
        try:
            # Get LLM response
            response = self._get_llm_response(analysis_prompt)
            
            # Parse response into structured understanding
            understanding = self._parse_analysis_response(response, matlab_file)
            
            return understanding
            
        except Exception as e:
            self.logger.error(f"Error analyzing file {matlab_file.path.name}: {e}")
            # Return basic understanding as fallback
            return CodeUnderstanding(
                purpose="Analysis failed",
                domain="Unknown",
                algorithms=[],
                data_flow={},
                complexity="Unknown",
                confidence=0.0,
                challenges=[f"Analysis error: {e}"],
                suggestions=["Manual review required"]
            )
    
    def _create_analysis_prompt(self, matlab_file: MATLABFile) -> str:
        """Create analysis prompt for LLM."""
        return f"""/no_think

You are an expert MATLAB and C++ developer specializing in signal processing and numerical algorithms. Analyze this MATLAB code thoroughly:

File: {matlab_file.path.name}
Size: {matlab_file.size} bytes
Functions: {len(matlab_file.functions)}
Dependencies: {len(matlab_file.dependencies)}

MATLAB Code:
```matlab
{matlab_file.content}
```

CRITICAL: Analyze the ACTUAL algorithm implementation, not just the function structure. Focus on:

1. Purpose: What does this code ACTUALLY do? What is the mathematical algorithm?
2. Domain: What field/domain does this belong to? (signal processing, ARMA filtering, etc.)
3. Algorithms: What SPECIFIC algorithms are used? (eigenvalue decomposition, matrix operations, regularization, etc.)
4. Data Flow: How does data flow through the code? What are the input/output transformations?
5. Mathematical Operations: What are the key mathematical operations? (matrix construction, eigendecomposition, regularization, etc.)
6. Complexity: How complex is the implementation? (Low/Medium/High)
7. Challenges: What challenges exist for C++ conversion?
8. Suggestions: What recommendations do you have for C++ implementation?

Be VERY specific about:
- Matrix dimensions and operations
- Mathematical algorithms (eigenvalue decomposition, regularization, etc.)
- Data transformations and processing steps
- The actual mathematical purpose of each operation

Do NOT provide generic analysis - analyze the SPECIFIC algorithm implemented in this code.
"""
    
    def _get_llm_response(self, prompt: str) -> str:
        """Get response from LLM."""
        try:
            messages = [
                {"role": "system", "content": "/no_think\n\nYou are an expert MATLAB and C++ developer with deep knowledge of mathematical computing, signal processing, and numerical methods. Analyze the provided MATLAB code thoroughly and provide detailed, accurate insights."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm_client.invoke(messages)
            return response
        except Exception as e:
            self.logger.error(f"Error getting LLM response: {e}")
            return f"Error in analysis: {e}"
    
    def _parse_analysis_response(self, response: str, matlab_file: MATLABFile) -> CodeUnderstanding:
        """Parse LLM response into structured understanding."""
        # This is a simplified parser - in practice, you'd want more robust parsing
        # For now, we'll extract basic information and create a structured response
        
        purpose = "Analyzed by LLM"
        domain = "General"
        complexity = "Medium"
        confidence = 0.8
        
        # Extract algorithms mentioned
        algorithms = []
        if "filter" in response.lower():
            algorithms.append("Filtering")
        if "matrix" in response.lower():
            algorithms.append("Matrix Operations")
        if "eigen" in response.lower():
            algorithms.append("Eigenvalue Analysis")
        
        # Extract challenges
        challenges = []
        if "indexing" in response.lower():
            challenges.append("MATLAB 1-based vs C++ 0-based indexing")
        if "memory" in response.lower():
            challenges.append("Memory management differences")
        
        # Extract suggestions
        suggestions = []
        if "eigen" in response.lower():
            suggestions.append("Use Eigen library for matrix operations")
        if "error" in response.lower():
            suggestions.append("Add comprehensive error handling")
        
        return CodeUnderstanding(
            purpose=purpose,
            domain=domain,
            algorithms=algorithms,
            data_flow={},
            complexity=complexity,
            confidence=confidence,
            challenges=challenges,
            suggestions=suggestions
        )
    
    def _extract_matlab_packages(self, file_analyses: List[Dict]) -> List[str]:
        """Extract MATLAB packages/toolboxes used."""
        packages = set()
        for analysis in file_analyses:
            for dep in analysis['parsed_structure'].dependencies:
                if '.' in dep:
                    package = dep.split('.')[0]
                    packages.add(package)
        return list(packages)
    
    def _extract_matlab_functions(self, file_analyses: List[Dict]) -> List[str]:
        """Extract MATLAB functions used."""
        functions = set()
        for analysis in file_analyses:
            for dep in analysis['parsed_structure'].dependencies:
                functions.add(dep)
        return list(functions)
    
    def _assess_overall_complexity(self, file_analyses: List[Dict]) -> str:
        """Assess overall project complexity."""
        complexities = [analysis['analysis'].complexity for analysis in file_analyses]
        if 'High' in complexities:
            return 'High'
        elif 'Medium' in complexities:
            return 'Medium'
        else:
            return 'Low'
    
    def _create_project_understanding(self, file_analyses: List[Dict]) -> ProjectUnderstanding:
        """Create overall project understanding."""
        if not file_analyses:
            return ProjectUnderstanding(
                main_purpose="Unknown",
                domain="Unknown",
                key_algorithms=[],
                architecture="Unknown",
                complexity_level="Unknown",
                conversion_challenges=[],
                recommendations=[],
                confidence=0.0
            )
        
        # Aggregate information from all file analyses
        all_algorithms = []
        all_challenges = []
        all_suggestions = []
        
        for analysis in file_analyses:
            understanding = analysis['analysis']
            all_algorithms.extend(understanding.algorithms)
            all_challenges.extend(understanding.challenges)
            all_suggestions.extend(understanding.suggestions)
        
        return ProjectUnderstanding(
            main_purpose="MATLAB to C++ conversion project",
            domain=file_analyses[0]['analysis'].domain if file_analyses else "General",
            key_algorithms=list(set(all_algorithms)),
            architecture="Modular C++ design",
            complexity_level=self._assess_overall_complexity(file_analyses),
            conversion_challenges=list(set(all_challenges)),
            recommendations=list(set(all_suggestions)),
            confidence=0.8
        )
