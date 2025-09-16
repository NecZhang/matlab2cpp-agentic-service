#!/usr/bin/env python3
"""
MATLAB2C++ Service Orchestrator

This orchestrator coordinates all the specialized agents to provide
a clean, modular MATLAB to C++ conversion service. It manages the
workflow and delegates specific tasks to appropriate agents.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
from loguru import logger

from matlab2cpp_agent.agents.matlab_content_analyzer import MATLABContentAnalyzerAgent
from matlab2cpp_agent.agents.conversion_planner import ConversionPlannerAgent, ConversionPlan
from matlab2cpp_agent.agents.cpp_generator import CppGeneratorAgent
from matlab2cpp_agent.agents.quality_assessor import QualityAssessorAgent, AssessmentResult
from matlab2cpp_agent.tools.llm_client import create_llm_client
from matlab2cpp_agent.utils.config import get_config

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

class MATLAB2CPPOrchestrator:
    """Orchestrator for MATLAB2C++ conversion service."""
    
    def __init__(self):
        """Initialize the orchestrator with all agents."""
        self.config = get_config()
        self.llm_client = create_llm_client(self.config.llm)
        
        # Initialize specialized agents
        self.content_analyzer = MATLABContentAnalyzerAgent(self.config.llm)
        self.conversion_planner = ConversionPlannerAgent(self.config.llm)
        self.cpp_generator = CppGeneratorAgent(self.config.llm)
        self.quality_assessor = QualityAssessorAgent(self.config.llm)
        
        self.logger = logger.bind(name="matlab2cpp_orchestrator")
        self.logger.info("MATLAB2C++ Orchestrator initialized with all agents")
    
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
        """Analyze MATLAB content using the content analyzer agent."""
        return self.content_analyzer.analyze_matlab_content(Path(matlab_path))
    
    def _create_conversion_plan(self, matlab_analysis: Dict[str, Any], 
                              request: ConversionRequest) -> ConversionPlan:
        """Create conversion plan using the conversion planner agent."""
        return self.conversion_planner.create_conversion_plan(
            matlab_analysis=matlab_analysis,
            project_name=request.project_name,
            cpp_standard=request.cpp_standard,
            include_tests=request.include_tests,
            additional_requirements=request.additional_requirements
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
        initial_code = self.cpp_generator.generate_cpp_code(
            matlab_analysis=matlab_analysis,
            conversion_plan=conversion_plan,
            project_name=request.project_name,
            cpp_standard=request.cpp_standard,
            target_quality_score=request.target_quality_score
        )
        
        if not initial_code:
            raise ValueError("Failed to generate initial C++ code")
        
        # Save initial code
        initial_files = self._save_generated_code(initial_code, output_dir, request.project_name, "v1")
        generated_files.extend(initial_files)
        
        # Assess initial code
        self.logger.info("Assessing initial code...")
        initial_assessment = self.quality_assessor.assess_code_quality(
            cpp_code=initial_code.get('implementation', ''),
            matlab_code=self._get_matlab_code_content(matlab_analysis),
            project_name=f"{request.project_name}_v1"
        )
        original_score = initial_assessment.metrics.overall_score
        
        # Save initial assessment
        initial_report = output_dir / f"{request.project_name}_v1_assessment_report.md"
        self.quality_assessor.generate_assessment_report(initial_assessment, initial_report)
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
            improved_code = self.cpp_generator.generate_improved_cpp_code(
                current_code=current_code,
                matlab_analysis=matlab_analysis,
                issues=current_assessment.issues,
                project_name=request.project_name
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
            improved_assessment = self.quality_assessor.assess_code_quality(
                cpp_code=improved_code.get('implementation', ''),
                matlab_code=self._get_matlab_code_content(matlab_analysis),
                project_name=f"{request.project_name}_v{improvement_turns + 1}"
            )
            
            # Save improved assessment
            improved_report = output_dir / f"{request.project_name}_v{improvement_turns + 1}_assessment_report.md"
            self.quality_assessor.generate_assessment_report(improved_assessment, improved_report)
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
        for file_analysis in matlab_analysis.get('file_analyses', []):
            content_parts.append(f"File: {file_analysis.get('file_path', 'Unknown')}")
            if 'parsed_structure' in file_analysis and hasattr(file_analysis['parsed_structure'], 'content'):
                content_parts.append(file_analysis['parsed_structure'].content)
            content_parts.append("")
        return "\n".join(content_parts)
