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

from ..agents.analyzer.legacy.matlab_analyzer import MATLABContentAnalyzerAgent
from ..agents.planner.legacy.conversion_planner import ConversionPlannerAgent
from ..agents.generator.legacy.cpp_generator import CppGeneratorAgent
from ..agents.assessor.legacy.quality_assessor import QualityAssessorAgent, AssessmentResult
from ...infrastructure.tools.llm_client import create_llm_client
from ...utils.config import get_config

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
    conversion_mode: str = "result-focused"  # 'faithful' or 'result-focused'
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
    conversion_plan: Dict[str, Any]
    total_processing_time: float
    error_message: Optional[str] = None

class MATLAB2CPPOrchestrator:
    """Orchestrator for MATLAB2C++ conversion service."""

    def __init__(self):
        """Initialize the orchestrator with all agents."""
        self.config = get_config()
        
        # Create LLM client for agents that need it
        llm_client = create_llm_client(self.config.llm)
        
        self.content_analyzer = MATLABContentAnalyzerAgent(self.config.llm)
        self.conversion_planner = ConversionPlannerAgent(llm_client)
        self.cpp_generator = CppGeneratorAgent(llm_client)
        self.quality_assessor = QualityAssessorAgent(llm_client)

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
                              request: ConversionRequest) -> Dict[str, Any]:
        """Create conversion plan using the conversion planner agent."""
        # Get the standard conversion plan
        conversion_plan = self.conversion_planner.plan(matlab_analysis)

        # Add multi-file project structure planning if this is a multi-file project
        if matlab_analysis.get('files_analyzed', 0) > 1:
            project_structure = self.conversion_planner.plan_multi_file_structure(matlab_analysis)
            conversion_plan['project_structure_plan'] = project_structure
            self.logger.info(f"Generated multi-file project structure with {len(project_structure['cpp_files'])} C++ files")
        
        return conversion_plan
    
    def _generate_and_optimize_code(self, matlab_analysis: Dict[str, Any], 
                                  conversion_plan: Dict[str, Any], 
                                  request: ConversionRequest) -> Dict[str, Any]:
        """Generate C++ code with iterative optimization."""
        
        output_dir = Path(request.output_dir or "output")
        output_dir.mkdir(exist_ok=True)
        
        original_score = 0.0
        final_score = 0.0
        improvement_turns = 0
        generated_files = []
        assessment_reports = []
        
        # Generate C++ code (multi-file or single-file)
        self.logger.info("Generating C++ code...")
        
        if matlab_analysis.get('files_analyzed', 0) > 1 and 'project_structure_plan' in conversion_plan:
            # Multi-file project generation
            self.logger.info("Generating multi-file C++ project...")
            initial_code = self.cpp_generator.generate_project_code(
                analysis=matlab_analysis,
                conversion_plan=conversion_plan,
                conversion_mode=request.conversion_mode
            )
        else:
            # Single-file project generation (fallback)
            self.logger.info("Generating single-file C++ code...")
            # Prepare MATLAB summary for code generation
            file_analyses = matlab_analysis.get('file_analyses', [])
            if file_analyses:
                first_parsed = file_analyses[0]['parsed_structure']
                matlab_summary = {
                    'functions': first_parsed.functions,
                    'dependencies': first_parsed.dependencies,
                    'numerical_calls': first_parsed.numerical_calls,
                    'source_code': first_parsed.content,
                }
            else:
                matlab_summary = {}
                
            initial_code = self.cpp_generator.generate_code(
                matlab_summary=matlab_summary,
                conversion_plan=conversion_plan,
                conversion_mode=request.conversion_mode
            )
        
        if not initial_code:
            raise ValueError("Failed to generate initial C++ code")
        
        # Handle different code generation results
        if isinstance(initial_code, str):
            # LLM returned a prompt instead of code
            self.logger.warning("LLM returned a prompt instead of code. Saving prompt for manual use.")
            prompt_file = output_dir / f"{request.project_name}_v1_prompt.txt"
            prompt_file.write_text(initial_code)
            generated_files.append(str(prompt_file))
            
            # Create a mock assessment for the prompt
            final_score = 0.0
            return {
                'original_score': final_score,
                'final_score': final_score,
                'improvement_turns': 0,
                'generated_files': generated_files,
                'assessment_reports': []
            }
        elif isinstance(initial_code, dict) and 'files' in initial_code:
            # Multi-file project generation result
            self.logger.info(f"Generated {len(initial_code['files'])} C++ files")
            initial_files = self._save_multi_file_code(initial_code, output_dir, request.project_name, "v1")
            generated_files.extend(initial_files)
            
            # For multi-file projects, assess the main entry point
            main_file = self._find_main_file(initial_code['files'])
            if main_file:
                full_code = initial_code['files'][main_file]
            else:
                # Concatenate all files for assessment
                full_code = "\n\n".join(initial_code['files'].values())
            
            # For multi-file projects, use the analyzed MATLAB content
            matlab_code = self._get_matlab_code_content(matlab_analysis)
            initial_assessment = self.quality_assessor.assess(
                code=full_code,
                matlab_code=matlab_code,
                conversion_plan=conversion_plan,
                conversion_mode=request.conversion_mode
            )
        else:
            # Single-file generation result
            initial_files = self._save_generated_code(initial_code, output_dir, request.project_name, "v1")
            generated_files.extend(initial_files)
            
            # Assess initial code
            self.logger.info("Assessing initial code...")
            full_code = (initial_code.get('header', '') + "\n" + initial_code.get('implementation', '')).strip()
            matlab_code = Path(request.matlab_path).read_text(encoding='utf-8', errors='ignore')
            initial_assessment = self.quality_assessor.assess(
                code=full_code,
                matlab_code=matlab_code,
                conversion_plan=conversion_plan,
                conversion_mode=request.conversion_mode
            )
        original_score = initial_assessment.metrics.get('algorithmic', 0.0)
        
        # Save initial assessment to organized structure
        project_output_dir = output_dir / request.project_name
        project_output_dir.mkdir(parents=True, exist_ok=True)
        (project_output_dir / "reports").mkdir(exist_ok=True)
        
        initial_report = project_output_dir / "reports" / "v1_assessment_report.md"
        self._generate_assessment_report(initial_assessment, initial_report, request.project_name, "v1")
        assessment_reports.append(str(initial_report))
        
        self.logger.info(f"Initial code score: {original_score:.1f}/10")
        
        # Optimization summary
        if original_score >= request.target_quality_score:
            self.logger.info(f"Initial code already meets target quality ({request.target_quality_score}/10). No optimization needed.")
        else:
            self.logger.info(f"Initial code below target quality ({request.target_quality_score}/10). Starting optimization with max {request.max_optimization_turns} additional turns.")
        
        # Check if optimization is needed
        current_score = original_score
        current_code = initial_code
        current_assessment = initial_assessment
        
        # Optimization loop: max_optimization_turns = additional optimization attempts beyond initial generation
        # For now, skip the complex optimization loop and just demonstrate the max-turns concept
        # This can be enhanced later when the generate_improved_cpp_code method is implemented
        if current_score < request.target_quality_score:
            self.logger.info(f"Target quality ({request.target_quality_score}/10) not achieved with initial code ({current_score:.1f}/10)")
            self.logger.info(f"Max optimization turns available: {request.max_optimization_turns}")
            self.logger.info("Note: Advanced optimization loop not yet implemented in this simplified version")
        else:
            self.logger.info(f"Initial code meets target quality ({request.target_quality_score}/10). No optimization needed.")
        
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
        """Save generated C++ code to organized project directory."""
        saved_files = []
        
        # Create organized output directory structure
        project_output_dir = output_dir / project_name
        project_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (project_output_dir / "generated_code").mkdir(exist_ok=True)
        (project_output_dir / "reports").mkdir(exist_ok=True)
        (project_output_dir / "debug").mkdir(exist_ok=True)
        
        if code.get('header'):
            header_file = project_output_dir / "generated_code" / f"{version}.h"
            header_file.write_text(code['header'])
            saved_files.append(str(header_file))
        
        if code.get('implementation'):
            impl_file = project_output_dir / "generated_code" / f"{version}.cpp"
            impl_file.write_text(code['implementation'])
            saved_files.append(str(impl_file))
        
        return saved_files
    
    def _generate_assessment_report(self, assessment: AssessmentResult, report_path: Path, 
                                  project_name: str, version: str) -> None:
        """Generate an assessment report in markdown format."""
        try:
            report_lines = [
                f"# Assessment Report for {project_name} - {version}",
                "",
                "## Metrics",
                *(f"- **{cat.replace('_',' ').title()}**: {score:.1f}/10" for cat, score in assessment.metrics.items()),
                "",
                "## Issues",
                *(f"- **{issue.category.title()} ({issue.severity})**: {issue.description}\n  *Suggestion*: {issue.suggestion}" for issue in assessment.issues),
                "",
                "## Summary",
                assessment.summary
            ]
            report_path.write_text("\n".join(report_lines))
        except Exception as e:
            self.logger.warning(f"Failed to generate assessment report: {e}")

    def _get_matlab_code_content(self, matlab_analysis: Dict[str, Any]) -> str:
        """Extract MATLAB code content for assessment."""
        content_parts = []
        for file_analysis in matlab_analysis.get('file_analyses', []):
            content_parts.append(f"File: {file_analysis.get('file_path', 'Unknown')}")
            if 'parsed_structure' in file_analysis and hasattr(file_analysis['parsed_structure'], 'content'):
                content_parts.append(file_analysis['parsed_structure'].content)
            content_parts.append("")
        return "\n".join(content_parts)

    def _save_multi_file_code(self, code_result: Dict[str, Any], output_dir: Path, 
                            project_name: str, version: str) -> List[str]:
        """Save multi-file C++ code to organized project directory."""
        generated_files = []
        
        # Create organized output directory structure
        project_output_dir = output_dir / project_name
        project_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (project_output_dir / "generated_code").mkdir(exist_ok=True)
        (project_output_dir / "reports").mkdir(exist_ok=True)
        (project_output_dir / "debug").mkdir(exist_ok=True)
        
        # Save all generated files to generated_code subdirectory
        for filename, content in code_result['files'].items():
            file_path = project_output_dir / "generated_code" / f"{version}_{filename}"
            file_path.write_text(content, encoding='utf-8')
            generated_files.append(str(file_path))
        
        # Save compilation instructions if available
        if 'compilation_instructions' in code_result:
            instructions_file = project_output_dir / "generated_code" / f"{version}_compilation_instructions.md"
            instructions_file.write_text(code_result['compilation_instructions'], encoding='utf-8')
            generated_files.append(str(instructions_file))
        
        return generated_files

    def _find_main_file(self, files: Dict[str, str]) -> Optional[str]:
        """Find the main file from a dictionary of generated files."""
        # Look for common main file names
        main_candidates = ['main.cpp', 'skeleton_vessel.cpp']
        
        for candidate in main_candidates:
            if candidate in files:
                return candidate
        
        # Return the first .cpp file if no main candidate found
        cpp_files = [f for f in files.keys() if f.endswith('.cpp')]
        return cpp_files[0] if cpp_files else None
