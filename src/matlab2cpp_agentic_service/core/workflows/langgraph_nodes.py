"""
LangGraph-Compatible Agent Nodes

This module transforms the existing agent classes into LangGraph-compatible nodes
that can be used in a graph-based workflow while preserving all existing functionality.
"""

import time
from pathlib import Path
from typing import Dict, Any
from loguru import logger

from ...infrastructure.state.conversion_state import ConversionState, ConversionStatus, add_processing_time, update_state_status
from ..agents.analyzer.legacy.matlab_analyzer import MATLABContentAnalyzerAgent
from ..agents.planner.legacy.conversion_planner import ConversionPlannerAgent
from ..agents.generator.legacy.cpp_generator import CppGeneratorAgent
from ..agents.assessor.legacy.quality_assessor import QualityAssessorAgent
from ...infrastructure.tools.llm_client import create_llm_client
from ...utils.config import get_config


class LangGraphAgentNodes:
    """
    LangGraph-compatible nodes that wrap existing agent functionality.
    
    Each node is a function that takes a state and returns an updated state,
    enabling the agents to work within a LangGraph workflow.
    """
    
    def __init__(self):
        """Initialize the agent nodes with existing agent classes."""
        self.config = get_config()
        
        # Create LLM client for agents that need it
        llm_client = create_llm_client(self.config.llm)
        
        # Initialize existing agents
        self.content_analyzer = MATLABContentAnalyzerAgent(self.config.llm)
        self.conversion_planner = ConversionPlannerAgent(llm_client)
        self.cpp_generator = CppGeneratorAgent(llm_client)
        self.quality_assessor = QualityAssessorAgent(llm_client)
        
        self.logger = logger.bind(name="langgraph_agent_nodes")
    
    def analyze_matlab_content(self, state: ConversionState) -> ConversionState:
        """
        Analyze MATLAB content using the content analyzer agent.
        
        Args:
            state: Current conversion state
            
        Returns:
            Updated state with analysis results
        """
        start_time = time.time()
        
        try:
            self.logger.info("Starting MATLAB content analysis...")
            state = update_state_status(state, ConversionStatus.ANALYZING)
            
            # Use existing agent functionality
            matlab_path = Path(state["request"].matlab_path)
            analysis_result = self.content_analyzer.analyze_matlab_content(matlab_path)
            
            # Update state with analysis results
            state["matlab_analysis"] = analysis_result
            state["is_multi_file"] = analysis_result.get('files_analyzed', 0) > 1
            
            self.logger.info(f"Analysis complete. Files analyzed: {analysis_result.get('files_analyzed', 1)}")
            self.logger.info(f"Multi-file project: {state['is_multi_file']}")
            
        except Exception as e:
            self.logger.error(f"Error in MATLAB content analysis: {e}")
            state["error_message"] = f"Analysis failed: {str(e)}"
            state = update_state_status(state, ConversionStatus.FAILED)
        
        # Record processing time
        duration = time.time() - start_time
        state = add_processing_time(state, "analysis", duration)
        
        return state
    
    def create_conversion_plan(self, state: ConversionState) -> ConversionState:
        """
        Create conversion plan using the conversion planner agent.
        
        Args:
            state: Current conversion state
            
        Returns:
            Updated state with conversion plan
        """
        start_time = time.time()
        
        try:
            self.logger.info("Creating conversion plan...")
            state = update_state_status(state, ConversionStatus.PLANNING)
            
            # Use existing agent functionality
            matlab_analysis = state["matlab_analysis"]
            conversion_plan = self.conversion_planner.plan(matlab_analysis)
            
            # Add multi-file project structure planning if needed
            if state["is_multi_file"]:
                project_structure = self.conversion_planner.plan_multi_file_structure(matlab_analysis)
                conversion_plan['project_structure_plan'] = project_structure
                state["project_structure_plan"] = project_structure
                self.logger.info(f"Generated multi-file project structure with {len(project_structure['cpp_files'])} C++ files")
            
            # Update state with planning results
            state["conversion_plan"] = conversion_plan
            
            self.logger.info("Conversion plan created successfully")
            
        except Exception as e:
            self.logger.error(f"Error in conversion planning: {e}")
            state["error_message"] = f"Planning failed: {str(e)}"
            state = update_state_status(state, ConversionStatus.FAILED)
        
        # Record processing time
        duration = time.time() - start_time
        state = add_processing_time(state, "planning", duration)
        
        return state
    
    def generate_cpp_code(self, state: ConversionState) -> ConversionState:
        """
        Generate C++ code using the C++ generator agent.
        
        Args:
            state: Current conversion state
            
        Returns:
            Updated state with generated code
        """
        start_time = time.time()
        
        try:
            self.logger.info("Generating C++ code...")
            state = update_state_status(state, ConversionStatus.GENERATING)
            
            # Use existing agent functionality
            matlab_analysis = state["matlab_analysis"]
            conversion_plan = state["conversion_plan"]
            conversion_mode = state["request"].conversion_mode
            
            if state["is_multi_file"] and 'project_structure_plan' in conversion_plan:
                # Multi-file project generation
                self.logger.info("Generating multi-file C++ project...")
                generated_code = self.cpp_generator.generate_project_code(
                    analysis=matlab_analysis,
                    conversion_plan=conversion_plan,
                    conversion_mode=conversion_mode
                )
            else:
                # Single-file project generation
                self.logger.info("Generating single-file C++ code...")
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
                
                generated_code = self.cpp_generator.generate_code(
                    matlab_summary=matlab_summary,
                    conversion_plan=conversion_plan,
                    conversion_mode=conversion_mode
                )
            
            # Update state with generated code
            state["generated_code"] = generated_code
            
            if isinstance(generated_code, dict) and 'files' in generated_code:
                self.logger.info(f"Generated {len(generated_code['files'])} C++ files")
            else:
                self.logger.info("Generated C++ code successfully")
            
        except Exception as e:
            self.logger.error(f"Error in C++ code generation: {e}")
            state["error_message"] = f"Code generation failed: {str(e)}"
            state = update_state_status(state, ConversionStatus.FAILED)
        
        # Record processing time
        duration = time.time() - start_time
        state = add_processing_time(state, "generation", duration)
        
        return state
    
    def assess_code_quality(self, state: ConversionState) -> ConversionState:
        """
        Assess code quality using the quality assessor agent.
        
        Args:
            state: Current conversion state
            
        Returns:
            Updated state with quality assessment
        """
        start_time = time.time()
        
        try:
            self.logger.info("Assessing code quality...")
            state = update_state_status(state, ConversionStatus.ASSESSING)
            
            # Use existing agent functionality
            generated_code = state["generated_code"]
            conversion_plan = state["conversion_plan"]
            conversion_mode = state["request"].conversion_mode
            
            # Prepare code for assessment
            if isinstance(generated_code, dict) and 'files' in generated_code:
                # Multi-file project - assess main file or concatenate all
                main_file = self._find_main_file(generated_code['files'])
                if main_file:
                    full_code = generated_code['files'][main_file]
                else:
                    full_code = "\n\n".join(generated_code['files'].values())
            else:
                # Single-file project
                full_code = (generated_code.get('header', '') + "\n" + generated_code.get('implementation', '')).strip()
            
            # Get MATLAB code for comparison
            matlab_code = self._get_matlab_code_content(state["matlab_analysis"])
            
            # Perform assessment
            assessment = self.quality_assessor.assess(
                code=full_code,
                matlab_code=matlab_code,
                conversion_plan=conversion_plan,
                conversion_mode=conversion_mode
            )
            
            # Update state with assessment results
            state["quality_scores"] = assessment.metrics
            state["assessment_reports"].append(f"Turn {state['current_turn']} assessment")
            
            self.logger.info(f"Quality assessment complete. Score: {assessment.metrics.get('algorithmic', 0.0):.1f}/10")
            
        except Exception as e:
            self.logger.error(f"Error in quality assessment: {e}")
            state["error_message"] = f"Quality assessment failed: {str(e)}"
            state = update_state_status(state, ConversionStatus.FAILED)
        
        # Record processing time
        duration = time.time() - start_time
        state = add_processing_time(state, "assessment", duration)
        
        return state
    
    def save_generated_code(self, state: ConversionState) -> ConversionState:
        """
        Save generated code to organized output structure.
        
        Args:
            state: Current conversion state
            
        Returns:
            Updated state with saved files information
        """
        start_time = time.time()
        
        try:
            self.logger.info("Saving generated code...")
            
            # Create organized output directory structure
            project_output_dir = Path(state["request"].output_dir) / state["request"].project_name
            project_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (project_output_dir / "generated_code").mkdir(exist_ok=True)
            (project_output_dir / "reports").mkdir(exist_ok=True)
            (project_output_dir / "debug").mkdir(exist_ok=True)
            
            state["project_output_dir"] = project_output_dir
            
            # Save generated code
            generated_code = state["generated_code"]
            version = f"v{state['current_turn'] + 1}"
            
            if isinstance(generated_code, dict) and 'files' in generated_code:
                # Multi-file project
                saved_files = self._save_multi_file_code(
                    generated_code, project_output_dir, state["request"].project_name, version
                )
            else:
                # Single-file project
                saved_files = self._save_single_file_code(
                    generated_code, project_output_dir, state["request"].project_name, version
                )
            
            state["generated_files"] = saved_files
            self.logger.info(f"Saved {len(saved_files)} files to {project_output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving generated code: {e}")
            state["error_message"] = f"File saving failed: {str(e)}"
            state = update_state_status(state, ConversionStatus.FAILED)
        
        # Record processing time
        duration = time.time() - start_time
        state = add_processing_time(state, "file_saving", duration)
        
        return state
    
    def _find_main_file(self, files: Dict[str, str]) -> str:
        """Find the main file from a dictionary of generated files."""
        main_candidates = ['main.cpp', 'skeleton_vessel.cpp']
        for candidate in main_candidates:
            if candidate in files:
                return candidate
        cpp_files = [f for f in files.keys() if f.endswith('.cpp')]
        return cpp_files[0] if cpp_files else None
    
    def _get_matlab_code_content(self, matlab_analysis: Dict[str, Any]) -> str:
        """Extract MATLAB code content from analysis."""
        content_parts = []
        file_analyses = matlab_analysis.get('file_analyses', [])
        for file_analysis in file_analyses:
            if 'parsed_structure' in file_analysis and hasattr(file_analysis['parsed_structure'], 'content'):
                content_parts.append(file_analysis['parsed_structure'].content)
            content_parts.append("")
        return "\n".join(content_parts)
    
    def _save_multi_file_code(self, code_result: Dict[str, Any], output_dir: Path, 
                            project_name: str, version: str) -> list:
        """Save multi-file C++ code to organized project directory."""
        saved_files = []
        
        # Save all generated files
        for filename, content in code_result['files'].items():
            file_path = output_dir / "generated_code" / f"{version}_{filename}"
            file_path.write_text(content, encoding='utf-8')
            saved_files.append(str(file_path))
        
        # Save compilation instructions if available
        if 'compilation_instructions' in code_result:
            instructions_file = output_dir / "generated_code" / f"{version}_compilation_instructions.md"
            instructions_file.write_text(code_result['compilation_instructions'], encoding='utf-8')
            saved_files.append(str(instructions_file))
        
        return saved_files
    
    def _save_single_file_code(self, code: Dict[str, str], output_dir: Path, 
                             project_name: str, version: str) -> list:
        """Save single-file C++ code to organized project directory."""
        saved_files = []
        
        if code.get('header'):
            header_file = output_dir / "generated_code" / f"{version}.h"
            header_file.write_text(code['header'])
            saved_files.append(str(header_file))
        
        if code.get('implementation'):
            impl_file = output_dir / "generated_code" / f"{version}.cpp"
            impl_file.write_text(code['implementation'])
            saved_files.append(str(impl_file))
        
        return saved_files
