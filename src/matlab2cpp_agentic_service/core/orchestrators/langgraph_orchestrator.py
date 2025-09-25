"""
LangGraph-Based MATLAB2C++ Orchestrator

This orchestrator uses LangGraph workflow for MATLAB to C++ conversion,
providing proper state management, conditional logic, and iterative optimization.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from loguru import logger

from ...infrastructure.state.conversion_state import (
    ConversionRequest, 
    ConversionResult, 
    ConversionStatus,
    create_initial_state,
    create_final_result
)
from ..workflows.langgraph_workflow import MATLAB2CPPLangGraphWorkflow


class MATLAB2CPPLangGraphOrchestrator:
    """
    LangGraph-based orchestrator for MATLAB2C++ conversion service.
    
    This orchestrator uses LangGraph workflow to manage the conversion process
    with proper state management, conditional logic, and iterative optimization.
    """
    
    def __init__(self):
        """Initialize the LangGraph orchestrator."""
        self.logger = logger.bind(name="langgraph_orchestrator")
        self.logger.info("LangGraph MATLAB2C++ Orchestrator initialized")
        
        # Initialize the workflow
        self.workflow = MATLAB2CPPLangGraphWorkflow()
    
    def convert_project(self, request: ConversionRequest) -> ConversionResult:
        """
        Convert MATLAB project to C++ using LangGraph workflow.
        
        Args:
            request: Conversion request with MATLAB path and requirements
            
        Returns:
            ConversionResult with generated files and assessment
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting LangGraph conversion for project: {request.project_name}")
            self.logger.info(f"MATLAB path: {request.matlab_path}")
            self.logger.info(f"Output directory: {request.output_dir}")
            self.logger.info(f"Max optimization turns: {request.max_optimization_turns}")
            self.logger.info(f"Target quality score: {request.target_quality_score}")
            self.logger.info(f"Conversion mode: {request.conversion_mode}")
            
            # Create initial state
            initial_state = create_initial_state(request)
            
            # Run the LangGraph workflow
            final_state = self.workflow.run_conversion_sync(initial_state)
            
            # Create final result
            result = create_final_result(final_state)
            
            # Log completion
            total_time = time.time() - start_time
            self.logger.info(f"LangGraph conversion completed in {total_time:.1f}s")
            self.logger.info(f"Final quality score: {result.final_score:.1f}/10")
            self.logger.info(f"Optimization turns used: {result.improvement_turns}")
            self.logger.info(f"Generated files: {len(result.generated_files)}")
            
            # Log workflow statistics
            workflow_stats = self.workflow.get_workflow_stats(final_state)
            self.logger.info("Workflow statistics:")
            for key, value in workflow_stats.items():
                self.logger.info(f"  {key}: {value}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in LangGraph conversion: {e}")
            return ConversionResult(
                status=ConversionStatus.FAILED,
                project_name=request.project_name,
                output_dir=request.output_dir,
                generated_files=[],
                original_score=0.0,
                final_score=0.0,
                improvement_turns=0,
                assessment_reports=[],
                conversion_plan=None,
                total_processing_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def convert_project_async(self, request: ConversionRequest) -> ConversionResult:
        """
        Convert MATLAB project to C++ using LangGraph workflow (async version).
        
        Args:
            request: Conversion request with MATLAB path and requirements
            
        Returns:
            ConversionResult with generated files and assessment
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting async LangGraph conversion for project: {request.project_name}")
            
            # Create initial state
            initial_state = create_initial_state(request)
            
            # Run the LangGraph workflow asynchronously
            final_state = await self.workflow.run_conversion(initial_state)
            
            # Create final result
            result = create_final_result(final_state)
            
            # Log completion
            total_time = time.time() - start_time
            self.logger.info(f"Async LangGraph conversion completed in {total_time:.1f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in async LangGraph conversion: {e}")
            return ConversionResult(
                status=ConversionStatus.FAILED,
                project_name=request.project_name,
                output_dir=request.output_dir,
                generated_files=[],
                original_score=0.0,
                final_score=0.0,
                improvement_turns=0,
                assessment_reports=[],
                conversion_plan=None,
                total_processing_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def get_workflow_diagram(self) -> str:
        """
        Get a visual representation of the workflow.
        
        Returns:
            Mermaid diagram of the workflow
        """
        return self.workflow.get_workflow_graph()
    
    def validate_request(self, request: ConversionRequest) -> bool:
        """
        Validate the conversion request.
        
        Args:
            request: Conversion request to validate
            
        Returns:
            True if request is valid, False otherwise
        """
        try:
            # Check if MATLAB path exists
            matlab_path = Path(request.matlab_path)
            if not matlab_path.exists():
                self.logger.error(f"MATLAB path does not exist: {matlab_path}")
                return False
            
            # Check if it's a file or directory
            if matlab_path.is_file():
                if not matlab_path.suffix == '.m':
                    self.logger.error(f"File is not a MATLAB file: {matlab_path}")
                    return False
            elif matlab_path.is_dir():
                # Check if directory contains .m files
                m_files = list(matlab_path.glob("*.m"))
                if not m_files:
                    self.logger.error(f"Directory contains no .m files: {matlab_path}")
                    return False
            
            # Validate output directory
            output_dir = Path(request.output_dir)
            if not output_dir.parent.exists():
                self.logger.error(f"Output directory parent does not exist: {output_dir.parent}")
                return False
            
            # Validate parameters
            if request.max_optimization_turns < 0:
                self.logger.error(f"Max optimization turns must be non-negative: {request.max_optimization_turns}")
                return False
            
            if not (0.0 <= request.target_quality_score <= 10.0):
                self.logger.error(f"Target quality score must be between 0.0 and 10.0: {request.target_quality_score}")
                return False
            
            if request.conversion_mode not in ["faithful", "result-focused"]:
                self.logger.error(f"Invalid conversion mode: {request.conversion_mode}")
                return False
            
            self.logger.info("Conversion request validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating request: {e}")
            return False
    
    def get_conversion_summary(self, result: ConversionResult) -> Dict[str, Any]:
        """
        Get a summary of the conversion result.
        
        Args:
            result: Conversion result to summarize
            
        Returns:
            Dictionary with conversion summary
        """
        return {
            "status": result.status.value,
            "quality_score": result.final_score,
            "improvement": result.final_score - result.original_score,
            "optimization_turns": result.improvement_turns,
            "generated_files": len(result.generated_files),
            "processing_time": result.total_processing_time,
            "has_errors": result.error_message is not None,
            "error_message": result.error_message
        }
