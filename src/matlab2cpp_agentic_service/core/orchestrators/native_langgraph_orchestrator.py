"""
Native LangGraph-Based MATLAB2C++ Orchestrator

This orchestrator uses truly native LangGraph agents and workflows for MATLAB to C++ conversion,
providing full utilization of LangGraph features including tools, memory, and state management.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from loguru import logger

from matlab2cpp_agentic_service.infrastructure.state.conversion_state import (
    ConversionRequest,
    ConversionResult,
    ConversionStatus,
    create_initial_state,
    create_final_result
)
from ..workflows.native_langgraph_workflow import NativeLangGraphMATLAB2CPPWorkflow


class NativeLangGraphMATLAB2CPPOrchestrator:
    """
    Native LangGraph-based orchestrator for MATLAB2C++ conversion service.

    This orchestrator uses truly native LangGraph agents and workflows to manage the conversion process
    with full utilization of LangGraph features including:
    - Native agent memory management
    - LangGraph tools integration
    - Advanced state management
    - Conditional logic and optimization
    - Human-in-the-loop capabilities
    """

    def __init__(self):
        """Initialize the native LangGraph orchestrator."""
        self.logger = logger.bind(name="native_langgraph_orchestrator")
        self.logger.info("Native LangGraph MATLAB2C++ Orchestrator initialized")

        # Initialize the native workflow
        self.workflow = NativeLangGraphMATLAB2CPPWorkflow()

    def convert_project(self, request: ConversionRequest) -> ConversionResult:
        """
        Convert MATLAB project to C++ using native LangGraph workflow.

        Args:
            request: Conversion request with MATLAB path and requirements

        Returns:
            ConversionResult with generated files and assessment
        """
        start_time = time.time()

        try:
            self.logger.info(f"Starting native LangGraph conversion for project: {request.project_name}")
            self.logger.info(f"MATLAB path: {request.matlab_path}")
            self.logger.info(f"Output directory: {request.output_dir}")
            self.logger.info(f"Max optimization turns: {request.max_optimization_turns}")
            self.logger.info(f"Target quality score: {request.target_quality_score}")
            self.logger.info(f"Conversion mode: {request.conversion_mode}")

            # Validate request
            if not self.validate_request(request):
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
                    error_message="Request validation failed"
                )

            # Create initial state
            initial_state = create_initial_state(request)

            # Run the native LangGraph workflow
            final_state = self.workflow.run_conversion_sync(initial_state)

            # Create final result
            result = create_final_result(final_state)

            # Log completion
            total_time = time.time() - start_time
            self.logger.info(f"Native LangGraph conversion completed in {total_time:.1f}s")
            self.logger.info(f"Final quality score: {result.final_score:.1f}/10")
            self.logger.info(f"Optimization turns used: {result.improvement_turns}")
            self.logger.info(f"Generated files: {len(result.generated_files)}")

            # Log workflow statistics
            workflow_stats = self.workflow.get_workflow_stats(final_state)
            self.logger.info("Native workflow statistics:")
            for key, value in workflow_stats.items():
                if key != "agent_performance":  # Log agent performance separately
                    self.logger.info(f"  {key}: {value}")

            # Log agent performance statistics
            agent_performance = workflow_stats.get("agent_performance", {})
            self.logger.info("Agent performance statistics:")
            for agent_name, perf in agent_performance.items():
                self.logger.info(f"  {agent_name}: {perf.get('total_operations', 0)} operations, "
                               f"{perf.get('success_rate', 0.0):.1%} success rate, "
                               f"{perf.get('avg_execution_time', 0.0):.2f}s avg time")

            # Log agent memory statistics
            memory_summary = self.workflow.get_agent_memory_summary()
            self.logger.info("Agent memory statistics:")
            for agent_name, memory in memory_summary.items():
                total_memory = (memory.get('short_term_memory_size', 0) + 
                              memory.get('long_term_memory_size', 0) + 
                              memory.get('context_memory_size', 0))
                self.logger.info(f"  {agent_name}: {total_memory} memory entries, "
                               f"{memory.get('operation_count', 0)} operations")

            return result

        except Exception as e:
            self.logger.error(f"Error in native LangGraph conversion: {e}")
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
        Convert MATLAB project to C++ using native LangGraph workflow (async version).

        Args:
            request: Conversion request with MATLAB path and requirements

        Returns:
            ConversionResult with generated files and assessment
        """
        start_time = time.time()

        try:
            self.logger.info(f"Starting async native LangGraph conversion for project: {request.project_name}")

            # Validate request
            if not self.validate_request(request):
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
                    error_message="Request validation failed"
                )

            # Create initial state
            initial_state = create_initial_state(request)

            # Run the native LangGraph workflow asynchronously
            final_state = await self.workflow.run_conversion(initial_state)

            # Create final result
            result = create_final_result(final_state)

            # Log completion
            total_time = time.time() - start_time
            self.logger.info(f"Async native LangGraph conversion completed in {total_time:.1f}s")

            return result

        except Exception as e:
            self.logger.error(f"Error in async native LangGraph conversion: {e}")
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
        Get a visual representation of the native workflow.

        Returns:
            Mermaid diagram of the native workflow
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

            self.logger.info("Native LangGraph conversion request validation passed")
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

    def get_agent_memory_summary(self) -> Dict[str, Any]:
        """
        Get summary of all agent memory states.

        Returns:
            Dictionary with agent memory summaries
        """
        return self.workflow.get_agent_memory_summary()

    def clear_agent_memory(self, memory_type: str = "all"):
        """
        Clear memory for all agents.

        Args:
            memory_type: Type of memory to clear ("short_term", "long_term", "context", "all")
        """
        self.workflow.clear_agent_memory(memory_type)
        self.logger.info(f"Cleared {memory_type} memory for all agents in native orchestrator")

    def get_workflow_capabilities(self) -> Dict[str, Any]:
        """
        Get information about the native LangGraph workflow capabilities.

        Returns:
            Dictionary with workflow capabilities
        """
        return {
            "workflow_type": "native_langgraph",
            "features": [
                "native_agent_memory_management",
                "langgraph_tools_integration",
                "advanced_state_management",
                "conditional_logic_optimization",
                "human_in_the_loop_capabilities",
                "streaming_support",
                "checkpointing_support",
                "parallel_execution_support",
                "agent_performance_tracking",
                "memory_persistence"
            ],
            "agents": [
                "LangGraphMATLABAnalyzerAgent",
                "LangGraphConversionPlannerAgent", 
                "LangGraphCppGeneratorAgent",
                "LangGraphQualityAssessorAgent"
            ],
            "tools": [
                "MATLABParserTool",
                "LLMAnalysisTool",
                "CodeGenerationTool",
                "QualityAssessmentTool"
            ],
            "memory_types": [
                "short_term_memory",
                "long_term_memory", 
                "context_memory",
                "performance_history"
            ]
        }
