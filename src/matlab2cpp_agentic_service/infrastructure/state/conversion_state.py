"""
Enhanced Conversion State for LangGraph Agents

This module provides the enhanced state management for LangGraph-based
MATLAB2C++ conversion workflows with advanced features like agent memory,
human feedback, and performance tracking.
"""

from typing import TypedDict, List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import time
import json

# Define the models directly here since they were moved from models directory

class ConversionStatus(Enum):
    """Conversion status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    GENERATING = "generating"
    ASSESSING = "assessing"
    OPTIMIZING = "optimizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ConversionRequest:
    """Request for MATLAB to C++ conversion."""
    matlab_path: Path
    project_name: str
    output_dir: Path
    max_optimization_turns: int = 2
    target_quality_score: float = 7.0
    conversion_mode: str = "result-focused"  # "faithful" or "result-focused"
    llm_config: Optional[Dict[str, Any]] = None
    custom_instructions: Optional[str] = None


@dataclass
class ConversionResult:
    """Result of MATLAB to C++ conversion."""
    status: Union[str, ConversionStatus]
    project_name: str
    output_dir: Path
    generated_files: List[Path]
    quality_score: float = 0.0
    original_score: float = 0.0
    final_score: float = 0.0
    improvement_turns: int = 0
    conversion_plan: Optional[Dict[str, Any]] = None
    total_processing_time: float = 0.0
    assessment_reports: List[Dict[str, Any]] = field(default_factory=list)
    final_code_path: Optional[Path] = None
    report_path: Optional[Path] = None
    errors: List[str] = field(default_factory=list)
    success: bool = False
    error_message: Optional[str] = None


class AgentMemory(TypedDict):
    """Individual agent memory structure."""
    short_term: Dict[str, Any]
    long_term: Dict[str, Any]
    context: Dict[str, Any]
    performance_history: List[Dict[str, Any]]
    last_operation: Optional[str]
    last_result: Optional[Dict[str, Any]]
    performance_summary: Dict[str, Any]


class HumanFeedback(TypedDict):
    """Human feedback structure."""
    feedback_type: str  # "approval", "rejection", "suggestion", "correction"
    message: str
    timestamp: float
    agent: str
    operation: str
    context: Dict[str, Any]
    action_required: bool


class SystemMetrics(TypedDict):
    """System performance metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_free_gb: float
    active_processes: int


class OperationResult(TypedDict):
    """Result of an individual operation."""
    operation_id: str
    agent: str
    operation_name: str
    result: Dict[str, Any]
    timestamp: float
    success: bool
    execution_time: float
    error_message: Optional[str]


class ConversionState(TypedDict):
    """
    Enhanced conversion state for LangGraph workflows.
    
    This state includes:
    - Core conversion data
    - Agent-specific memory
    - Human feedback integration
    - Performance tracking
    - Error recovery state
    - Streaming updates
    """
    # Core conversion data
    request: ConversionRequest
    result: ConversionResult
    
    # Enhanced state management
    agent_memory: Dict[str, AgentMemory]
    human_feedback: List[HumanFeedback]
    streaming_updates: List[Dict[str, Any]]
    
    # Advanced workflow control
    parallel_tasks: Dict[str, Any]
    checkpoint_data: Dict[str, Any]
    error_recovery_state: Optional[Dict[str, Any]]
    
    # Performance and monitoring
    agent_performance: Dict[str, Dict[str, float]]
    system_metrics: List[SystemMetrics]
    operation_results: Dict[str, OperationResult]
    
    # Workflow state
    current_agent: Optional[str]
    current_operation: Optional[str]
    workflow_step: str
    needs_human_intervention: bool
    optimization_complete: bool
    
    # Processing tracking
    processing_times: Dict[str, float]
    total_processing_time: float
    start_time: float
    
    # Error handling
    error_context: Dict[str, Any]
    retry_count: int
    max_retries: int
    
    # LangGraph workflow specific fields
    matlab_analysis: Optional[Dict[str, Any]]
    is_multi_file: bool
    conversion_plan: Optional[Dict[str, Any]]
    project_structure_plan: Optional[Dict[str, Any]]
    generated_code: Optional[Dict[str, Any]]
    quality_scores: Optional[Dict[str, float]]
    assessment_reports: List[str]
    current_turn: int
    project_output_dir: Optional[Path]
    generated_files: List[str]
    error_message: Optional[str]


def create_initial_state(request: ConversionRequest) -> ConversionState:
    """
    Create the initial state for LangGraph workflow.
    
    Args:
        request: Conversion request
        
    Returns:
        Initial conversion state
    """
    current_time = time.time()
    
    return ConversionState(
        # Core data
        request=request,
        result=ConversionResult(
            status=ConversionStatus.PENDING,
            project_name=request.project_name,
            output_dir=request.output_dir,
            generated_files=[],
            original_score=0.0,
            final_score=0.0,
            improvement_turns=0,
            assessment_reports=[],
            conversion_plan=None,
            total_processing_time=0.0
        ),
        
        # Enhanced state management
        agent_memory={},
        human_feedback=[],
        streaming_updates=[],
        
        # Advanced workflow control
        parallel_tasks={},
        checkpoint_data={},
        error_recovery_state=None,
        
        # Performance and monitoring
        agent_performance={},
        system_metrics=[],
        operation_results={},
        
        # Workflow state
        current_agent=None,
        current_operation=None,
        workflow_step="initialization",
        needs_human_intervention=False,
        optimization_complete=False,
        
        # Processing tracking
        processing_times={},
        total_processing_time=0.0,
        start_time=current_time,
        
        # Error handling
        error_context={},
        retry_count=0,
        max_retries=3,
        
        # LangGraph workflow specific fields
        matlab_analysis=None,
        is_multi_file=False,
        conversion_plan=None,
        project_structure_plan=None,
        generated_code=None,
        quality_scores=None,
        assessment_reports=[],
        current_turn=0,
        project_output_dir=None,
        generated_files=[],
        error_message=None
    )


def update_state_status(state: ConversionState, status: ConversionStatus, 
                       error_message: Optional[str] = None) -> ConversionState:
    """
    Update the status of the conversion result in the state.
    
    Args:
        state: Current conversion state
        status: New status
        error_message: Optional error message
        
    Returns:
        Updated conversion state
    """
    state["result"].status = status
    if error_message:
        state["result"].error_message = error_message
        state["error_context"]["last_error"] = error_message
        state["error_context"]["error_timestamp"] = time.time()
    
    # Update workflow step based on status
    status_to_step = {
        ConversionStatus.PENDING: "pending",
        ConversionStatus.ANALYZING: "analyzing",
        ConversionStatus.PLANNING: "planning", 
        ConversionStatus.GENERATING: "generating",
        ConversionStatus.ASSESSING: "assessing",
        ConversionStatus.OPTIMIZING: "optimizing",
        ConversionStatus.COMPLETED: "completed",
        ConversionStatus.FAILED: "failed"
    }
    
    state["workflow_step"] = status_to_step.get(status, "unknown")
    
    return state


def add_processing_time(state: ConversionState, operation: str, duration: float) -> ConversionState:
    """
    Add processing time for an operation.
    
    Args:
        state: Current conversion state
        operation: Operation name
        duration: Duration in seconds
        
    Returns:
        Updated conversion state
    """
    state["processing_times"][operation] = duration
    state["total_processing_time"] = sum(state["processing_times"].values())
    
    # Update result total processing time
    state["result"].total_processing_time = state["total_processing_time"]
    
    return state


def update_agent_memory(state: ConversionState, agent_name: str, 
                       memory_data: AgentMemory) -> ConversionState:
    """
    Update agent memory in the state.
    
    Args:
        state: Current conversion state
        agent_name: Name of the agent
        memory_data: Agent memory data
        
    Returns:
        Updated conversion state
    """
    state["agent_memory"][agent_name] = memory_data
    return state


def get_agent_memory(state: ConversionState, agent_name: str) -> Optional[AgentMemory]:
    """
    Get agent memory from the state.
    
    Args:
        state: Current conversion state
        agent_name: Name of the agent
        
    Returns:
        Agent memory or None if not found
    """
    return state["agent_memory"].get(agent_name)


def add_human_feedback(state: ConversionState, feedback: HumanFeedback) -> ConversionState:
    """
    Add human feedback to the state.
    
    Args:
        state: Current conversion state
        feedback: Human feedback data
        
    Returns:
        Updated conversion state
    """
    state["human_feedback"].append(feedback)
    
    # Set intervention flag if action is required
    if feedback.get("action_required", False):
        state["needs_human_intervention"] = True
    
    return state


def add_streaming_update(state: ConversionState, update: Dict[str, Any]) -> ConversionState:
    """
    Add streaming update to the state.
    
    Args:
        state: Current conversion state
        update: Streaming update data
        
    Returns:
        Updated conversion state
    """
    update["timestamp"] = time.time()
    state["streaming_updates"].append(update)
    
    # Keep only last 100 updates to prevent memory bloat
    if len(state["streaming_updates"]) > 100:
        state["streaming_updates"] = state["streaming_updates"][-100:]
    
    return state


def update_system_metrics(state: ConversionState, metrics: SystemMetrics) -> ConversionState:
    """
    Update system metrics in the state.
    
    Args:
        state: Current conversion state
        metrics: System metrics data
        
    Returns:
        Updated conversion state
    """
    state["system_metrics"].append(metrics)
    
    # Keep only last 50 metrics to prevent memory bloat
    if len(state["system_metrics"]) > 50:
        state["system_metrics"] = state["system_metrics"][-50:]
    
    return state


def add_operation_result(state: ConversionState, result: OperationResult) -> ConversionState:
    """
    Add operation result to the state.
    
    Args:
        state: Current conversion state
        result: Operation result data
        
    Returns:
        Updated conversion state
    """
    state["operation_results"][result["operation_id"]] = result
    
    # Update agent performance if available
    if "performance_metrics" in result["result"]:
        agent_name = result["agent"]
        if agent_name not in state["agent_performance"]:
            state["agent_performance"][agent_name] = {}
        
        perf_metrics = result["result"]["performance_metrics"]
        state["agent_performance"][agent_name].update(perf_metrics)
    
    return state


def create_checkpoint(state: ConversionState, checkpoint_name: str) -> ConversionState:
    """
    Create a checkpoint of the current state.
    
    Args:
        state: Current conversion state
        checkpoint_name: Name of the checkpoint
        
    Returns:
        Updated conversion state with checkpoint
    """
    checkpoint_data = {
        "timestamp": time.time(),
        "workflow_step": state["workflow_step"],
        "current_agent": state["current_agent"],
        "operation_results": state["operation_results"].copy(),
        "agent_memory": {k: v.copy() for k, v in state["agent_memory"].items()},
        "processing_times": state["processing_times"].copy(),
        "result": {
            "status": state["result"]["status"],
            "generated_files": state["result"]["generated_files"].copy(),
            "assessment_reports": state["result"]["assessment_reports"].copy()
        }
    }
    
    state["checkpoint_data"][checkpoint_name] = checkpoint_data
    return state


def restore_checkpoint(state: ConversionState, checkpoint_name: str) -> ConversionState:
    """
    Restore state from a checkpoint.
    
    Args:
        state: Current conversion state
        checkpoint_name: Name of the checkpoint to restore
        
    Returns:
        Restored conversion state
    """
    if checkpoint_name not in state["checkpoint_data"]:
        raise ValueError(f"Checkpoint not found: {checkpoint_name}")
    
    checkpoint = state["checkpoint_data"][checkpoint_name]
    
    # Restore key state components
    state["workflow_step"] = checkpoint["workflow_step"]
    state["current_agent"] = checkpoint["current_agent"]
    state["operation_results"] = checkpoint["operation_results"]
    state["agent_memory"] = checkpoint["agent_memory"]
    state["processing_times"] = checkpoint["processing_times"]
    
    # Restore result data
    state["result"]["status"] = checkpoint["result"]["status"]
    state["result"]["generated_files"] = checkpoint["result"]["generated_files"]
    state["result"]["assessment_reports"] = checkpoint["result"]["assessment_reports"]
    
    return state


def merge_agent_results(state: ConversionState, agent_name: str, 
                       results: Dict[str, Any]) -> ConversionState:
    """
    Merge agent results into the state.
    
    Args:
        state: Current conversion state
        agent_name: Name of the agent
        results: Results from the agent
        
    Returns:
        Updated conversion state
    """
    # Update agent memory with results
    if agent_name not in state["agent_memory"]:
        state["agent_memory"][agent_name] = {
            "short_term": {},
            "long_term": {},
            "context": {},
            "performance_history": [],
            "last_operation": None,
            "last_result": None,
            "performance_summary": {}
        }
    
    state["agent_memory"][agent_name]["last_result"] = results
    state["agent_memory"][agent_name]["last_operation"] = results.get("operation_name", "unknown")
    
    # Add to operation results
    operation_id = f"{agent_name}_{results.get('operation_name', 'unknown')}_{time.time()}"
    operation_result = OperationResult(
        operation_id=operation_id,
        agent=agent_name,
        operation_name=results.get("operation_name", "unknown"),
        result=results,
        timestamp=time.time(),
        success=results.get("success", True),
        execution_time=results.get("execution_time", 0.0),
        error_message=results.get("error_message")
    )
    
    return add_operation_result(state, operation_result)


def validate_state(state: ConversionState) -> Dict[str, Any]:
    """
    Validate the conversion state.
    
    Args:
        state: Conversion state to validate
        
    Returns:
        Validation result with success flag and any errors
    """
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "recommendations": []
    }
    
    # Check required fields
    required_fields = ["request", "result", "workflow_step", "start_time"]
    for field in required_fields:
        if field not in state:
            validation_result["errors"].append(f"Missing required field: {field}")
            validation_result["valid"] = False
    
    # Validate request
    if "request" in state:
        request = state["request"]
        if not hasattr(request, 'project_name') or not request.project_name:
            validation_result["errors"].append("Request missing project_name")
            validation_result["valid"] = False
    
    # Validate result
    if "result" in state:
        result = state["result"]
        if not hasattr(result, 'status'):
            validation_result["errors"].append("Result missing status")
            validation_result["valid"] = False
    
    # Check for potential issues
    if state.get("retry_count", 0) > state.get("max_retries", 3):
        validation_result["warnings"].append("Retry count exceeds maximum")
    
    if state.get("total_processing_time", 0) > 3600:  # More than 1 hour
        validation_result["warnings"].append("Processing time is very long")
    
    if len(state.get("streaming_updates", [])) > 50:
        validation_result["recommendations"].append("Consider reducing streaming update frequency")
    
    return validation_result


def create_final_result(state: ConversionState) -> ConversionResult:
    """
    Create the final conversion result from the state.
    
    Args:
        state: Final conversion state
        
    Returns:
        ConversionResult with all the final data
    """
    result = state["result"]
    
    # Update the result with final state data
    # Check both status and workflow_step for completion
    status = state.get("status")
    workflow_step = state.get("workflow_step", "")
    
    if status == ConversionStatus.COMPLETED or workflow_step == "completed":
        result.status = ConversionStatus.COMPLETED
    elif status == ConversionStatus.FAILED or workflow_step == "failed":
        result.status = ConversionStatus.FAILED
    else:
        result.status = ConversionStatus.FAILED  # Default to failed if unclear
    result.total_processing_time = state.get("total_processing_time", 0.0)
    
    # Update quality scores
    quality_scores = state.get("quality_scores", {})
    if quality_scores:
        result.final_score = quality_scores.get("overall", 0.0)
        result.original_score = result.final_score  # For now, assume same as final
    
    # Update generated files
    generated_files = state.get("generated_files", [])
    if generated_files:
        result.generated_files = [Path(f) for f in generated_files]
        # Set the main C++ file path
        cpp_files = [f for f in generated_files if f.endswith('.cpp')]
        if cpp_files:
            result.final_code_path = Path(cpp_files[0])
        
        # Set report path if available
        report_files = [f for f in generated_files if f.endswith('.md') or 'report' in f]
        if report_files:
            result.report_path = Path(report_files[0])
    
    # Update assessment reports
    result.assessment_reports = state.get("assessment_reports", [])
    
    # Set success flag based on status and generated files
    result.success = (result.status == ConversionStatus.COMPLETED and 
                     len(result.generated_files) > 0 and 
                     not state.get("error_message"))
    
    # Set errors if any
    if state.get("error_message"):
        result.errors = [state["error_message"]]
        result.error_message = state["error_message"]
    
    return result


def export_state_summary(state: ConversionState) -> Dict[str, Any]:
    """
    Export a summary of the conversion state.
    
    Args:
        state: Conversion state to summarize
        
    Returns:
        State summary dictionary
    """
    return {
        "workflow_step": state["workflow_step"],
        "current_agent": state["current_agent"],
        "processing_time": state["total_processing_time"],
        "status": state["result"]["status"].value if hasattr(state["result"]["status"], 'value') else str(state["result"]["status"]),
        "generated_files": len(state["result"]["generated_files"]),
        "agent_count": len(state["agent_memory"]),
        "operation_count": len(state["operation_results"]),
        "feedback_count": len(state["human_feedback"]),
        "update_count": len(state["streaming_updates"]),
        "needs_intervention": state["needs_human_intervention"],
        "retry_count": state["retry_count"],
        "timestamp": time.time()
    }
