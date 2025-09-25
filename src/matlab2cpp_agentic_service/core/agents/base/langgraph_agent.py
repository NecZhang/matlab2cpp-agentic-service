"""
Base LangGraph Agent Class

This module provides the foundational class for all LangGraph-native agents
in the MATLAB2C++ conversion service.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
import time
import asyncio
from loguru import logger

from ....infrastructure.state.conversion_state import ConversionState, AgentMemory
from ....infrastructure.tools.llm_client import LLMClient


@dataclass
class AgentConfig:
    """Configuration for LangGraph agents."""
    name: str
    description: str
    max_retries: int = 3
    timeout_seconds: float = 300.0
    enable_memory: bool = True
    enable_performance_tracking: bool = True
    tools: List[str] = field(default_factory=list)


@dataclass
class AgentMemory:
    """Memory structure for individual agents."""
    short_term: Dict[str, Any] = field(default_factory=dict)
    long_term: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    performance_history: List[Dict[str, Any]] = field(default_factory=list)


class BaseLangGraphAgent(ABC):
    """
    Base class for all LangGraph-native agents.
    
    This class provides:
    - Standardized agent interface
    - Memory management
    - Performance tracking
    - Error handling and retry logic
    - Tool integration
    """
    
    def __init__(self, config: AgentConfig, llm_client: LLMClient, tools: Optional[List[Any]] = None):
        """
        Initialize the base LangGraph agent.
        
        Args:
            config: Agent configuration
            llm_client: LLM client for agent operations
            tools: List of tools available to this agent
        """
        self.config = config
        self.llm_client = llm_client
        self.tools = tools or []
        self.memory = AgentMemory()
        self.performance_history = []
        self.logger = logger.bind(name=f"langgraph_agent.{config.name}")
        
        # Initialize performance tracking
        if config.enable_performance_tracking:
            self._start_time = None
            self._execution_count = 0
            
        self.logger.info(f"Initialized LangGraph agent: {config.name}")
    
    @abstractmethod
    def create_node(self) -> Callable[[ConversionState], ConversionState]:
        """
        Create the LangGraph node function.
        
        Returns:
            Callable that takes and returns ConversionState
        """
        pass
    
    @abstractmethod
    def get_tools(self) -> List[Any]:
        """
        Get available tools for this agent.
        
        Returns:
            List of tools available to this agent
        """
        pass
    
    def update_memory(self, key: str, value: Any, memory_type: str = "short_term"):
        """
        Update agent memory.
        
        Args:
            key: Memory key
            value: Memory value
            memory_type: Type of memory ("short_term", "long_term", "context")
        """
        if memory_type == "short_term":
            self.memory.short_term[key] = value
        elif memory_type == "long_term":
            self.memory.long_term[key] = value
        elif memory_type == "context":
            self.memory.context[key] = value
        else:
            raise ValueError(f"Invalid memory type: {memory_type}")
            
        self.logger.debug(f"Updated {memory_type} memory: {key}")
    
    def get_memory(self, key: str, memory_type: str = "short_term") -> Any:
        """
        Get value from agent memory.
        
        Args:
            key: Memory key
            memory_type: Type of memory to retrieve from
            
        Returns:
            Memory value or None if not found
        """
        if memory_type == "short_term":
            return self.memory.short_term.get(key)
        elif memory_type == "long_term":
            return self.memory.long_term.get(key)
        elif memory_type == "context":
            return self.memory.context.get(key)
        else:
            raise ValueError(f"Invalid memory type: {memory_type}")
    
    def clear_memory(self, memory_type: str = "short_term"):
        """
        Clear agent memory.
        
        Args:
            memory_type: Type of memory to clear ("short_term", "long_term", "context", "all")
        """
        if memory_type == "short_term":
            self.memory.short_term.clear()
        elif memory_type == "long_term":
            self.memory.long_term.clear()
        elif memory_type == "context":
            self.memory.context.clear()
        elif memory_type == "all":
            self.memory.short_term.clear()
            self.memory.long_term.clear()
            self.memory.context.clear()
        else:
            raise ValueError(f"Invalid memory type: {memory_type}")
            
        self.logger.debug(f"Cleared {memory_type} memory")
    
    def track_performance(self, operation: str, start_time: float, end_time: float, 
                         success: bool, metadata: Optional[Dict[str, Any]] = None):
        """
        Track agent performance metrics.
        
        Args:
            operation: Operation name
            start_time: Operation start time
            end_time: Operation end time
            success: Whether operation was successful
            metadata: Additional metadata
        """
        if not self.config.enable_performance_tracking:
            return
            
        metrics = {
            "operation": operation,
            "execution_time": end_time - start_time,
            "success": success,
            "timestamp": time.time(),
            "agent": self.config.name,
            "metadata": metadata or {}
        }
        
        self.memory.performance_history.append(metrics)
        
        # Keep only last 100 entries to prevent memory bloat
        if len(self.memory.performance_history) > 100:
            self.memory.performance_history = self.memory.performance_history[-100:]
        
        self.logger.debug(f"Tracked performance: {operation} - {metrics['execution_time']:.2f}s - {'✓' if success else '✗'}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary for this agent.
        
        Returns:
            Performance summary dictionary
        """
        if not self.memory.performance_history:
            return {"total_operations": 0, "success_rate": 0.0, "avg_execution_time": 0.0}
        
        total_ops = len(self.memory.performance_history)
        successful_ops = sum(1 for entry in self.memory.performance_history if entry["success"])
        avg_time = sum(entry["execution_time"] for entry in self.memory.performance_history) / total_ops
        
        return {
            "total_operations": total_ops,
            "success_rate": successful_ops / total_ops,
            "avg_execution_time": avg_time,
            "recent_operations": self.memory.performance_history[-5:]  # Last 5 operations
        }
    
    async def execute_with_retry(self, operation: Callable, *args, **kwargs) -> Any:
        """
        Execute an operation with retry logic.
        
        Args:
            operation: Operation to execute
            *args: Arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Operation result
            
        Raises:
            Exception: If all retries fail
        """
        last_exception = None
        
        for attempt in range(self.config.max_retries):
            try:
                start_time = time.time()
                
                # Execute the operation
                if asyncio.iscoroutinefunction(operation):
                    result = await asyncio.wait_for(operation(*args, **kwargs), timeout=self.config.timeout_seconds)
                else:
                    result = operation(*args, **kwargs)
                
                end_time = time.time()
                
                # Track successful execution
                self.track_performance(operation.__name__, start_time, end_time, True)
                
                if attempt > 0:
                    self.logger.info(f"Operation succeeded on attempt {attempt + 1}")
                
                return result
                
            except Exception as e:
                last_exception = e
                end_time = time.time()
                
                # Track failed execution
                self.track_performance(operation.__name__, start_time, end_time, False, 
                                     {"error": str(e), "attempt": attempt + 1})
                
                self.logger.warning(f"Operation failed on attempt {attempt + 1}: {e}")
                
                if attempt < self.config.max_retries - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
        
        # All retries failed
        self.logger.error(f"Operation failed after {self.config.max_retries} attempts")
        raise last_exception
    
    def get_state_context(self, state: ConversionState) -> Dict[str, Any]:
        """
        Extract relevant context from conversion state.
        
        Args:
            state: Current conversion state
            
        Returns:
            Context dictionary for this agent
        """
        return {
            "request": state["request"],
            "current_turn": state["current_turn"],
            "is_multi_file": state["is_multi_file"],
            "agent_memory": state.get("agent_memory", {}).get(self.config.name, {}),
            "previous_results": state.get("previous_results", []),
            "error_context": state.get("error_context", {})
        }
    
    def update_state_with_result(self, state: ConversionState, result: Dict[str, Any], 
                               operation_name: str) -> ConversionState:
        """
        Update conversion state with agent result.
        
        Args:
            state: Current conversion state
            result: Agent operation result
            operation_name: Name of the operation
            
        Returns:
            Updated conversion state
        """
        # Update agent memory in state
        if "agent_memory" not in state:
            state["agent_memory"] = {}
        
        state["agent_memory"][self.config.name] = {
            "short_term": self.memory.short_term.copy(),
            "long_term": self.memory.long_term.copy(),
            "context": self.memory.context.copy(),
            "last_operation": operation_name,
            "last_result": result,
            "performance_summary": self.get_performance_summary()
        }
        
        # Update operation results
        if "operation_results" not in state:
            state["operation_results"] = {}
        
        state["operation_results"][operation_name] = {
            "agent": self.config.name,
            "result": result,
            "timestamp": time.time(),
            "success": True
        }
        
        return state
    
    def log_operation_start(self, operation_name: str, context: Dict[str, Any] = None):
        """Log the start of an operation."""
        self.logger.info(f"Starting {operation_name}")
        if context:
            self.logger.debug(f"Operation context: {context}")
    
    def log_operation_end(self, operation_name: str, success: bool, duration: float):
        """Log the end of an operation."""
        status = "completed successfully" if success else "failed"
        self.logger.info(f"{operation_name} {status} in {duration:.2f}s")
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return f"LangGraphAgent(name={self.config.name}, tools={len(self.tools)}, memory_size={len(self.memory.short_term) + len(self.memory.long_term)})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the agent."""
        return f"BaseLangGraphAgent(config={self.config.name}, llm_client={type(self.llm_client).__name__}, tools={len(self.tools)})"
