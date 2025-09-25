"""
Legacy Agent Base Class

This module provides the base class for legacy agents in the MATLAB2C++ conversion service.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import time
from loguru import logger

from ....infrastructure.tools.llm_client import LLMClient


@dataclass
class LegacyAgentConfig:
    """Configuration for legacy agents."""
    name: str
    description: str
    max_retries: int = 3
    timeout_seconds: float = 300.0
    enable_logging: bool = True


class LegacyAgent(ABC):
    """
    Base class for legacy agents.
    
    This class provides the foundation for traditional agents that don't use
    LangGraph but still need common functionality.
    """
    
    def __init__(self, config: LegacyAgentConfig, llm_client: Optional[LLMClient] = None):
        """
        Initialize the legacy agent.
        
        Args:
            config: Agent configuration
            llm_client: Optional LLM client
        """
        self.config = config
        self.llm_client = llm_client
        self.logger = logger.bind(name=f"legacy_agent.{config.name}")
        
        if config.enable_logging:
            self.logger.info(f"Initialized legacy agent: {config.name}")
    
    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's main functionality.
        
        Args:
            input_data: Input data for the agent
            
        Returns:
            Agent execution result
        """
        pass
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if valid, False otherwise
        """
        return True
    
    def handle_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle errors during agent execution.
        
        Args:
            error: The exception that occurred
            context: Context information
            
        Returns:
            Error handling result
        """
        self.logger.error(f"Agent {self.config.name} failed: {error}", exc_info=True)
        return {
            "success": False,
            "error": str(error),
            "context": context
        }
    
    def execute_with_retry(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute agent with retry logic.
        
        Args:
            input_data: Input data for the agent
            
        Returns:
            Agent execution result
        """
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                # Validate input
                if not self.validate_input(input_data):
                    raise ValueError("Invalid input data")
                
                # Execute agent
                start_time = time.time()
                result = self.execute(input_data)
                execution_time = time.time() - start_time
                
                # Add execution metadata
                result["execution_time"] = execution_time
                result["attempt"] = attempt + 1
                result["agent"] = self.config.name
                
                if attempt > 0:
                    self.logger.info(f"Agent {self.config.name} succeeded on attempt {attempt + 1}")
                
                return result
                
            except Exception as e:
                last_error = e
                self.logger.warning(f"Agent {self.config.name} failed on attempt {attempt + 1}: {e}")
                
                if attempt < self.config.max_retries - 1:
                    # Wait before retry
                    time.sleep(2 ** attempt)
        
        # All retries failed
        return self.handle_error(last_error, {"input_data": input_data})
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get agent status information.
        
        Returns:
            Status information
        """
        return {
            "name": self.config.name,
            "description": self.config.description,
            "max_retries": self.config.max_retries,
            "timeout_seconds": self.config.timeout_seconds,
            "has_llm_client": self.llm_client is not None
        }
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return f"LegacyAgent(name={self.config.name})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the agent."""
        return f"LegacyAgent(config={self.config.name}, llm_client={type(self.llm_client).__name__ if self.llm_client else None})"

