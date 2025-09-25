"""
Agent Registry

This module provides a registry system for managing agents in the MATLAB2C++ conversion service.
"""

from typing import Dict, Any, List, Optional, Type, Union
from enum import Enum
from dataclasses import dataclass
from loguru import logger

from .legacy_agent import LegacyAgent, LegacyAgentConfig
from .langgraph_agent import BaseLangGraphAgent, AgentConfig


class AgentType(Enum):
    """Types of agents supported by the system."""
    LEGACY = "legacy"
    LANGGRAPH = "langgraph"


@dataclass
class AgentInfo:
    """Information about a registered agent."""
    name: str
    agent_type: AgentType
    agent_class: Type[Union[LegacyAgent, BaseLangGraphAgent]]
    config_class: Type[Union[LegacyAgentConfig, AgentConfig]]
    description: str
    dependencies: List[str]
    tags: List[str]


class AgentRegistry:
    """
    Registry for managing agents.
    
    This class provides a centralized way to register, discover, and instantiate
    agents in the MATLAB2C++ conversion service.
    """
    
    def __init__(self):
        """Initialize the agent registry."""
        self.agents: Dict[str, AgentInfo] = {}
        self.logger = logger.bind(name="agent_registry")
    
    def register_agent(self, agent_info: AgentInfo) -> bool:
        """
        Register an agent in the registry.
        
        Args:
            agent_info: Information about the agent
            
        Returns:
            True if registration was successful
        """
        try:
            if agent_info.name in self.agents:
                self.logger.warning(f"Agent '{agent_info.name}' is already registered, overwriting")
            
            self.agents[agent_info.name] = agent_info
            self.logger.info(f"Registered agent: {agent_info.name} ({agent_info.agent_type.value})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register agent '{agent_info.name}': {e}")
            return False
    
    def unregister_agent(self, name: str) -> bool:
        """
        Unregister an agent from the registry.
        
        Args:
            name: Name of the agent to unregister
            
        Returns:
            True if unregistration was successful
        """
        if name not in self.agents:
            self.logger.warning(f"Agent '{name}' not found in registry")
            return False
        
        agent_info = self.agents.pop(name)
        self.logger.info(f"Unregistered agent: {name}")
        return True
    
    def get_agent_info(self, name: str) -> Optional[AgentInfo]:
        """
        Get information about a registered agent.
        
        Args:
            name: Name of the agent
            
        Returns:
            Agent information or None if not found
        """
        return self.agents.get(name)
    
    def list_agents(self, agent_type: Optional[AgentType] = None, 
                   tags: Optional[List[str]] = None) -> List[AgentInfo]:
        """
        List registered agents with optional filtering.
        
        Args:
            agent_type: Filter by agent type
            tags: Filter by tags (agent must have all specified tags)
            
        Returns:
            List of matching agent information
        """
        agents = list(self.agents.values())
        
        # Filter by type
        if agent_type:
            agents = [a for a in agents if a.agent_type == agent_type]
        
        # Filter by tags
        if tags:
            agents = [a for a in agents if all(tag in a.tags for tag in tags)]
        
        return agents
    
    def create_agent(self, name: str, config: Union[LegacyAgentConfig, AgentConfig], 
                    **kwargs) -> Optional[Union[LegacyAgent, BaseLangGraphAgent]]:
        """
        Create an instance of a registered agent.
        
        Args:
            name: Name of the agent to create
            config: Agent configuration
            **kwargs: Additional arguments for agent creation
            
        Returns:
            Agent instance or None if creation failed
        """
        agent_info = self.get_agent_info(name)
        if not agent_info:
            self.logger.error(f"Agent '{name}' not found in registry")
            return None
        
        try:
            # Create agent instance
            if agent_info.agent_type == AgentType.LEGACY:
                agent = agent_info.agent_class(config, **kwargs)
            elif agent_info.agent_type == AgentType.LANGGRAPH:
                agent = agent_info.agent_class(config, **kwargs)
            else:
                raise ValueError(f"Unknown agent type: {agent_info.agent_type}")
            
            self.logger.info(f"Created agent instance: {name}")
            return agent
            
        except Exception as e:
            self.logger.error(f"Failed to create agent '{name}': {e}")
            return None
    
    def get_agent_dependencies(self, name: str) -> List[str]:
        """
        Get dependencies for an agent.
        
        Args:
            name: Name of the agent
            
        Returns:
            List of dependency names
        """
        agent_info = self.get_agent_info(name)
        return agent_info.dependencies if agent_info else []
    
    def validate_dependencies(self, name: str) -> bool:
        """
        Validate that all dependencies for an agent are registered.
        
        Args:
            name: Name of the agent
            
        Returns:
            True if all dependencies are satisfied
        """
        agent_info = self.get_agent_info(name)
        if not agent_info:
            return False
        
        missing_deps = []
        for dep in agent_info.dependencies:
            if dep not in self.agents:
                missing_deps.append(dep)
        
        if missing_deps:
            self.logger.error(f"Agent '{name}' has missing dependencies: {missing_deps}")
            return False
        
        return True
    
    def get_agents_by_tag(self, tag: str) -> List[AgentInfo]:
        """
        Get all agents with a specific tag.
        
        Args:
            tag: Tag to search for
            
        Returns:
            List of agents with the specified tag
        """
        return [agent for agent in self.agents.values() if tag in agent.tags]
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the registry.
        
        Returns:
            Registry summary
        """
        total_agents = len(self.agents)
        legacy_agents = len([a for a in self.agents.values() if a.agent_type == AgentType.LEGACY])
        langgraph_agents = len([a for a in self.agents.values() if a.agent_type == AgentType.LANGGRAPH])
        
        all_tags = set()
        for agent in self.agents.values():
            all_tags.update(agent.tags)
        
        return {
            "total_agents": total_agents,
            "legacy_agents": legacy_agents,
            "langgraph_agents": langgraph_agents,
            "available_tags": list(all_tags),
            "agent_names": list(self.agents.keys())
        }
    
    def export_registry(self) -> Dict[str, Any]:
        """
        Export registry information.
        
        Returns:
            Registry data for export
        """
        return {
            "agents": {
                name: {
                    "name": info.name,
                    "agent_type": info.agent_type.value,
                    "description": info.description,
                    "dependencies": info.dependencies,
                    "tags": info.tags
                }
                for name, info in self.agents.items()
            },
            "summary": self.get_registry_summary()
        }


# Global registry instance
_global_registry: Optional[AgentRegistry] = None


def get_global_registry() -> AgentRegistry:
    """
    Get the global agent registry instance.
    
    Returns:
        Global agent registry
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = AgentRegistry()
    return _global_registry


def register_agent(agent_info: AgentInfo) -> bool:
    """
    Register an agent in the global registry.
    
    Args:
        agent_info: Information about the agent
        
    Returns:
        True if registration was successful
    """
    return get_global_registry().register_agent(agent_info)


def create_agent(name: str, config: Union[LegacyAgentConfig, AgentConfig], 
                **kwargs) -> Optional[Union[LegacyAgent, BaseLangGraphAgent]]:
    """
    Create an agent from the global registry.
    
    Args:
        name: Name of the agent to create
        config: Agent configuration
        **kwargs: Additional arguments
        
    Returns:
        Agent instance or None if creation failed
    """
    return get_global_registry().create_agent(name, config, **kwargs)

