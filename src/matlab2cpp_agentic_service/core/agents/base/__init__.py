"""
Base Agent Classes

This module provides the foundational classes for all agents in the
MATLAB2C++ conversion service, supporting both legacy and LangGraph paradigms.
"""

from .legacy_agent import LegacyAgent, LegacyAgentConfig
from .langgraph_agent import BaseLangGraphAgent, AgentConfig
from .agent_registry import AgentRegistry, AgentType

__all__ = [
    "LegacyAgent",
    "LegacyAgentConfig",
    "BaseLangGraphAgent",
    "AgentConfig",
    "AgentRegistry",
    "AgentType"
]