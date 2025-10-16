"""
Base Agent Classes

This module provides the foundational classes for all agents in the
MATLAB2C++ conversion service.
"""

from .langgraph_agent import BaseLangGraphAgent, AgentConfig

__all__ = [
    "BaseLangGraphAgent",
    "AgentConfig"
]