"""Agent system for MATLAB to C++ conversion."""

# Base classes
from .base import BaseLangGraphAgent, LegacyAgent, AgentRegistry

# Legacy agents
from .analyzer.legacy import MATLABContentAnalyzerAgent, AgentTools
from .planner.legacy import ConversionPlannerAgent
from .generator.legacy import CppGeneratorAgent
from .assessor.legacy import QualityAssessorAgent
from .validator.legacy import ValidatorAgent

# LangGraph agents
from .analyzer.langgraph import LangGraphMATLABAnalyzerAgent
from .planner.langgraph import LangGraphConversionPlannerAgent
from .generator.langgraph import LangGraphCppGeneratorAgent
from .assessor.langgraph import LangGraphQualityAssessorAgent

__all__ = [
    # Base classes
    "BaseLangGraphAgent",
    "LegacyAgent", 
    "AgentRegistry",
    
    # Legacy agents
    "MATLABContentAnalyzerAgent",
    "AgentTools",
    "ConversionPlannerAgent",
    "CppGeneratorAgent",
    "QualityAssessorAgent",
    "ValidatorAgent",
    
    # LangGraph agents
    "LangGraphMATLABAnalyzerAgent",
    "LangGraphConversionPlannerAgent",
    "LangGraphCppGeneratorAgent",
    "LangGraphQualityAssessorAgent",
]