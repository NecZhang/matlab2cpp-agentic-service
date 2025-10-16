"""Basic tests for the MATLAB to C++ conversion agentic service."""

import pytest
from pathlib import Path
import sys

# Ensure proper module imports (use PYTHONPATH instead of sys.path manipulation)
# If running directly, the parent directory should be in PYTHONPATH
# Example: PYTHONPATH=/path/to/matlab2cpp_agentic_service/src pytest


def test_core_imports():
    """Test that core components can be imported."""
    # Test core orchestrators
    from matlab2cpp_agentic_service.core.orchestrators.native_langgraph_orchestrator import NativeLangGraphMATLAB2CPPOrchestrator
    assert NativeLangGraphMATLAB2CPPOrchestrator is not None
    
    # Test infrastructure components
    from matlab2cpp_agentic_service.infrastructure.state.conversion_state import ConversionRequest, ConversionResult
    assert ConversionRequest is not None
    assert ConversionResult is not None
    
    # Test utilities
    from matlab2cpp_agentic_service.utils.config import get_config
    from matlab2cpp_agentic_service.utils.logger import setup_logger
    assert get_config is not None
    assert setup_logger is not None


def test_agent_imports():
    """Test that agent components can be imported."""
    # Test streamlined agents (v0.3.0 architecture)
    from matlab2cpp_agentic_service.core.agents.streamlined.matlab_analyzer import MATLABAnalyzer
    from matlab2cpp_agentic_service.core.agents.streamlined.conversion_planner import ConversionPlanner
    from matlab2cpp_agentic_service.core.agents.streamlined.cpp_generator import CppGenerator
    from matlab2cpp_agentic_service.core.agents.streamlined.quality_assessor import QualityAssessor
    
    assert MATLABAnalyzer is not None
    assert ConversionPlanner is not None
    assert CppGenerator is not None
    assert QualityAssessor is not None


def test_infrastructure_imports():
    """Test that infrastructure components can be imported."""
    # Test tools
    from matlab2cpp_agentic_service.infrastructure.tools.langgraph_tools import CodeGenerationTool, QualityAssessmentTool
    assert CodeGenerationTool is not None
    assert QualityAssessmentTool is not None
    
    # Test LLM client
    from matlab2cpp_agentic_service.infrastructure.tools.llm_client import LLMClient
    assert LLMClient is not None
    
    # Test MATLAB parser
    from matlab2cpp_agentic_service.infrastructure.tools.matlab_parser import MATLABParser
    assert MATLABParser is not None


def test_cli_imports():
    """Test that CLI components can be imported."""
    from matlab2cpp_agentic_service.cli.commands import cli
    assert cli is not None


def test_project_structure():
    """Test that project structure is properly set up."""
    project_root = Path(__file__).parent.parent
    
    # Core directories
    assert (project_root / "src" / "matlab2cpp_agentic_service").exists()
    assert (project_root / "src" / "matlab2cpp_agentic_service" / "core").exists()
    assert (project_root / "src" / "matlab2cpp_agentic_service" / "infrastructure").exists()
    assert (project_root / "src" / "matlab2cpp_agentic_service" / "cli").exists()
    
    # Example directories
    assert (project_root / "examples" / "matlab_samples").exists()
    
    # Configuration
    assert (project_root / "config" / "default_config.yaml").exists()
    
    # Documentation
    assert (project_root / "README.md").exists()


def test_basic_functionality():
    """Test basic functionality without LLM calls."""
    # Test MATLAB parser
    from matlab2cpp_agentic_service.infrastructure.tools.matlab_parser import MATLABParser
    parser = MATLABParser()
    assert parser is not None
    
    # Test configuration loading
    from matlab2cpp_agentic_service.utils.config import get_config
    config = get_config()
    assert config is not None


if __name__ == "__main__":
    pytest.main([__file__])


