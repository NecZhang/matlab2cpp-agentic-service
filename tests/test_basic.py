"""Basic tests for the MATLAB to C++ conversion agentic service."""

import pytest
from pathlib import Path
import sys

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


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
    # Test LangGraph agents
    from matlab2cpp_agentic_service.core.agents.analyzer.langgraph.matlab_analyzer_agent import LangGraphMATLABAnalyzerAgent
    from matlab2cpp_agentic_service.core.agents.planner.langgraph.conversion_planner_agent import LangGraphConversionPlannerAgent
    from matlab2cpp_agentic_service.core.agents.generator.langgraph.cpp_generator_agent import LangGraphCppGeneratorAgent
    from matlab2cpp_agentic_service.core.agents.assessor.langgraph.quality_assessor_agent import LangGraphQualityAssessorAgent
    
    assert LangGraphMATLABAnalyzerAgent is not None
    assert LangGraphConversionPlannerAgent is not None
    assert LangGraphCppGeneratorAgent is not None
    assert LangGraphQualityAssessorAgent is not None


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
    assert (project_root / "README_DETAILS.md").exists()


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


