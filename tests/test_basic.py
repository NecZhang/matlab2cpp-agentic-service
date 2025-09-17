"""Basic tests for the MATLAB to C++ conversion agent."""

import pytest
from pathlib import Path
import sys

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from matlab2cpp_agentic_service.utils.config import Config, get_config
from matlab2cpp_agentic_service.tools.matlab_parser import MATLABParser
from matlab2cpp_agentic_service.agents.matlab_content_analyzer import MATLABContentAnalyzerAgent


def test_config_loading():
    """Test configuration loading."""
    config = get_config()
    assert config is not None
    assert config.llm.model == "gpt-4"
    assert config.analysis.max_file_size == 100000


def test_matlab_parser():
    """Test MATLAB parser basic functionality."""
    parser = MATLABParser()
    assert parser is not None


def test_matlab_analyzer():
    """Test MATLAB analyzer basic functionality."""
    # Note: This test would need LLM client setup to work properly
    # analyzer = MATLABContentAnalyzerAgent(llm_client)
    # assert analyzer is not None
    pass


def test_project_structure():
    """Test that project structure is properly set up."""
    project_root = Path(__file__).parent.parent
    assert (project_root / "src" / "matlab2cpp_agentic_service").exists()
    assert (project_root / "examples" / "matlab_samples").exists()
    assert (project_root / "config" / "default_config.yaml").exists()


if __name__ == "__main__":
    pytest.main([__file__])


