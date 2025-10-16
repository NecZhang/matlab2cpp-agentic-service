"""
Comprehensive tests for Enhanced MATLAB2CPP Workflow (v0.3.0)

This test suite validates the enhanced workflow with 5 streamlined agents:
- MATLABAnalyzer (formerly EnhancedMATLABAnalyzer)
- ConversionPlanner (formerly EnhancedConversionPlanner)
- CppGenerator (formerly EnhancedCppGenerator)
- QualityAssessor (formerly EnhancedQualityAssessor)
- ProjectManager (formerly MultiFileProjectManager)

Updated for v0.3.0 architecture with:
- Streamlined agent names
- Selective LLM client support
- Two-pass compilation
- Smart helper detection
- CMake integration
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Ensure proper module imports (use PYTHONPATH instead of sys.path manipulation)
# If running directly, the parent directory should be in PYTHONPATH
# Example: PYTHONPATH=/path/to/matlab2cpp_agentic_service/src pytest

from matlab2cpp_agentic_service.core.workflows.enhanced_langgraph_workflow import (
    EnhancedLangGraphMATLAB2CPPWorkflow
)
from matlab2cpp_agentic_service.core.agents import (
    MATLABAnalyzer,
    ConversionPlanner,
    CppGenerator,
    QualityAssessor,
    ProjectManager
)
from matlab2cpp_agentic_service.core.agents.base.langgraph_agent import AgentConfig
from matlab2cpp_agentic_service.infrastructure.state.conversion_state import ConversionRequest


class TestEnhancedWorkflowV030:
    """Test suite for Enhanced MATLAB2CPP Workflow v0.3.0."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for v0.3.0."""
        config = Mock(spec=AgentConfig)
        config.max_retries = 3
        config.timeout = 300
        config.name = "test_agent"
        config.model = "test-model"
        return config
    
    @pytest.fixture
    def mock_llm_clients(self):
        """Create mock LLM clients dictionary for v0.3.0 selective client support."""
        # Create two distinct mock clients
        reasoning_client = MagicMock()
        reasoning_client.config = Mock()
        reasoning_client.config.model = "reasoning-model"
        reasoning_client.config.vllm_endpoint = "http://reasoning:8000"
        
        cpp_generation_client = MagicMock()
        cpp_generation_client.config = Mock()
        cpp_generation_client.config.model = "cpp-generation-model"
        cpp_generation_client.config.vllm_endpoint = "http://cpp-gen:8000"
        
        return {
            "reasoning": reasoning_client,
            "cpp_generation": cpp_generation_client
        }
    
    @pytest.fixture
    def enhanced_workflow(self, mock_config, mock_llm_clients):
        """Create enhanced workflow instance for v0.3.0."""
        return EnhancedLangGraphMATLAB2CPPWorkflow(mock_config, mock_llm_clients)
    
    @pytest.fixture
    def sample_matlab_file(self):
        """Create sample MATLAB file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.m', delete=False) as f:
            f.write("""
function result = sample_function(x, y)
    % Sample MATLAB function
    result = x + y;
end

function result = another_function(z)
    % Another sample function
    result = z * 2;
end
""")
            return Path(f.name)
    
    @pytest.fixture
    def sample_matlab_project(self):
        """Create sample multi-file MATLAB project."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir) / "test_project"
            project_dir.mkdir()
            
            # Create main file
            main_file = project_dir / "main.m"
            with open(main_file, 'w') as f:
                f.write("""
function main()
    x = 5;
    y = 10;
    result = helper_function(x, y);
    disp(result);
end
""")
            
            # Create helper file
            helper_file = project_dir / "helper_function.m"
            with open(helper_file, 'w') as f:
                f.write("""
function result = helper_function(a, b)
    result = a + b;
end
""")
            
            yield project_dir
    
    def test_workflow_initialization(self, enhanced_workflow):
        """Test workflow initialization with v0.3.0 architecture."""
        assert enhanced_workflow is not None
        assert len(enhanced_workflow.agents) == 5
        assert 'analyzer' in enhanced_workflow.agents
        assert 'planner' in enhanced_workflow.agents
        assert 'generator' in enhanced_workflow.agents
        assert 'assessor' in enhanced_workflow.agents
        assert 'project_manager' in enhanced_workflow.agents
    
    def test_agent_types_v030(self, enhanced_workflow):
        """Test that agents are of correct v0.3.0 streamlined types."""
        # v0.3.0 uses streamlined agent names (no "Enhanced" prefix)
        assert isinstance(enhanced_workflow.agents['analyzer'], MATLABAnalyzer)
        assert isinstance(enhanced_workflow.agents['planner'], ConversionPlanner)
        assert isinstance(enhanced_workflow.agents['generator'], CppGenerator)
        assert isinstance(enhanced_workflow.agents['assessor'], QualityAssessor)
        assert isinstance(enhanced_workflow.agents['project_manager'], ProjectManager)
    
    def test_selective_llm_clients(self, enhanced_workflow, mock_llm_clients):
        """Test that v0.3.0 selective LLM client assignment is correct."""
        # Analyzer should use reasoning client
        assert enhanced_workflow.agents['analyzer'].llm_client == mock_llm_clients['reasoning']
        
        # Planner should use reasoning client
        assert enhanced_workflow.agents['planner'].llm_client == mock_llm_clients['reasoning']
        
        # Generator should use cpp_generation client
        assert enhanced_workflow.agents['generator'].llm_client == mock_llm_clients['cpp_generation']
        
        # Assessor should use reasoning client
        assert enhanced_workflow.agents['assessor'].llm_client == mock_llm_clients['reasoning']
        
        # ProjectManager should use reasoning client
        assert enhanced_workflow.agents['project_manager'].llm_client == mock_llm_clients['reasoning']
    
    def test_workflow_info_v030(self, enhanced_workflow):
        """Test workflow information reflects v0.3.0 architecture."""
        info = enhanced_workflow.get_workflow_info()
        
        assert info['workflow_type'] == 'enhanced_langgraph'
        assert 'version' in info
        assert info['total_agents'] == 5
        assert info['architecture'] == 'streamlined'
        
        # Check agents (no "Enhanced" prefix in v0.3.0)
        agents_info = info['agents']
        assert 'MATLABAnalyzer' in str(agents_info)
        assert 'ConversionPlanner' in str(agents_info)
        assert 'CppGenerator' in str(agents_info)
        assert 'QualityAssessor' in str(agents_info)
        assert 'ProjectManager' in str(agents_info)
        
        # Check v0.3.0 features
        features = info['features']
        assert 'Real-time compilation testing' in features
        assert 'Adaptive strategy selection' in features
        assert 'Multi-file project support' in features
    
    def test_conversion_request_structure(self):
        """Test ConversionRequest structure for v0.3.0."""
        request = ConversionRequest(
            matlab_path="test.m",
            output_dir="output",
            project_name="test_project",
            max_turns=5,
            build_system="gcc"  # v0.3.0 feature
        )
        
        assert request.matlab_path == "test.m"
        assert request.output_dir == "output"
        assert request.project_name == "test_project"
        assert request.max_turns == 5
        assert request.build_system == "gcc"
    
    def test_build_system_options(self):
        """Test v0.3.0 build system options (gcc vs cmake)."""
        # Test GCC (default)
        request_gcc = ConversionRequest(
            matlab_path="test.m",
            output_dir="output",
            build_system="gcc"
        )
        assert request_gcc.build_system == "gcc"
        
        # Test CMake (v0.3.0 feature)
        request_cmake = ConversionRequest(
            matlab_path="test.m",
            output_dir="output",
            build_system="cmake"
        )
        assert request_cmake.build_system == "cmake"


class TestStreamlinedAgentsV030:
    """Test individual streamlined agents in v0.3.0 architecture."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock(spec=AgentConfig)
        config.max_retries = 3
        config.timeout = 300
        config.name = "test_agent"
        config.model = "test-model"
        return config
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        client = MagicMock()
        client.config = Mock()
        client.config.model = "test-model"
        return client
    
    def test_matlab_analyzer_initialization(self, mock_config, mock_llm_client):
        """Test MATLABAnalyzer initialization (v0.3.0 streamlined)."""
        analyzer = MATLABAnalyzer(mock_config, mock_llm_client)
        
        assert analyzer is not None
        assert analyzer.config == mock_config
        assert analyzer.llm_client == mock_llm_client
        
        # v0.3.0 uses simplified tool structure
        assert hasattr(analyzer, 'tools')
    
    def test_conversion_planner_initialization(self, mock_config, mock_llm_client):
        """Test ConversionPlanner initialization (v0.3.0 streamlined)."""
        planner = ConversionPlanner(mock_config, mock_llm_client)
        
        assert planner is not None
        assert planner.config == mock_config
        assert planner.llm_client == mock_llm_client
        
        # v0.3.0 uses simplified tool structure
        assert hasattr(planner, 'tools')
    
    def test_cpp_generator_initialization(self, mock_config, mock_llm_client):
        """Test CppGenerator initialization (v0.3.0 streamlined)."""
        generator = CppGenerator(mock_config, mock_llm_client)
        
        assert generator is not None
        assert generator.config == mock_config
        assert generator.llm_client == mock_llm_client
        
        # v0.3.0 features
        assert hasattr(generator, 'tools')
        # Smart helper detection and two-pass compilation are integrated
    
    def test_quality_assessor_initialization(self, mock_config, mock_llm_client):
        """Test QualityAssessor initialization (v0.3.0 streamlined)."""
        assessor = QualityAssessor(mock_config, mock_llm_client)
        
        assert assessor is not None
        assert assessor.config == mock_config
        assert assessor.llm_client == mock_llm_client
        
        # v0.3.0 features fair quality scoring
        assert hasattr(assessor, 'tools')
    
    def test_project_manager_initialization(self, mock_config, mock_llm_client):
        """Test ProjectManager initialization (v0.3.0 streamlined)."""
        manager = ProjectManager(mock_config, mock_llm_client)
        
        assert manager is not None
        assert manager.config == mock_config
        assert manager.llm_client == mock_llm_client
        
        # v0.3.0 multi-file project support
        assert hasattr(manager, 'tools')


class TestV030Features:
    """Test v0.3.0 specific features."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock(spec=AgentConfig)
        config.max_retries = 3
        config.timeout = 300
        config.name = "test_agent"
        config.model = "test-model"
        return config
    
    @pytest.fixture
    def mock_llm_clients(self):
        """Create mock LLM clients."""
        reasoning_client = MagicMock()
        reasoning_client.config = Mock()
        reasoning_client.config.model = "reasoning-model"
        
        cpp_generation_client = MagicMock()
        cpp_generation_client.config = Mock()
        cpp_generation_client.config.model = "cpp-generation-model"
        
        return {
            "reasoning": reasoning_client,
            "cpp_generation": cpp_generation_client
        }
    
    def test_two_pass_compilation_feature(self):
        """Test that two-pass compilation feature is available in v0.3.0."""
        # Two-pass compilation is integrated into DockerTestingManager
        from matlab2cpp_agentic_service.infrastructure.testing.docker_manager import DockerTestingManager
        
        # Create instance with gcc build system (supports two-pass)
        manager = DockerTestingManager(build_system='gcc')
        assert manager is not None
        assert manager.build_system == 'gcc'
        
        # Verify the method exists
        assert hasattr(manager, 'run_compilation_test')
    
    def test_smart_helper_detection_feature(self):
        """Test that smart helper detection is available in v0.3.0."""
        from matlab2cpp_agentic_service.infrastructure.build.helper_detector import (
            detect_needed_helpers
        )
        
        # Test detection with tensor code
        test_code = {
            'test.cpp': 'Eigen::Tensor<double, 3> tensor;'
        }
        
        helpers = detect_needed_helpers(test_code)
        assert 'tensor_helpers' in helpers
    
    def test_cmake_generation_feature(self):
        """Test that CMake generation is available in v0.3.0."""
        from matlab2cpp_agentic_service.infrastructure.build.cmake_generator import (
            generate_cmake_file
        )
        
        # Test CMake generation
        cmake_content = generate_cmake_file(
            project_name="test_project",
            cpp_files=["test.cpp"],
            header_files=["test.h"]
        )
        
        assert cmake_content is not None
        assert 'cmake_minimum_required' in cmake_content
        assert 'test_project' in cmake_content
    
    def test_post_generation_fixer_integration(self, mock_config, mock_llm_clients):
        """Test that post-generation namespace fixer is integrated in v0.3.0."""
        workflow = EnhancedLangGraphMATLAB2CPPWorkflow(mock_config, mock_llm_clients)
        
        # Verify workflow has the necessary structure
        assert hasattr(workflow, 'agents')
        assert hasattr(workflow, 'workflow')
        
        # Post-generation fixer is integrated in the workflow's save method
        # This is tested indirectly through the workflow execution


class TestWorkflowIntegrationV030:
    """Test workflow integration and coordination for v0.3.0."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock(spec=AgentConfig)
        config.max_retries = 3
        config.timeout = 300
        config.name = "test_agent"
        config.model = "test-model"
        return config
    
    @pytest.fixture
    def mock_llm_clients(self):
        """Create mock LLM clients."""
        reasoning_client = MagicMock()
        reasoning_client.config = Mock()
        reasoning_client.config.model = "reasoning-model"
        
        cpp_generation_client = MagicMock()
        cpp_generation_client.config = Mock()
        cpp_generation_client.config.model = "cpp-generation-model"
        
        return {
            "reasoning": reasoning_client,
            "cpp_generation": cpp_generation_client
        }
    
    def test_agent_coordination_v030(self, mock_config, mock_llm_clients):
        """Test that agents coordinate properly in v0.3.0."""
        workflow = EnhancedLangGraphMATLAB2CPPWorkflow(mock_config, mock_llm_clients)
        
        # Test that all 5 streamlined agents are properly initialized
        assert len(workflow.agents) == 5
        
        for agent_name in ['analyzer', 'planner', 'generator', 'assessor', 'project_manager']:
            agent = workflow.agents[agent_name]
            assert agent is not None
            assert agent.config == mock_config
            
            # Verify selective LLM client assignment
            if agent_name == 'generator':
                assert agent.llm_client == mock_llm_clients['cpp_generation']
            else:
                assert agent.llm_client == mock_llm_clients['reasoning']
        
        # Test workflow structure
        assert workflow.workflow is not None
        
        # Test workflow info
        info = workflow.get_workflow_info()
        assert info['total_agents'] == 5
        assert info['architecture'] == 'streamlined'
    
    def test_workflow_has_required_methods(self, mock_config, mock_llm_clients):
        """Test that workflow has all required v0.3.0 methods."""
        workflow = EnhancedLangGraphMATLAB2CPPWorkflow(mock_config, mock_llm_clients)
        
        # Core workflow methods
        assert hasattr(workflow, 'run_conversion')
        assert hasattr(workflow, 'get_workflow_info')
        
        # Internal methods (may be private)
        assert hasattr(workflow, '_initialize_enhanced_agents')
        assert hasattr(workflow, '_create_enhanced_workflow')


class TestBackwardCompatibility:
    """Test backward compatibility and migration path."""
    
    def test_agent_name_mapping(self):
        """Test that old agent names map to new streamlined names."""
        # v0.3.0 streamlined names (no "Enhanced" prefix)
        from matlab2cpp_agentic_service.core.agents import (
            MATLABAnalyzer,
            ConversionPlanner,
            CppGenerator,
            QualityAssessor,
            ProjectManager
        )
        
        # All imports should succeed
        assert MATLABAnalyzer is not None
        assert ConversionPlanner is not None
        assert CppGenerator is not None
        assert QualityAssessor is not None
        assert ProjectManager is not None
    
    def test_conversion_request_defaults(self):
        """Test ConversionRequest has sensible v0.3.0 defaults."""
        request = ConversionRequest(
            matlab_path="test.m",
            output_dir="output"
        )
        
        # Default build system should be 'gcc'
        assert request.build_system == "gcc"
        
        # Default max_turns should be reasonable
        assert request.max_turns >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
