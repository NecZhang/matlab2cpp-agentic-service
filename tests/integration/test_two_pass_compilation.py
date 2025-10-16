"""
Integration tests for Two-Pass Compilation System.

Tests the Smart Detection (Pass 1) + Fallback (Pass 2) compilation strategy
that was introduced in v0.3.0.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from matlab2cpp_agentic_service.infrastructure.testing.docker_manager import DockerTestingManager
from matlab2cpp_agentic_service.infrastructure.testing.types import CompilationResult
from matlab2cpp_agentic_service.infrastructure.build.helper_detector import detect_needed_helpers


class TestTwoPassCompilation:
    """Test suite for two-pass compilation system."""
    
    @pytest.fixture
    def docker_manager(self):
        """Create a DockerTestingManager instance."""
        return DockerTestingManager(build_system='gcc')
    
    @pytest.fixture
    def simple_project_files(self):
        """Create a simple project that doesn't need helpers."""
        return {
            "main.cpp": """
                #include <Eigen/Dense>
                #include <iostream>
                
                int main() {
                    Eigen::MatrixXd m(2, 2);
                    m << 1, 2, 3, 4;
                    std::cout << m << std::endl;
                    return 0;
                }
            """,
            "main.h": """
                #pragma once
                #include <Eigen/Dense>
            """
        }
    
    @pytest.fixture
    def project_with_tensor_helpers(self):
        """Create a project that needs tensor helpers."""
        return {
            "main.cpp": """
                #include <unsupported/Eigen/CXX11/Tensor>
                #include "tensor_helpers.h"
                
                int main() {
                    Eigen::Tensor<double, 3> tensor(10, 10, 10);
                    auto slice = matlab::tensor::slice(tensor, 2, 5);
                    return 0;
                }
            """,
            "main.h": """
                #pragma once
                #include <unsupported/Eigen/CXX11/Tensor>
            """
        }
    
    def test_helper_detection_identifies_needed_helpers(self, project_with_tensor_helpers):
        """Test that helper detection correctly identifies needed helpers."""
        needed = detect_needed_helpers(project_with_tensor_helpers)
        assert "tensor_helpers" in needed
    
    def test_helper_detection_finds_no_helpers(self, simple_project_files):
        """Test that helper detection finds no helpers for simple project."""
        needed = detect_needed_helpers(simple_project_files)
        assert len(needed) == 0 or "tensor_helpers" not in needed
    
    def test_two_pass_strategy_exists(self, docker_manager):
        """Test that DockerTestingManager has two-pass compilation methods."""
        assert hasattr(docker_manager, 'run_compilation_test')
        assert hasattr(docker_manager, '_add_helper_libraries')
        assert hasattr(docker_manager, '_add_all_helper_libraries')
    
    @patch.object(DockerTestingManager, '_run_single_compilation_pass')
    def test_pass1_success_skips_pass2(self, mock_compile, docker_manager, simple_project_files):
        """Test that successful Pass 1 doesn't trigger Pass 2."""
        # Mock successful compilation
        mock_compile.return_value = CompilationResult(
            success=True,
            exit_code=0,
            stdout="Compilation successful",
            stderr="",
            errors=[],
            warnings=[]
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = docker_manager.run_compilation_test(
                simple_project_files,
                tmpdir,
                "test_project"
            )
            
            # Should only call compilation once (Pass 1 succeeded)
            assert mock_compile.call_count == 1
            assert result.success is True
    
    @patch.object(DockerTestingManager, '_run_single_compilation_pass')
    def test_pass1_failure_triggers_pass2(self, mock_compile, docker_manager, simple_project_files):
        """Test that failed Pass 1 triggers Pass 2 fallback."""
        # First call (Pass 1) fails, second call (Pass 2) succeeds
        mock_compile.side_effect = [
            CompilationResult(
                success=False,
                exit_code=1,
                stdout="",
                stderr="Compilation failed: missing helper",
                errors=["undefined reference to 'helper_function'"],
                warnings=[]
            ),
            CompilationResult(
                success=True,
                exit_code=0,
                stdout="Compilation successful",
                stderr="",
                errors=[],
                warnings=[]
            )
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = docker_manager.run_compilation_test(
                simple_project_files,
                tmpdir,
                "test_project"
            )
            
            # Should call compilation twice (Pass 1 failed, Pass 2 succeeded)
            assert mock_compile.call_count == 2
            assert result.success is True
    
    @patch.object(DockerTestingManager, '_run_single_compilation_pass')
    def test_both_passes_fail(self, mock_compile, docker_manager, simple_project_files):
        """Test behavior when both passes fail."""
        # Both passes fail
        mock_compile.return_value = CompilationResult(
            success=False,
            exit_code=1,
            stdout="",
            stderr="Compilation failed: syntax error",
            errors=["syntax error"],
            warnings=[]
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = docker_manager.run_compilation_test(
                simple_project_files,
                tmpdir,
                "test_project"
            )
            
            # Should try both passes
            assert mock_compile.call_count == 2
            assert result.success is False
    
    def test_add_helper_libraries_smart_detection(self, docker_manager, project_with_tensor_helpers):
        """Test that _add_helper_libraries uses smart detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Call the smart detection method
            files_with_helpers = docker_manager._add_helper_libraries(project_with_tensor_helpers, tmpdir)
            
            # Should include original files
            assert "main.cpp" in files_with_helpers
            assert "main.h" in files_with_helpers
            
            # Should include tensor helpers (detected)
            assert any("tensor_helpers" in f for f in files_with_helpers.keys())
    
    def test_add_all_helper_libraries_includes_everything(self, docker_manager, simple_project_files):
        """Test that _add_all_helper_libraries includes all helpers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Call the fallback method
            files_with_all_helpers = docker_manager._add_all_helper_libraries(simple_project_files, tmpdir)
            
            # Should include original files
            assert "main.cpp" in files_with_all_helpers
            assert "main.h" in files_with_all_helpers
            
            # Should include ALL helpers
            expected_helpers = [
                "tensor_helpers",
                "rk4_helpers",
                "msfm_helpers",
                "matlab_image_helpers",
                "matlab_array_utils",
                "pointmin_helpers"
            ]
            
            for helper in expected_helpers:
                assert any(helper in f for f in files_with_all_helpers.keys()), \
                    f"Expected {helper} to be included in fallback"


class TestTwoPassCompilationMetrics:
    """Test metrics and logging for two-pass compilation."""
    
    @pytest.fixture
    def docker_manager(self):
        return DockerTestingManager(build_system='gcc')
    
    @patch.object(DockerTestingManager, '_run_single_compilation_pass')
    def test_pass1_metrics(self, mock_compile, docker_manager):
        """Test that Pass 1 metrics are recorded."""
        mock_compile.return_value = CompilationResult(
            success=True,
            exit_code=0,
            stdout="Pass 1 succeeded with smart detection",
            stderr="",
            errors=[],
            warnings=[]
        )
        
        project_files = {"test.cpp": "int main() { return 0; }"}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = docker_manager.run_compilation_test(project_files, tmpdir, "test")
            
            # Verify result contains Pass 1 information
            assert result.success is True
            assert "smart detection" in result.stdout.lower() or mock_compile.call_count == 1
    
    @patch.object(DockerTestingManager, '_run_single_compilation_pass')
    def test_pass2_fallback_metrics(self, mock_compile, docker_manager):
        """Test that Pass 2 fallback metrics are recorded."""
        mock_compile.side_effect = [
            CompilationResult(success=False, exit_code=1, stdout="", stderr="Pass 1 failed", errors=["error"], warnings=[]),
            CompilationResult(success=True, exit_code=0, stdout="Pass 2 succeeded with all helpers", stderr="", errors=[], warnings=[])
        ]
        
        project_files = {"test.cpp": "int main() { return 0; }"}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = docker_manager.run_compilation_test(project_files, tmpdir, "test")
            
            # Verify result shows fallback was used
            assert result.success is True
            assert mock_compile.call_count == 2


class TestTwoPassEdgeCases:
    """Test edge cases in two-pass compilation."""
    
    @pytest.fixture
    def docker_manager(self):
        return DockerTestingManager(build_system='gcc')
    
    def test_empty_project(self, docker_manager):
        """Test two-pass compilation with empty project."""
        empty_project = {}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Should handle gracefully
            result = docker_manager.run_compilation_test(empty_project, tmpdir, "empty")
            # Result may fail, but should not crash
            assert result is not None
    
    def test_project_with_only_headers(self, docker_manager):
        """Test project with only header files."""
        header_only = {
            "utils.h": "#pragma once\nint add(int a, int b) { return a + b; }"
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = docker_manager.run_compilation_test(header_only, tmpdir, "header_only")
            # Should handle gracefully
            assert result is not None
    
    @patch.object(DockerTestingManager, '_run_single_compilation_pass')
    def test_pass1_with_warnings(self, mock_compile, docker_manager):
        """Test Pass 1 succeeds with warnings."""
        mock_compile.return_value = CompilationResult(
            success=True,
            exit_code=0,
            stdout="Compilation successful",
            stderr="warning: unused variable",
            errors=[],
            warnings=["unused variable 'x'"]
        )
        
        project = {"test.cpp": "int main() { int x; return 0; }"}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = docker_manager.run_compilation_test(project, tmpdir, "test")
            
            # Should succeed and not trigger Pass 2
            assert result.success is True
            assert mock_compile.call_count == 1
            assert len(result.warnings) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

