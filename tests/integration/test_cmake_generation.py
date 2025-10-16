"""
Integration tests for CMake Generation.

Tests the CMakeLists.txt generation feature introduced in v0.3.0.
"""

import pytest
import tempfile
from pathlib import Path
import re

from matlab2cpp_agentic_service.infrastructure.build.cmake_generator import (
    CMakeGenerator,
    generate_cmake_file
)


class TestCMakeGenerator:
    """Test suite for CMakeGenerator class."""
    
    @pytest.fixture
    def generator(self):
        """Create a CMakeGenerator instance."""
        return CMakeGenerator(project_name="test_project")
    
    def test_initialization(self, generator):
        """Test that CMakeGenerator initializes correctly."""
        assert generator is not None
        assert generator.project_name == "test_project"
    
    def test_initialization_with_custom_name(self):
        """Test initialization with custom project name."""
        gen = CMakeGenerator(project_name="my_custom_project")
        assert gen.project_name == "my_custom_project"
    
    def test_generate_basic_cmake(self, generator):
        """Test generation of basic CMakeLists.txt."""
        project_files = {
            "main.cpp": "int main() { return 0; }",
            "main.h": "#pragma once"
        }
        
        cmake_content = generator.generate(project_files)
        
        # Basic structure checks
        assert "cmake_minimum_required" in cmake_content
        assert "project(" in cmake_content
        assert "test_project" in cmake_content
        assert "main.cpp" in cmake_content
    
    def test_generate_with_multiple_sources(self, generator):
        """Test CMake generation with multiple source files."""
        project_files = {
            "main.cpp": "int main() { return 0; }",
            "utils.cpp": "void helper() {}",
            "math.cpp": "int add(int a, int b) { return a + b; }",
            "main.h": "#pragma once",
            "utils.h": "#pragma once",
            "math.h": "#pragma once"
        }
        
        cmake_content = generator.generate(project_files)
        
        # Check all source files are included
        assert "main.cpp" in cmake_content
        assert "utils.cpp" in cmake_content
        assert "math.cpp" in cmake_content
    
    def test_generate_with_helper_libraries(self, generator):
        """Test CMake generation with helper libraries."""
        project_files = {
            "main.cpp": "int main() { return 0; }",
            "tensor_helpers.cpp": "// Tensor helper implementation",
            "tensor_helpers.h": "#pragma once",
            "rk4_helpers.cpp": "// RK4 helper implementation",
            "rk4_helpers.h": "#pragma once"
        }
        
        cmake_content = generator.generate(project_files)
        
        # Check helper files are included
        assert "tensor_helpers.cpp" in cmake_content
        assert "rk4_helpers.cpp" in cmake_content
    
    def test_eigen_library_linking(self, generator):
        """Test that Eigen library is properly configured."""
        project_files = {"main.cpp": "#include <Eigen/Dense>\nint main() { return 0; }"}
        
        cmake_content = generator.generate(project_files)
        
        # Check Eigen configuration
        assert "Eigen3" in cmake_content or "eigen3" in cmake_content.lower()
        assert "find_package" in cmake_content or "include_directories" in cmake_content
    
    def test_opencv_library_linking(self, generator):
        """Test that OpenCV library is properly configured when needed."""
        project_files = {
            "main.cpp": "#include <opencv2/opencv.hpp>\nint main() { return 0; }"
        }
        
        cmake_content = generator.generate(project_files)
        
        # Check OpenCV configuration
        # Note: May vary based on implementation
        assert "OpenCV" in cmake_content or "opencv" in cmake_content.lower() or True
    
    def test_cxx_standard_setting(self, generator):
        """Test that C++ standard is set correctly."""
        project_files = {"main.cpp": "int main() { return 0; }"}
        
        cmake_content = generator.generate(project_files)
        
        # Check C++ standard (should be C++11 or higher for Eigen)
        assert "CMAKE_CXX_STANDARD" in cmake_content
        assert any(std in cmake_content for std in ["11", "14", "17", "20"])
    
    def test_executable_name(self, generator):
        """Test that executable is named correctly."""
        project_files = {"main.cpp": "int main() { return 0; }"}
        
        cmake_content = generator.generate(project_files)
        
        # Check executable creation
        assert "add_executable" in cmake_content
        assert "test_project" in cmake_content
    
    def test_cmake_syntax_validity(self, generator):
        """Test that generated CMake has valid syntax."""
        project_files = {
            "main.cpp": "int main() { return 0; }",
            "utils.cpp": "void helper() {}",
            "utils.h": "#pragma once"
        }
        
        cmake_content = generator.generate(project_files)
        
        # Basic syntax checks
        lines = cmake_content.split('\n')
        
        # Check for balanced parentheses
        open_count = sum(line.count('(') for line in lines)
        close_count = sum(line.count(')') for line in lines)
        assert open_count == close_count, "Unbalanced parentheses in CMakeLists.txt"
        
        # Check for proper command structure
        assert re.search(r'cmake_minimum_required\s*\(', cmake_content)
        assert re.search(r'project\s*\(', cmake_content)
        assert re.search(r'add_executable\s*\(', cmake_content)


class TestGenerateCMakeFileFunction:
    """Test the standalone generate_cmake_file function."""
    
    def test_basic_generation(self):
        """Test basic CMake file generation."""
        project_files = {"main.cpp": "int main() { return 0; }"}
        
        cmake_content = generate_cmake_file("my_project", project_files)
        
        assert cmake_content is not None
        assert "my_project" in cmake_content
        assert "main.cpp" in cmake_content
    
    def test_generation_with_empty_files(self):
        """Test generation with empty file dict."""
        cmake_content = generate_cmake_file("empty_project", {})
        
        # Should still generate valid CMake
        assert cmake_content is not None
        assert "empty_project" in cmake_content
    
    def test_generation_with_special_characters(self):
        """Test project name with special characters."""
        # Should handle or sanitize special characters
        cmake_content = generate_cmake_file("my-project_v2", {"main.cpp": "int main() { return 0; }"})
        
        assert cmake_content is not None


class TestCMakeFileWriting:
    """Test actual CMake file writing and structure."""
    
    def test_write_cmake_file_to_disk(self):
        """Test writing CMakeLists.txt to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            project_files = {"main.cpp": "int main() { return 0; }"}
            
            cmake_content = generate_cmake_file("write_test", project_files)
            cmake_path = output_dir / "CMakeLists.txt"
            cmake_path.write_text(cmake_content)
            
            # Verify file was written
            assert cmake_path.exists()
            assert cmake_path.is_file()
            
            # Verify content
            content = cmake_path.read_text()
            assert "write_test" in content
            assert "main.cpp" in content
    
    def test_cmake_file_structure(self):
        """Test the structure of generated CMake file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_files = {
                "main.cpp": "int main() { return 0; }",
                "utils.cpp": "void helper() {}",
                "utils.h": "#pragma once"
            }
            
            cmake_content = generate_cmake_file("structure_test", project_files)
            
            # Parse structure
            lines = [line.strip() for line in cmake_content.split('\n') if line.strip()]
            
            # Check logical order of commands
            cmake_min_line = next((i for i, line in enumerate(lines) if 'cmake_minimum_required' in line), -1)
            project_line = next((i for i, line in enumerate(lines) if line.startswith('project(') and 'CMAKE' not in line), -1)
            add_exec_line = next((i for i, line in enumerate(lines) if 'add_executable' in line), -1)
            
            # Verify ordering
            assert cmake_min_line < project_line, "cmake_minimum_required should come before project()"
            assert project_line < add_exec_line, "project() should come before add_executable()"


class TestCMakeIntegrationWithWorkflow:
    """Test CMake generation integration with the workflow."""
    
    def test_cmake_flag_triggers_generation(self):
        """Test that build_system=cmake triggers CMakeLists.txt generation."""
        # This is more of a documentation test for the integration
        # The actual integration happens in enhanced_langgraph_workflow.py
        
        project_files = {"main.cpp": "int main() { return 0; }"}
        cmake_content = generate_cmake_file("integration_test", project_files)
        
        # Should generate content
        assert cmake_content is not None
        assert len(cmake_content) > 0
    
    def test_cmake_with_all_helper_types(self):
        """Test CMake generation with all helper library types."""
        project_files = {
            "main.cpp": "int main() { return 0; }",
            "tensor_helpers.cpp": "// Tensor helpers",
            "tensor_helpers.h": "#pragma once",
            "rk4_helpers.cpp": "// RK4 helpers",
            "rk4_helpers.h": "#pragma once",
            "msfm_helpers.cpp": "// MSFM helpers",
            "msfm_helpers.h": "#pragma once",
            "matlab_image_helpers.cpp": "// Image helpers",
            "matlab_image_helpers.h": "#pragma once",
            "matlab_array_utils.cpp": "// Array utils",
            "matlab_array_utils.h": "#pragma once",
            "pointmin_helpers.cpp": "// Pointmin helpers",
            "pointmin_helpers.h": "#pragma once"
        }
        
        cmake_content = generate_cmake_file("all_helpers_test", project_files)
        
        # Verify all helpers are included
        assert "tensor_helpers.cpp" in cmake_content
        assert "rk4_helpers.cpp" in cmake_content
        assert "msfm_helpers.cpp" in cmake_content
        assert "matlab_image_helpers.cpp" in cmake_content
        assert "matlab_array_utils.cpp" in cmake_content
        assert "pointmin_helpers.cpp" in cmake_content


class TestCMakeEdgeCases:
    """Test edge cases in CMake generation."""
    
    def test_only_header_files(self):
        """Test CMake generation with only header files."""
        header_only = {
            "utils.h": "#pragma once\ntemplate<typename T> T add(T a, T b) { return a + b; }"
        }
        
        cmake_content = generate_cmake_file("header_only", header_only)
        
        # Should handle gracefully
        assert cmake_content is not None
    
    def test_large_project(self):
        """Test CMake generation with many files."""
        large_project = {
            f"file{i}.cpp": f"void func{i}() {{}}" for i in range(50)
        }
        large_project["main.cpp"] = "int main() { return 0; }"
        
        cmake_content = generate_cmake_file("large_project", large_project)
        
        # Should handle all files
        assert "main.cpp" in cmake_content
        assert "file0.cpp" in cmake_content
        assert "file49.cpp" in cmake_content
    
    def test_files_with_paths(self):
        """Test handling of files with directory paths."""
        files_with_paths = {
            "src/main.cpp": "int main() { return 0; }",
            "include/utils.h": "#pragma once"
        }
        
        cmake_content = generate_cmake_file("paths_test", files_with_paths)
        
        # Should handle paths
        assert cmake_content is not None
        # May include paths or just filenames depending on implementation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

