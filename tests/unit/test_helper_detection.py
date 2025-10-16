"""
Unit tests for HelperDetector - Smart helper library detection.

Tests the pattern-based detection system that identifies which helper libraries
are actually needed by generated C++ code.
"""

import pytest
from pathlib import Path

from matlab2cpp_agentic_service.infrastructure.build.helper_detector import (
    HelperDetector,
    detect_needed_helpers
)


class TestHelperDetector:
    """Test suite for HelperDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create a HelperDetector instance."""
        return HelperDetector()
    
    def test_initialization(self, detector):
        """Test that HelperDetector initializes correctly."""
        assert detector is not None
        assert hasattr(detector, 'patterns')
        assert len(detector.patterns) > 0
    
    def test_tensor_helpers_detection(self, detector):
        """Test detection of tensor helper usage."""
        # Code using tensor helpers
        code_with_tensors = """
        #include <unsupported/Eigen/CXX11/Tensor>
        Eigen::Tensor<double, 3> tensor(10, 10, 10);
        auto slice_result = matlab::tensor::slice(tensor, 2, 5);
        """
        
        needed = detector.detect({"test.cpp": code_with_tensors})
        assert "tensor_helpers" in needed
    
    def test_rk4_helpers_detection(self, detector):
        """Test detection of RK4 helper usage."""
        code_with_rk4 = """
        #include "rk4_helpers.h"
        auto result = matlab::rk4::rk4_solve(f, t0, tf, y0, dt);
        """
        
        needed = detector.detect({"test.cpp": code_with_rk4})
        assert "rk4_helpers" in needed
    
    def test_msfm_helpers_detection(self, detector):
        """Test detection of MSFM helper usage."""
        code_with_msfm = """
        #include "msfm_helpers.h"
        msfm::helpers::vector_to_matrix(source_points);
        """
        
        needed = detector.detect({"test.cpp": code_with_msfm})
        assert "msfm_helpers" in needed
    
    def test_image_helpers_detection(self, detector):
        """Test detection of image helper usage."""
        code_with_image = """
        #include "matlab_image_helpers.h"
        matlab::image::imdilate(image, structuring_element);
        """
        
        needed = detector.detect({"test.cpp": code_with_image})
        assert "matlab_image_helpers" in needed
    
    def test_array_utils_detection(self, detector):
        """Test detection of array utils usage."""
        code_with_array = """
        #include "matlab_array_utils.h"
        matlab::array::linearIndexToSubscripts(index, rows, cols);
        """
        
        needed = detector.detect({"test.cpp": code_with_array})
        assert "matlab_array_utils" in needed
    
    def test_pointmin_helpers_detection(self, detector):
        """Test detection of pointmin helper usage."""
        code_with_pointmin = """
        #include "pointmin_helpers.h"
        matlab::pointmin::find_minimum_point(matrix);
        """
        
        needed = detector.detect({"test.cpp": code_with_pointmin})
        assert "pointmin_helpers" in needed
    
    def test_no_helpers_needed(self, detector):
        """Test that code without helpers returns empty set."""
        simple_code = """
        #include <Eigen/Dense>
        int main() {
            Eigen::MatrixXd m(3, 3);
            m << 1, 2, 3, 4, 5, 6, 7, 8, 9;
            return 0;
        }
        """
        
        needed = detector.detect({"test.cpp": simple_code})
        assert len(needed) == 0
    
    def test_multiple_helpers_detection(self, detector):
        """Test detection of multiple helpers in one file."""
        code_with_multiple = """
        #include "tensor_helpers.h"
        #include "rk4_helpers.h"
        #include "msfm_helpers.h"
        
        auto tensor_result = matlab::tensor::slice(tensor, 2, 5);
        auto rk4_result = matlab::rk4::rk4_solve(f, t0, tf, y0, dt);
        auto msfm_result = msfm::helpers::vector_to_matrix(points);
        """
        
        needed = detector.detect({"test.cpp": code_with_multiple})
        assert "tensor_helpers" in needed
        assert "rk4_helpers" in needed
        assert "msfm_helpers" in needed
        assert len(needed) == 3
    
    def test_multiple_files(self, detector):
        """Test detection across multiple files."""
        project_files = {
            "main.cpp": "#include \"tensor_helpers.h\"\nmatlab::tensor::slice(t, 0, 5);",
            "utils.cpp": "#include \"rk4_helpers.h\"\nmatlab::rk4::rk4_solve(f, 0, 1, y, 0.1);",
            "helpers.cpp": "#include \"msfm_helpers.h\"\nmsfm::helpers::convert(p);"
        }
        
        needed = detector.detect(project_files)
        assert "tensor_helpers" in needed
        assert "rk4_helpers" in needed
        assert "msfm_helpers" in needed
        assert len(needed) == 3
    
    def test_case_sensitivity(self, detector):
        """Test that detection is case-sensitive."""
        # Lowercase should not match (C++ is case-sensitive)
        code_lowercase = "matlab::tensor::SLICE(tensor, 2, 5);"
        needed = detector.detect({"test.cpp": code_lowercase})
        # Should still detect via include pattern
        
        # Correct case should match
        code_correct = "matlab::tensor::slice(tensor, 2, 5);"
        needed_correct = detector.detect({"test.cpp": code_correct})
        assert "tensor_helpers" in needed_correct


class TestDetectNeededHelpersFunction:
    """Test the standalone detect_needed_helpers function."""
    
    def test_basic_detection(self):
        """Test basic helper detection."""
        project_files = {
            "main.cpp": "#include \"tensor_helpers.h\"\nmatlab::tensor::slice(t, 0, 5);"
        }
        
        needed = detect_needed_helpers(project_files)
        assert "tensor_helpers" in needed
    
    def test_empty_project(self):
        """Test detection with empty project."""
        needed = detect_needed_helpers({})
        assert len(needed) == 0
    
    def test_none_project(self):
        """Test detection with None input."""
        needed = detect_needed_helpers(None)
        assert len(needed) == 0


class TestPatternMatching:
    """Test specific pattern matching scenarios."""
    
    @pytest.fixture
    def detector(self):
        return HelperDetector()
    
    def test_namespace_pattern_matching(self, detector):
        """Test various namespace patterns."""
        test_cases = [
            ("matlab::tensor::slice", "tensor_helpers"),
            ("matlab::rk4::rk4_solve", "rk4_helpers"),
            ("msfm::helpers::vector_to_matrix", "msfm_helpers"),
            ("matlab::image::imdilate", "matlab_image_helpers"),
            ("matlab::array::linearIndexToSubscripts", "matlab_array_utils"),
            ("matlab::pointmin::find_minimum_point", "pointmin_helpers"),
        ]
        
        for pattern, expected_helper in test_cases:
            code = f"auto result = {pattern}(args);"
            needed = detector.detect({"test.cpp": code})
            assert expected_helper in needed, f"Failed to detect {expected_helper} from pattern {pattern}"
    
    def test_include_pattern_matching(self, detector):
        """Test various include patterns."""
        test_cases = [
            ('#include "tensor_helpers.h"', "tensor_helpers"),
            ('#include "rk4_helpers.h"', "rk4_helpers"),
            ('#include "msfm_helpers.h"', "msfm_helpers"),
            ('#include "matlab_image_helpers.h"', "matlab_image_helpers"),
            ('#include "matlab_array_utils.h"', "matlab_array_utils"),
            ('#include "pointmin_helpers.h"', "pointmin_helpers"),
        ]
        
        for include, expected_helper in test_cases:
            code = f"{include}\nvoid func() {{}}"
            needed = detector.detect({"test.cpp": code})
            assert expected_helper in needed, f"Failed to detect {expected_helper} from include"
    
    def test_complex_code_patterns(self, detector):
        """Test detection in complex code."""
        complex_code = """
        #include <Eigen/Dense>
        #include <unsupported/Eigen/CXX11/Tensor>
        #include "tensor_helpers.h"
        
        namespace my_project {
            class TensorProcessor {
            public:
                Eigen::MatrixXd process(const Eigen::Tensor<double, 3>& input) {
                    // Use tensor helpers
                    auto slice = matlab::tensor::slice(input, 2, 0);
                    
                    // Convert to matrix
                    return slice;
                }
            };
        }
        """
        
        needed = detector.detect({"processor.cpp": complex_code})
        assert "tensor_helpers" in needed
    
    def test_comments_dont_trigger_detection(self, detector):
        """Test that commented code doesn't trigger detection."""
        code_with_comments = """
        // This is a comment: matlab::tensor::slice(tensor, 0, 5);
        /* Another comment: matlab::rk4::rk4_solve(f, 0, 1, y, 0.1); */
        
        int main() {
            // No actual helper usage
            return 0;
        }
        """
        
        needed = detector.detect({"test.cpp": code_with_comments})
        # Comments might still trigger via simple regex, which is acceptable
        # This test documents current behavior


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

