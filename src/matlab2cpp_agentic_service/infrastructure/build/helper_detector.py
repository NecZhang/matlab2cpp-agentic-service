"""
Smart Helper Detection for MATLAB to C++ Conversion.

This module analyzes generated C++ code to detect which helper libraries
are actually used, enabling conditional inclusion to reduce file bloat.
"""

import re
from typing import Dict, Set
import logging


class HelperDetector:
    """Detects which helper libraries are actually used in generated code."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def detect_needed_helpers(self, generated_files: Dict[str, str]) -> Set[str]:
        """
        Analyze generated code to detect which helpers are actually needed.
        
        Args:
            generated_files: Dict mapping filename to content
        
        Returns:
            Set of helper names that are actually used
        """
        needed = set()
        
        # Combine all generated code (excluding helper files themselves)
        project_code = self._get_project_code(generated_files)
        
        # Check each helper
        if self._needs_tensor_helpers(project_code):
            needed.add('tensor_helpers')
            self.logger.info("  ✅ tensor_helpers: NEEDED")
        else:
            self.logger.info("  ⏭️  tensor_helpers: NOT NEEDED")
        
        if self._needs_rk4_helpers(project_code):
            needed.add('rk4_helpers')
            self.logger.info("  ✅ rk4_helpers: NEEDED")
        else:
            self.logger.info("  ⏭️  rk4_helpers: NOT NEEDED")
        
        if self._needs_image_helpers(project_code):
            needed.add('matlab_image_helpers')
            self.logger.info("  ✅ matlab_image_helpers: NEEDED")
        else:
            self.logger.info("  ⏭️  matlab_image_helpers: NOT NEEDED")
        
        if self._needs_msfm_helpers(project_code):
            needed.add('msfm_helpers')
            self.logger.info("  ✅ msfm_helpers: NEEDED")
        else:
            self.logger.info("  ⏭️  msfm_helpers: NOT NEEDED")
        
        if self._needs_array_utils(project_code):
            needed.add('matlab_array_utils')
            self.logger.info("  ✅ matlab_array_utils: NEEDED")
        else:
            self.logger.info("  ⏭️  matlab_array_utils: NOT NEEDED")
        
        self.logger.info(f"📊 Smart Detection Result: {len(needed)}/5 helpers needed")
        return needed
    
    def _get_project_code(self, generated_files: Dict[str, str]) -> str:
        """Extract project code (excluding helper libraries themselves)."""
        project_files = {}
        helper_patterns = [
            'tensor_helpers',
            'rk4_helpers',
            'matlab_image_helpers',
            'msfm_helpers',
            'matlab_array_utils'
        ]
        
        for filename, content in generated_files.items():
            # Skip helper library files themselves
            if not any(helper in filename for helper in helper_patterns):
                project_files[filename] = content
        
        return '\n'.join(project_files.values())
    
    def _needs_tensor_helpers(self, code: str) -> bool:
        """Check if code uses tensor helper functions."""
        patterns = [
            r'matlab::tensor::',           # Direct helper calls
            r'#include\s+"tensor_helpers\.h"',  # Explicit include
            r'#include\s+<tensor_helpers\.h>',  # Explicit include (angle brackets)
        ]
        
        for pattern in patterns:
            if re.search(pattern, code):
                return True
        
        return False
    
    def _needs_rk4_helpers(self, code: str) -> bool:
        """Check if code uses RK4 integration helpers."""
        patterns = [
            r'matlab::rk4::',              # Direct helper calls
            r'#include\s+"rk4_helpers\.h"',     # Explicit include
            r'#include\s+<rk4_helpers\.h>',     # Explicit include (angle brackets)
        ]
        
        for pattern in patterns:
            if re.search(pattern, code):
                return True
        
        return False
    
    def _needs_image_helpers(self, code: str) -> bool:
        """Check if code uses image processing helpers."""
        patterns = [
            r'matlab::image::',            # Direct helper calls
            r'#include\s+"matlab_image_helpers\.h"',  # Explicit include
            r'#include\s+<matlab_image_helpers\.h>',  # Explicit include (angle brackets)
        ]
        
        for pattern in patterns:
            if re.search(pattern, code):
                return True
        
        return False
    
    def _needs_msfm_helpers(self, code: str) -> bool:
        """Check if code uses MSFM (Fast Marching) helpers."""
        patterns = [
            r'msfm::helpers::',            # Direct helper calls
            r'#include\s+"msfm_helpers\.h"',    # Explicit include
            r'#include\s+<msfm_helpers\.h>',    # Explicit include (angle brackets)
        ]
        
        for pattern in patterns:
            if re.search(pattern, code):
                return True
        
        return False
    
    def _needs_array_utils(self, code: str) -> bool:
        """Check if code uses array utility helpers."""
        patterns = [
            r'matlab::array::',            # Direct helper calls
            r'#include\s+"matlab_array_utils\.h"',  # Explicit include
            r'#include\s+<matlab_array_utils\.h>',  # Explicit include (angle brackets)
        ]
        
        for pattern in patterns:
            if re.search(pattern, code):
                return True
        
        return False


def detect_needed_helpers(generated_files: Dict[str, str]) -> Set[str]:
    """
    Convenience function to detect needed helper libraries.
    
    Args:
        generated_files: Dict of filename -> content
    
    Returns:
        Set of helper names that are actually used
    
    Example:
        >>> needed = detect_needed_helpers({
        ...     'arma_filter.cpp': '...',
        ...     'main.cpp': '#include "tensor_helpers.h"\\nmatlab::tensor::slice(...)'
        ... })
        >>> print(needed)
        {'tensor_helpers'}
    """
    detector = HelperDetector()
    return detector.detect_needed_helpers(generated_files)

