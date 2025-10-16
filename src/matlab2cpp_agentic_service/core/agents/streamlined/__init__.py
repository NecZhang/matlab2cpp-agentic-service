"""
Streamlined Agents

The 5 streamlined agents that form the core of the MATLAB2C++ conversion system:
- MATLABAnalyzer: Advanced MATLAB code analysis with dependency detection
- ConversionPlanner: Multi-file project structure planning and coordination
- CppGenerator: C++ code generation with integrated compilation testing
- QualityAssessor: Comprehensive quality assessment with real compilation results
- ProjectManager: Complex multi-file project coordination and management
"""

from .matlab_analyzer import MATLABAnalyzer
from .conversion_planner import ConversionPlanner
from .cpp_generator import CppGenerator
from .quality_assessor import QualityAssessor
from .project_manager import ProjectManager

__all__ = [
    "MATLABAnalyzer",
    "ConversionPlanner", 
    "CppGenerator",
    "QualityAssessor",
    "ProjectManager"
]







