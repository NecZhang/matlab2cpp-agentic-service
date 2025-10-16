"""
MATLAB2C++ Agentic Service

A comprehensive LangGraph-based agentic service for converting MATLAB projects to C++
with multi-file support, intelligent optimization, and advanced conversion modes.

Version: 0.2.0
Author: Nec Zhang
"""

# Core business logic
from .core import *

# Infrastructure components
from .infrastructure import *

# Data models (commented out - models directory archived)
# from .models import *

# CLI interface
from .cli import cli

__version__ = "0.2.0"
__author__ = "Nec Zhang"

__all__ = [
    # Core components  
    "BaseLangGraphAgent",
    "NativeLangGraphMATLAB2CPPOrchestrator",
    
    # State
    "ConversionRequest",
    "ConversionStatus",
    
    # CLI
    "cli",
    
    # Metadata
    "__version__",
    "__author__"
]