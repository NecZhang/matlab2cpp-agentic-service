# MATLAB2C++ Agentic Service - Project Structure

## 📁 **Project Overview**

This is a native LangGraph-based agentic service for converting MATLAB code to C++ with multi-turn optimization capabilities.

## 🏗️ **Directory Structure**

```
matlab2cpp_agentic_service/
├── 📁 config/                    # Configuration files
│   └── default_config.yaml      # Default service configuration
├── 📁 examples/                  # Example files and samples
│   ├── env_files/               # Environment configuration files
│   └── matlab_samples/          # Sample MATLAB files for testing
├── 📁 output/                   # Generated C++ code outputs
│   ├── test_*.md                # Recent test outputs (kept for reference)
│   └── llm_response_*.txt       # Debug logs
├── 📁 src/                      # Source code
│   └── matlab2cpp_agentic_service/
│       ├── 📁 cli/              # Command-line interface
│       ├── 📁 core/             # Core business logic
│       │   ├── 📁 agents/       # AI agents (Function-First structure)
│       │   │   ├── 📁 analyzer/ # MATLAB analysis agents
│       │   │   │   ├── 📁 langgraph/    # Native LangGraph agents
│       │   │   │   └── 📁 legacy/       # Legacy agents (backward compatibility)
│       │   │   ├── 📁 assessor/ # Quality assessment agents
│       │   │   ├── 📁 base/     # Base agent classes and utilities
│       │   │   ├── 📁 generator/# C++ code generation agents
│       │   │   ├── 📁 planner/  # Conversion planning agents
│       │   │   └── 📁 validator/# Code validation agents
│       │   ├── 📁 orchestrators/# Workflow orchestration
│       │   └── 📁 workflows/    # LangGraph workflows
│       ├── 📁 infrastructure/   # Infrastructure components
│       │   ├── 📁 state/        # State management
│       │   └── 📁 tools/        # LangGraph tools and utilities
│       ├── 📁 models/           # Data models and schemas
│       ├── 📁 templates/        # Code templates
│       └── 📁 utils/            # Utility functions
├── 📁 tests/                    # Test suite
│   ├── 📁 fixtures/             # Test fixtures
│   ├── 📁 integration/          # Integration tests (moved from root)
│   └── 📁 unit/                 # Unit tests
├── 📄 pyproject.toml            # Project configuration
├── 📄 README.md                 # Project documentation
├── 📄 uv.lock                   # Dependency lock file
└── 📄 PROJECT_STRUCTURE.md      # This file
```

## 🔧 **Key Components**

### **Core Agents (Function-First Structure)**
- **Analyzer**: MATLAB code analysis and parsing
- **Assessor**: Code quality assessment and scoring
- **Generator**: C++ code generation
- **Planner**: Conversion strategy planning
- **Validator**: Generated code validation

### **Infrastructure**
- **LangGraph Tools**: Specialized tools for LLM interactions
- **State Management**: Conversion state and workflow management
- **CLI**: Command-line interface for service interaction

### **Workflows**
- **Native LangGraph Workflows**: Graph-based conversion workflows
- **Orchestrators**: High-level workflow orchestration

## 🚀 **Usage**

### **CLI Usage**
```bash
# Single file conversion
uv run python -m matlab2cpp_agentic_service.cli convert <matlab_file> <project_name> --output-dir <output_dir>

# Multi-file project conversion
uv run python -m matlab2cpp_agentic_service.cli convert <project_dir> <project_name> --output-dir <output_dir> --max-turns <turns>
```

### **Testing**
```bash
# Run all tests
uv run pytest tests/

# Run integration tests
uv run pytest tests/integration/
```

## 📋 **Recent Changes**

- ✅ Moved scattered test files to `tests/integration/`
- ✅ Cleaned up old test outputs (kept recent ones)
- ✅ Removed empty directories
- ✅ Consolidated duplicate output directories
- ✅ Organized Function-First agent structure

## 🔍 **Development Notes**

- **Native LangGraph**: The service uses native LangGraph agents for full framework utilization
- **Backward Compatibility**: Legacy agents are maintained for compatibility
- **Function-First**: Agents are organized by function rather than framework type
- **Clean Architecture**: Clear separation of concerns with infrastructure, core, and interface layers



