# 🚀 MATLAB to C++ Agentic Service - Detailed Documentation

A **native LangGraph-based** agentic system for converting MATLAB projects to C++ with intelligent analysis, planning, and iterative optimization using state-of-the-art language models. Built with **Function-First architecture** and **multi-turn optimization** capabilities.

## ✨ Features

- 🧠 **Native LangGraph Architecture**: Built from the ground up using LangGraph for true agentic workflows
- 🔄 **Multi-Turn Optimization**: Iterative code improvement with intelligent quality assessment and feedback loops
- 📁 **Multi-File Project Support**: Convert entire MATLAB projects with automatic dependency resolution and function call tree analysis
- 🏗️ **Function-First Agent Design**: Specialized agents organized by function (analyzer, planner, generator, assessor, validator)
- 📊 **Comprehensive Quality Assessment**: Multi-dimensional code quality evaluation with detailed reports
- 🎯 **Flexible Conversion Modes**: Support for different C++ standards and project structures
- 💻 **Modern CLI Interface**: Clean command-line interface with rich configuration support
- ⚙️ **vLLM Integration**: Optimized for self-hosted vLLM with configurable LLM providers
- 📂 **Organized Output**: Structured output with separate directories for code, reports, and debug information
- 🔧 **Robust JSON Parsing**: Enhanced error handling for reliable LLM response processing
- 🎛️ **Advanced State Management**: LangGraph-based state management with memory and performance tracking

## 📦 Installation

### Prerequisites

Make sure you have [uv](https://docs.astral.sh/uv/) installed:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Project Setup

```bash
# Clone the repository
git clone <repository-url>
cd matlab2cpp_agentic_service

# Install dependencies using uv
uv sync

# Install in development mode
uv pip install -e .
```

### Dependencies

The project uses `uv` for dependency management. Key dependencies include:
- 📝 `loguru` - Advanced logging
- 🔧 `pydantic` - Data validation and configuration management
- ⚙️ `pyyaml` - YAML configuration file support
- 🖱️ `click` - CLI framework
- 🌐 `python-dotenv` - Environment variable management
- 🤖 `langgraph` - Native graph-based agent workflows
- 🔗 `langchain-community` - LLM provider support
- 🌐 `httpx` - HTTP client for vLLM integration

## 🚀 Usage

### 🔧 Quick Setup

1. **Configure your environment**:
   ```bash
   # Copy the template and customize
   cp examples/env_files/env.vllm.remote .env
   # Edit .env with your vLLM server settings
   ```

2. **Test your configuration**:
   ```bash
   uv run python -m matlab2cpp_agentic_service.cli test-llm
   ```

### 💻 Command Line Interface

**Using the native LangGraph CLI**:

```bash
# 🔄 Convert a MATLAB project to C++
uv run python -m matlab2cpp_agentic_service.cli convert examples/matlab_samples/arma_filter.m my_project

# 🔍 Analyze a MATLAB project without conversion
uv run python -m matlab2cpp_agentic_service.cli analyze examples/matlab_samples/arma_filter.m --detailed

# ✅ Validate a converted C++ project
uv run python -m matlab2cpp_agentic_service.cli validate output/my_project

# 🧪 Test LLM connection
uv run python -m matlab2cpp_agentic_service.cli test-llm

# 📚 Show available examples
uv run python -m matlab2cpp_agentic_service.cli examples
```

### ⚙️ Configuration Options

**Using different configurations**:

```bash
# Use specific .env file
uv run python -m matlab2cpp_agentic_service.cli --env examples/env_files/env.vllm.remote convert examples/matlab_samples/arma_filter.m my_project

# Use YAML config file
uv run python -m matlab2cpp_agentic_service.cli --config config/default_config.yaml convert examples/matlab_samples/arma_filter.m my_project

# Verbose output with custom .env
uv run python -m matlab2cpp_agentic_service.cli --verbose --env examples/env_files/env.development convert examples/matlab_samples/arma_filter.m my_project
```

### 🎯 Advanced Conversion Options

```bash
# Convert with custom parameters
uv run python -m matlab2cpp_agentic_service.cli convert examples/matlab_samples/arma_filter.m my_project \
  --output-dir ./cpp_output \
  --max-turns 3 \
  --target-quality 8.0 \
  --cpp-standard C++20

# Multi-file MATLAB project conversion
uv run python -m matlab2cpp_agentic_service.cli convert examples/matlab_samples/skeleton_vessel my_multi_file_project \
  --max-turns 2 \
  --target-quality 7.5

# Disable optimization (max-turns=0)
uv run python -m matlab2cpp_agentic_service.cli convert examples/matlab_samples/arma_filter.m my_project \
  --max-turns 0

# JSON output for automation
uv run python -m matlab2cpp_agentic_service.cli convert examples/matlab_samples/arma_filter.m my_project --json-output
```

### 🐍 Programmatic Usage

```python
from matlab2cpp_agentic_service.core.orchestrators.native_langgraph_orchestrator import NativeLangGraphMATLAB2CPPOrchestrator
from matlab2cpp_agentic_service.infrastructure.state.conversion_state import ConversionRequest

# 🏗️ Initialize native LangGraph orchestrator
orchestrator = NativeLangGraphMATLAB2CPPOrchestrator()

# 📋 Create conversion request for single file
request = ConversionRequest(
    matlab_path=Path("examples/matlab_samples/arma_filter.m"),
    project_name="my_cpp_project",
    output_dir=Path("./output"),
    max_optimization_turns=2,
    target_quality_score=7.0,
    cpp_standard="C++17"
)

# 📋 Create conversion request for multi-file project
multi_file_request = ConversionRequest(
    matlab_path=Path("examples/matlab_samples/skeleton_vessel"),  # Directory path
    project_name="my_multi_file_project",
    output_dir=Path("./output"),
    max_optimization_turns=3,
    target_quality_score=8.0
)

# 🚀 Convert project
result = orchestrator.convert_project(request)

if result.status == "completed":
    print(f"✅ Conversion successful! Quality score: {result.final_score}/10")
    print(f"📁 Generated files: {result.generated_files}")
    print(f"📋 Assessment reports: {result.assessment_reports}")
    print(f"🔄 Optimization turns used: {result.improvement_turns}")
else:
    print(f"❌ Conversion failed: {result.error_message}")
```

## 🏗️ Architecture

The system uses a **native LangGraph architecture** with Function-First agent organization:

### 🔧 Core Components

- 🎯 **NativeLangGraphMATLAB2CPPOrchestrator**: Main orchestrator using LangGraph workflows
- 🔍 **MATLAB Analyzer Agents**: Analyze MATLAB code structure, content, and multi-file dependencies
- 📋 **Conversion Planner Agents**: Create comprehensive C++ conversion plans with project structure planning
- ⚡ **C++ Generator Agents**: Generate optimized C++ code for both single and multi-file projects
- 📊 **Quality Assessor Agents**: Evaluate code quality with multi-dimensional assessment
- ✅ **Validator Agents**: Validate converted C++ projects
- 🔧 **LangGraph Tools**: Specialized tools for LLM interactions with centralized prompts

### 📁 Function-First Project Structure

```
matlab2cpp_agentic_service/
├── 📄 pyproject.toml              # uv project configuration
├── 📄 README.md                   # This file
├── 📄 PROJECT_STRUCTURE.md        # Detailed project structure documentation
├── 📂 config/                     # Configuration files
│   └── default_config.yaml
├── 📂 examples/                   # Example files and samples
│   ├── 📁 env_files/              # Example .env configurations
│   │   ├── env.vllm.local         # Local vLLM server config
│   │   ├── env.vllm.remote        # Remote vLLM server config
│   │   ├── env.openai             # OpenAI API config
│   │   └── env.development        # Development mode config
│   └── 📁 matlab_samples/         # Example MATLAB projects
│       ├── arma_filter.m          # Single-file example
│       └── skeleton_vessel/       # Multi-file project example
├── 📂 src/matlab2cpp_agentic_service/
│   ├── 📁 cli/                    # Command-line interface
│   │   ├── run.py                 # Main CLI entry point
│   │   └── __main__.py            # CLI module entry point
│   ├── 📁 core/                   # Core business logic
│   │   ├── 📁 agents/             # Function-First agent organization
│   │   │   ├── 📁 analyzer/       # MATLAB analysis agents
│   │   │   │   ├── 📁 langgraph/  # Native LangGraph agents
│   │   │   │   └── 📁 legacy/     # Legacy agents (backward compatibility)
│   │   │   ├── 📁 assessor/       # Quality assessment agents
│   │   │   │   └── 📁 langgraph/  # Native LangGraph assessors
│   │   │   ├── 📁 base/           # Base agent classes and utilities
│   │   │   │   ├── agent_registry.py
│   │   │   │   ├── langgraph_agent.py
│   │   │   │   ├── memory_manager.py
│   │   │   │   └── performance_monitor.py
│   │   │   ├── 📁 generator/      # C++ code generation agents
│   │   │   │   └── 📁 langgraph/  # Native LangGraph generators
│   │   │   ├── 📁 planner/        # Conversion planning agents
│   │   │   │   └── 📁 langgraph/  # Native LangGraph planners
│   │   │   └── 📁 validator/      # Code validation agents
│   │   │       └── 📁 langgraph/  # Native LangGraph validators
│   │   ├── 📁 orchestrators/      # Workflow orchestration
│   │   │   └── native_langgraph_orchestrator.py
│   │   └── 📁 workflows/          # LangGraph workflows
│   │       └── native_langgraph_workflow.py
│   ├── 📁 infrastructure/         # Infrastructure components
│   │   ├── 📁 state/              # State management
│   │   │   ├── conversion_state.py
│   │   │   ├── shared_memory.py
│   │   │   └── state_validator.py
│   │   └── 📁 tools/              # LangGraph tools and utilities
│   │       ├── langgraph_tools.py # Centralized LangGraph tools with prompts
│   │       ├── llm_client.py      # LLM client for vLLM integration
│   │       └── matlab_parser.py   # MATLAB parsing utilities
│   ├── 📁 models/                 # Data models and schemas
│   ├── 📁 templates/              # Code templates
│   └── 📁 utils/                  # Utility functions
├── 📁 output/                     # Generated C++ projects (organized by project)
│   ├── project_name/
│   │   ├── generated_code/        # C++ files (.h, .cpp)
│   │   ├── reports/              # Assessment reports (.md)
│   │   └── debug/                # Debug information (.json, .txt)
├── 📁 tests/                      # Test suite
│   ├── 📁 fixtures/              # Test fixtures
│   ├── 📁 integration/           # Integration tests
│   └── 📁 unit/                  # Unit tests
└── 📄 uv.lock                     # Dependency lock file
```

## ⚙️ Configuration

The system supports multiple LLM providers and can be configured using `.env` files or environment variables.

### 🔧 Environment Configuration (.env)

**Recommended approach**: Use `.env` files for easy configuration management.

1. **Copy the template**:
   ```bash
   cp examples/env_files/env.vllm.remote .env
   ```

2. **Edit the `.env` file** with your settings:
   ```bash
   # For remote vLLM server
   LLM_PROVIDER=vllm
   VLLM_ENDPOINT=http://192.168.6.10:8002
   VLLM_MODEL_NAME=Qwen/Qwen3-32B-FP8
   VLLM_API_KEY=dummy_key
   ```

3. **Use example configurations**:
   ```bash
   # For local vLLM
   cp examples/env_files/env.vllm.local .env
   
   # For remote vLLM
   cp examples/env_files/env.vllm.remote .env
   
   # For OpenAI
   cp examples/env_files/env.openai .env
   
   # For development
   cp examples/env_files/env.development .env
   ```

### 🌐 Environment Variables

Alternatively, set environment variables directly:

```bash
# 🚀 For vLLM (recommended)
export VLLM_ENDPOINT="http://your-vllm-server:8002"
export VLLM_MODEL_NAME="Qwen/Qwen3-32B-FP8"
export LLM_PROVIDER="vllm"
export LLM_API_KEY="your-api-key"

# 🤖 For OpenAI
export OPENAI_API_KEY="your-openai-api-key"
export LLM_PROVIDER="openai"
export LLM_MODEL="gpt-4"
```

### 📋 Available Configuration Options

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `LLM_PROVIDER` | LLM provider (vllm, openai) | `vllm` | `vllm` |
| `VLLM_ENDPOINT` | vLLM server endpoint | `http://localhost:8000` | `http://192.168.6.10:8002` |
| `VLLM_MODEL_NAME` | vLLM model name | `Qwen/Qwen3-32B-FP8` | `Qwen/Qwen3-32B-FP8` |
| `OPENAI_API_KEY` | OpenAI API key | - | `sk-...` |
| `OPENAI_MODEL` | OpenAI model | `gpt-4` | `gpt-4-turbo` |
| `LLM_TIMEOUT` | Request timeout (seconds) | `1200` | `300` |
| `LLM_MAX_TOKENS` | Max tokens per request | `8000` | `4000` |
| `LLM_TEMPERATURE` | Response temperature | `0.1` | `0.2` |
| `DEFAULT_OUTPUT_DIR` | Output directory | `./output` | `./cpp_projects` |
| `DEFAULT_CPP_STANDARD` | C++ standard | `C++17` | `C++20` |
| `DEFAULT_MAX_TURNS` | Max optimization turns | `2` | `3` |
| `DEFAULT_TARGET_QUALITY` | Target quality score | `7.0` | `8.5` |
| `LOG_LEVEL` | Log level | `INFO` | `DEBUG` |

## 🔄 Conversion Process

### Native LangGraph Workflow

The system uses a **native LangGraph workflow** with the following stages:

1. 🔍 **Analysis**: MATLAB content is analyzed using LangGraph-native agents
2. 📋 **Planning**: A comprehensive C++ conversion plan is created with architecture recommendations
3. ⚡ **Generation**: Initial C++ code is generated following the conversion plan
4. 📊 **Assessment**: Code quality is evaluated across multiple dimensions
5. 🔄 **Optimization Check**: Intelligent decision on whether to continue optimization
6. ✅ **Validation**: Final code is validated for compilation and functionality

### Multi-Turn Optimization

The system supports **intelligent multi-turn optimization**:

- **Turn 0**: Initial generation
- **Turn 1-N**: Optimization turns based on quality assessment
- **Early Termination**: Stops if quality is declining or target is met
- **Max Turns Control**: Configurable maximum optimization turns (including `max-turns=0`)

### Multi-File Projects

1. 🔍 **Function Call Detection**: Analyzes all `.m` files to build dependency graphs
2. 📋 **Project Structure Planning**: Determines optimal C++ file organization
3. 🏗️ **Compilation Order Planning**: Uses topological sorting for correct compilation sequence
4. ⚡ **Multi-File Generation**: Generates coordinated C++ files with proper includes
5. 📊 **Project-Level Assessment**: Evaluates the entire project for quality
6. 🔄 **Iterative Optimization**: Improves code across multiple turns if needed

## 📊 Quality Assessment

The system provides comprehensive quality assessment including:
- 🎯 **Overall Quality**: Combined score from all assessment dimensions
- ⚡ **Performance**: Optimization opportunities and efficiency analysis
- 🛡️ **Error Handling**: Robustness and edge case coverage
- 🎨 **Code Style**: Syntax, formatting, and best practices compliance
- 🛠️ **Maintainability**: Code structure, documentation, and readability
- ✅ **Functional Equivalence**: Correctness compared to original MATLAB
- 📋 **Completeness**: Coverage of all MATLAB functionality

**Quality Score**: Overall score from 0-10 with detailed breakdown and improvement recommendations.

## 🛠️ Development

### Quick Start

```bash
# 🔧 Install development dependencies
uv sync --dev

# 🚀 Test LLM connection
uv run python -m matlab2cpp_agentic_service.cli test-llm

# 🧪 Test single file conversion
uv run python -m matlab2cpp_agentic_service.cli convert examples/matlab_samples/arma_filter.m test_project

# 🧪 Test multi-file conversion
uv run python -m matlab2cpp_agentic_service.cli convert examples/matlab_samples/skeleton_vessel test_multi_project
```

### Development Workflow

```bash
# 🚀 Run with development configuration
uv run python -m matlab2cpp_agentic_service.cli --env examples/env_files/env.development convert examples/matlab_samples/arma_filter.m dev_test

# 📦 Add new dependencies
uv add package-name

# 🔄 Update dependencies
uv sync

# 🧹 Clean virtual environment
uv sync --reinstall
```

### Testing

```bash
# Run all tests
uv run pytest tests/

# Run integration tests
uv run pytest tests/integration/

# Run specific test
uv run pytest tests/integration/test_multi_file_conversion.py
```

## 📚 Examples

See the `examples/` directory for sample MATLAB projects and their C++ conversions in the `output/` directory.

### 🎯 Quick Examples

**Single-File Conversion**:
```bash
# Setup configuration
cp examples/env_files/env.vllm.remote .env

# Convert the ARMA filter example
uv run python -m matlab2cpp_agentic_service.cli convert examples/matlab_samples/arma_filter.m arma_filter_cpp

# View results
ls output/arma_filter_cpp/
```

**Multi-File Project Conversion**:
```bash
# Convert entire MATLAB project with multiple files
uv run python -m matlab2cpp_agentic_service.cli convert examples/matlab_samples/skeleton_vessel skeleton_vessel_cpp \
  --max-turns 2

# View organized results
ls output/skeleton_vessel_cpp/
```

### 📋 Example Output Structure

**Single-File Projects**:
```
output/project_name/
├── generated_code/
│   ├── v1.h                    # Header file
│   └── v1.cpp                  # Implementation file
├── reports/
│   └── v1_assessment_report.md # Quality analysis
└── debug/                      # Debug information
```

**Multi-File Projects**:
```
output/project_name/
├── generated_code/
│   ├── v1_function1.h/.cpp     # Header/implementation pairs
│   ├── v1_function2.h/.cpp
│   ├── v1_main.cpp             # Main entry point
│   └── v1_compilation_instructions.md
├── reports/
│   └── v1_assessment_report.md # Project-level quality analysis
└── debug/                      # Debug information
```

## 🆕 Recent Improvements

### Version 0.2.0 - Native LangGraph Architecture
- ✅ **Native LangGraph Implementation**: Complete rewrite using LangGraph for true agentic workflows
- ✅ **Function-First Agent Organization**: Agents organized by function rather than framework type
- ✅ **Enhanced State Management**: LangGraph-based state management with memory and performance tracking
- ✅ **Robust JSON Parsing**: Enhanced error handling with centralized JSON cleaning utilities
- ✅ **Multi-Turn Optimization**: Intelligent optimization with early termination and declining quality detection
- ✅ **Clean Project Structure**: Organized, professional project structure with comprehensive documentation
- ✅ **Improved CLI**: Modern CLI with better error handling and configuration management
- ✅ **vLLM Integration**: Optimized for self-hosted vLLM with reliable connection handling

### Key Features
- 🧠 **Native LangGraph**: Built from the ground up using LangGraph for full framework utilization
- 🔄 **Multi-Turn Optimization**: Intelligent iterative improvement with quality-based decisions
- 📁 **Multi-File Support**: Complete support for MATLAB projects with multiple `.m` files
- 🏗️ **Function-First Architecture**: Clear separation of concerns with specialized agents
- 📊 **Quality Assessment**: Multi-dimensional quality evaluation with detailed reports
- 🎯 **Flexible Configuration**: Support for multiple LLM providers and configuration methods

## 🔧 Troubleshooting

### Common Issues

**LLM Connection Failed**:
```bash
# Test your configuration
uv run python -m matlab2cpp_agentic_service.cli test-llm

# Check your .env file
cat .env

# Try different configuration
cp examples/env_files/env.vllm.local .env
```

**Conversion Takes Too Long**:
```bash
# Use development configuration for faster processing
cp examples/env_files/env.development .env

# Reduce optimization turns
uv run python -m matlab2cpp_agentic_service.cli convert examples/matlab_samples/arma_filter.m my_project --max-turns 1
```

**Low Quality Scores**:
```bash
# Increase optimization turns
uv run python -m matlab2cpp_agentic_service.cli convert examples/matlab_samples/arma_filter.m my_project --max-turns 3 --target-quality 8.5

# Enable verbose output for debugging
uv run python -m matlab2cpp_agentic_service.cli --verbose convert examples/matlab_samples/arma_filter.m my_project
```

**Multi-File Project Issues**:
```bash
# Check if MATLAB project directory contains .m files
ls examples/matlab_samples/skeleton_vessel/

# Check organized output structure
ls output/my_project/
```

### Getting Help

1. **Check the logs**: Enable verbose mode with `--verbose` flag
2. **Test LLM connection**: Run `uv run python -m matlab2cpp_agentic_service.cli test-llm`
3. **Try example configurations**: Use pre-configured .env files in `examples/env_files/`
4. **Review assessment reports**: Check generated `.md` files in `output/` directory
5. **Check project structure**: See `PROJECT_STRUCTURE.md` for detailed architecture

## 📄 License

MIT License - see LICENSE file for details.

---

<div align="center">

**🚀 Built with ❤️ using native LangGraph architecture and modern Python tooling**

[![uv](https://img.shields.io/badge/uv-0.2.0-blue.svg)](https://docs.astral.sh/uv/)
[![Python](https://img.shields.io/badge/Python-3.12+-green.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Native-orange.svg)](https://langchain.com/langgraph)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Transform MATLAB code into production-ready C++ with native LangGraph agents, multi-file project support, and intelligent multi-turn optimization**

</div>
