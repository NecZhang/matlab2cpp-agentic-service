# 🚀 MATLAB to C++ Conversion Agent

A modern agentic system for converting MATLAB projects to C++ with intelligent analysis, planning, and iterative optimization.

## ✨ Features

- 🧠 **Intelligent Analysis**: Deep understanding of MATLAB code content and purpose using specialized agents
- 🏗️ **Modern Orchestrator Architecture**: Clean, modular design with specialized agents for each conversion step
- 🔄 **Iterative Optimization**: Automatic code improvement with quality assessment and feedback loops
- 📊 **Comprehensive Quality Assessment**: Multi-dimensional code quality evaluation with detailed reports
- 🎯 **Flexible Output**: Support for different C++ standards and project structures
- 💻 **CLI Interface**: Easy-to-use command-line interface for conversion and analysis

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
cd matlab2cpp_agent

# Install dependencies using uv
uv sync

# Install in development mode
uv pip install -e .
```

### Dependencies

The project uses `uv` for dependency management. Key dependencies include:
- 📝 `loguru` - Advanced logging
- 🔧 `pydantic` - Data validation
- ⚙️ `pyyaml` - Configuration files
- 🖱️ `click` - CLI framework

## 🚀 Usage

### 💻 Command Line Interface

**Using the general runner script (recommended)**:

```bash
# 🔄 Convert a MATLAB project to C++
uv run python run.py convert /path/to/matlab/project my_project

# 🔍 Analyze a MATLAB project without conversion
uv run python run.py analyze /path/to/matlab/project --detailed

# ✅ Validate a converted C++ project
uv run python run.py validate /path/to/cpp/project

# 🧪 Test LLM connection
uv run python run.py test-llm

# 📚 Show available examples
uv run python run.py examples
```

**Using custom .env configuration**:

```bash
# Use specific .env file
uv run python run.py --env examples/env_files/env.vllm.remote convert examples/matlab_samples/arma_filter.m my_project

# Use YAML config file
uv run python run.py --config my_config.yaml convert examples/matlab_samples/arma_filter.m my_project
```

### ⚙️ Advanced Options

```bash
# 🎯 Convert with custom options
uv run python run.py convert examples/matlab_samples/arma_filter.m my_project \
  --output-dir ./cpp_output \
  --max-turns 3 \
  --target-quality 8.0 \
  --cpp-standard C++20 \
  --include-tests

# 🔧 Verbose output with custom .env
uv run python run.py --verbose --env examples/env_files/env.development convert examples/matlab_samples/arma_filter.m my_project
```

### 🐍 Programmatic Usage

```python
from matlab2cpp_agent import MATLAB2CPPOrchestrator, ConversionRequest

# 🏗️ Initialize orchestrator
orchestrator = MATLAB2CPPOrchestrator()

# 📋 Create conversion request
request = ConversionRequest(
    matlab_path="/path/to/matlab/project",
    project_name="my_cpp_project",
    output_dir="./output",
    max_optimization_turns=2,
    target_quality_score=7.0,
    include_tests=True,
    cpp_standard="C++17"
)

# 🚀 Convert project
result = orchestrator.convert_project(request)

if result.status == "completed":
    print(f"✅ Conversion successful! Quality score: {result.final_score}/10")
    print(f"📁 Generated files: {result.generated_files}")
else:
    print(f"❌ Conversion failed: {result.error_message}")
```

## 🏗️ Architecture

The system uses a modern orchestrator pattern with specialized agents:

### 🔧 Core Components

- 🎯 **MATLAB2CPPOrchestrator**: Main service that coordinates the conversion process
- 🔍 **MATLABContentAnalyzerAgent**: Analyzes MATLAB code structure and content
- 📋 **ConversionPlannerAgent**: Creates comprehensive C++ conversion plans
- ⚡ **CppGeneratorAgent**: Generates optimized C++ code
- 📊 **QualityAssessorAgent**: Evaluates code quality and provides improvement suggestions
- ✅ **ValidatorAgent**: Validates converted C++ projects

### 📁 Project Structure

```
matlab2cpp_agent/
├── 📂 src/matlab2cpp_agent/
│   ├── 🤖 agents/              # Specialized conversion agents
│   │   ├── matlab_content_analyzer.py
│   │   ├── conversion_planner.py
│   │   ├── cpp_generator.py
│   │   ├── quality_assessor.py
│   │   └── validator.py
│   ├── 🎯 services/            # Main service orchestrator
│   │   └── matlab2cpp_orchestrator.py
│   ├── 🔧 tools/               # Analysis and conversion tools
│   │   ├── matlab_parser.py
│   │   └── llm_client.py
│   ├── ⚙️ utils/               # Utilities and configuration
│   │   ├── config.py
│   │   └── logger.py
│   ├── 💻 cli/                 # Command-line interfaces
│   │   └── general_converter.py
│   └── main.py              # Main CLI entry point
├── 📚 examples/                # Example MATLAB projects
├── 📁 output/                  # Generated C++ projects
├── 📄 pyproject.toml           # uv project configuration
└── 📖 README.md
```

## ⚙️ Configuration

The system supports multiple LLM providers and can be configured using `.env` files or environment variables.

### 🔧 Environment Configuration (.env)

**Recommended approach**: Use `.env` files for easy configuration management.

1. **Copy the template**:
   ```bash
   cp .env.template .env
   ```

2. **Edit the `.env` file** with your settings:
   ```bash
   # For local vLLM server
   LLM_PROVIDER=vllm
   VLLM_ENDPOINT=http://localhost:8000
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

1. 🔍 **Analysis**: MATLAB content is analyzed to understand structure, functions, and dependencies
2. 📋 **Planning**: A comprehensive C++ conversion plan is created with architecture and strategy
3. ⚡ **Generation**: Initial C++ code is generated following the conversion plan
4. 📊 **Assessment**: Code quality is evaluated across multiple dimensions
5. 🔄 **Optimization**: Code is iteratively improved based on assessment feedback
6. ✅ **Validation**: Final code is validated for compilation and functionality

## 📊 Quality Assessment

The system provides comprehensive quality assessment including:
- 🎯 **Code Quality**: Syntax, style, and best practices
- 🔄 **Functional Equivalence**: Correctness compared to original MATLAB
- ⚡ **Performance**: Optimization opportunities and efficiency
- 🛠️ **Maintainability**: Code structure and documentation
- ✅ **Completeness**: Coverage of all MATLAB functionality

## 🛠️ Development

### Quick Start

```bash
# 🔧 Install development dependencies
uv sync --dev

# ✅ Run syntax checks
uv run python3 -m py_compile src/matlab2cpp_agent/main.py

# 🧪 Test LLM connection
uv run python3 -m src.matlab2cpp_agent.main test-llm

# 🎨 Format code (if using black/isort)
uv run black src/
uv run isort src/

# 🔍 Type checking (if using mypy)
uv run mypy src/
```

### Development Workflow

```bash
# 🚀 Run the converter with uv
uv run python -m src.matlab2cpp_agent.main convert examples/matlab_samples/arma_filter.m arma_filter_test

# 📦 Add new dependencies
uv add package-name

# 🔄 Update dependencies
uv sync

# 🧹 Clean virtual environment
uv sync --reinstall
```

## 📚 Examples

See the `examples/` directory for sample MATLAB projects and their C++ conversions in the `output/` directory.

### 🎯 Quick Example

```bash
# Convert the ARMA filter example
uv run python -m src.matlab2cpp_agent.main convert examples/matlab_samples/arma_filter.m arma_filter_cpp --output ./output
```

## 📄 License

MIT License - see LICENSE file for details.

---

<div align="center">

**🚀 Built with ❤️ using modern Python tooling**

[![uv](https://img.shields.io/badge/uv-0.1.0-blue.svg)](https://docs.astral.sh/uv/)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>
