# 🚀 MATLAB to C++ Agentic Service

A modern agentic system for converting MATLAB projects to C++ with intelligent analysis, planning, and iterative optimization using state-of-the-art language models.

## ✨ Features

- 🧠 **Intelligent Analysis**: Deep understanding of MATLAB code content and purpose using specialized agents
- 🏗️ **Modern Orchestrator Architecture**: Clean, modular design with specialized agents for each conversion step
- 🔄 **Iterative Optimization**: Automatic code improvement with quality assessment and feedback loops
- 📊 **Comprehensive Quality Assessment**: Multi-dimensional code quality evaluation with detailed reports
- 🎯 **Flexible Output**: Support for different C++ standards and project structures
- 💻 **Unified CLI Interface**: Single `run.py` script for all operations with rich configuration support
- ⚙️ **Environment-Based Configuration**: Easy setup with `.env` files and YAML configuration support
- 🤖 **Multi-LLM Support**: Works with vLLM, OpenAI, and other LLM providers

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
- 🤖 `langchain` & `langgraph` - LLM integration and agent orchestration
- 🔗 `langchain-openai` & `langchain-community` - LLM provider support

## 🚀 Usage

### 🔧 Quick Setup

1. **Configure your environment**:
   ```bash
   # Copy the template and customize
   cp .env.template .env
   # Edit .env with your LLM provider settings
   ```

2. **Test your configuration**:
   ```bash
   uv run python run.py test-llm
   ```

### 💻 Command Line Interface

**Using the unified runner script (recommended)**:

```bash
# 🔄 Convert a MATLAB project to C++
uv run python run.py convert examples/matlab_samples/arma_filter.m my_project

# 🔍 Analyze a MATLAB project without conversion
uv run python run.py analyze examples/matlab_samples/arma_filter.m --detailed

# ✅ Validate a converted C++ project
uv run python run.py validate output/my_project

# 🧪 Test LLM connection
uv run python run.py test-llm

# 📚 Show available examples
uv run python run.py examples
```

### ⚙️ Configuration Options

**Using different .env configurations**:

```bash
# Use specific .env file
uv run python run.py --env examples/env_files/env.vllm.remote convert examples/matlab_samples/arma_filter.m my_project

# Use YAML config file
uv run python run.py --config config/default_config.yaml convert examples/matlab_samples/arma_filter.m my_project

# Verbose output with custom .env
uv run python run.py --verbose --env examples/env_files/env.development convert examples/matlab_samples/arma_filter.m my_project
```

### 🎯 Advanced Conversion Options

```bash
# Convert with custom parameters
uv run python run.py convert examples/matlab_samples/arma_filter.m my_project \
  --output-dir ./cpp_output \
  --max-turns 3 \
  --target-quality 8.0 \
  --cpp-standard C++20 \
  --include-tests

# JSON output for automation
uv run python run.py convert examples/matlab_samples/arma_filter.m my_project --json-output
```

### 🐍 Programmatic Usage

```python
from matlab2cpp_agentic_service import MATLAB2CPPOrchestrator, ConversionRequest

# 🏗️ Initialize orchestrator (uses .env configuration automatically)
orchestrator = MATLAB2CPPOrchestrator()

# 📋 Create conversion request
request = ConversionRequest(
    matlab_path="examples/matlab_samples/arma_filter.m",
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
    print(f"📋 Assessment reports: {result.assessment_reports}")
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
matlab2cpp_agentic_service/
├── 🚀 run.py                   # Unified runner script (main entry point)
├── 📄 .env.template            # Environment configuration template
├── 📂 config/                  # YAML configuration files
│   └── default_config.yaml
├── 📂 examples/
│   ├── 📁 env_files/           # Example .env configurations
│   │   ├── env.vllm.local      # Local vLLM server config
│   │   ├── env.vllm.remote     # Remote vLLM server config
│   │   ├── env.openai          # OpenAI API config
│   │   └── env.development     # Development mode config
│   └── 📁 matlab_samples/      # Example MATLAB projects
│       └── arma_filter.m
├── 📂 src/matlab2cpp_agentic_service/
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
│   │   ├── config.py           # Enhanced with .env support
│   │   └── logger.py
│   ├── 💻 cli/                 # Command-line interfaces
│   │   └── general_converter.py
│   └── main.py                 # Legacy CLI entry point
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

1. 🔍 **Analysis**: MATLAB content is analyzed using LLM-powered agents to understand structure, functions, and dependencies
2. 📋 **Planning**: A comprehensive C++ conversion plan is created with architecture and strategy recommendations
3. ⚡ **Generation**: Initial C++ code is generated following the conversion plan with proper headers and implementations
4. 📊 **Assessment**: Code quality is evaluated across multiple dimensions (algorithmic accuracy, performance, style, maintainability)
5. 🔄 **Optimization**: Code is iteratively improved based on assessment feedback (if enabled)
6. ✅ **Validation**: Final code is validated for compilation and functionality
7. 📋 **Reporting**: Detailed assessment reports and generated files are provided

## 📊 Quality Assessment

The system provides comprehensive quality assessment including:
- 🎯 **Algorithmic Accuracy**: Mathematical correctness and logic validation (10.0/10)
- ⚡ **Performance**: Optimization opportunities and efficiency analysis (10.0/10)
- 🛡️ **Error Handling**: Robustness and edge case coverage (10.0/10)
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

# 🚀 Use the unified runner for development
uv run python run.py --env examples/env_files/env.development test-llm

# 🧪 Test conversion with development settings
uv run python run.py --verbose --env examples/env_files/env.development convert examples/matlab_samples/arma_filter.m test_project

# ✅ Run syntax checks
uv run python3 -m py_compile run.py

# 🎨 Format code (if using black/isort)
uv run black src/ run.py
uv run isort src/ run.py

# 🔍 Type checking (if using mypy)
uv run mypy src/ run.py
```

### Development Workflow

```bash
# 🚀 Run the converter with development configuration
uv run python run.py --env examples/env_files/env.development convert examples/matlab_samples/arma_filter.m dev_test

# 📦 Add new dependencies
uv add package-name

# 🔄 Update dependencies
uv sync

# 🧹 Clean virtual environment
uv sync --reinstall
```

### Configuration for Development

Use the development environment configuration:
```bash
cp examples/env_files/env.development .env
# This enables DEBUG logging, profiling, and faster analysis
```

## 📚 Examples

See the `examples/` directory for sample MATLAB projects and their C++ conversions in the `output/` directory.

### 🎯 Quick Example

```bash
# Setup configuration
cp examples/env_files/env.vllm.remote .env

# Convert the ARMA filter example
uv run python run.py convert examples/matlab_samples/arma_filter.m arma_filter_cpp

# View results
ls output/arma_filter_cpp*
```

### 📋 Example Output

After conversion, you'll find:
- **Header file**: `arma_filter_cpp_v1.h` - Class declarations and interfaces
- **Implementation**: `arma_filter_cpp_v1.cpp` - Complete C++ implementation
- **Assessment report**: `arma_filter_cpp_v1_assessment_report.md` - Quality analysis and recommendations

### 🔍 Example Quality Assessment

```
Overall Score: 8.3/10

📈 Detailed Metrics:
  • Algorithmic Accuracy: 10.0/10
  • Performance: 10.0/10
  • Error Handling: 10.0/10
  • Code Style: 1.5/10
  • Maintainability: 1.5/10

🎉 Excellent quality! Ready for production use.
```

## 🔧 Troubleshooting

### Common Issues

**LLM Connection Failed**:
```bash
# Test your configuration
uv run python run.py test-llm

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
uv run python run.py convert examples/matlab_samples/arma_filter.m my_project --max-turns 1
```

**Low Quality Scores**:
```bash
# Increase optimization turns
uv run python run.py convert examples/matlab_samples/arma_filter.m my_project --max-turns 3 --target-quality 8.5

# Enable verbose output for debugging
uv run python run.py --verbose convert examples/matlab_samples/arma_filter.m my_project
```

### Getting Help

1. **Check the logs**: Enable verbose mode with `--verbose` flag
2. **Test LLM connection**: Run `uv run python run.py test-llm`
3. **Try example configurations**: Use pre-configured .env files in `examples/env_files/`
4. **Review assessment reports**: Check generated `.md` files in `output/` directory

## 📄 License

MIT License - see LICENSE file for details.

---

<div align="center">

**🚀 Built with ❤️ using modern Python tooling and AI agents**

[![uv](https://img.shields.io/badge/uv-0.1.0-blue.svg)](https://docs.astral.sh/uv/)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-orange.svg)](https://langchain.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Transform MATLAB code into production-ready C++ with AI-powered analysis and optimization**

</div>
