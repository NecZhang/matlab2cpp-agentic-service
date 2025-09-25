# 🚀 MATLAB to C++ Agentic Service

**Native LangGraph-based** agentic system for converting MATLAB projects to C++ with intelligent analysis, planning, and multi-turn optimization.

## ✨ Key Features

- 🧠 **Native LangGraph Architecture** - True agentic workflows with specialized agents
- 🔄 **Multi-Turn Optimization** - Iterative code improvement with quality assessment
- 📁 **Multi-File Project Support** - Convert entire MATLAB projects with dependency resolution
- 🎯 **Flexible Conversion Modes** - Support for different C++ standards and structures
- ⚙️ **vLLM Integration** - Optimized for self-hosted vLLM with configurable providers
- 📊 **Quality Assessment** - Multi-dimensional code quality evaluation
- 💻 **Modern CLI** - Clean command-line interface with rich configuration
- 🛡️ **Robust Error Handling** - Enhanced JSON parsing and error recovery
- 🎨 **Organized Output** - Structured project directories with reports and debug info

## 🚀 Quick Start

### 📦 Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone <repository-url>
cd matlab2cpp_agentic_service
uv sync
uv pip install -e .
```

### ⚙️ Configuration

```bash
# Setup environment
cp examples/env_files/env.vllm.remote .env
# Edit .env with your vLLM server settings

# Test connection
uv run python -m matlab2cpp_agentic_service.cli test-llm
```

### 💻 Usage

```bash
# 🔄 Convert single MATLAB file
uv run python -m matlab2cpp_agentic_service.cli convert examples/matlab_samples/arma_filter.m my_project

# 📁 Convert multi-file MATLAB project
uv run python -m matlab2cpp_agentic_service.cli convert examples/matlab_samples/skeleton_vessel my_project --max-turns 2

# 🔍 Analyze without conversion
uv run python -m matlab2cpp_agentic_service.cli analyze examples/matlab_samples/arma_filter.m --detailed

# ✅ Validate converted project
uv run python -m matlab2cpp_agentic_service.cli validate output/my_project
```

## 🎯 Advanced Options

```bash
# 🔧 Custom parameters
uv run python -m matlab2cpp_agentic_service.cli convert examples/matlab_samples/arma_filter.m my_project \
  --max-turns 3 \
  --target-quality 8.0 \
  --cpp-standard C++20

# 🛑 Disable optimization
uv run python -m matlab2cpp_agentic_service.cli convert examples/matlab_samples/arma_filter.m my_project --max-turns 0

# 📄 JSON output for automation
uv run python -m matlab2cpp_agentic_service.cli convert examples/matlab_samples/arma_filter.m my_project --json-output
```

## 🏗️ Architecture

**Function-First Agent Organization:**
- 🔍 **Analyzer Agents** - MATLAB code analysis and dependency resolution
- 📋 **Planner Agents** - C++ conversion planning and project structure
- ⚡ **Generator Agents** - C++ code generation with optimization
- 📊 **Assessor Agents** - Multi-dimensional quality assessment
- ✅ **Validator Agents** - Code validation and testing

**Native LangGraph Workflow:**
1. 🔍 **Analysis** → 📋 **Planning** → ⚡ **Generation** → 📊 **Assessment** → 🔄 **Optimization** → ✅ **Validation**

## 📊 Output Structure

```
output/project_name/
├── 📁 generated_code/           # C++ files (.h, .cpp)
├── 📋 reports/                 # Quality assessment reports
└── 🐛 debug/                   # Debug information
```

## ⚙️ Configuration

### 🌐 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM provider (vllm, openai) | `vllm` |
| `VLLM_ENDPOINT` | vLLM server endpoint | `http://localhost:8000` |
| `VLLM_MODEL_NAME` | vLLM model name | `Qwen/Qwen3-32B-FP8` |
| `LLM_TIMEOUT` | Request timeout (seconds) | `1200` |
| `DEFAULT_MAX_TURNS` | Max optimization turns | `2` |
| `DEFAULT_TARGET_QUALITY` | Target quality score | `7.0` |

### 📝 Example Configurations

```bash
# 🏠 Local vLLM
cp examples/env_files/env.vllm.local .env

# 🌐 Remote vLLM
cp examples/env_files/env.vllm.remote .env

# 🤖 OpenAI
cp examples/env_files/env.openai .env
```

## 🐍 Programmatic Usage

```python
from matlab2cpp_agentic_service.core.orchestrators.native_langgraph_orchestrator import NativeLangGraphMATLAB2CPPOrchestrator
from matlab2cpp_agentic_service.infrastructure.state.conversion_state import ConversionRequest

# 🏗️ Initialize orchestrator
orchestrator = NativeLangGraphMATLAB2CPPOrchestrator()

# 📋 Create request
request = ConversionRequest(
    matlab_path=Path("examples/matlab_samples/arma_filter.m"),
    project_name="my_cpp_project",
    output_dir=Path("./output"),
    max_optimization_turns=2,
    target_quality_score=7.0
)

# 🚀 Convert
result = orchestrator.convert_project(request)

if result.status == "completed":
    print(f"✅ Quality score: {result.final_score}/10")
    print(f"📁 Generated files: {result.generated_files}")
```

## 🔧 Troubleshooting

```bash
# 🧪 Test LLM connection
uv run python -m matlab2cpp_agentic_service.cli test-llm

# 🔍 Verbose output for debugging
uv run python -m matlab2cpp_agentic_service.cli --verbose convert examples/matlab_samples/arma_filter.m my_project

# 📁 Check example configurations
ls examples/env_files/
```

## 📚 Documentation

- 📖 **Detailed Documentation**: See `README_DETAILS.md` for comprehensive information
- 🏗️ **Architecture**: Function-First agent organization with native LangGraph workflows
- 🔄 **Multi-Turn Optimization**: Intelligent iterative improvement with quality-based decisions
- 📁 **Multi-File Support**: Complete support for MATLAB projects with dependency resolution

## 🆕 Recent Updates

**Version 0.2.0 - Native LangGraph Architecture:**
- ✅ Complete rewrite using LangGraph for true agentic workflows
- ✅ Function-First agent organization
- ✅ Enhanced state management with memory and performance tracking
- ✅ Robust JSON parsing with centralized error handling
- ✅ Intelligent multi-turn optimization
- ✅ Clean project structure with comprehensive documentation

## 📄 License

MIT License - see LICENSE file for details.

---

<div align="center">

**🚀 Built with ❤️ using native LangGraph architecture and modern Python tooling**

[![uv](https://img.shields.io/badge/uv-0.2.0-blue.svg)](https://docs.astral.sh/uv/)
[![Python](https://img.shields.io/badge/Python-3.12+-green.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Native-orange.svg)](https://langchain.com/langgraph)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Transform MATLAB code into production-ready C++ with native LangGraph agents and intelligent optimization**

</div>