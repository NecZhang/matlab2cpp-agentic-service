# MATLAB to C++ Conversion Agent

A LangGraph-based agentic system for converting MATLAB projects to C++ while maintaining functionality and ensuring validation.

## Features

- **LLM-Centric Analysis**: Deep understanding of MATLAB code content and purpose
- **Multi-Agent Architecture**: Specialized agents for analysis, mapping, generation, and validation
- **Content-First Approach**: Handles unclear MATLAB project structure through content analysis
- **Incremental Conversion**: Convert functions one by one with validation
- **Comprehensive Validation**: Unit tests and functional equivalence testing

## Installation

```bash
# Install dependencies
uv add langgraph langchain langchain-openai pydantic pyyaml jinja2 click pytest

# Install in development mode
uv pip install -e .
```

## Usage

```bash
# Convert a MATLAB project
matlab2cpp convert /path/to/matlab/project --output /path/to/cpp/project

# Analyze a MATLAB project without conversion
matlab2cpp analyze /path/to/matlab/project
```

## Project Structure

```
matlab2cpp_agent/
├── src/
│   ├── agents/          # LangGraph agents
│   ├── tools/           # Analysis and conversion tools
│   ├── workflows/       # LangGraph workflows
│   ├── templates/       # C++ project templates
│   └── utils/           # Utilities and configuration
├── examples/            # Example MATLAB projects
├── tests/              # Test suite
└── docs/               # Documentation
```

## Development

```bash
# Run tests
pytest

# Format code
black src/
isort src/

# Type checking
mypy src/
```

## License

MIT License
