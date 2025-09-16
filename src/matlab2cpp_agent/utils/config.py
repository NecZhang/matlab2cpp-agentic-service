"""Configuration management for the MATLAB to C++ conversion system."""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class LLMConfig(BaseModel):
    """Configuration for LLM services."""
    provider: str = Field(default="vllm", description="LLM provider (vllm, openai, anthropic, etc.)")
    model: str = Field(default="gpt-3.5-turbo", description="Model name to use")
    api_key: Optional[str] = Field(default=None, description="API key for LLM service")
    temperature: float = Field(default=0.1, description="Temperature for LLM responses")
    max_tokens: int = Field(default=4000, description="Maximum tokens per request")
    timeout: int = Field(default=300, description="Request timeout in seconds")
    
    # vLLM specific configuration
    vllm_endpoint: str = Field(default="http://localhost:8000/v1", description="vLLM server endpoint")
    vllm_model_name: Optional[str] = Field(default=None, description="Model name for vLLM")
    
    # OpenAI specific configuration
    openai_base_url: Optional[str] = Field(default=None, description="OpenAI API base URL")
    base_url: Optional[str] = Field(default=None, description="Base URL for API requests")


class AnalysisConfig(BaseModel):
    """Configuration for MATLAB code analysis."""
    max_file_size: int = Field(default=100000, description="Maximum file size to analyze (bytes)")
    chunk_size: int = Field(default=8000, description="Chunk size for large files")
    analysis_passes: int = Field(default=3, description="Number of analysis passes")
    confidence_threshold: float = Field(default=0.7, description="Minimum confidence for analysis")


class ConversionConfig(BaseModel):
    """Configuration for C++ conversion."""
    cpp_standard: str = Field(default="c++17", description="C++ standard to use")
    include_optimization: bool = Field(default=True, description="Include optimization suggestions")
    generate_tests: bool = Field(default=True, description="Generate unit tests")
    create_documentation: bool = Field(default=True, description="Generate documentation")


class Config(BaseModel):
    """Main configuration class."""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    conversion: ConversionConfig = Field(default_factory=ConversionConfig)
    
    # Paths
    project_root: Path = Field(default=Path.cwd())
    templates_dir: Path = Field(default=Path("templates"))
    output_dir: Path = Field(default=Path("output"))
    
    class Config:
        arbitrary_types_allowed = True


def load_config(config_path: Optional[Path] = None) -> Config:
    """Load configuration from file or environment variables."""
    config = Config()
    
    # Load from file if provided
    if config_path and config_path.exists():
        with open(config_path, 'r') as f:
            file_config = yaml.safe_load(f)
            config = Config(**file_config)
    
    # Override with environment variables
    # vLLM configuration
    if os.getenv("VLLM_ENDPOINT"):
        config.llm.vllm_endpoint = os.getenv("VLLM_ENDPOINT")
        config.llm.base_url = os.getenv("VLLM_ENDPOINT") + "/v1"
        config.llm.provider = "vllm"
    
    if os.getenv("VLLM_API_KEY"):
        config.llm.api_key = os.getenv("VLLM_API_KEY")
    
    if os.getenv("VLLM_MODEL_NAME"):
        config.llm.vllm_model_name = os.getenv("VLLM_MODEL_NAME")
        config.llm.model = os.getenv("VLLM_MODEL_NAME")
    
    # OpenAI configuration
    if os.getenv("OPENAI_API_KEY"):
        config.llm.api_key = os.getenv("OPENAI_API_KEY")
        if config.llm.provider == "vllm" and not os.getenv("VLLM_ENDPOINT"):
            config.llm.provider = "openai"
    
    if os.getenv("OPENAI_BASE_URL"):
        config.llm.openai_base_url = os.getenv("OPENAI_BASE_URL")
    
    # General LLM settings
    if os.getenv("LLM_MODEL"):
        config.llm.model = os.getenv("LLM_MODEL")
    
    if os.getenv("LLM_TEMPERATURE"):
        config.llm.temperature = float(os.getenv("LLM_TEMPERATURE"))
    
    if os.getenv("LLM_MAX_TOKENS"):
        config.llm.max_tokens = int(os.getenv("LLM_MAX_TOKENS"))
    
    if os.getenv("LLM_TIMEOUT"):
        config.llm.timeout = int(os.getenv("LLM_TIMEOUT"))
    
    if os.getenv("LLM_BASE_URL"):
        config.llm.base_url = os.getenv("LLM_BASE_URL")
    
    return config


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config
