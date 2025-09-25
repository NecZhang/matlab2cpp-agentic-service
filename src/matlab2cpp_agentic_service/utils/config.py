"""Configuration management for the MATLAB to C++ conversion system."""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict
import yaml
from dotenv import load_dotenv

def load_env_file(env_path: Optional[Path] = None) -> None:
    """Load environment variables from .env file."""
    if env_path is None:
        # Look for .env file in project root
        project_root = Path(__file__).parent.parent.parent.parent
        env_path = project_root / ".env"
    
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded environment variables from {env_path}")
    else:
        print(f"⚠️  No .env file found at {env_path}, using system environment variables")

# Load environment variables from .env file
load_env_file()


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
    cpp_standard: str = Field(default="C++17", description="C++ standard to use")
    include_optimization: bool = Field(default=True, description="Include optimization suggestions")
    generate_tests: bool = Field(default=True, description="Generate unit tests")
    create_documentation: bool = Field(default=True, description="Generate documentation")
    max_optimization_turns: int = Field(default=2, description="Maximum optimization turns")
    target_quality_score: float = Field(default=7.0, description="Target quality score (0-10)")


class LoggingConfig(BaseModel):
    """Configuration for logging."""
    log_level: str = Field(default="INFO", description="Log level: DEBUG, INFO, WARNING, ERROR")
    enable_console: bool = Field(default=True, description="Enable console logging")
    log_file: Optional[Path] = Field(default=None, description="Log file path")


class ProjectConfig(BaseModel):
    """Configuration for project settings."""
    default_output_dir: Path = Field(default=Path("./output"), description="Default output directory")
    cache_dir: Path = Field(default=Path("./.cache"), description="Cache directory")
    dev_mode: bool = Field(default=False, description="Enable development mode")
    enable_profiling: bool = Field(default=False, description="Enable performance profiling")


class Config(BaseModel):
    """Main configuration class."""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    conversion: ConversionConfig = Field(default_factory=ConversionConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    
    # Paths
    project_root: Path = Field(default=Path.cwd())
    templates_dir: Path = Field(default=Path("templates"))
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


def load_config(config_path: Optional[Path] = None, env_path: Optional[Path] = None) -> Config:
    """Load configuration from file or environment variables."""
    config = Config()
    
    # Load environment variables first
    if env_path:
        load_env_file(env_path)
    
    # Load from YAML file if provided
    if config_path and config_path.exists():
        with open(config_path, 'r') as f:
            file_config = yaml.safe_load(f)
            config = Config(**file_config)
    
    # Override with environment variables
    # LLM Provider
    if os.getenv("LLM_PROVIDER"):
        config.llm.provider = os.getenv("LLM_PROVIDER")
    
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
    
    if os.getenv("OPENAI_MODEL"):
        config.llm.model = os.getenv("OPENAI_MODEL")
    
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
    
    if os.getenv("LLM_API_KEY"):
        config.llm.api_key = os.getenv("LLM_API_KEY")
    
    # Analysis configuration
    if os.getenv("MAX_FILE_SIZE"):
        config.analysis.max_file_size = int(os.getenv("MAX_FILE_SIZE"))
    
    if os.getenv("CHUNK_SIZE"):
        config.analysis.chunk_size = int(os.getenv("CHUNK_SIZE"))
    
    if os.getenv("ANALYSIS_PASSES"):
        config.analysis.analysis_passes = int(os.getenv("ANALYSIS_PASSES"))
    
    if os.getenv("CONFIDENCE_THRESHOLD"):
        config.analysis.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD"))
    
    # Conversion configuration
    if os.getenv("DEFAULT_CPP_STANDARD"):
        config.conversion.cpp_standard = os.getenv("DEFAULT_CPP_STANDARD")
    
    if os.getenv("DEFAULT_MAX_TURNS"):
        config.conversion.max_optimization_turns = int(os.getenv("DEFAULT_MAX_TURNS"))
    
    if os.getenv("DEFAULT_TARGET_QUALITY"):
        config.conversion.target_quality_score = float(os.getenv("DEFAULT_TARGET_QUALITY"))
    
    if os.getenv("DEFAULT_INCLUDE_TESTS"):
        config.conversion.generate_tests = os.getenv("DEFAULT_INCLUDE_TESTS").lower() == "true"
    
    if os.getenv("INCLUDE_OPTIMIZATION"):
        config.conversion.include_optimization = os.getenv("INCLUDE_OPTIMIZATION").lower() == "true"
    
    if os.getenv("GENERATE_TESTS"):
        config.conversion.generate_tests = os.getenv("GENERATE_TESTS").lower() == "true"
    
    if os.getenv("CREATE_DOCUMENTATION"):
        config.conversion.create_documentation = os.getenv("CREATE_DOCUMENTATION").lower() == "true"
    
    # Logging configuration
    if os.getenv("LOG_LEVEL"):
        config.logging.log_level = os.getenv("LOG_LEVEL")
    
    if os.getenv("ENABLE_CONSOLE_LOGGING"):
        config.logging.enable_console = os.getenv("ENABLE_CONSOLE_LOGGING").lower() == "true"
    
    if os.getenv("LOG_FILE"):
        config.logging.log_file = Path(os.getenv("LOG_FILE"))
    
    # Project configuration
    if os.getenv("DEFAULT_OUTPUT_DIR"):
        config.project.default_output_dir = Path(os.getenv("DEFAULT_OUTPUT_DIR"))
    
    if os.getenv("CACHE_DIR"):
        config.project.cache_dir = Path(os.getenv("CACHE_DIR"))
    
    if os.getenv("DEV_MODE"):
        config.project.dev_mode = os.getenv("DEV_MODE").lower() == "true"
    
    if os.getenv("ENABLE_PROFILING"):
        config.project.enable_profiling = os.getenv("ENABLE_PROFILING").lower() == "true"
    
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
