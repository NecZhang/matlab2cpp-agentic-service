"""Main entry point for the MATLAB to C++ conversion agent."""

import click
from pathlib import Path
from typing import Optional
from loguru import logger

from .workflows.conversion_workflow import ConversionWorkflow
from .utils.config import get_config
from .utils.logger import setup_logger, get_logger


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
def cli(verbose: bool, config: Optional[Path]):
    """MATLAB to C++ Conversion Agent - Convert MATLAB projects to C++ with validation."""
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logger(log_level=log_level)
    
    # Load configuration
    if config:
        from .utils.config import load_config
        cfg = load_config(Path(config))
        from .utils.config import set_config
        set_config(cfg)
    
    logger.info("MATLAB to C++ Conversion Agent started")


@cli.command()
@click.argument('matlab_path', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o', type=click.Path(path_type=Path), help='Output directory for C++ project')
@click.option('--validate', is_flag=True, default=True, help='Run validation tests')
@click.option('--incremental', is_flag=True, help='Convert incrementally (function by function)')
def convert(matlab_path: Path, output: Optional[Path], validate: bool, incremental: bool):
    """Convert a MATLAB project to C++."""
    logger = get_logger("main")
    
    # Set default output path if not provided
    if not output:
        output = matlab_path.parent / f"{matlab_path.name}_cpp"
    
    logger.info(f"Converting MATLAB project: {matlab_path}")
    logger.info(f"Output directory: {output}")
    
    try:
        # Initialize workflow
        workflow = ConversionWorkflow()
        
        # Run conversion
        result = workflow.convert_project(matlab_path, output)
        
        # Display results
        if result["errors"]:
            logger.error("Conversion completed with errors:")
            for error in result["errors"]:
                logger.error(f"  - {error}")
        else:
            logger.success("Conversion completed successfully!")
        
        if result["warnings"]:
            logger.warning("Conversion completed with warnings:")
            for warning in result["warnings"]:
                logger.warning(f"  - {warning}")
        
        logger.info(f"C++ project created at: {output}")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise click.ClickException(f"Conversion failed: {e}")


@cli.command()
@click.argument('matlab_path', type=click.Path(exists=True, path_type=Path))
@click.option('--detailed', is_flag=True, help='Show detailed analysis')
def analyze(matlab_path: Path, detailed: bool):
    """Analyze a MATLAB project without converting."""
    logger = get_logger("main")
    
    logger.info(f"Analyzing MATLAB project: {matlab_path}")
    
    try:
        from .agents.matlab_analyzer import MATLABAnalyzerAgent
        
        # Initialize analyzer
        analyzer = MATLABAnalyzerAgent()
        
        # Analyze project
        understanding = analyzer.analyze_project(matlab_path)
        
        # Display results
        logger.info("Analysis Results:")
        logger.info(f"  Main Purpose: {understanding.main_purpose}")
        logger.info(f"  Domain: {understanding.domain}")
        logger.info(f"  Complexity: {understanding.complexity_level}")
        logger.info(f"  Confidence: {understanding.confidence:.2f}")
        
        if detailed:
            logger.info(f"  Key Algorithms: {', '.join(understanding.key_algorithms)}")
            logger.info(f"  Architecture: {understanding.architecture}")
            
            if understanding.conversion_challenges:
                logger.info("  Conversion Challenges:")
                for challenge in understanding.conversion_challenges:
                    logger.info(f"    - {challenge}")
            
            if understanding.recommendations:
                logger.info("  Recommendations:")
                for rec in understanding.recommendations:
                    logger.info(f"    - {rec}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise click.ClickException(f"Analysis failed: {e}")


@cli.command()
@click.argument('cpp_path', type=click.Path(exists=True, path_type=Path))
def validate(cpp_path: Path):
    """Validate a converted C++ project."""
    logger = get_logger("main")
    
    logger.info(f"Validating C++ project: {cpp_path}")
    
    try:
        from .agents.validator import ValidatorAgent
        
        # Initialize validator
        validator = ValidatorAgent()
        
        # Validate project
        results = validator.validate_project(cpp_path)
        
        # Display results
        if results.get("compilation_success", False):
            logger.success("C++ project compiles successfully!")
        else:
            logger.error("C++ project compilation failed:")
            for error in results.get("compilation_errors", []):
                logger.error(f"  - {error}")
        
        if results.get("tests_passed", False):
            logger.success("All tests passed!")
        else:
            logger.warning("Some tests failed:")
            for test in results.get("failed_tests", []):
                logger.warning(f"  - {test}")
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise click.ClickException(f"Validation failed: {e}")


@cli.command()
def test_llm():
    """Test LLM connection (vLLM or OpenAI)."""
    logger = get_logger("main")
    
    logger.info("Testing LLM connection...")
    
    try:
        from .tools.llm_client import test_llm_connection
        from .utils.config import get_config
        
        config = get_config()
        
        logger.info(f"Testing {config.llm.provider} connection...")
        logger.info(f"Endpoint: {config.llm.vllm_endpoint if config.llm.provider == 'vllm' else 'OpenAI API'}")
        logger.info(f"Model: {config.llm.model}")
        
        success = test_llm_connection(config.llm)
        
        if success:
            logger.success("✅ LLM connection test passed!")
        else:
            logger.error("❌ LLM connection test failed!")
            logger.info("Check your configuration:")
            logger.info("  - For vLLM: Set VLLM_ENDPOINT and VLLM_API_KEY")
            logger.info("  - For OpenAI: Set OPENAI_API_KEY")
            raise click.ClickException("LLM connection test failed")
        
    except Exception as e:
        logger.error(f"LLM test failed: {e}")
        raise click.ClickException(f"LLM test failed: {e}")


if __name__ == "__main__":
    cli()
