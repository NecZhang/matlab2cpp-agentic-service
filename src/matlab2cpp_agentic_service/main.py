"""Main entry point for the MATLAB to C++ conversion agent."""

import click
from pathlib import Path
from typing import Optional
from loguru import logger

from .services.matlab2cpp_orchestrator import MATLAB2CPPOrchestrator, ConversionRequest
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
        # Initialize orchestrator
        orchestrator = MATLAB2CPPOrchestrator()
        
        # Create conversion request
        request = ConversionRequest(
            matlab_path=str(matlab_path),
            project_name=matlab_path.stem,
            output_dir=str(output),
            max_optimization_turns=2,
            target_quality_score=7.0,
            include_tests=validate,
            cpp_standard="C++17"
        )
        
        # Run conversion
        result = orchestrator.convert_project(request)
        
        # Display results
        if result.status == "completed":
            logger.success("Conversion completed successfully!")
            logger.info(f"C++ project created at: {output}")
        else:
            logger.error(f"Conversion failed: {result.error_message}")
            raise click.ClickException(f"Conversion failed: {result.error_message}")
        
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
        from .agents.matlab_content_analyzer import MATLABContentAnalyzerAgent
        from .utils.config import get_config
        from .tools.llm_client import create_llm_client
        
        # Initialize analyzer
        config = get_config()
        analyzer = MATLABContentAnalyzerAgent(config.llm)
        
        # Analyze project
        analysis_result = analyzer.analyze_matlab_content(matlab_path)
        
        # Display results
        logger.info("Analysis Results:")
        logger.info(f"  Files Analyzed: {analysis_result.get('files_analyzed', 0)}")
        logger.info(f"  Total Functions: {analysis_result.get('total_functions', 0)}")
        logger.info(f"  Total Dependencies: {analysis_result.get('total_dependencies', 0)}")
        logger.info(f"  Complexity: {analysis_result.get('complexity_assessment', 'Unknown')}")
        
        if detailed:
            matlab_packages = analysis_result.get('matlab_packages_used', [])
            if matlab_packages:
                logger.info(f"  MATLAB Packages: {', '.join(matlab_packages)}")
            
            matlab_functions = analysis_result.get('matlab_functions_used', [])
            if matlab_functions:
                logger.info(f"  MATLAB Functions: {', '.join(matlab_functions[:10])}")  # Show first 10
                if len(matlab_functions) > 10:
                    logger.info(f"    ... and {len(matlab_functions) - 10} more")
            
            file_analyses = analysis_result.get('file_analyses', [])
            if file_analyses:
                logger.info("  File Analysis:")
                for file_analysis in file_analyses[:5]:  # Show first 5 files
                    file_path = file_analysis.get('file_path', 'Unknown')
                    logger.info(f"    - {file_path}")
                if len(file_analyses) > 5:
                    logger.info(f"    ... and {len(file_analyses) - 5} more files")
        
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
