#!/usr/bin/env python3
"""
🚀 MATLAB2C++ Agent - General Runner Script

A comprehensive interface for running the MATLAB to C++ conversion system.
This script provides easy access to all functionality including conversion,
analysis, validation, and testing.

Usage:
    python run.py convert <matlab_path> [options]
    python run.py analyze <matlab_path> [options]  
    python run.py validate <cpp_path> [options]
    python run.py test-llm [options]
    python run.py examples [options]
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Import the main components
from matlab2cpp_agentic_service import (
    MATLAB2CPPOrchestrator,
    ConversionRequest,
    ConversionResult,
    ConversionStatus
)
from matlab2cpp_agentic_service.utils.config import get_config, set_config, load_config
from matlab2cpp_agentic_service.utils.logger import setup_logger, get_logger
from matlab2cpp_agentic_service.tools.llm_client import test_llm_connection


class MATLAB2CPPRunner:
    """Main runner class for the MATLAB2C++ conversion system."""
    
    def __init__(self, verbose: bool = False, config_path: Optional[Path] = None, env_path: Optional[Path] = None):
        """Initialize the runner with configuration."""
        self.verbose = verbose
        
        # Load configuration first to get logging settings
        self.config = load_config(config_path, env_path)
        
        # Setup logging with configuration
        log_level = "DEBUG" if verbose else self.config.logging.log_level
        setup_logger(
            log_level=log_level,
            log_file=self.config.logging.log_file,
            enable_console=self.config.logging.enable_console
        )
        self.logger = get_logger("runner")
        
        # Set global config
        set_config(self.config)
        
        self.logger.info("🚀 MATLAB2C++ Agent Runner initialized")
        self.logger.info(f"   LLM Provider: {self.config.llm.provider}")
        self.logger.info(f"   Model: {self.config.llm.model}")
        if self.config.llm.vllm_endpoint:
            self.logger.info(f"   vLLM Endpoint: {self.config.llm.vllm_endpoint}")
    
    def setup_environment(self, llm_provider: str = None, **kwargs) -> bool:
        """
        Setup environment variables for LLM configuration.
        
        Args:
            llm_provider: LLM provider to use ('vllm', 'openai', etc.)
            **kwargs: Additional configuration parameters
            
        Returns:
            bool: True if setup successful, False otherwise
        """
        try:
            # Set LLM provider specific environment variables
            if llm_provider == 'vllm':
                if 'endpoint' in kwargs:
                    os.environ['VLLM_ENDPOINT'] = kwargs['endpoint']
                    os.environ['LLM_BASE_URL'] = f"{kwargs['endpoint']}/v1"
                if 'model' in kwargs:
                    os.environ['VLLM_MODEL_NAME'] = kwargs['model']
                if 'api_key' in kwargs:
                    os.environ['LLM_API_KEY'] = kwargs['api_key']
                os.environ['LLM_PROVIDER'] = 'vllm'
                
            elif llm_provider == 'openai':
                if 'api_key' in kwargs:
                    os.environ['OPENAI_API_KEY'] = kwargs['api_key']
                if 'model' in kwargs:
                    os.environ['LLM_MODEL'] = kwargs['model']
                os.environ['LLM_PROVIDER'] = 'openai'
            
            # Set general LLM parameters
            if 'max_tokens' in kwargs:
                os.environ['LLM_MAX_TOKENS'] = str(kwargs['max_tokens'])
            if 'timeout' in kwargs:
                os.environ['LLM_TIMEOUT'] = str(kwargs['timeout'])
            if 'temperature' in kwargs:
                os.environ['LLM_TEMPERATURE'] = str(kwargs['temperature'])
            
            # Reload configuration with new environment variables
            self.config = get_config()
            self.logger.info(f"✅ Environment configured for {llm_provider or 'default'} provider")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup environment: {e}")
            return False
    
    def test_llm_connection(self) -> bool:
        """Test LLM connection."""
        self.logger.info("🧪 Testing LLM connection...")
        try:
            success = test_llm_connection(self.config.llm)
            if success:
                self.logger.success("✅ LLM connection test passed!")
                return True
            else:
                self.logger.error("❌ LLM connection test failed!")
                return False
        except Exception as e:
            self.logger.error(f"❌ LLM test failed: {e}")
            return False
    
    def analyze_matlab_project(self, matlab_path: Path, detailed: bool = False) -> Dict[str, Any]:
        """
        Analyze a MATLAB project without converting.
        
        Args:
            matlab_path: Path to MATLAB file or project directory
            detailed: Whether to show detailed analysis
            
        Returns:
            Dict containing analysis results
        """
        self.logger.info(f"🔍 Analyzing MATLAB project: {matlab_path}")
        
        try:
            from matlab2cpp_agentic_service.agents.matlab_content_analyzer import MATLABContentAnalyzerAgent
            from matlab2cpp_agentic_service.tools.llm_client import create_llm_client
            
            # Initialize analyzer
            llm_client = create_llm_client(self.config.llm)
            analyzer = MATLABContentAnalyzerAgent(llm_client)
            
            # Analyze project
            result = analyzer.analyze_matlab_content(matlab_path)
            
            # Display results
            self.logger.info("📊 Analysis Results:")
            self.logger.info(f"   Files Analyzed: {result.get('files_analyzed', 0)}")
            self.logger.info(f"   Total Functions: {result.get('total_functions', 0)}")
            self.logger.info(f"   Total Dependencies: {result.get('total_dependencies', 0)}")
            self.logger.info(f"   Complexity: {result.get('complexity_assessment', 'Unknown')}")
            
            if detailed:
                self._display_detailed_analysis(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Analysis failed: {e}")
            raise
    
    def convert_matlab_project(
        self,
        matlab_path: Path,
        project_name: str,
        output_dir: Optional[Path] = None,
        max_turns: Optional[int] = None,
        target_quality: Optional[float] = None,
        cpp_standard: Optional[str] = None,
        include_tests: Optional[bool] = None
    ) -> ConversionResult:
        """
        Convert MATLAB project to C++.
        
        Args:
            matlab_path: Path to MATLAB file or project directory
            project_name: Name for the generated C++ project
            output_dir: Output directory for C++ files
            max_turns: Maximum optimization turns (uses config default if None)
            target_quality: Target quality score (0-10, uses config default if None)
            cpp_standard: C++ standard to use (uses config default if None)
            include_tests: Whether to include unit tests (uses config default if None)
            
        Returns:
            ConversionResult with conversion status and details
        """
        # Use configuration defaults if not provided
        output_dir = output_dir or self.config.project.default_output_dir
        max_turns = max_turns or self.config.conversion.max_optimization_turns
        target_quality = target_quality or self.config.conversion.target_quality_score
        cpp_standard = cpp_standard or self.config.conversion.cpp_standard
        include_tests = include_tests if include_tests is not None else self.config.conversion.generate_tests
        
        self.logger.info(f"🚀 Converting MATLAB project: {matlab_path}")
        self.logger.info(f"   Project Name: {project_name}")
        self.logger.info(f"   Output Directory: {output_dir}")
        self.logger.info(f"   Max Turns: {max_turns}")
        self.logger.info(f"   Target Quality: {target_quality}/10")
        self.logger.info(f"   C++ Standard: {cpp_standard}")
        self.logger.info(f"   Include Tests: {include_tests}")
        
        try:
            # Initialize orchestrator
            orchestrator = MATLAB2CPPOrchestrator()
            
            # Create conversion request
            request = ConversionRequest(
                matlab_path=str(matlab_path),
                project_name=project_name,
                output_dir=str(output_dir),
                max_optimization_turns=max_turns,
                target_quality_score=target_quality,
                include_tests=include_tests,
                cpp_standard=cpp_standard
            )
            
            # Perform conversion
            result = orchestrator.convert_project(request)
            
            # Display results
            self._display_conversion_result(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Conversion failed: {e}")
            raise
    
    def validate_cpp_project(self, cpp_path: Path) -> Dict[str, Any]:
        """
        Validate a converted C++ project.
        
        Args:
            cpp_path: Path to C++ project directory
            
        Returns:
            Dict containing validation results
        """
        self.logger.info(f"✅ Validating C++ project: {cpp_path}")
        
        try:
            from matlab2cpp_agentic_service.agents.validator import ValidatorAgent
            
            # Initialize validator
            validator = ValidatorAgent()
            
            # Validate project
            results = validator.validate_project(cpp_path)
            
            # Display results
            if results.get("compilation_success", False):
                self.logger.success("✅ C++ project compiles successfully!")
            else:
                self.logger.error("❌ C++ project compilation failed:")
                for error in results.get("compilation_errors", []):
                    self.logger.error(f"   - {error}")
            
            if results.get("tests_passed", False):
                self.logger.success("✅ All tests passed!")
            else:
                self.logger.warning("⚠️  Some tests failed:")
                for test in results.get("failed_tests", []):
                    self.logger.warning(f"   - {test}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Validation failed: {e}")
            raise
    
    def show_examples(self) -> None:
        """Show available examples and how to run them."""
        examples_dir = project_root / "examples"
        if not examples_dir.exists():
            self.logger.warning("📁 No examples directory found")
            return
        
        self.logger.info("📚 Available Examples:")
        self.logger.info("=" * 50)
        
        # Find MATLAB samples
        matlab_samples = examples_dir / "matlab_samples"
        if matlab_samples.exists():
            matlab_files = list(matlab_samples.glob("*.m"))
            if matlab_files:
                self.logger.info(f"🔬 MATLAB Samples ({len(matlab_files)}):")
                for file_path in matlab_files:
                    self.logger.info(f"   • {file_path.name}")
                    self.logger.info(f"     python run.py convert {file_path} {file_path.stem}_cpp")
        
        # Show output examples
        output_dir = project_root / "output"
        if output_dir.exists():
            output_projects = [d for d in output_dir.iterdir() if d.is_dir()]
            if output_projects:
                self.logger.info(f"\n📁 Generated C++ Projects ({len(output_projects)}):")
                for project_dir in output_projects:
                    self.logger.info(f"   • {project_dir.name}")
                    self.logger.info(f"     python run.py validate {project_dir}")
    
    def _display_detailed_analysis(self, result: Dict[str, Any]) -> None:
        """Display detailed analysis results."""
        matlab_packages = result.get('matlab_packages_used', [])
        if matlab_packages:
            self.logger.info(f"   MATLAB Packages: {', '.join(matlab_packages)}")
        
        matlab_functions = result.get('matlab_functions_used', [])
        if matlab_functions:
            self.logger.info(f"   MATLAB Functions: {', '.join(matlab_functions[:10])}")
            if len(matlab_functions) > 10:
                self.logger.info(f"     ... and {len(matlab_functions) - 10} more")
        
        file_analyses = result.get('file_analyses', [])
        if file_analyses:
            self.logger.info("   File Analysis:")
            for file_analysis in file_analyses[:5]:
                file_path = file_analysis.get('file_path', 'Unknown')
                self.logger.info(f"     - {file_path}")
            if len(file_analyses) > 5:
                self.logger.info(f"     ... and {len(file_analyses) - 5} more files")
    
    def _display_conversion_result(self, result: ConversionResult) -> None:
        """Display conversion results."""
        self.logger.info("\n📋 CONVERSION RESULTS")
        self.logger.info("=" * 50)
        
        if result.status == ConversionStatus.COMPLETED:
            self.logger.success("✅ Conversion completed successfully!")
        elif result.status == ConversionStatus.FAILED:
            self.logger.error("❌ Conversion failed!")
            if result.error_message:
                self.logger.error(f"   Error: {result.error_message}")
            return
        else:
            self.logger.warning(f"⚠️  Conversion status: {result.status.value}")
        
        self.logger.info(f"🏗️  Project: {result.project_name}")
        self.logger.info(f"⏱️  Total Time: {result.total_processing_time:.1f}s")
        self.logger.info(f"📊 Quality Scores:")
        self.logger.info(f"   Original: {result.original_score:.1f}/10")
        self.logger.info(f"   Final: {result.final_score:.1f}/10")
        self.logger.info(f"   Improvement: {result.final_score - result.original_score:+.1f} points")
        self.logger.info(f"🔄 Optimization Turns: {result.improvement_turns}")
        
        if result.generated_files:
            self.logger.info(f"\n📁 Generated Files ({len(result.generated_files)}):")
            for file_path in result.generated_files:
                self.logger.info(f"   • {file_path}")
        
        if result.assessment_reports:
            self.logger.info(f"\n📋 Assessment Reports ({len(result.assessment_reports)}):")
            for report_path in result.assessment_reports:
                self.logger.info(f"   • {report_path}")
        
        # Quality assessment
        if result.final_score >= 8.0:
            self.logger.success("\n🎉 Excellent quality! Ready for production use.")
        elif result.final_score >= 6.0:
            self.logger.success("\n✅ Good quality! Suitable for most use cases.")
        elif result.final_score >= 4.0:
            self.logger.warning("\n⚠️  Moderate quality. Consider manual review.")
        else:
            self.logger.error("\n❌ Low quality. Manual intervention recommended.")


def main():
    """Main entry point for the runner script."""
    parser = argparse.ArgumentParser(
        description="🚀 MATLAB2C++ Agent - General Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert MATLAB project
  python run.py convert examples/matlab_samples/arma_filter.m my_project
  
  # Convert with custom options
  python run.py convert examples/matlab_samples/arma_filter.m my_project \\
    --output-dir ./output \\
    --max-turns 3 \\
    --target-quality 8.0 \\
    --cpp-standard C++20
  
  # Analyze MATLAB project
  python run.py analyze examples/matlab_samples/arma_filter.m --detailed
  
  # Validate C++ project
  python run.py validate output/my_project
  
  # Test LLM connection
  python run.py test-llm
  
  # Setup vLLM environment
  python run.py test-llm --llm-provider vllm \\
    --endpoint http://192.168.6.10:8002 \\
    --model Qwen/Qwen3-32B-FP8
  
  # Show examples
  python run.py examples
        """
    )
    
    # Global options
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--config', '-c', type=Path, help='Configuration file path (YAML)')
    parser.add_argument('--env', '-e', type=Path, help='Environment file path (.env)')
    parser.add_argument('--json-output', action='store_true', help='Output results in JSON format')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert MATLAB project to C++')
    convert_parser.add_argument('matlab_path', type=Path, help='Path to MATLAB file or project')
    convert_parser.add_argument('project_name', help='Name for the generated C++ project')
    convert_parser.add_argument('--output-dir', '-o', type=Path, help='Output directory')
    convert_parser.add_argument('--max-turns', type=int, default=2, help='Max optimization turns')
    convert_parser.add_argument('--target-quality', type=float, default=7.0, help='Target quality score')
    convert_parser.add_argument('--cpp-standard', default='C++17', help='C++ standard')
    convert_parser.add_argument('--include-tests', action='store_true', default=True, help='Include tests')
    convert_parser.add_argument('--no-tests', dest='include_tests', action='store_false', help='Skip tests')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze MATLAB project')
    analyze_parser.add_argument('matlab_path', type=Path, help='Path to MATLAB file or project')
    analyze_parser.add_argument('--detailed', action='store_true', help='Show detailed analysis')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate C++ project')
    validate_parser.add_argument('cpp_path', type=Path, help='Path to C++ project')
    
    # Test LLM command
    test_llm_parser = subparsers.add_parser('test-llm', help='Test LLM connection')
    test_llm_parser.add_argument('--llm-provider', choices=['vllm', 'openai'], help='LLM provider')
    test_llm_parser.add_argument('--endpoint', help='LLM endpoint URL')
    test_llm_parser.add_argument('--model', help='Model name')
    test_llm_parser.add_argument('--api-key', help='API key')
    test_llm_parser.add_argument('--max-tokens', type=int, help='Max tokens')
    test_llm_parser.add_argument('--timeout', type=int, help='Timeout in seconds')
    test_llm_parser.add_argument('--temperature', type=float, help='Temperature')
    
    # Examples command
    examples_parser = subparsers.add_parser('examples', help='Show available examples')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        # Initialize runner
        runner = MATLAB2CPPRunner(verbose=args.verbose, config_path=args.config, env_path=args.env)
        
        # Handle LLM setup for test-llm command
        if args.command == 'test-llm' and args.llm_provider:
            setup_kwargs = {}
            if args.endpoint:
                setup_kwargs['endpoint'] = args.endpoint
            if args.model:
                setup_kwargs['model'] = args.model
            if args.api_key:
                setup_kwargs['api_key'] = args.api_key
            if args.max_tokens:
                setup_kwargs['max_tokens'] = args.max_tokens
            if args.timeout:
                setup_kwargs['timeout'] = args.timeout
            if args.temperature:
                setup_kwargs['temperature'] = args.temperature
            
            if not runner.setup_environment(args.llm_provider, **setup_kwargs):
                sys.exit(1)
        
        # Execute command
        if args.command == 'convert':
            result = runner.convert_matlab_project(
                matlab_path=args.matlab_path,
                project_name=args.project_name,
                output_dir=args.output_dir,
                max_turns=args.max_turns,
                target_quality=args.target_quality,
                cpp_standard=args.cpp_standard,
                include_tests=args.include_tests
            )
            
            if args.json_output:
                print(json.dumps({
                    'status': result.status.value,
                    'project_name': result.project_name,
                    'original_score': result.original_score,
                    'final_score': result.final_score,
                    'improvement_turns': result.improvement_turns,
                    'generated_files': result.generated_files,
                    'assessment_reports': result.assessment_reports,
                    'total_processing_time': result.total_processing_time,
                    'error_message': result.error_message
                }, indent=2))
            
            sys.exit(0 if result.status == ConversionStatus.COMPLETED else 1)
        
        elif args.command == 'analyze':
            result = runner.analyze_matlab_project(args.matlab_path, args.detailed)
            
            if args.json_output:
                print(json.dumps(result, indent=2))
        
        elif args.command == 'validate':
            result = runner.validate_cpp_project(args.cpp_path)
            
            if args.json_output:
                print(json.dumps(result, indent=2))
            
            sys.exit(0 if result.get('compilation_success', False) else 1)
        
        elif args.command == 'test-llm':
            success = runner.test_llm_connection()
            sys.exit(0 if success else 1)
        
        elif args.command == 'examples':
            runner.show_examples()
    
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
