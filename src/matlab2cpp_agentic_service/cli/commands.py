"""
CLI Commands for Streamlined MATLAB2C++ Agentic Service

This module provides command-line commands for the streamlined MATLAB2C++ conversion service
with 5 core agents: MATLABAnalyzer, ConversionPlanner, CppGenerator, QualityAssessor, and ProjectManager.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import click
import asyncio
from loguru import logger

from ..core.orchestrators import NativeLangGraphMATLAB2CPPOrchestrator
from ..infrastructure.state import ConversionRequest
from ..utils.config import get_config
from ..infrastructure.tools.llm_client import test_llm_connection


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, verbose):
    """
    MATLAB2C++ Streamlined Agentic Service CLI
    
    A streamlined agentic service for converting MATLAB projects to C++
    using 5 core agents with intelligent optimization.
    """
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    
    if verbose:
        logger.add(lambda msg: click.echo(msg, err=True), level="DEBUG")


@cli.command()
@click.argument('matlab_path', type=click.Path(exists=True))
@click.option('--project-name', '-p', required=True, help='Project name')
@click.option('--output-dir', '-o', default='output', help='Output directory')
@click.option('--max-turns', '-t', default=2, help='Maximum optimization turns')
@click.option('--quality-score', '-q', default=7.0, help='Target quality score')
@click.option('--conversion-mode', '-m', default='result-focused', 
              type=click.Choice(['faithful', 'result-focused']), help='Conversion mode')
@click.option('--build-system', '-b', default='gcc',
              type=click.Choice(['gcc', 'cmake']), help='Build system (gcc or cmake)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def convert(matlab_path: str, project_name: str, output_dir: str, 
            max_turns: int, quality_score: float, conversion_mode: str, 
            build_system: str, verbose: bool):
    """
    Convert MATLAB code to C++ using streamlined agentic workflow.
    
    MATLAB_PATH: Path to MATLAB file or directory
    """
    try:
        # Validate inputs
        matlab_path_obj = Path(matlab_path)
        if not matlab_path_obj.exists():
            click.echo(f"❌ Error: MATLAB path does not exist: {matlab_path}", err=True)
            return 1
        
        if not matlab_path_obj.suffix == '.m' and not matlab_path_obj.is_dir():
            click.echo(f"❌ Error: Invalid MATLAB path. Must be .m file or directory: {matlab_path}", err=True)
            return 1
        
        # Create conversion request
        request = ConversionRequest(
            matlab_path=matlab_path_obj,
            project_name=project_name,
            output_dir=Path(output_dir),
            max_optimization_turns=max_turns,
            target_quality_score=quality_score,
            conversion_mode=conversion_mode,
            build_system=build_system
        )
        
        # Initialize orchestrator
        click.echo("🚀 Initializing streamlined agentic workflow...")
        orchestrator = NativeLangGraphMATLAB2CPPOrchestrator()
        
        # Run conversion
        click.echo(f"🔄 Starting conversion: {matlab_path}")
        click.echo(f"📁 Project: {project_name}")
        click.echo(f"📂 Output: {output_dir}")
        click.echo(f"🔄 Max turns: {max_turns}")
        click.echo(f"🎯 Quality target: {quality_score}")
        click.echo(f"🔧 Mode: {conversion_mode}")
        click.echo(f"🏗️  Build system: {build_system}")
        
        result = asyncio.run(orchestrator.convert_project(request))
        
        # Display results
        click.echo("\n" + "="*60)
        click.echo("📊 CONVERSION RESULTS")
        click.echo("="*60)
        
        if result.status.value == "completed":
            click.echo("✅ Conversion completed successfully!")
            click.echo(f"📊 Quality score: {result.final_score:.1f}/10")
            click.echo(f"📁 Generated files: {len(result.generated_files)}")
            click.echo(f"⏱️  Processing time: {result.total_processing_time:.1f}s")
            
            if result.generated_files:
                click.echo("\n📄 Generated files:")
                for file_path in result.generated_files:
                    click.echo(f"  - {file_path}")
            
            # Show output directory
            click.echo(f"\n📂 Output directory: {result.output_dir}")
            click.echo(f"💡 Check the generated C++ files in: {result.output_dir}")
            
            # Show conversion report if available
            if result.report_path:
                click.echo(f"📋 Conversion report: {result.report_path}")
                click.echo(f"📖 Detailed analysis available in: {result.report_path}")
            
        else:
            click.echo("❌ Conversion failed!")
            click.echo(f"📊 Quality score: {result.final_score:.1f}/10")
            click.echo(f"📁 Generated files: {len(result.generated_files)}")
            if result.error_message:
                click.echo(f"❌ Error: {result.error_message}")
        
        return 0 if result.status.value == "completed" else 1
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        if verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        return 1


@cli.command()
@click.argument('matlab_path', type=click.Path(exists=True))
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def analyze(matlab_path: str, verbose: bool):
    """
    Analyze MATLAB code structure and dependencies.
    
    MATLAB_PATH: Path to MATLAB file or directory
    """
    try:
        from ..core.agents.streamlined.matlab_analyzer import MATLABAnalyzer
        from ..core.agents.base.langgraph_agent import AgentConfig
        from ..infrastructure.tools.llm_client import create_llm_client
        from ..utils.config import get_config
        
        # Initialize analyzer
        config = get_config()
        llm_client = create_llm_client(config.llm)
        agent_config = AgentConfig(name='cli_analyzer', description='CLI analyzer')
        analyzer = MATLABAnalyzer(agent_config, llm_client)
        
        # Create test state
        test_state = {
            'matlab_path': str(matlab_path),
            'project_name': Path(matlab_path).stem,
            'is_multi_file': False
        }
        
        click.echo(f"🔍 Analyzing MATLAB code: {matlab_path}")
        
        # Run analysis
        result = asyncio.run(analyzer.analyze_project(Path(matlab_path), test_state))
        
        # Display results
        matlab_analysis = result.get('matlab_analysis', {})
        file_analyses = matlab_analysis.get('file_analyses', [])
        function_call_tree = matlab_analysis.get('function_call_tree', {})
        is_multi_file = matlab_analysis.get('is_multi_file', False)
        
        click.echo("\n" + "="*60)
        click.echo("📊 ANALYSIS RESULTS")
        click.echo("="*60)
        click.echo(f"📁 Files analyzed: {len(file_analyses)}")
        click.echo(f"🔧 Functions found: {len(function_call_tree)}")
        click.echo(f"📂 Multi-file project: {is_multi_file}")
        
        if file_analyses:
            click.echo("\n📄 File details:")
            for i, analysis in enumerate(file_analyses):
                file_name = analysis.get('file_name', 'unknown')
                functions = analysis.get('functions', [])
                lines_of_code = analysis.get('lines_of_code', 0)
                click.echo(f"  {i+1}. {file_name}")
                click.echo(f"     - Functions: {len(functions)}")
                click.echo(f"     - Lines of code: {lines_of_code}")
                if functions:
                    click.echo(f"     - Function names: {', '.join(functions)}")
        
        if function_call_tree:
            click.echo("\n🔗 Function call tree:")
            for func_name, calls in function_call_tree.items():
                if calls:
                    click.echo(f"  {func_name} -> {', '.join(calls)}")
                else:
                    click.echo(f"  {func_name} -> (no calls)")
        
        return 0
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        if verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        return 1


@cli.command()
@click.option('--provider', type=click.Choice(['vllm', 'openai']), help='LLM provider')
@click.option('--endpoint', help='LLM endpoint URL')
@click.option('--model', help='Model name')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def test_llm(provider: Optional[str], endpoint: Optional[str], model: Optional[str], verbose: bool):
    """
    Test LLM connection and configuration.
    """
    try:
        click.echo("🧪 Testing LLM connection...")
        
        # Test LLM connection
        success, message = test_llm_connection()
        
        if success:
            click.echo("✅ LLM connection successful!")
            click.echo(f"📝 Message: {message}")
        else:
            click.echo("❌ LLM connection failed!")
            click.echo(f"📝 Error: {message}")
            return 1
        
        return 0
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        if verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        return 1


@cli.command()
def examples():
    """Show available MATLAB examples."""
    try:
        examples_dir = Path('examples/matlab_samples')
        if not examples_dir.exists():
            click.echo("❌ Examples directory not found: examples/matlab_samples")
            return 1
        
        click.echo("📁 Available MATLAB examples:")
        click.echo("="*40)
        
        # List single files
        single_files = list(examples_dir.glob('*.m'))
        if single_files:
            click.echo("\n📄 Single files:")
            for file_path in single_files:
                click.echo(f"  - {file_path.name}")
        
        # List directories
        directories = [d for d in examples_dir.iterdir() if d.is_dir()]
        if directories:
            click.echo("\n📂 Multi-file projects:")
            for dir_path in directories:
                matlab_files = list(dir_path.glob('*.m'))
                click.echo(f"  - {dir_path.name}/ ({len(matlab_files)} files)")
                for file_path in matlab_files:
                    click.echo(f"    - {file_path.name}")
        
        click.echo(f"\n💡 Usage examples:")
        click.echo(f"  # Convert single file:")
        click.echo(f"  matlab2cpp convert examples/matlab_samples/arma_filter.m -p arma_filter_test")
        click.echo(f"  # Convert multi-file project:")
        click.echo(f"  matlab2cpp convert examples/matlab_samples/skeleton_vessel -p skeleton_test")
        
        return 0
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        return 1


@cli.command()
def status():
    """Check system status and configuration."""
    try:
        click.echo("🔍 Checking system status...")
        click.echo("="*40)
        
        # Check configuration
        config = get_config()
        click.echo("✅ Configuration loaded")
        
        # Check LLM connection
        success, message = test_llm_connection()
        if success:
            click.echo("✅ LLM connection: OK")
        else:
            click.echo(f"❌ LLM connection: FAILED - {message}")
        
        # Check examples directory
        examples_dir = Path('examples/matlab_samples')
        if examples_dir.exists():
            click.echo("✅ Examples directory: OK")
        else:
            click.echo("❌ Examples directory: NOT FOUND")
        
        # Check output directory
        output_dir = Path('output')
        if output_dir.exists():
            click.echo("✅ Output directory: OK")
        else:
            click.echo("⚠️  Output directory: NOT FOUND (will be created)")
        
        click.echo("\n🎯 System ready for MATLAB to C++ conversion!")
        
        return 0
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        return 1


@cli.command()
def version():
    """Show version information."""
    try:
        from .. import __version__, __author__
        
        click.echo("MATLAB2C++ Streamlined Agentic Service")
        click.echo("="*40)
        click.echo(f"Version: {__version__}")
        click.echo(f"Author: {__author__}")
        click.echo(f"Architecture: 5 Streamlined Agents")
        click.echo(f"  - MATLABAnalyzer")
        click.echo(f"  - ConversionPlanner") 
        click.echo(f"  - CppGenerator")
        click.echo(f"  - QualityAssessor")
        click.echo(f"  - ProjectManager")
        
        return 0
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        return 1


if __name__ == '__main__':
    cli()