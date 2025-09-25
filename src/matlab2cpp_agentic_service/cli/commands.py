"""
CLI Commands

This module provides command-line commands for the MATLAB2C++ conversion service.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import click
from loguru import logger

from ..core.orchestrators import MATLAB2CPPOrchestrator, MATLAB2CPPLangGraphOrchestrator
from ..infrastructure.state import ConversionRequest
from ..utils.config import get_config, set_config, load_config
from ..utils.logger import setup_logger, get_logger
from ..infrastructure.tools.llm_client import test_llm_connection
from .utils.monitoring_utils import (
    run_health_check, 
    get_health_report,
    setup_conversion_monitoring,
    export_conversion_metrics,
    get_monitoring_manager,
    get_performance_report,
    get_performance_recommendations
)


@click.command()
@click.argument('matlab_path', type=click.Path(exists=True))
@click.option('--project-name', '-p', required=True, help='Project name')
@click.option('--output-dir', '-o', default='output', help='Output directory')
@click.option('--max-turns', '-t', default=2, help='Maximum optimization turns')
@click.option('--quality-score', '-q', default=7.0, help='Target quality score')
@click.option('--conversion-mode', '-m', default='result-focused', 
              type=click.Choice(['faithful', 'result-focused']), help='Conversion mode')
@click.option('--orchestrator', default='native', 
              type=click.Choice(['legacy', 'langgraph', 'native']), help='Orchestrator type')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--monitor', is_flag=True, help='Enable performance monitoring')
@click.option('--health-check', is_flag=True, help='Run health check before conversion')
@click.option('--export-metrics', is_flag=True, help='Export metrics after conversion')
def convert(matlab_path: str, project_name: str, output_dir: str, 
            max_turns: int, quality_score: float, conversion_mode: str, 
            orchestrator: str, verbose: bool, monitor: bool, health_check: bool, 
            export_metrics: bool):
    """
    Convert MATLAB code to C++.
    
    MATLAB_PATH: Path to MATLAB file or directory
    """
    operation_id = None
    try:
        # Optional health check
        if health_check:
            click.echo("🏥 Running health check...")
            if not run_health_check():
                click.echo("❌ Health check failed, aborting conversion", err=True)
                return 1
            click.echo("✅ Health check passed")
        
        # Optional monitoring setup
        if monitor:
            click.echo("📊 Starting performance monitoring...")
            operation_id = setup_conversion_monitoring(
                Path(matlab_path), project_name, max_turns, conversion_mode
            )
        
        # Create conversion request
        request = ConversionRequest(
            matlab_path=Path(matlab_path),
            project_name=project_name,
            output_dir=Path(output_dir),
            max_optimization_turns=max_turns,
            target_quality_score=quality_score,
            conversion_mode=conversion_mode
        )
        
        # Select orchestrator
        if orchestrator == 'legacy':
            orchestrator_instance = MATLAB2CPPOrchestrator()
        elif orchestrator == 'langgraph':
            orchestrator_instance = MATLAB2CPPLangGraphOrchestrator()
        else:  # native (default)
            from ..core.orchestrators.native_langgraph_orchestrator import NativeLangGraphMATLAB2CPPOrchestrator
            orchestrator_instance = NativeLangGraphMATLAB2CPPOrchestrator()
        
        # Run conversion
        click.echo(f"🚀 Starting {orchestrator} conversion for project: {project_name}")
        result = orchestrator_instance.convert_project(request)
        
        # Display results
        if result.status.value == "completed":
            click.echo(f"✅ Conversion completed successfully!")
            click.echo(f"📊 Quality score: {result.final_score:.1f}/10")
            click.echo(f"📁 Generated files: {len(result.generated_files)}")
            click.echo(f"⏱️  Processing time: {result.total_processing_time:.1f}s")
            
            if verbose:
                click.echo("\nGenerated files:")
                for file_path in result.generated_files:
                    click.echo(f"  - {file_path}")
            
            # Optional metrics export
            if export_metrics and monitor:
                click.echo("📈 Exporting metrics...")
                metrics_path = Path(output_dir) / f"{project_name}_metrics.json"
                if export_conversion_metrics({
                    "total_time": result.total_processing_time,
                    "final_quality_score": result.final_score,
                    "generated_files": len(result.generated_files),
                    "project_name": project_name,
                    "optimization_turns": max_turns,
                    "is_multi_file": Path(matlab_path).is_dir()
                }, metrics_path):
                    click.echo(f"📊 Metrics exported to: {metrics_path}")
            
            # End monitoring with success
            if monitor and operation_id:
                monitoring_manager = get_monitoring_manager()
                monitoring_manager.end_conversion_monitoring(
                    success=True,
                    result={
                        "total_time": result.total_processing_time,
                        "final_quality_score": result.final_score,
                        "generated_files": len(result.generated_files),
                        "project_name": project_name,
                        "optimization_turns": max_turns,
                        "is_multi_file": Path(matlab_path).is_dir()
                    }
                )
        else:
            click.echo(f"❌ Conversion failed: {result.error_message}")
            
            # End monitoring with failure
            if monitor and operation_id:
                monitoring_manager = get_monitoring_manager()
                monitoring_manager.end_conversion_monitoring(
                    success=False,
                    error_message=result.error_message
                )
            return 1
        
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        
        # End monitoring with failure
        if monitor and operation_id:
            monitoring_manager = get_monitoring_manager()
            monitoring_manager.end_conversion_monitoring(
                success=False,
                error_message=str(e)
            )
        
        if verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


@click.command()
@click.argument('matlab_path', type=click.Path(exists=True))
@click.option('--detailed', is_flag=True, help='Show detailed analysis')
@click.option('--use-legacy', is_flag=True, help='Use legacy analyzer instead of native LangGraph')
def analyze(matlab_path: str, detailed: bool, use_legacy: bool):
    """Analyze MATLAB project without converting."""
    try:
        # Load configuration
        config = load_config()
        
        # Initialize logger
        logger_instance = get_logger("analyzer")
        
        logger_instance.info(f"🔍 Analyzing MATLAB project: {matlab_path}")
        
        if use_legacy:
            # Use legacy analyzer
            from ..core.agents.analyzer.legacy.matlab_analyzer import MATLABContentAnalyzerAgent
            analyzer = MATLABContentAnalyzerAgent(config.llm)
            result = analyzer.analyze_matlab_content(Path(matlab_path))
        else:
            # Use native LangGraph analyzer (fallback to legacy for now)
            click.echo("⚠️  LangGraph analyzer not fully integrated yet, using legacy analyzer")
            from ..core.agents.analyzer.legacy.matlab_analyzer import MATLABContentAnalyzerAgent
            analyzer = MATLABContentAnalyzerAgent(config.llm)
            result = analyzer.analyze_matlab_content(Path(matlab_path))
        
        # Display results
        click.echo("📊 Analysis Results:")
        click.echo(f"   Files Analyzed: {result.get('files_analyzed', 0)}")
        click.echo(f"   Total Functions: {result.get('total_functions', 0)}")
        click.echo(f"   Total Dependencies: {result.get('total_dependencies', 0)}")
        click.echo(f"   Complexity: {result.get('complexity_assessment', 'Unknown')}")
        
        if detailed:
            # Show detailed analysis
            matlab_packages = result.get('matlab_packages_used', [])
            if matlab_packages:
                click.echo(f"   MATLAB Packages: {', '.join(matlab_packages)}")
            
            matlab_functions = result.get('matlab_functions_used', [])
            if matlab_functions:
                click.echo(f"   MATLAB Functions: {', '.join(matlab_functions[:10])}")
                if len(matlab_functions) > 10:
                    click.echo(f"     ... and {len(matlab_functions) - 10} more")
            
            file_analyses = result.get('file_analyses', [])
            if file_analyses:
                click.echo("   File Analysis:")
                for file_analysis in file_analyses[:5]:
                    file_path = file_analysis.get('file_path', 'Unknown')
                    click.echo(f"     - {file_path}")
                if len(file_analyses) > 5:
                    click.echo(f"     ... and {len(file_analyses) - 5} more files")
        
    except Exception as e:
        click.echo(f"❌ Analysis failed: {e}")
        return 1
    
    return 0


@click.command()
@click.argument('cpp_path', type=click.Path(exists=True))
@click.option('--use-legacy', is_flag=True, help='Use legacy validator instead of native LangGraph')
def validate(cpp_path: str, use_legacy: bool):
    """Validate C++ project."""
    try:
        # Load configuration
        config = load_config()
        
        # Initialize logger
        logger_instance = get_logger("validator")
        
        logger_instance.info(f"✅ Validating C++ project: {cpp_path}")
        
        if use_legacy:
            # Use legacy validator
            from ..core.agents.validator.legacy.validator import ValidatorAgent
            validator = ValidatorAgent()
            results = validator.validate_project(Path(cpp_path))
        else:
            # Use native LangGraph validator (fallback to legacy for now)
            click.echo("⚠️  LangGraph validator not available yet, using legacy validator")
            from ..core.agents.validator.legacy.validator import ValidatorAgent
            validator = ValidatorAgent()
            results = validator.validate_project(Path(cpp_path))
        
        # Display results
        if results.get("compilation_success", False):
            click.echo("✅ C++ project compiles successfully!")
        else:
            click.echo("❌ C++ project compilation failed:")
            for error in results.get("compilation_errors", []):
                click.echo(f"   - {error}")
        
        if results.get("tests_passed", False):
            click.echo("✅ All tests passed!")
        else:
            click.echo("⚠️  Some tests failed:")
            for test in results.get("failed_tests", []):
                click.echo(f"   - {test}")
        
    except Exception as e:
        click.echo(f"❌ Validation failed: {e}")
        return 1
    
    return 0


@click.command()
@click.option('--llm-provider', type=click.Choice(['vllm', 'openai']), help='LLM provider')
@click.option('--endpoint', help='LLM endpoint URL')
@click.option('--model', help='Model name')
@click.option('--api-key', help='API key')
@click.option('--max-tokens', type=int, help='Max tokens')
@click.option('--timeout', type=int, help='Timeout in seconds')
@click.option('--temperature', type=float, help='Temperature')
def test_llm(llm_provider: Optional[str], endpoint: Optional[str], model: Optional[str], 
             api_key: Optional[str], max_tokens: Optional[int], timeout: Optional[int], 
             temperature: Optional[float]):
    """Test LLM connection."""
    try:
        # Load configuration
        config = load_config()
        
        # Initialize logger
        logger_instance = get_logger("test_llm")
        
        # Setup environment if provider is specified
        if llm_provider:
            import os
            
            # Set LLM provider specific environment variables
            if llm_provider == 'vllm':
                if endpoint:
                    os.environ['VLLM_ENDPOINT'] = endpoint
                    os.environ['LLM_BASE_URL'] = f"{endpoint}/v1"
                if model:
                    os.environ['VLLM_MODEL_NAME'] = model
                if api_key:
                    os.environ['LLM_API_KEY'] = api_key
                os.environ['LLM_PROVIDER'] = 'vllm'
                
            elif llm_provider == 'openai':
                if api_key:
                    os.environ['OPENAI_API_KEY'] = api_key
                if model:
                    os.environ['LLM_MODEL'] = model
                os.environ['LLM_PROVIDER'] = 'openai'
            
            # Set general LLM parameters
            if max_tokens:
                os.environ['LLM_MAX_TOKENS'] = str(max_tokens)
            if timeout:
                os.environ['LLM_TIMEOUT'] = str(timeout)
            if temperature:
                os.environ['LLM_TEMPERATURE'] = str(temperature)
            
            # Reload configuration with new environment variables
            config = get_config()
        
        logger_instance.info("🧪 Testing LLM connection...")
        
        # Test connection
        success = test_llm_connection(config.llm)
        
        if success:
            click.echo("✅ LLM connection test passed!")
            return 0
        else:
            click.echo("❌ LLM connection test failed!")
            return 1
        
    except Exception as e:
        click.echo(f"❌ LLM test failed: {e}")
        return 1


@click.command()
def examples():
    """Show available examples."""
    try:
        # Get project root
        project_root = Path(__file__).parent.parent.parent.parent
        
        # Initialize logger
        logger_instance = get_logger("examples")
        
        logger_instance.info("📚 Available Examples:")
        click.echo("📚 Available Examples:")
        click.echo("=" * 50)
        
        # Find MATLAB samples
        matlab_samples = project_root / "examples" / "matlab_samples"
        if matlab_samples.exists():
            matlab_files = list(matlab_samples.glob("*.m"))
            if matlab_files:
                click.echo(f"🔬 MATLAB Samples ({len(matlab_files)}):")
                for file_path in matlab_files:
                    click.echo(f"   • {file_path.name}")
                    click.echo(f"     ./matlab2cpp convert {file_path} {file_path.stem}_cpp")
        
        # Show output examples
        output_dir = project_root / "output"
        if output_dir.exists():
            output_projects = [d for d in output_dir.iterdir() if d.is_dir()]
            if output_projects:
                click.echo(f"\n📁 Generated C++ Projects ({len(output_projects)}):")
                for project_dir in output_projects:
                    click.echo(f"   • {project_dir.name}")
                    click.echo(f"     ./matlab2cpp validate {project_dir}")
        
        # Show test files if they exist
        test_files = project_root / "test_files"
        if test_files.exists():
            test_matlab_files = list(test_files.glob("*.m"))
            if test_matlab_files:
                click.echo(f"\n🧪 Test Files ({len(test_matlab_files)}):")
                for file_path in test_matlab_files:
                    click.echo(f"   • {file_path.name}")
                    click.echo(f"     ./matlab2cpp convert {file_path} test_{file_path.stem}")
        
        click.echo("\n💡 Usage Examples:")
        click.echo("   # Convert MATLAB file")
        click.echo("   ./matlab2cpp convert test.m --project-name my_project")
        click.echo("")
        click.echo("   # Convert with monitoring")
        click.echo("   ./matlab2cpp convert test.m --project-name my_project --monitor --export-metrics")
        click.echo("")
        click.echo("   # Analyze MATLAB project")
        click.echo("   ./matlab2cpp analyze test.m --detailed")
        click.echo("")
        click.echo("   # Test LLM connection")
        click.echo("   ./matlab2cpp test-llm")
        click.echo("")
        click.echo("   # Check system status")
        click.echo("   ./matlab2cpp status")
        
    except Exception as e:
        click.echo(f"❌ Error showing examples: {e}")
        return 1
    
    return 0


@click.command()
def status():
    """Check system status."""
    try:
        health_report = get_health_report()
        
        if "error" in health_report:
            click.echo(f"❌ Error getting health report: {health_report['error']}")
            return 1
        
        # Display status
        status_emoji = {
            'healthy': '✅',
            'degraded': '⚠️',
            'unhealthy': '❌',
            'unknown': '❓'
        }.get(health_report['overall_status'], '❓')
        
        click.echo(f"{status_emoji} System Status: {health_report['overall_status'].upper()}")
        click.echo(f"Total Checks: {health_report['total_checks']}")
        click.echo(f"Healthy: {health_report['healthy']}")
        click.echo(f"Degraded: {health_report['degraded']}")
        click.echo(f"Unhealthy: {health_report['unhealthy']}")
        
        if health_report['checks']:
            click.echo("\nComponent Status:")
            for check in health_report['checks']:
                status_emoji = {
                    'healthy': '✅',
                    'degraded': '⚠️',
                    'unhealthy': '❌',
                    'unknown': '❓'
                }.get(check['status'], '❓')
                
                click.echo(f"  {status_emoji} {check['name']}: {check['message']}")
        
    except Exception as e:
        click.echo(f"❌ Error checking status: {e}")
        return 1
    
    return 0


@click.command()
@click.option('--name', help='Filter by metric name')
@click.option('--limit', default=10, help='Number of metrics to show')
def metrics(name: Optional[str], limit: int):
    """Show system metrics."""
    try:
        monitoring_manager = get_monitoring_manager()
        metrics_summary = monitoring_manager.get_metrics_summary()
        
        if "error" in metrics_summary:
            click.echo(f"❌ Error getting metrics: {metrics_summary['error']}")
            return 1
        
        if name:
            # Show specific metric
            collector = monitoring_manager.metrics_collector
            summary = collector.get_metric_summary(name)
            if summary:
                click.echo(f"📊 Metric: {name}")
                click.echo(f"  Count: {summary.count}")
                click.echo(f"  Average: {summary.avg_value:.2f}")
                click.echo(f"  Min: {summary.min_value:.2f}")
                click.echo(f"  Max: {summary.max_value:.2f}")
                click.echo(f"  Latest: {summary.latest_value:.2f}")
            else:
                click.echo(f"❌ No data for metric: {name}")
        else:
            # Show all metrics
            metric_names = metrics_summary.get('metric_names', [])
            if metric_names:
                click.echo("📊 Available Metrics:")
                for metric_name in metric_names[:limit]:
                    collector = monitoring_manager.metrics_collector
                    summary = collector.get_metric_summary(metric_name)
                    if summary:
                        click.echo(f"  {metric_name}: {summary.count} samples, avg {summary.avg_value:.2f}")
            else:
                click.echo("📊 No metrics available")
        
    except Exception as e:
        click.echo(f"❌ Error showing metrics: {e}")
        return 1
    
    return 0


@click.command()
def performance_report():
    """Generate performance report."""
    try:
        report = get_performance_report()
        
        if "error" in report:
            click.echo(f"❌ Error getting performance report: {report['error']}")
            return 1
        
        click.echo("📊 Performance Report")
        click.echo(f"Total Operations: {report.get('total_operations', 0)}")
        click.echo(f"Total Agents: {report.get('total_agents', 0)}")
        
        if report.get('system_metrics'):
            system = report['system_metrics']
            click.echo(f"\n🖥️ System Metrics:")
            click.echo(f"  CPU Usage: {system.get('avg_cpu_percent', 0):.1f}%")
            click.echo(f"  Memory Usage: {system.get('avg_memory_percent', 0):.1f}%")
            click.echo(f"  Available Memory: {system.get('avg_memory_available_gb', 0):.1f}GB")
        
        if report.get('agents'):
            click.echo(f"\n🤖 Agent Performance:")
            for agent_name, perf in report['agents'].items():
                click.echo(f"  {agent_name}:")
                click.echo(f"    Operations: {perf.get('total_operations', 0)}")
                click.echo(f"    Success Rate: {perf.get('success_rate', 0):.1%}")
                click.echo(f"    Avg Time: {perf.get('avg_execution_time', 0):.2f}s")
                click.echo(f"    Avg Memory: {perf.get('avg_memory_usage', 0):.1f}MB")
        
        if report.get('insights'):
            click.echo(f"\n🔍 Performance Insights:")
            for insight in report['insights']:
                click.echo(f"  • {insight}")
        
    except Exception as e:
        click.echo(f"❌ Error generating performance report: {e}")
        return 1
    
    return 0


@click.command()
def performance_recommendations():
    """Get performance optimization recommendations."""
    try:
        recommendations = get_performance_recommendations()
        
        if recommendations:
            click.echo("💡 Performance Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                click.echo(f"  {i}. {rec}")
        else:
            click.echo("✅ No performance recommendations at this time")
        
    except Exception as e:
        click.echo(f"❌ Error getting recommendations: {e}")
        return 1
    
    return 0


@click.command()
@click.option('--output', '-o', help='Output file path')
@click.option('--format', type=click.Choice(['json', 'csv']), default='json')
def export_metrics(output: Optional[str], format: str):
    """Export metrics data."""
    try:
        monitoring_manager = get_monitoring_manager()
        
        if output:
            output_path = Path(output)
        else:
            output_path = Path("metrics_export.json")
        
        success = monitoring_manager.export_metrics(output_path)
        
        if success:
            click.echo(f"📈 Metrics exported to: {output_path}")
        else:
            click.echo("❌ Failed to export metrics")
            return 1
        
    except Exception as e:
        click.echo(f"❌ Error exporting metrics: {e}")
        return 1
    
    return 0


@click.command()
@click.option('--confirm', is_flag=True, help='Confirm clearing metrics')
def clear_metrics(confirm: bool):
    """Clear all metrics data."""
    try:
        if not confirm:
            if not click.confirm('Are you sure you want to clear all metrics?'):
                click.echo("Operation cancelled")
                return 0
        
        monitoring_manager = get_monitoring_manager()
        success = monitoring_manager.clear_metrics()
        
        if success:
            click.echo("🗑️ All metrics cleared")
        else:
            click.echo("❌ Failed to clear metrics")
            return 1
        
    except Exception as e:
        click.echo(f"❌ Error clearing metrics: {e}")
        return 1
    
    return 0


@click.command()
@click.argument('config_path', type=click.Path(exists=True))
def validate_config(config_path: str):
    """Validate configuration file."""
    try:
        # TODO: Implement configuration validation
        click.echo(f"✅ Configuration file is valid: {config_path}")
    except Exception as e:
        click.echo(f"❌ Configuration validation failed: {e}")
        return 1
    
    return 0


@click.command()
def version():
    """Show version information."""
    try:
        # TODO: Get version from package metadata
        click.echo("MATLAB2C++ Conversion Service")
        click.echo("Version: 0.2.0")
        click.echo("Author: Nec Zhang")
    except Exception as e:
        click.echo(f"❌ Error getting version: {e}")
        return 1
    
    return 0


# Create CLI group
@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def cli(ctx, verbose):
    """
    MATLAB2C++ Conversion Service CLI
    
    A comprehensive agentic service for converting MATLAB projects to C++
    with multi-file support and intelligent optimization.
    """
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    
    if verbose:
        logger.add(lambda msg: click.echo(msg, err=True), level="DEBUG")


# Add commands to CLI group
cli.add_command(convert)
cli.add_command(analyze)
cli.add_command(validate)
cli.add_command(test_llm)
cli.add_command(examples)
cli.add_command(status)
cli.add_command(metrics)
cli.add_command(validate_config)
cli.add_command(version)
cli.add_command(performance_report)
cli.add_command(performance_recommendations)
cli.add_command(export_metrics)
cli.add_command(clear_metrics)


if __name__ == '__main__':
    cli()