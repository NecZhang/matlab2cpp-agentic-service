#!/usr/bin/env python3
"""
General MATLAB2C++ Converter CLI

Usage:
    python -m matlab2cpp_agent.cli.general_converter <matlab_path> <project_name> [options]
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from matlab2cpp_agent.services.matlab2cpp_service import (
    convert_matlab_project, 
    convert_matlab_script,
    ConversionRequest,
    ConversionStatus
)

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert MATLAB projects/scripts to C++ with comprehensive analysis and optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert MATLAB project directory
  python -m matlab2cpp_agent.cli.general_converter ./my_matlab_project my_cpp_project
  
  # Convert single MATLAB file
  python -m matlab2cpp_agent.cli.general_converter ./script.m my_script
  
  # Convert with custom options
  python -m matlab2cpp_agent.cli.general_converter ./project my_project \\
    --output-dir ./cpp_output \\
    --max-turns 3 \\
    --target-quality 8.0 \\
    --cpp-standard C++20 \\
    --include-tests
        """
    )
    
    parser.add_argument(
        "matlab_path",
        help="Path to MATLAB file or project directory"
    )
    
    parser.add_argument(
        "project_name",
        help="Name for the generated C++ project"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for generated C++ files (default: ./output)"
    )
    
    parser.add_argument(
        "--max-turns",
        type=int,
        default=2,
        help="Maximum optimization turns (default: 2)"
    )
    
    parser.add_argument(
        "--target-quality",
        type=float,
        default=7.0,
        help="Target quality score (0-10, default: 7.0)"
    )
    
    parser.add_argument(
        "--cpp-standard",
        type=str,
        default="C++17",
        help="C++ standard to use (default: C++17)"
    )
    
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Include unit tests in generated code"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Output results in JSON format"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    matlab_path = Path(args.matlab_path)
    if not matlab_path.exists():
        print(f"❌ Error: MATLAB path does not exist: {matlab_path}")
        sys.exit(1)
    
    if not matlab_path.is_file() and not matlab_path.is_dir():
        print(f"❌ Error: MATLAB path must be a file or directory: {matlab_path}")
        sys.exit(1)
    
    # Create conversion request
    request = ConversionRequest(
        matlab_path=matlab_path,
        project_name=args.project_name,
        output_dir=args.output_dir,
        max_optimization_turns=args.max_turns,
        target_quality_score=args.target_quality,
        include_tests=args.include_tests,
        cpp_standard=args.cpp_standard
    )
    
    print("🚀 MATLAB2C++ General Converter")
    print("=" * 50)
    print(f"📁 MATLAB Path: {matlab_path}")
    print(f"🏗️  Project Name: {args.project_name}")
    print(f"📤 Output Directory: {args.output_dir or './output'}")
    print(f"🔄 Max Optimization Turns: {args.max_turns}")
    print(f"🎯 Target Quality Score: {args.target_quality}/10")
    print(f"📋 C++ Standard: {args.cpp_standard}")
    print(f"🧪 Include Tests: {args.include_tests}")
    print("=" * 50)
    
    try:
        # Perform conversion
        result = convert_matlab_project(
            matlab_path, 
            args.project_name,
            output_dir=args.output_dir,
            max_optimization_turns=args.max_turns,
            target_quality_score=args.target_quality,
            include_tests=args.include_tests,
            cpp_standard=args.cpp_standard
        )
        
        # Display results
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
        else:
            display_conversion_result(result, args.verbose)
        
        # Exit with appropriate code
        if result.status == ConversionStatus.COMPLETED:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def display_conversion_result(result, verbose: bool = False):
    """Display conversion result in human-readable format."""
    
    print("\n📋 CONVERSION RESULTS")
    print("=" * 50)
    
    if result.status == ConversionStatus.COMPLETED:
        print("✅ Conversion completed successfully!")
    elif result.status == ConversionStatus.FAILED:
        print("❌ Conversion failed!")
        if result.error_message:
            print(f"   Error: {result.error_message}")
        return
    else:
        print(f"⚠️  Conversion status: {result.status.value}")
    
    print(f"🏗️  Project: {result.project_name}")
    print(f"⏱️  Total Time: {result.total_processing_time:.1f}s")
    print(f"📊 Quality Scores:")
    print(f"   Original: {result.original_score:.1f}/10")
    print(f"   Final: {result.final_score:.1f}/10")
    print(f"   Improvement: {result.final_score - result.original_score:+.1f} points")
    print(f"🔄 Optimization Turns: {result.improvement_turns}")
    
    if result.generated_files:
        print(f"\n📁 Generated Files ({len(result.generated_files)}):")
        for file_path in result.generated_files:
            print(f"   • {file_path}")
    
    if result.assessment_reports:
        print(f"\n📋 Assessment Reports ({len(result.assessment_reports)}):")
        for report_path in result.assessment_reports:
            print(f"   • {report_path}")
    
    if verbose and result.conversion_plan:
        print(f"\n📋 Conversion Plan:")
        print(f"   Dependencies: {', '.join(result.conversion_plan.dependencies)}")
        print(f"   Complexity: {result.conversion_plan.estimated_complexity}")
        print(f"   Steps: {len(result.conversion_plan.conversion_steps)}")
        
        if result.conversion_plan.conversion_steps:
            print("   Conversion Steps:")
            for i, step in enumerate(result.conversion_plan.conversion_steps, 1):
                print(f"     {i}. {step}")
    
    # Quality assessment
    if result.final_score >= 8.0:
        print("\n🎉 Excellent quality! Ready for production use.")
    elif result.final_score >= 6.0:
        print("\n✅ Good quality! Suitable for most use cases.")
    elif result.final_score >= 4.0:
        print("\n⚠️  Moderate quality. Consider manual review.")
    else:
        print("\n❌ Low quality. Manual intervention recommended.")

if __name__ == "__main__":
    main()
