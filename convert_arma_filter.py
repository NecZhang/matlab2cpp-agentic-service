#!/usr/bin/env python3
"""
Convert arma_filter.m using the modular MATLAB2C++ service
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set environment variables for vLLM
os.environ["VLLM_ENDPOINT"] = "http://192.168.6.10:8002"
os.environ["VLLM_MODEL_NAME"] = "Qwen/Qwen3-32B-FP8"
os.environ["LLM_PROVIDER"] = "vllm"
os.environ["LLM_BASE_URL"] = "http://192.168.6.10:8002/v1"
os.environ["LLM_API_KEY"] = "dummy_key"
os.environ["LLM_MAX_TOKENS"] = "8000"
os.environ["LLM_TIMEOUT"] = "1200"

from matlab2cpp_agent.services.matlab2cpp_orchestrator import (
    MATLAB2CPPOrchestrator,
    ConversionRequest
)

def convert_arma_filter():
    """Convert arma_filter.m using the modular service."""
    
    print("🚀 Converting arma_filter.m with Modular MATLAB2C++ Service")
    print("=" * 70)
    
    # Initialize the orchestrator
    print("🔧 Initializing orchestrator with all agents...")
    orchestrator = MATLAB2CPPOrchestrator()
    print("✅ Orchestrator initialized successfully")
    
    # Create conversion request
    request = ConversionRequest(
        matlab_path="examples/matlab_samples/arma_filter.m",
        project_name="arma_filter_modular",
        output_dir="output/arma_filter_conversion",
        max_optimization_turns=2,
        target_quality_score=7.0,
        include_tests=True,
        cpp_standard="C++17"
    )
    
    print(f"\n📋 Conversion Request:")
    print(f"   MATLAB Path: {request.matlab_path}")
    print(f"   Project Name: {request.project_name}")
    print(f"   Output Directory: {request.output_dir}")
    print(f"   Max Optimization Turns: {request.max_optimization_turns}")
    print(f"   Target Quality Score: {request.target_quality_score}/10")
    print(f"   Include Tests: {request.include_tests}")
    print(f"   C++ Standard: {request.cpp_standard}")
    
    # Check if MATLAB file exists
    matlab_file = Path(request.matlab_path)
    if not matlab_file.exists():
        print(f"❌ Error: MATLAB file not found: {matlab_file}")
        return
    
    print(f"\n✅ MATLAB file found: {matlab_file}")
    print(f"   File size: {matlab_file.stat().st_size} bytes")
    
    try:
        # Perform conversion
        print(f"\n🔄 Starting conversion process...")
        print("   This will take several minutes due to LLM processing...")
        
        result = orchestrator.convert_project(request)
        
        # Display results
        print_conversion_result(result)
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()

def print_conversion_result(result):
    """Print conversion result in a formatted way."""
    print(f"\n📋 CONVERSION RESULT")
    print("=" * 50)
    
    if result.status.value == "completed":
        print("✅ Conversion completed successfully!")
    else:
        print(f"❌ Conversion status: {result.status.value}")
        if result.error_message:
            print(f"   Error: {result.error_message}")
        return
    
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
    
    # Quality assessment
    if result.final_score >= 8.0:
        print(f"\n🎉 Excellent quality! Ready for production use.")
    elif result.final_score >= 6.0:
        print(f"\n✅ Good quality! Suitable for most use cases.")
    elif result.final_score >= 4.0:
        print(f"\n⚠️  Moderate quality. Consider manual review.")
    else:
        print(f"\n❌ Low quality. Manual intervention recommended.")

if __name__ == "__main__":
    convert_arma_filter()