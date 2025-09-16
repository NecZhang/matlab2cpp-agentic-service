#!/usr/bin/env python3
"""
Convert arma_filter.m using command line interface
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
os.environ["LLM_TIMEOUT"] = "600"

def main():
    """Run the conversion using command line interface."""
    
    print("🚀 Converting arma_filter.m using Command Line Interface")
    print("=" * 70)
    
    # Import the CLI module
    from matlab2cpp_agent.cli.general_converter import main as cli_main
    
    # Set up command line arguments
    sys.argv = [
        "general_converter",
        "examples/matlab_samples/arma_filter.m",
        "arma_filter_cli",
        "--output-dir", "output/arma_filter_cli",
        "--max-turns", "2",
        "--target-quality", "7.0",
        "--cpp-standard", "C++17",
        "--include-tests",
        "--verbose"
    ]
    
    print("📋 Command line arguments:")
    print("   MATLAB Path: examples/matlab_samples/arma_filter.m")
    print("   Project Name: arma_filter_cli")
    print("   Output Directory: output/arma_filter_cli")
    print("   Max Optimization Turns: 2")
    print("   Target Quality Score: 7.0/10")
    print("   C++ Standard: C++17")
    print("   Include Tests: Yes")
    print("   Verbose Output: Yes")
    
    print(f"\n🔄 Starting conversion...")
    
    try:
        # Run the CLI
        cli_main()
    except Exception as e:
        print(f"❌ CLI conversion failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
