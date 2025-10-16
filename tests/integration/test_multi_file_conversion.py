#!/usr/bin/env python3
"""
Test script for multi-file MATLAB project conversion using the native LangGraph service.
"""

import os
import sys
import time
from pathlib import Path

# Ensure proper module imports (use PYTHONPATH instead of sys.path manipulation)
# If running directly, the parent directory should be in PYTHONPATH
# Example: PYTHONPATH=/path/to/matlab2cpp_agentic_service/src pytest

from matlab2cpp_agentic_service.utils.config import get_config, load_config
from matlab2cpp_agentic_service.utils.logger import setup_logger, get_logger
from matlab2cpp_agentic_service.core.orchestrators.native_langgraph_orchestrator import NativeLangGraphMATLAB2CPPOrchestrator
from matlab2cpp_agentic_service.infrastructure.state.conversion_state import ConversionRequest

def test_multi_file_conversion():
    """Test multi-file MATLAB project conversion with native LangGraph service."""
    
    print("🧪 Multi-File MATLAB Project Conversion Test")
    print("=" * 60)
    
    # Setup logging
    setup_logger()
    logger = get_logger(__name__)
    
    # Load configuration
    print("📋 Loading configuration...")
    try:
        config = load_config()
        print("✅ Configuration loaded successfully")
        print(f"   LLM Provider: {getattr(config, 'llm_provider', 'unknown')}")
        if getattr(config, 'llm_provider', None) == 'vllm':
            print(f"   vLLM Endpoint: {getattr(config, 'vllm_endpoint', 'unknown')}")
            print(f"   Model: {getattr(config, 'vllm_model_name', 'unknown')}")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return False
    
    # Initialize native LangGraph orchestrator
    print("\n🔧 Initializing Native LangGraph Orchestrator...")
    try:
        orchestrator = NativeLangGraphMATLAB2CPPOrchestrator()
        print("✅ Native LangGraph orchestrator initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize orchestrator: {e}")
        return False
    
    # Test multi-file project
    matlab_project_path = "examples/matlab_samples/skeleton_vessel"
    project_name = "skeleton_vessel_multi_file"
    output_dir = "output"
    max_turns = 3
    target_quality = 6.0
    conversion_mode = "result-focused"
    
    print(f"\n🧪 Testing Multi-File Project Conversion...")
    print(f"   MATLAB Project: {matlab_project_path}")
    print(f"   Project Name: {project_name}")
    print(f"   Conversion Mode: {conversion_mode}")
    print(f"   Max Turns: {max_turns}")
    print(f"   Target Quality: {target_quality}")
    
    # Check if the project directory exists
    if not os.path.exists(matlab_project_path):
        print(f"❌ MATLAB project directory not found: {matlab_project_path}")
        return False
    
    # List MATLAB files in the project
    matlab_files = list(Path(matlab_project_path).glob("*.m"))
    print(f"   Found {len(matlab_files)} MATLAB files:")
    for i, file_path in enumerate(matlab_files, 1):
        print(f"     {i}. {file_path.name}")
    
    # Create conversion request
    request = ConversionRequest(
        matlab_path=matlab_project_path,
        project_name=project_name,
        output_dir=output_dir,
        max_optimization_turns=max_turns,
        target_quality_score=target_quality,
        conversion_mode=conversion_mode
    )
    
    # Run conversion
    print(f"\n🔄 Running Multi-File Conversion...")
    start_time = time.time()
    
    try:
        result = orchestrator.convert_project(request)
        
        end_time = time.time()
        conversion_time = end_time - start_time
        
        print(f"✅ Multi-file conversion completed in {conversion_time:.1f}s")
        
        # Analyze results
        print(f"\n📊 Multi-File Conversion Results Analysis...")
        print(f"   Status: {result.status}")
        print(f"   Original Score: {result.original_score}/10")
        print(f"   Final Score: {result.final_score}/10")
        print(f"   Quality Improvement: +{result.final_score - result.original_score:.1f}")
        print(f"   Optimization Turns: {result.improvement_turns}")
        print(f"   Generated Files: {len(result.generated_files) if result.generated_files else 0}")
        print(f"   Total Processing Time: {result.total_processing_time:.1f}s")
        
        if result.error_message:
            print(f"   Error: {result.error_message}")
        
        # Show generated files
        if result.generated_files:
            print(f"\n📁 Generated Files:")
            for i, file_path in enumerate(result.generated_files, 1):
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    print(f"   ✅ {file_path} ({file_size} bytes)")
                else:
                    print(f"   ❌ {file_path} (not found)")
        
        # Test agent memory
        print(f"\n🧠 Testing Agent Memory for Multi-File Project...")
        try:
            memory_summary = orchestrator.get_agent_memory_summary()
            total_memory = 0
            for agent_name, memory_info in memory_summary.items():
                total_memory += memory_info.get('short_term_memory_size', 0)
                total_memory += memory_info.get('long_term_memory_size', 0)
                total_memory += memory_info.get('context_memory_size', 0)
                print(f"   {agent_name}:")
                print(f"     Short-term: {memory_info.get('short_term_memory_size', 0)} entries")
                print(f"     Long-term: {memory_info.get('long_term_memory_size', 0)} entries")
                print(f"     Context: {memory_info.get('context_memory_size', 0)} entries")
                print(f"     Operations: {memory_info.get('operation_count', 0)}")
            print(f"   Total Memory Entries: {total_memory}")
        except Exception as e:
            print(f"   ❌ Error getting memory summary: {e}")
        
        # Test workflow capabilities
        print(f"\n🎯 Testing Multi-File Workflow Features...")
        try:
            workflow_info = orchestrator.get_workflow_info()
            print(f"   Workflow Type: {workflow_info.get('type', 'unknown')}")
            print(f"   Features: {workflow_info.get('features_count', 0)} native LangGraph features")
            print(f"   Agents: {workflow_info.get('agents_count', 0)} native agents")
            print(f"   Tools: {workflow_info.get('tools_count', 0)} LangGraph tools")
            print(f"   Memory Types: {workflow_info.get('memory_types_count', 0)} memory types")
        except Exception as e:
            print(f"   ❌ Error getting workflow info: {e}")
        
        # Summary
        print(f"\n🎉 Multi-File Conversion Test Results:")
        print("=" * 50)
        print(f"✅ Status: {result.status}")
        print(f"✅ Quality Score: {result.final_score}/10")
        print(f"✅ Improvement: +{result.final_score - result.original_score:.1f}")
        print(f"✅ Optimization Turns: {result.improvement_turns}")
        print(f"✅ Generated Files: {len(result.generated_files) if result.generated_files else 0}")
        print(f"✅ Processing Time: {result.total_processing_time:.1f}s")
        print(f"✅ Has Errors: {bool(result.error_message)}")
        if result.error_message:
            print(f"❌ Error Message: {result.error_message}")
        
        return result.status == "completed"
        
    except Exception as e:
        end_time = time.time()
        conversion_time = end_time - start_time
        print(f"❌ Multi-file conversion failed after {conversion_time:.1f}s: {e}")
        return False

if __name__ == "__main__":
    success = test_multi_file_conversion()
    if success:
        print("\n🎉 Multi-file conversion test passed!")
        sys.exit(0)
    else:
        print("\n❌ Multi-file conversion test failed!")
        sys.exit(1)
