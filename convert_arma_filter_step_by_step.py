#!/usr/bin/env python3
"""
Convert arma_filter.m using individual agents step by step
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

from matlab2cpp_agent.agents.matlab_content_analyzer import MATLABContentAnalyzerAgent
from matlab2cpp_agent.agents.conversion_planner import ConversionPlannerAgent
from matlab2cpp_agent.agents.cpp_generator import CppGeneratorAgent
from matlab2cpp_agent.agents.quality_assessor import QualityAssessorAgent
from matlab2cpp_agent.utils.config import get_config

def convert_arma_filter_step_by_step():
    """Convert arma_filter.m using individual agents step by step."""
    
    print("🚀 Converting arma_filter.m with Individual Agents (Step by Step)")
    print("=" * 70)
    
    # Get configuration
    config = get_config()
    
    # Initialize all agents
    print("🔧 Initializing individual agents...")
    content_analyzer = MATLABContentAnalyzerAgent(config.llm)
    conversion_planner = ConversionPlannerAgent(config.llm)
    cpp_generator = CppGeneratorAgent(config.llm)
    quality_assessor = QualityAssessorAgent(config.llm)
    print("✅ All agents initialized successfully")
    
    # Step 1: Analyze MATLAB content
    print(f"\n📊 Step 1: Analyzing MATLAB content...")
    matlab_path = Path("examples/matlab_samples/arma_filter.m")
    
    if not matlab_path.exists():
        print(f"❌ Error: MATLAB file not found: {matlab_path}")
        return
    
    try:
        matlab_analysis = content_analyzer.analyze_matlab_content(matlab_path)
        print(f"✅ Analysis complete:")
        print(f"   Files analyzed: {matlab_analysis['files_analyzed']}")
        print(f"   Total functions: {matlab_analysis['total_functions']}")
        print(f"   Total dependencies: {matlab_analysis['total_dependencies']}")
        print(f"   MATLAB packages: {matlab_analysis['matlab_packages_used']}")
        print(f"   Complexity: {matlab_analysis['complexity_assessment']}")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return
    
    # Step 2: Create conversion plan
    print(f"\n🏗️  Step 2: Creating conversion plan...")
    try:
        conversion_plan = conversion_planner.create_conversion_plan(
            matlab_analysis=matlab_analysis,
            project_name="arma_filter_step_by_step",
            cpp_standard="C++17",
            include_tests=True
        )
        print(f"✅ Conversion plan created:")
        print(f"   Dependencies: {conversion_plan.dependencies}")
        print(f"   Architecture: {conversion_plan.cpp_architecture}")
        print(f"   Conversion steps: {len(conversion_plan.conversion_steps)}")
    except Exception as e:
        print(f"❌ Planning failed: {e}")
        return
    
    # Step 3: Generate C++ code
    print(f"\n💻 Step 3: Generating C++ code...")
    try:
        cpp_code = cpp_generator.generate_cpp_code(
            matlab_analysis=matlab_analysis,
            conversion_plan=conversion_plan,
            project_name="arma_filter_step_by_step",
            cpp_standard="C++17",
            target_quality_score=7.0
        )
        
        if cpp_code:
            print(f"✅ C++ code generated:")
            print(f"   Header size: {len(cpp_code.get('header', ''))} characters")
            print(f"   Implementation size: {len(cpp_code.get('implementation', ''))} characters")
            
            # Save the code
            output_dir = Path("output/arma_filter_step_by_step")
            output_dir.mkdir(exist_ok=True)
            
            if cpp_code.get('header'):
                header_file = output_dir / "arma_filter_step_by_step.h"
                header_file.write_text(cpp_code['header'])
                print(f"   Header saved: {header_file}")
            
            if cpp_code.get('implementation'):
                impl_file = output_dir / "arma_filter_step_by_step.cpp"
                impl_file.write_text(cpp_code['implementation'])
                print(f"   Implementation saved: {impl_file}")
        else:
            print(f"❌ Code generation failed")
            return
    except Exception as e:
        print(f"❌ Code generation failed: {e}")
        return
    
    # Step 4: Assess code quality
    print(f"\n🔍 Step 4: Assessing code quality...")
    try:
        matlab_code_content = get_matlab_code_content(matlab_analysis)
        assessment = quality_assessor.assess_code_quality(
            cpp_code=cpp_code.get('implementation', ''),
            matlab_code=matlab_code_content,
            project_name="arma_filter_step_by_step"
        )
        
        print(f"✅ Quality assessment complete:")
        print(f"   Overall score: {assessment.metrics.overall_score:.1f}/10")
        print(f"   Issues found: {len(assessment.issues)}")
        print(f"   Suggestions: {len(assessment.suggestions)}")
        
        # Save assessment report
        report_file = output_dir / "quality_assessment_report.md"
        quality_assessor.generate_assessment_report(assessment, report_file)
        print(f"   Assessment report saved: {report_file}")
        
    except Exception as e:
        print(f"❌ Quality assessment failed: {e}")
        return
    
    print(f"\n🎉 Step-by-step conversion completed successfully!")
    print(f"📁 All files saved to: {output_dir}")

def get_matlab_code_content(matlab_analysis):
    """Extract MATLAB code content for assessment."""
    content_parts = []
    for file_analysis in matlab_analysis.get('file_analyses', []):
        content_parts.append(f"File: {file_analysis.get('file_path', 'Unknown')}")
        if 'parsed_structure' in file_analysis and hasattr(file_analysis['parsed_structure'], 'content'):
            content_parts.append(file_analysis['parsed_structure'].content)
        content_parts.append("")
    return "\n".join(content_parts)

if __name__ == "__main__":
    convert_arma_filter_step_by_step()
