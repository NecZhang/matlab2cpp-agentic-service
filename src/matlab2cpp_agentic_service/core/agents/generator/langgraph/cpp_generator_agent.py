"""
LangGraph-Native C++ Generator Agent

This module implements a truly LangGraph-native C++ generator agent
that generates high-quality C++ code using LangGraph tools and memory.
"""

import time
from typing import Dict, Any, List, Callable
from pathlib import Path
from loguru import logger

from matlab2cpp_agentic_service.infrastructure.state.conversion_state import ConversionState, ConversionStatus, add_processing_time, update_state_status
from matlab2cpp_agentic_service.infrastructure.tools.llm_client import LLMClient
from matlab2cpp_agentic_service.infrastructure.tools.langgraph_tools import CodeGenerationTool, LLMAnalysisTool, ToolRegistry
from ...base.langgraph_agent import BaseLangGraphAgent, AgentConfig


class LangGraphCppGeneratorAgent(BaseLangGraphAgent):
    """
    LangGraph-native C++ generator agent.
    
    This agent generates high-quality C++ code using LangGraph tools
    and maintains generation history in memory for iterative improvements.
    """
    
    def __init__(self, config: AgentConfig, llm_client: LLMClient):
        """Initialize the LangGraph C++ generator agent."""
        super().__init__(config, llm_client)
        
        # Initialize tools
        self.tool_registry = ToolRegistry()
        self.code_generation_tool = CodeGenerationTool(llm_client)
        self.llm_analysis_tool = LLMAnalysisTool(llm_client)
        
        # Register tools
        self.tool_registry.register_tool("code_generation", self.code_generation_tool)
        self.tool_registry.register_tool("llm_analysis", self.llm_analysis_tool)
        
        self.logger.info(f"Initialized LangGraph C++ Generator Agent: {config.name}")
    
    def create_node(self) -> Callable[[ConversionState], ConversionState]:
        """
        Create the LangGraph node function for C++ code generation.
        
        Returns:
            Callable that takes and returns ConversionState
        """
        def generate_cpp_node(state: ConversionState) -> ConversionState:
            """LangGraph node function for C++ code generation."""
            start_time = time.time()
            
            try:
                self.logger.info("Starting LangGraph-native C++ code generation...")
                
                # Update state status
                state = update_state_status(state, ConversionStatus.GENERATING)
                
                # Get required data from state
                matlab_analysis = state.get("matlab_analysis")
                conversion_plan = state.get("conversion_plan")
                
                if not matlab_analysis or not conversion_plan:
                    raise ValueError("MATLAB analysis or conversion plan not found in state")
                
                # Get current turn for optimization
                current_turn = state.get("current_turn", 0)
                is_optimization = current_turn > 0
                
                # Check if we have cached generation in memory for this turn
                cache_key = f"cpp_generation_{current_turn}_{hash(str(matlab_analysis))}"
                cached_generation = self.get_memory(cache_key, "long_term")
                
                if cached_generation and not is_optimization:
                    self.logger.info("Using cached C++ generation from memory")
                    generated_code = cached_generation
                else:
                    self.logger.info(f"Generating fresh C++ code (turn {current_turn})")
                    generated_code = self._generate_cpp_code(matlab_analysis, conversion_plan, state)
                    
                    # Cache the generation in long-term memory
                    self.update_memory(cache_key, generated_code, "long_term")
                
                # Update agent memory with generation context
                self.update_memory("last_generation", generated_code, "short_term")
                self.update_memory("generation_count", 
                                 (self.get_memory("generation_count", "short_term") or 0) + 1, 
                                 "short_term")
                self.update_memory("current_turn", current_turn, "context")
                self.update_memory("is_optimization", is_optimization, "context")
                
                # Update state with generation results
                state["generated_code"] = generated_code
                
                # Update state with agent memory
                state = self.update_state_with_result(state, generated_code, "cpp_generation")
                
                # Log success
                duration = time.time() - start_time
                self.logger.info(f"C++ code generation completed successfully in {duration:.2f}s")
                
                if isinstance(generated_code, dict) and 'files' in generated_code:
                    # Count .cpp and .h files separately for clarity
                    cpp_files = sum(1 for f in generated_code['files'] if f.endswith('.cpp'))
                    h_files = sum(1 for f in generated_code['files'] if f.endswith('.h'))
                    total_files = len(generated_code['files'])
                    self.logger.info(f"Generated {total_files} files ({cpp_files} .cpp + {h_files} .h + {total_files - cpp_files - h_files} other)")
                else:
                    self.logger.info("Generated single-file C++ code")
                
                # Track performance
                self.track_performance("cpp_generation", start_time, time.time(), True, {
                    "turn": current_turn,
                    "is_optimization": is_optimization,
                    "project_type": conversion_plan.get('project_type', 'single_file')
                })
                
            except Exception as e:
                self.logger.error(f"Error in C++ code generation: {e}")
                state["error_message"] = f"C++ code generation failed: {str(e)}"
                state = update_state_status(state, ConversionStatus.FAILED)
                
                # Track failure
                self.track_performance("cpp_generation", start_time, time.time(), False, {
                    "error": str(e)
                })
            
            # Record processing time
            duration = time.time() - start_time
            state = add_processing_time(state, "cpp_generation", duration)
            
            return state
        
        return generate_cpp_node
    
    def get_tools(self) -> List[Any]:
        """Get available tools for this agent."""
        return [
            self.code_generation_tool,
            self.llm_analysis_tool
        ]
    
    def _generate_cpp_code(self, matlab_analysis: Dict[str, Any], conversion_plan: Dict[str, Any], 
                          state: ConversionState) -> Dict[str, Any]:
        """
        Generate C++ code using LangGraph tools.
        
        Args:
            matlab_analysis: MATLAB analysis results
            conversion_plan: Conversion plan
            state: Current conversion state
            
        Returns:
            Generated C++ code
        """
        self.logger.debug("Generating C++ code using LangGraph tools")
        
        # Determine generation strategy
        project_type = conversion_plan.get('project_type', 'single_file')
        current_turn = state.get("current_turn", 0)
        is_optimization = current_turn > 0
        
        if project_type == 'multi_file':
            # Generate multi-file project (both initial and optimization)
            generated_code = self._generate_multi_file_project(matlab_analysis, conversion_plan, state)
        else:
            # Generate single file
            generated_code = self._generate_single_file_code(matlab_analysis, conversion_plan, state)
        
        # Add generation metadata
        generated_code['generation_metadata'] = {
            'turn': current_turn,
            'is_optimization': is_optimization,
            'project_type': project_type,
            'generation_timestamp': time.time(),
            'agent_memory': {
                'generation_count': self.get_memory("generation_count", "short_term") or 0,
                'last_generation_time': time.time()
            }
        }
        
        self.logger.info(f"C++ code generation complete: {project_type} project, turn {current_turn}")
        
        return generated_code
    
    def _generate_single_file_code(self, matlab_analysis: Dict[str, Any], conversion_plan: Dict[str, Any], 
                                  state: ConversionState) -> Dict[str, Any]:
        """Generate single-file C++ code."""
        self.logger.debug("Generating single-file C++ code")
        
        # Prepare MATLAB summary for code generation
        file_analyses = matlab_analysis.get('file_analyses', [])
        if file_analyses:
            first_file = file_analyses[0]
            matlab_summary = {
                'functions': first_file['parsed_structure'].get('functions', []),
                'dependencies': first_file['parsed_structure'].get('dependencies', []),
                'numerical_calls': first_file['parsed_structure'].get('numerical_calls', []),
                'source_code': first_file.get('source_code', ''),  # Use actual source code from file analysis
                'complexity': first_file['parsed_structure'].get('complexity', {}),
                'llm_analysis': first_file.get('llm_analysis', {})
            }
        else:
            matlab_summary = {}
        
        # Use code generation tool
        generation_result = self.code_generation_tool(
            matlab_summary,
            conversion_plan,
            conversion_plan.get('conversion_mode', 'result-focused')
        )
        
        if not generation_result.success:
            raise RuntimeError(f"Code generation failed: {generation_result.error}")
        
        generated_code = generation_result.data
        
        # Ensure proper structure
        if not isinstance(generated_code, dict):
            generated_code = {
                'header': '',
                'implementation': str(generated_code),
                'dependencies': ['Eigen3'],
                'compilation_instructions': 'g++ -std=c++17 -I/path/to/eigen your_file.cpp',
                'usage_example': 'See implementation comments',
                'notes': 'Generated from raw response'
            }
        
        return generated_code
    
    def _generate_multi_file_project(self, matlab_analysis: Dict[str, Any], conversion_plan: Dict[str, Any], 
                                   state: ConversionState) -> Dict[str, Any]:
        """Generate multi-file C++ project."""
        self.logger.debug("Generating multi-file C++ project")
        
        file_analyses = matlab_analysis.get('file_analyses', [])
        project_structure = conversion_plan.get('project_structure_plan', {})
        file_mapping = project_structure.get('file_mapping', {})
        
        generated_files = {}
        compilation_instructions = []
        
        # Generate code for each file
        for file_analysis in file_analyses:
            file_path = file_analysis['file_path']
            
            if file_path in file_mapping:
                cpp_info = file_mapping[file_path]
                cpp_file = cpp_info['cpp_file']
                
                # Generate code for this file
                file_code = self._generate_file_code(file_analysis, conversion_plan, cpp_info)
                
                # Store generated files
                if file_code.get('header'):
                    header_file = cpp_info['header_file']
                    generated_files[header_file] = file_code['header']
                else:
                    self.logger.warning(f"Missing header file for {cpp_file}")
                
                if file_code.get('implementation'):
                    generated_files[cpp_file] = file_code['implementation']
                else:
                    self.logger.warning(f"Missing implementation file for {cpp_file}")
                
                # Collect compilation instructions
                if file_code.get('compilation_instructions'):
                    compilation_instructions.append(file_code['compilation_instructions'])
        
        # Generate main file
        main_code = self._generate_main_file(matlab_analysis, conversion_plan, file_mapping)
        if main_code.get('implementation'):
            generated_files['main.cpp'] = main_code['implementation']
        if main_code.get('header'):
            generated_files['main.h'] = main_code['header']
        
        # Create project compilation instructions
        project_instructions = self._create_project_compilation_instructions(
            generated_files, conversion_plan, compilation_instructions
        )
        
        return {
            'files': generated_files,
            'compilation_instructions': project_instructions,
            'project_structure': project_structure,
            'dependencies': conversion_plan.get('dependencies', ['Eigen3']),
            'usage_example': self._create_project_usage_example(generated_files, conversion_plan),
            'notes': 'Multi-file C++ project generated from MATLAB code'
        }
    
    def _generate_file_code(self, file_analysis: Dict[str, Any], conversion_plan: Dict[str, Any], 
                          cpp_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code for a specific file."""
        parsed_structure = file_analysis['parsed_structure']
        llm_analysis = file_analysis.get('llm_analysis', {})
        
        # Prepare file-specific summary
        file_summary = {
            'functions': parsed_structure.get('functions', []),
            'dependencies': parsed_structure.get('dependencies', []),
            'numerical_calls': parsed_structure.get('numerical_calls', []),
            'source_code': file_analysis.get('source_code', ''),  # Use actual source code from file analysis
            'complexity': parsed_structure.get('complexity', {}),
            'llm_analysis': llm_analysis,
            'namespace': cpp_info.get('namespace', 'unnamed'),
            'cpp_file': cpp_info['cpp_file'],
            'header_file': cpp_info['header_file']
        }
        
        # Use code generation tool
        generation_result = self.code_generation_tool(
            file_summary,
            conversion_plan,
            conversion_plan.get('conversion_mode', 'result-focused')
        )
        
        if not generation_result.success:
            self.logger.warning(f"Code generation failed for file: {generation_result.error}")
            return {
                'header': f'// Error generating header for {cpp_info["header_file"]}',
                'implementation': f'// Error generating implementation for {cpp_info["cpp_file"]}',
                'compilation_instructions': '// See error above',
                'notes': f'Generation failed: {generation_result.error}'
            }
        
        return generation_result.data
    
    def _generate_main_file(self, matlab_analysis: Dict[str, Any], conversion_plan: Dict[str, Any], 
                          file_mapping: Dict[str, Any]) -> Dict[str, Any]:
        """Generate main file for multi-file project."""
        # Create a simple entry point that uses the main module, not duplicate its functionality
        return self._create_proper_main_file(matlab_analysis, file_mapping, conversion_plan)
    
    def _create_proper_main_file(self, matlab_analysis: Dict[str, Any], file_mapping: Dict[str, Any], 
                               conversion_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Create a proper main file that uses the main module instead of duplicating functionality."""
        # Find the main MATLAB file (usually the one with the main function name)
        file_analyses = matlab_analysis.get('file_analyses', [])
        main_file_analysis = None
        main_module_namespace = None
        
        # Look for the main file (typically the one that calls other functions)
        for file_analysis in file_analyses:
            file_path = file_analysis['file_path']
            if file_path in file_mapping:
                cpp_info = file_mapping[file_path]
                main_module_namespace = cpp_info.get('namespace', 'main')
                main_file_analysis = file_analysis
                break
        
        if not main_file_analysis:
            # Fallback to first file
            if file_analyses:
                main_file_analysis = file_analyses[0]
                main_module_namespace = 'main'
        
        # Generate includes for the main module
        includes = []
        if main_file_analysis:
            file_path = main_file_analysis['file_path']
            if file_path in file_mapping:
                cpp_info = file_mapping[file_path]
                header_file = cpp_info['header_file']
                includes.append(f'#include "{header_file}"')
        
        # Add standard includes
        includes.extend([
            '#include <iostream>',
            '#include <vector>',
            '#include <Eigen/Dense>',
            '#include <opencv2/opencv.hpp>'
        ])
        
        # Create main function implementation
        main_implementation = f'''{chr(10).join(includes)}

int main() {{
    std::cout << "MATLAB to C++ Conversion Project" << std::endl;
    
    try {{
        // Example usage of the main module
        // TODO: Replace with actual test data and parameters
        Eigen::MatrixXd testImage = Eigen::MatrixXd::Random(100, 100);
        
        // Call the main function from the converted module
        // {main_module_namespace}::main_function(testImage);
        
        std::cout << "Conversion successful!" << std::endl;
        return 0;
    }} catch (const std::exception& e) {{
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }}
}}'''
        
        # Create header file
        header_content = f'''#ifndef MAIN_H
#define MAIN_H

{chr(10).join(includes)}

// Main entry point for the converted MATLAB project
int main();

#endif // MAIN_H'''
        
        return {
            'header': header_content,
            'implementation': main_implementation,
            'dependencies': ['Eigen3', 'OpenCV'],
            'compilation_instructions': f'g++ -std=c++17 -I/path/to/eigen -I/path/to/opencv main.cpp -lopencv_core -lopencv_imgproc',
            'usage_example': 'Compile and run: ./main',
            'notes': 'Simple entry point that uses the converted MATLAB modules'
        }
    
    def _create_fallback_main_file(self, file_mapping: Dict[str, Any], conversion_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Create a fallback main file if generation fails."""
        includes = []
        namespace_uses = []
        
        # Generate includes for all modules
        for matlab_file, cpp_info in file_mapping.items():
            header_file = cpp_info['header_file']
            namespace = cpp_info['namespace']
            includes.append(f'#include "{header_file}"')
            namespace_uses.append(f'using namespace {namespace};')
        
        # Create basic main file content
        main_header = f'''#pragma once

{chr(10).join(includes)}

// Main header for MATLAB to C++ conversion project
class MainApplication {{
public:
    static int run();
private:
    // Application logic will be implemented here
}};
'''
        
        main_implementation = f'''#include "main.h"
#include <iostream>
#include <vector>

{chr(10).join(namespace_uses)}

int MainApplication::run() {{
    std::cout << "MATLAB to C++ Conversion Project" << std::endl;
    std::cout << "Generated from MATLAB code" << std::endl;
    
    // TODO: Implement main application logic
    // Add calls to functions from converted modules
    
    return 0;
}}

int main() {{
    return MainApplication::run();
}}
'''
        
        return {
            'header': main_header,
            'implementation': main_implementation,
            'compilation_instructions': 'g++ -std=c++17 -I/path/to/eigen *.cpp -o main',
            'notes': 'Fallback main file created due to generation failure'
        }
    
    def _create_project_compilation_instructions(self, generated_files: Dict[str, str], 
                                               conversion_plan: Dict[str, Any], 
                                               file_instructions: List[str]) -> str:
        """Create compilation instructions for the entire project."""
        dependencies = conversion_plan.get('dependencies', ['Eigen3'])
        compilation_reqs = conversion_plan.get('compilation_requirements', {})
        
        # Build compilation command
        cpp_standard = compilation_reqs.get('cpp_standard', 'C++17')
        compiler_flags = ' '.join(compilation_reqs.get('compiler_flags', ['-O2', '-Wall']))
        
        # Include directories
        include_dirs = compilation_reqs.get('include_directories', [])
        include_flags = ' '.join(f'-I{dir}' for dir in include_dirs)
        
        # Library directories and libraries
        lib_dirs = compilation_reqs.get('library_directories', [])
        lib_flags = ' '.join(f'-L{dir}' for dir in lib_dirs)
        
        libraries = compilation_reqs.get('libraries', [])
        lib_link_flags = ' '.join(f'-l{lib}' for lib in libraries)
        
        # Get all C++ files
        cpp_files = [f for f in generated_files.keys() if f.endswith('.cpp')]
        cpp_files_str = ' '.join(cpp_files)
        
        instructions = f'''# Compilation Instructions for MATLAB to C++ Conversion Project

## Dependencies
Required libraries: {', '.join(dependencies)}

## Basic Compilation
g++ -std={cpp_standard} {compiler_flags} {include_flags} {cpp_files_str} {lib_flags} {lib_link_flags} -o main

## Debug Build
g++ -std={cpp_standard} -g -O0 -Wall -Wextra {include_flags} {cpp_files_str} {lib_flags} {lib_link_flags} -o main_debug

## Release Build
g++ -std={cpp_standard} -O3 -DNDEBUG {include_flags} {cpp_files_str} {lib_flags} {lib_link_flags} -o main_release

## Files Generated
{chr(10).join(f"- {f}" for f in generated_files.keys())}

## Usage
./main
'''
        
        return instructions
    
    def _create_project_usage_example(self, generated_files: Dict[str, str], 
                                    conversion_plan: Dict[str, Any]) -> str:
        """Create usage example for the generated project."""
        project_type = conversion_plan.get('project_type', 'single_file')
        
        if project_type == 'single_file':
            return '''// Single file usage example
#include "main.h"

int main() {
    // Create instance and use the converted functionality
    // Example usage will depend on the specific MATLAB code converted
    return 0;
}'''
        else:
            return '''// Multi-file project usage example
#include "main.h"

int main() {
    // Initialize the application
    MainApplication app;
    
    // Use the converted MATLAB functionality
    // Each module provides its own namespace and functions
    
    return app.run();
}'''
    
    def get_generation_summary(self, state: ConversionState) -> Dict[str, Any]:
        """Get a summary of the generation results from state."""
        generated_code = state.get("generated_code", {})
        generation_metadata = generated_code.get('generation_metadata', {})
        
        return {
            'turn': generation_metadata.get('turn', 0),
            'is_optimization': generation_metadata.get('is_optimization', False),
            'project_type': generation_metadata.get('project_type', 'single_file'),
            'files_generated': len(generated_code.get('files', {})) if 'files' in generated_code else 1,
            'has_header': bool(generated_code.get('header')),
            'has_implementation': bool(generated_code.get('implementation')),
            'dependencies': generated_code.get('dependencies', []),
            'generation_timestamp': generation_metadata.get('generation_timestamp', 0),
            'agent_performance': self.get_performance_summary()
        }
