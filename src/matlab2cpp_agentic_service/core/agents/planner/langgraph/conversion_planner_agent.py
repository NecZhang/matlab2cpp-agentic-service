"""
LangGraph-Native Conversion Planner Agent

This module implements a truly LangGraph-native conversion planner agent
that creates comprehensive conversion plans using LangGraph tools and memory.
"""

import time
from typing import Dict, Any, List, Callable
from pathlib import Path
from loguru import logger

from matlab2cpp_agentic_service.infrastructure.state.conversion_state import ConversionState, ConversionStatus, add_processing_time, update_state_status
from matlab2cpp_agentic_service.infrastructure.tools.llm_client import LLMClient
from matlab2cpp_agentic_service.infrastructure.tools.langgraph_tools import LLMAnalysisTool, ToolRegistry
from ...base.langgraph_agent import BaseLangGraphAgent, AgentConfig


class LangGraphConversionPlannerAgent(BaseLangGraphAgent):
    """
    LangGraph-native conversion planner agent.
    
    This agent creates comprehensive conversion plans using LangGraph tools
    and maintains planning history in memory for iterative improvements.
    """
    
    def __init__(self, config: AgentConfig, llm_client: LLMClient):
        """Initialize the LangGraph conversion planner agent."""
        super().__init__(config, llm_client)
        
        # Initialize tools
        self.tool_registry = ToolRegistry()
        self.llm_analysis_tool = LLMAnalysisTool(llm_client)
        
        # Register tools
        self.tool_registry.register_tool("llm_analysis", self.llm_analysis_tool)
        
        self.logger.info(f"Initialized LangGraph Conversion Planner Agent: {config.name}")
    
    def create_node(self) -> Callable[[ConversionState], ConversionState]:
        """
        Create the LangGraph node function for conversion planning.
        
        Returns:
            Callable that takes and returns ConversionState
        """
        def plan_conversion_node(state: ConversionState) -> ConversionState:
            """LangGraph node function for conversion planning."""
            start_time = time.time()
            
            try:
                self.logger.info("Starting LangGraph-native conversion planning...")
                
                # Update state status
                state = update_state_status(state, ConversionStatus.PLANNING)
                
                # Get MATLAB analysis from state
                matlab_analysis = state.get("matlab_analysis")
                if not matlab_analysis:
                    raise ValueError("MATLAB analysis not found in state")
                
                # Check if we have cached plan in memory
                cache_key = f"conversion_plan_{hash(str(matlab_analysis))}"
                cached_plan = self.get_memory(cache_key, "long_term")
                
                if cached_plan and not self._should_refresh_plan(cached_plan, state):
                    self.logger.info("Using cached conversion plan from memory")
                    conversion_plan = cached_plan
                else:
                    self.logger.info("Creating fresh conversion plan")
                    conversion_plan = self._create_conversion_plan(matlab_analysis, state)
                    
                    # Cache the plan in long-term memory
                    self.update_memory(cache_key, conversion_plan, "long_term")
                
                # Add multi-file project structure planning if needed
                if state.get("is_multi_file", False):
                    project_structure = self._create_multi_file_structure(matlab_analysis, state)
                    conversion_plan['project_structure_plan'] = project_structure
                    state["project_structure_plan"] = project_structure
                    self.logger.info(f"Generated multi-file project structure with {len(project_structure['cpp_files'])} C++ files")
                
                # Update agent memory with planning context
                self.update_memory("last_plan", conversion_plan, "short_term")
                self.update_memory("planning_count", 
                                 (self.get_memory("planning_count", "short_term") or 0) + 1, 
                                 "short_term")
                self.update_memory("is_multi_file", state.get("is_multi_file", False), "context")
                
                # Update state with planning results
                state["conversion_plan"] = conversion_plan
                
                # Update state with agent memory
                state = self.update_state_with_result(state, conversion_plan, "conversion_planning")
                
                # Log success
                duration = time.time() - start_time
                self.logger.info(f"Conversion planning completed successfully in {duration:.2f}s")
                self.logger.info(f"Plan includes {len(conversion_plan.get('cpp_files', []))} C++ files")
                
                # Track performance
                self.track_performance("conversion_planning", start_time, time.time(), True, {
                    "is_multi_file": state.get("is_multi_file", False),
                    "cpp_files_count": len(conversion_plan.get('cpp_files', [])),
                    "complexity_level": matlab_analysis.get('complexity_assessment', 'Unknown')
                })
                
            except Exception as e:
                self.logger.error(f"Error in conversion planning: {e}")
                state["error_message"] = f"Conversion planning failed: {str(e)}"
                state = update_state_status(state, ConversionStatus.FAILED)
                
                # Track failure
                self.track_performance("conversion_planning", start_time, time.time(), False, {
                    "error": str(e)
                })
            
            # Record processing time
            duration = time.time() - start_time
            state = add_processing_time(state, "conversion_planning", duration)
            
            return state
        
        return plan_conversion_node
    
    def get_tools(self) -> List[Any]:
        """Get available tools for this agent."""
        return [self.llm_analysis_tool]
    
    def _create_conversion_plan(self, matlab_analysis: Dict[str, Any], state: ConversionState) -> Dict[str, Any]:
        """
        Create comprehensive conversion plan using LangGraph tools.
        
        Args:
            matlab_analysis: MATLAB analysis results
            state: Current conversion state
            
        Returns:
            Comprehensive conversion plan
        """
        self.logger.debug("Creating conversion plan using LangGraph tools")
        
        # Extract key information from analysis
        files_analyzed = matlab_analysis.get('files_analyzed', 1)
        total_functions = matlab_analysis.get('total_functions', 0)
        complexity_level = matlab_analysis.get('complexity_assessment', 'Medium')
        project_understanding = matlab_analysis.get('project_understanding', {})
        dependency_map = matlab_analysis.get('dependency_map', {})
        
        # Get conversion requirements from state
        request = state["request"]
        conversion_mode = request.conversion_mode
        target_quality = request.target_quality_score
        max_turns = request.max_optimization_turns
        
        # Create base conversion plan
        conversion_plan = {
            'conversion_mode': conversion_mode,
            'target_quality_score': target_quality,
            'max_optimization_turns': max_turns,
            'complexity_level': complexity_level,
            'files_analyzed': files_analyzed,
            'total_functions': total_functions,
            'cpp_files': [],
            'header_files': [],
            'dependencies': [],
            'compilation_requirements': {},
            'optimization_strategies': [],
            'conversion_guidelines': {},
            'quality_targets': {},
            'planning_timestamp': time.time()
        }
        
        # Determine C++ file structure
        if files_analyzed == 1:
            # Single file project
            conversion_plan['cpp_files'] = ['main.cpp']
            conversion_plan['header_files'] = ['main.h']
            conversion_plan['project_type'] = 'single_file'
        else:
            # Multi-file project
            conversion_plan['project_type'] = 'multi_file'
            self._plan_multi_file_structure(conversion_plan, matlab_analysis)
        
        # Set conversion guidelines based on mode
        conversion_plan['conversion_guidelines'] = self._create_conversion_guidelines(conversion_mode, complexity_level)
        
        # Set quality targets
        conversion_plan['quality_targets'] = self._create_quality_targets(conversion_mode, target_quality)
        
        # Set optimization strategies
        conversion_plan['optimization_strategies'] = self._create_optimization_strategies(complexity_level, max_turns)
        
        # Set dependencies and compilation requirements
        conversion_plan['dependencies'] = self._determine_dependencies(project_understanding, matlab_analysis)
        conversion_plan['compilation_requirements'] = self._create_compilation_requirements(conversion_mode, complexity_level)
        
        # Add domain-specific planning
        domain = project_understanding.get('domain', 'General')
        conversion_plan['domain_specific_planning'] = self._create_domain_specific_planning(domain, project_understanding)
        
        # Use LLM for advanced planning if available
        if self.llm_client:
            llm_planning = self._get_llm_planning_recommendations(matlab_analysis, conversion_plan)
            if llm_planning:
                conversion_plan['llm_recommendations'] = llm_planning
        
        self.logger.info(f"Created conversion plan: {len(conversion_plan['cpp_files'])} C++ files, {complexity_level} complexity")
        
        return conversion_plan
    
    def _plan_multi_file_structure(self, conversion_plan: Dict[str, Any], matlab_analysis: Dict[str, Any]):
        """Plan multi-file project structure."""
        file_analyses = matlab_analysis.get('file_analyses', [])
        dependency_map = matlab_analysis.get('dependency_map', {})
        
        cpp_files = []
        header_files = []
        
        # Create files based on MATLAB structure
        for file_analysis in file_analyses:
            file_path = Path(file_analysis['file_path'])
            base_name = file_path.stem
            
            # Convert MATLAB file to C++ files
            cpp_file = f"{base_name}.cpp"
            header_file = f"{base_name}.h"
            
            cpp_files.append(cpp_file)
            header_files.append(header_file)
        
        # Add main file if not present
        if 'main.cpp' not in cpp_files:
            cpp_files.insert(0, 'main.cpp')
            header_files.insert(0, 'main.h')
        
        conversion_plan['cpp_files'] = cpp_files
        conversion_plan['header_files'] = header_files
        
        # Create compilation order based on dependencies
        compilation_order = self._determine_compilation_order(dependency_map, cpp_files)
        conversion_plan['compilation_order'] = compilation_order
    
    def _create_conversion_guidelines(self, conversion_mode: str, complexity_level: str) -> Dict[str, Any]:
        """Create conversion guidelines based on mode and complexity."""
        if conversion_mode == "faithful":
            return {
                'priority': 'algorithmic_fidelity',
                'numerical_precision': 'exact',
                'indexing': 'preserve_matlab_logic',
                'optimization': 'minimal',
                'style': 'matlab_similar',
                'error_handling': 'basic'
            }
        else:  # result-focused
            return {
                'priority': 'performance_and_maintainability',
                'numerical_precision': 'stable',
                'indexing': 'cpp_conventions',
                'optimization': 'aggressive',
                'style': 'modern_cpp',
                'error_handling': 'comprehensive'
            }
    
    def _create_quality_targets(self, conversion_mode: str, target_quality: float) -> Dict[str, float]:
        """Create quality targets for different categories."""
        base_targets = {
            'algorithmic': target_quality,
            'performance': target_quality,
            'error_handling': target_quality,
            'style': target_quality,
            'maintainability': target_quality,
            'security': target_quality
        }
        
        if conversion_mode == "faithful":
            # Adjust targets for faithful mode
            base_targets['algorithmic'] = min(target_quality + 1.0, 10.0)
            base_targets['performance'] = max(target_quality - 1.0, 0.0)
        else:  # result-focused
            # Adjust targets for result-focused mode
            base_targets['performance'] = min(target_quality + 1.0, 10.0)
            base_targets['maintainability'] = min(target_quality + 1.0, 10.0)
        
        return base_targets
    
    def _create_optimization_strategies(self, complexity_level: str, max_turns: int) -> List[str]:
        """Create optimization strategies based on complexity and max turns."""
        strategies = []
        
        if complexity_level == "High":
            strategies.extend([
                "memory_optimization",
                "algorithmic_optimization",
                "parallelization_analysis"
            ])
        elif complexity_level == "Medium":
            strategies.extend([
                "performance_optimization",
                "code_structure_optimization"
            ])
        else:
            strategies.extend([
                "basic_optimization"
            ])
        
        if max_turns > 1:
            strategies.append("iterative_improvement")
        
        return strategies
    
    def _determine_dependencies(self, project_understanding: Dict[str, Any], matlab_analysis: Dict[str, Any]) -> List[str]:
        """Determine C++ dependencies based on analysis."""
        dependencies = ["Eigen3"]  # Always include Eigen
        
        # Add dependencies based on domain
        domain = project_understanding.get('domain', 'General')
        if domain == 'Signal processing':
            dependencies.extend(['FFTW3', 'OpenMP'])
        elif domain == 'Image processing':
            dependencies.extend(['OpenCV', 'libpng', 'libjpeg'])
        elif domain == 'Machine learning':
            dependencies.extend(['mlpack', 'Armadillo'])
        elif domain == 'Optimization':
            dependencies.extend(['NLopt', 'Ceres Solver'])
        
        # Add dependencies based on numerical calls
        numerical_calls = matlab_analysis.get('numerical_calls_used', [])
        if any(call in numerical_calls for call in ['fft', 'ifft']):
            dependencies.append('FFTW3')
        if any(call in numerical_calls for call in ['imread', 'imwrite']):
            dependencies.extend(['OpenCV', 'libpng'])
        
        return list(set(dependencies))  # Remove duplicates
    
    def _create_compilation_requirements(self, conversion_mode: str, complexity_level: str) -> Dict[str, Any]:
        """Create compilation requirements."""
        requirements = {
            'cpp_standard': 'C++17',
            'compiler_flags': ['-O2', '-Wall', '-Wextra'],
            'include_directories': [],
            'library_directories': [],
            'libraries': []
        }
        
        if conversion_mode == "result-focused":
            requirements['compiler_flags'].extend(['-O3', '-march=native'])
        
        if complexity_level == "High":
            requirements['compiler_flags'].extend(['-fopenmp'])
            requirements['libraries'].append('gomp')
        
        return requirements
    
    def _create_domain_specific_planning(self, domain: str, project_understanding: Dict[str, Any]) -> Dict[str, Any]:
        """Create domain-specific planning recommendations."""
        domain_planning = {
            'domain': domain,
            'specialized_libraries': [],
            'optimization_focus': [],
            'testing_strategies': [],
            'documentation_requirements': []
        }
        
        if domain == 'Signal processing':
            domain_planning['specialized_libraries'] = ['FFTW3', 'GSL']
            domain_planning['optimization_focus'] = ['fft_optimization', 'filter_design']
            domain_planning['testing_strategies'] = ['frequency_response_testing', 'noise_analysis']
        elif domain == 'Image processing':
            domain_planning['specialized_libraries'] = ['OpenCV', 'ITK']
            domain_planning['optimization_focus'] = ['vectorization', 'memory_access_patterns']
            domain_planning['testing_strategies'] = ['image_quality_metrics', 'performance_benchmarks']
        elif domain == 'Linear algebra':
            domain_planning['specialized_libraries'] = ['Eigen3', 'LAPACK']
            domain_planning['optimization_focus'] = ['matrix_operations', 'solver_selection']
            domain_planning['testing_strategies'] = ['numerical_accuracy_tests', 'condition_number_analysis']
        
        return domain_planning
    
    def _get_llm_planning_recommendations(self, matlab_analysis: Dict[str, Any], conversion_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Get LLM-based planning recommendations."""
        try:
            # Create a summary for LLM analysis
            analysis_summary = {
                'files_analyzed': matlab_analysis.get('files_analyzed', 1),
                'complexity_level': matlab_analysis.get('complexity_assessment', 'Medium'),
                'domain': matlab_analysis.get('project_understanding', {}).get('domain', 'General'),
                'key_algorithms': matlab_analysis.get('project_understanding', {}).get('key_algorithms', []),
                'conversion_mode': conversion_plan.get('conversion_mode', 'result-focused')
            }
            
            # Use LLM analysis tool for planning recommendations
            llm_result = self.llm_analysis_tool(
                str(analysis_summary),
                analysis_summary,
                "complexity"  # Use complexity analysis type for planning
            )
            
            if llm_result.success:
                return llm_result.data
            else:
                self.logger.warning(f"LLM planning analysis failed: {llm_result.error}")
                return {}
                
        except Exception as e:
            self.logger.warning(f"Error in LLM planning recommendations: {e}")
            return {}
    
    def _create_multi_file_structure(self, matlab_analysis: Dict[str, Any], state: ConversionState) -> Dict[str, Any]:
        """Create multi-file project structure plan."""
        file_analyses = matlab_analysis.get('file_analyses', [])
        dependency_map = matlab_analysis.get('dependency_map', {})
        
        cpp_files = []
        header_files = []
        file_mapping = {}
        
        # Create file structure
        for file_analysis in file_analyses:
            file_path = Path(file_analysis['file_path'])
            base_name = file_path.stem
            
            # Convert to C++ naming convention
            cpp_name = self._convert_to_cpp_naming(base_name)
            
            cpp_file = f"{cpp_name}.cpp"
            header_file = f"{cpp_name}.h"
            
            cpp_files.append(cpp_file)
            header_files.append(header_file)
            
            file_mapping[str(file_path)] = {
                'cpp_file': cpp_file,
                'header_file': header_file,
                'namespace': cpp_name,
                'functions': file_analysis['parsed_structure'].get('functions', [])
            }
        
        # Determine compilation order
        compilation_order = self._determine_compilation_order(dependency_map, cpp_files)
        
        # Create include dependencies
        include_dependencies = self._create_include_dependencies(file_mapping, dependency_map)
        
        return {
            'cpp_files': cpp_files,
            'header_files': header_files,
            'file_mapping': file_mapping,
            'compilation_order': compilation_order,
            'include_dependencies': include_dependencies,
            'namespace_strategy': 'file_based',
            'project_structure': 'modular'
        }
    
    def _convert_to_cpp_naming(self, matlab_name: str) -> str:
        """Convert MATLAB naming to C++ naming convention."""
        # Convert to snake_case or CamelCase based on preference
        import re
        
        # Handle common MATLAB naming patterns
        name = matlab_name.lower()
        
        # Replace common patterns
        name = re.sub(r'[^a-z0-9_]', '_', name)
        name = re.sub(r'_+', '_', name)
        name = name.strip('_')
        
        return name or 'unnamed'
    
    def _determine_compilation_order(self, dependency_map: Dict[str, Any], cpp_files: List[str]) -> List[str]:
        """Determine compilation order based on dependencies."""
        # Simple topological sort based on dependencies
        # For now, return files in original order
        # TODO: Implement proper topological sorting
        return cpp_files
    
    def _create_include_dependencies(self, file_mapping: Dict[str, Any], dependency_map: Dict[str, Any]) -> Dict[str, List[str]]:
        """Create include dependencies between files."""
        include_deps = {}
        
        for matlab_file, cpp_info in file_mapping.items():
            cpp_file = cpp_info['cpp_file']
            includes = []
            
            # Add dependencies based on function calls
            functions = cpp_info.get('functions', [])
            for func_name in functions:
                if func_name in dependency_map:
                    called_funcs = dependency_map[func_name].get('calls', [])
                    for called_func in called_funcs:
                        # Find which file contains the called function
                        for other_file, other_info in file_mapping.items():
                            if called_func in other_info.get('functions', []):
                                header_file = other_info['header_file']
                                if header_file not in includes:
                                    includes.append(header_file)
            
            include_deps[cpp_file] = includes
        
        return include_deps
    
    def _should_refresh_plan(self, cached_plan: Dict[str, Any], state: ConversionState) -> bool:
        """Determine if cached plan should be refreshed."""
        # Refresh if plan is older than 30 minutes
        plan_time = cached_plan.get('planning_timestamp', 0)
        return time.time() - plan_time > 1800
    
    def get_planning_summary(self, state: ConversionState) -> Dict[str, Any]:
        """Get a summary of the planning results from state."""
        conversion_plan = state.get("conversion_plan", {})
        
        return {
            'project_type': conversion_plan.get('project_type', 'unknown'),
            'cpp_files_count': len(conversion_plan.get('cpp_files', [])),
            'complexity_level': conversion_plan.get('complexity_level', 'Unknown'),
            'conversion_mode': conversion_plan.get('conversion_mode', 'result-focused'),
            'dependencies': conversion_plan.get('dependencies', []),
            'optimization_strategies': conversion_plan.get('optimization_strategies', []),
            'planning_timestamp': conversion_plan.get('planning_timestamp', 0),
            'agent_performance': self.get_performance_summary()
        }
