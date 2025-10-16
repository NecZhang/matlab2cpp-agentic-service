"""
Enhanced Conversion Planner Agent

This agent provides advanced conversion planning with:
- Multi-file project structure planning
- Optimal compilation order determination
- Namespace strategy selection
- Include dependency management
- Cross-file coordination planning
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Set, Tuple, Callable
from pathlib import Path
from collections import defaultdict

from ..base.langgraph_agent import BaseLangGraphAgent, AgentConfig
from ....infrastructure.tools.llm_client import LLMClient
from ....infrastructure.state.conversion_state import ConversionState


class ConversionPlanner(BaseLangGraphAgent):
    """
    Conversion planner with multi-file project support.
    
    Capabilities:
    - Multi-file project structure planning
    - Optimal compilation order determination
    - Namespace strategy selection
    - Include dependency management
    - Cross-file coordination planning
    """
    
    def __init__(self, config: AgentConfig, llm_client: LLMClient):
        super().__init__(config, llm_client)
        
        # Enhanced tools
        self.tools = [
            "project_structure_planner",
            "compilation_order_optimizer",
            "namespace_strategy_selector",
            "include_manager",
            "llm_analysis"
        ]
        
        # Planning strategies
        self.namespace_strategies = self._initialize_namespace_strategies()
        self.include_templates = self._initialize_include_templates()
        
        self.logger.info(f"Initialized Conversion Planner: {config.name}")
    
    def create_node(self) -> Callable[[ConversionState], ConversionState]:
        """Create the LangGraph node function for conversion planning."""
        async def plan_node(state: ConversionState) -> ConversionState:
            return await self.create_conversion_plan(state)
        return plan_node
    
    def get_tools(self) -> List[Any]:
        """Get available tools for conversion planning."""
        return [
            # Add planning tools as needed
        ]
    
    def _initialize_namespace_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize namespace strategies for different project types."""
        return {
            "file_based": {
                "description": "Each file gets its own namespace",
                "namespace_pattern": "{file_name}",
                "pros": ["Clear separation", "Easy to understand"],
                "cons": ["Potential conflicts", "Verbose usage"]
            },
            "function_based": {
                "description": "Namespace based on function categories",
                "namespace_pattern": "{category}",
                "pros": ["Logical grouping", "Clean API"],
                "cons": ["Requires categorization", "Complex planning"]
            },
            "unified": {
                "description": "Single namespace for entire project",
                "namespace_pattern": "project",
                "pros": ["Simple usage", "No conflicts"],
                "cons": ["Global pollution", "No organization"]
            },
            "hierarchical": {
                "description": "Hierarchical namespace structure",
                "namespace_pattern": "project::{module}",
                "pros": ["Best organization", "Scalable"],
                "cons": ["Complex structure", "Planning overhead"]
            }
        }
    
    def _initialize_include_templates(self) -> Dict[str, List[str]]:
        """Initialize include templates for different types of functionality."""
        return {
            "standard": [
                "#include <iostream>",
                "#include <vector>",
                "#include <string>",
                "#include <memory>",
                "#include <algorithm>"
            ],
            "eigen": [
                "#include <Eigen/Dense>",
                "#include <Eigen/Sparse>",
                "#include <Eigen/Core>"
            ],
            "opencv": [
                "#include <opencv2/opencv.hpp>",
                "#include <opencv2/imgproc.hpp>",
                "#include <opencv2/imgcodecs.hpp>"
            ],
            "mathematical": [
                "#include <cmath>",
                "#include <numeric>",
                "#include <complex>"
            ],
            "file_io": [
                "#include <fstream>",
                "#include <sstream>",
                "#include <filesystem>"
            ]
        }
    
    async def plan_conversion(self, matlab_analysis: Dict[str, Any], state: ConversionState) -> ConversionState:
        """
        Plan conversion for MATLAB project with advanced multi-file support.
        
        Args:
            matlab_analysis: Results from MATLAB analysis
            state: Current conversion state
            
        Returns:
            Updated state with conversion plan
        """
        start_time = time.time()
        
        try:
            self.logger.info("Starting enhanced conversion planning...")
            
            # Phase 1: Analyze project structure
            project_structure = self._analyze_project_structure(matlab_analysis)
            
            # Phase 2: Plan file organization
            file_organization = self._plan_file_organization(project_structure, matlab_analysis)
            
            # Phase 3: Determine compilation order
            compilation_order = self._optimize_compilation_order(project_structure, file_organization)
            
            # Phase 4: Plan namespace strategy
            namespace_strategy = self._plan_namespace_strategy(project_structure)
            
            # Phase 5: Plan include dependencies
            include_dependencies = self._plan_include_dependencies(project_structure, file_organization)
            support_files = self._plan_support_files(project_structure)
            
            # Phase 6: Create comprehensive conversion plan
            conversion_plan = {
                'project_structure': project_structure,
                'file_organization': file_organization,
                'compilation_order': compilation_order,
                'namespace_strategy': namespace_strategy,
                'include_dependencies': include_dependencies,
                'coordination_strategy': self._plan_coordination_strategy(project_structure),
                'optimization_hints': self._generate_optimization_hints(project_structure),
                'conversion_mode': self._determine_conversion_mode(project_structure),
                'planning_timestamp': time.time(),
                'support_files': support_files,
                'function_signatures': matlab_analysis.get('function_signatures', {})
            }
            
            # Update state
            state["conversion_plan"] = conversion_plan
            
            # Update memory
            self.update_memory("planning_count", 
                             (self.get_memory("planning_count", "short_term") or 0) + 1, 
                             "short_term")
            
            # Track performance
            execution_time = time.time() - start_time
            self.track_performance("plan_conversion", start_time, time.time(), True, 
                                 {"files_planned": project_structure['file_count']})
            
            self.logger.info(f"Enhanced conversion planning complete: "
                           f"{project_structure['file_count']} files, "
                           f"{execution_time:.2f}s")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Enhanced conversion planning failed: {e}")
            self.track_performance("plan_conversion", start_time, time.time(), False, 
                                 {"error": str(e)})
            raise
    
    def _analyze_project_structure(self, matlab_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze project structure and dependencies."""
        file_analyses = matlab_analysis.get('file_analyses', [])
        dependency_map = matlab_analysis.get('dependency_map', {})
        function_call_tree = matlab_analysis.get('function_call_tree', {})
        
        # Analyze file relationships
        file_dependencies = {}
        for file_analysis in file_analyses:
            file_name = file_analysis['file_name']
            functions = file_analysis.get('functions', [])
            function_calls = file_analysis.get('function_calls', {})
            
            # Find dependencies for this file
            file_deps = set()
            for func_name, called_functions in function_calls.items():
                for called_func in called_functions:
                    # Find which file defines this function
                    if called_func in dependency_map:
                        dep_file = Path(dependency_map[called_func]['defined_in']).name
                        if dep_file != file_name:
                            file_deps.add(dep_file)
            
            file_dependencies[file_name] = list(file_deps)
        
        # Determine project type
        is_multi_file = len(file_analyses) > 1
        has_complex_dependencies = any(len(deps) > 0 for deps in file_dependencies.values())
        
        helper_keywords = ('imdilate', 'imerode', 'imopen', 'imclose', 'bwmorph')
        msfm_keywords = ('msfm(', 'msfm2d1(')
        array_helper_keywords = ('findNonZero', 'linearIndexToSubscripts', 'subscriptsToLinear')
        pointmin_keywords = ('pointmin(')

        requires_image_helpers = False
        requires_msfm_helpers = False
        requires_array_helpers = False
        requires_pointmin_helpers = False
        for file_analysis in file_analyses:
            content = file_analysis.get('content', '')
            if content:
                content_lower = content.lower()
                if any(keyword in content_lower for keyword in helper_keywords):
                    requires_image_helpers = True
                if any(keyword in content for keyword in msfm_keywords):
                    requires_msfm_helpers = True
                if any(keyword in content for keyword in array_helper_keywords):
                    requires_array_helpers = True
                if any(keyword in content_lower for keyword in pointmin_keywords):
                    requires_pointmin_helpers = True
        
        return {
            'file_count': len(file_analyses),
            'is_multi_file': is_multi_file,
            'has_complex_dependencies': has_complex_dependencies,
            'file_dependencies': file_dependencies,
            'dependency_map': dependency_map,
            'function_call_tree': function_call_tree,
            'complexity_level': matlab_analysis.get('complexity_assessment', {}).get('complexity_level', 'simple'),
            'requires_image_helpers': requires_image_helpers,
            'requires_msfm_helpers': requires_msfm_helpers,
            'requires_array_helpers': requires_array_helpers,
            'requires_pointmin_helpers': requires_pointmin_helpers
        }
    
    def _plan_file_organization(self, project_structure: Dict[str, Any], matlab_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Plan file organization for C++ project."""
        file_count = project_structure['file_count']
        is_multi_file = project_structure['is_multi_file']
        file_signatures_map = {
            analysis.get('file_name'): analysis.get('function_signatures', {})
            for analysis in matlab_analysis.get('file_analyses', [])
        }
        
        if not is_multi_file:
            # Single file project
            return {
                'organization_type': 'single_file',
                'cpp_files': ['main.cpp'],
                'header_files': ['main.h'],
                'file_mapping': {'main.m': {'cpp_file': 'main.cpp', 'header_file': 'main.h'}}
            }
        
        # Multi-file project
        file_mapping = {}
        cpp_files = []
        header_files = []
        
        for file_name in project_structure['file_dependencies'].keys():
            base_name = file_name.replace('.m', '')
            cpp_name = self._convert_to_cpp_naming(base_name)
            
            cpp_file = f"{cpp_name}.cpp"
            header_file = f"{cpp_name}.h"
            
            cpp_files.append(cpp_file)
            header_files.append(header_file)
            
            file_mapping[file_name] = {
                'cpp_file': cpp_file,
                'header_file': header_file,
                'namespace': cpp_name,
                'signatures': file_signatures_map.get(file_name, {})
            }
        
        return {
            'organization_type': 'multi_file',
            'cpp_files': cpp_files,
            'header_files': header_files,
            'file_mapping': file_mapping
        }
    
    def _optimize_compilation_order(self, project_structure: Dict[str, Any], file_organization: Dict[str, Any]) -> List[str]:
        """Optimize compilation order using dependency analysis."""
        if not project_structure['is_multi_file']:
            return ['main.cpp']

        file_dependencies = project_structure['file_dependencies']
        file_mapping = file_organization['file_mapping']
        
        # Build dependency graph for files
        file_dep_graph = {}
        for file_name, deps in file_dependencies.items():
            cpp_file = file_mapping[file_name]['cpp_file']
            file_dep_graph[cpp_file] = [file_mapping[dep]['cpp_file'] for dep in deps if dep in file_mapping]
        
        # Topological sort for compilation order
        in_degree = {file: 0 for file in file_dep_graph}
        for file, deps in file_dep_graph.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[dep] += 1
        
        from collections import deque
        queue = deque([file for file, degree in in_degree.items() if degree == 0])
        result = []
        
        while queue:
            file = queue.popleft()
            result.append(file)
            
            for dep in file_dep_graph.get(file, []):
                if dep in in_degree:
                    in_degree[dep] -= 1
                    if in_degree[dep] == 0:
                        queue.append(dep)
        
        return result
    
    def _plan_namespace_strategy(self, project_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Plan namespace strategy based on project characteristics."""
        file_count = project_structure['file_count']
        complexity_level = project_structure['complexity_level']
        has_complex_dependencies = project_structure['has_complex_dependencies']
        
        # Select strategy based on project characteristics
        if file_count == 1:
            strategy_name = "unified"
        elif file_count <= 3 and not has_complex_dependencies:
            strategy_name = "file_based"
        elif complexity_level == "complex" or has_complex_dependencies:
            strategy_name = "hierarchical"
        else:
            strategy_name = "file_based"
        
        strategy = self.namespace_strategies[strategy_name].copy()
        strategy['selected_strategy'] = strategy_name
        
        return strategy
    
    def _plan_include_dependencies(self, project_structure: Dict[str, Any], 
                                 file_organization: Dict[str, Any]) -> Dict[str, List[str]]:
        """Plan include dependencies for each file."""
        include_dependencies = {}
        
        needs_image_helpers = project_structure.get('requires_image_helpers', False)
        helper_include = '#include "matlab_image_helpers.h"' if needs_image_helpers else None
        
        for file_name, mapping in file_organization.get('file_mapping', {}).items():
            cpp_file = mapping['cpp_file']
            header_file = mapping['header_file']
            
            # Start with standard includes
            includes = self.include_templates['standard'].copy()
            if helper_include and helper_include not in includes:
                includes.append(helper_include)
            
            # Add project-specific includes
            file_deps = project_structure['file_dependencies'].get(file_name, [])
            for dep_file in file_deps:
                dep_mapping = file_organization['file_mapping'].get(dep_file, {})
                if dep_mapping:
                    includes.append(f'#include "{dep_mapping["header_file"]}"')
            
            include_dependencies[cpp_file] = includes
        
        return include_dependencies
    
    def _plan_support_files(self, project_structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Plan additional support files required by the project."""
        support_files: List[Dict[str, Any]] = []
        
        if project_structure.get('requires_image_helpers', False):
            support_files.append({
                'id': 'matlab_image_helpers',
                'header': 'matlab_image_helpers.h',
                'source': 'matlab_image_helpers.cpp',
                'description': 'Helper functions providing MATLAB-style morphological operations (e.g., imdilate, imerode).'
            })
        if project_structure.get('requires_msfm_helpers', False):
            support_files.append({
                'id': 'msfm_helpers',
                'header': 'msfm_helpers.h',
                'source': 'msfm_helpers.cpp',
                'description': 'Helper overloads for MSFM interfaces (vector/matrix source point conversions).'
            })
        if project_structure.get('requires_array_helpers', False):
            support_files.append({
                'id': 'matlab_array_utils',
                'header': 'matlab_array_utils.h',
                'source': 'matlab_array_utils.cpp',
                'description': 'Utility functions for MATLAB-style linear indexing conversions.'
            })
        if project_structure.get('requires_pointmin_helpers', False):
            support_files.append({
                'id': 'pointmin_helpers',
                'header': 'pointmin_helpers.h',
                'source': 'pointmin_helpers.cpp',
                'description': 'Wrapper helpers for pointmin 2D/3D outputs.'
            })
        
        return support_files
    
    def _plan_coordination_strategy(self, project_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Plan coordination strategy for multi-file projects."""
        if not project_structure['is_multi_file']:
            return {'strategy': 'none', 'coordination_needed': False}
        
        file_count = project_structure['file_count']
        has_complex_dependencies = project_structure['has_complex_dependencies']
        
        if file_count <= 3 and not has_complex_dependencies:
            return {
                'strategy': 'simple_sequential',
                'coordination_needed': True,
                'coordination_level': 'low'
            }
        elif has_complex_dependencies:
            return {
                'strategy': 'dependency_aware',
                'coordination_needed': True,
                'coordination_level': 'high'
            }
        else:
            return {
                'strategy': 'parallel_with_coordination',
                'coordination_needed': True,
                'coordination_level': 'medium'
            }
    
    def _generate_optimization_hints(self, project_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimization hints based on project analysis."""
        hints = {
            'compilation_optimization': [],
            'runtime_optimization': [],
            'memory_optimization': [],
            'code_organization': []
        }
        
        complexity_level = project_structure['complexity_level']
        
        if complexity_level == 'complex':
            hints['compilation_optimization'].extend([
                'Consider using forward declarations',
                'Minimize header dependencies',
                'Use include guards'
            ])
            hints['runtime_optimization'].extend([
                'Optimize hot paths',
                'Consider caching strategies',
                'Profile performance bottlenecks'
            ])
        
        if project_structure['has_complex_dependencies']:
            hints['code_organization'].extend([
                'Use clear interface definitions',
                'Minimize circular dependencies',
                'Consider facade pattern for complex APIs'
            ])
        
        return hints
    
    def _determine_conversion_mode(self, project_structure: Dict[str, Any]) -> str:
        """Determine optimal conversion mode based on project characteristics."""
        file_count = project_structure['file_count']
        complexity_level = project_structure['complexity_level']
        has_complex_dependencies = project_structure['has_complex_dependencies']
        
        if file_count == 1 and complexity_level == 'simple':
            return 'single_file_simple'
        elif file_count == 1 and complexity_level in ['moderate', 'complex']:
            return 'single_file_complex'
        elif file_count > 1 and not has_complex_dependencies:
            return 'multi_file_simple'
        else:
            return 'multi_file_complex'
    
    def _convert_to_cpp_naming(self, matlab_name: str) -> str:
        """Convert MATLAB naming to C++ naming convention."""
        # Convert to snake_case or camelCase
        import re
        
        # Handle common MATLAB naming patterns
        if '_' in matlab_name:
            # Already has underscores, keep as is
            return matlab_name.lower()
        else:
            # Convert camelCase or PascalCase to snake_case
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', matlab_name)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    
    async def get_planning_summary(self, conversion_plan: Dict[str, Any]) -> str:
        """Generate human-readable planning summary."""
        summary = f"📋 Enhanced Conversion Planning Summary\n"
        
        project_structure = conversion_plan['project_structure']
        summary += f"Project type: {'Multi-file' if project_structure['is_multi_file'] else 'Single-file'}\n"
        summary += f"Files to convert: {project_structure['file_count']}\n"
        summary += f"Complexity level: {project_structure['complexity_level']}\n"
        
        file_organization = conversion_plan['file_organization']
        summary += f"Organization: {file_organization['organization_type']}\n"
        summary += f"C++ files: {len(file_organization['cpp_files'])}\n"
        summary += f"Header files: {len(file_organization['header_files'])}\n"
        
        namespace_strategy = conversion_plan['namespace_strategy']
        summary += f"Namespace strategy: {namespace_strategy['selected_strategy']}\n"
        
        coordination = conversion_plan['coordination_strategy']
        if coordination['coordination_needed']:
            summary += f"Coordination needed: {coordination['coordination_level']}\n"
        
        conversion_mode = conversion_plan['conversion_mode']
        summary += f"Conversion mode: {conversion_mode}\n"
        
        return summary
