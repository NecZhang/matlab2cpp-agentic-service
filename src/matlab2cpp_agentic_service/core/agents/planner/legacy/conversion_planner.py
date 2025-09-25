"""
Conversion Planner (LLM-first)
==============================

This agent generates a plan for converting MATLAB code to C++.  It always
attempts to use an LLM to produce a structured JSON plan based on a
summary of the analysis and a heuristic baseline.  If the LLM call
fails or returns invalid JSON, it falls back to the heuristic plan.
"""

from __future__ import annotations
from typing import Dict, Any, List, Set, Tuple
import json
from pathlib import Path

class ConversionPlannerAgent:
    def __init__(self, llm_client: Any | None = None) -> None:
        """
        Args:
            llm_client: A client implementing `get_completion(prompt: str) -> str`.
                This agent will always attempt to use the model to generate
                the plan.  If `llm_client` is None, it falls back to the
                heuristic plan immediately.
        """
        self.llm_client = llm_client

    def plan(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a conversion plan.  This planner always tries to use the LLM.
        If the LLM call fails or returns invalid JSON, it falls back to a
        deterministic heuristic plan.

        Args:
            analysis: The aggregated analysis output from the content analyzer.

        Returns:
            A dictionary with keys: project_structure, cpp_architecture,
            conversion_strategy, dependencies, conversion_steps.
        """
        heuristic = self._heuristic_plan(analysis)
        # Attempt LLM planning if a client is available
        if self.llm_client:
            try:
                prompt = self._create_plan_prompt(analysis, heuristic)
                response = self.llm_client.get_completion(prompt)
                # Parse JSON
                data = json.loads(response.strip())
                required = [
                    'project_structure', 'cpp_architecture',
                    'conversion_strategy', 'dependencies', 'conversion_steps',
                    'algorithmic_mapping', 'data_flow_preservation'
                ]
                if all(k in data for k in required):
                    # Validate and normalise
                    plan: Dict[str, Any] = {}
                    plan['project_structure']  = str(data['project_structure']).strip()
                    plan['cpp_architecture']   = str(data['cpp_architecture']).strip()
                    plan['conversion_strategy']= str(data['conversion_strategy']).strip()
                    deps = data['dependencies']
                    deps_list = [str(d).strip() for d in deps] if isinstance(deps, list) else heuristic['dependencies']
                    plan['dependencies'] = deps_list
                    steps = data['conversion_steps']
                    steps_list = [str(s).strip() for s in steps] if isinstance(steps, list) else heuristic['conversion_steps']
                    plan['conversion_steps'] = steps_list
                    plan['algorithmic_mapping'] = data.get('algorithmic_mapping', {})
                    plan['data_flow_preservation'] = data.get('data_flow_preservation', {})
                    return plan
            except Exception:
                pass  # ignore errors; fallback below
        # If no llm_client or call fails, return heuristic
        return heuristic

    # ----------------------------------------------------------------------
    # Heuristic baseline
    # ----------------------------------------------------------------------
    def _heuristic_plan(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        num_files = analysis.get('files_analyzed', 0)
        proj = analysis.get('project_understanding') or {}
        complexity = getattr(proj, 'complexity_level', 'Low')
        key_alg = getattr(proj, 'key_algorithms', [])
        domain = getattr(proj, 'domain', None)

        if len(key_alg) > 2 or complexity == 'High' or num_files > 3:
            arch = 'classes'
        elif len(key_alg) > 1 or complexity == 'Medium':
            arch = 'modular'
        else:
            arch = 'functions_only'

        deps: List[str] = ['Eigen']
        for alg in key_alg:
            u = alg.upper()
            if 'FFT' in u or 'FILTER' in u: deps.append('FFTW')
            if 'EIGENVALUE' in u or 'CHOLESKY' in u or 'QR' in u or 'SINGULAR' in u:
                if 'Eigen' not in deps: deps.append('Eigen')
            if 'ODE' in u: deps.append('Boost::odeint')
            if 'OPTIMIZATION' in u: deps.append('Ceres Solver')
            if 'MACHINE LEARNING' in u: deps.append('MLPACK')
            if 'IMAGE' in u: deps.append('OpenCV')
        if domain:
            d = domain.lower()
            if 'signal' in d: deps.append('FFTW')
            if 'image' in d: deps.append('OpenCV')
            if 'machine' in d: deps.append('MLPACK')
            if 'optimization' in d: deps.append('Ceres Solver')

        recs = getattr(proj, 'recommendations', [])
        if recs:
            strategy = ' ; '.join(recs)
        else:
            strategy = (
                "Translate MATLAB matrices and arrays to Eigen types. "
                "Prefer RAII and smart pointers. Avoid explicit inverses; use LDLT/LLT "
                "solvers. Select the smallest eigenvalue's eigenvector where applicable."
            )

        steps: List[str] = [
            "Parse all MATLAB functions and determine inputs, outputs and side effects.",
            "Map MATLAB numerical calls to C++ libraries: Eigen, FFTW, OpenCV, Boost/Ceres, MLPACK.",
            "Design C++ interfaces with 0‑based indexing, const references for inputs and RAII via smart pointers.",
            "Implement each MATLAB function as a separate C++ function or class method; group related functions into classes.",
            "Add try/catch blocks around critical operations (e.g. file I/O, solver calls) to handle exceptions gracefully.",
            "Write unit tests comparing C++ outputs against MATLAB on synthetic inputs.",
            "Profile and optimise: avoid explicit inverses, check solver.info(), parallelise loops, and release unused resources."
        ]

        return {
            'project_structure': 'single_file' if num_files == 1 else 'multi_file',
            'cpp_architecture': arch,
            'conversion_strategy': strategy,
            'dependencies': sorted(set(deps)),
            'conversion_steps': steps
        }

    # ----------------------------------------------------------------------
    # Prompt for the LLM
    # ----------------------------------------------------------------------
    def _create_plan_prompt(self, analysis: Dict[str, Any], heuristic: Dict[str, Any]) -> str:
        """
        Summarise the analysis and heuristic plan, then ask the LLM to produce
        a JSON plan.  Explicitly request the keys required to drive the conversion.
        """
        lines: List[str] = []
        lines.append("You are a senior software architect planning to convert a MATLAB project to modern C++. ")
        lines.append("Below is an analysis summary and a heuristic baseline plan. "
                     "Use these as context and output a refined plan in JSON "
                     "with keys: project_structure, cpp_architecture, conversion_strategy, dependencies, conversion_steps, algorithmic_mapping, data_flow_preservation.")
        lines.append("CRITICAL: Focus on preserving the exact algorithmic structure, including nested loops, matrix construction patterns, and mathematical operations.")

        # Summary details
        num_files = analysis.get('files_analyzed', 0)
        lines.append(f"Number of MATLAB files: {num_files}.")
        proj = analysis.get('project_understanding') or {}
        dom = getattr(proj, 'domain', None)
        comp = getattr(proj, 'complexity_level', None)
        key_alg = getattr(proj, 'key_algorithms', [])
        recs = getattr(proj, 'recommendations', [])
        if dom: lines.append(f"Domain: {dom}.")
        if comp: lines.append(f"Complexity: {comp}.")
        if key_alg: lines.append(f"Key algorithms: {', '.join(key_alg)}.")
        if recs: lines.append(f"Analysis recommendations: {', '.join(recs)}.")
        
        # Add detailed algorithmic analysis
        file_analyses = analysis.get('file_analyses', [])
        for i, file_analysis in enumerate(file_analyses):
            analysis_data = file_analysis.get('analysis', {})
            if hasattr(analysis_data, 'algorithmic_structure') and analysis_data.algorithmic_structure:
                lines.append(f"File {i+1} algorithmic structure: {analysis_data.algorithmic_structure}")
            if hasattr(analysis_data, 'pseudocode') and analysis_data.pseudocode:
                lines.append(f"File {i+1} pseudocode: {analysis_data.pseudocode}")
            if hasattr(analysis_data, 'data_flow') and analysis_data.data_flow:
                lines.append(f"File {i+1} data flow: {analysis_data.data_flow}")

        # Heuristic plan summary
        lines.append("\nHeuristic plan:")
        lines.append(f"  - Project structure: {heuristic['project_structure']}")
        lines.append(f"  - C++ architecture: {heuristic['cpp_architecture']}")
        lines.append(f"  - Conversion strategy: {heuristic['conversion_strategy']}")
        lines.append(f"  - Dependencies: {', '.join(heuristic['dependencies'])}")
        lines.append("  - Steps:")
        for step in heuristic['conversion_steps']:
            lines.append(f"    * {step}")

        # Final instruction
        lines.append("\nPlease return a JSON object with keys exactly: "
                     "project_structure, cpp_architecture, conversion_strategy, dependencies, conversion_steps, algorithmic_mapping, data_flow_preservation. "
                     "Make dependencies and conversion_steps lists.  Use the analysis and heuristic as guidance.")
        lines.append("IMPORTANT: The algorithmic_mapping should detail how each MATLAB operation maps to C++ equivalents, "
                     "and data_flow_preservation should explain how to maintain the exact data transformations.")
        
        # Add multi-file project structure information if available
        if 'function_call_tree' in analysis:
            lines.append("\n=== MULTI-FILE PROJECT STRUCTURE ===")
            call_tree = analysis['function_call_tree']
            lines.append("Function call tree:")
            for func, calls in call_tree['call_graph'].items():
                if calls:  # Only show functions that call other functions
                    lines.append(f"  {func} -> {calls}")
            
            lines.append("\nFunction definitions:")
            for func, file in call_tree['defined_functions'].items():
                lines.append(f"  {func} defined in {Path(file).name}")
        
        return "\n".join(lines)

    def plan_multi_file_structure(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Plan the C++ project structure for multi-file MATLAB projects.
        
        Args:
            analysis: The analysis containing function_call_tree and dependency_map
            
        Returns:
            Dictionary with cpp_files, include_dependencies, compilation_order, namespaces
        """
        if 'function_call_tree' not in analysis or 'dependency_map' not in analysis:
            return self._fallback_single_file_structure(analysis)
        
        call_tree = analysis['function_call_tree']
        dependency_map = analysis['dependency_map']
        
        # Determine compilation order based on dependencies
        compilation_order = self._determine_compilation_order(call_tree, dependency_map)
        
        # Plan C++ file organization
        cpp_files = self._plan_cpp_files(call_tree, dependency_map, compilation_order)
        
        # Plan include dependencies
        include_dependencies = self._plan_include_dependencies(cpp_files, call_tree)
        
        # Plan namespaces
        namespaces = self._plan_namespaces(cpp_files, analysis)
        
        return {
            'cpp_files': cpp_files,
            'include_dependencies': include_dependencies,
            'compilation_order': compilation_order,
            'namespaces': namespaces,
            'project_type': 'multi_file'
        }

    def _determine_compilation_order(self, call_tree: Dict[str, Any], dependency_map: Dict[str, Any]) -> List[str]:
        """
        Determine the order in which C++ files should be compiled based on dependencies.
        Dependencies come first, so they can be included by dependent files.
        """
        # Build dependency graph
        deps_graph: Dict[str, Set[str]] = {}
        for func, info in dependency_map.items():
            deps_graph[func] = set(info['calls'])
        
        # Topological sort to determine compilation order
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(func: str):
            if func in temp_visited:
                return  # Cycle detected, skip
            if func in visited:
                return
            
            temp_visited.add(func)
            for dep in deps_graph.get(func, set()):
                if dep in dependency_map:  # Only include functions defined in the project
                    visit(dep)
            
            temp_visited.remove(func)
            visited.add(func)
            order.append(func)
        
        # Visit all functions
        for func in dependency_map.keys():
            if func not in visited:
                visit(func)
        
        return order

    def _plan_cpp_files(self, call_tree: Dict[str, Any], dependency_map: Dict[str, Any], 
                       compilation_order: List[str]) -> List[Dict[str, Any]]:
        """
        Plan how MATLAB files should be organized into C++ files.
        """
        defined_functions = call_tree['defined_functions']
        cpp_files = []
        
        # Group functions by their MATLAB file
        file_groups: Dict[str, List[str]] = {}
        for func, file_path in defined_functions.items():
            file_name = Path(file_path).stem
            if file_name not in file_groups:
                file_groups[file_name] = []
            file_groups[file_name].append(func)
        
        # Create C++ file plans
        for matlab_file, functions in file_groups.items():
            # Determine if this should be a header/implementation pair or single file
            has_public_interface = any(
                len(dependency_map.get(func, {}).get('called_by', [])) > 0 
                for func in functions
            )
            
            if has_public_interface or len(functions) > 1:
                # Create header/implementation pair
                cpp_files.append({
                    'name': matlab_file,
                    'type': 'header_impl_pair',
                    'header_file': f"{matlab_file}.h",
                    'impl_file': f"{matlab_file}.cpp",
                    'functions': functions,
                    'matlab_source': matlab_file,
                    'public_functions': [f for f in functions if len(dependency_map.get(f, {}).get('called_by', [])) > 0],
                    'private_functions': [f for f in functions if len(dependency_map.get(f, {}).get('called_by', [])) == 0]
                })
            else:
                # Single file
                cpp_files.append({
                    'name': matlab_file,
                    'type': 'single_file',
                    'file': f"{matlab_file}.cpp",
                    'functions': functions,
                    'matlab_source': matlab_file
                })
        
        return cpp_files

    def _plan_include_dependencies(self, cpp_files: List[Dict[str, Any]], 
                                 call_tree: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Plan which header files each C++ file needs to include.
        """
        include_deps = {}
        defined_functions = call_tree['defined_functions']
        
        for cpp_file in cpp_files:
            file_includes = []
            
            if cpp_file['type'] == 'header_impl_pair':
                # Implementation file includes its own header
                file_includes.append(f'"{cpp_file["header_file"]}"')
                
                # Add includes for called functions
                for func in cpp_file['functions']:
                    for called_func in call_tree['call_graph'].get(func, []):
                        if called_func in defined_functions:
                            called_file = Path(defined_functions[called_func]).stem
                            if called_file != cpp_file['name']:
                                header_name = f"{called_file}.h"
                                if header_name not in file_includes:
                                    file_includes.append(f'"{header_name}"')
                
                include_deps[cpp_file['impl_file']] = file_includes
            else:
                # Single file - add includes for called functions
                for func in cpp_file['functions']:
                    for called_func in call_tree['call_graph'].get(func, []):
                        if called_func in defined_functions:
                            called_file = Path(defined_functions[called_func]).stem
                            if called_file != cpp_file['name']:
                                header_name = f"{called_file}.h"
                                if header_name not in file_includes:
                                    file_includes.append(f'"{header_name}"')
                
                include_deps[cpp_file['file']] = file_includes
        
        return include_deps

    def _plan_namespaces(self, cpp_files: List[Dict[str, Any]], analysis: Dict[str, Any]) -> Dict[str, str]:
        """
        Plan C++ namespaces based on the project structure and domain.
        """
        proj = analysis.get('project_understanding') or {}
        domain = getattr(proj, 'domain', 'General')
        
        # Create namespace based on domain and project structure
        if domain and domain != 'General':
            namespace = domain.lower().replace(' ', '_')
        else:
            namespace = 'matlab_converted'
        
        namespaces = {}
        for cpp_file in cpp_files:
            if cpp_file['type'] == 'header_impl_pair':
                namespaces[cpp_file['header_file']] = namespace
                namespaces[cpp_file['impl_file']] = namespace
            else:
                namespaces[cpp_file['file']] = namespace
        
        return namespaces

    def _fallback_single_file_structure(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback to single file structure if multi-file analysis is not available.
        """
        return {
            'cpp_files': [{
                'name': 'main',
                'type': 'single_file',
                'file': 'main.cpp',
                'functions': [],
                'matlab_source': 'main'
            }],
            'include_dependencies': {'main.cpp': []},
            'compilation_order': ['main'],
            'namespaces': {'main.cpp': 'matlab_converted'},
            'project_type': 'single_file'
        }
