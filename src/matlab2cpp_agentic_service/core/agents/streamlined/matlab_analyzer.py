"""
Enhanced MATLAB Analyzer Agent

This agent provides advanced MATLAB code analysis with:
- Advanced dependency detection for multi-file projects
- Function call graph analysis with topological sorting
- Complexity assessment for conversion planning
- Pattern recognition for common MATLAB→C++ issues
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Set, Tuple, Callable
from pathlib import Path
from collections import defaultdict, deque

from ..base.langgraph_agent import BaseLangGraphAgent, AgentConfig
from ....infrastructure.tools.llm_client import LLMClient
from ....infrastructure.state.conversion_state import ConversionState
from ....infrastructure.tools.matlab_parser import MATLABParser


class MATLABAnalyzer(BaseLangGraphAgent):
    """
    MATLAB analyzer with advanced dependency detection and pattern recognition.
    
    Capabilities:
    - Multi-file project analysis
    - Function call graph construction
    - Dependency resolution with topological sorting
    - Complexity assessment for conversion planning
    - Pattern recognition for common issues
    """
    
    def __init__(self, config: AgentConfig, llm_client: LLMClient):
        super().__init__(config, llm_client)
        
        # Enhanced tools
        self.tools = [
            "matlab_parser",
            "dependency_analyzer", 
            "complexity_assessor",
            "pattern_recognizer",
            "llm_analysis"
        ]
        
        # Multi-file project support
        self.matlab_parser = MATLABParser()
        self.dependency_graph = {}
        self.function_call_tree = {}
        self.complexity_metrics = {}
        
        # Pattern recognition database
        self.conversion_patterns = self._initialize_conversion_patterns()
        
        self.logger.info(f"Initialized MATLAB Analyzer: {config.name}")
    
    def create_node(self) -> Callable[[ConversionState], ConversionState]:
        """Create the LangGraph node function for MATLAB analysis."""
        async def analyze_node(state: ConversionState) -> ConversionState:
            return await self.analyze_matlab_content(state)
        return analyze_node
    
    def get_tools(self) -> List[Any]:
        """Get available tools for MATLAB analysis."""
        return [
            self.matlab_parser,
            # Add other tools as needed
        ]
    
    def _initialize_conversion_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns that commonly cause C++ conversion issues."""
        return {
            "template_issues": [
                "dynamic arrays",
                "cell arrays", 
                "struct arrays",
                "anonymous functions",
                "nested functions"
            ],
            "include_issues": [
                "file I/O operations",
                "image processing",
                "signal processing",
                "mathematical functions",
                "plotting functions"
            ],
            "memory_issues": [
                "large matrix operations",
                "recursive functions",
                "global variables",
                "persistent variables",
                "memory-intensive algorithms"
            ],
            "syntax_issues": [
                "vectorized operations",
                "array indexing",
                "string operations",
                "regular expressions",
                "handle graphics"
            ]
        }
    
    async def analyze_project(self, matlab_path: Path, state: ConversionState) -> ConversionState:
        """
        Analyze entire MATLAB project with dependency awareness.
        
        Args:
            matlab_path: Path to MATLAB file or directory
            state: Current conversion state
            
        Returns:
            Updated state with comprehensive analysis
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting enhanced analysis of: {matlab_path}")
            
            # Phase 1: Parse all MATLAB files
            parse_start = time.time()
            file_analyses = await self._parse_all_matlab_files(matlab_path)
            parse_time = time.time() - parse_start
            self.logger.debug(f"Phase 1 (parse files) completed in {parse_time:.4f}s")
            
            # Phase 2: Build function call graph
            graph_start = time.time()
            function_call_tree = self._build_function_call_graph(file_analyses)
            graph_time = time.time() - graph_start
            self.logger.debug(f"Phase 2 (build call graph) completed in {graph_time:.4f}s")
            
            # Phase 3: Resolve dependencies with topological sort
            deps_start = time.time()
            dependency_map = self._resolve_dependencies(file_analyses)
            compilation_order = self._topological_sort(dependency_map)
            deps_time = time.time() - deps_start
            self.logger.debug(f"Phase 3 (resolve dependencies) completed in {deps_time:.4f}s")
            
            # Phase 4: Assess complexity metrics
            complexity_start = time.time()
            complexity_assessment = self._assess_complexity(file_analyses)
            complexity_time = time.time() - complexity_start
            self.logger.debug(f"Phase 4 (assess complexity) completed in {complexity_time:.4f}s")
            
            # Phase 5: Identify conversion patterns and potential issues
            patterns_start = time.time()
            conversion_patterns = self._detect_conversion_patterns(file_analyses)
            patterns_time = time.time() - patterns_start
            self.logger.debug(f"Phase 5 (detect patterns) completed in {patterns_time:.4f}s")
            
            # Phase 6: Create comprehensive analysis result
            analysis_result = {
                'files_analyzed': len(file_analyses),
                'file_analyses': file_analyses,
                'function_call_tree': function_call_tree,
                'dependency_map': dependency_map,
                'compilation_order': compilation_order,
                'complexity_assessment': complexity_assessment,
                'conversion_patterns': conversion_patterns,
                'is_multi_file': len(file_analyses) > 1,
                'total_functions': sum(len(analysis.get('functions', [])) for analysis in file_analyses),
                'total_dependencies': len(dependency_map),
                'function_signatures': {
                    analysis['file_name']: analysis.get('function_signatures', {})
                    for analysis in file_analyses
                },
                'analysis_timestamp': time.time(),
                'agent_memory': {
                    'analysis_count': self.get_memory("analysis_count", "short_term") or 0,
                    'last_analysis_time': time.time()
                }
            }
            
            # LOG CALL GRAPH AND ENTRY POINTS (for debugging and issue analysis)
            self._log_call_graph_and_entry_points(analysis_result)
            
            # Update state
            state["matlab_analysis"] = analysis_result
            state["is_multi_file"] = analysis_result['is_multi_file']
            
            # Update memory
            self.update_memory("analysis_count", 
                             (self.get_memory("analysis_count", "short_term") or 0) + 1, 
                             "short_term")
            self.update_memory("last_analysis_time", time.time(), "short_term")
            
            # Track performance
            execution_time = time.time() - start_time
            self.track_performance("analyze_project", start_time, time.time(), True, 
                                 {"files_analyzed": len(file_analyses)})
            
            self.logger.info(f"Enhanced analysis complete: {len(file_analyses)} files, "
                           f"{analysis_result['total_functions']} functions, "
                           f"{execution_time:.2f}s")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Enhanced analysis failed: {e}")
            self.track_performance("analyze_project", start_time, time.time(), False, 
                                 {"error": str(e)})
            raise
    
    async def _parse_all_matlab_files(self, matlab_path: Path) -> List[Dict[str, Any]]:
        """Parse all MATLAB files in the project."""
        file_analyses = []
        
        if matlab_path.is_file() and matlab_path.suffix == '.m':
            # Single file
            analysis = await self._parse_single_file(matlab_path)
            file_analyses.append(analysis)
        elif matlab_path.is_dir():
            # Directory - find all .m files
            matlab_files = list(matlab_path.rglob("*.m"))
            for file_path in matlab_files:
                analysis = await self._parse_single_file(file_path)
                file_analyses.append(analysis)
        else:
            raise ValueError(f"Invalid MATLAB path: {matlab_path}")
        
        return file_analyses
    
    async def _parse_single_file(self, file_path: Path) -> Dict[str, Any]:
        """Parse a single MATLAB file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Use MATLAB parser
            parsed_structure = self.matlab_parser.parse_project(content)
            
            # Extract additional information
            functions = parsed_structure.get('functions', [])
            function_calls = parsed_structure.get('function_calls', {})
            variables = parsed_structure.get('variables', [])
            comments = parsed_structure.get('comments', [])
            
            # Calculate file metrics
            lines_of_code = len([line for line in content.split('\n') if line.strip()])
            complexity_score = self._calculate_file_complexity(parsed_structure)
            
            return {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'content': content,
                'parsed_structure': parsed_structure,
                'functions': functions,
                'function_calls': function_calls,
                'variables': variables,
                'comments': comments,
                'lines_of_code': lines_of_code,
                'complexity_score': complexity_score,
                'file_size': len(content)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to parse {file_path}: {e}")
            return {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'error': str(e),
                'functions': [],
                'function_calls': {},
                'variables': [],
                'complexity_score': 0.0
            }
    
    def _build_function_call_graph(self, file_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build comprehensive function call graph."""
        function_call_tree = defaultdict(list)
        defined_functions = set()
        
        # First pass: collect all defined functions
        for analysis in file_analyses:
            for func_name in analysis.get('functions', []):
                defined_functions.add(func_name)
        
        # Second pass: build call relationships
        for analysis in file_analyses:
            file_path = analysis['file_path']
            function_calls = analysis.get('function_calls', {})
            
            for func_name, called_functions in function_calls.items():
                # Only include calls to functions defined in the project
                project_calls = [f for f in called_functions if f in defined_functions]
                function_call_tree[func_name] = project_calls
        
        return dict(function_call_tree)
    
    def _resolve_dependencies(self, file_analyses: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Resolve function dependencies by mapping function calls to their definitions."""
        dependency_map = {}
        
        # First pass: collect all function definitions and their locations
        for analysis in file_analyses:
            file_path = analysis['file_path']
            functions = analysis.get('functions', [])
            
            for func_name in functions:
                if func_name not in dependency_map:
                    dependency_map[func_name] = {
                        'defined_in': file_path,
                        'called_by': [],
                        'calls': [],
                        'complexity': analysis.get('complexity_score', 0.0)
                    }
        
        # Second pass: build call relationships
        for analysis in file_analyses:
            function_calls = analysis.get('function_calls', {})
            
            for func_name, called_functions in function_calls.items():
                if func_name in dependency_map:
                    dependency_map[func_name]['calls'] = called_functions
                    
                    # Update 'called_by' relationships
                    for called_func in called_functions:
                        if called_func in dependency_map:
                            if func_name not in dependency_map[called_func]['called_by']:
                                dependency_map[called_func]['called_by'].append(func_name)
        
        return dependency_map
    
    def _topological_sort(self, dependency_map: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Perform topological sort to determine compilation order.
        Returns list of FILE NAMES (e.g., ['add_numbers.m', 'test_main.m']).
        """
        # Build file-level dependency graph from function dependencies
        file_deps = {}
        func_to_file = {}
        
        # Map functions to their defining files
        for func, info in dependency_map.items():
            defined_in = info.get('defined_in', '')
            func_to_file[func] = defined_in
            if defined_in not in file_deps:
                file_deps[defined_in] = set()
        
        # Build file-to-file dependencies based on function calls
        for func, info in dependency_map.items():
            file_path = info.get('defined_in', '')
            for called_func in info.get('calls', []):
                if called_func in func_to_file:
                    called_file = func_to_file[called_func]
                    if called_file != file_path:  # Don't self-depend
                        file_deps[file_path].add(called_file)
        
        # Topological sort on files using Kahn's algorithm
        in_degree = {f: 0 for f in file_deps}
        for file_path in file_deps:
            for dep in file_deps.get(file_path, []):
                if dep in file_deps:
                    in_degree[dep] += 1
        
        queue = deque([f for f, degree in in_degree.items() if degree == 0])
        result = []
        
        while queue:
            file_path = queue.popleft()
            # Extract just the filename (e.g., 'add_numbers.m' from full path)
            from pathlib import Path
            filename = Path(file_path).name
            result.append(filename)
            
            for dep in file_deps.get(file_path, []):
                if dep in file_deps:
                    in_degree[dep] -= 1
                    if in_degree[dep] == 0:
                        queue.append(dep)
        
        return result
    
    def _assess_complexity(self, file_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess overall project complexity."""
        total_functions = sum(len(analysis.get('functions', [])) for analysis in file_analyses)
        total_lines = sum(analysis.get('lines_of_code', 0) for analysis in file_analyses)
        avg_complexity = sum(analysis.get('complexity_score', 0) for analysis in file_analyses) / len(file_analyses) if file_analyses else 0
        
        # Determine complexity level
        if total_functions < 5 and total_lines < 200:
            complexity_level = "simple"
        elif total_functions < 15 and total_lines < 1000:
            complexity_level = "moderate"
        else:
            complexity_level = "complex"
        
        return {
            'total_functions': total_functions,
            'total_lines_of_code': total_lines,
            'average_complexity_score': avg_complexity,
            'complexity_level': complexity_level,
            'file_count': len(file_analyses),
            'conversion_difficulty': self._assess_conversion_difficulty(file_analyses)
        }
    
    def _assess_conversion_difficulty(self, file_analyses: List[Dict[str, Any]]) -> str:
        """Assess difficulty level for C++ conversion."""
        # Analyze patterns that make conversion difficult
        difficult_patterns = 0
        
        for analysis in file_analyses:
            content = analysis.get('content', '').lower()
            
            # Check for difficult patterns
            if any(pattern in content for pattern in ['cell', 'struct', 'handle']):
                difficult_patterns += 1
            if any(pattern in content for pattern in ['plot', 'figure', 'subplot']):
                difficult_patterns += 1
            if any(pattern in content for pattern in ['global', 'persistent']):
                difficult_patterns += 1
        
        if difficult_patterns == 0:
            return "easy"
        elif difficult_patterns < 3:
            return "moderate"
        else:
            return "difficult"
    
    def _detect_conversion_patterns(self, file_analyses: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Detect patterns that commonly cause C++ conversion issues."""
        detected_patterns = {category: [] for category in self.conversion_patterns.keys()}
        
        for analysis in file_analyses:
            content = analysis.get('content', '').lower()
            file_name = analysis.get('file_name', '')
            
            for category, patterns in self.conversion_patterns.items():
                for pattern in patterns:
                    if pattern in content:
                        detected_patterns[category].append({
                            'pattern': pattern,
                            'file': file_name,
                            'severity': self._get_pattern_severity(category, pattern)
                        })
        
        return detected_patterns
    
    def _get_pattern_severity(self, category: str, pattern: str) -> str:
        """Get severity level for detected pattern."""
        high_severity_patterns = {
            'template_issues': ['cell arrays', 'struct arrays', 'anonymous functions'],
            'include_issues': ['image processing', 'signal processing'],
            'memory_issues': ['large matrix operations', 'global variables'],
            'syntax_issues': ['vectorized operations', 'array indexing']
        }
        
        if pattern in high_severity_patterns.get(category, []):
            return "high"
        else:
            return "medium"
    
    def _calculate_file_complexity(self, parsed_structure: Dict[str, Any]) -> float:
        """Calculate complexity score for a single file."""
        functions = parsed_structure.get('functions', [])
        function_calls = parsed_structure.get('function_calls', {})
        variables = parsed_structure.get('variables', [])
        
        # Simple complexity scoring
        complexity = 0.0
        
        # Function complexity
        complexity += len(functions) * 0.5
        
        # Call complexity
        complexity += sum(len(calls) for calls in function_calls.values()) * 0.2
        
        # Variable complexity
        complexity += len(variables) * 0.1
        
        return min(complexity, 10.0)  # Cap at 10
    
    async def get_analysis_summary(self, analysis_result: Dict[str, Any]) -> str:
        """Generate human-readable analysis summary."""
        summary = f"📊 Enhanced MATLAB Analysis Summary\n"
        summary += f"Files analyzed: {analysis_result['files_analyzed']}\n"
        summary += f"Total functions: {analysis_result['total_functions']}\n"
        summary += f"Total dependencies: {analysis_result['total_dependencies']}\n"
        summary += f"Multi-file project: {'Yes' if analysis_result['is_multi_file'] else 'No'}\n"
        
        complexity = analysis_result['complexity_assessment']
        summary += f"Complexity level: {complexity['complexity_level']}\n"
        summary += f"Conversion difficulty: {complexity['conversion_difficulty']}\n"
        
        # Show detected patterns
        patterns = analysis_result['conversion_patterns']
        total_patterns = sum(len(pattern_list) for pattern_list in patterns.values())
        if total_patterns > 0:
            summary += f"Potential issues detected: {total_patterns}\n"
            for category, pattern_list in patterns.items():
                if pattern_list:
                    summary += f"  - {category}: {len(pattern_list)} issues\n"
        
        return summary
    
    def _log_call_graph_and_entry_points(self, analysis_result: Dict[str, Any]) -> None:
        """Log call graph and entry point analysis for debugging and issue resolution."""
        self.logger.info("=" * 80)
        self.logger.info("📊 CALL GRAPH & ENTRY POINT ANALYSIS")
        self.logger.info("=" * 80)
        
        file_analyses = analysis_result.get('file_analyses', [])
        function_call_tree = analysis_result.get('function_call_tree', {})
        compilation_order = analysis_result.get('compilation_order', [])
        
        # 1. List all files and their functions
        self.logger.info("\n📁 FILES AND FUNCTIONS:")
        for fa in file_analyses:
            filename = fa.get('file_name', 'unknown')
            functions = fa.get('functions', [])
            self.logger.info(f"  {filename}:")
            if functions:
                for func in functions:
                    if isinstance(func, dict):
                        func_name = func.get('name', 'unnamed')
                        func_type = func.get('type', 'unknown')
                        self.logger.info(f"    - {func_name} ({func_type})")
                    elif isinstance(func, str):
                        self.logger.info(f"    - {func}")
            else:
                self.logger.info(f"    (no functions detected)")
        
        # 2. Show function call graph
        self.logger.info("\n🔗 FUNCTION CALL GRAPH:")
        if function_call_tree:
            for caller, callees in function_call_tree.items():
                if callees:
                    self.logger.info(f"  {caller} → {', '.join(callees)}")
                else:
                    self.logger.info(f"  {caller} → (no calls)")
        else:
            self.logger.info("  (no function calls detected)")
        
        # 3. Detect entry points (functions not called by others)
        self.logger.info("\n🎯 ENTRY POINT DETECTION:")
        all_functions = set(function_call_tree.keys())
        called_functions = set()
        for callees in function_call_tree.values():
            called_functions.update(callees)
        
        entry_points = all_functions - called_functions
        
        if entry_points:
            self.logger.info(f"  Found {len(entry_points)} entry point(s):")
            for ep in sorted(entry_points):
                self.logger.info(f"    ✓ {ep} (top-level function)")
        else:
            self.logger.info("  No clear entry points detected")
            self.logger.info("  (All functions are called by others, or no calls detected)")
        
        # 4. Show compilation order
        self.logger.info("\n📋 COMPILATION ORDER:")
        if compilation_order:
            for idx, file in enumerate(compilation_order, 1):
                self.logger.info(f"  {idx}. {file}")
        else:
            self.logger.info("  (no specific order required)")
        
        # 5. Show complexity summary
        complexity = analysis_result.get('complexity_assessment', {})
        self.logger.info("\n⚙️  COMPLEXITY ASSESSMENT:")
        self.logger.info(f"  Level: {complexity.get('complexity_level', 'unknown')}")
        self.logger.info(f"  Conversion Difficulty: {complexity.get('conversion_difficulty', 'unknown')}")
        self.logger.info(f"  Total Functions: {analysis_result.get('total_functions', 0)}")
        self.logger.info(f"  Total Dependencies: {analysis_result.get('total_dependencies', 0)}")
        
        self.logger.info("=" * 80)
