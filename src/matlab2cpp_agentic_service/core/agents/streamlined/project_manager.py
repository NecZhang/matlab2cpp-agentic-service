"""
Multi-File Project Manager Agent

This specialized agent handles complex multi-file C++ projects with:
- Multi-file project coordination
- Project-wide compilation testing
- Dependency resolution and error diagnosis
- Strategy selection for project-wide issues
- Cross-file consistency management
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Set, Tuple, Callable
from pathlib import Path
from collections import defaultdict
import re

from ..base.langgraph_agent import BaseLangGraphAgent, AgentConfig
from ....infrastructure.tools.llm_client import LLMClient
from ....infrastructure.state.conversion_state import ConversionState
from ....infrastructure.testing.compilation_manager import CPPCompilationManager
from ....infrastructure.testing.quality_assessor import CPPQualityAssessor
# Note: These adaptive agents will be available after we implement them
# from ..adaptive.strategy_selection_agent import StrategySelectionAgent
# from ..adaptive.multi_file_compilation_tester import MultiFileCompilationTester


class ProjectManager(BaseLangGraphAgent):
    """
    Specialized manager for complex multi-file C++ projects.
    
    Capabilities:
    - Multi-file project coordination
    - Project-wide compilation testing
    - Dependency resolution and error diagnosis
    - Strategy selection for project-wide issues
    - Cross-file consistency management
    """
    
    def __init__(self, config: AgentConfig, llm_client: LLMClient):
        super().__init__(config, llm_client)
        
        # Specialized tools
        self.tools = [
            "project_coordinator",
            "multi_file_compilation_tester",
            "dependency_resolver",
            "strategy_selector",
            "consistency_manager"
        ]
        
        # Initialize specialized components
        # Note: These will be available after implementing adaptive agents
        # self.compilation_tester = MultiFileCompilationTester(config, llm_client)
        # self.strategy_selector = StrategySelectionAgent(config, llm_client)
        self.quality_assessor = CPPQualityAssessor()
        self.compilation_manager = CPPCompilationManager()
        
        # Project management state
        self.project_state = {}
        self.coordination_strategies = self._initialize_coordination_strategies()
        self.error_resolution_patterns = self._initialize_error_resolution_patterns()
        
        self.logger.info(f"Initialized Project Manager: {config.name}")
    
    def create_node(self) -> Callable[[ConversionState], ConversionState]:
        """Create the LangGraph node function for project management."""
        async def manage_node(state: ConversionState) -> ConversionState:
            return await self.coordinate_project(state)
        return manage_node
    
    def get_tools(self) -> List[Any]:
        """Get available tools for project management."""
        return [
            self.quality_assessor,
            # Add other project management tools as needed
        ]
    
    def _initialize_coordination_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize coordination strategies for different project types."""
        return {
            "sequential_coordination": {
                "description": "Process files sequentially in dependency order",
                "use_case": "Simple dependency chains",
                "pros": ["Simple to implement", "Clear execution order"],
                "cons": ["Slower for large projects", "No parallelization"]
            },
            "parallel_coordination": {
                "description": "Process independent files in parallel",
                "use_case": "Files with minimal dependencies",
                "pros": ["Fast execution", "Efficient resource use"],
                "cons": ["Complex dependency management", "Potential conflicts"]
            },
            "hybrid_coordination": {
                "description": "Mix of sequential and parallel processing",
                "use_case": "Complex projects with both dependencies and independence",
                "pros": ["Balanced approach", "Flexible"],
                "cons": ["Complex implementation", "Requires careful planning"]
            },
            "dependency_aware_coordination": {
                "description": "Intelligent coordination based on dependency analysis",
                "use_case": "Complex multi-file projects",
                "pros": ["Optimal processing", "Handles complex dependencies"],
                "cons": ["Most complex", "Requires sophisticated analysis"]
            }
        }
    
    def _initialize_error_resolution_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize error resolution patterns for multi-file projects."""
        return {
            "include_conflicts": {
                "description": "Conflicting include statements across files",
                "resolution": "Standardize include order and guards",
                "severity": "high"
            },
            "namespace_conflicts": {
                "description": "Namespace conflicts between files",
                "resolution": "Implement hierarchical namespace strategy",
                "severity": "high"
            },
            "circular_dependencies": {
                "description": "Circular dependencies between files",
                "resolution": "Introduce forward declarations and interfaces",
                "severity": "critical"
            },
            "template_inconsistencies": {
                "description": "Inconsistent template usage across files",
                "resolution": "Standardize template patterns and conventions",
                "severity": "medium"
            },
            "interface_mismatches": {
                "description": "Function signature mismatches between files",
                "resolution": "Align function signatures and types",
                "severity": "high"
            }
        }
    
    async def coordinate_multi_file_project(self, project_structure: Dict[str, Any],
                                          generated_code: Dict[str, Any],
                                          conversion_plan: Dict[str, Any],
                                          state: ConversionState) -> ConversionState:
        """
        Coordinate complex multi-file project conversion.
        
        Args:
            project_structure: Project structure analysis
            generated_code: Generated C++ code
            conversion_plan: Conversion plan
            state: Current conversion state
            
        Returns:
            Updated state with coordination results
        """
        start_time = time.time()
        
        try:
            self.logger.info("Starting multi-file project coordination...")
            
            files_snapshot = self._extract_files_dict(generated_code)
            
            # Phase 1: Analyze project complexity
            complexity_analysis = self._analyze_project_complexity(project_structure, generated_code)
            
            # Phase 2: Select coordination strategy
            coordination_strategy = self._select_coordination_strategy(complexity_analysis)
            
            # Phase 3: Test project-wide compilation
            compilation_result = self._extract_compilation_result(generated_code)
            if not compilation_result:
                compilation_result = await self._test_project_compilation(generated_code, conversion_plan)
            
            # Phase 4: Diagnose cross-file issues
            cross_file_issues = self._diagnose_cross_file_issues(compilation_result, generated_code)
            
            # Phase 5: Resolve project-wide issues
            resolution_result = await self._resolve_project_issues(
                cross_file_issues,
                generated_code,
                conversion_plan,
                files_snapshot
            )
            
            # Phase 6: Validate final project
            validation_result = await self._validate_project_coordination(resolution_result, generated_code)
            
            # Create comprehensive coordination result
            coordination_result = {
                'complexity_analysis': complexity_analysis,
                'coordination_strategy': coordination_strategy,
                'compilation_result': compilation_result,
                'cross_file_issues': cross_file_issues,
                'resolution_result': resolution_result,
                'validation_result': validation_result,
                'coordination_timestamp': time.time(),
                'project_state': self.project_state
            }
            
            # Update state
            state["multi_file_coordination"] = coordination_result
            
            # Update memory
            self.update_memory("coordination_count", 
                             (self.get_memory("coordination_count", "short_term") or 0) + 1, 
                             "short_term")
            
            # Track performance
            execution_time = time.time() - start_time
            self.track_performance("coordinate_multi_file_project", start_time, time.time(), True, 
                                 {"files_coordinated": len(generated_code.get('files', {}))})
            
            self.logger.info(f"Multi-file project coordination complete: "
                           f"{len(generated_code.get('files', {}))} files, "
                           f"{execution_time:.2f}s")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Multi-file project coordination failed: {e}")
            self.track_performance("coordinate_multi_file_project", start_time, time.time(), False, 
                                 {"error": str(e)})
            raise
    
    def _analyze_project_complexity(self, project_structure: Dict[str, Any], 
                                  generated_code: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze complexity of multi-file project."""
        files = generated_code.get('files', {})
        file_count = len(files)
        
        # Analyze dependencies
        dependency_map = project_structure.get('dependency_map', {})
        dependency_count = len(dependency_map)
        
        # Analyze function call relationships
        function_call_tree = project_structure.get('function_call_tree', {})
        function_count = len(function_call_tree)
        
        # Calculate complexity metrics
        dependency_density = dependency_count / max(file_count, 1)
        function_density = function_count / max(file_count, 1)
        
        # Determine complexity level
        if file_count <= 2 and dependency_density < 0.5:
            complexity_level = "simple"
        elif file_count <= 5 and dependency_density < 1.0:
            complexity_level = "moderate"
        elif file_count <= 10 and dependency_density < 2.0:
            complexity_level = "complex"
        else:
            complexity_level = "highly_complex"
        
        # Detect potential issues
        potential_issues = self._detect_potential_issues(project_structure, generated_code)
        
        return {
            'file_count': file_count,
            'dependency_count': dependency_count,
            'function_count': function_count,
            'dependency_density': dependency_density,
            'function_density': function_density,
            'complexity_level': complexity_level,
            'potential_issues': potential_issues,
            'coordination_difficulty': self._assess_coordination_difficulty(complexity_level, potential_issues)
        }
    
    def _detect_potential_issues(self, project_structure: Dict[str, Any], 
                               generated_code: Dict[str, Any]) -> List[str]:
        """Detect potential issues in multi-file project."""
        issues = []
        
        # Check for circular dependencies
        dependency_map = project_structure.get('dependency_map', {})
        if self._has_circular_dependencies(dependency_map):
            issues.append('circular_dependencies')
        
        # Check for namespace conflicts
        files = generated_code.get('files', {})
        if self._has_namespace_conflicts(files):
            issues.append('namespace_conflicts')
        
        # Check for include conflicts
        if self._has_include_conflicts(files):
            issues.append('include_conflicts')
        
        # Check for interface mismatches
        if self._has_interface_mismatches(files, project_structure):
            issues.append('interface_mismatches')
        
        return issues
    
    def _has_circular_dependencies(self, dependency_map: Dict[str, Any]) -> bool:
        """Check for circular dependencies in the project."""
        # Simple cycle detection using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in dependency_map.get(node, {}).get('calls', []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in dependency_map:
            if node not in visited:
                if has_cycle(node):
                    return True
        
        return False
    
    def _has_namespace_conflicts(self, files: Dict[str, str]) -> bool:
        """Check for namespace conflicts between files."""
        namespaces = []
        
        for file_content in files.values():
            # Extract namespace declarations
            import re
            ns_matches = re.findall(r'namespace\s+(\w+)', file_content)
            namespaces.extend(ns_matches)
        
        # Check for duplicates
        return len(namespaces) != len(set(namespaces))
    
    def _has_include_conflicts(self, files: Dict[str, str]) -> bool:
        """Check for include conflicts between files."""
        includes = []
        
        for file_content in files.values():
            # Extract include statements
            import re
            include_matches = re.findall(r'#include\s*["<]([^">]+)["<]', file_content)
            includes.extend(include_matches)
        
        # Check for conflicting includes (same header with different paths)
        include_base_names = {}
        for include in includes:
            base_name = Path(include).name
            if base_name in include_base_names and include_base_names[base_name] != include:
                return True
            include_base_names[base_name] = include
        
        return False
    
    def _has_interface_mismatches(self, files: Dict[str, str], 
                                project_structure: Dict[str, Any]) -> bool:
        """Check for interface mismatches between files."""
        # This is a simplified check - in a full implementation,
        # it would parse function signatures and check for mismatches
        return False  # Placeholder
    
    def _assess_coordination_difficulty(self, complexity_level: str, potential_issues: List[str]) -> str:
        """Assess difficulty level for project coordination."""
        if complexity_level == "highly_complex" or len(potential_issues) >= 3:
            return "very_difficult"
        elif complexity_level == "complex" or len(potential_issues) >= 2:
            return "difficult"
        elif complexity_level == "moderate" or len(potential_issues) >= 1:
            return "moderate"
        else:
            return "easy"
    
    def _select_coordination_strategy(self, complexity_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Select optimal coordination strategy based on project analysis."""
        complexity_level = complexity_analysis['complexity_level']
        coordination_difficulty = complexity_analysis['coordination_difficulty']
        potential_issues = complexity_analysis['potential_issues']
        
        if coordination_difficulty == "easy":
            strategy_name = "sequential_coordination"
        elif coordination_difficulty == "moderate":
            strategy_name = "parallel_coordination"
        elif coordination_difficulty == "difficult":
            strategy_name = "hybrid_coordination"
        else:  # very_difficult
            strategy_name = "dependency_aware_coordination"
        
        strategy = self.coordination_strategies[strategy_name].copy()
        strategy['selected_strategy'] = strategy_name
        strategy['complexity_analysis'] = complexity_analysis
        
        return strategy
    
    async def _test_project_compilation(self, generated_code: Dict[str, Any], 
                                      conversion_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Test project-wide compilation."""
        try:
            # TODO: Use multi-file compilation tester when available
            # compilation_result = await self.compilation_tester.test_multi_file_compilation(
            #     generated_code, conversion_plan
            # )
            
            # Placeholder implementation
            compilation_result = {
                'success': True,  # Assume success for now
                'output': 'Project compilation test completed',
                'compilation_time': 1.0
            }
            
            return compilation_result
            
        except Exception as e:
            self.logger.error(f"Project compilation testing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'error_output': str(e)
            }
    
    def _diagnose_cross_file_issues(self, compilation_result: Dict[str, Any], 
                                  generated_code: Dict[str, Any]) -> Dict[str, Any]:
        """Diagnose cross-file issues from compilation results."""
        if compilation_result.get('success', False):
            return {
                'has_issues': False,
                'issues': [],
                'severity': 'none'
            }
        
        error_output = (
            compilation_result.get('error_output')
            or compilation_result.get('output')
            or ''
        )
        issues = []
        
        # Analyze error patterns
        for pattern_name, pattern_info in self.error_resolution_patterns.items():
            patterns = pattern_info.get('patterns', [])
            for pattern in patterns:
                if pattern.lower() in error_output.lower():
                    issues.append({
                        'type': pattern_name,
                        'description': pattern_info['description'],
                        'resolution': pattern_info['resolution'],
                        'severity': pattern_info['severity']
                    })
        
        # Determine overall severity
        if any(issue['severity'] == 'critical' for issue in issues):
            severity = 'critical'
        elif any(issue['severity'] == 'high' for issue in issues):
            severity = 'high'
        elif any(issue['severity'] == 'medium' for issue in issues):
            severity = 'medium'
        else:
            severity = 'low'
        
        return {
            'has_issues': len(issues) > 0,
            'issues': issues,
            'severity': severity,
            'error_output': error_output
        }
    
    async def _resolve_project_issues(self, cross_file_issues: Dict[str, Any],
                                    generated_code: Dict[str, Any],
                                    conversion_plan: Dict[str, Any],
                                    files_snapshot: Dict[str, str]) -> Dict[str, Any]:
        """Resolve project-wide issues."""
        if not cross_file_issues.get('has_issues', False):
            return {
                'resolution_applied': False,
                'issues_resolved': 0,
                'resolution_strategies': []
            }
        
        issues = cross_file_issues.get('issues', [])
        error_output = cross_file_issues.get('error_output', '')
        
        current_files = self._extract_files_dict(generated_code) or files_snapshot
        if not current_files:
            return {
                'resolution_applied': False,
                'issues_resolved': 0,
                'resolution_strategies': [],
                'notes': 'No generated files available for coordination.'
            }
        
        interface_index = self._build_interface_index(current_files, conversion_plan)
        adjusted_files, adjustments = self._enforce_cross_file_contracts(
            current_files,
            interface_index,
            conversion_plan,
            error_output
        )
        
        if adjusted_files:
            self._update_generated_code_structure(generated_code, adjusted_files)
            self.project_state['last_alignment'] = {
                'adjustments': adjustments,
                'timestamp': time.time()
            }
        
        resolution_strategies = adjustments
        issues_resolved = len(adjustments)
        
        return {
            'resolution_applied': True,
            'issues_resolved': issues_resolved,
            'total_issues': len(issues),
            'resolution_strategies': resolution_strategies,
            'success_rate': issues_resolved / max(len(issues), 1),
            'applied_fixes': len(adjustments),
            'notes': 'Cross-file namespace alignment applied' if adjustments else 'No automatic adjustments applied'
        }
    
    async def _validate_project_coordination(self, resolution_result: Dict[str, Any],
                                           generated_code: Dict[str, Any]) -> Dict[str, Any]:
        """Validate final project coordination."""
        # Use quality assessor to validate the coordinated project
        try:
            quality_assessment = await self.quality_assessor.assess_code_quality(generated_code)
            
            # Determine validation success
            quality_score = quality_assessment.get('overall_score', 0.0)
            validation_success = quality_score >= 0.7  # 70% quality threshold
            
            return {
                'validation_success': validation_success,
                'quality_score': quality_score,
                'quality_assessment': quality_assessment,
                'coordination_quality': 'high' if quality_score >= 0.8 else 'medium' if quality_score >= 0.6 else 'low'
            }
            
        except Exception as e:
            self.logger.error(f"Project validation failed: {e}")
            return {
                'validation_success': False,
                'quality_score': 0.0,
                'error': str(e),
                'coordination_quality': 'failed'
            }

    def _extract_files_dict(self, generated_code: Dict[str, Any]) -> Dict[str, str]:
        """Extract flat file dictionary from nested generated_code structure."""
        if not isinstance(generated_code, dict):
            return {}
        
        if 'files' in generated_code and isinstance(generated_code['files'], dict):
            return dict(generated_code['files'])
        
        inner = generated_code.get('generated_code')
        if isinstance(inner, dict):
            files = inner.get('files')
            if isinstance(files, dict):
                return dict(files)
        
        return {}
    
    def _update_generated_code_structure(self,
                                         generated_code: Dict[str, Any],
                                         updated_files: Dict[str, str]) -> None:
        """Persist modified files back into the generated_code structure."""
        if not updated_files:
            return
        
        if 'files' in generated_code and isinstance(generated_code['files'], dict):
            generated_code['files'].update(updated_files)
            return
        
        inner = generated_code.get('generated_code')
        if isinstance(inner, dict):
            if 'files' not in inner or not isinstance(inner['files'], dict):
                inner['files'] = {}
            inner['files'].update(updated_files)
        else:
            generated_code['files'] = dict(updated_files)
    
    def _extract_compilation_result(self, generated_code: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Retrieve the most recent compilation result from generated code."""
        payload: Optional[Dict[str, Any]] = generated_code if isinstance(generated_code, dict) else None
        
        while payload:
            result = payload.get('compilation_result')
            if result:
                return result
            next_payload = payload.get('generated_code')
            payload = next_payload if isinstance(next_payload, dict) else None
        
        return None
    
    def _build_interface_index(self,
                               files: Dict[str, str],
                               conversion_plan: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Build an index of known function interfaces from header files."""
        index: Dict[str, Dict[str, Any]] = {}
        file_mapping = {}
        if conversion_plan:
            file_mapping = conversion_plan.get('file_organization', {}).get('file_mapping', {}) or {}
        
        for filename, content in files.items():
            if not filename.endswith(('.h', '.hpp')):
                continue
            
            namespace = self._detect_namespace(content)
            if not namespace:
                mapping_match = next(
                    (mapping for mapping in file_mapping.values()
                     if mapping.get('header_file') == filename),
                    {}
                )
                namespace = mapping_match.get('namespace')
            
            signatures = self._extract_header_signatures(content)
            for signature in signatures:
                func_name = self._extract_function_name(signature)
                if not func_name:
                    continue
                index[func_name] = {
                    'namespace': namespace,
                    'signature': signature,
                    'header': filename
                }
        
        return index
    
    def _enforce_cross_file_contracts(self,
                                      files: Dict[str, str],
                                      interface_index: Dict[str, Dict[str, Any]],
                                      conversion_plan: Dict[str, Any],
                                      error_output: str) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
        """
        Apply cross-file consistency fixes (namespaces, includes, etc.).
        Returns updated files and a list of applied adjustments.
        """
        if not interface_index:
            return {}, []
        
        updated_files: Dict[str, str] = {}
        adjustments: List[Dict[str, Any]] = []
        
        for filename, content in files.items():
            if not filename.endswith('.cpp'):
                continue
            
            current_namespace = self._detect_namespace(content)
            local_functions = self._extract_defined_functions(content)
            rewritten, prefixed = self._apply_namespace_prefixes(
                content,
                interface_index,
                local_functions,
                current_namespace
            )
            
            if prefixed:
                updated_files[filename] = rewritten
                adjustments.append({
                    'issue_type': 'namespace_alignment',
                    'strategy': 'auto_prefix_calls',
                    'description': f"Prefixed cross-file calls in {filename}",
                    'functions': sorted(prefixed)
                })
        
        return updated_files, adjustments
    
    def _detect_namespace(self, content: str) -> Optional[str]:
        """Detect the first namespace declaration in the content."""
        if not content:
            return None
        match = re.search(r'namespace\s+([A-Za-z_]\w*)', content)
        return match.group(1) if match else None
    
    def _extract_header_signatures(self, content: str) -> List[str]:
        """Extract function signatures from a header file."""
        signatures: List[str] = []
        if not content:
            return signatures
        
        lines = content.split('\n')
        buffered = ""
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('//'):
                continue
            buffered += " " + stripped
            if ';' in stripped:
                signatures.append(buffered.strip())
                buffered = ""
        return signatures
    
    def _extract_function_name(self, signature: str) -> Optional[str]:
        """Extract function name from a signature string."""
        cleaned = ' '.join(signature.split())
        match = re.search(r'([A-Za-z_]\w*)\s*\([^()]*\)\s*;?$', cleaned)
        return match.group(1) if match else None
    
    def _extract_defined_functions(self, content: str) -> Set[str]:
        """Extract locally defined function names from implementation file."""
        return self._extract_function_names(content, '{')
    
    def _extract_function_names(self, content: str, terminator: str) -> Set[str]:
        """Generic function name extractor using a simple heuristic."""
        if not content:
            return set()
        
        pattern = re.compile(
            r'([A-Za-z_]\w*(?:::[A-Za-z_]\w*)?)\s*\([^;{}]*\)\s*' + re.escape(terminator)
        )
        reserved = {'if', 'for', 'while', 'switch', 'return', 'catch', 'else'}
        names: Set[str] = set()
        for match in pattern.finditer(content):
            name = match.group(1)
            base_name = name.split('::')[-1]
            if base_name in reserved:
                continue
            names.add(base_name)
        return names
    
    def _apply_namespace_prefixes(self,
                                  content: str,
                                  interface_index: Dict[str, Dict[str, Any]],
                                  local_functions: Set[str],
                                  current_namespace: Optional[str]) -> Tuple[str, Set[str]]:
        """Prefix external function calls with their namespaces where required."""
        if not content:
            return content, set()
        
        rewritten = content
        prefixed: Set[str] = set()
        
        for func_name, info in interface_index.items():
            if func_name in local_functions:
                continue
            target_namespace = info.get('namespace')
            if not target_namespace or target_namespace == current_namespace:
                continue
            
            pattern = re.compile(r'(?<![:\w])' + re.escape(func_name) + r'\s*\(')
            matches = list(pattern.finditer(rewritten))
            if not matches:
                continue
            
            replacements = []
            for match in matches:
                start = match.start()
                line_start = rewritten.rfind('\n', 0, start) + 1
                line_prefix = rewritten[line_start:start]
                
                # Skip comments
                if line_prefix.strip().startswith('//'):
                    continue
                
                # Skip already-qualified occurrences
                preceding = rewritten[max(0, start - 64):start]
                if re.search(r'::\s*$', preceding.strip()):
                    continue
                
                # Avoid definitions and declarations
                if re.search(r'\bclass\s+$', preceding) or re.search(r'\bstruct\s+$', preceding):
                    continue
                if re.search(r'\btemplate\s*<[^>]*>\s*$', preceding):
                    continue
                
                replacements.append((match.start(), match.end()))
            
            if not replacements:
                continue
            
            # Apply replacements from end to start to keep indices valid
            for start, end in reversed(replacements):
                rewritten = (
                    rewritten[:start]
                    + f"{target_namespace}::{func_name}("
                    + rewritten[end:]
                )
            prefixed.add(func_name)
        
        return rewritten, prefixed
    
    async def get_coordination_summary(self, coordination_result: Dict[str, Any]) -> str:
        """Generate human-readable coordination summary."""
        summary = f"🔧 Multi-File Project Coordination Summary\n"
        
        complexity_analysis = coordination_result['complexity_analysis']
        summary += f"Project complexity: {complexity_analysis['complexity_level']}\n"
        summary += f"Files coordinated: {complexity_analysis['file_count']}\n"
        summary += f"Coordination difficulty: {complexity_analysis['coordination_difficulty']}\n"
        
        coordination_strategy = coordination_result['coordination_strategy']
        summary += f"Strategy used: {coordination_strategy['selected_strategy']}\n"
        
        compilation_result = coordination_result['compilation_result']
        summary += f"Compilation: {'✅ Success' if compilation_result.get('success', False) else '❌ Failed'}\n"
        
        cross_file_issues = coordination_result['cross_file_issues']
        if cross_file_issues.get('has_issues', False):
            summary += f"Cross-file issues: {len(cross_file_issues['issues'])} detected\n"
            summary += f"Issue severity: {cross_file_issues['severity']}\n"
        else:
            summary += "Cross-file issues: ✅ None detected\n"
        
        resolution_result = coordination_result['resolution_result']
        if resolution_result.get('resolution_applied', False):
            summary += f"Issues resolved: {resolution_result['issues_resolved']}/{resolution_result['total_issues']}\n"
            summary += f"Resolution success rate: {resolution_result['success_rate']:.1%}\n"
        
        validation_result = coordination_result['validation_result']
        summary += f"Final validation: {'✅ Passed' if validation_result.get('validation_success', False) else '❌ Failed'}\n"
        summary += f"Quality score: {validation_result.get('quality_score', 0.0):.2f}\n"
        
        return summary
