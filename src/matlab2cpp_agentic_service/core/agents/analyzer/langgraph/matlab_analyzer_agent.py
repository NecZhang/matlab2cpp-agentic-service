"""
LangGraph-Native MATLAB Content Analyzer Agent

This module implements a truly LangGraph-native MATLAB content analyzer agent
that fully utilizes LangGraph features including tools, memory, and state management.
"""

import time
from typing import Dict, Any, List, Callable
from pathlib import Path
from loguru import logger

from matlab2cpp_agentic_service.infrastructure.state.conversion_state import ConversionState, ConversionStatus, add_processing_time, update_state_status
from matlab2cpp_agentic_service.infrastructure.tools.llm_client import LLMClient
from matlab2cpp_agentic_service.infrastructure.tools.langgraph_tools import MATLABParserTool, LLMAnalysisTool, ToolRegistry
from ...base.langgraph_agent import BaseLangGraphAgent, AgentConfig


class LangGraphMATLABAnalyzerAgent(BaseLangGraphAgent):
    """
    LangGraph-native MATLAB content analyzer agent.
    
    This agent uses LangGraph tools and memory to analyze MATLAB code,
    providing comprehensive analysis with persistent memory across workflow runs.
    """
    
    def __init__(self, config: AgentConfig, llm_client: LLMClient):
        """Initialize the LangGraph MATLAB analyzer agent."""
        super().__init__(config, llm_client)
        
        # Initialize tools
        self.tool_registry = ToolRegistry()
        self.matlab_parser_tool = MATLABParserTool()
        self.llm_analysis_tool = LLMAnalysisTool(llm_client)
        
        # Register tools
        self.tool_registry.register_tool("matlab_parser", self.matlab_parser_tool)
        self.tool_registry.register_tool("llm_analysis", self.llm_analysis_tool)
        
        self.logger.info(f"Initialized LangGraph MATLAB Analyzer Agent: {config.name}")
    
    def create_node(self) -> Callable[[ConversionState], ConversionState]:
        """
        Create the LangGraph node function for MATLAB analysis.
        
        Returns:
            Callable that takes and returns ConversionState
        """
        def analyze_matlab_node(state: ConversionState) -> ConversionState:
            """LangGraph node function for MATLAB content analysis."""
            start_time = time.time()
            
            try:
                self.logger.info("Starting LangGraph-native MATLAB analysis...")
                
                # Update state status
                state = update_state_status(state, ConversionStatus.ANALYZING)
                
                # Get MATLAB path from state
                matlab_path = Path(state["request"].matlab_path)
                
                # Check if we have cached analysis in memory
                cache_key = f"matlab_analysis_{matlab_path.resolve()}"
                cached_analysis = self.get_memory(cache_key, "long_term")
                
                if cached_analysis and not self._should_refresh_analysis(cached_analysis):
                    self.logger.info("Using cached MATLAB analysis from memory")
                    analysis_result = cached_analysis
                else:
                    self.logger.info("Performing fresh MATLAB analysis")
                    analysis_result = self._perform_analysis(matlab_path)
                    
                    # Cache the analysis in long-term memory
                    self.update_memory(cache_key, analysis_result, "long_term")
                
                # Update agent memory with analysis context
                self.update_memory("last_analysis", analysis_result, "short_term")
                self.update_memory("analysis_count", 
                                 (self.get_memory("analysis_count", "short_term") or 0) + 1, 
                                 "short_term")
                
                # Update state with analysis results
                state["matlab_analysis"] = analysis_result
                state["is_multi_file"] = analysis_result.get('files_analyzed', 0) > 1
                
                # Update state with agent memory
                state = self.update_state_with_result(state, analysis_result, "matlab_analysis")
                
                # Log success
                duration = time.time() - start_time
                self.logger.info(f"MATLAB analysis completed successfully in {duration:.2f}s")
                self.logger.info(f"Files analyzed: {analysis_result.get('files_analyzed', 1)}")
                self.logger.info(f"Multi-file project: {state['is_multi_file']}")
                
                # Track performance
                self.track_performance("matlab_analysis", start_time, time.time(), True, {
                    "files_analyzed": analysis_result.get('files_analyzed', 1),
                    "is_multi_file": state["is_multi_file"]
                })
                
            except Exception as e:
                self.logger.error(f"Error in MATLAB analysis: {e}")
                state["error_message"] = f"MATLAB analysis failed: {str(e)}"
                state = update_state_status(state, ConversionStatus.FAILED)
                
                # Track failure
                self.track_performance("matlab_analysis", start_time, time.time(), False, {
                    "error": str(e)
                })
            
            # Record processing time
            duration = time.time() - start_time
            state = add_processing_time(state, "matlab_analysis", duration)
            
            return state
        
        return analyze_matlab_node
    
    def get_tools(self) -> List[Any]:
        """Get available tools for this agent."""
        return [
            self.matlab_parser_tool,
            self.llm_analysis_tool
        ]
    
    def _perform_analysis(self, matlab_path: Path) -> Dict[str, Any]:
        """
        Perform comprehensive MATLAB analysis using LangGraph tools.
        
        Args:
            matlab_path: Path to MATLAB file or directory
            
        Returns:
            Comprehensive analysis results
        """
        self.logger.debug(f"Analyzing MATLAB content at: {matlab_path}")
        
        # Get all MATLAB files
        matlab_files = self._get_matlab_files(matlab_path)
        if not matlab_files:
            raise ValueError(f"No MATLAB files found in {matlab_path}")
        
        # Analyze each file
        file_analyses = []
        total_functions = 0
        total_dependencies = 0
        all_dependencies = set()
        all_numerical_calls = set()
        function_call_tree = {}
        dependency_map = {}
        
        for file_path in matlab_files:
            self.logger.debug(f"Analyzing file: {file_path}")
            
            # Read file content
            file_content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Use MATLAB parser tool
            parse_result = self.matlab_parser_tool(file_content, str(file_path))
            
            if not parse_result.success:
                self.logger.warning(f"Failed to parse {file_path}: {parse_result.error}")
                continue
            
            parsed_structure = parse_result.data
            
            # Use LLM analysis tool for deeper understanding
            llm_result = self.llm_analysis_tool(
                file_content, 
                parsed_structure, 
                "algorithmic"
            )
            
            # Combine parsing and LLM analysis
            file_analysis = {
                'file_path': str(file_path),
                'source_code': file_content,  # Include source code for accurate conversion
                'parsed_structure': parsed_structure,
                'llm_analysis': llm_result.data if llm_result.success else {},
                'file_size': len(file_content),
                'line_count': parsed_structure.get('line_count', 0)
            }
            
            file_analyses.append(file_analysis)
            
            # Aggregate statistics
            total_functions += len(parsed_structure.get('functions', []))
            total_dependencies += len(parsed_structure.get('dependencies', []))
            all_dependencies.update(parsed_structure.get('dependencies', []))
            all_numerical_calls.update(parsed_structure.get('numerical_calls', []))
            
            # Build function call tree
            function_calls = parsed_structure.get('function_calls', {})
            for func_name, called_funcs in function_calls.items():
                function_call_tree[func_name] = called_funcs
                
                # Build dependency map
                if func_name not in dependency_map:
                    dependency_map[func_name] = {
                        'defined_in': str(file_path),
                        'called_by': [],
                        'calls': []
                    }
                dependency_map[func_name]['calls'] = called_funcs
        
        # Resolve dependency relationships
        for func_name, dep_info in dependency_map.items():
            for called_func in dep_info['calls']:
                if called_func in dependency_map:
                    if func_name not in dependency_map[called_func]['called_by']:
                        dependency_map[called_func]['called_by'].append(func_name)
        
        # Assess overall complexity
        complexity_assessment = self._assess_overall_complexity(file_analyses)
        
        # Create project understanding
        project_understanding = self._create_project_understanding(file_analyses)
        
        # Build comprehensive analysis result
        analysis_result = {
            'files_analyzed': len(file_analyses),
            'file_analyses': file_analyses,
            'total_functions': total_functions,
            'total_dependencies': total_dependencies,
            'matlab_functions_used': sorted(list(all_dependencies)),
            'numerical_calls_used': sorted(list(all_numerical_calls)),
            'complexity_assessment': complexity_assessment,
            'project_understanding': project_understanding,
            'function_call_tree': function_call_tree,
            'dependency_map': dependency_map,
            'analysis_timestamp': time.time(),
            'agent_memory': {
                'analysis_count': self.get_memory("analysis_count", "short_term") or 0,
                'last_analysis_time': time.time()
            }
        }
        
        self.logger.info(f"Analysis complete: {len(file_analyses)} files, {total_functions} functions")
        
        return analysis_result
    
    def _get_matlab_files(self, path: Path) -> List[Path]:
        """Get all MATLAB files from path."""
        if path.is_file() and path.suffix.lower() == '.m':
            return [path]
        elif path.is_dir():
            return list(path.glob('**/*.m'))
        else:
            return []
    
    def _should_refresh_analysis(self, cached_analysis: Dict[str, Any]) -> bool:
        """
        Determine if cached analysis should be refreshed.
        
        Args:
            cached_analysis: Previously cached analysis
            
        Returns:
            True if analysis should be refreshed
        """
        # Refresh if analysis is older than 1 hour
        analysis_time = cached_analysis.get('analysis_timestamp', 0)
        return time.time() - analysis_time > 3600
    
    def _assess_overall_complexity(self, file_analyses: List[Dict[str, Any]]) -> str:
        """Assess overall project complexity."""
        if not file_analyses:
            return "Unknown"
        
        complexity_levels = []
        total_files = len(file_analyses)
        total_functions = 0
        total_lines = 0
        
        for analysis in file_analyses:
            parsed = analysis['parsed_structure']
            complexity = parsed.get('complexity', {})
            complexity_levels.append(complexity.get('level', 'Low'))
            total_functions += len(parsed.get('functions', []))
            total_lines += complexity.get('total_lines', 0)
        
        # Determine overall complexity
        if 'High' in complexity_levels or total_files > 10 or total_functions > 50 or total_lines > 5000:
            return "High"
        elif 'Medium' in complexity_levels or total_files > 3 or total_functions > 10 or total_lines > 1000:
            return "Medium"
        else:
            return "Low"
    
    def _create_project_understanding(self, file_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create comprehensive project understanding."""
        if not file_analyses:
            return {
                'main_purpose': 'Unknown project',
                'domain': 'Unknown',
                'key_algorithms': [],
                'architecture': 'Unknown',
                'complexity_level': 'Unknown',
                'conversion_challenges': [],
                'recommendations': [],
                'confidence': 0.0
            }
        
        # Aggregate information from all files
        algorithms = set()
        domains = set()
        challenges = set()
        suggestions = set()
        
        for analysis in file_analyses:
            llm_analysis = analysis.get('llm_analysis', {})
            
            if 'algorithms' in llm_analysis:
                algorithms.update(llm_analysis['algorithms'])
            if 'domain' in llm_analysis:
                domains.add(llm_analysis['domain'])
            if 'challenges' in llm_analysis:
                challenges.update(llm_analysis['challenges'])
            if 'suggestions' in llm_analysis:
                suggestions.update(llm_analysis['suggestions'])
        
        # Determine primary domain
        domain = list(domains)[0] if len(domains) == 1 else 'Mixed' if domains else 'General'
        
        # Assess complexity level
        complexity_level = self._assess_overall_complexity(file_analyses)
        
        return {
            'main_purpose': 'MATLAB to C++ conversion project',
            'domain': domain,
            'key_algorithms': sorted(list(algorithms)),
            'architecture': 'Modular C++ design with Eigen integration',
            'complexity_level': complexity_level,
            'conversion_challenges': sorted(list(challenges)),
            'recommendations': sorted(list(suggestions)),
            'confidence': 0.8 if algorithms else 0.5
        }
    
    def get_analysis_summary(self, state: ConversionState) -> Dict[str, Any]:
        """
        Get a summary of the analysis results from state.
        
        Args:
            state: Current conversion state
            
        Returns:
            Analysis summary
        """
        matlab_analysis = state.get("matlab_analysis", {})
        
        return {
            'files_analyzed': matlab_analysis.get('files_analyzed', 0),
            'total_functions': matlab_analysis.get('total_functions', 0),
            'complexity_level': matlab_analysis.get('complexity_assessment', 'Unknown'),
            'is_multi_file': state.get('is_multi_file', False),
            'domain': matlab_analysis.get('project_understanding', {}).get('domain', 'Unknown'),
            'key_algorithms': matlab_analysis.get('project_understanding', {}).get('key_algorithms', []),
            'analysis_timestamp': matlab_analysis.get('analysis_timestamp', 0),
            'agent_performance': self.get_performance_summary()
        }
