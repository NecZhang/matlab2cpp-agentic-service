"""
Enhanced LangGraph MATLAB to C++ Conversion Workflow

This workflow integrates the 5 streamlined agents:
- EnhancedMATLABAnalyzer
- EnhancedConversionPlanner  
- EnhancedCppGenerator
- EnhancedQualityAssessor
- MultiFileProjectManager

Features:
- Real-time compilation testing
- Adaptive strategy selection
- Multi-file project support
- Learning from previous conversions
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

from langgraph.graph import StateGraph, END
from loguru import logger

from ..agents.base.langgraph_agent import AgentConfig
from ..agents import (
    MATLABAnalyzer,
    ConversionPlanner,
    CppGenerator,
    QualityAssessor,
    ProjectManager
)
from ...infrastructure.state.conversion_state import ConversionState
from ...infrastructure.tools.llm_client import LLMClient
from ..reporting.conversion_report_generator import ConversionReportGenerator


class EnhancedLangGraphMATLAB2CPPWorkflow:
    """
    Enhanced LangGraph workflow with streamlined agent architecture.
    
    Features:
    - 5 intelligent agents working in coordination
    - Real-time compilation testing
    - Adaptive strategy selection
    - Multi-file project support
    - Learning from previous conversions
    """
    
    def __init__(self, config: AgentConfig, llm_clients: Dict[str, LLMClient]):
        self.config = config
        self.llm_clients = llm_clients
        # Backward compatibility - provide default llm_client as reasoning client
        self.llm_client = llm_clients.get("reasoning")
        self.logger = logger.bind(name="enhanced_workflow")
        
        # Initialize 5 streamlined agents with selective LLM clients
        self.logger.info("DEBUG: About to call _initialize_enhanced_agents")
        self.agents = self._initialize_enhanced_agents()
        self.logger.info("DEBUG: _initialize_enhanced_agents completed successfully")
        
        # Create LangGraph workflow
        self.workflow = self._create_enhanced_workflow()
        
        # Initialize report generator
        self.report_generator = ConversionReportGenerator()
        self.report_generator.logger = self.logger
        
        # Workflow state
        self.workflow_state = {}
        self.performance_metrics = {}
        
        self.logger.info("Enhanced LangGraph MATLAB2CPP Workflow initialized with selective LLM clients")
    
    def _initialize_enhanced_agents(self) -> Dict[str, Any]:
        """Initialize the 5 streamlined agents with selective LLM clients."""
        agents = {}
        
        # Get reasoning client for most agents
        reasoning_client = self.llm_clients.get("reasoning")
        # Get C++ generation client for the generator
        cpp_generation_client = self.llm_clients.get("cpp_generation")
        
        self.logger.info(f"Available LLM clients: {list(self.llm_clients.keys())}")
        self.logger.info(f"Reasoning client: {reasoning_client}")
        self.logger.info(f"C++ generation client: {cpp_generation_client}")
        
        # Debug: Check what's actually in the clients dictionary
        self.logger.info("INFO: Starting client debug check...")
        for key, client in self.llm_clients.items():
            if hasattr(client, 'config'):
                self.logger.info(f"INFO: CLIENT DEBUG: {key} -> model: {getattr(client.config, 'model', 'unknown')}, endpoint: {getattr(client.config, 'vllm_endpoint', 'unknown')}")
            else:
                self.logger.info(f"INFO: CLIENT DEBUG: {key} -> no config attribute")
        self.logger.info("INFO: Finished client debug check...")
        
        # CRITICAL DEBUG: Check if the clients are actually different
        if reasoning_client is cpp_generation_client:
            self.logger.error("CRITICAL ERROR: Reasoning and coding clients are the SAME object!")
        else:
            self.logger.info("INFO: Reasoning and coding clients are different objects - good!")
        
        # CRITICAL DEBUG: Check what client the C++ Generator will receive
        self.logger.info(f"INFO: C++ Generator will receive client: {cpp_generation_client}")
        if hasattr(cpp_generation_client, 'config'):
            self.logger.info(f"INFO: C++ Generator client config - model: {getattr(cpp_generation_client.config, 'model', 'unknown')}, endpoint: {getattr(cpp_generation_client.config, 'vllm_endpoint', 'unknown')}")
        else:
            self.logger.info("INFO: C++ Generator client has no config attribute")
        
        # CRITICAL DEBUG: Check what client the C++ Generator will receive
        self.logger.info(f"INFO: C++ Generator will receive client: {cpp_generation_client}")
        if hasattr(cpp_generation_client, 'config'):
            self.logger.info(f"INFO: C++ Generator client config - model: {getattr(cpp_generation_client.config, 'model', 'unknown')}, endpoint: {getattr(cpp_generation_client.config, 'vllm_endpoint', 'unknown')}")
        else:
            self.logger.info("INFO: C++ Generator client has no config attribute")
        
        # CRITICAL DEBUG: Check what client the C++ Generator will receive
        self.logger.info(f"INFO: C++ Generator will receive client: {cpp_generation_client}")
        if hasattr(cpp_generation_client, 'config'):
            self.logger.info(f"INFO: C++ Generator client config - model: {getattr(cpp_generation_client.config, 'model', 'unknown')}, endpoint: {getattr(cpp_generation_client.config, 'vllm_endpoint', 'unknown')}")
        else:
            self.logger.info("INFO: C++ Generator client has no config attribute")
        
        # Debug: Check if clients are the same (fallback scenario)
        if reasoning_client is cpp_generation_client:
            self.logger.warning("WARNING: Reasoning and coding clients are the same object - fallback may have been triggered!")
        else:
            self.logger.info("INFO: Reasoning and coding clients are different objects - good!")
        
        # Initialize MATLAB Analyzer with reasoning client
        try:
            self.logger.info("Initializing MATLAB Analyzer...")
            agents['analyzer'] = MATLABAnalyzer(
                config=self.config,
                llm_client=reasoning_client
            )
            self.logger.info("MATLAB Analyzer initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize MATLAB Analyzer: {e}")
            raise
        
        # Initialize Conversion Planner with reasoning client
        try:
            self.logger.info("Initializing Conversion Planner...")
            agents['planner'] = ConversionPlanner(
                config=self.config,
                llm_client=reasoning_client
            )
            self.logger.info("Conversion Planner initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Conversion Planner: {e}")
            raise
        
        # Initialize C++ Generator with C++ generation client (direct model)
        try:
            self.logger.info("Initializing C++ Generator...")
            self.logger.info(f"DEBUG: cpp_generation_client type: {type(cpp_generation_client)}")
            if hasattr(cpp_generation_client, 'config'):
                self.logger.info(f"DEBUG: cpp_generation_client config - model: {getattr(cpp_generation_client.config, 'model', 'unknown')}")
                self.logger.info(f"DEBUG: cpp_generation_client config - endpoint: {getattr(cpp_generation_client.config, 'vllm_endpoint', 'unknown')}")
            
            # Debug: Check if this is actually the coding client
            expected_coding_model = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
            actual_model = getattr(cpp_generation_client.config, 'model', 'unknown') if hasattr(cpp_generation_client, 'config') else 'unknown'
            if actual_model == expected_coding_model:
                self.logger.info("✅ INFO: C++ Generator is receiving the correct coding client")
            else:
                self.logger.error(f"❌ INFO: C++ Generator is receiving WRONG client! Expected: {expected_coding_model}, Got: {actual_model}")
            
            agents['generator'] = CppGenerator(
                config=self.config,
                llm_client=cpp_generation_client
            )
            self.logger.info("C++ Generator initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize C++ Generator: {e}")
            raise
        
        # Initialize Quality Assessor with reasoning client
        try:
            self.logger.info("Initializing Quality Assessor...")
            agents['assessor'] = QualityAssessor(
                config=self.config,
                llm_client=reasoning_client
            )
            self.logger.info("Quality Assessor initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Quality Assessor: {e}")
            raise
        
        # Initialize Project Manager with reasoning client
        try:
            self.logger.info("Initializing Project Manager...")
            agents['project_manager'] = ProjectManager(
                config=self.config,
                llm_client=reasoning_client
            )
            self.logger.info("Project Manager initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Project Manager: {e}")
            raise
        
        self.logger.info("Initialized 5 streamlined agents with selective LLM clients")
        self.logger.info(f"- Analyzer: {type(agents['analyzer']).__name__} with reasoning model")
        self.logger.info(f"- Planner: {type(agents['planner']).__name__} with reasoning model")
        self.logger.info(f"- Generator: {type(agents['generator']).__name__} with direct model (Qwen Coder)")
        self.logger.info(f"- Assessor: {type(agents['assessor']).__name__} with reasoning model")
        self.logger.info(f"- Project Manager: {type(agents['project_manager']).__name__} with reasoning model")
        
        return agents
    
    def get_workflow_stats(self, state: ConversionState) -> Dict[str, Any]:
        """Get workflow statistics from the state."""
        return {
            'total_processing_time': state.get('total_processing_time', 0.0),
            'current_agent': state.get('current_agent'),
            'workflow_step': state.get('workflow_step'),
            'generated_files_count': len(state.get('generated_files', [])),
            'quality_score': state.get('quality_scores', {}).get('overall_score', 0.0),
            'is_multi_file': state.get('is_multi_file', False),
            'optimization_complete': state.get('optimization_complete', False)
        }
    
    def _initialize_enhanced_agents(self) -> Dict[str, Any]:
        """Initialize the 5 streamlined agents."""
        agents = {}
        
        # Get reasoning client for most agents
        reasoning_client = self.llm_clients.get("reasoning")
        # Get C++ generation client for the generator
        cpp_generation_client = self.llm_clients.get("cpp_generation")
        
        self.logger.info(f"Available LLM clients: {list(self.llm_clients.keys())}")
        self.logger.info(f"Reasoning client: {reasoning_client}")
        self.logger.info(f"C++ generation client: {cpp_generation_client}")
        
        # Debug: Check what's actually in the clients dictionary
        self.logger.info("INFO: Starting client debug check...")
        for key, client in self.llm_clients.items():
            if hasattr(client, 'config'):
                self.logger.info(f"INFO: CLIENT DEBUG: {key} -> model: {getattr(client.config, 'model', 'unknown')}, endpoint: {getattr(client.config, 'vllm_endpoint', 'unknown')}")
            else:
                self.logger.info(f"INFO: CLIENT DEBUG: {key} -> no config attribute")
        self.logger.info("INFO: Finished client debug check...")
        
        # CRITICAL DEBUG: Check if the clients are actually different
        if reasoning_client is cpp_generation_client:
            self.logger.error("CRITICAL ERROR: Reasoning and coding clients are the SAME object!")
        else:
            self.logger.info("INFO: Reasoning and coding clients are different objects - good!")
        
        # CRITICAL DEBUG: Check what client the C++ Generator will receive
        self.logger.info(f"INFO: C++ Generator will receive client: {cpp_generation_client}")
        if hasattr(cpp_generation_client, 'config'):
            self.logger.info(f"INFO: C++ Generator client config - model: {getattr(cpp_generation_client.config, 'model', 'unknown')}, endpoint: {getattr(cpp_generation_client.config, 'vllm_endpoint', 'unknown')}")
        else:
            self.logger.info("INFO: C++ Generator client has no config attribute")
        
        # Initialize MATLAB Analyzer with reasoning client
        try:
            self.logger.info("Initializing MATLAB Analyzer...")
            agents['analyzer'] = MATLABAnalyzer(
                config=self.config,
                llm_client=reasoning_client
            )
            self.logger.info("MATLAB Analyzer initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize MATLAB Analyzer: {e}")
            raise
        
        # Initialize Conversion Planner with reasoning client
        try:
            self.logger.info("Initializing Conversion Planner...")
            agents['planner'] = ConversionPlanner(
                config=self.config,
                llm_client=reasoning_client
            )
            self.logger.info("Conversion Planner initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Conversion Planner: {e}")
            raise
        
        # Initialize C++ Generator with C++ generation client (direct model)
        try:
            self.logger.info("Initializing C++ Generator...")
            self.logger.info(f"DEBUG: cpp_generation_client type: {type(cpp_generation_client)}")
            if hasattr(cpp_generation_client, 'config'):
                self.logger.info(f"DEBUG: cpp_generation_client config - model: {getattr(cpp_generation_client.config, 'model', 'unknown')}")
                self.logger.info(f"DEBUG: cpp_generation_client config - endpoint: {getattr(cpp_generation_client.config, 'vllm_endpoint', 'unknown')}")
            
            # Debug: Check if this is actually the coding client
            expected_coding_model = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
            actual_model = getattr(cpp_generation_client.config, 'model', 'unknown') if hasattr(cpp_generation_client, 'config') else 'unknown'
            if actual_model == expected_coding_model:
                self.logger.info("✅ INFO: C++ Generator is receiving the correct coding client")
            else:
                self.logger.error(f"❌ INFO: C++ Generator is receiving WRONG client! Expected: {expected_coding_model}, Got: {actual_model}")
            
            agents['generator'] = CppGenerator(
                config=self.config,
                llm_client=cpp_generation_client
            )
            self.logger.info("C++ Generator initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize C++ Generator: {e}")
            raise
        
        # Initialize Quality Assessor with reasoning client
        try:
            self.logger.info("Initializing Quality Assessor...")
            agents['assessor'] = QualityAssessor(
                config=self.config,
                llm_client=reasoning_client
            )
            self.logger.info("Quality Assessor initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Quality Assessor: {e}")
            raise
        
        # Initialize Project Manager with reasoning client
        try:
            self.logger.info("Initializing Project Manager...")
            agents['project_manager'] = ProjectManager(
                config=self.config,
                llm_client=reasoning_client
            )
            self.logger.info("Project Manager initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Project Manager: {e}")
            raise
        
        return agents
    
    def _create_enhanced_workflow(self) -> StateGraph:
        """Create enhanced LangGraph workflow."""
        workflow = StateGraph(ConversionState)
        
        # Add nodes
        workflow.add_node("analyze", self._analyze_node)
        workflow.add_node("plan", self._plan_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("assess", self._assess_node)
        workflow.add_node("coordinate", self._coordinate_node)
        
        # Add observation nodes for learning
        workflow.add_node("observe_analyze", self._observe_analyze_results_node)
        workflow.add_node("observe_plan", self._observe_plan_results_node)
        workflow.add_node("observe_generate", self._observe_generate_results_node)
        workflow.add_node("observe_assess", self._observe_assess_results_node)
        workflow.add_node("observe_coordinate", self._observe_coordinate_results_node)
        
        # Set entry point
        workflow.set_entry_point("analyze")
        
        # Add edges
        workflow.add_edge("analyze", "observe_analyze")
        workflow.add_edge("observe_analyze", "plan")
        workflow.add_edge("plan", "observe_plan")
        workflow.add_edge("observe_plan", "generate")
        workflow.add_edge("generate", "observe_generate")
        workflow.add_edge("observe_generate", "assess")
        workflow.add_edge("assess", "observe_assess")
        
        # Conditional routing for multi-file projects
        workflow.add_conditional_edges(
            "observe_assess",
            self._should_coordinate_project,
            {
                "coordinate": "coordinate",
                "complete": END
            }
        )
        
        workflow.add_edge("coordinate", "observe_coordinate")
        workflow.add_edge("observe_coordinate", END)
        
        return workflow.compile()
    
    async def _analyze_node(self, state: ConversionState) -> ConversionState:
        """Analyze MATLAB code with enhanced analyzer."""
        try:
            matlab_path_str = state.get("matlab_path", "")
            self.logger.info(f"DEBUG: matlab_path from state: '{matlab_path_str}'")
            matlab_path = Path(matlab_path_str)
            self.logger.info(f"DEBUG: matlab_path as Path: '{matlab_path}'")
            if not matlab_path.exists():
                raise ValueError(f"MATLAB path does not exist: {matlab_path}")
            
            self.logger.info(f"Starting MATLAB analysis: {matlab_path}")
            return await self.agents['analyzer'].analyze_project(matlab_path, state)
            
        except Exception as e:
            self.logger.error(f"MATLAB analysis failed: {e}")
            state["error"] = f"Analysis failed: {e}"
            return state
    
    async def _plan_node(self, state: ConversionState) -> ConversionState:
        """Plan conversion with enhanced planner."""
        try:
            matlab_analysis = state.get("matlab_analysis", {})
            if not matlab_analysis:
                raise ValueError("No MATLAB analysis available for planning")
            
            self.logger.info("Starting conversion planning")
            return await self.agents['planner'].plan_conversion(matlab_analysis, state)
            
        except Exception as e:
            self.logger.error(f"Conversion planning failed: {e}")
            state["error"] = f"Planning failed: {e}"
            return state
    
    async def _generate_node(self, state: ConversionState) -> ConversionState:
        """Generate C++ code with enhanced generator."""
        try:
            conversion_plan = state.get("conversion_plan", {})
            matlab_analysis = state.get("matlab_analysis", {})
            
            if not conversion_plan or not matlab_analysis:
                raise ValueError("Missing conversion plan or MATLAB analysis")
            
            self.logger.info("Starting C++ generation with testing")
            return await self.agents['generator'].generate_with_testing(
                conversion_plan, matlab_analysis, state
            )
            
        except Exception as e:
            self.logger.error(f"C++ generation failed: {e}")
            state["error"] = f"Generation failed: {e}"
            return state
    
    async def _assess_node(self, state: ConversionState) -> ConversionState:
        """Assess quality with enhanced assessor."""
        try:
            generated_code = state.get("generated_code", {})
            conversion_plan = state.get("conversion_plan", {})
            matlab_analysis = state.get("matlab_analysis", {})
            
            if not generated_code:
                raise ValueError("No generated code available for assessment")
            
            # Get compilation result from generation - handle nested structure
            compilation_result = generated_code.get('compilation_result', {})
            actual_generated_code = generated_code.get('generated_code', generated_code)
            
            self.logger.info(f"Starting quality assessment with compilation result: {compilation_result.get('success', 'unknown')}")
            self.logger.info(f"DEBUG: Compilation result keys: {list(compilation_result.keys()) if compilation_result else 'Empty'}")
            
            return await self.agents['assessor'].assess_with_compilation_results(
                actual_generated_code, compilation_result, matlab_analysis, conversion_plan, state
            )
            
        except Exception as e:
            self.logger.error(f"Quality assessment failed: {e}")
            state["error"] = f"Assessment failed: {e}"
            return state
    
    async def _coordinate_node(self, state: ConversionState) -> ConversionState:
        """Coordinate multi-file project with specialized manager."""
        try:
            project_structure = state.get("matlab_analysis", {}).get('project_structure', {})
            generated_code = state.get("generated_code", {})
            conversion_plan = state.get("conversion_plan", {})
            
            if not project_structure or not generated_code:
                raise ValueError("Missing project structure or generated code")
            
            self.logger.info("Starting multi-file project coordination")
            return await self.agents['project_manager'].coordinate_multi_file_project(
                project_structure, generated_code, conversion_plan, state
            )
            
        except Exception as e:
            self.logger.error(f"Multi-file coordination failed: {e}")
            state["error"] = f"Coordination failed: {e}"
            return state
    
    async def _observe_analyze_results_node(self, state: ConversionState) -> ConversionState:
        """Observe and learn from analysis results."""
        try:
            matlab_analysis = state.get("matlab_analysis", {})
            if matlab_analysis:
                # Update memory with analysis insights
                self.agents['analyzer'].update_memory(
                    "analysis_patterns", 
                    matlab_analysis.get('conversion_patterns', {}),
                    "long_term"
                )
                
                self.logger.info("Analysis results observed and stored")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Analysis observation failed: {e}")
            return state
    
    async def _observe_plan_results_node(self, state: ConversionState) -> ConversionState:
        """Observe and learn from planning results."""
        try:
            conversion_plan = state.get("conversion_plan", {})
            if conversion_plan:
                # Update memory with planning insights
                self.agents['planner'].update_memory(
                    "planning_strategies",
                    conversion_plan.get('coordination_strategy', {}),
                    "long_term"
                )
                
                self.logger.info("Planning results observed and stored")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Planning observation failed: {e}")
            return state
    
    async def _observe_generate_results_node(self, state: ConversionState) -> ConversionState:
        """Observe and learn from generation results."""
        try:
            generated_code = state.get("generated_code", {})
            if generated_code:
                # Update memory with generation insights
                generation_strategy = generated_code.get('generation_strategy', 'unknown')
                compilation_success_rate = generated_code.get('compilation_success_rate', 0.0)
                
                self.agents['generator'].update_memory(
                    "generation_performance",
                    {
                        "strategy": generation_strategy,
                        "success_rate": compilation_success_rate,
                        "timestamp": time.time()
                    },
                    "long_term"
                )
                
                self.logger.info("Generation results observed and stored")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Generation observation failed: {e}")
            return state
    
    async def _observe_assess_results_node(self, state: ConversionState) -> ConversionState:
        """Observe and learn from assessment results."""
        try:
            quality_assessment = state.get("quality_assessment", {})
            if quality_assessment:
                # Update memory with quality insights
                overall_score = quality_assessment.get('overall_quality_score', 0.0)
                quality_level = quality_assessment.get('quality_level', 'unknown')
                
                self.agents['assessor'].update_memory(
                    "quality_trends",
                    {
                        "score": overall_score,
                        "level": quality_level,
                        "timestamp": time.time()
                    },
                    "long_term"
                )
                
                self.logger.info("Assessment results observed and stored")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Assessment observation failed: {e}")
            return state
    
    async def _observe_coordinate_results_node(self, state: ConversionState) -> ConversionState:
        """Observe and learn from coordination results."""
        try:
            coordination_result = state.get("multi_file_coordination", {})
            if coordination_result:
                # Update memory with coordination insights
                coordination_strategy = coordination_result.get('coordination_strategy', {})
                validation_result = coordination_result.get('validation_result', {})
                
                self.agents['project_manager'].update_memory(
                    "coordination_performance",
                    {
                        "strategy": coordination_strategy.get('selected_strategy', 'unknown'),
                        "validation_success": validation_result.get('validation_success', False),
                        "timestamp": time.time()
                    },
                    "long_term"
                )
                
                self.logger.info("Coordination results observed and stored")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Coordination observation failed: {e}")
            return state
    
    def _should_coordinate_project(self, state: ConversionState) -> str:
        """Determine if multi-file project coordination is needed."""
        try:
            # Check if this is a multi-file project
            is_multi_file = state.get("is_multi_file", False)
            
            # Determine latest quality score (normalize to 0-1 scale)
            quality_assessment = state.get("quality_assessment") or {}
            quality_scores = state.get("quality_scores") or {}
            
            normalized_score = None
            
            # Prefer explicit 0-10 score
            score_10 = quality_assessment.get('overall_quality_score_10')
            if score_10 is not None:
                normalized_score = score_10 / 10.0
            else:
                # Fall back to 0-1 metrics
                score_1 = quality_assessment.get('overall_quality_score')
                if score_1 is not None:
                    normalized_score = score_1
            
            if normalized_score is None:
                # Fall back to legacy quality_scores structure (already 0-10)
                legacy_score_10 = quality_scores.get('overall_score')
                if legacy_score_10 is not None:
                    normalized_score = legacy_score_10 / 10.0
            
            # Inspect compilation outcome from generation step, if available
            compilation_failed = False
            generated_payload = state.get("generated_code") or {}
            compilation_result = None
            
            if isinstance(generated_payload, dict):
                compilation_result = generated_payload.get('compilation_result')
                if not compilation_result and 'generated_code' in generated_payload:
                    inner_payload = generated_payload.get('generated_code')
                    if isinstance(inner_payload, dict):
                        compilation_result = inner_payload.get('compilation_result')
            
            if isinstance(compilation_result, dict):
                compilation_failed = not compilation_result.get('success', False)
            
            # Decide on coordination: require it when quality is unknown/poor or compilation failed
            needs_coordination = is_multi_file and (
                compilation_failed or
                normalized_score is None or
                normalized_score < 0.8
            )
            
            if needs_coordination:
                self.logger.info("Multi-file project coordination needed")
                return "coordinate"
            
            self.logger.info("Multi-file project coordination not needed")
            return "complete"
                
        except Exception as e:
            self.logger.error(f"Coordination decision failed: {e}")
            return "complete"
    
    async def run_conversion(self, matlab_path: Path, **kwargs) -> Dict[str, Any]:
        """
        Run the enhanced MATLAB to C++ conversion workflow.
        
        Args:
            matlab_path: Path to MATLAB file or directory
            **kwargs: Additional parameters
            
        Returns:
            Conversion results
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting enhanced MATLAB to C++ conversion: {matlab_path}")
            
            # Initialize state
            matlab_path_obj = Path(matlab_path) if isinstance(matlab_path, str) else matlab_path
            request = kwargs.get('request')
            initial_state: ConversionState = {
                'request': request,
                'result': kwargs.get('result'),
                'matlab_path': str(matlab_path_obj),
                'project_name': request.project_name if request else kwargs.get('project_name', 'converted_project'),
                'build_system': request.build_system if request else kwargs.get('build_system', 'gcc'),
                'agent_memory': {},
                'human_feedback': [],
                'streaming_updates': [],
                'parallel_tasks': {},
                'checkpoint_data': {},
                'error_recovery_state': None,
                'agent_performance': {},
                'system_metrics': [],
                'operation_results': {},
                'current_agent': None,
                'current_operation': None,
                'workflow_step': 'initialization',
                'needs_human_intervention': False,
                'optimization_complete': False,
                'processing_times': {},
                'total_processing_time': 0.0,
                'start_time': time.time(),
                'error_context': {},
                'retry_count': 0,
                'max_retries': kwargs.get('max_optimization_turns', 3),
                'matlab_analysis': None,
                'is_multi_file': False,
                'conversion_plan': None,
                'project_structure_plan': None,
                'generated_code': None,
                'quality_scores': None,
                'assessment_reports': [],
                'current_turn': 0,
                'project_output_dir': str(Path(kwargs.get('output_dir', 'output')) / kwargs.get('project_name', 'unknown_project')),
                'generated_files': [],
                'error_message': None
            }
            
            # Debug logging
            self.logger.info(f"DEBUG: Initial state matlab_path: '{initial_state.get('matlab_path', 'NOT_FOUND')}'")
            
            # Run workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            # Calculate performance metrics
            execution_time = time.time() - start_time
            
            # Debug: Log final state
            self.logger.info(f"DEBUG: Final state type: {type(final_state)}")
            if final_state:
                self.logger.info(f"DEBUG: Final state keys: {list(final_state.keys()) if isinstance(final_state, dict) else 'Not a dict'}")
            else:
                self.logger.warning("DEBUG: Final state is None or empty")
            
            # Safely extract data from final_state
            if final_state is None:
                self.logger.error("Final state is None, creating empty state")
                final_state = {}
            
            generated_files = final_state.get('generated_files', [])
            generated_code = final_state.get('generated_code', {})
            
            # Extract file names from generated code
            if generated_code and isinstance(generated_code, dict):
                # Check if it's the nested structure from CppGenerator
                if 'generated_code' in generated_code and 'files' in generated_code['generated_code']:
                    generated_files = list(generated_code['generated_code']['files'].keys())
                elif 'files' in generated_code:
                    generated_files = list(generated_code['files'].keys())
                else:
                    generated_files = []
            
            # Also check matlab_analysis for file count
            matlab_analysis = final_state.get('matlab_analysis', {})
            if matlab_analysis and 'file_analyses' in matlab_analysis:
                file_count = len(matlab_analysis['file_analyses'])
                self.logger.info(f"MATLAB analysis found {file_count} files")
            
            # Save generated files to disk
            if generated_files and generated_code:
                self.logger.info(f"DEBUG: About to save files. generated_code type: {type(generated_code)}")
                self.logger.info(f"DEBUG: generated_code keys: {list(generated_code.keys()) if isinstance(generated_code, dict) else 'Not a dict'}")
                if isinstance(generated_code, dict) and 'generated_code' in generated_code:
                    self.logger.info(f"DEBUG: nested generated_code keys: {list(generated_code['generated_code'].keys())}")
                # Use project-specific output directory
                project_output_dir = final_state.get('project_output_dir', str(Path(kwargs.get('output_dir', 'output')) / kwargs.get('project_name', 'unknown_project')))
                await self._save_generated_files(generated_code, project_output_dir, final_state)
            
            # Safely extract quality score
            quality_scores = final_state.get('quality_scores') or {}
            quality_score = quality_scores.get('overall_score', 0.0) if quality_scores else 0.0
            
            self.performance_metrics = {
                'execution_time': execution_time,
                'success': 'error' not in final_state,
                'quality_score': quality_score,
                'is_multi_file': final_state.get('is_multi_file', False),
                'files_generated': len(generated_files),
                'timestamp': time.time()
            }
            
            # Generate comprehensive result
            try:
                execution_summary = await self._generate_execution_summary(final_state) if final_state else 'No execution summary available'
            except Exception as summary_error:
                self.logger.warning(f"Failed to generate execution summary: {summary_error}")
                execution_summary = 'Execution summary generation failed'
            
            # Generate comprehensive conversion report
            report_path = None
            try:
                # Use project-specific output directory
                project_output_dir = final_state.get('project_output_dir', str(Path(kwargs.get('output_dir', 'output')) / kwargs.get('project_name', 'unknown_project')))
                # Ensure final_state is a dict
                if not isinstance(final_state, dict):
                    self.logger.warning(f"final_state is not a dict: {type(final_state)}")
                    final_state = {}
                
                report_path = self.report_generator.generate_comprehensive_report(final_state, Path(project_output_dir))
                self.logger.info(f"Comprehensive conversion report generated: {report_path}")
            except Exception as report_error:
                self.logger.warning(f"Failed to generate conversion report: {report_error}")
                import traceback
                self.logger.warning(f"Report generation traceback: {traceback.format_exc()}")
            
            result = {
                'success': 'error' not in final_state,
                'state': final_state,
                'performance_metrics': self.performance_metrics,
                'workflow_version': 'enhanced_v1.0',
                'agents_used': list(self.agents.keys()),
                'generated_files': generated_files,
                'execution_summary': execution_summary,
                'conversion_report_path': str(report_path) if report_path else None
            }
            
            if 'error' in final_state:
                result['error'] = final_state['error']
            
            try:
                self.logger.info(f"Enhanced conversion completed: {execution_time:.2f}s, "
                               f"success={result['success']}, "
                               f"quality={self.performance_metrics['quality_score']:.2f}")
                
                return result
            except Exception as log_error:
                self.logger.error(f"Error in final logging: {log_error}")
                return result
            
        except Exception as e:
            self.logger.error(f"Enhanced conversion workflow failed: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'success': False,
                'error': str(e),
                'performance_metrics': {
                    'execution_time': time.time() - start_time,
                    'success': False,
                    'timestamp': time.time()
                }
            }
    
    async def _generate_execution_summary(self, final_state: ConversionState) -> str:
        """Generate human-readable execution summary."""
        summary_parts = []
        
        # Basic info
        summary_parts.append("🎯 Enhanced MATLAB to C++ Conversion Summary")
        summary_parts.append("=" * 50)
        
        # Project info
        project_name = final_state.get('project_name', 'unknown')
        is_multi_file = final_state.get('is_multi_file', False)
        summary_parts.append(f"Project: {project_name}")
        summary_parts.append(f"Type: {'Multi-file' if is_multi_file else 'Single-file'}")
        
        # Analysis results
        matlab_analysis = final_state.get('matlab_analysis', {})
        if matlab_analysis:
            summary_parts.append(f"Files analyzed: {matlab_analysis.get('files_analyzed', 0)}")
            summary_parts.append(f"Functions found: {matlab_analysis.get('total_functions', 0)}")
        
        # Generation results
        generated_code = final_state.get('generated_code', {})
        if generated_code:
            files = generated_code.get('files', {})
            summary_parts.append(f"Files generated: {len(files)}")
            
            generation_iterations = generated_code.get('generation_iterations', 0)
            compilation_success_rate = generated_code.get('compilation_success_rate', 0.0)
            summary_parts.append(f"Generation iterations: {generation_iterations}")
            summary_parts.append(f"Compilation success rate: {compilation_success_rate:.1%}")
        
        # Quality assessment
        quality_assessment = final_state.get('quality_assessment', {})
        if quality_assessment:
            overall_score = quality_assessment.get('overall_quality_score', 0.0)
            quality_level = quality_assessment.get('quality_level', 'unknown')
            summary_parts.append(f"Quality score: {overall_score:.2f} ({quality_level})")
        
        # Multi-file coordination
        coordination_result = final_state.get('multi_file_coordination', {})
        if coordination_result:
            validation_result = coordination_result.get('validation_result', {})
            validation_success = validation_result.get('validation_success', False)
            summary_parts.append(f"Multi-file validation: {'✅ Passed' if validation_success else '❌ Failed'}")
        
        # Performance metrics
        execution_time = self.performance_metrics.get('execution_time', 0.0)
        summary_parts.append(f"Execution time: {execution_time:.2f}s")
        
        return "\n".join(summary_parts)
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """Get information about the enhanced workflow."""
        return {
            'workflow_type': 'enhanced_langgraph',
            'version': '1.0',
            'agents': {
                'core_agents': [
                    'EnhancedMATLABAnalyzer',
                    'EnhancedConversionPlanner',
                    'EnhancedCppGenerator',
                    'EnhancedQualityAssessor'
                ],
                'specialized_agents': [
                    'MultiFileProjectManager'
                ]
            },
            'features': [
                'Real-time compilation testing',
                'Adaptive strategy selection',
                'Multi-file project support',
                'Learning from previous conversions',
                'Enhanced quality assessment',
                'Cross-file consistency checking'
            ],
            'total_agents': 5,
            'architecture': 'streamlined'
        }
    
    async def _save_generated_files(self, generated_code: Dict[str, Any], output_dir: str, final_state: Dict[str, Any]) -> None:
        """Save generated C++ files to disk."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Extract the actual files from the nested structure
            files_to_save = {}
            if 'generated_code' in generated_code and 'files' in generated_code['generated_code']:
                files_to_save = generated_code['generated_code']['files']
            elif 'files' in generated_code:
                files_to_save = generated_code['files']
            
            if files_to_save:
                self.logger.info(f"Saving {len(files_to_save)} generated files to {output_path}")
                
                # 🔧 POST-GENERATION FIX: Apply namespace fixes to test harness BEFORE saving
                if 'main.cpp' in files_to_save:
                    from ...infrastructure.fixing.targeted_error_fixer import TargetedErrorFixer
                    fixer = TargetedErrorFixer()
                    original_main = str(files_to_save['main.cpp'])
                    fixed_main = fixer._fix_helper_namespaces(
                        content=original_main,
                        errors=[],  # Preemptive fix, no errors yet
                        filename='main.cpp'
                    )
                    if fixed_main != original_main:
                        files_to_save['main.cpp'] = fixed_main
                        self.logger.info("✅ Applied post-generation namespace fixes to main.cpp")
                    else:
                        self.logger.info("✅ main.cpp namespace syntax already correct")
                
                # 🏗️ GENERATE CMakeLists.txt if build_system is cmake
                build_system = final_state.get('build_system', 'gcc')
                if build_system == 'cmake' and 'CMakeLists.txt' not in files_to_save:
                    from ...infrastructure.build import generate_cmake_file
                    project_name = final_state.get('project_name', 'converted_project')
                    is_multi_file = final_state.get('is_multi_file', False)
                    
                    self.logger.info(f"🏗️ Generating CMakeLists.txt for project: {project_name}")
                    cmake_content = generate_cmake_file(
                        project_name=project_name,
                        generated_files=files_to_save,
                        is_multi_file=is_multi_file
                    )
                    files_to_save['CMakeLists.txt'] = cmake_content
                    self.logger.info("✅ Generated CMakeLists.txt for CMake build")
                
                for filename, content in files_to_save.items():
                    file_path = output_path / filename
                    self.logger.info(f"DEBUG: Saving {filename} with content length: {len(content)}")
                    self.logger.info(f"DEBUG: Content preview: {repr(content[:200])}")
                    self.logger.info(f"DEBUG: Content type: {type(content)}")
                    if isinstance(content, dict):
                        self.logger.info(f"DEBUG: Content is dict with keys: {list(content.keys())}")
                    
                    # Apply post-processing fixes to the content before saving
                    from ...infrastructure.tools.langgraph_tools import CodeGenerationTool
                    # Create a temporary instance to call the method
                    temp_tool = CodeGenerationTool(None)
                    fixed_content = temp_tool._fix_corrupted_includes(str(content))
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(fixed_content)
                    self.logger.info(f"Saved: {file_path}")
            else:
                self.logger.warning("No files found to save in generated_code")
                self.logger.info(f"DEBUG: Generated code structure: {list(generated_code.keys())}")
                if 'files' in generated_code:
                    self.logger.info(f"DEBUG: Files key exists, content: {generated_code['files']}")
                else:
                    self.logger.info("DEBUG: No 'files' key in generated_code")
                
                # Also save compilation instructions if available
                if 'compilation_instructions' in generated_code:
                    instructions_path = output_path / 'compilation_instructions.txt'
                    with open(instructions_path, 'w', encoding='utf-8') as f:
                        f.write(generated_code['compilation_instructions'])
                    self.logger.info(f"Saved compilation instructions: {instructions_path}")
                
                # Save usage example if available
                if 'usage_example' in generated_code:
                    usage_path = output_path / 'usage_example.cpp'
                    with open(usage_path, 'w', encoding='utf-8') as f:
                        f.write(generated_code['usage_example'])
                    self.logger.info(f"Saved usage example: {usage_path}")
            
            # Save call graph and entry point analysis
            await self._save_call_graph_analysis(final_state, output_path)
            
            # Save compilation and execution logs
            await self._save_compilation_logs(generated_code, output_path, final_state)
                
        except Exception as e:
            self.logger.error(f"Failed to save generated files: {e}")
    
    async def _save_call_graph_analysis(self, final_state: Dict[str, Any], output_path: Path) -> None:
        """Save call graph and entry point analysis to output directory."""
        try:
            matlab_analysis = final_state.get('matlab_analysis', {})
            if not matlab_analysis:
                return
            
            analysis_path = output_path / 'matlab_analysis.txt'
            with open(analysis_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("MATLAB PROJECT ANALYSIS\n")
                f.write("=" * 80 + "\n\n")
                
                # Project overview
                f.write(f"Project Type: {'Multi-file' if matlab_analysis.get('is_multi_file', False) else 'Single-file'}\n")
                f.write(f"Files Analyzed: {matlab_analysis.get('files_analyzed', 0)}\n")
                f.write(f"Total Functions: {matlab_analysis.get('total_functions', 0)}\n")
                f.write(f"Total Dependencies: {matlab_analysis.get('total_dependencies', 0)}\n\n")
                
                # Files and functions
                f.write("=" * 80 + "\n")
                f.write("FILES AND FUNCTIONS\n")
                f.write("=" * 80 + "\n\n")
                
                file_analyses = matlab_analysis.get('file_analyses', [])
                for fa in file_analyses:
                    filename = fa.get('file_name', 'unknown')
                    functions = fa.get('functions', [])
                    f.write(f"{filename}:\n")
                    if functions:
                        for func in functions:
                            if isinstance(func, dict):
                                func_name = func.get('name', 'unnamed')
                                func_type = func.get('type', 'unknown')
                                f.write(f"  - {func_name} ({func_type})\n")
                            elif isinstance(func, str):
                                f.write(f"  - {func}\n")
                    else:
                        f.write(f"  (no functions detected)\n")
                    f.write("\n")
                
                # Call graph
                f.write("=" * 80 + "\n")
                f.write("FUNCTION CALL GRAPH\n")
                f.write("=" * 80 + "\n\n")
                
                function_call_tree = matlab_analysis.get('function_call_tree', {})
                if function_call_tree:
                    for caller, callees in sorted(function_call_tree.items()):
                        if callees:
                            f.write(f"{caller} calls:\n")
                            for callee in callees:
                                f.write(f"  → {callee}\n")
                        else:
                            f.write(f"{caller}:\n  → (no function calls)\n")
                        f.write("\n")
                else:
                    f.write("(no function calls detected)\n\n")
                
                # Entry points
                f.write("=" * 80 + "\n")
                f.write("ENTRY POINT DETECTION\n")
                f.write("=" * 80 + "\n\n")
                
                all_functions = set(function_call_tree.keys())
                called_functions = set()
                for callees in function_call_tree.values():
                    called_functions.update(callees)
                
                entry_points = all_functions - called_functions
                
                if entry_points:
                    f.write(f"Found {len(entry_points)} entry point(s):\n\n")
                    for ep in sorted(entry_points):
                        f.write(f"  ✓ {ep}\n")
                        f.write(f"    (Top-level function - not called by others)\n")
                        # Show what this entry point calls
                        if ep in function_call_tree and function_call_tree[ep]:
                            f.write(f"    Calls: {', '.join(function_call_tree[ep])}\n")
                        f.write("\n")
                    
                    f.write("\nRECOMMENDATION:\n")
                    f.write(f"  Use {', '.join(sorted(entry_points))} as main entry point(s) in main.cpp\n\n")
                else:
                    f.write("No clear entry points detected.\n")
                    f.write("(All functions are called by others, or no calls detected)\n\n")
                
                # Compilation order
                f.write("=" * 80 + "\n")
                f.write("COMPILATION ORDER\n")
                f.write("=" * 80 + "\n\n")
                
                compilation_order = matlab_analysis.get('compilation_order', [])
                if compilation_order:
                    for idx, file in enumerate(compilation_order, 1):
                        f.write(f"{idx}. {file}\n")
                else:
                    f.write("(no specific order required)\n")
                f.write("\n")
                
                # Complexity
                f.write("=" * 80 + "\n")
                f.write("COMPLEXITY ASSESSMENT\n")
                f.write("=" * 80 + "\n\n")
                
                complexity = matlab_analysis.get('complexity_assessment', {})
                f.write(f"Complexity Level: {complexity.get('complexity_level', 'unknown')}\n")
                f.write(f"Conversion Difficulty: {complexity.get('conversion_difficulty', 'unknown')}\n")
                f.write(f"Estimated Effort: {complexity.get('estimated_effort', 'unknown')}\n")
                
            self.logger.info(f"✅ Saved MATLAB analysis: {analysis_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save MATLAB analysis: {e}")
    
    async def _save_compilation_logs(self, generated_code: Dict[str, Any], output_path: Path, final_state: Dict[str, Any]) -> None:
        """Save compilation and execution logs to output directory for later analysis."""
        try:
            import time
            
            # Extract compilation result
            compilation_result = generated_code.get('compilation_result', {})
            
            if not compilation_result:
                self.logger.debug("No compilation result to save")
                return
            
            # Save compilation output log
            if 'output' in compilation_result and compilation_result['output']:
                compilation_log_path = output_path / 'compilation.log'
                with open(compilation_log_path, 'w', encoding='utf-8') as f:
                    f.write("="*80 + "\n")
                    f.write("DOCKER CONTAINER COMPILATION LOG\n")
                    f.write("="*80 + "\n\n")
                    f.write(f"Compilation Success: {compilation_result.get('success', False)}\n")
                    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    
                    # Determine project type from state, not from conversion_mode
                    is_multi_file = final_state.get('is_multi_file', False)
                    project_type = 'multi_file' if is_multi_file else 'single_file'
                    f.write(f"Project Type: {project_type}\n\n")
                    
                    f.write("="*80 + "\n")
                    f.write("COMPILER OUTPUT\n")
                    f.write("="*80 + "\n\n")
                    
                    # Split compilation output and execution output
                    full_output = compilation_result['output']
                    if "EXECUTION TEST START" in full_output:
                        # Split at execution marker
                        parts = full_output.split("EXECUTION TEST START")
                        compilation_only = parts[0].strip()
                        f.write(compilation_only)
                        f.write("\n\n[Execution output saved separately to execution.log]\n")
                    else:
                        f.write(full_output)
                    
                    f.write("\n")
                self.logger.info(f"✅ Saved compilation log: {compilation_log_path}")
            
            # Save compilation errors (if any)
            if 'errors' in compilation_result and compilation_result['errors']:
                errors_log_path = output_path / 'compilation_errors.txt'
                with open(errors_log_path, 'w', encoding='utf-8') as f:
                    f.write("="*80 + "\n")
                    f.write(f"COMPILATION ERRORS ({len(compilation_result['errors'])} total)\n")
                    f.write("="*80 + "\n\n")
                    for i, error in enumerate(compilation_result['errors'], 1):
                        error_str = str(error)
                        f.write(f"{i}. {error_str}\n")
                self.logger.info(f"✅ Saved {len(compilation_result['errors'])} errors to: {errors_log_path}")
            
            # Save compilation warnings (if any)
            if 'warnings' in compilation_result and compilation_result['warnings']:
                warnings_log_path = output_path / 'compilation_warnings.txt'
                with open(warnings_log_path, 'w', encoding='utf-8') as f:
                    f.write("="*80 + "\n")
                    f.write(f"COMPILATION WARNINGS ({len(compilation_result['warnings'])} total)\n")
                    f.write("="*80 + "\n\n")
                    for i, warning in enumerate(compilation_result['warnings'], 1):
                        warning_str = str(warning)
                        f.write(f"{i}. {warning_str}\n")
                self.logger.info(f"✅ Saved {len(compilation_result['warnings'])} warnings to: {warnings_log_path}")
            
            # Save log analysis if available
            if 'log_analysis' in compilation_result and compilation_result['log_analysis']:
                analysis = compilation_result['log_analysis']
                analysis_log_path = output_path / 'compilation_analysis.txt'
                with open(analysis_log_path, 'w', encoding='utf-8') as f:
                    f.write("="*80 + "\n")
                    f.write("COMPILATION LOG ANALYSIS\n")
                    f.write("="*80 + "\n\n")
                    
                    # Handle both CompilationAnalysis dataclass and dict formats
                    if hasattr(analysis, 'total_errors'):
                        # CompilationAnalysis dataclass
                        f.write(f"Total Errors: {analysis.total_errors}\n")
                        f.write(f"Total Warnings: {analysis.total_warnings}\n\n")
                        
                        if analysis.error_categories:
                            f.write("Error Patterns:\n")
                            for pattern, count in analysis.error_categories.items():
                                f.write(f"  - {pattern}: {count}\n")
                            f.write("\n")
                        
                        if analysis.improvement_suggestions:
                            f.write("Improvement Suggestions:\n")
                            for suggestion in analysis.improvement_suggestions:
                                f.write(f"  - {suggestion}\n")
                            f.write("\n")
                    else:
                        # Dictionary format (legacy)
                        f.write(f"Total Errors: {analysis.get('error_count', 0)}\n")
                        f.write(f"Total Warnings: {analysis.get('warning_count', 0)}\n\n")
                        
                        if 'error_patterns' in analysis:
                            f.write("Error Patterns:\n")
                            for pattern, count in analysis.get('error_patterns', {}).items():
                                f.write(f"  - {pattern}: {count}\n")
                            f.write("\n")
                        
                        if 'suggestions' in analysis:
                            f.write("Improvement Suggestions:\n")
                            for suggestion in analysis.get('suggestions', []):
                                f.write(f"  - {suggestion}\n")
                            f.write("\n")
                self.logger.info(f"✅ Saved compilation analysis: {analysis_log_path}")
            
            # Save execution logs if available (save even if failed for debugging)
            # Check both top level and inside compilation_result
            execution_result = None
            if 'execution_result' in generated_code:
                execution_result = generated_code['execution_result']
            elif 'execution_result' in compilation_result:
                execution_result = compilation_result['execution_result']
            
            if execution_result:
                execution_log_path = output_path / 'execution.log'
                with open(execution_log_path, 'w', encoding='utf-8') as f:
                    f.write("="*80 + "\n")
                    f.write("DOCKER CONTAINER EXECUTION LOG\n")
                    f.write("="*80 + "\n\n")
                    f.write(f"Execution Success: {execution_result.get('success', False)}\n")
                    f.write(f"Exit Code: {execution_result.get('exit_code', 'N/A')}\n")
                    f.write(f"Execution Time: {execution_result.get('execution_time', 0.0):.3f}s\n")
                    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    
                    # Output section
                    output_text = execution_result.get('output', '')
                    stdout_text = execution_result.get('stdout', '')
                    stderr_text = execution_result.get('stderr', '')
                    
                    if output_text or stdout_text or stderr_text:
                        f.write("="*80 + "\n")
                        f.write("PROGRAM OUTPUT\n")
                        f.write("="*80 + "\n\n")
                        if output_text:
                            f.write(output_text + "\n")
                        if stdout_text:
                            f.write("STDOUT:\n" + stdout_text + "\n")
                        if stderr_text:
                            f.write("STDERR:\n" + stderr_text + "\n")
                    else:
                        f.write("="*80 + "\n")
                        f.write("NO OUTPUT (Program may have exited without printing)\n")
                        f.write("="*80 + "\n\n")
                    
                    # Error information if failed
                    if not execution_result.get('success', False):
                        f.write("\n" + "="*80 + "\n")
                        f.write("EXECUTION ERRORS\n")
                        f.write("="*80 + "\n\n")
                        errors = execution_result.get('errors', [])
                        if errors:
                            for error in errors:
                                f.write(f"- {error}\n")
                        else:
                            f.write("Execution failed but no specific error captured.\n")
                            f.write("Binary may have been created but execution environment issue.\n")
                        f.write("\n")
                
                self.logger.info(f"✅ Saved execution log: {execution_log_path}")
        
        except Exception as e:
            self.logger.warning(f"Failed to save compilation logs: {e}")
            # Don't fail the whole process if log saving fails
    
    def get_agent_memory_summary(self) -> Dict[str, Any]:
        """Get summary of agent memory usage."""
        try:
            memory_summary = {}
            
            # Collect memory from all agents
            for agent_name, agent in self.agents.items():
                if hasattr(agent, 'get_memory_summary'):
                    memory_summary[agent_name] = agent.get_memory_summary()
                else:
                    # Fallback: create basic memory summary
                    memory_summary[agent_name] = {
                        'short_term_memory': getattr(agent, 'short_term_memory', {}),
                        'long_term_memory': getattr(agent, 'long_term_memory', {}),
                        'memory_enabled': getattr(agent, 'memory_enabled', False)
                    }
            
            return memory_summary
        except Exception as e:
            self.logger.error(f"Failed to get agent memory summary: {e}")
            return {}
