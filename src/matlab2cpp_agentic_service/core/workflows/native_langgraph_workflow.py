"""
Native LangGraph Workflow for MATLAB2C++ Conversion

This module creates a truly native LangGraph workflow that uses LangGraph-native agents
with full utilization of LangGraph features including tools, memory, and state management.
"""

from typing import Dict, Any, Literal
from pathlib import Path
from langgraph.graph import StateGraph, END
from loguru import logger

from matlab2cpp_agentic_service.infrastructure.state.conversion_state import (
    ConversionState, 
    ConversionStatus,
    update_state_status
)
from matlab2cpp_agentic_service.infrastructure.tools.llm_client import create_llm_client
from matlab2cpp_agentic_service.utils.config import get_config
from ..agents.analyzer.langgraph import LangGraphMATLABAnalyzerAgent
from ..agents.planner.langgraph import LangGraphConversionPlannerAgent
from ..agents.generator.langgraph import LangGraphCppGeneratorAgent
from ..agents.assessor.langgraph import LangGraphQualityAssessorAgent
from ..agents.base.langgraph_agent import AgentConfig


class NativeLangGraphMATLAB2CPPWorkflow:
    """
    Native LangGraph workflow for MATLAB to C++ conversion.
    
    This workflow uses truly LangGraph-native agents that fully utilize
    LangGraph features including:
    - Native agent memory management
    - LangGraph tools integration
    - Advanced state management
    - Conditional logic and optimization
    - Human-in-the-loop capabilities
    """
    
    def __init__(self):
        """Initialize the native LangGraph workflow."""
        self.config = get_config()
        self.logger = logger.bind(name="native_langgraph_workflow")
        
        # Create LLM client
        self.llm_client = create_llm_client(self.config.llm)
        
        # Initialize native LangGraph agents
        self.agents = self._initialize_native_agents()
        
        # Create the workflow graph
        self.workflow = self._create_native_workflow()
        
        self.logger.info("Native LangGraph MATLAB2C++ Workflow initialized")
    
    def _initialize_native_agents(self) -> Dict[str, Any]:
        """Initialize all native LangGraph agents."""
        agents = {}
        
        # MATLAB Analyzer Agent
        analyzer_config = AgentConfig(
            name="matlab_analyzer",
            description="Analyzes MATLAB code using LangGraph tools and memory",
            max_retries=3,
            timeout_seconds=300.0,
            enable_memory=True,
            enable_performance_tracking=True,
            tools=["matlab_parser", "llm_analysis"]
        )
        agents['analyzer'] = LangGraphMATLABAnalyzerAgent(analyzer_config, self.llm_client)
        
        # Conversion Planner Agent
        planner_config = AgentConfig(
            name="conversion_planner",
            description="Creates conversion plans using LangGraph tools and memory",
            max_retries=3,
            timeout_seconds=300.0,
            enable_memory=True,
            enable_performance_tracking=True,
            tools=["llm_analysis"]
        )
        agents['planner'] = LangGraphConversionPlannerAgent(planner_config, self.llm_client)
        
        # C++ Generator Agent
        generator_config = AgentConfig(
            name="cpp_generator",
            description="Generates C++ code using LangGraph tools and memory",
            max_retries=3,
            timeout_seconds=600.0,
            enable_memory=True,
            enable_performance_tracking=True,
            tools=["code_generation", "llm_analysis"]
        )
        agents['generator'] = LangGraphCppGeneratorAgent(generator_config, self.llm_client)
        
        # Quality Assessor Agent
        assessor_config = AgentConfig(
            name="quality_assessor",
            description="Assesses code quality using LangGraph tools and memory",
            max_retries=3,
            timeout_seconds=300.0,
            enable_memory=True,
            enable_performance_tracking=True,
            tools=["quality_assessment", "llm_analysis"]
        )
        agents['assessor'] = LangGraphQualityAssessorAgent(assessor_config, self.llm_client)
        
        self.logger.info(f"Initialized {len(agents)} native LangGraph agents")
        return agents
    
    def _create_native_workflow(self) -> StateGraph:
        """
        Create the native LangGraph workflow with native agents.
        
        Returns:
            Compiled StateGraph workflow using native agents
        """
        # Create the state graph
        workflow = StateGraph(ConversionState)
        
        # Add nodes using native agent node functions
        workflow.add_node("analyze", self.agents['analyzer'].create_node())
        workflow.add_node("plan", self.agents['planner'].create_node())
        workflow.add_node("generate", self.agents['generator'].create_node())
        workflow.add_node("assess", self.agents['assessor'].create_node())
        workflow.add_node("save", self._save_generated_code_node)
        
        # Add optimization control nodes
        workflow.add_node("check_optimization", self._check_optimization_node)
        workflow.add_node("increment_turn", self._increment_turn_node)
        workflow.add_node("optimize", self._optimize_code_node)
        
        # Define the workflow edges
        # Initial workflow: analyze -> plan -> generate -> assess -> save
        workflow.add_edge("analyze", "plan")
        workflow.add_edge("plan", "generate")
        workflow.add_edge("generate", "assess")
        workflow.add_edge("assess", "save")
        workflow.add_edge("save", "check_optimization")
        
        # Optimization decision point
        workflow.add_conditional_edges(
            "check_optimization",
            self._should_continue_optimization,
            {
                "continue": "increment_turn",
                "complete": END
            }
        )
        
        # Optimization loop: increment turn -> optimize -> assess -> save -> check
        workflow.add_edge("increment_turn", "optimize")
        workflow.add_edge("optimize", "assess")
        
        # Set entry point
        workflow.set_entry_point("analyze")
        
        # Compile the workflow
        return workflow.compile()
    
    def _save_generated_code_node(self, state: ConversionState) -> ConversionState:
        """
        Save generated code to organized output structure.
        
        Args:
            state: Current conversion state
            
        Returns:
            Updated state with saved files information
        """
        from pathlib import Path
        
        try:
            self.logger.info("Saving generated code...")
            
            # Create organized output directory structure
            project_output_dir = Path(state["request"].output_dir) / state["request"].project_name
            project_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (project_output_dir / "generated_code").mkdir(exist_ok=True)
            (project_output_dir / "reports").mkdir(exist_ok=True)
            (project_output_dir / "debug").mkdir(exist_ok=True)
            
            state["project_output_dir"] = project_output_dir
            
            # Save generated code
            generated_code = state["generated_code"]
            version = f"v{state['current_turn'] + 1}"
            
            if isinstance(generated_code, dict) and 'files' in generated_code:
                # Multi-file project
                saved_files = self._save_multi_file_code(
                    generated_code, project_output_dir, state["request"].project_name, version
                )
            else:
                # Single-file project
                saved_files = self._save_single_file_code(
                    generated_code, project_output_dir, state["request"].project_name, version
                )
            
            state["generated_files"] = saved_files
            self.logger.info(f"Saved {len(saved_files)} files to {project_output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving generated code: {e}")
            state["error_message"] = f"File saving failed: {str(e)}"
            state = update_state_status(state, ConversionStatus.FAILED)
        
        return state
    
    def _optimize_code_node(self, state: ConversionState) -> ConversionState:
        """
        Optimize generated code using native LangGraph generator agent.
        
        Args:
            state: Current conversion state
            
        Returns:
            Updated state with optimized code
        """
        try:
            self.logger.info(f"Optimizing code for turn {state['current_turn']}")
            
            # Use the native generator agent for optimization
            # The agent will use its memory to improve the code
            generator_node = self.agents['generator'].create_node()
            state = generator_node(state)
            
            self.logger.info("Code optimization completed")
            
        except Exception as e:
            self.logger.error(f"Error in code optimization: {e}")
            state["error_message"] = f"Code optimization failed: {str(e)}"
            state = update_state_status(state, ConversionStatus.FAILED)
        
        return state
    
    def _check_optimization_node(self, state: ConversionState) -> ConversionState:
        """
        Check if optimization should continue using native agent memory.
        
        Args:
            state: Current conversion state
            
        Returns:
            Updated state with optimization decision
        """
        self.logger.info("Checking optimization conditions...")
        
        # Get quality scores from state
        quality_scores = state.get("quality_scores", {})
        current_score = quality_scores.get('overall', 0.0)
        
        # Get optimization parameters
        target_score = state["request"].target_quality_score
        max_turns = state["request"].max_optimization_turns
        current_turn = state["current_turn"]
        
        # Check agent memory for optimization history
        assessor_agent = self.agents['assessor']
        assessment_trends = assessor_agent.get_memory("assessment_trends", "long_term") or {}
        
        # Update state flags
        state["target_quality_met"] = current_score >= target_score
        state["max_turns_reached"] = current_turn >= max_turns
        
        # Use agent memory to make intelligent optimization decisions
        if max_turns == 0:
            # No optimization allowed
            state["optimization_complete"] = True
            self.logger.info("No optimization turns requested (max_turns=0)")
        elif assessment_trends.get('improvement_trend') == 'declining' and current_turn > 1:
            # Stop optimization if quality is declining
            state["optimization_complete"] = True
            self.logger.info("Stopping optimization due to declining quality trend")
        elif state["target_quality_met"]:
            state["optimization_complete"] = True
            self.logger.info(f"Target quality achieved: {current_score:.1f} >= {target_score}")
        elif state["max_turns_reached"]:
            state["optimization_complete"] = True
            self.logger.info(f"Max optimization turns reached: {current_turn}/{max_turns}")
        else:
            state["optimization_complete"] = False
            self.logger.info(f"Continuing optimization: {current_score:.1f}/{target_score} (turn {current_turn}/{max_turns})")
        
        return state
    
    def _increment_turn_node(self, state: ConversionState) -> ConversionState:
        """
        Increment the optimization turn counter and update agent memory.
        
        Args:
            state: Current conversion state
            
        Returns:
            Updated state with incremented turn
        """
        state["current_turn"] += 1
        self.logger.info(f"Starting optimization turn {state['current_turn']}")
        
        # Update status for optimization
        state["status"] = ConversionStatus.OPTIMIZING
        
        # Update agent memory with turn information
        for agent_name, agent in self.agents.items():
            agent.update_memory("current_optimization_turn", state["current_turn"], "context")
        
        return state
    
    def _should_continue_optimization(self, state: ConversionState) -> Literal["continue", "complete"]:
        """
        Determine if optimization should continue using agent memory and trends.
        
        Args:
            state: Current conversion state
            
        Returns:
            "continue" if optimization should continue, "complete" if finished
        """
        if state["optimization_complete"]:
            return "complete"
        else:
            return "continue"
    
    def _save_multi_file_code(self, code_result: Dict[str, Any], output_dir: Path, 
                            project_name: str, version: str) -> list:
        """Save multi-file C++ code to organized project directory."""
        saved_files = []
        
        # Save all generated files
        for filename, content in code_result['files'].items():
            file_path = output_dir / "generated_code" / f"{version}_{filename}"
            file_path.write_text(content, encoding='utf-8')
            saved_files.append(str(file_path))
        
        # Save compilation instructions if available
        if 'compilation_instructions' in code_result:
            instructions_file = output_dir / "generated_code" / f"{version}_compilation_instructions.md"
            instructions_file.write_text(code_result['compilation_instructions'], encoding='utf-8')
            saved_files.append(str(instructions_file))
        
        return saved_files
    
    def _save_single_file_code(self, code: Dict[str, str], output_dir: Path, 
                             project_name: str, version: str) -> list:
        """Save single-file C++ code to organized project directory."""
        saved_files = []
        
        if code.get('header'):
            header_file = output_dir / "generated_code" / f"{version}.h"
            header_file.write_text(code['header'])
            saved_files.append(str(header_file))
        
        if code.get('implementation'):
            impl_file = output_dir / "generated_code" / f"{version}.cpp"
            impl_file.write_text(code['implementation'])
            saved_files.append(str(impl_file))
        
        return saved_files
    
    async def run_conversion(self, initial_state: ConversionState) -> ConversionState:
        """
        Run the complete native LangGraph conversion workflow.
        
        Args:
            initial_state: Initial state for the workflow
            
        Returns:
            Final state after workflow completion
        """
        try:
            self.logger.info("Starting native LangGraph conversion workflow...")
            
            # Run the workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            # Update final status
            if final_state["error_message"]:
                final_state["status"] = ConversionStatus.FAILED
            else:
                final_state["status"] = ConversionStatus.COMPLETED
            
            self.logger.info("Native LangGraph conversion workflow completed successfully")
            return final_state
            
        except Exception as e:
            self.logger.error(f"Error in native LangGraph workflow: {e}")
            initial_state["error_message"] = f"Workflow failed: {str(e)}"
            initial_state["status"] = ConversionStatus.FAILED
            return initial_state
    
    def run_conversion_sync(self, initial_state: ConversionState) -> ConversionState:
        """
        Run the complete native LangGraph conversion workflow synchronously.
        
        Args:
            initial_state: Initial state for the workflow
            
        Returns:
            Final state after workflow completion
        """
        try:
            self.logger.info("Starting native LangGraph conversion workflow...")
            
            # Run the workflow synchronously
            final_state = self.workflow.invoke(initial_state)
            
            # Update final status
            if final_state["error_message"]:
                final_state["status"] = ConversionStatus.FAILED
            else:
                final_state["status"] = ConversionStatus.COMPLETED
            
            self.logger.info("Native LangGraph conversion workflow completed successfully")
            return final_state
            
        except Exception as e:
            self.logger.error(f"Error in native LangGraph workflow: {e}")
            initial_state["error_message"] = f"Workflow failed: {str(e)}"
            initial_state["status"] = ConversionStatus.FAILED
            return initial_state
    
    def get_workflow_graph(self) -> str:
        """
        Get a visual representation of the native workflow graph.
        
        Returns:
            Mermaid diagram representation of the workflow
        """
        try:
            return self.workflow.get_graph().draw_mermaid()
        except Exception as e:
            self.logger.warning(f"Could not generate workflow graph: {e}")
            return "Native workflow graph not available"
    
    def get_workflow_stats(self, final_state: ConversionState) -> Dict[str, Any]:
        """
        Get workflow execution statistics including agent performance.
        
        Args:
            final_state: Final state from workflow execution
            
        Returns:
            Dictionary with workflow and agent statistics
        """
        stats = {
            "total_processing_time": final_state["total_processing_time"],
            "optimization_turns": final_state["current_turn"],
            "quality_scores": final_state["quality_scores"],
            "generated_files": len(final_state["generated_files"]),
            "is_multi_file": final_state["is_multi_file"],
            "processing_times": final_state["processing_times"],
            "status": final_state["status"].value,
            "error_message": final_state["error_message"],
            "agent_performance": {}
        }
        
        # Get performance statistics from each native agent
        for agent_name, agent in self.agents.items():
            stats["agent_performance"][agent_name] = agent.get_performance_summary()
        
        return stats
    
    def get_agent_memory_summary(self) -> Dict[str, Any]:
        """
        Get summary of all agent memory states.
        
        Returns:
            Dictionary with agent memory summaries
        """
        memory_summary = {}
        
        for agent_name, agent in self.agents.items():
            memory_summary[agent_name] = {
                "short_term_memory_size": len(agent.memory.short_term),
                "long_term_memory_size": len(agent.memory.long_term),
                "context_memory_size": len(agent.memory.context),
                "performance_history_size": len(agent.memory.performance_history),
                "last_operation": agent.get_memory("last_operation", "context"),
                "operation_count": agent.get_memory("operation_count", "short_term") or 0
            }
        
        return memory_summary
    
    def clear_agent_memory(self, memory_type: str = "all"):
        """
        Clear memory for all agents.
        
        Args:
            memory_type: Type of memory to clear ("short_term", "long_term", "context", "all")
        """
        for agent_name, agent in self.agents.items():
            agent.clear_memory(memory_type)
        
        self.logger.info(f"Cleared {memory_type} memory for all agents")
