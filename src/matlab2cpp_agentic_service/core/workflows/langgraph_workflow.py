"""
LangGraph Workflow for MATLAB2C++ Conversion

This module creates a LangGraph-based workflow that orchestrates the conversion process
with proper state management, conditional logic, and iterative optimization.
"""

from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END
from loguru import logger

from ...infrastructure.state.conversion_state import (
    ConversionState, 
    ConversionStatus
)
from .langgraph_nodes import LangGraphAgentNodes


class MATLAB2CPPLangGraphWorkflow:
    """
    LangGraph-based workflow for MATLAB to C++ conversion.
    
    This workflow provides:
    - Proper state management
    - Conditional logic for optimization
    - Iterative processing with feedback loops
    - Error handling and recovery
    """
    
    def __init__(self):
        """Initialize the LangGraph workflow."""
        self.agent_nodes = LangGraphAgentNodes()
        self.logger = logger.bind(name="langgraph_workflow")
        
        # Create the workflow graph
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """
        Create the LangGraph workflow with nodes and edges.
        
        Returns:
            Compiled StateGraph workflow
        """
        # Create the state graph
        workflow = StateGraph(ConversionState)
        
        # Add nodes (each agent becomes a node)
        workflow.add_node("analyze", self.agent_nodes.analyze_matlab_content)
        workflow.add_node("plan", self.agent_nodes.create_conversion_plan)
        workflow.add_node("generate", self.agent_nodes.generate_cpp_code)
        workflow.add_node("assess", self.agent_nodes.assess_code_quality)
        workflow.add_node("save", self.agent_nodes.save_generated_code)
        
        # Add optimization control nodes
        workflow.add_node("check_optimization", self._check_optimization_node)
        workflow.add_node("increment_turn", self._increment_turn_node)
        
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
        
        # Optimization loop: increment turn -> generate -> assess -> save -> check
        workflow.add_edge("increment_turn", "generate")
        
        # Set entry point
        workflow.set_entry_point("analyze")
        
        # Compile the workflow
        return workflow.compile()
    
    def _check_optimization_node(self, state: ConversionState) -> ConversionState:
        """
        Check if optimization should continue.
        
        Args:
            state: Current conversion state
            
        Returns:
            Updated state with optimization decision
        """
        self.logger.info("Checking optimization conditions...")
        
        # Update optimization conditions
        current_score = state["quality_scores"].get("algorithmic", 0.0)
        target_score = state["request"].target_quality_score
        max_turns = state["request"].max_optimization_turns
        current_turn = state["current_turn"]
        
        # Update state flags
        state["target_quality_met"] = current_score >= target_score
        state["max_turns_reached"] = current_turn >= max_turns
        state["optimization_complete"] = state["target_quality_met"] or state["max_turns_reached"]
        
        # Log optimization status
        if state["target_quality_met"]:
            self.logger.info(f"Target quality achieved: {current_score:.1f} >= {target_score}")
        elif state["max_turns_reached"]:
            self.logger.info(f"Max optimization turns reached: {current_turn}/{max_turns}")
        else:
            self.logger.info(f"Continuing optimization: {current_score:.1f}/{target_score} (turn {current_turn}/{max_turns})")
        
        return state
    
    def _increment_turn_node(self, state: ConversionState) -> ConversionState:
        """
        Increment the optimization turn counter.
        
        Args:
            state: Current conversion state
            
        Returns:
            Updated state with incremented turn
        """
        state["current_turn"] += 1
        self.logger.info(f"Starting optimization turn {state['current_turn']}")
        
        # Update status for optimization
        state["status"] = ConversionStatus.OPTIMIZING
        
        return state
    
    def _should_continue_optimization(self, state: ConversionState) -> Literal["continue", "complete"]:
        """
        Determine if optimization should continue.
        
        Args:
            state: Current conversion state
            
        Returns:
            "continue" if optimization should continue, "complete" if finished
        """
        if state["optimization_complete"]:
            return "complete"
        else:
            return "continue"
    
    async def run_conversion(self, initial_state: ConversionState) -> ConversionState:
        """
        Run the complete conversion workflow.
        
        Args:
            initial_state: Initial state for the workflow
            
        Returns:
            Final state after workflow completion
        """
        try:
            self.logger.info("Starting LangGraph conversion workflow...")
            
            # Run the workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            # Update final status
            if final_state["error_message"]:
                final_state["status"] = ConversionStatus.FAILED
            else:
                final_state["status"] = ConversionStatus.COMPLETED
            
            self.logger.info("LangGraph conversion workflow completed successfully")
            return final_state
            
        except Exception as e:
            self.logger.error(f"Error in LangGraph workflow: {e}")
            initial_state["error_message"] = f"Workflow failed: {str(e)}"
            initial_state["status"] = ConversionStatus.FAILED
            return initial_state
    
    def run_conversion_sync(self, initial_state: ConversionState) -> ConversionState:
        """
        Run the complete conversion workflow synchronously.
        
        Args:
            initial_state: Initial state for the workflow
            
        Returns:
            Final state after workflow completion
        """
        try:
            self.logger.info("Starting LangGraph conversion workflow...")
            
            # Run the workflow synchronously
            final_state = self.workflow.invoke(initial_state)
            
            # Update final status
            if final_state["error_message"]:
                final_state["status"] = ConversionStatus.FAILED
            else:
                final_state["status"] = ConversionStatus.COMPLETED
            
            self.logger.info("LangGraph conversion workflow completed successfully")
            return final_state
            
        except Exception as e:
            self.logger.error(f"Error in LangGraph workflow: {e}")
            initial_state["error_message"] = f"Workflow failed: {str(e)}"
            initial_state["status"] = ConversionStatus.FAILED
            return initial_state
    
    def get_workflow_graph(self) -> str:
        """
        Get a visual representation of the workflow graph.
        
        Returns:
            Mermaid diagram representation of the workflow
        """
        try:
            return self.workflow.get_graph().draw_mermaid()
        except Exception as e:
            self.logger.warning(f"Could not generate workflow graph: {e}")
            return "Workflow graph not available"
    
    def get_workflow_stats(self, final_state: ConversionState) -> Dict[str, Any]:
        """
        Get workflow execution statistics.
        
        Args:
            final_state: Final state from workflow execution
            
        Returns:
            Dictionary with workflow statistics
        """
        return {
            "total_processing_time": final_state["total_processing_time"],
            "optimization_turns": final_state["current_turn"],
            "quality_scores": final_state["quality_scores"],
            "generated_files": len(final_state["generated_files"]),
            "is_multi_file": final_state["is_multi_file"],
            "processing_times": final_state["processing_times"],
            "status": final_state["status"].value,
            "error_message": final_state["error_message"]
        }
