"""Main LangGraph workflow for MATLAB to C++ conversion."""

from typing import Dict, Any, List, TypedDict
from pathlib import Path
from langgraph.graph import StateGraph, END
from loguru import logger

from ..agents.matlab_analyzer import MATLABAnalyzerAgent, ProjectUnderstanding
from ..agents.code_mapper import CodeMapperAgent
from ..agents.cpp_generator import CppGeneratorAgent
from ..agents.validator import ValidatorAgent
from ..agents.project_manager import ProjectManagerAgent
from ..utils.config import get_config


class ConversionState(TypedDict):
    """State for the conversion workflow."""
    # Input
    matlab_project_path: Path
    output_path: Path
    
    # Analysis results
    project_understanding: ProjectUnderstanding
    file_understandings: List[Dict[str, Any]]
    function_understandings: List[Dict[str, Any]]
    
    # Conversion results
    cpp_project_structure: Dict[str, Any]
    generated_files: List[Dict[str, Any]]
    validation_results: List[Dict[str, Any]]
    
    # Workflow state
    current_step: str
    errors: List[str]
    warnings: List[str]
    progress: float


class ConversionWorkflow:
    """Main workflow for converting MATLAB projects to C++."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logger.bind(name="conversion_workflow")
        
        # Initialize agents
        self.matlab_analyzer = MATLABAnalyzerAgent()
        self.code_mapper = CodeMapperAgent()
        self.cpp_generator = CppGeneratorAgent()
        self.validator = ValidatorAgent()
        self.project_manager = ProjectManagerAgent()
        
        # Build workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(ConversionState)
        
        # Add nodes
        workflow.add_node("analyze_project", self._analyze_project_node)
        workflow.add_node("map_code", self._map_code_node)
        workflow.add_node("generate_cpp", self._generate_cpp_node)
        workflow.add_node("validate_conversion", self._validate_conversion_node)
        workflow.add_node("create_project", self._create_project_node)
        workflow.add_node("handle_errors", self._handle_errors_node)
        
        # Add edges
        workflow.set_entry_point("analyze_project")
        
        workflow.add_edge("analyze_project", "map_code")
        workflow.add_edge("map_code", "generate_cpp")
        workflow.add_edge("generate_cpp", "validate_conversion")
        workflow.add_edge("validate_conversion", "create_project")
        workflow.add_edge("create_project", END)
        
        # Add conditional edges for error handling
        workflow.add_conditional_edges(
            "analyze_project",
            self._should_continue,
            {
                "continue": "map_code",
                "error": "handle_errors"
            }
        )
        
        workflow.add_conditional_edges(
            "map_code",
            self._should_continue,
            {
                "continue": "generate_cpp",
                "error": "handle_errors"
            }
        )
        
        workflow.add_conditional_edges(
            "generate_cpp",
            self._should_continue,
            {
                "continue": "validate_conversion",
                "error": "handle_errors"
            }
        )
        
        workflow.add_conditional_edges(
            "validate_conversion",
            self._should_continue,
            {
                "continue": "create_project",
                "error": "handle_errors"
            }
        )
        
        workflow.add_edge("handle_errors", END)
        
        return workflow.compile()
    
    def convert_project(self, matlab_path: Path, output_path: Path) -> Dict[str, Any]:
        """Convert a MATLAB project to C++."""
        self.logger.info(f"Starting conversion: {matlab_path} -> {output_path}")
        
        # Initialize state
        initial_state = ConversionState(
            matlab_project_path=matlab_path,
            output_path=output_path,
            project_understanding=None,
            file_understandings=[],
            function_understandings=[],
            cpp_project_structure={},
            generated_files=[],
            validation_results=[],
            current_step="initialization",
            errors=[],
            warnings=[],
            progress=0.0
        )
        
        # Run workflow
        try:
            final_state = self.workflow.invoke(initial_state)
            self.logger.info("Conversion completed successfully")
            return final_state
        except Exception as e:
            self.logger.error(f"Conversion failed: {e}")
            raise
    
    def _analyze_project_node(self, state: ConversionState) -> ConversionState:
        """Analyze the MATLAB project."""
        self.logger.info("Analyzing MATLAB project...")
        state["current_step"] = "analysis"
        state["progress"] = 0.1
        
        try:
            # Analyze project
            project_understanding = self.matlab_analyzer.analyze_project(
                state["matlab_project_path"]
            )
            state["project_understanding"] = project_understanding
            
            state["progress"] = 0.2
            self.logger.info("Project analysis completed")
            
        except Exception as e:
            self.logger.error(f"Error in project analysis: {e}")
            state["errors"].append(f"Project analysis failed: {e}")
        
        return state
    
    def _map_code_node(self, state: ConversionState) -> ConversionState:
        """Map MATLAB code to C++ equivalents."""
        self.logger.info("Mapping MATLAB code to C++...")
        state["current_step"] = "mapping"
        state["progress"] = 0.3
        
        try:
            # Map code constructs
            mapping_results = self.code_mapper.map_project(
                state["project_understanding"]
            )
            state["file_understandings"] = mapping_results.get("file_mappings", [])
            state["function_understandings"] = mapping_results.get("function_mappings", [])
            
            state["progress"] = 0.5
            self.logger.info("Code mapping completed")
            
        except Exception as e:
            self.logger.error(f"Error in code mapping: {e}")
            state["errors"].append(f"Code mapping failed: {e}")
        
        return state
    
    def _generate_cpp_node(self, state: ConversionState) -> ConversionState:
        """Generate C++ code."""
        self.logger.info("Generating C++ code...")
        state["current_step"] = "generation"
        state["progress"] = 0.6
        
        try:
            # Generate C++ project
            cpp_results = self.cpp_generator.generate_project(
                state["file_understandings"],
                state["function_understandings"],
                state["output_path"]
            )
            state["cpp_project_structure"] = cpp_results.get("project_structure", {})
            state["generated_files"] = cpp_results.get("generated_files", [])
            
            state["progress"] = 0.8
            self.logger.info("C++ code generation completed")
            
        except Exception as e:
            self.logger.error(f"Error in C++ generation: {e}")
            state["errors"].append(f"C++ generation failed: {e}")
        
        return state
    
    def _validate_conversion_node(self, state: ConversionState) -> ConversionState:
        """Validate the conversion."""
        self.logger.info("Validating conversion...")
        state["current_step"] = "validation"
        state["progress"] = 0.9
        
        try:
            # Validate conversion
            validation_results = self.validator.validate_conversion(
                state["generated_files"],
                state["matlab_project_path"]
            )
            state["validation_results"] = validation_results
            
            state["progress"] = 0.95
            self.logger.info("Validation completed")
            
        except Exception as e:
            self.logger.error(f"Error in validation: {e}")
            state["errors"].append(f"Validation failed: {e}")
        
        return state
    
    def _create_project_node(self, state: ConversionState) -> ConversionState:
        """Create the final C++ project."""
        self.logger.info("Creating final C++ project...")
        state["current_step"] = "project_creation"
        
        try:
            # Create project structure
            self.project_manager.create_project(
                state["cpp_project_structure"],
                state["generated_files"],
                state["output_path"]
            )
            
            state["progress"] = 1.0
            self.logger.info("Project creation completed")
            
        except Exception as e:
            self.logger.error(f"Error in project creation: {e}")
            state["errors"].append(f"Project creation failed: {e}")
        
        return state
    
    def _handle_errors_node(self, state: ConversionState) -> ConversionState:
        """Handle errors in the workflow."""
        self.logger.error("Handling workflow errors...")
        state["current_step"] = "error_handling"
        
        # Log all errors
        for error in state["errors"]:
            self.logger.error(f"Workflow error: {error}")
        
        return state
    
    def _should_continue(self, state: ConversionState) -> str:
        """Determine if workflow should continue or handle errors."""
        if state["errors"]:
            return "error"
        return "continue"


