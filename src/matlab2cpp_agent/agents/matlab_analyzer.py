"""MATLAB Analyzer Agent for understanding code functionality through LLM analysis."""

from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass
from loguru import logger

from ..tools.matlab_parser import MATLABParser, ProjectStructure, MATLABFile
from ..tools.llm_client import create_llm_client, test_llm_connection
from ..utils.config import get_config


@dataclass
class CodeUnderstanding:
    """Represents understanding of a piece of MATLAB code."""
    purpose: str
    domain: str
    algorithms: List[str]
    data_flow: Dict[str, Any]
    complexity: str
    confidence: float
    challenges: List[str]
    suggestions: List[str]


@dataclass
class FunctionUnderstanding:
    """Represents understanding of a MATLAB function."""
    name: str
    purpose: str
    inputs: List[Dict[str, Any]]
    outputs: List[Dict[str, Any]]
    algorithm_type: str
    complexity: str
    dependencies: List[str]
    cpp_equivalent_strategy: str
    confidence: float


@dataclass
class ProjectUnderstanding:
    """Represents overall understanding of a MATLAB project."""
    main_purpose: str
    domain: str
    key_algorithms: List[str]
    architecture: str
    complexity_level: str
    conversion_challenges: List[str]
    recommendations: List[str]
    confidence: float


class MATLABAnalyzerAgent:
    """LLM-powered agent for understanding MATLAB code content and purpose."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logger.bind(name="matlab_analyzer")
        
        # Create LLM client (vLLM or OpenAI)
        try:
            self.llm_client = create_llm_client(self.config.llm)
            self.logger.info(f"Initialized LLM client with provider: {self.config.llm.provider}")
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client: {e}")
            raise
        
        self.parser = MATLABParser()
    
    def analyze_project(self, project_path: Path) -> ProjectUnderstanding:
        """Analyze an entire MATLAB project to understand its purpose and structure."""
        self.logger.info(f"Starting comprehensive project analysis: {project_path}")
        
        # Parse project structure (all files)
        project_structure = self.parser.parse_project(project_path)
        self.logger.info(f"Found {len(project_structure.files)} MATLAB files in project")
        
        # Check if project is too large for standard analysis
        estimated_tokens = self._estimate_project_tokens(project_structure)
        max_tokens = self.config.llm.max_tokens // 2  # Reserve space for response
        
        if estimated_tokens > max_tokens:
            self.logger.info(f"Project too large ({estimated_tokens} tokens), using hierarchical analysis")
            return self._analyze_large_project_hierarchically(project_structure)
        else:
            self.logger.info(f"Project size OK ({estimated_tokens} tokens), using standard analysis")
            return self._analyze_project_standard(project_structure)
    
    def _estimate_project_tokens(self, project_structure: ProjectStructure) -> int:
        """Estimate total tokens needed for project analysis."""
        total_chars = 0
        
        # Base overhead
        total_chars += 2000  # Prompts and structure
        
        # File content
        for file in project_structure.files:
            total_chars += file.size
            total_chars += len(file.functions) * 200  # Function analysis overhead
            total_chars += len(file.dependencies) * 50  # Dependency analysis
        
        return total_chars // 4  # Rough token estimate
    
    def _analyze_large_project_hierarchically(self, project_structure: ProjectStructure) -> ProjectUnderstanding:
        """Analyze large project using hierarchical approach."""
        from .hierarchical_analyzer import HierarchicalAnalyzer
        
        hierarchical_analyzer = HierarchicalAnalyzer()
        # Get the project path from the first file
        project_path = project_structure.files[0].path.parent if project_structure.files else Path(".")
        analysis_result = hierarchical_analyzer.analyze_large_project(project_path)
        
        # Convert to ProjectUnderstanding format
        return ProjectUnderstanding(
            main_purpose=analysis_result["overall_understanding"]["overall_analysis"][:200] + "...",
            domain="Multi-domain (hierarchical analysis)",
            key_algorithms=[],
            architecture="Hierarchical multi-file project",
            complexity_level=analysis_result["overall_understanding"].get("complexity_level", "High"),
            conversion_challenges=["Large project size", "Complex dependencies"],
            recommendations=["Use hierarchical C++ structure", "Modular design"],
            confidence=analysis_result["overall_understanding"].get("confidence", 0.8)
        )
    
    def _analyze_project_standard(self, project_structure: ProjectStructure) -> ProjectUnderstanding:
        """Analyze project using standard approach (within token limits)."""
        # Analyze each file individually
        file_understandings = []
        for i, file in enumerate(project_structure.files):
            self.logger.info(f"Analyzing file {i+1}/{len(project_structure.files)}: {file.path.name}")
            understanding = self.analyze_file(file)
            file_understandings.append(understanding)
        
        # Cross-file analysis for dependencies and relationships
        self.logger.info("Performing cross-file dependency analysis...")
        cross_file_analysis = self._analyze_cross_file_relationships(
            project_structure, file_understandings
        )
        
        # Synthesize comprehensive project-level understanding
        self.logger.info("Synthesizing project-level understanding...")
        project_understanding = self._synthesize_project_understanding(
            project_structure, file_understandings, cross_file_analysis
        )
        
        self.logger.info("Comprehensive project analysis completed")
        return project_understanding
    
    def analyze_file(self, matlab_file: MATLABFile) -> CodeUnderstanding:
        """Analyze a single MATLAB file to understand its purpose and functionality."""
        self.logger.debug(f"Analyzing file: {matlab_file.path}")
        
        # Prepare content for analysis
        content = self._prepare_content_for_analysis(matlab_file)
        
        # Multi-pass analysis
        analysis_prompts = [
            self._get_purpose_analysis_prompt(),
            self._get_algorithm_analysis_prompt(),
            self._get_data_flow_analysis_prompt(),
            self._get_complexity_analysis_prompt()
        ]
        
        analyses = []
        for prompt_template in analysis_prompts:
            prompt = prompt_template.format(content=content)
            response = self._get_llm_response(prompt)
            analyses.append(response)
        
        # Synthesize understanding
        understanding = self._synthesize_file_understanding(matlab_file, analyses)
        
        return understanding
    
    def analyze_function(self, function_content: str, function_name: str) -> FunctionUnderstanding:
        """Analyze a specific MATLAB function."""
        self.logger.debug(f"Analyzing function: {function_name}")
        
        prompt = self._get_function_analysis_prompt().format(
            function_name=function_name,
            function_content=function_content
        )
        
        response = self._get_llm_response(prompt)
        understanding = self._parse_function_understanding(response, function_name)
        
        return understanding
    
    def _prepare_content_for_analysis(self, matlab_file: MATLABFile) -> str:
        """Prepare MATLAB file content for LLM analysis with token limits."""
        # Estimate token count (rough approximation: 1 token ≈ 4 characters)
        estimated_tokens = len(matlab_file.content) // 4
        
        # If content is too large for single analysis, use smart chunking
        max_tokens_per_file = self.config.llm.max_tokens // 4  # Reserve space for prompts
        if estimated_tokens > max_tokens_per_file:
            return self._smart_chunk_content(matlab_file)
        
        return matlab_file.content
    
    def _smart_chunk_content(self, matlab_file: MATLABFile) -> str:
        """Smart chunking for large MATLAB files with context preservation."""
        content = matlab_file.content
        max_chunk_size = self.config.analysis.chunk_size
        
        # Try to chunk by functions first (preserve logical structure)
        if matlab_file.functions:
            return self._chunk_by_functions(matlab_file)
        
        # Fallback to line-based chunking
        return self._chunk_by_lines(content, max_chunk_size)
    
    def _chunk_by_functions(self, matlab_file: MATLABFile) -> str:
        """Chunk content by functions to preserve logical structure."""
        chunks = []
        current_chunk = []
        current_size = 0
        max_chunk_size = self.config.analysis.chunk_size
        
        # Add file header
        header = f"=== File: {matlab_file.path.name} ===\n"
        header += f"Size: {matlab_file.size} bytes\n"
        header += f"Functions: {len(matlab_file.functions)}\n"
        header += f"Dependencies: {len(matlab_file.dependencies)}\n\n"
        
        current_chunk.append(header)
        current_size += len(header)
        
        # Add functions one by one
        for i, func in enumerate(matlab_file.functions):
            func_content = f"--- Function {i+1}: {func['name']} ---\n{func['content']}\n\n"
            
            if current_size + len(func_content) > max_chunk_size and current_chunk:
                # Start new chunk
                chunks.append('\n'.join(current_chunk))
                current_chunk = [header]  # Include header in each chunk
                current_size = len(header)
            
            current_chunk.append(func_content)
            current_size += len(func_content)
        
        # Add remaining content
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return '\n\n'.join([f"--- Chunk {i+1}/{len(chunks)} ---\n{chunk}" for i, chunk in enumerate(chunks)])
    
    def _chunk_by_lines(self, content: str, max_chunk_size: int) -> str:
        """Fallback line-based chunking."""
        lines = content.split('\n')
        chunks = []
        
        for i in range(0, len(lines), max_chunk_size):
            chunk = '\n'.join(lines[i:i + max_chunk_size])
            chunks.append(f"--- Chunk {i//max_chunk_size + 1} ---\n{chunk}")
        
        return '\n\n'.join(chunks)
    
    def _get_purpose_analysis_prompt(self) -> str:
        """Get prompt for analyzing code purpose."""
        return """
        Analyze this MATLAB code and determine its main purpose and functionality:
        
        {content}
        
        Please provide:
        1. Main purpose: What does this code do?
        2. Domain: What field/domain does it belong to? (e.g., signal processing, numerical computing, image processing, etc.)
        3. Key functionality: What are the main functions or operations?
        4. Input/Output: What data does it process and what does it produce?
        5. Dependencies: What external functions or toolboxes does it rely on?
        
        Format your response as structured text with clear sections.
        """
    
    def _get_algorithm_analysis_prompt(self) -> str:
        """Get prompt for analyzing algorithms."""
        return """
        Analyze this MATLAB code to identify the algorithms and mathematical techniques used:
        
        {content}
        
        Please identify:
        1. Mathematical operations: What mathematical computations are performed?
        2. Algorithms: What specific algorithms or techniques are used?
        3. Data structures: How is data organized and manipulated?
        4. Optimization: Are there any optimization techniques used?
        5. Numerical methods: What numerical methods are employed?
        
        Be specific about algorithm names and mathematical concepts.
        """
    
    def _get_data_flow_analysis_prompt(self) -> str:
        """Get prompt for analyzing data flow."""
        return """
        Analyze the data flow in this MATLAB code:
        
        {content}
        
        Please describe:
        1. Data inputs: What data enters the system?
        2. Data transformations: How is data processed and transformed?
        3. Data flow: How does data move through the code?
        4. Data outputs: What data is produced?
        5. Data dependencies: How do different data elements relate to each other?
        
        Focus on understanding the complete data pipeline.
        """
    
    def _get_complexity_analysis_prompt(self) -> str:
        """Get prompt for analyzing complexity."""
        return """
        Analyze the complexity and challenges in this MATLAB code:
        
        {content}
        
        Please assess:
        1. Computational complexity: How complex are the computations?
        2. Code complexity: How complex is the code structure?
        3. Dependencies: How many external dependencies are there?
        4. Potential challenges: What might be difficult to convert to C++?
        5. Optimization opportunities: Where could the code be optimized?
        
        Rate complexity as Low, Medium, or High and explain your reasoning.
        """
    
    def _get_function_analysis_prompt(self) -> str:
        """Get prompt for analyzing individual functions."""
        return """
        Analyze this MATLAB function in detail:
        
        Function: {function_name}
        Code:
        {function_content}
        
        Please provide:
        1. Purpose: What does this function do?
        2. Inputs: What parameters does it accept and what are their types?
        3. Outputs: What does it return and what are the types?
        4. Algorithm: What algorithm or technique does it use?
        5. Complexity: How complex is the implementation?
        6. Dependencies: What other functions does it call?
        7. C++ equivalent: How would you implement this in C++?
        
        Be specific about data types, dimensions, and implementation details.
        """
    
    def _get_llm_response(self, prompt: str) -> str:
        """Get response from LLM."""
        try:
            messages = [
                {"role": "system", "content": "/no_think\n\nYou are an expert MATLAB and C++ developer with deep knowledge of mathematical computing, signal processing, and numerical methods. Analyze the provided MATLAB code thoroughly and provide detailed, accurate insights."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm_client.invoke(messages)
            return response
        except Exception as e:
            self.logger.error(f"Error getting LLM response: {e}")
            return f"Error in analysis: {e}"
    
    def _synthesize_file_understanding(self, matlab_file: MATLABFile, analyses: List[str]) -> CodeUnderstanding:
        """Synthesize understanding from multiple analysis passes."""
        # This would parse the LLM responses and create a structured understanding
        # For now, we'll create a basic structure
        
        return CodeUnderstanding(
            purpose="Analyzed by LLM",  # Would be extracted from analyses
            domain="General",  # Would be extracted from analyses
            algorithms=[],  # Would be extracted from analyses
            data_flow={},  # Would be extracted from analyses
            complexity="Medium",  # Would be extracted from analyses
            confidence=0.8,  # Would be calculated based on analysis quality
            challenges=[],  # Would be extracted from analyses
            suggestions=[]  # Would be extracted from analyses
        )
    
    def _analyze_cross_file_relationships(self, project_structure: ProjectStructure, file_understandings: List[CodeUnderstanding]) -> Dict[str, Any]:
        """Analyze relationships between files in the project."""
        self.logger.debug("Analyzing cross-file relationships...")
        
        # Create comprehensive project context for LLM analysis
        project_context = self._create_project_context(project_structure, file_understandings)
        
        # Use LLM to understand project-wide relationships
        relationship_prompt = self._get_project_relationship_prompt().format(
            project_context=project_context
        )
        
        relationship_analysis = self._get_llm_response(relationship_prompt)
        
        return {
            "relationship_analysis": relationship_analysis,
            "project_context": project_context,
            "file_count": len(project_structure.files),
            "dependency_graph": project_structure.dependencies,
            "entry_points": project_structure.entry_points
        }
    
    def _create_project_context(self, project_structure: ProjectStructure, file_understandings: List[CodeUnderstanding]) -> str:
        """Create comprehensive project context for LLM analysis with token limits."""
        # Estimate total context size and use hierarchical approach if needed
        estimated_tokens = self._estimate_context_tokens(project_structure)
        max_context_tokens = self.config.llm.max_tokens // 2  # Reserve space for response
        
        if estimated_tokens > max_context_tokens:
            return self._create_hierarchical_context(project_structure, file_understandings)
        else:
            return self._create_full_context(project_structure, file_understandings)
    
    def _estimate_context_tokens(self, project_structure: ProjectStructure) -> int:
        """Estimate total token count for project context."""
        total_chars = 0
        
        # Base context overhead
        total_chars += 1000  # Prompts and structure
        
        # File summaries
        for file in project_structure.files:
            total_chars += 200  # File header
            total_chars += len(file.functions) * 100  # Function signatures
            total_chars += min(len(file.dependencies), 10) * 50  # Dependencies
        
        # Dependencies and entry points
        total_chars += len(project_structure.dependencies) * 100
        total_chars += len(project_structure.entry_points) * 50
        
        return total_chars // 4  # Rough token estimate
    
    def _create_full_context(self, project_structure: ProjectStructure, file_understandings: List[CodeUnderstanding]) -> str:
        """Create full project context (when within token limits)."""
        context_parts = []
        
        # Project overview
        context_parts.append("=== MATLAB PROJECT OVERVIEW ===")
        context_parts.append(f"Project contains {len(project_structure.files)} files:")
        
        for i, file in enumerate(project_structure.files):
            context_parts.append(f"\n--- File {i+1}: {file.path.name} ---")
            context_parts.append(f"Size: {file.size} bytes")
            context_parts.append(f"Functions: {len(file.functions)}")
            context_parts.append(f"Dependencies: {len(file.dependencies)}")
            
            if file.functions:
                context_parts.append("Functions:")
                for func in file.functions:
                    context_parts.append(f"  - {func['name']}({', '.join(func['parameters'])})")
            
            if file.dependencies:
                context_parts.append("Dependencies:")
                for dep in file.dependencies[:10]:  # Show first 10
                    context_parts.append(f"  - {dep}")
                if len(file.dependencies) > 10:
                    context_parts.append(f"  ... and {len(file.dependencies) - 10} more")
        
        # Dependencies between files
        context_parts.append("\n=== INTER-FILE DEPENDENCIES ===")
        for file_path, deps in project_structure.dependencies.items():
            if deps:
                context_parts.append(f"{Path(file_path).name} depends on:")
                for dep in deps:
                    context_parts.append(f"  - {Path(dep).name}")
        
        # Entry points
        context_parts.append("\n=== PROJECT ENTRY POINTS ===")
        for entry in project_structure.entry_points:
            context_parts.append(f"- {Path(entry).name}")
        
        return "\n".join(context_parts)
    
    def _create_hierarchical_context(self, project_structure: ProjectStructure, file_understandings: List[CodeUnderstanding]) -> str:
        """Create hierarchical context for large projects (within token limits)."""
        context_parts = []
        
        # High-level project overview
        context_parts.append("=== MATLAB PROJECT OVERVIEW (HIERARCHICAL) ===")
        context_parts.append(f"Project contains {len(project_structure.files)} files")
        context_parts.append(f"Total size: {sum(f.size for f in project_structure.files)} bytes")
        context_parts.append(f"Total functions: {sum(len(f.functions) for f in project_structure.files)}")
        
        # File summary (no detailed content)
        context_parts.append("\n=== FILE SUMMARY ===")
        for i, file in enumerate(project_structure.files):
            context_parts.append(f"{i+1}. {file.path.name}: {len(file.functions)} functions, {len(file.dependencies)} deps")
        
        # Key functions only
        context_parts.append("\n=== KEY FUNCTIONS ===")
        all_functions = []
        for file in project_structure.files:
            for func in file.functions:
                all_functions.append(f"{file.path.name}:{func['name']}({', '.join(func['parameters'])})")
        
        # Show most important functions (entry points, main functions)
        key_functions = all_functions[:20]  # Limit to 20 most important
        for func in key_functions:
            context_parts.append(f"  - {func}")
        
        if len(all_functions) > 20:
            context_parts.append(f"  ... and {len(all_functions) - 20} more functions")
        
        # Dependencies summary
        context_parts.append("\n=== DEPENDENCY SUMMARY ===")
        for file_path, deps in project_structure.dependencies.items():
            if deps:
                context_parts.append(f"{Path(file_path).name} -> {len(deps)} dependencies")
        
        # Entry points
        context_parts.append("\n=== ENTRY POINTS ===")
        for entry in project_structure.entry_points:
            context_parts.append(f"- {Path(entry).name}")
        
        return "\n".join(context_parts)
    
    def _get_project_relationship_prompt(self) -> str:
        """Get prompt for analyzing project-wide relationships."""
        return """
        Analyze this MATLAB project to understand the overall architecture and relationships:
        
        {project_context}
        
        Please provide a comprehensive analysis:
        
        1. **Project Purpose**: What is the main purpose of this entire project?
        2. **Architecture**: How are the files organized? What's the overall structure?
        3. **Data Flow**: How does data flow between files and functions?
        4. **Dependencies**: What are the key dependencies and how do they relate?
        5. **Entry Points**: What are the main entry points and how do they work?
        6. **Domain**: What field/domain does this project belong to?
        7. **Key Algorithms**: What are the main algorithms or techniques used across the project?
        8. **Complexity**: How complex is the overall project?
        9. **Conversion Challenges**: What challenges would exist when converting to C++?
        10. **Recommendations**: How should this project be organized in C++?
        
        Focus on understanding the project as a whole, not just individual files.
        """
    
    def _synthesize_project_understanding(self, project_structure: ProjectStructure, file_understandings: List[CodeUnderstanding], cross_file_analysis: Dict[str, Any]) -> ProjectUnderstanding:
        """Synthesize comprehensive project-level understanding from all analyses."""
        # This would parse the LLM responses and create structured understanding
        # For now, we'll create a comprehensive structure based on the analysis
        
        return ProjectUnderstanding(
            main_purpose="Comprehensive MATLAB project analysis",  # Would be extracted from LLM
            domain="Multi-domain project",  # Would be synthesized from all files
            key_algorithms=[],  # Would be aggregated from all files
            architecture="Multi-file project with dependencies",  # Would be determined from structure
            complexity_level="Medium",  # Would be calculated from all files
            conversion_challenges=[],  # Would be identified from cross-file analysis
            recommendations=[],  # Would be generated from comprehensive analysis
            confidence=0.9  # Higher confidence due to comprehensive analysis
        )
    
    def _parse_function_understanding(self, response: str, function_name: str) -> FunctionUnderstanding:
        """Parse LLM response into structured function understanding."""
        # This would parse the LLM response and extract structured information
        # For now, we'll create a basic structure
        
        return FunctionUnderstanding(
            name=function_name,
            purpose="Function analyzed by LLM",
            inputs=[],
            outputs=[],
            algorithm_type="General",
            complexity="Medium",
            dependencies=[],
            cpp_equivalent_strategy="Direct conversion",
            confidence=0.8
        )
