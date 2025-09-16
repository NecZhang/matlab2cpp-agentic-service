"""Hierarchical analyzer for large MATLAB projects within token limits."""

from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass
from loguru import logger

from ..tools.matlab_parser import MATLABParser, ProjectStructure, MATLABFile
from ..tools.llm_client import create_llm_client
from ..utils.config import get_config


@dataclass
class ProjectSummary:
    """High-level project summary for large projects."""
    total_files: int
    total_functions: int
    total_size: int
    key_files: List[str]
    entry_points: List[str]
    main_domains: List[str]
    complexity_level: str


@dataclass
class FileGroup:
    """Group of related files for analysis."""
    name: str
    files: List[MATLABFile]
    purpose: str
    dependencies: List[str]


class HierarchicalAnalyzer:
    """Analyzer for large projects using hierarchical approach within token limits."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logger.bind(name="hierarchical_analyzer")
        self.llm_client = create_llm_client(self.config.llm)
        self.parser = MATLABParser()
    
    def analyze_large_project(self, project_path: Path) -> Dict[str, Any]:
        """Analyze large project using hierarchical approach."""
        self.logger.info(f"Starting hierarchical analysis of large project: {project_path}")
        
        # Step 1: Parse project structure
        project_structure = self.parser.parse_project(project_path)
        
        # Step 2: Create project summary
        project_summary = self._create_project_summary(project_structure)
        
        # Step 3: Group files by functionality
        file_groups = self._group_files_by_functionality(project_structure)
        
        # Step 4: Analyze each group
        group_analyses = []
        for group in file_groups:
            analysis = self._analyze_file_group(group)
            group_analyses.append(analysis)
        
        # Step 5: Synthesize overall understanding
        overall_understanding = self._synthesize_overall_understanding(
            project_summary, group_analyses
        )
        
        return {
            "project_summary": project_summary,
            "file_groups": file_groups,
            "group_analyses": group_analyses,
            "overall_understanding": overall_understanding,
            "analysis_method": "hierarchical"
        }
    
    def _create_project_summary(self, project_structure: ProjectStructure) -> ProjectSummary:
        """Create high-level project summary."""
        total_functions = sum(len(f.functions) for f in project_structure.files)
        total_size = sum(f.size for f in project_structure.files)
        
        # Identify key files (largest, most functions, entry points)
        key_files = []
        for file in project_structure.files:
            if (len(file.functions) > 0 or 
                file.size > total_size / len(project_structure.files) * 2 or
                str(file.path) in project_structure.entry_points):
                key_files.append(file.path.name)
        
        return ProjectSummary(
            total_files=len(project_structure.files),
            total_functions=total_functions,
            total_size=total_size,
            key_files=key_files,
            entry_points=[Path(ep).name for ep in project_structure.entry_points],
            main_domains=[],  # Will be filled by LLM analysis
            complexity_level="Unknown"  # Will be determined by LLM
        )
    
    def _group_files_by_functionality(self, project_structure: ProjectStructure) -> List[FileGroup]:
        """Group files by functionality to reduce context size."""
        groups = []
        
        # Group 1: Entry points and main scripts
        main_files = []
        for file in project_structure.files:
            if str(file.path) in project_structure.entry_points or len(file.functions) == 0:
                main_files.append(file)
        
        if main_files:
            groups.append(FileGroup(
                name="Main Scripts",
                files=main_files,
                purpose="Main entry points and scripts",
                dependencies=[]
            ))
        
        # Group 2: Core functionality files
        core_files = []
        for file in project_structure.files:
            if (len(file.functions) > 0 and 
                str(file.path) not in project_structure.entry_points and
                len(file.functions) <= 5):  # Small to medium functions
                core_files.append(file)
        
        if core_files:
            groups.append(FileGroup(
                name="Core Functions",
                files=core_files,
                purpose="Core functionality and utilities",
                dependencies=[]
            ))
        
        # Group 3: Complex modules
        complex_files = []
        for file in project_structure.files:
            if len(file.functions) > 5:  # Large modules
                complex_files.append(file)
        
        if complex_files:
            groups.append(FileGroup(
                name="Complex Modules",
                files=complex_files,
                purpose="Complex modules with many functions",
                dependencies=[]
            ))
        
        return groups
    
    def _analyze_file_group(self, group: FileGroup) -> Dict[str, Any]:
        """Analyze a group of files within token limits."""
        self.logger.info(f"Analyzing file group: {group.name}")
        
        # Create compact context for this group
        group_context = self._create_group_context(group)
        
        # Analyze with LLM
        prompt = self._get_group_analysis_prompt().format(
            group_name=group.name,
            group_context=group_context
        )
        
        try:
            response = self.llm_client.invoke([
                {"role": "system", "content": "You are an expert MATLAB developer. Analyze the provided file group and provide concise insights."},
                {"role": "user", "content": prompt}
            ])
            
            return {
                "group_name": group.name,
                "file_count": len(group.files),
                "analysis": response,
                "files": [f.path.name for f in group.files]
            }
        except Exception as e:
            self.logger.error(f"Error analyzing group {group.name}: {e}")
            return {
                "group_name": group.name,
                "file_count": len(group.files),
                "analysis": f"Analysis failed: {e}",
                "files": [f.path.name for f in group.files]
            }
    
    def _create_group_context(self, group: FileGroup) -> str:
        """Create compact context for file group."""
        context_parts = []
        
        context_parts.append(f"=== {group.name} ===")
        context_parts.append(f"Purpose: {group.purpose}")
        context_parts.append(f"Files: {len(group.files)}")
        
        for file in group.files:
            context_parts.append(f"\n--- {file.path.name} ---")
            context_parts.append(f"Size: {file.size} bytes")
            context_parts.append(f"Functions: {len(file.functions)}")
            
            if file.functions:
                context_parts.append("Functions:")
                for func in file.functions:
                    params = ', '.join(func['parameters']) if func['parameters'] else 'none'
                    context_parts.append(f"  - {func['name']}({params})")
            
            # Show only key dependencies
            if file.dependencies:
                key_deps = file.dependencies[:5]  # Limit to 5 most important
                context_parts.append(f"Dependencies: {', '.join(key_deps)}")
        
        return "\n".join(context_parts)
    
    def _get_group_analysis_prompt(self) -> str:
        """Get prompt for analyzing file groups."""
        return """
        Analyze this MATLAB file group and provide insights:
        
        Group: {group_name}
        Context:
        {group_context}
        
        Please provide:
        1. **Purpose**: What does this group do?
        2. **Domain**: What field/domain does it belong to?
        3. **Key Functions**: What are the main functions?
        4. **Dependencies**: What does it depend on?
        5. **Complexity**: How complex is this group?
        
        Keep your response concise and focused.
        """
    
    def _synthesize_overall_understanding(self, project_summary: ProjectSummary, 
                                        group_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize overall project understanding from group analyses."""
        # Create summary context
        summary_context = f"""
        Project Summary:
        - Total files: {project_summary.total_files}
        - Total functions: {project_summary.total_functions}
        - Total size: {project_summary.total_size} bytes
        - Key files: {', '.join(project_summary.key_files)}
        - Entry points: {', '.join(project_summary.entry_points)}
        
        Group Analyses:
        """
        
        for analysis in group_analyses:
            summary_context += f"\n--- {analysis['group_name']} ---\n"
            summary_context += f"Files: {analysis['file_count']}\n"
            summary_context += f"Analysis: {analysis['analysis'][:200]}...\n"
        
        # Get overall understanding from LLM
        prompt = f"""
        Based on this hierarchical analysis of a large MATLAB project, provide overall insights:
        
        {summary_context}
        
        Please provide:
        1. **Overall Purpose**: What is the main purpose of this project?
        2. **Architecture**: How is the project organized?
        3. **Domain**: What field does it belong to?
        4. **Complexity**: Overall complexity level
        5. **Key Components**: What are the main components?
        6. **Conversion Strategy**: How should this be converted to C++?
        """
        
        try:
            response = self.llm_client.invoke([
                {"role": "system", "content": "You are an expert in MATLAB and C++ development. Provide comprehensive project analysis."},
                {"role": "user", "content": prompt}
            ])
            
            return {
                "overall_analysis": response,
                "project_summary": project_summary,
                "analysis_method": "hierarchical",
                "confidence": 0.8
            }
        except Exception as e:
            self.logger.error(f"Error in overall synthesis: {e}")
            return {
                "overall_analysis": f"Synthesis failed: {e}",
                "project_summary": project_summary,
                "analysis_method": "hierarchical",
                "confidence": 0.5
            }


