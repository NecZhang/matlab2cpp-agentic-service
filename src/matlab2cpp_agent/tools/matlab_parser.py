"""LLM-centric MATLAB code parser focusing on content understanding."""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

from ..utils.config import get_config


@dataclass
class MATLABFile:
    """Represents a MATLAB file with its content and metadata."""
    path: Path
    content: str
    size: int
    functions: List[Dict[str, Any]]
    dependencies: List[str]
    variables: List[str]
    comments: List[str]


@dataclass
class CodeBlock:
    """Represents a logical code block within a MATLAB file."""
    start_line: int
    end_line: int
    content: str
    type: str  # 'function', 'script', 'initialization', 'main_logic', 'cleanup'
    purpose: str
    inputs: List[str]
    outputs: List[str]
    variables: List[str]


@dataclass
class ProjectStructure:
    """Represents the overall structure of a MATLAB project."""
    files: List[MATLABFile]
    dependencies: Dict[str, List[str]]
    entry_points: List[str]
    main_purpose: str
    domain: str


class MATLABParser:
    """LLM-centric MATLAB parser that focuses on content understanding."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logger.bind(name="matlab_parser")
    
    def parse_project(self, project_path: Path) -> ProjectStructure:
        """Parse an entire MATLAB project."""
        self.logger.info(f"Parsing MATLAB project: {project_path}")
        
        # Find all MATLAB files
        matlab_files = self._find_matlab_files(project_path)
        self.logger.info(f"Found {len(matlab_files)} MATLAB files")
        
        # Parse each file
        parsed_files = []
        for file_path in matlab_files:
            try:
                parsed_file = self.parse_file(file_path)
                parsed_files.append(parsed_file)
            except Exception as e:
                self.logger.error(f"Error parsing {file_path}: {e}")
                continue
        
        # Analyze project structure
        dependencies = self._analyze_dependencies(parsed_files)
        entry_points = self._find_entry_points(parsed_files)
        
        return ProjectStructure(
            files=parsed_files,
            dependencies=dependencies,
            entry_points=entry_points,
            main_purpose="",  # Will be filled by LLM analysis
            domain=""  # Will be filled by LLM analysis
        )
    
    def parse_file(self, file_path: Path) -> MATLABFile:
        """Parse a single MATLAB file."""
        self.logger.debug(f"Parsing file: {file_path}")
        
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        # Basic structural analysis
        functions = self._extract_functions(content)
        dependencies = self._extract_dependencies(content)
        variables = self._extract_variables(content)
        comments = self._extract_comments(content)
        
        return MATLABFile(
            path=file_path,
            content=content,
            size=len(content),
            functions=functions,
            dependencies=dependencies,
            variables=variables,
            comments=comments
        )
    
    def _find_matlab_files(self, project_path: Path) -> List[Path]:
        """Find all MATLAB files in the project."""
        matlab_files = []
        
        for pattern in ["**/*.m", "**/*.mat", "**/*.fig"]:
            matlab_files.extend(project_path.glob(pattern))
        
        # Filter out common non-code files
        exclude_patterns = [
            "*.asv",  # MATLAB autosave files
            "*~",     # Backup files
            "*.bak"   # Backup files
        ]
        
        filtered_files = []
        for file_path in matlab_files:
            if not any(file_path.name.endswith(pattern.replace("*", "")) for pattern in exclude_patterns):
                filtered_files.append(file_path)
        
        return sorted(filtered_files)
    
    def _extract_functions(self, content: str) -> List[Dict[str, Any]]:
        """Extract function definitions from MATLAB code."""
        functions = []
        
        # Pattern for function definitions
        function_pattern = r'function\s+(?:\[([^\]]*)\]\s*=\s*)?(\w+)\s*\(([^)]*)\)'
        
        for match in re.finditer(function_pattern, content, re.MULTILINE):
            return_vars = match.group(1)
            function_name = match.group(2)
            parameters = match.group(3)
            
            # Find function end
            start_pos = match.start()
            end_pos = self._find_function_end(content, start_pos)
            
            function_content = content[start_pos:end_pos]
            
            functions.append({
                'name': function_name,
                'parameters': [p.strip() for p in parameters.split(',') if p.strip()],
                'return_vars': [r.strip() for r in return_vars.split(',')] if return_vars else [],
                'content': function_content,
                'start_line': content[:start_pos].count('\n') + 1,
                'end_line': content[:end_pos].count('\n') + 1
            })
        
        return functions
    
    def _extract_dependencies(self, content: str) -> List[str]:
        """Extract dependencies from MATLAB code."""
        dependencies = set()
        
        # Common MATLAB functions and toolboxes
        patterns = [
            r'(\w+)\s*\(',  # Function calls
            r'addpath\s*\([^)]*\)',  # Path additions
            r'import\s+([^;]+)',  # Import statements
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            dependencies.update(matches)
        
        # Filter out common MATLAB built-ins that don't need special handling
        common_builtins = {
            'disp', 'fprintf', 'sprintf', 'num2str', 'str2num',
            'length', 'size', 'zeros', 'ones', 'eye', 'rand',
            'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs',
            'max', 'min', 'sum', 'mean', 'std', 'var',
            'plot', 'figure', 'subplot', 'xlabel', 'ylabel', 'title',
            'if', 'else', 'elseif', 'end', 'for', 'while', 'break', 'continue'
        }
        
        return [dep for dep in dependencies if dep not in common_builtins]
    
    def _extract_variables(self, content: str) -> List[str]:
        """Extract variable names from MATLAB code."""
        # Simple variable extraction - this will be enhanced by LLM analysis
        variable_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*='
        variables = re.findall(variable_pattern, content)
        return list(set(variables))
    
    def _extract_comments(self, content: str) -> List[str]:
        """Extract comments from MATLAB code."""
        comments = []
        lines = content.split('\n')
        
        for line in lines:
            # Remove leading whitespace
            stripped = line.lstrip()
            if stripped.startswith('%'):
                comments.append(stripped[1:].strip())
        
        return comments
    
    def _find_function_end(self, content: str, start_pos: int) -> int:
        """Find the end of a function definition."""
        lines = content[start_pos:].split('\n')
        brace_count = 0
        in_function = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            if stripped.startswith('function'):
                in_function = True
                continue
            
            if in_function:
                if stripped == 'end':
                    if brace_count == 0:
                        return start_pos + sum(len(l) + 1 for l in lines[:i+1])
                elif stripped.startswith('if') or stripped.startswith('for') or stripped.startswith('while'):
                    brace_count += 1
                elif stripped == 'end':
                    brace_count -= 1
        
        return len(content)
    
    def _analyze_dependencies(self, files: List[MATLABFile]) -> Dict[str, List[str]]:
        """Analyze dependencies between files."""
        dependencies = {}
        
        for file in files:
            file_deps = []
            for dep in file.dependencies:
                # Check if dependency is another file in the project
                for other_file in files:
                    if other_file.path.stem == dep:
                        file_deps.append(str(other_file.path))
            
            dependencies[str(file.path)] = file_deps
        
        return dependencies
    
    def _find_entry_points(self, files: List[MATLABFile]) -> List[str]:
        """Find potential entry points (main scripts) in the project."""
        entry_points = []
        
        for file in files:
            # Script files (no function definitions) are likely entry points
            if not file.functions:
                entry_points.append(str(file.path))
            # Files with main function or similar
            elif any(func['name'].lower() in ['main', 'run', 'start'] for func in file.functions):
                entry_points.append(str(file.path))
        
        return entry_points


