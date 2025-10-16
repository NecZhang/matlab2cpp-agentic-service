"""
API Knowledge Base - Dynamic context-aware API documentation retrieval.

This module provides relevant API documentation based on compilation errors,
helping the LLM fix issues without overwhelming it with unnecessary information.
"""

import re
import yaml
from typing import List, Dict, Set, Optional, Any
from pathlib import Path
from loguru import logger


class APIKnowledgeBase:
    """
    Dynamic API documentation retrieval system.
    
    Provides context-aware API docs based on:
    - Compilation errors
    - Code context
    - Libraries used
    
    General and extensible to any C++ library.
    """
    
    def __init__(self, knowledge_dir: Optional[Path] = None):
        """
        Initialize knowledge base.
        
        Args:
            knowledge_dir: Directory containing YAML knowledge files
                          Defaults to <module_dir>/knowledge_db/
        """
        self.logger = logger.bind(name="APIKnowledgeBase")
        
        # Set knowledge directory
        if knowledge_dir is None:
            knowledge_dir = Path(__file__).parent / "knowledge_db"
        self.knowledge_dir = Path(knowledge_dir)
        
        # Load all knowledge databases
        self.knowledge_db = {}
        self._load_all_knowledge()
        
        self.logger.info(f"API Knowledge Base initialized with {len(self.knowledge_db)} libraries")
    
    def _load_all_knowledge(self):
        """Load all YAML knowledge files from knowledge_dir."""
        if not self.knowledge_dir.exists():
            self.logger.warning(f"Knowledge directory not found: {self.knowledge_dir}")
            self.knowledge_dir.mkdir(parents=True, exist_ok=True)
            return
        
        for yaml_file in self.knowledge_dir.glob("*.yaml"):
            try:
                with open(yaml_file, 'r') as f:
                    data = yaml.safe_load(f)
                    library_name = yaml_file.stem
                    self.knowledge_db[library_name] = data
                    self.logger.debug(f"Loaded knowledge: {library_name}")
            except Exception as e:
                self.logger.error(f"Failed to load {yaml_file}: {e}")
    
    def get_relevant_docs(self,
                         error_messages: List[str],
                         current_code: str,
                         libraries_used: Optional[List[str]] = None) -> str:
        """
        Get relevant API documentation based on errors and code context.
        
        Args:
            error_messages: List of compilation error messages
            current_code: Current C++ code being fixed
            libraries_used: List of library names (e.g., ['eigen', 'opencv'])
                          If None, auto-detect from code
        
        Returns:
            Formatted API documentation string for LLM prompt
        """
        # Auto-detect libraries if not provided
        if libraries_used is None:
            libraries_used = self._detect_libraries(current_code)
        
        # Extract API-related keywords from errors
        api_keywords = self._extract_api_keywords(error_messages)
        
        # Extract types and methods from code
        code_keywords = self._extract_code_keywords(current_code)
        
        # Combine all keywords
        all_keywords = api_keywords | code_keywords
        
        # Retrieve relevant docs
        relevant_docs = []
        for lib in libraries_used:
            if lib in self.knowledge_db:
                docs = self._retrieve_library_docs(lib, all_keywords)
                if docs:
                    relevant_docs.append({
                        'library': lib,
                        'docs': docs
                    })
        
        # Format for LLM
        if not relevant_docs:
            return ""
        
        return self._format_docs_for_llm(relevant_docs)
    
    def _detect_libraries(self, code: str) -> List[str]:
        """Detect which libraries are used in the code."""
        libraries = set()
        
        # Check for includes
        include_patterns = {
            'eigen': r'#include\s+[<"](?:Eigen|unsupported/Eigen)',
            'opencv': r'#include\s+[<"]opencv',
            'boost': r'#include\s+[<"]boost',
            'std': r'#include\s+[<"](?:vector|string|iostream|algorithm|memory|tuple|utility)',
        }
        
        for lib, pattern in include_patterns.items():
            if re.search(pattern, code):
                libraries.add(lib)
        
        # Check for namespace usage
        if re.search(r'\bEigen::', code):
            libraries.add('eigen')
        if re.search(r'\bcv::', code):
            libraries.add('opencv')
        if re.search(r'\bstd::', code):
            libraries.add('std')
        
        return list(libraries)
    
    def _extract_api_keywords(self, error_messages: List[str]) -> Set[str]:
        """Extract API-related keywords from error messages."""
        keywords = set()
        
        for error in error_messages:
            # Extract quoted identifiers
            quoted = re.findall(r"'(\w+)'", error)
            keywords.update(quoted)
            
            # Extract type names from "is not a member of 'X'"
            member_match = re.search(r"'(\w+)' is not a member of '(\w+)'", error)
            if member_match:
                keywords.add(member_match.group(1))  # The missing member
                keywords.add(member_match.group(2))  # The parent type
            
            # Extract method names from "has no member named 'X'"
            method_match = re.search(r"has no member named '(\w+)'", error)
            if method_match:
                keywords.add(method_match.group(1))
            
            # Extract types from template errors
            template_types = re.findall(r'Eigen::(\w+)', error)
            keywords.update(template_types)
        
        return keywords
    
    def _extract_code_keywords(self, code: str) -> Set[str]:
        """Extract types and methods from code."""
        keywords = set()
        
        # Extract Eigen types
        eigen_types = re.findall(r'Eigen::(\w+)', code)
        keywords.update(eigen_types)
        
        # Extract method calls
        method_calls = re.findall(r'\.(\w+)\(', code)
        keywords.update(method_calls)
        
        # Extract template instantiations
        templates = re.findall(r'(\w+)<[^>]+>', code)
        keywords.update(templates)
        
        return keywords
    
    def _retrieve_library_docs(self, library: str, keywords: Set[str]) -> List[Dict[str, Any]]:
        """
        Retrieve documentation entries for a library that match keywords.
        """
        lib_data = self.knowledge_db.get(library, {})
        matched_docs = []
        
        # Search in types
        if 'types' in lib_data:
            for type_name, type_info in lib_data['types'].items():
                if type_name in keywords or any(kw in type_name for kw in keywords):
                    matched_docs.append({
                        'category': 'type',
                        'name': type_name,
                        'info': type_info
                    })
        
        # Search in methods
        if 'methods' in lib_data:
            for method_name, method_info in lib_data['methods'].items():
                if method_name in keywords or any(kw in method_name for kw in keywords):
                    matched_docs.append({
                        'category': 'method',
                        'name': method_name,
                        'info': method_info
                    })
        
        # Search in common_errors
        if 'common_errors' in lib_data:
            for error_pattern, fix_info in lib_data['common_errors'].items():
                # Check if any keyword matches this error pattern
                if any(kw in error_pattern for kw in keywords):
                    matched_docs.append({
                        'category': 'common_error',
                        'pattern': error_pattern,
                        'info': fix_info
                    })
        
        return matched_docs
    
    def _format_docs_for_llm(self, relevant_docs: List[Dict]) -> str:
        """
        Format retrieved documentation for LLM consumption.
        
        Creates a clean, readable format with examples and common pitfalls.
        """
        if not relevant_docs:
            return ""
        
        output = ["", "━" * 80, "📚 API REFERENCE (Relevant to Your Errors)", "━" * 80, ""]
        
        for lib_docs in relevant_docs:
            library = lib_docs['library']
            docs = lib_docs['docs']
            
            if not docs:
                continue
            
            output.append(f"### {library.upper()} API:")
            output.append("")
            
            # Group by category
            types = [d for d in docs if d['category'] == 'type']
            methods = [d for d in docs if d['category'] == 'method']
            errors = [d for d in docs if d['category'] == 'common_error']
            
            # Format types
            if types:
                output.append("**Types:**")
                for doc in types:
                    output.append(f"  • {doc['name']}")
                    info = doc['info']
                    if 'header' in info:
                        output.append(f"    Header: {info['header']}")
                    if 'purpose' in info:
                        output.append(f"    Purpose: {info['purpose']}")
                    if 'example' in info:
                        output.append(f"    Example:")
                        for line in info['example'].split('\n'):
                            output.append(f"      {line}")
                    output.append("")
            
            # Format methods
            if methods:
                output.append("**Methods:**")
                for doc in methods:
                    output.append(f"  • {doc['name']}")
                    info = doc['info']
                    if 'replacement' in info:
                        output.append(f"    Use instead: {info['replacement']}")
                    if 'example' in info:
                        output.append(f"    Example:")
                        for line in info['example'].split('\n'):
                            output.append(f"      {line}")
                    output.append("")
            
            # Format common errors
            if errors:
                output.append("**Common Errors & Fixes:**")
                for doc in errors:
                    info = doc['info']
                    output.append(f"  ❌ Error: {doc['pattern']}")
                    if 'fix' in info:
                        output.append(f"  ✅ Fix: {info['fix']}")
                    if 'example' in info:
                        output.append(f"     Example:")
                        for line in info['example'].split('\n'):
                            output.append(f"       {line}")
                    output.append("")
        
        output.append("━" * 80)
        output.append("")
        
        return '\n'.join(output)
    
    def add_library_knowledge(self, library_name: str, knowledge_dict: Dict[str, Any]):
        """
        Dynamically add knowledge for a new library.
        
        Args:
            library_name: Name of the library (e.g., 'eigen', 'opencv')
            knowledge_dict: Dictionary containing types, methods, common_errors
        """
        self.knowledge_db[library_name] = knowledge_dict
        self.logger.info(f"Added knowledge for library: {library_name}")
    
    def save_knowledge_to_yaml(self, library_name: str, output_path: Optional[Path] = None):
        """
        Save library knowledge to YAML file for persistence.
        
        Args:
            library_name: Name of the library
            output_path: Where to save (defaults to knowledge_dir/{library_name}.yaml)
        """
        if library_name not in self.knowledge_db:
            self.logger.error(f"No knowledge for library: {library_name}")
            return
        
        if output_path is None:
            output_path = self.knowledge_dir / f"{library_name}.yaml"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self.knowledge_db[library_name], f, default_flow_style=False)
        
        self.logger.info(f"Saved {library_name} knowledge to {output_path}")
    
    def get_preventive_guidance(self, libraries_used: List[str] = None) -> str:
        """
        Get preventive API guidance to include in INITIAL generation prompts.
        
        This provides upfront guidance to PREVENT errors, not just fix them.
        
        Args:
            libraries_used: List of library names (e.g., ['eigen', 'opencv'])
                          If None, returns guidance for all libraries
        
        Returns:
            Formatted guidance string for LLM prompt
        """
        if libraries_used is None:
            libraries_used = list(self.knowledge_db.keys())
        
        guidance_sections = []
        
        for lib in libraries_used:
            if lib not in self.knowledge_db:
                continue
            
            lib_data = self.knowledge_db[lib]
            lib_guidance = []
            
            # Add critical reminders (highest priority)
            if 'critical_reminders' in lib_data:
                lib_guidance.append("🚨 CRITICAL RULES:")
                for reminder in lib_data['critical_reminders']:
                    lib_guidance.append(f"  • {reminder}")
                lib_guidance.append("")
            
            # Add operator rules
            if 'operators' in lib_data:
                lib_guidance.append("⚙️  OPERATOR RULES:")
                operators = lib_data['operators']
                
                for op_category, op_data in operators.items():
                    if isinstance(op_data, dict) and 'description' in op_data:
                        lib_guidance.append(f"  {op_data['description']}")
                        
                        if 'examples' in op_data:
                            for example in op_data['examples'][:3]:  # Limit to 3 examples
                                if isinstance(example, dict):
                                    lib_guidance.append(f"    ❌ WRONG: {example.get('wrong', '')}")
                                    lib_guidance.append(f"    ✅ RIGHT: {example.get('correct', '')}")
                lib_guidance.append("")
            
            # Add namespace patterns
            if 'namespace_patterns' in lib_data:
                lib_guidance.append("📦 NAMESPACE RULES:")
                ns_patterns = lib_data['namespace_patterns']
                
                if 'duplicate_namespace_function' in ns_patterns:
                    dup = ns_patterns['duplicate_namespace_function']
                    lib_guidance.append(f"  • {dup.get('explanation', '')}")
                    if 'examples' in dup and dup['examples']:
                        example = dup['examples'][0]
                        lib_guidance.append(f"    ❌ WRONG: {example.get('wrong', '')}")
                        lib_guidance.append(f"    ✅ RIGHT: {example.get('best', example.get('correct', ''))}")
                lib_guidance.append("")
            
            # Add common variables (for reference)
            if 'common_variables' in lib_data:
                lib_guidance.append("📝 COMMON VARIABLE DECLARATIONS:")
                vars_data = lib_data['common_variables']
                common_vars = ['check', 'stepsize', 'tolerance', 'i', 'j', 'k']
                
                for var_name in common_vars:
                    if var_name in vars_data:
                        var_info = vars_data[var_name]
                        example = var_info.get('example', '')
                        if not example:
                            var_type = var_info.get('type', 'auto')
                            var_val = var_info.get('typical_value', '0')
                            example = f"{var_type} {var_name} = {var_val};"
                        lib_guidance.append(f"  • {example}")
                lib_guidance.append("")
            
            if lib_guidance:
                guidance_sections.append(f"\n{'═' * 80}")
                guidance_sections.append(f"📚 {lib.upper()} API GUIDANCE (Prevent Common Errors)")
                guidance_sections.append('═' * 80)
                guidance_sections.extend(lib_guidance)
        
        # Add hallucination warnings if available
        if 'hallucinations' in self.knowledge_db:
            hall_data = self.knowledge_db['hallucinations']
            guidance_sections.append(f"\n{'═' * 80}")
            guidance_sections.append("⚠️  METHODS THAT DON'T EXIST (Common Hallucinations)")
            guidance_sections.append('═' * 80)
            
            if 'eigen' in hall_data:
                guidance_sections.append("EIGEN:")
                eigen_halls = hall_data['eigen']
                for method, info in list(eigen_halls.items())[:5]:  # Limit to 5
                    if isinstance(info, dict) and 'correct_api' in info:
                        guidance_sections.append(f"  ❌ .{method}() - DOES NOT EXIST")
                        guidance_sections.append(f"  ✅ Use: {info['correct_api']}")
            
            if 'stdlib' in hall_data:
                guidance_sections.append("")
                guidance_sections.append("STANDARD LIBRARY:")
                for issue, info in hall_data['stdlib'].items():
                    if isinstance(info, dict) and 'correct' in info:
                        guidance_sections.append(f"  ✅ {info['correct']}")
            
            guidance_sections.append("")
        
        return "\n".join(guidance_sections)


