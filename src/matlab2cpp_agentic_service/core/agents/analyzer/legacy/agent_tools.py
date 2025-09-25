"""
Agent Tools and Utilities

This module provides shared tools and utilities for LangGraph agents
in the MATLAB2C++ conversion service.
"""

from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
import hashlib
from loguru import logger

from matlab2cpp_agentic_service.infrastructure.tools.llm_client import LLMClient
from matlab2cpp_agentic_service.infrastructure.tools.matlab_parser import MATLABParser


@dataclass
class ToolConfig:
    """Configuration for agent tools."""
    name: str
    description: str
    enabled: bool = True
    timeout_seconds: float = 60.0
    max_retries: int = 3
    parameters: Dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """Registry for managing agent tools."""
    
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.configs: Dict[str, ToolConfig] = {}
        self.logger = logger.bind(name="tool_registry")
    
    def register_tool(self, name: str, tool_func: Callable, config: ToolConfig):
        """Register a tool in the registry."""
        self.tools[name] = tool_func
        self.configs[name] = config
        self.logger.debug(f"Registered tool: {name}")
    
    def get_tool(self, name: str) -> Optional[Callable]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def get_tool_config(self, name: str) -> Optional[ToolConfig]:
        """Get tool configuration by name."""
        return self.configs.get(name)
    
    def list_tools(self) -> List[str]:
        """List all registered tools."""
        return list(self.tools.keys())
    
    def is_tool_enabled(self, name: str) -> bool:
        """Check if a tool is enabled."""
        config = self.configs.get(name)
        return config.enabled if config else False


class AgentTools:
    """
    Collection of shared tools for LangGraph agents.
    
    This class provides common utilities and tools that can be used
    across different agents in the conversion workflow.
    """
    
    def __init__(self, llm_client: LLMClient, tool_registry: Optional[ToolRegistry] = None):
        """
        Initialize agent tools.
        
        Args:
            llm_client: LLM client for tool operations
            tool_registry: Optional tool registry
        """
        self.llm_client = llm_client
        self.tool_registry = tool_registry or ToolRegistry()
        self.logger = logger.bind(name="agent_tools")
        
        # Initialize default tools
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register default tools."""
        # File operations tool
        file_tool_config = ToolConfig(
            name="file_operations",
            description="File reading, writing, and manipulation operations",
            parameters={"encoding": "utf-8", "backup": True}
        )
        self.tool_registry.register_tool("file_operations", self._file_operations, file_tool_config)
        
        # MATLAB parsing tool
        matlab_tool_config = ToolConfig(
            name="matlab_parser",
            description="MATLAB code parsing and analysis",
            parameters={"strict_mode": True, "include_comments": True}
        )
        self.tool_registry.register_tool("matlab_parser", self._matlab_parser, matlab_tool_config)
        
        # Code validation tool
        validation_tool_config = ToolConfig(
            name="code_validation",
            description="Code validation and syntax checking",
            parameters={"strict_mode": True, "include_warnings": True}
        )
        self.tool_registry.register_tool("code_validation", self._code_validation, validation_tool_config)
        
        # LLM interaction tool
        llm_tool_config = ToolConfig(
            name="llm_interaction",
            description="LLM interaction and completion generation",
            parameters={"max_tokens": 4000, "temperature": 0.1}
        )
        self.tool_registry.register_tool("llm_interaction", self._llm_interaction, llm_tool_config)
        
        # Performance measurement tool
        perf_tool_config = ToolConfig(
            name="performance_measurement",
            description="Performance measurement and profiling",
            parameters={"enable_memory_tracking": True}
        )
        self.tool_registry.register_tool("performance_measurement", self._performance_measurement, perf_tool_config)
    
    def _file_operations(self, operation: str, file_path: Union[str, Path], 
                        content: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """File operations tool."""
        try:
            file_path = Path(file_path)
            
            if operation == "read":
                if not file_path.exists():
                    return {"success": False, "error": f"File not found: {file_path}"}
                
                encoding = kwargs.get("encoding", "utf-8")
                content = file_path.read_text(encoding=encoding)
                return {"success": True, "content": content, "size": len(content)}
            
            elif operation == "write":
                if content is None:
                    return {"success": False, "error": "Content required for write operation"}
                
                # Create backup if requested
                if kwargs.get("backup", False) and file_path.exists():
                    backup_path = file_path.with_suffix(file_path.suffix + ".backup")
                    backup_path.write_text(file_path.read_text())
                
                encoding = kwargs.get("encoding", "utf-8")
                file_path.write_text(content, encoding=encoding)
                return {"success": True, "bytes_written": len(content.encode(encoding))}
            
            elif operation == "exists":
                return {"success": True, "exists": file_path.exists()}
            
            elif operation == "size":
                if not file_path.exists():
                    return {"success": False, "error": f"File not found: {file_path}"}
                return {"success": True, "size": file_path.stat().st_size}
            
            else:
                return {"success": False, "error": f"Unknown file operation: {operation}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _matlab_parser(self, matlab_code: str, **kwargs) -> Dict[str, Any]:
        """MATLAB parsing tool."""
        try:
            parser = MATLABParser()
            result = parser.parse(matlab_code)
            
            # Apply strict mode if requested
            if kwargs.get("strict_mode", True):
                # Validate parsed result
                if not result.get("functions"):
                    return {"success": False, "error": "No functions found in MATLAB code"}
            
            return {"success": True, "parsed_result": result}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _code_validation(self, code: str, language: str = "cpp", **kwargs) -> Dict[str, Any]:
        """Code validation tool."""
        try:
            validation_result = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "suggestions": []
            }
            
            if language.lower() == "cpp":
                # Basic C++ validation
                if "int main(" not in code and "int main (" not in code:
                    if kwargs.get("strict_mode", True):
                        validation_result["errors"].append("No main function found")
                        validation_result["valid"] = False
                    else:
                        validation_result["warnings"].append("No main function found")
                
                # Check for common issues
                if "using namespace std;" in code:
                    validation_result["warnings"].append("Consider avoiding 'using namespace std;'")
                
                # Check for memory leaks (basic)
                if "new " in code and "delete " not in code:
                    validation_result["warnings"].append("Potential memory leak: 'new' without 'delete'")
            
            elif language.lower() == "matlab":
                # Basic MATLAB validation
                if not any(keyword in code.lower() for keyword in ["function", "script", "classdef"]):
                    validation_result["errors"].append("No function, script, or class definition found")
                    validation_result["valid"] = False
            
            return {"success": True, "validation_result": validation_result}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _llm_interaction(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """LLM interaction tool."""
        try:
            max_tokens = kwargs.get("max_tokens", 4000)
            temperature = kwargs.get("temperature", 0.1)
            
            # Get completion from LLM
            completion = self.llm_client.get_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return {
                "success": True,
                "completion": completion,
                "tokens_used": len(completion.split()),
                "parameters": {"max_tokens": max_tokens, "temperature": temperature}
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _performance_measurement(self, operation_name: str, **kwargs) -> Dict[str, Any]:
        """Performance measurement tool."""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            
            # Get memory usage
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            # Get CPU usage
            cpu_percent = process.cpu_percent()
            
            return {
                "success": True,
                "operation": operation_name,
                "memory_mb": memory_mb,
                "cpu_percent": cpu_percent,
                "timestamp": time.time(),
                "pid": os.getpid()
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_tool(self, name: str) -> Optional[Callable]:
        """Get a tool by name."""
        return self.tool_registry.get_tool(name)
    
    def execute_tool(self, name: str, *args, **kwargs) -> Dict[str, Any]:
        """Execute a tool by name."""
        tool = self.get_tool(name)
        if not tool:
            return {"success": False, "error": f"Tool not found: {name}"}
        
        config = self.tool_registry.get_tool_config(name)
        if config and not config.enabled:
            return {"success": False, "error": f"Tool disabled: {name}"}
        
        try:
            # Merge tool parameters with kwargs
            if config:
                tool_kwargs = config.parameters.copy()
                tool_kwargs.update(kwargs)
                result = tool(*args, **tool_kwargs)
            else:
                result = tool(*args, **kwargs)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Tool execution failed: {name} - {e}")
            return {"success": False, "error": str(e)}
    
    def create_code_hash(self, code: str) -> str:
        """Create a hash for code content."""
        return hashlib.md5(code.encode('utf-8')).hexdigest()
    
    def format_code_for_llm(self, code: str, language: str = "cpp") -> str:
        """Format code for LLM consumption."""
        if language.lower() == "cpp":
            return f"```cpp\n{code}\n```"
        elif language.lower() == "matlab":
            return f"```matlab\n{code}\n```"
        else:
            return f"```{language}\n{code}\n```"
    
    def extract_code_from_llm_response(self, response: str, language: str = "cpp") -> str:
        """Extract code from LLM response."""
        import re
        
        # Look for code blocks
        pattern = rf"```{language}\s*\n(.*?)\n```"
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        
        # Fallback: look for any code block
        pattern = r"```\s*\n(.*?)\n```"
        match = re.search(pattern, response, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        # If no code blocks found, return the response as-is
        return response.strip()
    
    def validate_json_response(self, response: str) -> Dict[str, Any]:
        """Validate and parse JSON response."""
        try:
            # Try to parse as JSON
            parsed = json.loads(response)
            return {"success": True, "parsed": parsed}
        except json.JSONDecodeError as e:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    return {"success": True, "parsed": parsed}
                except json.JSONDecodeError:
                    pass
            
            return {"success": False, "error": f"Invalid JSON: {e}"}
    
    def get_tool_summary(self) -> Dict[str, Any]:
        """Get summary of available tools."""
        return {
            "total_tools": len(self.tool_registry.list_tools()),
            "enabled_tools": [name for name in self.tool_registry.list_tools() 
                            if self.tool_registry.is_tool_enabled(name)],
            "disabled_tools": [name for name in self.tool_registry.list_tools() 
                             if not self.tool_registry.is_tool_enabled(name)],
            "tool_descriptions": {name: config.description 
                                for name, config in self.tool_registry.configs.items()}
        }
