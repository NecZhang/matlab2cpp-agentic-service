"""C++ compiler integration for validation."""

import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger


class CppCompiler:
    """C++ compiler wrapper for validation."""
    
    def __init__(self, compiler: str = "g++"):
        self.compiler = compiler
        self.logger = logger.bind(name="cpp_compiler")
    
    def compile_project(self, project_path: Path, 
                       build_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Compile a C++ project."""
        self.logger.info(f"Compiling C++ project: {project_path}")
        
        if build_dir is None:
            build_dir = project_path / "build"
        
        build_dir.mkdir(exist_ok=True)
        
        try:
            # Try CMake first
            if (project_path / "CMakeLists.txt").exists():
                return self._compile_with_cmake(project_path, build_dir)
            else:
                return self._compile_directly(project_path, build_dir)
        except Exception as e:
            self.logger.error(f"Compilation failed: {e}")
            return {
                "success": False,
                "errors": [str(e)],
                "warnings": []
            }
    
    def _compile_with_cmake(self, project_path: Path, build_dir: Path) -> Dict[str, Any]:
        """Compile using CMake."""
        try:
            # Configure
            result = subprocess.run(
                ["cmake", ".."],
                cwd=build_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return {
                    "success": False,
                    "errors": [result.stderr],
                    "warnings": []
                }
            
            # Build
            result = subprocess.run(
                ["cmake", "--build", "."],
                cwd=build_dir,
                capture_output=True,
                text=True
            )
            
            return {
                "success": result.returncode == 0,
                "errors": [result.stderr] if result.stderr else [],
                "warnings": [result.stdout] if result.stdout else []
            }
            
        except Exception as e:
            return {
                "success": False,
                "errors": [str(e)],
                "warnings": []
            }
    
    def _compile_directly(self, project_path: Path, build_dir: Path) -> Dict[str, Any]:
        """Compile directly with compiler."""
        # Find source files
        source_files = list(project_path.glob("**/*.cpp"))
        
        if not source_files:
            return {
                "success": False,
                "errors": ["No source files found"],
                "warnings": []
            }
        
        try:
            # Compile
            cmd = [self.compiler, "-std=c++17", "-o", "main"] + [str(f) for f in source_files]
            result = subprocess.run(
                cmd,
                cwd=build_dir,
                capture_output=True,
                text=True
            )
            
            return {
                "success": result.returncode == 0,
                "errors": [result.stderr] if result.stderr else [],
                "warnings": [result.stdout] if result.stdout else []
            }
            
        except Exception as e:
            return {
                "success": False,
                "errors": [str(e)],
                "warnings": []
            }
    
    def run_tests(self, executable_path: Path) -> Dict[str, Any]:
        """Run tests on compiled executable."""
        self.logger.info(f"Running tests: {executable_path}")
        
        try:
            result = subprocess.run(
                [str(executable_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "errors": [result.stderr] if result.stderr else []
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "errors": ["Test execution timed out"]
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "errors": [str(e)]
            }


