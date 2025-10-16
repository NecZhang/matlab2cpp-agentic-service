"""
C++ compilation manager for testing framework.
"""

import os
import tempfile
from typing import Dict, List, Optional
from ...utils.logger import get_logger
from .types import CompilationResult, ProjectFiles
from .docker_manager import DockerTestingManager

logger = get_logger(__name__)


class CPPCompilationManager:
    """Manages C++ compilation testing."""
    
    def __init__(self, docker_manager: Optional[DockerTestingManager] = None, build_system: str = 'gcc'):
        """
        Initialize compilation manager.
        
        Args:
            docker_manager: Optional Docker manager instance
            build_system: Build system to use ('gcc' or 'cmake'). Default: 'gcc'
        """
        self.docker_manager = docker_manager or DockerTestingManager(build_system=build_system)
        self.build_system = build_system
        self.logger = logger
        self.logger.info(f"🔧 Compilation manager initialized with build_system={build_system}")
    
    def compile_project(self, 
                       project_files: Dict[str, str],
                       project_name: str,
                       timeout: int = 300) -> CompilationResult:
        """Compile a C++ project."""
        
        self.logger.info(f"Starting compilation for project: {project_name}")
        
        # Validate project files
        validation_result = self._validate_project_files(project_files)
        if not validation_result['valid']:
            return CompilationResult(
                success=False,
                output="",
                errors=validation_result['errors'],
                warnings=validation_result['warnings']
            )
        
        # Run compilation in Docker
        compilation_result = self.docker_manager.run_compilation_test(
            project_files, project_name, timeout
        )
        
        self.logger.info(f"Compilation {'successful' if compilation_result.success else 'failed'}")
        return compilation_result
    
    def _validate_project_files(self, project_files: Dict[str, str]) -> Dict[str, any]:
        """Validate project files before compilation."""
        errors = []
        warnings = []
        
        # Check for at least one C++ source file
        cpp_files = [f for f in project_files.keys() if f.endswith('.cpp')]
        if not cpp_files:
            errors.append("No C++ source files (.cpp) found")
        
        # Check for main function
        has_main = False
        for filename, content in project_files.items():
            if filename.endswith('.cpp') and 'int main(' in content:
                has_main = True
                break
        
        if not has_main:
            warnings.append("No main function found - may not be executable")
        
        # Check for common issues
        for filename, content in project_files.items():
            if filename.endswith('.cpp'):
                # Check for missing includes
                if '#include' not in content:
                    warnings.append(f"File {filename} has no include statements")
                
                # Check for potential issues
                if 'using namespace std;' in content:
                    warnings.append(f"File {filename} uses 'using namespace std' - consider avoiding")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def generate_makefile(self, 
                        project_files: Dict[str, str],
                        project_name: str) -> str:
        """Generate Makefile for the project."""
        
        cpp_files = [f for f in project_files.keys() if f.endswith('.cpp')]
        
        makefile_content = f"""# Makefile for {project_name}
CXX = g++-13
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -I/usr/include/eigen3 -I/usr/include/opencv4
LDFLAGS = -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lgomp

TARGET = {project_name}
SOURCES = {' '.join(cpp_files)}

$(TARGET): $(SOURCES)
\t$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

clean:
\trm -f $(TARGET)

.PHONY: clean
"""
        return makefile_content
    
    def extract_dependencies(self, project_files: Dict[str, str]) -> List[str]:
        """Extract dependencies from project files."""
        dependencies = set()
        
        for filename, content in project_files.items():
            if filename.endswith(('.cpp', '.h', '.hpp')):
                # Extract include statements
                lines = content.split('\n')
                for line in lines:
                    if line.strip().startswith('#include'):
                        include = line.strip()
                        if '<eigen' in include:
                            dependencies.add('eigen3')
                        elif '<opencv' in include:
                            dependencies.add('opencv4')
                        elif '<png' in include:
                            dependencies.add('libpng')
                        elif '<omp.h>' in include:
                            dependencies.add('libomp')
        
        return list(dependencies)
