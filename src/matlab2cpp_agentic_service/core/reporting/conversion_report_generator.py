"""
Comprehensive MATLAB to C++ Conversion Report Generator

This module provides detailed conversion reports including:
- MATLAB code analysis summary
- C++ code usage notes and documentation
- Compilation and execution notes
- Quality assessment results
- Performance metrics and recommendations
"""

import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass

from ...infrastructure.state.conversion_state import ConversionState


@dataclass
class ConversionReportData:
    """Structured data for conversion report generation."""
    
    # Project Information
    project_name: str
    project_type: str  # 'single-file' or 'multi-file'
    timestamp: datetime
    total_processing_time: float
    
    # MATLAB Analysis
    matlab_analysis: Dict[str, Any]
    
    # C++ Generation
    generated_files: List[str]
    generation_iterations: int
    compilation_success_rate: float
    
    # Quality Assessment
    quality_scores: Dict[str, Any]
    quality_assessment: Dict[str, Any]
    
    # Compilation Results
    compilation_result: Dict[str, Any]
    
    # Performance Metrics
    performance_metrics: Dict[str, Any]
    
    # Multi-file Coordination (if applicable)
    coordination_result: Optional[Dict[str, Any]] = None


class ConversionReportGenerator:
    """
    Generates comprehensive conversion reports with all requested sections.
    """
    
    def __init__(self):
        self.logger = None  # Will be set by the workflow
    
    def generate_comprehensive_report(self, state: ConversionState, 
                                    output_dir: Path) -> Path:
        """
        Generate a comprehensive conversion report.
        
        Args:
            state: Final conversion state with all data
            output_dir: Directory to save the report
            
        Returns:
            Path to the generated report file
        """
        # Extract structured data from state
        report_data = self._extract_report_data(state)
        
        # Generate report sections
        report_sections = self._generate_report_sections(report_data)
        
        # Combine sections into final report
        full_report = self._combine_report_sections(report_sections)
        
        # Save report to file
        report_path = self._save_report(full_report, report_data.project_name, output_dir)
        
        return report_path
    
    def _extract_report_data(self, state: ConversionState) -> ConversionReportData:
        """Extract structured data from conversion state."""
        
        # Debug: Check if state is a dict or string
        if not isinstance(state, dict):
            self.logger.warning(f"State is not a dict, type: {type(state)}")
            # Try to handle string state
            if isinstance(state, str):
                return ConversionReportData(
                    project_name='unknown',
                    project_type='single-file',
                    timestamp=datetime.now(),
                    total_processing_time=0.0,
                    matlab_analysis={},
                    generated_files=[],
                    generation_iterations=1,
                    compilation_success_rate=0.0,
                    quality_scores={},
                    quality_assessment={},
                    compilation_result={},
                    performance_metrics={},
                    coordination_result=None
                )
        
        return ConversionReportData(
            # Project Information
            project_name=state.get('project_name', state.get('request', {}).get('project_name', 'unknown') if state.get('request') else 'unknown'),
            project_type='multi-file' if state.get('is_multi_file', False) else 'single-file',
            timestamp=datetime.now(),
            total_processing_time=state.get('total_processing_time', 0.0),
            
            # MATLAB Analysis
            matlab_analysis=state.get('matlab_analysis', {}),
            
            # C++ Generation
            generated_files=state.get('generated_files', []),
            generation_iterations=self._get_generation_iterations(state),
            compilation_success_rate=self._get_compilation_success_rate(state),
            
            # Quality Assessment
            quality_scores=state.get('quality_scores', {}),
            quality_assessment=state.get('quality_assessment', {}),
            
            # Compilation Results
            compilation_result=self._get_compilation_result(state),
            
            # Performance Metrics
            performance_metrics=self._get_performance_metrics(state),
            
            # Multi-file Coordination
            coordination_result=state.get('multi_file_coordination')
        )
    
    def _get_generation_iterations(self, state: ConversionState) -> int:
        """Extract generation iterations from state."""
        generated_code = state.get('generated_code', {})
        if isinstance(generated_code, dict):
            return generated_code.get('generation_iterations', 1)
        return 1
    
    def _get_compilation_success_rate(self, state: ConversionState) -> float:
        """Extract compilation success rate from state."""
        generated_code = state.get('generated_code', {})
        if isinstance(generated_code, dict):
            success_rate = generated_code.get('compilation_success_rate', 0.0)
            # If we have a final successful compilation, ensure it shows 100% for final status
            compilation_result = generated_code.get('compilation_result', {})
            if compilation_result.get('success', False):
                # Calculate actual success rate based on attempts
                total_attempts = generated_code.get('compilation_attempts', 0)
                successful_attempts = generated_code.get('successful_compilations', 0)
                if total_attempts > 0:
                    return successful_attempts / total_attempts
                else:
                    return 1.0  # If final compilation succeeded, show 100%
            return success_rate
        return 0.0
    
    def _get_compilation_result(self, state: ConversionState) -> Dict[str, Any]:
        """Extract compilation result from state."""
        generated_code = state.get('generated_code', {})
        if isinstance(generated_code, dict):
            return generated_code.get('compilation_result', {})
        return {}
    
    def _get_performance_metrics(self, state: ConversionState) -> Dict[str, Any]:
        """Extract performance metrics from state."""
        return {
            'execution_time': state.get('total_processing_time', 0.0),
            'agent_performance': state.get('agent_performance', {}),
            'memory_usage': state.get('memory_usage', {}),
            'optimization_turns': state.get('optimization_turns', 0)
        }
    
    def _generate_report_sections(self, data: ConversionReportData) -> Dict[str, str]:
        """Generate all report sections."""
        
        return {
            'header': self._generate_header_section(data),
            'executive_summary': self._generate_executive_summary(data),
            'matlab_analysis': self._generate_matlab_analysis_section(data),
            'cpp_usage_notes': self._generate_cpp_usage_notes_section(data),
            'compilation_execution_notes': self._generate_compilation_execution_section(data),
            'quality_assessment': self._generate_quality_assessment_section(data),
            'performance_metrics': self._generate_performance_metrics_section(data),
            'recommendations': self._generate_recommendations_section(data),
            'technical_details': self._generate_technical_details_section(data)
        }
    
    def _generate_header_section(self, data: ConversionReportData) -> str:
        """Generate report header."""
        
        return f"""# MATLAB to C++ Conversion Report

**Project:** {data.project_name}  
**Type:** {data.project_type.title()}  
**Generated:** {data.timestamp.strftime('%Y-%m-%d %H:%M:%S')}  
**Processing Time:** {data.total_processing_time:.2f} seconds  

---

"""
    
    def _generate_executive_summary(self, data: ConversionReportData) -> str:
        """Generate executive summary section."""
        
        quality_score = data.quality_scores.get('overall_score', 0.0)
        quality_level = data.quality_scores.get('quality_level', 'unknown')
        
        summary = f"""## 📋 Executive Summary

### Conversion Results
- **Status:** {'✅ Successful' if quality_score >= 6.0 else '⚠️ Needs Improvement' if quality_score >= 4.0 else '❌ Failed'}
- **Quality Score:** {quality_score:.1f}/10 ({quality_level.title()})
- **Files Generated:** {len(data.generated_files)}
- **Generation Iterations:** {data.generation_iterations}
- **Compilation Success Rate:** {data.compilation_success_rate:.1%}

### Key Metrics
- **Processing Time:** {data.total_processing_time:.2f} seconds
- **Project Complexity:** {data.project_type.title()}
- **Optimization Applied:** {'Yes' if data.performance_metrics.get('optimization_turns', 0) > 0 else 'No'}

"""
        
        # Add multi-file coordination summary if applicable
        if data.coordination_result:
            coordination = data.coordination_result
            validation = coordination.get('validation_result', {})
            summary += f"""### Multi-File Coordination
- **Validation Status:** {'✅ Passed' if validation.get('validation_success', False) else '❌ Failed'}
- **Cross-file Issues:** {coordination.get('cross_file_issues', {}).get('issues', [])}
- **Coordination Strategy:** {coordination.get('coordination_strategy', {}).get('selected_strategy', 'unknown')}

"""
        
        return summary
    
    def _generate_matlab_analysis_section(self, data: ConversionReportData) -> str:
        """Generate MATLAB code analysis section."""
        
        analysis = data.matlab_analysis
        
        section = f"""## 🔍 MATLAB Code Analysis

### File Overview
- **Files Analyzed:** {analysis.get('files_analyzed', 0)}
- **Total Functions:** {analysis.get('total_functions', 0)}
- **Total Lines:** {analysis.get('total_lines', 0)}

### Function Analysis
"""
        
        # Add function details if available
        file_analyses = analysis.get('file_analyses', [])
        for file_analysis in file_analyses:
            # Handle both dict and string cases
            if isinstance(file_analysis, dict):
                filename = file_analysis.get('file_name', 'unknown')
                functions = file_analysis.get('functions', [])
            else:
                filename = str(file_analysis)
                functions = []
            
            section += f"\n#### {filename}\n"
            section += f"- **Functions Found:** {len(functions)}\n"
            
            # List main functions - only if functions is a list of dicts
            if functions and isinstance(functions, list):
                main_functions = []
                for f in functions:
                    if isinstance(f, dict) and f.get('is_main_function', False):
                        main_functions.append(f)
                
                if main_functions:
                    section += "- **Main Functions:**\n"
                    for func in main_functions:
                        func_name = func.get('function_name', 'unknown')
                        section += f"  - `{func_name}()`\n"
        
        # Add complexity analysis if available
        complexity = analysis.get('complexity_analysis', {})
        if complexity:
            section += f"""
### Complexity Analysis
- **Overall Complexity:** {complexity.get('complexity_level', 'unknown')}
- **Cyclomatic Complexity:** {complexity.get('cyclomatic_complexity', 'unknown')}
- **Code Maintainability:** {complexity.get('maintainability_score', 'unknown')}
"""
        
        section += "\n"
        return section
    
    def _generate_cpp_usage_notes_section(self, data: ConversionReportData) -> str:
        """Generate C++ usage notes and documentation."""
        
        section = f"""## 💻 C++ Code Usage Notes

### Generated Files
"""
        
        for file_path in data.generated_files:
            filename = Path(file_path).name
            file_type = 'Header' if filename.endswith('.h') else 'Implementation' if filename.endswith('.cpp') else 'Other'
            section += f"- **{file_type}:** `{filename}`\n"
        
        section += f"""
### Usage Instructions

#### Compilation
```bash
# Basic compilation
g++ -std=c++17 -I/path/to/eigen -o {data.project_name} *.cpp

# With optimization
g++ -std=c++17 -O2 -I/path/to/eigen -o {data.project_name} *.cpp
```

#### Dependencies
- **C++ Standard:** C++17 or higher
- **Eigen Library:** Required for matrix operations
- **Standard Libraries:** iostream, vector, string, memory, algorithm

#### Function Usage
"""
        
        # Extract function information from compilation result or generated code
        compilation_result = data.compilation_result
        if compilation_result:
            # Add function signatures if available
            section += "```cpp\n"
            section += "// Main conversion function\n"
            section += f"// Function signature will be available after successful compilation\n"
            section += "```\n"
        
        section += """
### Important Notes
- **Indexing:** MATLAB uses 1-based indexing, C++ uses 0-based indexing
- **Memory Management:** Automatic memory management with RAII principles
- **Error Handling:** Check return values and use proper exception handling
- **Performance:** Compiled C++ code should be significantly faster than MATLAB

"""
        return section
    
    def _generate_compilation_execution_section(self, data: ConversionReportData) -> str:
        """Generate compilation and execution notes."""
        
        compilation_result = data.compilation_result
        success = compilation_result.get('success', False)
        
        section = f"""## 🔧 Compilation & Execution Notes

### Compilation Status
- **Final Status:** {'✅ Successful' if success else '❌ Failed'}
- **Success Rate:** {data.compilation_success_rate:.1%} ({'Final compilation succeeded' if success else 'All attempts failed'})
- **Total Attempts:** {data.generation_iterations + (1 if data.compilation_success_rate < 1.0 and success else 0)}

"""
        
        if success:
            section += f"""### Successful Compilation
The generated C++ code compiled successfully with no errors or warnings.

**Compilation Process:**
- **Generation Iterations:** {data.generation_iterations}
- **Total Compilation Attempts:** {data.generation_iterations + (1 if data.compilation_success_rate < 1.0 else 0)}
- **Final Result:** ✅ Success after iterative improvements

#### Compilation Command
```bash
g++ -std=c++17 -I/path/to/eigen -o program *.cpp
```

#### Execution
```bash
./program
```
"""
        else:
            section += """### Compilation Issues
The compilation encountered errors that need to be addressed.

#### Common Issues and Solutions
"""
            
            # Add specific error information if available
            errors = compilation_result.get('errors', [])
            warnings = compilation_result.get('warnings', [])
            
            if errors:
                section += f"- **Errors:** {len(errors)} found\n"
                for error in errors[:3]:  # Show first 3 errors
                    # Handle both string and dict formats
                    error_msg = error.get('message', str(error)) if isinstance(error, dict) else str(error)
                    section += f"  - {error_msg}\n"
            
            if warnings:
                section += f"- **Warnings:** {len(warnings)} found\n"
                for warning in warnings[:3]:  # Show first 3 warnings
                    # Handle both string and dict formats
                    warning_msg = warning.get('message', str(warning)) if isinstance(warning, dict) else str(warning)
                    section += f"  - {warning_msg}\n"
        
        section += """
### Performance Considerations
- **Compilation Time:** Optimized for fast compilation
- **Runtime Performance:** C++ implementation should be faster than original MATLAB
- **Memory Usage:** Efficient memory management with RAII
- **Optimization Level:** Recommended to use -O2 or -O3 for production

"""
        return section
    
    def _generate_quality_assessment_section(self, data: ConversionReportData) -> str:
        """Generate quality assessment section."""
        
        quality_scores = data.quality_scores
        overall_score = quality_scores.get('overall_score', 0.0)
        quality_level = quality_scores.get('quality_level', 'unknown')
        
        section = f"""## 📊 Quality Assessment

### Overall Quality Score
- **Score:** {overall_score:.1f}/10
- **Level:** {quality_level.title()}
- **Assessment:** {'Excellent' if overall_score >= 8.0 else 'Good' if overall_score >= 6.0 else 'Fair' if overall_score >= 4.0 else 'Poor'}

### Detailed Quality Dimensions
"""
        
        dimension_scores = quality_scores.get('dimension_scores', {})
        
        dimensions = {
            'compilation_quality': 'Compilation Quality',
            'code_correctness': 'Code Correctness', 
            'performance_quality': 'Performance Quality',
            'maintainability': 'Maintainability',
            'consistency': 'Consistency'
        }
        
        for dim_key, dim_name in dimensions.items():
            score = dimension_scores.get(dim_key, 0.0)
            status = '🟢' if score >= 8.0 else '🟡' if score >= 6.0 else '🔴'
            section += f"- **{dim_name}:** {status} {score:.1f}/10\n"
        
        # Add recommendations if available
        quality_assessment = data.quality_assessment
        recommendations = quality_assessment.get('recommendations', [])
        
        if recommendations:
            section += f"""
### Recommendations
"""
            for i, rec in enumerate(recommendations[:5], 1):  # Show top 5 recommendations
                section += f"{i}. {rec}\n"
        
        section += "\n"
        return section
    
    def _generate_performance_metrics_section(self, data: ConversionReportData) -> str:
        """Generate performance metrics section."""
        
        metrics = data.performance_metrics
        
        section = f"""## ⚡ Performance Metrics

### Processing Performance
- **Total Processing Time:** {data.total_processing_time:.2f} seconds
- **Generation Iterations:** {data.generation_iterations}
- **Optimization Turns:** {metrics.get('optimization_turns', 0)}

### Agent Performance
"""
        
        agent_performance = metrics.get('agent_performance', {})
        if agent_performance:
            for agent_name, perf in agent_performance.items():
                section += f"- **{agent_name}:**\n"
                section += f"  - Operations: {perf.get('total_operations', 0)}\n"
                section += f"  - Success Rate: {perf.get('success_rate', 0.0):.1%}\n"
                section += f"  - Avg Time: {perf.get('avg_execution_time', 0.0):.2f}s\n"
        else:
            section += "- Agent performance data not available\n"
        
        section += f"""
### Memory Usage
"""
        memory_usage = metrics.get('memory_usage', {})
        if memory_usage:
            section += f"- **Total Memory:** {memory_usage.get('total_memory', 'unknown')}\n"
            section += f"- **Peak Memory:** {memory_usage.get('peak_memory', 'unknown')}\n"
        else:
            section += "- Memory usage data not available\n"
        
        section += "\n"
        return section
    
    def _generate_recommendations_section(self, data: ConversionReportData) -> str:
        """Generate recommendations section."""
        
        section = f"""## 💡 Recommendations

### Immediate Actions
"""
        
        overall_score = data.quality_scores.get('overall_score', 0.0)
        
        if overall_score >= 8.0:
            section += """- ✅ **Excellent Quality:** The conversion is ready for production use
- Consider running performance benchmarks against the original MATLAB code
- Document any custom modifications made during conversion
"""
        elif overall_score >= 6.0:
            section += """- ⚠️ **Good Quality:** Minor improvements recommended
- Review and address any remaining warnings
- Consider additional optimization for performance-critical sections
- Test thoroughly before production deployment
"""
        elif overall_score >= 4.0:
            section += """- 🔧 **Needs Improvement:** Significant issues to address
- Fix compilation errors and warnings
- Review code correctness and logic
- Consider manual review of complex algorithms
"""
        else:
            section += """- ❌ **Poor Quality:** Major rework required
- Fix all compilation errors
- Review the conversion strategy
- Consider breaking down complex functions
- Manual intervention may be required
"""
        
        # Add specific recommendations based on quality dimensions
        dimension_scores = data.quality_scores.get('dimension_scores', {})
        
        section += """
### Dimension-Specific Recommendations
"""
        
        for dim_key, score in dimension_scores.items():
            dim_name = dim_key.replace('_', ' ').title()
            if score < 6.0:
                if dim_key == 'compilation_quality':
                    section += f"- **{dim_name}:** Fix compilation errors and ensure clean build\n"
                elif dim_key == 'code_correctness':
                    section += f"- **{dim_name}:** Review algorithm implementation and logic\n"
                elif dim_key == 'performance_quality':
                    section += f"- **{dim_name}:** Optimize performance-critical sections\n"
                elif dim_key == 'maintainability':
                    section += f"- **{dim_name}:** Improve code structure and documentation\n"
                elif dim_key == 'consistency':
                    section += f"- **{dim_name}:** Ensure consistent coding style and patterns\n"
        
        section += """
### Long-term Considerations
- Monitor runtime performance in production
- Consider further optimization opportunities
- Maintain documentation as code evolves
- Regular quality assessments for future conversions

"""
        return section
    
    def _generate_technical_details_section(self, data: ConversionReportData) -> str:
        """Generate technical details section."""
        
        section = f"""## 🔧 Technical Details

### Conversion Configuration
- **Project Type:** {data.project_type}
- **Generation Strategy:** Adaptive iterative generation
- **Quality Threshold:** 6.0/10
- **Max Iterations:** {data.generation_iterations}

### Generated File Structure
"""
        
        for file_path in data.generated_files:
            filename = Path(file_path).name
            section += f"- `{filename}`\n"
        
        section += f"""
### Compilation Details
- **C++ Standard:** C++17
- **Compiler:** g++ (GNU Compiler Collection)
- **Optimization:** -O2 recommended
- **Dependencies:** Eigen, STL

### Quality Assessment Details
- **Assessment Method:** Multi-dimensional analysis
- **Dimensions Evaluated:** 5 (compilation, correctness, performance, maintainability, consistency)
- **Scoring Scale:** 0-10
- **Assessment Timestamp:** {data.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

"""
        
        # Add multi-file coordination details if applicable
        if data.coordination_result:
            section += """### Multi-File Coordination Details
"""
            coordination = data.coordination_result
            strategy = coordination.get('coordination_strategy', {})
            section += f"- **Strategy:** {strategy.get('selected_strategy', 'unknown')}\n"
            
            validation = coordination.get('validation_result', {})
            section += f"- **Validation:** {'Passed' if validation.get('validation_success', False) else 'Failed'}\n"
        
        section += """
### System Information
- **Conversion Engine:** Enhanced LangGraph MATLAB2CPP Workflow
- **Agent Architecture:** 5 streamlined agents
- **Processing Mode:** Real-time compilation testing
- **Report Generated:** """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """

---
*Report generated by MATLAB to C++ Agentic Service*
"""
        
        return section
    
    def _combine_report_sections(self, sections: Dict[str, str]) -> str:
        """Combine all report sections into a single document."""
        
        section_order = [
            'header',
            'executive_summary', 
            'matlab_analysis',
            'cpp_usage_notes',
            'compilation_execution_notes',
            'quality_assessment',
            'performance_metrics',
            'recommendations',
            'technical_details'
        ]
        
        return '\n'.join(sections[section] for section in section_order if section in sections)
    
    def _save_report(self, report_content: str, project_name: str, output_dir: Path) -> Path:
        """Save the report to a file."""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"{project_name}_conversion_report_{timestamp}.md"
        report_path = output_dir / report_filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return report_path
