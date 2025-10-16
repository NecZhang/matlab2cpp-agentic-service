"""
Enhanced Quality Assessor Agent

This agent provides advanced quality assessment with:
- Real compilation results integration
- Multi-file project quality assessment
- Cross-file consistency checking
- Performance and maintainability analysis
- Strategy recommendation based on quality metrics
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Set, Tuple, Callable
from pathlib import Path
import re

from ..base.langgraph_agent import BaseLangGraphAgent, AgentConfig
from ....infrastructure.tools.llm_client import LLMClient
from ....infrastructure.state.conversion_state import ConversionState
from ....infrastructure.testing.quality_assessor import CPPQualityAssessor
from ....infrastructure.testing.enhanced_quality_assessor import EnhancedQualityAssessor


class QualityAssessor(BaseLangGraphAgent):
    """
    Quality assessor with real compilation results and multi-file support.
    
    Capabilities:
    - Real compilation results integration
    - Multi-file project quality assessment
    - Cross-file consistency checking
    - Performance and maintainability analysis
    - Strategy recommendation based on quality metrics
    """
    
    def __init__(self, config: AgentConfig, llm_client: LLMClient):
        super().__init__(config, llm_client)
        
        # Enhanced tools
        self.tools = [
            "quality_assessment",
            "compilation_result_analyzer",
            "consistency_checker",
            "performance_analyzer",
            "llm_analysis"
        ]
        
        # Initialize quality assessors
        self.basic_assessor = CPPQualityAssessor()
        self.enhanced_assessor = None  # Will be initialized when needed
        
        # Quality metrics
        self.quality_dimensions = self._initialize_quality_dimensions()
        self.quality_thresholds = self._initialize_quality_thresholds()
        
        self.logger.info(f"Initialized Quality Assessor: {config.name}")
    
    def create_node(self) -> Callable[[ConversionState], ConversionState]:
        """Create the LangGraph node function for quality assessment."""
        async def assess_node(state: ConversionState) -> ConversionState:
            return await self.assess_code_quality(state)
        return assess_node
    
    def get_tools(self) -> List[Any]:
        """Get available tools for quality assessment."""
        return [
            self.basic_assessor,
            self.enhanced_assessor,
        ]
    
    def _initialize_quality_dimensions(self) -> Dict[str, Dict[str, Any]]:
        """Initialize quality assessment dimensions."""
        return {
            "compilation_quality": {
                "description": "Quality based on compilation success",
                "weight": 0.3,
                "metrics": ["compilation_success", "error_count", "warning_count"]
            },
            "code_correctness": {
                "description": "Correctness of generated code",
                "weight": 0.25,
                "metrics": ["syntax_correctness", "type_correctness", "logic_correctness"]
            },
            "performance_quality": {
                "description": "Performance characteristics",
                "weight": 0.2,
                "metrics": ["execution_time", "memory_usage", "algorithm_efficiency"]
            },
            "maintainability": {
                "description": "Code maintainability",
                "weight": 0.15,
                "metrics": ["code_readability", "documentation", "structure_quality"]
            },
            "consistency": {
                "description": "Consistency across files",
                "weight": 0.1,
                "metrics": ["naming_consistency", "style_consistency", "interface_consistency"]
            },
            "actual_code_quality": {
                "description": "Actual validation of generated code files",
                "weight": 0.15,
                "metrics": ["syntax_validation", "file_structure", "content_quality"]
            }
        }
    
    def _initialize_quality_thresholds(self) -> Dict[str, float]:
        """Initialize quality thresholds for different levels."""
        return {
            "excellent": 0.9,
            "good": 0.8,
            "acceptable": 0.7,
            "poor": 0.6,
            "unacceptable": 0.0
        }
    
    async def assess_with_compilation_results(self, generated_code: Dict[str, Any],
                                            compilation_result: Dict[str, Any],
                                            matlab_analysis: Dict[str, Any],
                                            conversion_plan: Dict[str, Any],
                                            state: ConversionState) -> ConversionState:
        """
        Assess quality using real compilation results and comprehensive analysis.
        
        Args:
            generated_code: Generated C++ code
            compilation_result: Compilation test results
            matlab_analysis: MATLAB analysis results
            conversion_plan: Conversion plan
            state: Current conversion state
            
        Returns:
            Updated state with quality assessment
        """
        start_time = time.time()
        
        try:
            self.logger.info("Starting enhanced quality assessment...")
            
            # Phase 1: Assess compilation quality
            compilation_quality = self._assess_compilation_quality(compilation_result)
            
            # Phase 2: Assess code correctness
            code_correctness = await self._assess_code_correctness(generated_code, matlab_analysis)
            
            # Phase 3: Assess performance quality
            performance_quality = self._assess_performance_quality(compilation_result)
            
            # Phase 4: Assess maintainability
            maintainability = self._assess_maintainability(generated_code, conversion_plan)
            
            # Phase 5: Assess consistency (for multi-file projects)
            consistency = self._assess_consistency(generated_code, conversion_plan)
            
            # Phase 6: Calculate overall quality score
            dimension_scores = {
                'compilation_quality': compilation_quality,
                'code_correctness': code_correctness,
                'performance_quality': performance_quality,
                'maintainability': maintainability,
                'consistency': consistency
            }
            
            # Add actual code quality validation
            actual_code_quality = self._validate_actual_code_quality(generated_code)
            dimension_scores['actual_code_quality'] = actual_code_quality
            
            self.logger.info(f"DEBUG: Dimension scores before calculation: {[(k, v.get('score', 'no score')) for k, v in dimension_scores.items()]}")
            overall_quality = self._calculate_overall_quality(dimension_scores)
            self.logger.info(f"DEBUG: Overall quality calculated: {overall_quality}")
            
            # Phase 7: Generate improvement recommendations
            recommendations = self._generate_improvement_recommendations({
                'compilation_quality': compilation_quality,
                'code_correctness': code_correctness,
                'performance_quality': performance_quality,
                'maintainability': maintainability,
                'consistency': consistency
            }, overall_quality)
            
            # Create comprehensive assessment result
            assessment_result = {
                'overall_quality_score': overall_quality['score'],
                'quality_level': overall_quality['level'],
                'dimension_scores': {
                    'compilation_quality': compilation_quality,
                    'code_correctness': code_correctness,
                    'performance_quality': performance_quality,
                    'maintainability': maintainability,
                    'consistency': consistency
                },
                'recommendations': recommendations,
                'assessment_timestamp': time.time(),
                'compilation_result': compilation_result
            }
            
            # Convert scores from 0-1 scale to 0-10 scale
            overall_score_10 = overall_quality['score'] * 10
            
            # Update state with both formats for compatibility
            state["quality_assessment"] = assessment_result
            state["quality_assessment"]["overall_quality_score_10"] = overall_score_10
            state["quality_scores"] = {
                "overall_score": overall_score_10,  # Now on 0-10 scale
                "quality_level": overall_quality['level'],
                "dimension_scores": {
                    'compilation_quality': compilation_quality['score'] * 10,
                    'code_correctness': code_correctness['score'] * 10,
                    'performance_quality': performance_quality['score'] * 10,
                    'maintainability': maintainability['score'] * 10,
                    'consistency': consistency['score'] * 10
                }
            }
            
            # Update memory
            self.update_memory("assessment_count", 
                             (self.get_memory("assessment_count", "short_term") or 0) + 1, 
                             "short_term")
            
            # Track performance
            execution_time = time.time() - start_time
            self.track_performance("assess_with_compilation_results", start_time, time.time(), True, 
                                 {"quality_score": overall_quality['score']})
            
            self.logger.info(f"Enhanced quality assessment complete: "
                           f"score={overall_score_10:.1f}/10, "
                           f"level={overall_quality['level']}, "
                           f"{execution_time:.2f}s")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Enhanced quality assessment failed: {e}")
            self.track_performance("assess_with_compilation_results", start_time, time.time(), False, 
                                 {"error": str(e)})
            raise
    
    def _assess_compilation_quality(self, compilation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Assess quality based on compilation results and actual code quality."""
        self.logger.info(f"DEBUG: Assessing compilation quality with result: {compilation_result}")
        success = compilation_result.get('success', False)
        error_output = compilation_result.get('output', '')
        
        # Find all warnings and errors
        all_warnings = re.findall(r'.*warning:.*', error_output, re.IGNORECASE)
        all_errors = re.findall(r'.*error:.*', error_output, re.IGNORECASE)
        
        # ✅ FIXED: Filter out library/system warnings (only count OUR code warnings)
        library_paths = [
            '/usr/include/eigen3',
            '/usr/include/opencv',
            '/usr/include/c++',
            '/usr/local/include'
        ]
        
        our_code_warnings = [
            w for w in all_warnings 
            if not any(lib_path in w for lib_path in library_paths)
        ]
        
        warning_count = len(our_code_warnings)
        total_warning_count = len(all_warnings)
        error_count = len(all_errors)
        
        if total_warning_count > warning_count:
            self.logger.debug(f"Filtered {total_warning_count - warning_count} library warnings, counting {warning_count} project warnings")
        
        self.logger.info(f"DEBUG: Compilation success: {success}, OUR warnings: {warning_count}, errors: {error_count}")
        
        # Calculate compilation score
        if success:
            base_score = 1.0
            warning_penalty = min(warning_count * 0.05, 0.2)  # Max 20% penalty for OUR warnings only
            compilation_score = base_score - warning_penalty
        else:
            # Penalize based on number of errors
            error_penalty = min(error_count * 0.1, 0.8)  # Max 80% penalty for errors
            compilation_score = max(0.2 - error_penalty, 0.0)  # Minimum 20% score
        
        return {
            'compilation_success': success,
            'compilation_score': compilation_score,
            'warning_count': warning_count,
            'error_count': error_count,
            'quality_notes': f"Compilation: {'Success' if success else 'Failed'} ({error_count} errors, {warning_count} warnings)",
            'score': compilation_score,
            'success': success,
            'details': {
                'compilation_output': compilation_result.get('output', ''),
                'error_output': error_output
            }
        }
    
    def _validate_actual_code_quality(self, generated_code: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the actual quality of generated code files."""
        try:
            files = generated_code.get('files', {})
            if not files:
                return {'score': 0.0, 'issues': ['No files generated']}
            
            total_score = 0.0
            total_files = 0
            all_issues = []
            
            for filename, content in files.items():
                if not content or content.strip() == '':
                    all_issues.append(f"{filename}: Empty file")
                    continue
                
                file_score = 1.0
                file_issues = []
                
                # Check for basic syntax issues
                if filename.endswith('.h'):
                    # Header file validation
                    if not content.strip().startswith('#ifndef'):
                        file_issues.append(f"{filename}: Missing header guard")
                        file_score -= 0.3
                    
                    if ');' in content and content.count(');') > content.count('('):
                        file_issues.append(f"{filename}: Malformed syntax (orphaned ');')")
                        file_score -= 0.5
                    
                    if 'namespace' in content and not content.strip().endswith('}'):
                        file_issues.append(f"{filename}: Incomplete namespace")
                        file_score -= 0.2
                
                elif filename.endswith('.cpp'):
                    # Implementation file validation
                    if '#include' not in content:
                        file_issues.append(f"{filename}: Missing includes")
                        file_score -= 0.2
                    
                    # Check if implementation contains header content (common parsing error)
                    if '#ifndef' in content and '#endif' in content:
                        file_issues.append(f"{filename}: Contains header content instead of implementation")
                        file_score -= 0.7
                    
                    if content.strip() == '':
                        file_issues.append(f"{filename}: Empty implementation")
                        file_score -= 0.8
                
                # Check for common malformed patterns
                if ');' == content.strip():
                    file_issues.append(f"{filename}: Only contains ');'")
                    file_score = 0.0
                
                total_score += max(0.0, file_score)
                total_files += 1
                all_issues.extend(file_issues)
            
            average_score = total_score / total_files if total_files > 0 else 0.0
            
            return {
                'score': average_score,
                'issues': all_issues,
                'file_count': total_files,
                'quality_notes': f"Code quality: {len(all_issues)} issues found in {total_files} files"
            }
            
        except Exception as e:
            self.logger.error(f"Error validating code quality: {e}")
            return {'score': 0.0, 'issues': [f'Validation error: {str(e)}']}
    
    async def _assess_code_correctness(self, generated_code: Dict[str, Any], 
                                     matlab_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess correctness of generated code."""
        files = generated_code.get('files', {})
        if not files:
            return {'score': 0.0, 'details': {'error': 'No files to assess'}}
        
        correctness_scores = []
        total_issues = 0
        
        for file_name, file_content in files.items():
            file_assessment = self._assess_file_correctness(file_content, file_name)
            correctness_scores.append(file_assessment['score'])
            total_issues += file_assessment['issue_count']
        
        # Calculate average correctness score
        avg_score = sum(correctness_scores) / len(correctness_scores) if correctness_scores else 0.0
        
        return {
            'score': avg_score,
            'file_scores': correctness_scores,
            'total_issues': total_issues,
            'details': {
                'files_assessed': len(files),
                'average_file_score': avg_score
            }
        }
    
    def _assess_file_correctness(self, file_content: str, file_name: str) -> Dict[str, Any]:
        """Assess correctness of a single file."""
        issues = []
        score = 1.0
        
        # Check for common syntax issues
        if file_content.count('{') != file_content.count('}'):
            issues.append('Mismatched braces')
            score -= 0.3
        
        if file_content.count('(') != file_content.count(')'):
            issues.append('Mismatched parentheses')
            score -= 0.3
        
        # Check for malformed includes
        include_lines = [line for line in file_content.split('\n') if '#include' in line]
        for line in include_lines:
            if line.strip().endswith('#include') or line.strip() == '#include':
                issues.append('Malformed include statement')
                score -= 0.2
                break
        
        # Check for template syntax issues
        if '<>' in file_content and 'template' not in file_content:
            issues.append('Potential template syntax issue')
            score -= 0.1
        
        # Check for namespace issues
        if '::' in file_content and 'namespace' not in file_content:
            issues.append('Potential namespace issue')
            score -= 0.1
        
        return {
            'score': max(score, 0.0),
            'issue_count': len(issues),
            'issues': issues,
            'file_name': file_name
        }
    
    def _assess_performance_quality(self, compilation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Assess performance characteristics."""
        # For now, use compilation time as a proxy for performance
        compilation_time = compilation_result.get('compilation_time', 0)
        execution_time = compilation_result.get('execution_time', 0)
        
        # Simple performance scoring
        if compilation_time > 30:  # More than 30 seconds
            compilation_score = 0.5
        elif compilation_time > 10:  # More than 10 seconds
            compilation_score = 0.8
        else:
            compilation_score = 1.0
        
        if execution_time > 5:  # More than 5 seconds
            execution_score = 0.5
        elif execution_time > 1:  # More than 1 second
            execution_score = 0.8
        else:
            execution_score = 1.0
        
        performance_score = (compilation_score + execution_score) / 2
        
        return {
            'score': performance_score,
            'compilation_time': compilation_time,
            'execution_time': execution_time,
            'details': {
                'compilation_score': compilation_score,
                'execution_score': execution_score
            }
        }
    
    def _assess_maintainability(self, generated_code: Dict[str, Any], 
                              conversion_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Assess code maintainability."""
        files = generated_code.get('files', {})
        if not files:
            return {'score': 0.0, 'details': {'error': 'No files to assess'}}
        
        maintainability_scores = []
        
        for file_name, file_content in files.items():
            file_maintainability = self._assess_file_maintainability(file_content, file_name)
            maintainability_scores.append(file_maintainability['score'])
        
        avg_score = sum(maintainability_scores) / len(maintainability_scores)
        
        return {
            'score': avg_score,
            'file_scores': maintainability_scores,
            'details': {
                'files_assessed': len(files),
                'average_file_score': avg_score
            }
        }
    
    def _assess_file_maintainability(self, file_content: str, file_name: str) -> Dict[str, Any]:
        """Assess maintainability of a single file."""
        score = 1.0
        
        # Check for documentation
        comment_lines = [line for line in file_content.split('\n') if line.strip().startswith('//')]
        total_lines = len([line for line in file_content.split('\n') if line.strip()])
        
        if total_lines > 0:
            comment_ratio = len(comment_lines) / total_lines
            if comment_ratio < 0.1:  # Less than 10% comments
                score -= 0.2
            elif comment_ratio < 0.05:  # Less than 5% comments
                score -= 0.4
        
        # Check for function length (long functions are harder to maintain)
        functions = re.findall(r'(?:^|\n)\s*(?:\w+\s+)*(\w+)\s*\([^)]*\)\s*\{', file_content)
        for func in functions:
            # Simple heuristic: count lines in function
            func_start = file_content.find(func)
            if func_start != -1:
                brace_start = file_content.find('{', func_start)
                if brace_start != -1:
                    brace_count = 1
                    pos = brace_start + 1
                    while pos < len(file_content) and brace_count > 0:
                        if file_content[pos] == '{':
                            brace_count += 1
                        elif file_content[pos] == '}':
                            brace_count -= 1
                        pos += 1
                    
                    func_content = file_content[brace_start:pos]
                    func_lines = len([line for line in func_content.split('\n') if line.strip()])
                    
                    if func_lines > 50:  # More than 50 lines
                        score -= 0.1
        
        # Check for variable naming consistency
        variables = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', file_content)
        camel_case_vars = sum(1 for var in variables if re.match(r'^[a-z][a-zA-Z0-9]*$', var))
        snake_case_vars = sum(1 for var in variables if re.match(r'^[a-z][a-z0-9_]*$', var))
        
        if len(variables) > 0:
            naming_consistency = max(camel_case_vars, snake_case_vars) / len(variables)
            if naming_consistency < 0.7:  # Less than 70% consistent
                score -= 0.1
        
        return {
            'score': max(score, 0.0),
            'file_name': file_name,
            'details': {
                'comment_ratio': comment_ratio if total_lines > 0 else 0,
                'naming_consistency': naming_consistency if len(variables) > 0 else 1.0
            }
        }
    
    def _assess_consistency(self, generated_code: Dict[str, Any], 
                          conversion_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Assess consistency across files."""
        files = generated_code.get('files', {})
        if len(files) <= 1:
            return {'score': 1.0, 'details': {'message': 'Single file - no consistency check needed'}}
        
        consistency_scores = []
        
        # Check naming consistency
        naming_consistency = self._check_naming_consistency(files)
        consistency_scores.append(naming_consistency)
        
        # Check style consistency
        style_consistency = self._check_style_consistency(files)
        consistency_scores.append(style_consistency)
        
        # Check interface consistency
        interface_consistency = self._check_interface_consistency(files)
        consistency_scores.append(interface_consistency)
        
        avg_score = sum(consistency_scores) / len(consistency_scores)
        
        return {
            'score': avg_score,
            'naming_consistency': naming_consistency,
            'style_consistency': style_consistency,
            'interface_consistency': interface_consistency,
            'details': {
                'files_checked': len(files),
                'average_consistency': avg_score
            }
        }
    
    def _check_naming_consistency(self, files: Dict[str, str]) -> float:
        """Check naming consistency across files."""
        all_variables = []
        for file_content in files.values():
            variables = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', file_content)
            all_variables.extend(variables)
        
        if not all_variables:
            return 1.0
        
        camel_case_count = sum(1 for var in all_variables if re.match(r'^[a-z][a-zA-Z0-9]*$', var))
        snake_case_count = sum(1 for var in all_variables if re.match(r'^[a-z][a-z0-9_]*$', var))
        
        max_style_count = max(camel_case_count, snake_case_count)
        return max_style_count / len(all_variables)
    
    def _check_style_consistency(self, files: Dict[str, str]) -> float:
        """Check style consistency across files."""
        styles = []
        
        for file_content in files.values():
            file_style = {
                'indentation': self._detect_indentation_style(file_content),
                'brace_style': self._detect_brace_style(file_content),
                'line_length': self._detect_line_length_style(file_content)
            }
            styles.append(file_style)
        
        # Compare styles across files
        if len(styles) <= 1:
            return 1.0
        
        consistency_score = 0.0
        for key in ['indentation', 'brace_style', 'line_length']:
            values = [style[key] for style in styles]
            if len(set(values)) == 1:  # All files have same style
                consistency_score += 1.0
            elif len(set(values)) <= 2:  # Most files have same style
                consistency_score += 0.7
        
        return consistency_score / 3  # 3 style aspects checked
    
    def _detect_indentation_style(self, content: str) -> str:
        """Detect indentation style (tabs vs spaces)."""
        lines = [line for line in content.split('\n') if line.strip()]
        tab_lines = sum(1 for line in lines if line.startswith('\t'))
        space_lines = sum(1 for line in lines if line.startswith(' '))
        
        if tab_lines > space_lines:
            return 'tabs'
        elif space_lines > tab_lines:
            return 'spaces'
        else:
            return 'mixed'
    
    def _detect_brace_style(self, content: str) -> str:
        """Detect brace style (K&R vs Allman)."""
        # Look for function definitions
        functions = re.findall(r'(?:^|\n)\s*(?:\w+\s+)*(\w+)\s*\([^)]*\)\s*(\{|\n\s*\{)', content)
        
        k_and_r_count = sum(1 for _, brace in functions if brace == '{')
        allman_count = sum(1 for _, brace in functions if brace == '\n')
        
        if k_and_r_count > allman_count:
            return 'k_and_r'
        elif allman_count > k_and_r_count:
            return 'allman'
        else:
            return 'mixed'
    
    def _detect_line_length_style(self, content: str) -> str:
        """Detect line length style."""
        lines = content.split('\n')
        long_lines = sum(1 for line in lines if len(line) > 80)
        
        if long_lines / len(lines) > 0.2:  # More than 20% long lines
            return 'long'
        else:
            return 'short'
    
    def _check_interface_consistency(self, files: Dict[str, str]) -> float:
        """Check interface consistency across files."""
        # This is a simplified check - in a full implementation,
        # it would check function signatures, class interfaces, etc.
        return 0.8  # Placeholder score
    
    def _calculate_overall_quality(self, dimension_scores: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall quality score."""
        total_score = 0.0
        total_weight = 0.0
        
        for dimension, info in self.quality_dimensions.items():
            weight = info['weight']
            score = dimension_scores[dimension]['score']
            
            total_score += score * weight
            total_weight += weight
        
        overall_score = total_score / total_weight if total_weight > 0 else 0.0
        
        # Determine quality level
        if overall_score >= self.quality_thresholds['excellent']:
            level = 'excellent'
        elif overall_score >= self.quality_thresholds['good']:
            level = 'good'
        elif overall_score >= self.quality_thresholds['acceptable']:
            level = 'acceptable'
        elif overall_score >= self.quality_thresholds['poor']:
            level = 'poor'
        else:
            level = 'unacceptable'
        
        return {
            'score': overall_score,
            'level': level,
            'dimension_breakdown': {
                dimension: {
                    'score': dimension_scores[dimension]['score'],
                    'weight': info['weight'],
                    'contribution': dimension_scores[dimension]['score'] * info['weight']
                }
                for dimension, info in self.quality_dimensions.items()
            }
        }
    
    def _generate_improvement_recommendations(self, dimension_scores: Dict[str, Dict[str, Any]], 
                                           overall_quality: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations based on quality scores."""
        recommendations = []
        
        for dimension, scores in dimension_scores.items():
            score = scores['score']
            threshold = self.quality_thresholds['acceptable']
            
            if score < threshold:
                if dimension == 'compilation_quality':
                    recommendations.append("Fix compilation errors and warnings")
                elif dimension == 'code_correctness':
                    recommendations.append("Review and fix syntax and logic errors")
                elif dimension == 'performance_quality':
                    recommendations.append("Optimize performance bottlenecks")
                elif dimension == 'maintainability':
                    recommendations.append("Improve code documentation and structure")
                elif dimension == 'consistency':
                    recommendations.append("Standardize naming and coding style")
        
        # Add general recommendations based on overall score
        overall_score = overall_quality['score']
        if overall_score < 0.7:
            recommendations.append("Consider regenerating code with improved strategy")
        elif overall_score < 0.8:
            recommendations.append("Minor improvements needed for production readiness")
        
        return recommendations
    
    async def get_assessment_summary(self, assessment_result: Dict[str, Any]) -> str:
        """Generate human-readable assessment summary."""
        summary = f"📊 Enhanced Quality Assessment Summary\n"
        
        overall_score = assessment_result['overall_quality_score']
        quality_level = assessment_result['quality_level']
        
        summary += f"Overall Quality Score: {overall_score:.2f} ({quality_level})\n"
        
        # Show dimension scores
        dimension_scores = assessment_result['dimension_scores']
        summary += "\nDimension Scores:\n"
        for dimension, scores in dimension_scores.items():
            summary += f"  - {dimension}: {scores['score']:.2f}\n"
        
        # Show recommendations
        recommendations = assessment_result['recommendations']
        if recommendations:
            summary += "\nRecommendations:\n"
            for i, rec in enumerate(recommendations, 1):
                summary += f"  {i}. {rec}\n"
        else:
            summary += "\n✅ No specific recommendations - code quality is good!\n"
        
        # Show compilation status
        compilation_result = assessment_result.get('compilation_result', {})
        if compilation_result:
            success = compilation_result.get('success', False)
            summary += f"\nCompilation Status: {'✅ Success' if success else '❌ Failed'}\n"
            
            if not success:
                error_count = compilation_result.get('error_count', 0)
                warning_count = compilation_result.get('warning_count', 0)
                summary += f"Errors: {error_count}, Warnings: {warning_count}\n"
        
        return summary
