"""
LangGraph-Native Quality Assessor Agent

This module implements a truly LangGraph-native quality assessor agent
that evaluates generated C++ code quality using LangGraph tools and memory.
"""

import time
from typing import Dict, Any, List, Callable
from pathlib import Path
from loguru import logger

from matlab2cpp_agentic_service.infrastructure.state.conversion_state import ConversionState, ConversionStatus, add_processing_time, update_state_status
from matlab2cpp_agentic_service.infrastructure.tools.llm_client import LLMClient
from matlab2cpp_agentic_service.infrastructure.tools.langgraph_tools import QualityAssessmentTool, LLMAnalysisTool, ToolRegistry
from ...base.langgraph_agent import BaseLangGraphAgent, AgentConfig


class LangGraphQualityAssessorAgent(BaseLangGraphAgent):
    """
    LangGraph-native quality assessor agent.
    
    This agent evaluates C++ code quality using LangGraph tools
    and maintains assessment history in memory for iterative improvements.
    """
    
    def __init__(self, config: AgentConfig, llm_client: LLMClient):
        """Initialize the LangGraph quality assessor agent."""
        super().__init__(config, llm_client)
        
        # Initialize tools
        self.tool_registry = ToolRegistry()
        self.quality_assessment_tool = QualityAssessmentTool(llm_client)
        self.llm_analysis_tool = LLMAnalysisTool(llm_client)
        
        # Register tools
        self.tool_registry.register_tool("quality_assessment", self.quality_assessment_tool)
        self.tool_registry.register_tool("llm_analysis", self.llm_analysis_tool)
        
        self.logger.info(f"Initialized LangGraph Quality Assessor Agent: {config.name}")
    
    def create_node(self) -> Callable[[ConversionState], ConversionState]:
        """
        Create the LangGraph node function for quality assessment.
        
        Returns:
            Callable that takes and returns ConversionState
        """
        def assess_quality_node(state: ConversionState) -> ConversionState:
            """LangGraph node function for quality assessment."""
            start_time = time.time()
            
            try:
                self.logger.info("Starting LangGraph-native quality assessment...")
                
                # Update state status
                state = update_state_status(state, ConversionStatus.ASSESSING)
                
                # Get required data from state
                generated_code = state.get("generated_code")
                matlab_analysis = state.get("matlab_analysis")
                conversion_plan = state.get("conversion_plan")
                
                if not generated_code:
                    raise ValueError("Generated code not found in state")
                
                # Get current turn for assessment tracking
                current_turn = state.get("current_turn", 0)
                is_optimization = current_turn > 0
                
                # Check if we have cached assessment in memory for this turn
                cache_key = f"quality_assessment_{current_turn}_{hash(str(generated_code))}"
                cached_assessment = self.get_memory(cache_key, "long_term")
                
                if cached_assessment and not is_optimization:
                    self.logger.info("Using cached quality assessment from memory")
                    assessment_result = cached_assessment
                else:
                    self.logger.info(f"Performing fresh quality assessment (turn {current_turn})")
                    assessment_result = self._assess_code_quality(generated_code, matlab_analysis, conversion_plan, state)
                    
                    # Cache the assessment in long-term memory
                    self.update_memory(cache_key, assessment_result, "long_term")
                
                # Update agent memory with assessment context
                self.update_memory("last_assessment", assessment_result, "short_term")
                self.update_memory("assessment_count", 
                                 (self.get_memory("assessment_count", "short_term") or 0) + 1, 
                                 "short_term")
                self.update_memory("current_turn", current_turn, "context")
                self.update_memory("is_optimization", is_optimization, "context")
                
                # Calculate quality scores
                quality_scores = self._calculate_quality_scores(assessment_result)
                
                # Update state with assessment results
                state["quality_scores"] = quality_scores
                state["assessment_reports"].append(f"Turn {current_turn} assessment")
                
                # Update state with agent memory
                state = self.update_state_with_result(state, assessment_result, "quality_assessment")
                
                # Log success
                duration = time.time() - start_time
                overall_score = quality_scores.get('overall', 0.0)
                self.logger.info(f"Quality assessment completed successfully in {duration:.2f}s")
                self.logger.info(f"Overall quality score: {overall_score:.1f}/10")
                
                # Track performance
                self.track_performance("quality_assessment", start_time, time.time(), True, {
                    'turn': current_turn,
                    'is_optimization': is_optimization,
                    'overall_score': overall_score,
                    'categories_assessed': len(quality_scores)
                })
                
            except Exception as e:
                self.logger.error(f"Error in quality assessment: {e}")
                state["error_message"] = f"Quality assessment failed: {str(e)}"
                state = update_state_status(state, ConversionStatus.FAILED)
                
                # Track failure
                self.track_performance("quality_assessment", start_time, time.time(), False, {
                    'error': str(e)
                })
            
            # Record processing time
            duration = time.time() - start_time
            state = add_processing_time(state, "quality_assessment", duration)
            
            return state
        
        return assess_quality_node
    
    def get_tools(self) -> List[Any]:
        """Get available tools for this agent."""
        return [
            self.quality_assessment_tool,
            self.llm_analysis_tool
        ]
    
    def _assess_code_quality(self, generated_code: Dict[str, Any], matlab_analysis: Dict[str, Any], 
                           conversion_plan: Dict[str, Any], state: ConversionState) -> Dict[str, Any]:
        """
        Assess code quality using LangGraph tools.
        
        Args:
            generated_code: Generated C++ code
            matlab_analysis: MATLAB analysis results
            conversion_plan: Conversion plan
            state: Current conversion state
            
        Returns:
            Comprehensive quality assessment
        """
        self.logger.debug("Assessing code quality using LangGraph tools")
        
        # Prepare code for assessment
        cpp_code = self._prepare_code_for_assessment(generated_code)
        matlab_code = self._prepare_matlab_code_for_assessment(matlab_analysis)
        
        # Use quality assessment tool
        assessment_result = self.quality_assessment_tool(cpp_code, matlab_code, conversion_plan)
        
        if not assessment_result.success:
            self.logger.warning(f"Quality assessment tool failed: {assessment_result.error}")
            # Create fallback assessment
            return self._create_fallback_assessment(cpp_code, conversion_plan)
        
        assessment_data = assessment_result.data
        
        # Enhance assessment with additional analysis
        enhanced_assessment = self._enhance_assessment(assessment_data, generated_code, conversion_plan, state)
        
        # Add assessment metadata
        enhanced_assessment['assessment_metadata'] = {
            'turn': state.get("current_turn", 0),
            'is_optimization': state.get("current_turn", 0) > 0,
            'assessment_timestamp': time.time(),
            'tool_used': 'quality_assessment_tool',
            'agent_memory': {
                'assessment_count': self.get_memory("assessment_count", "short_term") or 0,
                'last_assessment_time': time.time()
            }
        }
        
        self.logger.info(f"Quality assessment complete: {enhanced_assessment.get('overall_score', 0)}/10")
        
        return enhanced_assessment
    
    def _prepare_code_for_assessment(self, generated_code: Dict[str, Any]) -> str:
        """Prepare C++ code for assessment."""
        if isinstance(generated_code, dict) and 'files' in generated_code:
            # Multi-file project - concatenate all files
            all_code = []
            for filename, content in generated_code['files'].items():
                all_code.append(f"// File: {filename}")
                all_code.append(content)
                all_code.append("")
            return "\n".join(all_code)
        else:
            # Single-file project
            header = generated_code.get('header', '')
            implementation = generated_code.get('implementation', '')
            return f"{header}\n\n{implementation}".strip()
    
    def _prepare_matlab_code_for_assessment(self, matlab_analysis: Dict[str, Any]) -> str:
        """Prepare MATLAB code for assessment."""
        if not matlab_analysis:
            return ""
        
        file_analyses = matlab_analysis.get('file_analyses', [])
        if not file_analyses:
            return ""
        
        # Concatenate all MATLAB code
        matlab_code_parts = []
        for file_analysis in file_analyses:
            file_path = file_analysis['file_path']
            parsed_structure = file_analysis.get('parsed_structure', {})
            
            matlab_code_parts.append(f"// MATLAB File: {file_path}")
            # Note: We don't have the original MATLAB content in parsed_structure
            # This would need to be stored during analysis if needed for assessment
            matlab_code_parts.append("// Original MATLAB code not available for assessment")
            matlab_code_parts.append("")
        
        return "\n".join(matlab_code_parts)
    
    def _enhance_assessment(self, assessment_data: Dict[str, Any], generated_code: Dict[str, Any], 
                          conversion_plan: Dict[str, Any], state: ConversionState) -> Dict[str, Any]:
        """Enhance assessment with additional analysis."""
        enhanced_assessment = assessment_data.copy()
        
        # Add project-specific assessment
        project_type = conversion_plan.get('project_type', 'single_file')
        enhanced_assessment['project_specific_metrics'] = self._assess_project_specific_quality(
            generated_code, conversion_plan, project_type
        )
        
        # Add optimization-specific assessment if this is an optimization turn
        current_turn = state.get("current_turn", 0)
        if current_turn > 0:
            enhanced_assessment['optimization_metrics'] = self._assess_optimization_quality(
                generated_code, state, current_turn
            )
        
        # Add memory-based assessment trends
        enhanced_assessment['assessment_trends'] = self._analyze_assessment_trends(state)
        
        # Add recommendations based on assessment
        enhanced_assessment['recommendations'] = self._generate_recommendations(enhanced_assessment, conversion_plan)
        
        return enhanced_assessment
    
    def _assess_project_specific_quality(self, generated_code: Dict[str, Any], 
                                       conversion_plan: Dict[str, Any], project_type: str) -> Dict[str, Any]:
        """Assess project-specific quality metrics."""
        metrics = {
            'project_type': project_type,
            'file_organization': 0.0,
            'dependency_management': 0.0,
            'namespace_usage': 0.0,
            'header_organization': 0.0
        }
        
        if project_type == 'multi_file':
            # Assess multi-file specific metrics
            if 'files' in generated_code:
                files = generated_code['files']
                metrics['file_organization'] = min(10.0, len(files) * 2.0)  # Reward multiple files
                
                # Check for proper header/implementation separation
                header_files = [f for f in files.keys() if f.endswith('.h')]
                cpp_files = [f for f in files.keys() if f.endswith('.cpp')]
                
                if len(header_files) > 0 and len(cpp_files) > 0:
                    metrics['header_organization'] = 8.0
                else:
                    metrics['header_organization'] = 4.0
                
                # Check for namespace usage
                namespace_count = sum(1 for content in files.values() if 'namespace' in content)
                metrics['namespace_usage'] = min(10.0, namespace_count * 3.0)
        else:
            # Single file project metrics
            if generated_code.get('header') and generated_code.get('implementation'):
                metrics['header_organization'] = 8.0
            else:
                metrics['header_organization'] = 5.0
            
            metrics['file_organization'] = 6.0  # Neutral for single file
            metrics['namespace_usage'] = 5.0 if 'namespace' in str(generated_code) else 3.0
        
        # Assess dependency management
        dependencies = conversion_plan.get('dependencies', [])
        if dependencies:
            metrics['dependency_management'] = min(10.0, len(dependencies) * 2.0)
        else:
            metrics['dependency_management'] = 5.0
        
        return metrics
    
    def _assess_optimization_quality(self, generated_code: Dict[str, Any], 
                                   state: ConversionState, current_turn: int) -> Dict[str, Any]:
        """Assess optimization-specific quality metrics."""
        metrics = {
            'turn': current_turn,
            'improvement_detected': False,
            'optimization_effectiveness': 0.0,
            'new_optimizations': [],
            'regressions': []
        }
        
        # Compare with previous turn if available
        previous_assessment = self.get_memory(f"quality_assessment_{current_turn - 1}", "long_term")
        
        if previous_assessment:
            current_score = self._calculate_quality_scores(self._assess_code_quality(
                generated_code, state.get("matlab_analysis"), state.get("conversion_plan"), state
            )).get('algorithmic', 0.0)
            
            previous_score = previous_assessment.get('overall_score', 0.0)
            
            if current_score > previous_score:
                metrics['improvement_detected'] = True
                metrics['optimization_effectiveness'] = min(10.0, (current_score - previous_score) * 5.0)
                metrics['new_optimizations'] = ['Quality score improved']
            elif current_score < previous_score:
                metrics['regressions'] = ['Quality score decreased']
                metrics['optimization_effectiveness'] = 0.0
            else:
                metrics['optimization_effectiveness'] = 5.0  # No change
        
        return metrics
    
    def _analyze_assessment_trends(self, state: ConversionState) -> Dict[str, Any]:
        """Analyze assessment trends from memory."""
        trends = {
            'assessment_history': [],
            'improvement_trend': 'stable',
            'consistency_score': 0.0,
            'average_score': 0.0
        }
        
        # Get assessment history from memory
        assessment_count = self.get_memory("assessment_count", "short_term") or 0
        
        if assessment_count > 1:
            # Collect historical assessments
            for turn in range(assessment_count):
                cache_key = f"quality_assessment_{turn}"
                assessment = self.get_memory(cache_key, "long_term")
                if assessment:
                    score = assessment.get('overall_score', 0.0)
                    trends['assessment_history'].append({
                        'turn': turn,
                        'score': score,
                        'timestamp': assessment.get('assessment_metadata', {}).get('assessment_timestamp', 0)
                    })
            
            if trends['assessment_history']:
                scores = [h['score'] for h in trends['assessment_history']]
                trends['average_score'] = sum(scores) / len(scores)
                
                # Calculate improvement trend
                if len(scores) >= 2:
                    if scores[-1] > scores[0]:
                        trends['improvement_trend'] = 'improving'
                    elif scores[-1] < scores[0]:
                        trends['improvement_trend'] = 'declining'
                    else:
                        trends['improvement_trend'] = 'stable'
                
                # Calculate consistency
                if len(scores) > 1:
                    variance = sum((s - trends['average_score']) ** 2 for s in scores) / len(scores)
                    trends['consistency_score'] = max(0.0, 10.0 - variance * 2.0)
        
        return trends
    
    def _generate_recommendations(self, assessment: Dict[str, Any], conversion_plan: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on assessment results."""
        recommendations = []
        
        overall_score = assessment.get('overall_score', 0.0)
        
        # Score-based recommendations
        if overall_score < 5.0:
            recommendations.append("Code quality is below acceptable threshold. Consider major refactoring.")
        elif overall_score < 7.0:
            recommendations.append("Code quality is moderate. Focus on improving weak areas.")
        elif overall_score < 9.0:
            recommendations.append("Code quality is good. Minor improvements recommended.")
        else:
            recommendations.append("Code quality is excellent. Ready for production use.")
        
        # Category-specific recommendations
        categories = ['algorithmic_correctness', 'performance', 'error_handling', 'code_style', 'maintainability']
        
        for category in categories:
            if category in assessment:
                category_data = assessment[category]
                score = category_data.get('score', 0.0)
                
                if score < 6.0:
                    issues = category_data.get('issues', [])
                    suggestions = category_data.get('suggestions', [])
                    
                    if issues:
                        recommendations.append(f"{category.replace('_', ' ').title()} needs attention: {issues[0]}")
                    if suggestions:
                        recommendations.append(f"Recommendation for {category.replace('_', ' ')}: {suggestions[0]}")
        
        # Conversion mode specific recommendations
        conversion_mode = conversion_plan.get('conversion_mode', 'result-focused')
        if conversion_mode == 'faithful':
            if assessment.get('algorithmic_correctness', {}).get('score', 0.0) < 8.0:
                recommendations.append("For faithful mode, ensure algorithmic correctness is prioritized.")
        else:  # result-focused
            if assessment.get('performance', {}).get('score', 0.0) < 7.0:
                recommendations.append("For result-focused mode, consider performance optimizations.")
        
        return recommendations
    
    def _calculate_quality_scores(self, assessment: Dict[str, Any]) -> Dict[str, float]:
        """Calculate quality scores from assessment data."""
        scores = {}
        
        # Extract scores from assessment categories
        categories = ['algorithmic_correctness', 'performance', 'error_handling', 'code_style', 'maintainability']
        
        for category in categories:
            if category in assessment:
                category_data = assessment[category]
                if isinstance(category_data, dict):
                    scores[category] = category_data.get('score', 0.0)
                else:
                    scores[category] = float(category_data)
            else:
                scores[category] = 5.0  # Default score
        
        # Add overall score
        scores['overall'] = assessment.get('overall_score', sum(scores.values()) / len(scores))
        
        # Add project-specific scores if available
        project_metrics = assessment.get('project_specific_metrics', {})
        for metric, value in project_metrics.items():
            if isinstance(value, (int, float)):
                scores[f"project_{metric}"] = float(value)
        
        return scores
    
    def _create_fallback_assessment(self, cpp_code: str, conversion_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback assessment when tools fail."""
        self.logger.warning("Creating fallback quality assessment")
        
        # Basic heuristic assessment
        issues = []
        suggestions = []
        
        # Check for basic issues
        if 'using namespace std' in cpp_code:
            issues.append("'using namespace std' detected")
            suggestions.append("Remove 'using namespace std' and use std:: prefix")
        
        if '.inverse(' in cpp_code.lower():
            issues.append("Explicit matrix inverse detected")
            suggestions.append("Replace with factorization-based solver")
        
        if 'new ' in cpp_code or 'delete ' in cpp_code:
            issues.append("Raw pointer allocation detected")
            suggestions.append("Use smart pointers or RAII containers")
        
        # Calculate basic scores
        base_score = 7.0
        penalty = len(issues) * 1.0
        overall_score = max(0.0, base_score - penalty)
        
        return {
            'algorithmic_correctness': {
                'score': overall_score,
                'issues': issues,
                'suggestions': suggestions
            },
            'performance': {
                'score': 7.0,
                'issues': [],
                'suggestions': []
            },
            'error_handling': {
                'score': 6.0,
                'issues': ['Basic error handling'],
                'suggestions': ['Add comprehensive error handling']
            },
            'code_style': {
                'score': 7.0,
                'issues': [],
                'suggestions': []
            },
            'maintainability': {
                'score': 7.0,
                'issues': [],
                'suggestions': []
            },
            'overall_score': overall_score,
            'summary': f'Fallback assessment completed. {len(issues)} issues detected.',
            'recommendations': suggestions,
            'assessment_metadata': {
                'fallback_assessment': True,
                'assessment_timestamp': time.time()
            }
        }
    
    def get_assessment_summary(self, state: ConversionState) -> Dict[str, Any]:
        """Get a summary of the assessment results from state."""
        quality_scores = state.get("quality_scores", {})
        assessment_reports = state.get("assessment_reports", [])
        
        return {
            'overall_score': quality_scores.get('overall', 0.0),
            'category_scores': {k: v for k, v in quality_scores.items() if k != 'overall'},
            'assessment_count': len(assessment_reports),
            'latest_assessment': assessment_reports[-1] if assessment_reports else None,
            'agent_performance': self.get_performance_summary()
        }
