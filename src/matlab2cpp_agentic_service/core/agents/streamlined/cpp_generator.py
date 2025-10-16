"""
Enhanced C++ Generator Agent

This agent provides advanced C++ generation with:
- Integrated compilation testing during generation
- Real-time error feedback and correction
- Multi-file coordination during generation
- Adaptive prompt enhancement based on compilation results
- Strategy-based regeneration
"""

import asyncio
import re
import time
from typing import Dict, Any, List, Optional, Set, Tuple, Callable
from pathlib import Path

from ..base.langgraph_agent import BaseLangGraphAgent, AgentConfig
from ....infrastructure.tools.llm_client import LLMClient
from ....infrastructure.state.conversion_state import ConversionState
from ....infrastructure.tools.langgraph_tools import CodeGenerationTool
from ....infrastructure.testing.compilation_manager import CPPCompilationManager
from .compilation_log_analyzer import CompilationLogAnalyzer
from .error_fix_prompt_generator import ErrorFixPromptGenerator
from .domain_analyzer import DomainAnalyzer
from ....infrastructure.tools.syntax_fixer import RobustCppSyntaxFixer
from .error_fix_iterator import IterativeErrorFixer  # PRIORITY 3: Iterative fixing
from ....infrastructure.knowledge.api_knowledge_base import APIKnowledgeBase  # PHASE 1: API Knowledge
from ....infrastructure.fixing.targeted_error_fixer import TargetedErrorFixer  # PHASE 2: Targeted Fixing


class CppGenerator(BaseLangGraphAgent):
    """
    C++ generator with integrated compilation testing.
    
    Capabilities:
    - Real-time compilation testing during generation
    - Adaptive prompt enhancement based on compilation results
    - Multi-file coordination during generation
    - Error-driven code improvement
    - Strategy-based regeneration
    """
    
    def __init__(self, config: AgentConfig, llm_client: LLMClient):
        super().__init__(config, llm_client)
        
        # Enhanced tools
        self.tools = [
            "code_generation",
            "compilation_testing",  # Integrated!
            "error_diagnosis",
            "prompt_enhancement",
            "llm_analysis"
        ]
        
        # Initialize tools
        self.code_generation_tool = CodeGenerationTool(llm_client)
        self.compilation_manager = None  # Will be initialized with build_system from state
        self.log_analyzer = CompilationLogAnalyzer(config, llm_client)
        self.error_fix_generator = ErrorFixPromptGenerator(config, llm_client)
        self.domain_analyzer = DomainAnalyzer()
        self.syntax_fixer = RobustCppSyntaxFixer()  # NEW: Robust syntax fixing
        self.iterative_fixer = None  # Lazy init (needs docker_manager)
        self.api_knowledge_base = APIKnowledgeBase()  # PHASE 1: API Knowledge
        self.targeted_fixer = TargetedErrorFixer()  # PHASE 2: Targeted Fixing
        
        # Generation strategies
        self.generation_strategies = self._initialize_generation_strategies()
        self.error_patterns = self._initialize_error_patterns()
        
        # Performance tracking
        self.generation_iterations = 0
        self.compilation_attempts = 0
        self.successful_compilations = 0
        
        # Iterative fixing configuration
        self.enable_iterative_fixing = False  # Disabled - using built-in iteration instead
        self.iterative_fix_max_iterations = 3
        self.iterative_fix_threshold = 3  # Success if errors < 3
        
        templates_root = Path(__file__).resolve().parents[3] / "infrastructure" / "templates"
        self.helper_templates_dir = templates_root / "helpers"
        self._helper_cache: Dict[str, str] = {}

        self.logger.info(f"Initialized C++ Generator: {config.name}")
    
    def create_node(self) -> Callable[[ConversionState], ConversionState]:
        """Create the LangGraph node function for C++ generation."""
        async def generate_node(state: ConversionState) -> ConversionState:
            return await self.generate_with_testing(
                state.get("conversion_plan", {}),
                state.get("analysis_results", {}),
                state
            )
        return generate_node
    
    def get_tools(self) -> List[Any]:
        """Get available tools for C++ generation."""
        return [
            self.code_generation_tool,
            self.compilation_manager,
        ]
    
    def _initialize_generation_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize generation strategies for different scenarios."""
        return {
            "conservative": {
                "description": "Conservative approach with minimal C++ features",
                "features": ["basic_types", "simple_functions", "standard_headers"],
                "use_case": "Simple MATLAB code with basic operations"
            },
            "eigen_optimized": {
                "description": "Eigen-optimized approach for mathematical operations",
                "features": ["eigen_types", "vectorized_operations", "eigen_headers"],
                "use_case": "Mathematical computations, matrix operations"
            },
            "opencv_integrated": {
                "description": "OpenCV-integrated approach for image processing",
                "features": ["opencv_types", "image_processing", "opencv_headers"],
                "use_case": "Image processing, computer vision"
            },
            "performance_focused": {
                "description": "Performance-focused approach with optimizations",
                "features": ["memory_management", "algorithm_optimization", "performance_headers"],
                "use_case": "Performance-critical applications"
            },
            "multi_file_coordinated": {
                "description": "Multi-file coordinated approach",
                "features": ["namespace_management", "header_coordination", "dependency_management"],
                "use_case": "Complex multi-file projects"
            }
        }
    
    def _initialize_error_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize error patterns and their corresponding strategies."""
        return {
            "include_errors": {
                "patterns": ["fatal error:", "No such file or directory"],
                "strategy": "fix_includes",
                "severity": "high"
            },
            "template_errors": {
                "patterns": ["expected", "template", "typename"],
                "strategy": "fix_templates",
                "severity": "high"
            },
            "namespace_errors": {
                "patterns": ["'::'", "namespace", "unqualified-id"],
                "strategy": "fix_namespace",
                "severity": "medium"
            },
            "type_errors": {
                "patterns": ["cannot convert", "invalid conversion", "no matching function"],
                "strategy": "fix_types",
                "severity": "high"
            },
            "syntax_errors": {
                "patterns": ["expected", "before", "syntax error"],
                "strategy": "fix_syntax",
                "severity": "medium"
            }
        }
    
    async def generate_with_testing(self, conversion_plan: Dict[str, Any], 
                                  matlab_analysis: Dict[str, Any], 
                                  state: ConversionState) -> ConversionState:
        """
        Generate C++ code with integrated compilation testing.
        
        Args:
            conversion_plan: Conversion plan from planner
            matlab_analysis: MATLAB analysis results
            state: Current conversion state
            
        Returns:
            Updated state with generated code
        """
        start_time = time.time()
        
        try:
            self.logger.info("Starting enhanced C++ generation with testing...")
            
            # Check if this is a multi-file project
            is_multi_file = conversion_plan.get('project_structure', {}).get('is_multi_file', False)
            
            # Initialize strategy for both paths
            strategy = self._select_generation_strategy(conversion_plan, matlab_analysis)
            
            if is_multi_file:
                self.logger.info("📁 Multi-file project detected - using file-by-file generation")
                # Use file-by-file generation for multi-file projects
                generated_code = await self._generate_multi_file_project(
                    conversion_plan, matlab_analysis, state
                )
            else:
                self.logger.info("📄 Single-file project - using standard generation")
                # Generate code iteratively with compilation testing
                # Note: main.cpp is now generated INSIDE the iteration loop for proper error handling
                generated_code = await self._generate_and_test_iteratively(
                    conversion_plan, matlab_analysis, strategy, state
                )
            
            # Get the final compilation result for quality assessment
            final_compilation_result = None
            if generated_code and isinstance(generated_code, dict) and 'files' in generated_code:
                # The final compilation result is already in generated_code from the iteration loop
                # No need to test again - just extract it
                final_compilation_result = generated_code.get('compilation_result')
            elif not generated_code or not isinstance(generated_code, dict):
                self.logger.error(f"❌ Generated code is invalid: type={type(generated_code)}, value={generated_code}")
                # Create minimal fallback structure
                generated_code = {
                    'files': {},
                    'dependencies': [],
                    'compilation_instructions': '',
                    'usage_example': '',
                    'notes': 'Generation failed - no valid code produced',
                    'conversion_mode': 'single_file',
                    'raw_response': ''
                }
                
                # PRIORITY 3: Apply iterative error fixing if compilation failed
                if final_compilation_result and not final_compilation_result.get('success', False) and self.enable_iterative_fixing:
                    self.logger.info("\n" + "="*80)
                    self.logger.info("🔄 COMPILATION FAILED - Starting Iterative LLM Error Fixing")
                    self.logger.info("="*80 + "\n")
                    
                    try:
                        # Apply iterative fixing
                        fixed_files = await self._apply_iterative_error_fixing(
                            generated_code=generated_code,
                            matlab_analysis=matlab_analysis,
                            compilation_result=final_compilation_result,
                            project_name=conversion_plan.get('project_name', 'test_project')
                        )
                        
                        if fixed_files:
                            # Update generated code with fixed files
                            generated_code['files'] = fixed_files
                            
                            # Re-compile to get final result
                            final_compilation_result = await self._test_compilation(generated_code, conversion_plan)
                            
                            self.logger.info(f"✅ Iterative fixing complete. Final compilation: {'SUCCESS' if final_compilation_result.get('success') else 'FAILED'}")
                        
                    except Exception as e:
                        self.logger.error(f"Iterative error fixing failed: {e}")
                        # Continue with original code
            
            # Create comprehensive generation result
            generation_result = {
                'generated_code': generated_code,
                'compilation_result': final_compilation_result,
                'generation_strategy': strategy,
                'generation_iterations': self.generation_iterations,
                'compilation_attempts': self.compilation_attempts,
                'successful_compilations': self.successful_compilations,
                'compilation_success_rate': self.successful_compilations / max(self.compilation_attempts, 1),
                'generation_timestamp': time.time()
            }
            
            # Update state
            state["generated_code"] = generation_result
            
            # Update memory
            self.update_memory("generation_count", 
                             (self.get_memory("generation_count", "short_term") or 0) + 1, 
                             "short_term")
            
            # Track performance
            execution_time = time.time() - start_time
            files_count = len(generated_code.get('files', {})) if generated_code and isinstance(generated_code, dict) else 0
            self.track_performance("generate_with_testing", start_time, time.time(), True, 
                                 {"files_generated": files_count})
            
            self.logger.info(f"Enhanced C++ generation complete: "
                           f"{files_count} files, "
                           f"{self.generation_iterations} iterations, "
                           f"{execution_time:.2f}s")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Enhanced C++ generation failed: {e}")
            self.track_performance("generate_with_testing", start_time, time.time(), False, 
                                 {"error": str(e)})
            raise
    
    def _select_generation_strategy(self, conversion_plan: Dict[str, Any], 
                                  matlab_analysis: Dict[str, Any]) -> str:
        """Select optimal generation strategy based on project characteristics."""
        conversion_mode = conversion_plan.get('conversion_mode', 'single_file_simple')
        complexity_level = matlab_analysis.get('complexity_assessment', {}).get('complexity_level', 'simple')
        
        # Analyze MATLAB content for specific patterns
        content_patterns = self._analyze_content_patterns(matlab_analysis)
        
        if conversion_mode.startswith('multi_file'):
            return "multi_file_coordinated"
        elif 'eigen' in content_patterns or 'matrix' in content_patterns:
            return "eigen_optimized"
        elif 'opencv' in content_patterns or 'image' in content_patterns:
            return "opencv_integrated"
        elif complexity_level == 'complex':
            return "performance_focused"
        else:
            return "conservative"
    
    def _analyze_content_patterns(self, matlab_analysis: Dict[str, Any]) -> List[str]:
        """Analyze MATLAB content for specific patterns."""
        patterns = []
        
        file_analyses = matlab_analysis.get('file_analyses', [])
        for analysis in file_analyses:
            content = analysis.get('content', '').lower()
            
            if any(keyword in content for keyword in ['eigen', 'matrix', 'vector']):
                patterns.append('eigen')
            if any(keyword in content for keyword in ['opencv', 'image', 'imread', 'imshow']):
                patterns.append('opencv')
            if any(keyword in content for keyword in ['plot', 'figure', 'subplot']):
                patterns.append('plotting')
            if any(keyword in content for keyword in ['fft', 'ifft', 'conv']):
                patterns.append('signal_processing')
        
        return patterns
    
    async def _generate_multi_file_project(self, conversion_plan: Dict[str, Any],
                                           matlab_analysis: Dict[str, Any],
                                           state: ConversionState) -> Dict[str, Any]:
        """
        Generate multi-file project using file-by-file approach.
        
        Args:
            conversion_plan: Conversion plan with file mapping
            matlab_analysis: Analysis results for all files
            state: Current conversion state
            
        Returns:
            Generated code for all files
        """
        self.logger.info("=" * 80)
        self.logger.info("🔄 Starting File-by-File Multi-File Generation")
        self.logger.info("=" * 80)
        
        # Get file information from plan and analysis
        file_analyses = matlab_analysis.get('file_analyses', [])
        compilation_order = matlab_analysis.get('compilation_order', [])
        file_mapping = conversion_plan.get('file_organization', {}).get('file_mapping', {})
        
        self.logger.info(f"📊 Project Info:")
        self.logger.info(f"  - Total files: {len(file_analyses)}")
        self.logger.info(f"  - Compilation order: {compilation_order}")
        
        # Storage for all generated files
        all_generated_files = {}
        previously_generated = {}
        
        # Generate each file in compilation order
        for idx, matlab_file in enumerate(compilation_order, 1):
            self.logger.info("-" * 80)
            self.logger.info(f"📝 Generating {idx}/{len(compilation_order)}: {matlab_file}")
            self.logger.info("-" * 80)
            
            # Find the analysis for this specific file
            # Handle both 'filename' and 'filename.m' formats
            search_name = matlab_file if matlab_file.endswith('.m') else f"{matlab_file}.m"
            file_analysis = next((fa for fa in file_analyses if fa['file_name'] == search_name), None)
            
            if not file_analysis:
                self.logger.warning(f"⚠️  No analysis found for {matlab_file} (searched: {search_name}), skipping")
                continue
            
            # Get C++ file names for this MATLAB file
            # Ensure we have proper base name (without .m extension)
            base_name = matlab_file.replace('.m', '') if matlab_file.endswith('.m') else matlab_file
            cpp_mapping = file_mapping.get(matlab_file, {})
            cpp_file = cpp_mapping.get('cpp_file', f"{base_name}.cpp")
            header_file = cpp_mapping.get('header_file', f"{base_name}.h")
            namespace = cpp_mapping.get('namespace', base_name)
            
            self.logger.info(f"  MATLAB: {matlab_file}")
            self.logger.info(f"  Header: {header_file}")
            self.logger.info(f"  CPP:    {cpp_file}")
            self.logger.info(f"  Namespace: {namespace}")
            
            # Generate this file with context of previously generated files
            try:
                file_code = await self._generate_single_file_with_context(
                    matlab_file=matlab_file,
                    file_analysis=file_analysis,
                    conversion_plan=conversion_plan,
                    matlab_analysis=matlab_analysis,
                    previously_generated=previously_generated,
                    namespace=namespace,
                    target_cpp=cpp_file,
                    target_header=header_file,
                    state=state
                )
                
                if file_code and 'files' in file_code:
                    # Store the generated files
                    all_generated_files.update(file_code['files'])
                    previously_generated[matlab_file] = file_code['files']
                    self.logger.info(f"  ✅ Successfully generated {len(file_code['files'])} file(s)")
                else:
                    self.logger.error(f"  ❌ Failed to generate code for {matlab_file}")
                    
            except Exception as e:
                self.logger.error(f"  ❌ Exception generating {matlab_file}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())

        # Global LLM iteration DISABLED - proven to make things worse (4 → 9 errors)
        # Using Phase 2 targeted fixing instead
        self.logger.info("=" * 80)
        self.logger.info(f"✅ Multi-File Generation Complete: {len(all_generated_files)} files")
        self.logger.info(f"   Files: {list(all_generated_files.keys())}")
        self.logger.info("=" * 80)
        
        # PHASE 2: Apply targeted pattern-based fixes
        self.logger.info("=" * 80)
        self.logger.info("🔧 Applying Phase 2: Targeted Error Fixing")
        self.logger.info("=" * 80)
        
        fixed_files = await self._apply_targeted_fixes(
            all_generated_files,
            conversion_plan
        )
        
        helper_files = self._inject_support_helpers(fixed_files, conversion_plan)
        if helper_files:
            fixed_files.update(helper_files)
        
        # Test compilation of all generated files (without main.cpp)
        self.logger.info("=" * 80)
        self.logger.info("🧪 Testing Compilation of Generated Files")
        self.logger.info("=" * 80)
        
        test_result = await self._test_compilation(
            {'files': fixed_files}, 
            conversion_plan
        )
        
        # Generate main.cpp only if compilation succeeded (for multi-file projects)
        # Reason: main.cpp is for EXECUTION TESTING, not for broken code
        self.logger.info("=" * 80)
        if test_result.get('success', False):
            self.logger.info("✅ All files compile successfully")
            self.logger.info("🎯 Generating main.cpp Entry Point with Multi-File Analysis")
            self.logger.info("=" * 80)
            
            try:
                main_cpp = self._generate_main_entry_point_multifile(
                    conversion_plan, matlab_analysis, fixed_files
                )
                fixed_files['main.cpp'] = main_cpp
                self.logger.info("  ✅ Successfully generated main.cpp with entry point analysis")
                
                # Re-test compilation with main.cpp
                self.logger.info("🧪 Re-testing compilation with main.cpp included")
                final_test = await self._test_compilation(
                    {'files': fixed_files},
                    conversion_plan
                )
                
                if final_test.get('success', False):
                    self.logger.info("✅ FINAL COMPILATION SUCCESSFUL (with main.cpp)")
                    compilation_result = final_test
                else:
                    self.logger.warning("⚠️ main.cpp compilation failed, but keeping generated files")
                    compilation_result = final_test
                        
            except Exception as e:
                self.logger.error(f"  ❌ Failed to generate main.cpp: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                compilation_result = test_result
        else:
            # Compilation failed - DON'T generate main.cpp for multi-file projects
            # Reason: LLM would hang trying to make sense of broken code with massive context
            self.logger.warning("⚠️ Multi-file compilation failed with errors")
            self.logger.info("=" * 80)
            self.logger.info("ℹ️  Skipping main.cpp generation")
            self.logger.info("=" * 80)
            self.logger.info("📋 Reason: main.cpp is for EXECUTION TESTING of working code")
            self.logger.info("📋 Current status: Code has compilation errors")
            self.logger.info("📋 Next steps:")
            self.logger.info("   1. Review compilation_errors.txt for error details")
            self.logger.info("   2. Fix the compilation errors in generated C++ files")
            self.logger.info("   3. Re-run conversion once errors are resolved")
            self.logger.info("   4. main.cpp will be generated for working code")
            self.logger.info("=" * 80)
            compilation_result = test_result
        
        # Wrap all files in expected format
        result = {
            'files': fixed_files,
            'project_name': conversion_plan.get('project_name', 'unknown'),
            'is_multi_file': True,
            'file_count': len(fixed_files),
            'compilation_result': compilation_result
        }
        
        return result
    
    async def _generate_and_test_iteratively(self, conversion_plan: Dict[str, Any],
                                             matlab_analysis: Dict[str, Any],
                                             strategy: str,
                                             state: ConversionState) -> Dict[str, Any]:
        """Generate code iteratively with compilation testing."""
        # Get max iterations from state or config
        max_iterations = state.get('max_retries', self.config.max_retries or 3)
        current_iteration = 0
        previous_error_count = 999  # FIX #4: Track progress
        
        while current_iteration < max_iterations:
            current_iteration += 1
            self.generation_iterations = current_iteration
            
            self.logger.info(f"Generation iteration {current_iteration}/{max_iterations}")
            
            # Generate code for this iteration
            generated_code = await self._generate_code_for_iteration(
                conversion_plan, matlab_analysis, strategy, current_iteration, state
            )
            
            # Debug: Log generated code structure
            self.logger.info(f"Generated code structure: {list(generated_code.keys()) if generated_code else 'None'}")
            if generated_code and 'files' in generated_code:
                self.logger.info(f"Generated files: {list(generated_code['files'].keys()) if generated_code['files'] else 'None'}")
            
            # FIX #2: Apply syntax fixer after each iteration
            if generated_code and 'files' in generated_code:
                self.logger.info(f"🔧 Applying syntax fixer to iteration {current_iteration} code...")
                fixed_files = {}
                for filename, content in generated_code['files'].items():
                    if filename.endswith(('.h', '.cpp')) and isinstance(content, str):
                        if filename.endswith('.h'):
                            # Extract base name for the fixer
                            base_name = filename.replace('.h', '')
                            fixed_header, _ = self.syntax_fixer.fix_all_syntax_issues(content, '', base_name)
                            fixed_files[filename] = fixed_header
                        elif filename.endswith('.cpp'):
                            base_name = filename.replace('.cpp', '')
                            _, fixed_impl = self.syntax_fixer.fix_all_syntax_issues('', content, base_name)
                            fixed_files[filename] = fixed_impl
                        else:
                            fixed_files[filename] = content
                    else:
                        fixed_files[filename] = content
                generated_code['files'] = fixed_files
                self.logger.info(f"✅ Syntax fixer applied to {len(fixed_files)} files")
            
            # Validate generated code before compilation
            validation_result = self._validate_generated_code(generated_code)
            if not validation_result['valid']:
                self.logger.error(f"❌ CODE VALIDATION FAILED: {validation_result['errors']}")
                # Try to fix validation issues
                generated_code = self._fix_validation_issues(generated_code, validation_result['errors'])
            
            # Test compilation with detailed logging
            compilation_result = await self._test_compilation(generated_code, conversion_plan, state)
            
            # DETAILED COMPILATION LOGGING
            self.logger.info(f"ITERATION {current_iteration} COMPILATION RESULT: success={compilation_result.get('success', False)}")
            
            if compilation_result.get('success', False):
                # Before returning, check if we need to add main.cpp
                if 'main.cpp' not in generated_code.get('files', {}):
                    self.logger.info(f"🎯 Compilation successful, now generating main.cpp for execution")
                    try:
                        main_cpp = self._generate_main_entry_point(
                            conversion_plan, matlab_analysis, generated_code['files']
                        )
                        generated_code['files']['main.cpp'] = main_cpp
                        self.logger.info(f"  ✅ Generated main.cpp, re-testing compilation with it included")
                        
                        # Re-test compilation with main.cpp included
                        compilation_with_main = await self._test_compilation(generated_code, conversion_plan, state)
                        
                        if compilation_with_main.get('success', False):
                            self.logger.info(f"✅ COMPILATION SUCCESSFUL (with main.cpp) after {current_iteration} iterations")
                            # Store final compilation result
                            generated_code['compilation_result'] = compilation_with_main
                            return generated_code
                        else:
                            # main.cpp has errors, continue iteration to fix it
                            self.logger.warning(f"⚠️ main.cpp compilation failed, continuing iteration to fix it")
                            compilation_result = compilation_with_main
                            # Fall through to error handling below, which will try to fix main.cpp
                    except Exception as e:
                        self.logger.warning(f"  ⚠️ Failed to generate main.cpp: {e}, returning without it")
                        # Store compilation result even if main.cpp generation failed
                        generated_code['compilation_result'] = compilation_result
                        return generated_code
                else:
                    # main.cpp already exists and compilation succeeded
                    self.logger.info(f"✅ COMPILATION SUCCESSFUL after {current_iteration} iterations")
                    # Store final compilation result
                    generated_code['compilation_result'] = compilation_result
                    return generated_code
            
            # Log detailed compilation errors
            self.logger.error(f"❌ COMPILATION FAILED in iteration {current_iteration}")
            compilation_errors = compilation_result.get('errors', [])
            current_error_count = len([e for e in compilation_errors if isinstance(e, str) and 'error:' in e.lower()])
            self.logger.error(f"DETAILED COMPILATION ERRORS ({len(compilation_errors)} found, {current_error_count} actual errors):")
            for i, error in enumerate(compilation_errors, 1):
                self.logger.error(f"  ERROR {i}: {error}")
            
            # FIX #4: Check progress - stop if no improvement
            if current_error_count >= previous_error_count:
                self.logger.warning(f"⚠️ NO PROGRESS: Errors {previous_error_count} → {current_error_count}")
                self.logger.warning(f"Stopping iteration to prevent thrashing")
                break
            else:
                self.logger.info(f"✅ PROGRESS: Errors {previous_error_count} → {current_error_count}")
                previous_error_count = current_error_count
            
            # Log full compilation output for debugging
            full_output = compilation_result.get('output', '')
            self.logger.error(f"FULL COMPILATION OUTPUT ({len(full_output)} chars):")
            self.logger.error(f"--- START ITERATION {current_iteration} COMPILATION LOG ---")
            self.logger.error(full_output)
            self.logger.error(f"--- END ITERATION {current_iteration} COMPILATION LOG ---")
            
            # Analyze compilation errors and generate specific fixes
            error_analysis = self._analyze_compilation_errors(compilation_result)
            
            # REACTIVE KNOWLEDGE: Get API documentation for errors
            api_docs = ""
            try:
                # Extract error messages for knowledge retrieval
                error_messages = [err.get('message', '') if isinstance(err, dict) else str(err) 
                                for err in compilation_errors[:10]]  # Limit to top 10 errors
                
                # Get current code for context
                current_code = "\n\n".join([f"// {fname}\n{content}" 
                                           for fname, content in generated_code.get('files', {}).items()])
                
                # Call with correct signature: error_messages (list), current_code (str), libraries_used (list)
                api_docs = self.api_knowledge_base.get_relevant_docs(
                    error_messages=error_messages,
                    current_code=current_code,
                    libraries_used=['eigen']
                )
                if api_docs:
                    self.logger.info(f"✅ Retrieved API docs for error fixing ({len(api_docs)} chars)")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to retrieve API docs for errors: {e}")
            
            # Generate specific error fix prompts with LLM-powered analysis
            error_fix_prompt = self.error_fix_generator.generate_error_fix_prompt(
                compilation_errors,
                generated_code,
                matlab_analysis,
                api_knowledge=api_docs  # Inject API docs into error fix prompt
            )
            
            if error_fix_prompt:
                self.logger.info(f"Generated LLM-powered error fix prompt for {len(compilation_errors)} errors")
                # Log the error fix prompt for debugging
                self.logger.info(f"--- START ITERATION {current_iteration} ERROR FIX PROMPT ---")
                self.logger.info(error_fix_prompt[:1000] + "..." if len(error_fix_prompt) > 1000 else error_fix_prompt)
                self.logger.info(f"--- END ITERATION {current_iteration} ERROR FIX PROMPT ---")
                # Use the error fix prompt as the enhanced strategy
                strategy = error_fix_prompt
            else:
                self.logger.info(f"No specific error fix prompt generated, using fallback strategy")
                # Fallback to generic improvement
                improvement_strategy = self._select_improvement_strategy(error_analysis, strategy)
                
                # Use log analysis for enhanced feedback if available
                log_analysis = compilation_result.get('log_analysis')
                if log_analysis:
                    self.logger.info(f"Using log analysis for improvement: {len(log_analysis.improvement_suggestions)} suggestions")
                    # Enhance the improvement strategy with log analysis insights
                    improvement_strategy = self._enhance_strategy_with_log_analysis(
                        improvement_strategy, log_analysis, error_analysis
                    )
                
                # Enhance strategy based on errors
                strategy = self._enhance_strategy_with_errors(strategy, error_analysis, improvement_strategy)
            
            # Log the updated strategy for debugging
            self.logger.debug(f"Updated strategy for next iteration: {strategy[:200]}..." if len(strategy) > 200 else f"Updated strategy: {strategy}")
            
            self.logger.info(f"Compilation failed, applying strategy for iteration {current_iteration}")
        
        # If we reach here, all iterations failed
        self.logger.warning(f"All {max_iterations} generation iterations failed")
        return await self._generate_fallback_code(conversion_plan, matlab_analysis, state)
    
    async def _generate_code_for_iteration(self, conversion_plan: Dict[str, Any],
                                         matlab_analysis: Dict[str, Any],
                                         strategy: str,
                                         iteration: int,
                                         state: ConversionState) -> Dict[str, Any]:
        """Generate code for a specific iteration."""
        # Use LangGraph tools for code generation
        generation_prompt = self._build_enhanced_generation_prompt(
            conversion_plan, matlab_analysis, strategy, iteration
        )
        
        # Generate code using LLM with /no_think parameter (since we're using general LLM)
        # Append /no_think to the prompt to minimize reasoning text
        enhanced_prompt = generation_prompt + "\n\n/no_think"
        response = self.llm_client.get_completion(enhanced_prompt)
        
        # Parse generated code using the proper parsing method
        parsed_code = self.code_generation_tool._parse_generated_code(response, "multi_file")
        
        # Apply post-processing fixes to the parsed code
        if parsed_code.get('header'):
            parsed_code['header'] = self.code_generation_tool._fix_corrupted_includes(parsed_code['header'])
        if parsed_code.get('implementation'):
            parsed_code['implementation'] = self.code_generation_tool._fix_corrupted_includes(parsed_code['implementation'])
        
        # Extract header and implementation content
        header_content = parsed_code.get('header', '')
        implementation_content = parsed_code.get('implementation', '')
        
        # If header is empty but implementation has content, extract header from implementation
        if not header_content and implementation_content:
            header_content = self._extract_header_from_implementation(implementation_content)
        
        # If implementation is empty but we have content, use the full content as implementation
        if not implementation_content and response:
            implementation_content = self._extract_implementation_from_response(response)
        
        # Validate and fix the generated content
        header_content = self._validate_and_fix_header(header_content)
        implementation_content = self._validate_and_fix_implementation(implementation_content)
        
        # Convert to expected structure with 'files' key
        generated_code = {
            'files': {
                'arma_filter.h': header_content,
                'arma_filter.cpp': implementation_content
            },
            'dependencies': parsed_code.get('dependencies', []),
            'compilation_instructions': parsed_code.get('compilation_instructions', ''),
            'usage_example': parsed_code.get('usage_example', ''),
            'notes': parsed_code.get('notes', ''),
            'conversion_mode': parsed_code.get('conversion_mode', 'multi_file'),
            'raw_response': parsed_code.get('raw_response', response)
        }
        
        return generated_code
    
    def _validate_and_fix_header(self, header_content: str) -> str:
        """Validate and fix header content to ensure it's syntactically correct."""
        if not header_content or header_content.strip() == '':
            return self._generate_minimal_header()
        
        # Check for common issues and fix them
        lines = header_content.split('\n')
        fixed_lines = []
        
        # Ensure we have proper header guards (use dynamic guard name)
        has_header_guard = any('#ifndef' in line for line in lines)
        if not has_header_guard:
            # Try to extract namespace name for guard, default to GENERATED_H
            guard_name = 'GENERATED_H'
            for line in lines:
                if 'namespace ' in line:
                    import re
                    match = re.search(r'namespace\s+(\w+)', line)
                    if match:
                        guard_name = f"{match.group(1).upper()}_H"
                        break
            fixed_lines.append(f'#ifndef {guard_name}')
            fixed_lines.append(f'#define {guard_name}')
            fixed_lines.append('')
        
        # Process each line
        for line in lines:
            stripped = line.strip()
            
            # Fix malformed lines
            if stripped == ');' or stripped == '}' or stripped == '};':
                continue  # Skip orphaned closing syntax
            
            # Keep namespace declarations as-is (don't filter them)
            
            # Add proper includes
            if stripped.startswith('#include') and stripped not in ['#include <vector>', '#include <string>', '#include <Eigen/Dense>']:
                fixed_lines.append(line)
            elif not stripped.startswith('#include'):
                fixed_lines.append(line)
        
        # Fix missing namespace closing braces before #endif
        result_content = '\n'.join(fixed_lines)
        result_content = self._fix_namespace_closing_before_endif(result_content)
        
        # Ensure we close the header guard if missing
        if not any('#endif' in line for line in result_content.split('\n')):
            # Try to extract namespace and guard name
            import re
            namespace_match = re.search(r'namespace\s+(\w+)', result_content)
            guard_match = re.search(r'#ifndef\s+(\w+)', result_content)
            
            namespace_name = namespace_match.group(1) if namespace_match else 'generated'
            guard_name = guard_match.group(1) if guard_match else 'GENERATED_H'
            
            fixed_lines = result_content.split('\n')
            fixed_lines.append(f'}} // namespace {namespace_name}')
            fixed_lines.append('')
            fixed_lines.append(f'#endif // {guard_name}')
            result_content = '\n'.join(fixed_lines)
        
        return result_content
    
    def _fix_namespace_closing_before_endif(self, content: str) -> str:
        """
        Fix missing closing braces for namespaces before #endif.
        Common LLM error: namespace { ... #endif (missing })
        """
        import re
        
        # Find all namespace declarations
        namespace_pattern = r'namespace\s+(\w+)\s*\{'
        namespaces = re.findall(namespace_pattern, content)
        
        if not namespaces:
            return content
        
        # Split into lines
        lines = content.split('\n')
        endif_line = -1
        for i, line in enumerate(lines):
            if '#endif' in line:
                endif_line = i
                break
        
        if endif_line == -1:
            return content
        
        # Count open and closed braces before #endif
        content_before_endif = '\n'.join(lines[:endif_line])
        open_braces = content_before_endif.count('{')
        close_braces = content_before_endif.count('}')
        
        # If we have unclosed braces
        if open_braces > close_braces:
            missing_braces = open_braces - close_braces
            
            # Add closing braces with namespace comments before #endif
            closing_lines = []
            for namespace in reversed(namespaces[-missing_braces:]):
                closing_lines.append(f"}} // namespace {namespace}")
            
            # Add empty line for readability
            closing_lines.append('')
            
            # Insert before #endif
            lines[endif_line:endif_line] = closing_lines
            content = '\n'.join(lines)
            
            self.logger.info(f"  🔧 Fixed {missing_braces} missing namespace closing brace(s)")
        
        return content
    
    def _validate_and_fix_implementation(self, implementation_content: str) -> str:
        """Validate and fix implementation content to ensure it's syntactically correct."""
        if not implementation_content or implementation_content.strip() == '':
            return self._generate_minimal_implementation()
        
        # Check if implementation contains header content (common parsing error)
        if '#ifndef' in implementation_content and '#endif' in implementation_content:
            # This is header content in implementation file, extract the actual implementation
            lines = implementation_content.split('\n')
            implementation_lines = []
            in_implementation = False
            
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('#include "') or stripped.startswith('#include <'):
                    implementation_lines.append(line)
                elif stripped.startswith('namespace') or stripped.startswith('std::'):
                    implementation_lines.append(line)
                    in_implementation = True
                elif in_implementation:
                    implementation_lines.append(line)
            
            implementation_content = '\n'.join(implementation_lines)
        
        # This old validation is no longer needed - we handle includes properly now
        # Implementation files should have their own header included correctly
        
        return implementation_content
    
    def _add_type_aware_guidance(self, matlab_analysis: Dict[str, Any]) -> List[str]:
        """Add type-aware guidance to the generation prompt."""
        guidance = [
            "",
            "ARCHITECTURAL ANALYSIS AND LIBRARY SELECTION:",
            "=============================================",
            "",
            "CRITICAL DECISION: Choose optimal C++ approach based on MATLAB operations:",
            "",
            "ANALYZE MATLAB OPERATIONS IN CODE:",
            "- Matrix operations (multiplication, transpose, inverse) → Use Eigen library",
            "- Eigenvalue decomposition (eig) → Use Eigen::SelfAdjointEigenSolver",
            "- Complex linear algebra → Leverage optimized BLAS/LAPACK",
            "- 3D array processing → Consider memory-efficient data structures",
            "",
            "RECOMMENDED LIBRARY APPROACHES:",
            "",
            "OPTION 1: EIGEN LIBRARY (RECOMMENDED for this case):",
            "#include <Eigen/Dense>",
            "- MatrixXd for 2D matrices, VectorXd for vectors",
            "- Direct operations: A*B, A.transpose(), A.inverse()",
            "- Built-in eigenvalue solver: SelfAdjointEigenSolver",
            "- Memory efficient, cache-friendly, highly optimized",
            "",
            "OPTION 2: ARMADILLO LIBRARY:",
            "#include <armadillo>",
            "- mat/vec types with MATLAB-like syntax",
            "- Direct operations: A*B, A.t(), inv(A), eig_sym()",
            "- Good for MATLAB-to-C++ migration",
            "",
            "OPTION 3: MANUAL IMPLEMENTATION (NOT RECOMMENDED):",
            "- std::vector<std::vector<double>> approach",
            "- Manual loops for all operations",
            "- Poor performance, error-prone, memory inefficient",
            "",
            "PERFORMANCE ANALYSIS:",
            "- Eigen: 10-100x faster than manual implementation",
            "- Armadillo: 5-50x faster than manual implementation", 
            "- Manual: Slow, buggy, not production-ready",
            "",
            "DECISION CRITERIA:",
            "1. If MATLAB code has matrix operations → Use Eigen",
            "2. If MATLAB code has linear algebra → Use Eigen",
            "3. If MATLAB code has eigenvalue problems → Use Eigen",
            "4. Only use manual implementation for simple scalar operations",
            "",
            "ANALYSIS TASK:",
            "1. Analyze the MATLAB code for matrix/linear algebra operations",
            "2. Choose the most appropriate C++ library (Eigen recommended)",
            "3. Map MATLAB operations to optimized library calls",
            "4. Generate high-performance C++ code using proper libraries",
            "",
            "CRITICAL REQUIREMENTS:",
            "- Use Eigen library for matrix operations (NOT manual std::vector)",
            "- Leverage optimized BLAS/LAPACK operations",
            "- Implement efficient 3D array processing",
            "- Ensure type safety and avoid manual matrix implementations",
            "- Use size_t for loop indices to avoid signed/unsigned warnings",
            "",
            "GENERATE COMPLETE C++ CODE WITH:",
            "- Proper type declarations and operations",
            "- Helper functions for complex operations",
            "- Comprehensive comments explaining the conversion",
            "- Maintained mathematical correctness of the original MATLAB algorithm",
            ""
        ]
        return guidance
    
    def _analyze_matlab_operations(self, matlab_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze MATLAB operations to recommend optimal C++ library approach."""
        # Get source code from multiple possible locations
        source_code = self._extract_source_code(matlab_analysis)
        
        self.logger.info(f"DEBUG: Analyzing MATLAB operations in source code (length: {len(source_code)})")
        self.logger.info(f"DEBUG: Source code preview: {source_code[:200]}...")
        
        # Perform domain-specific analysis
        domain_analysis = self.domain_analyzer.analyze_domains(source_code)
        
        # Get library recommendations based on detected domains
        library_recommendations = self.domain_analyzer.get_library_recommendations(domain_analysis)
        
        # Select optimal library based on domain analysis
        recommended_library, reason, performance_gain = self.domain_analyzer.select_optimal_library(
            library_recommendations, domain_analysis
        )
        
        # Detect 3D arrays and update recommendations if needed
        has_3d_arrays = self.domain_analyzer.detect_3d_arrays(source_code)
        
        if has_3d_arrays and recommended_library == 'eigen':
            # Update Eigen recommendation to include Tensor support
            if 'eigen' in library_recommendations:
                library_recommendations['eigen']['priority'] = 'critical'
                library_recommendations['eigen']['reason'] += ' (includes 3D Tensor support)'
                if '<unsupported/Eigen/CXX11/Tensor>' not in library_recommendations['eigen']['headers']:
                    library_recommendations['eigen']['headers'].append('<unsupported/Eigen/CXX11/Tensor>')
                library_recommendations['eigen']['types'].extend(['Tensor<double, 3>', 'TensorMap<Tensor<double, 3>>'])
        
        return {
            "domain_analysis": domain_analysis,
            "library_recommendations": library_recommendations,
            "recommended_library": recommended_library,
            "reason": reason,
            "performance_gain": performance_gain,
            "source_code_length": len(source_code),
            "has_3d_arrays": has_3d_arrays
        }
    
    def _extract_source_code(self, matlab_analysis: Dict[str, Any]) -> str:
        """Extract source code from MATLAB analysis with fallback strategies."""
        # Try multiple extraction strategies
        if 'source_code' in matlab_analysis:
            return matlab_analysis['source_code']
        
        # Try file_analyses structure
        if 'file_analyses' in matlab_analysis:
            file_analyses = matlab_analysis['file_analyses']
            if isinstance(file_analyses, dict):
                for file_data in file_analyses.values():
                    if isinstance(file_data, dict):
                        if 'source_code' in file_data:
                            return file_data['source_code']
                        elif 'content' in file_data:
                            return file_data['content']
            elif isinstance(file_analyses, list):
                for analysis in file_analyses:
                    if isinstance(analysis, dict):
                        if 'source_code' in analysis:
                            return analysis['source_code']
                        elif 'content' in analysis:
                            return analysis['content']
        
        # Try analysis_results structure
        if 'analysis_results' in matlab_analysis:
            analysis_results = matlab_analysis['analysis_results']
            if isinstance(analysis_results, dict) and 'source_code' in analysis_results:
                return analysis_results['source_code']
        
        return ''
    
    
    def _generate_minimal_header(self, filename: str = "generated") -> str:
        """Generate a minimal valid header file."""
        guard_name = f"{filename.upper().replace('.', '_')}_H"
        return f'''#ifndef {guard_name}
#define {guard_name}

#include <vector>
#include <string>
#include <Eigen/Dense>

namespace {filename} {{
    // Function declarations will be added here
}} // namespace {filename}

#endif // {guard_name}'''
    
    def _generate_minimal_implementation(self, filename: str = "generated") -> str:
        """Generate a minimal valid implementation file."""
        return f'''#include "{filename}.h"
#include <iostream>

namespace {filename} {{
    // Function implementations will be added here
}} // namespace {filename}'''
    
    def _build_enhanced_generation_prompt(self, conversion_plan: Dict[str, Any],
                                        matlab_analysis: Dict[str, Any],
                                        strategy: str,
                                        iteration: int) -> str:
        """Build enhanced generation prompt based on strategy and iteration."""
        prompt_parts = []
        
        # Base prompt
        prompt_parts.append("Convert the following MATLAB code to C++ with high quality and correctness.")
        
        # Always analyze MATLAB operations for library recommendations
        operation_analysis = self._analyze_matlab_operations(matlab_analysis)
        self.logger.info(f"MATLAB Operation Analysis: {operation_analysis}")
        
        # Strategy-specific guidance
        if "CRITICAL ERROR ANALYSIS AND FIXING REQUIRED" in strategy:
            # This is an LLM-powered error fix prompt
            prompt_parts.append(f"\n{strategy}")
        elif "CRITICAL CODE GENERATION FIXES REQUIRED" in strategy:
            # This is a legacy error fix prompt
            prompt_parts.append(f"\n{strategy}")
        elif "IMPROVEMENT REQUIREMENTS:" in strategy:
            # This is an enhanced strategy from log analysis
            prompt_parts.append(f"\nENHANCED STRATEGY WITH FEEDBACK:\n{strategy}")
        else:
            # Add architectural analysis to prompt
            domain_analysis = operation_analysis.get('domain_analysis', {})
            library_recommendations = operation_analysis.get('library_recommendations', {})
            
            prompt_parts.extend([
                "",
                "MATLAB DOMAIN ANALYSIS RESULTS:",
                "===============================",
                f"Primary Domain: {domain_analysis.get('primary_domain', 'general')}",
                f"Complex Operations: {domain_analysis.get('has_complex_operations', False)}",
                f"Parallel Operations: {domain_analysis.get('has_parallel_operations', False)}",
                "",
                "DOMAIN SCORES:",
                *[f"- {domain}: {score}" for domain, score in domain_analysis.get('domain_scores', {}).items() if score > 0],
                "",
                f"Recommended Library: {operation_analysis['recommended_library']}",
                f"Reason: {operation_analysis['reason']}",
                f"Performance Gain: {operation_analysis['performance_gain']}",
                "",
                "AVAILABLE LIBRARY OPTIONS:",
                *[f"- {lib}: {info.get('reason', 'No description')} (Priority: {info.get('priority', 'unknown')})" 
                  for lib, info in library_recommendations.items()],
                ""
            ])
            
            # Add type-aware guidance for regular generation
            prompt_parts.extend(self._add_type_aware_guidance(matlab_analysis))
            strategy_info = self.generation_strategies.get(strategy, {})
            if strategy_info:
                prompt_parts.append(f"\nSTRATEGY: {strategy_info['description']}")
                prompt_parts.append(f"Use features: {', '.join(strategy_info['features'])}")
        
        # Iteration-specific guidance
        if iteration > 1:
            prompt_parts.append(f"\nIMPORTANT: This is iteration {iteration}. Focus on fixing compilation errors and improving code quality.")
            prompt_parts.append("Pay special attention to:")
            prompt_parts.append("- Correct include statements")
            prompt_parts.append("- Proper template syntax")
            prompt_parts.append("- Namespace usage")
            prompt_parts.append("- Type conversions")
        
        # Multi-file specific guidance
        if conversion_plan.get('project_structure', {}).get('is_multi_file', False):
            prompt_parts.append("\nMULTI-FILE PROJECT GUIDANCE:")
            prompt_parts.append("- Coordinate includes across files")
            prompt_parts.append("- Use consistent namespace strategy")
            prompt_parts.append("- Ensure proper header dependencies")
            prompt_parts.append("- Follow compilation order")
        
        # MATLAB code
        prompt_parts.append("\nMATLAB CODE TO CONVERT:")
        file_analyses = matlab_analysis.get('file_analyses', [])
        for analysis in file_analyses:
            prompt_parts.append(f"\n--- {analysis['file_name']} ---")
            prompt_parts.append(analysis.get('content', ''))
        
        # Conversion plan details
        prompt_parts.append("\nCONVERSION PLAN:")
        prompt_parts.append(f"- Conversion mode: {conversion_plan.get('conversion_mode', 'unknown')}")
        prompt_parts.append(f"- Namespace strategy: {conversion_plan.get('namespace_strategy', {}).get('selected_strategy', 'unknown')}")
        
        file_organization = conversion_plan.get('file_organization', {})
        if file_organization:
            prompt_parts.append(f"- C++ files: {file_organization.get('cpp_files', [])}")
            prompt_parts.append(f"- Header files: {file_organization.get('header_files', [])}")
        
        # Include dependencies
        include_deps = conversion_plan.get('include_dependencies', {})
        if include_deps:
            prompt_parts.append("\nINCLUDE DEPENDENCIES:")
            for file, includes in include_deps.items():
                prompt_parts.append(f"{file}: {includes}")
        
        # Enhanced output format instructions for general LLM code generation
        prompt_parts.append("\nCRITICAL OUTPUT FORMAT REQUIREMENTS:")
        prompt_parts.append("You MUST provide the C++ code in exactly two code blocks:")
        prompt_parts.append("1. First code block: ```cpp (header file content)")
        prompt_parts.append("2. Second code block: ```cpp (implementation file content)")
        prompt_parts.append("")
        prompt_parts.append("IMPORTANT - CODE GENERATION WITH GENERAL LLM:")
        prompt_parts.append("- You may use <think> tags for internal reasoning if needed")
        prompt_parts.append("- But provide the actual C++ code in the required code blocks")
        prompt_parts.append("- The <think> content will be filtered out during extraction")
        prompt_parts.append("- Focus on generating correct, compilable C++ code")
        prompt_parts.append("- Start with the first code block after any thinking")
        prompt_parts.append("")
        prompt_parts.append("REQUIRED STRUCTURE (NO EXCEPTIONS):")
        prompt_parts.append("```cpp")
        prompt_parts.append("// Header file content here")
        prompt_parts.append("```")
        prompt_parts.append("")
        prompt_parts.append("```cpp")
        prompt_parts.append("// Implementation file content here")
        prompt_parts.append("```")
        prompt_parts.append("")
        prompt_parts.append("START YOUR RESPONSE IMMEDIATELY WITH THE FIRST ```cpp BLOCK")
        prompt_parts.append("")
        prompt_parts.append("VALIDATION CHECKLIST - Before generating code, ensure:")
        prompt_parts.append("✅ All #include statements are complete and correct")
        prompt_parts.append("✅ Header guard names match the actual filename")
        prompt_parts.append("✅ All template syntax is properly closed (<> not >>)")
        prompt_parts.append("✅ All statements end with semicolons")
        prompt_parts.append("✅ All loops have complete conditions and increments")
        prompt_parts.append("✅ All function calls have proper parentheses")
        prompt_parts.append("✅ All variable declarations have proper types")
        prompt_parts.append("✅ All code is inside proper functions or classes (NO code at file scope)")
        prompt_parts.append("✅ All Eigen types use proper namespace (Eigen::MatrixXd, NOT just MatrixXd)")
        prompt_parts.append("✅ Function signatures match between header and implementation")
        prompt_parts.append(f"✅ Use the recommended library: {operation_analysis.get('recommended_library', 'Eigen')}")
        prompt_parts.append(f"✅ Target performance: {operation_analysis.get('performance_gain', '10-100x faster')}")
        
        # Add type alias guidance based on code complexity
        code_complexity_threshold = 16000  # 16k characters
        estimated_code_length = len(str(matlab_analysis)) + len(str(conversion_plan))
        
        if estimated_code_length > code_complexity_threshold:
            prompt_parts.extend([
                "",
                "CODE ORGANIZATION GUIDANCE:",
                "- Code is expected to be lengthy (>16k chars)",
                "- Use type aliases for complex types to improve readability",
                "- Example: using Tensor3d = Eigen::Tensor<double, 3>;",
                "- Apply aliases consistently throughout header and implementation",
            ])
        else:
            prompt_parts.extend([
                "",
                "CODE ORGANIZATION GUIDANCE:",
                "- Code is relatively short, direct type usage is acceptable",
                "- Type aliases optional but recommended for clarity",
                "- If using aliases, define them in the header file",
            ])
        
        # Add library-specific guidance
        recommended_library = operation_analysis.get('recommended_library', 'eigen')
        has_3d_arrays = operation_analysis.get('has_3d_arrays', False)
        
        if recommended_library == 'eigen':
            prompt_parts.extend([
                "",
                "EIGEN LIBRARY GUIDANCE:",
                "- Use #include <Eigen/Dense> for matrix operations",
                "- Use MatrixXd for 2D matrices, VectorXd for vectors",
                "- Use ArrayXXd for 2D arrays, ArrayXd for 1D arrays",
                "- CRITICAL: ArrayXXXd does NOT exist - use ArrayXXd or ArrayXd",
                "- CRITICAL: MatrixXXXd does NOT exist - use MatrixXd",
                "- CRITICAL: VectorXXXd does NOT exist - use VectorXd",
                "- Use .rows() and .cols() for matrix dimensions",
                "- Use .size() for vector length",
                "- Direct operations: A*B, A.transpose(), A.inverse()",
                "- Use SelfAdjointEigenSolver for eigenvalue decomposition",
                "- NEVER use 'ArrayXXXd', 'MatrixXXXd', or 'VectorXXXd' - these types don't exist!",
            ])
            
            if has_3d_arrays:
                prompt_parts.extend([
                    "",
                    "EIGEN TENSOR FOR 3D ARRAYS (CRITICAL):",
                    "- Use #include <unsupported/Eigen/CXX11/Tensor>",
                    "- Use Eigen::Tensor<double, 3> for 3D arrays/tensors",
                    "- Access dimensions: tensor.dimension(0), tensor.dimension(1), tensor.dimension(2)",
                    "- Access elements: tensor(i, j, k) - NOT tensor[i][j][k]",
                    "- NEVER use Array3D or Array3d for 3D arrays",
                    "- Array3d is a 1D array with 3 elements, NOT a 3D array!",
                    "",
                    "TENSOR OPERATIONS:",
                    "- Extract to vector: for (int k = 0; k < depth; k++) vec(k) = tensor(i, j, k);",
                    "- Write from vector: for (int k = 0; k < depth; k++) tensor(i, j, k) = vec(k);",
                    "- Get dimensions: int rows = tensor.dimension(0);",
                    "- Tensor is pass-by-reference: void func(Eigen::Tensor<double, 3>& data)",
                    "",
                    "EXAMPLE 3D TENSOR USAGE (COMPLETE FUNCTION):",
                    "```cpp",
                    "// Header file",
                    "#ifndef EXAMPLE_H",
                    "#define EXAMPLE_H",
                    "",
                    "#include <Eigen/Dense>",
                    "#include <unsupported/Eigen/CXX11/Tensor>",
                    "",
                    "namespace example {",
                    "    void process_3d(Eigen::Tensor<double, 3>& data, int p, int its);",
                    "}",
                    "",
                    "#endif // EXAMPLE_H",
                    "```",
                    "",
                    "```cpp",
                    "// Implementation file",
                    "#include \"example.h\"",
                    "",
                    "namespace example {",
                    "",
                    "void process_3d(Eigen::Tensor<double, 3>& data, int p, int its) {",
                    "    const int rows = data.dimension(0);",
                    "    const int cols = data.dimension(1);",
                    "    const int depth = data.dimension(2);",
                    "    ",
                    "    for (int i = 0; i < rows; i++) {",
                    "        for (int j = 0; j < cols; j++) {",
                    "            // Extract slice to Eigen vector",
                    "            Eigen::VectorXd slice(depth);",
                    "            for (int k = 0; k < depth; k++) {",
                    "                slice(k) = data(i, j, k);",
                    "            }",
                    "            ",
                    "            // Process with Eigen operations",
                    "            Eigen::VectorXd result = slice.array() * 2.0;",
                    "            ",
                    "            // Write back",
                    "            for (int k = 0; k < depth; k++) {",
                    "                data(i, j, k) = result(k);",
                    "            }",
                    "        }",
                    "    }",
                    "}",
                    "",
                    "} // namespace example",
                    "```",
                    "",
                    "CRITICAL: Generate COMPLETE header and implementation files, NOT code snippets!",
                ])
            else:
                prompt_parts.append("- Avoid manual std::vector<std::vector<double>> implementations")
        
        elif recommended_library == 'armadillo':
            prompt_parts.extend([
                "",
                "ARMADILLO LIBRARY GUIDANCE:",
                "- Use #include <armadillo> for MATLAB-like syntax",
                "- Use mat for matrices, vec for vectors",
                "- Direct operations: A*B, A.t(), inv(A), eig_sym()",
                "- Good for MATLAB-to-C++ migration"
            ])
        elif recommended_library == 'opencv':
            prompt_parts.extend([
                "",
                "OPENCV LIBRARY GUIDANCE:",
                "- Use #include <opencv2/opencv.hpp> for image processing",
                "- Use cv::Mat for matrices, cv::Scalar for values",
                "- Built-in image processing functions available"
            ])
        
        # Add critical MATLAB vs C++ differences (PRIORITY 1 FIX - applies to ALL libraries)
        prompt_parts.extend([
            "",
            "═══════════════════════════════════════════════════════════════",
            "CRITICAL: MATLAB vs C++ DIFFERENCES (MUST FOLLOW!)",
            "═══════════════════════════════════════════════════════════════",
            "",
            "1. TYPE CASTING IN std::min/max (CRITICAL!):",
            "   ❌ WRONG:  std::min(static_cast<int>(a), matrix.rows())",
            "   ✅ CORRECT: std::min(static_cast<int>(a), static_cast<int>(matrix.rows()))",
            "   ",
            "   RULE: BOTH arguments must have IDENTICAL types!",
            "   - .rows(), .cols(), .size() return Eigen::Index (long int)",
            "   - Cast BOTH sides when mixing with int:",
            "     std::min(static_cast<int>(x), static_cast<int>(y))",
            "   - Or use: auto x = static_cast<int>(matrix.rows());",
            "",
            "2. std::min/max ONLY TAKES 2 ARGUMENTS:",
            "   ❌ WRONG:  double val = std::min(a, b, c, d);",
            "   ✅ CORRECT: double val = std::min({a, b, c, d});  // C++11 initializer list",
            "   ✅ CORRECT: double val = std::min(std::min(a, b), std::min(c, d));",
            "   ",
            "   MATLAB min([a,b,c,d]) ≠ C++ std::min(a,b,c,d)",
            "",
            "3. BOOLEAN ARRAY TO DOUBLE CONVERSION:",
            "   ❌ WRONG:  Eigen::MatrixXd result = (A.array() < threshold);",
            "   ✅ CORRECT: Eigen::ArrayXXd result = (A.array() < threshold).cast<double>();",
            "   ",
            "   MATLAB auto-converts logical to double, C++ requires .cast<double>()",
            "   Also: Use ArrayXXd for element-wise ops, MatrixXd for matrix ops",
            "",
            "4. EIGEN MATRIX VS TENSOR API:",
            "   MatrixXd (2D):              Tensor<double, 3> (3D):",
            "   - .rows(), .cols()          - .dimension(0), .dimension(1), .dimension(2)",
            "   - mat(i, j)                 - tensor(i, j, k)",
            "   - No .dimension() method!   - No .rows()/.cols() methods!",
            "   ",
            "   ❌ WRONG: Fy.dimension(0)  // If Fy is MatrixXd",
            "   ✅ CORRECT: Fy.rows()      // For 2D matrices",
            "",
            "5. EIGEN MATRIX INITIALIZATION:",
            "   ❌ WRONG:  Eigen::MatrixXd T = Eigen::MatrixXd::Zero(r, c) - 1.0;",
            "   ✅ CORRECT: Eigen::MatrixXd T = Eigen::MatrixXd::Constant(r, c, -1.0);",
            "   ✅ CORRECT: Eigen::MatrixXd T(r, c); T.setConstant(-1.0);",
            "   ",
            "   Eigen doesn't support Python-style broadcasting on lazy expressions",
            "   For arithmetic with scalars, use .array() or Constant()",
            "",
            "6. FUNCTION DECLARATIONS (CRITICAL!):",
            "   ❌ WRONG: Assume function exists without declaring it",
            "   ✅ CORRECT: Declare ALL functions before calling them",
            "   ",
            "   If you call a function, YOU MUST:",
            "   - Declare it in the header file",
            "   - Implement it in the cpp file",
            "   - OR use a standard library function",
            "   ",
            "   Common MATLAB functions that DON'T exist in C++:",
            "   - imdilate() → Use OpenCV cv::dilate() or implement manually",
            "   - roots()    → Use Eigen::PolynomialSolver or implement",
            "   - conv()     → Implement convolution manually",
            "   - fft()      → Use FFTW library or Eigen FFT module",
            "",
            "7. STANDARD LIBRARY NAMESPACES:",
            "   ❌ WRONG: min(a, b)              // Ambiguous",
            "   ✅ CORRECT: std::min(a, b)       // Explicit std::",
            "   ",
            "   ❌ WRONG: sort(begin(vec), end(vec))",
            "   ✅ CORRECT: std::sort(std::begin(vec), std::end(vec))",
            "   ",
            "   Always use std:: prefix for standard library functions",
            "",
            "8. EIGEN TYPES & METHODS - WHAT EXISTS AND WHAT DOESN'T:",
            "   ✅ EXISTS:                  ❌ DOES NOT EXIST:",
            "   - Eigen::MatrixXd           - Eigen::MatrixXXXd",
            "   - Eigen::VectorXd           - Eigen::VectorXXXd",
            "   - Eigen::ArrayXXd           - Eigen::ArrayXXXd",
            "   - Eigen::ArrayXd            - Eigen::Array3D (use Tensor)",
            "   - Eigen::Tensor<double, 3>  - Eigen::Array3d<double>",
            "   ",
            "   EIGEN METHODS THAT DO NOT EXIST (FIX #3):",
            "   ❌ MatrixXd::Zero(a, b, c)    → Use Tensor<double,3> or Zero(a,b)",
            "   ❌ matrix.dimension()          → Use .rows()/.cols() for MatrixXd",
            "   ❌ matrix.tensor<T>()          → No conversion, use TensorMap",
            "   ❌ Tensor::Zero()              → Use .setZero() instead",
            "   ❌ tensor.rows()/cols()        → Use .dimension(i) for Tensor",
            "   ❌ array.hasNonZero()          → Use (array != 0).any()",
            "   ❌ matrix.slice()              → Use .block() or .segment()",
            "   ❌ matrix.flatten()            → Use .reshaped() or manual loop",
            "   ",
            "   MEMORIZE: No 'XXX' in type names, no fictional methods!",
            "",
            "9. RETURN TYPE CONSISTENCY:",
            "   If function signature says Eigen::MatrixXd:",
            "   ❌ WRONG: std::vector<std::vector<double>> result = func();",
            "   ✅ CORRECT: Eigen::MatrixXd result = func();",
            "   ",
            "   Match the return type EXACTLY across:",
            "   - Header declaration",
            "   - Implementation",
            "   - Function calls",
            "",
            "10. FUNCTION NAMING CONSISTENCY:",
            "    Use ONE convention (snake_case OR camelCase), not both:",
            "    ✅ CORRECT: namespace get_distance { void get_distance(...); }",
            "    ❌ WRONG:   namespace get_distance { void getDistance(...); }",
            "    ",
            "    Match namespace name with function name for consistency",
            "",
            "═══════════════════════════════════════════════════════════════",
            "BEFORE GENERATING CODE, ASK YOURSELF:",
            "═══════════════════════════════════════════════════════════════",
            "1. Did I cast BOTH arguments in std::min/max?",
            "2. Did I use std::min({a,b,c}) for more than 2 arguments?",
            "3. Did I add .cast<double>() after boolean comparisons?",
            "4. Did I use .rows()/.cols() for MatrixXd (not .dimension())?",
            "5. Did I use .dimension(i) for Tensor (not .rows()/.cols())?",
            "6. Did I declare ALL helper functions in the header?",
            "7. Did I use Constant() or .array() for scalar arithmetic?",
            "8. Did I match return types across header/impl/calls?",
            "9. Did I check no 'XXX' in Eigen type names?",
            "10. Did I use consistent naming (snake_case or camelCase)?",
            "",
            "If ANY answer is NO, FIX IT BEFORE PROCEEDING!",
            "═══════════════════════════════════════════════════════════════",
            "",
        ])
        
        prompt_parts.append("")
        prompt_parts.append("❌ DO NOT generate step-by-step instructions or explanations as code.")
        prompt_parts.append("❌ DO NOT generate numbered lists or bullet points as code.")
        prompt_parts.append("❌ DO NOT generate comments that explain what to do.")
        prompt_parts.append("")
        prompt_parts.append("✅ DO generate actual, compilable C++ code.")
        prompt_parts.append("✅ DO include proper #include statements.")
        prompt_parts.append("✅ DO include proper function definitions.")
        prompt_parts.append("✅ DO include proper class/namespace declarations.")
        prompt_parts.append("")
        prompt_parts.append("EXAMPLE OF CORRECT OUTPUT:")
        prompt_parts.append("```cpp")
        prompt_parts.append("#include <Eigen/Dense>")
        prompt_parts.append("#include <vector>")
        prompt_parts.append("")
        prompt_parts.append("namespace arma_filter {")
        prompt_parts.append("    void processMatrix(const Eigen::MatrixXd& input);")
        prompt_parts.append("}")
        prompt_parts.append("```")
        prompt_parts.append("")
        prompt_parts.append("```cpp")
        prompt_parts.append("#include \"arma_filter.h\"")
        prompt_parts.append("")
        prompt_parts.append("namespace arma_filter {")
        prompt_parts.append("    void processMatrix(const Eigen::MatrixXd& input) {")
        prompt_parts.append("        // Actual implementation here")
        prompt_parts.append("    }")
        prompt_parts.append("}")
        prompt_parts.append("```")
        
        # PREVENTIVE GUIDANCE: DISABLED - Causes LLM to make systematic syntax errors
        # Evidence: A/B testing shows 100% failure rate when enabled (2/2 tests)
        # Issue: LLM generates extra '}' tokens when guidance is present
        # See: CRITICAL_FINDING_PREVENTIVE_GUIDANCE_ISSUE.md
        # Reactive knowledge (during error fixing) is still enabled and working!
        # try:
        #     preventive_guidance = self.api_knowledge_base.get_preventive_guidance(
        #         libraries_used=['eigen']
        #     )
        #     if preventive_guidance:
        #         prompt_parts.append("")
        #         prompt_parts.append("=" * 80)
        #         prompt_parts.append("📚 API KNOWLEDGE BASE - CRITICAL RULES TO PREVENT ERRORS")
        #         prompt_parts.append("=" * 80)
        #         prompt_parts.append(preventive_guidance)
        #         prompt_parts.append("=" * 80)
        #         self.logger.info("✅ Preventive guidance injected into generation prompt")
        # except Exception as e:
        #     self.logger.warning(f"⚠️ Failed to get preventive guidance: {e}")
        
        return "\n".join(prompt_parts)
    
    def _validate_generated_code(self, generated_code: Dict[str, Any]) -> Dict[str, Any]:
        """Validate generated code for basic syntax and structure issues."""
        validation_errors = []
        
        if not generated_code:
            validation_errors.append("Generated code is empty or None")
            return {'valid': False, 'errors': validation_errors}
        
        if 'files' not in generated_code:
            validation_errors.append("Generated code missing 'files' structure")
            return {'valid': False, 'errors': validation_errors}
        
        files = generated_code['files']
        if not files:
            validation_errors.append("Generated code has empty files dictionary")
            return {'valid': False, 'errors': validation_errors}
        
        # Check each file
        for filename, content in files.items():
            if not content or not content.strip():
                validation_errors.append(f"File {filename} is empty")
                continue
            
            # Check for basic C++ syntax issues
            if filename.endswith('.h') or filename.endswith('.hpp'):
                if not any(keyword in content for keyword in ['#ifndef', '#define', '#pragma once']):
                    validation_errors.append(f"Header file {filename} missing include guards")
                
                if 'Eigen::' in content or 'MatrixXd' in content or 'VectorXd' in content:
                    if '#include <Eigen/' not in content and '#include "Eigen/' not in content:
                        validation_errors.append(f"Header file {filename} uses Eigen types but missing Eigen include")
            
            elif filename.endswith('.cpp') or filename.endswith('.cc'):
                if 'Eigen::' in content or 'MatrixXd' in content or 'VectorXd' in content:
                    if '#include <Eigen/' not in content and '#include "Eigen/' not in content:
                        validation_errors.append(f"Implementation file {filename} uses Eigen types but missing Eigen include")
            
            # Check for Tensor usage and includes
            if 'Eigen::Tensor' in content:
                if '#include <unsupported/Eigen/CXX11/Tensor>' not in content:
                    validation_errors.append(
                        f"File {filename} uses Eigen::Tensor but missing "
                        "#include <unsupported/Eigen/CXX11/Tensor>"
                    )
            
            # Check for incorrect 3D array types
            import re
            incorrect_types = [
                (r'Eigen::Array3D', 'Array3D does not exist'),
                (r'Eigen::Array3d\s*<', 'Array3d is not a template (it\'s a 1D array)'),
                (r'Eigen::ArrayXXXd', 'ArrayXXXd does not exist'),
            ]
            for pattern, message in incorrect_types:
                if re.search(pattern, content):
                    validation_errors.append(f"File {filename}: {message}")
        
        return {
            'valid': len(validation_errors) == 0,
            'errors': validation_errors
        }
    
    def _fix_validation_issues(self, generated_code: Dict[str, Any], validation_errors: List[str]) -> Dict[str, Any]:
        """Fix validation issues in generated code."""
        if not generated_code or 'files' not in generated_code:
            return generated_code
        
        fixed_code = generated_code.copy()
        files = generated_code['files']
        fixed_files = {}
        
        for filename, content in files.items():
            fixed_content = content
            
            # Fix missing Eigen includes
            if 'Eigen::' in content or 'MatrixXd' in content or 'VectorXd' in content:
                if '#include <Eigen/' not in content and '#include "Eigen/' not in content:
                    # Add Eigen include at the top
                    lines = content.split('\n')
                    include_line = '#include <Eigen/Dense>'
                    
                    # Find the right place to insert the include
                    insert_index = 0
                    for i, line in enumerate(lines):
                        if line.strip().startswith('#include'):
                            insert_index = i + 1
                        elif line.strip() and not line.strip().startswith('//'):
                            break
                    
                    lines.insert(insert_index, include_line)
                    if insert_index < len(lines) - 1:
                        lines.insert(insert_index + 1, '')
                    
                    fixed_content = '\n'.join(lines)
                    self.logger.info(f"Fixed missing Eigen include in {filename}")
            
            # Fix missing include guards for header files
            if filename.endswith('.h') or filename.endswith('.hpp'):
                if not any(keyword in fixed_content for keyword in ['#ifndef', '#define', '#pragma once']):
                    lines = fixed_content.split('\n')
                    guard_name = filename.upper().replace('.', '_').replace('/', '_')
                    
                    # Add include guards at the beginning
                    guard_lines = [
                        f'#ifndef {guard_name}',
                        f'#define {guard_name}',
                        ''
                    ]
                    
                    # Add closing guard at the end
                    lines.extend(['', f'#endif // {guard_name}'])
                    
                    # Insert opening guards at the beginning
                    lines = guard_lines + lines
                    
                    fixed_content = '\n'.join(lines)
                    self.logger.info(f"Fixed missing include guards in {filename}")
            
            # Fix Tensor includes
            if 'Eigen::Tensor' in fixed_content:
                if '#include <unsupported/Eigen/CXX11/Tensor>' not in fixed_content:
                    lines = fixed_content.split('\n')
                    insert_index = 0
                    
                    # Find where to insert (after Eigen/Dense if present)
                    for i, line in enumerate(lines):
                        if '#include <Eigen/Dense>' in line:
                            insert_index = i + 1
                            break
                        elif line.strip().startswith('#include'):
                            insert_index = i + 1
                    
                    lines.insert(insert_index, '#include <unsupported/Eigen/CXX11/Tensor>')
                    if insert_index < len(lines) - 1:
                        lines.insert(insert_index + 1, '')
                    
                    fixed_content = '\n'.join(lines)
                    self.logger.info(f"Fixed missing Eigen Tensor include in {filename}")
            
            fixed_files[filename] = fixed_content
        
        fixed_code['files'] = fixed_files
        return fixed_code
    
    def _parse_execution_from_compilation_logs(self, compilation_result: Any) -> Any:
        """Parse execution test results from compilation logs.
        
        The execution test now runs in the same container as compilation,
        so we need to parse its output from the compilation logs.
        """
        from ....infrastructure.testing.types import ExecutionResult
        
        output = compilation_result.output
        
        # Look for execution markers in the logs
        if "EXECUTION TEST START" in output:
            # Extract execution output
            parts = output.split("EXECUTION TEST START")
            if len(parts) > 1:
                execution_section = parts[1]
                
                # Check if execution passed or failed
                if "EXECUTION TEST PASSED" in execution_section:
                    # Extract stdout (everything between START and PASSED)
                    stdout = execution_section.split("EXECUTION TEST PASSED")[0].strip()
                    
                    self.logger.info(f"✅ Execution test PASSED")
                    self.logger.info(f"Execution output:\n{stdout}")
                    
                    return ExecutionResult(
                        success=True,
                        output=stdout,
                        errors=[],
                        execution_time=0.0,
                        return_code=0,
                        stdout=stdout,
                        stderr=""
                    )
                elif "EXECUTION TEST FAILED" in execution_section:
                    # Extract exit code
                    import re
                    exit_code_match = re.search(r'exit code: (\d+)', execution_section)
                    exit_code = int(exit_code_match.group(1)) if exit_code_match else 1
                    
                    stdout = execution_section.split("EXECUTION TEST FAILED")[0].strip()
                    
                    self.logger.warning(f"❌ Execution test FAILED (exit code: {exit_code})")
                    self.logger.info(f"Execution output:\n{stdout}")
                    
                    return ExecutionResult(
                        success=False,
                        output=stdout,
                        errors=[f"Execution failed with exit code {exit_code}"],
                        execution_time=0.0,
                        return_code=exit_code,
                        stdout=stdout,
                        stderr=""
                    )
        
        # No execution test found in logs (library without main())
        self.logger.info("ℹ️  No execution test found in logs (library compilation)")
        return ExecutionResult(
            success=False,
            output="No execution test performed (library without main function)",
            errors=[],
            execution_time=0.0,
            return_code=-1,
            stdout="",
            stderr=""
        )
    
    async def _test_compilation(self, generated_code: Dict[str, Any], 
                              conversion_plan: Dict[str, Any],
                              state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Test compilation of generated code."""
        self.compilation_attempts += 1
        
        try:
            # Initialize compilation_manager with build_system from state if not already done
            if self.compilation_manager is None:
                build_system = state.get('build_system', 'gcc') if state else 'gcc'
                self.compilation_manager = CPPCompilationManager(build_system=build_system)
                self.logger.info(f"🔧 Initialized compilation_manager with build_system={build_system}")
            
            # Prepare files for compilation
            files_to_test = generated_code.get('files', {})
            if not files_to_test:
                return {'success': False, 'error': 'No files to compile'}
            
            # Check if Docker is available
            if not self.compilation_manager.docker_manager.is_image_available():
                self.logger.warning("Docker testing not available, skipping compilation test")
                return {'success': True, 'output': 'Docker not available, compilation skipped', 'errors': []}
            
            # Test compilation using compilation manager
            project_name = conversion_plan.get('project_name', 'test_project')
            self.logger.info(f"Testing compilation for project: {project_name}")
            self.logger.info(f"Files to compile: {list(files_to_test.keys())}")
            
            compilation_result = self.compilation_manager.compile_project(
                files_to_test, project_name
            )
            
            self.logger.info(f"Compilation result: success={compilation_result.success}")
            if compilation_result.errors:
                self.logger.info(f"Compilation errors: {compilation_result.errors}")
            
            # Parse execution results from compilation logs
            # (Execution now happens IN THE SAME CONTAINER as compilation)
            execution_result = None
            if compilation_result.success and compilation_result.binary_path:
                self.logger.info(f"✅ Compilation successful, parsing execution results from logs...")
                execution_result = self._parse_execution_from_compilation_logs(compilation_result)
            
            # Perform advanced log analysis if compilation failed
            log_analysis = None
            if not compilation_result.success and compilation_result.output:
                try:
                    log_analysis = await self.log_analyzer.analyze_compilation_logs(
                        compilation_result.output, 
                        files_to_test, 
                        project_name
                    )
                    self.logger.info(f"Log analysis complete: {log_analysis.total_errors} errors, {log_analysis.total_warnings} warnings")
                    
                    # Log improvement suggestions
                    if log_analysis.improvement_suggestions:
                        self.logger.info(f"Improvement suggestions: {log_analysis.improvement_suggestions[:3]}")
                    
                    # Log LLM prompt enhancements
                    if log_analysis.llm_prompt_enhancements:
                        self.logger.info(f"LLM prompt enhancements available: {len(log_analysis.llm_prompt_enhancements)}")
                        
                except Exception as e:
                    self.logger.warning(f"Failed to analyze compilation logs: {e}")
            
            if compilation_result.success:
                self.successful_compilations += 1
            
            # Build result dictionary
            result = {
                'success': compilation_result.success, 
                'output': compilation_result.output, 
                'errors': compilation_result.errors,
                'log_analysis': log_analysis
            }
            
            # Add execution result if available
            if execution_result:
                result['execution_result'] = {
                    'success': execution_result.success,
                    'exit_code': execution_result.return_code,  # Correct attribute name
                    'output': execution_result.output,
                    'execution_time': execution_result.execution_time,
                    'stdout': execution_result.stdout,
                    'stderr': execution_result.stderr
                }
                self.logger.info(f"✅ Execution result added to compilation result")
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Compilation testing failed: {e}, continuing without compilation test")
            return {'success': True, 'output': f'Compilation test failed: {e}', 'errors': []}
    
    def _analyze_compilation_errors(self, compilation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze compilation errors and categorize them."""
        # Use advanced log analysis if available
        log_analysis = compilation_result.get('log_analysis')
        if log_analysis:
            return {
                'categories': list(log_analysis.error_categories.keys()),
                'severity': 'high' if log_analysis.total_errors > 0 else 'low',
                'patterns': [
                    {
                        'category': error.error_type,
                        'pattern': error.error_message,
                        'severity': error.severity
                    } for error in log_analysis.errors
                ],
                'raw_errors': compilation_result.get('output', ''),
                'improvement_suggestions': log_analysis.improvement_suggestions,
                'llm_prompt_enhancements': log_analysis.llm_prompt_enhancements,
                'code_quality_issues': log_analysis.code_quality_issues
            }
        
        # Fallback to basic analysis
        error_output = compilation_result.get('error_output', '')
        if not error_output:
            return {'categories': [], 'severity': 'none', 'patterns': []}
        
        # Categorize errors
        error_categories = []
        detected_patterns = []
        
        for category, pattern_info in self.error_patterns.items():
            patterns = pattern_info['patterns']
            for pattern in patterns:
                if pattern.lower() in error_output.lower():
                    error_categories.append(category)
                    detected_patterns.append({
                        'category': category,
                        'pattern': pattern,
                        'severity': pattern_info['severity']
                    })
                    break
        
        # Determine overall severity
        if any(p['severity'] == 'high' for p in detected_patterns):
            severity = 'high'
        elif any(p['severity'] == 'medium' for p in detected_patterns):
            severity = 'medium'
        else:
            severity = 'low'
        
        return {
            'categories': error_categories,
            'severity': severity,
            'patterns': detected_patterns,
            'raw_errors': error_output
        }
    
    def _select_improvement_strategy(self, error_analysis: Dict[str, Any], 
                                   current_strategy: str) -> str:
        """Select improvement strategy based on error analysis."""
        categories = error_analysis.get('categories', [])
        severity = error_analysis.get('severity', 'low')
        
        if 'include_errors' in categories:
            return 'fix_includes'
        elif 'template_errors' in categories:
            return 'fix_templates'
        elif 'namespace_errors' in categories:
            return 'fix_namespace'
        elif 'type_errors' in categories:
            return 'fix_types'
        elif 'syntax_errors' in categories:
            return 'fix_syntax'
        elif severity == 'high':
            return 'conservative_approach'
        else:
            return 'refine_existing'
    
    def _enhance_strategy_with_errors(self, current_strategy: str,
                                    error_analysis: Dict[str, Any],
                                    improvement_strategy: str) -> str:
        """Enhance generation strategy based on error analysis."""
        # For now, return current strategy
        # In a full implementation, this would modify the strategy based on errors
        return current_strategy
    
    def _enhance_strategy_with_log_analysis(self, 
                                          current_strategy: str,
                                          log_analysis,
                                          error_analysis: Dict[str, Any]) -> str:
        """Enhance generation strategy using detailed log analysis."""
        
        # Start with current strategy
        enhanced_strategy = current_strategy
        
        # Add specific improvements based on log analysis
        if log_analysis.improvement_suggestions:
            enhanced_strategy += f"\n\nIMPROVEMENT REQUIREMENTS:\n"
            for i, suggestion in enumerate(log_analysis.improvement_suggestions[:3], 1):
                enhanced_strategy += f"{i}. {suggestion}\n"
        
        # Add LLM prompt enhancements
        if log_analysis.llm_prompt_enhancements:
            enhanced_strategy += f"\n\nCODE GENERATION GUIDELINES:\n"
            for i, enhancement in enumerate(log_analysis.llm_prompt_enhancements[:3], 1):
                enhanced_strategy += f"{i}. {enhancement}\n"
        
        # Add code quality fixes
        if log_analysis.code_quality_issues:
            enhanced_strategy += f"\n\nQUALITY IMPROVEMENTS NEEDED:\n"
            for i, issue in enumerate(log_analysis.code_quality_issues[:2], 1):
                enhanced_strategy += f"{i}. {issue}\n"
        
        # Add specific error corrections based on error categories
        error_categories = list(log_analysis.error_categories.keys())
        if error_categories:
            enhanced_strategy += f"\n\nSPECIFIC ERROR FIXES REQUIRED:\n"
            if 'include' in error_categories:
                enhanced_strategy += "- Fix all #include statements with proper syntax\n"
            if 'syntax' in error_categories:
                enhanced_strategy += "- Correct all syntax errors (semicolons, braces, parentheses)\n"
            if 'type' in error_categories:
                enhanced_strategy += "- Fix type mismatches and add explicit conversions\n"
            if 'linker' in error_categories:
                enhanced_strategy += "- Resolve linking issues and function definitions\n"
        
        return enhanced_strategy
    
    def _extract_header_from_implementation(self, implementation_content: str) -> str:
        """Extract header content from implementation code."""
        try:
            lines = implementation_content.split('\n')
            header_lines = []
            
            for line in lines:
                line = line.strip()
                # Include includes, namespace declarations, class declarations, etc.
                if (line.startswith('#include') or 
                    line.startswith('#ifndef') or 
                    line.startswith('#define') or 
                    line.startswith('namespace') or
                    line.startswith('class ') or
                    line.startswith('struct ') or
                    line.endswith(';') and ('(' not in line or ')' in line)):
                    header_lines.append(line)
                # Stop at function definitions (lines with {)
                elif '{' in line and not line.startswith('//'):
                    break
            
            return '\n'.join(header_lines)
        except Exception as e:
            self.logger.warning(f"Failed to extract header from implementation: {e}")
            return ""
    
    def _extract_implementation_from_response(self, response: str) -> str:
        """Extract implementation content from raw response."""
        try:
            # Look for C++ code patterns in the response
            lines = response.split('\n')
            code_lines = []
            in_code_section = False
            
            for line in lines:
                line = line.strip()
                
                # Start capturing when we see C++ patterns
                if any(pattern in line for pattern in ['#include', 'namespace', 'class ', 'struct ', 'int ', 'void ', 'double ']):
                    in_code_section = True
                
                # Stop capturing at certain patterns
                if line.startswith('```') or line.startswith('---') or line.startswith('##'):
                    break
                
                if in_code_section and line and not line.startswith('//') and not line.startswith('/*'):
                    code_lines.append(line)
            
            return '\n'.join(code_lines)
        except Exception as e:
            self.logger.warning(f"Failed to extract implementation from response: {e}")
            return ""
    
    async def _generate_fallback_code(self, conversion_plan: Dict[str, Any],
                                    matlab_analysis: Dict[str, Any],
                                    state: ConversionState) -> Dict[str, Any]:
        """Generate fallback code when all iterations fail."""
        self.logger.warning("Generating fallback code with conservative approach")
        
        # Use most conservative strategy
        fallback_strategy = "conservative"
        
        # Generate with minimal features
        generation_prompt = self._build_enhanced_generation_prompt(
            conversion_plan, matlab_analysis, fallback_strategy, 999
        )
        
        # Add fallback instructions
        generation_prompt += "\n\nFALLBACK MODE: Generate minimal but correct C++ code that compiles."
        
        response = self.llm_client.get_completion(generation_prompt)
        
        parsed_code = self.code_generation_tool._parse_generated_code(response, "single_file")
        
        # Handle None case - LLM failed to generate valid code
        if parsed_code is None:
            self.logger.error("❌ Fallback generation failed: LLM returned invalid response")
            # Return minimal empty structure to avoid crashes
            return {
                'files': {},
                'dependencies': [],
                'compilation_instructions': '',
                'usage_example': '',
                'notes': 'Fallback generation failed',
                'conversion_mode': 'single_file',
                'raw_response': response
            }
        
        # Apply post-processing fixes to the parsed code
        if parsed_code.get('header'):
            parsed_code['header'] = self.code_generation_tool._fix_corrupted_includes(parsed_code['header'])
        if parsed_code.get('implementation'):
            parsed_code['implementation'] = self.code_generation_tool._fix_corrupted_includes(parsed_code['implementation'])
        
        # Convert to expected structure with 'files' key
        generated_code = {
            'files': {
                'arma_filter.h': parsed_code.get('header', ''),
                'arma_filter.cpp': parsed_code.get('implementation', '')
            },
            'dependencies': parsed_code.get('dependencies', []),
            'compilation_instructions': parsed_code.get('compilation_instructions', ''),
            'usage_example': parsed_code.get('usage_example', ''),
            'notes': parsed_code.get('notes', ''),
            'conversion_mode': parsed_code.get('conversion_mode', 'single_file'),
            'raw_response': parsed_code.get('raw_response', response)
        }
        
        return generated_code
    
    async def get_generation_summary(self, generation_result: Dict[str, Any]) -> str:
        """Generate human-readable generation summary."""
        summary = f"🔧 Enhanced C++ Generation Summary\n"
        
        generated_code = generation_result.get('generated_code', {})
        files = generated_code.get('files', {})
        summary += f"Files generated: {len(files)}\n"
        
        strategy = generation_result.get('generation_strategy', 'unknown')
        summary += f"Strategy used: {strategy}\n"
        
        iterations = generation_result.get('generation_iterations', 0)
        compilation_attempts = generation_result.get('compilation_attempts', 0)
        success_rate = generation_result.get('compilation_success_rate', 0.0)
        
        summary += f"Generation iterations: {iterations}\n"
        summary += f"Compilation attempts: {compilation_attempts}\n"
        summary += f"Compilation success rate: {success_rate:.1%}\n"
        
        if success_rate > 0.8:
            summary += "✅ High success rate achieved\n"
        elif success_rate > 0.5:
            summary += "⚠️ Moderate success rate\n"
        else:
            summary += "❌ Low success rate - may need manual review\n"
        
        return summary
    
    # ==================== Multi-File Generation Methods ====================
    
    async def _generate_single_file_with_context(self, matlab_file: str,
                                                 file_analysis: Dict[str, Any],
                                                 conversion_plan: Dict[str, Any],
                                                 matlab_analysis: Dict[str, Any],
                                                 previously_generated: Dict[str, Dict[str, str]],
                                                 namespace: str,
                                                 target_cpp: str,
                                                 target_header: str,
                                                 state: ConversionState) -> Dict[str, Any]:
        """
        Generate a single MATLAB file with context of previously generated files.
        
        Args:
            matlab_file: Name of MATLAB file to convert
            file_analysis: Analysis for this specific file
            conversion_plan: Overall conversion plan
            matlab_analysis: Full project analysis
            previously_generated: Dict of {matlab_file: {cpp_file: content}}
            namespace: Namespace for this file
            state: Conversion state
            
        Returns:
            Generated code dict with 'files' key
        """
        # Create a focused analysis for just this file
        single_file_analysis = {
            'file_analyses': [file_analysis],
            'file_count': 1,
            'function_count': len(file_analysis.get('functions', [])),
            'is_multi_file': False,  # Treat as single file for generation
            'compilation_order': [matlab_file]
        }
        
        # Determine generation strategy
        strategy = self._select_generation_strategy(conversion_plan, single_file_analysis)
        
        # Build context prompt with trimmed view of previously generated files
        sanitized_context = self._prepare_previous_context(previously_generated)
        context_prompt = self._build_file_context_prompt(
            matlab_file, sanitized_context, conversion_plan
        )
        
        # Generate with limited iterations (1-2) since we're doing file-by-file
        max_iterations = 2
        current_iteration = 0
        
        while current_iteration < max_iterations:
            current_iteration += 1
            
            self.logger.info(f"    Iteration {current_iteration}/{max_iterations} for {matlab_file}")
            
            # Build generation prompt
            generation_prompt = self._build_single_file_generation_prompt(
                matlab_file=matlab_file,
                file_analysis=file_analysis,
                conversion_plan=conversion_plan,
                matlab_analysis=single_file_analysis,
                strategy=strategy,
                context_prompt=context_prompt,
                namespace=namespace,
                iteration=current_iteration,
                previously_generated=sanitized_context
            )
            
            # Generate code using LLM
            enhanced_prompt = generation_prompt + "\n\n/no_think"
            response = self.llm_client.get_completion(enhanced_prompt)
            
            # Parse generated code
            parsed_code = self.code_generation_tool._parse_generated_code(response, "multi_file")
            
            # Extract and validate
            header_content = parsed_code.get('header', '')
            implementation_content = parsed_code.get('implementation', '')
            
            # Apply comprehensive syntax fixes
            if header_content or implementation_content:
                # First, apply old fixes for backwards compatibility
                if header_content:
                    header_content = self.code_generation_tool._fix_corrupted_includes(header_content)
                    header_content = self._validate_and_fix_header(header_content)
                
                if implementation_content:
                    implementation_content = self.code_generation_tool._fix_corrupted_includes(implementation_content)
                    implementation_content = self._validate_and_fix_implementation(implementation_content)
                
                # Then apply robust syntax fixes (NEW!)
                header_content, implementation_content = self.syntax_fixer.fix_all_syntax_issues(
                    header_content or '',
                    implementation_content or '',
                    matlab_file
                )
            
            # Generate filenames with proper extensions
            cpp_file = target_cpp
            header_file = target_header
            
            # Reconcile newly generated artifacts against known interfaces
            interface_map = self._build_interface_map(sanitized_context, conversion_plan)
            header_content, implementation_content = self._reconcile_generated_artifacts(
                header_content,
                implementation_content,
                namespace,
                file_analysis,
                interface_map
            )

            implementation_content = self._apply_required_includes(
                cpp_file,
                implementation_content,
                conversion_plan
            )
            implementation_content = self._normalize_internal_includes(
                implementation_content,
                conversion_plan
            )
            implementation_content = self._ensure_dependency_includes(
                cpp_file,
                implementation_content,
                conversion_plan
            )
            implementation_content = self._normalize_internal_includes(
                implementation_content,
                conversion_plan
            )
            if not header_content or not header_content.strip():
                header_content = self._synthesize_header_stub(
                    implementation_content,
                    namespace,
                    cpp_file,
                    file_analysis.get('function_signatures', {})
                )
            
            # Check if we got valid content
            if header_content and implementation_content:
                return {
                    'files': {
                        header_file: header_content,
                        cpp_file: implementation_content
                    }
                }
            else:
                self.logger.warning(f"    ⚠️  Iteration {current_iteration} produced incomplete code")
                if current_iteration < max_iterations:
                    strategy = "Fix: Ensure both header (.h) and implementation (.cpp) are generated with complete code."
        
        # Fallback: return whatever we have
        result_files = {}
        if header_content:
            result_files[header_file] = header_content
        if implementation_content:
            result_files[cpp_file] = implementation_content
            
        return {'files': result_files} if result_files else {'files': {}}
    
    def _build_file_context_prompt(self, current_file: str,
                                   previously_generated: Dict[str, Dict[str, str]],
                                   conversion_plan: Dict[str, Any]) -> str:
        """Build context prompt showing previously generated files."""
        if not previously_generated:
            return ""
        
        context_parts = ["\n📚 PREVIOUSLY GENERATED FILES (for reference):"]
        context_parts.append("Use these as reference for consistency and dependencies.\n")
        
        # FIX #1: Extract and show AVAILABLE FUNCTIONS
        available_functions = []
        for matlab_file, files_dict in previously_generated.items():
            for filename, content in files_dict.items():
                if filename.endswith('.h'):
                    # Extract function signatures
                    func_sigs = self._extract_function_signatures(content, filename)
                    available_functions.extend(func_sigs)
        
        if available_functions:
            context_parts.append("\n" + "=" * 70)
            context_parts.append("🔑 AVAILABLE FUNCTIONS (FIX #1 - DO NOT INVENT NEW NAMES!):")
            context_parts.append("=" * 70)
            context_parts.append("\nYou may ONLY call these functions (already defined):")
            for sig in available_functions:
                context_parts.append(f"  ✅ {sig}")
            context_parts.append("\n❌ DO NOT invent new function names like 'pointmin2d', 'rk42d', 'euler2d'!")
            context_parts.append("❌ Only use the exact function names listed above!")
            context_parts.append("=" * 70 + "\n")
        
        for matlab_file, files_dict in previously_generated.items():
            context_parts.append(f"\n--- From {matlab_file} ---")
            for filename, content in files_dict.items():
                if filename.endswith('.h'):
                    # Show header signatures only (first 50 lines)
                    lines = content.split('\n')[:50]
                    context_parts.append(f"\n// {filename} (header signatures)")
                    context_parts.append('\n'.join(lines))
                    if len(content.split('\n')) > 50:
                        context_parts.append("// ... (truncated)")
        
        return '\n'.join(context_parts)
    
    def _extract_function_signatures(self, header_content: str, filename: str) -> List[str]:
        """Extract function signatures from header file (FIX #1)."""
        signatures = []
        lines = header_content.split('\n')
        
        in_namespace = False
        namespace_name = ""
        
        for i, line in enumerate(lines):
            # Track namespace
            if 'namespace' in line and '{' in line:
                match = re.search(r'namespace\s+(\w+)', line)
                if match:
                    namespace_name = match.group(1)
                    in_namespace = True
            elif line.strip() == '}' and in_namespace:
                in_namespace = False
            
            # Look for function declarations (not #define, not //)
            if (in_namespace and 
                not line.strip().startswith('//') and 
                not line.strip().startswith('#') and
                not line.strip().startswith('*') and
                ('(' in line and ');' in line or 
                 (i+1 < len(lines) and '(' in line and ');' in lines[i+1]))):
                # Simple extraction
                sig = line.strip()
                if i+1 < len(lines) and ');' in lines[i+1]:
                    sig += ' ' + lines[i+1].strip()
                # Clean up and add namespace
                if namespace_name:
                    signatures.append(f"{namespace_name}::{sig}")
        
        return signatures
    
    def _prepare_previous_context(self,
                                  previously_generated: Dict[str, Dict[str, str]],
                                  max_files: int = 5,
                                  max_total_chars: int = 16000,
                                  per_file_chars: int = 4000) -> Dict[str, Dict[str, str]]:
        """Trim previously generated context to keep prompts bounded."""
        if not previously_generated:
            return {}
        
        selected: Dict[str, Dict[str, str]] = {}
        total_chars = 0
        
        items = list(previously_generated.items())
        for matlab_file, files_dict in items[-max_files:]:
            trimmed_files: Dict[str, str] = {}
            for filename, content in files_dict.items():
                text_content = content if isinstance(content, str) else str(content)
                truncated = self._truncate_content_for_prompt(text_content, per_file_chars)
                trimmed_files[filename] = truncated
                total_chars += len(truncated)
                if total_chars >= max_total_chars:
                    break
            if trimmed_files:
                selected[matlab_file] = trimmed_files
            if total_chars >= max_total_chars:
                break
        
        return selected
    
    def _truncate_content_for_prompt(self, content: str, max_length: int) -> str:
        """Truncate long content while preserving start/end context."""
        if len(content) <= max_length:
            return content
        head_len = max_length // 2
        tail_len = max_length - head_len
        return (
            f"{content[:head_len].rstrip()}\n"
            "/* ... truncated for prompt context ... */\n"
            f"{content[-tail_len:].lstrip()}"
        )
    
    def _build_interface_map(self,
                             previous_context: Dict[str, Dict[str, str]],
                             conversion_plan: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Collect known function interfaces from previously generated headers."""
        interface_map: Dict[str, Dict[str, Any]] = {}
        if not previous_context:
            return interface_map
        
        file_mapping = {}
        if conversion_plan:
            file_mapping = conversion_plan.get('file_organization', {}).get('file_mapping', {})
        
        for matlab_file, files in previous_context.items():
            for filename, content in files.items():
                if not filename.endswith('.h') or not content:
                    continue
                
                namespace = self._detect_namespace_from_content(content)
                if not namespace:
                    namespace = self._infer_namespace_from_plan(file_mapping, filename, matlab_file)
                
                signatures = self._extract_function_signatures(content, filename)
                for signature in signatures:
                    func_name = self._extract_function_name_from_signature(signature)
                    if not func_name:
                        continue
                    interface_map[func_name] = {
                        'namespace': namespace,
                        'signature': signature,
                        'header': filename
                    }
        
        return interface_map
    
    def _detect_namespace_from_content(self, content: str) -> Optional[str]:
        """Detect the first namespace declaration in the content."""
        if not content:
            return None
        match = re.search(r'namespace\s+([A-Za-z_]\w*)', content)
        return match.group(1) if match else None
    
    def _infer_namespace_from_plan(self,
                                   file_mapping: Dict[str, Any],
                                   header_filename: str,
                                   matlab_file: str) -> Optional[str]:
        """Infer namespace from conversion plan when not present in content."""
        if not file_mapping:
            return None
        
        candidates = [
            matlab_file,
            matlab_file.replace('.m', ''),
            Path(matlab_file).stem if matlab_file else None
        ]
        for key in candidates:
            if key and key in file_mapping:
                namespace = file_mapping[key].get('namespace')
                if namespace:
                    return namespace
        
        for mapping in file_mapping.values():
            if mapping.get('header_file') == header_filename:
                return mapping.get('namespace')
        
        return None
    
    def _extract_function_name_from_signature(self, signature: str) -> Optional[str]:
        """Extract the function name from a signature string."""
        if not signature:
            return None
        cleaned = ' '.join(signature.strip().split())
        match = re.search(r'([A-Za-z_]\w*)\s*\([^()]*\)\s*;?', cleaned)
        if match:
            return match.group(1)
        return None
    
    def _build_interface_contract(self,
                                  matlab_file: str,
                                  file_analysis: Dict[str, Any],
                                  conversion_plan: Dict[str, Any],
                                  namespace: str,
                                  previously_generated: Optional[Dict[str, Dict[str, str]]]) -> str:
        """Build a concise interface contract snippet for the prompt."""
        details: List[str] = []
        mapping = {}
        if conversion_plan:
            file_mapping = conversion_plan.get('file_organization', {}).get('file_mapping', {})
            if file_mapping:
                mapping = (file_mapping.get(matlab_file) or
                           file_mapping.get(matlab_file.replace('.m', '')) or
                           file_mapping.get(Path(matlab_file).stem if matlab_file else '')) or {}
        
        functions = file_analysis.get('functions', []) if file_analysis else []
        has_prev = bool(previously_generated)
        
        if not any([namespace, mapping, functions, has_prev]):
            return ""
        
        details.append("\n" + "=" * 70)
        details.append("🔒 INTERFACE CONTRACT")
        details.append("=" * 70)
        details.append(f"- Target namespace: {namespace or 'n/a'}")
        if mapping:
            header_file = mapping.get('header_file')
            cpp_file = mapping.get('cpp_file')
            if header_file:
                details.append(f"- Expected header: {header_file}")
            if cpp_file:
                details.append(f"- Expected implementation: {cpp_file}")
            includes = mapping.get('includes') or mapping.get('dependencies')
            if includes:
                details.append(f"- Required includes: {', '.join(includes)}")
        if functions:
            details.append(f"- MATLAB functions to expose: {', '.join(functions)}")
        if has_prev:
            file_count = sum(len(files) for files in previously_generated.values())
            details.append(f"- Existing project files available for reference: {file_count}")
        details.append("=" * 70)
        
        return '\n'.join(details)
    
    def _reconcile_generated_artifacts(self,
                                       header_content: str,
                                       implementation_content: str,
                                       namespace: str,
                                       file_analysis: Dict[str, Any],
                                       interface_map: Dict[str, Dict[str, Any]]) -> Tuple[str, str]:
        """Verify and lightly adjust generated code before saving."""
        header_content = header_content or ""
        implementation_content = implementation_content or ""
        
        expected_functions = set(file_analysis.get('functions', [])) if file_analysis else set()
        header_functions = self._extract_header_function_names(header_content)
        impl_functions = self._extract_implementation_function_names(implementation_content)
        
        # Log missing declarations/definitions
        if expected_functions:
            missing_header = [fn for fn in expected_functions if fn not in header_functions]
            missing_impl = [fn for fn in expected_functions if fn not in impl_functions]
            if missing_header:
                self.logger.warning(
                    f"⚠️  Header is missing declarations for: {', '.join(missing_header)}"
                )
            if missing_impl:
                self.logger.warning(
                    f"⚠️  Implementation is missing definitions for: {', '.join(missing_impl)}"
                )
        
        # Detect namespace conflicts with previously generated interfaces
        conflicting = [
            fn for fn in header_functions
            if fn in interface_map and interface_map[fn].get('namespace') not in (None, namespace)
        ]
        if conflicting:
            self.logger.warning(
                f"⚠️  Potential namespace conflicts detected for: {', '.join(conflicting)}"
            )
        
        # Ensure external calls use known namespaces
        implementation_content = self._apply_external_namespace_fixes(
            implementation_content,
            interface_map,
            local_functions=header_functions.union(expected_functions),
            current_namespace=namespace
        )
        
        return header_content, implementation_content
    
    def _extract_header_function_names(self, header_content: str) -> Set[str]:
        """Extract function names declared in a header."""
        return self._extract_function_names_generic(header_content, ';')
    
    def _extract_implementation_function_names(self, implementation_content: str) -> Set[str]:
        """Extract function names defined in an implementation file."""
        return self._extract_function_names_generic(implementation_content, '{')
    
    def _extract_function_names_generic(self, content: str, terminator: str) -> Set[str]:
        """Extract function names using a lightweight regex heuristic."""
        if not content:
            return set()
        
        pattern = re.compile(
            r'([A-Za-z_]\w*(?:::[A-Za-z_]\w*)?)\s*\([^;{}]*\)\s*' + re.escape(terminator)
        )
        reserved = {'if', 'for', 'while', 'switch', 'return', 'catch', 'else'}
        names: Set[str] = set()
        for match in pattern.findall(content):
            name = match.split('::')[-1]
            if name in reserved:
                continue
            names.add(name)
        return names
    
    def _apply_external_namespace_fixes(self,
                                        content: str,
                                        interface_map: Dict[str, Dict[str, Any]],
                                        local_functions: Set[str],
                                        current_namespace: Optional[str]) -> str:
        """Prefix known external calls with their namespaces to avoid ambiguity."""
        if not content or not interface_map:
            return content
        
        updated_content = content
        for func_name, info in interface_map.items():
            if func_name in local_functions:
                continue
            namespace = info.get('namespace')
            if not namespace or namespace == current_namespace:
                continue
            
            pattern = re.compile(r'(?<![:\w])' + re.escape(func_name) + r'\s*\(')
            
            def replace_call(match: re.Match) -> str:
                prefix_segment = match.string[max(0, match.start() - 64):match.start()]
                stripped = prefix_segment.rstrip()
                if stripped.endswith(f"{namespace}::"):
                    return match.group(0)
                if stripped and stripped[-1].isalnum():
                    # Likely a declaration or definition; leave untouched
                    return match.group(0)
                return f"{namespace}::{func_name}("
            
            updated_content = pattern.sub(replace_call, updated_content)
        
        return updated_content
    
    def _normalize_internal_includes(self,
                                     content: str,
                                     conversion_plan: Dict[str, Any]) -> str:
        """Normalize internal #include directives to match planned file names."""
        if not content:
            return content
        
        file_mapping = conversion_plan.get('file_organization', {}).get('file_mapping', {}) or {}
        canonical_map: Dict[str, str] = {}
        
        for matlab_file, mapping in file_mapping.items():
            header = mapping.get('header_file')
            cpp = mapping.get('cpp_file')
            if header:
                canonical_map[self._canonical_include_key(Path(header).stem)] = header
            if cpp:
                canonical_map[self._canonical_include_key(Path(cpp).stem)] = header or cpp
            
            matlab_stem = self._canonical_include_key(Path(matlab_file).stem)
            if header and matlab_stem not in canonical_map:
                canonical_map[matlab_stem] = header
        
        pattern = re.compile(r'#include\s+"([^"]+)"')
        
        def replacer(match: re.Match) -> str:
            include_path = match.group(1)
            canonical = self._canonical_include_key(Path(include_path).stem)
            updated = canonical_map.get(canonical)
            if updated:
                return f'#include "{updated}"'
            return match.group(0)
        
        return pattern.sub(replacer, content)
    
    def _canonical_include_key(self, name: str) -> str:
        """Generate a canonical key for include normalization."""
        return re.sub(r'[^a-z0-9]', '', name.lower())
    
    def _ensure_dependency_includes(self,
                                    cpp_filename: str,
                                    content: str,
                                    conversion_plan: Dict[str, Any]) -> str:
        """Ensure includes for MATLAB-level dependencies are present."""
        project_structure = conversion_plan.get('project_structure', {}) or {}
        file_dependencies = project_structure.get('file_dependencies', {}) or {}
        file_mapping = conversion_plan.get('file_organization', {}).get('file_mapping', {}) or {}
        
        matlab_file = None
        for m_file, mapping in file_mapping.items():
            if mapping.get('cpp_file') == cpp_filename:
                matlab_file = m_file
                break
        
        if not matlab_file:
            return content
        
        dependency_headers: List[str] = []
        for dep in file_dependencies.get(matlab_file, []):
            dep_header = file_mapping.get(dep, {}).get('header_file')
            if dep_header:
                dependency_headers.append(dep_header)
        
        updated = content
        for header in dependency_headers:
            updated = self._ensure_include_line(updated, f'#include "{header}"')
        return updated
    
    def _apply_required_includes(self,
                                 cpp_filename: str,
                                 implementation_content: str,
                                 conversion_plan: Dict[str, Any]) -> str:
        """Ensure required include directives from the conversion plan are present."""
        include_deps = conversion_plan.get('include_dependencies', {}) or {}
        required_includes = include_deps.get(cpp_filename)
        if not required_includes:
            return implementation_content
        
        updated_content = implementation_content
        for include in required_includes:
            include_line = include if include.strip().startswith('#include') else f'#include {include}'
            updated_content = self._ensure_include_line(updated_content, include_line)
        return updated_content
    
    def _inject_support_helpers(self,
                                generated_files: Dict[str, str],
                                conversion_plan: Dict[str, Any]) -> Dict[str, str]:
        """Inject additional support helper files needed by the generated project."""
        if not generated_files:
            return {}
        
        support_plan = conversion_plan.get('support_files', []) or []
        helper_ids = {helper.get('id') for helper in support_plan}
        plan_requires_image_helpers = 'matlab_image_helpers' in helper_ids
        plan_requires_msfm_helpers = 'msfm_helpers' in helper_ids
        plan_requires_array_helpers = 'matlab_array_utils' in helper_ids
        plan_requires_pointmin_helpers = 'pointmin_helpers' in helper_ids
        detected_helper_usage = self._requires_image_helpers(generated_files)
        
        if not plan_requires_image_helpers and not detected_helper_usage and 'matlab_image_helpers.h' not in ''.join(generated_files.keys()):
            helper_needed = False
        else:
            helper_needed = plan_requires_image_helpers or detected_helper_usage
        
        helper_header = 'matlab_image_helpers.h'
        helper_files: Dict[str, str] = {}
        msfm_helper_needed = plan_requires_msfm_helpers
        msfm_helper_header = 'msfm_helpers.h'
        array_helper_needed = plan_requires_array_helpers
        array_helper_header = 'matlab_array_utils.h'
        pointmin_helper_needed = plan_requires_pointmin_helpers
        pointmin_helper_header = 'pointmin_helpers.h'
        
        for filename, content in list(generated_files.items()):
            if not isinstance(content, str):
                continue
            if filename.startswith('matlab_image_helpers'):
                continue
            if not filename.endswith(('.cpp', '.cc', '.cxx', '.h', '.hpp')):
                continue
            
            updated_content = content
            needs_image_helpers = self._contains_image_helper_usage(content)
            needs_array_helpers = self._contains_array_helper_usage(content)
            needs_pointmin_helpers = self._contains_pointmin_helper_usage(content)

            if needs_image_helpers:
                helper_needed = True
                normalized_content, replaced_functions = self._normalize_image_helper_calls(updated_content)
                if replaced_functions:
                    self.logger.info(
                        f"   🔧 Normalized image helper calls in {filename}: {', '.join(sorted(replaced_functions))}"
                    )
                updated_content = normalized_content
                updated_content = self._ensure_include_line(updated_content, f'#include "{helper_header}"')
                updated_content = self._normalize_array_helper_calls(updated_content)
            
            if 'msfm::msfm' in updated_content:
                msfm_helper_needed = True
                updated_content = self._ensure_include_line(updated_content, f'#include "{msfm_helper_header}"')
            
            if needs_array_helpers:
                array_helper_needed = True
                updated_content = self._ensure_include_line(updated_content, f'#include "{array_helper_header}"')
                updated_content = self._normalize_array_helper_calls(updated_content)
            
            if needs_pointmin_helpers:
                pointmin_helper_needed = True
                updated_content = self._ensure_include_line(updated_content, f'#include "{pointmin_helper_header}"')
            
            if updated_content != content:
                generated_files[filename] = updated_content
        
        if helper_needed:
            helper_files.update(self._get_image_helper_files())
            self.logger.info("   📎 Injected MATLAB image helper support files")

        if msfm_helper_needed:
            helper_files.update(self._get_msfm_helper_files())
            self.logger.info("   📎 Injected MSFM helper overloads")
        
        if array_helper_needed:
            helper_files.update(self._get_array_helper_files())
            self.logger.info("   📎 Injected MATLAB array utility helpers")
        
        if pointmin_helper_needed:
            helper_files.update(self._get_pointmin_helper_files())
            self.logger.info("   📎 Injected pointmin helper wrappers")
        
        return helper_files

    def _requires_image_helpers(self, generated_files: Dict[str, str]) -> bool:
        """Determine if any generated file references image helper operations."""
        patterns = (
            'imdilate',
            'imerode',
            'imopen',
            'imclose',
            'dilate2d',
            'erode2d',
            'open2d',
            'close2d'
        )
        for filename, content in generated_files.items():
            if not isinstance(content, str):
                continue
            if filename.startswith('matlab_image_helpers'):
                continue
            lowered = content.lower()
            if any(pattern in lowered for pattern in patterns):
                return True
        return False
    
    def _contains_image_helper_usage(self, content: str) -> bool:
        """Check whether a single file uses image helper operations."""
        if not content:
            return False
        lowered = content.lower()
        return any(keyword in lowered for keyword in (
            'imdilate', 'imerode', 'imopen', 'imclose', 'dilate2d', 'erode2d', 'open2d', 'close2d',
            'dilateimage', 'erodeimage', 'openimage', 'closeimage'
        ))
    
    def _contains_array_helper_usage(self, content: str) -> bool:
        """Check whether array helper utilities are referenced."""
        if not content:
            return False
        keywords = (
            'linearIndexToSubscripts',
            'subscriptsToLinear',
            'findNonZero',
            'findNonzero',
            'matlab::array::findNonZero',
            'matlab::array::linearIndexToSubscripts',
            'matlab::array::subscriptsToLinear'
        )
        return any(keyword in content for keyword in keywords)
    
    def _contains_pointmin_helper_usage(self, content: str) -> bool:
        """Check whether pointmin helper wrappers are referenced."""
        if not content:
            return False
        keywords = ('pointmin::pointmin2d', 'pointmin::pointmin3d')
        return any(keyword in content for keyword in keywords)

    def _synthesize_header_stub(self,
                                implementation_content: str,
                                namespace: str,
                                cpp_filename: str,
                                signature_map: Dict[str, Dict[str, Any]]) -> str:
        """Create a fallback header when the LLM fails to emit one."""
        includes = [
            '#pragma once',
            '',
            f'#include "{cpp_filename}"'
        ]
        body: List[str] = []

        if signature_map:
            body.append('// Signature metadata extracted from MATLAB analysis:')
            for func_name, metadata in signature_map.items():
                inputs = ', '.join(metadata.get('inputs', [])) or 'void'
                outputs = ', '.join(metadata.get('outputs', [])) or '[]'
                body.append(f"// {func_name}({inputs}) -> {outputs}")
            body.append('')

        prototypes = self._extract_function_prototypes(implementation_content, namespace)
        if prototypes:
            if namespace:
                body.append(f'namespace {namespace} {{')
                body.append('')
                body.extend(prototypes)
                body.append('')
                body.append(f'}} // namespace {namespace}')
            else:
                body.extend(prototypes)
        else:
            body.append('// TODO: add declarations or move implementations into header if needed')

        return '\n'.join(includes + [''] + body) + '\n'

    def _extract_function_prototypes(self, implementation_content: str, namespace: str) -> List[str]:
        """Extract top-level function prototypes from implementation content."""
        prototypes: List[str] = []
        if not implementation_content:
            return prototypes
        # Remove namespace scopes to simplify detection
        pattern = re.compile(r'namespace\s+' + re.escape(namespace) + r'\s*\{(.*?)\}', re.DOTALL)
        scoped_content = implementation_content
        match = pattern.search(implementation_content)
        if match:
            scoped_content = match.group(1)
        function_pattern = re.compile(
            r'^\s*(?:template<[^>]+>\s*)?'
            r'(?:inline\s+)?(?:static\s+)?'
            r'([\w:\<\>\s\*&]+?)\s+'
            r'(\w+)\s*\(([^)]*)\)'
            r'\s*(?:const)?\s*\{',
            re.MULTILINE
        )
        for match in function_pattern.finditer(scoped_content):
            return_type = ' '.join(match.group(1).split())
            name = match.group(2)
            args = match.group(3).strip()
            if not args:
                args = 'void'
            prototype = f'{return_type} {name}({args});'
            prototypes.append(prototype)
        return prototypes
    
    def _normalize_image_helper_calls(self, content: str) -> Tuple[str, Set[str]]:
        """Normalize MATLAB image helper calls to use shared helper namespace."""
        if not content:
            return content, set()
        
        function_map = {
            'imdilate': 'dilate2d',
            'dilate2d': 'dilate2d',
            'dilateimage': 'dilate2d',
            'imerode': 'erode2d',
            'erode2d': 'erode2d',
            'erodeimage': 'erode2d',
            'imopen': 'open2d',
            'open2d': 'open2d',
            'openimage': 'open2d',
            'imclose': 'close2d',
            'close2d': 'close2d',
            'closeimage': 'close2d'
        }
        
        pattern = re.compile(
            r'(?<![A-Za-z0-9_])((?:[A-Za-z_]\w*::)*)'
            r'(imdilate|dilate2d|imerode|erode2d|imopen|open2d|imclose|close2d)\s*\('
        )
        
        replaced_functions: Set[str] = set()
        
        def replacer(match: re.Match) -> str:
            prefix = match.group(1) or ''
            func = match.group(2)
            func_lower = func.lower()
            target_func = function_map.get(func_lower, func)
            replaced_functions.add(target_func)
            
            if 'matlab::image::' in prefix:
                return f'{prefix}{target_func}('
            return f'matlab::image::{target_func}('
        
        rewritten = pattern.sub(replacer, content)
        return rewritten, replaced_functions

    def _normalize_array_helper_calls(self, content: str) -> str:
        """Prefix array helper calls with matlab::array namespace."""
        if not content:
            return content

        replacements = {
            'findNonZero': 'matlab::array::findNonZero',
            'findNonzero': 'matlab::array::findNonZero',
            'linearIndexToSubscripts': 'matlab::array::linearIndexToSubscripts',
            'subscriptsToLinear': 'matlab::array::subscriptsToLinear'
        }

        for source, target in replacements.items():
            pattern = re.compile(rf'(?<![\w:]){re.escape(source)}\s*\(')

            def repl(match: re.Match) -> str:
                source_text = match.string
                prefix = source_text[max(0, match.start() - len(target) - 2):match.start()]
                if prefix.endswith(f'{target}('):
                    return match.group(0)
                return f'{target}('

            content = pattern.sub(repl, content)

        return content

    def _ensure_include_line(self, content: str, include_line: str) -> str:
        """Ensure the include line exists in the given content."""
        if include_line in content:
            return content
        
        lines = content.splitlines()
        insert_idx = 0
        for idx, line in enumerate(lines):
            if line.strip().startswith('#include'):
                insert_idx = idx + 1
        
        lines.insert(insert_idx, include_line)
        new_content = "\n".join(lines)
        if content.endswith("\n"):
            new_content += "\n"
        return new_content

    def _load_helper_template(self, filename: str) -> str:
        """Load helper file content from template directory with caching."""
        cache_key = filename
        if cache_key in self._helper_cache:
            return self._helper_cache[cache_key]
        path = self.helper_templates_dir / filename
        content = path.read_text(encoding='utf-8')
        self._helper_cache[cache_key] = content
        return content
    
    def _get_image_helper_files(self) -> Dict[str, str]:
        """Return helper file contents for MATLAB image operations."""
        return {
            'matlab_image_helpers.h': self._load_helper_template('matlab_image_helpers.h'),
            'matlab_image_helpers.cpp': self._load_helper_template('matlab_image_helpers.cpp')
        }

    def _get_msfm_helper_files(self) -> Dict[str, str]:
        """Return helper overloads for msfm source point conversions."""
        return {
            'msfm_helpers.h': self._load_helper_template('msfm_helpers.h'),
            'msfm_helpers.cpp': self._load_helper_template('msfm_helpers.cpp')
        }


    def _get_array_helper_files(self) -> Dict[str, str]:
        """Return helper utilities for MATLAB-like array operations."""
        return {
            'matlab_array_utils.h': self._load_helper_template('matlab_array_utils.h'),
            'matlab_array_utils.cpp': self._load_helper_template('matlab_array_utils.cpp')
        }


    def _get_pointmin_helper_files(self) -> Dict[str, str]:
        """Return helper wrappers bridging pointmin multi-dimensional variants."""
        return {
            'pointmin_helpers.h': self._load_helper_template('pointmin_helpers.h'),
            'pointmin_helpers.cpp': self._load_helper_template('pointmin_helpers.cpp')
        }


    def _build_single_file_generation_prompt(self, matlab_file: str,
                                            file_analysis: Dict[str, Any],
                                            conversion_plan: Dict[str, Any],
                                            matlab_analysis: Dict[str, Any],
                                            strategy: str,
                                            context_prompt: str,
                                            namespace: str,
                                            iteration: int,
                                            previously_generated: Dict[str, Dict[str, str]] = None) -> str:
        """Build generation prompt for a single file."""
        prompt_parts = [
            f"Convert the following MATLAB file to C++:",
            f"\nMATLAB File: {matlab_file}",
            f"Namespace: {namespace}",
            f"\nGeneration Strategy: {strategy}",
        ]
        
        if context_prompt:
            prompt_parts.append(context_prompt)
        
        interface_contract = self._build_interface_contract(
            matlab_file,
            file_analysis,
            conversion_plan,
            namespace,
            previously_generated
        )
        if interface_contract:
            prompt_parts.append(interface_contract)
        
        # FIX #1 & #3: Add available functions and Eigen warnings to file-level generation
        # This prevents errors during generation (better than fixing after!)
        if hasattr(self, 'error_fix_generator') and previously_generated:
            # Convert previously_generated to the format expected by error_fix_generator
            # previously_generated is Dict[matlab_file, Dict[cpp_file, content]]
            # We need Dict['files', Dict[cpp_file, content]]
            all_generated_files = {}
            for matlab_f, files_dict in previously_generated.items():
                all_generated_files.update(files_dict)
            
            generated_code_dict = {'files': all_generated_files}
            
            # Build available functions context
            available_funcs = self.error_fix_generator._build_available_functions_context(generated_code_dict)
            if available_funcs:
                prompt_parts.append(available_funcs)
                self.logger.info(f"✅ Added available functions context ({len(all_generated_files)} files)")
            
            # Build Eigen API warnings
            eigen_warnings = self.error_fix_generator._build_eigen_api_warnings()
            prompt_parts.append(eigen_warnings)
            self.logger.info(f"✅ Added Eigen API warnings")
        
        # Add the standard generation guidance (reuse existing method)
        standard_prompt = self._build_enhanced_generation_prompt(
            conversion_plan, matlab_analysis, strategy, iteration
        )
        
        # Combine
        prompt_parts.append("\n")
        prompt_parts.append(standard_prompt)
        
        # Add single-file specific guidance
        prompt_parts.append(f"\n⚠️  CRITICAL: Generate ONLY the code for {matlab_file}")
        prompt_parts.append(f"Output must include:")
        prompt_parts.append(f"1. {matlab_file.replace('.m', '.h')} - Header file with declarations")
        prompt_parts.append(f"2. {matlab_file.replace('.m', '.cpp')} - Implementation file")
        prompt_parts.append(f"\nUse namespace: {namespace}")
        
        return '\n'.join(prompt_parts)
    
    def _generate_main_entry_point_multifile(self, conversion_plan: Dict[str, Any],
                                              matlab_analysis: Dict[str, Any],
                                              generated_files: Dict[str, str]) -> str:
        """
        Generate main.cpp for multi-file projects with entry point analysis.
        
        This method analyzes multiple MATLAB files to determine the entry point:
        1. Check for main.m or script files
        2. Analyze function call graph to find top-level functions
        3. Generate appropriate test harness based on entry point type
        
        Args:
            conversion_plan: Conversion plan
            matlab_analysis: MATLAB analysis with multiple files
            generated_files: Dict of {filename: content}
            
        Returns:
            Content of main.cpp with intelligent entry point selection
        """
        project_name = conversion_plan.get('project_name', 'unknown')
        
        # Analyze MATLAB files to determine entry point
        file_analyses = matlab_analysis.get('file_analyses', [])
        call_graph = matlab_analysis.get('call_graph', {})
        
        # Build entry point analysis for LLM
        entry_point_analysis = self._analyze_entry_points(file_analyses, call_graph)
        
        # Extract function signatures from ALL header files
        all_function_signatures = []
        for header_name, header_content in generated_files.items():
            if header_name.endswith('.h'):
                import re
                # Match function declarations
                pattern = r'\s+([\w:<>]+(?:\s*<[^>]+>)?)\s+([\w_]+)\s*\([^)]*\);'
                matches = re.finditer(pattern, header_content)
                for match in matches:
                    full_sig = match.group(0).strip()
                    namespace = header_name.replace('.h', '')
                    all_function_signatures.append(f"{namespace}::{full_sig}")
        
        # Get MATLAB source snippets for ALL files
        matlab_sources = ""
        for file_analysis in file_analyses:
            if 'content' in file_analysis:
                filename = file_analysis.get('file_name', 'unknown')
                matlab_sources += f"\n// MATLAB File: {filename}\n"
                matlab_sources += file_analysis['content'][:300]  # First 300 chars per file
                matlab_sources += "\n// ...\n"
        
        # Build comprehensive LLM prompt for multi-file projects
        header_files = [f for f in generated_files.keys() if f.endswith('.h')]
        
        prompt = f"""Generate a complete main.cpp file for a MULTI-FILE C++ project converted from MATLAB.

PROJECT: {project_name}
FILE COUNT: {len(file_analyses)} MATLAB files → {len(generated_files)} C++ files

ENTRY POINT ANALYSIS:
{entry_point_analysis}

ALL AVAILABLE FUNCTIONS (from generated headers):
{chr(10).join(all_function_signatures) if all_function_signatures else "// No function signatures found"}

HEADER FILES TO INCLUDE:
{chr(10).join([f'#include "{h}"' for h in header_files])}

MATLAB SOURCES OVERVIEW (to understand the workflow):
{matlab_sources[:2000]}

GENERATED HEADER PREVIEWS:
{chr(10).join([f'// === {name} ==={chr(10)}{content[:400]}{chr(10)}' for name, content in generated_files.items() if name.endswith('.h')])[:3000]}

REQUIREMENTS:
1. **Analyze the entry point** - Determine which function(s) should be called first based on:
   - Is there a main MATLAB script or main.m?
   - Which functions are top-level (not called by others)?
   - What is the logical workflow based on MATLAB file structure?

2. **Create realistic test data** that matches the entry point function parameters:
   - If it needs Eigen::Tensor, create a small test tensor (e.g., 3x3x10 or 5x5x20)
   - If it needs scalars, use reasonable values based on MATLAB context
   - If it needs strings or file paths, provide example values

3. **Call functions in correct order** if there are dependencies:
   - Initialize data structures first
   - Call processing functions in sequence
   - Display results

4. **Include all necessary headers**:
   - All generated headers
   - <iostream>, <Eigen/Dense>, <unsupported/Eigen/CXX11/Tensor>
   - Any other C++ standard library headers needed

5. **Print meaningful output** showing:
   - What function(s) are being tested
   - Input data summary
   - Output/results
   - Success message

6. **Handle exceptions properly** using ONLY standard exception types:
   - std::exception (NOT Eigen::EigenException or other non-existent types)
   - Return appropriate exit codes

Generate ONLY the complete main.cpp file. Make it actually RUN and demonstrate the multi-file project works:

```cpp"""

        try:
            # Call LLM to generate main.cpp with multi-file awareness
            self.logger.info(f"🤖 Generating intelligent main.cpp for multi-file project using LLM...")
            response = self.llm_client.get_completion(prompt)
            
            # Extract code from response
            import re
            cpp_match = re.search(r'```cpp\s*\n(.*?)\n```', response, re.DOTALL)
            if cpp_match:
                main_cpp = cpp_match.group(1)
                self.logger.info(f"✅ Generated multi-file main.cpp ({len(main_cpp)} chars)")
            else:
                code_match = re.search(r'```\s*\n(.*?)\n```', response, re.DOTALL)
                if code_match:
                    main_cpp = code_match.group(1)
                    self.logger.info(f"✅ Generated main.cpp from generic code block ({len(main_cpp)} chars)")
                else:
                    main_cpp = response
                    self.logger.warning(f"⚠️  No code blocks found, using entire LLM response")
            
            # Ensure it has basic structure
            if 'int main' not in main_cpp:
                raise ValueError("LLM-generated code missing 'int main'")
                
            return main_cpp
            
        except Exception as e:
            self.logger.warning(f"⚠️  LLM main.cpp generation failed: {e}, using fallback")
            return self._generate_fallback_main(project_name, header_files)
    
    def _analyze_entry_points(self, file_analyses: List[Dict], call_graph: Dict) -> str:
        """Analyze MATLAB files to determine entry point(s) for multi-file projects."""
        analysis = []
        
        # Check for main.m or script files
        main_files = []
        script_files = []
        function_files = []
        
        for file_analysis in file_analyses:
            filename = file_analysis.get('file_name', '')
            file_type = file_analysis.get('file_type', 'unknown')
            
            if 'main.m' in filename.lower():
                main_files.append(filename)
            elif file_type == 'script':
                script_files.append(filename)
            elif file_type == 'function':
                function_files.append(filename)
        
        if main_files:
            analysis.append(f"ENTRY POINT: Found main file(s): {', '.join(main_files)}")
            analysis.append("RECOMMENDATION: Call the main function from this file as entry point")
        elif script_files:
            analysis.append(f"ENTRY POINT: Found script file(s): {', '.join(script_files)}")
            analysis.append("RECOMMENDATION: Execute the script logic as entry point")
        else:
            # Analyze call graph to find top-level functions (not called by others)
            all_functions = set()
            called_functions = set()
            
            for caller, callees in call_graph.items():
                all_functions.add(caller)
                for callee in callees:
                    called_functions.add(callee)
            
            top_level = all_functions - called_functions
            
            if top_level:
                analysis.append(f"ENTRY POINT: Top-level functions (not called by others): {', '.join(list(top_level)[:3])}")
                analysis.append("RECOMMENDATION: Call these top-level functions as entry points")
            else:
                analysis.append("ENTRY POINT: No clear entry point detected")
                analysis.append("RECOMMENDATION: Create a demo calling the most significant function(s)")
        
        # Add function list
        analysis.append(f"\nAVAILABLE MATLAB FILES: {len(file_analyses)} files")
        for fa in file_analyses:
            filename = fa.get('file_name', 'unknown')
            functions = fa.get('functions', [])
            if functions:
                # Handle both dict and string function entries
                func_names = []
                for f in functions:
                    if isinstance(f, dict):
                        func_names.append(f.get('name', 'unnamed'))
                    elif isinstance(f, str):
                        func_names.append(f)
                    else:
                        func_names.append(str(f))
                analysis.append(f"  - {filename}: {', '.join(func_names)}")
        
        return '\n'.join(analysis)
    
    def _generate_fallback_main(self, project_name: str, header_files: List[str]) -> str:
        """Generate a simple fallback main.cpp if LLM generation fails."""
        includes = [f'#include "{h}"' for h in header_files]
        
        return f"""// main.cpp - Generated entry point for {project_name}
#include <iostream>
#include <exception>
{chr(10).join(includes)}

int main(int argc, char** argv) {{
    try {{
        std::cout << "Starting {project_name}..." << std::endl;
        
        // TODO: Call the appropriate entry point function(s)
        // Analyze the generated headers to determine correct function calls
        
        std::cout << "{project_name} completed successfully." << std::endl;
        return 0;
        
    }} catch (const std::exception& e) {{
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }} catch (...) {{
        std::cerr << "Unknown error occurred" << std::endl;
        return 2;
    }}
}}
"""
    
    def _generate_main_entry_point(self, conversion_plan: Dict[str, Any],
                                   matlab_analysis: Dict[str, Any],
                                   generated_files: Dict[str, str]) -> str:
        """
        Generate main.cpp entry point with LLM-generated test harness.
        
        Args:
            conversion_plan: Conversion plan
            matlab_analysis: MATLAB analysis
            generated_files: Dict of {filename: content}
            
        Returns:
            Content of main.cpp
        """
        # Find the main entry point (usually the file matching project name)
        project_name = conversion_plan.get('project_name', 'unknown')
        
        # Find all header files
        header_files = [f for f in generated_files.keys() if f.endswith('.h')]
        
        # Extract function signatures from header files
        function_signatures = []
        for header_name, header_content in generated_files.items():
            if header_name.endswith('.h'):
                # Extract function declarations (simple regex-based extraction)
                import re
                # Match function declarations like: type func_name(params);
                pattern = r'\s+([\w:<>]+(?:\s*<[^>]+>)?)\s+([\w_]+)\s*\([^)]*\);'
                matches = re.finditer(pattern, header_content)
                for match in matches:
                    function_signatures.append(match.group(0).strip())
        
        # Get MATLAB source code for context
        matlab_source = ""
        for file_analysis in matlab_analysis.get('file_analyses', []):
            if 'content' in file_analysis:
                matlab_source += f"\n// MATLAB Source:\n// {file_analysis.get('filename', 'unknown')}\n"
                matlab_source += file_analysis['content'][:500]  # First 500 chars
        
        # Build LLM prompt to generate test harness
        prompt = f"""Generate a complete main.cpp file with a working test harness for the converted MATLAB code.

PROJECT: {project_name}

AVAILABLE FUNCTIONS (from generated headers):
{chr(10).join(function_signatures) if function_signatures else "// No function signatures found"}

HEADER FILES TO INCLUDE:
{chr(10).join([f'#include "{h}"' for h in header_files])}

MATLAB SOURCE CONTEXT:
{matlab_source[:1000] if matlab_source else "// No MATLAB source available"}

GENERATED HEADER CONTENT (for context):
{chr(10).join([f'// {name}:{chr(10)}{content[:500]}' for name, content in generated_files.items() if name.endswith('.h')])[:2000]}

REQUIREMENTS:
1. Include all necessary headers (#include <iostream>, <Eigen/Dense>, <unsupported/Eigen/CXX11/Tensor>, etc.)
2. Create realistic test data that matches the function parameter types
3. Call the converted function(s) with the test data
4. Print meaningful output showing the function executed successfully
5. Handle exceptions properly using ONLY standard C++ exception types:
   - Use std::exception (CORRECT)
   - Do NOT use Eigen::EigenException (DOES NOT EXIST)
   - Do NOT invent custom exception types
6. If the function requires Eigen::Tensor, create a small test tensor (e.g., 3x3x10)
7. If the function requires scalar parameters, use reasonable test values (e.g., p=5, its=2)
8. Make the code actually RUN and demonstrate the conversion works

CRITICAL: Only catch std::exception or use catch(...) for unknown exceptions.

Generate ONLY the complete main.cpp file below. Start with includes, then main() function:

```cpp"""

        try:
            # Call LLM to generate main.cpp with test harness
            self.logger.info(f"🤖 Generating intelligent main.cpp with test harness using LLM...")
            response = self.llm_client.get_completion(prompt)
            
            # Extract code from response
            # Try to find ```cpp code blocks
            import re
            cpp_match = re.search(r'```cpp\s*\n(.*?)\n```', response, re.DOTALL)
            if cpp_match:
                main_cpp = cpp_match.group(1)
                self.logger.info(f"✅ Generated intelligent main.cpp ({len(main_cpp)} chars)")
            else:
                # Try to find any code block
                code_match = re.search(r'```\s*\n(.*?)\n```', response, re.DOTALL)
                if code_match:
                    main_cpp = code_match.group(1)
                    self.logger.info(f"✅ Generated main.cpp from generic code block ({len(main_cpp)} chars)")
                else:
                    # Use the entire response if no code blocks found
                    main_cpp = response
                    self.logger.warning(f"⚠️  No code blocks found, using entire LLM response")
            
            # Ensure it has basic structure
            if 'int main' not in main_cpp:
                raise ValueError("LLM-generated code missing 'int main'")
                
            return main_cpp
            
        except Exception as e:
            self.logger.warning(f"⚠️  LLM main.cpp generation failed: {e}, using fallback")
            # Fallback to simple template
            includes = [f'#include "{h}"' for h in header_files]
            
            main_cpp = f"""// main.cpp - Generated entry point for {project_name}
// This file provides a main() function to execute the converted MATLAB code

{chr(10).join(includes)}
#include <iostream>
#include <exception>

int main(int argc, char** argv) {{
    try {{
        std::cout << "Starting {project_name}..." << std::endl;
        
        // TODO: Call the appropriate entry point function from your converted code
        // Example: {project_name}::{project_name}_main();
        
        std::cout << "{project_name} completed successfully." << std::endl;
        return 0;
        
    }} catch (const std::exception& e) {{
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }} catch (...) {{
        std::cerr << "Unknown error occurred" << std::endl;
        return 2;
    }}
}}
"""
            return main_cpp

    # ==================== PRIORITY 3: Iterative Error Fixing ====================
    
    async def _apply_iterative_error_fixing(self,
                                           generated_code: Dict[str, Any],
                                           matlab_analysis: Dict[str, Any],
                                           compilation_result: Dict[str, Any],
                                           project_name: str) -> Optional[Dict[str, str]]:
        """
        Apply iterative LLM-based error fixing to generated code.
        
        Args:
            generated_code: Initially generated C++ files
            matlab_analysis: Original MATLAB analysis
            compilation_result: Compilation result from first attempt
            project_name: Project name for compilation
            
        Returns:
            Fixed files dict or None if fixing failed
        """
        # Lazy init iterative fixer
        if self.iterative_fixer is None:
            self.iterative_fixer = IterativeErrorFixer(self.llm_client, self.compilation_manager)
            self.iterative_fixer.max_iterations = self.iterative_fix_max_iterations
            self.iterative_fixer.success_threshold = self.iterative_fix_threshold
        
        # Extract MATLAB sources for reference
        matlab_sources = {}
        for file_analysis in matlab_analysis.get('file_analyses', []):
            matlab_sources[file_analysis['file_name']] = file_analysis.get('content', '')
        
        # Run iterative fixing
        fixed_files = self.iterative_fixer.fix_compilation_errors(
            generated_files=generated_code.get('files', {}),
            matlab_sources=matlab_sources,
            compilation_output=compilation_result.get('output', ''),
            project_name=project_name
        )
        
        return fixed_files if fixed_files else None
    
    # ==================== PHASE 2: TARGETED ERROR FIXING ====================
    
    async def _apply_targeted_fixes(self,
                                   generated_files: Dict[str, str],
                                   conversion_plan: Dict[str, Any]) -> Dict[str, str]:
        """
        Apply Phase 2 targeted fixes instead of global LLM iteration.
        
        Uses pattern-based surgical fixes - no full-file regeneration.
        Much faster and safer than LLM iteration.
        """
        self.logger.info("   📋 Testing compilation to identify errors...")
        
        # Test compilation to get errors
        compilation_result = await self._test_compilation(
            {'files': generated_files},
            conversion_plan
        )
        
        if compilation_result.get('success', False):
            self.logger.info("   ✅ Already compiles successfully - no fixes needed!")
            return generated_files
        
        # Extract errors
        errors = compilation_result.get('errors', [])
        if isinstance(errors, list) and errors:
            error_count = len([e for e in errors if 'error:' in str(e).lower()])
            self.logger.info(f"   ❌ Found {error_count} errors to fix")
            
            # REACTIVE KNOWLEDGE: Get API docs for targeted fixing
            try:
                error_messages = [str(e) for e in errors[:10]]  # Top 10 errors
                
                # Get current code for context
                current_code = "\n\n".join([f"// {fname}\n{content}" 
                                           for fname, content in generated_files.items()])
                
                # Call with correct signature
                api_docs = self.api_knowledge_base.get_relevant_docs(
                    error_messages=error_messages,
                    current_code=current_code,
                    libraries_used=['eigen']
                )
                if api_docs:
                    self.logger.info(f"   📚 Retrieved API docs for targeted fixing ({len(api_docs)} chars)")
                    # Log a snippet of the guidance
                    snippet = api_docs[:200] + "..." if len(api_docs) > 200 else api_docs
                    self.logger.debug(f"   API guidance snippet: {snippet}")
            except Exception as e:
                self.logger.warning(f"   ⚠️ Failed to get API docs: {e}")
                api_docs = ""
            
            # Apply targeted fixes (Note: TargetedErrorFixer doesn't yet use API docs,
            # but we're logging them for future integration)
            self.logger.info("   🔧 Applying pattern-based fixes...")
            fixed_files = self.targeted_fixer.fix_compilation_errors(
                generated_files,
                errors
            )
            
            # Log what was fixed
            fix_summary = self.targeted_fixer.get_fix_summary()
            if fix_summary:
                self.logger.info(f"   ✅ Applied {len(fix_summary)} types of fixes:")
                for fix_type, count in fix_summary.items():
                    self.logger.info(f"      - {fix_type}: {count} fix(es)")
            else:
                self.logger.warning("   ⚠️  No pattern-based fixes applicable")
            
            # Test again to see if fixes helped
            self.logger.info("   📋 Retesting compilation after fixes...")
            retest_result = await self._test_compilation(
                {'files': fixed_files},
                conversion_plan
            )
            
            if retest_result.get('success', False):
                self.logger.info("   🎉 COMPILATION SUCCESS after targeted fixes!")
            else:
                new_errors = retest_result.get('errors', [])
                new_error_count = len([e for e in new_errors if 'error:' in str(e).lower()])
                improvement = error_count - new_error_count
                
                if improvement > 0:
                    self.logger.info(f"   ✅ Errors reduced: {error_count} → {new_error_count} (-{improvement})")
                elif improvement == 0:
                    self.logger.warning(f"   ⚠️  No improvement: {error_count} errors remain")
                else:
                    self.logger.warning(f"   ⚠️  Errors increased: {error_count} → {new_error_count}")
            
            return fixed_files
        else:
            self.logger.warning("   ⚠️  Could not extract error list")
            return generated_files
    
    # ==================== GLOBAL ITERATION FOR MULTI-FILE (DISABLED) ====================
    
    async def _global_error_fixing_iteration(self,
                                            generated_files: Dict[str, str],
                                            conversion_plan: Dict[str, Any],
                                            matlab_analysis: Dict[str, Any],
                                            state: ConversionState) -> Dict[str, Any]:
        """
        Global error-fixing iteration for multi-file projects.
        Fixes cross-file errors that only appear after all files are generated.
        
        This addresses 3 root causes:
        1. MATLAB semantics - shows MATLAB source for context
        2. Cross-file type inference - includes all headers for context
        3. Global iteration - NOW EXISTS for multi-file!
        """
        max_iterations = 3
        best_result = {
            'files': generated_files.copy(),
            'project_name': conversion_plan.get('project_name', 'unknown'),
            'is_multi_file': True,
            'file_count': len(generated_files)
        }
        best_error_count = 999
        
        for iteration in range(1, max_iterations + 1):
            self.logger.info(f"🔄 Global Iteration {iteration}/{max_iterations}")
            
            # Step 1: Test compilation
            self.logger.info(f"   📋 Testing compilation...")
            compilation_result = await self._test_compilation(
                {'files': generated_files}, 
                conversion_plan
            )
            
            # Step 2: Check if successful
            if compilation_result.get('success', False):
                self.logger.info(f"✅ COMPILATION SUCCESS after {iteration} global iteration(s)!")
                return {
                    'files': generated_files,
                    'project_name': conversion_plan.get('project_name', 'unknown'),
                    'is_multi_file': True,
                    'file_count': len(generated_files),
                    'compilation_result': compilation_result
                }
            
            # Step 3: Extract and analyze errors
            compilation_output = compilation_result.get('output', '')
            errors = self._extract_errors_from_output(compilation_output)
            error_count = len(errors)
            
            self.logger.info(f"   ❌ Errors found: {error_count}")
            
            # Step 4: Check progress
            if error_count >= best_error_count:
                self.logger.warning(f"⚠️  No progress: {best_error_count} → {error_count}")
                self.logger.warning(f"   Stopping global iteration")
                break
            
            # Update best result
            best_error_count = error_count
            best_result = {
                'files': generated_files.copy(),
                'project_name': conversion_plan.get('project_name', 'unknown'),
                'is_multi_file': True,
                'file_count': len(generated_files)
            }
            
            # Step 5: Group errors by file
            errors_by_file = self._group_errors_by_file(errors)
            
            if not errors_by_file:
                self.logger.warning(f"   ⚠️  Could not group errors by file")
                break
            
            # Step 6: Select top 3 files to fix
            files_to_fix = sorted(errors_by_file.items(), 
                                 key=lambda x: len(x[1]), 
                                 reverse=True)[:3]
            
            self.logger.info(f"   🎯 Files to fix: {[f[0] for f in files_to_fix]}")
            
            # Step 7: Regenerate problematic files
            for filename, file_errors in files_to_fix:
                self.logger.info(f"   → Fixing {filename} ({len(file_errors)} errors)")
                
                # Find corresponding MATLAB file
                matlab_file = self._find_matlab_source_for_cpp(filename, matlab_analysis)
                
                if not matlab_file:
                    self.logger.warning(f"      ⚠️  Could not find MATLAB source for {filename}")
                    continue
                
                # Generate error fix prompt with cross-file context
                error_fix_prompt = self._build_global_error_fix_prompt(
                    filename,
                    file_errors,
                    generated_files,
                    matlab_analysis,
                    matlab_file
                )
                
                # Regenerate this file with error context
                fixed_code = await self._regenerate_file_with_errors(
                    matlab_file,
                    error_fix_prompt,
                    generated_files,
                    matlab_analysis,
                    state
                )
                
                if fixed_code and 'files' in fixed_code:
                    # Update generated files
                    generated_files.update(fixed_code['files'])
                    self.logger.info(f"      ✅ Regenerated {filename}")
                else:
                    self.logger.warning(f"      ⚠️  Failed to regenerate {filename}")
        
        # Return best result
        self.logger.info("=" * 80)
        self.logger.info(f"✅ Global Iteration Complete")
        self.logger.info(f"   Best error count: {best_error_count}")
        self.logger.info(f"   Files: {list(best_result['files'].keys())}")
        self.logger.info("=" * 80)
        
        return best_result
    
    def _extract_errors_from_output(self, output: str) -> List[str]:
        """
        Extract actual error lines from compilation output.
        Filters out context lines, notes, and instantiation traces to avoid inflated counts.
        
        Only counts lines matching pattern: filename.cpp:line:col: error:
        This ensures consistency with the compilation analyzer's error count.
        """
        errors = []
        for line in output.split('\n'):
            line_stripped = line.strip()
            
            # Only include lines that are actual errors (file:line:col: error:)
            # Skip context lines ("In instantiation"), notes, and suggestions
            if 'error:' in line_stripped.lower():
                # Must match pattern: filename.cpp:line:col: error:
                if re.match(r'^[a-zA-Z0-9_]+\.(?:cpp|h):\d+:\d+:', line_stripped):
                    errors.append(line_stripped)
                # Also catch static assertion errors (different format)
                elif 'static assertion failed' in line_stripped:
                    errors.append(line_stripped)
        
        return errors
    
    def _group_errors_by_file(self, errors: List[str]) -> Dict[str, List[str]]:
        """Group compilation errors by filename."""
        errors_by_file = {}
        
        for error in errors:
            # Extract filename from error message
            # Pattern: "filename.cpp:line:col: error: message"
            match = re.search(r'(\w+\.(?:cpp|h)):\d+:\d+:', str(error))
            if match:
                filename = match.group(1)
                if filename not in errors_by_file:
                    errors_by_file[filename] = []
                errors_by_file[filename].append(error)
        
        return errors_by_file
    
    def _find_matlab_source_for_cpp(self, cpp_filename: str, matlab_analysis: Dict) -> Optional[str]:
        """Find the MATLAB source file for a C++ file."""
        # Remove .cpp or .h extension
        base_name = cpp_filename.replace('.cpp', '').replace('.h', '')
        
        # Look for corresponding .m file
        file_analyses = matlab_analysis.get('file_analyses', [])
        for analysis in file_analyses:
            matlab_file = analysis.get('file_name', '')
            if base_name in matlab_file or matlab_file.replace('.m', '') == base_name:
                return matlab_file
        
        return None
    
    def _build_global_error_fix_prompt(self,
                                      filename: str,
                                      file_errors: List[str],
                                      all_generated_files: Dict[str, str],
                                      matlab_analysis: Dict[str, Any],
                                      matlab_file: str) -> str:
        """
        Build a comprehensive error-fixing prompt with cross-file context.
        
        This is the KEY to fixing cross-file errors:
        - Shows MATLAB source (addresses MATLAB semantics)
        - Shows ALL generated headers (addresses cross-file type inference)
        - Shows specific compilation errors
        
        OPTIMIZATION: Limit MATLAB source to 150 lines to reduce prompt size
        - Covers 80% of files completely (based on skeleton_vessel analysis)
        - Reduces timeout risk while maintaining context quality
        """
        # Get MATLAB source (limit to 150 lines to reduce prompt size)
        matlab_source = ""
        for analysis in matlab_analysis.get('file_analyses', []):
            if analysis.get('file_name') == matlab_file:
                full_source = analysis.get('content', '')
                lines = full_source.split('\n')
                if len(lines) > 150:
                    matlab_source = '\n'.join(lines[:150]) + f"\n... (truncated {len(lines) - 150} lines)"
                else:
                    matlab_source = full_source
                break
        
        # Get current C++ code
        current_code = all_generated_files.get(filename, '')
        
        # Build list of available headers (only function signatures, not full headers)
        available_headers = []
        for fname, content in all_generated_files.items():
            if fname.endswith('.h') and fname != filename:
                # Extract key signatures
                available_headers.append(f"// {fname}")
                # Extract function signatures
                for line in content.split('\n'):
                    if '(' in line and ';' in line and not line.strip().startswith('//'):
                        available_headers.append(f"  {line.strip()}")
        
        # Format errors (limit to first 10 to reduce prompt size)
        errors_to_show = file_errors[:10]
        error_list = "\n".join(f"  {i+1}. {err}" for i, err in enumerate(errors_to_show))
        if len(file_errors) > 10:
            error_list += f"\n  ... and {len(file_errors) - 10} more errors"
        
        # PHASE 1: Get relevant API documentation based on errors
        api_docs = self.api_knowledge_base.get_relevant_docs(
            error_messages=file_errors,
            current_code=current_code,
            libraries_used=None  # Auto-detect
        )
        
        prompt = f"""
🔧 GLOBAL ERROR FIXING - Multi-File Project

You are fixing compilation errors in a multi-file C++ project converted from MATLAB.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 FILE TO FIX: {filename}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❌ COMPILATION ERRORS ({len(file_errors)} errors):
{error_list}
{api_docs}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📄 ORIGINAL MATLAB CODE ({matlab_file}):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```matlab
{matlab_source}
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📚 AVAILABLE FUNCTIONS (from other files):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{chr(10).join(available_headers) if available_headers else "  (none)"}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔧 CURRENT C++ CODE (with errors):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```cpp
{current_code}
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 YOUR TASK:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Fix ALL compilation errors in {filename} by:

1. **Check function signatures** - Use EXACT signatures from "Available Functions"
2. **Check variable types** - Match types from MATLAB semantics and other files
3. **Check return types** - Ensure consistency with function declarations
4. **Check const correctness** - Remove 'const' from modified variables
5. **Fix Eigen API calls** - Use correct Eigen methods (no .hasNonZero(), .hasInf(), etc.)

CRITICAL RULES:
✅ Use ACTUAL function names from "Available Functions" above
✅ Match EXACT types from other files (check return types!)
✅ Refer to MATLAB source to understand original intent
✅ Generate COMPLETE, VALID C++ code
✅ Include proper headers, namespace, and all required code

❌ DO NOT invent function names
❌ DO NOT guess types
❌ DO NOT use non-existent Eigen methods

Generate the FIXED C++ code for {filename} below:
"""
        return prompt
    
    async def _regenerate_file_with_errors(self,
                                          matlab_file: str,
                                          error_fix_prompt: str,
                                          all_generated_files: Dict[str, str],
                                          matlab_analysis: Dict[str, Any],
                                          state: ConversionState) -> Optional[Dict[str, Any]]:
        """
        Regenerate a single file with error-fixing context.
        Uses the error_fix_prompt which includes cross-file context.
        """
        try:
            # Call LLM with error-fixing prompt (not async, despite method being async)
            response = self.llm_client.get_completion(error_fix_prompt)
            
            # Parse response using existing code generation tool
            # Use 'single_file' mode for error-fixing regeneration
            parsed = self.code_generation_tool._parse_generated_code(response, conversion_mode='single_file')
            
            if not parsed or 'files' not in parsed or not parsed['files']:
                self.logger.warning(f"      ⚠️  LLM returned no valid code")
                return None
            
            # Apply syntax fixer to regenerated code
            fixed_files = {}
            for filename, content in parsed['files'].items():
                if filename.endswith('.h'):
                    base_name = filename.replace('.h', '')
                    fixed_header, _ = self.syntax_fixer.fix_all_syntax_issues(content, '', base_name)
                    fixed_files[filename] = fixed_header
                elif filename.endswith('.cpp'):
                    base_name = filename.replace('.cpp', '')
                    _, fixed_impl = self.syntax_fixer.fix_all_syntax_issues('', content, base_name)
                    fixed_files[filename] = fixed_impl
                else:
                    fixed_files[filename] = content
            
            return {'files': fixed_files}
            
        except Exception as e:
            self.logger.error(f"      ❌ Exception regenerating file: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
