"""
MATLAB Content Analyzer
=========================================

Combines deterministic parsing/heuristics with optional LLM analysis.  The
prompt sent to the LLM is expanded to request a deeper algorithmic
explanation, pseudocode, complexity reasoning, data-flow analysis and
specific C++ translation suggestions.
"""

from __future__ import annotations
import re, hashlib, json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List

try:
    from loguru import logger
except Exception:
    import logging
    _log = logging.getLogger("matlab_content_analyzer")
    _log.setLevel(logging.INFO)
    if not _log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        _log.addHandler(h)
    class _LoggerShim:
        def bind(self, **kwargs): return self
        def info(self, msg): _log.info(msg)
        def warning(self, msg): _log.warning(msg)
    logger = _LoggerShim()

from matlab2cpp_agentic_service.infrastructure.tools.matlab_parser import MATLABParser

@dataclass
class CodeUnderstanding:
    purpose: str
    domain: str
    algorithms: List[str]
    algorithmic_structure: Dict[str, Any]
    data_flow: Dict[str, Any]
    complexity: str
    confidence: float
    challenges: List[str]
    suggestions: List[str]

@dataclass
class ProjectUnderstanding:
    main_purpose: str
    domain: str
    key_algorithms: List[str]
    architecture: str
    complexity_level: str
    conversion_challenges: List[str]
    recommendations: List[str]
    confidence: float

@dataclass
class MATLABFile:
    path: Path
    content: str
    functions: List[str]
    dependencies: List[str]
    numerical_calls: List[str]
    function_calls: Dict[str, List[str]]
    size: int

class MATLABContentAnalyzerAgent:
    def __init__(self, llm_config: Any | None = None) -> None:
        self.parser = MATLABParser()
        self.logger = logger.bind(name="matlab_content_analyzer_agent")
        self.logger.info("MATLAB Content Analyzer Agent (enhanced prompt) initialized")
        self.llm_config = llm_config
        # Create LLM client if config is provided
        if llm_config:
            from matlab2cpp_agentic_service.infrastructure.tools.llm_client import create_llm_client
            self.llm_client = create_llm_client(llm_config)
        else:
            self.llm_client = None
        self._analysis_cache: Dict[tuple[str, str], CodeUnderstanding] = {}

    def analyze_matlab_content(self, path: Path) -> Dict[str, Any]:
        files = self._get_matlab_files(path)
        if not files:
            raise ValueError(f"No MATLAB files found in {path}")
        analyses: List[Dict[str, Any]] = []
        for f in files:
            mf = self._load_matlab_file(f)
            key = (str(f.resolve()), hashlib.md5(mf.content.encode('utf-8')).hexdigest())
            if key in self._analysis_cache:
                cu = self._analysis_cache[key]
            else:
                cu = self._analyze_file_content(mf)
                self._analysis_cache[key] = cu
            analyses.append({'file_path': str(f), 'analysis': cu, 'parsed_structure': mf})
        total_funcs = sum(len(a['parsed_structure'].functions) for a in analyses)
        total_deps  = sum(len(a['parsed_structure'].dependencies) for a in analyses)
        all_deps    = [dep for a in analyses for dep in a['parsed_structure'].dependencies]
        complexity  = self._assess_overall_complexity(analyses)
        project     = self._create_project_understanding(analyses)
        # Build function call tree and dependency resolution
        call_tree = self._build_function_call_tree(analyses)
        dependency_map = self._resolve_dependencies(analyses)
        
        return {
            'files_analyzed': len(analyses),
            'file_analyses': analyses,
            'total_functions': total_funcs,
            'total_dependencies': total_deps,
            'matlab_functions_used': sorted(set(all_deps)),
            'complexity_assessment': complexity,
            'project_understanding': project,
            'function_call_tree': call_tree,
            'dependency_map': dependency_map
        }

    def _get_matlab_files(self, p: Path) -> List[Path]:
        if p.is_file() and p.suffix.lower() == '.m':
            return [p]
        return list(p.glob('**/*.m')) if p.is_dir() else []

    def _load_matlab_file(self, p: Path) -> MATLABFile:
        content = p.read_text(encoding='utf-8', errors='ignore')
        summary = self.parser.parse_project(content)
        return MATLABFile(
            path=p, content=content,
            functions=summary['functions'],
            dependencies=summary['dependencies'],
            numerical_calls=summary['numerical_calls'],
            function_calls=summary['function_calls'],
            size=len(content)
        )

    def _analyze_file_content(self, mf: MATLABFile) -> CodeUnderstanding:
        # Heuristics as before
        purpose, domain = "General MATLAB function", "General"
        algorithms: List[str] = []
        suggestions: List[str] = []
        challenges: List[str] = ["MATLAB 1-based vs C++ 0-based indexing"]

        for call in set(mf.numerical_calls):
            c = call.lower()
            if c in {"fft","ifft","fft2","ifft2","dct","idct","filter","conv","conv2"}:
                if purpose == "General MATLAB function":
                    purpose, domain = "Signal processing function", "Signal processing"
                algorithms.append("FFT/Filter")
                suggestions.append("Use a C++ DSP or FFT library (e.g. FFTW, Eigen FFT)")
            elif c in {"imfilter","imread","imwrite","imshow","imresize","imrotate","medfilt2","rgb2gray"}:
                purpose, domain = "Image processing function", "Image processing"
                algorithms.append("Image processing operations")
                suggestions.append("Use OpenCV or Eigen with libpng/jpg for image ops")
                challenges.append("Handle image formats and channel order differences")
            elif c == "eig":
                purpose, domain = "Linear algebra function","Linear algebra"
                algorithms.append("Eigenvalue decomposition")
                suggestions.append("Use Eigen::SelfAdjointEigenSolver and select the smallest eigenvalue")
                challenges.append("Ensure correct eigenvector selection (smallest eigenvalue)")
            elif c == "svd":
                algorithms.append("Singular Value Decomposition")
                suggestions.append("Use Eigen::BDCSVD or JacobiSVD")
            elif c == "qr":
                algorithms.append("QR Decomposition")
                suggestions.append("Use Eigen::HouseholderQR or ColPivHouseholderQR")
            elif c == "chol":
                algorithms.append("Cholesky Decomposition")
                suggestions.append("Use Eigen::LLT or LDLT")
            elif c in {"ode45","ode23","ode15s","ode113"}:
                purpose, domain = "Numerical integration function","Numerical analysis"
                algorithms.append("Ordinary differential equation solver")
                suggestions.append("Consider Boost::odeint or other C++ ODE solvers")
            elif c in {"fminsearch","fminunc","lsqnonlin"}:
                purpose, domain = "Optimization function","Optimization"
                algorithms.append("Nonlinear optimization")
                suggestions.append("Use Ceres Solver or NLopt")
            elif c in {"svmtrain","svmpredict","kmeans","pca"}:
                purpose, domain = "Machine learning function","Machine learning"
                algorithms.append("Machine learning algorithm")
                suggestions.append("Use mlpack or another C++ ML library")
            elif c in {"inv","pinv"}:
                algorithms.append("Matrix inversion")
                suggestions.append("Avoid explicit inverses; use LDLT/LLT solver")
                challenges.append("Matrix inversion is numerically unstable")

        algorithms = list(dict.fromkeys(algorithms))
        suggestions = list(dict.fromkeys(suggestions))
        challenges  = list(dict.fromkeys(challenges))

        lines = len(mf.content.splitlines())
        loops = len(re.findall(r'\bfor\b|\bwhile\b', mf.content, re.IGNORECASE))
        conds = len(re.findall(r'\bif\b|\bswitch\b', mf.content, re.IGNORECASE))
        if lines > 800 or loops > 20:
            complexity = "High"
        elif lines > 300 or loops > 5 or conds > 10:
            complexity = "Medium"
        else:
            complexity = "Low"

        heur = CodeUnderstanding(
            purpose, domain, algorithms, {}, {},
            complexity, 0.8, challenges, suggestions
        )

        # Use the LLM if available
        if self.llm_client:
            try:
                prompt = self._create_analysis_prompt(mf, heur)
                result = self._perform_llm_analysis(prompt)
                if result:
                    return result
            except Exception as exc:
                self.logger.warning(f"LLM analysis failed: {exc}")
        return heur

    def _create_analysis_prompt(self, mf: MATLABFile, heur: CodeUnderstanding) -> str:
        """
        Build a rich prompt instructing the LLM to analyse the actual algorithm
        in detail.  Requests pseudocode, loop/conditional descriptions,
        matrix operations, data flow and complexity reasoning, plus pitfalls.
        """
        header = (
            "You are a domain expert analysing MATLAB code to guide its conversion "
            "to numerically stable C++.  Focus on the real algorithm, not on "
            "comments or docstrings.  CRITICAL: Analyze the complete algorithmic flow, "
            "including nested loops, matrix construction patterns, and mathematical operations.\n\n"
            "IMPORTANT: Return ONLY a valid JSON object. Do not include any text before or after the JSON.\n\n"
            "Describe the following:\n"
            "- purpose: one sentence summarising what the function does.\n"
            "- domain: a concise phrase indicating the domain (e.g. 'signal processing').\n"
            "- algorithms: the main numerical or computational methods employed.\n"
            "- algorithmic_structure: detailed breakdown of the algorithm including:\n"
            "  * nested loop structure and iteration patterns\n"
            "  * matrix construction methods (Toeplitz, Hankel, etc.)\n"
            "  * mathematical operations and their sequence\n"
            "  * data transformations and flow\n"
            "- pseudocode: a short but precise pseudocode listing key steps, loops, conditionals, "
            "and matrix operations (1-based indexing in MATLAB).\n"
            "- data_flow: describe how data structures (arrays, matrices, variables) are created, "
            "transformed and passed through the function.\n"
            "- complexity: big‑O or qualitative complexity (Low/Medium/High) with justification "
            "(e.g. nested loops, large matrices).\n"
            "- challenges: potential obstacles in translating to C++ (e.g. dynamic indexing, 1-based "
            "vs. 0-based indexing, explicit inverses, memory usage).\n"
            "- suggestions: concrete recommendations for a robust C++ implementation (e.g. use "
            "Eigen::SelfAdjointEigenSolver, use .ldlt().solve() instead of .inverse(), avoid "
            "malloc/free; consider parallelisation).\n\n"
            "Return a JSON object with keys: purpose, domain, algorithms, algorithmic_structure, "
            "pseudocode, data_flow, complexity, challenges, suggestions.\n"
        )

        summary = []
        if mf.functions:
            summary.append("Functions defined: " + ", ".join(mf.functions) + ".")
        if mf.dependencies:
            summary.append("External calls: " + ", ".join(mf.dependencies) + ".")
        if mf.numerical_calls:
            summary.append("Numerical calls: " + ", ".join(mf.numerical_calls) + ".")
        summary_text = "\n".join(summary)

        hints = []
        if heur.purpose != "General MATLAB function":
            hints.append(f"Heuristic purpose: {heur.purpose}.")
        if heur.domain != "General":
            hints.append(f"Heuristic domain: {heur.domain}.")
        if heur.algorithms:
            hints.append("Heuristic algorithms: " + ", ".join(heur.algorithms) + ".")
        hints.append(f"Heuristic complexity: {heur.complexity}.")
        hints.append("Known challenges: " + ", ".join(heur.challenges) + ".")
        if heur.suggestions:
            hints.append("Known suggestions: " + ", ".join(heur.suggestions) + ".")
        hints_text = "\n".join(hints)

        return (
            header +
            "\nMATLAB code:\n" + mf.content.strip() + "\n\n"
            "Parsed summary:\n" + summary_text + "\n\n"
            "Heuristic hints:\n" + hints_text + "\n"
        )

    def _perform_llm_analysis(self, prompt: str) -> CodeUnderstanding | None:
        try:
            response = self.llm_client.get_completion(prompt)
        except Exception as exc:
            self.logger.warning(f"LLM call failed: {exc}")
            return None
        if not response: return None
        
        # Try to extract JSON from response (handle various formats)
        json_data = self._extract_json_from_response(response)
        if not json_data:
            self.logger.warning("Could not extract valid JSON from LLM response")
            return None
            
        try:
            # Validate and extract required fields with fallbacks
            purpose = self._safe_get_string(json_data, 'purpose', "General MATLAB function")
            domain = self._safe_get_string(json_data, 'domain', "General")
            algorithms = self._safe_get_list(json_data, 'algorithms', [])
            algorithmic_structure = self._safe_get_dict(json_data, 'algorithmic_structure', {})
            data_flow = self._safe_get_dict(json_data, 'data_flow', {})
            complexity = self._safe_get_string(json_data, 'complexity', "Medium").capitalize()
            challenges = self._safe_get_list(json_data, 'challenges', ["MATLAB 1-based vs C++ 0-based indexing"])
            suggestions = self._safe_get_list(json_data, 'suggestions', [])
            
            return CodeUnderstanding(
                purpose=purpose,
                domain=domain,
                algorithms=algorithms,
                algorithmic_structure=algorithmic_structure,
                data_flow=data_flow,
                complexity=complexity,
                confidence=0.9,
                challenges=challenges,
                suggestions=suggestions
            )
        except Exception as exc:
            self.logger.warning(f"Failed to create CodeUnderstanding from LLM response: {exc}")
            return None
    
    def _extract_json_from_response(self, response: str) -> dict | None:
        """Extract JSON from LLM response, handling various formats."""
        import re
        
        # Try direct JSON parsing first
        try:
            return json.loads(response.strip())
        except:
            pass
        
        # Try to find JSON block in markdown or other formats
        json_patterns = [
            r'```json\s*(.*?)\s*```',  # Markdown JSON block
            r'```\s*(.*?)\s*```',      # Generic code block
            r'\{.*\}',                 # Any JSON object
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match.strip())
                except:
                    continue
        
        return None
    
    def _safe_get_string(self, data: dict, key: str, default: str) -> str:
        """Safely extract string value from dict."""
        value = data.get(key, default)
        return str(value).strip() if value else default
    
    def _safe_get_list(self, data: dict, key: str, default: list) -> list:
        """Safely extract list value from dict."""
        value = data.get(key, default)
        if isinstance(value, list):
            return [str(item).strip() for item in value if item]
        elif isinstance(value, str):
            return [value.strip()] if value.strip() else default
        return default
    
    def _safe_get_dict(self, data: dict, key: str, default: dict) -> dict:
        """Safely extract dict value from dict."""
        value = data.get(key, default)
        return value if isinstance(value, dict) else default

    def _assess_overall_complexity(self, analyses: List[Dict[str, Any]]) -> str:
        levels = [a['analysis'].complexity for a in analyses]
        return 'High' if 'High' in levels else 'Medium' if 'Medium' in levels else 'Low'

    def _create_project_understanding(self, analyses: List[Dict[str, Any]]) -> ProjectUnderstanding:
        if not analyses:
            return ProjectUnderstanding(
                main_purpose="Unknown project", domain="Unknown",
                key_algorithms=[], architecture="Unknown",
                complexity_level="Unknown",
                conversion_challenges=[], recommendations=[], confidence=0.0
            )
        algorithms = []; challenges = []; suggestions = []; domains = []
        for a in analyses:
            cu = a['analysis']
            algorithms.extend(cu.algorithms)
            challenges.extend(cu.challenges)
            suggestions.extend(cu.suggestions)
            domains.append(cu.domain)
        domain = domains[0] if len(set(domains)) == 1 else 'Mixed'
        return ProjectUnderstanding(
            main_purpose="MATLAB to C++ conversion project",
            domain=domain,
            key_algorithms=sorted(set(algorithms)),
            architecture="Modular C++ design",
            complexity_level=self._assess_overall_complexity(analyses),
            conversion_challenges=sorted(set(challenges)),
            recommendations=sorted(set(suggestions)),
            confidence=0.8
        )

    def _build_function_call_tree(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build a function call tree showing which functions call which other functions.
        
        Returns:
            Dict with 'call_graph' (function -> list of called functions) and 
            'defined_functions' (function -> file where it's defined)
        """
        call_graph = {}
        defined_functions = {}
        
        # First pass: collect all function definitions
        for analysis in analyses:
            file_path = analysis['file_path']
            parsed = analysis['parsed_structure']
            
            for func_name in parsed.functions:
                defined_functions[func_name] = file_path
        
        # Second pass: build call graph
        for analysis in analyses:
            parsed = analysis['parsed_structure']
            
            # Get function calls from the parsed structure
            if hasattr(parsed, 'function_calls'):
                for func_name, called_functions in parsed.function_calls.items():
                    call_graph[func_name] = called_functions
            else:
                # Fallback: if function_calls not available, use dependencies
                for func_name in parsed.functions:
                    call_graph[func_name] = []
        
        return {
            'call_graph': call_graph,
            'defined_functions': defined_functions
        }

    def _resolve_dependencies(self, analyses: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Resolve function dependencies by mapping function calls to their definitions.
        
        Returns:
            Dict mapping function name -> {'defined_in': file, 'called_by': [functions], 'calls': [functions]}
        """
        dependency_map = {}
        
        # First pass: collect all function definitions and their locations
        for analysis in analyses:
            file_path = analysis['file_path']
            parsed = analysis['parsed_structure']
            
            for func_name in parsed.functions:
                if func_name not in dependency_map:
                    dependency_map[func_name] = {
                        'defined_in': file_path,
                        'called_by': [],
                        'calls': []
                    }
        
        # Second pass: build call relationships
        for analysis in analyses:
            parsed = analysis['parsed_structure']
            
            if hasattr(parsed, 'function_calls'):
                for func_name, called_functions in parsed.function_calls.items():
                    if func_name in dependency_map:
                        dependency_map[func_name]['calls'] = called_functions
                        
                        # Update 'called_by' relationships
                        for called_func in called_functions:
                            if called_func in dependency_map:
                                if func_name not in dependency_map[called_func]['called_by']:
                                    dependency_map[called_func]['called_by'].append(func_name)
        
        return dependency_map
