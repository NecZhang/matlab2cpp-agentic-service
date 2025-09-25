# /src/matlab2cpp_agentic_service/tools/matlab_parser.py
# or /src/matlab2cpp_agent/agents/matlab_parser.py
"""
A lightweight MATLAB parser for the agentic conversion service.

This parser identifies:
  - function definitions
  - external dependencies (user-defined or toolbox functions)
  - special numerical calls (FFT, eig, etc.)
It uses simple regular expressions rather than a full MATLAB grammar.

Two key sets control classification:
  - BUILTIN_FUNCTIONS: functions that map to trivial C++ operations and should never be reported as dependencies.
  - NUMERICAL_FUNCTIONS: calls that require special handling (e.g. FFT, SVD, ODE solvers) and will drive library selection.
"""

import re
from typing import List, Dict, Set, Tuple, Optional

class MATLABParser:
    """
    Extracts basic structure from MATLAB code.

    Methods
    -------
    parse_functions(code: str) -> List[str]
        Return names of functions defined in the file.

    parse_dependencies(code: str) -> (Set[str], Set[str])
        Return two sets:
            - external dependencies (calls not in BUILTIN_FUNCTIONS)
            - numerical calls (calls in NUMERICAL_FUNCTIONS)

    parse_function_calls(code: str) -> Dict[str, List[str]]
        Return a mapping of function name -> list of functions it calls.

    parse_project(code: str) -> Dict[str, object]
        Summarise the file with keys: functions, dependencies,
        numerical_calls, function_calls and num_lines.
    """

    # Built‑ins: common math, random, logging/plotting, image ops, etc.
    BUILTIN_FUNCTIONS: Set[str] = {
        'zeros','ones','eye','size','sum','mean','abs','disp','sqrt','pi','sin','cos','tan',
        'floor','ceil','round','min','max','length','numel','reshape','linspace','mod',
        'real','imag','conj','log','log10','exp','log2','exp2','rand','randn','tic','toc',
        'subplot','plot','title','xlabel','ylabel','surf','mesh','image','imagesc','axis',
        'legend','rgb2gray','meshgrid'
    }

    # Numerical routines needing C++ library support (Eigen, FFTW, OpenCV, etc.)
    NUMERICAL_FUNCTIONS: Set[str] = {
        'eig','svd','chol','qr','lu','inv','pinv','squeeze',        # linear algebra
        'fft','ifft','fft2','ifft2','fftshift','ifftshift','conv','conv2','filter','dct','idct',
        'imfilter','imread','imwrite','imshow','imresize','imrotate','medfilt2','imadjust','rgb2gray',
        'ode45','ode23','ode15s','ode113','fminsearch','fminunc','lsqnonlin','optimset',
        'svmtrain','svmpredict','kmeans','pca','var','cov','corr','hist','cconv'
    }

    FUNCTION_DEF_PATTERN = re.compile(r'function\s+[^=]*=\s*(\w+)\s*\(')
    FUNCTION_CALL_PATTERN = re.compile(r'(?P<name>\b\w+\b)\s*\(')
    # Pattern to match function definitions with their body
    FUNCTION_WITH_BODY_PATTERN = re.compile(
        r'function\s+(?:[^=\n]*=\s*)?(\w+)\s*\([^)]*\)(.*?)(?=\nfunction|\Z)', 
        re.DOTALL
    )

    def parse_functions(self, code: str) -> List[str]:
        """Return a list of function names defined in the MATLAB code."""
        return [m.group(1) for m in self.FUNCTION_DEF_PATTERN.finditer(code)]

    def parse_dependencies(self, code: str) -> Tuple[Set[str], Set[str]]:
        """
        Parse function calls and split them into dependencies and numerical calls.

        Returns:
            deps   -- calls not in BUILTIN_FUNCTIONS or NUMERICAL_FUNCTIONS
            numeric-- calls in NUMERICAL_FUNCTIONS
        """
        deps: Set[str] = set()
        numeric: Set[str] = set()
        for match in self.FUNCTION_CALL_PATTERN.finditer(code):
            name = match.group('name')
            if name in self.BUILTIN_FUNCTIONS:
                continue
            if name in self.NUMERICAL_FUNCTIONS:
                numeric.add(name)
            else:
                deps.add(name)
        return deps, numeric

    def parse_function_calls(self, code: str) -> Dict[str, List[str]]:
        """
        Parse function calls within each function definition.
        
        Returns a mapping of function name -> list of functions it calls.
        This helps build a call tree for multi-file projects.
        """
        function_calls: Dict[str, List[str]] = {}
        
        # Find all function definitions with their bodies
        for match in self.FUNCTION_WITH_BODY_PATTERN.finditer(code):
            func_name = match.group(1)
            func_body = match.group(2)
            
            # Extract function calls from the function body
            calls = set()
            for call_match in self.FUNCTION_CALL_PATTERN.finditer(func_body):
                called_func = call_match.group('name')
                
                # Skip built-ins and numerical functions (they don't need dependency tracking)
                if (called_func not in self.BUILTIN_FUNCTIONS and 
                    called_func not in self.NUMERICAL_FUNCTIONS and
                    called_func != func_name):  # Skip self-calls
                    calls.add(called_func)
            
            function_calls[func_name] = sorted(list(calls))
        
        return function_calls

    def parse_project(self, code: str) -> Dict[str, object]:
        """
        Summarise the MATLAB file.

        Returns a dictionary with:
            'functions'      -- list of defined function names
            'dependencies'   -- sorted list of external calls
            'numerical_calls'-- sorted list of special numerical calls
            'function_calls' -- mapping of function name -> list of functions it calls
            'num_lines'      -- number of lines in the file
        """
        funcs = self.parse_functions(code)
        deps, nums = self.parse_dependencies(code)
        func_calls = self.parse_function_calls(code)
        return {
            'functions': funcs,
            'dependencies': sorted(deps),
            'numerical_calls': sorted(nums),
            'function_calls': func_calls,
            'num_lines': len(code.splitlines())
        }
