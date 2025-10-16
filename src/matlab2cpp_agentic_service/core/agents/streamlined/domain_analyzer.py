"""
Domain Analysis Module for MATLAB to C++ Conversion

This module provides domain-specific analysis for MATLAB code to recommend
optimal C++ libraries and approaches for different application domains.
"""

from typing import Dict, Any, Tuple


class DomainAnalyzer:
    """Analyzes MATLAB code for different application domains."""
    
    def __init__(self):
        """Initialize domain analyzer."""
        self.domain_patterns = self._initialize_domain_patterns()
        self.library_mappings = self._initialize_library_mappings()
    
    def _initialize_domain_patterns(self) -> Dict[str, Dict[str, list]]:
        """Initialize patterns for different domains."""
        return {
            'linear_algebra': {
                'matrix_ops': ['*', "'", 'inv(', 'eig(', 'svd(', 'qr(', 'chol(', 'lu(', 'eye(', 'zeros('],
                'vector_ops': ['dot(', 'cross(', 'norm(', 'length(', 'size('],
                'decomposition': ['eig(', 'svd(', 'qr(', 'chol(', 'lu(', 'schur('],
                'solving': ['\\', 'mldivide(', 'mrdivide(', 'linsolve(']
            },
            'signal_processing': {
                'transforms': ['fft(', 'ifft(', 'fft2(', 'ifft2(', 'dct(', 'idct(', 'dwt(', 'idwt('],
                'filtering': ['filter(', 'filtfilt(', 'butter(', 'cheby1(', 'cheby2(', 'ellip('],
                'convolution': ['conv(', 'conv2(', 'convn(', 'xcorr(', 'autocorr('],
                'windowing': ['hann(', 'hamming(', 'blackman(', 'kaiser(', 'bartlett(']
            },
            'image_processing': {
                'basic_ops': ['imread(', 'imwrite(', 'imshow(', 'imagesc(', 'colormap('],
                'filtering': ['imfilter(', 'medfilt2(', 'wiener2(', 'fspecial('],
                'morphology': ['imdilate(', 'imerode(', 'imopen(', 'imclose(', 'bwmorph('],
                'segmentation': ['watershed(', 'regionprops(', 'bwlabel(', 'bwareaopen(']
            },
            'optimization': {
                'minimization': ['fminsearch(', 'fminunc(', 'fmincon(', 'linprog(', 'quadprog('],
                'maximization': ['fmaxbnd(', 'fminimax(', 'fseminf('],
                'global_opt': ['ga(', 'simulannealbnd(', 'patternsearch('],
                'curve_fitting': ['fit(', 'lsqcurvefit(', 'lsqnonlin(']
            },
            'statistics': {
                'descriptive': ['mean(', 'median(', 'std(', 'var(', 'min(', 'max(', 'sum('],
                'distributions': ['normpdf(', 'normcdf(', 'chi2pdf(', 'tpdf(', 'fpdf('],
                'hypothesis': ['ttest(', 'ttest2(', 'anova1(', 'kstest(', 'lillietest('],
                'regression': ['regress(', 'polyfit(', 'fitlm(', 'glmfit(']
            },
            'control_systems': {
                'modeling': ['tf(', 'ss(', 'zpk(', 'frd(', 'pid('],
                'analysis': ['bode(', 'nyquist(', 'margin(', 'rlocus(', 'step('],
                'design': ['lqr(', 'place(', 'acker(', 'sisotool(', 'pidtune('],
                'simulation': ['lsim(', 'initial(', 'impulse(', 'sim(']
            },
            'data_analysis': {
                'import_export': ['readtable(', 'writetable(', 'readmatrix(', 'writematrix('],
                'manipulation': ['table(', 'array2table(', 'table2array(', 'join(', 'outerjoin('],
                'visualization': ['plot(', 'scatter(', 'histogram(', 'boxplot(', 'heatmap('],
                'clustering': ['kmeans(', 'linkage(', 'cluster(', 'silhouette(']
            },
            'simulation': {
                'ode_solving': ['ode45(', 'ode23(', 'ode113(', 'ode15s(', 'ode23s('],
                'pde_solving': ['pdepe(', 'pdeval(', 'pdeplot('],
                'monte_carlo': ['rand(', 'randn(', 'randi(', 'random('],
                'time_series': ['timeseries(', 'resample(', 'detrend(']
            }
        }
    
    def _initialize_library_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Initialize library mappings for different domains."""
        return {
            'linear_algebra': {
                'eigen': {
                    'priority': 'high',
                    'reason': 'Matrix operations and linear algebra detected',
                    'performance_gain': '10-100x faster than manual implementation',
                    'headers': ['<Eigen/Dense>', '<Eigen/Sparse>'],
                    'types': ['MatrixXd', 'VectorXd', 'MatrixXf', 'VectorXf'],
                    'threshold': 2
                },
                'armadillo': {
                    'priority': 'medium',
                    'reason': 'MATLAB-like syntax for linear algebra',
                    'performance_gain': '5-50x faster than manual implementation',
                    'headers': ['<armadillo>'],
                    'types': ['mat', 'vec', 'cx_mat', 'cx_vec'],
                    'threshold': 3
                }
            },
            'signal_processing': {
                'fftw': {
                    'priority': 'high',
                    'reason': 'Fast Fourier Transform operations detected',
                    'performance_gain': '5-20x faster than basic FFT',
                    'headers': ['<fftw3.h>'],
                    'types': ['fftw_complex', 'fftw_plan'],
                    'threshold': 1
                },
                'kiss_fft': {
                    'priority': 'medium',
                    'reason': 'Lightweight FFT library',
                    'performance_gain': '2-5x faster than basic FFT',
                    'headers': ['<kiss_fft.h>'],
                    'types': ['kiss_fft_cpx', 'kiss_fft_cfg'],
                    'threshold': 2
                }
            },
            'image_processing': {
                'opencv': {
                    'priority': 'high',
                    'reason': 'Image processing operations detected',
                    'performance_gain': '10-100x faster than manual implementation',
                    'headers': ['<opencv2/opencv.hpp>', '<opencv2/imgproc.hpp>'],
                    'types': ['cv::Mat', 'cv::Scalar', 'cv::Point'],
                    'threshold': 1
                },
                'vxl': {
                    'priority': 'medium',
                    'reason': 'Computer vision and image analysis',
                    'performance_gain': '5-20x faster than manual implementation',
                    'headers': ['<vil/vil_image_view.h>'],
                    'types': ['vil_image_view', 'vil_memory_chunk'],
                    'threshold': 2
                }
            },
            'optimization': {
                'nlopt': {
                    'priority': 'high',
                    'reason': 'Non-linear optimization detected',
                    'performance_gain': '3-10x faster than basic optimization',
                    'headers': ['<nlopt.h>'],
                    'types': ['nlopt_opt', 'nlopt_result'],
                    'threshold': 0
                },
                'ceres': {
                    'priority': 'medium',
                    'reason': 'Non-linear least squares optimization',
                    'performance_gain': '5-15x faster than basic optimization',
                    'headers': ['<ceres/ceres.h>'],
                    'types': ['ceres::Problem', 'ceres::Solver'],
                    'threshold': 1
                }
            },
            'statistics': {
                'boost_math': {
                    'priority': 'medium',
                    'reason': 'Statistical distributions and functions detected',
                    'performance_gain': '2-5x faster than basic implementations',
                    'headers': ['<boost/math/distributions.hpp>'],
                    'types': ['boost::math::normal_distribution', 'boost::math::chi_squared_distribution'],
                    'threshold': 2
                }
            },
            'control_systems': {
                'control_toolbox': {
                    'priority': 'high',
                    'reason': 'Control systems operations detected',
                    'performance_gain': '5-20x faster than manual implementation',
                    'headers': ['<control_toolbox.hpp>'],
                    'types': ['TransferFunction', 'StateSpace', 'PIDController'],
                    'threshold': 1
                }
            }
        }
    
    def analyze_domains(self, source_code: str) -> Dict[str, Any]:
        """Analyze MATLAB code for different application domains."""
        domains = {}
        
        for domain_name, patterns in self.domain_patterns.items():
            domains[domain_name] = self._analyze_single_domain(source_code, patterns)
        
        # Calculate domain scores
        domain_scores = {domain: analysis['score'] for domain, analysis in domains.items()}
        primary_domain = max(domain_scores, key=domain_scores.get) if domain_scores else 'general'
        
        return {
            'domains': domains,
            'domain_scores': domain_scores,
            'primary_domain': primary_domain,
            'has_complex_operations': any(analysis['has_complex_ops'] for analysis in domains.values()),
            'has_parallel_operations': any(analysis['has_parallel_ops'] for analysis in domains.values())
        }
    
    def _analyze_single_domain(self, source_code: str, patterns: Dict[str, list]) -> Dict[str, Any]:
        """Analyze a single domain."""
        scores = {}
        for category, ops in patterns.items():
            scores[category] = sum(1 for op in ops if op in source_code)
        
        total_score = sum(scores.values())
        has_complex_ops = total_score > 2
        has_parallel_ops = any(op in source_code for op in ['parfor', 'parfeval', 'gpuArray', 'UseParallel'])
        
        # Calculate confidence based on domain-specific thresholds
        confidence_threshold = 8.0  # Default threshold
        if total_score > 0:
            confidence_threshold = max(2.0, total_score * 2.0)
        
        return {
            'score': total_score,
            'category_scores': scores,
            'has_complex_ops': has_complex_ops,
            'has_parallel_ops': has_parallel_ops,
            'confidence': min(total_score / confidence_threshold, 1.0)
        }
    
    def get_library_recommendations(self, domain_analysis: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Get library recommendations based on domain analysis."""
        recommendations = {}
        
        for domain_name, domain_info in domain_analysis['domains'].items():
            if domain_name in self.library_mappings and domain_info['score'] > 0:
                domain_libraries = self.library_mappings[domain_name]
                
                for lib_name, lib_info in domain_libraries.items():
                    if domain_info['score'] >= lib_info['threshold']:
                        recommendations[lib_name] = lib_info.copy()
        
        # Default recommendation for general cases
        if not recommendations:
            recommendations['standard'] = {
                'priority': 'low',
                'reason': 'Simple operations detected - standard C++ libraries sufficient',
                'performance_gain': 'No significant performance difference',
                'headers': ['<vector>', '<algorithm>', '<numeric>'],
                'types': ['std::vector', 'std::array', 'std::string'],
                'threshold': 0
            }
        
        return recommendations
    
    def detect_3d_arrays(self, source_code: str) -> bool:
        """Detect if MATLAB code uses 3D arrays."""
        import re
        
        patterns = [
            r'\(\s*\w+\s*,\s*\w+\s*,\s*:\s*\)',  # (i, j, :) - slice notation
            r'\(\s*:\s*,\s*:\s*,\s*\w+\s*\)',    # (:, :, k) - slice notation
            r'\(\s*:\s*,\s*\w+\s*,\s*:\s*\)',    # (:, j, :) - slice notation
            r'size\(\s*\w+\s*,\s*3\s*\)',        # size(x, 3) - third dimension
            r'ndims\(\s*\w+\s*\)\s*[>==]\s*3',   # ndims(x) >= 3
            r'squeeze\s*\(',                      # squeeze often used with 3D
        ]
        
        for pattern in patterns:
            if re.search(pattern, source_code):
                return True
        
        return False
    
    def select_optimal_library(self, recommendations: Dict[str, Dict[str, Any]], 
                             domain_analysis: Dict[str, Any]) -> Tuple[str, str, str]:
        """Select the optimal library based on recommendations and domain analysis."""
        if not recommendations:
            return 'standard', 'No specific domain detected', 'No performance gain expected'
        
        # Sort by priority (high > medium > low)
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        sorted_recommendations = sorted(
            recommendations.items(),
            key=lambda x: priority_order.get(x[1]['priority'], 0),
            reverse=True
        )
        
        # Select the highest priority recommendation
        best_library, best_info = sorted_recommendations[0]
        
        return best_library, best_info['reason'], best_info['performance_gain']


