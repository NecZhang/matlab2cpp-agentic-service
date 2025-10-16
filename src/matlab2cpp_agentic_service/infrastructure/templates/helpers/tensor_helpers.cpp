#include "tensor_helpers.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace matlab {
namespace tensor {

// ============================================================================
// 3D TENSOR CREATION & INITIALIZATION
// ============================================================================

Eigen::Tensor<double, 3> zeros(int rows, int cols, int depth) {
    Eigen::Tensor<double, 3> result(rows, cols, depth);
    result.setZero();
    return result;
}

Eigen::Tensor<double, 3> ones(int rows, int cols, int depth) {
    Eigen::Tensor<double, 3> result(rows, cols, depth);
    result.setConstant(1.0);
    return result;
}

Eigen::Tensor<double, 3> constant(int rows, int cols, int depth, double value) {
    Eigen::Tensor<double, 3> result(rows, cols, depth);
    result.setConstant(value);
    return result;
}

// ============================================================================
// 3D SLICING & EXTRACTION
// ============================================================================

Eigen::MatrixXd slice(const Eigen::Tensor<double, 3>& tensor, int index, int dimension) {
    const int rows = tensor.dimension(0);
    const int cols = tensor.dimension(1);
    const int depth = tensor.dimension(2);
    
    if (dimension == 2) {
        // Extract tensor(:, :, index) -> rows x cols matrix
        if (index < 0 || index >= depth) {
            throw std::out_of_range("Slice index out of bounds for dimension 2");
        }
        
        // ✅ FIXED: Evaluate expression template to Tensor<double, 2> first
        Eigen::Tensor<double, 2> slice_2d = tensor.chip(index, 2);
        Eigen::MatrixXd result(rows, cols);
        
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result(i, j) = slice_2d(i, j);
            }
        }
        return result;
        
    } else if (dimension == 1) {
        // Extract tensor(:, index, :) -> rows x depth matrix
        if (index < 0 || index >= cols) {
            throw std::out_of_range("Slice index out of bounds for dimension 1");
        }
        
        // ✅ FIXED: Evaluate expression template to Tensor<double, 2> first
        Eigen::Tensor<double, 2> slice_2d = tensor.chip(index, 1);
        Eigen::MatrixXd result(rows, depth);
        
        for (int i = 0; i < rows; ++i) {
            for (int k = 0; k < depth; ++k) {
                result(i, k) = slice_2d(i, k);
            }
        }
        return result;
        
    } else if (dimension == 0) {
        // Extract tensor(index, :, :) -> cols x depth matrix
        if (index < 0 || index >= rows) {
            throw std::out_of_range("Slice index out of bounds for dimension 0");
        }
        
        // ✅ FIXED: Evaluate expression template to Tensor<double, 2> first
        Eigen::Tensor<double, 2> slice_2d = tensor.chip(index, 0);
        Eigen::MatrixXd result(cols, depth);
        
        for (int j = 0; j < cols; ++j) {
            for (int k = 0; k < depth; ++k) {
                result(j, k) = slice_2d(j, k);
            }
        }
        return result;
        
    } else {
        throw std::invalid_argument("Dimension must be 0, 1, or 2");
    }
}

Eigen::Tensor<double, 2> slice_tensor(const Eigen::Tensor<double, 3>& tensor, int index, int dimension) {
    return tensor.chip(index, dimension);
}

void set_slice(Eigen::Tensor<double, 3>& tensor, const Eigen::MatrixXd& slice, 
               int index, int dimension) {
    const int rows = tensor.dimension(0);
    const int cols = tensor.dimension(1);
    const int depth = tensor.dimension(2);
    
    if (dimension == 2) {
        // Set tensor(:, :, index) = slice
        if (index < 0 || index >= depth) {
            throw std::out_of_range("Slice index out of bounds for dimension 2");
        }
        if (slice.rows() != rows || slice.cols() != cols) {
            throw std::invalid_argument("Slice dimensions don't match tensor dimensions");
        }
        
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                tensor(i, j, index) = slice(i, j);
            }
        }
        
    } else if (dimension == 1) {
        // Set tensor(:, index, :) = slice
        if (index < 0 || index >= cols) {
            throw std::out_of_range("Slice index out of bounds for dimension 1");
        }
        if (slice.rows() != rows || slice.cols() != depth) {
            throw std::invalid_argument("Slice dimensions don't match tensor dimensions");
        }
        
        for (int i = 0; i < rows; ++i) {
            for (int k = 0; k < depth; ++k) {
                tensor(i, index, k) = slice(i, k);
            }
        }
        
    } else if (dimension == 0) {
        // Set tensor(index, :, :) = slice
        if (index < 0 || index >= rows) {
            throw std::out_of_range("Slice index out of bounds for dimension 0");
        }
        if (slice.rows() != cols || slice.cols() != depth) {
            throw std::invalid_argument("Slice dimensions don't match tensor dimensions");
        }
        
        for (int j = 0; j < cols; ++j) {
            for (int k = 0; k < depth; ++k) {
                tensor(index, j, k) = slice(j, k);
            }
        }
        
    } else {
        throw std::invalid_argument("Dimension must be 0, 1, or 2");
    }
}

// ============================================================================
// GRADIENT COMPUTATION
// ============================================================================

std::tuple<Eigen::Tensor<double, 3>, Eigen::Tensor<double, 3>, Eigen::Tensor<double, 3>>
gradient3d(const Eigen::Tensor<double, 3>& volume) {
    const int rows = volume.dimension(0);
    const int cols = volume.dimension(1);
    const int depth = volume.dimension(2);
    
    Eigen::Tensor<double, 3> Gx(rows, cols, depth);
    Eigen::Tensor<double, 3> Gy(rows, cols, depth);
    Eigen::Tensor<double, 3> Gz(rows, cols, depth);
    
    Gx.setZero();
    Gy.setZero();
    Gz.setZero();
    
    // Compute gradients using central differences (interior points)
    // and forward/backward differences (boundary points)
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            for (int k = 0; k < depth; ++k) {
                // Gradient in X direction
                if (i == 0) {
                    // Forward difference
                    Gx(i, j, k) = volume(i+1, j, k) - volume(i, j, k);
                } else if (i == rows - 1) {
                    // Backward difference
                    Gx(i, j, k) = volume(i, j, k) - volume(i-1, j, k);
                } else {
                    // Central difference
                    Gx(i, j, k) = (volume(i+1, j, k) - volume(i-1, j, k)) / 2.0;
                }
                
                // Gradient in Y direction
                if (j == 0) {
                    Gy(i, j, k) = volume(i, j+1, k) - volume(i, j, k);
                } else if (j == cols - 1) {
                    Gy(i, j, k) = volume(i, j, k) - volume(i, j-1, k);
                } else {
                    Gy(i, j, k) = (volume(i, j+1, k) - volume(i, j-1, k)) / 2.0;
                }
                
                // Gradient in Z direction
                if (k == 0) {
                    Gz(i, j, k) = volume(i, j, k+1) - volume(i, j, k);
                } else if (k == depth - 1) {
                    Gz(i, j, k) = volume(i, j, k) - volume(i, j, k-1);
                } else {
                    Gz(i, j, k) = (volume(i, j, k+1) - volume(i, j, k-1)) / 2.0;
                }
            }
        }
    }
    
    return std::make_tuple(Gx, Gy, Gz);
}

Eigen::Tensor<double, 3> gradient_magnitude(const Eigen::Tensor<double, 3>& volume) {
    auto [Gx, Gy, Gz] = gradient3d(volume);
    
    const int rows = volume.dimension(0);
    const int cols = volume.dimension(1);
    const int depth = volume.dimension(2);
    
    Eigen::Tensor<double, 3> magnitude(rows, cols, depth);
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            for (int k = 0; k < depth; ++k) {
                double gx = Gx(i, j, k);
                double gy = Gy(i, j, k);
                double gz = Gz(i, j, k);
                magnitude(i, j, k) = std::sqrt(gx*gx + gy*gy + gz*gz);
            }
        }
    }
    
    return magnitude;
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd>
gradient2d(const Eigen::MatrixXd& matrix) {
    const int rows = matrix.rows();
    const int cols = matrix.cols();
    
    Eigen::MatrixXd Gx = Eigen::MatrixXd::Zero(rows, cols);
    Eigen::MatrixXd Gy = Eigen::MatrixXd::Zero(rows, cols);
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Gradient in X direction
            if (i == 0) {
                Gx(i, j) = matrix(i+1, j) - matrix(i, j);
            } else if (i == rows - 1) {
                Gx(i, j) = matrix(i, j) - matrix(i-1, j);
            } else {
                Gx(i, j) = (matrix(i+1, j) - matrix(i-1, j)) / 2.0;
            }
            
            // Gradient in Y direction
            if (j == 0) {
                Gy(i, j) = matrix(i, j+1) - matrix(i, j);
            } else if (j == cols - 1) {
                Gy(i, j) = matrix(i, j) - matrix(i, j-1);
            } else {
                Gy(i, j) = (matrix(i, j+1) - matrix(i, j-1)) / 2.0;
            }
        }
    }
    
    return std::make_tuple(Gx, Gy);
}

// ============================================================================
// INTERPOLATION
// ============================================================================

double interp2_linear(const Eigen::MatrixXd& matrix, double x, double y) {
    const int rows = matrix.rows();
    const int cols = matrix.cols();
    
    // Clamp to valid range
    x = std::max(0.0, std::min(x, static_cast<double>(rows - 1)));
    y = std::max(0.0, std::min(y, static_cast<double>(cols - 1)));
    
    int x0 = static_cast<int>(std::floor(x));
    int x1 = std::min(x0 + 1, rows - 1);
    int y0 = static_cast<int>(std::floor(y));
    int y1 = std::min(y0 + 1, cols - 1);
    
    double dx = x - x0;
    double dy = y - y0;
    
    // Bilinear interpolation
    double v00 = matrix(x0, y0);
    double v01 = matrix(x0, y1);
    double v10 = matrix(x1, y0);
    double v11 = matrix(x1, y1);
    
    double v0 = v00 * (1 - dy) + v01 * dy;
    double v1 = v10 * (1 - dy) + v11 * dy;
    
    return v0 * (1 - dx) + v1 * dx;
}

double interp3_linear(const Eigen::Tensor<double, 3>& tensor, double x, double y, double z) {
    const int rows = tensor.dimension(0);
    const int cols = tensor.dimension(1);
    const int depth = tensor.dimension(2);
    
    // Clamp to valid range
    x = std::max(0.0, std::min(x, static_cast<double>(rows - 1)));
    y = std::max(0.0, std::min(y, static_cast<double>(cols - 1)));
    z = std::max(0.0, std::min(z, static_cast<double>(depth - 1)));
    
    int x0 = static_cast<int>(std::floor(x));
    int x1 = std::min(x0 + 1, rows - 1);
    int y0 = static_cast<int>(std::floor(y));
    int y1 = std::min(y0 + 1, cols - 1);
    int z0 = static_cast<int>(std::floor(z));
    int z1 = std::min(z0 + 1, depth - 1);
    
    double dx = x - x0;
    double dy = y - y0;
    double dz = z - z0;
    
    // Trilinear interpolation
    double v000 = tensor(x0, y0, z0);
    double v001 = tensor(x0, y0, z1);
    double v010 = tensor(x0, y1, z0);
    double v011 = tensor(x0, y1, z1);
    double v100 = tensor(x1, y0, z0);
    double v101 = tensor(x1, y0, z1);
    double v110 = tensor(x1, y1, z0);
    double v111 = tensor(x1, y1, z1);
    
    double v00 = v000 * (1 - dz) + v001 * dz;
    double v01 = v010 * (1 - dz) + v011 * dz;
    double v10 = v100 * (1 - dz) + v101 * dz;
    double v11 = v110 * (1 - dz) + v111 * dz;
    
    double v0 = v00 * (1 - dy) + v01 * dy;
    double v1 = v10 * (1 - dy) + v11 * dy;
    
    return v0 * (1 - dx) + v1 * dx;
}

// ============================================================================
// TENSOR ↔ MATRIX CONVERSION
// ============================================================================

Eigen::Tensor<double, 3> matrix_to_tensor(const Eigen::MatrixXd& matrix) {
    const int rows = matrix.rows();
    const int cols = matrix.cols();
    
    Eigen::Tensor<double, 3> result(rows, cols, 1);
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result(i, j, 0) = matrix(i, j);
        }
    }
    
    return result;
}

Eigen::Tensor<double, 3> stack_matrices(const std::vector<Eigen::MatrixXd>& matrices) {
    if (matrices.empty()) {
        throw std::invalid_argument("Cannot stack empty vector of matrices");
    }
    
    const int rows = matrices[0].rows();
    const int cols = matrices[0].cols();
    const int depth = matrices.size();
    
    // Verify all matrices have same dimensions
    for (size_t i = 1; i < matrices.size(); ++i) {
        if (matrices[i].rows() != rows || matrices[i].cols() != cols) {
            throw std::invalid_argument("All matrices must have the same dimensions");
        }
    }
    
    Eigen::Tensor<double, 3> result(rows, cols, depth);
    
    for (int k = 0; k < depth; ++k) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result(i, j, k) = matrices[k](i, j);
            }
        }
    }
    
    return result;
}

Eigen::MatrixXd tensor_to_matrix(const Eigen::Tensor<double, 3>& tensor, int flatten_dim) {
    const int rows = tensor.dimension(0);
    const int cols = tensor.dimension(1);
    const int depth = tensor.dimension(2);
    
    if (flatten_dim == 2) {
        // Flatten along depth: result is (rows*depth) x cols
        Eigen::MatrixXd result(rows * depth, cols);
        
        for (int k = 0; k < depth; ++k) {
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    result(i + k * rows, j) = tensor(i, j, k);
                }
            }
        }
        return result;
        
    } else if (flatten_dim == 1) {
        // Flatten along cols: result is (rows*cols) x depth
        Eigen::MatrixXd result(rows * cols, depth);
        
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                for (int k = 0; k < depth; ++k) {
                    result(i * cols + j, k) = tensor(i, j, k);
                }
            }
        }
        return result;
        
    } else if (flatten_dim == 0) {
        // Flatten along rows: result is rows x (cols*depth)
        Eigen::MatrixXd result(rows, cols * depth);
        
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                for (int k = 0; k < depth; ++k) {
                    result(i, j * depth + k) = tensor(i, j, k);
                }
            }
        }
        return result;
        
    } else {
        throw std::invalid_argument("Flatten dimension must be 0, 1, or 2");
    }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

std::tuple<int, int, int> size3d(const Eigen::Tensor<double, 3>& tensor) {
    return std::make_tuple(
        tensor.dimension(0),
        tensor.dimension(1),
        tensor.dimension(2)
    );
}

bool in_bounds(const Eigen::Tensor<double, 3>& tensor, int i, int j, int k) {
    return i >= 0 && i < tensor.dimension(0) &&
           j >= 0 && j < tensor.dimension(1) &&
           k >= 0 && k < tensor.dimension(2);
}

void clamp_to_bounds(const Eigen::Tensor<double, 3>& tensor, int& i, int& j, int& k) {
    i = std::max(0, std::min(i, static_cast<int>(tensor.dimension(0)) - 1));
    j = std::max(0, std::min(j, static_cast<int>(tensor.dimension(1)) - 1));
    k = std::max(0, std::min(k, static_cast<int>(tensor.dimension(2)) - 1));
}

std::tuple<double, double> minmax(const Eigen::Tensor<double, 3>& tensor) {
    const int rows = tensor.dimension(0);
    const int cols = tensor.dimension(1);
    const int depth = tensor.dimension(2);
    
    double min_val = tensor(0, 0, 0);
    double max_val = tensor(0, 0, 0);
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            for (int k = 0; k < depth; ++k) {
                double val = tensor(i, j, k);
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
            }
        }
    }
    
    return std::make_tuple(min_val, max_val);
}

void normalize(Eigen::Tensor<double, 3>& tensor) {
    auto [min_val, max_val] = minmax(tensor);
    
    if (max_val == min_val) {
        tensor.setZero();
        return;
    }
    
    const int rows = tensor.dimension(0);
    const int cols = tensor.dimension(1);
    const int depth = tensor.dimension(2);
    
    double range = max_val - min_val;
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            for (int k = 0; k < depth; ++k) {
                tensor(i, j, k) = (tensor(i, j, k) - min_val) / range;
            }
        }
    }
}

}  // namespace tensor
}  // namespace matlab


