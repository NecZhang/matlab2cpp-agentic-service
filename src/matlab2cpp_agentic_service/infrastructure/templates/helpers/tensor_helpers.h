#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <tuple>

namespace matlab {
namespace tensor {

// ============================================================================
// 3D TENSOR CREATION & INITIALIZATION
// ============================================================================

/**
 * Create a 3D tensor filled with zeros
 * MATLAB equivalent: zeros(rows, cols, depth)
 */
Eigen::Tensor<double, 3> zeros(int rows, int cols, int depth);

/**
 * Create a 3D tensor filled with ones
 * MATLAB equivalent: ones(rows, cols, depth)
 */
Eigen::Tensor<double, 3> ones(int rows, int cols, int depth);

/**
 * Create a 3D tensor filled with a constant value
 */
Eigen::Tensor<double, 3> constant(int rows, int cols, int depth, double value);

// ============================================================================
// 3D SLICING & EXTRACTION
// ============================================================================

/**
 * Extract a 2D slice from a 3D tensor along a specific dimension
 * 
 * @param tensor The 3D tensor to slice
 * @param index The index along the dimension to extract
 * @param dimension The dimension to slice along (0, 1, or 2)
 * @return 2D matrix view of the slice
 * 
 * MATLAB equivalents:
 *   slice(tensor, index, 2) -> tensor(:, :, index)  % Extract XY plane
 *   slice(tensor, index, 1) -> tensor(:, index, :)  % Extract XZ plane
 *   slice(tensor, index, 0) -> tensor(index, :, :)  % Extract YZ plane
 */
Eigen::MatrixXd slice(const Eigen::Tensor<double, 3>& tensor, int index, int dimension);

/**
 * Extract a 2D slice as a Tensor (rank-2)
 * Returns Eigen::Tensor<double, 2> instead of MatrixXd
 */
Eigen::Tensor<double, 2> slice_tensor(const Eigen::Tensor<double, 3>& tensor, int index, int dimension);

/**
 * Set a 2D slice in a 3D tensor
 * 
 * @param tensor The 3D tensor to modify (modified in-place)
 * @param slice The 2D data to insert
 * @param index The index along the dimension
 * @param dimension The dimension to slice along (0, 1, or 2)
 * 
 * MATLAB equivalent:
 *   tensor(:, :, k) = slice;
 */
void set_slice(Eigen::Tensor<double, 3>& tensor, const Eigen::MatrixXd& slice, 
               int index, int dimension);

// ============================================================================
// GRADIENT COMPUTATION
// ============================================================================

/**
 * Compute 3D gradient of a volume
 * 
 * @param volume Input 3D volume
 * @return Tuple of (Gx, Gy, Gz) gradient tensors
 * 
 * MATLAB equivalent: [Gx, Gy, Gz] = gradient(volume)
 * 
 * Uses central differences:
 *   Gx(i,j,k) = (volume(i+1,j,k) - volume(i-1,j,k)) / 2
 *   Gy(i,j,k) = (volume(i,j+1,k) - volume(i,j-1,k)) / 2
 *   Gz(i,j,k) = (volume(i,j,k+1) - volume(i,j,k-1)) / 2
 */
std::tuple<Eigen::Tensor<double, 3>, Eigen::Tensor<double, 3>, Eigen::Tensor<double, 3>>
gradient3d(const Eigen::Tensor<double, 3>& volume);

/**
 * Compute gradient magnitude at each voxel
 * 
 * @param volume Input 3D volume
 * @return Gradient magnitude tensor: sqrt(Gx^2 + Gy^2 + Gz^2)
 * 
 * MATLAB equivalent:
 *   [Gx, Gy, Gz] = gradient(volume);
 *   mag = sqrt(Gx.^2 + Gy.^2 + Gz.^2);
 */
Eigen::Tensor<double, 3> gradient_magnitude(const Eigen::Tensor<double, 3>& volume);

/**
 * Compute 2D gradient of a matrix
 * 
 * @param matrix Input 2D matrix
 * @return Tuple of (Gx, Gy) gradient matrices
 * 
 * MATLAB equivalent: [Gx, Gy] = gradient(matrix)
 */
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd>
gradient2d(const Eigen::MatrixXd& matrix);

// ============================================================================
// INTERPOLATION
// ============================================================================

/**
 * Bilinear interpolation at a specific point
 * 
 * @param matrix Input 2D matrix
 * @param x X-coordinate (can be fractional)
 * @param y Y-coordinate (can be fractional)
 * @return Interpolated value
 * 
 * MATLAB equivalent: interp2(matrix, x, y, 'linear')
 */
double interp2_linear(const Eigen::MatrixXd& matrix, double x, double y);

/**
 * Trilinear interpolation at a specific point in 3D
 * 
 * @param tensor Input 3D tensor
 * @param x X-coordinate (can be fractional)
 * @param y Y-coordinate (can be fractional)
 * @param z Z-coordinate (can be fractional)
 * @return Interpolated value
 * 
 * MATLAB equivalent: interp3(tensor, x, y, z, 'linear')
 */
double interp3_linear(const Eigen::Tensor<double, 3>& tensor, double x, double y, double z);

// ============================================================================
// TENSOR ↔ MATRIX CONVERSION
// ============================================================================

/**
 * Convert a 2D matrix to a 3D tensor with depth=1
 * 
 * @param matrix Input 2D matrix (rows x cols)
 * @return 3D tensor (rows x cols x 1)
 * 
 * MATLAB equivalent: reshape(matrix, [rows, cols, 1])
 */
Eigen::Tensor<double, 3> matrix_to_tensor(const Eigen::MatrixXd& matrix);

/**
 * Stack multiple 2D matrices into a 3D tensor
 * 
 * @param matrices Vector of 2D matrices (all must have same dimensions)
 * @return 3D tensor (rows x cols x num_matrices)
 * 
 * MATLAB equivalent: cat(3, matrix1, matrix2, matrix3, ...)
 */
Eigen::Tensor<double, 3> stack_matrices(const std::vector<Eigen::MatrixXd>& matrices);

/**
 * Reshape a 3D tensor to a 2D matrix (flattening one dimension)
 * 
 * @param tensor Input 3D tensor
 * @param flatten_dim Dimension to flatten (0, 1, or 2)
 * @return 2D matrix
 * 
 * MATLAB equivalent: reshape(tensor, [], size(tensor, dim))
 */
Eigen::MatrixXd tensor_to_matrix(const Eigen::Tensor<double, 3>& tensor, int flatten_dim);

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Get dimensions of a 3D tensor as a tuple
 * 
 * @param tensor Input 3D tensor
 * @return Tuple of (rows, cols, depth)
 * 
 * MATLAB equivalent: [rows, cols, depth] = size(tensor)
 */
std::tuple<int, int, int> size3d(const Eigen::Tensor<double, 3>& tensor);

/**
 * Check if coordinates are within tensor bounds
 * 
 * @param tensor Input 3D tensor
 * @param i Row index
 * @param j Column index
 * @param k Depth index
 * @return true if (i,j,k) is within bounds
 */
bool in_bounds(const Eigen::Tensor<double, 3>& tensor, int i, int j, int k);

/**
 * Clamp coordinates to tensor bounds
 * 
 * @param tensor Input 3D tensor
 * @param i Row index (will be clamped)
 * @param j Column index (will be clamped)
 * @param k Depth index (will be clamped)
 */
void clamp_to_bounds(const Eigen::Tensor<double, 3>& tensor, int& i, int& j, int& k);

/**
 * Find minimum and maximum values in a 3D tensor
 * 
 * @param tensor Input 3D tensor
 * @return Tuple of (min_value, max_value)
 * 
 * MATLAB equivalent: [min_val, max_val] = bounds(tensor(:))
 */
std::tuple<double, double> minmax(const Eigen::Tensor<double, 3>& tensor);

/**
 * Normalize a 3D tensor to [0, 1] range
 * 
 * @param tensor Input 3D tensor (modified in-place)
 * 
 * MATLAB equivalent: tensor = (tensor - min(tensor(:))) / (max(tensor(:)) - min(tensor(:)))
 */
void normalize(Eigen::Tensor<double, 3>& tensor);

}  // namespace tensor
}  // namespace matlab


