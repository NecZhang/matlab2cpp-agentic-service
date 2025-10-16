#include "matlab_array_utils.h"

namespace matlab {
namespace array {

std::vector<int> findNonZero(const Eigen::MatrixXd& matrix) {
    std::vector<int> indices;
    indices.reserve(static_cast<std::size_t>(matrix.size()));
    const int rows = static_cast<int>(matrix.rows());
    const int cols = static_cast<int>(matrix.cols());

    for (int col = 0; col < cols; ++col) {
        for (int row = 0; row < rows; ++row) {
            if (matrix(row, col) != 0.0) {
                int linear_index = col * rows + row + 1;  // MATLAB 1-based column-major index
                indices.push_back(linear_index);
            }
        }
    }

    return indices;
}

void linearIndexToSubscripts(int rows, int cols, int index, int& row, int& col) {
    // ✅ FIXED: Add bounds checking using cols parameter (eliminates warning + improves safety)
    if (index < 1 || index > rows * cols) {
        throw std::out_of_range("Linear index out of bounds for matrix dimensions");
    }
    
    const int zero_based = index - 1;
    row = (zero_based % rows) + 1;
    col = (zero_based / rows) + 1;
}

int subscriptsToLinear(int rows, int cols, int row, int col) {
    // ✅ FIXED: Add bounds checking using cols parameter (eliminates warning + improves safety)
    if (row < 1 || row > rows || col < 1 || col > cols) {
        throw std::out_of_range("Subscripts out of bounds for matrix dimensions");
    }
    
    return (col - 1) * rows + row;
}

}  // namespace array
}  // namespace matlab

