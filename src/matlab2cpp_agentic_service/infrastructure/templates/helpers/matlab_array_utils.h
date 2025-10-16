#pragma once

#include <Eigen/Dense>

#include <vector>

namespace matlab {
namespace array {

std::vector<int> findNonZero(const Eigen::MatrixXd& matrix);
void linearIndexToSubscripts(int rows, int cols, int index, int& row, int& col);
int subscriptsToLinear(int rows, int cols, int row, int col);

}  // namespace array
}  // namespace matlab

