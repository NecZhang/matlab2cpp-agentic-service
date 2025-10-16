#pragma once

#include <Eigen/Dense>

#include <stdexcept>
#include <vector>

namespace msfm {
namespace helpers {

// ✅ FIXED: Renamed to avoid ambiguity with generated msfm::msfm() function
// These are just type conversion utilities, not the main MSFM implementation

Eigen::MatrixXd convert_single_point(const Eigen::VectorXd& point);
Eigen::MatrixXd convert_single_point(const Eigen::Vector2d& point);
Eigen::MatrixXd convert_point_list(const std::vector<Eigen::VectorXd>& points);
Eigen::MatrixXd convert_point_list(const std::vector<Eigen::Vector2d>& points);

// Convert vector of 2D integer points to Nx2 matrix (for MSFM source points)
Eigen::MatrixXd vector_to_matrix(const std::vector<Eigen::Vector2i>& points);

}  // namespace helpers
}  // namespace msfm

