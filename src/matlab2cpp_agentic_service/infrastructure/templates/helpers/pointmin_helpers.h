#pragma once

#include "pointmin.h"

#include <tuple>
#include <unsupported/Eigen/CXX11/Tensor>

namespace pointmin {

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> pointmin2d(const Eigen::MatrixXd& image);
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> pointmin3d(const Eigen::Tensor<double, 3>& image);

}  // namespace pointmin

