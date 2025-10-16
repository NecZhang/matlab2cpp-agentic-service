#include "pointmin_helpers.h"

#include <stdexcept>

namespace pointmin {

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> pointmin2d(const Eigen::MatrixXd& image) {
    auto result = pointmin(image);
    return std::make_tuple(std::get<1>(result), std::get<0>(result));
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> pointmin3d(const Eigen::Tensor<double, 3>&) {
    throw std::runtime_error("pointmin3d helper is not implemented yet");
}

}  // namespace pointmin

