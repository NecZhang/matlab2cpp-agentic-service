#pragma once

#include <Eigen/Dense>

namespace matlab {
namespace image {

Eigen::MatrixXd dilate2d(const Eigen::MatrixXd& image,
                         const Eigen::MatrixXd& structuring_element);

Eigen::MatrixXd erode2d(const Eigen::MatrixXd& image,
                        const Eigen::MatrixXd& structuring_element);

Eigen::MatrixXd open2d(const Eigen::MatrixXd& image,
                       const Eigen::MatrixXd& structuring_element);

Eigen::MatrixXd close2d(const Eigen::MatrixXd& image,
                        const Eigen::MatrixXd& structuring_element);

}  // namespace image
}  // namespace matlab

