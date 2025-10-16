#include "msfm_helpers.h"

namespace msfm {
namespace helpers {

// ✅ FIXED: Simplified to just type converters, no circular msfm() calls

Eigen::MatrixXd convert_single_point(const Eigen::VectorXd& point) {
    if (point.size() == 0) {
        return Eigen::MatrixXd::Zero(0, 0);
    }
    Eigen::MatrixXd result(point.size(), 1);
    result.col(0) = point;
    return result;
}

Eigen::MatrixXd convert_single_point(const Eigen::Vector2d& point) {
    Eigen::MatrixXd result(2, 1);
    result.col(0) = point;
    return result;
}

Eigen::MatrixXd convert_point_list(const std::vector<Eigen::VectorXd>& points) {
    if (points.empty()) {
        return Eigen::MatrixXd::Zero(0, 0);
    }

    const Eigen::Index dimension = points.front().size();
    Eigen::MatrixXd result(dimension, static_cast<Eigen::Index>(points.size()));

    for (Eigen::Index idx = 0; idx < static_cast<Eigen::Index>(points.size()); ++idx) {
        if (points[idx].size() != dimension) {
            throw std::invalid_argument("msfm helper: inconsistent source point dimensions");
        }
        result.col(idx) = points[idx];
    }

    return result;
}

Eigen::MatrixXd convert_point_list(const std::vector<Eigen::Vector2d>& points) {
    if (points.empty()) {
        return Eigen::MatrixXd::Zero(2, 0);
    }

    Eigen::MatrixXd result(2, static_cast<Eigen::Index>(points.size()));
    for (Eigen::Index idx = 0; idx < static_cast<Eigen::Index>(points.size()); ++idx) {
        result.col(idx) = points[idx];
    }
    return result;
}

Eigen::MatrixXd vector_to_matrix(const std::vector<Eigen::Vector2i>& points) {
    if (points.empty()) {
        return Eigen::MatrixXd(0, 2);  // 0 rows, 2 columns
    }
    
    Eigen::MatrixXd result(points.size(), 2);
    for (size_t i = 0; i < points.size(); ++i) {
        result(i, 0) = static_cast<double>(points[i].x());
        result(i, 1) = static_cast<double>(points[i].y());
    }
    return result;
}

}  // namespace helpers
}  // namespace msfm

