#include "matlab_image_helpers.h"

#include <algorithm>
#include <limits>

namespace matlab {
namespace image {

namespace {
inline int clamp_index(int value, int min_value, int max_value) {
    return std::max(min_value, std::min(value, max_value));
}

inline bool is_active_element(const Eigen::MatrixXd& structuring_element, int row, int col) {
    return structuring_element(row, col) != 0.0;
}
}  // namespace

Eigen::MatrixXd dilate2d(const Eigen::MatrixXd& image,
                         const Eigen::MatrixXd& structuring_element) {
    if (structuring_element.size() == 0) {
        return image;
    }

    const int rows = static_cast<int>(image.rows());
    const int cols = static_cast<int>(image.cols());
    const int se_rows = static_cast<int>(structuring_element.rows());
    const int se_cols = static_cast<int>(structuring_element.cols());
    const int row_radius = se_rows / 2;
    const int col_radius = se_cols / 2;

    Eigen::MatrixXd result(rows, cols);

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            double max_value = -std::numeric_limits<double>::infinity();
            bool has_active = false;

            for (int se_r = 0; se_r < se_rows; ++se_r) {
                for (int se_c = 0; se_c < se_cols; ++se_c) {
                    if (!is_active_element(structuring_element, se_r, se_c)) {
                        continue;
                    }

                    const int image_r = clamp_index(r + se_r - row_radius, 0, rows - 1);
                    const int image_c = clamp_index(c + se_c - col_radius, 0, cols - 1);
                    max_value = std::max(max_value, image(image_r, image_c));
                    has_active = true;
                }
            }

            result(r, c) = has_active ? max_value : image(r, c);
        }
    }

    return result;
}

Eigen::MatrixXd erode2d(const Eigen::MatrixXd& image,
                        const Eigen::MatrixXd& structuring_element) {
    if (structuring_element.size() == 0) {
        return image;
    }

    const int rows = static_cast<int>(image.rows());
    const int cols = static_cast<int>(image.cols());
    const int se_rows = static_cast<int>(structuring_element.rows());
    const int se_cols = static_cast<int>(structuring_element.cols());
    const int row_radius = se_rows / 2;
    const int col_radius = se_cols / 2;

    Eigen::MatrixXd result(rows, cols);

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            double min_value = std::numeric_limits<double>::infinity();
            bool has_active = false;

            for (int se_r = 0; se_r < se_rows; ++se_r) {
                for (int se_c = 0; se_c < se_cols; ++se_c) {
                    if (!is_active_element(structuring_element, se_r, se_c)) {
                        continue;
                    }

                    const int image_r = clamp_index(r + se_r - row_radius, 0, rows - 1);
                    const int image_c = clamp_index(c + se_c - col_radius, 0, cols - 1);
                    min_value = std::min(min_value, image(image_r, image_c));
                    has_active = true;
                }
            }

            result(r, c) = has_active ? min_value : image(r, c);
        }
    }

    return result;
}

Eigen::MatrixXd open2d(const Eigen::MatrixXd& image,
                       const Eigen::MatrixXd& structuring_element) {
    return dilate2d(erode2d(image, structuring_element), structuring_element);
}

Eigen::MatrixXd close2d(const Eigen::MatrixXd& image,
                        const Eigen::MatrixXd& structuring_element) {
    return erode2d(dilate2d(image, structuring_element), structuring_element);
}

}  // namespace image
}  // namespace matlab

