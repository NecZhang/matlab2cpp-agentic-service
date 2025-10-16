#include "rk4_helpers.h"
#include "tensor_helpers.h"
#include <cmath>
#include <algorithm>

namespace matlab {
namespace rk4 {

// ============================================================================
// RUNGE-KUTTA 4TH ORDER INTEGRATION
// ============================================================================

Eigen::VectorXd rk4_step(const ODEFunction& f, double t, const Eigen::VectorXd& y, double h) {
    // Classic RK4 formula
    Eigen::VectorXd k1 = f(t, y);
    Eigen::VectorXd k2 = f(t + h/2.0, y + h*k1/2.0);
    Eigen::VectorXd k3 = f(t + h/2.0, y + h*k2/2.0);
    Eigen::VectorXd k4 = f(t + h, y + h*k3);
    
    return y + h/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4);
}

std::pair<std::vector<double>, std::vector<Eigen::VectorXd>>
rk4_integrate(const ODEFunction& f, const Eigen::Vector2d& t_span, 
              const Eigen::VectorXd& y0, int num_steps) {
    double t0 = t_span(0);
    double tf = t_span(1);
    double h = (tf - t0) / num_steps;
    
    std::vector<double> t_points;
    std::vector<Eigen::VectorXd> y_points;
    
    t_points.reserve(num_steps + 1);
    y_points.reserve(num_steps + 1);
    
    double t = t0;
    Eigen::VectorXd y = y0;
    
    t_points.push_back(t);
    y_points.push_back(y);
    
    for (int i = 0; i < num_steps; ++i) {
        y = rk4_step(f, t, y, h);
        t += h;
        
        t_points.push_back(t);
        y_points.push_back(y);
    }
    
    return std::make_pair(t_points, y_points);
}

// ============================================================================
// GRADIENT FIELD TRACING (for shortest path algorithms)
// ============================================================================

Eigen::MatrixXd trace_gradient_field(
    const Eigen::MatrixXd& gradient_x,
    const Eigen::MatrixXd& gradient_y,
    const Eigen::Vector2d& start_point,
    double step_size,
    int max_iterations,
    double convergence_threshold
) {
    std::vector<Eigen::Vector2d> path_points;
    path_points.reserve(max_iterations);
    
    Eigen::Vector2d current_point = start_point;
    path_points.push_back(current_point);
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Check if we're still within bounds
        if (!is_in_bounds(gradient_x, current_point(0), current_point(1), 2.0)) {
            break;
        }
        
        // RK4 integration step
        // k1 = gradient at current point
        Eigen::Vector2d k1 = interpolate_gradient(gradient_x, gradient_y, 
                                                   current_point(0), current_point(1));
        
        // k2 = gradient at midpoint using k1
        Eigen::Vector2d mid1 = current_point + step_size * k1 / 2.0;
        Eigen::Vector2d k2 = interpolate_gradient(gradient_x, gradient_y, 
                                                   mid1(0), mid1(1));
        
        // k3 = gradient at midpoint using k2
        Eigen::Vector2d mid2 = current_point + step_size * k2 / 2.0;
        Eigen::Vector2d k3 = interpolate_gradient(gradient_x, gradient_y, 
                                                   mid2(0), mid2(1));
        
        // k4 = gradient at endpoint using k3
        Eigen::Vector2d end = current_point + step_size * k3;
        Eigen::Vector2d k4 = interpolate_gradient(gradient_x, gradient_y, 
                                                   end(0), end(1));
        
        // Compute next point using RK4 formula
        Eigen::Vector2d next_point = current_point + 
            step_size / 6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4);
        
        // Check convergence
        double movement = (next_point - current_point).norm();
        if (movement < convergence_threshold) {
            break;
        }
        
        current_point = next_point;
        path_points.push_back(current_point);
    }
    
    // Convert to matrix format (2 x num_points)
    Eigen::MatrixXd result(2, path_points.size());
    for (size_t i = 0; i < path_points.size(); ++i) {
        result(0, i) = path_points[i](0);
        result(1, i) = path_points[i](1);
    }
    
    return result;
}

Eigen::MatrixXd trace_path_rk4(
    const GradientFieldFunction& velocity_func,
    const Eigen::Vector2d& start_point,
    double step_size,
    int max_iterations,
    double convergence_threshold
) {
    std::vector<Eigen::Vector2d> path_points;
    path_points.reserve(max_iterations);
    
    Eigen::Vector2d current_point = start_point;
    path_points.push_back(current_point);
    
    // Dummy gradient fields (will be passed to velocity_func)
    Eigen::MatrixXd dummy_gx, dummy_gy;
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        // RK4 integration with custom velocity function
        Eigen::Vector2d k1 = velocity_func(current_point(0), current_point(1), dummy_gx, dummy_gy);
        
        Eigen::Vector2d mid1 = current_point + step_size * k1 / 2.0;
        Eigen::Vector2d k2 = velocity_func(mid1(0), mid1(1), dummy_gx, dummy_gy);
        
        Eigen::Vector2d mid2 = current_point + step_size * k2 / 2.0;
        Eigen::Vector2d k3 = velocity_func(mid2(0), mid2(1), dummy_gx, dummy_gy);
        
        Eigen::Vector2d end = current_point + step_size * k3;
        Eigen::Vector2d k4 = velocity_func(end(0), end(1), dummy_gx, dummy_gy);
        
        Eigen::Vector2d next_point = current_point + 
            step_size / 6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4);
        
        double movement = (next_point - current_point).norm();
        if (movement < convergence_threshold) {
            break;
        }
        
        current_point = next_point;
        path_points.push_back(current_point);
    }
    
    Eigen::MatrixXd result(2, path_points.size());
    for (size_t i = 0; i < path_points.size(); ++i) {
        result(0, i) = path_points[i](0);
        result(1, i) = path_points[i](1);
    }
    
    return result;
}

// ============================================================================
// 3D GRADIENT FIELD TRACING
// ============================================================================

Eigen::MatrixXd trace_gradient_field_3d(
    const Eigen::Tensor<double, 3>& gradient_x,
    const Eigen::Tensor<double, 3>& gradient_y,
    const Eigen::Tensor<double, 3>& gradient_z,
    const Eigen::Vector3d& start_point,
    double step_size,
    int max_iterations,
    double convergence_threshold
) {
    std::vector<Eigen::Vector3d> path_points;
    path_points.reserve(max_iterations);
    
    Eigen::Vector3d current_point = start_point;
    path_points.push_back(current_point);
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        if (!is_in_bounds_3d(gradient_x, current_point(0), current_point(1), current_point(2), 2.0)) {
            break;
        }
        
        // RK4 integration in 3D
        Eigen::Vector3d k1 = interpolate_gradient_3d(gradient_x, gradient_y, gradient_z,
                                                      current_point(0), current_point(1), current_point(2));
        
        Eigen::Vector3d mid1 = current_point + step_size * k1 / 2.0;
        Eigen::Vector3d k2 = interpolate_gradient_3d(gradient_x, gradient_y, gradient_z,
                                                      mid1(0), mid1(1), mid1(2));
        
        Eigen::Vector3d mid2 = current_point + step_size * k2 / 2.0;
        Eigen::Vector3d k3 = interpolate_gradient_3d(gradient_x, gradient_y, gradient_z,
                                                      mid2(0), mid2(1), mid2(2));
        
        Eigen::Vector3d end = current_point + step_size * k3;
        Eigen::Vector3d k4 = interpolate_gradient_3d(gradient_x, gradient_y, gradient_z,
                                                      end(0), end(1), end(2));
        
        Eigen::Vector3d next_point = current_point + 
            step_size / 6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4);
        
        double movement = (next_point - current_point).norm();
        if (movement < convergence_threshold) {
            break;
        }
        
        current_point = next_point;
        path_points.push_back(current_point);
    }
    
    Eigen::MatrixXd result(3, path_points.size());
    for (size_t i = 0; i < path_points.size(); ++i) {
        result(0, i) = path_points[i](0);
        result(1, i) = path_points[i](1);
        result(2, i) = path_points[i](2);
    }
    
    return result;
}

// ============================================================================
// INTERPOLATION HELPERS
// ============================================================================

Eigen::Vector2d interpolate_gradient(
    const Eigen::MatrixXd& gradient_x,
    const Eigen::MatrixXd& gradient_y,
    double x, double y
) {
    // Use bilinear interpolation from tensor_helpers
    double gx = tensor::interp2_linear(gradient_x, x, y);
    double gy = tensor::interp2_linear(gradient_y, x, y);
    
    return Eigen::Vector2d(gx, gy);
}

Eigen::Vector3d interpolate_gradient_3d(
    const Eigen::Tensor<double, 3>& gradient_x,
    const Eigen::Tensor<double, 3>& gradient_y,
    const Eigen::Tensor<double, 3>& gradient_z,
    double x, double y, double z
) {
    // Use trilinear interpolation from tensor_helpers
    double gx = tensor::interp3_linear(gradient_x, x, y, z);
    double gy = tensor::interp3_linear(gradient_y, x, y, z);
    double gz = tensor::interp3_linear(gradient_z, x, y, z);
    
    return Eigen::Vector3d(gx, gy, gz);
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

bool is_in_bounds(const Eigen::MatrixXd& field, double x, double y, double margin) {
    return x >= margin && x < field.rows() - margin &&
           y >= margin && y < field.cols() - margin;
}

bool is_in_bounds_3d(const Eigen::Tensor<double, 3>& field, 
                     double x, double y, double z, double margin) {
    return x >= margin && x < field.dimension(0) - margin &&
           y >= margin && y < field.dimension(1) - margin &&
           z >= margin && z < field.dimension(2) - margin;
}

double compute_path_length(const Eigen::MatrixXd& path) {
    if (path.cols() < 2) {
        return 0.0;
    }
    
    double total_length = 0.0;
    
    for (int i = 1; i < path.cols(); ++i) {
        Eigen::VectorXd diff = path.col(i) - path.col(i-1);
        total_length += diff.norm();
    }
    
    return total_length;
}

Eigen::MatrixXd resample_path(const Eigen::MatrixXd& path, int num_points) {
    if (path.cols() < 2 || num_points < 2) {
        return path;
    }
    
    // Compute cumulative arc length
    std::vector<double> arc_lengths;
    arc_lengths.reserve(path.cols());
    arc_lengths.push_back(0.0);
    
    double total_length = 0.0;
    for (int i = 1; i < path.cols(); ++i) {
        Eigen::VectorXd diff = path.col(i) - path.col(i-1);
        total_length += diff.norm();
        arc_lengths.push_back(total_length);
    }
    
    if (total_length == 0.0) {
        return path;
    }
    
    // Resample at uniform intervals
    Eigen::MatrixXd resampled(path.rows(), num_points);
    
    for (int i = 0; i < num_points; ++i) {
        double target_length = (total_length * i) / (num_points - 1);
        
        // Find segment containing this arc length
        int seg = 0;
        for (size_t j = 1; j < arc_lengths.size(); ++j) {
            if (arc_lengths[j] >= target_length) {
                seg = j - 1;
                break;
            }
        }
        
        // Linear interpolation within segment
        double t = 0.0;
        if (arc_lengths[seg+1] > arc_lengths[seg]) {
            t = (target_length - arc_lengths[seg]) / 
                (arc_lengths[seg+1] - arc_lengths[seg]);
        }
        
        resampled.col(i) = (1.0 - t) * path.col(seg) + t * path.col(seg+1);
    }
    
    return resampled;
}

}  // namespace rk4
}  // namespace matlab


