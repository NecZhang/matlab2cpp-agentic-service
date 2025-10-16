#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>  // ✅ FIXED: Add Tensor support BEFORE using it
#include <functional>
#include <vector>

namespace matlab {
namespace rk4 {

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

/**
 * Function type for ODE right-hand side: dy/dt = f(t, y)
 * 
 * @param t Current time
 * @param y Current state vector
 * @return Derivative dy/dt
 */
using ODEFunction = std::function<Eigen::VectorXd(double t, const Eigen::VectorXd& y)>;

/**
 * Function type for gradient field evaluation: [dx/dt, dy/dt] = f(x, y, GradientField)
 * 
 * @param x Current x position
 * @param y Current y position
 * @param gradient_x X-component of gradient field
 * @param gradient_y Y-component of gradient field
 * @return Velocity vector [dx/dt, dy/dt]
 */
using GradientFieldFunction = std::function<Eigen::Vector2d(
    double x, double y,
    const Eigen::MatrixXd& gradient_x,
    const Eigen::MatrixXd& gradient_y
)>;

// ============================================================================
// RUNGE-KUTTA 4TH ORDER INTEGRATION
// ============================================================================

/**
 * Single step of RK4 integration
 * 
 * @param f ODE function dy/dt = f(t, y)
 * @param t Current time
 * @param y Current state
 * @param h Step size
 * @return Next state y(t + h)
 * 
 * MATLAB equivalent: [t, y] = ode45(@f, [t0 tf], y0)
 * 
 * RK4 formula:
 *   k1 = f(t, y)
 *   k2 = f(t + h/2, y + h*k1/2)
 *   k3 = f(t + h/2, y + h*k2/2)
 *   k4 = f(t + h, y + h*k3)
 *   y_next = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
 */
Eigen::VectorXd rk4_step(const ODEFunction& f, double t, const Eigen::VectorXd& y, double h);

/**
 * RK4 integration over a time interval
 * 
 * @param f ODE function dy/dt = f(t, y)
 * @param t_span Time interval [t0, tf]
 * @param y0 Initial state
 * @param num_steps Number of integration steps
 * @return Pair of (time_points, solution_states)
 * 
 * MATLAB equivalent: [t, y] = ode45(@f, [t0 tf], y0)
 */
std::pair<std::vector<double>, std::vector<Eigen::VectorXd>>
rk4_integrate(const ODEFunction& f, const Eigen::Vector2d& t_span, 
              const Eigen::VectorXd& y0, int num_steps);

// ============================================================================
// GRADIENT FIELD TRACING (for shortest path algorithms)
// ============================================================================

/**
 * Trace a path through a gradient field using RK4
 * 
 * @param gradient_x X-component of gradient field (rows x cols matrix)
 * @param gradient_y Y-component of gradient field (rows x cols matrix)
 * @param start_point Starting position [x, y]
 * @param step_size Integration step size
 * @param max_iterations Maximum number of steps
 * @param convergence_threshold Stop when movement < threshold
 * @return Matrix of path points (2 x num_points): [x_coords; y_coords]
 * 
 * MATLAB equivalent (from shortestpath.m):
 *   path = trace_gradient_field(Gx, Gy, start_point, stepsize, max_iter)
 * 
 * This is used for tracing shortest paths in distance fields computed by
 * Fast Marching Method (FMM/MSFM).
 */
Eigen::MatrixXd trace_gradient_field(
    const Eigen::MatrixXd& gradient_x,
    const Eigen::MatrixXd& gradient_y,
    const Eigen::Vector2d& start_point,
    double step_size,
    int max_iterations = 10000,
    double convergence_threshold = 0.01
);

/**
 * Trace a path with custom velocity function
 * 
 * @param velocity_func Function that computes velocity at (x, y)
 * @param start_point Starting position [x, y]
 * @param step_size Integration step size
 * @param max_iterations Maximum number of steps
 * @param convergence_threshold Stop when movement < threshold
 * @return Matrix of path points (2 x num_points)
 */
Eigen::MatrixXd trace_path_rk4(
    const GradientFieldFunction& velocity_func,
    const Eigen::Vector2d& start_point,
    double step_size,
    int max_iterations = 10000,
    double convergence_threshold = 0.01
);

// ============================================================================
// 3D GRADIENT FIELD TRACING
// ============================================================================

/**
 * Trace a path through a 3D gradient field using RK4
 * 
 * @param gradient_x X-component of gradient field (3D tensor)
 * @param gradient_y Y-component of gradient field (3D tensor)
 * @param gradient_z Z-component of gradient field (3D tensor)
 * @param start_point Starting position [x, y, z]
 * @param step_size Integration step size
 * @param max_iterations Maximum number of steps
 * @param convergence_threshold Stop when movement < threshold
 * @return Matrix of path points (3 x num_points): [x_coords; y_coords; z_coords]
 * 
 * MATLAB equivalent:
 *   path = trace_3d_gradient(Gx, Gy, Gz, start_point, stepsize)
 */
Eigen::MatrixXd trace_gradient_field_3d(
    const Eigen::Tensor<double, 3>& gradient_x,
    const Eigen::Tensor<double, 3>& gradient_y,
    const Eigen::Tensor<double, 3>& gradient_z,
    const Eigen::Vector3d& start_point,
    double step_size,
    int max_iterations = 10000,
    double convergence_threshold = 0.01
);

// ============================================================================
// INTERPOLATION HELPERS (for gradient evaluation)
// ============================================================================

/**
 * Evaluate gradient field at a point using bilinear interpolation
 * 
 * @param gradient_x X-component of gradient field
 * @param gradient_y Y-component of gradient field
 * @param x X-coordinate (can be fractional)
 * @param y Y-coordinate (can be fractional)
 * @return Gradient vector [gx, gy] at (x, y)
 * 
 * This is used internally by trace_gradient_field to evaluate gradients
 * at non-integer positions during RK4 integration.
 */
Eigen::Vector2d interpolate_gradient(
    const Eigen::MatrixXd& gradient_x,
    const Eigen::MatrixXd& gradient_y,
    double x, double y
);

/**
 * Evaluate 3D gradient field at a point using trilinear interpolation
 * 
 * @param gradient_x X-component of gradient field
 * @param gradient_y Y-component of gradient field
 * @param gradient_z Z-component of gradient field
 * @param x X-coordinate (can be fractional)
 * @param y Y-coordinate (can be fractional)
 * @param z Z-coordinate (can be fractional)
 * @return Gradient vector [gx, gy, gz] at (x, y, z)
 */
Eigen::Vector3d interpolate_gradient_3d(
    const Eigen::Tensor<double, 3>& gradient_x,
    const Eigen::Tensor<double, 3>& gradient_y,
    const Eigen::Tensor<double, 3>& gradient_z,
    double x, double y, double z
);

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Check if a point is within field bounds
 * 
 * @param field Gradient field matrix
 * @param x X-coordinate
 * @param y Y-coordinate
 * @param margin Safety margin from boundaries
 * @return true if point is within bounds
 */
bool is_in_bounds(const Eigen::MatrixXd& field, double x, double y, double margin = 1.0);

/**
 * Check if a 3D point is within field bounds
 */
bool is_in_bounds_3d(const Eigen::Tensor<double, 3>& field, 
                     double x, double y, double z, double margin = 1.0);

/**
 * Compute path length from a sequence of points
 * 
 * @param path Matrix of path points (2 x num_points or 3 x num_points)
 * @return Total Euclidean length of the path
 * 
 * MATLAB equivalent: pathlen = sum(sqrt(sum(diff(path, 1, 2).^2, 1)))
 */
double compute_path_length(const Eigen::MatrixXd& path);

/**
 * Resample a path to have uniform spacing
 * 
 * @param path Input path (2 x num_points)
 * @param num_points Desired number of output points
 * @return Resampled path with uniform spacing
 */
Eigen::MatrixXd resample_path(const Eigen::MatrixXd& path, int num_points);

}  // namespace rk4
}  // namespace matlab


