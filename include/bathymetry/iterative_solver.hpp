#pragma once

/// @file iterative_solver.hpp
/// @brief Standalone iterative solver using an iterative method

#include "bathymetry/iterative_method.hpp"
#include "core/types.hpp"
#include <memory>
#include <vector>

namespace drifter {

/// @brief Result of an iterative solve
struct IterativeSolveResult {
    int iterations = 0;            ///< Number of iterations performed
    Real final_residual = 0.0;     ///< Final residual norm
    Real relative_residual = 0.0;  ///< Final relative residual
    bool converged = false;        ///< Whether tolerance was reached
    std::vector<Real> residual_history; ///< Residual norm per iteration
};

/// @brief Standalone iterative solver using an iterative method
///
/// Wraps an IIterativeMethod and provides full solve capability with
/// convergence monitoring and stopping criteria.
class IterativeSolver {
public:
    /// @brief Construct solver with method and configuration
    /// @param method Iterative method to use (ownership transferred)
    /// @param config Solver configuration
    explicit IterativeSolver(std::unique_ptr<IIterativeMethod> method,
                             const IterativeMethodConfig &config = {});

    /// @brief Solve Qx = b iteratively
    /// @param x Initial guess (modified to solution)
    /// @param b Right-hand side
    /// @return Solve result with convergence info
    IterativeSolveResult solve(VecX &x, const VecX &b);

    /// @brief Get the underlying method
    const IIterativeMethod &method() const { return *method_; }

    /// @brief Get the configuration
    const IterativeMethodConfig &config() const { return config_; }

private:
    std::unique_ptr<IIterativeMethod> method_;
    IterativeMethodConfig config_;
};

} // namespace drifter
