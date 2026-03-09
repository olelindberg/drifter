#include "bathymetry/iterative_solver.hpp"
#include <cmath>
#include <stdexcept>

namespace drifter {

IterativeSolver::IterativeSolver(std::unique_ptr<IIterativeMethod> method,
                                 const IterativeMethodConfig &config)
    : method_(std::move(method)), config_(config) {
    if (!method_) {
        throw std::invalid_argument(
            "IterativeSolver: method cannot be nullptr");
    }
}

IterativeSolveResult IterativeSolver::solve(VecX &x, const VecX &b) {
    IterativeSolveResult result;
    result.residual_history.reserve(config_.max_iterations);

    const SpMat &Q = method_->matrix();

    // Compute initial residual
    VecX r = b - Q * x;
    Real r_norm = r.norm();
    Real b_norm = b.norm();
    Real r0_norm = r_norm;

    // Handle zero RHS
    if (b_norm < 1e-30) {
        b_norm = 1.0;
    }

    result.residual_history.push_back(r_norm);

    // Check if already converged
    if (r_norm / b_norm < config_.tolerance) {
        result.iterations = 0;
        result.final_residual = r_norm;
        result.relative_residual = r_norm / b_norm;
        result.converged = true;
        return result;
    }

    // Iterative solve
    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        // Apply one iteration
        method_->apply(x, b, 1);

        // Compute residual
        r = b - Q * x;
        r_norm = r.norm();

        result.residual_history.push_back(r_norm);

        // Check convergence
        Real rel_residual = r_norm / b_norm;
        if (rel_residual < config_.tolerance) {
            result.iterations = iter + 1;
            result.final_residual = r_norm;
            result.relative_residual = rel_residual;
            result.converged = true;
            return result;
        }
    }

    // Did not converge within max iterations
    result.iterations = config_.max_iterations;
    result.final_residual = r_norm;
    result.relative_residual = r_norm / b_norm;
    result.converged = false;
    return result;
}

} // namespace drifter
