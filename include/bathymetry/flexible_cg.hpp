#pragma once

/// @file flexible_cg.hpp
/// @brief Flexible Conjugate Gradient for variable preconditioners
///
/// FCG (Notay 2000) handles non-stationary preconditioners by using
/// a modified beta computation that doesn't assume M is fixed.

#include "bathymetry/schur_preconditioner.hpp"
#include "core/types.hpp"
#include <functional>
#include <vector>

namespace drifter {

/// @brief Result of FCG solve
struct FCGResult {
    int iterations = 0;            ///< Number of iterations performed
    Real final_residual = 0.0;     ///< Final ||r||_{M^{-1}}
    Real initial_residual = 0.0;   ///< Initial ||r_0||_{M^{-1}}
    Real relative_residual = 0.0;  ///< final / initial
    bool converged = false;        ///< True if tolerance was met
    std::vector<Real> residual_history; ///< Per-iteration ||r||_{M^{-1}}
};

/// @brief Flexible Conjugate Gradient solver
///
/// Solves S * x = b where S is SPD and accessed only via matvec.
/// Supports variable (non-stationary) preconditioners.
///
/// Standard CG computes beta = r_{k+1}^T r_{k+1} / r_k^T r_k which assumes
/// the preconditioner M is the same at each iteration. FCG uses:
///
///   beta_k = (z_{k+1}, r_{k+1}) / (z_k, r_k)
///
/// where z = M^{-1} r. This remains valid even when M varies.
///
/// Algorithm (Notay 2000):
/// @code
/// r_0 = b - S*x_0
/// z_0 = M^{-1} * r_0
/// p_0 = z_0
/// for k = 0, 1, ...
///     q_k = S * p_k
///     alpha_k = (z_k, r_k) / (p_k, q_k)
///     x_{k+1} = x_k + alpha_k * p_k
///     r_{k+1} = r_k - alpha_k * q_k
///     check convergence: ||r_{k+1}||_{M^{-1}} / ||r_0||_{M^{-1}} < tol
///     z_{k+1} = M^{-1} * r_{k+1}
///     beta_k = (z_{k+1}, r_{k+1}) / (z_k, r_k)
///     p_{k+1} = z_{k+1} + beta_k * p_k
/// @endcode
///
/// Note: FCG is mathematically equivalent to standard PCG when M is fixed,
/// but slightly more expensive due to storing z_k.
class FlexibleCG {
public:
    /// @brief Construct FCG solver
    /// @param matvec Function computing S * v
    /// @param precond Schur preconditioner (may be variable)
    /// @param tol Relative tolerance for ||r||_{M^{-1}} / ||r_0||_{M^{-1}}
    /// @param max_iter Maximum iterations
    FlexibleCG(std::function<VecX(const VecX&)> matvec,
               const ISchurPreconditioner& precond,
               Real tol = 1e-6,
               int max_iter = 1000);

    /// @brief Solve S * x = b
    /// @param x Initial guess (modified in place with solution)
    /// @param b Right-hand side
    /// @return Solve result with convergence info
    FCGResult solve(VecX& x, const VecX& b);

    /// @brief Enable verbose output to stdout
    void set_verbose(bool v) { verbose_ = v; }

private:
    std::function<VecX(const VecX&)> matvec_;
    const ISchurPreconditioner& precond_;
    Real tol_;
    int max_iter_;
    bool verbose_ = false;
};

} // namespace drifter
