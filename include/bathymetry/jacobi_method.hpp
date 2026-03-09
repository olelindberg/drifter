#pragma once

/// @file jacobi_method.hpp
/// @brief Weighted Jacobi iterative method

#include "bathymetry/iterative_method.hpp"
#include "core/types.hpp"

namespace drifter {

/// @brief Weighted Jacobi iterative method
///
/// Implements the damped Jacobi iteration:
///   x_{k+1} = x_k + omega * D^{-1} * (b - Q*x_k)
///
/// where D is the diagonal of Q and omega is the damping parameter.
class JacobiMethod : public IIterativeMethod {
public:
    /// @brief Construct Jacobi method
    /// @param Q System matrix (reference, must outlive this object)
    /// @param omega Damping parameter (typically 0.6-0.8)
    explicit JacobiMethod(const SpMat &Q, Real omega = 0.8);

    /// @brief Apply Jacobi iterations
    /// @param x Solution vector (modified in-place)
    /// @param b Right-hand side
    /// @param iters Number of iterations
    void apply(VecX &x, const VecX &b, int iters) const override;

    /// @brief Get the system matrix
    const SpMat &matrix() const override { return Q_; }

    /// @brief Get the damping parameter
    Real omega() const { return omega_; }

private:
    const SpMat &Q_;
    VecX diag_inv_;
    Real omega_;
};

} // namespace drifter
