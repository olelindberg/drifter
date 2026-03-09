#pragma once

/// @file iterative_method.hpp
/// @brief Abstract interface for iterative methods (smoothers/solvers)

#include "core/types.hpp"
#include <vector>

namespace drifter {

/// @brief Configuration for iterative method solve
struct IterativeMethodConfig {
    int max_iterations = 1000;
    Real tolerance = 1e-6;
    Real omega = 0.8; ///< Damping parameter
};

/// @brief Abstract interface for iterative methods (smoothers/solvers)
///
/// This interface provides a common abstraction for iterative methods
/// that can be used as standalone solvers or as smoothers in multigrid.
class IIterativeMethod {
public:
    virtual ~IIterativeMethod() = default;

    /// @brief Apply iterations of the method
    /// @param x Solution vector (modified in-place)
    /// @param b Right-hand side
    /// @param iters Number of iterations to apply
    virtual void apply(VecX &x, const VecX &b, int iters) const = 0;

    /// @brief Get the system matrix for residual computation
    virtual const SpMat &matrix() const = 0;
};

} // namespace drifter
