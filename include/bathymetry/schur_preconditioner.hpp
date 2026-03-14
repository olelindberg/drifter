#pragma once

/// @file schur_preconditioner.hpp
/// @brief Abstract interface for Schur complement preconditioners
///
/// All preconditioners approximate M_S^{-1} where the Schur complement
/// S = C * A^{-1} * C^T arises from eliminating primal variables in the
/// KKT system. The preconditioner must be SPD for use with CG.

#include "core/types.hpp"

namespace drifter {

/// @brief Abstract interface for Schur complement preconditioners
///
/// Preconditioners for the Schur complement system S * lambda = rhs
/// where S = C * A^{-1} * C^T. The apply() method computes z = M_S^{-1} * r.
///
/// For preconditioned CG, we monitor the preconditioned residual norm:
///   ||r||_{M_S^{-1}} = sqrt(r^T * z)
/// which is monotonically decreasing (unlike ||r||_2).
///
/// Usage in PCG:
/// @code
/// VecX r = rhs - S * lambda;
/// VecX z = precond.apply(r);
/// Real rz = r.dot(z);  // ||r||^2_{M^{-1}}
/// @endcode
class ISchurPreconditioner {
public:
    virtual ~ISchurPreconditioner() = default;

    /// @brief Apply preconditioner: z = M_S^{-1} * r
    /// @param r Input residual vector (size = num_constraints)
    /// @return Preconditioned vector z (size = num_constraints)
    virtual VecX apply(const VecX& r) const = 0;

    /// @brief Check if preconditioner is variable (non-stationary)
    ///
    /// Variable preconditioners (e.g., multigrid V-cycle) may produce
    /// different results for the same input. These require Flexible CG
    /// (FCG) instead of standard CG.
    ///
    /// @return true if preconditioner is variable
    virtual bool is_variable() const { return false; }

    /// @brief Number of constraints (dimension of Schur complement)
    /// @return Size of lambda vector
    virtual Index num_constraints() const = 0;
};

} // namespace drifter
