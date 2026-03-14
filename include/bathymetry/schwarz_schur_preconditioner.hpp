#pragma once

/// @file schwarz_schur_preconditioner.hpp
/// @brief Schwarz-based Schur complement preconditioner
///
/// Uses Schwarz smoothers directly (without multigrid hierarchy) to approximate
/// Q^{-1} in the Schur complement. This serves as a diagnostic tool to verify
/// that the smoothers work correctly as Schur preconditioners.

#include "bathymetry/iterative_method.hpp"
#include "bathymetry/schur_preconditioner.hpp"
#include "core/types.hpp"
#include <memory>

namespace drifter {

/// @brief Schwarz-based Schur complement preconditioner
///
/// Approximates M_S^{-1} * r ≈ C * M_Q^{-1} * C^T * r
/// where M_Q is a Schwarz smoother applied for multiple iterations.
///
/// This preconditioner uses Schwarz methods (additive, multiplicative, or colored)
/// directly to approximate Q^{-1}, without the multigrid hierarchy. It serves as
/// a diagnostic tool to isolate whether convergence issues are due to:
/// 1. The smoothers themselves (Schwarz not approximating Q^{-1} well)
/// 2. The multigrid hierarchy (coarse grid correction, transfer operators)
///
/// If SchwarzSchur converges well but MultigridSchur doesn't, the issue is in
/// the multigrid hierarchy. If both fail, the issue is in the Q^{-1} approximation.
///
/// Usage:
/// @code
/// // Create Schwarz smoother (requires element blocks)
/// auto smoother = std::make_unique<ColoredSchwarzMethod>(Q, element_dofs, block_lu, colors);
///
/// // Create Schur preconditioner
/// SchwarzSchurPreconditioner precond(std::move(smoother), Q, C, 10);
///
/// // Use in FCG
/// FlexibleCG fcg(schur_matvec, precond, tol, max_iter);
/// @endcode
class SchwarzSchurPreconditioner : public ISchurPreconditioner {
public:
    /// @brief Construct Schwarz-based Schur preconditioner
    /// @param smoother Schwarz smoother (takes ownership)
    /// @param Q System matrix (n_free x n_free, SPD)
    /// @param C Constraint matrix (n_c x n_free)
    /// @param num_iterations Number of Schwarz iterations per apply (default: 10)
    SchwarzSchurPreconditioner(std::unique_ptr<IIterativeMethod> smoother,
                                const SpMat &Q, const SpMat &C,
                                int num_iterations = 10);

    /// @brief Apply preconditioner: z = M_S^{-1} * r ≈ C * M_Q^{-1} * C^T * r
    /// @param r Input residual vector (size = num_constraints)
    /// @return Preconditioned vector z (size = num_constraints)
    VecX apply(const VecX &r) const override;

    /// @brief Check if preconditioner is variable
    /// @return false (Schwarz smoothers are stationary)
    bool is_variable() const override { return false; }

    /// @brief Number of constraints
    Index num_constraints() const override { return C_.rows(); }

    /// @brief Get the number of Schwarz iterations used
    int num_iterations() const { return num_iterations_; }

private:
    std::unique_ptr<IIterativeMethod> smoother_;
    SpMat Q_;
    SpMat C_;
    SpMat Ct_; ///< Cached transpose of C
    int num_iterations_;
};

} // namespace drifter
