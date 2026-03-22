#pragma once

/// @file diagonal_approx_cg_schur_preconditioner.hpp
/// @brief Diagonal-approximation Schur preconditioner with inner CG solve

#include "bathymetry/schur_preconditioner.hpp"
#include "core/types.hpp"

namespace drifter {

/// @brief Diagonal-approximation Schur complement preconditioner
///
/// Assembles M_S = C * diag(Q)^{-1} * C^T where diag(Q) is the diagonal
/// of the system matrix Q. This approximates the true Schur complement
/// S = C * Q^{-1} * C^T but is much cheaper to assemble (sparse matmul
/// instead of n_c Schur matvecs).
///
/// Assembly process (during construction):
///   1. Extract diag(Q) from system matrix
///   2. Build sparse diagonal D = diag(1/diag(Q))
///   3. M_S = C * D * C^T via sparse matrix multiplication
///
/// Application: Solve M_S * z = r using inner diagonal-preconditioned CG.
///
/// Setup cost: O(nnz(C)) sparse matmul
/// Apply cost: O(inner_iterations * nnz(M_S))
///
/// This is a variable preconditioner (inner CG is iterative) and requires
/// Flexible CG (FCG) for the outer solve.
class DiagonalApproxCGSchurPreconditioner : public ISchurPreconditioner {
public:
    /// @brief Setup diagonal-approximation preconditioner
    /// @param Q System matrix on free DOFs (n_free x n_free)
    /// @param C Constraint matrix on free DOFs (n_c x n_free)
    /// @param inner_tolerance CG tolerance for M_S^{-1} solve (default: 1e-6)
    /// @param inner_max_iterations Max inner CG iterations (default: 100)
    DiagonalApproxCGSchurPreconditioner(const SpMat& Q, const SpMat& C,
                                        Real inner_tolerance = 1e-6,
                                        int inner_max_iterations = 100);

    /// @brief Apply preconditioner: z = M_S^{-1} * r via inner CG
    VecX apply(const VecX& r) const override;

    /// @brief This is a variable preconditioner (inner CG is iterative)
    bool is_variable() const override { return true; }

    /// @brief Number of constraints
    Index num_constraints() const override { return n_c_; }

    /// @brief Get assembled M_S matrix for inspection (testing)
    const SpMat& assembled_matrix() const { return M_S_; }

private:
    SpMat M_S_;            ///< Assembled C * diag(Q)^{-1} * C^T
    VecX diag_M_S_inv_;    ///< 1/diag(M_S) for inner CG preconditioning
    Index n_c_;            ///< Number of constraints
    Real inner_tol_;       ///< Inner CG tolerance
    int inner_max_iter_;   ///< Max inner CG iterations
};

} // namespace drifter
