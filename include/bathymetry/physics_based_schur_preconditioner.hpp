#pragma once

/// @file physics_based_schur_preconditioner.hpp
/// @brief Physics-based Schur complement preconditioner using smoothness Hessian

#include "bathymetry/schur_preconditioner.hpp"
#include "core/types.hpp"
#include <Eigen/SparseCholesky>

namespace drifter {

/// @brief Physics-based Schur complement preconditioner
///
/// Assembles M_S = C * K^{-1} * C^T where K is the smoothness Hessian
/// (thin plate energy without data fitting term). This captures the
/// essential physics of constraint coupling.
///
/// Assembly process (during construction):
///   For each column i = 0..n_c-1:
///     1. Solve K * v = C^T * e_i
///     2. M_S[:, i] = C * v
///
/// The resulting M_S is SPD (since K is SPD and C has full row rank)
/// and is factored with SimplicialLLT for efficient application.
///
/// Setup cost: O(n_c) K-solves + Cholesky factorization of M_S
/// Apply cost: O(n_c^2) triangular solves (or sparse if M_S is sparse)
///
/// This preconditioner gives mesh-independent convergence when alpha
/// (the smoothness weight) dominates the data fitting term.
class PhysicsBasedSchurPreconditioner : public ISchurPreconditioner {
public:
    /// @brief Setup physics-based preconditioner
    /// @param K_reduced Smoothness Hessian on free DOFs (SPD, n_free x n_free)
    /// @param C Constraint matrix on free DOFs (n_c x n_free)
    /// @param regularization Relative diagonal regularization to handle rank-deficient M_S
    ///                       (when C has redundant constraints). Default: 1e-6
    /// @throws std::runtime_error if K factorization fails
    /// @throws std::runtime_error if M_S Cholesky factorization fails
    PhysicsBasedSchurPreconditioner(const SpMat& K_reduced, const SpMat& C,
                                    Real regularization = 1e-6);

    /// @brief Apply preconditioner: z = M_S^{-1} * r
    VecX apply(const VecX& r) const override;

    /// @brief Number of constraints
    Index num_constraints() const override { return num_constraints_; }

    /// @brief Get assembled M_S matrix for inspection (testing)
    const SpMat& assembled_matrix() const { return M_S_; }

    /// @brief Check if setup succeeded (always true if constructor didn't throw)
    bool is_valid() const { return is_valid_; }

private:
    Index num_constraints_;
    SpMat M_S_;                                ///< Assembled preconditioner matrix
    Eigen::SimplicialLLT<SpMat> cholesky_;     ///< Cholesky factorization of M_S
    bool is_valid_ = false;
};

} // namespace drifter
