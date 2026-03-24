#pragma once

/// @file gauss_seidel_schur_preconditioner.hpp
/// @brief Symmetric Gauss-Seidel Schur complement preconditioner

#include "bathymetry/schur_preconditioner.hpp"
#include "core/types.hpp"
#include <functional>

namespace drifter {

/// @brief Symmetric Gauss-Seidel Schur complement preconditioner
///
/// Assembles the Schur complement matrix S explicitly via n_c matvecs,
/// then applies symmetric Gauss-Seidel (forward + backward sweep) to
/// approximate S^{-1} * r.
///
/// Setup cost: O(n_c) Schur matvecs + O(n_c^2) storage
/// Apply cost: O(nnz(S) * num_iterations) per application
///
/// This is stronger than diagonal preconditioning but simpler than
/// physics-based (no factorization required). Good for small to medium
/// constraint counts.
class GaussSeidelSchurPreconditioner : public ISchurPreconditioner {
public:
    /// @brief Setup preconditioner by assembling S explicitly
    /// @param schur_matvec Function computing S * v = C * A^{-1} * C^T * v
    /// @param num_constraints Number of constraints (rows in C)
    /// @param num_iterations Number of symmetric GS sweeps per apply (default: 5)
    explicit GaussSeidelSchurPreconditioner(
        std::function<VecX(const VecX&)> schur_matvec,
        Index num_constraints,
        int num_iterations = 5);

    /// @brief Apply preconditioner: z ≈ S^{-1} * r via symmetric GS
    VecX apply(const VecX& r) const override;

    /// @brief Number of constraints
    Index num_constraints() const override { return num_constraints_; }

    /// @brief Get number of GS iterations
    int num_iterations() const { return num_iterations_; }

    /// @brief Get assembled Schur complement matrix for inspection (testing)
    const SpMat& schur_matrix() const { return S_; }

private:
    Index num_constraints_;
    int num_iterations_;
    SpMat S_;      ///< Assembled Schur complement matrix
    VecX diag_;    ///< Diagonal of S for GS sweeps
};

} // namespace drifter
