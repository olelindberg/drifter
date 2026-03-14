#pragma once

/// @file diagonal_schur_preconditioner.hpp
/// @brief Diagonal Schur complement preconditioner

#include "bathymetry/schur_preconditioner.hpp"
#include "core/types.hpp"
#include <functional>

namespace drifter {

/// @brief Diagonal Schur complement preconditioner
///
/// Computes M_S = diag(S) via n_c matvecs during setup:
///   d_i = e_i^T * S * e_i
/// where S = C * A^{-1} * C^T is never formed explicitly.
///
/// This is a cheap but weak preconditioner. Use as fallback when
/// physics-based preconditioner is too expensive or for comparison.
///
/// Setup cost: O(n_c) Schur matvecs
/// Apply cost: O(n_c) element-wise division
class DiagonalSchurPreconditioner : public ISchurPreconditioner {
public:
    /// @brief Setup diagonal preconditioner by extracting diag(S)
    /// @param schur_matvec Function computing S * v = C * A^{-1} * C^T * v
    /// @param num_constraints Number of constraints (rows in C)
    explicit DiagonalSchurPreconditioner(
        std::function<VecX(const VecX&)> schur_matvec,
        Index num_constraints);

    /// @brief Apply preconditioner: z = diag(S)^{-1} * r
    VecX apply(const VecX& r) const override;

    /// @brief Number of constraints
    Index num_constraints() const override { return diag_inv_.size(); }

    /// @brief Get inverse diagonal for inspection (testing)
    const VecX& diagonal_inverse() const { return diag_inv_; }

    /// @brief Get diagonal for inspection (testing)
    VecX diagonal() const;

private:
    VecX diag_inv_; ///< 1/diag(S), with safeguard for near-zero entries
};

} // namespace drifter
