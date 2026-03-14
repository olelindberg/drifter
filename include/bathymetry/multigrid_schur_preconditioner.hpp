#pragma once

/// @file multigrid_schur_preconditioner.hpp
/// @brief Multigrid V-cycle Schur complement preconditioner

#include "bathymetry/bezier_multigrid_preconditioner.hpp"
#include "bathymetry/schur_preconditioner.hpp"
#include "core/types.hpp"

namespace drifter {

/// @brief Multigrid V-cycle Schur complement preconditioner
///
/// Applies M_S^{-1} * v ≈ C * Q^{-1} * C^T * v
/// where Q^{-1} is approximated by iterative refinement with multigrid
/// V-cycles:
///   x_0 = 0
///   for i = 1..num_vcycles:
///     x_i = x_{i-1} + mg.apply(b - Q * x_{i-1})
///
/// This preconditioner uses a fixed number of V-cycles for iterative
/// refinement. While technically non-stationary, the variation is small
/// enough to use standard PCG instead of Flexible CG.
///
/// Setup cost: MG hierarchy construction (done externally)
/// Apply cost: num_vcycles V-cycles + sparse matvecs
class MultigridSchurPreconditioner : public ISchurPreconditioner {
public:
  /// @brief Setup MG V-cycle Schur preconditioner
  /// @param mg_precond Reference to multigrid preconditioner for Q (must
  /// outlive this object)
  /// @param Q System matrix on free DOFs (n_free x n_free, SPD)
  /// @param C Constraint matrix on free DOFs (n_c x n_free)
  /// @param num_vcycles Number of V-cycles for iterative refinement (default:
  /// 5)
  MultigridSchurPreconditioner(const BezierMultigridPreconditioner &mg_precond,
                               const SpMat &Q, const SpMat &C,
                               int num_vcycles = 5);

  /// @brief Apply preconditioner: z ≈ C * Q^{-1} * C^T * r
  VecX apply(const VecX &r) const override;

  /// @brief Treated as stationary for standard PCG
  bool is_variable() const override { return true; }

  /// @brief Number of constraints
  Index num_constraints() const override { return C_.rows(); }

private:
  const BezierMultigridPreconditioner &mg_precond_;
  SpMat Q_;         ///< System matrix (n_free x n_free)
  SpMat C_;         ///< Constraint matrix (n_c x n_free)
  SpMat Ct_;        ///< C^T cached for efficiency (n_free x n_c)
  int num_vcycles_; ///< Number of V-cycles for iterative refinement
};

} // namespace drifter
