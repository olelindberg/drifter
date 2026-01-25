#pragma once

/// @file bezier_multigrid_solver.hpp
/// @brief Geometric multigrid solver for Bezier bathymetry KKT systems
///
/// Implements V-cycle multigrid for KKT saddle-point systems arising from
/// constrained Bezier surface fitting with C² continuity constraints.
///
/// The solver leverages the octree hierarchy to build a sequence of coarse
/// grids and uses restriction/prolongation operators to transfer between levels.

#include "bathymetry/quadtree_adapter.hpp"
#include "core/types.hpp"
#include <vector>
#include <memory>

namespace drifter {

/// @brief Configuration for multigrid solver
struct MultigridConfig {
  int max_iterations = 100;        ///< Maximum V-cycle iterations
  Real tolerance = 1e-6;            ///< Convergence tolerance (relative residual)
  int num_presmooth = 2;            ///< Pre-smoothing iterations per level
  int num_postsmooth = 2;           ///< Post-smoothing iterations per level
  Real smoother_omega = 0.7;        ///< Relaxation parameter for smoother
  int coarsest_level = 0;           ///< Coarsest level (0 = base grid)
  bool use_direct_coarse_solve = true; ///< Use SparseLU on coarsest level
};

/// @brief Geometric multigrid solver for Bezier bathymetry KKT systems
///
/// Solves KKT systems of the form:
///   [Q   A^T] [x]   [f]
///   [A    0 ] [λ] = [g]
///
/// using V-cycle multigrid with:
/// - Jacobi smoother for primal variables
/// - Uzawa iteration for constraint multipliers
/// - L2 projection for restriction/prolongation
/// - Galerkin coarse-grid operators
class BezierMultigridSolver {
public:
  /// @brief Construct multigrid solver from octree hierarchy
  /// @param quadtree Finest level quadtree mesh
  /// @param config Multigrid configuration
  explicit BezierMultigridSolver(const QuadtreeAdapter& quadtree,
                                  const MultigridConfig& config = {});

  /// @brief Solve KKT system using V-cycle multigrid
  /// @param KKT Full KKT matrix [Q A^T; A 0]
  /// @param x Solution vector [primal; dual]
  /// @param rhs Right-hand side [f; g]
  /// @return Number of iterations performed
  int solve(const SpMat& KKT, VecX& x, const VecX& rhs);

  /// @brief Get convergence history (residual norms)
  const std::vector<Real>& residual_history() const { return residual_history_; }

  /// @brief Get final relative residual
  Real final_residual() const {
    return residual_history_.empty() ? 0.0 : residual_history_.back();
  }

  /// @brief Get number of levels in hierarchy
  int num_levels() const {
    // Placeholder: return 1 until grid hierarchy is built in Step 2.2
    return grids_.empty() ? 1 : static_cast<int>(grids_.size());
  }

private:
  // Configuration
  MultigridConfig config_;

  // Finest grid reference (stored for now, hierarchy built in Step 2.2)
  const QuadtreeAdapter* finest_grid_;

  // Grid hierarchy (level 0 = coarsest, level L-1 = finest)
  std::vector<std::unique_ptr<QuadtreeAdapter>> grids_;

  // Operators for each level (restriction_[l]: level l → level l-1)
  std::vector<SpMat> restriction_ops_;    // fine → coarse
  std::vector<SpMat> prolongation_ops_;   // coarse → fine
  std::vector<SpMat> operators_;          // KKT matrices per level

  // Convergence tracking
  std::vector<Real> residual_history_;

  // Internal methods

  /// @brief Extract grid hierarchy from octree
  void build_grid_hierarchy(const QuadtreeAdapter& finest_grid);

  /// @brief Build restriction operator from fine to coarse
  /// @param fine_level Fine grid level index
  /// @return Sparse restriction matrix
  SpMat build_restriction(int fine_level);

  /// @brief Build prolongation operator from coarse to fine
  /// @param coarse_level Coarse grid level index
  /// @return Sparse prolongation matrix
  SpMat build_prolongation(int coarse_level);

  /// @brief Build coarse-grid operator via Galerkin projection
  /// @param fine_operator Fine grid KKT matrix
  /// @param restriction Restriction operator R
  /// @param prolongation Prolongation operator P
  /// @return Coarse KKT operator (R * A_fine * P)
  SpMat build_coarse_operator(const SpMat& fine_operator,
                                const SpMat& restriction,
                                const SpMat& prolongation);

  /// @brief Perform one V-cycle
  /// @param level Current level (0 = coarsest)
  /// @param x Current solution
  /// @param rhs Right-hand side
  void v_cycle(int level, VecX& x, const VecX& rhs);

  /// @brief Smooth solution using Jacobi iteration
  /// @param A System matrix (KKT)
  /// @param x Solution vector
  /// @param b Right-hand side
  /// @param num_iterations Number of smoothing iterations
  void smooth(const SpMat& A, VecX& x, const VecX& b, int num_iterations);

  /// @brief Direct solve on coarsest level
  /// @param A Coarsest KKT matrix
  /// @param x Solution vector
  /// @param b Right-hand side
  void direct_solve(const SpMat& A, VecX& x, const VecX& b);

  /// @brief Compute residual norm
  /// @param A System matrix
  /// @param x Current solution
  /// @param b Right-hand side
  /// @return ||b - A*x||
  Real compute_residual(const SpMat& A, const VecX& x, const VecX& b);
};

// ============================================================================
// Performance Characteristics
// ============================================================================
//
// Complexity Analysis:
// - Single V-cycle: O(n) work per iteration (n = total DOFs)
// - Convergence: Typically 5-15 iterations for KKT systems
// - Total complexity: O(k·n) where k = iterations
//
// Comparison to Direct Solve:
// - SparseLU: O(n^1.5 - n^2.1) complexity
// - Multigrid: O(n) complexity
// - Crossover: ~1,000-5,000 DOFs depending on system
//
// Memory Usage:
// - Grid hierarchy: ~2-3× finest grid storage
// - Operators: One sparse matrix per level
// - Total: Acceptable for modern workstations (GB RAM)
//
// Scalability:
// - Weak scaling: O(n) for increasing mesh size
// - Strong scaling: Limited by sequential SparseLU on coarsest grid
// - Future: Replace coarsest solve with iterative for better scaling
//
// Current Limitations (TODO):
// - Grid hierarchy extraction incomplete (Step 2.2)
// - Restriction/prolongation use placeholder implementations
// - Optimal for uniform meshes when fully implemented
//
// ============================================================================

} // namespace drifter
