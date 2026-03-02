#pragma once

/// @file bezier_multigrid_preconditioner.hpp
/// @brief Geometric multigrid preconditioner for CG Bezier bathymetry smoother
///
/// Implements a V-cycle multigrid preconditioner using the natural quadtree
/// hierarchy. The thin-plate Hessian H is elliptic (4th order biharmonic-like),
/// making it ideal for multigrid preconditioning.
///
/// Key features:
/// - Geometric coarsening via Morton code parent grouping
/// - Bezier subdivision matrices for prolongation
/// - Galerkin projection for coarse-level operators
/// - Weighted Jacobi smoothing

#include "bathymetry/cg_cubic_bezier_dof_manager.hpp"
#include "bathymetry/cubic_bezier_basis_2d.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "core/types.hpp"
#include <Eigen/SparseLU>
#include <memory>
#include <vector>

namespace drifter {

/// @brief Smoother type for multigrid
enum class SmootherType {
  Jacobi,
  MultiplicativeSchwarz,
  AdditiveSchwarz,
  ColoredMultiplicativeSchwarz
};

/// @brief Detailed timing profile for multigrid preconditioner (all times in ms)
struct MultigridProfile {
  // =========================================================================
  // Setup phase breakdown
  // =========================================================================
  double setup_total_ms = 0.0;          ///< Total setup() time
  double setup_finest_init_ms = 0.0;    ///< Finest level init (diag_inv)
  double setup_element_blocks_ms = 0.0; ///< Element block LU factorizations
  double setup_prolongation_ms = 0.0;   ///< build_prolongation() for all levels
  double setup_galerkin_ms = 0.0;       ///< Galerkin projection (R*Q*P)
  double setup_coarse_lu_ms = 0.0;      ///< Coarsest level LU factorization

  // =========================================================================
  // Apply phase breakdown (accumulated over all apply() calls)
  // =========================================================================
  double apply_total_ms = 0.0; ///< Total time in apply() calls

  // V-cycle components (accumulated)
  double vcycle_pre_smooth_ms = 0.0;  ///< Pre-smoothing total time
  double vcycle_residual_ms = 0.0;    ///< Residual computation (r = b - Q*x)
  double vcycle_restrict_ms = 0.0;    ///< Restriction (r_c = R*r) total time
  double vcycle_prolong_ms = 0.0;     ///< Prolongation (e = P*e_c) total time
  double vcycle_post_smooth_ms = 0.0; ///< Post-smoothing total time
  double vcycle_coarse_solve_ms = 0.0; ///< Coarsest level direct solve time

  // Schwarz smoother breakdown (accumulated over all smooth_schwarz calls)
  double schwarz_matvec_ms = 0.0;        ///< Full mat-vec Qx = Q*x
  double schwarz_gather_ms = 0.0;        ///< Gather local residual r_local
  double schwarz_local_solve_ms = 0.0;   ///< Local LU solve dx = Q_block^{-1}*r
  double schwarz_scatter_update_ms = 0.0; ///< Scatter + incremental Qx update

  // Jacobi smoother timing (accumulated)
  double jacobi_total_ms = 0.0; ///< Total Jacobi smoother time

  // =========================================================================
  // Operation counters
  // =========================================================================
  int apply_calls = 0;            ///< Number of apply() invocations
  int vcycle_calls = 0;           ///< Total V-cycle calls (including recursive)
  int schwarz_iterations = 0;     ///< Total Schwarz smoother iterations
  int jacobi_iterations = 0;      ///< Total Jacobi smoother iterations
  int schwarz_element_solves = 0; ///< Total element block solves
  int matvec_products = 0;        ///< Total sparse mat-vec products
  int coarse_solves = 0;          ///< Coarsest level direct solves

  // =========================================================================
  // Helper methods
  // =========================================================================
  double schwarz_total_ms() const {
    return schwarz_matvec_ms + schwarz_gather_ms + schwarz_local_solve_ms +
           schwarz_scatter_update_ms;
  }

  double vcycle_smoothing_ms() const {
    return vcycle_pre_smooth_ms + vcycle_post_smooth_ms;
  }
};

/// @brief Configuration for geometric multigrid preconditioner
struct MultigridConfig {
  /// Number of V-cycle levels (including finest)
  int num_levels = 3;

  /// Pre-smoothing iterations
  int pre_smoothing = 1;

  /// Post-smoothing iterations
  int post_smoothing = 1;

  /// Damping parameter for weighted Jacobi smoother
  Real jacobi_omega = 0.8;

  /// Smoother type (Jacobi or MultiplicativeSchwarz)
  SmootherType smoother_type = SmootherType::MultiplicativeSchwarz;

  /// Maximum DOFs on coarsest level before using direct solver
  Index coarsest_direct_size = 200;

  /// Enable verbose logging during setup
  bool verbose = false;
};

/// @brief Data for a single multigrid level
struct MultigridLevel {
  /// System matrix at this level
  SpMat Q;

  /// Restriction operator from finer level (rows = this level DOFs)
  SpMat R;

  /// Prolongation operator to finer level (cols = this level DOFs)
  SpMat P;

  /// Inverse diagonal of Q for Jacobi smoother
  VecX diag_inv;

  /// Number of DOFs at this level
  Index num_dofs = 0;

  /// Direct solver for coarsest level
  std::unique_ptr<Eigen::SparseLU<SpMat>> solver;

  /// Element block LU factorizations for multiplicative Schwarz smoother
  std::vector<Eigen::PartialPivLU<MatX>> element_block_lu;

  /// Free DOF indices for each element block
  std::vector<std::vector<Index>> element_free_dofs;

  /// Element indices grouped by color (graph coloring, typically 4-8 colors)
  /// elements_by_color[c] contains indices of elements with color c
  std::vector<std::vector<Index>> elements_by_color;

  /// Number of colors used in coloring
  int num_colors = 0;
};

/// @brief Geometric multigrid preconditioner for CG Bezier bathymetry smoother
///
/// Builds a multilevel hierarchy by grouping fine elements by their parent
/// Morton codes. Uses Bezier subdivision matrices for transfer operators
/// and Galerkin projection (Q_c = R*Q*P) for coarse-level matrices.
///
/// Usage:
/// @code
///   BezierMultigridPreconditioner mg(config);
///   mg.setup(Q_fine, mesh, dof_manager);
///   // In CG iteration:
///   VecX z = mg.apply(r);  // z = M^{-1} * r
/// @endcode
class BezierMultigridPreconditioner {
public:
  /// @brief Construct with configuration
  explicit BezierMultigridPreconditioner(const MultigridConfig &config = {});

  /// @brief Setup multigrid hierarchy
  /// @param Q_fine Fine-level system matrix (condensed, on free DOFs)
  /// @param mesh Quadtree mesh
  /// @param dof_manager DOF manager for fine level
  void setup(const SpMat &Q_fine, const QuadtreeAdapter &mesh,
             const CGCubicBezierDofManager &dof_manager);

  /// @brief Apply V-cycle preconditioner: z = M^{-1} * r
  /// @param r Residual vector
  /// @return Preconditioned vector
  VecX apply(const VecX &r) const;

  /// @brief Number of levels in hierarchy
  int num_levels() const { return static_cast<int>(levels_.size()); }

  /// @brief Check if setup has been called
  bool is_setup() const { return !levels_.empty(); }

  /// @brief Get configuration
  const MultigridConfig &config() const { return config_; }

  /// @brief Get level data (for debugging/testing)
  const MultigridLevel &level(int l) const { return levels_[l]; }

  /// @brief Set profile for detailed timing instrumentation
  /// @param profile Pointer to profile struct (null to disable profiling)
  void set_profile(MultigridProfile *profile) { profile_ = profile; }

  /// @brief Get current profile pointer
  MultigridProfile *profile() const { return profile_; }

private:
  MultigridConfig config_;
  std::vector<MultigridLevel> levels_;
  MultigridProfile *profile_ = nullptr; ///< Optional profiling (null = disabled)

  /// Cached Bezier subdivision matrices (4x4 for cubic)
  MatX S_left_;  // [0, 0.5] subdivision
  MatX S_right_; // [0.5, 1] subdivision

  /// @brief Build coarse level from fine level
  /// @param fine_level Index of fine level
  /// @param mesh Quadtree mesh (for Morton codes)
  /// @param fine_dof_manager DOF manager for fine level
  void build_coarse_level(int fine_level, const QuadtreeAdapter &mesh,
                          const CGCubicBezierDofManager &fine_dof_manager);

  /// @brief Build prolongation operator for a single coarsening step
  /// @param mesh Quadtree mesh
  /// @param fine_dof_manager DOF manager for fine level
  /// @param coarse_num_dofs Output: number of coarse DOFs
  /// @return Global prolongation matrix (fine_dofs x coarse_dofs)
  SpMat build_prolongation(const QuadtreeAdapter &mesh,
                           const CGCubicBezierDofManager &fine_dof_manager,
                           Index &coarse_num_dofs);

  /// @brief V-cycle recursion
  /// @param level Current level (0 = finest)
  /// @param x Solution vector (modified in place)
  /// @param b Right-hand side
  void v_cycle(int level, VecX &x, const VecX &b) const;

  /// @brief Weighted Jacobi smoothing
  /// @param level Level index
  /// @param x Solution vector (modified in place)
  /// @param b Right-hand side
  /// @param iters Number of smoothing iterations
  void smooth_jacobi(int level, VecX &x, const VecX &b, int iters) const;

  /// @brief Multiplicative Schwarz (block Gauss-Seidel) smoothing
  /// @param level Level index
  /// @param x Solution vector (modified in place)
  /// @param b Right-hand side
  /// @param iters Number of smoothing iterations
  void smooth_schwarz(int level, VecX &x, const VecX &b, int iters) const;

  /// @brief Additive Schwarz (parallel Jacobi-style) smoothing
  /// @param level Level index
  /// @param x Solution vector (modified in place)
  /// @param b Right-hand side
  /// @param iters Number of smoothing iterations
  /// @details Accumulates all element corrections before applying, eliminating
  ///          the expensive incremental Qx updates of multiplicative Schwarz.
  void smooth_schwarz_additive(int level, VecX &x, const VecX &b,
                               int iters) const;

  /// @brief Colored multiplicative Schwarz smoothing
  /// @param level Level index
  /// @param x Solution vector (modified in place)
  /// @param b Right-hand side
  /// @param iters Number of smoothing iterations
  /// @details Uses 4-color checkerboard pattern. Same-colored elements don't
  ///          share DOFs, enabling batched updates within each color while
  ///          maintaining Gauss-Seidel information exchange between colors.
  void smooth_schwarz_colored(int level, VecX &x, const VecX &b,
                              int iters) const;

  /// @brief Build element block LU factorizations for Schwarz smoother
  /// @param level Level index
  /// @param dof_manager DOF manager for DOF-to-element mapping
  void build_element_blocks(int level,
                            const CGCubicBezierDofManager &dof_manager);

  /// @brief Build element coloring for colored Schwarz smoother using graph coloring
  /// @param level Level index
  /// @details Uses greedy graph coloring based on DOF adjacency. Elements that
  ///          share DOFs get different colors. Works for both uniform and
  ///          adaptive meshes. Typically produces 4-8 colors.
  void build_element_coloring(int level);

  /// @brief Compute Kronecker product of two matrices
  static MatX kronecker_product(const MatX &A, const MatX &B);
};

} // namespace drifter
