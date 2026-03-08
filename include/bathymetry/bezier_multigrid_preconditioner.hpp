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
#include <map>
#include <memory>
#include <tuple>
#include <vector>

namespace drifter {

/// @brief Smoother type for multigrid
enum class SmootherType {
  Jacobi,
  MultiplicativeSchwarz,
  AdditiveSchwarz,
  ColoredMultiplicativeSchwarz
};

/// @brief Strategy for building coarse-level system matrices
enum class CoarseGridStrategy {
  /// Q_c = R * Q_f * P (algebraic Galerkin projection)
  Galerkin,
  /// Assemble from cached element matrices (exact, includes data fitting)
  /// Requires element_matrix_cache to be set via set_element_matrix_cache()
  CachedRediscretization
};

/// @brief Strategy for building transfer operators (P and R)
enum class TransferOperatorStrategy {
  /// R = L2 projection (M_c^{-1} * P^T * M_f), P = R^T
  /// Symmetric multigrid but can have large negative weights
  L2Projection,
  /// P = pure Bezier subdivision, R = P^T normalized
  /// All non-negative weights (de Casteljau convex combinations)
  BezierSubdivision
};

/// @brief Detailed timing profile for multigrid preconditioner (all times in
/// ms)
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
  double vcycle_pre_smooth_ms = 0.0;   ///< Pre-smoothing total time
  double vcycle_residual_ms = 0.0;     ///< Residual computation (r = b - Q*x)
  double vcycle_restrict_ms = 0.0;     ///< Restriction (r_c = R*r) total time
  double vcycle_prolong_ms = 0.0;      ///< Prolongation (e = P*e_c) total time
  double vcycle_post_smooth_ms = 0.0;  ///< Post-smoothing total time
  double vcycle_coarse_solve_ms = 0.0; ///< Coarsest level direct solve time

  // Schwarz smoother breakdown (accumulated over all smooth_schwarz calls)
  double schwarz_matvec_ms = 0.0;      ///< Full mat-vec Qx = Q*x
  double schwarz_gather_ms = 0.0;      ///< Gather local residual r_local
  double schwarz_local_solve_ms = 0.0; ///< Local LU solve dx = Q_block^{-1}*r
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
  int num_levels = 100;

  /// Minimum tree level for coarsest multigrid level (0 = 1x1, 1 = 2x2, 2 = 4x4
  /// elements) Coarsening stops when composite grid nodes reach this tree
  /// level.
  int min_tree_level = 2;

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

  /// Strategy for building coarse-level system matrices
  /// Default: Galerkin (preserves original behavior)
  CoarseGridStrategy coarse_grid_strategy = CoarseGridStrategy::Galerkin;

  /// Strategy for building transfer operators (P and R)
  /// Default: L2Projection (preserves original behavior)
  TransferOperatorStrategy transfer_strategy = TransferOperatorStrategy::L2Projection;
};

/// @brief Active node in a composite multigrid level
///
/// Represents a node that contributes DOFs at a specific MG level.
/// Can be either a leaf node (passes through unchanged at this level)
/// or an internal node (has children at finer MG level that get subdivided).
struct CompositeGridNode {
  /// Pointer to the underlying quadtree node
  const QuadtreeNode *tree_node = nullptr;

  /// True if this node's 4 children exist at finer MG level (subdivision)
  /// False if this node passes through unchanged (identity prolongation)
  bool is_subdivided = false;

  /// Global DOF indices at this MG level (16 DOFs for cubic Bezier)
  std::array<Index, 16> dof_indices;

  CompositeGridNode() { dof_indices.fill(-1); }
};

/// @brief Composite grid level for adaptive multigrid
///
/// Represents all active nodes at one MG level. For adaptive meshes,
/// this includes both internal nodes (parents of finer elements) and
/// leaf nodes that pass through unchanged.
struct CompositeGridLevel {
  /// Active nodes at this level
  std::vector<CompositeGridNode> nodes;

  /// Total unique DOFs at this level
  Index num_dofs = 0;

  /// Map from tree node pointer to index in nodes vector
  std::map<const QuadtreeNode *, Index> node_index_map;

  /// Maximum tree depth among nodes at this level
  int max_depth = 0;
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

  // =========================================================================
  // Composite grid structure for adaptive meshes
  // =========================================================================

  /// Indices of nodes that are subdivided (have 4 children at finer level)
  std::vector<Index> subdivision_node_indices;

  /// Indices of nodes that pass through unchanged (coarse leaves)
  std::vector<Index> passthrough_node_indices;

  /// Composite grid structure for this level
  CompositeGridLevel composite_grid;
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

  /// @brief Get local L2 restriction matrix (16x64) for testing
  /// @details Maps 4 child elements (each 16 DOFs) to 1 parent (16 DOFs)
  ///          in reference space [0,1]x[0,1]
  const MatX &R_L2_local() const { return R_L2_local_; }

  /// @brief Get local L2 prolongation matrix (64x16) for testing
  /// @details Maps 1 parent (16 DOFs) to 4 child elements (each 16 DOFs)
  ///          in reference space [0,1]x[0,1]. P = R^T
  const MatX &P_L2_local() const { return P_L2_local_; }

  /// @brief Set profile for detailed timing instrumentation
  /// @param profile Pointer to profile struct (null to disable profiling)
  void set_profile(MultigridProfile *profile) { profile_ = profile; }

  /// @brief Get current profile pointer
  MultigridProfile *profile() const { return profile_; }

  /// @brief Set element matrix cache for CachedRediscretization strategy
  /// @param cache Pointer to map owned by adaptive smoother (persists across refinement)
  /// @note Only used when config.coarse_grid_strategy == CachedRediscretization
  void set_element_matrix_cache(
      const std::map<std::tuple<uint64_t, int, int>, MatX>* cache) {
    element_matrix_cache_ = cache;
  }

private:
  MultigridConfig config_;
  std::vector<MultigridLevel> levels_;
  MultigridProfile *profile_ =
      nullptr; ///< Optional profiling (null = disabled)

  /// External element matrix cache for CachedRediscretization strategy
  /// Key: (morton, level_x, level_y) -> element matrix Q_elem
  const std::map<std::tuple<uint64_t, int, int>, MatX>* element_matrix_cache_ =
      nullptr;

  /// Cached references for composite hierarchy building (valid only during
  /// setup)
  const QuadtreeAdapter *mesh_ = nullptr;
  const CGCubicBezierDofManager *dof_manager_ = nullptr;

  /// Cached Bezier subdivision matrices (4x4 for cubic)
  MatX S_left_;  // [0, 0.5] subdivision
  MatX S_right_; // [0.5, 1] subdivision

  // =========================================================================
  // L2 projection (full weighting) transfer operators
  // =========================================================================

  /// Cached 1D Bernstein mass matrix (4x4 for cubic)
  MatX M_1D_;

  /// Cached 2D mass matrix (16x16, tensor product)
  MatX M_2D_;

  /// Cached inverse of 2D mass matrix
  MatX M_2D_inv_;

  /// Cached local L2 restriction matrix (16x64, coarse × 4 children)
  MatX R_L2_local_;

  /// Cached local L2 prolongation matrix (64x16, 4 children × coarse)
  MatX P_L2_local_;

  /// @brief Build 1D cubic Bernstein mass matrix
  /// @return 4x4 mass matrix M[i,j] = ∫₀¹ Bᵢ³(t) · Bⱼ³(t) dt
  static MatX build_bernstein_mass_1d();

  /// @brief Build local L2 operators (restriction and prolongation)
  /// @details Computes R_L2_local_ = M_c⁻¹ · P_bezier^T · M_f
  ///          and P_L2_local_ = R_L2_local_^T
  void build_l2_operators_local();

  /// @brief Build global restriction operator using L2 projection
  /// @param mg_level The coarse MG level index
  /// @return Restriction matrix R: fine -> coarse
  SpMat build_restriction_l2(int mg_level);

  /// @brief Assemble coarse level matrix from cached element matrices
  /// @param mg_level The coarse MG level index
  /// @return Assembled system matrix for this level
  /// @note Requires element_matrix_cache_ to be set
  SpMat assemble_from_cached_matrices(int mg_level);

  /// @brief Build prolongation from tree level L to level L+1
  /// @param tree_level Tree level (0 = root/coarsest)
  /// @param mesh Quadtree mesh
  /// @param leaf_dof_manager DOF manager for leaf level
  /// @param coarse_num_dofs Output: number of DOFs at tree_level
  /// @param expected_fine_dofs Expected number of DOFs at tree_level+1 (0 to
  /// auto-compute)
  /// @return Prolongation matrix P: level -> level+1 (coarse_dofs x fine_dofs
  /// rows)
  SpMat
  build_prolongation_for_level(int tree_level, const QuadtreeAdapter &mesh,
                               const CGCubicBezierDofManager &leaf_dof_manager,
                               Index &coarse_num_dofs,
                               Index expected_fine_dofs = 0);

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

  /// @brief Build element coloring for colored Schwarz smoother using graph
  /// coloring
  /// @param level Level index
  /// @details Uses greedy graph coloring based on DOF adjacency. Elements that
  ///          share DOFs get different colors. Works for both uniform and
  ///          adaptive meshes. Typically produces 4-8 colors.
  void build_element_coloring(int level);

  /// @brief Compute Kronecker product of two matrices
  static MatX kronecker_product(const MatX &A, const MatX &B);

  // =========================================================================
  // Composite grid methods for adaptive meshes
  // =========================================================================

  /// @brief Build composite grid hierarchy from all leaves
  /// @param mesh Quadtree mesh
  /// @param dof_manager DOF manager for leaf level
  /// @details Initializes finest level with all leaves, then iteratively
  ///          builds coarser levels by coarsening complete sibling groups.
  void build_composite_hierarchy(const QuadtreeAdapter &mesh,
                                 const CGCubicBezierDofManager &dof_manager);

  /// @brief Build one coarse composite level from finer level
  /// @param mg_level The coarse MG level index to build
  /// @details Groups fine nodes by parent, coarsens complete sibling groups
  ///          of 4, and passes through incomplete groups unchanged.
  void build_composite_level_from_finer(int mg_level);

  /// @brief Build prolongation using composite grid structure
  /// @param mg_level The coarse MG level index
  /// @return Prolongation matrix P: coarse -> fine
  /// @details Builds P with subdivision entries for coarsened nodes and
  ///          identity entries for pass-through nodes.
  SpMat build_prolongation_composite(int mg_level);

  /// @brief Assign DOFs to node using position-based deduplication
  /// @param node The composite grid node to assign DOFs to
  /// @param bounds Element bounds for DOF positioning
  /// @param position_to_dof Map from quantized position to DOF index
  /// @param num_dofs Current DOF count (incremented for new DOFs)
  void assign_dofs_from_bounds(
      CompositeGridNode &node, const QuadBounds &bounds,
      std::map<std::pair<int64_t, int64_t>, Index> &position_to_dof,
      Index &num_dofs);

  /// @brief Find minimum leaf depth in mesh
  /// @param mesh Quadtree mesh
  /// @return Minimum tree depth among all leaves
  static int compute_min_leaf_depth(const QuadtreeAdapter &mesh);

  /// @brief Get child index (0-3) based on position within parent
  /// @param child Child node
  /// @param parent Parent node
  /// @return Child index: 0=(0,0), 1=(1,0), 2=(0,1), 3=(1,1)
  static int get_child_quadrant(const QuadtreeNode *child,
                                const QuadtreeNode *parent);

  /// @brief Recursively add prolongation entries from coarse node to all
  /// descendant leaves
  /// @param tree_node Current tree node (start with coarse node's child)
  /// @param coarse_node The coarse composite grid node
  /// @param fine_grid The fine composite grid level
  /// @param P_accumulated Accumulated subdivision matrix from coarse to current
  /// node
  /// @param P_local Subdivision matrices for each quadrant
  /// @param triplets Output triplet list for prolongation matrix
  /// @param processed_fine_dofs Set of fine DOFs already processed (to avoid
  /// duplicates)
  /// @param parent Parent of tree_node (for quadrant determination)
  void add_recursive_prolongation(
      const QuadtreeNode *tree_node, const CompositeGridNode &coarse_node,
      const CompositeGridLevel &fine_grid, const MatX &P_accumulated,
      const std::array<MatX, 4> &P_local,
      std::vector<Eigen::Triplet<Real>> &triplets,
      std::set<Index> &processed_fine_dofs,
      const QuadtreeNode *parent) const;
};

} // namespace drifter
