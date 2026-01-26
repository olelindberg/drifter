#pragma once

/// @file bezier_hierarchical_solver.hpp
/// @brief Hierarchical local correction solver for Bezier bathymetry fitting
///
/// Implements a hierarchical approach to solving Bezier bathymetry smoothing
/// problems. Instead of solving one large global KKT system, it solves a
/// sequence of smaller problems: a global coarse solution followed by local
/// corrections in refined subdomains.
///
/// The key insight is that at refinement boundaries, the correction vanishes
/// (delta_b = 0) and the Laplacian matches the coarser level (sigma_l = sigma_{l-1}),
/// ensuring C^2 continuity across scales.
///
/// Algorithm:
///   b(x) = b_0(x) + delta_b_1(x) + delta_b_2(x) + ... + delta_b_L(x)
///
///   1. Solve global coarse problem for b_0 (elements with max_level = 0)
///   2. For each refinement level l = 1, ..., L:
///      a. Identify subdomains Omega_l (elements with max_level = l)
///      b. Solve local correction with BCs: delta_b = 0, sigma = sigma_{l-1}
///      c. Update: b_l = b_{l-1} + delta_b_l
///   3. Final bathymetry: b = b_L

#include "bathymetry/bezier_basis_2d.hpp"
#include "bathymetry/bezier_c2_constraints.hpp"
#include "bathymetry/bezier_data_fitting.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "bathymetry/thin_plate_hessian.hpp"
#include "core/types.hpp"
#include <map>
#include <memory>
#include <set>
#include <vector>

namespace drifter {

// Forward declarations
struct BezierSmootherConfig;

/// @brief Configuration for hierarchical solver
struct HierarchicalConfig {
  Real tolerance = 1e-6;           ///< Convergence tolerance for local solves
  bool parallel_subdomains = true; ///< Enable parallel subdomain solves
  bool verbose = false;            ///< Print progress information
};

/// @brief Information about a subdomain at a specific refinement level
struct SubdomainInfo {
  int level;                       ///< Refinement level of this subdomain
  std::vector<Index> elements;     ///< Element indices in this subdomain
  std::vector<Index> boundary_dofs;///< DOFs on subdomain boundary (where delta_b = 0)
  std::vector<Index> interior_dofs;///< DOFs interior to subdomain

  /// Check if a DOF is on the subdomain boundary
  bool is_boundary_dof(Index dof) const {
    return std::binary_search(boundary_dofs.begin(), boundary_dofs.end(), dof);
  }
};

/// @brief Hierarchical local correction solver for Bezier bathymetry
///
/// Solves Bezier bathymetry fitting problems using a hierarchical approach:
/// - Level 0: Solve global coarse problem on coarsest elements
/// - Level l: Solve local corrections in refined regions with boundary conditions
///            that ensure C^2 continuity with coarser levels
///
/// This approach is particularly efficient for AMR meshes where refinement
/// is localized (e.g., near coastlines).
class BezierHierarchicalSolver {
public:
  /// @brief Construct hierarchical solver from quadtree mesh
  /// @param mesh 2D quadtree mesh
  /// @param config Smoother configuration (lambda, etc.)
  /// @param hlc_config Hierarchical solver configuration
  explicit BezierHierarchicalSolver(
      const QuadtreeAdapter &mesh,
      const BezierSmootherConfig &config,
      const HierarchicalConfig &hlc_config = {});

  /// @brief Set the data fitting assembler (borrowed from BezierBathymetrySmoother)
  /// @param assembler Data fitting assembler with bathymetry data
  void set_data_assembler(BezierDataFittingAssembler *assembler) {
    data_assembler_ = assembler;
  }

  /// @brief Solve using hierarchical local correction
  void solve();

  /// @brief Get the solution vector (all control point z-values)
  const VecX &solution() const { return solution_; }

  /// @brief Get convergence history (residual at each level)
  const std::vector<Real> &residual_history() const { return residual_history_; }

  /// @brief Get number of refinement levels (distinct levels present in mesh)
  int num_levels() const { return static_cast<int>(subdomains_.size()); }

  /// @brief Get subdomain info for a level
  const SubdomainInfo &subdomain(int level) const { return subdomains_.at(level); }

  /// @brief Evaluate Laplacian (sigma = z_xx + z_yy) at a point
  /// @param x, y Physical coordinates
  /// @return Laplacian value
  Real evaluate_laplacian(Real x, Real y) const;

private:
  // References
  const QuadtreeAdapter &mesh_;
  const BezierSmootherConfig &config_;
  HierarchicalConfig hlc_config_;

  // Borrowed components
  BezierDataFittingAssembler *data_assembler_ = nullptr;

  // Owned components
  std::unique_ptr<BezierBasis2D> basis_;
  std::unique_ptr<ThinPlateHessian> hessian_;

  // Level hierarchy
  std::map<int, SubdomainInfo> subdomains_; ///< Subdomains by level
  int max_level_ = 0;                       ///< Maximum refinement level

  // Solution state
  VecX solution_;     ///< Current solution b (control point z-values)
  VecX laplacian_;    ///< Current Laplacian sigma = Delta(b) at DOFs

  // Convergence tracking
  std::vector<Real> residual_history_;

  // =========================================================================
  // Hierarchy building
  // =========================================================================

  /// @brief Build level hierarchy from quadtree
  ///
  /// Groups elements by their maximum refinement level and identifies
  /// boundary DOFs for each subdomain.
  void build_level_hierarchy();

  /// @brief Find boundary DOFs for a subdomain
  ///
  /// Boundary DOFs are those on edges where:
  /// - EdgeNeighborInfo.type == FineToCoarse (refinement boundary)
  /// - EdgeNeighborInfo.type == Boundary (domain boundary)
  ///
  /// @param subdomain Subdomain to analyze
  /// @return Sorted vector of global DOF indices on subdomain boundary
  std::vector<Index> find_boundary_dofs(const SubdomainInfo &subdomain) const;

  // =========================================================================
  // Coarse global solve
  // =========================================================================

  /// @brief Solve global coarse problem (level 0)
  ///
  /// Solves the standard Bezier bathymetry problem on coarsest elements:
  ///   minimize: x^T H x + lambda * ||B x - b||^2
  ///   subject to: A_c2 x = 0 (C^2 continuity)
  void solve_coarse_global();

  // =========================================================================
  // Local correction solves
  // =========================================================================

  /// @brief Solve local corrections for a refinement level
  /// @param level Refinement level (1, 2, ..., max_level)
  void solve_level_corrections(int level);

  /// @brief Solve local correction problem for a subdomain
  ///
  /// Solves for delta_b in subdomain with boundary conditions:
  /// - delta_b = 0 at subdomain boundary (Dirichlet)
  /// - sigma = sigma_{l-1} at boundary (Laplacian matching)
  ///
  /// @param subdomain Subdomain information
  /// @return Local correction vector (DOFs in subdomain order)
  VecX solve_local_problem(const SubdomainInfo &subdomain);

  /// @brief Build local thin plate Hessian for subdomain
  /// @param subdomain Subdomain information
  /// @return Sparse Hessian matrix (local DOFs x local DOFs)
  SpMat build_local_hessian(const SubdomainInfo &subdomain) const;

  /// @brief Build local data fitting matrices with residual
  ///
  /// Builds normal equations for fitting residual (d - b_{l-1}) instead of d.
  ///
  /// @param subdomain Subdomain information
  /// @param AtWA Output: B^T W B matrix (local DOFs x local DOFs)
  /// @param AtWb Output: B^T W (d - b_{l-1}) vector
  void build_local_data_fitting_residual(const SubdomainInfo &subdomain,
                                          SpMat &AtWA, VecX &AtWb) const;

  /// @brief Build local C^2 constraint matrix
  ///
  /// Builds constraints only for vertices/edges within the subdomain.
  /// Boundary DOFs are handled separately via Dirichlet BCs.
  ///
  /// @param subdomain Subdomain information
  /// @return Sparse constraint matrix (local constraints x local DOFs)
  SpMat build_local_constraints(const SubdomainInfo &subdomain) const;

  /// @brief Apply boundary conditions to local problem
  ///
  /// Applies:
  /// 1. delta_b = 0 at boundary DOFs (Dirichlet elimination)
  /// 2. sigma = sigma_{l-1} at boundary (optional Laplacian matching)
  ///
  /// @param Q Local Q matrix (modified in place)
  /// @param c Local c vector (modified in place)
  /// @param A_constraints Local constraint matrix (modified in place)
  /// @param b_constraints Constraint RHS (modified in place)
  /// @param boundary_dofs Boundary DOF indices (in local numbering)
  void apply_correction_boundary_conditions(
      SpMat &Q, VecX &c, SpMat &A_constraints, VecX &b_constraints,
      const std::vector<Index> &boundary_dofs) const;

  /// @brief Solve local KKT system
  /// @param Q Objective Hessian
  /// @param c Objective gradient
  /// @param A Constraint matrix
  /// @param b Constraint RHS
  /// @return Solution vector
  VecX solve_local_kkt(const SpMat &Q, const VecX &c,
                        const SpMat &A, const VecX &b) const;

  // =========================================================================
  // Laplacian computation
  // =========================================================================

  /// @brief Compute Laplacian at all DOFs
  ///
  /// Computes sigma = z_xx + z_yy at each control point location
  /// using the current solution.
  void compute_laplacian();

  /// @brief Evaluate Laplacian within an element
  /// @param elem Element index
  /// @param u, v Parameter coordinates in [0,1]^2
  /// @return sigma = z_xx/dx^2 + z_yy/dy^2
  Real evaluate_laplacian_in_element(Index elem, Real u, Real v) const;

  // =========================================================================
  // Utility methods
  // =========================================================================

  /// @brief Get global DOF index
  Index global_dof(Index elem, int local_dof) const {
    return elem * BezierBasis2D::NDOF + local_dof;
  }

  /// @brief Get element and local DOF from global DOF
  void global_to_local_dof(Index global, Index &elem, int &local) const {
    elem = global / BezierBasis2D::NDOF;
    local = static_cast<int>(global % BezierBasis2D::NDOF);
  }

  /// @brief Map global DOF to local DOF within subdomain
  /// @param global_dof Global DOF index
  /// @param subdomain Subdomain info
  /// @return Local DOF index within subdomain, or -1 if not in subdomain
  int global_to_subdomain_dof(Index global_dof,
                               const SubdomainInfo &subdomain) const;

  /// @brief Map subdomain DOF to global DOF
  /// @param subdomain_dof Local DOF index within subdomain
  /// @param subdomain Subdomain info
  /// @return Global DOF index
  Index subdomain_to_global_dof(int subdomain_dof,
                                 const SubdomainInfo &subdomain) const;

  /// @brief Find element containing point
  /// @param x, y Physical coordinates
  /// @return Element index, or -1 if not found
  Index find_element(Real x, Real y) const;

  /// @brief Find coarse element containing point at or below given level
  /// @param x, y Physical coordinates
  /// @param max_level Maximum level to search (inclusive)
  /// @return Element index, or -1 if not found
  Index find_coarse_element(Real x, Real y, int max_level) const;

  /// @brief Initialize fine level DOFs by interpolating from coarse level
  /// @param fine_level The level to initialize
  ///
  /// For each fine element, finds the containing coarse element and evaluates
  /// the coarse Bezier surface at each fine DOF position.
  void interpolate_from_coarse_level(int fine_level);

  /// @brief Compute scale factor for combining H and AtWA
  /// @param H Thin plate Hessian
  /// @param AtWA Data fitting normal equations
  /// @return Scale factor alpha = ||AtWA|| / ||H||
  Real compute_scale_factor(const SpMat &H, const SpMat &AtWA) const;
};

// ============================================================================
// Performance Characteristics
// ============================================================================
//
// Complexity Analysis:
// - Coarse global solve: O(n_coarse^1.5) via SparseLU
// - Per-level local solve: O(n_local^1.5) via SparseLU
// - Total: O(sum_l n_l^1.5) where n_l = DOFs at level l
//
// For AMR with localized refinement:
// - n_coarse ~ n_total / 2^(2*L) where L = max_level
// - n_local ~ n_total / 2^(2*(L-l)) at level l
// - Much better than global O(n_total^1.5) for deep AMR
//
// Comparison to V-cycle Multigrid:
// - Multigrid: O(n) but iterative (5-15 iterations)
// - HLC: Direct solve at each level, one-way pass
// - HLC better when subdomains are small (localized refinement)
// - Multigrid better for uniform or near-uniform meshes
//
// Parallelism:
// - Disconnected subdomains at same level can be solved independently
// - Natural for OpenMP parallel_for over subdomains
//
// ============================================================================

} // namespace drifter
