#pragma once

/// @file cg_cubic_bezier_bathymetry_smoother.hpp
/// @brief CG cubic Bezier bathymetry smoother with C¹ continuity constraints
///
/// Uses Continuous Galerkin assembly where DOFs at element boundaries are
/// shared. Cubic Bezier (degree 3, 4×4 = 16 DOFs) with C¹ constraints.

#include "bathymetry/cg_bezier_smoother_base.hpp"
#include "bathymetry/cg_cubic_bezier_dof_manager.hpp"
#include "bathymetry/cubic_bezier_basis_2d.hpp"
#include "bathymetry/cubic_thin_plate_hessian.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "core/types.hpp"
#include "mesh/seabed_surface.hpp"
#include <functional>
#include <memory>
#include <optional>

namespace drifter {

// Forward declarations
class OctreeAdapter;
class BathymetrySource;
struct BathymetryPoint;
struct CGCubicIterationProfile;

/// @brief Timing profile for solve phase (all times in milliseconds)
struct CGCubicSolveProfile {
  double matrix_build_ms = 0.0; ///< Q matrix construction
  double constraint_build_ms =
      0.0; ///< C¹ edge constraint assembly (in DOF manager)
  double kkt_assembly_ms = 0.0;          ///< KKT system build
  double sparse_lu_compute_ms = 0.0;     ///< SparseLU factorization
  double sparse_lu_solve_ms = 0.0;       ///< SparseLU back-substitution
  double constraint_projection_ms = 0.0; ///< Constraint projection solve

  // Iterative solver timings
  double inner_cg_setup_ms = 0.0; ///< ICC preconditioner setup
  int outer_cg_iterations = 0;    ///< Number of outer CG iterations
  double outer_cg_total_ms = 0.0; ///< Total outer CG time
  int inner_cg_total_calls = 0;   ///< Total inner solve calls
};

/// @brief Configuration for CG cubic Bezier bathymetry smoother
struct CGCubicBezierSmootherConfig {
  /// Data fitting weight relative to smoothness
  Real lambda = 0.01;

  /// Gauss points per direction for data sampling
  int ngauss_data = 4;

  /// Gauss points for energy integration
  int ngauss_energy = 4;

  /// Ridge regularization parameter
  Real ridge_epsilon = 1e-4;

  /// Optional elevation bounds
  std::optional<Real> lower_bound;
  std::optional<Real> upper_bound;

  /// Maximum iterations for bound constraint solver
  int max_bound_iterations = 50;

  /// Tolerance for bound constraint satisfaction
  Real bound_tolerance = 1e-10;

  /// Number of Gauss points per edge for C¹ edge constraints
  int edge_ngauss = 4;

  /// Use constraint condensation for hanging nodes (smaller KKT system)
  /// If false, uses original full KKT system with all constraints
  bool use_condensation = true;

  /// Use iterative Schur complement solver (default: false = SparseLU)
  bool use_iterative_solver = false;

  /// Tolerance for outer Schur complement CG
  Real schur_cg_tolerance = 1e-6;

  /// Max iterations for outer Schur complement CG
  int schur_cg_max_iterations = 1000;

  /// Tolerance for inner ICC-CG solves
  Real inner_cg_tolerance = 1e-10;

  /// Max iterations for inner ICC-CG solves
  int inner_cg_max_iterations = 500;

  /// Initial diagonal shift for ICC preconditioner (robustness for small
  /// lambda)
  Real icc_shift = 1e-3;
};

/// @brief CG cubic Bezier bathymetry smoother with C¹ continuity
///
/// Fits cubic Bezier surfaces (16 DOFs per element) to bathymetry data
/// using Continuous Galerkin assembly with optional C¹ constraints.
class CGCubicBezierBathymetrySmoother : public CGBezierSmootherBase {
public:
  /// @brief Construct smoother for a quadtree mesh
  explicit CGCubicBezierBathymetrySmoother(
      const QuadtreeAdapter &mesh,
      const CGCubicBezierSmootherConfig &config = {});

  /// @brief Construct smoother from octree (uses bottom face)
  explicit CGCubicBezierBathymetrySmoother(
      const OctreeAdapter &octree,
      const CGCubicBezierSmootherConfig &config = {});

  // =========================================================================
  // Data input - inherited from base: set_bathymetry_data, set_scattered_points
  // =========================================================================

  // =========================================================================
  // Configuration
  // =========================================================================

  void set_smoothing_weight(Real lambda) { config_.lambda = lambda; }

  void set_bounds(Real lower, Real upper) {
    config_.lower_bound = lower;
    config_.upper_bound = upper;
  }

  void clear_bounds() {
    config_.lower_bound = std::nullopt;
    config_.upper_bound = std::nullopt;
  }

  const CGCubicBezierSmootherConfig &config() const { return config_; }

  // =========================================================================
  // Solve
  // =========================================================================

  void solve();

  /// @brief Set profile for timing solve phase
  /// @param profile Pointer to profile struct (null to disable profiling)
  void set_solve_profile(CGCubicSolveProfile *profile) {
    solve_profile_ = profile;
  }

  /// @brief Set profile for timing assembly operations (hessian, data fitting)
  /// @param profile Pointer to iteration profile struct (null to disable)
  void set_profile(CGCubicIterationProfile *profile) { profile_ = profile; }

  // =========================================================================
  // Solution evaluation - inherited from base: evaluate, evaluate_gradient,
  // solution
  // =========================================================================

  // =========================================================================
  // Transfer and output
  // =========================================================================

  // transfer_to_seabed inherited from base
  void write_vtk(const std::string &filename, int resolution = 8) const;
  void write_control_points_vtk(const std::string &filename) const;

  // =========================================================================
  // Diagnostics
  // =========================================================================

  // data_residual, regularization_energy, objective_value inherited from base
  Real constraint_violation() const;

  // num_global_dofs, num_free_dofs, num_constraints, mesh inherited from base
  const CGCubicBezierDofManager &dof_manager() const { return *dof_manager_; }

  // element_coefficients() implemented in base class

  /// @brief Get reference to the Bezier basis
  /// @return Reference to the CubicBezierBasis2D
  const BezierBasis2DBase &get_basis() const { return *basis_; }

protected:
  // =========================================================================
  // CGBezierSmootherBase virtual method implementations
  // =========================================================================

  void
  set_bathymetry_data_impl(std::function<Real(Real, Real)> bathy_func) override;
  Index dof_manager_num_global_dofs() const override {
    return dof_manager_->num_global_dofs();
  }
  Index dof_manager_num_free_dofs() const override {
    return dof_manager_->num_free_dofs();
  }
  Index dof_manager_num_constraints() const override {
    return dof_manager_->num_constraints();
  }
  const std::vector<Index> &element_global_dofs(Index elem) const override {
    return dof_manager_->element_dofs(elem);
  }
  const BezierBasis2DBase &basis() const override { return *basis_; }
  int ngauss_data() const override { return config_.ngauss_data; }
  Real lambda() const override { return config_.lambda; }
  Real ridge_epsilon() const override { return config_.ridge_epsilon; }

private:
  CGCubicBezierSmootherConfig config_;

  std::unique_ptr<CubicBezierBasis2D> basis_;
  std::unique_ptr<CubicThinPlateHessian> thin_plate_hessian_;
  std::unique_ptr<CGCubicBezierDofManager> dof_manager_;

  CGCubicSolveProfile *solve_profile_ = nullptr;
  CGCubicIterationProfile *profile_ = nullptr;

  void init_components();
  void solve_with_constraints();
  void solve_with_constraints_direct();    // SparseLU-based direct solver
  void solve_with_constraints_iterative(); // Schur complement CG with ICC
  void
  solve_with_constraints_full_kkt(); // Original implementation for comparison

  // =========================================================================
  // Shared helpers for constrained solve
  // =========================================================================

  /// @brief Condensed system after hanging node elimination
  struct CondensedSystem {
    SpMat Q_reduced; ///< Condensed stiffness matrix (num_free × num_free)
    VecX b_reduced;  ///< Condensed RHS vector (num_free)
    SpMat A_edge;    ///< Edge constraints on free DOFs (num_edge × num_free)
    Index num_dofs;  ///< Total global DOFs
    Index num_free;  ///< Free DOFs after hanging node elimination
    Index num_edge;  ///< Number of edge derivative constraints
  };

  /// @brief Build condensed system by eliminating hanging node constraints
  CondensedSystem build_condensed_system();

  /// @brief Recover full solution from free DOF solution
  /// @param x_free Solution on free DOFs
  /// @param sys Condensed system (for dimensions)
  void recover_solution_from_free(const VecX &x_free,
                                  const CondensedSystem &sys);

  // =========================================================================
  // Constraint matrix assembly
  // =========================================================================

  /// @brief Assemble hanging node constraints on global DOFs
  /// @return Sparse matrix A of size (num_hanging_constraints ×
  /// num_global_dofs)
  SpMat assemble_A_hanging() const;

  /// @brief Assemble edge derivative constraints on global DOFs
  /// @return Sparse matrix A of size (num_edge_constraints × num_global_dofs)
  SpMat assemble_A_edge() const;

  /// @brief Assemble all constraints (hanging + edge) on global DOFs
  /// @return Sparse matrix A of size (num_total_constraints × num_global_dofs)
  SpMat assemble_A() const;

  /// @brief Assemble edge constraints on free DOFs (after hanging node
  /// condensation)
  /// @param expand_dof Function mapping global DOF to (free_index, weight)
  /// pairs
  /// @return Sparse matrix A of size (num_edge_constraints × num_free_dofs)
  SpMat assemble_A_edge_free(
      const std::function<std::vector<std::pair<Index, Real>>(Index)>
          &expand_dof) const;
};

} // namespace drifter
