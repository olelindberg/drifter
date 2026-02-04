#pragma once

/// @file cg_linear_bezier_bathymetry_smoother.hpp
/// @brief CG linear Bezier bathymetry smoother with Laplacian smoothing
///
/// Uses Continuous Galerkin assembly where DOFs at element boundaries are
/// shared. Linear Bezier (degree 1, 2x2 = 4 DOFs) with C0 continuity from
/// shared DOFs. Uses Dirichlet/Laplace energy for smoothing instead of thin
/// plate energy.

#include "bathymetry/cg_linear_bezier_dof_manager.hpp"
#include "bathymetry/dirichlet_hessian.hpp"
#include "bathymetry/linear_bezier_basis_2d.hpp"
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
struct CGLinearIterationProfile;

/// @brief Configuration for CG linear Bezier bathymetry smoother
struct CGLinearBezierSmootherConfig {
  /// Data fitting weight relative to smoothness
  /// lambda = 0: pure Dirichlet energy (smooth surface, ignores data)
  /// lambda = 1: balanced smoothing and data fitting
  /// lambda > 1: closer to data
  Real lambda = 1.0;

  /// Gauss points per direction for data sampling
  int ngauss_data = 2;

  /// Gauss points for energy integration
  int ngauss_energy = 2;

  /// Ridge regularization parameter
  Real ridge_epsilon = 1e-4;

  /// Optional elevation bounds
  std::optional<Real> lower_bound;
  std::optional<Real> upper_bound;

  /// Maximum iterations for bound constraint solver
  int max_bound_iterations = 50;

  /// Tolerance for bound constraint satisfaction
  Real bound_tolerance = 1e-10;
};

/// @brief CG linear Bezier bathymetry smoother with Laplacian energy
///
/// Fits bilinear Bezier surfaces (4 DOFs per element) to bathymetry data
/// using Continuous Galerkin assembly. Uses Dirichlet energy (gradient
/// magnitude) for smoothing, which is the natural choice for piecewise
/// linear surfaces (thin plate energy would be zero for linear elements).
class CGLinearBezierBathymetrySmoother {
public:
  /// @brief Construct smoother for a quadtree mesh
  explicit CGLinearBezierBathymetrySmoother(
      const QuadtreeAdapter &mesh,
      const CGLinearBezierSmootherConfig &config = {});

  /// @brief Construct smoother from octree (uses bottom face)
  explicit CGLinearBezierBathymetrySmoother(
      const OctreeAdapter &octree,
      const CGLinearBezierSmootherConfig &config = {});

  // =========================================================================
  // Data input
  // =========================================================================

  void set_bathymetry_data(const BathymetrySource &source);
  void set_bathymetry_data(std::function<Real(Real, Real)> bathy_func);
  void set_scattered_points(const std::vector<Vec3> &points);
  void set_scattered_points(const std::vector<BathymetryPoint> &points);

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

  const CGLinearBezierSmootherConfig &config() const { return config_; }

  /// @brief Set profiling target for timing sub-operations
  /// @param profile Pointer to profile struct (nullptr to disable profiling)
  void set_profile(CGLinearIterationProfile *profile) { profile_ = profile; }

  // =========================================================================
  // Solve
  // =========================================================================

  void solve();
  bool is_solved() const { return solved_; }

  // =========================================================================
  // Solution evaluation
  // =========================================================================

  Real evaluate(Real x, Real y) const;
  Vec2 evaluate_gradient(Real x, Real y) const;
  const VecX &solution() const { return solution_; }

  /// @brief Get element control point values
  /// @return 4 control point z-values for this element
  VecX element_coefficients(Index elem) const;

  // =========================================================================
  // Transfer and output
  // =========================================================================

  void transfer_to_seabed(SeabedSurface &seabed) const;
  void write_vtk(const std::string &filename, int resolution = 6) const;
  void write_control_points_vtk(const std::string &filename) const;

  // =========================================================================
  // Diagnostics
  // =========================================================================

  Real data_residual() const;
  Real regularization_energy() const;
  Real objective_value() const;
  Real constraint_violation() const;

  Index num_global_dofs() const { return dof_manager_->num_global_dofs(); }
  Index num_free_dofs() const { return dof_manager_->num_free_dofs(); }
  Index num_constraints() const { return dof_manager_->num_constraints(); }

  const QuadtreeAdapter &mesh() const { return *quadtree_; }
  const CGLinearBezierDofManager &dof_manager() const { return *dof_manager_; }

private:
  std::unique_ptr<QuadtreeAdapter> quadtree_owned_;
  const QuadtreeAdapter *quadtree_ = nullptr;

  CGLinearBezierSmootherConfig config_;

  std::unique_ptr<LinearBezierBasis2D> basis_;
  std::unique_ptr<DirichletHessian> dirichlet_hessian_;
  std::unique_ptr<CGLinearBezierDofManager> dof_manager_;

  CGLinearIterationProfile *profile_ = nullptr;

  VecX solution_;
  bool solved_ = false;
  bool data_set_ = false;

  SpMat H_global_;
  SpMat BtWB_global_;
  VecX BtWd_global_;
  Real dTWd_global_ = 0;
  Real alpha_ = 0; // Cached scale normalization factor

  void init_components();
  void assemble_dirichlet_hessian();
  void assemble_data_fitting(std::function<Real(Real, Real)> bathy_func);
  void solve_unconstrained();
  void solve_with_constraints();
  Index find_element(Real x, Real y) const;
  Index find_element_with_fallback(Real x, Real y) const;
  Real evaluate_in_element(Index elem, Real x, Real y) const;
  Vec2 evaluate_gradient_in_element(Index elem, Real x, Real y) const;
};

} // namespace drifter
