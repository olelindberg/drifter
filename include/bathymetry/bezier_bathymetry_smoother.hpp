#pragma once

/// @file bezier_bathymetry_smoother.hpp
/// @brief Main solver for Bezier bathymetry surface fitting with C² continuity
///
/// Implements constrained optimization to fit quintic Bezier surfaces to
/// bathymetry data with thin plate spline regularization and C² continuity
/// enforcement at element interfaces.
///
/// Optimization problem:
///   minimize: ||B·x - b||²_W + λ·x^T·H·x
///   subject to: A_eq·x = 0  (C² continuity)
///               lower ≤ x ≤ upper  (optional bounds)

#include "bathymetry/bezier_basis_2d.hpp"
#include "bathymetry/bezier_c2_constraints.hpp"
#include "bathymetry/bezier_data_fitting.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "bathymetry/thin_plate_hessian.hpp"
#include "core/types.hpp"
#include "mesh/seabed_surface.hpp"
#include <functional>
#include <memory>
#include <optional>

namespace drifter {

// Forward declarations
class OctreeAdapter;

/// @brief Configuration for Bezier bathymetry smoother
struct BezierSmootherConfig {
  Real lambda =
      0.01; ///< Data fitting weight relative to smoothness (lower = smoother)
  Real gradient_weight =
      0.0; ///< Weight for gradient penalty (first derivative regularization)
  int ngauss_data = 6;   ///< Gauss points per direction for data sampling
  int ngauss_energy = 6; ///< Gauss points for energy integration

  /// Optional elevation bounds (only applied if both are set)
  std::optional<Real> lower_bound;
  std::optional<Real> upper_bound;

  /// Maximum iterations for bound constraint solver
  int max_bound_iterations = 50;

  /// Tolerance for bound constraint satisfaction
  Real bound_tolerance = 1e-10;

  /// Enable Dirichlet boundary conditions on domain boundary
  /// When enabled, boundary DOFs are constrained to match the input bathymetry
  /// data
  bool enable_boundary_dirichlet = true;
};

/// @brief Bezier bathymetry smoother with C² continuity
///
/// Fits quintic Bezier surfaces (36 DOFs per element) to bathymetry data
/// using constrained optimization. Features:
///   - Weighted least-squares data fitting
///   - Thin plate spline regularization
///   - C² continuity at element interfaces via equality constraints
///   - Optional bound constraints (elevation limits)
///   - Support for both scattered points and gridded data
class BezierBathymetrySmoother {
public:
  /// @brief Construct smoother for a mesh
  /// @param mesh 2D quadtree mesh
  /// @param config Configuration parameters
  explicit BezierBathymetrySmoother(const QuadtreeAdapter &mesh,
                                    const BezierSmootherConfig &config = {});

  /// @brief Construct smoother from octree (uses bottom face)
  /// @param octree 3D mesh
  /// @param config Configuration parameters
  explicit BezierBathymetrySmoother(const OctreeAdapter &octree,
                                    const BezierSmootherConfig &config = {});

  // =========================================================================
  // Data input
  // =========================================================================

  /// @brief Set bathymetry data from a BathymetrySource
  /// @param source Bathymetry data source (e.g., GeoTIFF)
  void set_bathymetry_data(const BathymetrySource &source);

  /// @brief Set bathymetry data from a function
  /// @param bathy_func Function taking (x, y) -> depth
  void set_bathymetry_data(std::function<Real(Real, Real)> bathy_func);

  /// @brief Set scattered XYZ data points
  /// @param points Vector of (x, y, z) points
  void set_scattered_points(const std::vector<Vec3> &points);

  /// @brief Set scattered data with weights
  /// @param points Vector of BathymetryPoint (x, y, z, weight)
  void set_scattered_points(const std::vector<BathymetryPoint> &points);

  // =========================================================================
  // Configuration
  // =========================================================================

  /// @brief Set smoothing weight (lambda)
  void set_smoothing_weight(Real lambda) { config_.lambda = lambda; }

  /// @brief Set elevation bounds for bound-constrained optimization
  void set_bounds(Real lower, Real upper) {
    config_.lower_bound = lower;
    config_.upper_bound = upper;
  }

  /// @brief Clear elevation bounds
  void clear_bounds() {
    config_.lower_bound = std::nullopt;
    config_.upper_bound = std::nullopt;
  }

  /// @brief Get configuration
  const BezierSmootherConfig &config() const { return config_; }

  // =========================================================================
  // Solve
  // =========================================================================

  /// @brief Solve the constrained optimization problem
  ///
  /// Solves the KKT system for the equality-constrained QP:
  ///   [Q   A^T] [x]   [-c]
  ///   [A    0 ] [y] = [ 0]
  ///
  /// If bounds are specified, uses active-set iteration.
  void solve();

  /// @brief Check if solution is available
  bool is_solved() const { return solved_; }

  // =========================================================================
  // Solution evaluation
  // =========================================================================

  /// @brief Evaluate smoothed bathymetry at (x, y)
  /// @param x, y Physical coordinates
  /// @return Smoothed depth value
  Real evaluate(Real x, Real y) const;

  /// @brief Evaluate gradient of smoothed bathymetry
  /// @param x, y Physical coordinates
  /// @return Gradient (dz/dx, dz/dy)
  Vec2 evaluate_gradient(Real x, Real y) const;

  /// @brief Get full solution vector (all control point z-values)
  const VecX &solution() const { return solution_; }

  /// @brief Get element control point values
  /// @param elem Element index
  /// @return 36 control point z-values for this element
  VecX element_coefficients(Index elem) const;

  // =========================================================================
  // Transfer and output
  // =========================================================================

  /// @brief Transfer solution to SeabedSurface
  ///
  /// Since both use Bezier/Bernstein representation, transfer is direct.
  /// @param seabed Target seabed surface
  void transfer_to_seabed(SeabedSurface &seabed) const;

  /// @brief Write smoothed bathymetry to VTK file
  /// @param filename Base filename (will append .vtu)
  /// @param resolution Subdivisions per element for visualization
  void write_vtk(const std::string &filename, int resolution = 10) const;

  /// @brief Write control points grid to VTK file
  ///
  /// Outputs the 6×6 Bezier control points per element as vertices
  /// with their z-values, connected by a structured grid for visualization.
  /// @param filename Base filename (will append .vtu)
  void write_control_points_vtk(const std::string &filename) const;

  // =========================================================================
  // Diagnostics
  // =========================================================================

  /// @brief Get data fitting residual ||B*x - b||²_W
  Real data_residual() const;

  /// @brief Get regularization energy x^T * H * x
  Real regularization_energy() const;

  /// @brief Get total objective value
  Real objective_value() const;

  /// @brief Get constraint violation ||A_eq * x||
  Real constraint_violation() const;

  /// @brief Get number of DOFs
  Index num_dofs() const { return data_assembler_->total_dofs(); }

  /// @brief Get number of constraints
  Index num_constraints() const {
    return constraint_builder_->num_constraints();
  }

  /// @brief Get the mesh
  const QuadtreeAdapter &mesh() const { return *quadtree_; }

private:
  /// Mesh (owned or referenced)
  std::unique_ptr<QuadtreeAdapter> quadtree_owned_;
  const QuadtreeAdapter *quadtree_ = nullptr;

  /// Configuration
  BezierSmootherConfig config_;

  /// Components
  std::unique_ptr<BezierBasis2D> basis_;
  std::unique_ptr<ThinPlateHessian> hessian_;
  std::unique_ptr<BezierC2ConstraintBuilder> constraint_builder_;
  std::unique_ptr<BezierDataFittingAssembler> data_assembler_;

  /// Solution
  VecX solution_;
  bool solved_ = false;

  /// Cached matrices for diagnostics
  mutable SpMat B_cached_;
  mutable VecX b_cached_;
  mutable VecX w_cached_;
  mutable bool cache_valid_ = false;

  /// Initialize components
  void init_components();

  /// Solve without bound constraints (KKT system)
  void solve_kkt();

  /// Solve with bound constraints (active-set method)
  void solve_with_bounds();

  /// Apply active-set iteration for bounds
  void apply_bound_constraints();

  /// Build the global regularization Hessian
  MatX build_global_hessian() const;

  /// Find element containing point
  Index find_element(Real x, Real y) const;

  /// Evaluate within an element
  Real evaluate_in_element(Index elem, Real x, Real y) const;

  /// Evaluate gradient within an element
  Vec2 evaluate_gradient_in_element(Index elem, Real x, Real y) const;
};

} // namespace drifter
