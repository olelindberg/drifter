#pragma once

/// @file cg_cubic_bezier_bathymetry_smoother.hpp
/// @brief CG cubic Bezier bathymetry smoother with C¹ continuity constraints
///
/// Uses Continuous Galerkin assembly where DOFs at element boundaries are shared.
/// Cubic Bezier (degree 3, 4×4 = 16 DOFs) with optional C¹ constraints.

#include "bathymetry/cubic_bezier_basis_2d.hpp"
#include "bathymetry/cg_cubic_bezier_dof_manager.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "bathymetry/cubic_thin_plate_hessian.hpp"
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

/// @brief Configuration for CG cubic Bezier bathymetry smoother
struct CGCubicBezierSmootherConfig {
    /// Data fitting weight relative to smoothness
    Real lambda = 0.01;

    /// Weight for gradient penalty
    Real gradient_weight = 0.0;

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

    /// Enable C¹ edge constraints (z_n matching at Gauss points)
    bool enable_c1_edge_constraints = false;

    /// Number of Gauss points per edge for edge constraints
    int edge_ngauss = 4;
};

/// @brief CG cubic Bezier bathymetry smoother with C¹ continuity
///
/// Fits cubic Bezier surfaces (16 DOFs per element) to bathymetry data
/// using Continuous Galerkin assembly with optional C¹ constraints.
class CGCubicBezierBathymetrySmoother {
public:
    /// @brief Construct smoother for a quadtree mesh
    explicit CGCubicBezierBathymetrySmoother(const QuadtreeAdapter& mesh,
                                              const CGCubicBezierSmootherConfig& config = {});

    /// @brief Construct smoother from octree (uses bottom face)
    explicit CGCubicBezierBathymetrySmoother(const OctreeAdapter& octree,
                                              const CGCubicBezierSmootherConfig& config = {});

    // =========================================================================
    // Data input
    // =========================================================================

    void set_bathymetry_data(const BathymetrySource& source);
    void set_bathymetry_data(std::function<Real(Real, Real)> bathy_func);
    void set_scattered_points(const std::vector<Vec3>& points);
    void set_scattered_points(const std::vector<BathymetryPoint>& points);

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

    const CGCubicBezierSmootherConfig& config() const { return config_; }

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
    const VecX& solution() const { return solution_; }

    /// @brief Get element control point values
    /// @return 16 control point z-values for this element
    VecX element_coefficients(Index elem) const;

    // =========================================================================
    // Transfer and output
    // =========================================================================

    void transfer_to_seabed(SeabedSurface& seabed) const;
    void write_vtk(const std::string& filename, int resolution = 8) const;
    void write_control_points_vtk(const std::string& filename) const;

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

    const QuadtreeAdapter& mesh() const { return *quadtree_; }
    const CGCubicBezierDofManager& dof_manager() const { return *dof_manager_; }

private:
    std::unique_ptr<QuadtreeAdapter> quadtree_owned_;
    const QuadtreeAdapter* quadtree_ = nullptr;

    CGCubicBezierSmootherConfig config_;

    std::unique_ptr<CubicBezierBasis2D> basis_;
    std::unique_ptr<CubicThinPlateHessian> thin_plate_hessian_;
    std::unique_ptr<CGCubicBezierDofManager> dof_manager_;

    VecX solution_;
    bool solved_ = false;
    bool data_set_ = false;

    SpMat H_global_;
    SpMat BtWB_global_;
    VecX BtWd_global_;
    Real dTWd_global_ = 0;

    void init_components();
    void assemble_thin_plate_hessian();
    void assemble_data_fitting(std::function<Real(Real, Real)> bathy_func);
    void solve_unconstrained();
    void solve_with_constraints();
    Index find_element(Real x, Real y) const;
    Real evaluate_in_element(Index elem, Real x, Real y) const;
    Vec2 evaluate_gradient_in_element(Index elem, Real x, Real y) const;
};

}  // namespace drifter
