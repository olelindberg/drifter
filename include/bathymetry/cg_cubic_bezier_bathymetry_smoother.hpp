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
    double matrix_build_ms = 0.0;          ///< Q matrix construction
    double constraint_build_ms = 0.0;      ///< C¹ edge constraint assembly (in DOF manager)
    double kkt_assembly_ms = 0.0;          ///< KKT system build
    double sparse_lu_compute_ms = 0.0;     ///< SparseLU factorization
    double sparse_lu_solve_ms = 0.0;       ///< SparseLU back-substitution
    double constraint_projection_ms = 0.0; ///< Constraint projection solve
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
};

/// @brief CG cubic Bezier bathymetry smoother with C¹ continuity
///
/// Fits cubic Bezier surfaces (16 DOFs per element) to bathymetry data
/// using Continuous Galerkin assembly with optional C¹ constraints.
class CGCubicBezierBathymetrySmoother : public CGBezierSmootherBase {
public:
    /// @brief Construct smoother for a quadtree mesh
    explicit CGCubicBezierBathymetrySmoother(const QuadtreeAdapter &mesh,
                                             const CGCubicBezierSmootherConfig &config = {});

    /// @brief Construct smoother from octree (uses bottom face)
    explicit CGCubicBezierBathymetrySmoother(const OctreeAdapter &octree,
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
    void set_solve_profile(CGCubicSolveProfile* profile) { solve_profile_ = profile; }

    /// @brief Set profile for timing assembly operations (hessian, data fitting)
    /// @param profile Pointer to iteration profile struct (null to disable)
    void set_profile(CGCubicIterationProfile* profile) { profile_ = profile; }

    // =========================================================================
    // Solution evaluation - inherited from base: evaluate, evaluate_gradient, solution
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

protected:
    // =========================================================================
    // CGBezierSmootherBase virtual method implementations
    // =========================================================================

    void set_bathymetry_data_impl(std::function<Real(Real, Real)> bathy_func) override;
    Index dof_manager_num_global_dofs() const override { return dof_manager_->num_global_dofs(); }
    Index dof_manager_num_free_dofs() const override { return dof_manager_->num_free_dofs(); }
    Index dof_manager_num_constraints() const override { return dof_manager_->num_constraints(); }
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

    CGCubicSolveProfile* solve_profile_ = nullptr;
    CGCubicIterationProfile* profile_ = nullptr;

    void init_components();
    void solve_with_constraints();
};

} // namespace drifter
