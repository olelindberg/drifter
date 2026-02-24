#pragma once

/// @file cg_linear_bezier_bathymetry_smoother.hpp
/// @brief CG linear Bezier bathymetry smoother with Laplacian smoothing
///
/// Uses Continuous Galerkin assembly where DOFs at element boundaries are
/// shared. Linear Bezier (degree 1, 2x2 = 4 DOFs) with C0 continuity from
/// shared DOFs. Uses Dirichlet/Laplace energy for smoothing instead of thin
/// plate energy.

#include "bathymetry/cg_bezier_smoother_base.hpp"
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
class CGLinearBezierBathymetrySmoother : public CGBezierSmootherBase {
public:
    /// @brief Construct smoother for a quadtree mesh
    explicit CGLinearBezierBathymetrySmoother(const QuadtreeAdapter &mesh,
                                              const CGLinearBezierSmootherConfig &config = {});

    /// @brief Construct smoother from octree (uses bottom face)
    explicit CGLinearBezierBathymetrySmoother(const OctreeAdapter &octree,
                                              const CGLinearBezierSmootherConfig &config = {});

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

    const CGLinearBezierSmootherConfig &config() const { return config_; }

    /// @brief Set profiling target for timing sub-operations
    /// @param profile Pointer to profile struct (nullptr to disable profiling)
    void set_profile(CGLinearIterationProfile* profile) { profile_ = profile; }

    // =========================================================================
    // Solve
    // =========================================================================

    void solve();

    // =========================================================================
    // Solution evaluation - inherited from base: evaluate, evaluate_gradient, solution
    // =========================================================================

    // =========================================================================
    // Transfer and output
    // =========================================================================

    // transfer_to_seabed inherited from base
    void write_vtk(const std::string &filename, int resolution = 6) const;
    void write_control_points_vtk(const std::string &filename) const;

    // =========================================================================
    // Diagnostics
    // =========================================================================

    // data_residual, regularization_energy inherited from base
    Real objective_value() const;
    Real constraint_violation() const;

    // num_global_dofs, num_free_dofs, num_constraints, mesh inherited from base
    const CGLinearBezierDofManager &dof_manager() const { return *dof_manager_; }

    // element_coefficients() declared in base class as public pure virtual
    VecX element_coefficients(Index elem) const override;

protected:
    // =========================================================================
    // CGBezierSmootherBase virtual method implementations
    // =========================================================================

    void set_bathymetry_data_impl(std::function<Real(Real, Real)> bathy_func) override;
    Real evaluate_scalar(const VecX &coeffs, Real u, Real v) const override;
    Vec2 evaluate_gradient_uv(const VecX &coeffs, Real u, Real v) const override;
    Index dof_manager_num_global_dofs() const override { return dof_manager_->num_global_dofs(); }
    Index dof_manager_num_free_dofs() const override { return dof_manager_->num_free_dofs(); }
    Index dof_manager_num_constraints() const override { return dof_manager_->num_constraints(); }
    const std::vector<Index> &element_global_dofs(Index elem) const override {
        return dof_manager_->element_dofs(elem);
    }

private:
    CGLinearBezierSmootherConfig config_;

    std::unique_ptr<LinearBezierBasis2D> basis_;
    std::unique_ptr<DirichletHessian> dirichlet_hessian_;
    std::unique_ptr<CGLinearBezierDofManager> dof_manager_;

    CGLinearIterationProfile* profile_ = nullptr;

    Real alpha_ = 0; ///< Cached scale normalization factor

    void init_components();
    void assemble_data_fitting(std::function<Real(Real, Real)> bathy_func);
    void solve_unconstrained();
    void solve_with_constraints();
};

} // namespace drifter
