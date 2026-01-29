#pragma once

/// @file cg_bezier_bathymetry_smoother.hpp
/// @brief CG Bezier bathymetry smoother with natural cross-element coupling
///
/// Uses Continuous Galerkin assembly where DOFs at element boundaries are shared.
/// This provides natural cross-element coupling through the energy functional,
/// eliminating the need for explicit C² continuity constraints at conforming
/// interfaces. Only hanging node constraints are needed for non-conforming meshes.
///
/// Advantages over DG Bezier approach:
///   - Natural smoothing across element boundaries (no ridges)
///   - Fewer DOFs (shared at boundaries)
///   - Simpler constraint system (only hanging nodes)

#include "bathymetry/bezier_basis_2d.hpp"
#include "bathymetry/cg_bezier_dof_manager.hpp"
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
class BathymetrySource;
struct BathymetryPoint;

/// @brief Configuration for CG Bezier bathymetry smoother
struct CGBezierSmootherConfig {
    /// Data fitting weight relative to smoothness (lower = smoother)
    /// In the ShipMesh formulation: Q = α·H + λ·(BᵀWB + εI)
    Real lambda = 0.01;

    /// Weight for gradient penalty (first derivative regularization)
    Real gradient_weight = 0.0;

    /// Gauss points per direction for data sampling
    int ngauss_data = 6;

    /// Gauss points for energy integration
    int ngauss_energy = 6;

    /// Ridge regularization parameter (Tikhonov)
    Real ridge_epsilon = 1e-4;

    /// Optional elevation bounds (only applied if both are set)
    std::optional<Real> lower_bound;
    std::optional<Real> upper_bound;

    /// Maximum iterations for bound constraint solver
    int max_bound_iterations = 50;

    /// Tolerance for bound constraint satisfaction
    Real bound_tolerance = 1e-10;

    /// Enable C² continuity constraints at shared vertices
    /// When enabled, all derivatives up to order (2,2) are constrained to match
    /// at element vertices: z_u, z_v, z_uu, z_uv, z_vv, z_uuv, z_uvv, z_uuvv
    /// Note: z value is already shared via CG DOFs, so 8 constraints per vertex pair.
    bool enable_c2_constraints = false;

    /// Enable edge derivative constraints at Gauss points along shared edges
    /// When enabled, normal derivatives (z_n and z_nn) are constrained to match
    /// at Gauss quadrature points along each conforming interior edge.
    /// This complements vertex constraints by enforcing smoothness along entire edges.
    bool enable_edge_constraints = false;

    /// Number of Gauss points per edge for edge constraints (2, 3, or 4)
    int edge_ngauss = 4;
};

/// @brief CG Bezier bathymetry smoother with natural cross-element coupling
///
/// Fits quintic Bezier surfaces to bathymetry data using Continuous Galerkin
/// assembly. Key differences from DG BezierBathymetrySmoother:
///   - DOFs at element boundaries are SHARED (not duplicated)
///   - Thin plate energy assembled globally with cross-element coupling
///   - No explicit C² constraints needed at conforming interfaces
///   - Only hanging node constraints for non-conforming (AMR) meshes
///
/// This approach naturally eliminates ridges at element boundaries because
/// shared DOFs contribute to the energy of multiple elements, creating
/// smooth transitions through energy minimization.
class CGBezierBathymetrySmoother {
public:
    /// @brief Construct smoother for a quadtree mesh
    /// @param mesh 2D quadtree mesh
    /// @param config Configuration parameters
    explicit CGBezierBathymetrySmoother(const QuadtreeAdapter& mesh,
                                         const CGBezierSmootherConfig& config = {});

    /// @brief Construct smoother from octree (uses bottom face)
    /// @param octree 3D mesh
    /// @param config Configuration parameters
    explicit CGBezierBathymetrySmoother(const OctreeAdapter& octree,
                                         const CGBezierSmootherConfig& config = {});

    // =========================================================================
    // Data input
    // =========================================================================

    /// @brief Set bathymetry data from a BathymetrySource
    void set_bathymetry_data(const BathymetrySource& source);

    /// @brief Set bathymetry data from a function
    void set_bathymetry_data(std::function<Real(Real, Real)> bathy_func);

    /// @brief Set scattered XYZ data points
    void set_scattered_points(const std::vector<Vec3>& points);

    /// @brief Set scattered data with weights
    void set_scattered_points(const std::vector<BathymetryPoint>& points);

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
    const CGBezierSmootherConfig& config() const { return config_; }

    // =========================================================================
    // Solve
    // =========================================================================

    /// @brief Solve the optimization problem
    ///
    /// If the mesh is conforming (no hanging nodes), solves unconstrained:
    ///   minimize: xᵀQx + cᵀx
    ///
    /// If there are hanging nodes, solves KKT system:
    ///   [Q   Aᵀ] [x]   [-c]
    ///   [A    0] [μ] = [ 0]
    void solve();

    /// @brief Check if solution is available
    bool is_solved() const { return solved_; }

    // =========================================================================
    // Solution evaluation
    // =========================================================================

    /// @brief Evaluate smoothed bathymetry at (x, y)
    Real evaluate(Real x, Real y) const;

    /// @brief Evaluate gradient of smoothed bathymetry
    Vec2 evaluate_gradient(Real x, Real y) const;

    /// @brief Get full solution vector (all global DOF values)
    const VecX& solution() const { return solution_; }

    /// @brief Get element control point values
    /// @param elem Element index
    /// @return 36 control point z-values for this element (from global DOFs)
    VecX element_coefficients(Index elem) const;

    // =========================================================================
    // Transfer and output
    // =========================================================================

    /// @brief Transfer solution to SeabedSurface
    void transfer_to_seabed(SeabedSurface& seabed) const;

    /// @brief Write smoothed bathymetry to VTK file
    void write_vtk(const std::string& filename, int resolution = 10) const;

    /// @brief Write control points grid to VTK file
    void write_control_points_vtk(const std::string& filename) const;

    // =========================================================================
    // Diagnostics
    // =========================================================================

    /// @brief Get data fitting residual
    Real data_residual() const;

    /// @brief Get regularization energy xᵀHx
    Real regularization_energy() const;

    /// @brief Get total objective value
    Real objective_value() const;

    /// @brief Get constraint violation (for hanging nodes)
    Real constraint_violation() const;

    /// @brief Get number of global DOFs
    Index num_global_dofs() const { return dof_manager_->num_global_dofs(); }

    /// @brief Get number of free DOFs (after constraint elimination)
    Index num_free_dofs() const { return dof_manager_->num_free_dofs(); }

    /// @brief Get number of constraints (hanging nodes only)
    Index num_constraints() const { return dof_manager_->num_constraints(); }

    /// @brief Get the mesh
    const QuadtreeAdapter& mesh() const { return *quadtree_; }

    /// @brief Get the DOF manager
    const CGBezierDofManager& dof_manager() const { return *dof_manager_; }

private:
    /// Mesh (owned or referenced)
    std::unique_ptr<QuadtreeAdapter> quadtree_owned_;
    const QuadtreeAdapter* quadtree_ = nullptr;

    /// Configuration
    CGBezierSmootherConfig config_;

    /// Components
    std::unique_ptr<BezierBasis2D> basis_;
    std::unique_ptr<ThinPlateHessian> thin_plate_hessian_;
    std::unique_ptr<CGBezierDofManager> dof_manager_;

    /// Solution (global DOF values)
    VecX solution_;
    bool solved_ = false;
    bool data_set_ = false;

    /// Assembled matrices (cached)
    SpMat H_global_;       ///< Global thin plate Hessian
    SpMat BtWB_global_;    ///< Global data fitting normal matrix
    VecX BtWd_global_;     ///< Global data fitting RHS
    Real dTWd_global_ = 0; ///< Data self-product for residual computation

    /// Initialize components
    void init_components();

    /// Assemble global thin plate Hessian with CG assembly
    void assemble_thin_plate_hessian();

    /// Assemble data fitting matrices with CG assembly
    void assemble_data_fitting(std::function<Real(Real, Real)> bathy_func);

    /// Solve unconstrained (no hanging nodes)
    void solve_unconstrained();

    /// Solve with hanging node constraints (KKT system)
    void solve_with_constraints();

    /// Find element containing point
    Index find_element(Real x, Real y) const;

    /// Evaluate within an element
    Real evaluate_in_element(Index elem, Real x, Real y) const;

    /// Evaluate gradient within an element
    Vec2 evaluate_gradient_in_element(Index elem, Real x, Real y) const;
};

}  // namespace drifter
