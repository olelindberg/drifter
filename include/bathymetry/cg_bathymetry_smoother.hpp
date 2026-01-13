#pragma once

// CGBathymetrySmoother - Main interface for CG bathymetry smoothing
//
// This class provides the top-level interface for smoothing noisy bathymetry
// data using a C² continuous Galerkin (CG) finite element method. The
// smoothing is based on minimizing a thin plate spline energy functional:
//
//   E(u) = α∫|∇²u|² dA + β∫(u - u_data)² dA
//
// where α controls smoothness (curvature penalty) and β controls data fidelity.
//
// Usage:
//   OctreeAdapter octree(...);  // 3D mesh
//   CGBathymetrySmoother smoother(octree, alpha, beta);
//   smoother.set_bathymetry_data(geotiff_function);
//   smoother.solve();
//
//   // Evaluate smoothed bathymetry
//   Real depth = smoother.evaluate(x, y);
//
//   // Transfer to DG representation
//   smoother.transfer_to_seabed(seabed_surface);

#include "core/types.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "bathymetry/quintic_basis_2d.hpp"
#include "bathymetry/cg_dof_manager.hpp"
#include "bathymetry/biharmonic_assembler.hpp"
#include "mesh/seabed_surface.hpp"
#include <memory>
#include <functional>

namespace drifter {

// Forward declarations
class OctreeAdapter;

/// @brief C² continuous Galerkin bathymetry smoother
///
/// Provides thin plate spline smoothing of bathymetry data on a 2D mesh
/// that mirrors the bottom face of a 3D octree mesh.
class CGBathymetrySmoother {
public:
    /// @brief Construct smoother from octree mesh
    /// @param octree 3D mesh (bottom face used for 2D smoothing)
    /// @param alpha Smoothing weight (curvature penalty)
    /// @param beta Data fitting weight
    CGBathymetrySmoother(const OctreeAdapter& octree, Real alpha, Real beta);

    /// @brief Construct smoother from standalone 2D mesh
    /// @param mesh 2D quadtree mesh
    /// @param alpha Smoothing weight
    /// @param beta Data fitting weight
    CGBathymetrySmoother(const QuadtreeAdapter& mesh, Real alpha, Real beta);

    /// @brief Set bathymetry data from a function
    /// @param bathy_func Function returning depth at (x, y)
    void set_bathymetry_data(std::function<Real(Real, Real)> bathy_func);

    /// @brief Set bathymetry data from a BathymetrySource object
    /// @param bathy Bathymetry data source
    void set_bathymetry_data(std::unique_ptr<BathymetrySource> bathy);

    /// @brief Solve the smoothing problem
    ///
    /// Assembles and solves the biharmonic system:
    ///   (α K_biharm + β M) u = β M u_data
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
    /// @return Gradient (∂u/∂x, ∂u/∂y)
    Vec2 evaluate_gradient(Real x, Real y) const;

    /// @brief Get solution at a DOF
    /// @param dof Global DOF index
    /// @return Solution value
    Real solution_at_dof(Index dof) const;

    /// @brief Get full solution vector
    const VecX& solution() const { return solution_global_; }

    // =========================================================================
    // Transfer to DG representation
    // =========================================================================

    /// @brief Transfer smoothed bathymetry to SeabedSurface
    ///
    /// Samples the CG solution at the DG element DOF positions and
    /// stores the result in the SeabedSurface for use in the 3D simulation.
    ///
    /// @param seabed Target seabed surface
    void transfer_to_seabed(SeabedSurface& seabed) const;

    // =========================================================================
    // Parameter access
    // =========================================================================

    /// Get smoothing weight
    Real alpha() const { return alpha_; }

    /// Get data fitting weight
    Real beta() const { return beta_; }

    /// Get the 2D mesh
    const QuadtreeAdapter& mesh() const { return *quadtree_; }

    /// Get number of DOFs
    Index num_dofs() const { return dof_manager_->num_global_dofs(); }

    /// Get DOF manager
    const CGDofManager& dof_manager() const { return *dof_manager_; }

private:
    /// Mesh (owned or referenced)
    std::unique_ptr<QuadtreeAdapter> quadtree_owned_;
    const QuadtreeAdapter* quadtree_ = nullptr;

    /// Basis functions
    QuinticBasis2D basis_;

    /// DOF manager
    std::unique_ptr<CGDofManager> dof_manager_;

    /// Assembler
    std::unique_ptr<BiharmonicAssembler> assembler_;

    /// Bathymetry data
    std::unique_ptr<BathymetrySource> bathy_;

    /// Smoothing parameters
    Real alpha_;
    Real beta_;

    /// Solution
    VecX solution_free_;
    VecX solution_global_;
    bool solved_ = false;

    /// Initialize components
    void init_components();

    /// Find element containing point
    Index find_element(Real x, Real y) const;

    /// Evaluate solution within an element
    Real evaluate_in_element(Index elem, Real x, Real y) const;

    /// Evaluate gradient within an element
    Vec2 evaluate_gradient_in_element(Index elem, Real x, Real y) const;
};

}  // namespace drifter
