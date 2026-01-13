#pragma once

// BiharmonicAssembler - Assembly for CG bathymetry smoothing
//
// Assembles the global stiffness matrix and RHS vector for the
// variational form of the biharmonic smoothing problem:
//
//   Find u such that: α∫∇²u·∇²v + β∫u·v = β∫u_data·v  for all v
//
// This is a thin plate spline formulation that minimizes:
//   α∫|κ|² dA + β∫(u - u_data)² dA
// where κ = ∇²u is the curvature (Laplacian) of the surface.
//
// The assembly follows wobbler's CG2DBiharmonicIntegration pattern:
// - Element loop with local matrix computation
// - Gauss quadrature for integration
// - Sparse triplet assembly
//
// Usage:
//   BiharmonicAssembler assembler(quadtree, basis, dofs, alpha, beta);
//   SpMat K = assembler.assemble_stiffness();
//   VecX f = assembler.assemble_rhs(bathymetry_data);

#include "core/types.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "bathymetry/quintic_basis_2d.hpp"
#include "bathymetry/cg_dof_manager.hpp"
#include <functional>
#include <memory>

namespace drifter {

/// @brief Bathymetry data source interface
///
/// Provides depth values at arbitrary (x, y) coordinates.
class BathymetrySource {
public:
    virtual ~BathymetrySource() = default;

    /// Evaluate depth at (x, y)
    virtual Real evaluate(Real x, Real y) const = 0;
};

/// @brief Simple function-based bathymetry data
class FunctionBathymetry : public BathymetrySource {
public:
    explicit FunctionBathymetry(std::function<Real(Real, Real)> func)
        : func_(std::move(func)) {}

    Real evaluate(Real x, Real y) const override {
        return func_(x, y);
    }

private:
    std::function<Real(Real, Real)> func_;
};

/// @brief Assembly for biharmonic bathymetry smoothing
///
/// Implements the variational form assembly for the thin plate spline
/// smoothing problem on a 2D quintic CG mesh.
class BiharmonicAssembler {
public:
    /// @brief Construct assembler
    /// @param mesh 2D quadtree mesh
    /// @param basis Quintic basis functions
    /// @param dofs DOF manager
    /// @param alpha Smoothing weight (curvature penalty)
    /// @param beta Data fitting weight
    BiharmonicAssembler(const QuadtreeAdapter& mesh,
                        const QuinticBasis2D& basis,
                        const CGDofManager& dofs,
                        Real alpha,
                        Real beta);

    // =========================================================================
    // Global assembly
    // =========================================================================

    /// @brief Assemble global stiffness matrix
    ///
    /// K_ij = α∫∇²φ_i·∇²φ_j dA + β∫φ_i·φ_j dA
    ///
    /// @return Sparse matrix (num_global_dofs x num_global_dofs)
    SpMat assemble_stiffness() const;

    /// @brief Assemble global RHS vector
    ///
    /// f_i = β∫u_data·φ_i dA
    ///
    /// @param bathy Bathymetry data source
    /// @return RHS vector (num_global_dofs)
    VecX assemble_rhs(const BathymetrySource& bathy) const;

    /// @brief Assemble reduced system (after constraint elimination)
    ///
    /// @param bathy Bathymetry data source
    /// @param[out] K_red Reduced stiffness matrix (num_free_dofs x num_free_dofs)
    /// @param[out] f_red Reduced RHS vector (num_free_dofs)
    void assemble_reduced_system(const BathymetrySource& bathy,
                                 SpMat& K_red, VecX& f_red) const;

    // =========================================================================
    // Element-level operations
    // =========================================================================

    /// @brief Compute element biharmonic stiffness matrix
    ///
    /// K^e_ij = α∫∇²φ_i·∇²φ_j dA  (Laplacian product)
    ///
    /// @param elem Element index
    /// @return Local stiffness matrix (36 x 36)
    MatX element_biharmonic(Index elem) const;

    /// @brief Compute element mass matrix
    ///
    /// M^e_ij = β∫φ_i·φ_j dA
    ///
    /// @param elem Element index
    /// @return Local mass matrix (36 x 36)
    MatX element_mass(Index elem) const;

    /// @brief Compute element stiffness (biharmonic + mass)
    ///
    /// K^e = α * K^e_biharmonic + β * M^e
    ///
    /// @param elem Element index
    /// @return Local stiffness matrix (36 x 36)
    MatX element_stiffness(Index elem) const;

    /// @brief Compute element RHS vector
    ///
    /// f^e_i = β∫u_data·φ_i dA
    ///
    /// @param elem Element index
    /// @param bathy Bathymetry data source
    /// @return Local RHS vector (36)
    VecX element_rhs(Index elem, const BathymetrySource& bathy) const;

    // =========================================================================
    // Jacobian and geometry
    // =========================================================================

    /// @brief Compute Jacobian at Gauss points
    /// @param elem Element index
    /// @return Vector of Jacobian determinants at Gauss points
    VecX compute_jacobian(Index elem) const;

    /// @brief Get smoothing weight
    Real alpha() const { return alpha_; }

    /// @brief Get data fitting weight
    Real beta() const { return beta_; }

private:
    const QuadtreeAdapter& mesh_;
    const QuinticBasis2D& basis_;
    const CGDofManager& dofs_;
    Real alpha_;  // Smoothing weight
    Real beta_;   // Data fitting weight

    // Cached quadrature data
    VecX gauss_nodes_;    // 1D Gauss nodes
    VecX gauss_weights_;  // 1D Gauss weights
    int num_gauss_1d_;    // Number of 1D Gauss points

    // Precomputed basis values at Gauss points
    MatX phi_at_gauss_;      // Basis functions (n_gauss x 36)
    MatX lap_at_gauss_;      // Laplacians (n_gauss x 36)

    /// Initialize quadrature data
    void init_quadrature();

    /// Map reference point to physical coordinates
    Vec2 map_to_physical(Index elem, Real xi, Real eta) const;
};

}  // namespace drifter
