#pragma once

// BiharmonicAssembler - Assembly for CG bathymetry smoothing with IPDG
//
// Assembles the global stiffness matrix and RHS vector for the
// variational form of the biharmonic smoothing problem with
// interior penalty for C¹ continuity:
//
//   Find u such that: α∫∇²u·∇²v + β∫u·v + γ∫[[∂u/∂n]][[∂v/∂n]] = β∫u_data·v
//
// This is a thin plate spline formulation that minimizes:
//   α∫|κ|² dA + β∫(u - u_data)² dA
// where κ = ∇²u is the curvature (Laplacian) of the surface.
//
// The IPDG penalty term γ/h∫[[∂u/∂n]]² weakly enforces C¹ continuity
// at element interfaces (both conforming and non-conforming).
//
// Usage:
//   BiharmonicAssembler assembler(quadtree, basis, dofs, alpha, beta, penalty);
//   SpMat K = assembler.assemble_stiffness();
//   VecX f = assembler.assemble_rhs(bathymetry_data);

#include "core/types.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "bathymetry/lagrange_basis_2d.hpp"
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

/// @brief Assembly for biharmonic bathymetry smoothing with IPDG
///
/// Implements the variational form assembly for the thin plate spline
/// smoothing problem on a 2D CG mesh with interior penalty for C¹ continuity.
class BiharmonicAssembler {
public:
    /// @brief Construct assembler
    /// @param mesh 2D quadtree mesh
    /// @param basis Lagrange basis functions
    /// @param dofs DOF manager
    /// @param alpha Smoothing weight (curvature penalty)
    /// @param beta Data fitting weight
    /// @param penalty IPDG penalty parameter for C¹ continuity (default 500)
    BiharmonicAssembler(const QuadtreeAdapter& mesh,
                        const LagrangeBasis2D& basis,
                        const CGDofManager& dofs,
                        Real alpha,
                        Real beta,
                        Real penalty = 500.0);

    // =========================================================================
    // Global assembly
    // =========================================================================

    /// @brief Assemble global stiffness matrix
    ///
    /// K_ij = α∫∇²φ_i·∇²φ_j dA + β∫φ_i·φ_j dA + penalty terms
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
    /// @return Local stiffness matrix (ndof x ndof)
    MatX element_biharmonic(Index elem) const;

    /// @brief Compute element mass matrix
    ///
    /// M^e_ij = β∫φ_i·φ_j dA
    ///
    /// @param elem Element index
    /// @return Local mass matrix (ndof x ndof)
    MatX element_mass(Index elem) const;

    /// @brief Compute element stiffness (biharmonic + mass)
    ///
    /// K^e = α * K^e_biharmonic + β * M^e
    ///
    /// @param elem Element index
    /// @return Local stiffness matrix (ndof x ndof)
    MatX element_stiffness(Index elem) const;

    /// @brief Compute element RHS vector
    ///
    /// f^e_i = β∫u_data·φ_i dA
    ///
    /// @param elem Element index
    /// @param bathy Bathymetry data source
    /// @return Local RHS vector (ndof)
    VecX element_rhs(Index elem, const BathymetrySource& bathy) const;

    // =========================================================================
    // IPDG penalty assembly
    // =========================================================================

    /// @brief Assemble IPDG penalty contributions to stiffness
    ///
    /// Adds penalty terms for normal derivative jumps at all interior edges.
    /// Handles both conforming and non-conforming (hanging node) interfaces.
    ///
    /// @param triplets Output triplet list to add to
    void assemble_ipdg_penalty(std::vector<Eigen::Triplet<Real>>& triplets) const;

    /// @brief Compute edge penalty matrix for a conforming edge
    ///
    /// Penalty term: (γ/h) ∫_edge [[∂u/∂n]]·[[∂v/∂n]] ds
    ///
    /// @param elem_left Left element index
    /// @param elem_right Right element index
    /// @param edge_left Edge ID on left element (0-3)
    /// @param edge_right Edge ID on right element (0-3)
    /// @return Pair of matrices (K_LL, K_LR) for left-left and left-right coupling
    std::pair<MatX, MatX> edge_penalty_conforming(
        Index elem_left, Index elem_right,
        int edge_left, int edge_right) const;

    /// @brief Compute edge penalty for non-conforming edge (fine side)
    ///
    /// @param elem_fine Fine element index
    /// @param elem_coarse Coarse element index
    /// @param edge_fine Edge ID on fine element
    /// @param edge_coarse Edge ID on coarse element
    /// @param subedge_idx Which sub-edge of coarse edge (0 or 1)
    /// @return Pair of matrices (K_FF, K_FC) for fine-fine and fine-coarse coupling
    std::pair<MatX, MatX> edge_penalty_nonconforming(
        Index elem_fine, Index elem_coarse,
        int edge_fine, int edge_coarse,
        int subedge_idx) const;

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

    /// @brief Get IPDG penalty parameter
    Real penalty() const { return penalty_; }

    /// @brief Get number of DOFs per element
    int num_element_dofs() const { return basis_.num_dofs(); }

private:
    const QuadtreeAdapter& mesh_;
    const LagrangeBasis2D& basis_;
    const CGDofManager& dofs_;
    Real alpha_;    // Smoothing weight
    Real beta_;     // Data fitting weight
    Real penalty_;  // IPDG penalty parameter

    // Cached quadrature data
    VecX gauss_nodes_;    // 1D Gauss nodes
    VecX gauss_weights_;  // 1D Gauss weights
    int num_gauss_1d_;    // Number of 1D Gauss points

    // Precomputed basis values at Gauss points
    MatX phi_at_gauss_;      // Basis functions (n_gauss x ndof)
    MatX lap_at_gauss_;      // Laplacians (n_gauss x ndof)

    /// Initialize quadrature data
    void init_quadrature();

    /// Map reference point to physical coordinates
    Vec2 map_to_physical(Index elem, Real xi, Real eta) const;

    /// Map parameter t in [-1,1] to sub-interval for non-conforming edges
    /// @param t Parameter in [-1, 1]
    /// @param subedge_idx 0 for first half, 1 for second half
    /// @return Mapped parameter in [-1, 1] for the coarse edge
    Real map_to_coarse_edge(Real t, int subedge_idx) const;
};

}  // namespace drifter
