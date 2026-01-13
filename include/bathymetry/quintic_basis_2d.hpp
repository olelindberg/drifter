#pragma once

/// @file quintic_basis_2d.hpp
/// @brief 2D tensor-product quintic Lagrange basis for CG bathymetry smoothing
///
/// This provides quintic (order 5) basis functions on quadrilateral elements
/// for the continuous Galerkin formulation of the biharmonic smoothing problem.
/// Uses LGL (Legendre-Gauss-Lobatto) nodes for optimal interpolation properties.

#include "core/types.hpp"
#include <vector>

namespace drifter {

/// @brief 2D tensor-product quintic Lagrange basis on reference element [-1,1]^2
///
/// Uses 6 LGL nodes per direction (order 5), giving 36 DOFs per element.
/// Provides basis function evaluation, gradients, Laplacian, and Hessians
/// needed for C^1 continuity enforcement and biharmonic operator assembly.
///
/// DOF indexing: dof = i + N1D * j where i is xi-direction, j is eta-direction
class QuinticBasis2D {
public:
    static constexpr int ORDER = 5;
    static constexpr int N1D = 6;       ///< Nodes per direction (order + 1)
    static constexpr int NDOF = 36;     ///< Total DOFs per element (6 x 6)

    /// @brief Construct quintic basis with LGL nodes
    QuinticBasis2D();

    // =========================================================================
    // Node access
    // =========================================================================

    /// Polynomial order
    int order() const { return ORDER; }

    /// Number of nodes in 1D
    int num_nodes_1d() const { return N1D; }

    /// Total number of DOFs per element
    int num_dofs() const { return NDOF; }

    /// 1D LGL nodes in [-1, 1]
    const VecX& nodes_1d() const { return nodes_; }

    /// 1D quadrature weights
    const VecX& weights_1d() const { return weights_; }

    /// Get 2D node position for DOF index in reference coordinates
    /// @param dof DOF index (0 to 35)
    /// @return (xi, eta) in [-1, 1]^2
    Vec2 node_position(int dof) const;

    // =========================================================================
    // DOF indexing
    // =========================================================================

    /// Convert (i, j) tensor indices to linear DOF index
    /// @param i Index in xi direction (0 to 5)
    /// @param j Index in eta direction (0 to 5)
    static int dof_index(int i, int j) { return i + N1D * j; }

    /// Extract (i, j) from linear DOF index
    static void dof_ij(int dof, int& i, int& j) {
        j = dof / N1D;
        i = dof % N1D;
    }

    // =========================================================================
    // Basis function evaluation
    // =========================================================================

    /// Evaluate all 36 basis functions at point (xi, eta)
    /// @param xi First reference coordinate in [-1, 1]
    /// @param eta Second reference coordinate in [-1, 1]
    /// @return Vector of 36 basis function values
    VecX evaluate(Real xi, Real eta) const;

    /// Evaluate gradient of all basis functions at (xi, eta)
    /// @return Matrix (36 x 2): column 0 = d/dxi, column 1 = d/deta
    MatX evaluate_gradient(Real xi, Real eta) const;

    /// Evaluate Laplacian of all basis functions at (xi, eta)
    /// @return Vector of 36 values: d^2/dxi^2 + d^2/deta^2
    VecX evaluate_laplacian(Real xi, Real eta) const;

    /// Evaluate second derivatives of all basis functions at (xi, eta)
    /// @param d2_dxi2 Output: d^2 phi / dxi^2 (36 values)
    /// @param d2_deta2 Output: d^2 phi / deta^2 (36 values)
    /// @param d2_dxideta Output: d^2 phi / dxi deta (36 values)
    void evaluate_second_derivatives(Real xi, Real eta,
                                     VecX& d2_dxi2, VecX& d2_deta2, VecX& d2_dxideta) const;

    /// Evaluate Hessian matrices of all basis functions at (xi, eta)
    /// @return Vector of 36 2x2 matrices: H[dof](i,j) = d^2 phi_dof / dx_i dx_j
    std::vector<Mat2> evaluate_hessian(Real xi, Real eta) const;

    // =========================================================================
    // Pre-computed derivative matrices at nodes
    // =========================================================================

    /// 1D first derivative matrix: D[i,j] = d(phi_j)/dxi at node i
    const MatX& derivative_matrix_1d() const { return D_; }

    /// 1D second derivative matrix: D2[i,j] = d^2(phi_j)/dxi^2 at node i
    const MatX& second_derivative_matrix_1d() const { return D2_; }

    // =========================================================================
    // Quadrature (for integration)
    // =========================================================================

    /// 2D quadrature weights for tensor-product rule
    /// Weight for DOF (i,j) = weights_1d(i) * weights_1d(j)
    Real quadrature_weight(int i, int j) const { return weights_(i) * weights_(j); }

    /// Total 2D quadrature weight for a DOF
    Real quadrature_weight(int dof) const {
        int i, j;
        dof_ij(dof, i, j);
        return quadrature_weight(i, j);
    }

    // =========================================================================
    // Edge and corner DOF identification
    // =========================================================================

    /// Get DOF indices on an edge
    /// @param edge_id 0: xi=-1 (left), 1: xi=+1 (right), 2: eta=-1 (bottom), 3: eta=+1 (top)
    /// @return Vector of 6 DOF indices along the edge
    std::vector<int> edge_dofs(int edge_id) const;

    /// Get corner DOF index
    /// @param corner_id 0: (-1,-1), 1: (+1,-1), 2: (-1,+1), 3: (+1,+1)
    int corner_dof(int corner_id) const;

    /// Check if DOF is on boundary (any edge)
    bool is_boundary_dof(int dof) const;

    /// Check if DOF is in interior (not on any edge)
    bool is_interior_dof(int dof) const { return !is_boundary_dof(dof); }

    /// Get interior DOF indices (16 DOFs for quintic)
    std::vector<int> interior_dofs() const;

private:
    VecX nodes_;      ///< 6 LGL nodes in [-1, 1]
    VecX weights_;    ///< LGL quadrature weights
    VecX bary_;       ///< Barycentric weights for interpolation
    MatX D_;          ///< 1D derivative matrix (6 x 6)
    MatX D2_;         ///< 1D second derivative matrix (6 x 6)

    /// Compute LGL nodes and weights
    void compute_nodes_and_weights();

    /// Compute derivative matrices
    void compute_derivative_matrices();

    /// Evaluate 1D basis functions at point xi using barycentric formula
    VecX evaluate_1d(Real xi) const;

    /// Evaluate 1D basis function derivatives at point xi
    VecX evaluate_derivative_1d(Real xi) const;

    /// Evaluate 1D basis function second derivatives at point xi
    VecX evaluate_second_derivative_1d(Real xi) const;
};

}  // namespace drifter
