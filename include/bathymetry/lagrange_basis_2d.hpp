#pragma once

/// @file lagrange_basis_2d.hpp
/// @brief Flexible-order 2D tensor-product Lagrange basis for CG bathymetry
/// smoothing
///
/// This provides Lagrange basis functions on quadrilateral elements with
/// configurable polynomial order for the continuous Galerkin formulation of the
/// biharmonic smoothing. Uses LGL (Legendre-Gauss-Lobatto) nodes for optimal
/// interpolation properties.
///
/// Default order is 3 (bicubic), which is the minimum for biharmonic on
/// rectangles.

#include "core/types.hpp"
#include "dg/basis_hexahedron.hpp"
#include <vector>

namespace drifter {

/// @brief 2D tensor-product Lagrange basis on reference element [-1,1]^2
///
/// Uses LGL nodes for the specified order, giving (order+1)^2 DOFs per element.
/// Provides basis function evaluation, gradients, Laplacian, and Hessians
/// needed for biharmonic operator assembly.
///
/// DOF indexing: dof = i + n1d * j where i is xi-direction, j is eta-direction
class LagrangeBasis2D {
public:
    /// @brief Construct Lagrange basis with specified order
    /// @param order Polynomial order (default 3 = bicubic, minimum for
    /// biharmonic)
    explicit LagrangeBasis2D(int order = 3);

    // =========================================================================
    // Order and size queries
    // =========================================================================

    /// Polynomial order
    int order() const { return order_; }

    /// Number of nodes in 1D (order + 1)
    int num_nodes_1d() const { return n1d_; }

    /// Total number of DOFs per element ((order+1)^2)
    int num_dofs() const { return ndof_; }

    // =========================================================================
    // Node access
    // =========================================================================

    /// 1D LGL nodes in [-1, 1]
    const VecX &nodes_1d() const { return basis_1d_.nodes; }

    /// 1D quadrature weights
    const VecX &weights_1d() const { return basis_1d_.weights; }

    /// Get 2D node position for DOF index in reference coordinates
    /// @param dof DOF index (0 to ndof-1)
    /// @return (xi, eta) in [-1, 1]^2
    Vec2 node_position(int dof) const;

    // =========================================================================
    // DOF indexing
    // =========================================================================

    /// Convert (i, j) tensor indices to linear DOF index
    /// @param i Index in xi direction (0 to order)
    /// @param j Index in eta direction (0 to order)
    int dof_index(int i, int j) const { return i + n1d_ * j; }

    /// Extract (i, j) from linear DOF index
    void dof_ij(int dof, int &i, int &j) const {
        j = dof / n1d_;
        i = dof % n1d_;
    }

    // =========================================================================
    // Basis function evaluation
    // =========================================================================

    /// Evaluate all basis functions at point (xi, eta)
    /// @param xi First reference coordinate in [-1, 1]
    /// @param eta Second reference coordinate in [-1, 1]
    /// @return Vector of ndof basis function values
    VecX evaluate(Real xi, Real eta) const;

    /// Evaluate gradient of all basis functions at (xi, eta)
    /// @return Matrix (ndof x 2): column 0 = d/dxi, column 1 = d/deta
    MatX evaluate_gradient(Real xi, Real eta) const;

    /// Evaluate Laplacian of all basis functions at (xi, eta)
    /// @return Vector of ndof values: d^2/dxi^2 + d^2/deta^2
    VecX evaluate_laplacian(Real xi, Real eta) const;

    /// Evaluate second derivatives of all basis functions at (xi, eta)
    /// @param d2_dxi2 Output: d^2 phi / dxi^2 (ndof values)
    /// @param d2_deta2 Output: d^2 phi / deta^2 (ndof values)
    /// @param d2_dxideta Output: d^2 phi / dxi deta (ndof values)
    void evaluate_second_derivatives(Real xi, Real eta, VecX &d2_dxi2, VecX &d2_deta2,
                                     VecX &d2_dxideta) const;

    /// Evaluate Hessian matrices of all basis functions at (xi, eta)
    /// @return Vector of ndof 2x2 matrices: H[dof](i,j) = d^2 phi_dof / dx_i
    /// dx_j
    std::vector<Mat2> evaluate_hessian(Real xi, Real eta) const;

    // =========================================================================
    // Pre-computed derivative matrices at nodes
    // =========================================================================

    /// 1D first derivative matrix: D[i,j] = d(phi_j)/dxi at node i
    const MatX &derivative_matrix_1d() const { return basis_1d_.D; }

    /// 1D second derivative matrix: D2[i,j] = d^2(phi_j)/dxi^2 at node i
    const MatX &second_derivative_matrix_1d() const { return D2_; }

    // =========================================================================
    // Quadrature (for integration)
    // =========================================================================

    /// 2D quadrature weights for tensor-product rule
    /// Weight for DOF (i,j) = weights_1d(i) * weights_1d(j)
    Real quadrature_weight(int i, int j) const {
        return basis_1d_.weights(i) * basis_1d_.weights(j);
    }

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
    /// @param edge_id 0: xi=-1 (left), 1: xi=+1 (right), 2: eta=-1 (bottom), 3:
    /// eta=+1 (top)
    /// @return Vector of n1d DOF indices along the edge
    std::vector<int> edge_dofs(int edge_id) const;

    /// Get corner DOF index
    /// @param corner_id 0: (-1,-1), 1: (+1,-1), 2: (-1,+1), 3: (+1,+1)
    int corner_dof(int corner_id) const;

    /// Check if DOF is on boundary (any edge)
    bool is_boundary_dof(int dof) const;

    /// Check if DOF is in interior (not on any edge)
    bool is_interior_dof(int dof) const { return !is_boundary_dof(dof); }

    /// Get interior DOF indices
    std::vector<int> interior_dofs() const;

    // =========================================================================
    // Edge evaluation for IPDG
    // =========================================================================

    /// Evaluate normal derivative of all basis functions at a point on an edge
    /// @param edge_id 0: xi=-1, 1: xi=+1, 2: eta=-1, 3: eta=+1
    /// @param t Parameter along edge in [-1, 1]
    /// @return Vector of ndof normal derivative values (outward normal)
    VecX evaluate_normal_derivative_at_edge(int edge_id, Real t) const;

private:
    int order_; ///< Polynomial order
    int n1d_; ///< Nodes per direction (order + 1)
    int ndof_; ///< Total DOFs per element (n1d^2)
    LagrangeBasis1D basis_1d_; ///< 1D LGL basis
    MatX D2_; ///< 1D second derivative matrix

    /// Evaluate 1D second derivative at arbitrary point
    VecX evaluate_second_derivative_1d(Real xi) const;
};

} // namespace drifter
