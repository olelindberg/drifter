#pragma once

// 3D Hexahedron basis functions for DG-FEM with staggered nodal grids
// LGL (Legendre-Gauss-Lobatto) for velocity fields - includes boundary nodes
// GL (Gauss-Legendre) for tracer fields - interior nodes only
//
// Adapted from wobbler: Galerkin2DQuadrilateral and LagrangePolynomial1D

#include "core/types.hpp"
#include "dg/face_connection.hpp"
#include <array>
#include <memory>
#include <vector>

namespace drifter {

/// @brief Compute Legendre polynomial and its derivative using recurrence
/// @details Uses the three-term recurrence relation for Legendre polynomials
/// @param n Polynomial degree
/// @param x Evaluation point in [-1, 1]
/// @param L Output: P_n(x)
/// @param dL Output: P'_n(x)
inline void legendre_poly_and_derivative(int n, Real x, Real& L, Real& dL) {
    if (n == 0) {
        L = 1.0;
        dL = 0.0;
    } else if (n == 1) {
        L = x;
        dL = 1.0;
    } else {
        Real Lnm2 = 1.0;
        Real Lnm1 = x;
        Real dLnm2 = 0.0;
        Real dLnm1 = 1.0;

        for (int k = 2; k <= n; ++k) {
            L = (2.0 * k - 1.0) / k * x * Lnm1 - (k - 1.0) / k * Lnm2;
            dL = dLnm2 + (2.0 * k - 1.0) * Lnm1;

            Lnm2 = Lnm1;
            Lnm1 = L;
            dLnm2 = dLnm1;
            dLnm1 = dL;
        }
    }
}

/// @brief Compute Gauss-Legendre (GL) nodes and weights
/// @details Interior nodes only, optimal for integration
/// @param n Number of nodes
/// @param nodes Output: node positions in [-1, 1]
/// @param weights Output: quadrature weights
void compute_gauss_legendre_nodes(int n, VecX& nodes, VecX& weights);

/// @brief Compute Legendre-Gauss-Lobatto (LGL) nodes and weights
/// @details Includes endpoint nodes at -1 and +1
/// @param n Number of nodes (order + 1)
/// @param nodes Output: node positions in [-1, 1]
/// @param weights Output: quadrature weights
void compute_gauss_lobatto_nodes(int n, VecX& nodes, VecX& weights);

/// @brief Compute barycentric weights for Lagrange interpolation
/// @param nodes Node positions
/// @return Barycentric weights
VecX compute_barycentric_weights(const VecX& nodes);

/// @brief Compute 1D derivative matrix for Lagrange interpolation
/// @details D_ij = l'_j(x_i) where l_j is the j-th Lagrange basis function
/// @param nodes Node positions
/// @return Derivative matrix (n x n)
MatX compute_derivative_matrix_1d(const VecX& nodes);

/// @brief Compute 1D interpolation matrix from one grid to another
/// @param from_nodes Source node positions
/// @param to_nodes Target node positions
/// @return Interpolation matrix (n_to x n_from)
MatX compute_interpolation_matrix_1d(const VecX& from_nodes, const VecX& to_nodes);

/// @brief 1D Lagrange basis data for a single direction
/// @details Contains nodes, weights, derivative matrix, and interpolation
struct LagrangeBasis1D {
    int order;            ///< Polynomial order (n_nodes - 1)
    VecX nodes;           ///< Node positions in [-1, 1]
    VecX weights;         ///< Quadrature weights
    VecX bary_weights;    ///< Barycentric weights for interpolation
    MatX D;               ///< Derivative matrix D_ij = l'_j(x_i)

    /// Construct LGL basis
    static LagrangeBasis1D create_lgl(int order);

    /// Construct GL basis
    static LagrangeBasis1D create_gl(int order);

    /// Number of nodes
    int num_nodes() const { return order + 1; }

    /// Evaluate Lagrange basis functions at point xi
    VecX evaluate(Real xi) const;

    /// Evaluate derivative of Lagrange basis functions at point xi
    VecX evaluate_derivative(Real xi) const;
};

/// @brief 3D Hexahedron basis using tensor-product Lagrange polynomials
/// @details Supports staggered grids: LGL for velocity, GL for tracers
class HexahedronBasis {
public:
    /// @brief Construct hexahedron basis
    /// @param order Polynomial order (same in all directions)
    /// @param lgl_for_velocity Use LGL nodes for velocity grid (default true)
    /// @param gl_for_tracers Use GL nodes for tracer grid (default true)
    explicit HexahedronBasis(int order,
                             bool lgl_for_velocity = true,
                             bool gl_for_tracers = true);

    /// Polynomial order
    int order() const { return order_; }

    /// Number of DOFs per element for velocity grid
    int num_dofs_velocity() const { return num_dofs_lgl_; }

    /// Number of DOFs per element for tracer grid
    int num_dofs_tracer() const { return num_dofs_gl_; }

    /// Number of DOFs per face for velocity grid
    int num_face_dofs_velocity() const { return (order_ + 1) * (order_ + 1); }

    /// Number of DOFs per face for tracer grid
    int num_face_dofs_tracer() const { return (order_ + 1) * (order_ + 1); }

    // =========================================================================
    // LGL (velocity) grid operators
    // =========================================================================

    /// Get 1D LGL basis
    const LagrangeBasis1D& lgl_basis_1d() const { return lgl_1d_; }

    /// Get LGL nodes in reference element
    const std::vector<Vec3>& lgl_nodes() const { return lgl_nodes_3d_; }

    /// Differentiation matrix in xi direction (LGL grid)
    const MatX& D_xi_lgl() const { return D_xi_lgl_; }

    /// Differentiation matrix in eta direction (LGL grid)
    const MatX& D_eta_lgl() const { return D_eta_lgl_; }

    /// Differentiation matrix in zeta direction (LGL grid)
    const MatX& D_zeta_lgl() const { return D_zeta_lgl_; }

    /// Mass matrix on LGL grid (diagonal for tensor-product)
    const MatX& mass_lgl() const { return mass_lgl_; }

    /// Inverse mass matrix on LGL grid
    const MatX& mass_inv_lgl() const { return mass_inv_lgl_; }

    // =========================================================================
    // GL (tracer) grid operators
    // =========================================================================

    /// Get 1D GL basis
    const LagrangeBasis1D& gl_basis_1d() const { return gl_1d_; }

    /// Get GL nodes in reference element
    const std::vector<Vec3>& gl_nodes() const { return gl_nodes_3d_; }

    /// Differentiation matrix in xi direction (GL grid)
    const MatX& D_xi_gl() const { return D_xi_gl_; }

    /// Differentiation matrix in eta direction (GL grid)
    const MatX& D_eta_gl() const { return D_eta_gl_; }

    /// Differentiation matrix in zeta direction (GL grid)
    const MatX& D_zeta_gl() const { return D_zeta_gl_; }

    /// Mass matrix on GL grid (diagonal for tensor-product)
    const MatX& mass_gl() const { return mass_gl_; }

    /// Inverse mass matrix on GL grid
    const MatX& mass_inv_gl() const { return mass_inv_gl_; }

    // =========================================================================
    // Grid interpolation operators
    // =========================================================================

    /// Interpolate from LGL to GL grid (for advection: u from LGL, apply to tracer on GL)
    const MatX& lgl_to_gl() const { return lgl_to_gl_; }

    /// Interpolate from GL to LGL grid
    const MatX& gl_to_lgl() const { return gl_to_lgl_; }

    // =========================================================================
    // Face interpolation operators
    // =========================================================================

    /// Interpolation matrix from LGL volume nodes to face quadrature points
    /// @param face_id Face ID (0-5): 0,1=xi faces, 2,3=eta faces, 4,5=zeta faces
    const MatX& interp_to_face_lgl(int face_id) const { return interp_to_face_lgl_[face_id]; }

    /// Interpolation matrix from GL volume nodes to face quadrature points
    /// @param face_id Face ID (0-5)
    const MatX& interp_to_face_gl(int face_id) const { return interp_to_face_gl_[face_id]; }

    /// Get face quadrature nodes in 2D reference coordinates (tangent directions)
    /// @param face_id Face ID (0-5)
    const std::vector<Vec2>& face_quad_nodes(int face_id) const { return face_quad_nodes_[face_id]; }

    /// Get face quadrature weights
    /// @param face_id Face ID (0-5)
    const VecX& face_quad_weights(int face_id) const { return face_quad_weights_[face_id]; }

    // =========================================================================
    // Sub-face interpolation for non-conforming interfaces
    // =========================================================================

    /// Interpolation to a sub-face for non-conforming interfaces
    /// @param face_id Face ID (0-5)
    /// @param subface_idx Sub-face index (0-3 for 2x2, 0-1 for 2x1 or 1x2)
    /// @param conn_type Face connection type
    /// @param use_lgl Use LGL grid (true) or GL grid (false)
    /// @return Interpolation matrix from volume to sub-face quadrature points
    MatX interp_to_subface(int face_id, int subface_idx,
                           FaceConnectionType conn_type,
                           bool use_lgl) const;

    // =========================================================================
    // Evaluation at arbitrary points
    // =========================================================================

    /// Evaluate LGL basis functions at a point in reference coordinates
    /// @param xi Reference coordinates (xi, eta, zeta) in [-1, 1]^3
    /// @return Vector of basis function values (length = num_dofs_velocity)
    VecX evaluate_lgl(const Vec3& xi) const;

    /// Evaluate GL basis functions at a point in reference coordinates
    VecX evaluate_gl(const Vec3& xi) const;

    /// Evaluate gradient of LGL basis functions at a point
    /// @param xi Reference coordinates
    /// @return Matrix (num_dofs x 3) with columns [dN/dxi, dN/deta, dN/dzeta]
    MatX evaluate_gradient_lgl(const Vec3& xi) const;

    /// Evaluate gradient of GL basis functions at a point
    MatX evaluate_gradient_gl(const Vec3& xi) const;

    // =========================================================================
    // DOF indexing
    // =========================================================================

    /// Convert (i, j, k) indices to linear DOF index
    /// @param i Index in xi direction (0 to order)
    /// @param j Index in eta direction (0 to order)
    /// @param k Index in zeta direction (0 to order)
    static int dof_index(int i, int j, int k, int order) {
        return i + (order + 1) * (j + (order + 1) * k);
    }

    /// Extract (i, j, k) indices from linear DOF index
    static void dof_indices(int dof, int order, int& i, int& j, int& k) {
        int np = order + 1;
        k = dof / (np * np);
        int rem = dof % (np * np);
        j = rem / np;
        i = rem % np;
    }

private:
    int order_;
    int num_dofs_lgl_;
    int num_dofs_gl_;

    // 1D basis data
    LagrangeBasis1D lgl_1d_;
    LagrangeBasis1D gl_1d_;

    // 3D node positions
    std::vector<Vec3> lgl_nodes_3d_;
    std::vector<Vec3> gl_nodes_3d_;

    // 3D differentiation matrices (tensor product structure)
    MatX D_xi_lgl_, D_eta_lgl_, D_zeta_lgl_;
    MatX D_xi_gl_, D_eta_gl_, D_zeta_gl_;

    // Mass matrices (diagonal for tensor-product basis)
    MatX mass_lgl_, mass_inv_lgl_;
    MatX mass_gl_, mass_inv_gl_;

    // Grid interpolation
    MatX lgl_to_gl_;
    MatX gl_to_lgl_;

    // Face interpolation (for each of 6 faces)
    std::array<MatX, 6> interp_to_face_lgl_;
    std::array<MatX, 6> interp_to_face_gl_;
    std::array<std::vector<Vec2>, 6> face_quad_nodes_;
    std::array<VecX, 6> face_quad_weights_;

    // Build operators
    void build_1d_operators();
    void build_3d_nodes();
    void build_3d_differentiation_matrices();
    void build_mass_matrices();
    void build_grid_interpolation();
    void build_face_interpolation();
};

// =============================================================================
// Inline implementations
// =============================================================================

inline VecX LagrangeBasis1D::evaluate(Real xi) const {
    int n = num_nodes();
    VecX phi(n);

    // Use barycentric formula for stable evaluation
    // phi_j(xi) = (w_j / (xi - x_j)) / sum_k(w_k / (xi - x_k))

    VecX temp(n);
    Real sum = 0.0;
    bool at_node = false;
    int node_idx = -1;

    for (int j = 0; j < n; ++j) {
        Real diff = xi - nodes(j);
        if (std::abs(diff) < 1e-15) {
            at_node = true;
            node_idx = j;
            break;
        }
        temp(j) = bary_weights(j) / diff;
        sum += temp(j);
    }

    if (at_node) {
        phi.setZero();
        phi(node_idx) = 1.0;
    } else {
        for (int j = 0; j < n; ++j) {
            phi(j) = temp(j) / sum;
        }
    }

    return phi;
}

inline VecX LagrangeBasis1D::evaluate_derivative(Real xi) const {
    int n = num_nodes();
    VecX dphi(n);

    // First evaluate the basis functions
    VecX phi = evaluate(xi);

    // Then compute derivatives using the formula:
    // l'_j(x) = l_j(x) * sum_{k != j} 1/(x - x_k) - l_j(x) * sum_{k} 1/(x - x_k)
    // Or use the derivative matrix: dphi = D * phi (at nodes)

    // For arbitrary points, use:
    // l'_j(x) = (w_j / (x - x_j)) * (sum_k l_k(x)/(x - x_k) - l_j(x)/(x - x_j)) / sum

    bool at_node = false;
    int node_idx = -1;

    for (int j = 0; j < n; ++j) {
        if (std::abs(xi - nodes(j)) < 1e-15) {
            at_node = true;
            node_idx = j;
            break;
        }
    }

    if (at_node) {
        // Use the derivative matrix row
        dphi = D.row(node_idx).transpose();
    } else {
        // General formula for arbitrary points
        Real sum_phi_over_diff = 0.0;
        for (int k = 0; k < n; ++k) {
            sum_phi_over_diff += phi(k) / (xi - nodes(k));
        }

        for (int j = 0; j < n; ++j) {
            Real diff = xi - nodes(j);
            dphi(j) = phi(j) * (sum_phi_over_diff - 1.0 / diff) / diff * (-1.0);
            // Correct formula: l'_j(x) = l_j(x) * sum_{k!=j} 1/(x - x_k)
            Real inner_sum = 0.0;
            for (int k = 0; k < n; ++k) {
                if (k != j) {
                    inner_sum += 1.0 / (xi - nodes(k));
                }
            }
            dphi(j) = phi(j) * inner_sum;
        }
    }

    return dphi;
}

inline VecX HexahedronBasis::evaluate_lgl(const Vec3& xi) const {
    VecX phi_xi = lgl_1d_.evaluate(xi(0));
    VecX phi_eta = lgl_1d_.evaluate(xi(1));
    VecX phi_zeta = lgl_1d_.evaluate(xi(2));

    int np = order_ + 1;
    VecX phi(num_dofs_lgl_);

    for (int k = 0; k < np; ++k) {
        for (int j = 0; j < np; ++j) {
            for (int i = 0; i < np; ++i) {
                int idx = dof_index(i, j, k, order_);
                phi(idx) = phi_xi(i) * phi_eta(j) * phi_zeta(k);
            }
        }
    }

    return phi;
}

inline VecX HexahedronBasis::evaluate_gl(const Vec3& xi) const {
    VecX phi_xi = gl_1d_.evaluate(xi(0));
    VecX phi_eta = gl_1d_.evaluate(xi(1));
    VecX phi_zeta = gl_1d_.evaluate(xi(2));

    int np = order_ + 1;
    VecX phi(num_dofs_gl_);

    for (int k = 0; k < np; ++k) {
        for (int j = 0; j < np; ++j) {
            for (int i = 0; i < np; ++i) {
                int idx = dof_index(i, j, k, order_);
                phi(idx) = phi_xi(i) * phi_eta(j) * phi_zeta(k);
            }
        }
    }

    return phi;
}

inline MatX HexahedronBasis::evaluate_gradient_lgl(const Vec3& xi) const {
    VecX phi_xi = lgl_1d_.evaluate(xi(0));
    VecX phi_eta = lgl_1d_.evaluate(xi(1));
    VecX phi_zeta = lgl_1d_.evaluate(xi(2));

    VecX dphi_xi = lgl_1d_.evaluate_derivative(xi(0));
    VecX dphi_eta = lgl_1d_.evaluate_derivative(xi(1));
    VecX dphi_zeta = lgl_1d_.evaluate_derivative(xi(2));

    int np = order_ + 1;
    MatX grad(num_dofs_lgl_, 3);

    for (int k = 0; k < np; ++k) {
        for (int j = 0; j < np; ++j) {
            for (int i = 0; i < np; ++i) {
                int idx = dof_index(i, j, k, order_);
                grad(idx, 0) = dphi_xi(i) * phi_eta(j) * phi_zeta(k);
                grad(idx, 1) = phi_xi(i) * dphi_eta(j) * phi_zeta(k);
                grad(idx, 2) = phi_xi(i) * phi_eta(j) * dphi_zeta(k);
            }
        }
    }

    return grad;
}

inline MatX HexahedronBasis::evaluate_gradient_gl(const Vec3& xi) const {
    VecX phi_xi = gl_1d_.evaluate(xi(0));
    VecX phi_eta = gl_1d_.evaluate(xi(1));
    VecX phi_zeta = gl_1d_.evaluate(xi(2));

    VecX dphi_xi = gl_1d_.evaluate_derivative(xi(0));
    VecX dphi_eta = gl_1d_.evaluate_derivative(xi(1));
    VecX dphi_zeta = gl_1d_.evaluate_derivative(xi(2));

    int np = order_ + 1;
    MatX grad(num_dofs_gl_, 3);

    for (int k = 0; k < np; ++k) {
        for (int j = 0; j < np; ++j) {
            for (int i = 0; i < np; ++i) {
                int idx = dof_index(i, j, k, order_);
                grad(idx, 0) = dphi_xi(i) * phi_eta(j) * phi_zeta(k);
                grad(idx, 1) = phi_xi(i) * dphi_eta(j) * phi_zeta(k);
                grad(idx, 2) = phi_xi(i) * phi_eta(j) * dphi_zeta(k);
            }
        }
    }

    return grad;
}

}  // namespace drifter
