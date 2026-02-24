#pragma once

/// @file linear_bezier_basis_2d.hpp
/// @brief 2D tensor-product linear Bezier basis for bathymetry surface fitting
///
/// This provides linear (degree 1) Bezier basis functions on quadrilateral
/// elements for bathymetry representation with C0 continuity.
/// Uses analytic Bernstein polynomial evaluation.

#include "bathymetry/bezier_basis_2d_base.hpp"
#include "core/types.hpp"
#include <array>
#include <vector>

namespace drifter {

/// @brief 2D tensor-product linear Bezier basis on reference element [0,1]^2
///
/// Uses degree 1 Bezier (2 control points per direction), giving 4 DOFs per
/// element. Provides basis function evaluation and first derivatives for
/// Dirichlet energy (Laplacian smoothing).
///
/// DOF indexing: dof = i + N1D * j where i is u-direction, j is v-direction
/// Control points are at corners: u_i = i for i=0,1
///
/// DOF layout:
/// ```
///   j=1: [1]──[3]   ← top (v=1)
///         │    │
///   j=0: [0]──[2]   ← bottom (v=0)
///        ↑    ↑
///      left  right
///     (u=0) (u=1)
/// ```
class LinearBezierBasis2D : public BezierBasis2DBase {
public:
    static constexpr int DEGREE = 1; ///< Polynomial degree
    static constexpr int N1D = 2; ///< Control points per direction (degree + 1)
    static constexpr int NDOF = 4; ///< Total DOFs per element (2 x 2)

    /// @brief Construct linear Bezier basis
    LinearBezierBasis2D() = default;

    // =========================================================================
    // Basic properties
    // =========================================================================

    /// Polynomial degree
    int degree() const override { return DEGREE; }

    /// Number of control points in 1D
    int num_nodes_1d() const override { return N1D; }

    /// Total number of DOFs per element
    int num_dofs() const override { return NDOF; }

    // =========================================================================
    // DOF indexing
    // =========================================================================

    /// Convert (i, j) tensor indices to linear DOF index
    /// @param i Index in u direction (0 or 1)
    /// @param j Index in v direction (0 or 1)
    /// Layout: DOF = j + N1D * i to match diagram [0,1] bottom, [2,3] top with u
    /// varying
    static int dof_index(int i, int j) { return j + N1D * i; }

    /// Extract (i, j) from linear DOF index
    /// Layout: DOF = j + N1D * i, so i = dof / N1D, j = dof % N1D
    static void dof_ij(int dof, int &i, int &j) {
        i = dof / N1D;
        j = dof % N1D;
    }

    /// Get control point position for DOF index (at corners of [0,1]^2)
    /// @param dof DOF index (0 to 3)
    /// @return (u, v) in [0, 1]^2
    Vec2 control_point_position(int dof) const override;

    // =========================================================================
    // Basis function evaluation (parameter domain [0,1]^2)
    // =========================================================================

    /// Evaluate all 4 basis functions at point (u, v)
    /// @param u First parameter in [0, 1]
    /// @param v Second parameter in [0, 1]
    /// @return Vector of 4 basis function values
    VecX evaluate(Real u, Real v) const override;

    /// Evaluate 1D Bernstein basis B_{i,n}(t)
    /// @param n Degree
    /// @param t Parameter in [0, 1]
    /// @return Vector of (n+1) Bernstein basis values
    VecX evaluate_bernstein_1d(int n, Real t) const;

    // =========================================================================
    // First derivatives (for Dirichlet energy)
    // =========================================================================

    /// Evaluate du (partial derivative w.r.t u) of all basis functions
    /// @return Vector of 4 values: d phi_k / du
    VecX evaluate_du(Real u, Real v) const override;

    /// Evaluate dv (partial derivative w.r.t v) of all basis functions
    /// @return Vector of 4 values: d phi_k / dv
    VecX evaluate_dv(Real u, Real v) const override;

    /// Evaluate gradient of all basis functions at (u, v)
    /// @return Matrix (4 x 2): column 0 = d/du, column 1 = d/dv
    MatX evaluate_gradient(Real u, Real v) const override;

    // =========================================================================
    // 1D Bernstein derivative helpers
    // =========================================================================

    /// Evaluate k-th derivative of 1D Bernstein basis of degree n
    /// @param n Degree
    /// @param t Parameter in [0, 1]
    /// @param k Derivative order
    /// @return Vector of (n+1) derivative values
    VecX evaluate_bernstein_derivative_1d(int n, Real t, int k) const;

    // =========================================================================
    // Conversion utilities
    // =========================================================================

    /// Convert from reference [-1,1]^2 to parameter [0,1]^2
    static Vec2 ref_to_param(Real xi, Real eta) {
        return Vec2((xi + 1.0) / 2.0, (eta + 1.0) / 2.0);
    }

    /// Convert from parameter [0,1]^2 to reference [-1,1]^2
    static Vec2 param_to_ref(Real u, Real v) { return Vec2(2.0 * u - 1.0, 2.0 * v - 1.0); }

    // =========================================================================
    // Scalar interpolation
    // =========================================================================

    /// Evaluate scalar field at (u,v) given control point values using bilinear
    /// interpolation
    /// @param coeffs 4 control point values indexed as i + 2*j
    /// @param u, v Parameters in [0, 1]^2
    /// @return Interpolated scalar value
    Real evaluate_scalar(const VecX &coeffs, Real u, Real v) const override;

    // =========================================================================
    // Corner and edge access (for constraint building)
    // =========================================================================

    /// Get DOF index for corner
    /// @param corner_id 0: (0,0), 1: (1,0), 2: (0,1), 3: (1,1) in parameter space
    int corner_dof(int corner_id) const override;

    /// Get corner ID for a DOF index (inverse of corner_dof)
    /// @param dof DOF index (0, 1, 2, or 3)
    /// @return Corner ID (0-3), or -1 if not valid
    int dof_to_corner(int dof) const override;

    /// Get DOF indices along an edge (2 DOFs - both are corners)
    /// @param edge_id 0: u=0 (left), 1: u=1 (right), 2: v=0 (bottom), 3: v=1
    /// (top)
    std::vector<int> edge_dofs(int edge_id) const override;

    /// Get (u, v) parameter values at corner
    /// @param corner_id 0: (0,0), 1: (1,0), 2: (0,1), 3: (1,1)
    Vec2 corner_param(int corner_id) const override;

    // =========================================================================
    // Non-conforming interface support (Bezier subdivision)
    // =========================================================================

    /// @brief Compute Bezier extraction matrix for a sub-interval
    ///
    /// For 2:1 refinement: [0, 0.5] or [0.5, 1]
    ///
    /// @param t0 Start of sub-interval
    /// @param t1 End of sub-interval
    /// @return (N1D x N1D) matrix S where new_ctrl_pts = S * original_ctrl_pts
    MatX compute_1d_extraction_matrix(Real t0, Real t1) const;
};

} // namespace drifter
