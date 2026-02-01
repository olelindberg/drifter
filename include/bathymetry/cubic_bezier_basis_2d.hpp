#pragma once

/// @file cubic_bezier_basis_2d.hpp
/// @brief 2D tensor-product cubic Bezier basis for bathymetry surface fitting
///
/// This provides cubic (degree 3) Bezier basis functions on quadrilateral
/// elements for smooth bathymetry representation with C¹ continuity.
/// Uses analytic Bernstein polynomial evaluation.

#include "core/types.hpp"
#include <vector>
#include <array>

namespace drifter {

/// @brief 2D tensor-product cubic Bezier basis on reference element [0,1]^2
///
/// Uses degree 3 Bezier (4 control points per direction), giving 16 DOFs per element.
/// Provides basis function evaluation and derivatives up to 2nd order for thin plate
/// energy and C¹ constraint enforcement at element vertices.
///
/// DOF indexing: dof = i + N1D * j where i is u-direction, j is v-direction
/// Control points are uniformly spaced on [0,1]: u_i = i/3 for i=0..3
///
/// DOF layout:
/// ```
///   j=3: [3]  [7]  [11] [15]   ← top (v=1)
///   j=2: [2]  [6]  [10] [14]
///   j=1: [1]  [5]  [9]  [13]
///   j=0: [0]  [4]  [8]  [12]   ← bottom (v=0)
///        ↑              ↑
///      left           right
///     (u=0)          (u=1)
/// ```
class CubicBezierBasis2D {
public:
    static constexpr int DEGREE = 3;    ///< Polynomial degree
    static constexpr int N1D = 4;       ///< Control points per direction (degree + 1)
    static constexpr int NDOF = 16;     ///< Total DOFs per element (4 x 4)

    /// @brief Construct cubic Bezier basis
    CubicBezierBasis2D();

    // =========================================================================
    // Basic properties
    // =========================================================================

    /// Polynomial degree
    int degree() const { return DEGREE; }

    /// Number of control points in 1D
    int num_nodes_1d() const { return N1D; }

    /// Total number of DOFs per element
    int num_dofs() const { return NDOF; }

    // =========================================================================
    // DOF indexing
    // =========================================================================

    /// Convert (i, j) tensor indices to linear DOF index
    /// @param i Index in u direction (0 to 3)
    /// @param j Index in v direction (0 to 3)
    static int dof_index(int i, int j) { return i + N1D * j; }

    /// Extract (i, j) from linear DOF index
    static void dof_ij(int dof, int& i, int& j) {
        j = dof / N1D;
        i = dof % N1D;
    }

    /// Get control point position for DOF index (uniformly spaced on [0,1]^2)
    /// @param dof DOF index (0 to 15)
    /// @return (u, v) in [0, 1]^2
    Vec2 control_point_position(int dof) const;

    // =========================================================================
    // Basis function evaluation (parameter domain [0,1]^2)
    // =========================================================================

    /// Evaluate all 16 basis functions at point (u, v)
    /// @param u First parameter in [0, 1]
    /// @param v Second parameter in [0, 1]
    /// @return Vector of 16 basis function values
    VecX evaluate(Real u, Real v) const;

    /// Evaluate 1D Bernstein basis B_{i,n}(t) using de Casteljau recurrence
    /// @param n Degree
    /// @param t Parameter in [0, 1]
    /// @return Vector of (n+1) Bernstein basis values
    VecX evaluate_bernstein_1d(int n, Real t) const;

    // =========================================================================
    // First derivatives (for C¹ constraints and gradient penalty)
    // =========================================================================

    /// Evaluate du (partial derivative w.r.t u) of all basis functions
    /// @return Vector of 16 values: d phi_k / du
    VecX evaluate_du(Real u, Real v) const;

    /// Evaluate dv (partial derivative w.r.t v) of all basis functions
    /// @return Vector of 16 values: d phi_k / dv
    VecX evaluate_dv(Real u, Real v) const;

    /// Evaluate gradient of all basis functions at (u, v)
    /// @return Matrix (16 x 2): column 0 = d/du, column 1 = d/dv
    MatX evaluate_gradient(Real u, Real v) const;

    // =========================================================================
    // Second derivatives (for thin plate energy)
    // =========================================================================

    /// Evaluate d^2/du^2 of all basis functions
    VecX evaluate_d2u(Real u, Real v) const;

    /// Evaluate d^2/dv^2 of all basis functions
    VecX evaluate_d2v(Real u, Real v) const;

    /// Evaluate d^2/dudv of all basis functions
    VecX evaluate_d2uv(Real u, Real v) const;

    /// Evaluate all second derivatives at once
    void evaluate_second_derivatives(Real u, Real v,
                                     VecX& d2u, VecX& d2v, VecX& d2uv) const;

    // =========================================================================
    // Mixed partial (for C¹ constraint z_uv)
    // =========================================================================

    /// Evaluate arbitrary mixed partial derivative d^(nu+nv) / du^nu dv^nv
    /// @param u, v Evaluation point in [0,1]^2
    /// @param nu Derivative order in u (0 to 3)
    /// @param nv Derivative order in v (0 to 3)
    /// @return Vector of 16 values
    VecX evaluate_derivative(Real u, Real v, int nu, int nv) const;

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
    static Vec2 param_to_ref(Real u, Real v) {
        return Vec2(2.0 * u - 1.0, 2.0 * v - 1.0);
    }

    // =========================================================================
    // Scalar interpolation
    // =========================================================================

    /// Evaluate scalar field at (u,v) given control point values using de Casteljau
    /// @param coeffs 16 control point values indexed as i + 4*j
    /// @param u, v Parameters in [0, 1]^2
    /// @return Interpolated scalar value
    Real evaluate_scalar(const VecX& coeffs, Real u, Real v) const;

    // =========================================================================
    // Corner and edge access (for constraint building)
    // =========================================================================

    /// Get DOF index for corner
    /// @param corner_id 0: (0,0), 1: (1,0), 2: (0,1), 3: (1,1) in parameter space
    int corner_dof(int corner_id) const;

    /// Get corner ID for a DOF index (inverse of corner_dof)
    /// @param dof DOF index (0, 3, 12, or 15 for corners)
    /// @return Corner ID (0-3), or -1 if not a corner DOF
    int dof_to_corner(int dof) const;

    /// Get DOF indices along an edge (4 DOFs)
    /// @param edge_id 0: u=0 (left), 1: u=1 (right), 2: v=0 (bottom), 3: v=1 (top)
    std::vector<int> edge_dofs(int edge_id) const;

    /// Get (u, v) parameter values at corner
    /// @param corner_id 0: (0,0), 1: (1,0), 2: (0,1), 3: (1,1)
    Vec2 corner_param(int corner_id) const;

    // =========================================================================
    // Non-conforming interface support (Bezier subdivision)
    // =========================================================================

    /// @brief Compute Bezier extraction matrix for a sub-interval
    ///
    /// For 2:1 refinement: [0, 0.5] or [0.5, 1]
    ///
    /// @param t0 Start of sub-interval
    /// @param t1 End of sub-interval
    /// @return (N1D × N1D) matrix S where new_ctrl_pts = S * original_ctrl_pts
    MatX compute_1d_extraction_matrix(Real t0, Real t1) const;

private:
    /// Precomputed binomial coefficients C(n,k) for n <= 2*DEGREE
    std::array<std::array<Real, 2*DEGREE+1>, 2*DEGREE+1> binomial_;

    /// Initialize binomial coefficient table
    void compute_binomial_coefficients();

    /// Get binomial coefficient C(n,k) from precomputed table
    Real binom(int n, int k) const;
};

}  // namespace drifter
