#pragma once

/// @file bezier_basis_2d.hpp
/// @brief 2D tensor-product Bezier basis for bathymetry surface fitting
///
/// This provides quintic (degree 5) Bezier basis functions on quadrilateral
/// elements for smooth bathymetry representation with C^2 continuity.
/// Uses analytic Bernstein polynomial evaluation without OpenCASCADE dependency.

#include "core/types.hpp"
#include <vector>
#include <array>

namespace drifter {

/// @brief 2D tensor-product Bezier basis on reference element [0,1]^2
///
/// Uses degree 5 Bezier (6 control points per direction), giving 36 DOFs per element.
/// Provides basis function evaluation and derivatives up to 4th order for C^2
/// constraint enforcement at element vertices.
///
/// DOF indexing: dof = i + N1D * j where i is u-direction, j is v-direction
/// Control points are uniformly spaced on [0,1]: u_i = i/5 for i=0..5
class BezierBasis2D {
public:
    static constexpr int DEGREE = 5;    ///< Polynomial degree
    static constexpr int N1D = 6;       ///< Control points per direction (degree + 1)
    static constexpr int NDOF = 36;     ///< Total DOFs per element (6 x 6)

    /// @brief Construct quintic Bezier basis
    BezierBasis2D();

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
    /// @param i Index in u direction (0 to 5)
    /// @param j Index in v direction (0 to 5)
    static int dof_index(int i, int j) { return i + N1D * j; }

    /// Extract (i, j) from linear DOF index
    static void dof_ij(int dof, int& i, int& j) {
        j = dof / N1D;
        i = dof % N1D;
    }

    /// Get control point position for DOF index (uniformly spaced on [0,1]^2)
    /// @param dof DOF index (0 to 35)
    /// @return (u, v) in [0, 1]^2
    Vec2 control_point_position(int dof) const;

    // =========================================================================
    // Basis function evaluation (parameter domain [0,1]^2)
    // =========================================================================

    /// Evaluate all 36 basis functions at point (u, v)
    /// @param u First parameter in [0, 1]
    /// @param v Second parameter in [0, 1]
    /// @return Vector of 36 basis function values
    VecX evaluate(Real u, Real v) const;

    /// Evaluate 1D Bernstein basis B_{i,n}(t) using de Casteljau recurrence
    /// @param n Degree
    /// @param t Parameter in [0, 1]
    /// @return Vector of (n+1) Bernstein basis values
    VecX evaluate_bernstein_1d(int n, Real t) const;

    // =========================================================================
    // First derivatives
    // =========================================================================

    /// Evaluate du (partial derivative w.r.t u) of all basis functions
    /// @return Vector of 36 values: d phi_k / du
    VecX evaluate_du(Real u, Real v) const;

    /// Evaluate dv (partial derivative w.r.t v) of all basis functions
    /// @return Vector of 36 values: d phi_k / dv
    VecX evaluate_dv(Real u, Real v) const;

    /// Evaluate gradient of all basis functions at (u, v)
    /// @return Matrix (36 x 2): column 0 = d/du, column 1 = d/dv
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
    // Higher derivatives (for C^2 constraints at vertices)
    // =========================================================================

    /// Evaluate arbitrary mixed partial derivative d^(nu+nv) / du^nu dv^nv
    /// @param u, v Evaluation point in [0,1]^2
    /// @param nu Derivative order in u
    /// @param nv Derivative order in v
    /// @return Vector of 36 values
    VecX evaluate_derivative(Real u, Real v, int nu, int nv) const;

    /// Third derivatives needed for C^2 constraints
    VecX evaluate_d3uuv(Real u, Real v) const;   ///< d^3/du^2 dv
    VecX evaluate_d3uvv(Real u, Real v) const;   ///< d^3/du dv^2

    /// Fourth derivative d^4/du^2 dv^2 (needed for full C^2 constraints)
    VecX evaluate_d4uuvv(Real u, Real v) const;

    // =========================================================================
    // 1D Bernstein derivative helpers
    // =========================================================================

    /// Evaluate k-th derivative of 1D Bernstein basis of degree n
    /// Uses the relation: d^k B_{i,n} / dt^k = n!/(n-k)! * sum of lower degree Bernstein
    /// @param n Degree
    /// @param t Parameter in [0, 1]
    /// @param k Derivative order
    /// @return Vector of (n+1) derivative values
    VecX evaluate_bernstein_derivative_1d(int n, Real t, int k) const;

    // =========================================================================
    // Conversion utilities
    // =========================================================================

    /// Get matrix to convert from reference [-1,1]^2 to parameter [0,1]^2
    /// u = (xi + 1) / 2, v = (eta + 1) / 2
    static Vec2 ref_to_param(Real xi, Real eta) {
        return Vec2((xi + 1.0) / 2.0, (eta + 1.0) / 2.0);
    }

    /// Get matrix to convert from parameter [0,1]^2 to reference [-1,1]^2
    static Vec2 param_to_ref(Real u, Real v) {
        return Vec2(2.0 * u - 1.0, 2.0 * v - 1.0);
    }

    // =========================================================================
    // Scalar interpolation
    // =========================================================================

    /// Evaluate scalar field at (u,v) given control point values using de Casteljau
    /// This is numerically stable and efficient for single-point evaluation
    /// @param coeffs 36 control point values indexed as i + 6*j
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
    /// @param dof DOF index (0, 5, 30, or 35 for corners)
    /// @return Corner ID (0-3), or -1 if not a corner DOF
    int dof_to_corner(int dof) const;

    /// Get DOF indices along an edge (6 DOFs)
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
    /// For a degree-n Bezier curve on [0,1], this computes the (n+1)×(n+1) matrix S
    /// such that new_ctrl_pts = S * original_ctrl_pts gives the control points
    /// for the Bezier curve restricted to [t0, t1].
    ///
    /// Uses the de Casteljau subdivision algorithm. For the common 2:1 refinement case:
    /// - [0, 0.5]: Left half subdivision (binomial coefficients / 2^k)
    /// - [0.5, 1]: Right half subdivision
    ///
    /// @param t0 Start of sub-interval (0 ≤ t0 < t1)
    /// @param t1 End of sub-interval (t0 < t1 ≤ 1)
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
