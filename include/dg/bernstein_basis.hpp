#pragma once

// Bernstein polynomial basis for bounded interpolation
//
// Bernstein polynomials have the convex hull property: the interpolated
// value at any point lies within the convex hull of the control points.
// This guarantees that interpolation is bounded by min/max of the data.
//
// For seabed interpolation, this prevents spurious oscillations and
// ensures depth values stay within physically meaningful bounds.
//
// Usage:
//   BernsteinBasis1D basis(order);
//   VecX phi = basis.evaluate(xi);  // Basis functions at xi in [-1,1]
//
// Conversion from Lagrange to Bernstein:
//   VecX bernstein_coeffs = lagrange_to_bernstein(order) * lagrange_coeffs;

#include "core/types.hpp"
#include <memory>
#include <vector>

namespace drifter {

/// @brief Compute binomial coefficient C(n, k)
inline Real binomial(int n, int k) {
    if (k < 0 || k > n) return 0.0;
    if (k == 0 || k == n) return 1.0;

    // Use symmetry: C(n,k) = C(n, n-k)
    if (k > n - k) k = n - k;

    Real result = 1.0;
    for (int i = 0; i < k; ++i) {
        result *= static_cast<Real>(n - i) / static_cast<Real>(i + 1);
    }
    return result;
}

/// @brief 1D Bernstein polynomial basis
///
/// Bernstein polynomials of degree n on [0,1]:
///   B_{i,n}(t) = C(n,i) * t^i * (1-t)^(n-i)
///
/// On reference interval [-1,1], we remap: t = (xi + 1) / 2
///
/// Key property (convex hull): For any t in [0,1]:
///   sum_i B_{i,n}(t) = 1  (partition of unity)
///   B_{i,n}(t) >= 0       (non-negativity)
///
/// Therefore: min(c_i) <= sum_i c_i * B_{i,n}(t) <= max(c_i)
class BernsteinBasis1D {
public:
    /// @brief Construct Bernstein basis of given order
    /// @param order Polynomial order (degree n, so n+1 basis functions)
    explicit BernsteinBasis1D(int order);

    /// @brief Polynomial order (degree)
    int order() const { return order_; }

    /// @brief Number of basis functions (order + 1)
    int num_modes() const { return order_ + 1; }

    /// @brief Evaluate all Bernstein basis functions at xi in [-1, 1]
    /// @param xi Evaluation point in reference coordinates [-1, 1]
    /// @return Vector of basis function values B_i(xi)
    VecX evaluate(Real xi) const;

    /// @brief Evaluate derivatives of Bernstein basis functions
    /// @param xi Evaluation point in reference coordinates [-1, 1]
    /// @return Vector of derivative values dB_i/dxi
    VecX evaluate_derivative(Real xi) const;

    /// @brief Get conversion matrix from Lagrange (at LGL nodes) to Bernstein coefficients
    /// @return Matrix M such that bernstein_coeffs = M * lagrange_coeffs
    const MatX& lagrange_to_bernstein_matrix() const { return L2B_; }

    /// @brief Get conversion matrix from Bernstein to Lagrange coefficients
    /// @return Matrix M such that lagrange_coeffs = M * bernstein_coeffs
    const MatX& bernstein_to_lagrange_matrix() const { return B2L_; }

private:
    int order_;
    MatX L2B_;  // Lagrange to Bernstein conversion
    MatX B2L_;  // Bernstein to Lagrange conversion

    void build_conversion_matrices();
};

/// @brief 3D tensor-product Bernstein basis for hexahedral elements
class BernsteinBasis3D {
public:
    /// @brief Construct 3D Bernstein basis
    /// @param order Polynomial order (same in all directions)
    explicit BernsteinBasis3D(int order);

    /// @brief Polynomial order
    int order() const { return order_; }

    /// @brief Number of DOFs ((order+1)^3)
    int num_dofs() const { return num_dofs_; }

    /// @brief Evaluate all 3D Bernstein basis functions at a point
    /// @param xi Reference coordinates (xi, eta, zeta) in [-1, 1]^3
    /// @return Vector of basis function values
    VecX evaluate(const Vec3& xi) const;

    /// @brief Evaluate on bottom face (zeta = -1), returning 2D basis values
    /// @param xi, eta Reference coordinates in [-1, 1]^2
    /// @return Vector of basis function values for 3D DOFs evaluated at zeta=-1
    VecX evaluate_bottom_face(Real xi, Real eta) const;

    /// @brief Get conversion matrix from 3D Lagrange to Bernstein
    const MatX& lagrange_to_bernstein_matrix() const { return L2B_3d_; }

    /// @brief Access 1D basis
    const BernsteinBasis1D& basis_1d() const { return basis_1d_; }

private:
    int order_;
    int num_dofs_;
    BernsteinBasis1D basis_1d_;
    MatX L2B_3d_;  // 3D Lagrange to Bernstein conversion

    void build_3d_conversion_matrix();
};

/// @brief Interpolation method for seabed surface
enum class SeabedInterpolation {
    Lagrange,   // Standard Lagrange interpolation (may overshoot)
    Bernstein   // Bernstein basis with convex hull property (bounded)
};

/// @brief Seabed interpolator with selectable basis
///
/// Provides interpolation of seabed data stored at element DOFs.
/// Supports both Lagrange (standard) and Bernstein (bounded) interpolation.
class SeabedInterpolator {
public:
    /// @brief Construct interpolator
    /// @param order Polynomial order
    /// @param method Interpolation method (Lagrange or Bernstein)
    SeabedInterpolator(int order, SeabedInterpolation method = SeabedInterpolation::Bernstein);

    /// @brief Interpolate 3D coordinates on bottom face
    /// @param coords Interleaved coordinates [x0,y0,z0, x1,y1,z1, ...] at DOFs
    /// @param xi, eta Reference coordinates on bottom face [-1,1]^2
    /// @return Interpolated (x, y, z) point
    Vec3 evaluate_point(const VecX& coords, Real xi, Real eta) const;

    /// @brief Interpolate scalar field on bottom face
    /// @param data Scalar values at DOFs
    /// @param xi, eta Reference coordinates on bottom face [-1,1]^2
    /// @return Interpolated scalar value
    Real evaluate_scalar(const VecX& data, Real xi, Real eta) const;

    /// @brief Evaluate 2D scalar field at a point (only bottom face, no z-dependence)
    /// @param data_2d Scalar values at 2D DOFs (n1d x n1d)
    /// @param xi, eta Reference coordinates [-1,1]^2
    /// @return Interpolated scalar value
    Real evaluate_scalar_2d(const VecX& data_2d, Real xi, Real eta) const;

    /// @brief Convert Lagrange DOF values to Bernstein control points
    /// @param lagrange_data Data at Lagrange DOF nodes
    /// @return Bernstein control point values
    VecX to_bernstein(const VecX& lagrange_data) const;

    /// @brief Get the interpolation method
    SeabedInterpolation method() const { return method_; }

    /// @brief Get polynomial order
    int order() const { return order_; }

    /// @brief Get 1D LGL nodes
    const VecX& lgl_nodes() const { return lgl_nodes_; }

private:
    int order_;
    SeabedInterpolation method_;

    // Lagrange basis data (for Lagrange mode)
    VecX lgl_nodes_;
    VecX lgl_weights_;

    // Bernstein basis (for Bernstein mode)
    std::unique_ptr<BernsteinBasis3D> bernstein_basis_;

    // Cached conversion matrix
    MatX L2B_3d_;

    // Evaluation helpers
    VecX evaluate_lagrange_1d(Real xi) const;
    VecX evaluate_lagrange_bottom_face(Real xi, Real eta) const;
};


// =============================================================================
// Inline implementations
// =============================================================================

inline VecX BernsteinBasis1D::evaluate(Real xi) const {
    // Remap from [-1, 1] to [0, 1]
    Real t = 0.5 * (xi + 1.0);

    int n = order_;
    VecX B(n + 1);

    // Use de Casteljau-like recurrence for stability
    // B_{i,n}(t) = C(n,i) * t^i * (1-t)^(n-i)

    Real one_minus_t = 1.0 - t;

    // Start with B_{0,0} = 1
    std::vector<Real> prev(1, 1.0);

    for (int degree = 1; degree <= n; ++degree) {
        std::vector<Real> curr(degree + 1);

        // B_{0,d} = (1-t) * B_{0,d-1}
        curr[0] = one_minus_t * prev[0];

        // B_{i,d} = t * B_{i-1,d-1} + (1-t) * B_{i,d-1}
        for (int i = 1; i < degree; ++i) {
            curr[i] = t * prev[i-1] + one_minus_t * prev[i];
        }

        // B_{d,d} = t * B_{d-1,d-1}
        curr[degree] = t * prev[degree - 1];

        prev = std::move(curr);
    }

    for (int i = 0; i <= n; ++i) {
        B(i) = prev[i];
    }

    return B;
}

inline VecX BernsteinBasis1D::evaluate_derivative(Real xi) const {
    // Derivative: dB_{i,n}/dt = n * (B_{i-1,n-1}(t) - B_{i,n-1}(t))
    // Chain rule: dB/dxi = dB/dt * dt/dxi = dB/dt * 0.5

    Real t = 0.5 * (xi + 1.0);
    int n = order_;
    VecX dB(n + 1);

    if (n == 0) {
        dB(0) = 0.0;
        return dB;
    }

    // First compute B_{i,n-1}(t)
    BernsteinBasis1D lower_basis(n - 1);
    VecX B_lower = lower_basis.evaluate(xi);

    // dB_{0,n}/dt = n * (0 - B_{0,n-1}) = -n * B_{0,n-1}
    dB(0) = -static_cast<Real>(n) * B_lower(0);

    // dB_{i,n}/dt = n * (B_{i-1,n-1} - B_{i,n-1})
    for (int i = 1; i < n; ++i) {
        dB(i) = static_cast<Real>(n) * (B_lower(i-1) - B_lower(i));
    }

    // dB_{n,n}/dt = n * (B_{n-1,n-1} - 0) = n * B_{n-1,n-1}
    dB(n) = static_cast<Real>(n) * B_lower(n - 1);

    // Apply chain rule: dt/dxi = 0.5
    dB *= 0.5;

    return dB;
}

}  // namespace drifter
