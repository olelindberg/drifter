#pragma once

/// @file thin_plate_hessian.hpp
/// @brief Thin plate spline energy functional for Bezier surface regularization
///
/// Computes the combined thin plate + gradient energy:
///   E = integral[(z_uu + z_vv)^2 + 2*z_uv^2] + α*integral[z_u^2 + z_v^2]
///
/// The thin plate term penalizes curvature, the gradient term penalizes slope.
/// Together they provide smooth bathymetry representation without oscillations.
///
/// The energy is quadratic in the control point values: E(z) = z^T * H * z
/// where H is the precomputed Hessian matrix (36x36 for quintic Bezier).

#include "bathymetry/bezier_basis_2d.hpp"
#include "core/types.hpp"
#include <memory>

namespace drifter {

/// @brief Computes thin plate spline energy Hessian for Bezier patches
///
/// For a Bezier surface z(u,v) = sum_k c_k * B_k(u,v), the thin plate energy
/// is:
///   E = integral_{[0,1]^2} [(z_uu + z_vv)^2 + 2*z_uv^2] dudv
///     = integral [(z_uu)^2 + 2*z_uu*z_vv + (z_vv)^2 + 2*(z_uv)^2] dudv
///
/// This can be written as E = c^T * H * c where H is a symmetric positive
/// semi-definite matrix that depends only on the basis functions (not on the
/// coefficients).
///
/// The Hessian is computed using Gauss-Legendre quadrature:
///   H = D2U^T * W * D2U + D2V^T * W * D2V + 2 * D2U^T * W * D2V + 2 * D2UV^T *
///   W * D2UV
/// where D2U, D2V, D2UV are derivative evaluation matrices at Gauss points
/// and W is the diagonal matrix of quadrature weights.
class ThinPlateHessian {
public:
    /// @brief Construct thin plate Hessian with specified quadrature order
    /// @param ngauss Number of Gauss points per direction (default 6 for
    /// quintic)
    /// @param gradient_weight Weight for gradient penalty term (default 1.0)
    explicit ThinPlateHessian(int ngauss = 6, Real gradient_weight = 1.0);

    /// @brief Get the precomputed element Hessian matrix (36 x 36)
    ///
    /// This is for a single element with unit size [0,1]^2.
    /// For an element with physical size (dx, dy), scale by:
    ///   H_physical = H / (dx * dy) * scaling_factors
    /// where scaling_factors account for derivative chain rule.
    const MatX &element_hessian() const { return H_; }

    /// @brief Evaluate thin plate energy for given control point z-values
    /// @param coeffs Vector of 36 control point z-values
    /// @return Energy value E = coeffs^T * H * coeffs
    Real energy(const VecX &coeffs) const;

    /// @brief Evaluate gradient of thin plate energy (2 * H * coeffs)
    /// @param coeffs Vector of 36 control point z-values
    /// @return Gradient vector (36 values)
    VecX gradient(const VecX &coeffs) const;

    /// @brief Compute scaled Hessian for a physical element
    ///
    /// For an element with dimensions (dx, dy), the derivatives scale as:
    ///   d/du_physical = (1/dx) * d/du_param
    ///   d^2/du^2_physical = (1/dx^2) * d^2/du^2_param
    ///
    /// The energy integral scales as dx*dy (Jacobian), so:
    ///   E_physical = (dx*dy) * integral[(1/dx^2 * z_uu + 1/dy^2 * z_vv)^2 +
    ///   ...] du dv
    ///
    /// @param dx Element width
    /// @param dy Element height
    /// @return Scaled Hessian matrix
    MatX scaled_hessian(Real dx, Real dy) const;

    /// @brief Number of Gauss points per direction
    int num_gauss_points() const { return ngauss_; }

    /// @brief Access derivative matrices (for debugging/testing)
    const MatX &d2u_matrix() const { return D2U_; }
    const MatX &d2v_matrix() const { return D2V_; }
    const MatX &d2uv_matrix() const { return D2UV_; }
    const MatX &d1u_matrix() const { return D1U_; }
    const MatX &d1v_matrix() const { return D1V_; }

    /// @brief Get gradient weight
    Real gradient_weight() const { return gradient_weight_; }

private:
    int ngauss_;           ///< Number of Gauss points per direction
    Real gradient_weight_; ///< Weight for gradient penalty term
    MatX H_;               ///< Precomputed Hessian (36 x 36)
    MatX D2U_;             ///< d^2/du^2 at Gauss points (ngauss^2 x 36)
    MatX D2V_;             ///< d^2/dv^2 at Gauss points (ngauss^2 x 36)
    MatX D2UV_;            ///< d^2/dudv at Gauss points (ngauss^2 x 36)
    MatX D1U_;             ///< d/du at Gauss points (ngauss^2 x 36)
    MatX D1V_;             ///< d/dv at Gauss points (ngauss^2 x 36)
    MatX H_u_u_;           ///< Gradient Hessian component D1U^T * W * D1U
    MatX H_v_v_;           ///< Gradient Hessian component D1V^T * W * D1V
    VecX gauss_weights_;   ///< Quadrature weights (ngauss^2)
    VecX gauss_nodes_;     ///< 1D Gauss nodes in [0,1]

    std::unique_ptr<BezierBasis2D> basis_;

    /// Build derivative evaluation matrices at Gauss points
    void build_derivative_matrices();

    /// Build the Hessian from derivative matrices
    void build_hessian();

    /// Compute Gauss-Legendre nodes and weights on [0,1]
    void compute_gauss_quadrature();
};

} // namespace drifter
