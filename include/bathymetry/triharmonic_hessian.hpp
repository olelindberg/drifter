#pragma once

/// @file triharmonic_hessian.hpp
/// @brief Triharmonic energy functional for Bezier surface regularization
///
/// Computes the triharmonic energy:
///   E = integral[(z_uuu + z_uvv)^2 + (z_uuv + z_vvv)^2] du dv
///
/// This corresponds to minimizing |grad(Laplacian z)|^2, producing surfaces
/// where the curvature gradient is minimized. This is the variational form
/// of the triharmonic equation nabla^6 z = 0.
///
/// The energy is quadratic in the control point values: E(z) = z^T * H * z
/// where H is the precomputed Hessian matrix (36x36 for quintic Bezier).
///
/// Compared to thin plate (biharmonic) energy which minimizes curvature,
/// triharmonic energy minimizes the rate of change of curvature, producing
/// smoother surfaces with more gradual transitions.

#include "bathymetry/bezier_basis_2d.hpp"
#include "core/types.hpp"
#include <memory>

namespace drifter {

/// @brief Computes triharmonic energy Hessian for Bezier patches
///
/// For a Bezier surface z(u,v) = sum_k c_k * B_k(u,v), the triharmonic energy
/// is:
///   E = integral_{[0,1]^2} |grad(Laplacian z)|^2 du dv
///     = integral [(z_uuu + z_uvv)^2 + (z_uuv + z_vvv)^2] du dv
///
/// This can be written as E = c^T * H * c where H is a symmetric positive
/// semi-definite matrix that depends only on the basis functions (not on the
/// coefficients).
///
/// The Hessian is computed using Gauss-Legendre quadrature:
///   H = G1^T * W * G1 + G2^T * W * G2
/// where G1 = D3UUU + D3UVV and G2 = D3UUV + D3VVV are the gradient of
/// Laplacian evaluation matrices at Gauss points, and W is the diagonal matrix
/// of quadrature weights.
class TriharmonicHessian {
public:
    /// @brief Construct triharmonic Hessian with specified quadrature order
    /// @param ngauss Number of Gauss points per direction (default 4, minimum
    /// 3)
    /// @param gradient_weight Weight for gradient penalty term (default 0.0)
    explicit TriharmonicHessian(int ngauss = 4, Real gradient_weight = 0.0);

    /// @brief Get the precomputed element Hessian matrix (36 x 36)
    ///
    /// This is for a single element with unit size [0,1]^2.
    /// For an element with physical size (dx, dy), use scaled_hessian(dx, dy).
    const MatX &element_hessian() const { return H_; }

    /// @brief Evaluate triharmonic energy for given control point z-values
    /// @param coeffs Vector of 36 control point z-values
    /// @return Energy value E = coeffs^T * H * coeffs
    Real energy(const VecX &coeffs) const;

    /// @brief Evaluate gradient of triharmonic energy (2 * H * coeffs)
    /// @param coeffs Vector of 36 control point z-values
    /// @return Gradient vector (36 values)
    VecX gradient(const VecX &coeffs) const;

    /// @brief Compute scaled Hessian for a physical element
    ///
    /// For an element with dimensions (dx, dy), the third derivatives scale as:
    ///   d^3/dx^3_physical = (1/dx^3) * d^3/du^3_param
    ///   d^3/dx^2 dy = (1/dx^2/dy) * d^3/du^2 dv_param
    ///   etc.
    ///
    /// The energy integral scales as dx*dy (Jacobian), combined with derivative
    /// scaling.
    ///
    /// @param dx Element width
    /// @param dy Element height
    /// @return Scaled Hessian matrix
    MatX scaled_hessian(Real dx, Real dy) const;

    /// @brief Number of Gauss points per direction
    int num_gauss_points() const { return ngauss_; }

    /// @brief Access derivative matrices (for debugging/testing)
    const MatX &d3uuu_matrix() const { return D3UUU_; }
    const MatX &d3uuv_matrix() const { return D3UUV_; }
    const MatX &d3uvv_matrix() const { return D3UVV_; }
    const MatX &d3vvv_matrix() const { return D3VVV_; }
    const MatX &d1u_matrix() const { return D1U_; }
    const MatX &d1v_matrix() const { return D1V_; }

    /// @brief Get gradient weight
    Real gradient_weight() const { return gradient_weight_; }

private:
    int ngauss_;           ///< Number of Gauss points per direction
    Real gradient_weight_; ///< Weight for gradient penalty term
    MatX H_;               ///< Precomputed Hessian (36 x 36)
    MatX D3UUU_;           ///< d^3/du^3 at Gauss points (ngauss^2 x 36)
    MatX D3UUV_;           ///< d^3/du^2 dv at Gauss points (ngauss^2 x 36)
    MatX D3UVV_;           ///< d^3/du dv^2 at Gauss points (ngauss^2 x 36)
    MatX D3VVV_;           ///< d^3/dv^3 at Gauss points (ngauss^2 x 36)
    MatX D1U_;             ///< d/du at Gauss points (ngauss^2 x 36)
    MatX D1V_;             ///< d/dv at Gauss points (ngauss^2 x 36)

    // Component Hessians for scaled assembly
    MatX H_uuu_uuu_; ///< D3UUU^T * W * D3UUU
    MatX H_uuv_uuv_; ///< D3UUV^T * W * D3UUV
    MatX H_uvv_uvv_; ///< D3UVV^T * W * D3UVV
    MatX H_vvv_vvv_; ///< D3VVV^T * W * D3VVV
    MatX H_uuu_uvv_; ///< D3UUU^T * W * D3UVV (cross term)
    MatX H_uuv_vvv_; ///< D3UUV^T * W * D3VVV (cross term)
    MatX H_u_u_;     ///< Gradient Hessian component D1U^T * W * D1U
    MatX H_v_v_;     ///< Gradient Hessian component D1V^T * W * D1V

    VecX gauss_weights_; ///< Quadrature weights (ngauss^2)
    VecX gauss_nodes_;   ///< 1D Gauss nodes in [0,1]

    std::unique_ptr<BezierBasis2D> basis_;

    /// Build derivative evaluation matrices at Gauss points
    void build_derivative_matrices();

    /// Build the Hessian from derivative matrices
    void build_hessian();

    /// Compute Gauss-Legendre nodes and weights on [0,1]
    void compute_gauss_quadrature();
};

} // namespace drifter
