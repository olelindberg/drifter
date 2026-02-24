#pragma once

/// @file cubic_thin_plate_hessian.hpp
/// @brief Thin plate spline energy functional for cubic Bezier surface
/// regularization
///
/// Computes the thin plate energy for cubic Bezier patches:
///   E = integral[(z_uu + z_vv)^2 + 2*z_uv^2]
///
/// The energy is quadratic in the control point values: E(z) = z^T * H * z
/// where H is the precomputed Hessian matrix (16x16 for cubic Bezier).

#include "bathymetry/bezier_hessian_base.hpp"
#include "bathymetry/cubic_bezier_basis_2d.hpp"
#include "core/types.hpp"
#include <memory>

namespace drifter {

/// @brief Computes thin plate spline energy Hessian for cubic Bezier patches
///
/// For a cubic Bezier surface z(u,v) = sum_k c_k * B_k(u,v), the thin plate
/// energy is:
///   E = integral_{[0,1]^2} [(z_uu + z_vv)^2 + 2*z_uv^2] dudv
///
/// This can be written as E = c^T * H * c where H is a 16x16 symmetric positive
/// semi-definite matrix that depends only on the basis functions.
class CubicThinPlateHessian : public BezierHessianBase {
public:
    /// @brief Construct thin plate Hessian with specified quadrature order
    /// @param ngauss Number of Gauss points per direction (default 4 for cubic)
    explicit CubicThinPlateHessian(int ngauss = 4);

    // BezierHessianBase interface
    int num_dofs() const override { return CubicBezierBasis2D::NDOF; }
    const MatX &element_hessian() const override { return H_; }

    /// @brief Evaluate thin plate energy for given control point z-values
    /// @param coeffs Vector of 16 control point z-values
    /// @return Energy value E = coeffs^T * H * coeffs
    Real energy(const VecX &coeffs) const;

    /// @brief Evaluate gradient of thin plate energy (2 * H * coeffs)
    /// @param coeffs Vector of 16 control point z-values
    /// @return Gradient vector (16 values)
    VecX gradient(const VecX &coeffs) const;

    /// @brief Compute scaled Hessian for a physical element
    ///
    /// For an element with dimensions (dx, dy), applies proper derivative
    /// scaling.
    ///
    /// @param dx Element width
    /// @param dy Element height
    /// @return Scaled Hessian matrix (16 x 16)
    MatX scaled_hessian(Real dx, Real dy) const override;

    /// @brief Number of Gauss points per direction
    int num_gauss_points() const { return ngauss_; }

    /// @brief Access derivative matrices (for debugging/testing)
    const MatX &d2u_matrix() const { return D2U_; }
    const MatX &d2v_matrix() const { return D2V_; }
    const MatX &d2uv_matrix() const { return D2UV_; }

private:
    int ngauss_; ///< Number of Gauss points per direction
    MatX H_; ///< Precomputed Hessian (16 x 16)
    MatX D2U_; ///< d^2/du^2 at Gauss points (ngauss^2 x 16)
    MatX D2V_; ///< d^2/dv^2 at Gauss points (ngauss^2 x 16)
    MatX D2UV_; ///< d^2/dudv at Gauss points (ngauss^2 x 16)
    VecX gauss_weights_; ///< Quadrature weights (ngauss^2)
    VecX gauss_nodes_; ///< 1D Gauss nodes in [0,1]

    std::unique_ptr<CubicBezierBasis2D> basis_;

    /// Build derivative evaluation matrices at Gauss points
    void build_derivative_matrices();

    /// Build the Hessian from derivative matrices
    void build_hessian();

    /// Compute Gauss-Legendre nodes and weights on [0,1]
    void compute_gauss_quadrature();
};

} // namespace drifter
