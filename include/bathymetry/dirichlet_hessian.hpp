#pragma once

/// @file dirichlet_hessian.hpp
/// @brief Dirichlet energy functional for linear Bezier surface regularization
///
/// Computes the Dirichlet/Laplace energy for linear Bezier patches:
///   E = integral[z_u^2 + z_v^2] du dv
///
/// This is the natural smoothness term for piecewise linear surfaces,
/// analogous to minimizing the surface gradient magnitude.
///
/// The energy is quadratic in the control point values: E(z) = z^T * H * z
/// where H is the precomputed Hessian matrix (4x4 for linear Bezier).

#include "bathymetry/linear_bezier_basis_2d.hpp"
#include "core/types.hpp"
#include <memory>

namespace drifter {

/// @brief Computes Dirichlet energy Hessian for linear Bezier patches
///
/// For a linear Bezier surface z(u,v) = sum_k c_k * B_k(u,v), the Dirichlet
/// energy is:
///   E = integral_{[0,1]^2} [z_u^2 + z_v^2] dudv
///
/// This can be written as E = c^T * H * c where H is a 4x4 symmetric positive
/// semi-definite matrix that depends only on the basis functions.
class DirichletHessian {
public:
    /// @brief Construct Dirichlet Hessian with specified quadrature order
    /// @param ngauss Number of Gauss points per direction (default 2 for linear)
    explicit DirichletHessian(int ngauss = 2);

    /// @brief Get the precomputed element Hessian matrix (4 x 4)
    ///
    /// This is for a single element with unit size [0,1]^2.
    const MatX &element_hessian() const { return H_; }

    /// @brief Evaluate Dirichlet energy for given control point z-values
    /// @param coeffs Vector of 4 control point z-values
    /// @return Energy value E = coeffs^T * H * coeffs
    Real energy(const VecX &coeffs) const;

    /// @brief Evaluate gradient of Dirichlet energy (2 * H * coeffs)
    /// @param coeffs Vector of 4 control point z-values
    /// @return Gradient vector (4 values)
    VecX gradient(const VecX &coeffs) const;

    /// @brief Compute scaled Hessian for a physical element
    ///
    /// For an element with dimensions (dx, dy), applies proper derivative
    /// scaling:
    ///   z_x = z_u / dx, z_y = z_v / dy
    ///   E_physical = integral (z_x^2 + z_y^2) dx dy
    ///              = (dy/dx) * H_u_u + (dx/dy) * H_v_v
    ///
    /// @param dx Element width
    /// @param dy Element height
    /// @return Scaled Hessian matrix (4 x 4)
    MatX scaled_hessian(Real dx, Real dy) const;

    /// @brief Number of Gauss points per direction
    int num_gauss_points() const { return ngauss_; }

    /// @brief Access derivative matrices (for debugging/testing)
    const MatX &d1u_matrix() const { return D1U_; }
    const MatX &d1v_matrix() const { return D1V_; }

    /// @brief Access Hessian components (for debugging)
    const MatX &h_u_u() const { return H_u_u_; }
    const MatX &h_v_v() const { return H_v_v_; }

private:
    int ngauss_; ///< Number of Gauss points per direction
    MatX H_; ///< Precomputed Hessian (4 x 4)
    MatX D1U_; ///< d/du at Gauss points (ngauss^2 x 4)
    MatX D1V_; ///< d/dv at Gauss points (ngauss^2 x 4)
    MatX H_u_u_; ///< D1U^T * W * D1U component
    MatX H_v_v_; ///< D1V^T * W * D1V component
    VecX gauss_weights_; ///< Quadrature weights (ngauss^2)
    VecX gauss_nodes_; ///< 1D Gauss nodes in [0,1]

    LinearBezierBasis2D basis_;

    /// Build derivative evaluation matrices at Gauss points
    void build_derivative_matrices();

    /// Build the Hessian from derivative matrices
    void build_hessian();

    /// Compute Gauss-Legendre nodes and weights on [0,1]
    void compute_gauss_quadrature();
};

} // namespace drifter
