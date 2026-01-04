#pragma once

// Vertical velocity diagnosis for sigma-coordinate ocean model
//
// In a hydrostatic model, the vertical velocity is not prognostic but
// diagnosed from the continuity equation (incompressibility constraint):
//
//   div(u) = du/dx + dv/dy + dw/dz = 0
//
// In sigma coordinates, this becomes:
//   d(Hu)/dx + d(Hv)/dy + d(H*omega)/dsigma = 0
//
// where omega = dsigma/dt is the "sigma velocity" (rate of change of sigma
// following a fluid parcel).
//
// Integrating vertically from bottom (sigma = -1) with BC omega(-1) = 0:
//   omega(sigma) = -(1/H) * integral_{-1}^{sigma} [d(Hu)/dx + d(Hv)/dy] dsigma'
//
// The physical vertical velocity w is then:
//   w = H*omega + u*dz/dx|_sigma + v*dz/dy|_sigma + dz/dt|_sigma

#include "core/types.hpp"
#include "dg/basis_hexahedron.hpp"
#include "dg/quadrature_3d.hpp"
#include <vector>

namespace drifter {

// Forward declarations
class DG3DElementOperator;

/// @brief Vertical velocity diagnosis from continuity equation
class VerticalVelocityDiagnosis {
public:
    /// @brief Construct vertical velocity diagnosis
    /// @param basis Hexahedron basis
    /// @param quad Volume quadrature
    VerticalVelocityDiagnosis(const HexahedronBasis& basis,
                               const GaussQuadrature3D& quad);

    /// @brief Get the basis
    const HexahedronBasis& basis() const { return basis_; }

    /// @brief Get the quadrature
    const GaussQuadrature3D& quadrature() const { return quad_; }

    // =========================================================================
    // Omega (sigma velocity) diagnosis
    // =========================================================================

    /// @brief Diagnose omega from horizontal divergence
    /// @details omega(sigma) = omega(-1) - (1/H) * integral_{-1}^{sigma} div_h(Hu, Hv) dsigma'
    ///          with BC: omega(-1) = 0 (no flow through bottom)
    /// @param div_Hu Horizontal divergence at DOFs (d(Hu)/dx + d(Hv)/dy)
    /// @param H Water depth at DOFs
    /// @param[out] omega Sigma velocity at DOFs
    void diagnose_omega(const VecX& div_Hu, const VecX& H, VecX& omega) const;

    /// @brief Diagnose omega from velocity and depth fields
    /// @param Hu H*u at element DOFs
    /// @param Hv H*v at element DOFs
    /// @param H Water depth at DOFs
    /// @param dHu_dx x-derivative of Hu at DOFs
    /// @param dHv_dy y-derivative of Hv at DOFs
    /// @param[out] omega Sigma velocity at DOFs
    void diagnose_omega_from_velocity(const VecX& Hu, const VecX& Hv,
                                       const VecX& H,
                                       const VecX& dHu_dx, const VecX& dHv_dy,
                                       VecX& omega) const;

    // =========================================================================
    // Physical vertical velocity
    // =========================================================================

    /// @brief Convert omega to physical vertical velocity w
    /// @details w = H*omega + u*dz/dx|_sigma + v*dz/dy|_sigma + dz/dt|_sigma
    ///            = H*omega + u*(deta/dx + sigma*dH/dx) + v*(deta/dy + sigma*dH/dy)
    ///              + deta/dt*(1 + sigma)
    /// @param omega Sigma velocity at DOFs
    /// @param u, v Horizontal velocity at DOFs
    /// @param H Water depth at DOFs
    /// @param dz_dx, dz_dy z-gradients at constant sigma
    /// @param dz_dt Time derivative of z at constant sigma (mesh velocity)
    /// @param[out] w Physical vertical velocity at DOFs
    void omega_to_physical_w(const VecX& omega,
                              const VecX& u, const VecX& v,
                              const VecX& H,
                              const VecX& dz_dx, const VecX& dz_dy,
                              const VecX& dz_dt,
                              VecX& w) const;

    /// @brief Simplified conversion assuming fixed bathymetry
    /// @param omega Sigma velocity
    /// @param u, v Horizontal velocity
    /// @param eta Free surface elevation
    /// @param h Bathymetry (fixed)
    /// @param sigma Sigma values at DOFs
    /// @param deta_dx, deta_dy Free surface gradients
    /// @param deta_dt Free surface time derivative
    /// @param[out] w Physical vertical velocity
    void omega_to_w_simple(const VecX& omega,
                           const VecX& u, const VecX& v,
                           const VecX& eta, const VecX& h,
                           const VecX& sigma,
                           const VecX& deta_dx, const VecX& deta_dy,
                           const VecX& deta_dt,
                           VecX& w) const;

    // =========================================================================
    // Boundary conditions
    // =========================================================================

    /// @brief Surface kinematic boundary condition for omega
    /// @details omega(0) = (deta/dt + u*deta/dx + v*deta/dy) / H
    ///          This ensures the free surface is a material surface
    /// @param deta_dt Free surface time derivative
    /// @param u_surf, v_surf Surface velocity
    /// @param deta_dx, deta_dy Free surface gradient
    /// @param H Water depth
    /// @return Surface omega value
    static Real surface_omega(Real deta_dt, Real u_surf, Real v_surf,
                              Real deta_dx, Real deta_dy, Real H);

    /// @brief Bottom kinematic boundary condition
    /// @details omega(-1) = -u*dh/dx - v*dh/dy (= 0 for flat bottom)
    ///          For sigma-following bottom, omega(-1) = 0 by definition
    /// @param u_bot, v_bot Bottom velocity
    /// @param dh_dx, dh_dy Bathymetry gradient
    /// @return Bottom omega value (usually 0)
    static Real bottom_omega(Real u_bot, Real v_bot,
                             Real dh_dx, Real dh_dy);

    // =========================================================================
    // Verification
    // =========================================================================

    /// @brief Check continuity equation residual
    /// @details After diagnosis, div_h(Hu,Hv) + d(H*omega)/dsigma should be ~0
    /// @param div_Hu Horizontal divergence
    /// @param H Water depth
    /// @param omega Diagnosed sigma velocity
    /// @param dHomega_dsigma Vertical derivative of H*omega
    /// @return L2 norm of residual
    Real continuity_residual(const VecX& div_Hu, const VecX& H,
                              const VecX& omega, const VecX& dHomega_dsigma) const;

private:
    const HexahedronBasis& basis_;
    const GaussQuadrature3D& quad_;

    int n_vert_;   // Number of vertical levels
    int n_horiz_;  // Number of horizontal nodes per level

    // Quadrature weights for vertical integration
    VecX vertical_integration_weights_;

    // Integration matrix for vertical direction
    // I(k,j) = integral from sigma(-1) to sigma(k) of phi_j dsigma
    MatX vertical_integration_matrix_;

    /// @brief Build vertical integration weights and matrix
    void build_vertical_integration();

    /// @brief Get 3D node index from horizontal index and vertical level
    Index node_index_3d(int i_horiz, int k_vert) const {
        return static_cast<Index>(i_horiz * n_vert_ + k_vert);
    }

    /// @brief Get horizontal index from 3D node index
    int horizontal_index(Index idx_3d) const {
        return static_cast<int>(idx_3d) / n_vert_;
    }

    /// @brief Get vertical level from 3D node index
    int vertical_level(Index idx_3d) const {
        return static_cast<int>(idx_3d) % n_vert_;
    }
};

/// @brief Alternative method: diagnose omega column by column
/// @details More efficient for column-based vertical integration
class ColumnVerticalVelocity {
public:
    /// @brief Construct column-based vertical velocity diagnosis
    /// @param order Polynomial order in vertical
    ColumnVerticalVelocity(int order);

    /// @brief Diagnose omega for a single column
    /// @param div_Hu_col Horizontal divergence at vertical levels (bottom to top)
    /// @param H_col Water depth at vertical levels
    /// @param omega_bottom Bottom boundary condition (usually 0)
    /// @param[out] omega_col Sigma velocity at vertical levels
    void diagnose_column(const VecX& div_Hu_col, const VecX& H_col,
                          Real omega_bottom, VecX& omega_col) const;

    /// @brief Get vertical integration weights
    const VecX& integration_weights() const { return weights_; }

    /// @brief Get sigma values at nodes
    const VecX& sigma_nodes() const { return sigma_nodes_; }

private:
    int order_;
    int n_levels_;

    VecX sigma_nodes_;    // Sigma at LGL nodes [-1, 0]
    VecX weights_;        // LGL quadrature weights (scaled for [-1, 0])
    MatX integration_matrix_;  // Cumulative integration from bottom
};

}  // namespace drifter
