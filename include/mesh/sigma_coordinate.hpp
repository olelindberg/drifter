#pragma once

// Sigma-coordinate transformation for terrain-following vertical coordinates
// Used in ocean modeling where the vertical coordinate follows both the
// free surface (eta) and bathymetry (h).
//
// The sigma coordinate maps the physical depth z to a normalized range [-1, 0]:
//   sigma = (z - eta) / H  where H = eta + h (total water depth)
//   z = eta + sigma * H
//
// At surface: sigma = 0, z = eta
// At bottom:  sigma = -1, z = -h

#include "core/types.hpp"
#include <cmath>
#include <functional>

namespace drifter {

/// @brief Vertical stretching function types for sigma coordinates
enum class SigmaStretchType {
    Uniform,        ///< Uniform spacing in sigma
    Surface,        ///< Enhanced resolution near surface (wind-driven mixing)
    Bottom,         ///< Enhanced resolution near bottom (BBL)
    SurfaceBottom,  ///< Enhanced at both surface and bottom (Song & Haidvogel)
    ROMS            ///< ROMS-style stretching (UCLA ocean model)
};

/// @brief Parameters for sigma stretching functions
struct SigmaStretchParams {
    Real theta_s = 5.0;   ///< Surface stretching parameter (0-10)
    Real theta_b = 0.4;   ///< Bottom stretching parameter (0-1)
    Real hc = 200.0;      ///< Critical depth (meters) for ROMS stretching
};

/// @brief Sigma-coordinate transformation class
/// @details Provides coordinate transforms, metric terms, and stretching functions
class SigmaCoordinate {
public:
    // =========================================================================
    // Core coordinate transforms
    // =========================================================================

    /// @brief Transform physical z to sigma coordinate
    /// @param z Physical depth (negative below mean sea level)
    /// @param eta Free surface elevation (positive upward)
    /// @param h Bathymetry depth (positive downward, h > 0)
    /// @return Sigma coordinate in [-1, 0]
    static Real z_to_sigma(Real z, Real eta, Real h) {
        const Real H = eta + h;
        if (H < 1e-10) return -0.5;  // Degenerate case
        return (z - eta) / H;
    }

    /// @brief Transform sigma to physical z coordinate
    /// @param sigma Sigma coordinate in [-1, 0]
    /// @param eta Free surface elevation
    /// @param h Bathymetry depth
    /// @return Physical z coordinate
    static Real sigma_to_z(Real sigma, Real eta, Real h) {
        const Real H = eta + h;
        return eta + sigma * H;
    }

    /// @brief Total water column depth
    static Real total_depth(Real eta, Real h) {
        return eta + h;
    }

    // =========================================================================
    // Metric terms for transformed equations
    // =========================================================================

    /// @brief Jacobian dz/dsigma = H (total depth)
    static Real dz_dsigma(Real eta, Real h) {
        return eta + h;
    }

    /// @brief Inverse Jacobian dsigma/dz = 1/H
    static Real dsigma_dz(Real eta, Real h) {
        const Real H = eta + h;
        return (H > 1e-10) ? 1.0 / H : 0.0;
    }

    /// @brief Horizontal metric term dsigma/dx at constant z
    /// @details For sigma = (z - eta) / H where H = eta + h:
    ///          dsigma/dx|_z = -(deta/dx + sigma * dH/dx) / H
    static Real dsigma_dx(Real sigma, Real eta, Real h,
                          Real deta_dx, Real dh_dx) {
        const Real H = eta + h;
        if (H < 1e-10) return 0.0;
        const Real dH_dx = deta_dx + dh_dx;
        return -(deta_dx + sigma * dH_dx) / H;
    }

    /// @brief Horizontal metric term dsigma/dy at constant z
    /// @details For sigma = (z - eta) / H where H = eta + h:
    ///          dsigma/dy|_z = -(deta/dy + sigma * dH/dy) / H
    static Real dsigma_dy(Real sigma, Real eta, Real h,
                          Real deta_dy, Real dh_dy) {
        const Real H = eta + h;
        if (H < 1e-10) return 0.0;
        const Real dH_dy = deta_dy + dh_dy;
        return -(deta_dy + sigma * dH_dy) / H;
    }

    /// @brief Time derivative metric term dsigma/dt at fixed (x,y,z)
    /// @details dsigma/dt|_z = -(1/H) * deta/dt * (1 + sigma)
    static Real dsigma_dt(Real sigma, Real eta, Real h, Real deta_dt) {
        const Real H = eta + h;
        if (H < 1e-10) return 0.0;
        return -(deta_dt * (1.0 + sigma)) / H;
    }

    /// @brief Physical z derivative with respect to x at constant sigma
    /// @details dz/dx|_sigma = deta/dx + sigma * dH/dx
    static Real dz_dx_at_sigma(Real sigma, Real deta_dx, Real dh_dx) {
        return deta_dx + sigma * (deta_dx + dh_dx);
    }

    /// @brief Physical z derivative with respect to y at constant sigma
    static Real dz_dy_at_sigma(Real sigma, Real deta_dy, Real dh_dy) {
        return deta_dy + sigma * (deta_dy + dh_dy);
    }

    /// @brief Physical z derivative with respect to t at constant sigma
    /// @details dz/dt|_sigma = deta/dt * (1 + sigma)
    static Real dz_dt_at_sigma(Real sigma, Real deta_dt) {
        return deta_dt * (1.0 + sigma);
    }

    // =========================================================================
    // Vertical stretching functions
    // =========================================================================

    /// @brief Apply vertical stretching to uniform sigma
    /// @param sigma_uniform Uniform sigma in [-1, 0]
    /// @param type Stretching type
    /// @param params Stretching parameters
    /// @return Stretched sigma value
    static Real apply_stretching(Real sigma_uniform, SigmaStretchType type,
                                 const SigmaStretchParams& params = SigmaStretchParams()) {
        switch (type) {
            case SigmaStretchType::Uniform:
                return sigma_uniform;
            case SigmaStretchType::Surface:
                return stretch_surface(sigma_uniform, params.theta_s);
            case SigmaStretchType::Bottom:
                return stretch_bottom(sigma_uniform, params.theta_b);
            case SigmaStretchType::SurfaceBottom:
                return stretch_song_haidvogel(sigma_uniform, params.theta_s, params.theta_b);
            case SigmaStretchType::ROMS:
                return stretch_roms(sigma_uniform, params.theta_s, params.theta_b);
            default:
                return sigma_uniform;
        }
    }

    /// @brief Get derivative of stretched sigma with respect to uniform sigma
    /// @details Needed for Jacobian computation
    static Real stretching_derivative(Real sigma_uniform, SigmaStretchType type,
                                      const SigmaStretchParams& params = SigmaStretchParams()) {
        // Use central difference for general case
        constexpr Real eps = 1e-6;
        Real sigma_plus = std::min(sigma_uniform + eps, 0.0);
        Real sigma_minus = std::max(sigma_uniform - eps, -1.0);
        Real C_plus = apply_stretching(sigma_plus, type, params);
        Real C_minus = apply_stretching(sigma_minus, type, params);
        return (C_plus - C_minus) / (sigma_plus - sigma_minus);
    }

    /// @brief Generate array of sigma values with stretching
    /// @param n Number of sigma levels
    /// @param type Stretching type
    /// @param params Stretching parameters
    /// @return Vector of sigma values from -1 to 0
    static VecX generate_sigma_levels(int n, SigmaStretchType type,
                                      const SigmaStretchParams& params = SigmaStretchParams()) {
        VecX sigma(n);
        for (int k = 0; k < n; ++k) {
            // Uniform sigma from -1 to 0
            Real sigma_uniform = -1.0 + static_cast<Real>(k) / (n - 1);
            sigma(k) = apply_stretching(sigma_uniform, type, params);
        }
        return sigma;
    }

    // =========================================================================
    // Vectorized operations for efficiency
    // =========================================================================

    /// @brief Transform array of z values to sigma (same eta, h for all)
    static void z_to_sigma_array(const VecX& z, Real eta, Real h, VecX& sigma) {
        const Real H = eta + h;
        const Real H_inv = (H > 1e-10) ? 1.0 / H : 0.0;
        sigma = (z.array() - eta) * H_inv;
    }

    /// @brief Transform array of sigma values to z
    static void sigma_to_z_array(const VecX& sigma, Real eta, Real h, VecX& z) {
        const Real H = eta + h;
        z = eta + sigma.array() * H;
    }

    /// @brief Compute sigma at all DOFs given eta, h at horizontal nodes
    /// @details For 3D element with tensor-product structure
    /// @param eta_horiz Free surface at 2D horizontal nodes (n_horiz)
    /// @param h_horiz Bathymetry at 2D horizontal nodes (n_horiz)
    /// @param sigma_1d Sigma values at vertical nodes (n_vert)
    /// @param sigma_3d Output: sigma at all 3D nodes (n_horiz * n_vert)
    static void compute_sigma_3d(const VecX& eta_horiz, const VecX& h_horiz,
                                  const VecX& sigma_1d, VecX& sigma_3d) {
        const int n_horiz = static_cast<int>(eta_horiz.size());
        const int n_vert = static_cast<int>(sigma_1d.size());
        sigma_3d.resize(n_horiz * n_vert);

        // Tensor product: sigma_3d(i,k) = sigma_1d(k) for all horizontal nodes i
        // The actual z values depend on eta, h, but sigma itself is just the reference
        for (int i = 0; i < n_horiz; ++i) {
            for (int k = 0; k < n_vert; ++k) {
                sigma_3d(i * n_vert + k) = sigma_1d(k);
            }
        }
    }

    /// @brief Compute physical z at all DOFs
    static void compute_z_3d(const VecX& eta_horiz, const VecX& h_horiz,
                              const VecX& sigma_1d, VecX& z_3d) {
        const int n_horiz = static_cast<int>(eta_horiz.size());
        const int n_vert = static_cast<int>(sigma_1d.size());
        z_3d.resize(n_horiz * n_vert);

        for (int i = 0; i < n_horiz; ++i) {
            const Real H = eta_horiz(i) + h_horiz(i);
            for (int k = 0; k < n_vert; ++k) {
                z_3d(i * n_vert + k) = eta_horiz(i) + sigma_1d(k) * H;
            }
        }
    }

private:
    // =========================================================================
    // Stretching function implementations
    // =========================================================================

    /// @brief Surface-enhanced stretching (tanh-based)
    /// @details C(s) = sinh(theta_s * s) / sinh(theta_s)
    static Real stretch_surface(Real s, Real theta_s) {
        if (theta_s < 1e-6) return s;
        return std::sinh(theta_s * s) / std::sinh(theta_s);
    }

    /// @brief Bottom-enhanced stretching
    /// @details Uses exponential-type stretching that concentrates points near bottom (s=-1)
    ///          C(s) = (exp(theta_b*(s+1)) - 1) / (exp(theta_b) - 1) - 1
    ///          Endpoints: C(-1) = -1, C(0) = 0
    ///          For theta_b > 0: points are pushed toward bottom (C(s) < s for s in (-1, 0))
    static Real stretch_bottom(Real s, Real theta_b) {
        if (theta_b < 1e-6) return s;
        // Exponential stretching that gives more resolution near bottom
        // At s=-1: (exp(0) - 1) / (exp(theta_b) - 1) - 1 = 0 - 1 = -1 ✓
        // At s=0: (exp(theta_b) - 1) / (exp(theta_b) - 1) - 1 = 1 - 1 = 0 ✓
        // For s in (-1, 0): result < s (points pushed toward bottom)
        const Real exp_tb = std::exp(theta_b);
        const Real exp_arg = std::exp(theta_b * (s + 1.0));
        return (exp_arg - 1.0) / (exp_tb - 1.0) - 1.0;
    }

    /// @brief Song and Haidvogel (1994) stretching
    /// @details Combined surface and bottom stretching
    static Real stretch_song_haidvogel(Real s, Real theta_s, Real theta_b) {
        // C(s) = (1 - b) * C_surface(s) + b * C_bottom(s)
        // where b = (cosh(theta_s) - 1) / (cosh(theta_s) + 1) * theta_b
        const Real Cs = stretch_surface(s, theta_s);

        if (theta_b < 1e-6) return Cs;

        const Real cosh_ts = std::cosh(theta_s);
        const Real b = theta_b * (cosh_ts - 1.0) / (cosh_ts + 1.0);
        const Real Cb = stretch_bottom(s, theta_b);

        return (1.0 - b) * Cs + b * Cb;
    }

    /// @brief ROMS-style vertical stretching (Shchepetkin & McWilliams, 2005)
    /// @details More refined stretching with better bottom layer resolution
    static Real stretch_roms(Real s, Real theta_s, Real theta_b) {
        // ROMS Vtransform = 2, Vstretching = 4
        // Cs = (1 - cosh(theta_s * s)) / (cosh(theta_s) - 1)  for surface
        // Combined with bottom stretching

        Real Csur = 0.0;
        if (theta_s > 0.0) {
            const Real csrf = (1.0 - std::cosh(theta_s * s)) /
                              (std::cosh(theta_s) - 1.0);
            Csur = csrf;
        } else {
            Csur = -s * s;
        }

        Real Cbot = 0.0;
        if (theta_b > 0.0) {
            const Real exp_b = std::exp(theta_b * (s + 1.0));
            Cbot = (exp_b - 1.0) / (std::exp(theta_b) - 1.0) - 1.0;
        } else {
            Cbot = s;
        }

        // Weight surface and bottom
        const Real weight = (s + 1.0);  // 0 at bottom, 1 at surface
        return weight * Csur + (1.0 - weight) * Cbot;
    }
};

/// @brief Compute all sigma-coordinate metric terms at element DOFs
/// @details Computes dsigma/dx, dsigma/dy, dsigma/dt at all DOF locations
class SigmaMetrics {
public:
    SigmaMetrics() = default;

    /// @brief Initialize metrics storage
    /// @param num_dofs Number of DOFs per element
    void resize(int num_dofs) {
        dsigma_dx_.resize(num_dofs);
        dsigma_dy_.resize(num_dofs);
        dsigma_dt_.resize(num_dofs);
        H_.resize(num_dofs);
        H_inv_.resize(num_dofs);
    }

    /// @brief Update all metric terms
    /// @param sigma Sigma values at DOFs
    /// @param eta Free surface at DOFs
    /// @param h Bathymetry at DOFs
    /// @param deta_dx, deta_dy Free surface gradients
    /// @param dh_dx, dh_dy Bathymetry gradients
    /// @param deta_dt Free surface time derivative
    void update(const VecX& sigma, const VecX& eta, const VecX& h,
                const VecX& deta_dx, const VecX& deta_dy,
                const VecX& dh_dx, const VecX& dh_dy,
                const VecX& deta_dt) {
        const int n = static_cast<int>(sigma.size());
        resize(n);

        for (int i = 0; i < n; ++i) {
            H_(i) = SigmaCoordinate::total_depth(eta(i), h(i));
            H_inv_(i) = (H_(i) > 1e-10) ? 1.0 / H_(i) : 0.0;
            dsigma_dx_(i) = SigmaCoordinate::dsigma_dx(
                sigma(i), eta(i), h(i), deta_dx(i), dh_dx(i));
            dsigma_dy_(i) = SigmaCoordinate::dsigma_dy(
                sigma(i), eta(i), h(i), deta_dy(i), dh_dy(i));
            dsigma_dt_(i) = SigmaCoordinate::dsigma_dt(
                sigma(i), eta(i), h(i), deta_dt(i));
        }
    }

    /// @brief Update metrics without time derivative (steady state)
    void update_steady(const VecX& sigma, const VecX& eta, const VecX& h,
                       const VecX& deta_dx, const VecX& deta_dy,
                       const VecX& dh_dx, const VecX& dh_dy) {
        VecX zero_dt = VecX::Zero(sigma.size());
        update(sigma, eta, h, deta_dx, deta_dy, dh_dx, dh_dy, zero_dt);
    }

    // Accessors
    const VecX& dsigma_dx() const { return dsigma_dx_; }
    const VecX& dsigma_dy() const { return dsigma_dy_; }
    const VecX& dsigma_dt() const { return dsigma_dt_; }
    const VecX& H() const { return H_; }
    const VecX& H_inv() const { return H_inv_; }

private:
    VecX dsigma_dx_;
    VecX dsigma_dy_;
    VecX dsigma_dt_;
    VecX H_;
    VecX H_inv_;
};

}  // namespace drifter
