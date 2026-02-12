#pragma once

// Numerical Flux Interface for Discontinuous Galerkin Methods
//
// Numerical fluxes are used at element interfaces to couple the
// discontinuous approximation. They provide upwinding and stability.
//
// For conservation law: du/dt + div(F(u)) = 0
// The numerical flux F* approximates F(u)·n at interfaces.
//
// Common choices:
// - Lax-Friedrichs (Rusanov): Simple, diffusive
// - HLLC: Accurate for shallow water
// - Roe: Contact-preserving
// - Central: Non-dissipative (unstable without filtering)

#include "core/types.hpp"
#include <functional>
#include <memory>

namespace drifter {

/// @brief Numerical flux function signature
/// F*(U_L, U_R, n) returns the numerical flux in direction n
using NumericalFluxFunc =
    std::function<VecX(const VecX &, const VecX &, const Vec3 &)>;

/// @brief Physical flux function (for advection-like equations)
/// Returns flux tensor F(U) where F[d] is flux in direction d
using PhysicalFluxFunc = std::function<Tensor3(const VecX &)>;

/// @brief Base class for numerical fluxes
class NumericalFlux {
public:
    virtual ~NumericalFlux() = default;

    /// @brief Compute numerical flux
    /// @param U_L Left state
    /// @param U_R Right state
    /// @param n Outward normal
    /// @return Numerical flux F*
    virtual VecX
    flux(const VecX &U_L, const VecX &U_R, const Vec3 &n) const = 0;

    /// @brief Get function wrapper for this flux
    NumericalFluxFunc as_function() const {
        return [this](const VecX &U_L, const VecX &U_R, const Vec3 &n) {
            return this->flux(U_L, U_R, n);
        };
    }

    /// @brief Number of conserved variables
    virtual int num_vars() const = 0;
};

/// @brief Lax-Friedrichs (Rusanov) numerical flux
/// @details F* = 0.5 * (F(U_L) + F(U_R)) - 0.5 * lambda_max * (U_R - U_L)
///          where lambda_max is the maximum wave speed
class LaxFriedrichsFlux : public NumericalFlux {
public:
    /// @brief Construct with physical flux function and wave speed estimator
    /// @param physical_flux Physical flux F(U)
    /// @param max_wave_speed Function to compute maximum wave speed
    /// @param num_vars Number of conserved variables
    LaxFriedrichsFlux(
        PhysicalFluxFunc physical_flux,
        std::function<Real(const VecX &, const VecX &, const Vec3 &)>
            max_wave_speed,
        int num_vars);

    VecX flux(const VecX &U_L, const VecX &U_R, const Vec3 &n) const override;
    int num_vars() const override { return num_vars_; }

private:
    PhysicalFluxFunc physical_flux_;
    std::function<Real(const VecX &, const VecX &, const Vec3 &)>
        max_wave_speed_;
    int num_vars_;
};

/// @brief Central flux (no dissipation)
/// @details F* = 0.5 * (F(U_L) + F(U_R))
class CentralFlux : public NumericalFlux {
public:
    CentralFlux(PhysicalFluxFunc physical_flux, int num_vars);

    VecX flux(const VecX &U_L, const VecX &U_R, const Vec3 &n) const override;
    int num_vars() const override { return num_vars_; }

private:
    PhysicalFluxFunc physical_flux_;
    int num_vars_;
};

/// @brief Upwind flux for linear advection
/// @details F* = a * U_upwind where upwind is determined by sign of a·n
class UpwindFlux : public NumericalFlux {
public:
    /// @brief Construct upwind flux
    /// @param velocity Advection velocity
    UpwindFlux(const Vec3 &velocity);

    VecX flux(const VecX &U_L, const VecX &U_R, const Vec3 &n) const override;
    int num_vars() const override { return 1; }

    void set_velocity(const Vec3 &velocity) { velocity_ = velocity; }

private:
    Vec3 velocity_;
};

/// @brief Shallow water HLLC flux
/// @details Harten-Lax-van Leer-Contact flux for shallow water equations
///          Preserves intermediate (contact) waves for improved accuracy
class ShallowWaterHLLCFlux : public NumericalFlux {
public:
    /// @brief Construct HLLC flux for shallow water
    /// @param g Gravitational acceleration
    ShallowWaterHLLCFlux(Real g = 9.81);

    VecX flux(const VecX &U_L, const VecX &U_R, const Vec3 &n) const override;
    int num_vars() const override { return 3; } // h, hu, hv

    /// @brief Set gravity
    void set_gravity(Real g) { g_ = g; }

private:
    Real g_;

    /// @brief Compute wave speeds for HLLC
    void wave_speeds(
        Real h_L, Real u_L, Real h_R, Real u_R, Real &S_L, Real &S_R,
        Real &S_star) const;

    /// @brief Compute flux from primitive variables
    Vec3 physical_flux(Real h, Real u, Real v, Real g, Real nx, Real ny) const;
};

/// @brief Shallow water Roe flux with entropy fix
class ShallowWaterRoeFlux : public NumericalFlux {
public:
    ShallowWaterRoeFlux(Real g = 9.81, Real entropy_fix = 0.1);

    VecX flux(const VecX &U_L, const VecX &U_R, const Vec3 &n) const override;
    int num_vars() const override { return 3; }

private:
    Real g_;
    Real entropy_fix_;

    /// @brief Roe-averaged values
    void roe_average(
        Real h_L, Real u_L, Real v_L, Real h_R, Real u_R, Real v_R, Real &h_roe,
        Real &u_roe, Real &v_roe) const;
};

/// @brief Local Lax-Friedrichs for general conservation laws
/// @details Uses local maximum wave speed at each interface
template <int NVARS> class LocalLaxFriedrichs : public NumericalFlux {
public:
    using StateVec = Eigen::Matrix<Real, NVARS, 1>;
    using FluxMat = Eigen::Matrix<Real, NVARS, 3>; // Flux in each direction

    /// @brief Physical flux function type
    using PhysFluxFunc = std::function<FluxMat(const StateVec &)>;

    /// @brief Maximum wave speed function type
    using WaveSpeedFunc = std::function<Real(const StateVec &, const Vec3 &)>;

    LocalLaxFriedrichs(PhysFluxFunc physical_flux, WaveSpeedFunc max_wave_speed)
        : physical_flux_(physical_flux), max_wave_speed_(max_wave_speed) {}

    VecX flux(const VecX &U_L, const VecX &U_R, const Vec3 &n) const override {
        StateVec uL = U_L;
        StateVec uR = U_R;

        FluxMat F_L = physical_flux_(uL);
        FluxMat F_R = physical_flux_(uR);

        // Flux in normal direction
        StateVec Fn_L = F_L * n;
        StateVec Fn_R = F_R * n;

        // Maximum wave speed
        Real lambda = std::max(max_wave_speed_(uL, n), max_wave_speed_(uR, n));

        // Lax-Friedrichs flux
        StateVec result = 0.5 * (Fn_L + Fn_R) - 0.5 * lambda * (uR - uL);
        return result;
    }

    int num_vars() const override { return NVARS; }

private:
    PhysFluxFunc physical_flux_;
    WaveSpeedFunc max_wave_speed_;
};

/// @brief Factory for creating common numerical fluxes
class NumericalFluxFactory {
public:
    /// @brief Create Lax-Friedrichs flux for shallow water
    static std::unique_ptr<NumericalFlux>
    shallow_water_lax_friedrichs(Real g = 9.81);

    /// @brief Create HLLC flux for shallow water
    static std::unique_ptr<NumericalFlux> shallow_water_hllc(Real g = 9.81);

    /// @brief Create Roe flux for shallow water
    static std::unique_ptr<NumericalFlux> shallow_water_roe(Real g = 9.81);

    /// @brief Create upwind flux for advection
    static std::unique_ptr<NumericalFlux>
    advection_upwind(const Vec3 &velocity);

    /// @brief Create central flux for diffusion
    static std::unique_ptr<NumericalFlux>
    central(PhysicalFluxFunc phys_flux, int nvars);
};

// =============================================================================
// Inline implementations
// =============================================================================

inline LaxFriedrichsFlux::LaxFriedrichsFlux(
    PhysicalFluxFunc physical_flux,
    std::function<Real(const VecX &, const VecX &, const Vec3 &)>
        max_wave_speed,
    int num_vars)
    : physical_flux_(std::move(physical_flux)),
      max_wave_speed_(std::move(max_wave_speed)), num_vars_(num_vars) {}

inline VecX
LaxFriedrichsFlux::flux(const VecX &U_L, const VecX &U_R, const Vec3 &n) const {
    Tensor3 F_L = physical_flux_(U_L);
    Tensor3 F_R = physical_flux_(U_R);

    // Compute flux in normal direction
    VecX Fn_L = VecX::Zero(num_vars_);
    VecX Fn_R = VecX::Zero(num_vars_);

    for (int i = 0; i < num_vars_; ++i) {
        for (int d = 0; d < 3; ++d) {
            Fn_L(i) += F_L[d](i, 0) * n(d);
            Fn_R(i) += F_R[d](i, 0) * n(d);
        }
    }

    Real lambda = max_wave_speed_(U_L, U_R, n);
    return 0.5 * (Fn_L + Fn_R) - 0.5 * lambda * (U_R - U_L);
}

inline CentralFlux::CentralFlux(PhysicalFluxFunc physical_flux, int num_vars)
    : physical_flux_(std::move(physical_flux)), num_vars_(num_vars) {}

inline VecX
CentralFlux::flux(const VecX &U_L, const VecX &U_R, const Vec3 &n) const {
    Tensor3 F_L = physical_flux_(U_L);
    Tensor3 F_R = physical_flux_(U_R);

    VecX Fn_L = VecX::Zero(num_vars_);
    VecX Fn_R = VecX::Zero(num_vars_);

    for (int i = 0; i < num_vars_; ++i) {
        for (int d = 0; d < 3; ++d) {
            Fn_L(i) += F_L[d](i, 0) * n(d);
            Fn_R(i) += F_R[d](i, 0) * n(d);
        }
    }

    return 0.5 * (Fn_L + Fn_R);
}

inline UpwindFlux::UpwindFlux(const Vec3 &velocity) : velocity_(velocity) {}

inline VecX
UpwindFlux::flux(const VecX &U_L, const VecX &U_R, const Vec3 &n) const {
    Real vn = velocity_.dot(n);
    VecX result(1);
    result(0) = vn * ((vn >= 0) ? U_L(0) : U_R(0));
    return result;
}

inline ShallowWaterHLLCFlux::ShallowWaterHLLCFlux(Real g) : g_(g) {}

inline VecX ShallowWaterHLLCFlux::flux(
    const VecX &U_L, const VecX &U_R, const Vec3 &n) const {
    // State: [h, hu, hv]
    Real h_L = U_L(0);
    Real hu_L = U_L(1);
    Real hv_L = U_L(2);

    Real h_R = U_R(0);
    Real hu_R = U_R(1);
    Real hv_R = U_R(2);

    // Avoid division by zero
    Real h_min = 1e-10;
    if (h_L < h_min)
        h_L = h_min;
    if (h_R < h_min)
        h_R = h_min;

    Real u_L = hu_L / h_L;
    Real v_L = hv_L / h_L;
    Real u_R = hu_R / h_R;
    Real v_R = hv_R / h_R;

    Real nx = n(0);
    Real ny = n(1);

    // Normal velocities
    Real un_L = u_L * nx + v_L * ny;
    Real un_R = u_R * nx + v_R * ny;

    // Wave speeds
    Real c_L = std::sqrt(g_ * h_L);
    Real c_R = std::sqrt(g_ * h_R);

    Real S_L, S_R, S_star;
    wave_speeds(h_L, un_L, h_R, un_R, S_L, S_R, S_star);

    // Physical flux
    Vec3 F_L = physical_flux(h_L, u_L, v_L, g_, nx, ny);
    Vec3 F_R = physical_flux(h_R, u_R, v_R, g_, nx, ny);

    VecX result(3);

    if (S_L >= 0) {
        // Left state
        result(0) = F_L(0);
        result(1) = F_L(1);
        result(2) = F_L(2);
    } else if (S_R <= 0) {
        // Right state
        result(0) = F_R(0);
        result(1) = F_R(1);
        result(2) = F_R(2);
    } else {
        // Star region
        // HLLC intermediate states
        Real h_star_L = h_L * (S_L - un_L) / (S_L - S_star);
        Real h_star_R = h_R * (S_R - un_R) / (S_R - S_star);

        if (S_star >= 0) {
            // Left star
            Vec3 U_star_L;
            U_star_L(0) = h_star_L;
            U_star_L(1) = h_star_L * (u_L + (S_star - un_L) * nx);
            U_star_L(2) = h_star_L * (v_L + (S_star - un_L) * ny);

            Vec3 U_L_vec;
            U_L_vec(0) = h_L;
            U_L_vec(1) = hu_L;
            U_L_vec(2) = hv_L;

            Vec3 F_star = F_L + S_L * (U_star_L - U_L_vec);
            result = F_star;
        } else {
            // Right star
            Vec3 U_star_R;
            U_star_R(0) = h_star_R;
            U_star_R(1) = h_star_R * (u_R + (S_star - un_R) * nx);
            U_star_R(2) = h_star_R * (v_R + (S_star - un_R) * ny);

            Vec3 U_R_vec;
            U_R_vec(0) = h_R;
            U_R_vec(1) = hu_R;
            U_R_vec(2) = hv_R;

            Vec3 F_star = F_R + S_R * (U_star_R - U_R_vec);
            result = F_star;
        }
    }

    return result;
}

inline void ShallowWaterHLLCFlux::wave_speeds(
    Real h_L, Real u_L, Real h_R, Real u_R, Real &S_L, Real &S_R,
    Real &S_star) const {

    Real c_L = std::sqrt(g_ * h_L);
    Real c_R = std::sqrt(g_ * h_R);

    // Einfeldt wave speed estimates
    S_L = std::min(u_L - c_L, u_R - c_R);
    S_R = std::max(u_L + c_L, u_R + c_R);

    // Contact wave speed using standard HLLC formula for shallow water
    // S_star = [h_L*u_L*(S_L - u_L) - h_R*u_R*(S_R - u_R) + 0.5*g*(h_R^2 -
    // h_L^2)]
    //          / [h_L*(S_L - u_L) - h_R*(S_R - u_R)]
    Real denom = h_L * (S_L - u_L) - h_R * (S_R - u_R);
    if (std::abs(denom) > 1e-10) {
        Real numer = h_L * u_L * (S_L - u_L) - h_R * u_R * (S_R - u_R) +
                     0.5 * g_ * (h_R * h_R - h_L * h_L);
        S_star = numer / denom;
    } else {
        S_star = 0.5 * (u_L + u_R);
    }
}

inline Vec3 ShallowWaterHLLCFlux::physical_flux(
    Real h, Real u, Real v, Real g, Real nx, Real ny) const {

    Real un = u * nx + v * ny;
    Vec3 F;
    F(0) = h * un;
    F(1) = h * u * un + 0.5 * g * h * h * nx;
    F(2) = h * v * un + 0.5 * g * h * h * ny;
    return F;
}

} // namespace drifter
