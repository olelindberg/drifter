#pragma once

// Mode Splitting for Ocean Circulation Models
//
// The barotropic/baroclinic mode splitting separates the fast surface gravity
// waves (barotropic mode) from the slow internal dynamics (baroclinic mode).
//
// Barotropic mode (2D, depth-averaged):
//   deta/dt + div(H * U_bar) = 0
//   d(H*U_bar)/dt + ... = -gH*grad(eta) + integral(baroclinic forces)
//
// Baroclinic mode (3D, deviations from depth-average):
//   u' = u - U_bar
//   Driven by baroclinic pressure gradient, Coriolis, diffusion
//
// Time stepping:
//   - Barotropic: small dt_baro (due to fast surface waves)
//   - Baroclinic: large dt_3D = N * dt_baro (where N ~ 10-100)
//
// The barotropic equations are subcycled within each 3D time step.

#include "core/types.hpp"
#include "dg/basis_hexahedron.hpp"
#include "dg/operators_3d.hpp"
#include "dg/quadrature_3d.hpp"
#include "physics/primitive_equations.hpp"
#include <memory>
#include <vector>

namespace drifter {

/// @brief Parameters for mode splitting
struct ModeSplittingParams {
    int subcycles = 30; ///< Number of barotropic subcycles per baroclinic step
    Real barotropic_cfl = 0.8; ///< CFL for barotropic mode
    bool use_predictor = true; ///< Use predictor-corrector for barotropic
    bool forward_averaging =
        true; ///< Average barotropic solution forward in time
};

/// @brief Barotropic (depth-averaged) state
struct BarotropicState {
    VecX eta;    ///< Free surface elevation
    VecX HU_bar; ///< Depth-integrated x-transport
    VecX HV_bar; ///< Depth-integrated y-transport

    // Derived
    VecX H;     ///< Water depth (eta + h)
    VecX U_bar; ///< Depth-averaged x-velocity
    VecX V_bar; ///< Depth-averaged y-velocity

    void resize(int n);
    void update_derived(const VecX &h);
};

/// @brief Baroclinic (3D deviation) state
struct BaroclinicState {
    VecX u_prime; ///< Baroclinic x-velocity (u - U_bar)
    VecX v_prime; ///< Baroclinic y-velocity (v - V_bar)
    VecX HT;      ///< Heat content
    VecX HS;      ///< Salt content
    VecX omega;   ///< Sigma velocity

    void resize(int n);
};

/// @brief Tendencies for barotropic mode
struct BarotropicTendencies {
    VecX deta_dt;
    VecX dHU_bar_dt;
    VecX dHV_bar_dt;

    void resize(int n);
    void set_zero();
};

/// @brief Mode splitting time stepper for a single element
class ModeSplittingElement {
public:
    /// @brief Construct mode splitting element
    ModeSplittingElement(
        const HexahedronBasis &basis, const GaussQuadrature3D &quad,
        const OceanConstants &constants = OceanConstants());

    /// @brief Set bathymetry
    void set_bathymetry(const VecX &h, const VecX &dh_dx, const VecX &dh_dy);

    /// @brief Get bathymetry
    const VecX &bathymetry() const { return h_; }

    /// @brief Set Coriolis parameter
    void set_coriolis(const VecX &f);

    /// @brief Compute 2D horizontal divergence
    /// @param HU_bar Depth-integrated x-transport
    /// @param HV_bar Depth-integrated y-transport
    /// @return Divergence at horizontal DOFs
    VecX compute_2d_divergence(const VecX &HU_bar, const VecX &HV_bar) const;

    // =========================================================================
    // Vertical integration
    // =========================================================================

    /// @brief Compute depth-averaged velocity from 3D field
    /// @param u_3d 3D velocity field
    /// @param[out] U_bar Depth-averaged velocity (horizontal DOFs)
    void depth_average(const VecX &u_3d, VecX &U_bar) const;

    /// @brief Compute depth-integrated transport
    /// @param u_3d 3D velocity field
    /// @param H Water depth at DOFs
    /// @param[out] HU_bar Depth-integrated transport
    void depth_integrate(const VecX &u_3d, const VecX &H, VecX &HU_bar) const;

    /// @brief Compute baroclinic deviation
    /// @param u_3d 3D velocity
    /// @param U_bar Depth-averaged velocity
    /// @param[out] u_prime Baroclinic deviation (u - U_bar)
    void
    compute_deviation(const VecX &u_3d, const VecX &U_bar, VecX &u_prime) const;

    // =========================================================================
    // Barotropic RHS
    // =========================================================================

    /// @brief Compute barotropic RHS for free surface
    /// @details deta/dt = -div(H * U_bar)
    void barotropic_eta_rhs(const BarotropicState &state, VecX &deta_dt) const;

    /// @brief Compute barotropic RHS for momentum
    /// @details d(HU_bar)/dt = -gH*grad(eta) + f*HV_bar + ...
    void barotropic_momentum_rhs(
        const BarotropicState &state,
        const VecX &forcing_x, // From baroclinic
        const VecX &forcing_y, VecX &dHU_bar_dt, VecX &dHV_bar_dt) const;

    /// @brief Compute full barotropic RHS
    void barotropic_rhs(
        const BarotropicState &state, const VecX &forcing_x,
        const VecX &forcing_y, BarotropicTendencies &tendency) const;

    // =========================================================================
    // Baroclinic forcing
    // =========================================================================

    /// @brief Compute baroclinic pressure gradient forcing
    /// @details Integral of (1/rho_0) * grad_h(p') over depth
    ///          where p' is baroclinic pressure
    void baroclinic_pressure_forcing(
        const VecX &rho, const VecX &H, VecX &forcing_x, VecX &forcing_y) const;

    /// @brief Compute bottom stress contribution to barotropic
    void bottom_stress_forcing(
        const VecX &u_bot, const VecX &v_bot, Real Cd, VecX &tau_x,
        VecX &tau_y) const;

    /// @brief Compute wind stress contribution
    void wind_stress_forcing(
        const VecX &tau_wind_x, const VecX &tau_wind_y, const VecX &H,
        VecX &forcing_x, VecX &forcing_y) const;

    // =========================================================================
    // Time stepping
    // =========================================================================

    /// @brief Single barotropic step (forward Euler)
    void barotropic_step_euler(
        Real dt, const VecX &forcing_x, const VecX &forcing_y,
        BarotropicState &state) const;

    /// @brief Barotropic predictor-corrector step
    void barotropic_step_predictor_corrector(
        Real dt, const VecX &forcing_x, const VecX &forcing_y,
        BarotropicState &state) const;

    /// @brief Subcycle barotropic equations
    /// @param dt_3d Baroclinic time step
    /// @param n_subcycles Number of barotropic subcycles
    /// @param forcing_x, forcing_y Baroclinic forcing (constant during
    /// subcycling)
    /// @param state Barotropic state (updated in place)
    /// @param eta_avg Time-averaged eta over subcycling (for 3D pressure
    /// gradient)
    void subcycle(
        Real dt_3d, int n_subcycles, const VecX &forcing_x,
        const VecX &forcing_y, BarotropicState &state, VecX &eta_avg) const;

private:
    const HexahedronBasis &basis_;
    const GaussQuadrature3D &quad_;
    OceanConstants constants_;

    VecX h_; // Bathymetry
    VecX dh_dx_;
    VecX dh_dy_;
    VecX f_; // Coriolis

    int n_horiz_; // Horizontal DOFs per element
    int n_vert_;  // Vertical levels

    // Vertical integration weights (LGL weights scaled for sigma [-1,0])
    VecX sigma_weights_;

    // 2D DG operators for barotropic equations
    // (Horizontal gradients and divergence)
    MatX D_x_2d_; // Horizontal x-derivative (on 2D slice)
    MatX D_y_2d_; // Horizontal y-derivative

    void build_2d_operators();
};

/// @brief Global mode splitting solver
class ModeSplittingSolver {
public:
    /// @brief Construct solver
    ModeSplittingSolver(
        int order, const OceanConstants &constants = OceanConstants(),
        const ModeSplittingParams &params = ModeSplittingParams());

    /// @brief Initialize with mesh
    void initialize(
        int num_elements, const std::vector<VecX> &bathymetry,
        const std::vector<VecX> &dh_dx, const std::vector<VecX> &dh_dy,
        const std::vector<VecX> &coriolis);

    /// @brief Set splitting parameters
    void set_params(const ModeSplittingParams &params) { params_ = params; }

    /// @brief Get splitting parameters
    const ModeSplittingParams &params() const { return params_; }

    // =========================================================================
    // Decomposition
    // =========================================================================

    /// @brief Split 3D state into barotropic and baroclinic components
    void decompose(
        const std::vector<PrimitiveState> &full_state,
        std::vector<BarotropicState> &baro_state,
        std::vector<BaroclinicState> &clinic_state) const;

    /// @brief Recombine barotropic and baroclinic into full 3D state
    void recombine(
        const std::vector<BarotropicState> &baro_state,
        const std::vector<BaroclinicState> &clinic_state,
        std::vector<PrimitiveState> &full_state) const;

    // =========================================================================
    // Time stepping
    // =========================================================================

    /// @brief Perform one full 3D time step with mode splitting
    /// @param dt_3d Baroclinic time step
    /// @param full_state Full 3D state (updated in place)
    void step(Real dt_3d, std::vector<PrimitiveState> &full_state);

    /// @brief Compute barotropic CFL time step
    Real compute_barotropic_dt(
        const std::vector<BarotropicState> &states, Real dx_min) const;

    /// @brief Compute baroclinic CFL time step
    Real compute_baroclinic_dt(
        const std::vector<PrimitiveState> &states, Real dx_min) const;

private:
    int order_;
    OceanConstants constants_;
    ModeSplittingParams params_;

    HexahedronBasis basis_;
    GaussQuadrature3D quad_;

    std::vector<std::unique_ptr<ModeSplittingElement>> elements_;

    // Cached forcing terms
    std::vector<VecX> baroclinic_forcing_x_;
    std::vector<VecX> baroclinic_forcing_y_;

    /// @brief Compute baroclinic forcing for barotropic
    void compute_baroclinic_forcing(
        const std::vector<BaroclinicState> &clinic_state,
        const std::vector<PrimitiveState> &full_state);

    /// @brief Subcycle barotropic equations for all elements
    void subcycle_barotropic(
        Real dt_3d, std::vector<BarotropicState> &baro_state,
        std::vector<VecX> &eta_avg);

    /// @brief Update baroclinic state
    void update_baroclinic(
        Real dt_3d, const std::vector<VecX> &eta_avg,
        std::vector<BaroclinicState> &clinic_state,
        std::vector<PrimitiveState> &full_state);
};

} // namespace drifter
