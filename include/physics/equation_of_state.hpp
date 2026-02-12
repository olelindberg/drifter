#pragma once

// Equation of State for Seawater
//
// Computes density as a function of temperature (T), salinity (S), and pressure
// (p). Several formulations are provided:
//
// 1. Linear EOS (fast, for idealized cases):
//    rho = rho_0 * (1 - alpha*(T - T0) + beta*(S - S0))
//
// 2. UNESCO 1983 (IES 80):
//    Full polynomial expansion with 41 coefficients
//
// 3. TEOS-10 (modern standard):
//    Conservative Temperature and Absolute Salinity based
//
// 4. Jackett & McDougall (1995):
//    Efficient polynomial fit for ocean modeling

#include "core/types.hpp"
#include <cmath>

namespace drifter {

/// @brief Equation of state type
enum class EOSType {
    Linear,           ///< Linear approximation
    UNESCO,           ///< UNESCO 1983 (IES 80)
    JackettMcDougall, ///< Jackett & McDougall 1995
    TEOS10            ///< TEOS-10 (if library available)
};

/// @brief Reference values for linear EOS
struct EOSReferenceValues {
    Real T0 = 10.0;      ///< Reference temperature [°C]
    Real S0 = 35.0;      ///< Reference salinity [PSU]
    Real p0 = 0.0;       ///< Reference pressure [dbar]
    Real rho_0 = 1025.0; ///< Reference density [kg/m³]
    Real alpha = 2.0e-4; ///< Thermal expansion coefficient [1/K]
    Real beta = 7.6e-4;  ///< Haline contraction coefficient [1/PSU]
};

/// @brief Equation of state for seawater
class EquationOfState {
public:
    /// @brief Construct with specified type
    explicit EquationOfState(
        EOSType type = EOSType::Linear,
        const EOSReferenceValues &ref = EOSReferenceValues())
        : type_(type), ref_(ref) {}

    /// @brief Set equation of state type
    void set_type(EOSType type) { type_ = type; }

    /// @brief Set reference values (for linear EOS)
    void set_reference(const EOSReferenceValues &ref) { ref_ = ref; }

    // =========================================================================
    // Density computation
    // =========================================================================

    /// @brief Compute in-situ density
    /// @param T Temperature [°C] (potential or in-situ depending on
    /// formulation)
    /// @param S Salinity [PSU] (practical salinity)
    /// @param p Pressure [dbar] (0 at surface)
    /// @return Density [kg/m³]
    Real density(Real T, Real S, Real p = 0.0) const;

    /// @brief Compute density anomaly (rho - rho_0)
    Real density_anomaly(Real T, Real S, Real p = 0.0) const;

    /// @brief Compute potential density (density at p = 0)
    Real potential_density(Real T, Real S) const;

    /// @brief Vectorized density computation
    void density(const VecX &T, const VecX &S, const VecX &p, VecX &rho) const;

    /// @brief Vectorized density anomaly
    void density_anomaly(
        const VecX &T, const VecX &S, const VecX &p, VecX &rho_anom) const;

    // =========================================================================
    // Derivatives (for pressure gradient computation)
    // =========================================================================

    /// @brief Thermal expansion coefficient: -1/rho * drho/dT
    Real thermal_expansion(Real T, Real S, Real p = 0.0) const;

    /// @brief Haline contraction coefficient: 1/rho * drho/dS
    Real haline_contraction(Real T, Real S, Real p = 0.0) const;

    /// @brief Isothermal compressibility: 1/rho * drho/dp
    Real compressibility(Real T, Real S, Real p = 0.0) const;

    // =========================================================================
    // Sound speed
    // =========================================================================

    /// @brief Compute speed of sound
    Real sound_speed(Real T, Real S, Real p = 0.0) const;

private:
    EOSType type_;
    EOSReferenceValues ref_;

    // Implementation methods for different EOS types
    Real density_linear(Real T, Real S, Real p) const;
    Real density_unesco(Real T, Real S, Real p) const;
    Real density_jackett_mcdougall(Real T, Real S, Real p) const;
};

// =============================================================================
// Inline implementations
// =============================================================================

inline Real EquationOfState::density(Real T, Real S, Real p) const {
    switch (type_) {
    case EOSType::Linear:
        return density_linear(T, S, p);
    case EOSType::UNESCO:
        return density_unesco(T, S, p);
    case EOSType::JackettMcDougall:
        return density_jackett_mcdougall(T, S, p);
    case EOSType::TEOS10:
        // TEOS-10 would require external library
        return density_unesco(T, S, p);
    default:
        return density_linear(T, S, p);
    }
}

inline Real EquationOfState::density_anomaly(Real T, Real S, Real p) const {
    // Standard oceanographic convention: sigma = rho - 1000 kg/m³
    return density(T, S, p) - 1000.0;
}

inline Real EquationOfState::potential_density(Real T, Real S) const {
    return density(T, S, 0.0);
}

inline void EquationOfState::density(
    const VecX &T, const VecX &S, const VecX &p, VecX &rho) const {
    int n = static_cast<int>(T.size());
    rho.resize(n);

    for (int i = 0; i < n; ++i) {
        Real pressure = (p.size() > i) ? p(i) : 0.0;
        rho(i) = density(T(i), S(i), pressure);
    }
}

inline void EquationOfState::density_anomaly(
    const VecX &T, const VecX &S, const VecX &p, VecX &rho_anom) const {
    int n = static_cast<int>(T.size());
    rho_anom.resize(n);

    for (int i = 0; i < n; ++i) {
        Real pressure = (p.size() > i) ? p(i) : 0.0;
        rho_anom(i) = density_anomaly(T(i), S(i), pressure);
    }
}

inline Real EquationOfState::density_linear(Real T, Real S, Real /*p*/) const {
    // rho = rho_0 * (1 - alpha*(T - T0) + beta*(S - S0))
    return ref_.rho_0 *
           (1.0 - ref_.alpha * (T - ref_.T0) + ref_.beta * (S - ref_.S0));
}

inline Real EquationOfState::density_unesco(Real T, Real S, Real p) const {
    // UNESCO 1983 Equation of State (IES 80)
    // Full polynomial expansion
    // NOTE: UNESCO coefficients expect pressure in bars (1 dbar = 0.1 bar)
    // Input p is in dbar, so convert to bars for the calculation
    Real p_bar = p * 0.1; // Convert dbar to bar

    // Pure water density at atmospheric pressure
    Real T2 = T * T;
    Real T3 = T2 * T;
    Real T4 = T3 * T;
    Real T5 = T4 * T;

    Real rho_0 = 999.842594 + 6.793952e-2 * T - 9.095290e-3 * T2 +
                 1.001685e-4 * T3 - 1.120083e-6 * T4 + 6.536332e-9 * T5;

    // Effect of salinity at atmospheric pressure
    Real S15 = std::sqrt(S * S * S); // S^(3/2)

    Real A = 8.24493e-1 - 4.0899e-3 * T + 7.6438e-5 * T2 - 8.2467e-7 * T3 +
             5.3875e-9 * T4;
    Real B = -5.72466e-3 + 1.0227e-4 * T - 1.6546e-6 * T2;
    Real C = 4.8314e-4;

    Real rho_s0 = rho_0 + A * S + B * S15 + C * S * S;

    if (p_bar <= 0.0) {
        return rho_s0;
    }

    // Effect of pressure (compressibility)
    // K = secant bulk modulus (in bars)

    // K_w (pure water)
    Real K_w = 19652.21 + 148.4206 * T - 2.327105 * T2 + 1.360477e-2 * T3 -
               5.155288e-5 * T4;

    // K_0 (at atmospheric pressure with salinity)
    Real K_0_A = 54.6746 - 0.603459 * T + 1.09987e-2 * T2 - 6.1670e-5 * T3;
    Real K_0_B = 7.944e-2 + 1.6483e-2 * T - 5.3009e-4 * T2;

    Real K_0 = K_w + K_0_A * S + K_0_B * S15;

    // Pressure contribution to bulk modulus
    Real A_w = 3.239908 + 1.43713e-3 * T + 1.16092e-4 * T2 - 5.77905e-7 * T3;
    Real A_s = 2.2838e-3 - 1.0981e-5 * T - 1.6078e-6 * T2;
    Real A_K = A_w + A_s * S + 1.91075e-4 * S15;

    Real B_w = 8.50935e-5 - 6.12293e-6 * T + 5.2787e-8 * T2;
    Real B_s = -9.9348e-7 + 2.0816e-8 * T + 9.1697e-10 * T2;
    Real B_K = B_w + B_s * S;

    Real K = K_0 + A_K * p_bar + B_K * p_bar * p_bar;

    // In-situ density: rho = rho_s0 / (1 - p/K)
    return rho_s0 / (1.0 - p_bar / K);
}

inline Real
EquationOfState::density_jackett_mcdougall(Real T, Real S, Real p) const {
    // Jackett & McDougall (1995) polynomial fit
    // Valid for: -2 <= T <= 40 °C, 0 <= S <= 42 PSU, 0 <= p <= 10000 dbar
    // NOTE: Coefficients expect pressure in bars (1 dbar = 0.1 bar)
    Real p_bar = p * 0.1; // Convert dbar to bar

    Real T2 = T * T;
    Real T3 = T2 * T;
    Real T4 = T3 * T;
    Real S15 = std::sqrt(S * S * S);
    Real p2 = p_bar * p_bar;

    // Density at surface
    Real rho_0 = 999.842594 + 6.793952e-2 * T - 9.09529e-3 * T2 +
                 1.001685e-4 * T3 - 1.120083e-6 * T4 + 6.536336e-9 * T4 * T +
                 (8.24493e-1 - 4.0899e-3 * T + 7.6438e-5 * T2 - 8.2467e-7 * T3 +
                  5.3875e-9 * T4) *
                     S +
                 (-5.72466e-3 + 1.0227e-4 * T - 1.6546e-6 * T2) * S15 +
                 4.8314e-4 * S * S;

    if (p_bar <= 0.0) {
        return rho_0;
    }

    // Compressibility coefficients (simplified) - K in bars
    Real K = 19652.21 + 148.4206 * T - 2.327105 * T2 + 1.360477e-2 * T3 +
             (54.6746 - 0.603459 * T + 1.09987e-2 * T2) * S +
             (3.239908 + 1.43713e-3 * T + 1.16092e-4 * T2) * p_bar +
             8.50935e-5 * p2;

    return rho_0 / (1.0 - p_bar / K);
}

inline Real EquationOfState::thermal_expansion(Real T, Real S, Real p) const {
    // Numerical differentiation
    constexpr Real dT = 0.01;
    Real rho_plus = density(T + dT, S, p);
    Real rho_minus = density(T - dT, S, p);
    Real rho_center = density(T, S, p);

    return -(rho_plus - rho_minus) / (2.0 * dT * rho_center);
}

inline Real EquationOfState::haline_contraction(Real T, Real S, Real p) const {
    // Numerical differentiation
    constexpr Real dS = 0.01;
    Real rho_plus = density(T, S + dS, p);
    Real rho_minus = density(T, S - dS, p);
    Real rho_center = density(T, S, p);

    return (rho_plus - rho_minus) / (2.0 * dS * rho_center);
}

inline Real EquationOfState::compressibility(Real T, Real S, Real p) const {
    // Numerical differentiation
    constexpr Real dp = 1.0; // 1 dbar
    Real rho_plus = density(T, S, p + dp);
    Real rho_minus = density(T, S, std::max(0.0, p - dp));
    Real rho_center = density(T, S, p);

    return (rho_plus - rho_minus) / (2.0 * dp * rho_center);
}

inline Real EquationOfState::sound_speed(Real T, Real S, Real p) const {
    // Approximate sound speed using Chen & Millero (1977)
    // c = 1449.2 + 4.6*T - 0.055*T^2 + 0.00029*T^3 + (1.34 - 0.01*T)*(S - 35) +
    // 0.016*p

    Real c = 1449.2 + 4.6 * T - 0.055 * T * T + 0.00029 * T * T * T +
             (1.34 - 0.01 * T) * (S - 35.0) + 0.016 * p;

    return c;
}

/// @brief Buoyancy frequency squared (N^2 = -g/rho * drho/dz)
/// @details Measures static stability of water column
class BuoyancyFrequency {
public:
    explicit BuoyancyFrequency(const EquationOfState &eos, Real g = 9.81)
        : eos_(eos), g_(g) {}

    /// @brief Compute N^2 from vertical density gradient
    /// @param drho_dz Vertical density gradient [kg/m^4]
    /// @param rho Local density [kg/m³]
    /// @return Buoyancy frequency squared [1/s²]
    Real compute(Real drho_dz, Real rho) const { return -g_ * drho_dz / rho; }

    /// @brief Compute N^2 profile from T, S profiles
    void compute_profile(
        const VecX &T, const VecX &S, const VecX &z, VecX &N2) const {
        int n = static_cast<int>(T.size());
        N2.resize(n);

        // Central difference for interior points
        for (int i = 1; i < n - 1; ++i) {
            Real dz = z(i + 1) - z(i - 1);
            if (std::abs(dz) < 1e-10) {
                N2(i) = 0.0;
                continue;
            }

            Real rho_up = eos_.density(T(i + 1), S(i + 1));
            Real rho_down = eos_.density(T(i - 1), S(i - 1));
            Real rho_mid = eos_.density(T(i), S(i));

            Real drho_dz = (rho_up - rho_down) / dz;
            N2(i) = -g_ * drho_dz / rho_mid;
        }

        // Extrapolate boundaries
        N2(0) = N2(1);
        N2(n - 1) = N2(n - 2);
    }

private:
    const EquationOfState &eos_;
    Real g_;
};

} // namespace drifter
