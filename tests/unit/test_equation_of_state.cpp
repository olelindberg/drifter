#include <gtest/gtest.h>
#include "physics/equation_of_state.hpp"
#include "../test_utils.hpp"
#include <cmath>

using namespace drifter;
using namespace drifter::testing;

class EquationOfStateTest : public DrifterTestBase {
protected:
    void SetUp() override {
        DrifterTestBase::SetUp();
    }
};

// Test reference density
TEST_F(EquationOfStateTest, ReferenceDensity) {
    // All EOS should give approximately 1025 kg/m³ for typical seawater
    Real T = 10.0;  // 10°C
    Real S = 35.0;  // 35 PSU

    EquationOfState linear_eos(EOSType::Linear);
    EquationOfState unesco_eos(EOSType::UNESCO);
    EquationOfState jm_eos(EOSType::JackettMcDougall);

    Real rho_linear = linear_eos.density(T, S);
    Real rho_unesco = unesco_eos.density(T, S);
    Real rho_jm = jm_eos.density(T, S);

    // All should be within 5 kg/m³ of 1025
    EXPECT_NEAR(rho_linear, 1025.0, 10.0);
    EXPECT_NEAR(rho_unesco, 1027.0, 5.0);  // UNESCO is more accurate
    EXPECT_NEAR(rho_jm, 1027.0, 5.0);
}

// Test that density increases with salinity
TEST_F(EquationOfStateTest, DensityIncreaseWithSalinity) {
    Real T = 15.0;

    std::vector<EOSType> types = {
        EOSType::Linear,
        EOSType::UNESCO,
        EOSType::JackettMcDougall
    };

    for (auto type : types) {
        EquationOfState eos(type);

        Real rho_low_S = eos.density(T, 30.0);
        Real rho_high_S = eos.density(T, 40.0);

        EXPECT_GT(rho_high_S, rho_low_S)
            << "EOS type " << static_cast<int>(type);
    }
}

// Test that density decreases with temperature (for T > 4°C)
TEST_F(EquationOfStateTest, DensityDecreaseWithTemperature) {
    Real S = 35.0;

    std::vector<EOSType> types = {
        EOSType::Linear,
        EOSType::UNESCO,
        EOSType::JackettMcDougall
    };

    for (auto type : types) {
        EquationOfState eos(type);

        Real rho_cold = eos.density(10.0, S);
        Real rho_warm = eos.density(25.0, S);

        EXPECT_GT(rho_cold, rho_warm)
            << "EOS type " << static_cast<int>(type);
    }
}

// Test thermal expansion coefficient is positive (for typical ocean temps)
TEST_F(EquationOfStateTest, ThermalExpansionPositive) {
    Real T = 15.0;
    Real S = 35.0;

    std::vector<EOSType> types = {
        EOSType::Linear,
        EOSType::UNESCO,
        EOSType::JackettMcDougall
    };

    for (auto type : types) {
        EquationOfState eos(type);

        Real alpha = eos.thermal_expansion(T, S);

        // Thermal expansion should be positive for T > 4°C
        EXPECT_GT(alpha, 0.0)
            << "EOS type " << static_cast<int>(type);

        // Typical value around 1-3 × 10^-4 K^-1
        EXPECT_GT(alpha, 0.5e-4)
            << "EOS type " << static_cast<int>(type);
        EXPECT_LT(alpha, 5.0e-4)
            << "EOS type " << static_cast<int>(type);
    }
}

// Test haline contraction coefficient is positive
TEST_F(EquationOfStateTest, HalineContractionPositive) {
    Real T = 15.0;
    Real S = 35.0;

    std::vector<EOSType> types = {
        EOSType::Linear,
        EOSType::UNESCO,
        EOSType::JackettMcDougall
    };

    for (auto type : types) {
        EquationOfState eos(type);

        Real beta = eos.haline_contraction(T, S);

        // Haline contraction should be positive
        EXPECT_GT(beta, 0.0)
            << "EOS type " << static_cast<int>(type);

        // Typical value around 7 × 10^-4 PSU^-1
        EXPECT_GT(beta, 5.0e-4)
            << "EOS type " << static_cast<int>(type);
        EXPECT_LT(beta, 10.0e-4)
            << "EOS type " << static_cast<int>(type);
    }
}

// Test linear EOS coefficients
TEST_F(EquationOfStateTest, LinearEOSCoefficients) {
    EOSReferenceValues ref;
    ref.rho_0 = 1025.0;
    ref.alpha = 2.0e-4;
    ref.beta = 7.5e-4;
    ref.T0 = 10.0;
    ref.S0 = 35.0;

    EquationOfState eos(EOSType::Linear, ref);

    // At reference T, S: rho = rho0
    EXPECT_NEAR(eos.density(ref.T0, ref.S0), ref.rho_0, TOLERANCE);

    // Increase T by 1: rho decreases by alpha * rho0
    Real rho_T_plus = eos.density(ref.T0 + 1.0, ref.S0);
    EXPECT_NEAR(rho_T_plus, ref.rho_0 * (1.0 - ref.alpha), LOOSE_TOLERANCE);

    // Increase S by 1: rho increases by beta * rho0
    Real rho_S_plus = eos.density(ref.T0, ref.S0 + 1.0);
    EXPECT_NEAR(rho_S_plus, ref.rho_0 * (1.0 + ref.beta), LOOSE_TOLERANCE);
}

// Test density anomaly (sigma)
TEST_F(EquationOfStateTest, DensityAnomaly) {
    Real T = 15.0;
    Real S = 35.0;

    EquationOfState eos(EOSType::UNESCO);

    Real rho = eos.density(T, S);
    Real sigma = eos.density_anomaly(T, S);

    // sigma = rho - 1000
    EXPECT_NEAR(sigma, rho - 1000.0, TOLERANCE);
}

// Test UNESCO EOS at known values
TEST_F(EquationOfStateTest, UNESCOKnownValues) {
    EquationOfState eos(EOSType::UNESCO);

    // Standard seawater at 10°C, 35 PSU, surface pressure
    // Reference: UNESCO 1983
    Real rho = eos.density(10.0, 35.0, 0.0);

    // Should be approximately 1026.952 kg/m³
    EXPECT_NEAR(rho, 1026.95, 0.1);
}

// Test sound speed
TEST_F(EquationOfStateTest, SoundSpeed) {
    EquationOfState eos(EOSType::UNESCO);

    Real T = 10.0;
    Real S = 35.0;
    Real p = 0.0;

    Real c = eos.sound_speed(T, S, p);

    // Sound speed in seawater should be around 1500 m/s
    EXPECT_GT(c, 1480.0);
    EXPECT_LT(c, 1550.0);
}

// Test that pressure increases density
TEST_F(EquationOfStateTest, PressureEffect) {
    EquationOfState eos(EOSType::UNESCO);

    Real T = 10.0;
    Real S = 35.0;

    Real rho_surface = eos.density(T, S, 0.0);
    Real rho_deep = eos.density(T, S, 4000.0);  // 4000 dbar ≈ 4000m

    // Deep water should be denser due to compression
    EXPECT_GT(rho_deep, rho_surface);

    // Difference should be roughly 2% for 4000m
    Real relative_diff = (rho_deep - rho_surface) / rho_surface;
    EXPECT_GT(relative_diff, 0.01);
    EXPECT_LT(relative_diff, 0.03);
}

// Test potential density
TEST_F(EquationOfStateTest, PotentialDensity) {
    EquationOfState eos(EOSType::UNESCO);

    Real T = 10.0;
    Real S = 35.0;

    Real rho_pot = eos.potential_density(T, S);

    // Potential density should be similar to surface density
    Real rho_surface = eos.density(T, S, 0.0);
    EXPECT_NEAR(rho_pot, rho_surface, 1.0);
}
