// Integration tests for initial conditions

#include <gtest/gtest.h>
#include "solver/simulation_driver.hpp"
#include "test_integration_fixtures.hpp"

using namespace drifter;
using namespace drifter::testing;

TEST_F(SimulationTest, QuiescentInitialCondition) {
    int num_elements = 4;
    int dofs_per_element = 64;  // Order 3: 4^3

    auto states = create_quiescent_initial_condition(
        num_elements, dofs_per_element, 15.0, 35.0);

    EXPECT_EQ(states.size(), num_elements);

    for (const auto& state : states) {
        EXPECT_EQ(state.Hu.size(), dofs_per_element);

        // Check quiescent conditions
        EXPECT_NEAR(state.Hu.norm(), 0.0, TOLERANCE);
        EXPECT_NEAR(state.Hv.norm(), 0.0, TOLERANCE);
        EXPECT_NEAR(state.eta.norm(), 0.0, TOLERANCE);

        // Check constant T and S
        EXPECT_NEAR(state.T(0), 15.0, TOLERANCE);
        EXPECT_NEAR(state.S(0), 35.0, TOLERANCE);
    }
}

TEST_F(SimulationTest, KelvinWaveInitialCondition) {
    int num_elements = 4;
    int dofs_per_element = 64;

    // Create position arrays
    std::vector<VecX> x_pos(num_elements), y_pos(num_elements);
    for (int e = 0; e < num_elements; ++e) {
        x_pos[e].resize(dofs_per_element);
        y_pos[e].resize(dofs_per_element);
        for (int i = 0; i < dofs_per_element; ++i) {
            x_pos[e](i) = e * 25000.0 + i * 100.0;  // x position
            y_pos[e](i) = 10000.0;  // y position (offshore)
        }
    }

    Real amplitude = 0.1;
    Real wavelength = 100000.0;
    Real depth = 100.0;

    auto states = create_kelvin_wave_initial_condition(
        num_elements, dofs_per_element, x_pos, y_pos,
        amplitude, wavelength, depth);

    EXPECT_EQ(states.size(), num_elements);

    for (const auto& state : states) {
        // Kelvin wave should have non-zero eta
        EXPECT_GT(state.eta.cwiseAbs().maxCoeff(), 0.0);

        // Along-shore velocity (u) should be non-zero
        EXPECT_GT(state.u.cwiseAbs().maxCoeff(), 0.0);

        // Cross-shore velocity (v) should be zero for Kelvin wave
        EXPECT_NEAR(state.v.norm(), 0.0, LOOSE_TOLERANCE);
    }
}

TEST_F(SimulationTest, LockExchangeInitialCondition) {
    int num_elements = 4;
    int dofs_per_element = 64;

    // Create position arrays
    std::vector<VecX> x_pos(num_elements);
    for (int e = 0; e < num_elements; ++e) {
        x_pos[e].resize(dofs_per_element);
        for (int i = 0; i < dofs_per_element; ++i) {
            x_pos[e](i) = (e * dofs_per_element + i) /
                          static_cast<Real>(num_elements * dofs_per_element);
        }
    }

    Real T_cold = 10.0;
    Real T_warm = 20.0;
    Real x_interface = 0.5;

    auto states = create_lock_exchange_initial_condition(
        num_elements, dofs_per_element, x_pos, T_cold, T_warm, x_interface);

    EXPECT_EQ(states.size(), num_elements);

    // Check that we have both cold and warm water
    Real min_T = 100.0, max_T = -100.0;
    for (const auto& state : states) {
        min_T = std::min(min_T, state.T.minCoeff());
        max_T = std::max(max_T, state.T.maxCoeff());
    }

    EXPECT_NEAR(min_T, T_cold, LOOSE_TOLERANCE);
    EXPECT_NEAR(max_T, T_warm, LOOSE_TOLERANCE);
}
