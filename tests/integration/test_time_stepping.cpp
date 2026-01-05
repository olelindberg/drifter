// Integration tests for time stepping algorithms

#include <gtest/gtest.h>
#include "solver/time_stepper.hpp"
#include "test_integration_fixtures.hpp"
#include <cmath>

using namespace drifter;
using namespace drifter::testing;

TEST_F(SimulationTest, TimeStepperRK3SSP) {
    // Test that SSP-RK3 correctly advances a simple ODE
    // dy/dt = -y, y(0) = 1  =>  y(t) = exp(-t)

    auto stepper = TimeStepper::rk3_ssp();
    ASSERT_EQ(stepper->order(), 3);
    ASSERT_EQ(stepper->num_stages(), 3);

    VecX y(1);
    y(0) = 1.0;

    auto rhs = [](Real t, const VecX& U, VecX& dUdt) {
        dUdt = -U;
    };

    Real t = 0.0;
    Real dt = 0.01;
    int n_steps = 100;  // Integrate to t=1

    for (int i = 0; i < n_steps; ++i) {
        stepper->step(t, dt, y, rhs);
        t += dt;
    }

    Real expected = std::exp(-1.0);
    EXPECT_NEAR(y(0), expected, 1e-4);  // 3rd order should be accurate
}

TEST_F(SimulationTest, TimeStepperRK4) {
    // Test classic RK4
    auto stepper = TimeStepper::rk4_classic();
    ASSERT_EQ(stepper->order(), 4);

    VecX y(1);
    y(0) = 1.0;

    auto rhs = [](Real t, const VecX& U, VecX& dUdt) {
        dUdt = -U;
    };

    Real t = 0.0;
    Real dt = 0.1;  // Larger step, RK4 should handle it
    int n_steps = 10;

    for (int i = 0; i < n_steps; ++i) {
        stepper->step(t, dt, y, rhs);
        t += dt;
    }

    Real expected = std::exp(-1.0);
    EXPECT_NEAR(y(0), expected, 1e-6);  // 4th order should be very accurate
}

TEST_F(SimulationTest, AdaptiveTimeController) {
    AdaptiveTimeController controller(0.5, 1e-10, 1e6);

    // Test CFL-based time step computation
    Real max_wave_speed = 100.0;  // m/s (fast gravity wave)
    Real min_element_size = 1000.0;  // m

    Real dt = controller.compute_dt(max_wave_speed, min_element_size);

    // dt should be approximately CFL * dx / c = 0.5 * 1000 / 100 = 5s
    EXPECT_NEAR(dt, 5.0, 0.1);

    // Test with zero wave speed (shouldn't crash)
    Real dt_zero = controller.compute_dt(0.0, min_element_size);
    EXPECT_GT(dt_zero, 0.0);  // Should return dt_max
}
