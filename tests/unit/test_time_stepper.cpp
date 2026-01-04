#include <gtest/gtest.h>
#include "solver/time_stepper.hpp"
#include "../test_utils.hpp"
#include <cmath>

using namespace drifter;
using namespace drifter::testing;

class TimeStepperTest : public DrifterTestBase {
protected:
    void SetUp() override {
        DrifterTestBase::SetUp();
    }
};

// Test that Forward Euler converges with O(h) for simple ODE
TEST_F(TimeStepperTest, ForwardEulerConvergence) {
    // Test ODE: du/dt = -u, u(0) = 1, exact solution: u(t) = exp(-t)
    auto rhs = [](Real t, const VecX& U, VecX& dUdt) {
        dUdt = -U;
    };

    auto stepper = TimeStepper::forward_euler();
    EXPECT_EQ(stepper->order(), 1);
    EXPECT_EQ(stepper->num_stages(), 1);

    Real t_end = 1.0;
    Real exact = std::exp(-t_end);

    // Test convergence with decreasing dt
    Real error_prev = 1.0;
    for (Real dt : {0.1, 0.05, 0.025}) {
        VecX U(1);
        U(0) = 1.0;
        Real t = 0.0;

        while (t < t_end - 1e-10) {
            Real step = std::min(dt, t_end - t);
            stepper->step(t, step, U, rhs);
            t += step;
        }

        Real error = std::abs(U(0) - exact);

        // Error should decrease by factor of ~2 when dt halves (order 1)
        if (dt < 0.1) {
            Real ratio = error_prev / error;
            EXPECT_GT(ratio, 1.8);  // Should be ~2 for first order
            EXPECT_LT(ratio, 2.2);
        }
        error_prev = error;
    }
}

// Test RK2 midpoint convergence (O(h^2))
TEST_F(TimeStepperTest, RK2MidpointConvergence) {
    auto rhs = [](Real t, const VecX& U, VecX& dUdt) {
        dUdt = -U;
    };

    auto stepper = TimeStepper::rk2_midpoint();
    EXPECT_EQ(stepper->order(), 2);
    EXPECT_EQ(stepper->num_stages(), 2);

    Real t_end = 1.0;
    Real exact = std::exp(-t_end);

    Real error_prev = 1.0;
    for (Real dt : {0.1, 0.05, 0.025}) {
        VecX U(1);
        U(0) = 1.0;
        Real t = 0.0;

        while (t < t_end - 1e-10) {
            Real step = std::min(dt, t_end - t);
            stepper->step(t, step, U, rhs);
            t += step;
        }

        Real error = std::abs(U(0) - exact);

        // Error should decrease by factor of ~4 when dt halves (order 2)
        if (dt < 0.1) {
            Real ratio = error_prev / error;
            EXPECT_GT(ratio, 3.5);
            EXPECT_LT(ratio, 4.5);
        }
        error_prev = error;
    }
}

// Test SSP-RK3 convergence (O(h^3))
TEST_F(TimeStepperTest, SSPRK3Convergence) {
    auto rhs = [](Real t, const VecX& U, VecX& dUdt) {
        dUdt = -U;
    };

    auto stepper = TimeStepper::rk3_ssp();
    EXPECT_EQ(stepper->order(), 3);
    EXPECT_EQ(stepper->num_stages(), 3);

    Real t_end = 1.0;
    Real exact = std::exp(-t_end);

    Real error_prev = 1.0;
    for (Real dt : {0.2, 0.1, 0.05}) {
        VecX U(1);
        U(0) = 1.0;
        Real t = 0.0;

        while (t < t_end - 1e-10) {
            Real step = std::min(dt, t_end - t);
            stepper->step(t, step, U, rhs);
            t += step;
        }

        Real error = std::abs(U(0) - exact);

        // Error should decrease by factor of ~8 when dt halves (order 3)
        if (dt < 0.2) {
            Real ratio = error_prev / error;
            EXPECT_GT(ratio, 6.0);
            EXPECT_LT(ratio, 10.0);
        }
        error_prev = error;
    }
}

// Test Classic RK4 convergence (O(h^4))
TEST_F(TimeStepperTest, RK4Convergence) {
    auto rhs = [](Real t, const VecX& U, VecX& dUdt) {
        dUdt = -U;
    };

    auto stepper = TimeStepper::rk4_classic();
    EXPECT_EQ(stepper->order(), 4);
    EXPECT_EQ(stepper->num_stages(), 4);

    Real t_end = 1.0;
    Real exact = std::exp(-t_end);

    Real error_prev = 1.0;
    for (Real dt : {0.2, 0.1, 0.05}) {
        VecX U(1);
        U(0) = 1.0;
        Real t = 0.0;

        while (t < t_end - 1e-10) {
            Real step = std::min(dt, t_end - t);
            stepper->step(t, step, U, rhs);
            t += step;
        }

        Real error = std::abs(U(0) - exact);

        // Error should decrease by factor of ~16 when dt halves (order 4)
        if (dt < 0.2) {
            Real ratio = error_prev / error;
            EXPECT_GT(ratio, 12.0);
            EXPECT_LT(ratio, 20.0);
        }
        error_prev = error;
    }
}

// Test harmonic oscillator (system of ODEs)
TEST_F(TimeStepperTest, HarmonicOscillator) {
    // du/dt = v, dv/dt = -u
    // Exact: u(t) = cos(t), v(t) = -sin(t) for u(0)=1, v(0)=0
    auto rhs = [](Real t, const VecX& U, VecX& dUdt) {
        dUdt.resize(2);
        dUdt(0) = U(1);   // du/dt = v
        dUdt(1) = -U(0);  // dv/dt = -u
    };

    auto stepper = TimeStepper::rk4_classic();

    VecX U(2);
    U(0) = 1.0;  // u(0) = 1
    U(1) = 0.0;  // v(0) = 0

    Real t = 0.0;
    Real dt = 0.01;
    Real t_end = 2.0 * 3.14159265358979;  // One period

    while (t < t_end - 1e-10) {
        Real step = std::min(dt, t_end - t);
        stepper->step(t, step, U, rhs);
        t += step;
    }

    // After one period, should return to initial state
    EXPECT_NEAR(U(0), 1.0, 1e-4);  // u ≈ 1
    EXPECT_NEAR(U(1), 0.0, 1e-4);  // v ≈ 0
}

// Test that RKDG selects appropriate scheme
TEST_F(TimeStepperTest, RKDGOrderSelection) {
    auto rk1 = TimeStepper::rkdg(1);
    EXPECT_EQ(rk1->order(), 1);

    auto rk2 = TimeStepper::rkdg(2);
    EXPECT_EQ(rk2->order(), 2);

    auto rk3 = TimeStepper::rkdg(3);
    EXPECT_EQ(rk3->order(), 3);

    auto rk4 = TimeStepper::rkdg(4);
    EXPECT_EQ(rk4->order(), 4);

    auto rk5 = TimeStepper::rkdg(5);
    EXPECT_EQ(rk5->order(), 4);  // Falls back to RK4 for order >= 4
}

// Test AdaptiveTimeController
TEST_F(TimeStepperTest, AdaptiveTimeController) {
    AdaptiveTimeController controller(0.5, 1e-10, 1000.0);

    // Test CFL-based dt computation
    Real wave_speed = 100.0;  // m/s
    Real dx = 10.0;           // m

    Real dt = controller.compute_dt(wave_speed, dx);
    EXPECT_NEAR(dt, 0.5 * dx / wave_speed, 1e-10);  // CFL = 0.5

    // Test clamping to dt_max
    Real dt_slow = controller.compute_dt(0.001, 10.0);
    EXPECT_EQ(dt_slow, 1000.0);  // Clamped to dt_max

    // Test clamping to dt_min
    Real dt_fast = controller.compute_dt(1e15, 1e-20);
    EXPECT_EQ(dt_fast, 1e-10);  // Clamped to dt_min
}

// Test LSRK45 (low-storage RK)
TEST_F(TimeStepperTest, LSRK45Convergence) {
    auto rhs = [](Real t, const VecX& U, VecX& dUdt) {
        dUdt = -U;
    };

    LSRK45 stepper;
    EXPECT_EQ(stepper.order(), 4);
    EXPECT_EQ(stepper.num_stages(), 5);

    VecX U(1);
    U(0) = 1.0;
    Real t = 0.0;
    Real dt = 0.1;
    Real t_end = 1.0;

    while (t < t_end - 1e-10) {
        Real step = std::min(dt, t_end - t);
        stepper.step(t, step, U, rhs);
        t += step;
    }

    Real exact = std::exp(-t_end);
    EXPECT_NEAR(U(0), exact, 1e-5);
}

// Test vector valued ODE
TEST_F(TimeStepperTest, VectorODE) {
    // du1/dt = -u1 + u2, du2/dt = u1 - u2
    // Conservation: u1 + u2 = const
    auto rhs = [](Real t, const VecX& U, VecX& dUdt) {
        dUdt.resize(2);
        dUdt(0) = -U(0) + U(1);
        dUdt(1) = U(0) - U(1);
    };

    auto stepper = TimeStepper::rk3_ssp();

    VecX U(2);
    U(0) = 1.0;
    U(1) = 0.0;
    Real initial_sum = U.sum();

    Real t = 0.0;
    Real dt = 0.1;

    for (int i = 0; i < 100; ++i) {
        stepper->step(t, dt, U, rhs);
        t += dt;

        // Check conservation
        EXPECT_NEAR(U.sum(), initial_sum, 1e-10);
    }

    // Equilibrium should be u1 = u2 = 0.5
    EXPECT_NEAR(U(0), 0.5, 0.01);
    EXPECT_NEAR(U(1), 0.5, 0.01);
}
