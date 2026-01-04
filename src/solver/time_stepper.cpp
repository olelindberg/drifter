#include "solver/time_stepper.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace drifter {

// =============================================================================
// Forward Euler (RK1)
// =============================================================================

class ForwardEuler : public TimeStepper {
public:
    void step(Real t, Real dt, VecX& U, const RHSFunction& rhs) override {
        k_.resize(U.size());
        rhs(t, U, k_);
        U += dt * k_;
    }

    int order() const override { return 1; }
    int num_stages() const override { return 1; }

private:
    VecX k_;
};

// =============================================================================
// RK2 Midpoint Method
// =============================================================================

class RK2Midpoint : public TimeStepper {
public:
    void step(Real t, Real dt, VecX& U, const RHSFunction& rhs) override {
        k1_.resize(U.size());
        k2_.resize(U.size());
        U_tmp_.resize(U.size());

        // Stage 1: k1 = f(t, u)
        rhs(t, U, k1_);

        // Stage 2: k2 = f(t + dt/2, u + dt/2 * k1)
        U_tmp_ = U + 0.5 * dt * k1_;
        rhs(t + 0.5 * dt, U_tmp_, k2_);

        // Update: u_new = u + dt * k2
        U += dt * k2_;
    }

    int order() const override { return 2; }
    int num_stages() const override { return 2; }

private:
    VecX k1_, k2_, U_tmp_;
};

// =============================================================================
// SSP-RK3 (Strong Stability Preserving, 3rd order)
// =============================================================================
// Optimal 3-stage, 3rd order SSP scheme (Shu-Osher form)
// u(1) = u(n) + dt * L(u(n))
// u(2) = 3/4 * u(n) + 1/4 * (u(1) + dt * L(u(1)))
// u(n+1) = 1/3 * u(n) + 2/3 * (u(2) + dt * L(u(2)))

void SSPRK3::step(Real t, Real dt, VecX& U, const RHSFunction& rhs) {
    U1_.resize(U.size());
    U2_.resize(U.size());
    k_.resize(U.size());

    // Stage 1: U1 = U + dt * L(U)
    rhs(t, U, k_);
    U1_ = U + dt * k_;

    // Stage 2: U2 = 3/4 * U + 1/4 * (U1 + dt * L(U1))
    rhs(t + dt, U1_, k_);
    U2_ = 0.75 * U + 0.25 * (U1_ + dt * k_);

    // Stage 3: U_new = 1/3 * U + 2/3 * (U2 + dt * L(U2))
    rhs(t + 0.5 * dt, U2_, k_);
    U = (1.0 / 3.0) * U + (2.0 / 3.0) * (U2_ + dt * k_);
}

// =============================================================================
// Classic RK4
// =============================================================================

class RK4Classic : public TimeStepper {
public:
    void step(Real t, Real dt, VecX& U, const RHSFunction& rhs) override {
        k1_.resize(U.size());
        k2_.resize(U.size());
        k3_.resize(U.size());
        k4_.resize(U.size());
        U_tmp_.resize(U.size());

        // Stage 1: k1 = f(t, u)
        rhs(t, U, k1_);

        // Stage 2: k2 = f(t + dt/2, u + dt/2 * k1)
        U_tmp_ = U + 0.5 * dt * k1_;
        rhs(t + 0.5 * dt, U_tmp_, k2_);

        // Stage 3: k3 = f(t + dt/2, u + dt/2 * k2)
        U_tmp_ = U + 0.5 * dt * k2_;
        rhs(t + 0.5 * dt, U_tmp_, k3_);

        // Stage 4: k4 = f(t + dt, u + dt * k3)
        U_tmp_ = U + dt * k3_;
        rhs(t + dt, U_tmp_, k4_);

        // Update: u_new = u + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        U += (dt / 6.0) * (k1_ + 2.0 * k2_ + 2.0 * k3_ + k4_);
    }

    int order() const override { return 4; }
    int num_stages() const override { return 4; }

private:
    VecX k1_, k2_, k3_, k4_, U_tmp_;
};

// =============================================================================
// Low-Storage RK45 (Carpenter-Kennedy 2N storage)
// =============================================================================
// 5-stage, 4th order, optimized for DG

void LSRK45::step(Real t, Real dt, VecX& U, const RHSFunction& rhs) {
    int n = static_cast<int>(U.size());
    resid_.resize(n);
    tmp_.resize(n);
    resid_.setZero();

    for (int stage = 0; stage < 5; ++stage) {
        // Compute RHS at current stage time
        Real t_stage = t + C[stage] * dt;
        rhs(t_stage, U, tmp_);

        // Update residual: resid = A[s] * resid + dt * tmp
        resid_ = A[stage] * resid_ + dt * tmp_;

        // Update solution: U = U + B[s] * resid
        U += B[stage] * resid_;
    }
}

// =============================================================================
// Dormand-Prince RK45 (Adaptive)
// =============================================================================

class RK45DormandPrince : public TimeStepper {
public:
    void step(Real t, Real dt, VecX& U, const RHSFunction& rhs) override {
        // Standard Dormand-Prince coefficients
        k1_.resize(U.size());
        k2_.resize(U.size());
        k3_.resize(U.size());
        k4_.resize(U.size());
        k5_.resize(U.size());
        k6_.resize(U.size());
        U_tmp_.resize(U.size());

        // k1 = f(t, u)
        rhs(t, U, k1_);

        // k2 = f(t + dt/5, u + dt/5 * k1)
        U_tmp_ = U + (dt / 5.0) * k1_;
        rhs(t + dt / 5.0, U_tmp_, k2_);

        // k3 = f(t + 3*dt/10, u + 3*dt/40 * k1 + 9*dt/40 * k2)
        U_tmp_ = U + dt * (3.0 / 40.0 * k1_ + 9.0 / 40.0 * k2_);
        rhs(t + 3.0 * dt / 10.0, U_tmp_, k3_);

        // k4 = f(t + 4*dt/5, ...)
        U_tmp_ = U + dt * (44.0 / 45.0 * k1_ - 56.0 / 15.0 * k2_ + 32.0 / 9.0 * k3_);
        rhs(t + 4.0 * dt / 5.0, U_tmp_, k4_);

        // k5 = f(t + 8*dt/9, ...)
        U_tmp_ = U + dt * (19372.0 / 6561.0 * k1_ - 25360.0 / 2187.0 * k2_ +
                          64448.0 / 6561.0 * k3_ - 212.0 / 729.0 * k4_);
        rhs(t + 8.0 * dt / 9.0, U_tmp_, k5_);

        // k6 = f(t + dt, ...)
        U_tmp_ = U + dt * (9017.0 / 3168.0 * k1_ - 355.0 / 33.0 * k2_ +
                          46732.0 / 5247.0 * k3_ + 49.0 / 176.0 * k4_ -
                          5103.0 / 18656.0 * k5_);
        rhs(t + dt, U_tmp_, k6_);

        // 5th order solution (used for advancing)
        U += dt * (35.0 / 384.0 * k1_ + 500.0 / 1113.0 * k3_ +
                   125.0 / 192.0 * k4_ - 2187.0 / 6784.0 * k5_ +
                   11.0 / 84.0 * k6_);

        // 4th order solution would be:
        // U4 = U + dt * (5179/57600*k1 + 7571/16695*k3 + 393/640*k4 -
        //               92097/339200*k5 + 187/2100*k6 + 1/40*k7)
        // Error estimate = U5 - U4 (for adaptive stepping)
    }

    int order() const override { return 5; }
    int num_stages() const override { return 6; }

private:
    VecX k1_, k2_, k3_, k4_, k5_, k6_, U_tmp_;
};

// =============================================================================
// Optimized RKDG (order-matched to DG polynomial order)
// =============================================================================

class RKDG : public TimeStepper {
public:
    explicit RKDG(int order) : order_(order) {
        // Select optimal RK scheme for DG order
        if (order <= 1) {
            inner_ = std::make_unique<ForwardEuler>();
        } else if (order == 2) {
            inner_ = std::make_unique<RK2Midpoint>();
        } else if (order == 3) {
            inner_ = std::make_unique<SSPRK3>();
        } else {
            inner_ = std::make_unique<RK4Classic>();
        }
    }

    void step(Real t, Real dt, VecX& U, const RHSFunction& rhs) override {
        inner_->step(t, dt, U, rhs);
    }

    int order() const override { return inner_->order(); }
    int num_stages() const override { return inner_->num_stages(); }

private:
    int order_;
    std::unique_ptr<TimeStepper> inner_;
};

// =============================================================================
// Factory methods
// =============================================================================

std::unique_ptr<TimeStepper> TimeStepper::forward_euler() {
    return std::make_unique<ForwardEuler>();
}

std::unique_ptr<TimeStepper> TimeStepper::rk2_midpoint() {
    return std::make_unique<RK2Midpoint>();
}

std::unique_ptr<TimeStepper> TimeStepper::rk3_ssp() {
    return std::make_unique<SSPRK3>();
}

std::unique_ptr<TimeStepper> TimeStepper::rk4_classic() {
    return std::make_unique<RK4Classic>();
}

std::unique_ptr<TimeStepper> TimeStepper::rk45_dormand_prince() {
    return std::make_unique<RK45DormandPrince>();
}

std::unique_ptr<TimeStepper> TimeStepper::rkdg(int order) {
    return std::make_unique<RKDG>(order);
}

// =============================================================================
// AdaptiveTimeController implementation
// =============================================================================

AdaptiveTimeController::AdaptiveTimeController(Real cfl_target, Real dt_min, Real dt_max)
    : cfl_(cfl_target), dt_min_(dt_min), dt_max_(dt_max) {}

Real AdaptiveTimeController::compute_dt(Real max_wave_speed, Real min_element_size) const {
    if (max_wave_speed < 1e-14) {
        return dt_max_;  // Stationary - use maximum timestep
    }

    Real dt = cfl_ * min_element_size / max_wave_speed;
    return std::clamp(dt, dt_min_, dt_max_);
}

Real AdaptiveTimeController::adjust_dt(Real dt, Real error, Real tolerance) const {
    if (error < 1e-14) {
        return std::min(dt * 2.0, dt_max_);  // Error very small, increase dt
    }

    // Standard error-based adjustment: dt_new = dt * (tol / err)^(1/p)
    // Using p = 4 for 4th order method
    Real ratio = std::pow(tolerance / error, 0.25);
    Real dt_new = safety_factor_ * dt * ratio;

    return std::clamp(dt_new, dt_min_, dt_max_);
}

// =============================================================================
// IMEX Time Stepper implementation
// =============================================================================

void IMEXTimeStepper::step(Real t, Real dt, VecX& U,
                            const RHSFunction& explicit_rhs,
                            const ImplicitSolver& implicit_solve) {
    // Simple IMEX-RK2 (Ascher-Ruuth-Spiteri ARS(2,2,2))
    // Combines explicit treatment of advection with implicit treatment of stiff terms

    k1_.resize(U.size());
    k2_.resize(U.size());
    tmp_.resize(U.size());

    // Stage 1
    // Explicit: k1 = f_explicit(t, U)
    explicit_rhs(t, U, k1_);

    // Implicit: solve (I - gamma*dt*L_implicit) * U1 = U + gamma*dt*k1
    Real gamma = 1.0 - 1.0 / std::sqrt(2.0);  // SDIRK parameter

    tmp_ = U + gamma * dt * k1_;
    implicit_solve(t + gamma * dt, gamma * dt, tmp_);

    // Stage 2
    // Explicit: k2 = f_explicit(t + dt, tmp)
    explicit_rhs(t + dt, tmp_, k2_);

    // Final update
    // U_new = U + dt * ((1-gamma)*k1 + gamma*k2) + implicit correction
    U = U + dt * ((1.0 - gamma) * k1_ + gamma * k2_);
    implicit_solve(t + dt, gamma * dt, U);
}

}  // namespace drifter
