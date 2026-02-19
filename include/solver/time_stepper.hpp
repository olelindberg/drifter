#pragma once

#include "core/types.hpp"
#include <functional>
#include <memory>

namespace drifter {

// Right-hand side function type: dU/dt = RHS(t, U)
using RHSFunction = std::function<void(Real t, const VecX &U, VecX &dUdt)>;

// Time integration schemes
class TimeStepper {
public:
    virtual ~TimeStepper() = default;

    // Advance solution from t to t + dt
    virtual void step(Real t, Real dt, VecX &U, const RHSFunction &rhs) = 0;

    // Order of accuracy
    virtual int order() const = 0;

    // Number of stages (for Runge-Kutta methods)
    virtual int num_stages() const = 0;

    // Factory methods
    static std::unique_ptr<TimeStepper> forward_euler();
    static std::unique_ptr<TimeStepper> rk2_midpoint();
    static std::unique_ptr<TimeStepper> rk3_ssp(); // Strong stability preserving
    static std::unique_ptr<TimeStepper> rk4_classic();
    static std::unique_ptr<TimeStepper> rk45_dormand_prince(); // Adaptive
    static std::unique_ptr<TimeStepper> rkdg(int order); // Optimized for DG
};

// Explicit Strong Stability Preserving Runge-Kutta (ideal for DG)
class SSPRK3 : public TimeStepper {
public:
    void step(Real t, Real dt, VecX &U, const RHSFunction &rhs) override;
    int order() const override { return 3; }
    int num_stages() const override { return 3; }

private:
    VecX U1_, U2_, k_; // Storage for intermediate stages
};

// Low-storage Runge-Kutta (memory efficient for large problems)
class LSRK45 : public TimeStepper {
public:
    void step(Real t, Real dt, VecX &U, const RHSFunction &rhs) override;
    int order() const override { return 4; }
    int num_stages() const override { return 5; }

private:
    VecX resid_, tmp_;
    // 2N-storage RK coefficients
    static constexpr std::array<Real, 5> A = {0.0, -0.4178904745, -1.192151694643, -1.697784692471,
                                              -1.514183444257};
    static constexpr std::array<Real, 5> B = {0.1496590219993, 0.3792103129999, 0.8229550293869,
                                              0.6994504559488, 0.1530572479681};
    static constexpr std::array<Real, 5> C = {0.0, 0.1496590219993, 0.3704009573644,
                                              0.6222557631345, 0.9582821306748};
};

// Adaptive time stepping controller
class AdaptiveTimeController {
public:
    AdaptiveTimeController(Real cfl_target = 0.5, Real dt_min = 1.0e-10, Real dt_max = 1.0e6);

    // Compute stable time step based on CFL condition
    Real compute_dt(Real max_wave_speed, Real min_element_size) const;

    // Adjust dt based on embedded error estimate
    Real adjust_dt(Real dt, Real error, Real tolerance) const;

    void set_cfl(Real cfl) { cfl_ = cfl; }

private:
    Real cfl_;
    Real dt_min_, dt_max_;
    Real safety_factor_ = 0.9;
};

// IMEX schemes for stiff source terms
class IMEXTimeStepper {
public:
    using ImplicitSolver = std::function<void(Real t, Real dt, VecX &U)>;

    void step(Real t, Real dt, VecX &U, const RHSFunction &explicit_rhs,
              const ImplicitSolver &implicit_solve);

private:
    VecX k1_, k2_, tmp_;
};

} // namespace drifter
