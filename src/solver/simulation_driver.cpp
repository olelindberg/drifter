#include "solver/simulation_driver.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <omp.h>

namespace drifter {

// =============================================================================
// SimulationDriver implementation
// =============================================================================

SimulationDriver::SimulationDriver(const SimulationConfig &config)
    : config_(config), time_(config.t_start), dt_(config.dt_initial), total_steps_(0),
      eos_(EOSType::Linear), coriolis_(1e-4, 0.0, 0.0),
      dt_controller_(config.cfl, config.dt_min, config.dt_max), num_elements_(0),
      order_(config.polynomial_order), dx_min_(1.0),
      next_output_time_(config.t_start + config.output_interval),
      next_diagnostic_time_(config.t_start + config.diagnostic_interval),
      next_checkpoint_time_(config.t_start + config.checkpoint_interval) {
    // Create time stepper
    if (config.use_ssp_rk) {
        time_stepper_ = TimeStepper::rk3_ssp();
    } else {
        time_stepper_ = TimeStepper::rk4_classic();
    }
}

void SimulationDriver::initialize(int num_elements, const std::vector<VecX> &bathymetry,
                                  const std::vector<VecX> &dh_dx, const std::vector<VecX> &dh_dy,
                                  const std::vector<VecX> &y_positions,
                                  const std::vector<PrimitiveState> &initial_states) {

    num_elements_ = num_elements;
    bathymetry_ = bathymetry;
    states_ = initial_states;

    // Estimate minimum element size from DOF count
    // Assuming unit element in reference space, physical size varies
    int dofs_per_elem = initial_states[0].Hu.size();
    int n1d = order_ + 1;
    dx_min_ = 2.0 / n1d; // Reference element size / nodes

    // Create physics solvers
    OceanConstants constants;

    primitive_solver_ = std::make_unique<PrimitiveEquationsSolver>(order_, constants);
    primitive_solver_->initialize(num_elements, bathymetry, dh_dx, dh_dy, coriolis_, y_positions);

    if (config_.use_mode_splitting) {
        ModeSplittingParams ms_params;
        ms_params.subcycles = config_.barotropic_subcycles;

        mode_splitting_solver_ =
            std::make_unique<ModeSplittingSolver>(order_, constants, ms_params);

        // Compute Coriolis at each element
        std::vector<VecX> coriolis_vals(num_elements);
        for (int e = 0; e < num_elements; ++e) {
            coriolis_.compute(y_positions[e], coriolis_vals[e]);
        }

        mode_splitting_solver_->initialize(num_elements, bathymetry, dh_dx, dh_dy, coriolis_vals);
    }
}

void SimulationDriver::set_coriolis(const CoriolisParameter &coriolis) { coriolis_ = coriolis; }

void SimulationDriver::set_eos(const EquationOfState &eos) { eos_ = eos; }

void SimulationDriver::set_boundary_conditions(const std::vector<BoundaryCondition> &bcs) {
    if (primitive_solver_) {
        primitive_solver_->set_boundary_conditions(bcs);
    }
}

void SimulationDriver::set_face_connections(
    const std::vector<std::vector<FaceConnection>> &connections) {
    if (primitive_solver_) {
        primitive_solver_->set_face_connections(connections);
    }
}

void SimulationDriver::set_output_callback(OutputCallback callback) {
    output_callback_ = std::move(callback);
}

void SimulationDriver::set_diagnostic_callback(DiagnosticCallback callback) {
    diagnostic_callback_ = std::move(callback);
}

void SimulationDriver::run() { run_for(config_.t_end - time_); }

void SimulationDriver::run_for(Real duration) {
    Real t_end = time_ + duration;

    while (time_ < t_end) {
        // Limit final step to hit exact end time
        Real dt_to_end = t_end - time_;
        if (dt_ > dt_to_end) {
            dt_ = dt_to_end;
        }

        step();
    }
}

Real SimulationDriver::step() {
    // Compute adaptive time step
    Real max_wave_speed = compute_max_wave_speed();
    dt_ = dt_controller_.compute_dt(max_wave_speed, dx_min_);

    // Mode-split or standard time stepping
    if (config_.use_mode_splitting && mode_splitting_solver_) {
        // Mode-split step handles its own time stepping
        mode_splitting_solver_->step(dt_, states_);
    } else {
        // Standard RK time stepping
        VecX U;
        pack_state(states_, U);

        auto rhs = [this](Real t, const VecX &U_in, VecX &dUdt) {
            this->compute_rhs(t, U_in, dUdt);
        };

        time_stepper_->step(time_, dt_, U, rhs);
        unpack_state(U, states_);
    }

    // Update time
    time_ += dt_;
    total_steps_++;

    // Handle outputs
    if (time_ >= next_output_time_) {
        do_output();
        next_output_time_ += config_.output_interval;
    }

    if (config_.compute_diagnostics && time_ >= next_diagnostic_time_) {
        do_diagnostics();
        next_diagnostic_time_ += config_.diagnostic_interval;
    }

    if (time_ >= next_checkpoint_time_) {
        do_checkpoint();
        next_checkpoint_time_ += config_.checkpoint_interval;
    }

    return dt_;
}

Real SimulationDriver::compute_max_wave_speed() const {
    Real c_max = 0.0;

    for (const auto &state : states_) {
        // Surface gravity wave speed: c = sqrt(g * H)
        Real H_max = state.H.maxCoeff();
        Real c_gravity = std::sqrt(9.81 * H_max);

        // Advection speed
        Real u_max = std::max(state.u.cwiseAbs().maxCoeff(), state.v.cwiseAbs().maxCoeff());

        c_max = std::max(c_max, c_gravity + u_max);
    }

    return c_max;
}

void SimulationDriver::compute_rhs(Real t, const VecX &U, VecX &dUdt) {
    // Unpack state
    std::vector<PrimitiveState> temp_states;
    unpack_state(U, temp_states);

    // Compute tendencies
    std::vector<PrimitiveTendencies> tendencies(num_elements_);

    primitive_solver_->compute_rhs(temp_states, tendencies);
    primitive_solver_->apply_interface_fluxes(temp_states, tendencies);
    primitive_solver_->apply_boundary_conditions(temp_states, tendencies, t);

    // Pack tendencies into output
    int state_size = states_[0].Hu.size();
    int vars_per_elem = 5 * state_size; // Hu, Hv, eta, HT, HS

    dUdt.resize(num_elements_ * vars_per_elem);

    for (int e = 0; e < num_elements_; ++e) {
        int offset = e * vars_per_elem;

        dUdt.segment(offset, state_size) = tendencies[e].dHu_dt;
        dUdt.segment(offset + state_size, state_size) = tendencies[e].dHv_dt;
        dUdt.segment(offset + 2 * state_size, state_size) = tendencies[e].deta_dt;
        dUdt.segment(offset + 3 * state_size, state_size) = tendencies[e].dHT_dt;
        dUdt.segment(offset + 4 * state_size, state_size) = tendencies[e].dHS_dt;
    }
}

void SimulationDriver::pack_state(const std::vector<PrimitiveState> &states, VecX &U) const {
    int state_size = states[0].Hu.size();
    int vars_per_elem = 5 * state_size;

    U.resize(num_elements_ * vars_per_elem);

    for (int e = 0; e < num_elements_; ++e) {
        int offset = e * vars_per_elem;
        U.segment(offset, state_size) = states[e].Hu;
        U.segment(offset + state_size, state_size) = states[e].Hv;
        U.segment(offset + 2 * state_size, state_size) = states[e].eta;
        U.segment(offset + 3 * state_size, state_size) = states[e].HT;
        U.segment(offset + 4 * state_size, state_size) = states[e].HS;
    }
}

void SimulationDriver::unpack_state(const VecX &U, std::vector<PrimitiveState> &states) const {
    int state_size = states_.empty() ? U.size() / (5 * num_elements_) : states_[0].Hu.size();
    int vars_per_elem = 5 * state_size;

    states.resize(num_elements_);

    for (int e = 0; e < num_elements_; ++e) {
        states[e].resize(state_size);
        int offset = e * vars_per_elem;
        states[e].Hu = U.segment(offset, state_size);
        states[e].Hv = U.segment(offset + state_size, state_size);
        states[e].eta = U.segment(offset + 2 * state_size, state_size);
        states[e].HT = U.segment(offset + 3 * state_size, state_size);
        states[e].HS = U.segment(offset + 4 * state_size, state_size);

        // Update derived quantities
        if (e < static_cast<int>(bathymetry_.size())) {
            states[e].update_derived(bathymetry_[e]);
        }
    }
}

SimulationDiagnostics SimulationDriver::compute_diagnostics() const {
    SimulationDiagnostics diag;
    diag.time = time_;
    diag.time_steps = total_steps_;

    // Initialize with extreme values
    diag.total_mass = 0.0;
    diag.total_energy = 0.0;
    diag.max_velocity = 0.0;
    diag.max_eta = -1e30;
    diag.min_eta = 1e30;
    diag.max_temperature = -1e30;
    diag.min_temperature = 1e30;

    for (const auto &state : states_) {
        // Mass: integral of H
        diag.total_mass += state.H.sum();

        // Energy: KE + PE = 0.5*H*(u^2+v^2) + 0.5*g*eta^2
        for (int i = 0; i < state.H.size(); ++i) {
            Real KE = 0.5 * state.H(i) * (state.u(i) * state.u(i) + state.v(i) * state.v(i));
            Real PE = 0.5 * 9.81 * state.eta(i) * state.eta(i);
            diag.total_energy += KE + PE;
        }

        // Max velocity
        Real u_max = std::sqrt(state.u.cwiseAbs2().maxCoeff() + state.v.cwiseAbs2().maxCoeff());
        diag.max_velocity = std::max(diag.max_velocity, u_max);

        // Eta range
        diag.max_eta = std::max(diag.max_eta, state.eta.maxCoeff());
        diag.min_eta = std::min(diag.min_eta, state.eta.minCoeff());

        // Temperature range
        diag.max_temperature = std::max(diag.max_temperature, state.T.maxCoeff());
        diag.min_temperature = std::min(diag.min_temperature, state.T.minCoeff());
    }

    // Actual CFL
    Real max_wave_speed = compute_max_wave_speed();
    diag.cfl_actual = (max_wave_speed > 0) ? dt_ * max_wave_speed / dx_min_ : 0.0;

    return diag;
}

void SimulationDriver::do_output() {
    if (output_callback_) {
        output_callback_(time_, states_);
    }
}

void SimulationDriver::do_diagnostics() {
    if (diagnostic_callback_) {
        auto diag = compute_diagnostics();
        diagnostic_callback_(diag);
    }
}

void SimulationDriver::do_checkpoint() {
    // Placeholder for checkpoint writing
    // Would serialize states_ to disk
}

// =============================================================================
// Initial condition factories
// =============================================================================

std::vector<PrimitiveState> create_quiescent_initial_condition(int num_elements,
                                                               int dofs_per_element,
                                                               Real initial_temperature,
                                                               Real initial_salinity) {

    std::vector<PrimitiveState> states(num_elements);

    for (int e = 0; e < num_elements; ++e) {
        states[e].resize(dofs_per_element);

        // Zero velocity
        states[e].Hu.setZero();
        states[e].Hv.setZero();
        states[e].u.setZero();
        states[e].v.setZero();
        states[e].omega.setZero();

        // Flat free surface
        states[e].eta.setZero();

        // Constant temperature and salinity
        states[e].T.setConstant(initial_temperature);
        states[e].S.setConstant(initial_salinity);

        // H and HT, HS will be set when bathymetry is provided
        states[e].H.setConstant(100.0); // Default 100m depth
        states[e].HT = states[e].H.cwiseProduct(states[e].T);
        states[e].HS = states[e].H.cwiseProduct(states[e].S);

        // Density from linear EOS
        Real T0 = 10.0, S0 = 35.0, rho_0 = 1025.0;
        Real alpha = 2.0e-4, beta = 7.6e-4;
        for (int i = 0; i < dofs_per_element; ++i) {
            states[e].rho(i) =
                rho_0 * (1.0 - alpha * (initial_temperature - T0) + beta * (initial_salinity - S0));
        }
    }

    return states;
}

std::vector<PrimitiveState> create_kelvin_wave_initial_condition(
    int num_elements, int dofs_per_element, const std::vector<VecX> &x_positions,
    const std::vector<VecX> &y_positions, Real amplitude, Real wavelength, Real depth) {

    std::vector<PrimitiveState> states =
        create_quiescent_initial_condition(num_elements, dofs_per_element);

    Real k = 2.0 * 3.14159265358979323846 / wavelength;
    Real f = 1e-4; // Coriolis parameter
    Real c = std::sqrt(9.81 * depth); // Phase speed
    Real Rd = c / f; // Rossby radius

    for (int e = 0; e < num_elements; ++e) {
        for (int i = 0; i < dofs_per_element; ++i) {
            Real x = x_positions[e](i);
            Real y = y_positions[e](i);

            // Kelvin wave: eta = A * exp(-y/Rd) * cos(k*x)
            Real decay = std::exp(-y / Rd);
            states[e].eta(i) = amplitude * decay * std::cos(k * x);

            // Along-shore velocity: u = (g/c) * eta
            states[e].u(i) = (9.81 / c) * states[e].eta(i);
            states[e].v(i) = 0.0;

            // Update conserved variables
            states[e].H(i) = states[e].eta(i) + depth;
            states[e].Hu(i) = states[e].H(i) * states[e].u(i);
            states[e].Hv(i) = 0.0;
        }
    }

    return states;
}

std::vector<PrimitiveState>
create_lock_exchange_initial_condition(int num_elements, int dofs_per_element,
                                       const std::vector<VecX> &x_positions, Real T_cold,
                                       Real T_warm, Real x_interface) {

    std::vector<PrimitiveState> states =
        create_quiescent_initial_condition(num_elements, dofs_per_element);

    for (int e = 0; e < num_elements; ++e) {
        for (int i = 0; i < dofs_per_element; ++i) {
            Real x = x_positions[e](i);

            // Cold water on left, warm on right
            if (x < x_interface) {
                states[e].T(i) = T_cold;
            } else {
                states[e].T(i) = T_warm;
            }

            // Update HT and density
            states[e].HT(i) = states[e].H(i) * states[e].T(i);

            Real T0 = 10.0, rho_0 = 1025.0, alpha = 2.0e-4;
            states[e].rho(i) = rho_0 * (1.0 - alpha * (states[e].T(i) - T0));
        }
    }

    return states;
}

} // namespace drifter
