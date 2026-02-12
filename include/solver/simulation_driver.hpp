#pragma once

// Simulation Driver for DRIFTER Ocean Model
//
// Ties together all components:
// - Physics: primitive equations, mode splitting, equation of state
// - Time integration: RK schemes, adaptive CFL
// - I/O: checkpointing, output writers
// - Diagnostics: conservation, stability monitoring
//
// Supports:
// - Mode-split time stepping (barotropic subcycling)
// - Adaptive time stepping based on CFL
// - Periodic output and checkpointing
// - Parallel execution via OpenMP

#include "core/types.hpp"
#include "dg/basis_hexahedron.hpp"
#include "dg/operators_3d.hpp"
#include "dg/quadrature_3d.hpp"
#include "physics/equation_of_state.hpp"
#include "physics/mode_splitting.hpp"
#include "physics/primitive_equations.hpp"
#include "solver/time_stepper.hpp"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace drifter {

/// @brief Simulation configuration parameters
struct SimulationConfig {
    // Time stepping
    Real t_start = 0.0;     ///< Start time [s]
    Real t_end = 86400.0;   ///< End time [s] (default 1 day)
    Real dt_initial = 10.0; ///< Initial time step [s]
    Real cfl = 0.5;         ///< CFL number
    Real dt_min = 1e-6;     ///< Minimum time step [s]
    Real dt_max = 3600.0;   ///< Maximum time step [s]

    // Mode splitting
    bool use_mode_splitting = true;
    int barotropic_subcycles = 30;

    // Output
    Real output_interval = 3600.0; ///< Output interval [s] (default 1 hour)
    Real checkpoint_interval = 86400.0; ///< Checkpoint interval [s]
    std::string output_prefix = "drifter";

    // Diagnostics
    bool compute_diagnostics = true;
    Real diagnostic_interval = 600.0; ///< Diagnostic interval [s]

    // Solver options
    int polynomial_order = 3;
    bool use_ssp_rk = true; ///< Use SSP-RK3 (true) or classic RK4 (false)
};

/// @brief Diagnostic quantities for monitoring
struct SimulationDiagnostics {
    Real time;
    Real total_mass;
    Real total_energy;
    Real max_velocity;
    Real max_eta;
    Real min_eta;
    Real max_temperature;
    Real min_temperature;
    Real cfl_actual;
    int time_steps;
};

/// @brief Callback types for user hooks
using OutputCallback =
    std::function<void(Real time, const std::vector<PrimitiveState> &states)>;
using DiagnosticCallback =
    std::function<void(const SimulationDiagnostics &diag)>;

/// @brief Main simulation driver
class SimulationDriver {
public:
    /// @brief Construct simulation driver
    /// @param config Simulation configuration
    SimulationDriver(const SimulationConfig &config = SimulationConfig());

    /// @brief Initialize with mesh and initial conditions
    /// @param num_elements Number of elements
    /// @param bathymetry Bathymetry at each element's DOFs
    /// @param dh_dx Bathymetry x-gradient
    /// @param dh_dy Bathymetry y-gradient
    /// @param y_positions Y-coordinates at DOFs (for Coriolis)
    /// @param initial_states Initial primitive states
    void initialize(
        int num_elements, const std::vector<VecX> &bathymetry,
        const std::vector<VecX> &dh_dx, const std::vector<VecX> &dh_dy,
        const std::vector<VecX> &y_positions,
        const std::vector<PrimitiveState> &initial_states);

    /// @brief Set Coriolis parameter
    void set_coriolis(const CoriolisParameter &coriolis);

    /// @brief Set equation of state
    void set_eos(const EquationOfState &eos);

    /// @brief Set boundary conditions
    void set_boundary_conditions(const std::vector<BoundaryCondition> &bcs);

    /// @brief Set face connections for DG
    void set_face_connections(
        const std::vector<std::vector<FaceConnection>> &connections);

    /// @brief Register output callback
    void set_output_callback(OutputCallback callback);

    /// @brief Register diagnostic callback
    void set_diagnostic_callback(DiagnosticCallback callback);

    // =========================================================================
    // Running the simulation
    // =========================================================================

    /// @brief Run simulation from t_start to t_end
    void run();

    /// @brief Run simulation for specified duration
    void run_for(Real duration);

    /// @brief Advance one time step
    /// @return Actual time step taken
    Real step();

    /// @brief Get current simulation time
    Real current_time() const { return time_; }

    /// @brief Get current time step
    Real current_dt() const { return dt_; }

    /// @brief Get current state
    const std::vector<PrimitiveState> &current_state() const { return states_; }

    /// @brief Get mutable state (for external forcing)
    std::vector<PrimitiveState> &state() { return states_; }

    /// @brief Check if simulation is complete
    bool is_complete() const { return time_ >= config_.t_end; }

    // =========================================================================
    // Diagnostics
    // =========================================================================

    /// @brief Compute and return diagnostics
    SimulationDiagnostics compute_diagnostics() const;

    /// @brief Get total number of time steps taken
    int total_steps() const { return total_steps_; }

    /// @brief Get configuration
    const SimulationConfig &config() const { return config_; }

private:
    SimulationConfig config_;

    // Current state
    Real time_;
    Real dt_;
    int total_steps_;
    std::vector<PrimitiveState> states_;

    // Physics solvers
    std::unique_ptr<PrimitiveEquationsSolver> primitive_solver_;
    std::unique_ptr<ModeSplittingSolver> mode_splitting_solver_;
    EquationOfState eos_;
    CoriolisParameter coriolis_;

    // Time integration
    std::unique_ptr<TimeStepper> time_stepper_;
    AdaptiveTimeController dt_controller_;

    // Mesh data
    int num_elements_;
    int order_;
    std::vector<VecX> bathymetry_;
    Real dx_min_; // Minimum element size

    // Callbacks
    OutputCallback output_callback_;
    DiagnosticCallback diagnostic_callback_;

    // Timing for output
    Real next_output_time_;
    Real next_diagnostic_time_;
    Real next_checkpoint_time_;

    /// @brief Compute maximum wave speed for CFL
    Real compute_max_wave_speed() const;

    /// @brief Compute RHS for time stepping (wraps physics solvers)
    void compute_rhs(Real t, const VecX &U, VecX &dUdt);

    /// @brief Pack states into single vector
    void pack_state(const std::vector<PrimitiveState> &states, VecX &U) const;

    /// @brief Unpack single vector into states
    void unpack_state(const VecX &U, std::vector<PrimitiveState> &states) const;

    /// @brief Handle output (called at output intervals)
    void do_output();

    /// @brief Handle diagnostics (called at diagnostic intervals)
    void do_diagnostics();

    /// @brief Handle checkpoint (called at checkpoint intervals)
    void do_checkpoint();
};

/// @brief Simple test case: quiescent ocean
std::vector<PrimitiveState> create_quiescent_initial_condition(
    int num_elements, int dofs_per_element, Real initial_temperature = 15.0,
    Real initial_salinity = 35.0);

/// @brief Test case: barotropic Kelvin wave
std::vector<PrimitiveState> create_kelvin_wave_initial_condition(
    int num_elements, int dofs_per_element,
    const std::vector<VecX> &x_positions, const std::vector<VecX> &y_positions,
    Real amplitude = 0.1, Real wavelength = 100000.0, Real depth = 100.0);

/// @brief Test case: lock exchange (gravity current)
std::vector<PrimitiveState> create_lock_exchange_initial_condition(
    int num_elements, int dofs_per_element,
    const std::vector<VecX> &x_positions, Real T_cold = 10.0,
    Real T_warm = 20.0, Real x_interface = 0.5);

} // namespace drifter
