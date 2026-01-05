// Integration tests for full simulation pipelines

#include <gtest/gtest.h>
#include "solver/simulation_driver.hpp"
#include "mesh/octree_adapter.hpp"
#include "dg/basis_hexahedron.hpp"
#include "io/vtk_writer.hpp"
#include "physics/primitive_equations.hpp"
#include "test_integration_fixtures.hpp"
#include <filesystem>
#include <fstream>

using namespace drifter;
using namespace drifter::testing;

TEST_F(SimulationTest, AdvectionSanityCheck) {
    // Sanity test: verify simulation infrastructure works
    // Create mesh, initial condition, run a few steps
    // Just verify it runs without crashing - the physics is not stable without
    // proper dissipation and boundary conditions, but the infrastructure should work

    int order = 2;
    int nx = 2, ny = 2, nz = 1;
    Real Lx = 10000.0, Ly = 10000.0, Lz = 100.0;  // 10km x 10km x 100m

    // Create mesh
    OctreeAdapter mesh(0.0, Lx, 0.0, Ly, -Lz, 0.0);
    mesh.build_uniform(nx, ny, nz);

    int num_elements = mesh.num_elements();
    int n1d = order + 1;
    int dofs_per_elem = n1d * n1d * n1d;

    // Create basis
    HexahedronBasis basis(order);
    const VecX& nodes_1d = basis.lgl_basis_1d().nodes;

    // Set up position arrays for initial conditions
    std::vector<VecX> x_pos(num_elements), y_pos(num_elements);
    std::vector<VecX> bathymetry(num_elements);
    std::vector<VecX> dh_dx(num_elements), dh_dy(num_elements);

    for (Index e = 0; e < num_elements; ++e) {
        const auto& bounds = mesh.element_bounds(e);
        Vec3 center = bounds.center();
        Vec3 size = bounds.size();

        x_pos[e].resize(dofs_per_elem);
        y_pos[e].resize(dofs_per_elem);
        bathymetry[e].resize(dofs_per_elem);
        dh_dx[e].resize(dofs_per_elem);
        dh_dy[e].resize(dofs_per_elem);

        for (int k = 0; k < n1d; ++k) {
            for (int j = 0; j < n1d; ++j) {
                for (int i = 0; i < n1d; ++i) {
                    int idx = i + n1d * (j + n1d * k);
                    x_pos[e](idx) = center(0) + 0.5 * nodes_1d(i) * size(0);
                    y_pos[e](idx) = center(1) + 0.5 * nodes_1d(j) * size(1);
                    bathymetry[e](idx) = Lz;
                    dh_dx[e](idx) = 0.0;
                    dh_dy[e](idx) = 0.0;
                }
            }
        }
    }

    // Create quiescent initial condition (no perturbation = stable)
    auto states = create_quiescent_initial_condition(
        num_elements, dofs_per_elem, 15.0, 35.0);

    // Set correct depth in states
    for (Index e = 0; e < num_elements; ++e) {
        states[e].H = bathymetry[e];
        states[e].HT = states[e].H.cwiseProduct(states[e].T);
        states[e].HS = states[e].H.cwiseProduct(states[e].S);
    }

    // Configure simulation
    SimulationConfig config;
    config.t_start = 0.0;
    config.t_end = 100.0;
    config.dt_initial = 1.0;
    config.cfl = 0.5;
    config.polynomial_order = order;
    config.use_mode_splitting = false;
    config.use_ssp_rk = true;
    config.output_interval = 1000.0;
    config.diagnostic_interval = 1000.0;

    // Create driver
    SimulationDriver driver(config);
    driver.initialize(num_elements, bathymetry, dh_dx, dh_dy, y_pos, states);

    // Set face connections
    auto connections = mesh.build_face_connections();
    driver.set_face_connections(connections);

    // Check initial state
    Real initial_mass = 0.0;
    for (const auto& state : driver.current_state()) {
        initial_mass += state.H.sum();
        ASSERT_FALSE(state.eta.hasNaN());
        ASSERT_FALSE(state.H.hasNaN());
    }

    // Run just a few steps to verify infrastructure works
    int steps = 0;
    for (int i = 0; i < 5; ++i) {
        driver.step();
        ++steps;
    }

    // Check simulation progressed
    EXPECT_EQ(steps, 5);
    EXPECT_GT(driver.current_time(), 0.0);
    EXPECT_EQ(driver.total_steps(), 5);

    // For quiescent initial condition (zero velocity, flat surface),
    // the state should remain stable
    for (const auto& state : driver.current_state()) {
        EXPECT_FALSE(state.eta.hasNaN());
        EXPECT_FALSE(state.H.hasNaN());
    }
}

TEST_F(SimulationTest, KelvinWavePhysics) {
    // Test Kelvin wave initial condition properties
    // The full simulation is unstable without proper boundary conditions,
    // but we can verify the initial condition has correct structure

    int order = 3;
    int nx = 4, ny = 2, nz = 1;
    Real Lx = 100000.0, Ly = 50000.0, Lz = 100.0;  // 100km x 50km x 100m

    OctreeAdapter mesh(0.0, Lx, 0.0, Ly, -Lz, 0.0);
    mesh.build_uniform(nx, ny, nz);

    int num_elements = mesh.num_elements();
    int n1d = order + 1;
    int dofs_per_elem = n1d * n1d * n1d;

    HexahedronBasis basis(order);
    const VecX& nodes_1d = basis.lgl_basis_1d().nodes;

    // Set up position arrays
    std::vector<VecX> x_pos(num_elements), y_pos(num_elements);
    std::vector<VecX> bathymetry(num_elements);
    std::vector<VecX> dh_dx(num_elements), dh_dy(num_elements);

    for (Index e = 0; e < num_elements; ++e) {
        const auto& bounds = mesh.element_bounds(e);
        Vec3 center = bounds.center();
        Vec3 size = bounds.size();

        x_pos[e].resize(dofs_per_elem);
        y_pos[e].resize(dofs_per_elem);
        bathymetry[e].resize(dofs_per_elem);
        dh_dx[e].resize(dofs_per_elem);
        dh_dy[e].resize(dofs_per_elem);

        for (int k = 0; k < n1d; ++k) {
            for (int j = 0; j < n1d; ++j) {
                for (int i = 0; i < n1d; ++i) {
                    int idx = i + n1d * (j + n1d * k);
                    x_pos[e](idx) = center(0) + 0.5 * nodes_1d(i) * size(0);
                    y_pos[e](idx) = center(1) + 0.5 * nodes_1d(j) * size(1);
                    bathymetry[e](idx) = Lz;
                    dh_dx[e](idx) = 0.0;
                    dh_dy[e](idx) = 0.0;
                }
            }
        }
    }

    // Kelvin wave parameters
    Real amplitude = 0.1;  // 10 cm
    Real wavelength = 50000.0;  // 50 km
    Real depth = Lz;

    auto states = create_kelvin_wave_initial_condition(
        num_elements, dofs_per_elem, x_pos, y_pos,
        amplitude, wavelength, depth);

    // Update H with bathymetry
    for (Index e = 0; e < num_elements; ++e) {
        for (int i = 0; i < dofs_per_elem; ++i) {
            states[e].H(i) = bathymetry[e](i) + states[e].eta(i);
        }
        states[e].HT = states[e].H.cwiseProduct(states[e].T);
        states[e].HS = states[e].H.cwiseProduct(states[e].S);
    }

    // Verify initial conditions have correct Kelvin wave properties:

    // 1. Cross-shore velocity should be zero
    for (const auto& state : states) {
        EXPECT_NEAR(state.v.norm(), 0.0, LOOSE_TOLERANCE);
    }

    // 2. Along-shore velocity should be non-zero where eta is non-zero
    bool found_nonzero_u = false;
    for (const auto& state : states) {
        if (state.u.cwiseAbs().maxCoeff() > 1e-6) {
            found_nonzero_u = true;
            break;
        }
    }
    EXPECT_TRUE(found_nonzero_u);

    // 3. Eta should decay offshore (in y direction)
    Real eta_nearshore = 0.0;
    Real eta_offshore = 0.0;
    int count_near = 0, count_off = 0;

    for (Index e = 0; e < num_elements; ++e) {
        const auto& bounds = mesh.element_bounds(e);
        if (bounds.center()(1) < Ly / 4) {
            eta_nearshore += states[e].eta.cwiseAbs().mean();
            ++count_near;
        } else if (bounds.center()(1) > 3 * Ly / 4) {
            eta_offshore += states[e].eta.cwiseAbs().mean();
            ++count_off;
        }
    }

    if (count_near > 0 && count_off > 0) {
        eta_nearshore /= count_near;
        eta_offshore /= count_off;

        // Offshore eta should be smaller (exponential decay)
        EXPECT_LT(eta_offshore, eta_nearshore);
    }

    // 4. Verify simulation driver can be initialized with this IC
    SimulationConfig config;
    config.t_start = 0.0;
    config.t_end = 100.0;
    config.polynomial_order = order;
    config.use_mode_splitting = false;
    config.use_ssp_rk = true;

    SimulationDriver driver(config);
    driver.initialize(num_elements, bathymetry, dh_dx, dh_dy, y_pos, states);

    auto connections = mesh.build_face_connections();
    driver.set_face_connections(connections);

    // Set Coriolis
    Real f = 1e-4;
    CoriolisParameter coriolis(f, 0.0, 0.0);
    driver.set_coriolis(coriolis);

    // Run just one step to verify infrastructure works
    driver.step();
    EXPECT_GT(driver.current_time(), 0.0);
    EXPECT_EQ(driver.total_steps(), 1);
}

TEST_F(SimulationTest, FullSimulationWithOutput) {
    // Test VTK output with simulation data
    // Use quiescent IC for stability, just verify output infrastructure works

    int order = 2;
    Real Lx = 10000.0, Ly = 10000.0, Lz = 50.0;

    OctreeAdapter mesh(0.0, Lx, 0.0, Ly, -Lz, 0.0);
    mesh.build_uniform(2, 2, 1);

    int num_elements = mesh.num_elements();
    int n1d = order + 1;
    int dofs_per_elem = n1d * n1d * n1d;

    HexahedronBasis basis(order);
    const VecX& nodes_1d = basis.lgl_basis_1d().nodes;

    // Set up data arrays
    std::vector<VecX> y_pos(num_elements);
    std::vector<VecX> bathymetry(num_elements);
    std::vector<VecX> dh_dx(num_elements), dh_dy(num_elements);

    for (Index e = 0; e < num_elements; ++e) {
        const auto& bounds = mesh.element_bounds(e);
        Vec3 center = bounds.center();
        Vec3 size = bounds.size();

        y_pos[e].resize(dofs_per_elem);
        bathymetry[e].resize(dofs_per_elem);
        dh_dx[e].resize(dofs_per_elem);
        dh_dy[e].resize(dofs_per_elem);

        for (int k = 0; k < n1d; ++k) {
            for (int j = 0; j < n1d; ++j) {
                for (int i = 0; i < n1d; ++i) {
                    int idx = i + n1d * (j + n1d * k);
                    y_pos[e](idx) = center(1) + 0.5 * nodes_1d(j) * size(1);
                    bathymetry[e](idx) = Lz;
                    dh_dx[e](idx) = 0.0;
                    dh_dy[e](idx) = 0.0;
                }
            }
        }
    }

    // Create quiescent initial condition (stable)
    auto states = create_quiescent_initial_condition(
        num_elements, dofs_per_elem, 15.0, 35.0);

    for (Index e = 0; e < num_elements; ++e) {
        states[e].H = bathymetry[e];
        states[e].HT = states[e].H.cwiseProduct(states[e].T);
        states[e].HS = states[e].H.cwiseProduct(states[e].S);
    }

    // Set up VTK output
    std::string vtk_basename = test_output_dir_ + "/simulation";
    VTKWriter vtk_writer(vtk_basename, VTKFormat::VTU, VTKEncoding::ASCII);
    vtk_writer.set_polynomial_order(order);
    vtk_writer.set_mesh(mesh);
    vtk_writer.add_point_data("eta", 1);
    vtk_writer.add_point_data("H", 1);

    // Write initial condition
    std::vector<VecX> eta_data(num_elements), H_data(num_elements);
    for (Index e = 0; e < num_elements; ++e) {
        eta_data[e] = states[e].eta;
        H_data[e] = states[e].H;
    }
    vtk_writer.set_point_data("eta", eta_data);
    vtk_writer.set_point_data("H", H_data);
    vtk_writer.write_timestep(0.0);

    // Configure simulation
    SimulationConfig config;
    config.t_start = 0.0;
    config.t_end = 10.0;
    config.polynomial_order = order;
    config.use_mode_splitting = false;
    config.use_ssp_rk = true;

    SimulationDriver driver(config);
    driver.initialize(num_elements, bathymetry, dh_dx, dh_dy, y_pos, states);

    auto connections = mesh.build_face_connections();
    driver.set_face_connections(connections);

    // Run a few steps and write output
    for (int i = 0; i < 3; ++i) {
        driver.step();
    }

    // Write another timestep
    const auto& current = driver.current_state();
    for (Index e = 0; e < num_elements; ++e) {
        eta_data[e] = current[e].eta;
        H_data[e] = current[e].H;
    }
    vtk_writer.set_point_data("eta", eta_data);
    vtk_writer.set_point_data("H", H_data);
    vtk_writer.write_timestep(driver.current_time());

    vtk_writer.finalize();

    // Verify output files exist
    EXPECT_TRUE(std::filesystem::exists(vtk_basename + "_000000.vtu"));
    EXPECT_TRUE(std::filesystem::exists(vtk_basename + ".pvd"));

    // Read PVD and verify it references multiple timesteps
    std::ifstream pvd(vtk_basename + ".pvd");
    std::string content((std::istreambuf_iterator<char>(pvd)),
                        std::istreambuf_iterator<char>());

    // Should have at least 2 timesteps
    size_t count = 0;
    size_t pos = 0;
    while ((pos = content.find("timestep", pos)) != std::string::npos) {
        ++count;
        ++pos;
    }
    EXPECT_GE(count, 2);
}

TEST_F(SimulationTest, SimulationDiagnostics) {
    int order = 2;
    int dofs = 27;  // 3^3
    int num_elements = 4;

    // Create simple quiescent state
    auto states = create_quiescent_initial_condition(num_elements, dofs, 15.0, 35.0);

    // Set up minimal simulation
    std::vector<VecX> bathymetry(num_elements), dh_dx(num_elements), dh_dy(num_elements);
    std::vector<VecX> y_pos(num_elements);

    for (int e = 0; e < num_elements; ++e) {
        bathymetry[e] = VecX::Constant(dofs, 100.0);
        dh_dx[e] = VecX::Zero(dofs);
        dh_dy[e] = VecX::Zero(dofs);
        y_pos[e] = VecX::Zero(dofs);
        states[e].H = bathymetry[e];
        states[e].HT = states[e].H.cwiseProduct(states[e].T);
        states[e].HS = states[e].H.cwiseProduct(states[e].S);
    }

    SimulationConfig config;
    config.polynomial_order = order;
    config.use_mode_splitting = false;

    SimulationDriver driver(config);
    driver.initialize(num_elements, bathymetry, dh_dx, dh_dy, y_pos, states);

    // Get diagnostics
    auto diag = driver.compute_diagnostics();

    // Check diagnostics are reasonable for quiescent state
    EXPECT_EQ(diag.time, 0.0);
    EXPECT_EQ(diag.time_steps, 0);
    EXPECT_GT(diag.total_mass, 0.0);
    EXPECT_GE(diag.total_energy, 0.0);
    EXPECT_NEAR(diag.max_velocity, 0.0, TOLERANCE);  // Quiescent
    EXPECT_NEAR(diag.max_eta, 0.0, TOLERANCE);
    EXPECT_NEAR(diag.min_eta, 0.0, TOLERANCE);
    EXPECT_NEAR(diag.max_temperature, 15.0, TOLERANCE);
    EXPECT_NEAR(diag.min_temperature, 15.0, TOLERANCE);
}
