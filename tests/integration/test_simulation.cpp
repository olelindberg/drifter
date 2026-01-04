// Integration tests for simulation driver and physics
// Tests that the full simulation pipeline works correctly

#include <gtest/gtest.h>
#include "solver/simulation_driver.hpp"
#include "solver/time_stepper.hpp"
#include "mesh/octree_adapter.hpp"
#include "dg/basis_hexahedron.hpp"
#include "dg/quadrature_3d.hpp"
#include "dg/operators_3d.hpp"
#include "io/vtk_writer.hpp"
#include "physics/primitive_equations.hpp"
#include "physics/mode_splitting.hpp"
#include "../test_utils.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>

using namespace drifter;
using namespace drifter::testing;

class SimulationTest : public DrifterTestBase {
protected:
    void SetUp() override {
        DrifterTestBase::SetUp();

        // Create test output directory
        test_output_dir_ = "/tmp/drifter_test_output";
        std::filesystem::create_directories(test_output_dir_);
    }

    void TearDown() override {
        // Clean up test output files
        std::filesystem::remove_all(test_output_dir_);
        DrifterTestBase::TearDown();
    }

    std::string test_output_dir_;
};

// =============================================================================
// Simple sanity checks
// =============================================================================

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

// =============================================================================
// Mesh and basis tests
// =============================================================================

TEST_F(SimulationTest, OctreeMeshCreation) {
    // Create a simple 2x2x2 mesh
    OctreeAdapter mesh(0.0, 1000.0, 0.0, 1000.0, -100.0, 0.0);
    mesh.build_uniform(2, 2, 2);

    EXPECT_EQ(mesh.num_elements(), 8);

    // Check element bounds
    for (Index e = 0; e < mesh.num_elements(); ++e) {
        const auto& bounds = mesh.element_bounds(e);
        EXPECT_GE(bounds.xmin, 0.0);
        EXPECT_LE(bounds.xmax, 1000.0);
        EXPECT_GE(bounds.ymin, 0.0);
        EXPECT_LE(bounds.ymax, 1000.0);
        EXPECT_GE(bounds.zmin, -100.0);
        EXPECT_LE(bounds.zmax, 0.0);
    }
}

TEST_F(SimulationTest, FaceConnections) {
    // Create mesh and check face connections
    OctreeAdapter mesh(0.0, 1000.0, 0.0, 1000.0, -100.0, 0.0);
    mesh.build_uniform(2, 2, 1);  // 2x2x1 = 4 elements

    auto connections = mesh.build_face_connections();
    EXPECT_EQ(connections.size(), 4);

    // Each element should have 6 faces
    for (const auto& elem_conns : connections) {
        EXPECT_EQ(elem_conns.size(), 6);
    }

    // Count boundary vs interior faces
    int boundary_count = 0;
    int interior_count = 0;

    for (Index e = 0; e < mesh.num_elements(); ++e) {
        for (int f = 0; f < 6; ++f) {
            if (connections[e][f].is_boundary()) {
                ++boundary_count;
            } else {
                ++interior_count;
            }
        }
    }

    // For 2x2x1 mesh:
    // - 4 elements * 6 faces = 24 total face slots
    // - Bottom/top faces: 4*2 = 8 boundary
    // - Side faces: depends on arrangement
    EXPECT_GT(boundary_count, 0);
    EXPECT_GT(interior_count, 0);
}

// =============================================================================
// Initial condition tests
// =============================================================================

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

// =============================================================================
// VTK Output tests
// =============================================================================

TEST_F(SimulationTest, VTKWriterCreation) {
    std::string basename = test_output_dir_ + "/test_vtk";

    VTKWriter writer(basename, VTKFormat::VTU, VTKEncoding::ASCII);
    writer.set_polynomial_order(2);

    // Create simple mesh
    OctreeAdapter mesh(0.0, 100.0, 0.0, 100.0, -10.0, 0.0);
    mesh.build_uniform(2, 2, 1);

    writer.set_mesh(mesh);
    writer.add_point_data("test_scalar", 1);

    // Set some data
    int n_per_elem = 3 * 3 * 3;  // Order 2
    std::vector<VecX> data(mesh.num_elements());
    for (Index e = 0; e < mesh.num_elements(); ++e) {
        data[e] = VecX::Constant(n_per_elem, static_cast<Real>(e));
    }
    writer.set_point_data("test_scalar", data);

    // Write file
    writer.write(0, 0.0);

    // Check file exists
    std::string filename = writer.get_filename(0);
    EXPECT_TRUE(std::filesystem::exists(filename));

    // Check file has content
    std::ifstream file(filename);
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    EXPECT_GT(content.size(), 100);  // Should have substantial content
    EXPECT_NE(content.find("VTKFile"), std::string::npos);
    EXPECT_NE(content.find("UnstructuredGrid"), std::string::npos);
}

TEST_F(SimulationTest, VTKLegacyWriter) {
    std::string filename = test_output_dir_ + "/test_legacy.vtk";

    VTKLegacyWriter writer(filename);

    // Write simple cube
    std::vector<Vec3> points = {
        {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
        {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}
    };
    writer.write_points(points);

    std::vector<std::array<Index, 8>> cells = {
        {0, 1, 2, 3, 4, 5, 6, 7}
    };
    writer.write_hexahedra(cells);

    VecX scalar(8);
    scalar << 0, 1, 2, 3, 4, 5, 6, 7;
    writer.add_point_scalar("node_id", scalar);

    writer.close();

    // Check file exists and has content
    EXPECT_TRUE(std::filesystem::exists(filename));

    std::ifstream file(filename);
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("POINTS 8"), std::string::npos);
    EXPECT_NE(content.find("CELLS 1"), std::string::npos);
    EXPECT_NE(content.find("POINT_DATA"), std::string::npos);
}

TEST_F(SimulationTest, VTKPVDCollection) {
    std::string basename = test_output_dir_ + "/time_series";

    VTKWriter writer(basename, VTKFormat::VTU, VTKEncoding::ASCII);
    writer.set_polynomial_order(1);

    OctreeAdapter mesh(0.0, 10.0, 0.0, 10.0, -1.0, 0.0);
    mesh.build_uniform(1, 1, 1);
    writer.set_mesh(mesh);

    writer.add_point_data("eta", 1);

    // Write multiple timesteps
    for (int t = 0; t < 3; ++t) {
        std::vector<VecX> eta(1);
        eta[0] = VecX::Constant(8, static_cast<Real>(t) * 0.1);
        writer.set_point_data("eta", eta);
        writer.write_timestep(static_cast<Real>(t));
    }

    // Finalize creates PVD file
    writer.finalize();

    // Check PVD file exists
    std::string pvd_file = basename + ".pvd";
    EXPECT_TRUE(std::filesystem::exists(pvd_file));

    std::ifstream file(pvd_file);
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("Collection"), std::string::npos);
    EXPECT_NE(content.find("timestep"), std::string::npos);
}

// =============================================================================
// Simple advection test (sanity check)
// =============================================================================

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

// =============================================================================
// Kelvin wave test
// =============================================================================

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

// =============================================================================
// Full simulation with VTK output
// =============================================================================

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

// =============================================================================
// Diagnostics test
// =============================================================================

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

// =============================================================================
// GeoTIFF Bathymetry Mesh Generation Test
// =============================================================================

#include "mesh/geotiff_reader.hpp"
#include "mesh/coastline_refinement.hpp"

// Configurable bathymetry GeoTIFF file path
// Available options:
//   1. Danish Depth Model (50m resolution, large file):
//      "/home/ole/Projects/SeaMesh/data/Klimadatastyrelsen/DanmarksDybdeModel/2024/ddm_50m.dybde.tiff"
//   2. EMODNET bathymetry (blended):
//      "/home/ole/Projects/SeaMesh/data/bathymetry-blender/ddm_emodnet_bathymetry.tif"
const std::string BATHYMETRY_GEOTIFF_PATH =
    "/home/ole/Projects/SeaMesh/data/bathymetry-blender/ddm_emodnet_bathymetry.tif";

// Create VTK point storage ordering: maps VTK point ID to tensor-product index
// VTK stores points with 8 corners first, then remaining points by layer
std::vector<int> create_vtk_point_ordering(int order) {
    int n = order + 1;
    std::vector<int> ordering;

    auto tensor_idx = [n](int i, int j, int k) { return i + j * n + k * n * n; };

    // 8 corners first (VTK's corner ordering)
    ordering.push_back(tensor_idx(0, 0, 0));
    ordering.push_back(tensor_idx(order, 0, 0));
    ordering.push_back(tensor_idx(0, order, 0));
    ordering.push_back(tensor_idx(order, order, 0));
    ordering.push_back(tensor_idx(0, 0, order));
    ordering.push_back(tensor_idx(order, 0, order));
    ordering.push_back(tensor_idx(0, order, order));
    ordering.push_back(tensor_idx(order, order, order));

    if (order < 2) return ordering;

    // Remaining points in layer order (k varies slowest)
    for (int k = 0; k <= order; ++k) {
        for (int j = 0; j <= order; ++j) {
            for (int i = 0; i <= order; ++i) {
                bool is_corner = (i == 0 || i == order) &&
                                 (j == 0 || j == order) &&
                                 (k == 0 || k == order);
                if (!is_corner) {
                    ordering.push_back(tensor_idx(i, j, k));
                }
            }
        }
    }

    return ordering;
}

// VTK Lagrange hexahedron connectivity ordering for order 3
// This is the node ordering VTK expects in the CONNECTIVITY section
// Extracted from VTK's vtkCellTypeSource output
const std::vector<int> VTK_LAGRANGE_HEX_CONNECTIVITY_ORDER3 = {
    0, 1, 3, 2, 4, 5, 7, 6, 8,
    9, 13, 17, 18, 19, 10, 14, 52, 53,
    57, 61, 62, 63, 54, 58, 20, 36, 23,
    39, 35, 51, 32, 48, 24, 28, 40, 44,
    27, 31, 43, 47, 21, 22, 37, 38, 33,
    34, 49, 50, 11, 12, 15, 16, 55, 56,
    59, 60, 25, 26, 29, 30, 41, 42, 45,
    46
};

// Get VTK Lagrange hex connectivity for given order
// For now only order 3 is supported
std::vector<int> get_vtk_lagrange_hex_connectivity(int order) {
    if (order == 3) {
        return VTK_LAGRANGE_HEX_CONNECTIVITY_ORDER3;
    }
    // For other orders, would need to generate dynamically
    // For now, return empty and caller should handle
    return {};
}

TEST_F(SimulationTest, GeoTiffBathymetryMesh) {
    // Test loading real Danish bathymetry data and generating mesh
    // Outputs VTK file for visualization in ParaView

    // Check if GDAL is available
    if (!GeoTiffReader::is_available()) {
        GTEST_SKIP() << "GDAL not available, skipping GeoTIFF test";
    }

    // Use configurable bathymetry file path
    std::string geotiff_path = BATHYMETRY_GEOTIFF_PATH;

    // Check if file exists
    if (!std::filesystem::exists(geotiff_path)) {
        GTEST_SKIP() << "GeoTIFF file not found: " << geotiff_path;
    }

    // Load bathymetry
    GeoTiffReader reader;
    BathymetryData bathy = reader.load(geotiff_path);

    ASSERT_TRUE(bathy.is_valid()) << "Failed to load GeoTIFF: " << reader.last_error();

    std::cout << "Loaded bathymetry: " << bathy.sizex << " x " << bathy.sizey << " pixels\n";
    std::cout << "Bounds: x=[" << bathy.xmin << ", " << bathy.xmax << "], "
              << "y=[" << bathy.ymin << ", " << bathy.ymax << "]\n";

    // Create mesh generator with a subset of the domain for testing
    // Use a smaller region to keep the test fast
    auto bathy_ptr = std::make_shared<BathymetryData>(std::move(bathy));

    BathymetryMeshGenerator generator(bathy_ptr);

    BathymetryMeshGenerator::Config config;
    // Use coarse resolution for fast test
    config.base_nx = 20;
    config.base_ny = 20;
    config.base_nz = 3;
    config.mask_land = true;
    config.min_depth = 2.0;  // Only cells with at least 2m depth
    config.zmin = -1.0;      // Sigma coordinates
    config.zmax = 0.0;

    // Focus on a smaller region (e.g., around Kattegat)
    // The full Danish model is quite large
    Real center_x = (bathy_ptr->xmin + bathy_ptr->xmax) / 2;
    Real center_y = (bathy_ptr->ymin + bathy_ptr->ymax) / 2;
    Real region_size = 100000.0;  // 100 km region

    config.xmin = center_x - region_size / 2;
    config.xmax = center_x + region_size / 2;
    config.ymin = center_y - region_size / 2;
    config.ymax = center_y + region_size / 2;

    generator.set_config(config);

    // Generate mesh elements
    auto elements = generator.generate();

    std::cout << "Generated " << elements.size() << " mesh elements\n";

    // Skip if no water elements found in this region
    if (elements.empty()) {
        GTEST_SKIP() << "No water elements in selected region";
    }

    EXPECT_GT(elements.size(), 0);

    // Create high-order DG basis for node positions
    const int poly_order = 3;  // Polynomial order for visualization
    HexahedronBasis basis(poly_order);
    const auto& ref_nodes = basis.lgl_nodes();
    const int nodes_per_elem = static_cast<int>(ref_nodes.size());
    const int n1d = poly_order + 1;  // Nodes per direction

    std::cout << "Using polynomial order " << poly_order
              << " (" << nodes_per_elem << " nodes per element)\n";

    // Create VTK output for visualization
    std::string vtk_path = test_output_dir_ + "/danish_bathymetry";
    std::ofstream vtk_file(vtk_path + ".vtk");

    // Write VTK 5.1 header (required for Lagrange hexahedra)
    vtk_file << "# vtk DataFile Version 5.1\n";
    vtk_file << "DRIFTER Danish Bathymetry Mesh - High Order Elements\n";
    vtk_file << "ASCII\n";
    vtk_file << "DATASET UNSTRUCTURED_GRID\n";

    // Get VTK point ordering: maps VTK point ID to tensor-product index
    auto vtk_point_order = create_vtk_point_ordering(poly_order);

    // Collect all points with bathymetry-following coordinates
    // First collect in tensor-product order, then reorder for VTK
    std::vector<Vec3> all_points;
    std::vector<Real> point_depths;

    for (size_t e = 0; e < elements.size(); ++e) {
        const auto& bounds = elements[e];

        // Element dimensions in horizontal
        Real dx = bounds.xmax - bounds.xmin;
        Real dy = bounds.ymax - bounds.ymin;
        Real dz_sigma = bounds.zmax - bounds.zmin;

        // Collect points in tensor-product order first
        std::vector<Vec3> elem_points(nodes_per_elem);
        std::vector<Real> elem_depths(nodes_per_elem);

        for (size_t i = 0; i < ref_nodes.size(); ++i) {
            const auto& ref_node = ref_nodes[i];
            // Map reference coordinates [-1,1]^3 to physical coordinates
            // Horizontal: simple linear mapping
            Real x = bounds.xmin + 0.5 * (ref_node.x() + 1.0) * dx;
            Real y = bounds.ymin + 0.5 * (ref_node.y() + 1.0) * dy;

            // Sigma coordinate in [zmin, zmax]
            Real sigma = bounds.zmin + 0.5 * (ref_node.z() + 1.0) * dz_sigma;

            // Get local bathymetry depth at this (x,y) position
            Real local_depth = bathy_ptr->get_depth(x, y);

            // Physical z from sigma: z = sigma * H(x,y)
            // sigma=0 at surface, sigma=-1 at bottom
            Real z = sigma * local_depth;

            elem_points[i] = Vec3(x, y, z);
            elem_depths[i] = local_depth;
        }

        // Reorder to VTK point order (corners first, then remaining by layer)
        for (int vtk_id = 0; vtk_id < nodes_per_elem; ++vtk_id) {
            int tensor_idx = vtk_point_order[vtk_id];
            all_points.push_back(elem_points[tensor_idx]);
            point_depths.push_back(elem_depths[tensor_idx]);
        }
    }

    // Write points
    vtk_file << "POINTS " << all_points.size() << " double\n";
    for (const auto& p : all_points) {
        vtk_file << p.x() << " " << p.y() << " " << p.z() << "\n";
    }

    // Write cells as VTK_LAGRANGE_HEXAHEDRON (cell type 72) using VTK 5.1 format
    size_t num_cells = elements.size();

    // Get the VTK connectivity ordering
    auto vtk_connectivity = get_vtk_lagrange_hex_connectivity(poly_order);

    // VTK 5.1 format: CELLS <num_offsets> <num_connectivity_entries>
    // num_offsets = num_cells + 1, num_connectivity = num_cells * nodes_per_elem
    size_t num_offsets = num_cells + 1;
    size_t num_connectivity = num_cells * nodes_per_elem;
    vtk_file << "\nCELLS " << num_offsets << " " << num_connectivity << "\n";
    vtk_file << "OFFSETS vtktypeint64\n";
    for (size_t e = 0; e <= num_cells; ++e) {
        vtk_file << (e * nodes_per_elem) << " ";
    }
    vtk_file << "\n";
    vtk_file << "CONNECTIVITY vtktypeint64\n";
    for (size_t e = 0; e < num_cells; ++e) {
        Index base = static_cast<Index>(e * nodes_per_elem);
        for (int conn_idx : vtk_connectivity) {
            vtk_file << (base + conn_idx) << " ";
        }
        vtk_file << "\n";
    }

    // Cell types: VTK_LAGRANGE_HEXAHEDRON = 72
    vtk_file << "\nCELL_TYPES " << num_cells << "\n";
    for (size_t e = 0; e < num_cells; ++e) {
        vtk_file << "72\n";
    }

    // Point data: depth at each node
    vtk_file << "\nPOINT_DATA " << all_points.size() << "\n";
    vtk_file << "SCALARS depth double 1\n";
    vtk_file << "LOOKUP_TABLE default\n";
    for (Real d : point_depths) {
        vtk_file << d << "\n";
    }

    // Cell data: element ID
    vtk_file << "\nCELL_DATA " << num_cells << "\n";
    vtk_file << "SCALARS element_id double 1\n";
    vtk_file << "LOOKUP_TABLE default\n";
    for (size_t e = 0; e < num_cells; ++e) {
        vtk_file << e << "\n";
    }

    vtk_file.close();

    // Verify output file exists
    std::string vtk_filename = vtk_path + ".vtk";
    EXPECT_TRUE(std::filesystem::exists(vtk_filename));

    std::cout << "VTK output written to: " << vtk_filename << "\n";
    std::cout << "Open in ParaView to visualize the Danish bathymetry mesh\n";
    std::cout << "High-order Lagrange hexahedra with bathymetry-following bottom surface\n";

    // Copy to a persistent location for easy access
    std::string persistent_path = "/tmp/danish_bathymetry.vtk";
    std::filesystem::copy_file(vtk_filename, persistent_path,
                                std::filesystem::copy_options::overwrite_existing);
    std::cout << "Also copied to: " << persistent_path << "\n";
}

TEST_F(SimulationTest, GeoTiffFullDomainMesh) {
    // Test generating mesh for the full Danish bathymetry domain
    // This creates a coarser mesh suitable for overview visualization

    if (!GeoTiffReader::is_available()) {
        GTEST_SKIP() << "GDAL not available, skipping GeoTIFF test";
    }

    // Use configurable bathymetry file path
    std::string geotiff_path = BATHYMETRY_GEOTIFF_PATH;

    if (!std::filesystem::exists(geotiff_path)) {
        GTEST_SKIP() << "GeoTIFF file not found: " << geotiff_path;
    }

    GeoTiffReader reader;
    BathymetryData bathy = reader.load(geotiff_path);
    ASSERT_TRUE(bathy.is_valid());

    auto bathy_ptr = std::make_shared<BathymetryData>(std::move(bathy));

    BathymetryMeshGenerator generator(bathy_ptr);

    BathymetryMeshGenerator::Config config;
    // Coarse mesh for full domain
    config.base_nx = 50;
    config.base_ny = 50;
    config.base_nz = 1;  // Single layer for 2D overview
    config.mask_land = true;
    config.min_depth = 1.0;

    generator.set_config(config);

    auto elements = generator.generate();

    std::cout << "Full domain mesh: " << elements.size() << " elements\n";

    if (elements.empty()) {
        GTEST_SKIP() << "No water elements generated";
    }

    // Create high-order DG basis for node positions
    const int poly_order = 3;
    HexahedronBasis basis(poly_order);
    const auto& ref_nodes = basis.lgl_nodes();
    const int nodes_per_elem = static_cast<int>(ref_nodes.size());

    // Write VTK for full domain with high-order elements
    std::string vtk_path = "/tmp/danish_bathymetry_full.vtk";
    std::ofstream vtk_file(vtk_path);

    // VTK 5.1 format (required for Lagrange hexahedra)
    vtk_file << "# vtk DataFile Version 5.1\n";
    vtk_file << "DRIFTER Danish Bathymetry Full Domain - High Order Elements\n";
    vtk_file << "ASCII\n";
    vtk_file << "DATASET UNSTRUCTURED_GRID\n";

    // Get VTK point ordering: maps VTK point ID to tensor-product index
    auto vtk_point_order = create_vtk_point_ordering(poly_order);

    // Collect all points with bathymetry-following coordinates
    // First collect in tensor-product order, then reorder for VTK
    std::vector<Vec3> all_points;
    std::vector<Real> point_depths;

    for (const auto& bounds : elements) {
        Real dx = bounds.xmax - bounds.xmin;
        Real dy = bounds.ymax - bounds.ymin;
        Real dz_sigma = bounds.zmax - bounds.zmin;

        // Collect points in tensor-product order first
        std::vector<Vec3> elem_points(nodes_per_elem);
        std::vector<Real> elem_depths(nodes_per_elem);

        for (size_t i = 0; i < ref_nodes.size(); ++i) {
            const auto& ref_node = ref_nodes[i];
            Real x = bounds.xmin + 0.5 * (ref_node.x() + 1.0) * dx;
            Real y = bounds.ymin + 0.5 * (ref_node.y() + 1.0) * dy;
            Real sigma = bounds.zmin + 0.5 * (ref_node.z() + 1.0) * dz_sigma;

            Real local_depth = bathy_ptr->get_depth(x, y);
            Real z = sigma * local_depth;

            elem_points[i] = Vec3(x, y, z);
            elem_depths[i] = local_depth;
        }

        // Reorder to VTK point order (corners first, then remaining by layer)
        for (int vtk_id = 0; vtk_id < nodes_per_elem; ++vtk_id) {
            int tensor_idx = vtk_point_order[vtk_id];
            all_points.push_back(elem_points[tensor_idx]);
            point_depths.push_back(elem_depths[tensor_idx]);
        }
    }

    // Write points
    vtk_file << "POINTS " << all_points.size() << " double\n";
    for (const auto& p : all_points) {
        vtk_file << p.x() << " " << p.y() << " " << p.z() << "\n";
    }

    // Write cells as VTK_LAGRANGE_HEXAHEDRON using VTK 5.1 format
    size_t num_cells = elements.size();
    auto vtk_connectivity = get_vtk_lagrange_hex_connectivity(poly_order);

    // VTK 5.1 format: CELLS <num_offsets> <num_connectivity_entries>
    size_t num_offsets = num_cells + 1;
    size_t num_connectivity = num_cells * nodes_per_elem;
    vtk_file << "\nCELLS " << num_offsets << " " << num_connectivity << "\n";
    vtk_file << "OFFSETS vtktypeint64\n";
    for (size_t e = 0; e <= num_cells; ++e) {
        vtk_file << (e * nodes_per_elem) << " ";
    }
    vtk_file << "\n";
    vtk_file << "CONNECTIVITY vtktypeint64\n";
    for (size_t e = 0; e < num_cells; ++e) {
        Index base = static_cast<Index>(e * nodes_per_elem);
        for (int conn_idx : vtk_connectivity) {
            vtk_file << (base + conn_idx) << " ";
        }
        vtk_file << "\n";
    }

    vtk_file << "\nCELL_TYPES " << num_cells << "\n";
    for (size_t e = 0; e < num_cells; ++e) {
        vtk_file << "72\n";
    }

    vtk_file << "\nPOINT_DATA " << all_points.size() << "\n";
    vtk_file << "SCALARS depth double 1\n";
    vtk_file << "LOOKUP_TABLE default\n";
    for (Real d : point_depths) {
        vtk_file << d << "\n";
    }

    vtk_file.close();

    EXPECT_TRUE(std::filesystem::exists(vtk_path));
    std::cout << "Full domain VTK: " << vtk_path << "\n";
}

// Configurable land polygon GeoPackage path
const std::string LAND_POLYGON_GPKG_PATH =
    "/home/ole/Projects/SeaMesh/data/Klimadatastyrelsen/TopografiskLandpolygon/landpolygon.gpkg";
const std::string LAND_POLYGON_LAYER = "landpolygon_2500";
const std::string LAND_POLYGON_PROJECTION = "EPSG:3034";

TEST_F(SimulationTest, CoastlineAdaptiveOctreeMesh) {
    // Test generating a coastline-adaptive multiresolution octree mesh
    // Elements are refined near the coastline (land polygon boundary)
    // This reproduces the SeaMesh coastline-adaptive example using land polygons

    if (!GeoTiffReader::is_available()) {
        GTEST_SKIP() << "GDAL not available, skipping GeoTIFF test";
    }

    // Load bathymetry for domain bounds and depth values
    std::string geotiff_path = BATHYMETRY_GEOTIFF_PATH;

    if (!std::filesystem::exists(geotiff_path)) {
        GTEST_SKIP() << "GeoTIFF file not found: " << geotiff_path;
    }

    GeoTiffReader reader;
    BathymetryData bathy = reader.load(geotiff_path);
    ASSERT_TRUE(bathy.is_valid());

    std::cout << "Loaded bathymetry: " << bathy.sizex << " x " << bathy.sizey << " pixels\n";
    std::cout << "Bounds: x=[" << bathy.xmin << ", " << bathy.xmax << "], "
              << "y=[" << bathy.ymin << ", " << bathy.ymax << "]\n";

    auto bathy_ptr = std::make_shared<BathymetryData>(std::move(bathy));

    // Load land polygon for coastline refinement
    if (!std::filesystem::exists(LAND_POLYGON_GPKG_PATH)) {
        GTEST_SKIP() << "Land polygon file not found: " << LAND_POLYGON_GPKG_PATH;
    }

    std::cout << "Loading land polygon from: " << LAND_POLYGON_GPKG_PATH << "\n";

    CoastlineReader coastline_reader;
    bool loaded = coastline_reader.load(LAND_POLYGON_GPKG_PATH,
                                         LAND_POLYGON_LAYER,
                                         LAND_POLYGON_PROJECTION);
    ASSERT_TRUE(loaded) << "Failed to load land polygon: " << coastline_reader.last_error();

    // Swap X/Y coordinates (required for EPSG:3034)
    coastline_reader.swap_xy();

    // Remove small islands (area < 1 km²)
    const double min_area = 1.0e6;  // 1 km² in m²
    coastline_reader.remove_small_polygons(min_area);

    std::cout << "Loaded " << coastline_reader.num_polygons() << " land polygons\n";
    auto bbox = coastline_reader.bounding_box();
    std::cout << "Coastline bounds: x=[" << bg::get<0>(bbox.min_corner()) << ", "
              << bg::get<0>(bbox.max_corner()) << "], y=["
              << bg::get<1>(bbox.min_corner()) << ", "
              << bg::get<1>(bbox.max_corner()) << "]\n";

    // Build R-tree index for fast coastline intersection queries
    auto coastline_index = std::make_shared<CoastlineIndex>();
    coastline_index->build(coastline_reader.polygons());
    std::cout << "Built coastline R-tree with " << coastline_index->num_segments() << " segments\n";

    // Build adaptive octree using coastline refinement criteria
    OctreeAdapter octree(bathy_ptr->xmin, bathy_ptr->xmax,
                         bathy_ptr->ymin, bathy_ptr->ymax,
                         -1.0, 0.0);  // Sigma coordinates

    // Maximum refinement levels per axis
    const int max_level_x = 6;
    const int max_level_y = 6;
    const int max_level_z = 0;  // No vertical refinement for 2D view

    // Create coastline refinement criterion
    CoastlineRefinement coastline_criterion(coastline_index, max_level_x);

    // Build adaptive mesh: refine where element intersects coastline
    octree.build_adaptive(
        [&coastline_criterion](const ElementBounds& bounds) -> bool {
            // Use R-tree based intersection test
            return coastline_criterion.should_refine(bounds, 0);
        },
        max_level_x, max_level_y, max_level_z);

    // Balance the octree to ensure 2:1 constraint
    octree.balance();

    std::cout << "Adaptive octree mesh: " << octree.num_elements() << " elements\n";
    ASSERT_GT(octree.num_elements(), 0);

    // Create high-order DG basis for node positions
    const int poly_order = 3;
    HexahedronBasis basis(poly_order);
    const auto& ref_nodes = basis.lgl_nodes();
    const int nodes_per_elem = static_cast<int>(ref_nodes.size());

    // Write VTK output for visualization
    std::string vtk_path = "/tmp/danish_bathymetry_adaptive.vtk";
    std::ofstream vtk_file(vtk_path);

    // VTK 5.1 format (required for Lagrange hexahedra)
    vtk_file << "# vtk DataFile Version 5.1\n";
    vtk_file << "DRIFTER Coastline Adaptive Octree Mesh - High Order Elements\n";
    vtk_file << "ASCII\n";
    vtk_file << "DATASET UNSTRUCTURED_GRID\n";

    // Get VTK point ordering
    auto vtk_point_order = create_vtk_point_ordering(poly_order);

    // Collect all points with bathymetry-following coordinates
    std::vector<Vec3> all_points;
    std::vector<Real> point_depths;
    std::vector<Real> point_levels;  // Store refinement level for visualization

    octree.for_each_element([&](Index e, const OctreeNode& node) {
        const auto& bounds = node.bounds;

        Real dx = bounds.xmax - bounds.xmin;
        Real dy = bounds.ymax - bounds.ymin;
        Real dz_sigma = bounds.zmax - bounds.zmin;

        // Collect points in tensor-product order first
        std::vector<Vec3> elem_points(nodes_per_elem);
        std::vector<Real> elem_depths(nodes_per_elem);

        for (size_t i = 0; i < ref_nodes.size(); ++i) {
            const auto& ref_node = ref_nodes[i];
            Real x = bounds.xmin + 0.5 * (ref_node.x() + 1.0) * dx;
            Real y = bounds.ymin + 0.5 * (ref_node.y() + 1.0) * dy;
            Real sigma = bounds.zmin + 0.5 * (ref_node.z() + 1.0) * dz_sigma;

            Real local_depth = bathy_ptr->get_depth(x, y);
            Real z = sigma * local_depth;

            elem_points[i] = Vec3(x, y, z);
            elem_depths[i] = local_depth;
        }

        // Reorder to VTK point order
        for (int vtk_id = 0; vtk_id < nodes_per_elem; ++vtk_id) {
            int tensor_idx = vtk_point_order[vtk_id];
            all_points.push_back(elem_points[tensor_idx]);
            point_depths.push_back(elem_depths[tensor_idx]);
            point_levels.push_back(static_cast<Real>(node.level.level_x));
        }
    });

    // Write points
    vtk_file << "POINTS " << all_points.size() << " double\n";
    for (const auto& p : all_points) {
        vtk_file << p.x() << " " << p.y() << " " << p.z() << "\n";
    }

    // Write cells as VTK_LAGRANGE_HEXAHEDRON using VTK 5.1 format
    size_t num_cells = octree.num_elements();
    auto vtk_connectivity = get_vtk_lagrange_hex_connectivity(poly_order);

    size_t num_offsets = num_cells + 1;
    size_t num_connectivity = num_cells * nodes_per_elem;
    vtk_file << "\nCELLS " << num_offsets << " " << num_connectivity << "\n";
    vtk_file << "OFFSETS vtktypeint64\n";
    for (size_t e = 0; e <= num_cells; ++e) {
        vtk_file << (e * nodes_per_elem) << " ";
    }
    vtk_file << "\n";
    vtk_file << "CONNECTIVITY vtktypeint64\n";
    for (size_t e = 0; e < num_cells; ++e) {
        Index base = static_cast<Index>(e * nodes_per_elem);
        for (int conn_idx : vtk_connectivity) {
            vtk_file << (base + conn_idx) << " ";
        }
        vtk_file << "\n";
    }

    vtk_file << "\nCELL_TYPES " << num_cells << "\n";
    for (size_t e = 0; e < num_cells; ++e) {
        vtk_file << "72\n";
    }

    // Point data
    vtk_file << "\nPOINT_DATA " << all_points.size() << "\n";
    vtk_file << "SCALARS depth double 1\n";
    vtk_file << "LOOKUP_TABLE default\n";
    for (Real d : point_depths) {
        vtk_file << d << "\n";
    }
    vtk_file << "SCALARS refinement_level double 1\n";
    vtk_file << "LOOKUP_TABLE default\n";
    for (Real lev : point_levels) {
        vtk_file << lev << "\n";
    }

    // Cell data: refinement level
    vtk_file << "\nCELL_DATA " << num_cells << "\n";
    vtk_file << "SCALARS cell_level double 1\n";
    vtk_file << "LOOKUP_TABLE default\n";
    octree.for_each_element([&](Index e, const OctreeNode& node) {
        vtk_file << node.level.level_x << "\n";
    });

    vtk_file.close();

    EXPECT_TRUE(std::filesystem::exists(vtk_path));
    std::cout << "Coastline-adaptive VTK: " << vtk_path << "\n";
    std::cout << "Elements refined near coastline with max level " << max_level_x << "\n";
}
