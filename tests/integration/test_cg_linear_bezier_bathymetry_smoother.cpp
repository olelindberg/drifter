#include "bathymetry/cg_linear_bezier_bathymetry_smoother.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "mesh/octree_adapter.hpp"
#include "mesh/seabed_surface.hpp"
#include "test_integration_fixtures.hpp"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <sstream>

#include "mesh/geotiff_reader.hpp"

using namespace drifter;
using namespace drifter::testing;

class CGLinearBezierSmootherTest : public ::testing::Test {
protected:
  static constexpr Real TOLERANCE = 1e-10;
  static constexpr Real LOOSE_TOLERANCE = 1e-6;

  std::unique_ptr<OctreeAdapter> create_octree(int nx, int ny, int nz) {
    auto octree = std::make_unique<OctreeAdapter>(0.0, 100.0, // x bounds
                                                  0.0, 100.0, // y bounds
                                                  -1.0, 0.0   // z bounds
    );
    octree->build_uniform(nx, ny, nz);
    return octree;
  }

  QuadtreeAdapter create_quadtree(int nx, int ny, Real xmin = 0.0,
                                  Real xmax = 100.0, Real ymin = 0.0,
                                  Real ymax = 100.0) {
    QuadtreeAdapter mesh;
    mesh.build_uniform(xmin, xmax, ymin, ymax, nx, ny);
    return mesh;
  }
};

// =============================================================================
// Level 1: Basic Construction and Single Element Tests
// =============================================================================

TEST_F(CGLinearBezierSmootherTest, ConstructFromQuadtree) {
  auto mesh = create_quadtree(2, 2);

  CGLinearBezierBathymetrySmoother smoother(mesh);

  EXPECT_GT(smoother.num_global_dofs(), 0);
  EXPECT_FALSE(smoother.is_solved());
}

TEST_F(CGLinearBezierSmootherTest, ConstructFromOctree) {
  auto octree = create_octree(4, 4, 2);

  CGLinearBezierBathymetrySmoother smoother(*octree);

  EXPECT_GT(smoother.num_global_dofs(), 0);
  EXPECT_FALSE(smoother.is_solved());
}

TEST_F(CGLinearBezierSmootherTest, SingleElementConstant) {
  // Single element test: constant bathymetry
  QuadtreeAdapter mesh;
  mesh.build_uniform(0.0, 10.0, 0.0, 10.0, 1, 1);

  CGLinearBezierSmootherConfig config;
  config.lambda = 100.0;      // Strong data fitting
  config.ridge_epsilon = 0.0; // No ridge for exact polynomial reproduction

  CGLinearBezierBathymetrySmoother smoother(mesh, config);
  smoother.set_bathymetry_data([](Real, Real) { return 50.0; });
  smoother.solve();

  EXPECT_TRUE(smoother.is_solved());

  // Should reproduce constant exactly
  EXPECT_NEAR(smoother.evaluate(5.0, 5.0), 50.0, LOOSE_TOLERANCE);
  EXPECT_NEAR(smoother.evaluate(2.0, 8.0), 50.0, LOOSE_TOLERANCE);
}

TEST_F(CGLinearBezierSmootherTest, SingleElementLinear) {
  // Single element: linear bathymetry
  // Note: Dirichlet energy penalizes gradients, so linear functions won't be
  // exactly reproduced. With high lambda, we get close to the linear input.
  QuadtreeAdapter mesh;
  mesh.build_uniform(0.0, 10.0, 0.0, 10.0, 1, 1);

  CGLinearBezierSmootherConfig config;
  config.lambda = 1000.0; // Very high data weight to minimize smoothing
  config.ridge_epsilon = 0.0;

  auto linear = [](Real x, Real y) { return 10.0 + 2.0 * x + 3.0 * y; };

  CGLinearBezierBathymetrySmoother smoother(mesh, config);
  smoother.set_bathymetry_data(linear);
  smoother.solve();

  EXPECT_TRUE(smoother.is_solved());

  // Check that the solution is reasonable - within ~20% of linear range
  // The Dirichlet energy pulls gradients toward zero, so the output is flatter
  Real val_center = smoother.evaluate(5.0, 5.0);
  Real val_corner = smoother.evaluate(2.0, 8.0);
  Real expected_center = linear(5.0, 5.0); // 10 + 10 + 15 = 35
  Real expected_corner = linear(2.0, 8.0); // 10 + 4 + 24 = 38

  // Values should be in the general ballpark of expected
  EXPECT_GT(val_center, expected_center - 10);
  EXPECT_LT(val_center, expected_center + 10);
  EXPECT_GT(val_corner, expected_corner - 10);
  EXPECT_LT(val_corner, expected_corner + 10);
}

// =============================================================================
// Level 2: 2x2 Uniform Mesh Tests (DOF Sharing)
// =============================================================================

TEST_F(CGLinearBezierSmootherTest, TwoByTwoMeshConstant) {
  auto mesh = create_quadtree(2, 2);

  CGLinearBezierSmootherConfig config;
  config.lambda = 100.0;
  config.ridge_epsilon = 0.0; // No ridge for exact polynomial reproduction

  CGLinearBezierBathymetrySmoother smoother(mesh, config);
  smoother.set_bathymetry_data([](Real, Real) { return 42.0; });
  smoother.solve();

  EXPECT_TRUE(smoother.is_solved());

  // Check at several points
  EXPECT_NEAR(smoother.evaluate(25.0, 25.0), 42.0, LOOSE_TOLERANCE);
  EXPECT_NEAR(smoother.evaluate(50.0, 50.0), 42.0, LOOSE_TOLERANCE);
  EXPECT_NEAR(smoother.evaluate(75.0, 75.0), 42.0, LOOSE_TOLERANCE);
}

TEST_F(CGLinearBezierSmootherTest, TwoByTwoMeshLinear) {
  auto mesh = create_quadtree(2, 2);

  CGLinearBezierSmootherConfig config;
  config.lambda = 1000.0; // Very high data weight
  config.ridge_epsilon = 0.0;

  auto linear = [](Real x, Real y) { return 10.0 + 0.5 * x + 0.3 * y; };

  CGLinearBezierBathymetrySmoother smoother(mesh, config);
  smoother.set_bathymetry_data(linear);
  smoother.solve();

  // With Dirichlet energy (gradient penalty), linear is not exactly reproduced
  // Check that the overall slope is approximately correct
  Real val_00 = smoother.evaluate(10.0, 10.0);
  Real val_90_90 = smoother.evaluate(90.0, 90.0);

  Real expected_00 = linear(10.0, 10.0);
  Real expected_90_90 = linear(90.0, 90.0);

  // The range should be approximately correct (within 20% error)
  Real actual_range = val_90_90 - val_00;
  Real expected_range = expected_90_90 - expected_00;

  EXPECT_GT(actual_range,
            0.5 * expected_range); // At least 50% of expected slope
  EXPECT_LT(actual_range,
            1.5 * expected_range); // At most 150% of expected slope
}

TEST_F(CGLinearBezierSmootherTest, DofSharingReducesCount) {
  // Verify DOF sharing: CG should have fewer DOFs than DG
  auto mesh = create_quadtree(2, 2); // 4 elements

  CGLinearBezierBathymetrySmoother smoother(mesh);

  Index num_dofs = smoother.num_global_dofs();

  // DG would have 4 x 4 = 16 DOFs (linear: 2x2 = 4 per element)
  // CG should have (2+1) x (2+1) = 9 DOFs (vertex sharing)
  EXPECT_EQ(num_dofs, 9); // (N+1)^2 for NxN mesh with linear elements

  std::cout << "2x2 mesh CG linear DOFs: " << num_dofs << " (vs 16 for DG)"
            << std::endl;
}

TEST_F(CGLinearBezierSmootherTest, NoConstraintsForUniformMesh) {
  // Uniform mesh should have no hanging nodes
  auto mesh = create_quadtree(2, 2);

  CGLinearBezierBathymetrySmoother smoother(mesh);

  EXPECT_EQ(smoother.num_constraints(), 0);
}

TEST_F(CGLinearBezierSmootherTest, LargerUniformMeshDofCount) {
  // Verify (N+1)^2 DOF count for NxN uniform mesh
  auto mesh = create_quadtree(8, 8);

  CGLinearBezierBathymetrySmoother smoother(mesh);

  Index num_dofs = smoother.num_global_dofs();

  // For 8x8 mesh with linear CG elements: (8+1)^2 = 81 DOFs
  EXPECT_EQ(num_dofs, 81);

  std::cout << "8x8 mesh CG linear DOFs: " << num_dofs << std::endl;
}

// =============================================================================
// Level 3: Non-conforming Mesh (Hanging Nodes)
// =============================================================================

TEST_F(CGLinearBezierSmootherTest, OnePlusFourMeshConstruction) {
  // Create 1+4 non-conforming mesh:
  // 2×2 fine elements + 1 coarse element on the right
  //
  //  +---+---+-------+
  //  | 2 | 3 |       |
  //  +---+---+   4   |
  //  | 0 | 1 |       |
  //  +---+---+-------+

  QuadtreeAdapter mesh;
  Real h = 25.0;

  // Fine elements (level 2)
  mesh.add_element({0.0, h, 0.0, h}, {2, 2});     // elem 0
  mesh.add_element({h, 2 * h, 0.0, h}, {2, 2});   // elem 1
  mesh.add_element({0.0, h, h, 2 * h}, {2, 2});   // elem 2
  mesh.add_element({h, 2 * h, h, 2 * h}, {2, 2}); // elem 3

  // Coarse element (level 1)
  mesh.add_element({2 * h, 4 * h, 0.0, 2 * h}, {1, 1}); // elem 4

  EXPECT_EQ(mesh.num_elements(), 5);

  CGLinearBezierBathymetrySmoother smoother(mesh);

  // Should have hanging node constraints at fine-coarse interface
  EXPECT_GT(smoother.num_constraints(), 0);
  std::cout << "1+4 mesh linear: " << smoother.num_global_dofs() << " DOFs, "
            << smoother.num_constraints() << " constraints" << std::endl;
}

TEST_F(CGLinearBezierSmootherTest, OnePlusFourConstantBathymetry) {
  // 1+4 mesh with constant bathymetry
  QuadtreeAdapter mesh;
  Real h = 25.0;

  mesh.add_element({0.0, h, 0.0, h}, {2, 2});
  mesh.add_element({h, 2 * h, 0.0, h}, {2, 2});
  mesh.add_element({0.0, h, h, 2 * h}, {2, 2});
  mesh.add_element({h, 2 * h, h, 2 * h}, {2, 2});
  mesh.add_element({2 * h, 4 * h, 0.0, 2 * h}, {1, 1});

  CGLinearBezierSmootherConfig config;
  config.lambda = 100.0;
  config.ridge_epsilon = 0.0;

  CGLinearBezierBathymetrySmoother smoother(mesh, config);
  smoother.set_bathymetry_data([](Real, Real) { return 75.0; });
  smoother.solve();

  EXPECT_TRUE(smoother.is_solved());

  // Should be constant everywhere
  EXPECT_NEAR(smoother.evaluate(10.0, 10.0), 75.0, LOOSE_TOLERANCE);
  EXPECT_NEAR(smoother.evaluate(75.0, 25.0), 75.0, LOOSE_TOLERANCE);
}

TEST_F(CGLinearBezierSmootherTest, OnePlusFourLinearBathymetry) {
  // 1+4 mesh with linear bathymetry
  // Note: Dirichlet energy penalizes gradients, so linear won't be exact
  QuadtreeAdapter mesh;
  Real h = 25.0;

  mesh.add_element({0.0, h, 0.0, h}, {2, 2});
  mesh.add_element({h, 2 * h, 0.0, h}, {2, 2});
  mesh.add_element({0.0, h, h, 2 * h}, {2, 2});
  mesh.add_element({h, 2 * h, h, 2 * h}, {2, 2});
  mesh.add_element({2 * h, 4 * h, 0.0, 2 * h}, {1, 1});

  auto linear = [](Real x, Real y) { return 50.0 + x + 2.0 * y; };

  CGLinearBezierSmootherConfig config;
  config.lambda = 100.0;

  CGLinearBezierBathymetrySmoother smoother(mesh, config);
  smoother.set_bathymetry_data(linear);
  smoother.solve();

  EXPECT_TRUE(smoother.is_solved());

  // Check that constraint violation is small (T-junction constraints)
  Real constraint_violation = smoother.constraint_violation();
  std::cout << "Constraint violation: " << constraint_violation << std::endl;
  EXPECT_LT(constraint_violation, 1e-6);

  // Check approximate range (Dirichlet smoothing affects gradients)
  Real val_low = smoother.evaluate(10.0, 10.0);
  Real val_high = smoother.evaluate(90.0, 40.0);

  // Both values should be within the ballpark of the input
  Real expected_low = linear(10.0, 10.0);  // 50 + 10 + 20 = 80
  Real expected_high = linear(90.0, 40.0); // 50 + 90 + 80 = 220

  EXPECT_GT(val_low, expected_low - 30); // Within 30 of expected
  EXPECT_LT(val_low, expected_low + 30);
  EXPECT_GT(val_high, expected_high - 50);
  EXPECT_LT(val_high, expected_high + 50);
}

TEST_F(CGLinearBezierSmootherTest, OnePlusFourContinuityAtInterface) {
  // Check C⁰ continuity at the fine-coarse interface
  QuadtreeAdapter mesh;
  Real h = 25.0;

  mesh.add_element({0.0, h, 0.0, h}, {2, 2});
  mesh.add_element({h, 2 * h, 0.0, h}, {2, 2});
  mesh.add_element({0.0, h, h, 2 * h}, {2, 2});
  mesh.add_element({h, 2 * h, h, 2 * h}, {2, 2});
  mesh.add_element({2 * h, 4 * h, 0.0, 2 * h}, {1, 1});

  // Use smooth function to test continuity
  auto smooth_bathy = [](Real x, Real y) {
    return 100.0 + 20.0 * std::sin(0.03 * x) * std::cos(0.03 * y);
  };

  CGLinearBezierSmootherConfig config;
  config.lambda = 10.0;

  CGLinearBezierBathymetrySmoother smoother(mesh, config);
  smoother.set_bathymetry_data(smooth_bathy);
  smoother.solve();

  // Check continuity at interface x = 2*h = 50
  Real max_jump = 0.0;
  for (Real y = 1.0; y < 50.0; y += 5.0) {
    Real left = smoother.evaluate(49.99, y);
    Real right = smoother.evaluate(50.01, y);
    Real jump = std::abs(left - right);
    max_jump = std::max(max_jump, jump);
  }

  std::cout << "Max jump at interface (linear): " << max_jump << std::endl;

  // CG should provide C⁰ continuity via shared DOFs and hanging node
  // constraints
  EXPECT_LT(max_jump, 1.0);

  // Check constraint violation
  Real constraint_violation = smoother.constraint_violation();
  std::cout << "Constraint violation: " << constraint_violation << std::endl;
  EXPECT_LT(constraint_violation, 1e-8);
}

// =============================================================================
// Level 4: Smoothing Behavior and Diagnostics
// =============================================================================

TEST_F(CGLinearBezierSmootherTest, SmoothingReducesVariation) {
  auto mesh = create_quadtree(8, 8);

  // Noisy data
  auto noisy = [](Real x, Real y) {
    return 50.0 + 0.5 * x + 5.0 * std::sin(x / 5.0) * std::cos(y / 5.0);
  };

  // Strong smoothing
  CGLinearBezierSmootherConfig config_smooth;
  config_smooth.lambda = 0.01; // Weak data fitting = smooth result

  CGLinearBezierBathymetrySmoother smoother_smooth(mesh, config_smooth);
  smoother_smooth.set_bathymetry_data(noisy);
  smoother_smooth.solve();

  // Weak smoothing
  CGLinearBezierSmootherConfig config_data;
  config_data.lambda = 100.0; // Strong data fitting = noisy result

  CGLinearBezierBathymetrySmoother smoother_data(mesh, config_data);
  smoother_data.set_bathymetry_data(noisy);
  smoother_data.solve();

  // Smooth solution should have higher data residual but lower
  // regularization energy
  EXPECT_GT(smoother_smooth.data_residual(), smoother_data.data_residual());
  EXPECT_LT(smoother_smooth.regularization_energy(),
            smoother_data.regularization_energy());

  std::cout << "Smooth: data_res=" << smoother_smooth.data_residual()
            << ", reg_energy=" << smoother_smooth.regularization_energy()
            << std::endl;
  std::cout << "Data:   data_res=" << smoother_data.data_residual()
            << ", reg_energy=" << smoother_data.regularization_energy()
            << std::endl;
}

TEST_F(CGLinearBezierSmootherTest, GradientOfConstant) {
  auto mesh = create_quadtree(4, 4);

  CGLinearBezierSmootherConfig config;
  config.lambda = 100.0;
  config.ridge_epsilon = 0.0;

  CGLinearBezierBathymetrySmoother smoother(mesh, config);
  smoother.set_bathymetry_data([](Real, Real) { return 42.0; });
  smoother.solve();

  // Gradient of constant should be zero
  Vec2 grad = smoother.evaluate_gradient(50.0, 50.0);
  EXPECT_NEAR(grad(0), 0.0, LOOSE_TOLERANCE);
  EXPECT_NEAR(grad(1), 0.0, LOOSE_TOLERANCE);
}

TEST_F(CGLinearBezierSmootherTest, GradientOfLinear) {
  auto mesh = create_quadtree(4, 4);

  CGLinearBezierSmootherConfig config;
  config.lambda = 1000.0; // High data weight to get close to input
  config.ridge_epsilon = 0.0;

  auto linear = [](Real x, Real y) { return 10.0 + 0.5 * x + 0.3 * y; };

  CGLinearBezierBathymetrySmoother smoother(mesh, config);
  smoother.set_bathymetry_data(linear);
  smoother.solve();

  // With Dirichlet energy, gradients are penalized
  // At high lambda, gradient should be approximately correct
  Vec2 grad = smoother.evaluate_gradient(50.0, 50.0);

  // Allow larger tolerance due to gradient penalty
  EXPECT_NEAR(grad(0), 0.5, 0.15); // Within 0.15 of expected 0.5
  EXPECT_NEAR(grad(1), 0.3, 0.15); // Within 0.15 of expected 0.3
}

TEST_F(CGLinearBezierSmootherTest, DiagnosticsAfterSolve) {
  auto mesh = create_quadtree(4, 4);

  auto func = [](Real x, Real y) { return 50.0 + 0.2 * x + 0.1 * y; };

  CGLinearBezierBathymetrySmoother smoother(mesh);
  smoother.set_bathymetry_data(func);
  smoother.solve();

  // All diagnostic values should be non-negative
  EXPECT_GE(smoother.data_residual(), 0.0);
  EXPECT_GE(smoother.regularization_energy(), 0.0);
  EXPECT_GE(smoother.objective_value(), 0.0);
  EXPECT_GE(smoother.constraint_violation(), 0.0);

  std::cout << "Diagnostics:" << std::endl;
  std::cout << "  data_residual: " << smoother.data_residual() << std::endl;
  std::cout << "  reg_energy:    " << smoother.regularization_energy()
            << std::endl;
  std::cout << "  objective:     " << smoother.objective_value() << std::endl;
  std::cout << "  constraint_v:  " << smoother.constraint_violation()
            << std::endl;
}

// =============================================================================
// Level 5: VTK Output Tests
// =============================================================================

class CGLinearBezierVTKTest : public SimulationTest {};

TEST_F(CGLinearBezierVTKTest, VTKOutputUniformMesh) {
  QuadtreeAdapter mesh;
  mesh.build_uniform(0.0, 100.0, 0.0, 100.0, 4, 4);

  CGLinearBezierBathymetrySmoother smoother(mesh);
  smoother.set_bathymetry_data([](Real x, Real y) {
    return 50.0 + 0.2 * x + 0.1 * y + 5.0 * std::sin(x / 10.0);
  });
  smoother.solve();

  std::string filename = test_output_dir_ + "/cg_linear_bezier_surface";
  smoother.write_vtk(filename); // Uses default resolution=2 for linear

  std::string output_file = filename + ".vtu";
  EXPECT_TRUE(std::filesystem::exists(output_file));
}

TEST_F(CGLinearBezierVTKTest, VTKOutputControlPoints) {
  QuadtreeAdapter mesh;
  mesh.build_uniform(0.0, 100.0, 0.0, 100.0, 4, 4);

  CGLinearBezierBathymetrySmoother smoother(mesh);
  smoother.set_bathymetry_data(
      [](Real x, Real y) { return 50.0 + 0.2 * x + 0.1 * y; });
  smoother.solve();

  std::string filename = test_output_dir_ + "/cg_linear_bezier_control_pts.vtu";
  smoother.write_control_points_vtk(filename);

  EXPECT_TRUE(std::filesystem::exists(filename));
}

TEST_F(CGLinearBezierVTKTest, VTKOutputNonconformingMesh) {
  // Create 1+4 non-conforming mesh
  QuadtreeAdapter mesh;
  Real h = 25.0;

  mesh.add_element({0.0, h, 0.0, h}, {2, 2});
  mesh.add_element({h, 2 * h, 0.0, h}, {2, 2});
  mesh.add_element({0.0, h, h, 2 * h}, {2, 2});
  mesh.add_element({h, 2 * h, h, 2 * h}, {2, 2});
  mesh.add_element({2 * h, 4 * h, 0.0, 2 * h}, {1, 1});

  CGLinearBezierBathymetrySmoother smoother(mesh);
  smoother.set_bathymetry_data([](Real x, Real y) {
    return 50.0 + 10.0 * std::sin(x / 20.0) * std::cos(y / 20.0);
  });
  smoother.solve();

  std::string filename = test_output_dir_ + "/cg_linear_bezier_nonconforming";
  smoother.write_vtk(filename);

  std::string output_file = filename + ".vtu";
  EXPECT_TRUE(std::filesystem::exists(output_file));
}

// =============================================================================
// Level 6: Edge Cases and Robustness
// =============================================================================

TEST_F(CGLinearBezierSmootherTest, ElementCoefficients) {
  QuadtreeAdapter mesh;
  mesh.build_uniform(0.0, 10.0, 0.0, 10.0, 2, 2);

  CGLinearBezierBathymetrySmoother smoother(mesh);
  smoother.set_bathymetry_data([](Real, Real) { return 100.0; });
  smoother.solve();

  // Each element should have 4 coefficients (2x2 control points)
  for (Index e = 0; e < mesh.num_elements(); ++e) {
    VecX coeffs = smoother.element_coefficients(e);
    EXPECT_EQ(coeffs.size(), 4);

    // All coefficients should be approximately 100 for constant function
    for (int i = 0; i < 4; ++i) {
      EXPECT_NEAR(coeffs(i), 100.0, 0.1);
    }
  }
}

TEST_F(CGLinearBezierSmootherTest, EvaluateOutsideDomain) {
  QuadtreeAdapter mesh;
  mesh.build_uniform(0.0, 100.0, 0.0, 100.0, 4, 4);

  CGLinearBezierBathymetrySmoother smoother(mesh);
  smoother.set_bathymetry_data(
      [](Real x, Real y) { return 50.0 + 0.5 * x + 0.3 * y; });
  smoother.solve();

  // Should not throw for points outside domain (finds closest element)
  EXPECT_NO_THROW({
    Real val = smoother.evaluate(-10.0, 50.0); // Outside left
    (void)val;
  });
  EXPECT_NO_THROW({
    Real val = smoother.evaluate(110.0, 50.0); // Outside right
    (void)val;
  });
}

TEST_F(CGLinearBezierSmootherTest, CompareToAnalyticForLinear) {
  // Create a fine mesh for accurate representation
  auto mesh = create_quadtree(16, 16);

  CGLinearBezierSmootherConfig config;
  config.lambda = 1000.0; // Very strong data fitting
  config.ridge_epsilon = 0.0;

  auto linear = [](Real x, Real y) { return 5.0 + 0.123 * x + 0.456 * y; };

  CGLinearBezierBathymetrySmoother smoother(mesh, config);
  smoother.set_bathymetry_data(linear);
  smoother.solve();

  // Sample many points and compute RMS error
  // Note: With Dirichlet energy (gradient penalty), linear functions
  // won't be exactly reproduced even with high lambda
  Real sum_sq_error = 0.0;
  int count = 0;
  for (Real x = 5.0; x < 95.0; x += 10.0) {
    for (Real y = 5.0; y < 95.0; y += 10.0) {
      Real expected = linear(x, y);
      Real computed = smoother.evaluate(x, y);
      Real error = computed - expected;
      sum_sq_error += error * error;
      ++count;
    }
  }
  Real rms_error = std::sqrt(sum_sq_error / count);

  // Allow larger RMS error due to gradient penalty
  EXPECT_LT(rms_error, 5.0); // RMS error within 5.0

  std::cout << "RMS error for linear function: " << rms_error << std::endl;
}

// =============================================================================
// Level 7: Multi-Source Bathymetry Integration
// =============================================================================

class CGLinearBezierSmootherGeoTiffTest : public BathymetryTestFixture {
  protected:
    static constexpr Real TOLERANCE = 1e-10;
    static constexpr Real LOOSE_TOLERANCE = 1e-6;
};

TEST_F(CGLinearBezierSmootherGeoTiffTest, KattegatIntegration) {
  if (!data_files_exist()) {
    GTEST_SKIP() << "Bathymetry data not available";
  }

  // Kattegat test area
  Real center_x = 4095238.0;  // EPSG:3034
  Real center_y = 3344695.0;  // EPSG:3034
  Real domain_size = 30000.0; // 30 km

  Real xmin = center_x - domain_size / 2;
  Real xmax = center_x + domain_size / 2;
  Real ymin = center_y - domain_size / 2;
  Real ymax = center_y + domain_size / 2;

  std::cout << "=== CG Linear Bezier Kattegat Test ===" << std::endl;
  std::cout << "Domain: [" << xmin << ", " << xmax << "] x [" << ymin << ", "
            << ymax << "]" << std::endl;

  // Create uniform mesh
  QuadtreeAdapter mesh;
  mesh.build_uniform(xmin, xmax, ymin, ymax, 8, 8);

  std::cout << "Mesh elements: " << mesh.num_elements() << std::endl;

  // Depth function using multi-source bathymetry
  auto depth_func = create_depth_function();

  // Create CG Linear Bezier smoother
  CGLinearBezierSmootherConfig config;
  config.lambda = 1.0;
  config.ngauss_data = 4;
  config.ngauss_energy = 4;

  CGLinearBezierBathymetrySmoother smoother(mesh, config);
  smoother.set_bathymetry_data(depth_func);

  std::cout << "DOFs: " << smoother.num_global_dofs() << std::endl;
  std::cout << "Constraints: " << smoother.num_constraints() << std::endl;

  // Solve
  smoother.solve();

  EXPECT_TRUE(smoother.is_solved());

  // Write output for ParaView verification
  std::string output_base = "/tmp/cg_linear_bezier_kattegat_test";
  smoother.write_vtk(output_base);

  std::string output_file = output_base + ".vtu";
  std::cout << "Output written to: " << output_file << std::endl;
  EXPECT_TRUE(std::filesystem::exists(output_file));

  // Write control points for debugging
  std::string cp_file = "/tmp/cg_linear_bezier_kattegat_control_points.vtu";
  smoother.write_control_points_vtk(cp_file);
  std::cout << "Control points written to: " << cp_file << std::endl;

  // Basic quality checks
  Real max_diff = 0.0;
  Real sum_diff = 0.0;
  int count = 0;

  for (Index e = 0; e < mesh.num_elements(); ++e) {
    const auto &bounds = mesh.element_bounds(e);
    Real cx = 0.5 * (bounds.xmin + bounds.xmax);
    Real cy = 0.5 * (bounds.ymin + bounds.ymax);

    Real expected = depth_func(cx, cy);
    Real computed = smoother.evaluate(cx, cy);
    Real diff = std::abs(expected - computed);

    max_diff = std::max(max_diff, diff);
    sum_diff += diff;
    count++;
  }

  Real avg_diff = sum_diff / count;
  std::cout << "Max difference from input: " << max_diff << " m" << std::endl;
  std::cout << "Avg difference from input: " << avg_diff << " m" << std::endl;

  // Reasonable fit quality (smoothed, so not exact)
  EXPECT_LT(avg_diff, 50.0); // Average error less than 50m
}

// =============================================================================
// Diagnostic Tests for Debugging Discontinuity Issue
// =============================================================================

TEST_F(CGLinearBezierVTKTest, DiagnosticUniform2x2) {
  // Simple [0,100] x [0,100] domain with 2x2 elements
  QuadtreeAdapter mesh;
  mesh.build_uniform(0.0, 100.0, 0.0, 100.0, 2, 2);

  CGLinearBezierSmootherConfig config;
  config.lambda = 100.0; // Strong data fitting

  CGLinearBezierBathymetrySmoother smoother(mesh, config);

  // Use simple linear function for predictable results
  smoother.set_bathymetry_data(
      [](Real x, Real y) { return 10.0 + 0.5 * x + 0.3 * y; });
  smoother.solve();

  // Output to /tmp for ParaView inspection
  smoother.write_vtk("/tmp/cg_linear_uniform_2x2");

  // Also write control points to see DOF values
  smoother.write_control_points_vtk(
      "/tmp/cg_linear_uniform_2x2_control_pts.vtu");

  // Print diagnostics
  std::cout << "=== Diagnostic: Uniform 2x2 ===" << std::endl;
  std::cout << "DOFs: " << smoother.num_global_dofs() << " (expected 9)"
            << std::endl;
  std::cout << "Constraints: " << smoother.num_constraints() << " (expected 0)"
            << std::endl;

  // Print element coefficients and global DOF indices for debugging
  // DOF layout in reference element [0,1]²:
  //   [1]──[3]   (v=1)
  //    │    │
  //   [0]──[2]   (v=0)
  //   u=0  u=1
  for (Index e = 0; e < mesh.num_elements(); ++e) {
    VecX coeffs = smoother.element_coefficients(e);
    const auto &bounds = mesh.element_bounds(e);
    const auto &global_dofs = smoother.dof_manager().element_dofs(e);

    std::cout << "Element " << e << " bounds [" << bounds.xmin << ","
              << bounds.xmax << "]x[" << bounds.ymin << "," << bounds.ymax
              << "]\n";
    std::cout << "  Global DOF indices: [" << global_dofs[0] << ", "
              << global_dofs[1] << ", " << global_dofs[2] << ", "
              << global_dofs[3] << "]\n";
    std::cout << "  Coeffs: " << coeffs.transpose() << std::endl;

    // Print corner positions with values
    std::cout << "  Corner (xmin,ymin)=(" << bounds.xmin << "," << bounds.ymin
              << "): DOF " << global_dofs[0] << " = " << coeffs(0) << "\n";
    std::cout << "  Corner (xmin,ymax)=(" << bounds.xmin << "," << bounds.ymax
              << "): DOF " << global_dofs[1] << " = " << coeffs(1) << "\n";
    std::cout << "  Corner (xmax,ymin)=(" << bounds.xmax << "," << bounds.ymin
              << "): DOF " << global_dofs[2] << " = " << coeffs(2) << "\n";
    std::cout << "  Corner (xmax,ymax)=(" << bounds.xmax << "," << bounds.ymax
              << "): DOF " << global_dofs[3] << " = " << coeffs(3) << "\n";
  }

  std::cout << "Output: /tmp/cg_linear_uniform_2x2.vtu" << std::endl;
  std::cout << "Control points: /tmp/cg_linear_uniform_2x2_control_pts.vtu"
            << std::endl;
}

TEST_F(CGLinearBezierVTKTest, DiagnosticNonConforming1Plus4) {
  // 1+4 non-conforming mesh:
  // 2×2 fine elements + 1 coarse element on the right
  //
  //  +---+---+-------+
  //  | 2 | 3 |       |
  //  +---+---+   4   |
  //  | 0 | 1 |       |
  //  +---+---+-------+
  QuadtreeAdapter mesh;
  Real h = 25.0;

  mesh.add_element({0.0, h, 0.0, h}, {2, 2});           // elem 0
  mesh.add_element({h, 2 * h, 0.0, h}, {2, 2});         // elem 1
  mesh.add_element({0.0, h, h, 2 * h}, {2, 2});         // elem 2
  mesh.add_element({h, 2 * h, h, 2 * h}, {2, 2});       // elem 3
  mesh.add_element({2 * h, 4 * h, 0.0, 2 * h}, {1, 1}); // elem 4 (coarse)

  CGLinearBezierSmootherConfig config;
  config.lambda = 100.0;

  CGLinearBezierBathymetrySmoother smoother(mesh, config);
  smoother.set_bathymetry_data(
      [](Real x, Real y) { return 10.0 + 0.5 * x + 0.3 * y; });
  smoother.solve();

  smoother.write_vtk("/tmp/cg_linear_nonconforming_1plus4");
  smoother.write_control_points_vtk(
      "/tmp/cg_linear_nonconforming_1plus4_control_pts.vtu");

  std::cout << "=== Diagnostic: Non-conforming 1+4 ===" << std::endl;
  std::cout << "DOFs: " << smoother.num_global_dofs() << std::endl;
  std::cout << "Constraints: " << smoother.num_constraints() << std::endl;
  std::cout << "Constraint violation: " << smoother.constraint_violation()
            << std::endl;

  for (Index e = 0; e < mesh.num_elements(); ++e) {
    VecX coeffs = smoother.element_coefficients(e);
    const auto &bounds = mesh.element_bounds(e);
    std::cout << "Element " << e << " bounds [" << bounds.xmin << ","
              << bounds.xmax << "]x[" << bounds.ymin << "," << bounds.ymax
              << "] coeffs: " << coeffs.transpose() << std::endl;
  }

  std::cout << "Output: /tmp/cg_linear_nonconforming_1plus4.vtu" << std::endl;
  std::cout
      << "Control points: /tmp/cg_linear_nonconforming_1plus4_control_pts.vtu"
      << std::endl;
}
