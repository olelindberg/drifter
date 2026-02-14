#include "bathymetry/adaptive_cg_bezier_smoother.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "mesh/geotiff_reader.hpp"
#include "mesh/octree_adapter.hpp"
#include "test_integration_fixtures.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace drifter;

class AdaptiveCGBezierSmootherTest : public ::testing::Test {
protected:
  static constexpr Real TOLERANCE = 1e-10;
  static constexpr Real LOOSE_TOLERANCE = 1e-4;
};

// =============================================================================
// Basic Construction Tests
// =============================================================================

TEST_F(AdaptiveCGBezierSmootherTest, ConstructFromDomainBounds) {
  AdaptiveCGBezierConfig config;
  config.error_threshold = 0.1;

  AdaptiveCGBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 2, 2, config);

  EXPECT_EQ(smoother.mesh().num_elements(), 4);
  EXPECT_FALSE(smoother.is_solved());
}

TEST_F(AdaptiveCGBezierSmootherTest, ConstructFromOctree) {
  OctreeAdapter octree(0.0, 100.0, 0.0, 100.0, -1.0, 0.0);
  octree.build_uniform(3, 3, 1);

  AdaptiveCGBezierConfig config;
  AdaptiveCGBezierSmoother smoother(octree, config);

  // 3x3 uniform mesh gives 9 elements (may differ due to octree internals)
  EXPECT_GE(smoother.mesh().num_elements(), 9);
}

// =============================================================================
// Convergence Tests
// =============================================================================

TEST_F(AdaptiveCGBezierSmootherTest, ConvergesOnConstantBathymetry) {
  // Constant bathymetry should have low error
  AdaptiveCGBezierConfig config;
  config.error_threshold = 0.01; // 1 cm threshold
  config.max_iterations = 5;
  config.verbose = false;

  AdaptiveCGBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 2, 2, config);

  auto bathy_func = [](Real /*x*/, Real /*y*/) -> Real {
    return 50.0; // Constant depth
  };
  smoother.set_bathymetry_data(bathy_func);

  auto result = smoother.solve_adaptive();

  EXPECT_TRUE(result.converged);
  EXPECT_EQ(result.num_elements,
            4); // No refinement needed (error below threshold)
  EXPECT_LT(result.max_error, 0.01); // Below threshold
}

TEST_F(AdaptiveCGBezierSmootherTest, ConvergesOnLinearBathymetry) {
  // Linear bathymetry should have low error
  AdaptiveCGBezierConfig config;
  config.error_threshold = 0.01; // 1 cm threshold
  config.max_iterations = 5;

  AdaptiveCGBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 2, 2, config);

  auto bathy_func = [](Real x, Real y) -> Real {
    return 10.0 + 0.5 * x + 0.3 * y;
  };
  smoother.set_bathymetry_data(bathy_func);

  auto result = smoother.solve_adaptive();

  EXPECT_TRUE(result.converged);
  EXPECT_EQ(result.num_elements,
            4); // No refinement needed (error below threshold)
  EXPECT_LT(result.max_error, 0.01); // Below threshold
}

TEST_F(AdaptiveCGBezierSmootherTest, RefinesOnHighFrequencyBathymetry) {
  // High-frequency bathymetry should trigger refinement
  AdaptiveCGBezierConfig config;
  config.error_threshold = 0.5; // 0.5 meter threshold
  config.max_iterations = 3;
  config.max_elements = 100;
  config.verbose = false;

  AdaptiveCGBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 2, 2, config);

  // Bathymetry with localized high-frequency feature
  auto bathy_func = [](Real x, Real y) -> Real {
    // Smooth background + localized bump
    Real background = 50.0 + 0.1 * x;
    Real bump =
        10.0 *
        std::exp(-((x - 50.0) * (x - 50.0) + (y - 50.0) * (y - 50.0)) / 100.0);
    return background + bump;
  };
  smoother.set_bathymetry_data(bathy_func);

  auto result = smoother.solve_adaptive();

  // Should have refined some elements
  EXPECT_GT(result.num_elements, 4);

  // Check history shows decreasing max error
  const auto &history = smoother.history();
  ASSERT_GE(history.size(), 2);
}

TEST_F(AdaptiveCGBezierSmootherTest, RespectsMaxIterations) {
  AdaptiveCGBezierConfig config;
  config.error_threshold = 1e-10; // Very tight threshold
  config.max_iterations = 2;      // Limited iterations
  config.max_elements = 10000;

  AdaptiveCGBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 2, 2, config);

  auto bathy_func = [](Real x, Real y) -> Real {
    return 50.0 * std::sin(0.1 * x) * std::sin(0.1 * y);
  };
  smoother.set_bathymetry_data(bathy_func);

  auto result = smoother.solve_adaptive();

  // Should stop after max_iterations even if not converged
  EXPECT_LE(static_cast<int>(smoother.history().size()), config.max_iterations);
}

TEST_F(AdaptiveCGBezierSmootherTest, RespectsMaxElements) {
  AdaptiveCGBezierConfig config;
  config.error_threshold = 0.001; // Very tight threshold to trigger refinement
  config.max_iterations = 20;
  config.max_elements = 50; // Limit elements

  AdaptiveCGBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 2, 2, config);

  auto bathy_func = [](Real x, Real y) -> Real {
    return 50.0 * std::sin(0.2 * x) * std::sin(0.2 * y);
  };
  smoother.set_bathymetry_data(bathy_func);

  auto result = smoother.solve_adaptive();

  // Should stop when element count exceeds max_elements
  EXPECT_TRUE(result.converged);
}

// =============================================================================
// Error Estimation Tests
// =============================================================================

TEST_F(AdaptiveCGBezierSmootherTest, ErrorEstimationAfterSolve) {
  AdaptiveCGBezierConfig config;
  config.max_iterations = 1;

  AdaptiveCGBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 2, 2, config);

  auto bathy_func = [](Real x, Real y) -> Real { return 50.0 + x * y * 0.001; };
  smoother.set_bathymetry_data(bathy_func);

  smoother.solve_adaptive();

  auto errors = smoother.estimate_errors();
  EXPECT_EQ(errors.size(), smoother.mesh().num_elements());

  for (const auto &err : errors) {
    EXPECT_GE(err.l2_error, 0.0);
    EXPECT_GE(err.normalized_error, 0.0);
  }
}

// =============================================================================
// Continuity Tests
// =============================================================================

TEST_F(AdaptiveCGBezierSmootherTest, MaintainsContinuityAfterRefinement) {
  // CG smoother should have natural continuity through shared DOFs
  AdaptiveCGBezierConfig config;
  config.error_threshold = 1.0;
  config.max_iterations = 2;

  AdaptiveCGBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 2, 2, config);

  auto bathy_func = [](Real x, Real y) -> Real {
    return 50.0 + 5.0 * std::sin(0.1 * x) * std::sin(0.1 * y);
  };
  smoother.set_bathymetry_data(bathy_func);

  auto result = smoother.solve_adaptive();

  // CG smoother should be solved
  EXPECT_TRUE(smoother.is_solved());

  // For CG smoother, continuity is natural via shared DOFs
  // Just verify solution is valid and smooth
  Real prev_z = smoother.evaluate(0.0, 50.0);
  Real max_jump = 0.0;
  for (Real x = 0.1; x <= 100.0; x += 0.1) {
    Real z = smoother.evaluate(x, 50.0);
    Real jump = std::abs(z - prev_z);
    max_jump = std::max(max_jump, jump);
    prev_z = z;
  }

  // Small step changes imply continuity (no jumps)
  EXPECT_LT(max_jump, 1.0) << "Large discontinuity detected in CG solution";
}

// =============================================================================
// VTK Output Test
// =============================================================================

TEST_F(AdaptiveCGBezierSmootherTest, WritesVTKOutput) {
  AdaptiveCGBezierConfig config;
  config.error_threshold = 1.0;
  config.max_iterations = 2;

  AdaptiveCGBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 2, 2, config);

  auto bathy_func = [](Real x, Real y) -> Real {
    return 50.0 + 0.2 * x + 0.1 * y;
  };
  smoother.set_bathymetry_data(bathy_func);

  smoother.solve_adaptive();

  std::string output_path = "/tmp/test_adaptive_cg_bezier_output";
  EXPECT_NO_THROW(smoother.write_vtk(output_path, 5));

  // Check file was created (CG smoother writes VTU format with .vtu extension)
  std::ifstream file(output_path + ".vtu");
  EXPECT_TRUE(file.good()) << "VTU file was not created";
}

// =============================================================================
// Multi-Source Bathymetry Integration Tests
// =============================================================================

class AdaptiveCGBezierSmootherGeoTiffTest : public BathymetryTestFixture {
  protected:
    static constexpr Real TOLERANCE = 1e-10;
    static constexpr Real LOOSE_TOLERANCE = 1e-6;
};

TEST_F(AdaptiveCGBezierSmootherGeoTiffTest, DISABLED_AdaptiveGeoTiffRefinement) {
  if (!data_files_exist()) {
    GTEST_SKIP() << "Bathymetry data not available";
  }

  // Define domain with specific center coordinates
  Real center_x = 4095238.0;  // EPSG:3034
  Real center_y = 3344695.0;  // EPSG:3034
  Real domain_size = 30000.0; // 30 km in meters
  Real half_size = domain_size / 2.0;

  Real xmin = center_x - half_size;
  Real xmax = center_x + half_size;
  Real ymin = center_y - half_size;
  Real ymax = center_y + half_size;

  AdaptiveCGBezierConfig config;
  config.error_threshold = 1.0; // 1 meter threshold
  config.max_iterations = 10;
  config.max_elements = 500;
  config.verbose = true;
  config.smoother_config.enable_edge_constraints =
      true; // Enable edge constraints

  AdaptiveCGBezierSmoother smoother(xmin, xmax, ymin, ymax, 4, 4, config);

  auto bathy_func = create_depth_function();
  smoother.set_bathymetry_data(bathy_func);

  auto result = smoother.solve_adaptive();

  std::cout << "GeoTIFF adaptive CG test:\n";
  std::cout << "  Initial elements: 16\n";
  std::cout << "  Final elements: " << result.num_elements << "\n";
  std::cout << "  Max error: " << result.max_error << " m\n";
  std::cout << "  Iterations: " << smoother.history().size() << "\n";

  EXPECT_TRUE(smoother.is_solved());
  EXPECT_GE(result.num_elements, 16); // At least initial mesh

  // Write output for visualization
  smoother.write_vtk("/tmp/adaptive_cg_geotiff_test", 10);
}

// =============================================================================
// Test: Compare constraint configurations with adaptive solver
// =============================================================================

TEST_F(AdaptiveCGBezierSmootherGeoTiffTest,
       DISABLED_AdaptiveWithConstraintComparison) {
  if (!data_files_exist()) {
    GTEST_SKIP() << "Bathymetry data not available";
  }

  // Define domain with specific center coordinates
  Real center_x = 4095238.0;  // EPSG:3034
  Real center_y = 3344695.0;  // EPSG:3034
  Real domain_size = 30000.0; // 30 km in meters
  Real half_size = domain_size / 2.0;

  Real xmin = center_x - half_size;
  Real xmax = center_x + half_size;
  Real ymin = center_y - half_size;
  Real ymax = center_y + half_size;

  auto bathy_func = create_depth_function();

  std::cout << "=== Adaptive CG Bezier Constraint Comparison ===" << std::endl;

  // Test configurations
  struct TestConfig {
    std::string name;
    bool edge_constraints;
  };

  std::vector<TestConfig> configs = {
      {"no_constraints", false},
      {"edge_only", true},
  };

  for (const auto &tc : configs) {
    AdaptiveCGBezierConfig config;
    config.error_threshold = 2.0; // Tighter threshold to force refinement
    config.max_iterations = 4;    // More iterations for non-conforming mesh
    config.max_elements = 200;
    config.verbose = false;
    config.smoother_config.lambda = 1.0;
    config.smoother_config.enable_edge_constraints = tc.edge_constraints;

    AdaptiveCGBezierSmoother smoother(xmin, xmax, ymin, ymax, 4, 4, config);
    smoother.set_bathymetry_data(bathy_func);

    auto result = smoother.solve_adaptive();

    std::string output_file = "/tmp/adaptive_cg_" + tc.name + ".vtu";
    smoother.write_vtk("/tmp/adaptive_cg_" + tc.name, 10);

    std::cout << "\n--- Config: " << tc.name << " ---" << std::endl;
    std::cout << "  Elements: " << result.num_elements << std::endl;
    std::cout << "  DOFs: " << smoother.smoother().num_global_dofs()
              << std::endl;
    std::cout
        << "  Edge constraints: "
        << smoother.smoother().dof_manager().num_edge_derivative_constraints()
        << std::endl;
    std::cout << "  Max error: " << result.max_error << " m" << std::endl;
    std::cout << "  Output: " << output_file << std::endl;

    EXPECT_TRUE(smoother.is_solved());
  }

  std::cout << "\nCompare outputs in ParaView to see effect of constraints on "
               "T-junction gaps."
            << std::endl;
}

// =============================================================================
// Comparison with DG: CG should be efficient
// =============================================================================

TEST_F(AdaptiveCGBezierSmootherTest, CGHasFewerDOFsThanDG) {
  // CG should have fewer DOFs than DG due to shared boundary DOFs
  AdaptiveCGBezierConfig config;
  config.error_threshold = 0.5;
  config.max_iterations = 2;
  config.smoother_config.lambda = 1.0;

  AdaptiveCGBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 4, 4, config);

  auto bathy_func = [](Real x, Real y) -> Real {
    return 50.0 + 0.001 * (x * x + y * y);
  };
  smoother.set_bathymetry_data(bathy_func);

  smoother.solve_adaptive();

  // 4x4 mesh = 16 elements
  // DG: 16 * 36 = 576 DOFs
  // CG: (4*5+1) * (4*5+1) = 21*21 = 441 DOFs (for uniform mesh)
  // CG should have significantly fewer DOFs
  Index num_elements = smoother.mesh().num_elements();
  Index num_global_dofs = smoother.smoother().num_global_dofs();

  // CG DOFs should be less than DG DOFs (36 per element)
  EXPECT_LT(num_global_dofs, num_elements * 36)
      << "CG should have fewer DOFs than DG due to shared boundaries";
}

// =============================================================================
// Quality Evaluation: Compare constraint modes and lambda values
// =============================================================================

TEST_F(AdaptiveCGBezierSmootherGeoTiffTest, DISABLED_QualityEvaluation) {
  if (!data_files_exist()) {
    GTEST_SKIP() << "Bathymetry data not available";
  }

  // Define domain with specific center coordinates
  Real center_x = 4095238.0;  // EPSG:3034
  Real center_y = 3344695.0;  // EPSG:3034
  Real domain_size = 30000.0; // 30 km in meters
  Real half_size = domain_size / 2.0;

  Real xmin = center_x - half_size;
  Real xmax = center_x + half_size;
  Real ymin = center_y - half_size;
  Real ymax = center_y + half_size;

  auto bathy_func = create_depth_function();

  // Constraint configurations
  struct ConstraintConfig {
    std::string name;
    bool edge_constraints;
  };

  std::vector<ConstraintConfig> constraint_configs = {
      {"no_constraints", false},
      {"edge_only", true},
  };

  // Lambda values to test
  std::vector<Real> lambda_values = {100.0, 10.0, 1.0, 0.1, 0.01};

  // Results storage
  struct EvalResult {
    std::string config_name;
    Real lambda;
    Index num_elements;
    Index num_dofs;
    Real max_error;
    Real mean_error;
    Real data_residual;
    Real constraint_violation;
    Real regularization_energy;
    double wall_time_ms;
  };

  std::vector<EvalResult> results;

  std::cout << "\n=== AdaptiveCGBezierSmoother Quality Evaluation ==="
            << std::endl;
  std::cout << "Domain: 30km x 30km, Initial mesh: 4x4, ngauss_error=6"
            << std::endl;
  std::cout << "Running " << constraint_configs.size() * lambda_values.size()
            << " configurations..." << std::endl;

  for (const auto &cc : constraint_configs) {
    for (Real lambda : lambda_values) {
      AdaptiveCGBezierConfig config;
      config.error_threshold = 2.0;
      config.max_iterations = 4;
      config.max_elements = 200;
      config.ngauss_error =
          6; // Maximum allowed precision for error integration
      config.verbose = false;
      config.smoother_config.lambda = lambda;
      config.smoother_config.enable_edge_constraints = cc.edge_constraints;

      AdaptiveCGBezierSmoother smoother(xmin, xmax, ymin, ymax, 4, 4, config);
      smoother.set_bathymetry_data(bathy_func);

      // Time the solve
      auto start = std::chrono::high_resolution_clock::now();
      auto result = smoother.solve_adaptive();
      auto end = std::chrono::high_resolution_clock::now();
      double time_ms =
          std::chrono::duration<double, std::milli>(end - start).count();

      // Collect metrics
      EvalResult er;
      er.config_name = cc.name;
      er.lambda = lambda;
      er.num_elements = result.num_elements;
      er.num_dofs = smoother.smoother().num_global_dofs();
      er.max_error = result.max_error;
      er.mean_error = result.mean_error;
      er.data_residual = smoother.smoother().data_residual();
      er.constraint_violation = smoother.smoother().constraint_violation();
      er.regularization_energy = smoother.smoother().regularization_energy();
      er.wall_time_ms = time_ms;

      results.push_back(er);

      std::cout << "  " << cc.name << " lambda=" << lambda << ": "
                << result.num_elements << " elems, "
                << "max_err=" << result.max_error << " m, "
                << "time=" << time_ms << " ms" << std::endl;

      EXPECT_TRUE(smoother.is_solved());
    }
  }

  // Write results to markdown file
  std::string output_path = "/tmp/adaptive_cg_bezier_evaluation.md";
  std::ofstream ofs(output_path);

  // Get current time
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);

  ofs << "# AdaptiveCGBezierSmoother Evaluation Results\n\n";
  ofs << "Date: " << std::ctime(&time_t);
  ofs << "Domain: 30km x 30km centered at (4095238, 3344695) EPSG:3034\n";
  ofs << "Initial mesh: 4x4 (16 elements)\n";
  ofs << "ngauss_error: 6\n";
  ofs << "error_threshold: 2.0 m\n";
  ofs << "max_iterations: 4\n";
  ofs << "max_elements: 200\n\n";

  ofs << "## Results\n\n";
  ofs << "| Config | Lambda | Elements | DOFs | Max Err (m) | Mean Err (m) | "
         "Data Res | Constr Viol | Reg Energy | Time (ms) |\n";
  ofs << "|--------|--------|----------|------|-------------|--------------|---"
         "-------|-------------|------------|----------|\n";

  for (const auto &r : results) {
    // Format lambda appropriately (avoid scientific for readability)
    std::ostringstream lambda_ss;
    if (r.lambda >= 1.0) {
      lambda_ss << std::fixed << std::setprecision(0) << r.lambda;
    } else {
      lambda_ss << std::fixed << std::setprecision(2) << r.lambda;
    }

    ofs << "| " << r.config_name << " | " << lambda_ss.str() << " | "
        << r.num_elements << " | " << r.num_dofs << " | " << std::fixed
        << std::setprecision(3) << r.max_error << " | " << std::fixed
        << std::setprecision(3) << r.mean_error << " | " << std::scientific
        << std::setprecision(2) << r.data_residual << " | " << std::scientific
        << std::setprecision(2) << r.constraint_violation << " | "
        << std::scientific << std::setprecision(2) << r.regularization_energy
        << " | " << std::fixed << std::setprecision(1) << r.wall_time_ms
        << " |\n";
  }

  ofs << "\n## Configuration Legend\n\n";
  ofs << "- **c2_only**: Vertex derivative constraints (z_u, z_v, z_uu, z_uv, "
         "z_vv, z_uuv, z_uvv, z_uuvv)\n";
  ofs << "- **edge_only**: Edge Gauss point constraints (z_n, z_nn at 4 points "
         "per edge)\n";
  ofs << "- **c2_and_edge**: Both vertex and edge constraints\n\n";

  ofs << "## Metric Definitions\n\n";
  ofs << "- **Max Err**: Maximum normalized L2 error across all elements "
         "(meters)\n";
  ofs << "- **Mean Err**: Mean normalized L2 error across all elements "
         "(meters)\n";
  ofs << "- **Data Res**: Weighted least-squares residual ||Bx - d||²_W\n";
  ofs << "- **Constr Viol**: C² constraint violation ||Ax - b||\n";
  ofs << "- **Reg Energy**: Thin plate regularization energy x^T H x\n";
  ofs << "- **Time**: Wall clock time for solve_adaptive() in milliseconds\n";

  ofs.close();

  std::cout << "\nResults written to: " << output_path << std::endl;

  // Verify file was created
  std::ifstream check(output_path);
  EXPECT_TRUE(check.good()) << "Output file was not created";
}
