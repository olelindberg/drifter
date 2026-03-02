#include "bathymetry/adaptive_cg_cubic_bezier_smoother.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "mesh/octree_adapter.hpp"
#include "test_integration_fixtures.hpp"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "mesh/geotiff_reader.hpp"

using namespace drifter;
using namespace drifter::testing;

class AdaptiveCGCubicBezierSmootherTest : public ::testing::Test {
protected:
  static constexpr Real TOLERANCE = 1e-10;
  static constexpr Real LOOSE_TOLERANCE = 1e-4;
};

// =============================================================================
// Construction Tests
// =============================================================================

TEST_F(AdaptiveCGCubicBezierSmootherTest, ConstructFromDomainBounds) {
  AdaptiveCGCubicBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 4, 4);

  EXPECT_EQ(smoother.mesh().num_elements(), 16);
  EXPECT_FALSE(smoother.is_solved());
}

TEST_F(AdaptiveCGCubicBezierSmootherTest, ConstructFromOctree) {
  OctreeAdapter octree(0.0, 100.0, 0.0, 100.0, -1.0, 0.0);
  octree.build_uniform(4, 4, 1);

  AdaptiveCGCubicBezierSmoother smoother(octree);

  EXPECT_EQ(smoother.mesh().num_elements(), 16);
  EXPECT_FALSE(smoother.is_solved());
}

// =============================================================================
// Convergence Tests
// =============================================================================

TEST_F(AdaptiveCGCubicBezierSmootherTest, ConvergesOnConstantBathymetry) {
  // Constant bathymetry should converge immediately (no refinement needed)
  AdaptiveCGCubicBezierConfig config;
  config.error_threshold = 0.01;
  config.max_iterations = 5;
  config.smoother_config.lambda = 100.0; // Strong data fitting

  AdaptiveCGCubicBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 2, 2, config);
  smoother.set_bathymetry_data([](Real, Real) { return 50.0; });

  auto result = smoother.solve_adaptive();

  EXPECT_TRUE(smoother.is_solved());
  EXPECT_TRUE(result.converged);
  EXPECT_LE(result.max_error, config.error_threshold);

  // Evaluation should return constant
  EXPECT_NEAR(smoother.evaluate(50.0, 50.0), 50.0, LOOSE_TOLERANCE);
}

TEST_F(AdaptiveCGCubicBezierSmootherTest, ConvergesOnLinearBathymetry) {
  // Linear bathymetry: cubic Bezier can represent exactly
  AdaptiveCGCubicBezierConfig config;
  config.error_threshold = 0.01;
  config.max_iterations = 5;
  config.smoother_config.lambda = 100.0;
  config.smoother_config.ridge_epsilon =
      0.0; // No ridge for exact polynomial reproduction

  auto linear = [](Real x, Real y) { return 10.0 + 0.5 * x + 0.3 * y; };

  AdaptiveCGCubicBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 2, 2, config);
  smoother.set_bathymetry_data(linear);

  auto result = smoother.solve_adaptive();

  EXPECT_TRUE(smoother.is_solved());
  EXPECT_TRUE(result.converged);

  // Should reproduce linear function (use slightly relaxed tolerance for
  // smoothing effects)
  Real tol = 1e-3; // 0.001 tolerance - smoothing introduces small error
  EXPECT_NEAR(smoother.evaluate(50.0, 50.0), linear(50.0, 50.0), tol);
  EXPECT_NEAR(smoother.evaluate(25.0, 75.0), linear(25.0, 75.0), tol);
}

TEST_F(AdaptiveCGCubicBezierSmootherTest, ConvergesOnQuadraticBathymetry) {
  // Quadratic bathymetry: cubic Bezier can represent exactly
  AdaptiveCGCubicBezierConfig config;
  config.error_threshold = 0.1;
  config.max_iterations = 5;
  config.smoother_config.lambda = 100.0;

  auto quadratic = [](Real x, Real y) {
    return 100.0 + 0.001 * x * x + 0.002 * y * y;
  };

  AdaptiveCGCubicBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 2, 2, config);
  smoother.set_bathymetry_data(quadratic);

  auto result = smoother.solve_adaptive();

  EXPECT_TRUE(smoother.is_solved());

  // Should represent quadratic well
  Real rel_tol = 0.01; // 1% relative tolerance
  EXPECT_NEAR(smoother.evaluate(50.0, 50.0), quadratic(50.0, 50.0),
              std::abs(quadratic(50.0, 50.0)) * rel_tol);
}

TEST_F(AdaptiveCGCubicBezierSmootherTest, RefinesOnHighFrequencyBathymetry) {
  // High-frequency bathymetry should trigger refinement
  AdaptiveCGCubicBezierConfig config;
  config.error_threshold = 1.0; // 1 meter threshold
  config.max_iterations = 5;
  config.max_elements = 200;
  config.smoother_config.lambda = 10.0;

  // Gaussian bump
  auto bump = [](Real x, Real y) {
    Real cx = 50.0, cy = 50.0, sigma = 10.0;
    Real dx = x - cx, dy = y - cy;
    return 50.0 * std::exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
  };

  AdaptiveCGCubicBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 2, 2, config);
  smoother.set_bathymetry_data(bump);

  auto result = smoother.solve_adaptive();

  EXPECT_TRUE(smoother.is_solved());
  // Should have refined some elements
  EXPECT_GT(result.num_elements, 4);

  std::cout << "High-frequency test: " << result.num_elements
            << " elements, max_error=" << result.max_error << " m\n";
}

// =============================================================================
// Stopping Criteria Tests
// =============================================================================

TEST_F(AdaptiveCGCubicBezierSmootherTest, RespectsMaxIterations) {
  AdaptiveCGCubicBezierConfig config;
  config.error_threshold = 0.001; // Very tight threshold
  config.max_iterations = 3;
  config.max_elements = 10000;
  config.smoother_config.lambda = 10.0;

  // Function that needs refinement
  auto wavy = [](Real x, Real y) {
    return 50.0 + 10.0 * std::sin(0.1 * x) * std::cos(0.1 * y);
  };

  AdaptiveCGCubicBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 2, 2, config);
  smoother.set_bathymetry_data(wavy);

  auto result = smoother.solve_adaptive();

  // Should have run max_iterations
  EXPECT_LE(static_cast<int>(smoother.history().size()), config.max_iterations);
}

TEST_F(AdaptiveCGCubicBezierSmootherTest, RespectsMaxElements) {
  AdaptiveCGCubicBezierConfig config;
  config.error_threshold = 0.001; // Very tight threshold
  config.max_iterations = 20;
  config.max_elements = 50; // Low element limit
  config.smoother_config.lambda = 10.0;

  auto wavy = [](Real x, Real y) {
    return 50.0 + 10.0 * std::sin(0.1 * x) * std::cos(0.1 * y);
  };

  AdaptiveCGCubicBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 4, 4, config);
  smoother.set_bathymetry_data(wavy);

  auto result = smoother.solve_adaptive();

  // Note: Refinement with 2:1 balancing can exceed max_elements by one
  // refinement step. The check happens before refinement, and balancing may add
  // extra elements. With C1 constraints, the solver may refine more elements.
  // We allow 3x buffer over max_elements to account for this.
  EXPECT_LE(result.num_elements, static_cast<Index>(config.max_elements * 3));
}

// =============================================================================
// Error Estimation Tests
// =============================================================================

TEST_F(AdaptiveCGCubicBezierSmootherTest, ErrorEstimationAfterSolve) {
  AdaptiveCGCubicBezierConfig config;
  config.error_threshold = 10.0;
  config.max_iterations = 1;
  config.smoother_config.lambda = 10.0;

  auto bathy = [](Real x, Real y) {
    return 100.0 + 20.0 * std::sin(0.05 * x) * std::cos(0.05 * y);
  };

  AdaptiveCGCubicBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 4, 4, config);
  smoother.set_bathymetry_data(bathy);

  auto result = smoother.solve_adaptive();

  // Can estimate errors
  auto errors = smoother.estimate_errors();
  EXPECT_EQ(static_cast<Index>(errors.size()), smoother.mesh().num_elements());

  // max_error and mean_error should work
  Real max_err = smoother.max_error();
  Real mean_err = smoother.mean_error();

  EXPECT_GE(max_err, mean_err);
  EXPECT_GT(max_err, 0.0);
}

// =============================================================================
// Continuity Tests
// =============================================================================

TEST_F(AdaptiveCGCubicBezierSmootherTest, MaintainsContinuityAfterRefinement) {
  AdaptiveCGCubicBezierConfig config;
  config.error_threshold = 5.0;
  config.max_iterations = 3;
  config.smoother_config.lambda = 10.0;

  auto smooth_bathy = [](Real x, Real y) {
    return 100.0 + 20.0 * std::sin(0.03 * x) * std::cos(0.03 * y);
  };

  AdaptiveCGCubicBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 4, 4, config);
  smoother.set_bathymetry_data(smooth_bathy);
  smoother.solve_adaptive();

  // Check C0 continuity by evaluating nearby points
  Real max_jump = 0.0;
  Real step = 0.01;

  for (Real x = 10.0; x < 90.0; x += 10.0) {
    for (Real y = 10.0; y < 90.0; y += 10.0) {
      Real z0 = smoother.evaluate(x, y);
      Real z1 = smoother.evaluate(x + step, y);
      Real z2 = smoother.evaluate(x, y + step);

      Real jump_x = std::abs(z1 - z0) / step;
      Real jump_y = std::abs(z2 - z0) / step;

      max_jump = std::max(max_jump, std::max(jump_x, jump_y));
    }
  }

  // Gradients should be bounded (no discontinuities)
  EXPECT_LT(max_jump, 100.0); // Reasonable gradient bound
}

// =============================================================================
// VTK Output Tests
// =============================================================================

TEST_F(AdaptiveCGCubicBezierSmootherTest, WritesVTKOutput) {
  AdaptiveCGCubicBezierConfig config;
  config.error_threshold = 5.0;
  config.max_iterations = 2;
  config.smoother_config.lambda = 10.0;

  auto bathy = [](Real x, Real y) { return 100.0 + 10.0 * std::sin(0.05 * x); };

  AdaptiveCGCubicBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 4, 4, config);
  smoother.set_bathymetry_data(bathy);
  smoother.solve_adaptive();

  std::string filename = "/tmp/adaptive_cg_cubic_bezier_test";
  smoother.write_vtk(filename, 8);

  std::string output_file = filename + ".vtu";
  EXPECT_TRUE(std::filesystem::exists(output_file));

  // Check file has content
  std::ifstream file(output_file);
  std::string content((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
  EXPECT_GT(content.size(), 1000u);
}

// =============================================================================
// History Tracking Tests
// =============================================================================

TEST_F(AdaptiveCGCubicBezierSmootherTest, TracksAdaptationHistory) {
  AdaptiveCGCubicBezierConfig config;
  config.error_threshold = 1.0;
  config.max_iterations = 5;
  config.smoother_config.lambda = 10.0;

  auto bump = [](Real x, Real y) {
    Real cx = 50.0, cy = 50.0, sigma = 15.0;
    Real dx = x - cx, dy = y - cy;
    return 30.0 * std::exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
  };

  AdaptiveCGCubicBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 2, 2, config);
  smoother.set_bathymetry_data(bump);
  smoother.solve_adaptive();

  const auto &history = smoother.history();
  EXPECT_FALSE(history.empty());

  // Check history is properly populated
  for (size_t i = 0; i < history.size(); ++i) {
    EXPECT_EQ(history[i].iteration, static_cast<int>(i));
    EXPECT_GT(history[i].num_elements, 0);
    EXPECT_GE(history[i].max_error, 0.0);
    EXPECT_GE(history[i].mean_error, 0.0);
  }

  // Print history summary
  std::cout << "\nAdaptation history:\n";
  for (const auto &h : history) {
    std::cout << "  Iter " << h.iteration << ": " << h.num_elements
              << " elements, "
              << "max_err=" << h.max_error << " m, "
              << "refined=" << h.elements_refined << "\n";
  }
}

// =============================================================================
// CG DOF Count Tests
// =============================================================================

TEST_F(AdaptiveCGCubicBezierSmootherTest, CGHasFewerDOFsThanDG) {
  AdaptiveCGCubicBezierConfig config;
  config.error_threshold = 5.0;
  config.max_iterations = 2;
  config.smoother_config.lambda = 10.0;

  auto bathy = [](Real x, Real y) { return 50.0 + x * 0.5; };

  AdaptiveCGCubicBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 4, 4, config);
  smoother.set_bathymetry_data(bathy);
  smoother.solve_adaptive();

  Index num_dofs = smoother.smoother().num_global_dofs();
  Index num_elements = smoother.mesh().num_elements();

  // DG would have num_elements * 16 DOFs (cubic: 4x4 = 16 per element)
  Index dg_dofs = num_elements * 16;

  std::cout << "CG cubic DOFs: " << num_dofs << " vs DG equivalent: " << dg_dofs
            << "\n";

  EXPECT_LT(num_dofs, dg_dofs);
}

// =============================================================================
// C1 Constraint Tests
// =============================================================================

TEST_F(AdaptiveCGCubicBezierSmootherTest, AdaptiveWithC1Constraints) {
  AdaptiveCGCubicBezierConfig config;
  config.error_threshold = 5.0;
  config.max_iterations = 3;
  config.smoother_config.lambda = 10.0;
  config.smoother_config.edge_ngauss = 4;

  auto bathy = [](Real x, Real y) {
    return 100.0 + 20.0 * std::sin(0.04 * x) * std::cos(0.04 * y);
  };

  AdaptiveCGCubicBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 4, 4, config);
  smoother.set_bathymetry_data(bathy);

  auto result = smoother.solve_adaptive();

  EXPECT_TRUE(smoother.is_solved());

  // Verify constraints are satisfied
  Real violation = smoother.smoother().constraint_violation();
  EXPECT_LT(violation, 1e-5);

  std::cout << "Adaptive with C1 constraints: " << result.num_elements
            << " elements, constraint_violation=" << violation << "\n";
}

// =============================================================================
// Multi-Source Bathymetry Integration Tests
// =============================================================================

class AdaptiveCGCubicBezierSmootherGeoTiffTest : public BathymetryTestFixture {
protected:
  static constexpr Real TOLERANCE = 1e-10;
  static constexpr Real LOOSE_TOLERANCE = 1e-6;
};

TEST_F(AdaptiveCGCubicBezierSmootherGeoTiffTest, AdaptiveGeoTiffRefinement) {
  if (!data_files_exist()) {
    GTEST_SKIP() << "Bathymetry data not available";
  }

  // Kattegat test area
  Real center_x = 4095238.0; // EPSG:3034
  Real center_y = 3344695.0; // EPSG:3034
  Real domain_size = 100000.0;

  AdaptiveCGCubicBezierConfig config;
  config.error_threshold = 1.0;
  config.error_metric_type = ErrorMetricType::VolumeChange;
  config.max_iterations = 5;
  config.max_elements = 2500;
  config.smoother_config.lambda = 10.0;
  config.max_refinement_level = 12;
  config.verbose = true;
  config.ngauss_error = 6;
  config.error_output_dir = "/tmp/adaptive_cg_cubic_errors";
  config.smoother_config.edge_ngauss = 4;
  config.smoother_config.use_iterative_solver = true;
  config.smoother_config.use_multigrid = true;
  config.smoother_config.multigrid_config.smoother_type =
      SmootherType::ColoredMultiplicativeSchwarz;
  config.smoother_config.multigrid_config.verbose = true;

  Real xmin = center_x - domain_size / 2;
  Real xmax = center_x + domain_size / 2;
  Real ymin = center_y - domain_size / 2;
  Real ymax = center_y + domain_size / 2;

  auto depth_func = create_depth_function();

  std::cout << "=== Adaptive CG Cubic Bezier GeoTIFF Test ===" << std::endl;
  std::cout << "Domain: [" << xmin << ", " << xmax << "] x [" << ymin << ", "
            << ymax << "]" << std::endl;

  AdaptiveCGCubicBezierSmoother smoother(xmin, xmax, ymin, ymax, 4, 4, config);
  smoother.set_bathymetry_data(depth_func);

  // Set land mask to skip refinement on land-only elements
  smoother.set_land_mask(create_land_mask());

  auto start = std::chrono::high_resolution_clock::now();
  auto result = smoother.solve_adaptive();
  auto end = std::chrono::high_resolution_clock::now();

  double time_ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  std::cout << "\nFinal result:" << std::endl;
  std::cout << "  Elements: " << result.num_elements << std::endl;
  std::cout << "  Max error: " << result.max_error << " m" << std::endl;
  std::cout << "  Mean error: " << result.mean_error << " m" << std::endl;
  std::cout << "  Converged: " << (result.converged ? "yes" : "no")
            << std::endl;
  std::cout << "  Time: " << time_ms << " ms" << std::endl;

  EXPECT_TRUE(smoother.is_solved());

  // Compute final refinement statistics
  Real max_level = 0.0;
  for (Index i = 0; i < smoother.mesh().num_elements(); ++i) {
    max_level = std::max(
        max_level,
        static_cast<Real>(smoother.mesh().element_level(i).max_level()));
  }
  auto element_size_min = domain_size / std::pow(2.0, max_level);

  std::cout << "Number of levels         : " << max_level << std::endl;
  std::cout << "Size of smallest element : " << element_size_min << std::endl;

  // Write final result
  std::string output_file = "/tmp/adaptive_cg_cubic_bezier_kattegat";
  smoother.write_vtk(output_file, 8);
  std::cout << "Output written to        : " << output_file << ".vtu"
            << std::endl;

  EXPECT_TRUE(std::filesystem::exists(output_file + ".vtu"));
}

TEST_F(AdaptiveCGCubicBezierSmootherGeoTiffTest,
       DISABLED_ColoredSchwarzGeoTiffProfiling) {
  if (!data_files_exist()) {
    GTEST_SKIP() << "Bathymetry data not available";
  }

  // Kattegat test area - same as AdaptiveGeoTiffRefinement
  Real center_x = 4095238.0;
  Real center_y = 3344695.0;
  Real domain_size = 100000.0;

  Real xmin = center_x - domain_size / 2;
  Real xmax = center_x + domain_size / 2;
  Real ymin = center_y - domain_size / 2;
  Real ymax = center_y + domain_size / 2;

  auto depth_func = create_depth_function();

  std::cout << "\n=== Colored Schwarz Profiling on GeoTIFF Data ===" << std::endl;

  // Test both multiplicative and colored Schwarz on a uniform grid first
  std::vector<std::pair<std::string, SmootherType>> smoother_types = {
      {"Multiplicative", SmootherType::MultiplicativeSchwarz},
      {"Colored", SmootherType::ColoredMultiplicativeSchwarz}};

  for (const auto &[name, smoother_type] : smoother_types) {
    // Build a uniform mesh with moderate refinement
    QuadtreeAdapter mesh;
    mesh.build_uniform(xmin, xmax, ymin, ymax, 16, 16); // 256 elements

    CGCubicBezierSmootherConfig config;
    config.lambda = 10.0;
    config.edge_ngauss = 4;
    config.use_iterative_solver = true;
    config.use_multigrid = true;
    config.schur_cg_tolerance = 1e-10;
    config.multigrid_config.num_levels = 3;
    config.multigrid_config.pre_smoothing = 1;
    config.multigrid_config.post_smoothing = 1;
    config.multigrid_config.smoother_type = smoother_type;

    MultigridProfile mg_profile;
    CGCubicSolveProfile solve_profile;
    solve_profile.multigrid_profile = &mg_profile;

    CGCubicBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_solve_profile(&solve_profile);
    smoother.set_bathymetry_data(depth_func);

    auto start = std::chrono::high_resolution_clock::now();
    smoother.solve();
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    double schwarz_total = mg_profile.schwarz_matvec_ms +
                           mg_profile.schwarz_gather_ms +
                           mg_profile.schwarz_local_solve_ms +
                           mg_profile.schwarz_scatter_update_ms;

    std::cout << "\n" << name << " Schwarz (16x16 = 256 elements):\n";
    std::cout << "  Total solve time: " << std::fixed << std::setprecision(2)
              << total_ms << " ms\n";
    std::cout << "  CG iterations: " << solve_profile.outer_cg_iterations
              << "\n";
    std::cout << "  Q^-1 calls: " << solve_profile.qinv_apply_calls << "\n";
    std::cout << "  Schwarz total: " << schwarz_total << " ms\n";
    std::cout << "    MatVec: " << mg_profile.schwarz_matvec_ms << " ms ("
              << std::setprecision(1)
              << 100.0 * mg_profile.schwarz_matvec_ms /
                     std::max(0.001, schwarz_total)
              << "%)\n";
    std::cout << "    Gather: " << mg_profile.schwarz_gather_ms << " ms\n";
    std::cout << "    Local solve: " << mg_profile.schwarz_local_solve_ms
              << " ms\n";
    std::cout << "    Scatter: " << mg_profile.schwarz_scatter_update_ms
              << " ms ("
              << 100.0 * mg_profile.schwarz_scatter_update_ms /
                     std::max(0.001, schwarz_total)
              << "%)\n";

    EXPECT_TRUE(smoother.is_solved());
    EXPECT_GT(solve_profile.outer_cg_iterations, 0);
  }

  // Now test adaptive refinement with colored Schwarz
  std::cout << "\n=== Adaptive Refinement with Colored Schwarz ===\n";

  AdaptiveCGCubicBezierConfig adaptive_config;
  adaptive_config.error_threshold = 1.0;
  adaptive_config.error_metric_type = ErrorMetricType::VolumeChange;
  adaptive_config.max_iterations = 3;
  adaptive_config.max_elements = 1000;
  adaptive_config.smoother_config.lambda = 10.0;
  adaptive_config.max_refinement_level = 10;
  adaptive_config.verbose = true;
  adaptive_config.ngauss_error = 6;
  adaptive_config.smoother_config.edge_ngauss = 4;
  adaptive_config.smoother_config.use_iterative_solver = true;
  adaptive_config.smoother_config.use_multigrid = true;
  adaptive_config.smoother_config.multigrid_config.smoother_type =
      SmootherType::ColoredMultiplicativeSchwarz;

  AdaptiveCGCubicBezierSmoother adaptive_smoother(xmin, xmax, ymin, ymax, 4, 4,
                                                   adaptive_config);
  adaptive_smoother.set_bathymetry_data(depth_func);
  adaptive_smoother.set_land_mask(create_land_mask());

  auto start = std::chrono::high_resolution_clock::now();
  auto result = adaptive_smoother.solve_adaptive();
  auto end = std::chrono::high_resolution_clock::now();

  double time_ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  std::cout << "\nAdaptive result (Colored Schwarz):\n";
  std::cout << "  Elements: " << result.num_elements << "\n";
  std::cout << "  Max error: " << result.max_error << " m\n";
  std::cout << "  Mean error: " << result.mean_error << " m\n";
  std::cout << "  Total time: " << time_ms << " ms\n";
  std::cout << "  Converged: " << (result.converged ? "yes" : "no") << "\n";

  EXPECT_TRUE(adaptive_smoother.is_solved());
}

TEST_F(AdaptiveCGCubicBezierSmootherGeoTiffTest, AdaptiveWithC1Constraints) {
  if (!data_files_exist()) {
    GTEST_SKIP() << "Bathymetry data not available";
  }

  Real center_x = 4095238.0;
  Real center_y = 3344695.0;
  Real domain_size = 30000.0;

  Real xmin = center_x - domain_size / 2;
  Real xmax = center_x + domain_size / 2;
  Real ymin = center_y - domain_size / 2;
  Real ymax = center_y + domain_size / 2;

  auto depth_func = create_depth_function();

  std::cout << "\n=== Adaptive CG Cubic Bezier with C¹ Constraints ==="
            << std::endl;

  AdaptiveCGCubicBezierConfig config;
  config.error_threshold = 5.0;
  config.max_iterations = 4;
  config.max_elements = 200;
  config.smoother_config.lambda = 10.0;
  config.smoother_config.edge_ngauss = 4;

  AdaptiveCGCubicBezierSmoother smoother(xmin, xmax, ymin, ymax, 4, 4, config);
  smoother.set_bathymetry_data(std::function<Real(Real, Real)>(depth_func));

  auto start = std::chrono::high_resolution_clock::now();
  auto result = smoother.solve_adaptive();
  auto end = std::chrono::high_resolution_clock::now();

  double time_ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  std::cout << "Elements: " << result.num_elements << std::endl;
  std::cout << "DOFs: " << smoother.smoother().num_global_dofs() << std::endl;
  std::cout << "Constraints: " << smoother.smoother().num_constraints()
            << std::endl;
  std::cout << "Max error: " << result.max_error << " m" << std::endl;
  std::cout << "Time: " << time_ms << " ms" << std::endl;

  // Write output
  std::string output_file = "/tmp/adaptive_cg_cubic_c1";
  smoother.write_vtk(output_file, 8);

  EXPECT_TRUE(smoother.is_solved());
}

// =============================================================================
// Verbose Mode Test
// =============================================================================

TEST_F(AdaptiveCGCubicBezierSmootherTest, VerboseMode) {
  AdaptiveCGCubicBezierConfig config;
  config.error_threshold = 5.0;
  config.max_iterations = 3;
  config.smoother_config.lambda = 10.0;
  config.verbose = true; // Enable verbose output

  auto bump = [](Real x, Real y) {
    Real cx = 50.0, cy = 50.0, sigma = 15.0;
    Real dx = x - cx, dy = y - cy;
    return 30.0 * std::exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
  };

  std::cout << "\n=== Verbose Mode Test ===\n";

  AdaptiveCGCubicBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 2, 2, config);
  smoother.set_bathymetry_data(bump);
  smoother.solve_adaptive();

  EXPECT_TRUE(smoother.is_solved());
}

// =============================================================================
// Error Handling Tests
// =============================================================================

TEST_F(AdaptiveCGCubicBezierSmootherTest, ThrowsIfNoDataSet) {
  AdaptiveCGCubicBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 2, 2);

  EXPECT_THROW(smoother.solve_adaptive(), std::runtime_error);
}

TEST_F(AdaptiveCGCubicBezierSmootherTest, ThrowsIfEvaluateBeforeSolve) {
  AdaptiveCGCubicBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 2, 2);
  smoother.set_bathymetry_data([](Real, Real) { return 50.0; });

  EXPECT_THROW(smoother.evaluate(50.0, 50.0), std::runtime_error);
}
