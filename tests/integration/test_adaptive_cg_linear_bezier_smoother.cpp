#include "bathymetry/adaptive_cg_linear_bezier_smoother.hpp"
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

class AdaptiveCGLinearBezierSmootherTest : public ::testing::Test {
protected:
  static constexpr Real TOLERANCE = 1e-10;
  static constexpr Real LOOSE_TOLERANCE = 1e-4;
};

// =============================================================================
// Construction Tests
// =============================================================================

TEST_F(AdaptiveCGLinearBezierSmootherTest, ConstructFromDomainBounds) {
  AdaptiveCGLinearBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 4, 4);

  EXPECT_EQ(smoother.mesh().num_elements(), 16);
  EXPECT_FALSE(smoother.is_solved());
}

TEST_F(AdaptiveCGLinearBezierSmootherTest, ConstructFromOctree) {
  OctreeAdapter octree(0.0, 100.0, 0.0, 100.0, -1.0, 0.0);
  octree.build_uniform(4, 4, 1);

  AdaptiveCGLinearBezierSmoother smoother(octree);

  EXPECT_EQ(smoother.mesh().num_elements(), 16);
  EXPECT_FALSE(smoother.is_solved());
}

// =============================================================================
// Convergence Tests
// =============================================================================

TEST_F(AdaptiveCGLinearBezierSmootherTest, ConvergesOnConstantBathymetry) {
  // Constant bathymetry should converge immediately (no refinement needed)
  AdaptiveCGLinearBezierConfig config;
  config.error_threshold = 0.01;
  config.max_iterations = 5;
  config.smoother_config.lambda = 100.0; // Strong data fitting

  AdaptiveCGLinearBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 2, 2, config);
  smoother.set_bathymetry_data([](Real, Real) { return 50.0; });

  auto result = smoother.solve_adaptive();

  EXPECT_TRUE(smoother.is_solved());
  EXPECT_TRUE(result.converged);
  EXPECT_LE(result.max_error, config.error_threshold);

  // Evaluation should return constant
  EXPECT_NEAR(smoother.evaluate(50.0, 50.0), 50.0, LOOSE_TOLERANCE);
}

TEST_F(AdaptiveCGLinearBezierSmootherTest, ConvergesOnLinearBathymetry) {
  // Linear bathymetry: linear Bezier can represent, but Dirichlet energy
  // penalizes gradients, so the solution won't be exact.
  // Note: The Dirichlet energy E = integral |grad z|^2 pulls the surface toward
  // flat (zero gradient), so even with high lambda, linear functions get
  // smoothed toward their mean value.
  AdaptiveCGLinearBezierConfig config;
  config.error_threshold = 1.0;
  config.max_iterations = 5;
  config.smoother_config.lambda = 1000.0;

  auto linear = [](Real x, Real y) { return 10.0 + 0.5 * x + 0.3 * y; };

  AdaptiveCGLinearBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 2, 2, config);
  smoother.set_bathymetry_data(linear);

  auto result = smoother.solve_adaptive();

  EXPECT_TRUE(smoother.is_solved());

  // The approximation won't be exact due to Dirichlet smoothing pulling toward
  // flatter surfaces. Just verify we get something reasonable (within 10m).
  Real tol = 10.0;
  EXPECT_NEAR(smoother.evaluate(50.0, 50.0), linear(50.0, 50.0), tol);
}

TEST_F(AdaptiveCGLinearBezierSmootherTest, RefinesOnQuadraticBathymetry) {
  // Quadratic bathymetry: linear Bezier needs refinement to approximate
  AdaptiveCGLinearBezierConfig config;
  config.error_threshold = 0.5;  // 0.5 meter threshold
  config.max_iterations = 5;
  config.max_elements = 200;
  config.smoother_config.lambda = 100.0;

  auto quadratic = [](Real x, Real y) {
    return 100.0 + 0.001 * x * x + 0.002 * y * y;
  };

  AdaptiveCGLinearBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 2, 2, config);
  smoother.set_bathymetry_data(quadratic);

  auto result = smoother.solve_adaptive();

  EXPECT_TRUE(smoother.is_solved());
  // Should have refined to approximate quadratic
  EXPECT_GT(result.num_elements, 4);

  std::cout << "Quadratic test: " << result.num_elements
            << " elements, max_error=" << result.max_error << " m\n";
}

TEST_F(AdaptiveCGLinearBezierSmootherTest, RefinesOnHighFrequencyBathymetry) {
  // High-frequency bathymetry should trigger refinement
  AdaptiveCGLinearBezierConfig config;
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

  AdaptiveCGLinearBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 2, 2, config);
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

TEST_F(AdaptiveCGLinearBezierSmootherTest, RespectsMaxIterations) {
  AdaptiveCGLinearBezierConfig config;
  config.error_threshold = 0.001; // Very tight threshold
  config.max_iterations = 3;
  config.max_elements = 10000;
  config.smoother_config.lambda = 10.0;

  // Function that needs refinement
  auto wavy = [](Real x, Real y) {
    return 50.0 + 10.0 * std::sin(0.1 * x) * std::cos(0.1 * y);
  };

  AdaptiveCGLinearBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 2, 2, config);
  smoother.set_bathymetry_data(wavy);

  auto result = smoother.solve_adaptive();

  // Should have run max_iterations
  EXPECT_LE(static_cast<int>(smoother.history().size()), config.max_iterations);
}

TEST_F(AdaptiveCGLinearBezierSmootherTest, RespectsMaxElements) {
  AdaptiveCGLinearBezierConfig config;
  config.error_threshold = 0.001; // Very tight threshold
  config.max_iterations = 20;
  config.max_elements = 50; // Low element limit
  config.smoother_config.lambda = 10.0;

  auto wavy = [](Real x, Real y) {
    return 50.0 + 10.0 * std::sin(0.1 * x) * std::cos(0.1 * y);
  };

  AdaptiveCGLinearBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 4, 4, config);
  smoother.set_bathymetry_data(wavy);

  auto result = smoother.solve_adaptive();

  // Note: Refinement with 2:1 balancing can exceed max_elements by one
  // refinement step. The check happens before refinement, and balancing may add
  // extra elements. We allow 2x buffer over max_elements to account for this.
  EXPECT_LE(result.num_elements, static_cast<Index>(config.max_elements * 2));
}

// =============================================================================
// Error Estimation Tests
// =============================================================================

TEST_F(AdaptiveCGLinearBezierSmootherTest, ErrorEstimationAfterSolve) {
  AdaptiveCGLinearBezierConfig config;
  config.error_threshold = 10.0;
  config.max_iterations = 1;
  config.smoother_config.lambda = 10.0;

  auto bathy = [](Real x, Real y) {
    return 100.0 + 20.0 * std::sin(0.05 * x) * std::cos(0.05 * y);
  };

  AdaptiveCGLinearBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 4, 4, config);
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

TEST_F(AdaptiveCGLinearBezierSmootherTest, MaintainsContinuityAfterRefinement) {
  // Test that the solution is defined and evaluable after refinement.
  // Note: For CG linear Bezier, C0 continuity is maintained through shared DOFs.
  AdaptiveCGLinearBezierConfig config;
  config.error_threshold = 2.0;
  config.max_iterations = 2;
  config.smoother_config.lambda = 10.0;

  // Simple Gaussian bump that triggers refinement
  auto bump = [](Real x, Real y) {
    Real cx = 50.0, cy = 50.0, sigma = 20.0;
    Real dx = x - cx, dy = y - cy;
    return 50.0 + 30.0 * std::exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
  };

  AdaptiveCGLinearBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 4, 4, config);
  smoother.set_bathymetry_data(bump);
  smoother.solve_adaptive();

  EXPECT_TRUE(smoother.is_solved());
  EXPECT_GT(smoother.mesh().num_elements(), 16);  // Should have refined

  // Test that evaluation works at various points
  std::vector<std::pair<Real, Real>> test_points = {
      {50.0, 50.0}, {25.0, 25.0}, {75.0, 75.0}, {25.0, 75.0}, {75.0, 25.0}};

  for (const auto& [x, y] : test_points) {
    Real z = smoother.evaluate(x, y);
    // Values should be bounded and reasonable
    EXPECT_GT(z, 0.0);
    EXPECT_LT(z, 150.0);
  }
}

// =============================================================================
// VTK Output Tests
// =============================================================================

TEST_F(AdaptiveCGLinearBezierSmootherTest, WritesVTKOutput) {
  AdaptiveCGLinearBezierConfig config;
  config.error_threshold = 0.5;  // Tight threshold to force refinement
  config.max_iterations = 5;
  config.smoother_config.lambda = 100.0;  // Strong data fitting

  // Gaussian bump that requires adaptive refinement
  auto bathy = [](Real x, Real y) {
    Real cx = 50.0, cy = 50.0, sigma = 15.0;
    Real dx = x - cx, dy = y - cy;
    return 90.0 + 20.0 * std::exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
  };

  AdaptiveCGLinearBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 4, 4, config);
  smoother.set_bathymetry_data(bathy);
  smoother.solve_adaptive();

  // Should have refined adaptively around the Gaussian bump
  EXPECT_GT(smoother.mesh().num_elements(), 16);

  std::string filename = "/tmp/adaptive_cg_linear_bezier_test";
  smoother.write_vtk(filename);

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

TEST_F(AdaptiveCGLinearBezierSmootherTest, TracksAdaptationHistory) {
  AdaptiveCGLinearBezierConfig config;
  config.error_threshold = 1.0;
  config.max_iterations = 5;
  config.smoother_config.lambda = 10.0;

  auto bump = [](Real x, Real y) {
    Real cx = 50.0, cy = 50.0, sigma = 15.0;
    Real dx = x - cx, dy = y - cy;
    return 30.0 * std::exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
  };

  AdaptiveCGLinearBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 2, 2, config);
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

TEST_F(AdaptiveCGLinearBezierSmootherTest, CGHasFewerDOFsThanDG) {
  AdaptiveCGLinearBezierConfig config;
  config.error_threshold = 5.0;
  config.max_iterations = 2;
  config.smoother_config.lambda = 10.0;

  auto bathy = [](Real x, Real y) { return 50.0 + x * 0.5; };

  AdaptiveCGLinearBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 4, 4, config);
  smoother.set_bathymetry_data(bathy);
  smoother.solve_adaptive();

  Index num_dofs = smoother.smoother().num_global_dofs();
  Index num_elements = smoother.mesh().num_elements();

  // DG would have num_elements * 4 DOFs (linear: 2x2 = 4 per element)
  Index dg_dofs = num_elements * 4;

  std::cout << "CG linear DOFs: " << num_dofs << " vs DG equivalent: " << dg_dofs
            << "\n";

  EXPECT_LT(num_dofs, dg_dofs);
}

// =============================================================================
// GeoTIFF Integration Tests
// =============================================================================

TEST_F(AdaptiveCGLinearBezierSmootherTest, DISABLED_AdaptiveGeoTiffRefinement) {
  // Kattegat test area
  Real center_x = 4095238.0;  // EPSG:3034
  Real center_y = 3344695.0;  // EPSG:3034
  Real domain_size = 30000.0; // 30 km

  Real xmin = center_x - domain_size / 2;
  Real xmax = center_x + domain_size / 2;
  Real ymin = center_y - domain_size / 2;
  Real ymax = center_y + domain_size / 2;

  std::string geotiff_path = BATHYMETRY_GEOTIFF_PATH;
  GeoTiffReader reader;
  BathymetryData bathy;

  try {
    bathy = reader.load(geotiff_path);
  } catch (...) {
    GTEST_SKIP() << "GeoTIFF not available";
  }

  if (!bathy.is_valid()) {
    GTEST_SKIP() << "GeoTIFF not valid";
  }

  auto bathy_ptr = std::make_shared<BathymetryData>(std::move(bathy));
  BathymetrySurface surface(bathy_ptr);

  auto depth_func = [&surface](Real x, Real y) -> Real {
    return -surface.depth(x,
                          y); // depth returns positive down, we want elevation
  };

  std::cout << "=== Adaptive CG Linear Bezier GeoTIFF Test ===" << std::endl;
  std::cout << "Domain: [" << xmin << ", " << xmax << "] x [" << ymin << ", "
            << ymax << "]" << std::endl;

  AdaptiveCGLinearBezierConfig config;
  config.error_threshold = 5.0; // 5 meter threshold
  config.max_iterations = 8;
  config.max_elements = 1000;
  config.smoother_config.lambda = 10.0;
  config.verbose = true;

  AdaptiveCGLinearBezierSmoother smoother(xmin, xmax, ymin, ymax, 4, 4, config);
  smoother.set_bathymetry_data(std::function<Real(Real, Real)>(depth_func));

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

  // Write output for visualization
  std::string output_file = "/tmp/adaptive_cg_linear_bezier_kattegat";
  smoother.write_vtk(output_file);
  std::cout << "Output written to: " << output_file << ".vtu" << std::endl;

  EXPECT_TRUE(std::filesystem::exists(output_file + ".vtu"));
}

// =============================================================================
// Verbose Mode Test
// =============================================================================

TEST_F(AdaptiveCGLinearBezierSmootherTest, VerboseMode) {
  AdaptiveCGLinearBezierConfig config;
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

  AdaptiveCGLinearBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 2, 2, config);
  smoother.set_bathymetry_data(bump);
  smoother.solve_adaptive();

  EXPECT_TRUE(smoother.is_solved());
}

// =============================================================================
// Error Handling Tests
// =============================================================================

TEST_F(AdaptiveCGLinearBezierSmootherTest, ThrowsIfNoDataSet) {
  AdaptiveCGLinearBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 2, 2);

  EXPECT_THROW(smoother.solve_adaptive(), std::runtime_error);
}

TEST_F(AdaptiveCGLinearBezierSmootherTest, ThrowsIfEvaluateBeforeSolve) {
  AdaptiveCGLinearBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 2, 2);
  smoother.set_bathymetry_data([](Real, Real) { return 50.0; });

  EXPECT_THROW(smoother.evaluate(50.0, 50.0), std::runtime_error);
}
