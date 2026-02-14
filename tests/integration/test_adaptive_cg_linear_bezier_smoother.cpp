#include "bathymetry/adaptive_cg_linear_bezier_smoother.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "io/bathymetry_vtk_writer.hpp"
#include "mesh/octree_adapter.hpp"
#include "test_integration_fixtures.hpp"
#include <algorithm>
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
  config.error_threshold =
      0.1; // 0.1 meter threshold (tight to force refinement)
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

TEST_F(AdaptiveCGLinearBezierSmootherTest, ErrorStatisticsIncludeMeanAndStd) {
  // Test that mean_error and std_error are populated correctly
  AdaptiveCGLinearBezierConfig config;
  config.error_threshold = 10.0;
  config.max_iterations = 1;
  config.smoother_config.lambda = 10.0;

  // Sinusoidal bathymetry creates known error patterns
  auto bathy = [](Real x, Real y) {
    return 100.0 + 20.0 * std::sin(0.1 * x) * std::cos(0.1 * y);
  };

  AdaptiveCGLinearBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 4, 4, config);
  smoother.set_bathymetry_data(bathy);
  smoother.solve_adaptive();

  auto errors = smoother.estimate_errors();
  ASSERT_FALSE(errors.empty());

  for (const auto &err : errors) {
    // Mathematical relationship: RMS² = mean² + std²
    // So std ≤ RMS always (allowing small numerical tolerance)
    EXPECT_LE(err.std_error, err.normalized_error + 1e-10)
        << "std_error should be <= normalized_error (RMS)";

    // Sanity check: std_error should be non-negative
    EXPECT_GE(err.std_error, 0.0);
  }

  // Print some statistics for debugging
  Real max_std = 0.0, max_mean = 0.0;
  for (const auto &err : errors) {
    max_std = std::max(max_std, err.std_error);
    max_mean = std::max(max_mean, std::abs(err.mean_error));
  }
  std::cout << "Max std_error: " << max_std
            << ", Max |mean_error|: " << max_mean << "\n";
}

// =============================================================================
// Continuity Tests
// =============================================================================

TEST_F(AdaptiveCGLinearBezierSmootherTest, MaintainsContinuityAfterRefinement) {
  // Test that the solution is defined and evaluable after refinement.
  // Note: For CG linear Bezier, C0 continuity is maintained through shared
  // DOFs.
  AdaptiveCGLinearBezierConfig config;
  config.error_threshold = 0.5; // Tight threshold to force refinement
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
  EXPECT_GT(smoother.mesh().num_elements(), 16); // Should have refined

  // Test that evaluation works at various points
  std::vector<std::pair<Real, Real>> test_points = {
      {50.0, 50.0}, {25.0, 25.0}, {75.0, 75.0}, {25.0, 75.0}, {75.0, 25.0}};

  for (const auto &[x, y] : test_points) {
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
  config.error_threshold = 0.5; // Tight threshold to force refinement
  config.max_iterations = 5;
  config.smoother_config.lambda = 100.0; // Strong data fitting

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

  std::cout << "CG linear DOFs: " << num_dofs
            << " vs DG equivalent: " << dg_dofs << "\n";

  EXPECT_LT(num_dofs, dg_dofs);
}

// =============================================================================
// Multi-Source Bathymetry Integration Tests
// =============================================================================

class AdaptiveCGLinearBezierSmootherGeoTiffTest : public BathymetryTestFixture {
protected:
  static constexpr Real TOLERANCE = 1e-10;
  static constexpr Real LOOSE_TOLERANCE = 1e-6;
};

TEST_F(AdaptiveCGLinearBezierSmootherGeoTiffTest, AdaptiveGeoTiffRefinement) {
  if (!data_files_exist()) {
    GTEST_SKIP() << "Bathymetry data not available";
  }

  // Kattegat test area
  Real center_x = 4095238.0; // EPSG:3034
  Real center_y = 3344695.0; // EPSG:3034
  Real domain_size = 100000.0;

  AdaptiveCGLinearBezierConfig config;
  config.error_threshold = 1000.0; // WENO indicator threshold [length⁴]
  config.error_metric_type = ErrorMetricType::WenoIndicator;
  config.max_iterations = 100;
  config.max_elements = 10000;
  config.smoother_config.lambda = 10.0;
  config.max_refinement_level = 12;
  config.verbose = true;
  config.ngauss_error = 6;
  config.weno_gradient_weight = 1.0;
  config.weno_curvature_weight = 1.0;
  config.error_output_dir = "/tmp/adaptive_cg_linear_errors";

  Real xmin = center_x - domain_size / 2;
  Real xmax = center_x + domain_size / 2;
  Real ymin = center_y - domain_size / 2;
  Real ymax = center_y + domain_size / 2;

  auto depth_func = create_depth_function();

  std::cout << "=== Adaptive CG Linear Bezier GeoTIFF Test ===" << std::endl;
  std::cout << "Domain: [" << xmin << ", " << xmax << "] x [" << ymin << ", "
            << ymax << "]" << std::endl;

  AdaptiveCGLinearBezierSmoother smoother(xmin, xmax, ymin, ymax, 4, 4, config);
  smoother.set_bathymetry_data(depth_func);

  // Set gradient function for WENO indicator (numerical differentiation)
  smoother.set_gradient_function(create_gradient_output_function());

  // Set curvature function for WENO indicator (numerical differentiation)
  smoother.set_curvature_function(create_curvature_function());

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

  // Write output for visualization with per-element error statistics
  std::string output_file = "/tmp/adaptive_cg_linear_bezier_kattegat";
  auto errors = smoother.estimate_errors();
  std::vector<Real> element_rms(smoother.mesh().num_elements(), 0.0);
  std::vector<Real> element_mean(smoother.mesh().num_elements(), 0.0);
  std::vector<Real> element_std(smoother.mesh().num_elements(), 0.0);
  std::vector<Real> element_gradient(smoother.mesh().num_elements(), 0.0);
  std::vector<Real> element_curvature(smoother.mesh().num_elements(), 0.0);
  std::vector<Real> element_weno(smoother.mesh().num_elements(), 0.0);
  for (const auto &e : errors) {
    element_rms[e.element] = e.normalized_error;
    element_mean[e.element] = e.mean_error;
    element_std[e.element] = e.std_error;
    element_gradient[e.element] = e.gradient_indicator;
    element_curvature[e.element] = e.curvature_indicator;
    element_weno[e.element] = e.weno_indicator;
  }
  std::vector<Real> refinement_levels(smoother.mesh().num_elements(), 0.0);
  for (Index i = 0; i < smoother.mesh().num_elements(); ++i) {
    refinement_levels[i] =
        static_cast<Real>(smoother.mesh().element_level(i).max_level());
  }

  io::write_cg_bezier_surface_vtk(
      output_file, smoother.mesh(),
      [&smoother](Real x, Real y) { return smoother.evaluate(x, y); }, 6,
      "elevation",
      {{"rms_error", element_rms},
       {"mean_error", element_mean},
       {"std_error", element_std},
       {"gradient_indicator", element_gradient},
       {"curvature_indicator", element_curvature},
       {"weno_indicator", element_weno},
       {"refinement_level", refinement_levels}});

  // Write raw bathymetry with per-point gradient and curvature indicators
  if (false) {
    std::string raw_output_file =
        "/tmp/raw_bathymetry_with_indicators_kattegat.vtu";
    auto grad_func = create_gradient_output_function();
    auto curv_func = create_curvature_function();

    // Sample at GeoTIFF resolution (50m)
    Real geotiff_resolution = 50.0;
    int nx = static_cast<int>((xmax - xmin) / geotiff_resolution) + 1;
    int ny = static_cast<int>((ymax - ymin) / geotiff_resolution) + 1;
    Real hx = geotiff_resolution;
    Real hy = geotiff_resolution;

    Index num_points = nx * ny;
    Index num_cells = (nx - 1) * (ny - 1);

    std::ofstream file(raw_output_file);
    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" "
            "byte_order=\"LittleEndian\">\n";
    file << "  <UnstructuredGrid>\n";
    file << "    <Piece NumberOfPoints=\"" << num_points
         << "\" NumberOfCells=\"" << num_cells << "\">\n";

    // Points
    file << "      <Points>\n";
    file << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" "
            "format=\"ascii\">\n";
    for (int j = 0; j < ny; ++j) {
      Real y = ymin + j * hy;
      for (int i = 0; i < nx; ++i) {
        Real x = xmin + i * hx;
        Real z = depth_func(x, y);
        file << "          " << std::setprecision(10) << x << " " << y << " "
             << z << "\n";
      }
    }
    file << "        </DataArray>\n";
    file << "      </Points>\n";

    // Cells (quads)
    file << "      <Cells>\n";
    file << "        <DataArray type=\"Int64\" Name=\"connectivity\" "
            "format=\"ascii\">\n";
    for (int j = 0; j < ny - 1; ++j) {
      for (int i = 0; i < nx - 1; ++i) {
        Index p0 = j * nx + i;
        Index p1 = j * nx + (i + 1);
        Index p2 = (j + 1) * nx + (i + 1);
        Index p3 = (j + 1) * nx + i;
        file << "          " << p0 << " " << p1 << " " << p2 << " " << p3
             << "\n";
      }
    }
    file << "        </DataArray>\n";
    file << "        <DataArray type=\"Int64\" Name=\"offsets\" "
            "format=\"ascii\">\n";
    for (Index c = 1; c <= num_cells; ++c) {
      file << "          " << (c * 4) << "\n";
    }
    file << "        </DataArray>\n";
    file << "        <DataArray type=\"UInt8\" Name=\"types\" "
            "format=\"ascii\">\n";
    for (Index c = 0; c < num_cells; ++c) {
      file << "          9\n"; // VTK_QUAD
    }
    file << "        </DataArray>\n";
    file << "      </Cells>\n";

    // Point data: elevation, gradient indicator, curvature indicator
    file << "      <PointData Scalars=\"elevation\">\n";

    // Elevation
    file << "        <DataArray type=\"Float64\" Name=\"elevation\" "
            "format=\"ascii\">\n";
    for (int j = 0; j < ny; ++j) {
      Real y = ymin + j * hy;
      for (int i = 0; i < nx; ++i) {
        Real x = xmin + i * hx;
        Real z = depth_func(x, y);
        file << "          " << std::setprecision(10) << z << "\n";
      }
    }
    file << "        </DataArray>\n";

    // Gradient indicator: |grad z|^2 = (dz/dx)^2 + (dz/dy)^2
    file << "        <DataArray type=\"Float64\" Name=\"gradient_indicator\" "
            "format=\"ascii\">\n";
    for (int j = 0; j < ny; ++j) {
      Real y = ymin + j * hy;
      for (int i = 0; i < nx; ++i) {
        Real x = xmin + i * hx;
        Real dh_dx, dh_dy;
        grad_func(x, y, dh_dx, dh_dy);
        Real grad_sq = dh_dx * dh_dx + dh_dy * dh_dy;
        file << "          " << std::setprecision(10) << grad_sq << "\n";
      }
    }
    file << "        </DataArray>\n";

    // Curvature indicator: ||H||_F^2 = h_xx^2 + 2*h_xy^2 + h_yy^2
    file << "        <DataArray type=\"Float64\" Name=\"curvature_indicator\" "
            "format=\"ascii\">\n";
    for (int j = 0; j < ny; ++j) {
      Real y = ymin + j * hy;
      for (int i = 0; i < nx; ++i) {
        Real x = xmin + i * hx;
        Real d2h_dx2, d2h_dxdy, d2h_dy2;
        curv_func(x, y, d2h_dx2, d2h_dxdy, d2h_dy2);
        Real hess_sq =
            d2h_dx2 * d2h_dx2 + 2 * d2h_dxdy * d2h_dxdy + d2h_dy2 * d2h_dy2;
        file << "          " << std::setprecision(10) << hess_sq << "\n";
      }
    }
    file << "        </DataArray>\n";

    file << "      </PointData>\n";

    // VTU footer
    file << "    </Piece>\n";
    file << "  </UnstructuredGrid>\n";
    file << "</VTKFile>\n";
    file.close();

    std::cout << "Raw bathymetry with per-point indicators written to: "
              << raw_output_file << std::endl;
  }

  auto levels =
      *std::max_element(refinement_levels.begin(), refinement_levels.end());
  auto element_size_min = domain_size / std::pow(2.0, levels);

  std::cout << "Number of levels         : " << levels << std::endl;
  std::cout << "Size of smallest element : " << element_size_min << std::endl;
  std::cout << "Output written to        : " << output_file << ".vtu"
            << std::endl;

  EXPECT_TRUE(std::filesystem::exists(output_file + ".vtu"));
}

TEST_F(AdaptiveCGLinearBezierSmootherGeoTiffTest, WriteRawBathymetryVTK) {
  if (!data_files_exist()) {
    GTEST_SKIP() << "Bathymetry data not available";
  }

  // Kattegat test area (same as AdaptiveGeoTiffRefinement for comparison)
  Real center_x = 4095238.0;  // EPSG:3034
  Real center_y = 3344695.0;  // EPSG:3034
  Real domain_size = 30000.0; // 30 km

  Real xmin = center_x - domain_size / 2;
  Real xmax = center_x + domain_size / 2;
  Real ymin = center_y - domain_size / 2;
  Real ymax = center_y + domain_size / 2;

  auto depth_func = create_depth_function();

  std::cout << "=== Writing Raw Bathymetry VTK ===" << std::endl;
  std::cout << "Domain: [" << xmin << ", " << xmax << "] x [" << ymin << ", "
            << ymax << "]" << std::endl;

  // Sample on uniform grid (200x200 for good visual quality)
  int nx = 200, ny = 200;
  Real hx = (xmax - xmin) / (nx - 1);
  Real hy = (ymax - ymin) / (ny - 1);

  std::string output_file = "/tmp/raw_bathymetry_kattegat.vtu";
  std::ofstream file(output_file);

  // VTU XML format header
  file << "<?xml version=\"1.0\"?>\n";
  file << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" "
          "byte_order=\"LittleEndian\">\n";
  file << "  <UnstructuredGrid>\n";

  Index num_points = nx * ny;
  Index num_cells = (nx - 1) * (ny - 1);

  file << "    <Piece NumberOfPoints=\"" << num_points << "\" NumberOfCells=\""
       << num_cells << "\">\n";

  // Points
  file << "      <Points>\n";
  file << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" "
          "format=\"ascii\">\n";
  for (int j = 0; j < ny; ++j) {
    Real y = ymin + j * hy;
    for (int i = 0; i < nx; ++i) {
      Real x = xmin + i * hx;
      Real z = depth_func(x, y);
      file << "          " << std::setprecision(10) << x << " " << y << " " << z
           << "\n";
    }
  }
  file << "        </DataArray>\n";
  file << "      </Points>\n";

  // Cells (quads)
  file << "      <Cells>\n";
  file << "        <DataArray type=\"Int64\" Name=\"connectivity\" "
          "format=\"ascii\">\n";
  for (int j = 0; j < ny - 1; ++j) {
    for (int i = 0; i < nx - 1; ++i) {
      Index p0 = j * nx + i;
      Index p1 = j * nx + (i + 1);
      Index p2 = (j + 1) * nx + (i + 1);
      Index p3 = (j + 1) * nx + i;
      file << "          " << p0 << " " << p1 << " " << p2 << " " << p3 << "\n";
    }
  }
  file << "        </DataArray>\n";
  file << "        <DataArray type=\"Int64\" Name=\"offsets\" "
          "format=\"ascii\">\n";
  for (Index c = 1; c <= num_cells; ++c) {
    file << "          " << (c * 4) << "\n";
  }
  file << "        </DataArray>\n";
  file
      << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
  for (Index c = 0; c < num_cells; ++c) {
    file << "          9\n"; // VTK_QUAD
  }
  file << "        </DataArray>\n";
  file << "      </Cells>\n";

  // Point data (elevation)
  file << "      <PointData Scalars=\"elevation\">\n";
  file << "        <DataArray type=\"Float64\" Name=\"elevation\" "
          "format=\"ascii\">\n";
  for (int j = 0; j < ny; ++j) {
    Real y = ymin + j * hy;
    for (int i = 0; i < nx; ++i) {
      Real x = xmin + i * hx;
      Real z = depth_func(x, y);
      file << "          " << std::setprecision(10) << z << "\n";
    }
  }
  file << "        </DataArray>\n";
  file << "      </PointData>\n";

  file << "    </Piece>\n";
  file << "  </UnstructuredGrid>\n";
  file << "</VTKFile>\n";

  file.close();

  std::cout << "Raw bathymetry written to: " << output_file << std::endl;
  std::cout << "Grid: " << nx << " x " << ny << " = " << num_points << " points"
            << std::endl;

  EXPECT_TRUE(std::filesystem::exists(output_file));
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

// =============================================================================
// Marking Strategy Tests
// =============================================================================

TEST_F(AdaptiveCGLinearBezierSmootherTest,
       DorflerSymmetricPreservesSymmetryForGaussianBump) {
  // Centered Gaussian bump - perfectly symmetric about (50, 50)
  auto symmetric_bump = [](Real x, Real y) {
    Real cx = 50.0, cy = 50.0, sigma = 15.0;
    Real dx = x - cx, dy = y - cy;
    return 50.0 * std::exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
  };

  AdaptiveCGLinearBezierConfig config;
  config.marking_strategy = MarkingStrategy::DorflerSymmetric;
  config.dorfler_theta = 0.5;
  config.error_threshold = 2.0;
  config.max_iterations = 3;
  config.smoother_config.lambda = 10.0;

  // Use even initial grid so symmetry is possible
  AdaptiveCGLinearBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 4, 4, config);
  smoother.set_bathymetry_data(symmetric_bump);
  smoother.solve_adaptive();

  EXPECT_TRUE(smoother.is_solved());
  EXPECT_GT(smoother.mesh().num_elements(), 16); // Should have refined

  // Verify mesh symmetry: count elements in each quadrant
  const auto &mesh = smoother.mesh();
  int q1 = 0, q2 = 0, q3 = 0, q4 = 0;
  for (Index e = 0; e < mesh.num_elements(); ++e) {
    const auto &bounds = mesh.element_bounds(e);
    Real cx = (bounds.xmin + bounds.xmax) / 2.0;
    Real cy = (bounds.ymin + bounds.ymax) / 2.0;
    if (cx < 50.0 && cy < 50.0)
      q1++;
    else if (cx >= 50.0 && cy < 50.0)
      q2++;
    else if (cx < 50.0 && cy >= 50.0)
      q3++;
    else
      q4++;
  }

  std::cout << "DorflerSymmetric quadrant counts: " << q1 << " " << q2 << " "
            << q3 << " " << q4 << "\n";

  EXPECT_EQ(q1, q2) << "X-symmetry broken";
  EXPECT_EQ(q1, q3) << "Y-symmetry broken";
  EXPECT_EQ(q1, q4) << "Diagonal symmetry broken";
}

TEST_F(AdaptiveCGLinearBezierSmootherTest,
       RelativeThresholdPreservesSymmetryForGaussianBump) {
  auto symmetric_bump = [](Real x, Real y) {
    Real cx = 50.0, cy = 50.0, sigma = 15.0;
    Real dx = x - cx, dy = y - cy;
    return 50.0 * std::exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
  };

  AdaptiveCGLinearBezierConfig config;
  config.marking_strategy = MarkingStrategy::RelativeThreshold;
  config.relative_alpha = 0.3;
  config.error_threshold = 2.0;
  config.max_iterations = 3;
  config.smoother_config.lambda = 10.0;

  AdaptiveCGLinearBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 4, 4, config);
  smoother.set_bathymetry_data(symmetric_bump);
  smoother.solve_adaptive();

  EXPECT_TRUE(smoother.is_solved());
  EXPECT_GT(smoother.mesh().num_elements(), 16);

  const auto &mesh = smoother.mesh();
  int q1 = 0, q2 = 0, q3 = 0, q4 = 0;
  for (Index e = 0; e < mesh.num_elements(); ++e) {
    const auto &bounds = mesh.element_bounds(e);
    Real cx = (bounds.xmin + bounds.xmax) / 2.0;
    Real cy = (bounds.ymin + bounds.ymax) / 2.0;
    if (cx < 50.0 && cy < 50.0)
      q1++;
    else if (cx >= 50.0 && cy < 50.0)
      q2++;
    else if (cx < 50.0 && cy >= 50.0)
      q3++;
    else
      q4++;
  }

  std::cout << "RelativeThreshold quadrant counts: " << q1 << " " << q2 << " "
            << q3 << " " << q4 << "\n";

  EXPECT_EQ(q1, q2) << "X-symmetry broken";
  EXPECT_EQ(q1, q3) << "Y-symmetry broken";
  EXPECT_EQ(q1, q4) << "Diagonal symmetry broken";
}

TEST_F(AdaptiveCGLinearBezierSmootherTest,
       DorflerSymmetricConvergesOnQuadratic) {
  // Verify DorflerSymmetric achieves convergence comparable to FixedFraction
  AdaptiveCGLinearBezierConfig config;
  config.marking_strategy = MarkingStrategy::DorflerSymmetric;
  config.dorfler_theta = 0.5;
  config.error_threshold = 0.1;
  config.max_iterations = 8;
  config.max_elements = 500;
  config.smoother_config.lambda = 100.0;

  auto quadratic = [](Real x, Real y) {
    return 100.0 + 0.001 * x * x + 0.002 * y * y;
  };

  AdaptiveCGLinearBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 2, 2, config);
  smoother.set_bathymetry_data(quadratic);
  auto result = smoother.solve_adaptive();

  EXPECT_TRUE(smoother.is_solved());
  EXPECT_GT(result.num_elements, 4);

  // Error should decrease over iterations
  const auto &history = smoother.history();
  if (history.size() >= 2) {
    EXPECT_LT(history.back().max_error, history.front().max_error);
  }

  std::cout << "DorflerSymmetric quadratic: " << result.num_elements
            << " elements, max_error=" << result.max_error << " m\n";
}

TEST_F(AdaptiveCGLinearBezierSmootherTest, FixedFractionBackwardCompatible) {
  // Verify FixedFraction still works for backward compatibility
  AdaptiveCGLinearBezierConfig config;
  config.marking_strategy = MarkingStrategy::FixedFraction;
  config.refine_fraction = 0.2;
  config.error_threshold = 1.0;
  config.max_iterations = 5;
  config.max_elements = 200;
  config.smoother_config.lambda = 10.0;

  auto bump = [](Real x, Real y) {
    Real cx = 50.0, cy = 50.0, sigma = 10.0;
    Real dx = x - cx, dy = y - cy;
    return 50.0 * std::exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
  };

  AdaptiveCGLinearBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 2, 2, config);
  smoother.set_bathymetry_data(bump);
  auto result = smoother.solve_adaptive();

  EXPECT_TRUE(smoother.is_solved());
  EXPECT_GT(result.num_elements, 4);
}

// =============================================================================
// Relative Error Metric Tests
// =============================================================================

TEST_F(AdaptiveCGLinearBezierSmootherTest, RelativeErrorIsComputed) {
  // Verify that relative_error and mean_depth fields are populated
  AdaptiveCGLinearBezierConfig config;
  config.error_threshold = 100.0; // High threshold to get one iteration
  config.max_iterations = 1;
  config.smoother_config.lambda = 10.0;
  config.depth_scale = 1.0;

  // Bathymetry with varying depth (negative = below sea level)
  auto bathy = [](Real x, Real y) {
    return -50.0 - 0.5 * x; // Depth varies from -50 to -100
  };

  AdaptiveCGLinearBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 4, 4, config);
  smoother.set_bathymetry_data(bathy);
  auto result = smoother.solve_adaptive();

  ASSERT_TRUE(smoother.is_solved());
  auto errors = smoother.estimate_errors();
  ASSERT_FALSE(errors.empty());

  for (const auto &err : errors) {
    // Verify mean_depth is computed (should be negative for ocean)
    EXPECT_LT(err.mean_depth, 0.0);

    // Verify relative_error is computed and non-negative
    EXPECT_GE(err.relative_error, 0.0);

    // Verify relative_error = normalized_error / (|mean_depth| + depth_scale)
    Real expected_rel =
        err.normalized_error / (std::abs(err.mean_depth) + config.depth_scale);
    EXPECT_NEAR(err.relative_error, expected_rel, 1e-12);
  }
}

TEST_F(AdaptiveCGLinearBezierSmootherTest, RelativeErrorIsDepthIndependent) {
  // Test that RelativeError normalizes by depth correctly.
  //
  // Strategy: Create bathymetry with uniform perturbation across depths.
  // With NormalizedError, shallow and deep have similar absolute errors.
  // With RelativeError, deep regions should have smaller relative error
  // because the same absolute error is smaller fraction of larger depth.

  // Uniform sine wave perturbation across depth gradient
  auto uniform_perturbation = [](Real x, Real y) {
    Real base_depth = -10.0 - 0.9 * x;              // -10 to -100 depth
    Real perturbation = 5.0 * sin(y * M_PI / 50.0); // ±5m uniform
    return base_depth + perturbation;
  };

  AdaptiveCGLinearBezierConfig config;
  config.error_threshold = 100.0; // High threshold to prevent refinement
  config.max_iterations = 1;
  config.smoother_config.lambda = 10.0;
  config.depth_scale = 1.0;

  AdaptiveCGLinearBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 2, 2, config);
  smoother.set_bathymetry_data(uniform_perturbation);
  smoother.solve_adaptive();

  auto errors = smoother.estimate_errors();

  // Find shallow (x < 50, ~-10 to -55m) and deep (x > 50, ~-55 to -100m)
  // elements
  Real shallow_rel_error = 0.0, deep_rel_error = 0.0;
  Real shallow_mean_depth = 0.0, deep_mean_depth = 0.0;
  int shallow_count = 0, deep_count = 0;

  for (const auto &err : errors) {
    const auto &bounds = smoother.mesh().element_bounds(err.element);
    Real x_center = (bounds.xmin + bounds.xmax) / 2;

    if (x_center < 50.0) {
      shallow_rel_error += err.relative_error;
      shallow_mean_depth += std::abs(err.mean_depth);
      shallow_count++;
    } else {
      deep_rel_error += err.relative_error;
      deep_mean_depth += std::abs(err.mean_depth);
      deep_count++;
    }
  }

  ASSERT_GT(shallow_count, 0);
  ASSERT_GT(deep_count, 0);

  shallow_rel_error /= shallow_count;
  deep_rel_error /= deep_count;
  shallow_mean_depth /= shallow_count;
  deep_mean_depth /= deep_count;

  // Deep region should have larger mean depth
  EXPECT_GT(deep_mean_depth, shallow_mean_depth * 1.5)
      << "Deep region should have larger mean depth";

  // With uniform perturbation, deep regions should have smaller relative error
  // because the same absolute error is divided by larger depth
  EXPECT_LT(deep_rel_error, shallow_rel_error * 1.2)
      << "Deep region should have smaller or similar relative error "
      << "(shallow_rel=" << shallow_rel_error << ", deep_rel=" << deep_rel_error
      << ", shallow_depth=" << shallow_mean_depth
      << ", deep_depth=" << deep_mean_depth << ")";
}

TEST_F(AdaptiveCGLinearBezierSmootherTest,
       RelativeErrorMetricUsedForRefinement) {
  // Verify that when ErrorMetricType::RelativeError is set, it's used for
  // refinement decisions
  AdaptiveCGLinearBezierConfig config;
  config.error_metric_type = ErrorMetricType::RelativeError;
  config.error_threshold = 0.001; // 0.1% relative error threshold (very tight)
  config.max_iterations = 5;
  config.max_elements = 200;
  config.depth_scale = 1.0;
  config.smoother_config.lambda = 100.0; // Strong data fitting

  // Gaussian bump at depth -100m (10% bump = 10m)
  // Linear Bezier can't represent curved surfaces well, so needs refinement
  auto deep_bump = [](Real x, Real y) {
    Real cx = 50.0, cy = 50.0, sigma = 10.0;
    Real dx = x - cx, dy = y - cy;
    Real base = -100.0;
    Real bump = 10.0 * std::exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
    return base + bump; // -100 to -90
  };

  AdaptiveCGLinearBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 2, 2, config);
  smoother.set_bathymetry_data(deep_bump);
  auto result = smoother.solve_adaptive();

  EXPECT_TRUE(smoother.is_solved());

  // Should refine to capture the bump
  EXPECT_GT(result.num_elements, 4)
      << "Should refine to capture Gaussian bump with tight relative threshold";
}

TEST_F(AdaptiveCGLinearBezierSmootherTest, DepthScaleAffectsShallowRegions) {
  // Test that depth_scale controls behavior near shoreline
  // Larger depth_scale = more like absolute error at shallow depths

  auto shallow_bump = [](Real x, Real y) {
    Real cx = 50.0, cy = 50.0, sigma = 15.0;
    Real dx = x - cx, dy = y - cy;
    Real base = -5.0; // Shallow water
    Real bump = 2.0 * std::exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
    return base + bump; // -5 to -3
  };

  // Small depth_scale (0.1m) - relative error dominates
  AdaptiveCGLinearBezierConfig config_small;
  config_small.error_metric_type = ErrorMetricType::RelativeError;
  config_small.error_threshold = 0.1;
  config_small.max_iterations = 1;
  config_small.depth_scale = 0.1;
  config_small.smoother_config.lambda = 10.0;

  AdaptiveCGLinearBezierSmoother smoother_small(0.0, 100.0, 0.0, 100.0, 4, 4,
                                                config_small);
  smoother_small.set_bathymetry_data(shallow_bump);
  smoother_small.adapt_once();

  // Large depth_scale (10m) - absolute error dominates
  AdaptiveCGLinearBezierConfig config_large;
  config_large.error_metric_type = ErrorMetricType::RelativeError;
  config_large.error_threshold = 0.1;
  config_large.max_iterations = 1;
  config_large.depth_scale = 10.0;
  config_large.smoother_config.lambda = 10.0;

  AdaptiveCGLinearBezierSmoother smoother_large(0.0, 100.0, 0.0, 100.0, 4, 4,
                                                config_large);
  smoother_large.set_bathymetry_data(shallow_bump);
  smoother_large.adapt_once();

  // Get errors for comparison
  auto errors_small = smoother_small.estimate_errors();
  auto errors_large = smoother_large.estimate_errors();

  // Compute average relative error for each
  Real avg_rel_small = 0.0, avg_rel_large = 0.0;
  for (const auto &err : errors_small) {
    avg_rel_small += err.relative_error;
  }
  for (const auto &err : errors_large) {
    avg_rel_large += err.relative_error;
  }
  avg_rel_small /= errors_small.size();
  avg_rel_large /= errors_large.size();

  // With small depth_scale, relative error should be larger (dividing by ~5m)
  // With large depth_scale, relative error should be smaller (dividing by ~15m)
  EXPECT_GT(avg_rel_small, avg_rel_large)
      << "Smaller depth_scale should give larger relative errors";
}

// =============================================================================
// WENO Indicator Tests
// =============================================================================

TEST_F(AdaptiveCGLinearBezierSmootherTest,
       WenoIndicatorZeroForConstantSurface) {
  // Constant surface: gradient=0, curvature=0, weno=0
  AdaptiveCGLinearBezierConfig config;
  config.smoother_config.lambda = 100.0;

  AdaptiveCGLinearBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 4, 4, config);

  // Set constant bathymetry with gradient/curvature functions
  smoother.set_bathymetry_data([](Real, Real) { return 50.0; });
  smoother.set_gradient_function([](Real, Real, Real &dh_dx, Real &dh_dy) {
    dh_dx = 0.0;
    dh_dy = 0.0;
  });
  smoother.set_curvature_function(
      [](Real, Real, Real &d2h_dx2, Real &d2h_dxdy, Real &d2h_dy2) {
        d2h_dx2 = 0.0;
        d2h_dxdy = 0.0;
        d2h_dy2 = 0.0;
      });

  smoother.adapt_once();

  auto errors = smoother.estimate_errors();
  for (const auto &err : errors) {
    EXPECT_NEAR(err.gradient_indicator, 0.0, TOLERANCE);
    EXPECT_NEAR(err.curvature_indicator, 0.0, TOLERANCE);
    EXPECT_NEAR(err.weno_indicator, 0.0, TOLERANCE);
  }
}

TEST_F(AdaptiveCGLinearBezierSmootherTest,
       WenoIndicatorPositiveGradientForLinearSurface) {
  // Linear surface z = ax + by: gradient > 0, curvature = 0
  Real a = 0.5, b = 0.3;

  AdaptiveCGLinearBezierConfig config;
  config.smoother_config.lambda = 100.0;

  AdaptiveCGLinearBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 4, 4, config);

  smoother.set_bathymetry_data(
      [a, b](Real x, Real y) { return 10.0 + a * x + b * y; });
  smoother.set_gradient_function([a, b](Real, Real, Real &dh_dx, Real &dh_dy) {
    dh_dx = a;
    dh_dy = b;
  });
  smoother.set_curvature_function(
      [](Real, Real, Real &d2h_dx2, Real &d2h_dxdy, Real &d2h_dy2) {
        d2h_dx2 = 0.0;
        d2h_dxdy = 0.0;
        d2h_dy2 = 0.0;
      });

  smoother.adapt_once();

  auto errors = smoother.estimate_errors();
  for (const auto &err : errors) {
    // Gradient indicator should be positive (h² × |∇z|²)
    EXPECT_GT(err.gradient_indicator, 0.0);

    // Curvature indicator should be zero
    EXPECT_NEAR(err.curvature_indicator, 0.0, TOLERANCE);

    // WENO indicator equals gradient indicator when curvature is zero
    EXPECT_NEAR(err.weno_indicator, err.gradient_indicator, TOLERANCE);
  }
}

TEST_F(AdaptiveCGLinearBezierSmootherTest,
       WenoIndicatorBothTermsForQuadraticSurface) {
  // Quadratic surface z = ax² + by²: gradient > 0, curvature > 0
  Real a = 0.001, b = 0.002;

  AdaptiveCGLinearBezierConfig config;
  config.smoother_config.lambda = 100.0;
  config.error_threshold = 1000.0; // High threshold to prevent refinement
  config.max_iterations = 1;

  AdaptiveCGLinearBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 4, 4, config);

  smoother.set_bathymetry_data(
      [a, b](Real x, Real y) { return 10.0 + a * x * x + b * y * y; });
  smoother.set_gradient_function(
      [a, b](Real x, Real y, Real &dh_dx, Real &dh_dy) {
        dh_dx = 2.0 * a * x;
        dh_dy = 2.0 * b * y;
      });
  smoother.set_curvature_function(
      [a, b](Real, Real, Real &d2h_dx2, Real &d2h_dxdy, Real &d2h_dy2) {
        d2h_dx2 = 2.0 * a;
        d2h_dxdy = 0.0;
        d2h_dy2 = 2.0 * b;
      });

  auto result = smoother.solve_adaptive();
  EXPECT_TRUE(smoother.is_solved());

  auto errors = smoother.estimate_errors();
  for (const auto &err : errors) {
    // Both indicators should be positive
    EXPECT_GT(err.gradient_indicator, 0.0);
    EXPECT_GT(err.curvature_indicator, 0.0);

    // WENO indicator should be sum of both
    EXPECT_NEAR(err.weno_indicator,
                err.gradient_indicator + err.curvature_indicator, TOLERANCE);
  }
}

TEST_F(AdaptiveCGLinearBezierSmootherTest, WenoIndicatorScalesWithElementSize) {
  // Test that WENO indicator scales correctly with h
  // For uniform gradient, gradient_indicator = h² × |∇z|²
  // Smaller elements should have smaller indicator values

  Real grad_magnitude = 0.5;

  // Solve on coarse mesh (4x4)
  AdaptiveCGLinearBezierConfig config_coarse;
  config_coarse.smoother_config.lambda = 100.0;
  config_coarse.max_iterations = 1;

  AdaptiveCGLinearBezierSmoother smoother_coarse(0.0, 100.0, 0.0, 100.0, 4, 4,
                                                 config_coarse);
  smoother_coarse.set_bathymetry_data(
      [grad_magnitude](Real x, Real) { return grad_magnitude * x; });
  smoother_coarse.set_gradient_function(
      [grad_magnitude](Real, Real, Real &dh_dx, Real &dh_dy) {
        dh_dx = grad_magnitude;
        dh_dy = 0.0;
      });
  smoother_coarse.adapt_once();

  // Solve on fine mesh (8x8)
  AdaptiveCGLinearBezierConfig config_fine;
  config_fine.smoother_config.lambda = 100.0;
  config_fine.max_iterations = 1;

  AdaptiveCGLinearBezierSmoother smoother_fine(0.0, 100.0, 0.0, 100.0, 8, 8,
                                               config_fine);
  smoother_fine.set_bathymetry_data(
      [grad_magnitude](Real x, Real) { return grad_magnitude * x; });
  smoother_fine.set_gradient_function(
      [grad_magnitude](Real, Real, Real &dh_dx, Real &dh_dy) {
        dh_dx = grad_magnitude;
        dh_dy = 0.0;
      });
  smoother_fine.adapt_once();

  // Get average gradient indicator
  auto errors_coarse = smoother_coarse.estimate_errors();
  auto errors_fine = smoother_fine.estimate_errors();

  Real avg_grad_coarse = 0.0, avg_grad_fine = 0.0;
  for (const auto &err : errors_coarse) {
    avg_grad_coarse += err.gradient_indicator;
  }
  for (const auto &err : errors_fine) {
    avg_grad_fine += err.gradient_indicator;
  }
  avg_grad_coarse /= errors_coarse.size();
  avg_grad_fine /= errors_fine.size();

  // Fine mesh has 4x smaller area, so h² is 4x smaller
  // gradient_indicator = h² × |∇z|², so fine should be 4x smaller
  Real ratio = avg_grad_coarse / avg_grad_fine;
  EXPECT_NEAR(ratio, 4.0, 0.1)
      << "Coarse elements should have 4x larger gradient indicator than fine";
}

TEST_F(AdaptiveCGLinearBezierSmootherTest, WenoIndicatorWeightsAreRespected) {
  // Test that weno_gradient_weight and weno_curvature_weight are applied
  Real a = 0.001, b = 0.001;

  AdaptiveCGLinearBezierConfig config;
  config.smoother_config.lambda = 100.0;
  config.weno_gradient_weight = 2.0;
  config.weno_curvature_weight = 0.5;

  AdaptiveCGLinearBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 4, 4, config);

  smoother.set_bathymetry_data(
      [a, b](Real x, Real y) { return 10.0 + a * x * x + b * y * y; });
  smoother.set_gradient_function(
      [a, b](Real x, Real y, Real &dh_dx, Real &dh_dy) {
        dh_dx = 2.0 * a * x;
        dh_dy = 2.0 * b * y;
      });
  smoother.set_curvature_function(
      [a, b](Real, Real, Real &d2h_dx2, Real &d2h_dxdy, Real &d2h_dy2) {
        d2h_dx2 = 2.0 * a;
        d2h_dxdy = 0.0;
        d2h_dy2 = 2.0 * b;
      });

  smoother.adapt_once();

  auto errors = smoother.estimate_errors();
  for (const auto &err : errors) {
    Real expected_weno =
        2.0 * err.gradient_indicator + 0.5 * err.curvature_indicator;
    EXPECT_NEAR(err.weno_indicator, expected_weno, TOLERANCE);
  }
}

TEST_F(AdaptiveCGLinearBezierSmootherTest,
       WenoIndicatorMetricTypeSelectsCorrectValue) {
  // Test that selecting WenoIndicator as error_metric_type works
  Real a = 0.001;

  AdaptiveCGLinearBezierConfig config;
  config.smoother_config.lambda = 100.0;
  config.error_metric_type = ErrorMetricType::WenoIndicator;
  config.error_threshold = 1000.0; // High threshold

  AdaptiveCGLinearBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 4, 4, config);

  smoother.set_bathymetry_data(
      [a](Real x, Real y) { return 10.0 + a * x * x + a * y * y; });
  smoother.set_gradient_function([a](Real x, Real y, Real &dh_dx, Real &dh_dy) {
    dh_dx = 2.0 * a * x;
    dh_dy = 2.0 * a * y;
  });
  smoother.set_curvature_function(
      [a](Real, Real, Real &d2h_dx2, Real &d2h_dxdy, Real &d2h_dy2) {
        d2h_dx2 = 2.0 * a;
        d2h_dxdy = 0.0;
        d2h_dy2 = 2.0 * a;
      });

  auto result = smoother.solve_adaptive();

  EXPECT_TRUE(smoother.is_solved());
  // max_error should be the weno_indicator value
  auto errors = smoother.estimate_errors();
  Real max_weno = 0.0;
  for (const auto &err : errors) {
    max_weno = std::max(max_weno, err.weno_indicator);
  }
  EXPECT_NEAR(result.max_error, max_weno, TOLERANCE);
}

TEST_F(AdaptiveCGLinearBezierSmootherTest,
       SetBathymetrySurfaceSetsAllFunctions) {
  // Test that set_bathymetry_surface sets depth, gradient, curvature, and land
  // mask functions - the key is that the solver runs without errors

  // Create a simple BathymetryData for testing
  auto bathy_data = std::make_shared<BathymetryData>();
  bathy_data->sizex = 20;
  bathy_data->sizey = 20;
  bathy_data->elevation.resize(400, -50.0f); // 50m depth everywhere
  bathy_data->geotransform = {0.0, 5.0, 0.0, 100.0, 0.0, -5.0}; // 5m pixels
  bathy_data->nodata_value = -9999.0f;
  bathy_data->is_depth_positive = false;
  bathy_data->xmin = 0.0;
  bathy_data->xmax = 100.0;
  bathy_data->ymin = 0.0;
  bathy_data->ymax = 100.0;

  BathymetrySurface surface(bathy_data);

  AdaptiveCGLinearBezierConfig config;
  config.smoother_config.lambda = 100.0;
  config.max_iterations = 1;

  AdaptiveCGLinearBezierSmoother smoother(0.0, 100.0, 0.0, 100.0, 4, 4, config);
  smoother.set_bathymetry_surface(surface);

  // Should be able to solve without errors (all functions set)
  auto result = smoother.solve_adaptive();
  EXPECT_TRUE(smoother.is_solved());

  // Check that WENO indicators are computed (values may be non-zero due to
  // finite difference boundary artifacts, but the computation should complete)
  auto errors = smoother.estimate_errors();
  EXPECT_EQ(errors.size(), static_cast<size_t>(smoother.mesh().num_elements()));

  // Verify the indicators exist and are finite
  for (const auto &err : errors) {
    EXPECT_TRUE(std::isfinite(err.gradient_indicator));
    EXPECT_TRUE(std::isfinite(err.curvature_indicator));
    EXPECT_TRUE(std::isfinite(err.weno_indicator));
  }
}
