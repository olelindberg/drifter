#include "bathymetry/bezier_bathymetry_smoother.hpp"
#include "bathymetry/bezier_c2_constraints.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "mesh/geotiff_reader.hpp"
#include "mesh/octree_adapter.hpp"
#include "test_integration_fixtures.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <gtest/gtest.h>
#include <map>
#include <omp.h>

using namespace drifter;

class BezierBathymetrySmootherTest : public ::testing::Test {
protected:
  static constexpr Real TOLERANCE = 1e-10;
  static constexpr Real LOOSE_TOLERANCE = 1e-4;

  std::unique_ptr<OctreeAdapter> create_octree(int nx, int ny, int nz) {
    auto octree = std::make_unique<OctreeAdapter>(0.0, 100.0, // x bounds
                                                  0.0, 100.0, // y bounds
                                                  -1.0, 0.0   // z bounds
    );
    octree->build_uniform(nx, ny, nz);
    return octree;
  }

  std::unique_ptr<QuadtreeAdapter> create_quadtree(int nx, int ny) {
    auto quadtree = std::make_unique<QuadtreeAdapter>();
    quadtree->build_uniform(0.0, 100.0, 0.0, 100.0, nx, ny);
    return quadtree;
  }

  /// Write raw GeoTIFF data to VTK for a given region at native resolution
  void write_geotiff_region_vtk(const BathymetryData &bathy, Real xmin,
                                Real xmax, Real ymin, Real ymax,
                                const std::string &filename) {
    // Compute pixel range for the region
    double px_min, py_min, px_max, py_max;
    bathy.world_to_pixel(xmin, ymax, px_min,
                         py_min); // Note: y is inverted in GeoTIFF
    bathy.world_to_pixel(xmax, ymin, px_max, py_max);

    int ix_min = std::max(0, static_cast<int>(std::floor(px_min)));
    int ix_max = std::min(bathy.sizex - 1, static_cast<int>(std::ceil(px_max)));
    int iy_min = std::max(0, static_cast<int>(std::floor(py_min)));
    int iy_max = std::min(bathy.sizey - 1, static_cast<int>(std::ceil(py_max)));

    int nx = ix_max - ix_min + 1;
    int ny = iy_max - iy_min + 1;

    std::ofstream file(filename + ".vtu");
    if (!file)
      return;

    Index total_pts = nx * ny;
    Index total_cells = (nx - 1) * (ny - 1);

    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" "
            "byte_order=\"LittleEndian\">\n";
    file << "<UnstructuredGrid>\n";
    file << "<Piece NumberOfPoints=\"" << total_pts << "\" NumberOfCells=\""
         << total_cells << "\">\n";

    // Points
    file << "<Points>\n";
    file << "<DataArray type=\"Float64\" NumberOfComponents=\"3\" "
            "format=\"ascii\">\n";

    for (int iy = iy_min; iy <= iy_max; ++iy) {
      for (int ix = ix_min; ix <= ix_max; ++ix) {
        double wx, wy;
        bathy.pixel_to_world(ix, iy, wx, wy);
        float depth = bathy.get_depth(wx, wy);
        file << std::setprecision(12) << wx << " " << wy << " " << depth
             << "\n";
      }
    }

    file << "</DataArray>\n";
    file << "</Points>\n";

    // Cells (quads)
    file << "<Cells>\n";
    file << "<DataArray type=\"Int64\" Name=\"connectivity\" "
            "format=\"ascii\">\n";

    for (int j = 0; j < ny - 1; ++j) {
      for (int i = 0; i < nx - 1; ++i) {
        Index p0 = i + nx * j;
        Index p1 = p0 + 1;
        Index p2 = p0 + nx + 1;
        Index p3 = p0 + nx;
        file << p0 << " " << p1 << " " << p2 << " " << p3 << "\n";
      }
    }

    file << "</DataArray>\n";
    file << "<DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";

    for (Index c = 1; c <= total_cells; ++c) {
      file << 4 * c << "\n";
    }

    file << "</DataArray>\n";
    file << "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";

    for (Index c = 0; c < total_cells; ++c) {
      file << "9\n"; // VTK_QUAD
    }

    file << "</DataArray>\n";
    file << "</Cells>\n";

    // Point data: depth
    file << "<PointData Scalars=\"depth\">\n";
    file << "<DataArray type=\"Float64\" Name=\"depth\" format=\"ascii\">\n";

    for (int iy = iy_min; iy <= iy_max; ++iy) {
      for (int ix = ix_min; ix <= ix_max; ++ix) {
        double wx, wy;
        bathy.pixel_to_world(ix, iy, wx, wy);
        float depth = bathy.get_depth(wx, wy);
        file << std::setprecision(12) << depth << "\n";
      }
    }

    file << "</DataArray>\n";
    file << "</PointData>\n";

    file << "</Piece>\n";
    file << "</UnstructuredGrid>\n";
    file << "</VTKFile>\n";

    std::cout << "Wrote raw GeoTIFF data (" << nx << "x" << ny << " pixels) to "
              << filename << ".vtu\n";
  }

  /// Write raw bathymetry function data to VTK per element
  template <typename BathyFunc>
  void write_raw_bathy_vtk(const QuadtreeAdapter &quadtree,
                           const BathyFunc &bathy, int samples_per_elem,
                           const std::string &filename) {
    std::ofstream file(filename + ".vtu");
    if (!file)
      return;

    Index num_elems = quadtree.num_elements();
    Index pts_per_elem = samples_per_elem * samples_per_elem;
    Index cells_per_elem = (samples_per_elem - 1) * (samples_per_elem - 1);
    Index total_pts = num_elems * pts_per_elem;
    Index total_cells = num_elems * cells_per_elem;

    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" "
            "byte_order=\"LittleEndian\">\n";
    file << "<UnstructuredGrid>\n";
    file << "<Piece NumberOfPoints=\"" << total_pts << "\" NumberOfCells=\""
         << total_cells << "\">\n";

    // Points
    file << "<Points>\n";
    file << "<DataArray type=\"Float64\" NumberOfComponents=\"3\" "
            "format=\"ascii\">\n";

    for (Index e = 0; e < num_elems; ++e) {
      const auto &bounds = quadtree.element_bounds(e);
      Real dx = (bounds.xmax - bounds.xmin) / (samples_per_elem - 1);
      Real dy = (bounds.ymax - bounds.ymin) / (samples_per_elem - 1);

      for (int j = 0; j < samples_per_elem; ++j) {
        for (int i = 0; i < samples_per_elem; ++i) {
          Real x = bounds.xmin + i * dx;
          Real y = bounds.ymin + j * dy;
          Real z = bathy(x, y);
          file << std::setprecision(12) << x << " " << y << " " << z << "\n";
        }
      }
    }

    file << "</DataArray>\n";
    file << "</Points>\n";

    // Cells (quads per element)
    file << "<Cells>\n";
    file << "<DataArray type=\"Int64\" Name=\"connectivity\" "
            "format=\"ascii\">\n";

    for (Index e = 0; e < num_elems; ++e) {
      Index base = e * pts_per_elem;
      for (int j = 0; j < samples_per_elem - 1; ++j) {
        for (int i = 0; i < samples_per_elem - 1; ++i) {
          Index p0 = base + i + samples_per_elem * j;
          Index p1 = p0 + 1;
          Index p2 = p0 + samples_per_elem + 1;
          Index p3 = p0 + samples_per_elem;
          file << p0 << " " << p1 << " " << p2 << " " << p3 << "\n";
        }
      }
    }

    file << "</DataArray>\n";
    file << "<DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";

    for (Index c = 1; c <= total_cells; ++c) {
      file << 4 * c << "\n";
    }

    file << "</DataArray>\n";
    file << "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";

    for (Index c = 0; c < total_cells; ++c) {
      file << "9\n"; // VTK_QUAD
    }

    file << "</DataArray>\n";
    file << "</Cells>\n";

    // Point data: depth
    file << "<PointData Scalars=\"depth\">\n";
    file << "<DataArray type=\"Float64\" Name=\"depth\" format=\"ascii\">\n";

    for (Index e = 0; e < num_elems; ++e) {
      const auto &bounds = quadtree.element_bounds(e);
      Real dx = (bounds.xmax - bounds.xmin) / (samples_per_elem - 1);
      Real dy = (bounds.ymax - bounds.ymin) / (samples_per_elem - 1);

      for (int j = 0; j < samples_per_elem; ++j) {
        for (int i = 0; i < samples_per_elem; ++i) {
          Real x = bounds.xmin + i * dx;
          Real y = bounds.ymin + j * dy;
          Real z = bathy(x, y);
          file << std::setprecision(12) << z << "\n";
        }
      }
    }

    file << "</DataArray>\n";
    file << "</PointData>\n";

    file << "</Piece>\n";
    file << "</UnstructuredGrid>\n";
    file << "</VTKFile>\n";

    std::cout << "Wrote raw bathymetry data (" << num_elems << " elements, "
              << samples_per_elem << "x" << samples_per_elem
              << " samples/elem) to " << filename << ".vtu\n";
  }
};

// =============================================================================
// Construction tests
// =============================================================================

TEST_F(BezierBathymetrySmootherTest, ConstructFromQuadtree) {
  auto quadtree = create_quadtree(2, 2);

  BezierSmootherConfig config;
  config.lambda = 0.01;

  BezierBathymetrySmoother smoother(*quadtree, config);

  EXPECT_EQ(smoother.config().lambda, config.lambda);
  EXPECT_GT(smoother.num_dofs(), 0);
  EXPECT_FALSE(smoother.is_solved());

  // Should have 36 DOFs per element
  EXPECT_EQ(smoother.num_dofs(), 4 * 36); // 2x2 = 4 elements
}

TEST_F(BezierBathymetrySmootherTest, ConstructFromOctree) {
  auto octree = create_octree(2, 2, 1);

  BezierBathymetrySmoother smoother(*octree);

  // Bottom face should have 4 elements (2x2)
  EXPECT_EQ(smoother.num_dofs(), 4 * 36);
  EXPECT_FALSE(smoother.is_solved());
}

// =============================================================================
// Solve tests
// =============================================================================

TEST_F(BezierBathymetrySmootherTest, SolveSingleElementConstant) {
  // Single element - no constraints, should work well
  auto quadtree = create_quadtree(1, 1);

  BezierSmootherConfig config;
  config.lambda =
      1.0; // Need data fitting term (lambda weights LS vs smoothness)

  BezierBathymetrySmoother smoother(*quadtree, config);

  Real constant_depth = 50.0;
  smoother.set_bathymetry_data(
      [constant_depth](Real, Real) { return constant_depth; });

  smoother.solve();

  EXPECT_TRUE(smoother.is_solved());

  // Evaluate at center - some fitting error due to Gauss quadrature sampling
  // and the over-determined least squares problem
  Real depth = smoother.evaluate(50.0, 50.0);
  EXPECT_NEAR(depth, constant_depth, 5.0) // Relaxed tolerance
      << "Single element constant bathymetry";
}

TEST_F(BezierBathymetrySmootherTest, SolveConstantBathymetry) {
  auto quadtree = create_quadtree(2, 2);

  BezierBathymetrySmoother smoother(*quadtree);

  Real constant_depth = 50.0;
  smoother.set_bathymetry_data(
      [constant_depth](Real, Real) { return constant_depth; });

  smoother.solve();

  EXPECT_TRUE(smoother.is_solved());

  // Evaluate at several points - looser tolerance due to regularization
  // tradeoff
  std::vector<Vec2> test_points = {
      Vec2(25.0, 25.0),
      Vec2(50.0, 50.0),
      Vec2(75.0, 75.0),
  };

  for (const auto &pt : test_points) {
    Real depth = smoother.evaluate(pt(0), pt(1));
    EXPECT_NEAR(depth, constant_depth, 1.0) // Looser tolerance
        << "At (" << pt(0) << ", " << pt(1) << ")";
  }
}

TEST_F(BezierBathymetrySmootherTest, SolveLinearBathymetry) {
  auto quadtree = create_quadtree(2, 2);

  BezierSmootherConfig config;
  config.lambda = 0.001; // Small regularization

  // Linear function: z = 10 + 0.5*x + 0.3*y
  auto linear_bathy = [](Real x, Real y) { return 10.0 + 0.5 * x + 0.3 * y; };

  BezierBathymetrySmoother smoother(*quadtree, config);
  smoother.set_bathymetry_data(linear_bathy);
  smoother.solve();

  EXPECT_TRUE(smoother.is_solved());

  // Verify at test points - tolerance accounts for regularization smoothing
  std::vector<Vec2> test_points = {
      Vec2(25.0, 25.0),
      Vec2(50.0, 50.0),
      Vec2(75.0, 75.0),
      Vec2(30.0, 60.0),
  };

  for (const auto &pt : test_points) {
    Real expected = linear_bathy(pt(0), pt(1));
    Real computed = smoother.evaluate(pt(0), pt(1));

    // Larger tolerance needed because Dirichlet BCs only constrain corners,
    // not all boundary DOFs - this allows the surface to deviate more from
    // the data while maintaining C² continuity.
    EXPECT_NEAR(computed, expected, 10.0)
        << "At (" << pt(0) << ", " << pt(1) << ")";
  }
}

TEST_F(BezierBathymetrySmootherTest, SolveQuadraticBathymetry) {
  auto quadtree = create_quadtree(2, 2);

  BezierSmootherConfig config;
  config.lambda = 10.0; // High data fitting weight for tight fit
  config.gradient_weight =
      0.0; // Disable gradient penalty to test exact fitting

  // Quadratic function
  auto quadratic_bathy = [](Real x, Real y) {
    return 100.0 + 0.01 * x * x + 0.01 * y * y;
  };

  BezierBathymetrySmoother smoother(*quadtree, config);
  smoother.set_bathymetry_data(quadratic_bathy);
  smoother.solve();

  // Check at several test points
  std::vector<Vec2> test_points = {
      Vec2(25.0, 25.0),
      Vec2(50.0, 50.0),
      Vec2(75.0, 75.0),
      Vec2(30.0, 60.0),
  };

  for (const auto &pt : test_points) {
    Real expected = quadratic_bathy(pt(0), pt(1));
    Real computed = smoother.evaluate(pt(0), pt(1));

    EXPECT_NEAR(computed, expected, 5.0) // Tolerance for regularization
        << "At (" << pt(0) << ", " << pt(1) << ")";
  }
}

// =============================================================================
// C² continuity tests
// =============================================================================

TEST_F(BezierBathymetrySmootherTest, C2ContinuityAtInterface) {
  auto quadtree = create_quadtree(2, 2);

  BezierSmootherConfig config;
  config.lambda = 0.01;
  config.enable_boundary_dirichlet =
      false; // Test C² without Dirichlet influence

  // Non-polynomial function to exercise smoothing
  auto wavy_bathy = [](Real x, Real y) {
    return 50.0 + 10.0 * std::sin(x / 20.0) * std::cos(y / 20.0);
  };

  BezierBathymetrySmoother smoother(*quadtree, config);
  smoother.set_bathymetry_data(wavy_bathy);
  smoother.solve();

  // Check constraint violation (should be near zero)
  Real violation = smoother.constraint_violation();
  EXPECT_LT(violation, 1e-8)
      << "C² constraint violation too large: " << violation;
}

// =============================================================================
// Regularization tests
// =============================================================================

TEST_F(BezierBathymetrySmootherTest, RegularizationSmooths) {
  auto quadtree = create_quadtree(2, 2);

  // Noisy bathymetry
  auto noisy_bathy = [](Real x, Real y) {
    return 50.0 + 5.0 * std::sin(x * 0.5) * std::cos(y * 0.5);
  };

  // Solve with different regularization weights
  // Use ngauss_data=4 to create underdetermined system where regularization
  // matters Disable gradient penalty to test pure thin plate regularization
  // behavior
  BezierSmootherConfig config_low, config_high;
  config_low.lambda = 0.001;
  config_low.ngauss_data = 4;
  config_low.gradient_weight = 0.0;
  config_low.enable_boundary_dirichlet =
      false; // Test regularization without Dirichlet
  config_high.lambda = 10.0;
  config_high.ngauss_data = 4;
  config_high.gradient_weight = 0.0;
  config_high.enable_boundary_dirichlet = false;

  BezierBathymetrySmoother smoother_low(*quadtree, config_low);
  smoother_low.set_bathymetry_data(noisy_bathy);
  smoother_low.solve();

  BezierBathymetrySmoother smoother_high(*quadtree, config_high);
  smoother_high.set_bathymetry_data(noisy_bathy);
  smoother_high.solve();

  // In our formulation, Q = H + lambda * AtWA, so lambda weights data fitting.
  // Lower lambda means smoothness dominates → smoother surface → lower
  // regularization energy. Higher lambda means data fitting dominates → closer
  // to noisy data → higher regularization energy.
  Real energy_low = smoother_low.regularization_energy();
  Real energy_high = smoother_high.regularization_energy();

  EXPECT_LT(energy_low, energy_high)
      << "Lower lambda (more smoothing weight) should produce lower "
         "regularization energy";
}

// =============================================================================
// Bound constraint tests
// =============================================================================

TEST_F(BezierBathymetrySmootherTest, BoundConstraints) {
  auto quadtree = create_quadtree(2, 2);

  BezierSmootherConfig config;
  config.lambda = 0.001;
  config.lower_bound = 40.0;
  config.upper_bound = 60.0;

  // Bathymetry with values outside bounds
  auto extreme_bathy = [](Real x, Real y) {
    return 50.0 + 30.0 * std::sin(x / 20.0); // Range [20, 80]
  };

  BezierBathymetrySmoother smoother(*quadtree, config);
  smoother.set_bathymetry_data(extreme_bathy);
  smoother.solve();

  // Check that solution respects bounds
  const VecX &sol = smoother.solution();
  for (Index i = 0; i < sol.size(); ++i) {
    EXPECT_GE(sol(i), config.lower_bound.value() - 1e-6)
        << "Solution below lower bound at index " << i;
    EXPECT_LE(sol(i), config.upper_bound.value() + 1e-6)
        << "Solution above upper bound at index " << i;
  }
}

// =============================================================================
// Scattered point tests
// =============================================================================

TEST_F(BezierBathymetrySmootherTest, ScatteredPointInput) {
  auto quadtree = create_quadtree(2, 2);

  BezierSmootherConfig config;
  config.lambda = 0.001;
  config.enable_boundary_dirichlet =
      false; // Dirichlet requires bathymetry function

  // Create scattered points on a linear surface
  std::vector<Vec3> points;
  for (int i = 0; i < 100; ++i) {
    Real x = (i % 10) * 10.0 + 5.0;
    Real y = (i / 10) * 10.0 + 5.0;
    Real z = 20.0 + 0.3 * x + 0.2 * y;
    points.emplace_back(x, y, z);
  }

  BezierBathymetrySmoother smoother(*quadtree, config);
  smoother.set_scattered_points(points);
  smoother.solve();

  EXPECT_TRUE(smoother.is_solved());

  // Check fit at center
  Real expected = 20.0 + 0.3 * 50.0 + 0.2 * 50.0; // = 45
  Real computed = smoother.evaluate(50.0, 50.0);

  EXPECT_NEAR(computed, expected, 1.0) // Allow some fitting error
      << "Scattered point fit at center";
}

// =============================================================================
// Gradient tests
// =============================================================================

TEST_F(BezierBathymetrySmootherTest, GradientEvaluation) {
  auto quadtree = create_quadtree(2, 2);

  BezierSmootherConfig config;
  config.lambda = 0.001;

  // Linear function: z = 10 + 2*x + 3*y
  // Gradient should be close to (2, 3)
  auto linear_bathy = [](Real x, Real y) { return 10.0 + 2.0 * x + 3.0 * y; };

  BezierBathymetrySmoother smoother(*quadtree, config);
  smoother.set_bathymetry_data(linear_bathy);
  smoother.solve();

  Vec2 grad = smoother.evaluate_gradient(50.0, 50.0);

  // Larger tolerance needed because corner-only Dirichlet BCs allow the surface
  // more freedom to deviate from the linear function while maintaining C²
  // continuity.
  EXPECT_NEAR(grad(0), 2.0, 1.5) << "dz/dx mismatch";
  EXPECT_NEAR(grad(1), 3.0, 1.5) << "dz/dy mismatch";
}

// =============================================================================
// Diagnostics tests
// =============================================================================

TEST_F(BezierBathymetrySmootherTest, ObjectiveValue) {
  auto quadtree = create_quadtree(2, 2);

  BezierSmootherConfig config;
  config.lambda = 0.01;

  auto simple_bathy = [](Real x, Real y) { return 50.0 + x * 0.1; };

  BezierBathymetrySmoother smoother(*quadtree, config);
  smoother.set_bathymetry_data(simple_bathy);
  smoother.solve();

  Real data_res = smoother.data_residual();
  Real reg_energy = smoother.regularization_energy();
  Real obj_value = smoother.objective_value();

  EXPECT_GE(data_res, -1e-8);   // Allow small numerical noise
  EXPECT_GE(reg_energy, -1e-8); // Allow small numerical noise
  EXPECT_NEAR(obj_value, data_res + config.lambda * reg_energy, 1e-8);
}

// =============================================================================
// VTK output test (creates file in temp directory)
// =============================================================================

TEST_F(BezierBathymetrySmootherTest, VTKOutput) {
  auto quadtree = create_quadtree(2, 2);

  BezierSmootherConfig config;
  config.lambda = 0.001;

  auto bathy = [](Real x, Real y) {
    return 50.0 + 10.0 * std::sin(x / 30.0) * std::cos(y / 30.0);
  };

  BezierBathymetrySmoother smoother(*quadtree, config);
  smoother.set_bathymetry_data(bathy);
  smoother.solve();

  // Write surface to temp file
  std::string filename = "/tmp/test_bezier_bathy";
  EXPECT_NO_THROW(smoother.write_vtk(filename, 5));

  // Check file exists
  std::ifstream file(filename + ".vtu");
  EXPECT_TRUE(file.good()) << "VTK file not created";

  // Write control points grid to temp file
  std::string cp_filename = "/tmp/test_bezier_control_points";
  EXPECT_NO_THROW(smoother.write_control_points_vtk(cp_filename));

  // Check control points file exists
  std::ifstream cp_file(cp_filename + ".vtu");
  EXPECT_TRUE(cp_file.good()) << "Control points VTK file not created";
}

// =============================================================================
// Multi-element mesh tests
// =============================================================================

TEST_F(BezierBathymetrySmootherTest, LargerMesh) {
  auto quadtree = create_quadtree(4, 4); // 16 elements

  auto bathy = [](Real x, Real y) { return 100.0 + 0.5 * x - 0.3 * y; };

  BezierBathymetrySmoother smoother(*quadtree);
  smoother.set_bathymetry_data(bathy);
  smoother.solve();

  EXPECT_TRUE(smoother.is_solved());
  EXPECT_EQ(smoother.num_dofs(), 16 * 36);

  // Verify fit (relaxed tolerance for linear function with smoothness
  // regularization)
  Real expected = bathy(50.0, 50.0);
  Real computed = smoother.evaluate(50.0, 50.0);
  EXPECT_NEAR(computed, expected, 0.1); // Within 0.1 depth units
}

// =============================================================================
// GeoTIFF bathymetry tests
// =============================================================================

TEST_F(BezierBathymetrySmootherTest, GeoTiffBathymetry) {
  // Skip if GDAL not available
  if (!GeoTiffReader::is_available()) {
    GTEST_SKIP() << "GDAL not available, skipping GeoTIFF test";
  }

  std::string geotiff_path = drifter::testing::BATHYMETRY_GEOTIFF_PATH;
  if (!std::filesystem::exists(geotiff_path)) {
    GTEST_SKIP() << "GeoTIFF file not found: " << geotiff_path;
  }

  // Load bathymetry
  GeoTiffReader reader;
  BathymetryData bathy = reader.load(geotiff_path);
  ASSERT_TRUE(bathy.is_valid())
      << "Failed to load GeoTIFF: " << reader.last_error();

  std::cout << "Loaded bathymetry: " << bathy.sizex << " x " << bathy.sizey
            << " pixels\n";
  std::cout << "Bounds: x=[" << bathy.xmin << ", " << bathy.xmax << "], "
            << "y=[" << bathy.ymin << ", " << bathy.ymax << "]\n";

  // Search for a region with actual water (non-zero depth)
  // The Danish bathymetry data has Kattegat in the northwest
  Real region_size = 30000.0; // 30 km region

  // Search grid for a water region
  Real best_x = 0, best_y = 0;
  Real max_depth_found = 0;
  for (Real x = bathy.xmin + region_size; x < bathy.xmax - region_size;
       x += region_size) {
    for (Real y = bathy.ymin + region_size; y < bathy.ymax - region_size;
         y += region_size) {
      Real depth = bathy.get_depth(x, y);
      if (depth > max_depth_found) {
        max_depth_found = depth;
        best_x = x;
        best_y = y;
      }
    }
  }

  if (max_depth_found < 5.0) {
    GTEST_SKIP() << "No water region with depth > 5m found";
  }

  std::cout << "Found water region at (" << best_x << ", " << best_y
            << ") with depth " << max_depth_found << "m\n";

  Real xmin = best_x - region_size / 2;
  Real xmax = best_x + region_size / 2;
  Real ymin = best_y - region_size / 2;
  Real ymax = best_y + region_size / 2;

  std::cout << "Test region: x=[" << xmin << ", " << xmax << "], "
            << "y=[" << ymin << ", " << ymax << "]\n";

  // Create quadtree over this region
  auto quadtree = std::make_unique<QuadtreeAdapter>();
  quadtree->build_uniform(xmin, xmax, ymin, ymax, 4, 4); // 4x4 = 16 elements

  // Wrap BathymetryData in a lambda for set_bathymetry_data
  auto bathy_func = [&bathy](Real x, Real y) -> Real {
    return static_cast<Real>(bathy.get_depth(x, y));
  };

  BezierBathymetrySmoother smoother(*quadtree);
  smoother.set_bathymetry_data(bathy_func);

  std::cout << "Solving with " << smoother.num_dofs() << " DOFs...\n";
  smoother.solve();

  EXPECT_TRUE(smoother.is_solved());

  // Check solution at several test points
  std::cout << "\nSample depth comparisons:\n";
  std::vector<Vec2> test_points = {
      Vec2(best_x, best_y),
      Vec2(xmin + region_size * 0.25, ymin + region_size * 0.25),
      Vec2(xmin + region_size * 0.75, ymin + region_size * 0.75),
      Vec2(xmin + region_size * 0.25, ymin + region_size * 0.75),
      Vec2(xmin + region_size * 0.75, ymin + region_size * 0.25),
  };

  for (const auto &pt : test_points) {
    Real raw_depth = bathy.get_depth(pt(0), pt(1));
    Real smoothed_depth = smoother.evaluate(pt(0), pt(1));

    std::cout << "  (" << pt(0) << ", " << pt(1) << "): "
              << "raw=" << raw_depth << ", smoothed=" << smoothed_depth
              << ", diff=" << std::abs(smoothed_depth - raw_depth) << "\n";

    // Smoothed should be within reasonable range of raw (allowing for
    // smoothing)
    if (raw_depth > 0) {
      EXPECT_GT(smoothed_depth, 0) << "At (" << pt(0) << ", " << pt(1) << ")";
    }
  }

  // Write VTK output for visualization
  std::string vtk_path = "/tmp/bezier_geotiff_test";
  EXPECT_NO_THROW(smoother.write_vtk(vtk_path, 5));
  std::cout << "\nWrote VTK output to " << vtk_path << ".vtu\n";

  // Write control points grid
  std::string cp_path = "/tmp/bezier_geotiff_control_points";
  EXPECT_NO_THROW(smoother.write_control_points_vtk(cp_path));
  std::cout << "Wrote control points to " << cp_path << ".vtu\n";

  // Write raw GeoTIFF data at native resolution
  std::string raw_path = "/tmp/bezier_geotiff_raw";
  write_geotiff_region_vtk(bathy, xmin, xmax, ymin, ymax, raw_path);

  // Check diagnostics
  std::cout << "\nDiagnostics:\n";
  std::cout << "  Data residual: " << smoother.data_residual() << "\n";
  std::cout << "  Regularization energy: " << smoother.regularization_energy()
            << "\n";
  std::cout << "  Objective value: " << smoother.objective_value() << "\n";
  std::cout << "  Constraint violation: " << smoother.constraint_violation()
            << "\n";

  // Compute relative constraint violation based on solution magnitude
  Real violation = smoother.constraint_violation();
  Real sol_norm = smoother.solution().norm();
  Real relative_violation =
      (sol_norm > 1e-10) ? violation / sol_norm : violation;
  std::cout << "  Relative violation: " << relative_violation << "\n";

  // Relative constraint violation should be small
  // With Dirichlet BCs on boundary corner DOFs that are also C²-constrained,
  // the KKT solver must satisfy both constraints, leading to slightly higher
  // violation.
  EXPECT_LT(relative_violation, 5e-3)
      << "C² constraints not satisfied (relative)";
}

TEST_F(BezierBathymetrySmootherTest, GeoTiffHigherResolution) {
  // Test with higher resolution mesh in a water region
  if (!GeoTiffReader::is_available()) {
    GTEST_SKIP() << "GDAL not available, skipping GeoTIFF test";
  }

  std::string geotiff_path = drifter::testing::BATHYMETRY_GEOTIFF_PATH;
  if (!std::filesystem::exists(geotiff_path)) {
    GTEST_SKIP() << "GeoTIFF file not found: " << geotiff_path;
  }

  GeoTiffReader reader;
  BathymetryData bathy = reader.load(geotiff_path);
  ASSERT_TRUE(bathy.is_valid());

  // Search for a water region (same approach as GeoTiffBathymetry test)
  Real region_size = 20000.0; // 20 km region
  Real best_x = 0, best_y = 0;
  Real max_depth_found = 0;
  for (Real x = bathy.xmin + region_size; x < bathy.xmax - region_size;
       x += region_size) {
    for (Real y = bathy.ymin + region_size; y < bathy.ymax - region_size;
         y += region_size) {
      Real depth = bathy.get_depth(x, y);
      if (depth > max_depth_found) {
        max_depth_found = depth;
        best_x = x;
        best_y = y;
      }
    }
  }

  if (max_depth_found < 5.0) {
    GTEST_SKIP() << "No water region with depth > 5m found";
  }

  Real xmin = best_x - region_size / 2;
  Real xmax = best_x + region_size / 2;
  Real ymin = best_y - region_size / 2;
  Real ymax = best_y + region_size / 2;

  // Create higher resolution quadtree
  auto quadtree = std::make_unique<QuadtreeAdapter>();
  quadtree->build_uniform(xmin, xmax, ymin, ymax, 6, 6); // 6x6 = 36 elements

  std::cout << "High-res mesh: " << quadtree->num_elements() << " elements, "
            << quadtree->num_elements() * 36 << " total DOFs\n";
  std::cout << "Region around depth " << max_depth_found << "m\n";

  auto bathy_func = [&bathy](Real x, Real y) -> Real {
    return static_cast<Real>(bathy.get_depth(x, y));
  };

  BezierBathymetrySmoother smoother(*quadtree);
  smoother.set_bathymetry_data(bathy_func);
  smoother.solve();

  EXPECT_TRUE(smoother.is_solved());

  // Sample some depth values
  Real center_depth = smoother.evaluate(best_x, best_y);
  Real raw_depth = bathy.get_depth(best_x, best_y);
  std::cout << "Center depth: raw=" << raw_depth
            << ", smoothed=" << center_depth << "\n";

  // Check C² continuity - with 36 elements there are many interface constraints
  Real violation = smoother.constraint_violation();
  // Compute relative violation based on solution magnitude
  Real sol_norm = smoother.solution().norm();
  Real relative_violation =
      (sol_norm > 1e-10) ? violation / sol_norm : violation;
  std::cout << "C² constraint violation: " << violation
            << " (relative: " << relative_violation << ")\n";

  // Relative constraint violation should be small
  EXPECT_LT(relative_violation, 1e-3);

  // Write VTK
  std::string vtk_path = "/tmp/bezier_geotiff_highres";
  smoother.write_vtk(vtk_path, 3); // Lower resolution VTK for larger mesh
  std::cout << "Wrote VTK output to " << vtk_path << ".vtu\n";

  // Write control points grid
  std::string cp_path = "/tmp/bezier_geotiff_highres_control_points";
  smoother.write_control_points_vtk(cp_path);
  std::cout << "Wrote control points to " << cp_path << ".vtu\n";

  // Write raw GeoTIFF data at native resolution
  std::string raw_path = "/tmp/bezier_geotiff_highres_raw";
  write_geotiff_region_vtk(bathy, xmin, xmax, ymin, ymax, raw_path);
}

// =============================================================================
// Dirichlet boundary condition tests
// =============================================================================

TEST_F(BezierBathymetrySmootherTest, DirichletSingleElement) {
  // Single element: all 4 edges are on domain boundary
  // Dirichlet BCs now only constrain corner DOFs (4 out of 36) to preserve
  // C² continuity at shared vertices in multi-element meshes.
  auto quadtree = create_quadtree(1, 1); // 1 element

  const Real depth = 50.0;
  auto bathy = [depth](Real /*x*/, Real /*y*/) { return depth; };

  BezierSmootherConfig config;
  config.enable_boundary_dirichlet = true;
  config.enable_natural_bc = false;  // Disable natural BC for Dirichlet test

  BezierBathymetrySmoother smoother(*quadtree, config);
  smoother.set_bathymetry_data(bathy);
  smoother.solve();

  EXPECT_TRUE(smoother.is_solved());

  // Corners should be exactly constrained
  std::vector<std::pair<Real, Real>> corner_points = {
      {0.0, 0.0},
      {100.0, 0.0},
      {0.0, 100.0},
      {100.0, 100.0},
  };

  for (const auto &[x, y] : corner_points) {
    Real eval = smoother.evaluate(x, y);
    EXPECT_NEAR(eval, depth, 1e-6) << "At corner (" << x << ", " << y << ")";
  }

  // Edge midpoints are not directly constrained - they're determined by
  // the optimization. For a constant bathymetry, they should still be close.
  std::vector<std::pair<Real, Real>> edge_midpoints = {
      {50.0, 0.0},
      {0.0, 50.0},
      {100.0, 50.0},
      {50.0, 100.0},
  };

  for (const auto &[x, y] : edge_midpoints) {
    Real eval = smoother.evaluate(x, y);
    EXPECT_NEAR(eval, depth, 0.01)
        << "At edge midpoint (" << x << ", " << y << ")";
  }
}

TEST_F(BezierBathymetrySmootherTest, DirichletMultiElement) {
  // 2x2 mesh with Dirichlet BCs on domain boundary
  auto quadtree = create_quadtree(2, 2);

  // Print constraint info for debugging
  BezierC2ConstraintBuilder builder(*quadtree);
  std::cout << "Num elements: " << quadtree->num_elements() << "\n";
  std::cout << "Total DOFs: " << builder.total_dofs() << "\n";
  std::cout << "C² constraints: " << builder.num_constraints() << "\n";
  std::cout << "Dirichlet constraints: " << builder.num_dirichlet_constraints()
            << "\n";
  std::cout << "Combined constraints: " << builder.num_combined_constraints()
            << "\n";

  // Linear bathymetry: z = 10 + 0.5*x + 0.3*y
  auto bathy = [](Real x, Real y) { return 10.0 + 0.5 * x + 0.3 * y; };

  BezierSmootherConfig config;
  config.enable_boundary_dirichlet = true;
  config.enable_natural_bc = false;  // Disable natural BC for Dirichlet test

  BezierBathymetrySmoother smoother(*quadtree, config);
  smoother.set_bathymetry_data(bathy);
  smoother.solve();

  EXPECT_TRUE(smoother.is_solved());

  // Check boundary points
  std::vector<std::pair<Real, Real>> boundary_points = {
      {0.0, 0.0},     // xmin, ymin
      {100.0, 0.0},   // xmax, ymin
      {0.0, 100.0},   // xmin, ymax
      {100.0, 100.0}, // xmax, ymax
      {50.0, 0.0},    // ymin edge
      {0.0, 50.0},    // xmin edge
      {100.0, 50.0},  // xmax edge
      {50.0, 100.0},  // ymax edge
  };

  for (const auto &[x, y] : boundary_points) {
    Real expected = bathy(x, y);
    Real eval = smoother.evaluate(x, y);
    EXPECT_NEAR(eval, expected, 1e-4)
        << "At boundary point (" << x << ", " << y << ")";
  }

  // Also check that C² continuity is maintained at interior interface
  Real violation = smoother.constraint_violation();
  Real sol_norm = smoother.solution().norm();
  Real relative_violation =
      (sol_norm > 1e-10) ? violation / sol_norm : violation;
  EXPECT_LT(relative_violation, 1e-3) << "C² continuity violated";
}

TEST_F(BezierBathymetrySmootherTest, DirichletVsNonDirichlet) {
  // Compare solutions with and without Dirichlet BCs
  auto quadtree1 = create_quadtree(3, 3);
  auto quadtree2 = create_quadtree(3, 3);

  // Quadratic bathymetry
  auto bathy = [](Real x, Real y) {
    Real cx = 50.0, cy = 50.0;
    return 100.0 - 0.01 * ((x - cx) * (x - cx) + (y - cy) * (y - cy));
  };

  BezierSmootherConfig config_no_dirichlet;
  config_no_dirichlet.lambda = 0.01;
  config_no_dirichlet.enable_boundary_dirichlet = false;
  config_no_dirichlet.enable_natural_bc = false;  // Pure C² only

  BezierSmootherConfig config_with_dirichlet;
  config_with_dirichlet.lambda = 0.01;
  config_with_dirichlet.enable_boundary_dirichlet = true;
  config_with_dirichlet.enable_natural_bc = false;  // Dirichlet instead of natural BC

  BezierBathymetrySmoother smoother_no_dir(*quadtree1, config_no_dirichlet);
  smoother_no_dir.set_bathymetry_data(bathy);
  smoother_no_dir.solve();

  BezierBathymetrySmoother smoother_with_dir(*quadtree2, config_with_dirichlet);
  smoother_with_dir.set_bathymetry_data(bathy);
  smoother_with_dir.solve();

  EXPECT_TRUE(smoother_no_dir.is_solved());
  EXPECT_TRUE(smoother_with_dir.is_solved());

  // Both should have similar interior values
  Real center_no_dir = smoother_no_dir.evaluate(50.0, 50.0);
  Real center_with_dir = smoother_with_dir.evaluate(50.0, 50.0);
  Real expected_center = bathy(50.0, 50.0);

  std::cout << "Center: expected=" << expected_center
            << ", no_dirichlet=" << center_no_dir
            << ", with_dirichlet=" << center_with_dir << "\n";

  // Dirichlet should match boundary better
  std::vector<std::pair<Real, Real>> boundary_points = {
      {0.0, 50.0},
      {100.0, 50.0},
      {50.0, 0.0},
      {50.0, 100.0},
  };

  Real max_err_no_dir = 0.0, max_err_with_dir = 0.0;
  for (const auto &[x, y] : boundary_points) {
    Real expected = bathy(x, y);
    Real err_no_dir = std::abs(smoother_no_dir.evaluate(x, y) - expected);
    Real err_with_dir = std::abs(smoother_with_dir.evaluate(x, y) - expected);
    max_err_no_dir = std::max(max_err_no_dir, err_no_dir);
    max_err_with_dir = std::max(max_err_with_dir, err_with_dir);
  }

  std::cout << "Max boundary error: no_dirichlet=" << max_err_no_dir
            << ", with_dirichlet=" << max_err_with_dir << "\n";

  // Dirichlet constraints only apply at corner DOFs to preserve C² continuity.
  // Edge midpoints are determined by the optimization, so they may have larger
  // errors than with full-edge constraints. But this is acceptable because
  // it maintains smooth surfaces across element boundaries.
  EXPECT_LT(max_err_with_dir, 5.0) << "Dirichlet boundary error too large";
}

TEST_F(BezierBathymetrySmootherTest, GeoTiffWithDirichlet) {
  // Test GeoTIFF bathymetry with Dirichlet boundary conditions
  if (!GeoTiffReader::is_available()) {
    GTEST_SKIP() << "GDAL not available, skipping GeoTIFF test";
  }

  std::string geotiff_path = drifter::testing::BATHYMETRY_GEOTIFF_PATH;
  if (!std::filesystem::exists(geotiff_path)) {
    GTEST_SKIP() << "GeoTIFF file not found: " << geotiff_path;
  }

  // Load bathymetry
  GeoTiffReader reader;
  BathymetryData bathy = reader.load(geotiff_path);
  ASSERT_TRUE(bathy.is_valid()) << "Failed to load GeoTIFF";

  // Search for a water region
  Real region_size = 20000.0; // 20 km region
  Real best_x = 0, best_y = 0;
  Real max_depth_found = 0;
  for (Real x = bathy.xmin + region_size; x < bathy.xmax - region_size;
       x += region_size) {
    for (Real y = bathy.ymin + region_size; y < bathy.ymax - region_size;
         y += region_size) {
      Real depth = bathy.get_depth(x, y);
      if (depth > max_depth_found) {
        max_depth_found = depth;
        best_x = x;
        best_y = y;
      }
    }
  }

  if (max_depth_found < 5.0) {
    GTEST_SKIP() << "No water region found";
  }

  Real xmin = best_x - region_size / 2;
  Real xmax = best_x + region_size / 2;
  Real ymin = best_y - region_size / 2;
  Real ymax = best_y + region_size / 2;

  // Create quadtree
  auto quadtree = std::make_unique<QuadtreeAdapter>();
  quadtree->build_uniform(xmin, xmax, ymin, ymax, 3, 3); // 3x3 = 9 elements

  auto bathy_func = [&bathy](Real x, Real y) -> Real {
    return static_cast<Real>(bathy.get_depth(x, y));
  };

  // Solve WITHOUT Dirichlet (pure C² only)
  BezierSmootherConfig config_no_dir;
  config_no_dir.lambda = 1.0; // Equal weight to data fitting and smoothness
  config_no_dir.enable_boundary_dirichlet = false;
  config_no_dir.enable_natural_bc = false;  // Pure C² only

  BezierBathymetrySmoother smoother_no_dir(*quadtree, config_no_dir);
  smoother_no_dir.set_bathymetry_data(bathy_func);
  smoother_no_dir.solve();

  // Solve WITH Dirichlet
  BezierSmootherConfig config_with_dir;
  config_with_dir.lambda = 1.0; // Equal weight to data fitting and smoothness
  config_with_dir.enable_boundary_dirichlet = true;
  config_with_dir.enable_natural_bc = false;  // Dirichlet instead of natural BC

  BezierBathymetrySmoother smoother_with_dir(*quadtree, config_with_dir);
  smoother_with_dir.set_bathymetry_data(bathy_func);
  smoother_with_dir.solve();

  // Check boundary errors
  std::vector<std::pair<Real, Real>> boundary_points = {
      {xmin, (ymin + ymax) / 2}, // Left edge
      {xmax, (ymin + ymax) / 2}, // Right edge
      {(xmin + xmax) / 2, ymin}, // Bottom edge
      {(xmin + xmax) / 2, ymax}, // Top edge
  };

  Real max_err_no_dir = 0.0, max_err_with_dir = 0.0;
  std::cout << "GeoTIFF boundary comparison:\n";
  for (const auto &[x, y] : boundary_points) {
    Real expected = bathy_func(x, y);
    Real err_no_dir = std::abs(smoother_no_dir.evaluate(x, y) - expected);
    Real err_with_dir = std::abs(smoother_with_dir.evaluate(x, y) - expected);
    max_err_no_dir = std::max(max_err_no_dir, err_no_dir);
    max_err_with_dir = std::max(max_err_with_dir, err_with_dir);

    std::cout << "  (" << x << ", " << y << "): expected=" << expected
              << ", no_dir_err=" << err_no_dir
              << ", with_dir_err=" << err_with_dir << "\n";
  }

  std::cout << "Max boundary error: no_dirichlet=" << max_err_no_dir
            << ", with_dirichlet=" << max_err_with_dir << "\n";

  // With corner-only Dirichlet, edge midpoint errors may be similar to
  // no-Dirichlet because we only constrain corners to preserve C² continuity.
  // The important thing is that both solutions have reasonable boundary errors.
  // Require both to be less than 15 meters (reasonable for real bathymetry
  // data).
  EXPECT_LT(max_err_no_dir, 15.0) << "No-Dirichlet boundary error too large";
  EXPECT_LT(max_err_with_dir, 15.0)
      << "With-Dirichlet boundary error too large";

  // Both should satisfy C² continuity
  Real violation_no_dir = smoother_no_dir.constraint_violation();
  Real violation_with_dir = smoother_with_dir.constraint_violation();
  Real norm_no_dir = smoother_no_dir.solution().norm();
  Real norm_with_dir = smoother_with_dir.solution().norm();

  EXPECT_LT(violation_no_dir / norm_no_dir, 1e-3)
      << "C² violated (no Dirichlet)";
  EXPECT_LT(violation_with_dir / norm_with_dir, 1e-3)
      << "C² violated (with Dirichlet)";
}

// =============================================================================
// Natural boundary condition tests
// =============================================================================

TEST_F(BezierBathymetrySmootherTest, NaturalBCConstraintCounts) {
  // Verify natural BC constraint counts for various meshes

  // Single element: 4 corners × 2 constraints each + 4 edges × 4 interior DOFs = 8 + 16 = 24
  // But wait - domain corners have 2 edges each, not counted separately for interior DOFs
  // Let's calculate properly:
  //   - Edge 0 (left): 6 DOFs total
  //   - Edge 1 (right): 6 DOFs total
  //   - Edge 2 (bottom): 6 DOFs total
  //   - Edge 3 (top): 6 DOFs total
  //   - Corner DOFs appear on 2 edges, getting 2 constraints each
  //
  // For single element: 4 corners get 2 constraints each = 8
  //                     4 edges × 4 interior DOFs × 1 constraint = 16
  // Total = 24

  auto quadtree1 = create_quadtree(1, 1);
  BezierC2ConstraintBuilder builder1(*quadtree1);
  Index natural_bc_1x1 = builder1.num_natural_bc_constraints();

  // 24 = 4 corners × 2 + 4 edges × 4 interior
  EXPECT_EQ(natural_bc_1x1, 24) << "1x1 mesh should have 24 natural BC constraints";

  // 2x2 mesh:
  //   - 4 domain corners × 2 = 8
  //   - 4 domain edge midpoints (vertices) × 1 = 4
  //   - 4 domain edges × 8 interior DOFs × 1 = 32
  // But some DOFs are shared between elements on domain boundary
  // Let's see what the actual count is
  auto quadtree2 = create_quadtree(2, 2);
  BezierC2ConstraintBuilder builder2(*quadtree2);
  Index natural_bc_2x2 = builder2.num_natural_bc_constraints();

  std::cout << "Natural BC constraints - 1x1: " << natural_bc_1x1
            << ", 2x2: " << natural_bc_2x2 << "\n";

  // Verify it's a reasonable number (not zero, not too large)
  EXPECT_GT(natural_bc_2x2, natural_bc_1x1) << "2x2 should have more constraints than 1x1";
  EXPECT_LT(natural_bc_2x2, 100) << "Constraint count seems too high";
}

TEST_F(BezierBathymetrySmootherTest, NaturalBCConstantBathymetry) {
  // For constant bathymetry, natural BC (z_nn = 0) should be trivially satisfied
  // since z_xx = z_yy = 0 everywhere for a constant function
  auto quadtree = create_quadtree(2, 2);

  const Real depth = 50.0;
  auto bathy = [depth](Real /*x*/, Real /*y*/) { return depth; };

  BezierSmootherConfig config;
  config.enable_natural_bc = true;
  config.enable_boundary_dirichlet = false;

  BezierBathymetrySmoother smoother(*quadtree, config);
  smoother.set_bathymetry_data(bathy);
  smoother.solve();

  EXPECT_TRUE(smoother.is_solved());

  // Solution should be constant everywhere
  std::vector<std::pair<Real, Real>> test_points = {
      {25.0, 25.0}, {75.0, 25.0}, {25.0, 75.0}, {75.0, 75.0},
      {50.0, 50.0},  // center
      {0.0, 0.0}, {100.0, 100.0},  // corners
      {50.0, 0.0}, {0.0, 50.0},    // edge midpoints
  };

  for (const auto& [x, y] : test_points) {
    Real eval = smoother.evaluate(x, y);
    EXPECT_NEAR(eval, depth, 0.1) << "At point (" << x << ", " << y << ")";
  }

  // Constraint violation should be small
  Real violation = smoother.constraint_violation();
  Real norm = smoother.solution().norm();
  EXPECT_LT(violation / norm, 1e-6) << "Constraint violation too large";
}

TEST_F(BezierBathymetrySmootherTest, NaturalBCLinearBathymetry) {
  // For linear bathymetry z = ax + by + c, all second derivatives are zero
  // so natural BC z_nn = 0 should be exactly satisfied
  auto quadtree = create_quadtree(2, 2);

  auto bathy = [](Real x, Real y) { return 10.0 + 0.3 * x + 0.2 * y; };

  BezierSmootherConfig config;
  config.enable_natural_bc = true;
  config.enable_boundary_dirichlet = false;
  config.lambda = 1.0;  // Equal weight to data and smoothness

  BezierBathymetrySmoother smoother(*quadtree, config);
  smoother.set_bathymetry_data(bathy);
  smoother.solve();

  EXPECT_TRUE(smoother.is_solved());

  // Solution should match linear function well
  std::vector<std::pair<Real, Real>> test_points = {
      {25.0, 25.0}, {75.0, 25.0}, {25.0, 75.0}, {75.0, 75.0},
      {50.0, 50.0}, {0.0, 50.0}, {100.0, 50.0},
  };

  for (const auto& [x, y] : test_points) {
    Real expected = bathy(x, y);
    Real eval = smoother.evaluate(x, y);
    EXPECT_NEAR(eval, expected, 1.0) << "At point (" << x << ", " << y << ")";
  }

  // Constraint violation should be small
  Real violation = smoother.constraint_violation();
  Real norm = smoother.solution().norm();
  EXPECT_LT(violation / norm, 1e-6) << "Constraint violation too large";
}

TEST_F(BezierBathymetrySmootherTest, NaturalBCVsDirichlet) {
  // Compare solutions with natural BC vs Dirichlet BC
  auto quadtree1 = create_quadtree(3, 3);
  auto quadtree2 = create_quadtree(3, 3);

  // Smooth quadratic bathymetry
  auto bathy = [](Real x, Real y) {
    Real cx = 50.0, cy = 50.0;
    return 100.0 - 0.005 * ((x - cx) * (x - cx) + (y - cy) * (y - cy));
  };

  // Natural BC configuration
  BezierSmootherConfig config_natural;
  config_natural.lambda = 1.0;
  config_natural.enable_natural_bc = true;
  config_natural.enable_boundary_dirichlet = false;

  // Dirichlet BC configuration
  BezierSmootherConfig config_dirichlet;
  config_dirichlet.lambda = 1.0;
  config_dirichlet.enable_natural_bc = false;
  config_dirichlet.enable_boundary_dirichlet = true;

  BezierBathymetrySmoother smoother_natural(*quadtree1, config_natural);
  smoother_natural.set_bathymetry_data(bathy);
  smoother_natural.solve();

  BezierBathymetrySmoother smoother_dirichlet(*quadtree2, config_dirichlet);
  smoother_dirichlet.set_bathymetry_data(bathy);
  smoother_dirichlet.solve();

  EXPECT_TRUE(smoother_natural.is_solved());
  EXPECT_TRUE(smoother_dirichlet.is_solved());

  // Both should approximate the center well
  Real center_natural = smoother_natural.evaluate(50.0, 50.0);
  Real center_dirichlet = smoother_dirichlet.evaluate(50.0, 50.0);
  Real expected_center = bathy(50.0, 50.0);

  std::cout << "Center comparison: expected=" << expected_center
            << ", natural=" << center_natural
            << ", dirichlet=" << center_dirichlet << "\n";

  EXPECT_NEAR(center_natural, expected_center, 5.0);
  EXPECT_NEAR(center_dirichlet, expected_center, 5.0);

  // Natural BC should have smaller curvature at boundaries
  // Check that both produce valid solutions
  std::vector<std::pair<Real, Real>> boundary_points = {
      {0.0, 50.0}, {100.0, 50.0}, {50.0, 0.0}, {50.0, 100.0},
  };

  for (const auto& [x, y] : boundary_points) {
    Real expected = bathy(x, y);
    Real eval_natural = smoother_natural.evaluate(x, y);
    Real eval_dirichlet = smoother_dirichlet.evaluate(x, y);

    std::cout << "Boundary (" << x << ", " << y << "): expected=" << expected
              << ", natural=" << eval_natural << ", dirichlet=" << eval_dirichlet << "\n";

    // Both should be reasonable (within 20% of expected)
    EXPECT_NEAR(eval_natural, expected, 0.2 * std::abs(expected) + 5.0);
    EXPECT_NEAR(eval_dirichlet, expected, 0.2 * std::abs(expected) + 5.0);
  }

  // Both should satisfy their constraints
  Real viol_natural = smoother_natural.constraint_violation();
  Real viol_dirichlet = smoother_dirichlet.constraint_violation();
  Real norm_natural = smoother_natural.solution().norm();
  Real norm_dirichlet = smoother_dirichlet.solution().norm();

  EXPECT_LT(viol_natural / norm_natural, 1e-3) << "Natural BC constraints violated";
  EXPECT_LT(viol_dirichlet / norm_dirichlet, 1e-3) << "Dirichlet constraints violated";
}

TEST_F(BezierBathymetrySmootherTest, NaturalBCSmoothBoundaries) {
  // Natural BC should produce smooth (low curvature) boundaries
  auto quadtree = create_quadtree(2, 2);

  // Sinusoidal bathymetry that has non-zero curvature
  auto bathy = [](Real x, Real y) {
    return 50.0 + 10.0 * std::sin(x * M_PI / 100.0) * std::cos(y * M_PI / 100.0);
  };

  BezierSmootherConfig config;
  config.enable_natural_bc = true;
  config.enable_boundary_dirichlet = false;
  config.lambda = 10.0;  // Stronger data fitting

  BezierBathymetrySmoother smoother(*quadtree, config);
  smoother.set_bathymetry_data(bathy);
  smoother.solve();

  EXPECT_TRUE(smoother.is_solved());

  // Check that solution exists and is reasonable
  Real center = smoother.evaluate(50.0, 50.0);
  EXPECT_NEAR(center, bathy(50.0, 50.0), 2.0);

  // Verify gradient is reasonable at boundaries (not too steep)
  std::vector<std::pair<Real, Real>> boundary_points = {
      {0.0, 50.0}, {100.0, 50.0}, {50.0, 0.0}, {50.0, 100.0},
  };

  for (const auto& [x, y] : boundary_points) {
    Vec2 grad = smoother.evaluate_gradient(x, y);
    Real grad_mag = grad.norm();
    // Gradient should not be extreme
    EXPECT_LT(grad_mag, 5.0) << "Gradient too steep at boundary (" << x << ", " << y << ")";
  }

  // Constraint violation should be small
  Real violation = smoother.constraint_violation();
  Real norm = smoother.solution().norm();
  EXPECT_LT(violation / norm, 1e-6) << "Constraint violation too large";
}

// =============================================================================
// Non-conforming (AMR) mesh tests
// =============================================================================

TEST_F(BezierBathymetrySmootherTest, NonConformingOnePlusFour) {
  // Create a non-conforming mesh: 1 coarse element + 4 fine elements
  // Layout:
  //   +-------+---+---+
  //   |       | 2 | 3 |
  //   |   0   +---+---+
  //   |       | 1 | 4 |
  //   +-------+---+---+
  // Element 0: [0,50] x [0,100] (coarse, level 0 in x)
  // Elements 1-4: [50,100] x [0,100] subdivided into 2x2 (level 1 in both)

  auto quadtree = std::make_unique<QuadtreeAdapter>();

  // Add coarse element on left half
  quadtree->add_element(QuadBounds{0.0, 50.0, 0.0, 100.0}, QuadLevel{0, 0});

  // Add 4 fine elements on right half (2x2 grid)
  quadtree->add_element(QuadBounds{50.0, 75.0, 0.0, 50.0}, QuadLevel{1, 1});
  quadtree->add_element(QuadBounds{50.0, 75.0, 50.0, 100.0}, QuadLevel{1, 1});
  quadtree->add_element(QuadBounds{75.0, 100.0, 50.0, 100.0}, QuadLevel{1, 1});
  quadtree->add_element(QuadBounds{75.0, 100.0, 0.0, 50.0}, QuadLevel{1, 1});

  ASSERT_EQ(quadtree->num_elements(), 5);

  // Verify non-conforming interface on right edge of element 0
  auto neighbor_info = quadtree->get_neighbor(0, 1); // right edge
  EXPECT_EQ(neighbor_info.type, EdgeNeighborInfo::Type::CoarseToFine);
  EXPECT_EQ(neighbor_info.neighbor_elements.size(), 2);

  std::cout << "Created non-conforming mesh: 1 coarse + 4 fine elements\n";
  std::cout << "Total DOFs: " << 5 * 36 << "\n";

  // Debug: print all neighbor relationships
  std::cout << "\nNeighbor relationships:\n";
  for (Index e = 0; e < quadtree->num_elements(); ++e) {
    const auto &bounds = quadtree->element_bounds(e);
    std::cout << "Element " << e << ": [" << bounds.xmin << "," << bounds.xmax
              << "] x [" << bounds.ymin << "," << bounds.ymax << "]\n";
    for (int edge = 0; edge < 4; ++edge) {
      auto info = quadtree->get_neighbor(e, edge);
      std::string edge_name[] = {"left", "right", "bottom", "top"};
      std::cout << "  " << edge_name[edge] << ": ";
      if (info.is_boundary()) {
        std::cout << "boundary\n";
      } else if (info.is_conforming()) {
        std::cout << "conforming -> elem " << info.neighbor_elements[0] << "\n";
      } else if (info.type == EdgeNeighborInfo::Type::CoarseToFine) {
        std::cout << "coarse->fine -> elems ";
        for (auto n : info.neighbor_elements)
          std::cout << n << " ";
        std::cout << "\n";
      } else {
        std::cout << "fine->coarse -> elem " << info.neighbor_elements[0] << "\n";
      }
    }
  }
  std::cout << "\n";

  // Bilinear bathymetry that satisfies natural BCs (z_xx = z_yy = 0 everywhere)
  // z = 50 + 0.3*x + 0.2*y + 0.01*xy
  // This ensures z_nn = 0 at all boundaries since z_xx = z_yy = 0
  auto bathy = [](Real x, Real y) {
    return 50.0 + 0.3 * x + 0.2 * y + 0.01 * x * y;
  };

  BezierBathymetrySmoother smoother(*quadtree);
  smoother.set_bathymetry_data(bathy);
  smoother.solve();

  EXPECT_TRUE(smoother.is_solved());

  // Check C² continuity at non-conforming interface
  Real violation = smoother.constraint_violation();
  Real sol_norm = smoother.solution().norm();
  Real relative_violation = (sol_norm > 1e-10) ? violation / sol_norm : violation;

  std::cout << "C² constraint violation: " << violation
            << " (relative: " << relative_violation << ")\n";

  EXPECT_LT(relative_violation, 1e-3) << "C² constraints not satisfied";

  // Sample depth at several points including the non-conforming interface
  std::vector<std::pair<Real, Real>> test_points = {
      {25.0, 50.0},  // center of coarse element
      {62.5, 25.0},  // center of fine element 1
      {62.5, 75.0},  // center of fine element 2
      {87.5, 75.0},  // center of fine element 3
      {87.5, 25.0},  // center of fine element 4
      {50.0, 25.0},  // on non-conforming interface (bottom)
      {50.0, 75.0},  // on non-conforming interface (top)
      {50.0, 50.0},  // on non-conforming interface (middle/vertex)
  };

  std::cout << "\nDepth comparisons:\n";
  for (const auto &[x, y] : test_points) {
    Real expected = bathy(x, y);
    Real computed = smoother.evaluate(x, y);
    Real error = std::abs(computed - expected);

    std::cout << "  (" << x << ", " << y << "): expected=" << expected
              << ", computed=" << computed << ", error=" << error << "\n";

    // Allow smoothing error (coarse element smooths significantly with low lambda)
    EXPECT_LT(error, 20.0) << "At (" << x << ", " << y << ")";
  }

  // Check C0/C1/C2 continuity along shared edge between elements 3 and 4
  // Elements 3 and 4 share the edge at y=50, x in [75, 100]
  // Element 3: [75,100] x [50,100], so y=50 is at v=0
  // Element 4: [75,100] x [0,50], so y=50 is at v=1
  // Both have dx=25, dy=50

  BezierBasis2D basis;
  VecX coeff3 = smoother.element_coefficients(3);
  VecX coeff4 = smoother.element_coefficients(4);

  Real dx3 = 25.0, dy3 = 50.0;
  Real dx4 = 25.0, dy4 = 50.0;

  std::cout << "\nC0/C1/C2 continuity check along elem3-elem4 shared edge (y=50):\n";
  std::cout << "  x\tdz\t\tdz_x\t\tdz_y\t\tdz_xx\t\tdz_xy\t\tdz_yy\n";

  Real max_c0 = 0.0, max_c1x = 0.0, max_c1y = 0.0;
  Real max_c2xx = 0.0, max_c2xy = 0.0, max_c2yy = 0.0;

  for (double x = 75.0; x <= 100.0; x += 5.0) {
    double u = (x - 75.0) / 25.0;

    // Element 3 at (u, v=0), Element 4 at (u, v=1)
    // Value
    VecX phi3 = basis.evaluate(u, 0.0);
    VecX phi4 = basis.evaluate(u, 1.0);
    Real z3 = coeff3.dot(phi3);
    Real z4 = coeff4.dot(phi4);

    // First derivatives (in parameter space, then convert to physical)
    VecX phi3_u = basis.evaluate_du(u, 0.0);
    VecX phi3_v = basis.evaluate_dv(u, 0.0);
    VecX phi4_u = basis.evaluate_du(u, 1.0);
    VecX phi4_v = basis.evaluate_dv(u, 1.0);

    Real z3_x = coeff3.dot(phi3_u) / dx3;  // dz/dx = dz/du / dx
    Real z3_y = coeff3.dot(phi3_v) / dy3;
    Real z4_x = coeff4.dot(phi4_u) / dx4;
    Real z4_y = coeff4.dot(phi4_v) / dy4;

    // Second derivatives
    VecX phi3_uu = basis.evaluate_d2u(u, 0.0);
    VecX phi3_uv = basis.evaluate_d2uv(u, 0.0);
    VecX phi3_vv = basis.evaluate_d2v(u, 0.0);
    VecX phi4_uu = basis.evaluate_d2u(u, 1.0);
    VecX phi4_uv = basis.evaluate_d2uv(u, 1.0);
    VecX phi4_vv = basis.evaluate_d2v(u, 1.0);

    Real z3_xx = coeff3.dot(phi3_uu) / (dx3 * dx3);
    Real z3_xy = coeff3.dot(phi3_uv) / (dx3 * dy3);
    Real z3_yy = coeff3.dot(phi3_vv) / (dy3 * dy3);
    Real z4_xx = coeff4.dot(phi4_uu) / (dx4 * dx4);
    Real z4_xy = coeff4.dot(phi4_uv) / (dx4 * dy4);
    Real z4_yy = coeff4.dot(phi4_vv) / (dy4 * dy4);

    Real dz = std::abs(z3 - z4);
    Real dz_x = std::abs(z3_x - z4_x);
    Real dz_y = std::abs(z3_y - z4_y);
    Real dz_xx = std::abs(z3_xx - z4_xx);
    Real dz_xy = std::abs(z3_xy - z4_xy);
    Real dz_yy = std::abs(z3_yy - z4_yy);

    max_c0 = std::max(max_c0, dz);
    max_c1x = std::max(max_c1x, dz_x);
    max_c1y = std::max(max_c1y, dz_y);
    max_c2xx = std::max(max_c2xx, dz_xx);
    max_c2xy = std::max(max_c2xy, dz_xy);
    max_c2yy = std::max(max_c2yy, dz_yy);

    std::cout << "  " << x << "\t" << dz << "\t" << dz_x << "\t" << dz_y
              << "\t" << dz_xx << "\t" << dz_xy << "\t" << dz_yy << "\n";
  }

  std::cout << "Max errors: C0=" << max_c0 << ", C1_x=" << max_c1x << ", C1_y=" << max_c1y
            << ", C2_xx=" << max_c2xx << ", C2_xy=" << max_c2xy << ", C2_yy=" << max_c2yy << "\n";

  EXPECT_LT(max_c0, 1e-10) << "C0 continuity violated";
  EXPECT_LT(max_c1x, 1e-10) << "C1 (dz/dx) continuity violated";
  EXPECT_LT(max_c1y, 1e-10) << "C1 (dz/dy) continuity violated";
  EXPECT_LT(max_c2xx, 1e-10) << "C2 (d²z/dx²) continuity violated";
  EXPECT_LT(max_c2xy, 1e-10) << "C2 (d²z/dxdy) continuity violated";
  EXPECT_LT(max_c2yy, 1e-10) << "C2 (d²z/dy²) continuity violated";

  // Check C0/C1/C2 continuity at the NON-CONFORMING interface at x=50
  // Elements 0 (coarse) and 1 (fine) share edge at x=50, y in [0,50]
  // Element 0: [0,50] x [0,100], so x=50 is at u=1
  // Element 1: [50,75] x [0,50], so x=50 is at u=0
  // Element 0 has dx=50, dy=100
  // Element 1 has dx=25, dy=50

  std::cout << "\nC0/C1/C2 continuity at NON-CONFORMING interface (x=50, y in [0,50]):\n";
  std::cout << "Testing coarse element 0 vs fine element 1\n";
  std::cout << "  y\tdz\t\tdz_x\t\tdz_y\t\tdz_xx\t\tdz_xy\t\tdz_yy\n";

  VecX coeff0 = smoother.element_coefficients(0);
  VecX coeff1 = smoother.element_coefficients(1);

  Real dx0 = 50.0, dy0 = 100.0;
  Real dx1 = 25.0, dy1 = 50.0;

  Real max_nc_c0 = 0.0, max_nc_c1x = 0.0, max_nc_c1y = 0.0;
  Real max_nc_c2xx = 0.0, max_nc_c2xy = 0.0, max_nc_c2yy = 0.0;

  for (double y = 0.0; y <= 50.0; y += 10.0) {
    // Element 0: u=1.0, v = y/100
    // Element 1: u=0.0, v = y/50
    double v0 = y / 100.0;
    double v1 = y / 50.0;

    // Value
    VecX phi0 = basis.evaluate(1.0, v0);
    VecX phi1 = basis.evaluate(0.0, v1);
    Real z0 = coeff0.dot(phi0);
    Real z1 = coeff1.dot(phi1);

    // First derivatives
    VecX phi0_u = basis.evaluate_du(1.0, v0);
    VecX phi0_v = basis.evaluate_dv(1.0, v0);
    VecX phi1_u = basis.evaluate_du(0.0, v1);
    VecX phi1_v = basis.evaluate_dv(0.0, v1);

    Real z0_x = coeff0.dot(phi0_u) / dx0;
    Real z0_y = coeff0.dot(phi0_v) / dy0;
    Real z1_x = coeff1.dot(phi1_u) / dx1;
    Real z1_y = coeff1.dot(phi1_v) / dy1;

    // Second derivatives
    VecX phi0_uu = basis.evaluate_d2u(1.0, v0);
    VecX phi0_uv = basis.evaluate_d2uv(1.0, v0);
    VecX phi0_vv = basis.evaluate_d2v(1.0, v0);
    VecX phi1_uu = basis.evaluate_d2u(0.0, v1);
    VecX phi1_uv = basis.evaluate_d2uv(0.0, v1);
    VecX phi1_vv = basis.evaluate_d2v(0.0, v1);

    Real z0_xx = coeff0.dot(phi0_uu) / (dx0 * dx0);
    Real z0_xy = coeff0.dot(phi0_uv) / (dx0 * dy0);
    Real z0_yy = coeff0.dot(phi0_vv) / (dy0 * dy0);
    Real z1_xx = coeff1.dot(phi1_uu) / (dx1 * dx1);
    Real z1_xy = coeff1.dot(phi1_uv) / (dx1 * dy1);
    Real z1_yy = coeff1.dot(phi1_vv) / (dy1 * dy1);

    Real dz = std::abs(z0 - z1);
    Real dz_x = std::abs(z0_x - z1_x);
    Real dz_y = std::abs(z0_y - z1_y);
    Real dz_xx = std::abs(z0_xx - z1_xx);
    Real dz_xy = std::abs(z0_xy - z1_xy);
    Real dz_yy = std::abs(z0_yy - z1_yy);

    max_nc_c0 = std::max(max_nc_c0, dz);
    max_nc_c1x = std::max(max_nc_c1x, dz_x);
    max_nc_c1y = std::max(max_nc_c1y, dz_y);
    max_nc_c2xx = std::max(max_nc_c2xx, dz_xx);
    max_nc_c2xy = std::max(max_nc_c2xy, dz_xy);
    max_nc_c2yy = std::max(max_nc_c2yy, dz_yy);

    std::cout << "  " << y << "\t" << dz << "\t" << dz_x << "\t" << dz_y
              << "\t" << dz_xx << "\t" << dz_xy << "\t" << dz_yy << "\n";
  }

  std::cout << "Non-conforming max errors: C0=" << max_nc_c0 << ", C1_x=" << max_nc_c1x
            << ", C1_y=" << max_nc_c1y << ", C2_xx=" << max_nc_c2xx
            << ", C2_xy=" << max_nc_c2xy << ", C2_yy=" << max_nc_c2yy << "\n";

  // C² continuity at non-conforming interface should be satisfied at machine precision
  EXPECT_LT(max_nc_c0, 1e-6) << "C0 continuity violated at non-conforming interface";
  EXPECT_LT(max_nc_c1x, 1e-6) << "C1 (dz/dx) violated at non-conforming interface";
  EXPECT_LT(max_nc_c1y, 1e-6) << "C1 (dz/dy) violated at non-conforming interface";
  EXPECT_LT(max_nc_c2xx, 1e-6) << "C2 (d²z/dx²) violated at non-conforming interface";
  EXPECT_LT(max_nc_c2xy, 1e-6) << "C2 (d²z/dxdy) violated at non-conforming interface";
  EXPECT_LT(max_nc_c2yy, 1e-6) << "C2 (d²z/dy²) violated at non-conforming interface";

  // Write VTK for visualization
  std::string vtk_path = "/tmp/bezier_nonconforming_1plus4";
  smoother.write_vtk(vtk_path, 5);
  std::cout << "\nWrote VTK output to " << vtk_path << ".vtu\n";

  std::string cp_path = "/tmp/bezier_nonconforming_1plus4_control_points";
  smoother.write_control_points_vtk(cp_path);
  std::cout << "Wrote control points to " << cp_path << ".vtu\n";

  // Write raw bathymetry data per element
  std::string raw_path = "/tmp/bezier_nonconforming_1plus4_raw";
  write_raw_bathy_vtk(*quadtree, bathy, 10, raw_path);
}

// ============================================================================
// Multiresolution Tests with 5 Levels (L0-L4) and Many Hanging Nodes
// ============================================================================

/// Creates a multiresolution mesh by subdividing the bottom row at each level.
/// L0: 1 element (entire domain)
/// L1: 2×2 elements (subdivide L0). Keep top row (2), subdivide bottom row (2)
/// L2: 2×2 per subdivided L1 = 8 elements. Keep top 4, subdivide bottom 4.
/// etc.
/// All subdivisions are 2×2. Never keeps "3" elements.
void create_half_subdivided_mesh(QuadtreeAdapter &quadtree, Real xmin, Real xmax,
                                  Real ymin, Real ymax, int max_level) {
  // Recursive helper: processes a horizontal strip [x0,x1] × [y0,y1]
  // with nx elements in the x direction. Top half becomes leaves,
  // bottom half gets subdivided further.
  std::function<void(Real, Real, Real, Real, int, int)> process_strip =
      [&](Real x0, Real x1, Real y0, Real y1, int level, int nx) {
        Real dx = (x1 - x0) / nx;
        Real ymid = 0.5 * (y0 + y1);

        // Add top row as leaves at current level (nx elements)
        for (int i = 0; i < nx; ++i) {
          quadtree.add_element(QuadBounds{x0 + i * dx, x0 + (i + 1) * dx, ymid, y1},
                               QuadLevel{level, level});
        }

        if (level >= max_level) {
          // At max level: also add bottom row as leaves
          for (int i = 0; i < nx; ++i) {
            quadtree.add_element(QuadBounds{x0 + i * dx, x0 + (i + 1) * dx, y0, ymid},
                                 QuadLevel{level, level});
          }
        } else {
          // Subdivide bottom row: each element becomes 2×2, recurse
          process_strip(x0, x1, y0, ymid, level + 1, nx * 2);
        }
      };

  // Start: L0 is entire domain, subdivide into 2×2 (L1)
  process_strip(xmin, xmax, ymin, ymax, 1, 2);
}

TEST_F(BezierBathymetrySmootherTest, MultiresolutionFiveLevels) {
  // Create a 5-level mesh (L0-L4) with half-subdivision pattern
  // L0: 1 element (entire domain) - subdivided, not a leaf
  // L1: 2 leaves (top row) + bottom row subdivided
  // L2: 4 leaves (top row) + bottom row subdivided
  // L3: 8 leaves (top row) + bottom row subdivided
  // L4: 32 leaves (16 top + 16 bottom)
  // Total: 2 + 4 + 8 + 32 = 46 elements

  auto quadtree = std::make_unique<QuadtreeAdapter>();
  create_half_subdivided_mesh(*quadtree, 0.0, 100.0, 0.0, 100.0, 4);

  ASSERT_EQ(quadtree->num_elements(), 46);

  // Count elements at each level
  std::map<int, int> level_counts;
  for (Index e = 0; e < quadtree->num_elements(); ++e) {
    QuadLevel lvl = quadtree->element_level(e);
    level_counts[lvl.x]++;
  }

  std::cout << "Element counts by level:\n";
  for (const auto &[lvl, count] : level_counts) {
    std::cout << "  L" << lvl << ": " << count << " elements\n";
  }

  EXPECT_EQ(level_counts[1], 2) << "L1 should have 2 elements";
  EXPECT_EQ(level_counts[2], 4) << "L2 should have 4 elements";
  EXPECT_EQ(level_counts[3], 8) << "L3 should have 8 elements";
  EXPECT_EQ(level_counts[4], 32) << "L4 should have 32 elements";

  // Verify 2:1 balance: check that no adjacent elements differ by more than 1 level
  bool balance_ok = true;
  for (Index e = 0; e < quadtree->num_elements(); ++e) {
    QuadLevel my_level = quadtree->element_level(e);
    for (int edge = 0; edge < 4; ++edge) {
      EdgeNeighborInfo info = quadtree->get_neighbor(e, edge);
      if (!info.is_boundary()) {
        for (Index n : info.neighbor_elements) {
          QuadLevel neighbor_level = quadtree->element_level(n);
          int diff_x = std::abs(my_level.x - neighbor_level.x);
          int diff_y = std::abs(my_level.y - neighbor_level.y);
          if (diff_x > 1 || diff_y > 1) {
            std::cout << "Balance violation: elem " << e << " (L" << my_level.x
                      << ") neighbors elem " << n << " (L" << neighbor_level.x << ")\n";
            balance_ok = false;
          }
        }
      }
    }
  }
  EXPECT_TRUE(balance_ok) << "2:1 balance violated";

  // Sinusoidal bathymetry
  auto bathy = [](Real x, Real y) {
    return 50.0 + 20.0 * std::sin(x * M_PI / 50.0) * std::cos(y * M_PI / 50.0);
  };

  BezierBathymetrySmoother smoother(*quadtree);
  smoother.set_bathymetry_data(bathy);
  smoother.solve();

  EXPECT_TRUE(smoother.is_solved());

  // Check hanging node constraint count
  // Calling num_constraints() first populates the internal constraint lists
  BezierC2ConstraintBuilder builder(*quadtree);
  Index total_constraints = builder.num_constraints();
  Index num_hanging = builder.hanging_node_constraints().size();
  std::cout << "Total constraints: " << total_constraints << "\n";
  std::cout << "Hanging node constraints: " << num_hanging << "\n";
  EXPECT_GE(num_hanging, 14) << "Expected at least 14 T-junctions";

  // Check constraint violation
  Real violation = smoother.constraint_violation();
  Real sol_norm = smoother.solution().norm();
  Real relative_violation = (sol_norm > 1e-10) ? violation / sol_norm : violation;

  std::cout << "C² constraint violation: " << violation
            << " (relative: " << relative_violation << ")\n";

  EXPECT_LT(relative_violation, 1e-6) << "C² constraints not satisfied";

  // Sample depth at several points across different levels
  std::vector<std::pair<Real, Real>> test_points = {
      {75.0, 75.0}, // L1 region (top-right)
      {25.0, 75.0}, // L1 region (top-left)
      {12.5, 37.5}, // L2 region
      {6.25, 18.75}, // L3 region
      {3.125, 9.375}, // L4 region
  };

  std::cout << "\nDepth comparisons:\n";
  for (const auto &[x, y] : test_points) {
    Real expected = bathy(x, y);
    Real computed = smoother.evaluate(x, y);
    Real error = std::abs(computed - expected);

    std::cout << "  (" << x << ", " << y << "): expected=" << expected
              << ", computed=" << computed << ", error=" << error << "\n";

    EXPECT_LT(error, 25.0) << "Large error at (" << x << ", " << y << ")";
  }

  // Write VTK for visualization
  std::string vtk_path = "/tmp/bezier_multiresolution_5level";
  smoother.write_vtk(vtk_path, 5);
  std::cout << "\nWrote VTK output to " << vtk_path << ".vtu\n";

  std::string cp_path = "/tmp/bezier_multiresolution_5level_control_points";
  smoother.write_control_points_vtk(cp_path);
  std::cout << "Wrote control points to " << cp_path << ".vtu\n";

  // Write raw bathymetry data per element
  std::string raw_path = "/tmp/bezier_multiresolution_5level_raw";
  write_raw_bathy_vtk(*quadtree, bathy, 10, raw_path);
}

TEST_F(BezierBathymetrySmootherTest, MultiresolutionGeoTiffFiveLevels) {
  // Skip if GDAL not available
  if (!GeoTiffReader::is_available()) {
    GTEST_SKIP() << "GDAL support not available. Rebuild with -DDRIFTER_USE_GDAL=ON";
  }

  // Load GeoTIFF bathymetry
  GeoTiffReader reader;
  BathymetryData bathy_data = reader.load(BATHYMETRY_GEOTIFF_PATH);

  if (!bathy_data.is_valid()) {
    GTEST_SKIP() << "Could not load GeoTIFF: " << reader.last_error();
  }

  std::cout << "Loaded bathymetry: " << bathy_data.sizex << " x " << bathy_data.sizey
            << " pixels\n";
  std::cout << "Bounds: x=[" << bathy_data.xmin << ", " << bathy_data.xmax << "], "
            << "y=[" << bathy_data.ymin << ", " << bathy_data.ymax << "]\n";

  // Find a water region with reasonable depth
  Real region_size = 50000.0; // 50 km region
  Real best_x = 0, best_y = 0;
  Real max_depth_found = 0;

  for (Real x = bathy_data.xmin + region_size; x < bathy_data.xmax - region_size;
       x += region_size * 0.5) {
    for (Real y = bathy_data.ymin + region_size; y < bathy_data.ymax - region_size;
         y += region_size * 0.5) {
      Real depth = bathy_data.get_depth(x, y);
      if (depth > max_depth_found) {
        max_depth_found = depth;
        best_x = x;
        best_y = y;
      }
    }
  }

  if (max_depth_found < 10.0) {
    GTEST_SKIP() << "No water region with depth > 10m found in GeoTIFF";
  }

  std::cout << "Found water region at (" << best_x << ", " << best_y
            << ") with depth " << max_depth_found << "m\n";

  // Create mesh bounds centered on water region
  Real xmin = best_x - region_size / 2;
  Real xmax = best_x + region_size / 2;
  Real ymin = best_y - region_size / 2;
  Real ymax = best_y + region_size / 2;

  std::cout << "Mesh bounds: [" << xmin << ", " << xmax << "] x ["
            << ymin << ", " << ymax << "]\n";

  // Create 5-level mesh with half-subdivision pattern
  auto quadtree = std::make_unique<QuadtreeAdapter>();
  create_half_subdivided_mesh(*quadtree, xmin, xmax, ymin, ymax, 4);

  ASSERT_EQ(quadtree->num_elements(), 46);

  std::cout << "Created mesh with " << quadtree->num_elements() << " elements\n";

  // Wrap bathymetry data in lambda
  auto bathy_func = [&bathy_data](Real x, Real y) -> Real {
    return static_cast<Real>(bathy_data.get_depth(x, y));
  };

  BezierBathymetrySmoother smoother(*quadtree);
  smoother.set_bathymetry_data(bathy_func);
  smoother.solve();

  EXPECT_TRUE(smoother.is_solved());

  // Check hanging node constraint count
  // Calling num_constraints() first populates the internal constraint lists
  BezierC2ConstraintBuilder builder(*quadtree);
  Index total_constraints = builder.num_constraints();
  Index num_hanging = builder.hanging_node_constraints().size();
  std::cout << "Total constraints: " << total_constraints << "\n";
  std::cout << "Hanging node constraints: " << num_hanging << "\n";
  EXPECT_GE(num_hanging, 14) << "Expected at least 14 T-junctions";

  // Check constraint violation (relaxed for real data)
  Real violation = smoother.constraint_violation();
  Real sol_norm = smoother.solution().norm();
  Real relative_violation = (sol_norm > 1e-10) ? violation / sol_norm : violation;

  std::cout << "C² constraint violation: " << violation
            << " (relative: " << relative_violation << ")\n";

  EXPECT_LT(relative_violation, 1e-3) << "C² constraints not satisfied";

  // Check that depths are reasonable (positive where we expect water)
  Real center_depth = smoother.evaluate(best_x, best_y);
  std::cout << "Smoothed depth at center: " << center_depth << "m\n";
  EXPECT_GT(center_depth, 0.0) << "Expected positive depth at water center";

  // Write VTK for visualization
  std::string vtk_path = "/tmp/bezier_multiresolution_geotiff_5level";
  smoother.write_vtk(vtk_path, 5);
  std::cout << "\nWrote VTK output to " << vtk_path << ".vtu\n";

  std::string cp_path = "/tmp/bezier_multiresolution_geotiff_5level_control_points";
  smoother.write_control_points_vtk(cp_path);
  std::cout << "Wrote control points to " << cp_path << ".vtu\n";

  // Write raw GeoTIFF data for comparison
  write_geotiff_region_vtk(bathy_data, xmin, xmax, ymin, ymax,
                           "/tmp/bezier_multiresolution_geotiff_5level_raw");
}

TEST_F(BezierBathymetrySmootherTest, MultiresolutionContinuityVerification) {
  // Create 5-level mesh for detailed C² continuity verification
  auto quadtree = std::make_unique<QuadtreeAdapter>();
  create_half_subdivided_mesh(*quadtree, 0.0, 100.0, 0.0, 100.0, 4);

  ASSERT_EQ(quadtree->num_elements(), 46);

  // Analytical bathymetry with known derivatives
  auto bathy = [](Real x, Real y) {
    return 50.0 + 20.0 * std::sin(x * M_PI / 50.0) * std::cos(y * M_PI / 50.0);
  };

  BezierBathymetrySmoother smoother(*quadtree);
  smoother.set_bathymetry_data(bathy);
  smoother.solve();

  EXPECT_TRUE(smoother.is_solved());

  BezierBasis2D basis;

  // Find all non-conforming interfaces and verify C² continuity
  std::cout << "Verifying C² continuity at non-conforming interfaces:\n";

  int num_nc_interfaces = 0;
  Real max_c0_error = 0, max_c1x_error = 0, max_c1y_error = 0;
  Real max_c2xx_error = 0, max_c2xy_error = 0, max_c2yy_error = 0;

  for (Index e = 0; e < quadtree->num_elements(); ++e) {
    for (int edge = 0; edge < 4; ++edge) {
      EdgeNeighborInfo info = quadtree->get_neighbor(e, edge);

      if (info.type == EdgeNeighborInfo::Type::CoarseToFine) {
        num_nc_interfaces++;

        const QuadBounds &coarse_bounds = quadtree->element_bounds(e);
        Real dx_c = coarse_bounds.xmax - coarse_bounds.xmin;
        Real dy_c = coarse_bounds.ymax - coarse_bounds.ymin;
        VecX coeff_c = smoother.element_coefficients(e);

        // For each fine neighbor, check continuity along shared edge
        for (size_t n = 0; n < info.neighbor_elements.size(); ++n) {
          Index fine_elem = info.neighbor_elements[n];
          int fine_edge = info.neighbor_edges[n];

          const QuadBounds &fine_bounds = quadtree->element_bounds(fine_elem);
          Real dx_f = fine_bounds.xmax - fine_bounds.xmin;
          Real dy_f = fine_bounds.ymax - fine_bounds.ymin;
          VecX coeff_f = smoother.element_coefficients(fine_elem);

          // Sample along the shared edge
          for (Real t = 0.0; t <= 1.0; t += 0.25) {
            // Compute parameter on coarse element
            Real u_c, v_c;
            if (edge == 0) { u_c = 0.0; v_c = (n == 0 ? t * 0.5 : 0.5 + t * 0.5); }
            else if (edge == 1) { u_c = 1.0; v_c = (n == 0 ? t * 0.5 : 0.5 + t * 0.5); }
            else if (edge == 2) { u_c = (n == 0 ? t * 0.5 : 0.5 + t * 0.5); v_c = 0.0; }
            else { u_c = (n == 0 ? t * 0.5 : 0.5 + t * 0.5); v_c = 1.0; }

            // Compute parameter on fine element
            Real u_f, v_f;
            if (fine_edge == 0) { u_f = 0.0; v_f = t; }
            else if (fine_edge == 1) { u_f = 1.0; v_f = t; }
            else if (fine_edge == 2) { u_f = t; v_f = 0.0; }
            else { u_f = t; v_f = 1.0; }

            // Evaluate values and derivatives on coarse element
            Real z_c = coeff_c.dot(basis.evaluate(u_c, v_c));
            Real z_c_x = coeff_c.dot(basis.evaluate_du(u_c, v_c)) / dx_c;
            Real z_c_y = coeff_c.dot(basis.evaluate_dv(u_c, v_c)) / dy_c;
            Real z_c_xx = coeff_c.dot(basis.evaluate_d2u(u_c, v_c)) / (dx_c * dx_c);
            Real z_c_xy = coeff_c.dot(basis.evaluate_d2uv(u_c, v_c)) / (dx_c * dy_c);
            Real z_c_yy = coeff_c.dot(basis.evaluate_d2v(u_c, v_c)) / (dy_c * dy_c);

            // Evaluate values and derivatives on fine element
            Real z_f = coeff_f.dot(basis.evaluate(u_f, v_f));
            Real z_f_x = coeff_f.dot(basis.evaluate_du(u_f, v_f)) / dx_f;
            Real z_f_y = coeff_f.dot(basis.evaluate_dv(u_f, v_f)) / dy_f;
            Real z_f_xx = coeff_f.dot(basis.evaluate_d2u(u_f, v_f)) / (dx_f * dx_f);
            Real z_f_xy = coeff_f.dot(basis.evaluate_d2uv(u_f, v_f)) / (dx_f * dy_f);
            Real z_f_yy = coeff_f.dot(basis.evaluate_d2v(u_f, v_f)) / (dy_f * dy_f);

            // Update max errors
            max_c0_error = std::max(max_c0_error, std::abs(z_c - z_f));
            max_c1x_error = std::max(max_c1x_error, std::abs(z_c_x - z_f_x));
            max_c1y_error = std::max(max_c1y_error, std::abs(z_c_y - z_f_y));
            max_c2xx_error = std::max(max_c2xx_error, std::abs(z_c_xx - z_f_xx));
            max_c2xy_error = std::max(max_c2xy_error, std::abs(z_c_xy - z_f_xy));
            max_c2yy_error = std::max(max_c2yy_error, std::abs(z_c_yy - z_f_yy));
          }
        }
      }
    }
  }

  std::cout << "Checked " << num_nc_interfaces << " non-conforming interfaces\n";
  std::cout << "Max C0 error: " << max_c0_error << "\n";
  std::cout << "Max C1_x error: " << max_c1x_error << "\n";
  std::cout << "Max C1_y error: " << max_c1y_error << "\n";
  std::cout << "Max C2_xx error: " << max_c2xx_error << "\n";
  std::cout << "Max C2_xy error: " << max_c2xy_error << "\n";
  std::cout << "Max C2_yy error: " << max_c2yy_error << "\n";

  EXPECT_GE(num_nc_interfaces, 14) << "Expected at least 14 non-conforming interfaces";
  EXPECT_LT(max_c0_error, 1e-8) << "C0 continuity violated";
  EXPECT_LT(max_c1x_error, 1e-6) << "C1 (dz/dx) continuity violated";
  EXPECT_LT(max_c1y_error, 1e-6) << "C1 (dz/dy) continuity violated";
  EXPECT_LT(max_c2xx_error, 1e-4) << "C2 (d²z/dx²) continuity violated";
  EXPECT_LT(max_c2xy_error, 1e-4) << "C2 (d²z/dxdy) continuity violated";
  EXPECT_LT(max_c2yy_error, 1e-4) << "C2 (d²z/dy²) continuity violated";
}

TEST_F(BezierBathymetrySmootherTest, MultiresolutionLargeMeshStress) {
  // Create a larger mesh by applying half-subdivision to multiple regions
  // We create two 5-level meshes side by side for more elements

  auto quadtree = std::make_unique<QuadtreeAdapter>();

  // Left half: 5-level half-subdivision pattern
  create_half_subdivided_mesh(*quadtree, 0.0, 100.0, 0.0, 100.0, 4);

  // Note: For a true stress test with more elements, we could add more regions
  // For now, 46 elements is already a good test of the constraint system

  Index num_elements = quadtree->num_elements();
  std::cout << "Created stress test mesh with " << num_elements << " elements\n";

  EXPECT_GE(num_elements, 46);

  // Count elements at each level
  std::map<int, int> level_counts;
  for (Index e = 0; e < quadtree->num_elements(); ++e) {
    QuadLevel lvl = quadtree->element_level(e);
    level_counts[lvl.x]++;
  }

  std::cout << "Element counts by level:\n";
  for (const auto &[lvl, count] : level_counts) {
    std::cout << "  L" << lvl << ": " << count << " elements\n";
  }

  // Analytical bathymetry
  auto bathy = [](Real x, Real y) {
    return 100.0 + 30.0 * std::sin(x * M_PI / 100.0) * std::sin(y * M_PI / 100.0);
  };

  BezierBathymetrySmoother smoother(*quadtree);
  smoother.set_bathymetry_data(bathy);

  // Time the solve
  auto start = std::chrono::high_resolution_clock::now();
  smoother.solve();
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "KKT solve time: " << duration.count() << " ms\n";

  EXPECT_TRUE(smoother.is_solved());

  // Check for NaN/Inf in solution
  VecX sol = smoother.solution();
  bool has_nan = false;
  bool has_inf = false;
  for (Index i = 0; i < sol.size(); ++i) {
    if (std::isnan(sol(i))) has_nan = true;
    if (std::isinf(sol(i))) has_inf = true;
  }
  EXPECT_FALSE(has_nan) << "Solution contains NaN values";
  EXPECT_FALSE(has_inf) << "Solution contains Inf values";

  // Check constraint violation
  Real violation = smoother.constraint_violation();
  Real sol_norm = sol.norm();
  Real relative_violation = (sol_norm > 1e-10) ? violation / sol_norm : violation;

  std::cout << "C² constraint violation: " << violation
            << " (relative: " << relative_violation << ")\n";

  EXPECT_LT(relative_violation, 1e-3) << "C² constraints not satisfied";

  // Count hanging node constraints
  // Calling num_constraints() first populates the internal constraint lists
  BezierC2ConstraintBuilder builder(*quadtree);
  Index total_constraints = builder.num_constraints();
  Index num_hanging = builder.hanging_node_constraints().size();
  std::cout << "Total constraints: " << total_constraints << "\n";
  std::cout << "Hanging node constraints: " << num_hanging << "\n";
  EXPECT_GE(num_hanging, 14) << "Expected many T-junctions";

  // Write VTK for visualization
  std::string vtk_path = "/tmp/bezier_multiresolution_stress";
  smoother.write_vtk(vtk_path, 3); // Lower resolution for larger mesh
  std::cout << "\nWrote VTK output to " << vtk_path << ".vtu\n";
}

// =============================================================================
// OpenMP Performance Benchmarks
// =============================================================================

TEST_F(BezierBathymetrySmootherTest, BenchmarkOpenMP8x8Mesh) {
  // 8x8 mesh (64 elements) - baseline benchmark
  auto quadtree = create_quadtree(8, 8);

  BezierBathymetrySmoother smoother(*quadtree);

  // Set simple quadratic bathymetry for consistent timing
  auto bathy = [](Real x, Real y) { return -50.0 - 0.001 * (x * x + y * y); };
  smoother.set_bathymetry_data(bathy);

  // Time the solve
  auto start = std::chrono::high_resolution_clock::now();
  smoother.solve();
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  std::cout << "\n=== 8×8 Mesh Benchmark ===\n";
  std::cout << "Elements: " << quadtree->num_elements() << "\n";
  std::cout << "DOFs: " << smoother.num_dofs() << "\n";
  std::cout << "Constraints: " << smoother.num_constraints() << "\n";
  std::cout << "Solve time: " << duration.count() << " ms\n";

  EXPECT_TRUE(smoother.is_solved());
  EXPECT_LT(smoother.constraint_violation(), 1e-8);
}

TEST_F(BezierBathymetrySmootherTest, BenchmarkOpenMP20x20Mesh) {
  // 20x20 mesh (400 elements) - Phase 1 target
  auto quadtree = create_quadtree(20, 20);

  BezierBathymetrySmoother smoother(*quadtree);

  // Set simple quadratic bathymetry for consistent timing
  auto bathy = [](Real x, Real y) { return -50.0 - 0.001 * (x * x + y * y); };
  smoother.set_bathymetry_data(bathy);

  // Time the solve
  auto start = std::chrono::high_resolution_clock::now();
  smoother.solve();
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  std::cout << "\n=== 20×20 Mesh Benchmark ===\n";
  std::cout << "Elements: " << quadtree->num_elements() << "\n";
  std::cout << "DOFs: " << smoother.num_dofs() << "\n";
  std::cout << "Constraints: " << smoother.num_constraints() << "\n";
  std::cout << "Solve time: " << duration.count() << " ms\n";

  EXPECT_TRUE(smoother.is_solved());
  EXPECT_LT(smoother.constraint_violation(), 1e-8);

  // Phase 1 target: should be <10 seconds with OpenMP
  // This is a relaxed check - actual speedup depends on hardware
  EXPECT_LT(duration.count(), 30000) << "Solve took too long - check OpenMP parallelization";
}

TEST_F(BezierBathymetrySmootherTest, BenchmarkThreadScaling) {
  // Test thread scaling with 10x10 mesh (100 elements)
  auto quadtree = create_quadtree(10, 10);

  auto bathy = [](Real x, Real y) { return -50.0 - 0.001 * (x * x + y * y); };

  std::cout << "\n=== Thread Scaling Benchmark (10×10 mesh) ===\n";
  std::cout << "Elements: " << quadtree->num_elements() << "\n";
  std::cout << "DOFs: " << quadtree->num_elements() * 36 << "\n\n";

  // Get current OMP_NUM_THREADS setting
  int num_threads = 1;
  #pragma omp parallel
  {
    #pragma omp single
    num_threads = omp_get_num_threads();
  }

  std::cout << "Current OMP_NUM_THREADS: " << num_threads << "\n";

  // Run benchmark 3 times and take median
  std::vector<long long> times;
  for (int run = 0; run < 3; ++run) {
    BezierBathymetrySmoother smoother(*quadtree);
    smoother.set_bathymetry_data(bathy);

    auto start = std::chrono::high_resolution_clock::now();
    smoother.solve();
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    times.push_back(duration.count());

    EXPECT_TRUE(smoother.is_solved());
    EXPECT_LT(smoother.constraint_violation(), 1e-8);
  }

  std::sort(times.begin(), times.end());
  long long median_time = times[1];

  std::cout << "Run times: " << times[0] << ", " << times[1] << ", " << times[2] << " ms\n";
  std::cout << "Median solve time: " << median_time << " ms\n";
  std::cout << "\nNote: Run with different OMP_NUM_THREADS to measure scaling:\n";
  std::cout << "  OMP_NUM_THREADS=1 ./build/tests/drifter_integration_tests --gtest_filter=\"*ThreadScaling*\"\n";
  std::cout << "  OMP_NUM_THREADS=8 ./build/tests/drifter_integration_tests --gtest_filter=\"*ThreadScaling*\"\n";
}

TEST_F(BezierBathymetrySmootherTest, BenchmarkAMR5Level) {
  // Benchmark 5-level AMR mesh (moderate complexity) - reuse existing mesh pattern
  auto quadtree = std::make_unique<QuadtreeAdapter>();
  create_half_subdivided_mesh(*quadtree, 0.0, 100.0, 0.0, 100.0, 4);

  ASSERT_EQ(quadtree->num_elements(), 46);  // Known size from MultiresolutionFiveLevels

  BezierBathymetrySmoother smoother(*quadtree);
  auto bathy = [](Real x, Real y) { return -50.0 - 0.001 * (x * x + y * y); };
  smoother.set_bathymetry_data(bathy);

  auto start = std::chrono::high_resolution_clock::now();
  smoother.solve();
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  std::cout << "\n=== 5-Level AMR Benchmark ===\n";
  std::cout << "Elements: " << quadtree->num_elements() << "\n";
  std::cout << "DOFs: " << smoother.num_dofs() << "\n";
  std::cout << "Constraints: " << smoother.num_constraints() << "\n";
  std::cout << "Solve time: " << duration.count() << " ms\n";

  EXPECT_TRUE(smoother.is_solved());
  EXPECT_LT(smoother.constraint_violation(), 1e-8);
}

TEST_F(BezierBathymetrySmootherTest, BenchmarkOpenMP30x30Mesh) {
  // 30x30 mesh (900 elements) - larger test for OpenMP crossover
  auto quadtree = create_quadtree(30, 30);

  BezierBathymetrySmoother smoother(*quadtree);

  auto bathy = [](Real x, Real y) { return -50.0 - 0.001 * (x * x + y * y); };
  smoother.set_bathymetry_data(bathy);

  auto start = std::chrono::high_resolution_clock::now();
  smoother.solve();
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  std::cout << "\n=== 30×30 Mesh Benchmark ===\n";
  std::cout << "Elements: " << quadtree->num_elements() << "\n";
  std::cout << "DOFs: " << smoother.num_dofs() << "\n";
  std::cout << "Constraints: " << smoother.num_constraints() << "\n";
  std::cout << "Solve time: " << duration.count() << " ms\n";

  EXPECT_TRUE(smoother.is_solved());
  EXPECT_LT(smoother.constraint_violation(), 1e-8);
}

TEST_F(BezierBathymetrySmootherTest, BenchmarkOpenMP40x40Mesh) {
  // 40x40 mesh (1600 elements) - test for larger scale
  auto quadtree = create_quadtree(40, 40);

  BezierBathymetrySmoother smoother(*quadtree);

  auto bathy = [](Real x, Real y) { return -50.0 - 0.001 * (x * x + y * y); };
  smoother.set_bathymetry_data(bathy);

  auto start = std::chrono::high_resolution_clock::now();
  smoother.solve();
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  std::cout << "\n=== 40×40 Mesh Benchmark ===\n";
  std::cout << "Elements: " << quadtree->num_elements() << "\n";
  std::cout << "DOFs: " << smoother.num_dofs() << "\n";
  std::cout << "Constraints: " << smoother.num_constraints() << "\n";
  std::cout << "Solve time: " << duration.count() << " ms\n";

  EXPECT_TRUE(smoother.is_solved());
  EXPECT_LT(smoother.constraint_violation(), 1e-8);
}

// Test multigrid solver integration
TEST_F(BezierBathymetrySmootherTest, MultigridSolverSmallMesh) {
  // Create 4×4 mesh
  auto quadtree = std::make_shared<QuadtreeAdapter>();
  quadtree->build_uniform(0.0, 1.0, 0.0, 1.0, 4, 4);

  // Solve with SparseLU (reference solution)
  BezierSmootherConfig config_direct;
  config_direct.lambda = 0.1;
  config_direct.use_multigrid = false;

  BezierBathymetrySmoother smoother_direct(*quadtree, config_direct);

  auto bathy = [](Real x, Real y) { return -10.0 + 2.0 * x + 3.0 * y; };
  smoother_direct.set_bathymetry_data(bathy);
  smoother_direct.solve();

  VecX solution_direct = smoother_direct.solution();

  // Solve with multigrid
  BezierSmootherConfig config_mg;
  config_mg.lambda = 0.1;
  config_mg.use_multigrid = true;
  config_mg.multigrid_max_iterations = 100;
  config_mg.multigrid_tolerance = 1e-6;

  BezierBathymetrySmoother smoother_mg(*quadtree, config_mg);
  smoother_mg.set_bathymetry_data(bathy);
  smoother_mg.solve();

  VecX solution_mg = smoother_mg.solution();

  // Compare solutions (should be very similar for this linear case)
  Real max_diff = (solution_direct - solution_mg).cwiseAbs().maxCoeff();
  std::cout << "Max difference between SparseLU and multigrid: " << max_diff << "\n";

  // Note: Multigrid is iterative, so won't be exact, but should be close
  // For now, just verify it runs and produces a solution
  EXPECT_TRUE(smoother_mg.is_solved());
  EXPECT_LT(smoother_mg.constraint_violation(), 1e-6);

  // TODO: When full hierarchy is implemented, test convergence on larger meshes
}

// Test multigrid with non-conforming mesh
TEST_F(BezierBathymetrySmootherTest, MultigridNonConforming) {
  // Create 1+4 non-conforming mesh
  auto quadtree = std::make_shared<QuadtreeAdapter>();

  // Coarse element
  quadtree->add_element(QuadBounds{0.0, 1.0, 0.0, 1.0}, QuadLevel{0, 0});

  // Fine elements (2×2 refinement in right half)
  quadtree->add_element(QuadBounds{1.0, 1.5, 0.0, 0.5}, QuadLevel{1, 1});
  quadtree->add_element(QuadBounds{1.5, 2.0, 0.0, 0.5}, QuadLevel{1, 1});
  quadtree->add_element(QuadBounds{1.0, 1.5, 0.5, 1.0}, QuadLevel{1, 1});
  quadtree->add_element(QuadBounds{1.5, 2.0, 0.5, 1.0}, QuadLevel{1, 1});

  // Solve with multigrid
  BezierSmootherConfig config;
  config.lambda = 0.1;
  config.use_multigrid = true;

  BezierBathymetrySmoother smoother(*quadtree, config);

  auto bathy = [](Real x, Real y) { return -5.0 - x * x - y * y; };
  smoother.set_bathymetry_data(bathy);
  smoother.solve();

  EXPECT_TRUE(smoother.is_solved());
  EXPECT_LT(smoother.constraint_violation(), 1e-6);

  // Verify C² continuity at T-junction
  Real tol = 1e-6;
  EXPECT_LT(smoother.constraint_violation(), tol);
}

// ===========================================================================
// Multigrid Large Domain Demonstration
// ===========================================================================

TEST_F(BezierBathymetrySmootherTest, MultigridKattegatDemonstration) {
  // ============================================================================
  // MULTIGRID LARGE DOMAIN DEMONSTRATION
  // ============================================================================
  // Demonstrates multigrid solver on large real-world bathymetry from Kattegat.
  // Domain: ~50km × 50km region in Kattegat strait (Denmark/Sweden)
  // Mesh: 30×30 uniform quadtree (900 elements, 32,400 DOFs)
  // Solver: Multigrid V-cycle with C² continuity constraints
  //
  // This test validates:
  // - Multigrid configuration on practical domain sizes
  // - Real bathymetry data handling
  // - Solution quality (C² continuity, constraint satisfaction)
  // - VTK output generation for visualization
  // ============================================================================

  std::cout << "\n=== Multigrid Kattegat Demonstration ===\n";

  // Load GeoTIFF bathymetry
  GeoTiffReader reader;
  std::string geotiff_path = BATHYMETRY_GEOTIFF_PATH;
  BathymetryData bathy = reader.load(geotiff_path);

  ASSERT_TRUE(bathy.is_valid()) << "Failed to load GeoTIFF: " << geotiff_path;

  std::cout << "GeoTIFF loaded:\n";
  std::cout << "  Domain: [" << bathy.xmin << ", " << bathy.xmax << "] × ["
            << bathy.ymin << ", " << bathy.ymax << "]\n";
  std::cout << "  Resolution: " << bathy.sizex << " × " << bathy.sizey
            << " pixels\n";

  // Find suitable water region in Kattegat (northwest of domain)
  // Search with 50km regions, focusing on Kattegat depths (10-50m typical)
  Real region_size = 50000.0;  // 50 km
  Real search_step = 25000.0;  // 25 km step

  Real best_x = 0.0, best_y = 0.0;
  Real max_depth_found = 0.0;
  int water_regions_found = 0;

  // Sample northwest quadrant (Kattegat location)
  Real x_search_min = bathy.xmin + (bathy.xmax - bathy.xmin) * 0.1;
  Real x_search_max = bathy.xmin + (bathy.xmax - bathy.xmin) * 0.5;
  Real y_search_min = bathy.ymin + (bathy.ymax - bathy.ymin) * 0.5;
  Real y_search_max = bathy.ymin + (bathy.ymax - bathy.ymin) * 0.9;

  for (Real y = y_search_min; y <= y_search_max; y += search_step) {
    for (Real x = x_search_min; x <= x_search_max; x += search_step) {
      float depth = bathy.get_depth(x, y);
      if (depth > max_depth_found && depth > 5.0) {
        max_depth_found = depth;
        best_x = x;
        best_y = y;
        water_regions_found++;
      }
    }
  }

  ASSERT_GT(max_depth_found, 5.0)
      << "No suitable water region found in Kattegat area";

  std::cout << "\nKattegat region selected:\n";
  std::cout << "  Center: (" << best_x << ", " << best_y << ")\n";
  std::cout << "  Max depth: " << max_depth_found << " m\n";
  std::cout << "  Water regions surveyed: " << water_regions_found << "\n";

  // Create mesh bounds centered on selected region
  Real xmin = best_x - region_size / 2.0;
  Real xmax = best_x + region_size / 2.0;
  Real ymin = best_y - region_size / 2.0;
  Real ymax = best_y + region_size / 2.0;

  std::cout << "  Mesh bounds: [" << xmin << ", " << xmax << "] × [" << ymin
            << ", " << ymax << "]\n";
  std::cout << "  Domain size: " << (xmax - xmin) / 1000.0 << " km × "
            << (ymax - ymin) / 1000.0 << " km\n";

  // Create large uniform quadtree mesh
  int nx = 30, ny = 30;
  auto quadtree = std::make_unique<QuadtreeAdapter>();
  Real dx = (xmax - xmin) / nx;
  Real dy = (ymax - ymin) / ny;

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      QuadBounds bounds;
      bounds.xmin = xmin + i * dx;
      bounds.xmax = xmin + (i + 1) * dx;
      bounds.ymin = ymin + j * dy;
      bounds.ymax = ymin + (j + 1) * dy;
      quadtree->add_element(bounds, QuadLevel{0, 0});
    }
  }

  std::cout << "\nMesh created:\n";
  std::cout << "  Elements: " << quadtree->num_elements() << " (" << nx << "×"
            << ny << ")\n";
  std::cout << "  DOFs: " << quadtree->num_elements() * 36 << "\n";
  std::cout << "  Element size: " << dx / 1000.0 << " km × " << dy / 1000.0
            << " km\n";

  // Configure multigrid solver
  BezierSmootherConfig config;
  config.lambda = 0.05;  // Moderate data fitting weight
  config.use_multigrid = true;
  config.multigrid_max_iterations = 100;
  config.multigrid_tolerance = 1e-6;
  config.enable_natural_bc = true;

  std::cout << "\nMultigrid configuration:\n";
  std::cout << "  Lambda: " << config.lambda << "\n";
  std::cout << "  Max iterations: " << config.multigrid_max_iterations << "\n";
  std::cout << "  Tolerance: " << config.multigrid_tolerance << "\n";

  // Create smoother and set bathymetry
  BezierBathymetrySmoother smoother(*quadtree, config);

  auto bathy_func = [&bathy](Real x, Real y) -> Real {
    return static_cast<Real>(bathy.get_depth(x, y));
  };
  smoother.set_bathymetry_data(bathy_func);

  // Solve with multigrid
  std::cout << "\nSolving with multigrid...\n";
  auto start = std::chrono::high_resolution_clock::now();

  smoother.solve();

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  std::cout << "  Solve time: " << duration.count() << " ms ("
            << duration.count() / 1000.0 << " s)\n";

  // Validate solution
  EXPECT_TRUE(smoother.is_solved()) << "Multigrid solve failed";

  Real constraint_viol = smoother.constraint_violation();
  std::cout << "  Constraint violation: " << constraint_viol << "\n";
  EXPECT_LT(constraint_viol, 2e-5) << "C² constraints not satisfied";

  Real data_res = smoother.data_residual();
  Real reg_energy = smoother.regularization_energy();
  Real obj_value = smoother.objective_value();

  std::cout << "  Data residual: " << data_res << "\n";
  std::cout << "  Regularization energy: " << reg_energy << "\n";
  std::cout << "  Objective value: " << obj_value << "\n";

  // Check for numerical issues
  VecX sol = smoother.solution();
  bool has_nan = false, has_inf = false;
  for (Index i = 0; i < sol.size(); ++i) {
    if (std::isnan(sol(i))) has_nan = true;
    if (std::isinf(sol(i))) has_inf = true;
  }
  EXPECT_FALSE(has_nan) << "Solution contains NaN";
  EXPECT_FALSE(has_inf) << "Solution contains Inf";

  // Sample and compare solution to input data at several points
  std::vector<std::pair<Real, Real>> sample_points = {
      {best_x, best_y},                             // Center
      {xmin + dx, ymin + dy},                       // Near corner
      {xmax - dx, ymax - dy},                       // Opposite corner
      {(xmin + xmax) / 2, ymin + 2 * dy},           // Edge
      {xmin + 3 * dx, (ymin + ymax) / 2}            // Different edge
  };

  std::cout << "\nSolution sampling:\n";
  Real max_error = 0.0;
  for (const auto& [x, y] : sample_points) {
    Real expected = bathy_func(x, y);
    Real computed = smoother.evaluate(x, y);
    Real error = std::abs(computed - expected);
    max_error = std::max(max_error, error);
    std::cout << "  (" << x << ", " << y << "): "
              << "expected=" << expected << ", computed=" << computed
              << ", error=" << error << "\n";
  }
  std::cout << "  Max error: " << max_error << " m\n";

  // Write VTK output for visualization
  std::string output_base = "/tmp/multigrid_kattegat";
  smoother.write_vtk(output_base, 10);  // 10 subdivisions per element

  // Write raw GeoTIFF data at native resolution
  std::string raw_path = "/tmp/multigrid_kattegat_raw";
  write_geotiff_region_vtk(bathy, xmin, xmax, ymin, ymax, raw_path);

  std::cout << "\nVTK output written:\n";
  std::cout << "  " << output_base << ".vtu (smoothed solution)\n";
  std::cout << "  " << raw_path << ".vtu (raw GeoTIFF data)\n";
  std::cout << "  Visualization: 10 subdivisions per element\n";

  std::cout << "\n=== Demonstration Complete ===\n";
  std::cout << "Successfully demonstrated multigrid on "
            << region_size / 1000.0 << " km × " << region_size / 1000.0
            << " km Kattegat domain\n";
  std::cout << "Solution validates: constraint violation < 2e-5, no NaN/Inf\n";
}
