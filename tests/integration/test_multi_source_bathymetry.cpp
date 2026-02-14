#include "bathymetry/adaptive_cg_linear_bezier_smoother.hpp"
#include "io/bathymetry_vtk_writer.hpp"
#include "mesh/multi_source_bathymetry.hpp"
#include "test_integration_fixtures.hpp"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>
#include <iostream>

#include <ogr_spatialref.h>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>

using namespace drifter;
using namespace drifter::testing;

// =============================================================================
// Test fixture
// =============================================================================

class MultiSourceBathymetryTest : public ::testing::Test {
protected:
  static constexpr Real TOLERANCE = 1e-6;

  // Data paths
  const std::string data_dir = "/home/ole/Projects/drifter/data/input/";
  const std::string primary_file = data_dir + "ddm_50m.dybde-emodnet.tif";
  const std::vector<std::string> tile_files = {
      data_dir + "C4_2024.tif", data_dir + "C5_2024.tif",
      data_dir + "C6_2024.tif", data_dir + "C7_2024.tif",
      data_dir + "D4_2024.tif", data_dir + "D5_2024.tif",
      data_dir + "D6_2024.tif", data_dir + "D7_2024.tif",
      data_dir + "E4_2024.tif", data_dir + "E5_2024.tif",
      data_dir + "E6_2024.tif", data_dir + "E7_2024.tif",
  };

  bool data_files_exist() const {
    if (!std::filesystem::exists(primary_file))
      return false;
    for (const auto &tile : tile_files) {
      if (!std::filesystem::exists(tile))
        return false;
    }
    return true;
  }
};

// =============================================================================
// Basic construction tests
// =============================================================================

TEST_F(MultiSourceBathymetryTest, IsAvailable) {
  EXPECT_TRUE(MultiSourceBathymetry::is_available());
}

TEST_F(MultiSourceBathymetryTest, ConstructWithValidFiles) {
  if (!data_files_exist()) {
    GTEST_SKIP() << "Data files not available";
  }

  MultiSourceBathymetry bathy(primary_file, tile_files);
  EXPECT_EQ(bathy.num_sources(), 1 + tile_files.size());
}

TEST_F(MultiSourceBathymetryTest, ConstructWithMissingPrimaryThrows) {
  EXPECT_THROW(MultiSourceBathymetry("nonexistent.tif", {}),
               std::runtime_error);
}

TEST_F(MultiSourceBathymetryTest, ConstructWithMissingTileThrows) {
  if (!std::filesystem::exists(primary_file)) {
    GTEST_SKIP() << "Primary file not available";
  }

  EXPECT_THROW(MultiSourceBathymetry(primary_file, {"nonexistent.tif"}),
               std::runtime_error);
}

// =============================================================================
// Lookup tests
// =============================================================================

TEST_F(MultiSourceBathymetryTest, EvaluateInsidePrimaryDomain) {
  if (!data_files_exist()) {
    GTEST_SKIP() << "Data files not available";
  }

  MultiSourceBathymetry bathy(primary_file, tile_files);

  // Get bounds and test center point
  Real xmin, xmax, ymin, ymax;
  bathy.get_bounds(xmin, xmax, ymin, ymax);

  Real cx = 0.5 * (xmin + xmax);
  Real cy = 0.5 * (ymin + ymax);

  // Should not throw - point is inside primary domain
  EXPECT_NO_THROW({
    Real depth = bathy.evaluate(cx, cy);
    // Depth should be non-negative (either water or land)
    EXPECT_GE(depth, 0.0);
  });
}

TEST_F(MultiSourceBathymetryTest, ContainsInsidePrimary) {
  if (!data_files_exist()) {
    GTEST_SKIP() << "Data files not available";
  }

  MultiSourceBathymetry bathy(primary_file, tile_files);

  Real xmin, xmax, ymin, ymax;
  bathy.get_bounds(xmin, xmax, ymin, ymax);

  Real cx = 0.5 * (xmin + xmax);
  Real cy = 0.5 * (ymin + ymax);

  EXPECT_TRUE(bathy.contains(cx, cy));
}

TEST_F(MultiSourceBathymetryTest, IsLandReturnsCorrectly) {
  if (!data_files_exist()) {
    GTEST_SKIP() << "Data files not available";
  }

  MultiSourceBathymetry bathy(primary_file, tile_files);

  Real xmin, xmax, ymin, ymax;
  bathy.get_bounds(xmin, xmax, ymin, ymax);

  Real cx = 0.5 * (xmin + xmax);
  Real cy = 0.5 * (ymin + ymax);

  // is_land should return true iff depth == 0
  bool is_land = bathy.is_land(cx, cy);
  Real depth = bathy.evaluate(cx, cy);

  if (is_land) {
    EXPECT_EQ(depth, 0.0);
  } else {
    EXPECT_GT(depth, 0.0);
  }
}

TEST_F(MultiSourceBathymetryTest, OutsideAllSourcesThrows) {
  if (!data_files_exist()) {
    GTEST_SKIP() << "Data files not available";
  }

  MultiSourceBathymetry bathy(primary_file, tile_files);

  // Point far outside any reasonable domain (EPSG:3034 coordinates)
  EXPECT_THROW(bathy.evaluate(-1e9, -1e9), std::out_of_range);
}

TEST_F(MultiSourceBathymetryTest, ContainsOutsideDomain) {
  if (!data_files_exist()) {
    GTEST_SKIP() << "Data files not available";
  }

  MultiSourceBathymetry bathy(primary_file, tile_files);

  // Point far outside
  EXPECT_FALSE(bathy.contains(-1e9, -1e9));
}

// =============================================================================
// BathymetrySource interface test
// =============================================================================

TEST_F(MultiSourceBathymetryTest, WorksAsBathymetrySource) {
  if (!data_files_exist()) {
    GTEST_SKIP() << "Data files not available";
  }

  MultiSourceBathymetry bathy(primary_file, tile_files);

  // Cast to BathymetrySource interface
  const BathymetrySource &source = bathy;

  Real xmin, xmax, ymin, ymax;
  bathy.get_bounds(xmin, xmax, ymin, ymax);

  Real cx = 0.5 * (xmin + xmax);
  Real cy = 0.5 * (ymin + ymax);

  // Should work via interface
  Real depth = source.evaluate(cx, cy);
  EXPECT_GE(depth, 0.0);
}

// =============================================================================
// Bounds test
// =============================================================================

TEST_F(MultiSourceBathymetryTest, GetBoundsReturnsValidBounds) {
  if (!data_files_exist()) {
    GTEST_SKIP() << "Data files not available";
  }

  MultiSourceBathymetry bathy(primary_file, tile_files);

  Real xmin, xmax, ymin, ymax;
  bathy.get_bounds(xmin, xmax, ymin, ymax);

  EXPECT_LT(xmin, xmax);
  EXPECT_LT(ymin, ymax);
}

// =============================================================================
// Large domain adaptive refinement test
// =============================================================================

TEST_F(MultiSourceBathymetryTest, LargeDomainAdaptiveRefinement) {
  if (!data_files_exist()) {
    GTEST_SKIP() << "Data files not available";
  }

  // Large domain in EPSG:3034 (2000 km x 2000 km)
  // Covers approximately: 50°N to 66°N, 5°W to 30°E
  Real xmin = 3200000.0, xmax = 5200000.0;
  Real ymin = 2600000.0, ymax = 4600000.0;

  std::cout << "=== Large Domain Adaptive Bathymetry Test ===" << std::endl;
  std::cout << "Domain: [" << xmin << ", " << xmax << "] x [" << ymin << ", "
            << ymax << "]" << std::endl;
  std::cout << "Size: " << (xmax - xmin) / 1000.0 << " km x "
            << (ymax - ymin) / 1000.0 << " km" << std::endl;

  // Load multi-source bathymetry
  MultiSourceBathymetry bathy(primary_file, tile_files);

  // Wrap for smoother interface - handle out-of-domain as land
  auto depth_func = [&bathy](Real x, Real y) -> Real {
    try {
      return bathy.evaluate(x, y);
    } catch (const std::out_of_range &) {
      return 0.0; // Outside domain = land
    }
  };

  auto is_land_func = [&bathy](Real x, Real y) -> bool {
    try {
      return bathy.is_land(x, y);
    } catch (const std::out_of_range &) {
      return true; // Outside domain = land
    }
  };

  // Configure adaptive smoother - shallow for initial testing
  AdaptiveCGLinearBezierConfig config;
  config.max_refinement_level =
      5;                         // Shallow mesh (coarsest ~250km, finest ~8km)
  config.error_threshold = 10.0; // Loose threshold (meters)
  config.max_iterations = 10;
  config.max_elements = 5000;
  config.smoother_config.lambda = 10.0;
  config.verbose = true;

  // Start with coarse 8x8 grid (64 elements, each ~250km)
  auto start = std::chrono::high_resolution_clock::now();

  AdaptiveCGLinearBezierSmoother smoother(xmin, xmax, ymin, ymax, 8, 8, config);
  smoother.set_bathymetry_data(std::function<Real(Real, Real)>(depth_func));
  smoother.set_land_mask(std::function<bool(Real, Real)>(is_land_func));

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
  std::cout << "  Time: " << time_ms / 1000.0 << " s" << std::endl;

  EXPECT_TRUE(smoother.is_solved());

  // Write VTK output with per-element error data
  std::string output_file = "/tmp/large_domain_bathymetry";

  auto errors = smoother.estimate_errors();
  std::vector<Real> element_rms(smoother.mesh().num_elements(), 0.0);
  std::vector<Real> element_std(smoother.mesh().num_elements(), 0.0);
  std::vector<Real> refinement_levels(smoother.mesh().num_elements(), 0.0);

  for (const auto &e : errors) {
    element_rms[e.element] = e.normalized_error;
    element_std[e.element] = e.std_error;
  }
  for (Index i = 0; i < smoother.mesh().num_elements(); ++i) {
    refinement_levels[i] =
        static_cast<Real>(smoother.mesh().element_level(i).max_level());
  }

  io::write_cg_bezier_surface_vtk(
      output_file, smoother.mesh(),
      [&smoother](Real x, Real y) { return smoother.evaluate(x, y); }, 6,
      "depth",
      {{"rms_error", element_rms},
       {"std_error", element_std},
       {"refinement_level", refinement_levels}});

  std::cout << "Output written to: " << output_file << ".vtu" << std::endl;

  EXPECT_TRUE(std::filesystem::exists(output_file + ".vtu"));
}

// =============================================================================
// Large domain with polygon mask test
// =============================================================================

TEST_F(MultiSourceBathymetryTest, LargeDomainWithPolygonMask) {
  if (!data_files_exist()) {
    GTEST_SKIP() << "Data files not available";
  }

  // Define North Sea / Baltic Sea polygon in WGS84 (lon, lat)
  // This simple polygon has two exclusion lines:
  // - North: Peterhead (Scotland) to Obrestad Fyr (Norway)
  // - South: Dover pointing SE towards French coast
  // Plus generous domain bounds to include all Baltic/Gulf areas
  std::vector<std::pair<Real, Real>> polygon_wgs84 = {
      {-1.8, 57.5}, // Peterhead, Scotland (north exclusion line start)
      {5.5, 58.5},  // Obrestad Fyr, Norway (north exclusion line end)
      {30.0, 66.0}, // Domain NE corner (Gulf of Bothnia/Finland extent)
      {30.0, 54.0}, // Domain SE corner (Baltic south)
      {1.8, 50.8},  // French coast (south exclusion line end)
      {1.3, 51.1},  // Dover, UK (south exclusion line start)
  };

  // Transform polygon from WGS84 (EPSG:4326) to EPSG:3034
  std::vector<std::pair<Real, Real>> polygon_3034;

  OGRSpatialReference srcSRS, dstSRS;
  srcSRS.SetFromUserInput("EPSG:4326");
  dstSRS.SetFromUserInput("EPSG:3034");

  // Use traditional GIS order (lon, lat) for EPSG:4326
  srcSRS.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
  dstSRS.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);

  auto to_3034 = std::unique_ptr<OGRCoordinateTransformation>(
      OGRCreateCoordinateTransformation(&srcSRS, &dstSRS));

  if (!to_3034) {
    GTEST_SKIP() << "Failed to create coordinate transformation";
  }

  std::cout << "=== Polygon Coordinate Transformation ===" << std::endl;
  for (const auto &pt : polygon_wgs84) {
    double x = pt.first;  // lon
    double y = pt.second; // lat
    if (to_3034->Transform(1, &x, &y)) {
      polygon_3034.emplace_back(x, y);
      std::cout << "  WGS84(" << pt.first << ", " << pt.second
                << ") -> EPSG:3034(" << x << ", " << y << ")" << std::endl;
    } else {
      GTEST_SKIP() << "Coordinate transformation failed for point";
    }
  }

  // Write polygon boundary to VTP file for ParaView visualization
  std::string polygon_output = "/tmp/north_sea_baltic_boundary";
  io::write_polygon_vtk(polygon_output, polygon_3034, 0.0, "north_sea_baltic");
  std::cout << "Polygon boundary written to: " << polygon_output << ".vtp"
            << std::endl;

  EXPECT_TRUE(std::filesystem::exists(polygon_output + ".vtp"));

  // Now run the same adaptive bathymetry as LargeDomainAdaptiveRefinement
  Real xmin = 3200000.0, xmax = 5200000.0;
  Real ymin = 2600000.0, ymax = 4600000.0;

  std::cout << "\n=== Large Domain Adaptive Bathymetry with Polygon Mask ==="
            << std::endl;
  std::cout << "Domain: [" << xmin << ", " << xmax << "] x [" << ymin << ", "
            << ymax << "]" << std::endl;

  // Load multi-source bathymetry
  MultiSourceBathymetry bathy(primary_file, tile_files);

  // Create boost::geometry polygon for masking refinement
  namespace bg = boost::geometry;
  using Point2D = bg::model::point<double, 2, bg::cs::cartesian>;
  using Polygon2D = bg::model::polygon<Point2D, true>;

  Polygon2D poly;
  for (const auto &pt : polygon_3034) {
    bg::append(poly.outer(), Point2D(pt.first, pt.second));
  }
  // Close the polygon
  if (!polygon_3034.empty()) {
    bg::append(poly.outer(),
               Point2D(polygon_3034[0].first, polygon_3034[0].second));
  }
  bg::correct(poly);

  // Depth function: return 0 outside polygon, actual bathymetry inside
  auto depth_func = [&bathy, &poly](Real x, Real y) -> Real {
    Point2D p(x, y);
    // Outside polygon = depth 0
    if (!bg::within(p, poly)) {
      return 0.0;
    }
    // Inside polygon = actual bathymetry
    try {
      return bathy.evaluate(x, y);
    } catch (const std::out_of_range &) {
      return 0.0;
    }
  };

  // Land mask: outside polygon = "land" (controls refinement)
  // With original is_element_on_land() logic: refine if ANY corner is inside
  auto is_land_func = [&poly](Real x, Real y) -> bool {
    Point2D p(x, y);
    return !bg::within(p, poly); // Outside polygon = land
  };

  // Configure adaptive smoother
  AdaptiveCGLinearBezierConfig config;
  config.max_refinement_level = 15;
  config.error_threshold = 0.01;
  config.max_iterations = 100;
  config.max_elements = 10000;
  config.smoother_config.lambda = 1000.0;
  config.verbose = true;
  config.error_metric_type = ErrorMetricType::RelativeError;

  // Start with single element covering entire domain
  auto start = std::chrono::high_resolution_clock::now();

  AdaptiveCGLinearBezierSmoother smoother(xmin, xmax, ymin, ymax, 1, 1, config);
  smoother.set_bathymetry_data(std::function<Real(Real, Real)>(depth_func));
  smoother.set_land_mask(std::function<bool(Real, Real)>(is_land_func));

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
  std::cout << "  Time: " << time_ms / 1000.0 << " s" << std::endl;

  EXPECT_TRUE(smoother.is_solved());

  // Compute per-element membership (polygon already created above for
  // masking)
  Index num_elements = smoother.mesh().num_elements();
  std::vector<Real> in_region(num_elements, 0.0);
  Index elements_in_region = 0;

  for (Index e = 0; e < num_elements; ++e) {
    const auto &bounds = smoother.mesh().element_bounds(e);
    Real cx = 0.5 * (bounds.xmin + bounds.xmax);
    Real cy = 0.5 * (bounds.ymin + bounds.ymax);
    Point2D center(cx, cy);

    if (bg::within(center, poly)) {
      in_region[e] = 1.0;
      elements_in_region++;
    }
  }

  std::cout << "Elements in North Sea/Baltic region: " << elements_in_region
            << " / " << num_elements << std::endl;

  EXPECT_GT(elements_in_region, 0);

  // Write VTK output with per-element error and region membership data
  std::string output_file = "/tmp/large_domain_bathymetry_polygon";

  auto errors = smoother.estimate_errors();
  std::vector<Real> element_rms(num_elements, 0.0);
  std::vector<Real> element_std(num_elements, 0.0);
  std::vector<Real> refinement_levels(num_elements, 0.0);

  for (const auto &e : errors) {
    element_rms[e.element] = e.normalized_error;
    element_std[e.element] = e.std_error;
  }
  for (Index i = 0; i < num_elements; ++i) {
    refinement_levels[i] =
        static_cast<Real>(smoother.mesh().element_level(i).max_level());
  }

  io::write_cg_bezier_surface_vtk(
      output_file, smoother.mesh(),
      [&smoother](Real x, Real y) { return smoother.evaluate(x, y); }, 6,
      "depth",
      {{"rms_error", element_rms},
       {"std_error", element_std},
       {"refinement_level", refinement_levels},
       {"in_north_sea_baltic", in_region}});

  std::cout << "Output written to: " << output_file << ".vtu" << std::endl;

  EXPECT_TRUE(std::filesystem::exists(output_file + ".vtu"));
}
