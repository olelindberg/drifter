#include "bathymetry/pixel_max_error_estimator.hpp"
#include "bathymetry/linear_bezier_surface.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "mesh/geotiff_reader.hpp"
#include <gtest/gtest.h>

using namespace drifter;

/// Mock surface that returns a constant value
class ConstantSurface {
  public:
  explicit ConstantSurface(Real value) : value_(value) {}
  Real evaluate(Real /*x*/, Real /*y*/) const { return value_; }

  private:
  Real value_;
};

/// Create synthetic BathymetryData for testing
static BathymetryData create_test_bathymetry(int sizex, int sizey, Real xmin, Real xmax, Real ymin, Real ymax, std::function<float(double, double)> elevation_func, float nodata_value = -9999.0f) {
  BathymetryData data;
  data.sizex             = sizex;
  data.sizey             = sizey;
  data.nodata_value      = nodata_value;
  data.is_depth_positive = false; // elevation format

    // Compute pixel size
  Real dx = (xmax - xmin) / sizex;
  Real dy = (ymax - ymin) / sizey;

    // Set geotransform (origin at top-left, y decreasing)
  data.geotransform[0] = xmin;        // top-left X
  data.geotransform[1] = dx;          // pixel width
  data.geotransform[2] = 0.0;         // row rotation
  data.geotransform[3] = ymax;        // top-left Y
  data.geotransform[4] = 0.0;         // column rotation
  data.geotransform[5] = -dy;         // pixel height (negative)

  data.xmin = xmin;
  data.xmax = xmax;
  data.ymin = ymin;
  data.ymax = ymax;

    // Generate elevation data
  data.elevation.resize(sizex * sizey);
  for (int py = 0; py < sizey; ++py) {
    for (int px = 0; px < sizex; ++px) {
      double wx, wy;
      data.pixel_center_to_world(px, py, wx, wy);
      data.elevation[py * sizex + px] = elevation_func(wx, wy);
    }
  }

  return data;
}

class PixelMaxErrorEstimatorTest : public ::testing::Test {
  protected:
  void SetUp() override {
        // Create a 10x10 pixel domain from (0,0) to (100,100)
        // Each pixel is 10m x 10m
  }
};

TEST_F(PixelMaxErrorEstimatorTest, ConstantSurfaceZeroError) {
    // Surface returns constant -50.0 (50m depth as elevation)
  ConstantSurface surface(-50.0);

    // Create bathymetry with matching constant values
  auto data = create_test_bathymetry(10, 10, 0.0, 100.0, 0.0, 100.0, [](double /*x*/, double /*y*/) { return -50.0f; });

    // Create single-element mesh covering entire domain
  QuadtreeAdapter mesh;
  mesh.build_uniform(0.0, 100.0, 0.0, 100.0, 1, 1);

  PixelErrorEstimator estimator(surface, data, mesh);

    // Max error should be exactly zero
  auto errors = estimator.estimate_all();
  EXPECT_EQ(errors.size(), 1u);
  EXPECT_NEAR(errors[0].max_pixel_error, 0.0, 1e-10);

    // Should have 100 pixels (10x10)
  EXPECT_EQ(estimator.total_pixel_count(), 100);
}

TEST_F(PixelMaxErrorEstimatorTest, KnownErrorMagnitude) {
    // Surface returns 0.0 everywhere
  ConstantSurface surface(0.0);

    // Create bathymetry with constant -1.0 (1m error at every pixel)
  auto data = create_test_bathymetry(10, 10, 0.0, 100.0, 0.0, 100.0, [](double /*x*/, double /*y*/) { return -1.0f; });

  QuadtreeAdapter mesh;
  mesh.build_uniform(0.0, 100.0, 0.0, 100.0, 1, 1);

  PixelErrorEstimator estimator(surface, data, mesh);

  auto errors = estimator.estimate_all();
  EXPECT_EQ(errors.size(), 1u);

    // Max error should be 1.0 (same as RMSE in uniform error case)
  EXPECT_NEAR(errors[0].max_pixel_error, 1.0, 1e-10);
}

TEST_F(PixelMaxErrorEstimatorTest, MaxErrorDiffersFromRMSE) {
    // This is the key test: max error differs from RMSE
    // Surface returns 0.0 everywhere
  ConstantSurface surface(0.0);

    // Create bathymetry where most pixels have small errors, but one has a large error
    // First half: error = 1.0, Second half: error = 5.0
  auto data = create_test_bathymetry(10, 10, 0.0, 100.0, 0.0, 100.0, [](double x, double /*y*/) { return x < 50.0 ? -1.0f : -5.0f; });

  QuadtreeAdapter mesh;
  mesh.build_uniform(0.0, 100.0, 0.0, 100.0, 1, 1);

  PixelErrorEstimator estimator(surface, data, mesh);

  auto errors = estimator.estimate_all();
  EXPECT_EQ(errors.size(), 1u);

    // Max error should be 5.0 (the maximum absolute error)
  EXPECT_NEAR(errors[0].max_pixel_error, 5.0, 1e-10);

    // RMSE would be sqrt((50*1 + 50*25)/100) = sqrt(1300/100) = sqrt(13) ~ 3.6
    // This confirms max error is different from RMSE
  EXPECT_NEAR(errors[0].pixel_rmse, std::sqrt(13.0), 1e-10);
  EXPECT_GT(errors[0].max_pixel_error, errors[0].pixel_rmse);
}

TEST_F(PixelMaxErrorEstimatorTest, PerElementMaxTracking) {
    // Surface returns 0.0 everywhere
  ConstantSurface surface(0.0);

    // Element 0 (x < 50): error = 2.0
    // Element 1 (x >= 50): error = 4.0
  auto data = create_test_bathymetry(10, 10, 0.0, 100.0, 0.0, 100.0, [](double x, double /*y*/) { return x < 50.0 ? -2.0f : -4.0f; });

    // 2 elements side by side
  QuadtreeAdapter mesh;
  mesh.build_uniform(0.0, 100.0, 0.0, 100.0, 2, 1);

  PixelErrorEstimator estimator(surface, data, mesh);

  auto errors = estimator.estimate_all();
  EXPECT_EQ(errors.size(), 2u);

    // Each element should have its own max error
  EXPECT_NEAR(errors[0].max_pixel_error, 2.0, 1e-10);
  EXPECT_NEAR(errors[1].max_pixel_error, 4.0, 1e-10);
}

TEST_F(PixelMaxErrorEstimatorTest, NoDataHandling) {
    // Surface returns -50.0
  ConstantSurface surface(-50.0);

    // Create bathymetry with some NoData values
  auto data = create_test_bathymetry(10, 10, 0.0, 100.0, 0.0, 100.0, [](double x, double y) {
        // Mark top-right quadrant as NoData
    if (x > 50.0 && y > 50.0) {
      return -9999.0f; // NoData
    }
    return -50.0f;
  });

  QuadtreeAdapter mesh;
  mesh.build_uniform(0.0, 100.0, 0.0, 100.0, 1, 1);

  PixelErrorEstimator estimator(surface, data, mesh);

    // Should count only 75 pixels (100 - 25 NoData)
  EXPECT_EQ(estimator.total_pixel_count(), 75);

    // Max error should still be 0 for valid pixels
  auto errors = estimator.estimate_all();
  EXPECT_NEAR(errors[0].max_pixel_error, 0.0, 1e-10);
}

TEST_F(PixelMaxErrorEstimatorTest, GlobalMaxError) {
    // Element 0: max error = 1.0, Element 1: max error = 3.0
  auto data = create_test_bathymetry(10, 10, 0.0, 100.0, 0.0, 100.0, [](double x, double /*y*/) { return x < 50.0 ? -1.0f : -3.0f; });

  QuadtreeAdapter mesh;
  mesh.build_uniform(0.0, 100.0, 0.0, 100.0, 2, 1);

    // Test with PixelMaxErrorEstimator which has global_max_error() method
  LinearBezierSurface lin_surface(mesh);
  lin_surface.fit(data);

  PixelMaxErrorEstimator max_estimator(lin_surface, data, mesh);

    // Global max should be the maximum across all elements
    // Note: with a fitted surface the error may not be exactly what we set above
    // but global_max_error() should return the max of all per-element max errors
  Real global_max = max_estimator.global_max_error();
  auto errors     = max_estimator.estimate_all();

  Real manual_max = 0.0;
  for (const auto &err : errors) {
    manual_max = std::max(manual_max, err.error);
  }

  EXPECT_NEAR(global_max, manual_max, 1e-10);
}

TEST_F(PixelMaxErrorEstimatorTest, ElementErrorInterface) {
    // Test that PixelMaxErrorEstimator correctly implements ElementErrorEstimator interface
  auto data = create_test_bathymetry(10, 10, 0.0, 100.0, 0.0, 100.0, [](double x, double /*y*/) { return x < 50.0 ? -2.0f : -4.0f; });

  QuadtreeAdapter mesh;
  mesh.build_uniform(0.0, 100.0, 0.0, 100.0, 2, 1);

    // Fit a linear surface to use with PixelMaxErrorEstimator
  LinearBezierSurface lin_surface(mesh);
  lin_surface.fit(data);

  PixelMaxErrorEstimator estimator(lin_surface, data, mesh);

    // Test interface methods
  EXPECT_EQ(estimator.num_elements(), 2);

    // Test estimate_element returns ElementError with max_pixel_error as error
  auto elem_err = estimator.estimate_element(0);
  EXPECT_EQ(elem_err.element, 0);
  EXPECT_GT(elem_err.sample_count, 0);
  EXPECT_GT(elem_err.area, 0.0);

    // max_error() should return the maximum error across all elements
    // (this is inherited from ElementErrorEstimator base class)
  Real max_err    = estimator.max_error();
  auto all_errors = estimator.estimate_all();

  Real expected_max = 0.0;
  for (const auto &err : all_errors) {
    expected_max = std::max(expected_max, err.error);
  }
  EXPECT_NEAR(max_err, expected_max, 1e-10);
}

TEST_F(PixelMaxErrorEstimatorTest, DepthPositiveConvention) {
    // Surface returns -50 (elevation, 50m below sea level)
  ConstantSurface surface(-50.0);

    // Create bathymetry with depth convention (positive = water depth)
  auto data              = create_test_bathymetry(10, 10, 0.0, 100.0, 0.0, 100.0, [](double /*x*/, double /*y*/) { return 50.0f; });
  data.is_depth_positive = true; // Switch to depth convention

  QuadtreeAdapter mesh;
  mesh.build_uniform(0.0, 100.0, 0.0, 100.0, 1, 1);

  PixelErrorEstimator estimator(surface, data, mesh);

    // Depth 50 should convert to elevation -50, matching surface
  auto errors = estimator.estimate_all();
  EXPECT_NEAR(errors[0].max_pixel_error, 0.0, 1e-10);
}

TEST_F(PixelMaxErrorEstimatorTest, ZeroPixelElement) {
  ConstantSurface surface(0.0);

    // Create a 2x2 raster in the bottom-left corner only
  BathymetryData data;
  data.sizex             = 2;
  data.sizey             = 2;
  data.nodata_value      = -9999.0f;
  data.is_depth_positive = false;

    // Raster covers (5, 5) to (15, 15) - entirely in first element
  data.geotransform[0] = 5.0;  // xmin
  data.geotransform[1] = 5.0;  // dx (pixel width)
  data.geotransform[2] = 0.0;
  data.geotransform[3] = 15.0; // ymax
  data.geotransform[4] = 0.0;
  data.geotransform[5] = -5.0; // dy (negative)

  data.xmin = 5.0;
  data.xmax = 15.0;
  data.ymin = 5.0;
  data.ymax = 15.0;

  data.elevation = {-1.0f, -1.0f, -1.0f, -1.0f};

    // 4 elements (2x2): element 0 covers (0,0) to (50,50), others are outside raster
  QuadtreeAdapter mesh;
  mesh.build_uniform(0.0, 100.0, 0.0, 100.0, 2, 2);

  PixelErrorEstimator estimator(surface, data, mesh);

  auto errors          = estimator.estimate_all();
  int zero_pixel_count = 0;
  for (const auto &err : errors) {
    if (err.pixel_count == 0) {
            // Elements with zero pixels should have max_pixel_error = 0
      EXPECT_NEAR(err.max_pixel_error, 0.0, 1e-10);
      zero_pixel_count++;
    }
  }
  EXPECT_EQ(zero_pixel_count, 3);
}
