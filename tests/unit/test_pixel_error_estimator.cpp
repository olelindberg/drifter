#include "bathymetry/pixel_error_estimator.hpp"
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

/// Mock surface that returns a linear function z = a*x + b*y + c
class LinearSurface {
public:
    LinearSurface(Real a, Real b, Real c) : a_(a), b_(b), c_(c) {}
    Real evaluate(Real x, Real y) const { return a_ * x + b_ * y + c_; }

private:
    Real a_, b_, c_;
};

/// Mock surface that returns a quadratic function
class QuadraticSurface {
public:
    QuadraticSurface(Real a, Real b) : a_(a), b_(b) {}
    Real evaluate(Real x, Real y) const { return a_ * x * x + b_ * y * y; }

private:
    Real a_, b_;
};

/// Create synthetic BathymetryData for testing
BathymetryData create_test_bathymetry(int sizex, int sizey, Real xmin, Real xmax, Real ymin,
                                      Real ymax, std::function<float(double, double)> elevation_func,
                                      float nodata_value = -9999.0f) {
    BathymetryData data;
    data.sizex = sizex;
    data.sizey = sizey;
    data.nodata_value = nodata_value;
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

class PixelErrorEstimatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a 10x10 pixel domain from (0,0) to (100,100)
        // Each pixel is 10m x 10m
    }
};

TEST_F(PixelErrorEstimatorTest, ConstantSurfaceZeroError) {
    // Surface returns constant -50.0 (50m depth as elevation)
    ConstantSurface surface(-50.0);

    // Create bathymetry with matching constant values
    auto data = create_test_bathymetry(10, 10, 0.0, 100.0, 0.0, 100.0,
                                       [](double /*x*/, double /*y*/) { return -50.0f; });

    // Create single-element mesh covering entire domain
    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 100.0, 0.0, 100.0, 1, 1);

    PixelErrorEstimator estimator(surface, data, mesh);

    // Error should be exactly zero
    EXPECT_NEAR(estimator.global_rmse(), 0.0, 1e-10);
    EXPECT_NEAR(estimator.max_element_rmse(), 0.0, 1e-10);

    // Should have 100 pixels (10x10)
    EXPECT_EQ(estimator.total_pixel_count(), 100);
}

TEST_F(PixelErrorEstimatorTest, LinearSurfaceZeroError) {
    // Surface returns z = 0.1*x + 0.2*y - 100
    LinearSurface surface(0.1, 0.2, -100.0);

    // Create bathymetry with matching linear function
    auto data = create_test_bathymetry(
        10, 10, 0.0, 100.0, 0.0, 100.0,
        [](double x, double y) { return static_cast<float>(0.1 * x + 0.2 * y - 100.0); });

    // Create 2x2 mesh
    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 100.0, 0.0, 100.0, 2, 2);

    PixelErrorEstimator estimator(surface, data, mesh);

    // Error should be zero (surface matches data exactly)
    EXPECT_NEAR(estimator.global_rmse(), 0.0, 1e-10);
    EXPECT_NEAR(estimator.max_element_rmse(), 0.0, 1e-10);
}

TEST_F(PixelErrorEstimatorTest, KnownErrorMagnitude) {
    // Surface returns 0.0 everywhere
    ConstantSurface surface(0.0);

    // Create bathymetry with constant -1.0 (1m error at every pixel)
    auto data = create_test_bathymetry(10, 10, 0.0, 100.0, 0.0, 100.0,
                                       [](double /*x*/, double /*y*/) { return -1.0f; });

    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 100.0, 0.0, 100.0, 1, 1);

    PixelErrorEstimator estimator(surface, data, mesh);

    // RMSE should be 1.0 (sqrt(1^2) = 1)
    EXPECT_NEAR(estimator.global_rmse(), 1.0, 1e-10);
    EXPECT_NEAR(estimator.max_element_rmse(), 1.0, 1e-10);
}

TEST_F(PixelErrorEstimatorTest, PixelCountAccuracy) {
    ConstantSurface surface(0.0);

    // 20x20 = 400 pixels
    auto data = create_test_bathymetry(20, 20, 0.0, 100.0, 0.0, 100.0,
                                       [](double /*x*/, double /*y*/) { return 0.0f; });

    // 4x4 = 16 elements, each should have 25 pixels
    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 100.0, 0.0, 100.0, 4, 4);

    PixelErrorEstimator estimator(surface, data, mesh);

    EXPECT_EQ(estimator.total_pixel_count(), 400);

    // Check individual element pixel counts
    auto errors = estimator.estimate_all();
    EXPECT_EQ(errors.size(), 16u);

    // Each 25x25m element contains 5x5=25 pixels
    for (const auto& err : errors) {
        EXPECT_EQ(err.pixel_count, 25);
    }
}

TEST_F(PixelErrorEstimatorTest, NoDataHandling) {
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
    mesh.build_uniform(0.0, 100.0, 0.0, 100.0, 2, 2);

    PixelErrorEstimator estimator(surface, data, mesh);

    // Should count only 75 pixels (100 - 25 NoData)
    EXPECT_EQ(estimator.total_pixel_count(), 75);

    // Global RMSE should still be 0 for valid pixels
    EXPECT_NEAR(estimator.global_rmse(), 0.0, 1e-10);
}

TEST_F(PixelErrorEstimatorTest, UndersampledElements) {
    ConstantSurface surface(0.0);

    // Only 4 pixels (2x2) in domain
    auto data = create_test_bathymetry(2, 2, 0.0, 100.0, 0.0, 100.0,
                                       [](double /*x*/, double /*y*/) { return 0.0f; });

    // 4 elements (2x2), each has only 1 pixel
    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 100.0, 0.0, 100.0, 2, 2);

    PixelErrorEstimator estimator(surface, data, mesh);

    // Each element has only 1 pixel, so all are undersampled (< 4 pixels)
    EXPECT_EQ(estimator.count_undersampled_elements(4), 4);
    EXPECT_EQ(estimator.count_undersampled_elements(2), 4);
    EXPECT_EQ(estimator.count_undersampled_elements(1), 0);
}

TEST_F(PixelErrorEstimatorTest, MinRecommendedElementSize) {
    ConstantSurface surface(0.0);

    // 10x10 pixels over 100x100m domain = 10m pixel spacing
    auto data = create_test_bathymetry(10, 10, 0.0, 100.0, 0.0, 100.0,
                                       [](double /*x*/, double /*y*/) { return 0.0f; });

    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 100.0, 0.0, 100.0, 1, 1);

    PixelErrorEstimator estimator(surface, data, mesh);

    EXPECT_NEAR(estimator.min_recommended_element_size(), 10.0, 1e-10);
}

TEST_F(PixelErrorEstimatorTest, QuadraticSurfaceNonZeroError) {
    // Linear surface approximation
    LinearSurface surface(0.0, 0.0, 0.0); // z = 0 everywhere

    // Bathymetry with quadratic variation
    auto data = create_test_bathymetry(10, 10, 0.0, 100.0, 0.0, 100.0, [](double x, double y) {
        // Quadratic: z = 0.001 * (x-50)^2 + 0.001 * (y-50)^2
        double dx = x - 50.0;
        double dy = y - 50.0;
        return static_cast<float>(0.001 * dx * dx + 0.001 * dy * dy);
    });

    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 100.0, 0.0, 100.0, 1, 1);

    PixelErrorEstimator estimator(surface, data, mesh);

    // Error should be non-zero (linear can't capture quadratic)
    EXPECT_GT(estimator.global_rmse(), 0.0);
}

TEST_F(PixelErrorEstimatorTest, ElementErrorSumSquaredCorrect) {
    ConstantSurface surface(0.0);

    // Create bathymetry where first half has -1.0 and second half has -2.0
    auto data = create_test_bathymetry(10, 10, 0.0, 100.0, 0.0, 100.0, [](double x, double /*y*/) {
        return x < 50.0 ? -1.0f : -2.0f;
    });

    // 2 elements side by side
    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 100.0, 0.0, 100.0, 2, 1);

    PixelErrorEstimator estimator(surface, data, mesh);

    // Element 0 (x < 50): 50 pixels * 1^2 = 50, RMSE = sqrt(50/50) = 1.0
    // Element 1 (x >= 50): 50 pixels * 2^2 = 200, RMSE = sqrt(200/50) = 2.0
    auto errors = estimator.estimate_all();
    EXPECT_EQ(errors.size(), 2u);

    EXPECT_NEAR(errors[0].pixel_rmse, 1.0, 1e-10);
    EXPECT_NEAR(errors[1].pixel_rmse, 2.0, 1e-10);

    // Global RMSE: sqrt((50*1 + 200) / 100) = sqrt(250/100) = sqrt(2.5)
    EXPECT_NEAR(estimator.global_rmse(), std::sqrt(2.5), 1e-10);
}

TEST_F(PixelErrorEstimatorTest, DepthPositiveConvention) {
    // Surface returns -50 (elevation, 50m below sea level)
    ConstantSurface surface(-50.0);

    // Create bathymetry with depth convention (positive = water depth)
    auto data = create_test_bathymetry(10, 10, 0.0, 100.0, 0.0, 100.0,
                                       [](double /*x*/, double /*y*/) { return 50.0f; });
    data.is_depth_positive = true; // Switch to depth convention

    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 100.0, 0.0, 100.0, 1, 1);

    PixelErrorEstimator estimator(surface, data, mesh);

    // Depth 50 should convert to elevation -50, matching surface
    EXPECT_NEAR(estimator.global_rmse(), 0.0, 1e-10);
}

TEST_F(PixelErrorEstimatorTest, ZeroPixelElement) {
    ConstantSurface surface(0.0);

    // Create a 2x2 raster in the bottom-left corner only
    // Pixels will be entirely within the first element
    BathymetryData data;
    data.sizex = 2;
    data.sizey = 2;
    data.nodata_value = -9999.0f;
    data.is_depth_positive = false;

    // Raster covers (5, 5) to (15, 15) - entirely in first element
    // Pixel centers at (7.5, 12.5), (12.5, 12.5), (7.5, 7.5), (12.5, 7.5)
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

    data.elevation = {0.0f, 0.0f, 0.0f, 0.0f};

    // 4 elements (2x2): element 0 covers (0,0) to (50,50), others are outside raster
    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 100.0, 0.0, 100.0, 2, 2);

    PixelErrorEstimator estimator(surface, data, mesh);

    // Only element 0 has pixels (all 4 pixels are in element 0)
    EXPECT_EQ(estimator.total_pixel_count(), 4);

    // 3 elements have 0 pixels, so they're undersampled at threshold 1
    EXPECT_EQ(estimator.count_undersampled_elements(1), 3);

    auto errors = estimator.estimate_all();
    int zero_pixel_count = 0;
    for (const auto& err : errors) {
        if (err.pixel_count == 0) {
            EXPECT_NEAR(err.pixel_rmse, 0.0, 1e-10); // No error for empty elements
            zero_pixel_count++;
        }
    }
    EXPECT_EQ(zero_pixel_count, 3);
}

