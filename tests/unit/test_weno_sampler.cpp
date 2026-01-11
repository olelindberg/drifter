#include <gtest/gtest.h>
#include "bathymetry/weno_sampler.hpp"
#include "../test_utils.hpp"
#include <cmath>

using namespace drifter;
using namespace drifter::testing;

class WENO5SamplerTest : public DrifterTestBase {
protected:
    void SetUp() override {
        DrifterTestBase::SetUp();

        // Create a test bathymetry dataset
        int size = 128;
        std::vector<float> elevation(size * size);

        // Smooth bathymetry: h(x,y) = 50 + 20*sin(pi*x/128)*cos(pi*y/128)
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                Real fx = static_cast<Real>(x) / size;
                Real fy = static_cast<Real>(y) / size;
                // Store as negative elevation (depth is positive)
                elevation[y * size + x] = static_cast<float>(
                    -(50.0 + 20.0 * std::sin(M_PI * fx) * std::cos(M_PI * fy)));
            }
        }

        smooth_bathy_ = std::make_shared<BathymetryData>();
        smooth_bathy_->sizex = size;
        smooth_bathy_->sizey = size;
        smooth_bathy_->elevation = std::move(elevation);
        smooth_bathy_->geotransform = {0.0, 1.0, 0.0, static_cast<double>(size), 0.0, -1.0};
        smooth_bathy_->nodata_value = -9999.0f;
        smooth_bathy_->is_depth_positive = false;
        smooth_bathy_->xmin = 0.0;
        smooth_bathy_->xmax = static_cast<double>(size);
        smooth_bathy_->ymin = 0.0;
        smooth_bathy_->ymax = static_cast<double>(size);
    }

    std::shared_ptr<BathymetryData> create_step_bathymetry(int size) {
        std::vector<float> elevation(size * size);

        // Step function: shallow on left, deep on right
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                elevation[y * size + x] = (x < size / 2) ? -20.0f : -80.0f;
            }
        }

        auto bathy = std::make_shared<BathymetryData>();
        bathy->sizex = size;
        bathy->sizey = size;
        bathy->elevation = std::move(elevation);
        bathy->geotransform = {0.0, 1.0, 0.0, static_cast<double>(size), 0.0, -1.0};
        bathy->nodata_value = -9999.0f;
        bathy->is_depth_positive = false;
        bathy->xmin = 0.0;
        bathy->xmax = static_cast<double>(size);
        bathy->ymin = 0.0;
        bathy->ymax = static_cast<double>(size);

        return bathy;
    }

    std::shared_ptr<BathymetryData> smooth_bathy_;
};

// Test basic sampling in smooth region
TEST_F(WENO5SamplerTest, SmoothSampling) {
    WENO5Sampler sampler(smooth_bathy_);

    // Sample at center of domain
    Real x = 64.0, y = 64.0;

    Real depth = sampler.sample(x, y);

    // Expected depth should be in the range of our test bathymetry: 30-70
    EXPECT_GT(depth, 30.0) << "Depth should be above minimum";
    EXPECT_LT(depth, 70.0) << "Depth should be below maximum";
}

// Test sampling near boundaries
TEST_F(WENO5SamplerTest, BoundarySampling) {
    WENO5Sampler sampler(smooth_bathy_);

    // Sample near corner - falls back to bilinear
    Real depth_corner = sampler.sample(5.0, 5.0);

    // Should get a valid depth, not crash
    EXPECT_GT(depth_corner, 0.0);
    EXPECT_LT(depth_corner, 100.0);
}

// Test WENO non-oscillatory property near discontinuity
TEST_F(WENO5SamplerTest, NonOscillatoryNearDiscontinuity) {
    int size = 128;
    auto step_bathy = create_step_bathymetry(size);

    WENO5Sampler sampler(step_bathy);

    Real h_shallow = 20.0;
    Real h_deep = 80.0;

    // Sample across the discontinuity
    for (Real x = 50.0; x <= 78.0; x += 2.0) {
        Real depth = sampler.sample(x, 64.0);

        // WENO should not overshoot beyond the data bounds
        EXPECT_GE(depth, h_shallow - 5.0)
            << "Depth at x=" << x << " undershoots: " << depth;
        EXPECT_LE(depth, h_deep + 5.0)
            << "Depth at x=" << x << " overshoots: " << depth;
    }
}

// Test gradient computation
TEST_F(WENO5SamplerTest, GradientComputation) {
    // Create bathymetry with known gradient: h = 0.1*x + 50
    int size = 128;
    std::vector<float> elevation(size * size);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            elevation[y * size + x] = static_cast<float>(-(0.1 * x + 50.0));
        }
    }

    auto linear_bathy = std::make_shared<BathymetryData>();
    linear_bathy->sizex = size;
    linear_bathy->sizey = size;
    linear_bathy->elevation = std::move(elevation);
    linear_bathy->geotransform = {0.0, 1.0, 0.0, static_cast<double>(size), 0.0, -1.0};
    linear_bathy->nodata_value = -9999.0f;
    linear_bathy->is_depth_positive = false;
    linear_bathy->xmin = 0.0;
    linear_bathy->xmax = static_cast<double>(size);
    linear_bathy->ymin = 0.0;
    linear_bathy->ymax = static_cast<double>(size);

    WENO5Sampler sampler(linear_bathy);

    Real dh_dx, dh_dy;
    sampler.sample_gradient(64.0, 64.0, dh_dx, dh_dy);

    // The gradient in x should be positive (depth increases with x)
    EXPECT_GT(dh_dx, 0.05) << "Gradient in x should be positive";
    EXPECT_LT(std::abs(dh_dy), 0.5) << "Gradient in y should be small";
}

// Test batch sampling
TEST_F(WENO5SamplerTest, BatchSampling) {
    WENO5Sampler sampler(smooth_bathy_);

    std::vector<Vec2> points = {
        {30.0, 30.0},
        {50.0, 50.0},
        {70.0, 70.0},
        {90.0, 90.0}
    };

    std::vector<Real> values;
    sampler.sample_batch(points, values);

    ASSERT_EQ(values.size(), 4);

    // All values should be reasonable depths
    for (size_t i = 0; i < values.size(); ++i) {
        EXPECT_GT(values[i], 0.0) << "Depth at point " << i << " should be positive";
        EXPECT_LT(values[i], 100.0) << "Depth at point " << i << " should be reasonable";
    }

    // Compare with individual samples
    for (size_t i = 0; i < points.size(); ++i) {
        Real single = sampler.sample(points[i](0), points[i](1));
        EXPECT_NEAR(values[i], single, 1e-10)
            << "Batch and single sampling should match at point " << i;
    }
}

// Test parameter setters
TEST_F(WENO5SamplerTest, ParameterSetters) {
    WENO5Sampler sampler(smooth_bathy_);

    // Default values
    EXPECT_NEAR(sampler.epsilon(), 1e-6, 1e-12);
    EXPECT_NEAR(sampler.power(), 2.0, 1e-12);

    // Set new values
    sampler.set_epsilon(1e-8);
    sampler.set_power(3.0);

    EXPECT_NEAR(sampler.epsilon(), 1e-8, 1e-12);
    EXPECT_NEAR(sampler.power(), 3.0, 1e-12);
}

// Test WENO5 accuracy for smooth data
TEST_F(WENO5SamplerTest, SmoothDataAccuracy) {
    // Create polynomial bathymetry that WENO5 should reproduce accurately
    int size = 128;
    std::vector<float> elevation(size * size);

    // h(x,y) = 50 + 0.01*x^2 (quadratic - WENO5 should handle well)
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            elevation[y * size + x] = static_cast<float>(-(50.0 + 0.01 * x * x));
        }
    }

    auto poly_bathy = std::make_shared<BathymetryData>();
    poly_bathy->sizex = size;
    poly_bathy->sizey = size;
    poly_bathy->elevation = std::move(elevation);
    poly_bathy->geotransform = {0.0, 1.0, 0.0, static_cast<double>(size), 0.0, -1.0};
    poly_bathy->nodata_value = -9999.0f;
    poly_bathy->is_depth_positive = false;
    poly_bathy->xmin = 0.0;
    poly_bathy->xmax = static_cast<double>(size);
    poly_bathy->ymin = 0.0;
    poly_bathy->ymax = static_cast<double>(size);

    WENO5Sampler sampler(poly_bathy);

    // Test at several interior points
    std::vector<Real> test_x = {30.0, 50.0, 70.0, 90.0};
    for (Real x : test_x) {
        Real depth = sampler.sample(x, 64.0);
        Real expected = 50.0 + 0.01 * x * x;

        EXPECT_NEAR(depth, expected, 2.0)
            << "Quadratic should be reproduced accurately at x=" << x;
    }
}
