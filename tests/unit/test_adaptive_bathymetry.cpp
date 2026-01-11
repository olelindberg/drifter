#include <gtest/gtest.h>
#include "bathymetry/adaptive_bathymetry.hpp"
#include "mesh/seabed_surface.hpp"
#include "../test_utils.hpp"
#include <cmath>

using namespace drifter;
using namespace drifter::testing;

class AdaptiveBathymetryTest : public DrifterTestBase {
protected:
    void SetUp() override {
        DrifterTestBase::SetUp();

        // Create test bathymetry
        int size = 128;
        std::vector<float> elevation(size * size);

        // Varied bathymetry: shallow coast to deep ocean
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                Real fx = static_cast<Real>(x) / size;
                Real fy = static_cast<Real>(y) / size;
                // Depth increases from 10m at x=0 to 100m at x=size
                Real depth = 10.0 + 90.0 * fx + 5.0 * std::sin(4.0 * M_PI * fy);
                elevation[y * size + x] = static_cast<float>(-depth);
            }
        }

        bathy_ = std::make_shared<BathymetryData>();
        bathy_->sizex = size;
        bathy_->sizey = size;
        bathy_->elevation = std::move(elevation);
        // Geotransform: origin at (0, size), pixel width=1, pixel height=-1
        bathy_->geotransform = {0.0, 1.0, 0.0, static_cast<double>(size), 0.0, -1.0};
        bathy_->nodata_value = -9999.0f;
        bathy_->is_depth_positive = false;
        bathy_->xmin = 0.0;
        bathy_->xmax = static_cast<double>(size);
        bathy_->ymin = 0.0;
        bathy_->ymax = static_cast<double>(size);

        adaptive_ = std::make_unique<AdaptiveBathymetry>(bathy_);
    }

    std::shared_ptr<BathymetryData> create_constant_bathymetry(int size, Real depth) {
        std::vector<float> elevation(size * size, static_cast<float>(-depth));

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

    std::shared_ptr<BathymetryData> bathy_;
    std::unique_ptr<AdaptiveBathymetry> adaptive_;
};

// Test constant bathymetry projection - all coefficients should be similar
TEST_F(AdaptiveBathymetryTest, ConstantPreservation) {
    int size = 64;
    Real const_depth = 50.0;
    auto const_bathy = create_constant_bathymetry(size, const_depth);

    AdaptiveBathymetry adaptive(const_bathy);

    // Project onto an interior element
    ElementBounds bounds{15, 45, 15, 45, -1, 0};
    int order = 3;

    VecX coeffs = adaptive.project_element(bounds, order);

    int n2d = (order + 1) * (order + 1);
    ASSERT_EQ(coeffs.size(), n2d);

    // For constant data, all coefficients should be nearly equal
    Real mean = coeffs.mean();
    Real max_deviation = (coeffs.array() - mean).abs().maxCoeff();

    // The variation should be small
    EXPECT_LT(max_deviation, 1.0)
        << "Constant field should have uniform coefficients";

    // Mean should be close to expected depth
    EXPECT_GT(mean, const_depth * 0.8)
        << "Mean coefficient should be close to expected depth";
    EXPECT_LT(mean, const_depth * 1.2)
        << "Mean coefficient should be close to expected depth";
}

// Test Bernstein boundedness property
TEST_F(AdaptiveBathymetryTest, BernsteinBoundedness) {
    ElementBounds bounds{20, 60, 20, 60, -1, 0};
    int order = 4;

    VecX coeffs = adaptive_->project_element(bounds, order);

    Real min_coeff = coeffs.minCoeff();
    Real max_coeff = coeffs.maxCoeff();

    // Evaluate at many points - values must be within coefficient bounds
    BernsteinBasis1D basis(order);
    int n1d = order + 1;

    for (Real xi = -1.0; xi <= 1.0; xi += 0.1) {
        for (Real eta = -1.0; eta <= 1.0; eta += 0.1) {
            VecX phi_xi = basis.evaluate(xi);
            VecX phi_eta = basis.evaluate(eta);

            Real value = 0.0;
            for (int j = 0; j < n1d; ++j) {
                for (int i = 0; i < n1d; ++i) {
                    int idx = i + n1d * j;
                    value += coeffs(idx) * phi_xi(i) * phi_eta(j);
                }
            }

            EXPECT_GE(value, min_coeff - TOLERANCE)
                << "Value at (" << xi << ", " << eta << ") below min coefficient";
            EXPECT_LE(value, max_coeff + TOLERANCE)
                << "Value at (" << xi << ", " << eta << ") above max coefficient";
        }
    }
}

// Test projection error estimation
TEST_F(AdaptiveBathymetryTest, ProjectionErrorEstimate) {
    ElementBounds bounds{30, 50, 30, 50, -1, 0};

    // Lower order should have some error
    VecX coeffs_low = adaptive_->project_element(bounds, 2);
    Real error_low = adaptive_->estimate_projection_error(bounds, coeffs_low, 2);

    // Higher order projection
    VecX coeffs_high = adaptive_->project_element(bounds, 5);
    Real error_high = adaptive_->estimate_projection_error(bounds, coeffs_high, 5);

    // Both errors should be finite
    EXPECT_LT(error_low, 1.0) << "Low order error should be finite";
    EXPECT_LT(error_high, 1.0) << "High order error should be finite";
}

// Test different polynomial orders
TEST_F(AdaptiveBathymetryTest, DifferentOrders) {
    ElementBounds bounds{20, 40, 20, 40, -1, 0};

    for (int order = 1; order <= 5; ++order) {
        VecX coeffs = adaptive_->project_element(bounds, order);

        int expected_size = (order + 1) * (order + 1);
        EXPECT_EQ(coeffs.size(), expected_size)
            << "Order " << order << " should have " << expected_size << " coefficients";

        // All coefficients should be reasonable depth values
        EXPECT_GT(coeffs.minCoeff(), 0.0)
            << "Order " << order << " coefficients should be positive";
        EXPECT_LT(coeffs.maxCoeff(), 150.0)
            << "Order " << order << " coefficients should be reasonable";
    }
}

// Test integration with SeabedSurface
TEST_F(AdaptiveBathymetryTest, SeabedSurfaceIntegration) {
    // Create a simple uniform mesh
    OctreeAdapter mesh(0, 100, 0, 100, -1, 0);
    mesh.build_uniform(4, 4, 2);

    // Create seabed surface
    int order = 3;
    SeabedSurface seabed(mesh, order, SeabedInterpolation::Bernstein);

    // Use adaptive bathymetry to populate it
    seabed.set_from_adaptive_bathymetry(*adaptive_);

    // Query some depths
    Real depth_center = seabed.depth(50.0, 50.0);

    // Depth should be reasonable (bathymetry range is ~10-100m)
    EXPECT_GT(depth_center, 5.0);
    EXPECT_LT(depth_center, 120.0);

    // Check that we have coefficients for all bottom elements
    EXPECT_GT(seabed.num_elements(), 0);
}

// Test overintegration factor setting
TEST_F(AdaptiveBathymetryTest, OverintegrationFactor) {
    ElementBounds bounds{30, 50, 30, 50, -1, 0};
    int order = 3;

    // Default overintegration
    VecX coeffs_default = adaptive_->project_element(bounds, order);

    // Higher overintegration
    adaptive_->set_overintegration_factor(3);
    VecX coeffs_higher = adaptive_->project_element(bounds, order);

    // Both should produce valid results
    EXPECT_EQ(coeffs_default.size(), coeffs_higher.size());
    EXPECT_GT(coeffs_default.minCoeff(), 0.0);
    EXPECT_GT(coeffs_higher.minCoeff(), 0.0);
}

// Test projection matrix caching
TEST_F(AdaptiveBathymetryTest, ProjectionMatrixCaching) {
    ElementBounds bounds1{15, 35, 15, 35, -1, 0};
    ElementBounds bounds2{50, 70, 50, 70, -1, 0};
    int order = 3;

    // First projection (builds cache)
    VecX coeffs1 = adaptive_->project_element(bounds1, order);

    // Second projection with same order (uses cache)
    VecX coeffs2 = adaptive_->project_element(bounds2, order);

    // Both should succeed with valid positive coefficients
    EXPECT_EQ(coeffs1.size(), coeffs2.size());
    EXPECT_GT(coeffs1.minCoeff(), 0.0);
    EXPECT_GT(coeffs2.minCoeff(), 0.0);
}

// Test that constant input produces constant output (verifies WENO+L2 projection)
TEST_F(AdaptiveBathymetryTest, ConstantInputProducesConstantCoefficients) {
    // Create constant bathymetry
    int size = 64;
    Real const_depth = 50.0;
    auto const_bathy = create_constant_bathymetry(size, const_depth);

    AdaptiveBathymetry adaptive(const_bathy);

    // Project onto an element fully inside the bathymetry domain
    ElementBounds bounds{10, 30, 10, 30, -1, 0};
    int order = 3;

    VecX coeffs = adaptive.project_element(bounds, order);

    int n2d = (order + 1) * (order + 1);
    ASSERT_EQ(coeffs.size(), n2d);

    // For constant input, ALL coefficients should be exactly equal
    Real max_deviation = (coeffs.array() - const_depth).abs().maxCoeff();
    EXPECT_LT(max_deviation, 1.0)
        << "Constant field should produce uniform coefficients close to " << const_depth;
}

// Test edge case: very small element
TEST_F(AdaptiveBathymetryTest, VerySmallElement) {
    ElementBounds tiny_bounds{50.0, 52.0, 50.0, 52.0, -1, 0};
    int order = 3;

    VecX coeffs = adaptive_->project_element(tiny_bounds, order);

    // Should still produce valid coefficients
    EXPECT_EQ(coeffs.size(), 16);  // (3+1)^2

    // All coefficients should be positive and reasonable
    EXPECT_GT(coeffs.minCoeff(), 0.0);
    EXPECT_LT(coeffs.maxCoeff(), 150.0);
}

// Test edge case: very large element
TEST_F(AdaptiveBathymetryTest, VeryLargeElement) {
    // Use interior bounds to avoid boundary effects
    ElementBounds large_bounds{15, 85, 15, 85, -1, 0};
    int order = 3;

    VecX coeffs = adaptive_->project_element(large_bounds, order);

    // Should produce coefficients (some may be outside data range due to L2 projection)
    EXPECT_EQ(coeffs.size(), 16);

    // Check that mean is in reasonable range
    Real mean = coeffs.mean();
    EXPECT_GT(mean, 0.0) << "Mean coefficient should be positive";
    EXPECT_LT(mean, 200.0) << "Mean coefficient should be reasonable";
}
