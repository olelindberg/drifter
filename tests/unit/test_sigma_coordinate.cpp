#include <gtest/gtest.h>
#include "mesh/sigma_coordinate.hpp"
#include "../test_utils.hpp"
#include <cmath>

using namespace drifter;
using namespace drifter::testing;

class SigmaCoordinateTest : public DrifterTestBase {
protected:
    void SetUp() override {
        DrifterTestBase::SetUp();
    }
};

// Test sigma to z transformation at boundaries
TEST_F(SigmaCoordinateTest, SigmaToZBoundaries) {
    Real eta = 1.0;  // Surface elevation
    Real h = 10.0;   // Bathymetry depth

    // At sigma = 0 (surface): z = eta
    Real z_surface = SigmaCoordinate::sigma_to_z(0.0, eta, h);
    EXPECT_NEAR(z_surface, eta, TOLERANCE);

    // At sigma = -1 (bottom): z = -h
    Real z_bottom = SigmaCoordinate::sigma_to_z(-1.0, eta, h);
    EXPECT_NEAR(z_bottom, -h, TOLERANCE);
}

// Test z to sigma transformation
TEST_F(SigmaCoordinateTest, ZToSigma) {
    Real eta = 2.0;
    Real h = 20.0;

    // At z = eta (surface): sigma = 0
    Real sigma_surface = SigmaCoordinate::z_to_sigma(eta, eta, h);
    EXPECT_NEAR(sigma_surface, 0.0, TOLERANCE);

    // At z = -h (bottom): sigma = -1
    Real sigma_bottom = SigmaCoordinate::z_to_sigma(-h, eta, h);
    EXPECT_NEAR(sigma_bottom, -1.0, TOLERANCE);

    // At mid-depth: sigma = -0.5
    Real z_mid = 0.5 * (eta - h);
    Real sigma_mid = SigmaCoordinate::z_to_sigma(z_mid, eta, h);
    EXPECT_NEAR(sigma_mid, -0.5, TOLERANCE);
}

// Test roundtrip sigma -> z -> sigma
TEST_F(SigmaCoordinateTest, Roundtrip) {
    Real eta = 1.5;
    Real h = 15.0;

    for (Real sigma = -1.0; sigma <= 0.0; sigma += 0.1) {
        Real z = SigmaCoordinate::sigma_to_z(sigma, eta, h);
        Real sigma_back = SigmaCoordinate::z_to_sigma(z, eta, h);
        EXPECT_NEAR(sigma_back, sigma, TOLERANCE) << "sigma = " << sigma;
    }
}

// Test total water depth
TEST_F(SigmaCoordinateTest, TotalDepth) {
    Real eta = 2.0;
    Real h = 18.0;

    Real H = SigmaCoordinate::total_depth(eta, h);
    EXPECT_NEAR(H, 20.0, TOLERANCE);
}

// Test dsigma/dx metric term
TEST_F(SigmaCoordinateTest, DSigmaDx) {
    Real sigma = -0.5;
    Real eta = 1.0;
    Real h = 10.0;
    Real deta_dx = 0.1;
    Real dh_dx = 0.2;

    Real dsigma_dx = SigmaCoordinate::dsigma_dx(sigma, eta, h, deta_dx, dh_dx);

    // Manual calculation:
    // H = eta + h = 11
    // dsigma/dx = -1/H * (sigma * dH/dx + deta/dx)
    //           = -1/H * (sigma * (deta/dx + dh/dx) + deta/dx)
    // Note: dH/dx = deta/dx (fixed bathymetry) but we're given dh_dx as well
    Real H = eta + h;
    Real expected = -(1.0 / H) * ((1.0 + sigma) * deta_dx + sigma * dh_dx);

    EXPECT_NEAR(dsigma_dx, expected, LOOSE_TOLERANCE);
}

// Test uniform stretching (no change)
TEST_F(SigmaCoordinateTest, UniformStretching) {
    SigmaStretchParams params;

    for (Real sigma = -1.0; sigma <= 0.0; sigma += 0.2) {
        Real stretched = SigmaCoordinate::apply_stretching(
            sigma, SigmaStretchType::Uniform, params);
        EXPECT_NEAR(stretched, sigma, TOLERANCE) << "sigma = " << sigma;
    }
}

// Test surface stretching concentrates nodes near surface
TEST_F(SigmaCoordinateTest, SurfaceStretching) {
    SigmaStretchParams params;
    params.theta_s = 5.0;  // Strong surface stretching

    // With surface stretching, sigma values should be biased toward surface (sigma=0)
    Real sigma_uniform = -0.5;
    Real sigma_stretched = SigmaCoordinate::apply_stretching(
        sigma_uniform, SigmaStretchType::Surface, params);

    // Surface stretching should push values toward 0 (surface)
    // So stretched sigma should be > uniform sigma (closer to 0)
    EXPECT_GT(sigma_stretched, sigma_uniform);
}

// Test bottom stretching concentrates nodes near bottom
TEST_F(SigmaCoordinateTest, BottomStretching) {
    SigmaStretchParams params;
    params.theta_b = 5.0;  // Strong bottom stretching

    // With bottom stretching, sigma values should be biased toward bottom (sigma=-1)
    Real sigma_uniform = -0.5;
    Real sigma_stretched = SigmaCoordinate::apply_stretching(
        sigma_uniform, SigmaStretchType::Bottom, params);

    // Bottom stretching should push values toward -1 (bottom)
    // So stretched sigma should be < uniform sigma (closer to -1)
    EXPECT_LT(sigma_stretched, sigma_uniform);
}

// Test that stretching preserves endpoints
TEST_F(SigmaCoordinateTest, StretchingPreservesEndpoints) {
    SigmaStretchParams params;
    params.theta_s = 3.0;
    params.theta_b = 2.0;

    std::vector<SigmaStretchType> types = {
        SigmaStretchType::Uniform,
        SigmaStretchType::Surface,
        SigmaStretchType::Bottom,
        SigmaStretchType::SurfaceBottom,
        SigmaStretchType::ROMS
    };

    for (auto type : types) {
        // Surface (sigma = 0) should remain at 0
        Real at_surface = SigmaCoordinate::apply_stretching(0.0, type, params);
        EXPECT_NEAR(at_surface, 0.0, TOLERANCE);

        // Bottom (sigma = -1) should remain at -1
        Real at_bottom = SigmaCoordinate::apply_stretching(-1.0, type, params);
        EXPECT_NEAR(at_bottom, -1.0, TOLERANCE);
    }
}

// Test SigmaMetrics class
TEST_F(SigmaCoordinateTest, SigmaMetricsComputation) {
    // Create simple test data
    int n = 5;
    VecX sigma(n), eta(n), h(n), deta_dx(n), deta_dy(n), dh_dx(n), dh_dy(n), deta_dt(n);

    for (int i = 0; i < n; ++i) {
        sigma(i) = -1.0 + 1.0 * i / (n - 1);  // From -1 to 0
        eta(i) = 1.0 + 0.1 * i;
        h(i) = 10.0 + i;
        deta_dx(i) = 0.01;
        deta_dy(i) = 0.02;
        dh_dx(i) = 0.0;  // Flat bottom
        dh_dy(i) = 0.0;
        deta_dt(i) = 0.0;
    }

    SigmaMetrics metrics;
    metrics.resize(n);
    metrics.update(sigma, eta, h, deta_dx, deta_dy, dh_dx, dh_dy, deta_dt);

    // Check H = eta + h
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(metrics.H()(i), eta(i) + h(i), TOLERANCE);
    }
}

// Test Jacobian computation
TEST_F(SigmaCoordinateTest, JacobianDeterminant) {
    Real eta = 1.0;
    Real h = 10.0;

    // For flat surface and bathymetry, Jacobian = H (total depth)
    Real H = SigmaCoordinate::total_depth(eta, h);

    // dz/dsigma = H, so Jacobian determinant for 1D vertical = H
    Real dz_dsigma = SigmaCoordinate::dz_dsigma(eta, h);
    EXPECT_NEAR(dz_dsigma, H, TOLERANCE);
}
