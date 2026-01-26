#include "bathymetry/quadtree_adapter.hpp"
#include "bathymetry/thb_spline/thb_surface.hpp"
#include "mesh/geotiff_reader.hpp"
#include "mesh/octree_adapter.hpp"
#include "test_integration_fixtures.hpp"
#include <filesystem>
#include <gtest/gtest.h>
#include <cmath>
#include <iostream>

namespace drifter {
namespace {

constexpr Real TOLERANCE = 1e-6;
constexpr Real LOOSE_TOLERANCE = 1e-3;

// =============================================================================
// THBSurface Integration Tests
// =============================================================================

class THBSurfaceTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Create a simple uniform quadtree for testing
        quadtree_ = std::make_unique<QuadtreeAdapter>();
        quadtree_->build_uniform(0.0, 10.0, 0.0, 10.0, 4, 4);
    }

    std::unique_ptr<QuadtreeAdapter> quadtree_;
};

TEST_F(THBSurfaceTest, ConstructFromQuadtree) {
    THBSurfaceConfig config;
    config.verbose = false;

    THBSurface surface(*quadtree_, config);

    EXPECT_GT(surface.num_active_dofs(), 0);
    EXPECT_GE(surface.max_level(), 0);
    EXPECT_DOUBLE_EQ(surface.domain_min_x(), 0.0);
    EXPECT_DOUBLE_EQ(surface.domain_max_x(), 10.0);
}

TEST_F(THBSurfaceTest, FitConstantFunction) {
    THBSurfaceConfig config;
    config.ngauss = 4;
    config.verbose = false;

    THBSurface surface(*quadtree_, config);

    // Fit constant bathymetry
    const Real constant_depth = 100.0;
    surface.set_bathymetry_data([=](Real, Real) { return constant_depth; });
    surface.solve();

    EXPECT_TRUE(surface.is_solved());

    // Evaluate at several points - should return constant
    for (int j = 0; j < 10; ++j) {
        for (int i = 0; i < 10; ++i) {
            Real x = 0.5 + i;
            Real y = 0.5 + j;
            Real depth = surface.evaluate(x, y);
            EXPECT_NEAR(depth, constant_depth, TOLERANCE)
                << "at (" << x << ", " << y << ")";
        }
    }
}

TEST_F(THBSurfaceTest, FitLinearFunction) {
    THBSurfaceConfig config;
    config.ngauss = 4;
    config.verbose = false;

    THBSurface surface(*quadtree_, config);

    // Fit linear bathymetry: z = 10 + 2*x + 3*y
    auto linear = [](Real x, Real y) { return 10.0 + 2.0 * x + 3.0 * y; };
    surface.set_bathymetry_data(linear);
    surface.solve();

    EXPECT_TRUE(surface.is_solved());

    // Cubic B-splines can exactly represent linear functions
    for (int j = 0; j < 10; ++j) {
        for (int i = 0; i < 10; ++i) {
            Real x = 0.5 + i;
            Real y = 0.5 + j;
            Real expected = linear(x, y);
            Real actual = surface.evaluate(x, y);
            EXPECT_NEAR(actual, expected, TOLERANCE)
                << "at (" << x << ", " << y << ")";
        }
    }
}

TEST_F(THBSurfaceTest, FitQuadraticFunction) {
    THBSurfaceConfig config;
    config.ngauss = 6;  // More points for better fit
    config.verbose = false;

    THBSurface surface(*quadtree_, config);

    // Fit quadratic bathymetry: z = 100 - (x-5)^2 - (y-5)^2
    auto quadratic = [](Real x, Real y) {
        return 100.0 - (x - 5.0) * (x - 5.0) - (y - 5.0) * (y - 5.0);
    };
    surface.set_bathymetry_data(quadratic);
    surface.solve();

    EXPECT_TRUE(surface.is_solved());

    // Cubic B-splines can exactly represent quadratic functions
    for (int j = 1; j < 9; ++j) {
        for (int i = 1; i < 9; ++i) {
            Real x = 0.5 + i;
            Real y = 0.5 + j;
            Real expected = quadratic(x, y);
            Real actual = surface.evaluate(x, y);
            EXPECT_NEAR(actual, expected, LOOSE_TOLERANCE)
                << "at (" << x << ", " << y << ")";
        }
    }
}

TEST_F(THBSurfaceTest, SmoothingReducesOscillations) {
    THBSurfaceConfig config_no_smooth;
    config_no_smooth.smoothing_weight = 0.0;
    config_no_smooth.ngauss = 4;
    config_no_smooth.verbose = false;

    THBSurfaceConfig config_smooth;
    config_smooth.smoothing_weight = 0.1;
    config_smooth.ngauss = 4;
    config_smooth.verbose = false;

    THBSurface surface_no_smooth(*quadtree_, config_no_smooth);
    THBSurface surface_smooth(*quadtree_, config_smooth);

    // Noisy data
    auto noisy = [](Real x, Real y) {
        return 50.0 + 5.0 * std::sin(x * 2.0) * std::cos(y * 2.0) +
               2.0 * std::sin(x * 10.0) * std::sin(y * 10.0);
    };

    surface_no_smooth.set_bathymetry_data(noisy);
    surface_no_smooth.solve();

    surface_smooth.set_bathymetry_data(noisy);
    surface_smooth.solve();

    // Both should fit reasonably well at sample points
    // But smoothed version should have smoother gradients
    Real grad_sum_no_smooth = 0.0;
    Real grad_sum_smooth = 0.0;

    for (int j = 1; j < 9; ++j) {
        for (int i = 1; i < 9; ++i) {
            Real x = 0.5 + i;
            Real y = 0.5 + j;

            Vec2 grad_ns = surface_no_smooth.evaluate_gradient(x, y);
            Vec2 grad_s = surface_smooth.evaluate_gradient(x, y);

            grad_sum_no_smooth += grad_ns.squaredNorm();
            grad_sum_smooth += grad_s.squaredNorm();
        }
    }

    // Smoothed surface should have smaller gradient magnitude on average
    EXPECT_LT(grad_sum_smooth, grad_sum_no_smooth);
}

TEST_F(THBSurfaceTest, EvaluateBeforeSolveThrows) {
    THBSurfaceConfig config;
    THBSurface surface(*quadtree_, config);

    surface.set_bathymetry_data([](Real, Real) { return 50.0; });

    // Should throw because solve() hasn't been called
    EXPECT_THROW(surface.evaluate(5.0, 5.0), std::runtime_error);
}

TEST_F(THBSurfaceTest, SolveBeforeDataThrows) {
    THBSurfaceConfig config;
    THBSurface surface(*quadtree_, config);

    // Should throw because data hasn't been set
    EXPECT_THROW(surface.solve(), std::runtime_error);
}

// =============================================================================
// THBSurface with Adaptive Mesh Tests
// =============================================================================

class THBSurfaceAdaptiveTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Create a non-uniform quadtree: one coarse + four fine elements
        // Layout:
        // +-------+---+---+
        // |       | 3 | 4 |
        // |   1   +---+---+
        // |       | 2 | 5 |
        // +-------+---+---+
        quadtree_ = std::make_unique<QuadtreeAdapter>();

        QuadLevel level0 = {0, 0};
        QuadLevel level1 = {1, 1};

        // Coarse element on left half
        quadtree_->add_element(QuadBounds{0.0, 5.0, 0.0, 10.0}, level0);

        // Four fine elements on right half
        quadtree_->add_element(QuadBounds{5.0, 7.5, 0.0, 5.0}, level1);
        quadtree_->add_element(QuadBounds{5.0, 7.5, 5.0, 10.0}, level1);
        quadtree_->add_element(QuadBounds{7.5, 10.0, 0.0, 5.0}, level1);
        quadtree_->add_element(QuadBounds{7.5, 10.0, 5.0, 10.0}, level1);
    }

    std::unique_ptr<QuadtreeAdapter> quadtree_;
};

TEST_F(THBSurfaceAdaptiveTest, ConstructFromNonUniformMesh) {
    THBSurfaceConfig config;
    config.verbose = false;

    THBSurface surface(*quadtree_, config);

    // Should have multiple levels
    EXPECT_GT(surface.max_level(), 0);
    EXPECT_GT(surface.num_active_dofs(), 0);
}

TEST_F(THBSurfaceAdaptiveTest, FitConstantOnNonUniform) {
    THBSurfaceConfig config;
    // Need ngauss >= 5 to get enough data points (25) for the ~20 active
    // level 0 functions in the single coarse element
    config.ngauss = 5;
    config.verbose = false;

    THBSurface surface(*quadtree_, config);

    const Real constant = 75.0;
    surface.set_bathymetry_data([=](Real, Real) { return constant; });
    surface.solve();

    // Should still fit constant well
    for (int j = 0; j < 10; ++j) {
        for (int i = 0; i < 10; ++i) {
            Real x = 0.5 + i;
            Real y = 0.5 + j;
            Real depth = surface.evaluate(x, y);
            EXPECT_NEAR(depth, constant, TOLERANCE)
                << "at (" << x << ", " << y << ")";
        }
    }
}

TEST_F(THBSurfaceAdaptiveTest, SmoothWithinRegions) {
    // NOTE: With single-level THB evaluation, we have C² continuity WITHIN each
    // level region, but NOT across level boundaries (where we switch from level 0
    // to level 1 functions). The boundary at x=5 will have discontinuous derivatives.
    //
    // This is a known limitation of the simplified single-level approach.
    // Standard THB with truncation would maintain C² but has other issues with
    // boundary functions on bounded domains.

    THBSurfaceConfig config;
    config.ngauss = 5;
    config.verbose = false;

    THBSurface surface(*quadtree_, config);

    // Smooth function
    auto smooth = [](Real x, Real y) {
        return 50.0 + 10.0 * std::sin(x * 0.5) * std::cos(y * 0.5);
    };
    surface.set_bathymetry_data(smooth);
    surface.solve();

    const Real h = 0.01;

    // Check points WITHIN each level region (not on the level boundary at x=5)
    // These should have smooth second derivatives
    std::vector<std::pair<Real, Real>> test_points = {
        {2.5, 5.0},  // Center of coarse region
        {7.5, 5.0},  // Center of fine region
        {2.5, 2.5},  // Corner of coarse region
        {7.5, 7.5},  // Center of fine region
    };

    for (const auto& [x, y] : test_points) {
        Real f_xph = surface.evaluate(x + h, y);
        Real f_xmh = surface.evaluate(x - h, y);
        Real f_xy = surface.evaluate(x, y);
        Real f_yph = surface.evaluate(x, y + h);
        Real f_ymh = surface.evaluate(x, y - h);

        Real d2f_dx2 = (f_xph - 2.0 * f_xy + f_xmh) / (h * h);
        Real d2f_dy2 = (f_yph - 2.0 * f_xy + f_ymh) / (h * h);

        // Within regions, derivatives should be finite and reasonable
        EXPECT_TRUE(std::isfinite(d2f_dx2)) << "at (" << x << ", " << y << ")";
        EXPECT_TRUE(std::isfinite(d2f_dy2)) << "at (" << x << ", " << y << ")";
        EXPECT_LT(std::abs(d2f_dx2), 100.0) << "at (" << x << ", " << y << ")";
        EXPECT_LT(std::abs(d2f_dy2), 100.0) << "at (" << x << ", " << y << ")";
    }
}

// =============================================================================
// THBSurface from OctreeAdapter Tests
// =============================================================================

TEST(THBSurfaceOctreeTest, ConstructFromOctree) {
    // Create a simple octree
    OctreeAdapter octree(0.0, 10.0, 0.0, 10.0, -100.0, 0.0);
    octree.build_uniform(4, 4, 2);

    THBSurfaceConfig config;
    config.verbose = false;

    THBSurface surface(octree, config);

    EXPECT_GT(surface.num_active_dofs(), 0);
    EXPECT_DOUBLE_EQ(surface.domain_min_x(), 0.0);
    EXPECT_DOUBLE_EQ(surface.domain_max_x(), 10.0);
}

TEST(THBSurfaceOctreeTest, FitConstantFromOctree) {
    OctreeAdapter octree(0.0, 10.0, 0.0, 10.0, -100.0, 0.0);
    octree.build_uniform(3, 3, 2);

    THBSurfaceConfig config;
    config.ngauss = 4;
    config.verbose = false;

    THBSurface surface(octree, config);

    const Real depth = 50.0;
    surface.set_bathymetry_data([=](Real, Real) { return depth; });
    surface.solve();

    Real actual = surface.evaluate(5.0, 5.0);
    EXPECT_NEAR(actual, depth, TOLERANCE);
}

// =============================================================================
// THBSurface GeoTIFF Integration Tests
// =============================================================================

TEST(THBSurfaceGeoTiffTest, KattegatBathymetry30km) {
    // Skip if GDAL not available
    if (!GeoTiffReader::is_available()) {
        GTEST_SKIP() << "GDAL not available, skipping GeoTIFF test";
    }

    std::string geotiff_path = testing::BATHYMETRY_GEOTIFF_PATH;
    if (!std::filesystem::exists(geotiff_path)) {
        GTEST_SKIP() << "GeoTIFF file not found: " << geotiff_path;
    }

    std::cout << "\n=== THB-Spline Kattegat 30km Test ===\n";

    // Load bathymetry
    GeoTiffReader reader;
    BathymetryData bathy = reader.load(geotiff_path);
    ASSERT_TRUE(bathy.is_valid()) << "Failed to load GeoTIFF: " << reader.last_error();

    std::cout << "GeoTIFF loaded:\n";
    std::cout << "  Domain: [" << bathy.xmin << ", " << bathy.xmax << "] x ["
              << bathy.ymin << ", " << bathy.ymax << "]\n";

    // Define 30x30 km domain centered in Kattegat
    // Using EPSG:3034 coordinates (same as other tests)
    Real center_x = 4095238.0;
    Real center_y = 3344695.0;
    Real domain_size = 30000.0;  // 30 km
    Real half_size = domain_size / 2.0;

    Real xmin = center_x - half_size;
    Real xmax = center_x + half_size;
    Real ymin = center_y - half_size;
    Real ymax = center_y + half_size;

    std::cout << "  Test domain: [" << xmin << ", " << xmax << "] x ["
              << ymin << ", " << ymax << "] (30x30 km)\n";

    // Create adaptive quadtree with multiple refinement levels
    // Coarse base mesh with refinement in center
    auto quadtree = std::make_unique<QuadtreeAdapter>();

    // Create a 4x4 base grid with refinement in center 2x2
    Real dx = domain_size / 4.0;
    Real dy = domain_size / 4.0;

    QuadLevel coarse_level = {0, 0};
    QuadLevel fine_level = {1, 1};

    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            Real ex_min = xmin + i * dx;
            Real ex_max = xmin + (i + 1) * dx;
            Real ey_min = ymin + j * dy;
            Real ey_max = ymin + (j + 1) * dy;

            // Refine center 2x2 region
            bool is_center = (i >= 1 && i <= 2 && j >= 1 && j <= 2);

            if (is_center) {
                // Split into 2x2 fine elements
                Real mx = (ex_min + ex_max) / 2.0;
                Real my = (ey_min + ey_max) / 2.0;
                quadtree->add_element(QuadBounds{ex_min, mx, ey_min, my}, fine_level);
                quadtree->add_element(QuadBounds{mx, ex_max, ey_min, my}, fine_level);
                quadtree->add_element(QuadBounds{ex_min, mx, my, ey_max}, fine_level);
                quadtree->add_element(QuadBounds{mx, ex_max, my, ey_max}, fine_level);
            } else {
                quadtree->add_element(QuadBounds{ex_min, ex_max, ey_min, ey_max},
                                      coarse_level);
            }
        }
    }

    std::cout << "  Quadtree: " << quadtree->num_elements() << " elements\n";
    std::cout << "    Coarse elements (7.5 km): 12\n";
    std::cout << "    Fine elements (3.75 km): 16\n";

    // Create THB surface
    THBSurfaceConfig config;
    config.ngauss = 5;  // Need enough points for the basis functions
    config.smoothing_weight = 0.0;
    config.verbose = true;

    THBSurface surface(*quadtree, config);

    std::cout << "  THB hierarchy:\n";
    std::cout << "    Max level: " << surface.max_level() << "\n";
    std::cout << "    Active DOFs: " << surface.num_active_dofs() << "\n";

    // Wrap BathymetryData in a lambda for set_bathymetry_data
    auto bathy_func = [&bathy](Real x, Real y) -> Real {
        return static_cast<Real>(bathy.get_depth(x, y));
    };

    // Fit to bathymetry
    surface.set_bathymetry_data(bathy_func);
    surface.solve();

    EXPECT_TRUE(surface.is_solved());

    // Evaluate at some test points and compare to raw bathymetry
    std::cout << "\n  Sample evaluations:\n";
    Real max_error = 0.0;
    Real sum_error = 0.0;
    int num_samples = 0;

    for (int j = 0; j < 10; ++j) {
        for (int i = 0; i < 10; ++i) {
            Real x = xmin + (i + 0.5) * domain_size / 10.0;
            Real y = ymin + (j + 0.5) * domain_size / 10.0;

            Real thb_depth = surface.evaluate(x, y);
            Real raw_depth = bathy_func(x, y);

            Real error = std::abs(thb_depth - raw_depth);
            max_error = std::max(max_error, error);
            sum_error += error;
            num_samples++;

            if (i == 5 && j == 5) {
                std::cout << "    Center (" << x << ", " << y << "): THB="
                          << thb_depth << " m, raw=" << raw_depth << " m\n";
            }
        }
    }

    Real mean_error = sum_error / num_samples;
    std::cout << "  Fitting error: max=" << max_error << " m, mean=" << mean_error
              << " m\n";

    // Write VTK output
    std::string vtk_path = "/tmp/thb_kattegat_30km";
    surface.write_vtk(vtk_path, 20);
    std::cout << "  Wrote VTK: " << vtk_path << ".vtk\n";

    EXPECT_TRUE(std::filesystem::exists(vtk_path + ".vtk"));

    // Reasonable fitting error for real bathymetry data
    EXPECT_LT(max_error, 50.0) << "Max fitting error too large";
    EXPECT_LT(mean_error, 10.0) << "Mean fitting error too large";
}

TEST(THBSurfaceGeoTiffTest, KattegatMultiLevel) {
    // Test with multiple refinement levels (similar to Bezier multiresolution tests)
    if (!GeoTiffReader::is_available()) {
        GTEST_SKIP() << "GDAL not available";
    }

    std::string geotiff_path = testing::BATHYMETRY_GEOTIFF_PATH;
    if (!std::filesystem::exists(geotiff_path)) {
        GTEST_SKIP() << "GeoTIFF file not found: " << geotiff_path;
    }

    std::cout << "\n=== THB-Spline Multi-Level Kattegat Test ===\n";

    GeoTiffReader reader;
    BathymetryData bathy = reader.load(geotiff_path);
    ASSERT_TRUE(bathy.is_valid());

    // Same 30x30 km domain
    Real center_x = 4095238.0;
    Real center_y = 3344695.0;
    Real domain_size = 30000.0;
    Real half_size = domain_size / 2.0;

    Real xmin = center_x - half_size;
    Real xmax = center_x + half_size;
    Real ymin = center_y - half_size;
    Real ymax = center_y + half_size;

    // Create quadtree with 3 levels of refinement
    // Level 0: 15 km elements (2x2 base)
    // Level 1: 7.5 km elements
    // Level 2: 3.75 km elements (finest in center)
    auto quadtree = std::make_unique<QuadtreeAdapter>();

    QuadLevel level0 = {0, 0};
    QuadLevel level1 = {1, 1};
    QuadLevel level2 = {2, 2};

    Real base_size = domain_size / 2.0;  // 15 km

    for (int bj = 0; bj < 2; ++bj) {
        for (int bi = 0; bi < 2; ++bi) {
            Real bx_min = xmin + bi * base_size;
            Real bx_max = xmin + (bi + 1) * base_size;
            Real by_min = ymin + bj * base_size;
            Real by_max = ymin + (bj + 1) * base_size;

            // Refine center quadrant (bi=0, bj=0 touches center)
            bool is_center_quadrant = (bi == 1 && bj == 1);

            if (is_center_quadrant) {
                // Level 1: 4 elements of 7.5 km
                Real mid_x = (bx_min + bx_max) / 2.0;
                Real mid_y = (by_min + by_max) / 2.0;

                // Further refine bottom-left into level 2
                Real q_size = base_size / 4.0;  // 3.75 km
                for (int qj = 0; qj < 2; ++qj) {
                    for (int qi = 0; qi < 2; ++qi) {
                        Real qx_min = bx_min + qi * q_size;
                        Real qx_max = bx_min + (qi + 1) * q_size;
                        Real qy_min = by_min + qj * q_size;
                        Real qy_max = by_min + (qj + 1) * q_size;
                        quadtree->add_element(QuadBounds{qx_min, qx_max, qy_min, qy_max},
                                              level2);
                    }
                }

                // Rest at level 1
                quadtree->add_element(QuadBounds{mid_x, bx_max, by_min, mid_y}, level1);
                quadtree->add_element(QuadBounds{bx_min, mid_x, mid_y, by_max}, level1);
                quadtree->add_element(QuadBounds{mid_x, bx_max, mid_y, by_max}, level1);
            } else {
                // Keep at level 0
                quadtree->add_element(QuadBounds{bx_min, bx_max, by_min, by_max}, level0);
            }
        }
    }

    std::cout << "  Quadtree: " << quadtree->num_elements() << " elements\n";
    std::cout << "    Level 0 (15 km): 3 elements\n";
    std::cout << "    Level 1 (7.5 km): 3 elements\n";
    std::cout << "    Level 2 (3.75 km): 4 elements\n";

    // Create and fit THB surface
    THBSurfaceConfig config;
    config.ngauss = 6;  // More points for coarse elements
    config.verbose = true;

    THBSurface surface(*quadtree, config);

    std::cout << "  THB hierarchy:\n";
    std::cout << "    Max level: " << surface.max_level() << "\n";
    std::cout << "    Active DOFs: " << surface.num_active_dofs() << "\n";

    // Wrap BathymetryData in a lambda
    auto bathy_func = [&bathy](Real x, Real y) -> Real {
        return static_cast<Real>(bathy.get_depth(x, y));
    };

    surface.set_bathymetry_data(bathy_func);
    surface.solve();

    EXPECT_TRUE(surface.is_solved());

    // Write VTK output
    std::string vtk_path = "/tmp/thb_kattegat_multilevel";
    surface.write_vtk(vtk_path, 20);
    std::cout << "  Wrote VTK: " << vtk_path << ".vtk\n";

    EXPECT_TRUE(std::filesystem::exists(vtk_path + ".vtk"));

    // Evaluate fitting error
    Real max_error = 0.0;
    Real sum_error = 0.0;
    int num_samples = 0;
    for (int j = 0; j < 20; ++j) {
        for (int i = 0; i < 20; ++i) {
            Real x = xmin + (i + 0.5) * domain_size / 20.0;
            Real y = ymin + (j + 0.5) * domain_size / 20.0;
            Real thb_depth = surface.evaluate(x, y);
            Real raw_depth = bathy_func(x, y);
            Real err = std::abs(thb_depth - raw_depth);
            max_error = std::max(max_error, err);
            sum_error += err;
            num_samples++;
        }
    }

    Real mean_error = sum_error / num_samples;
    std::cout << "  Fitting error: max=" << max_error << " m, mean=" << mean_error << " m\n";

    // With coarse 15km elements at level 0, larger errors are expected in those regions
    // The mean error should still be reasonable
    EXPECT_LT(max_error, 80.0) << "Multi-level max fitting error too large";
    EXPECT_LT(mean_error, 15.0) << "Multi-level mean fitting error too large";
}

}  // namespace
}  // namespace drifter
