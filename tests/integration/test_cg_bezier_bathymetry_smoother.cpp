#include <gtest/gtest.h>
#include "bathymetry/cg_bezier_bathymetry_smoother.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "mesh/octree_adapter.hpp"
#include "mesh/seabed_surface.hpp"
#include "test_integration_fixtures.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <filesystem>

#ifdef DRIFTER_HAS_GDAL
#include "mesh/geotiff_reader.hpp"
#endif

using namespace drifter;
using namespace drifter::testing;

class CGBezierSmootherTest : public ::testing::Test {
protected:
    static constexpr Real TOLERANCE = 1e-10;
    static constexpr Real LOOSE_TOLERANCE = 1e-6;

    std::unique_ptr<OctreeAdapter> create_octree(int nx, int ny, int nz) {
        auto octree = std::make_unique<OctreeAdapter>(
            0.0, 100.0,   // x bounds
            0.0, 100.0,   // y bounds
            -1.0, 0.0     // z bounds
        );
        octree->build_uniform(nx, ny, nz);
        return octree;
    }

    QuadtreeAdapter create_quadtree(int nx, int ny,
                                     Real xmin = 0.0, Real xmax = 100.0,
                                     Real ymin = 0.0, Real ymax = 100.0) {
        QuadtreeAdapter mesh;
        mesh.build_uniform(xmin, xmax, ymin, ymax, nx, ny);
        return mesh;
    }
};

// =============================================================================
// Level 1: Basic Construction and Single Element Tests
// =============================================================================

TEST_F(CGBezierSmootherTest, ConstructFromQuadtree) {
    auto mesh = create_quadtree(2, 2);

    CGBezierBathymetrySmoother smoother(mesh);

    EXPECT_GT(smoother.num_global_dofs(), 0);
    EXPECT_FALSE(smoother.is_solved());
}

TEST_F(CGBezierSmootherTest, ConstructFromOctree) {
    auto octree = create_octree(4, 4, 2);

    CGBezierBathymetrySmoother smoother(*octree);

    EXPECT_GT(smoother.num_global_dofs(), 0);
    EXPECT_FALSE(smoother.is_solved());
}

TEST_F(CGBezierSmootherTest, SingleElementConstant) {
    // Single element test: constant bathymetry
    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 10.0, 0.0, 10.0, 1, 1);

    CGBezierSmootherConfig config;
    config.lambda = 100.0;  // Strong data fitting
    config.ridge_epsilon = 0.0;  // No ridge for exact polynomial reproduction

    CGBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data([](Real, Real) { return 50.0; });
    smoother.solve();

    EXPECT_TRUE(smoother.is_solved());

    // Should reproduce constant exactly
    EXPECT_NEAR(smoother.evaluate(5.0, 5.0), 50.0, LOOSE_TOLERANCE);
    EXPECT_NEAR(smoother.evaluate(2.0, 8.0), 50.0, LOOSE_TOLERANCE);
}

TEST_F(CGBezierSmootherTest, SingleElementLinear) {
    // Single element: linear bathymetry
    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 10.0, 0.0, 10.0, 1, 1);

    CGBezierSmootherConfig config;
    config.lambda = 100.0;
    config.ridge_epsilon = 0.0;  // No ridge for exact polynomial reproduction

    auto linear = [](Real x, Real y) { return 10.0 + 2.0 * x + 3.0 * y; };

    CGBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data(linear);
    smoother.solve();

    // Quintic can represent linear exactly
    EXPECT_NEAR(smoother.evaluate(5.0, 5.0), linear(5.0, 5.0), LOOSE_TOLERANCE);
    EXPECT_NEAR(smoother.evaluate(2.0, 8.0), linear(2.0, 8.0), LOOSE_TOLERANCE);
}

TEST_F(CGBezierSmootherTest, SingleElementQuadratic) {
    // Single element: quadratic bathymetry
    // Note: Quadratics have non-zero thin plate energy (curvature), so there's a
    // small trade-off between smoothing and data fitting even with high lambda.
    // We use a looser tolerance (0.1% relative error) for this test.
    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 10.0, 0.0, 10.0, 1, 1);

    CGBezierSmootherConfig config;
    config.lambda = 100.0;
    config.ridge_epsilon = 0.0;

    auto quadratic = [](Real x, Real y) { return 100.0 + 0.1 * x * x + 0.2 * y * y; };

    CGBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data(quadratic);
    smoother.solve();

    // Use 0.1% relative tolerance for quadratics (smoothing affects non-linear functions)
    Real rel_tol = 0.001;
    EXPECT_NEAR(smoother.evaluate(5.0, 5.0), quadratic(5.0, 5.0),
                std::abs(quadratic(5.0, 5.0)) * rel_tol);
    EXPECT_NEAR(smoother.evaluate(3.0, 7.0), quadratic(3.0, 7.0),
                std::abs(quadratic(3.0, 7.0)) * rel_tol);
}

// =============================================================================
// Level 1: 2×2 Uniform Mesh Tests (DOF Sharing)
// =============================================================================

TEST_F(CGBezierSmootherTest, TwoByTwoMeshConstant) {
    auto mesh = create_quadtree(2, 2);

    CGBezierSmootherConfig config;
    config.lambda = 100.0;
    config.ridge_epsilon = 0.0;  // No ridge for exact polynomial reproduction

    CGBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data([](Real, Real) { return 42.0; });
    smoother.solve();

    EXPECT_TRUE(smoother.is_solved());

    // Check at several points
    EXPECT_NEAR(smoother.evaluate(25.0, 25.0), 42.0, LOOSE_TOLERANCE);
    EXPECT_NEAR(smoother.evaluate(50.0, 50.0), 42.0, LOOSE_TOLERANCE);
    EXPECT_NEAR(smoother.evaluate(75.0, 75.0), 42.0, LOOSE_TOLERANCE);
}

TEST_F(CGBezierSmootherTest, TwoByTwoMeshLinear) {
    auto mesh = create_quadtree(2, 2);

    CGBezierSmootherConfig config;
    config.lambda = 100.0;
    config.ridge_epsilon = 0.0;  // No ridge for exact polynomial reproduction

    auto linear = [](Real x, Real y) { return 10.0 + 0.5 * x + 0.3 * y; };

    CGBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data(linear);
    smoother.solve();

    // Should reproduce linear exactly across all elements
    for (Real x = 10.0; x < 100.0; x += 20.0) {
        for (Real y = 10.0; y < 100.0; y += 20.0) {
            EXPECT_NEAR(smoother.evaluate(x, y), linear(x, y), LOOSE_TOLERANCE)
                << "At (" << x << ", " << y << ")";
        }
    }
}

TEST_F(CGBezierSmootherTest, DofSharingReducesCount) {
    // Verify DOF sharing: CG should have fewer DOFs than DG
    auto mesh = create_quadtree(2, 2);  // 4 elements

    CGBezierBathymetrySmoother smoother(mesh);

    Index num_dofs = smoother.num_global_dofs();

    // DG would have 4 × 36 = 144 DOFs
    // CG should have significantly fewer due to shared edges/corners
    // For a 2×2 mesh with quintic:
    //   - Interior vertex (1): shared by 4 elements
    //   - Edge midpoints (4): shared by 2 elements
    //   - Corners (4): unique to 1 element each
    //   - Interior edge DOFs: shared between 2 elements
    // Expected: much less than 144
    EXPECT_LT(num_dofs, 144);
    EXPECT_GT(num_dofs, 36);  // At least one element's worth

    std::cout << "2×2 mesh CG DOFs: " << num_dofs << " (vs 144 for DG)" << std::endl;
}

TEST_F(CGBezierSmootherTest, NoConstraintsForUniformMesh) {
    // Uniform mesh should have no hanging nodes
    auto mesh = create_quadtree(2, 2);

    CGBezierBathymetrySmoother smoother(mesh);

    // Uniform mesh = conforming = no constraints
    EXPECT_EQ(smoother.num_constraints(), 0);
}

// =============================================================================
// Level 2: Non-conforming 1+4 Mesh Tests
// =============================================================================

TEST_F(CGBezierSmootherTest, OnePlusFourMeshConstruction) {
    // Create 1+4 non-conforming mesh:
    // 2×2 fine elements + 1 coarse element on the right
    //
    //  +---+---+-------+
    //  | 2 | 3 |       |
    //  +---+---+   4   |
    //  | 0 | 1 |       |
    //  +---+---+-------+

    QuadtreeAdapter mesh;
    Real h = 25.0;

    // Fine elements (level 2)
    mesh.add_element({0.0, h, 0.0, h}, {2, 2});       // elem 0
    mesh.add_element({h, 2*h, 0.0, h}, {2, 2});       // elem 1
    mesh.add_element({0.0, h, h, 2*h}, {2, 2});       // elem 2
    mesh.add_element({h, 2*h, h, 2*h}, {2, 2});       // elem 3

    // Coarse element (level 1)
    mesh.add_element({2*h, 4*h, 0.0, 2*h}, {1, 1});   // elem 4

    EXPECT_EQ(mesh.num_elements(), 5);

    CGBezierBathymetrySmoother smoother(mesh);

    // Should have hanging node constraints at fine-coarse interface
    EXPECT_GT(smoother.num_constraints(), 0);

    std::cout << "1+4 mesh: " << smoother.num_global_dofs() << " DOFs, "
              << smoother.num_constraints() << " constraints" << std::endl;
}

TEST_F(CGBezierSmootherTest, OnePlusFourConstantBathymetry) {
    // 1+4 mesh with constant bathymetry
    QuadtreeAdapter mesh;
    Real h = 25.0;

    mesh.add_element({0.0, h, 0.0, h}, {2, 2});
    mesh.add_element({h, 2*h, 0.0, h}, {2, 2});
    mesh.add_element({0.0, h, h, 2*h}, {2, 2});
    mesh.add_element({h, 2*h, h, 2*h}, {2, 2});
    mesh.add_element({2*h, 4*h, 0.0, 2*h}, {1, 1});

    CGBezierSmootherConfig config;
    config.lambda = 100.0;
    config.ridge_epsilon = 0.0;  // No ridge for exact polynomial reproduction

    CGBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data([](Real, Real) { return 100.0; });
    smoother.solve();

    EXPECT_TRUE(smoother.is_solved());

    // Should be constant everywhere
    EXPECT_NEAR(smoother.evaluate(10.0, 10.0), 100.0, LOOSE_TOLERANCE);
    EXPECT_NEAR(smoother.evaluate(75.0, 25.0), 100.0, LOOSE_TOLERANCE);
}

TEST_F(CGBezierSmootherTest, OnePlusFourLinearBathymetry) {
    // 1+4 mesh with linear bathymetry
    QuadtreeAdapter mesh;
    Real h = 25.0;

    mesh.add_element({0.0, h, 0.0, h}, {2, 2});
    mesh.add_element({h, 2*h, 0.0, h}, {2, 2});
    mesh.add_element({0.0, h, h, 2*h}, {2, 2});
    mesh.add_element({h, 2*h, h, 2*h}, {2, 2});
    mesh.add_element({2*h, 4*h, 0.0, 2*h}, {1, 1});

    auto linear = [](Real x, Real y) { return 50.0 + x + 2.0 * y; };

    CGBezierSmootherConfig config;
    config.lambda = 100.0;
    config.ridge_epsilon = 0.0;  // No ridge for exact polynomial reproduction

    CGBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data(linear);
    smoother.solve();

    // Should reproduce linear across both fine and coarse elements
    std::vector<Vec2> test_points = {
        {10.0, 10.0}, {25.0, 25.0}, {40.0, 40.0},  // Fine region
        {60.0, 25.0}, {75.0, 40.0}, {90.0, 10.0},  // Coarse region
        {50.0, 25.0},  // At interface
    };

    for (const auto& pt : test_points) {
        Real expected = linear(pt(0), pt(1));
        Real computed = smoother.evaluate(pt(0), pt(1));
        EXPECT_NEAR(computed, expected, LOOSE_TOLERANCE)
            << "At (" << pt(0) << ", " << pt(1) << ")";
    }
}

TEST_F(CGBezierSmootherTest, OnePlusFourContinuityAtInterface) {
    // Check C⁰ continuity at the fine-coarse interface
    QuadtreeAdapter mesh;
    Real h = 25.0;

    mesh.add_element({0.0, h, 0.0, h}, {2, 2});
    mesh.add_element({h, 2*h, 0.0, h}, {2, 2});
    mesh.add_element({0.0, h, h, 2*h}, {2, 2});
    mesh.add_element({h, 2*h, h, 2*h}, {2, 2});
    mesh.add_element({2*h, 4*h, 0.0, 2*h}, {1, 1});

    // Use smooth function to test continuity
    auto smooth_bathy = [](Real x, Real y) {
        return 100.0 + 20.0 * std::sin(0.03 * x) * std::cos(0.03 * y);
    };

    CGBezierSmootherConfig config;
    config.lambda = 10.0;

    CGBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data(smooth_bathy);
    smoother.solve();

    // Check continuity at interface x = 2*h = 50
    Real max_jump = 0.0;
    for (Real y = 1.0; y < 50.0; y += 5.0) {
        Real left = smoother.evaluate(49.99, y);
        Real right = smoother.evaluate(50.01, y);
        Real jump = std::abs(left - right);
        max_jump = std::max(max_jump, jump);
    }

    std::cout << "Max jump at interface: " << max_jump << std::endl;

    // CG should provide C⁰ continuity via shared DOFs
    // With energy coupling, should be smooth (small jumps)
    EXPECT_LT(max_jump, 1.0);
}

// =============================================================================
// Smoothing Effect Tests
// =============================================================================

TEST_F(CGBezierSmootherTest, SmoothingReducesVariation) {
    auto mesh = create_quadtree(4, 4);

    // Noisy bathymetry
    auto noisy = [](Real x, Real y) {
        return 50.0 + 5.0 * std::sin(0.5 * x) * std::sin(0.5 * y);
    };

    // Low smoothing
    CGBezierSmootherConfig config_low;
    config_low.lambda = 100.0;  // High data weight

    CGBezierBathymetrySmoother low_smooth(mesh, config_low);
    low_smooth.set_bathymetry_data(noisy);
    low_smooth.solve();

    // High smoothing
    CGBezierSmootherConfig config_high;
    config_high.lambda = 0.01;  // Low data weight, high smoothing

    CGBezierBathymetrySmoother high_smooth(mesh, config_high);
    high_smooth.set_bathymetry_data(noisy);
    high_smooth.solve();

    // Compute variation
    Real var_low = 0.0, var_high = 0.0;
    Real mean_low = 0.0, mean_high = 0.0;
    int count = 0;

    for (Real x = 10.0; x < 90.0; x += 10.0) {
        for (Real y = 10.0; y < 90.0; y += 10.0) {
            mean_low += low_smooth.evaluate(x, y);
            mean_high += high_smooth.evaluate(x, y);
            count++;
        }
    }
    mean_low /= count;
    mean_high /= count;

    for (Real x = 10.0; x < 90.0; x += 10.0) {
        for (Real y = 10.0; y < 90.0; y += 10.0) {
            Real v1 = low_smooth.evaluate(x, y) - mean_low;
            Real v2 = high_smooth.evaluate(x, y) - mean_high;
            var_low += v1 * v1;
            var_high += v2 * v2;
        }
    }

    std::cout << "Variance - low smooth: " << var_low << ", high smooth: " << var_high << std::endl;

    // High smoothing should reduce variance
    EXPECT_LT(var_high, var_low);
}

// =============================================================================
// Gradient Tests
// =============================================================================

TEST_F(CGBezierSmootherTest, GradientOfConstant) {
    auto mesh = create_quadtree(2, 2);

    CGBezierSmootherConfig config;
    config.lambda = 100.0;
    config.ridge_epsilon = 0.0;  // No ridge for exact polynomial reproduction

    CGBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data([](Real, Real) { return 42.0; });
    smoother.solve();

    Vec2 grad = smoother.evaluate_gradient(50.0, 50.0);

    EXPECT_NEAR(grad(0), 0.0, LOOSE_TOLERANCE);
    EXPECT_NEAR(grad(1), 0.0, LOOSE_TOLERANCE);
}

TEST_F(CGBezierSmootherTest, GradientOfLinear) {
    auto mesh = create_quadtree(2, 2);

    CGBezierSmootherConfig config;
    config.lambda = 100.0;
    config.ridge_epsilon = 0.0;  // No ridge for exact polynomial reproduction

    // z = 10 + 2x + 3y, gradient = (2, 3)
    CGBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data([](Real x, Real y) {
        return 10.0 + 2.0 * x + 3.0 * y;
    });
    smoother.solve();

    Vec2 grad = smoother.evaluate_gradient(50.0, 50.0);

    EXPECT_NEAR(grad(0), 2.0, LOOSE_TOLERANCE);
    EXPECT_NEAR(grad(1), 3.0, LOOSE_TOLERANCE);
}

// =============================================================================
// Error Handling Tests
// =============================================================================

TEST_F(CGBezierSmootherTest, SolveBeforeDataThrows) {
    auto mesh = create_quadtree(2, 2);
    CGBezierBathymetrySmoother smoother(mesh);

    EXPECT_THROW(smoother.solve(), std::runtime_error);
}

TEST_F(CGBezierSmootherTest, EvaluateBeforeSolveThrows) {
    auto mesh = create_quadtree(2, 2);
    CGBezierBathymetrySmoother smoother(mesh);
    smoother.set_bathymetry_data([](Real, Real) { return 1.0; });

    EXPECT_THROW(smoother.evaluate(50.0, 50.0), std::runtime_error);
}

// =============================================================================
// VTK Output Tests
// =============================================================================

TEST_F(CGBezierSmootherTest, VTKOutputUniformMesh) {
    auto mesh = create_quadtree(4, 4);

    auto bathy = [](Real x, Real y) {
        return 100.0 + 20.0 * std::sin(0.03 * x) * std::cos(0.03 * y);
    };

    CGBezierSmootherConfig config;
    config.lambda = 10.0;

    CGBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data(bathy);
    smoother.solve();

    std::string filename = "/tmp/cg_bezier_uniform.vtk";
    smoother.write_vtk(filename, 10);

    EXPECT_TRUE(std::filesystem::exists(filename));

    std::ifstream file(filename);
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    EXPECT_GT(content.size(), 1000);
}

TEST_F(CGBezierSmootherTest, VTKOutputOnePlusFourMesh) {
    // 1+4 mesh
    QuadtreeAdapter mesh;
    Real h = 25.0;

    mesh.add_element({0.0, h, 0.0, h}, {2, 2});
    mesh.add_element({h, 2*h, 0.0, h}, {2, 2});
    mesh.add_element({0.0, h, h, 2*h}, {2, 2});
    mesh.add_element({h, 2*h, h, 2*h}, {2, 2});
    mesh.add_element({2*h, 4*h, 0.0, 2*h}, {1, 1});

    auto bathy = [](Real x, Real y) {
        return 100.0 + 20.0 * std::sin(0.03 * x) * std::cos(0.03 * y);
    };

    CGBezierSmootherConfig config;
    config.lambda = 10.0;

    CGBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data(bathy);
    smoother.solve();

    std::string filename = "/tmp/cg_bezier_1plus4.vtk";
    smoother.write_vtk(filename, 10);

    EXPECT_TRUE(std::filesystem::exists(filename));

    // Also write control points
    std::string cp_filename = "/tmp/cg_bezier_1plus4_control_points.vtk";
    smoother.write_control_points_vtk(cp_filename);

    EXPECT_TRUE(std::filesystem::exists(cp_filename));
}

// =============================================================================
// Level 4: Kattegat GeoTIFF Integration Test
// =============================================================================

#ifdef DRIFTER_HAS_GDAL
TEST_F(CGBezierSmootherTest, KattegatGeoTiffIntegration) {
    // Kattegat test area
    Real center_x = 4095238.0;  // EPSG:3034
    Real center_y = 3344695.0;  // EPSG:3034
    Real domain_size = 30000.0; // 30 km

    Real xmin = center_x - domain_size / 2;
    Real xmax = center_x + domain_size / 2;
    Real ymin = center_y - domain_size / 2;
    Real ymax = center_y + domain_size / 2;

    std::string geotiff_path = BATHYMETRY_GEOTIFF_PATH;

    if (!GeoTiffReader::is_available()) {
        GTEST_SKIP() << "GDAL not available";
    }

    if (!std::filesystem::exists(geotiff_path)) {
        GTEST_SKIP() << "GeoTIFF file not found: " << geotiff_path;
    }

    GeoTiffReader reader;
    BathymetryData bathy = reader.load(geotiff_path);
    if (!bathy.is_valid()) {
        GTEST_SKIP() << "Could not load GeoTIFF file: " << reader.last_error();
    }

    auto bathy_ptr = std::make_shared<BathymetryData>(std::move(bathy));
    BathymetrySurface surface(bathy_ptr);

    std::cout << "=== CG Bezier Kattegat Test ===" << std::endl;
    std::cout << "Domain: [" << xmin << ", " << xmax << "] x [" << ymin << ", " << ymax << "]" << std::endl;

    // Create uniform mesh (no adaptive refinement for simplicity)
    QuadtreeAdapter mesh;
    mesh.build_uniform(xmin, xmax, ymin, ymax, 8, 8);

    std::cout << "Mesh elements: " << mesh.num_elements() << std::endl;

    // Depth function using BathymetrySurface
    auto depth_func = [&surface](Real x, Real y) -> Real {
        return -surface.depth(x, y);  // depth returns positive down, we want elevation
    };

    // Create CG Bezier smoother
    // lambda controls data fitting weight: lower = smoother, higher = closer to data
    CGBezierSmootherConfig config;
    config.lambda = 1.0;
    config.ngauss_data = 6;
    config.ngauss_energy = 6;

    CGBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data(std::function<Real(Real, Real)>(depth_func));

    std::cout << "DOFs: " << smoother.num_global_dofs() << std::endl;
    std::cout << "Constraints: " << smoother.num_constraints() << std::endl;

    // Solve
    smoother.solve();

    EXPECT_TRUE(smoother.is_solved());

    // Write output for ParaView verification
    std::string output_file = "/tmp/cg_bezier_kattegat_test.vtk";
    smoother.write_vtk(output_file, 10);

    std::cout << "Output written to: " << output_file << std::endl;
    EXPECT_TRUE(std::filesystem::exists(output_file));

    // Write control points for debugging
    std::string cp_file = "/tmp/cg_bezier_kattegat_control_points.vtk";
    smoother.write_control_points_vtk(cp_file);

    std::cout << "Control points written to: " << cp_file << std::endl;

    // Write raw bathymetry data to uniform grid for comparison
    {
        std::string raw_file = "/tmp/cg_bezier_kattegat_raw_data.vtk";
        std::ofstream file(raw_file);

        // Use comparable resolution to smoothed output (8x8 elements * 10 points = 80x80)
        int nx = 80, ny = 80;
        Real hx = (xmax - xmin) / (nx - 1);
        Real hy = (ymax - ymin) / (ny - 1);

        file << "# vtk DataFile Version 3.0\n";
        file << "Raw Bathymetry Data\n";
        file << "ASCII\n";
        file << "DATASET STRUCTURED_GRID\n";
        file << "DIMENSIONS " << nx << " " << ny << " 1\n";
        file << "POINTS " << (nx * ny) << " double\n";

        for (int j = 0; j < ny; ++j) {
            Real y = ymin + j * hy;
            for (int i = 0; i < nx; ++i) {
                Real x = xmin + i * hx;
                Real z = depth_func(x, y);
                file << x << " " << y << " " << z << "\n";
            }
        }

        file << "POINT_DATA " << (nx * ny) << "\n";
        file << "SCALARS elevation double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int j = 0; j < ny; ++j) {
            Real y = ymin + j * hy;
            for (int i = 0; i < nx; ++i) {
                Real x = xmin + i * hx;
                Real z = depth_func(x, y);
                file << z << "\n";
            }
        }

        file.close();
        std::cout << "Raw data written to: " << raw_file << std::endl;
    }

    // Basic quality checks
    Real max_diff = 0.0;
    Real sum_diff = 0.0;
    int count = 0;

    for (Index e = 0; e < mesh.num_elements(); ++e) {
        const auto& bounds = mesh.element_bounds(e);
        Real cx = 0.5 * (bounds.xmin + bounds.xmax);
        Real cy = 0.5 * (bounds.ymin + bounds.ymax);

        Real expected = depth_func(cx, cy);
        Real computed = smoother.evaluate(cx, cy);
        Real diff = std::abs(expected - computed);

        max_diff = std::max(max_diff, diff);
        sum_diff += diff;
        count++;
    }

    Real avg_diff = sum_diff / count;
    std::cout << "Max difference from input: " << max_diff << " m" << std::endl;
    std::cout << "Avg difference from input: " << avg_diff << " m" << std::endl;

    // Reasonable fit quality (smoothed, so not exact)
    EXPECT_LT(avg_diff, 50.0);  // Average error less than 50m
}

TEST_F(CGBezierSmootherTest, KattegatWithC1C2Constraints) {
    // Test with C¹ and C² continuity constraints
    Real center_x = 4095238.0;
    Real center_y = 3344695.0;
    Real domain_size = 30000.0;

    Real xmin = center_x - domain_size / 2;
    Real xmax = center_x + domain_size / 2;
    Real ymin = center_y - domain_size / 2;
    Real ymax = center_y + domain_size / 2;

    std::string geotiff_path = BATHYMETRY_GEOTIFF_PATH;

    if (!GeoTiffReader::is_available()) {
        GTEST_SKIP() << "GDAL not available";
    }

    if (!std::filesystem::exists(geotiff_path)) {
        GTEST_SKIP() << "GeoTIFF file not found: " << geotiff_path;
    }

    GeoTiffReader reader;
    BathymetryData bathy = reader.load(geotiff_path);
    if (!bathy.is_valid()) {
        GTEST_SKIP() << "Could not load GeoTIFF file: " << reader.last_error();
    }

    auto bathy_ptr = std::make_shared<BathymetryData>(std::move(bathy));
    BathymetrySurface surface(bathy_ptr);

    std::cout << "=== CG Bezier with C¹/C² Constraints ===" << std::endl;

    QuadtreeAdapter mesh;
    mesh.build_uniform(xmin, xmax, ymin, ymax, 8, 8);

    auto depth_func = [&surface](Real x, Real y) -> Real {
        return -surface.depth(x, y);
    };

    // Test configurations
    struct TestConfig {
        bool c2;
        bool edge;
        std::string suffix;
    };
    std::vector<TestConfig> configs = {
        {false, false, "no_constraints"},
        {true, false, "c2_constraints"},
        {false, true, "edge_only_constraints"},
        {true, true, "c2_edge_constraints"}
    };

    for (const auto& tc : configs) {
        CGBezierSmootherConfig config;
        config.lambda = 1.0;
        config.ngauss_data = 6;
        config.ngauss_energy = 6;
        config.enable_c2_constraints = tc.c2;
        config.enable_edge_constraints = tc.edge;
        config.edge_ngauss = 4;

        CGBezierBathymetrySmoother smoother(mesh, config);
        smoother.set_bathymetry_data(std::function<Real(Real, Real)>(depth_func));

        std::cout << "\n--- Config: " << tc.suffix << " ---" << std::endl;
        std::cout << "DOFs: " << smoother.num_global_dofs() << std::endl;
        std::cout << "Vertex derivative constraints: "
                  << smoother.dof_manager().num_vertex_derivative_constraints() << std::endl;
        std::cout << "Edge derivative constraints: "
                  << smoother.dof_manager().num_edge_derivative_constraints() << std::endl;

        smoother.solve();
        EXPECT_TRUE(smoother.is_solved());

        std::string output_file = "/tmp/cg_bezier_kattegat_" + tc.suffix + ".vtk";
        smoother.write_vtk(output_file, 10);
        std::cout << "Output: " << output_file << std::endl;

        // Check constraint violation
        Real violation = smoother.constraint_violation();
        std::cout << "Constraint violation: " << violation << std::endl;

        // Compute data fitting error
        Real max_diff = 0.0;
        Real sum_diff = 0.0;
        int count = 0;

        for (Index e = 0; e < mesh.num_elements(); ++e) {
            const auto& bounds = mesh.element_bounds(e);
            Real cx = 0.5 * (bounds.xmin + bounds.xmax);
            Real cy = 0.5 * (bounds.ymin + bounds.ymax);

            Real expected = depth_func(cx, cy);
            Real computed = smoother.evaluate(cx, cy);
            Real diff = std::abs(expected - computed);

            max_diff = std::max(max_diff, diff);
            sum_diff += diff;
            count++;
        }

        Real avg_diff = sum_diff / count;
        std::cout << "Max diff: " << max_diff << " m, Avg diff: " << avg_diff << " m" << std::endl;
    }

    std::cout << "\nCompare outputs in ParaView to see effect of constraints on boundary kinks." << std::endl;
}
#endif

// =============================================================================
// Diagnostics Tests
// =============================================================================

TEST_F(CGBezierSmootherTest, DiagnosticsAfterSolve) {
    auto mesh = create_quadtree(2, 2);

    CGBezierSmootherConfig config;
    config.lambda = 10.0;

    CGBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data([](Real x, Real y) {
        return 50.0 + x * 0.5;
    });
    smoother.solve();

    // These should not throw
    Real data_res = smoother.data_residual();
    Real reg_energy = smoother.regularization_energy();
    Real obj_val = smoother.objective_value();

    std::cout << "Data residual: " << data_res << std::endl;
    std::cout << "Regularization energy: " << reg_energy << std::endl;
    std::cout << "Objective value: " << obj_val << std::endl;

    // Objective should be positive or zero
    EXPECT_GE(obj_val, 0.0);
}

TEST_F(CGBezierSmootherTest, ConstraintViolationNearZero) {
    // 1+4 mesh with constraints
    QuadtreeAdapter mesh;
    Real h = 25.0;

    mesh.add_element({0.0, h, 0.0, h}, {2, 2});
    mesh.add_element({h, 2*h, 0.0, h}, {2, 2});
    mesh.add_element({0.0, h, h, 2*h}, {2, 2});
    mesh.add_element({h, 2*h, h, 2*h}, {2, 2});
    mesh.add_element({2*h, 4*h, 0.0, 2*h}, {1, 1});

    CGBezierSmootherConfig config;
    config.lambda = 10.0;

    CGBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data([](Real x, Real y) { return 50.0 + x; });
    smoother.solve();

    Real violation = smoother.constraint_violation();
    std::cout << "Constraint violation: " << violation << std::endl;

    // Constraints should be satisfied to high precision
    EXPECT_LT(violation, 1e-6);
}
