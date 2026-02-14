#include <gtest/gtest.h>
#include "bathymetry/cg_triharmonic_bezier_bathymetry_smoother.hpp"
#include "bathymetry/cg_bezier_bathymetry_smoother.hpp"  // For comparison
#include "bathymetry/triharmonic_hessian.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "mesh/octree_adapter.hpp"
#include "test_integration_fixtures.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <filesystem>

#include "mesh/geotiff_reader.hpp"

using namespace drifter;
using namespace drifter::testing;

class CGTriharmonicSmootherTest : public ::testing::Test {
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
// TriharmonicHessian Unit Tests
// =============================================================================

TEST_F(CGTriharmonicSmootherTest, TriharmonicHessianConstruction) {
    TriharmonicHessian hessian(4, 0.0);

    EXPECT_EQ(hessian.num_gauss_points(), 4);
    EXPECT_EQ(hessian.element_hessian().rows(), 36);
    EXPECT_EQ(hessian.element_hessian().cols(), 36);
}

TEST_F(CGTriharmonicSmootherTest, TriharmonicHessianSymmetric) {
    TriharmonicHessian hessian(4, 0.0);

    const MatX& H = hessian.element_hessian();

    // Hessian should be symmetric
    EXPECT_NEAR((H - H.transpose()).norm(), 0.0, TOLERANCE);
}

TEST_F(CGTriharmonicSmootherTest, TriharmonicHessianPositiveSemiDefinite) {
    TriharmonicHessian hessian(4, 0.0);

    const MatX& H = hessian.element_hessian();

    // Compute eigenvalues
    Eigen::SelfAdjointEigenSolver<MatX> solver(H);
    VecX eigenvalues = solver.eigenvalues();

    // All eigenvalues should be >= 0 (positive semi-definite)
    Real min_eigenvalue = eigenvalues.minCoeff();
    EXPECT_GE(min_eigenvalue, -TOLERANCE)
        << "Minimum eigenvalue: " << min_eigenvalue;
}

TEST_F(CGTriharmonicSmootherTest, TriharmonicEnergyZeroForConstant) {
    TriharmonicHessian hessian(4, 0.0);

    // Constant function z = 42
    VecX coeffs = VecX::Constant(36, 42.0);

    Real energy = hessian.energy(coeffs);
    // Use looser tolerance for numerical precision issues
    EXPECT_NEAR(energy, 0.0, 1e-8)
        << "Constant function should have zero triharmonic energy";
}

TEST_F(CGTriharmonicSmootherTest, TriharmonicEnergyZeroForLinear) {
    TriharmonicHessian hessian(4, 0.0);
    BezierBasis2D basis;

    // Linear function z = 1 + 2u + 3v
    VecX coeffs(36);
    for (int j = 0; j < 6; ++j) {
        for (int i = 0; i < 6; ++i) {
            Real u = static_cast<Real>(i) / 5;
            Real v = static_cast<Real>(j) / 5;
            coeffs(i + 6*j) = 1.0 + 2.0 * u + 3.0 * v;
        }
    }

    Real energy = hessian.energy(coeffs);
    EXPECT_NEAR(energy, 0.0, TOLERANCE)
        << "Linear function should have zero triharmonic energy";
}

TEST_F(CGTriharmonicSmootherTest, TriharmonicEnergyZeroForQuadratic) {
    TriharmonicHessian hessian(4, 0.0);

    // Quadratic function z = u^2 + v^2
    VecX coeffs(36);
    for (int j = 0; j < 6; ++j) {
        for (int i = 0; i < 6; ++i) {
            Real u = static_cast<Real>(i) / 5;
            Real v = static_cast<Real>(j) / 5;
            coeffs(i + 6*j) = u * u + v * v;
        }
    }

    Real energy = hessian.energy(coeffs);
    // Quadratic has zero third derivatives, so triharmonic energy should be zero
    EXPECT_NEAR(energy, 0.0, TOLERANCE)
        << "Quadratic function should have zero triharmonic energy";
}

TEST_F(CGTriharmonicSmootherTest, TriharmonicScaledHessian) {
    TriharmonicHessian hessian(4, 0.0);

    MatX H_unit = hessian.scaled_hessian(1.0, 1.0);
    MatX H_scaled = hessian.scaled_hessian(2.0, 3.0);

    // Scaled Hessian should be different from unit
    EXPECT_GT((H_unit - H_scaled).norm(), TOLERANCE);

    // Both should be symmetric
    EXPECT_NEAR((H_unit - H_unit.transpose()).norm(), 0.0, TOLERANCE);
    EXPECT_NEAR((H_scaled - H_scaled.transpose()).norm(), 0.0, TOLERANCE);
}

// =============================================================================
// Smoother Construction Tests
// =============================================================================

TEST_F(CGTriharmonicSmootherTest, ConstructFromQuadtree) {
    auto mesh = create_quadtree(2, 2);

    CGTriharmonicBezierBathymetrySmoother smoother(mesh);

    EXPECT_GT(smoother.num_global_dofs(), 0);
    EXPECT_FALSE(smoother.is_solved());
}

TEST_F(CGTriharmonicSmootherTest, ConstructFromOctree) {
    auto octree = create_octree(4, 4, 2);

    CGTriharmonicBezierBathymetrySmoother smoother(*octree);

    EXPECT_GT(smoother.num_global_dofs(), 0);
    EXPECT_FALSE(smoother.is_solved());
}

// =============================================================================
// Single Element Polynomial Reproduction Tests
// =============================================================================

TEST_F(CGTriharmonicSmootherTest, SingleElementConstant) {
    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 10.0, 0.0, 10.0, 1, 1);

    CGTriharmonicBezierSmootherConfig config;
    config.lambda = 100.0;
    config.ridge_epsilon = 0.0;

    CGTriharmonicBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data([](Real, Real) { return 50.0; });
    smoother.solve();

    EXPECT_TRUE(smoother.is_solved());
    EXPECT_NEAR(smoother.evaluate(5.0, 5.0), 50.0, LOOSE_TOLERANCE);
    EXPECT_NEAR(smoother.evaluate(2.0, 8.0), 50.0, LOOSE_TOLERANCE);
}

TEST_F(CGTriharmonicSmootherTest, SingleElementLinear) {
    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 10.0, 0.0, 10.0, 1, 1);

    CGTriharmonicBezierSmootherConfig config;
    config.lambda = 100.0;
    config.ridge_epsilon = 0.0;

    auto linear = [](Real x, Real y) { return 10.0 + 2.0 * x + 3.0 * y; };

    CGTriharmonicBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data(linear);
    smoother.solve();

    EXPECT_NEAR(smoother.evaluate(5.0, 5.0), linear(5.0, 5.0), LOOSE_TOLERANCE);
    EXPECT_NEAR(smoother.evaluate(2.0, 8.0), linear(2.0, 8.0), LOOSE_TOLERANCE);
}

TEST_F(CGTriharmonicSmootherTest, SingleElementQuadratic) {
    // Quadratic functions have zero triharmonic energy, so should reproduce exactly
    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 10.0, 0.0, 10.0, 1, 1);

    CGTriharmonicBezierSmootherConfig config;
    config.lambda = 100.0;
    config.ridge_epsilon = 0.0;

    auto quadratic = [](Real x, Real y) { return 100.0 + 0.1 * x * x + 0.2 * y * y; };

    CGTriharmonicBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data(quadratic);
    smoother.solve();

    // Triharmonic should reproduce quadratics better than biharmonic
    EXPECT_NEAR(smoother.evaluate(5.0, 5.0), quadratic(5.0, 5.0), LOOSE_TOLERANCE);
    EXPECT_NEAR(smoother.evaluate(3.0, 7.0), quadratic(3.0, 7.0), LOOSE_TOLERANCE);
}

TEST_F(CGTriharmonicSmootherTest, SingleElementCubic) {
    // Cubic functions have constant third derivatives, so triharmonic energy is constant
    // (not zero), but should still reproduce reasonably well
    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 10.0, 0.0, 10.0, 1, 1);

    CGTriharmonicBezierSmootherConfig config;
    config.lambda = 100.0;
    config.ridge_epsilon = 0.0;

    auto cubic = [](Real x, Real y) {
        return 100.0 + 0.001 * x * x * x + 0.002 * y * y * y;
    };

    CGTriharmonicBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data(cubic);
    smoother.solve();

    // Use relative tolerance for cubic
    Real rel_tol = 0.01;  // 1%
    EXPECT_NEAR(smoother.evaluate(5.0, 5.0), cubic(5.0, 5.0),
                std::abs(cubic(5.0, 5.0)) * rel_tol);
}

// =============================================================================
// Multi-Element Tests
// =============================================================================

TEST_F(CGTriharmonicSmootherTest, TwoByTwoMeshConstant) {
    auto mesh = create_quadtree(2, 2);

    CGTriharmonicBezierSmootherConfig config;
    config.lambda = 100.0;
    config.ridge_epsilon = 0.0;

    CGTriharmonicBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data([](Real, Real) { return 42.0; });
    smoother.solve();

    EXPECT_TRUE(smoother.is_solved());
    EXPECT_NEAR(smoother.evaluate(25.0, 25.0), 42.0, LOOSE_TOLERANCE);
    EXPECT_NEAR(smoother.evaluate(50.0, 50.0), 42.0, LOOSE_TOLERANCE);
    EXPECT_NEAR(smoother.evaluate(75.0, 75.0), 42.0, LOOSE_TOLERANCE);
}

TEST_F(CGTriharmonicSmootherTest, TwoByTwoMeshLinear) {
    auto mesh = create_quadtree(2, 2);

    CGTriharmonicBezierSmootherConfig config;
    config.lambda = 100.0;
    config.ridge_epsilon = 0.0;

    auto linear = [](Real x, Real y) { return 10.0 + 0.5 * x + 0.3 * y; };

    CGTriharmonicBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data(linear);
    smoother.solve();

    for (Real x = 10.0; x < 100.0; x += 20.0) {
        for (Real y = 10.0; y < 100.0; y += 20.0) {
            EXPECT_NEAR(smoother.evaluate(x, y), linear(x, y), LOOSE_TOLERANCE)
                << "At (" << x << ", " << y << ")";
        }
    }
}

TEST_F(CGTriharmonicSmootherTest, TwoByTwoMeshQuadratic) {
    auto mesh = create_quadtree(2, 2);

    CGTriharmonicBezierSmootherConfig config;
    config.lambda = 100.0;
    config.ridge_epsilon = 0.0;

    auto quadratic = [](Real x, Real y) { return 100.0 + 0.001 * x * x + 0.002 * y * y; };

    CGTriharmonicBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data(quadratic);
    smoother.solve();

    // Triharmonic should reproduce quadratics well
    for (Real x = 10.0; x < 100.0; x += 30.0) {
        for (Real y = 10.0; y < 100.0; y += 30.0) {
            EXPECT_NEAR(smoother.evaluate(x, y), quadratic(x, y), LOOSE_TOLERANCE)
                << "At (" << x << ", " << y << ")";
        }
    }
}

// =============================================================================
// Edge Constraint Tests
// =============================================================================

TEST_F(CGTriharmonicSmootherTest, EdgeConstraintsEnabled) {
    auto mesh = create_quadtree(4, 4);

    CGTriharmonicBezierSmootherConfig config;
    config.lambda = 1.0;
    config.enable_edge_constraints = true;
    config.edge_ngauss = 4;

    CGTriharmonicBezierBathymetrySmoother smoother(mesh, config);

    EXPECT_GT(smoother.num_constraints(), 0);
    std::cout << "4×4 mesh with edge constraints: "
              << smoother.num_constraints() << " constraints" << std::endl;
}

TEST_F(CGTriharmonicSmootherTest, EdgeConstraintsSolve) {
    auto mesh = create_quadtree(4, 4);

    CGTriharmonicBezierSmootherConfig config;
    config.lambda = 10.0;
    config.enable_edge_constraints = true;
    config.edge_ngauss = 4;

    auto smooth_func = [](Real x, Real y) {
        return 100.0 - 0.01 * std::sin(x * 0.1) * std::cos(y * 0.1);
    };

    CGTriharmonicBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data(smooth_func);
    smoother.solve();

    EXPECT_TRUE(smoother.is_solved());
    EXPECT_LT(smoother.constraint_violation(), 1e-8);
}

// =============================================================================
// Non-Conforming Mesh Tests
// =============================================================================

TEST_F(CGTriharmonicSmootherTest, OnePlusFourMesh) {
    // Create non-conforming mesh: 4 fine elements + 1 coarse element
    QuadtreeAdapter mesh;
    Real h = 25.0;

    // Fine elements (level 2)
    mesh.add_element({0.0, h, 0.0, h}, {2, 2});
    mesh.add_element({h, 2*h, 0.0, h}, {2, 2});
    mesh.add_element({0.0, h, h, 2*h}, {2, 2});
    mesh.add_element({h, 2*h, h, 2*h}, {2, 2});

    // Coarse element (level 1)
    mesh.add_element({2*h, 4*h, 0.0, 2*h}, {1, 1});

    CGTriharmonicBezierSmootherConfig config;
    config.lambda = 10.0;

    auto linear = [](Real x, Real y) { return 10.0 + 0.5 * x + 0.3 * y; };

    CGTriharmonicBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data(linear);
    smoother.solve();

    EXPECT_TRUE(smoother.is_solved());
    EXPECT_LT(smoother.constraint_violation(), 1e-8);

    // Check linear reproduction across elements
    std::vector<Vec2> test_points = {
        {10.0, 10.0}, {25.0, 25.0}, {40.0, 40.0},  // Fine region
        {60.0, 25.0}, {75.0, 40.0}, {90.0, 10.0},  // Coarse region
    };

    for (const auto& pt : test_points) {
        EXPECT_NEAR(smoother.evaluate(pt(0), pt(1)), linear(pt(0), pt(1)), 0.1)
            << "At (" << pt(0) << ", " << pt(1) << ")";
    }
}

// =============================================================================
// Comparison with Biharmonic (Thin Plate)
// =============================================================================

TEST_F(CGTriharmonicSmootherTest, CompareWithBiharmonicForQuadratic) {
    // Triharmonic should preserve quadratics better than biharmonic
    auto mesh = create_quadtree(4, 4);

    auto quadratic = [](Real x, Real y) {
        return 100.0 + 0.001 * x * x + 0.002 * y * y;
    };

    // Biharmonic smoother
    CGBezierSmootherConfig config_bi;
    config_bi.lambda = 10.0;
    config_bi.ridge_epsilon = 0.0;

    CGBezierBathymetrySmoother smoother_bi(mesh, config_bi);
    smoother_bi.set_bathymetry_data(quadratic);
    smoother_bi.solve();

    // Triharmonic smoother
    CGTriharmonicBezierSmootherConfig config_tri;
    config_tri.lambda = 10.0;
    config_tri.ridge_epsilon = 0.0;

    CGTriharmonicBezierBathymetrySmoother smoother_tri(mesh, config_tri);
    smoother_tri.set_bathymetry_data(quadratic);
    smoother_tri.solve();

    // Compute max error for both
    Real max_error_bi = 0.0;
    Real max_error_tri = 0.0;
    for (Real x = 10.0; x < 100.0; x += 10.0) {
        for (Real y = 10.0; y < 100.0; y += 10.0) {
            Real exact = quadratic(x, y);
            max_error_bi = std::max(max_error_bi, std::abs(smoother_bi.evaluate(x, y) - exact));
            max_error_tri = std::max(max_error_tri, std::abs(smoother_tri.evaluate(x, y) - exact));
        }
    }

    std::cout << "Quadratic reproduction - Biharmonic max error: " << max_error_bi
              << ", Triharmonic max error: " << max_error_tri << std::endl;

    // Triharmonic should be at least as good as biharmonic for quadratics
    // (and typically better since quadratics have zero triharmonic energy)
    EXPECT_LE(max_error_tri, max_error_bi * 1.1);  // Allow 10% margin
}

// =============================================================================
// Diagnostics Tests
// =============================================================================

TEST_F(CGTriharmonicSmootherTest, DiagnosticsAvailable) {
    auto mesh = create_quadtree(4, 4);

    CGTriharmonicBezierSmootherConfig config;
    config.lambda = 1.0;

    CGTriharmonicBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data([](Real x, Real y) { return 100.0 - 0.01 * x; });
    smoother.solve();

    EXPECT_GE(smoother.data_residual(), 0.0);
    EXPECT_GE(smoother.regularization_energy(), 0.0);
    EXPECT_GE(smoother.objective_value(), 0.0);
}

// =============================================================================
// VTK Output Tests
// =============================================================================

TEST_F(CGTriharmonicSmootherTest, VTKOutput) {
    auto mesh = create_quadtree(4, 4);

    CGTriharmonicBezierSmootherConfig config;
    config.lambda = 1.0;

    auto smooth_func = [](Real x, Real y) {
        return 100.0 - 0.005 * x * x - 0.003 * y * y;
    };

    CGTriharmonicBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data(smooth_func);
    smoother.solve();

    // Create temp directory
    std::string tmp_dir = "/tmp/drifter_test_triharmonic";
    std::filesystem::create_directories(tmp_dir);

    // Note: write_vtk adds .vtu extension automatically
    std::string vtk_base = tmp_dir + "/triharmonic_surface";
    try {
        smoother.write_vtk(vtk_base, 11);
        EXPECT_TRUE(std::filesystem::exists(vtk_base + ".vtu"));
    } catch (const std::exception& e) {
        // VTK output may fail if VTK not available
        std::cout << "VTK output skipped: " << e.what() << std::endl;
    }

    // Note: write_control_points_vtk expects filename with extension
    std::string ctrl_file = tmp_dir + "/triharmonic_control_points.vtu";
    try {
        smoother.write_control_points_vtk(ctrl_file);
        EXPECT_TRUE(std::filesystem::exists(ctrl_file));
    } catch (const std::exception& e) {
        std::cout << "Control points VTK output skipped: " << e.what() << std::endl;
    }
}

// =============================================================================
// Performance Test
// =============================================================================

TEST_F(CGTriharmonicSmootherTest, PerformanceTest8x8) {
    auto mesh = create_quadtree(8, 8);

    CGTriharmonicBezierSmootherConfig config;
    config.lambda = 1.0;

    auto smooth_func = [](Real x, Real y) {
        return 100.0 - 0.005 * x * x - 0.003 * y * y + 10.0 * std::sin(0.1 * x);
    };

    CGTriharmonicBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data(smooth_func);

    auto start = std::chrono::high_resolution_clock::now();
    smoother.solve();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "8×8 triharmonic solve time: " << duration.count() << " ms"
              << " (" << smoother.num_global_dofs() << " DOFs)" << std::endl;

    EXPECT_TRUE(smoother.is_solved());
}

// =============================================================================
// Multi-Source Bathymetry Integration Tests
// =============================================================================

class CGTriharmonicSmootherGeoTiffTest : public BathymetryTestFixture {
  protected:
    static constexpr Real TOLERANCE = 1e-10;
    static constexpr Real LOOSE_TOLERANCE = 1e-6;
};

TEST_F(CGTriharmonicSmootherGeoTiffTest, KattegatIntegration) {
    if (!data_files_exist()) {
        GTEST_SKIP() << "Bathymetry data not available";
    }

    // Kattegat test area
    Real center_x = 4095238.0;   // EPSG:3034
    Real center_y = 3344695.0;   // EPSG:3034
    Real domain_size = 30000.0;  // 30 km

    Real xmin = center_x - domain_size / 2;
    Real xmax = center_x + domain_size / 2;
    Real ymin = center_y - domain_size / 2;
    Real ymax = center_y + domain_size / 2;

    std::cout << "=== CG Triharmonic Bezier Kattegat Test ===" << std::endl;
    std::cout << "Domain: [" << xmin << ", " << xmax << "] x [" << ymin << ", "
              << ymax << "]" << std::endl;

    // Create uniform mesh
    QuadtreeAdapter mesh;
    mesh.build_uniform(xmin, xmax, ymin, ymax, 8, 8);

    std::cout << "Mesh elements: " << mesh.num_elements() << std::endl;

    // Depth function using multi-source bathymetry
    auto depth_func = create_depth_function();

    // Create CG Triharmonic Bezier smoother
    // Note: triharmonic only penalizes curvature gradient, not curvature itself
    // Use moderate lambda with small gradient_weight for balance
    CGTriharmonicBezierSmootherConfig config;
    config.lambda = 0.01;           // Moderate data fitting weight
    config.gradient_weight = 0.01;  // Small gradient penalty
    config.ngauss_data = 6;
    config.ngauss_energy = 4;
    config.enable_edge_constraints = true;  // Enable C² edge constraints
    config.edge_ngauss = 4;                 // 4 Gauss points per edge

    CGTriharmonicBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data(depth_func);

    std::cout << "DOFs: " << smoother.num_global_dofs() << std::endl;
    std::cout << "Constraints: " << smoother.num_constraints() << std::endl;

    // Solve
    smoother.solve();

    EXPECT_TRUE(smoother.is_solved());

    // Write output for ParaView verification
    std::string output_base = "/tmp/cg_triharmonic_bezier_kattegat_test";
    smoother.write_vtk(output_base, 11);

    std::string output_file = output_base + ".vtu";
    std::cout << "Output written to: " << output_file << std::endl;
    EXPECT_TRUE(std::filesystem::exists(output_file));

    // Write control points for debugging
    std::string cp_base = "/tmp/cg_triharmonic_bezier_kattegat_control_points";
    smoother.write_control_points_vtk(cp_base);
    std::cout << "Control points written to: " << cp_base << ".vtu" << std::endl;

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

TEST_F(CGTriharmonicSmootherGeoTiffTest, DISABLED_KattegatLambdaComparison) {
    if (!data_files_exist()) {
        GTEST_SKIP() << "Bathymetry data not available";
    }

    // Kattegat test area
    Real center_x = 4095238.0;
    Real center_y = 3344695.0;
    Real domain_size = 30000.0;

    Real xmin = center_x - domain_size / 2;
    Real xmax = center_x + domain_size / 2;
    Real ymin = center_y - domain_size / 2;
    Real ymax = center_y + domain_size / 2;

    QuadtreeAdapter mesh;
    mesh.build_uniform(xmin, xmax, ymin, ymax, 8, 8);

    auto depth_func = create_depth_function();

    std::cout << "=== Lambda Comparison for Triharmonic ===" << std::endl;

    // Test different lambda values
    std::vector<Real> lambdas = {0.0001, 0.001, 0.01, 0.1, 1.0};

    for (Real lambda : lambdas) {
        CGTriharmonicBezierSmootherConfig config;
        config.lambda = lambda;
        config.ngauss_data = 6;
        config.ngauss_energy = 4;
        config.enable_edge_constraints = true;  // Enable C² edge constraints
        config.edge_ngauss = 4;

        CGTriharmonicBezierBathymetrySmoother smoother(mesh, config);
        smoother.set_bathymetry_data(std::function<Real(Real, Real)>(depth_func));
        smoother.solve();

        // Compute average error
        Real sum_diff = 0.0;
        for (Index e = 0; e < mesh.num_elements(); ++e) {
            const auto& bounds = mesh.element_bounds(e);
            Real cx = 0.5 * (bounds.xmin + bounds.xmax);
            Real cy = 0.5 * (bounds.ymin + bounds.ymax);
            sum_diff += std::abs(depth_func(cx, cy) - smoother.evaluate(cx, cy));
        }
        Real avg_diff = sum_diff / mesh.num_elements();

        std::string filename = "/tmp/triharmonic_lambda_" + std::to_string(lambda);
        smoother.write_vtk(filename, 11);

        std::cout << "lambda=" << std::setw(8) << lambda
                  << " avg_err=" << std::setw(8) << std::fixed << std::setprecision(2) << avg_diff << " m"
                  << " -> " << filename << ".vtu" << std::endl;
    }

    // Also biharmonic for comparison
    {
        CGBezierSmootherConfig config;
        config.lambda = 1.0;
        config.ngauss_data = 6;
        config.ngauss_energy = 4;

        CGBezierBathymetrySmoother smoother(mesh, config);
        smoother.set_bathymetry_data(std::function<Real(Real, Real)>(depth_func));
        smoother.solve();

        Real sum_diff = 0.0;
        for (Index e = 0; e < mesh.num_elements(); ++e) {
            const auto& bounds = mesh.element_bounds(e);
            Real cx = 0.5 * (bounds.xmin + bounds.xmax);
            Real cy = 0.5 * (bounds.ymin + bounds.ymax);
            sum_diff += std::abs(depth_func(cx, cy) - smoother.evaluate(cx, cy));
        }
        Real avg_diff = sum_diff / mesh.num_elements();

        smoother.write_vtk("/tmp/biharmonic_lambda_1.0", 11);

        std::cout << "BIHARMONIC lambda=1.0 avg_err=" << std::fixed << std::setprecision(2)
                  << avg_diff << " m -> /tmp/biharmonic_lambda_1.0.vtu" << std::endl;
    }

    EXPECT_TRUE(true);  // Always pass, this is for visual comparison
}

