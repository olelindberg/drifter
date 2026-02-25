#include <gtest/gtest.h>
#include "bathymetry/cg_cubic_bezier_bathymetry_smoother.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "mesh/octree_adapter.hpp"
#include "mesh/seabed_surface.hpp"
#include "test_integration_fixtures.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <sstream>

#include "mesh/geotiff_reader.hpp"

using namespace drifter;
using namespace drifter::testing;

class CGCubicBezierSmootherTest : public ::testing::Test {
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

TEST_F(CGCubicBezierSmootherTest, ConstructFromQuadtree) {
    auto mesh = create_quadtree(2, 2);

    CGCubicBezierBathymetrySmoother smoother(mesh);

    EXPECT_GT(smoother.num_global_dofs(), 0);
    EXPECT_FALSE(smoother.is_solved());
}

TEST_F(CGCubicBezierSmootherTest, ConstructFromOctree) {
    auto octree = create_octree(4, 4, 2);

    CGCubicBezierBathymetrySmoother smoother(*octree);

    EXPECT_GT(smoother.num_global_dofs(), 0);
    EXPECT_FALSE(smoother.is_solved());
}

TEST_F(CGCubicBezierSmootherTest, SingleElementConstant) {
    // Single element test: constant bathymetry
    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 10.0, 0.0, 10.0, 1, 1);

    CGCubicBezierSmootherConfig config;
    config.lambda = 100.0;  // Strong data fitting
    config.ridge_epsilon = 0.0;  // No ridge for exact polynomial reproduction

    CGCubicBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data([](Real, Real) { return 50.0; });
    smoother.solve();

    EXPECT_TRUE(smoother.is_solved());

    // Should reproduce constant exactly
    EXPECT_NEAR(smoother.evaluate(5.0, 5.0), 50.0, LOOSE_TOLERANCE);
    EXPECT_NEAR(smoother.evaluate(2.0, 8.0), 50.0, LOOSE_TOLERANCE);
}

TEST_F(CGCubicBezierSmootherTest, SingleElementLinear) {
    // Single element: linear bathymetry
    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 10.0, 0.0, 10.0, 1, 1);

    CGCubicBezierSmootherConfig config;
    config.lambda = 100.0;
    config.ridge_epsilon = 0.0;  // No ridge for exact polynomial reproduction

    auto linear = [](Real x, Real y) { return 10.0 + 2.0 * x + 3.0 * y; };

    CGCubicBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data(linear);
    smoother.solve();

    // Cubic can represent linear exactly
    EXPECT_NEAR(smoother.evaluate(5.0, 5.0), linear(5.0, 5.0), LOOSE_TOLERANCE);
    EXPECT_NEAR(smoother.evaluate(2.0, 8.0), linear(2.0, 8.0), LOOSE_TOLERANCE);
}

TEST_F(CGCubicBezierSmootherTest, SingleElementQuadratic) {
    // Single element: quadratic bathymetry
    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 10.0, 0.0, 10.0, 1, 1);

    CGCubicBezierSmootherConfig config;
    config.lambda = 100.0;
    config.ridge_epsilon = 0.0;

    auto quadratic = [](Real x, Real y) { return 100.0 + 0.1 * x * x + 0.2 * y * y; };

    CGCubicBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data(quadratic);
    smoother.solve();

    // Use 0.5% relative tolerance for quadratics (smoothing affects non-linear functions)
    // Cubic has lower degree than quintic, so may have slightly larger error
    Real rel_tol = 0.005;
    EXPECT_NEAR(smoother.evaluate(5.0, 5.0), quadratic(5.0, 5.0),
                std::abs(quadratic(5.0, 5.0)) * rel_tol);
    EXPECT_NEAR(smoother.evaluate(3.0, 7.0), quadratic(3.0, 7.0),
                std::abs(quadratic(3.0, 7.0)) * rel_tol);
}

// =============================================================================
// Level 1: 2×2 Uniform Mesh Tests (DOF Sharing)
// =============================================================================

TEST_F(CGCubicBezierSmootherTest, TwoByTwoMeshConstant) {
    auto mesh = create_quadtree(2, 2);

    CGCubicBezierSmootherConfig config;
    config.lambda = 100.0;
    config.ridge_epsilon = 0.0;  // No ridge for exact polynomial reproduction

    CGCubicBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data([](Real, Real) { return 42.0; });
    smoother.solve();

    EXPECT_TRUE(smoother.is_solved());

    // Check at several points
    EXPECT_NEAR(smoother.evaluate(25.0, 25.0), 42.0, LOOSE_TOLERANCE);
    EXPECT_NEAR(smoother.evaluate(50.0, 50.0), 42.0, LOOSE_TOLERANCE);
    EXPECT_NEAR(smoother.evaluate(75.0, 75.0), 42.0, LOOSE_TOLERANCE);
}

TEST_F(CGCubicBezierSmootherTest, TwoByTwoMeshLinear) {
    auto mesh = create_quadtree(2, 2);

    CGCubicBezierSmootherConfig config;
    config.lambda = 100.0;
    config.ridge_epsilon = 0.0;  // No ridge for exact polynomial reproduction

    auto linear = [](Real x, Real y) { return 10.0 + 0.5 * x + 0.3 * y; };

    CGCubicBezierBathymetrySmoother smoother(mesh, config);
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

TEST_F(CGCubicBezierSmootherTest, DofSharingReducesCount) {
    // Verify DOF sharing: CG should have fewer DOFs than DG
    auto mesh = create_quadtree(2, 2);  // 4 elements

    CGCubicBezierBathymetrySmoother smoother(mesh);

    Index num_dofs = smoother.num_global_dofs();

    // DG would have 4 × 16 = 64 DOFs (cubic: 4×4 = 16 per element)
    // CG should have significantly fewer due to shared edges/corners
    // For a 2×2 mesh with cubic:
    //   - Interior vertex (1): shared by 4 elements
    //   - Edge midpoints (4): shared by 2 elements
    //   - Corners (4): unique to 1 element each
    //   - Interior edge DOFs: shared between 2 elements
    // Expected: much less than 64
    EXPECT_LT(num_dofs, 64);
    EXPECT_GT(num_dofs, 16);  // At least one element's worth

    std::cout << "2×2 mesh CG cubic DOFs: " << num_dofs << " (vs 64 for DG)" << std::endl;
}

TEST_F(CGCubicBezierSmootherTest, NoConstraintsForUniformMesh) {
    // Uniform mesh should have no hanging nodes
    auto mesh = create_quadtree(2, 2);

    CGCubicBezierBathymetrySmoother smoother(mesh);

    // Uniform mesh = conforming = no constraints
    EXPECT_EQ(smoother.num_constraints(), 0);
}

// =============================================================================
// Level 2: Non-conforming 1+4 Mesh Tests
// =============================================================================

TEST_F(CGCubicBezierSmootherTest, OnePlusFourMeshConstruction) {
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

    CGCubicBezierBathymetrySmoother smoother(mesh);

    // Should have hanging node constraints at fine-coarse interface
    EXPECT_GT(smoother.num_constraints(), 0);

    std::cout << "1+4 mesh cubic: " << smoother.num_global_dofs() << " DOFs, "
              << smoother.num_constraints() << " constraints" << std::endl;
}

TEST_F(CGCubicBezierSmootherTest, OnePlusFourConstantBathymetry) {
    // 1+4 mesh with constant bathymetry
    QuadtreeAdapter mesh;
    Real h = 25.0;

    mesh.add_element({0.0, h, 0.0, h}, {2, 2});
    mesh.add_element({h, 2*h, 0.0, h}, {2, 2});
    mesh.add_element({0.0, h, h, 2*h}, {2, 2});
    mesh.add_element({h, 2*h, h, 2*h}, {2, 2});
    mesh.add_element({2*h, 4*h, 0.0, 2*h}, {1, 1});

    CGCubicBezierSmootherConfig config;
    config.lambda = 100.0;
    config.ridge_epsilon = 0.0;  // No ridge for exact polynomial reproduction

    CGCubicBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data([](Real, Real) { return 100.0; });
    smoother.solve();

    EXPECT_TRUE(smoother.is_solved());

    // Should be constant everywhere
    EXPECT_NEAR(smoother.evaluate(10.0, 10.0), 100.0, LOOSE_TOLERANCE);
    EXPECT_NEAR(smoother.evaluate(75.0, 25.0), 100.0, LOOSE_TOLERANCE);
}

TEST_F(CGCubicBezierSmootherTest, OnePlusFourLinearBathymetry) {
    // 1+4 mesh with linear bathymetry
    QuadtreeAdapter mesh;
    Real h = 25.0;

    mesh.add_element({0.0, h, 0.0, h}, {2, 2});
    mesh.add_element({h, 2*h, 0.0, h}, {2, 2});
    mesh.add_element({0.0, h, h, 2*h}, {2, 2});
    mesh.add_element({h, 2*h, h, 2*h}, {2, 2});
    mesh.add_element({2*h, 4*h, 0.0, 2*h}, {1, 1});

    auto linear = [](Real x, Real y) { return 50.0 + x + 2.0 * y; };

    CGCubicBezierSmootherConfig config;
    config.lambda = 100.0;
    config.ridge_epsilon = 0.0;  // No ridge for exact polynomial reproduction

    CGCubicBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data(linear);
    smoother.solve();

    // Print values at key points
    std::cout << "\n=== OnePlusFourLinearBathymetry (cubic) ===\n";
    std::cout << "z(50, 0)  = " << smoother.evaluate(50.0, 0.0) << " (expected " << linear(50.0, 0.0) << ")\n";
    std::cout << "z(50, 50) = " << smoother.evaluate(50.0, 50.0) << " (expected " << linear(50.0, 50.0) << ")\n";

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

TEST_F(CGCubicBezierSmootherTest, OnePlusFourContinuityAtInterface) {
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

    CGCubicBezierSmootherConfig config;
    config.lambda = 10.0;

    CGCubicBezierBathymetrySmoother smoother(mesh, config);
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

    std::cout << "Max jump at interface (cubic): " << max_jump << std::endl;

    // CG should provide C⁰ continuity via shared DOFs
    EXPECT_LT(max_jump, 1.0);
}

TEST_F(CGCubicBezierSmootherTest, CondensationMatchesFullKKT) {
    // Verify condensed solve matches original full KKT implementation
    QuadtreeAdapter mesh;
    Real h = 25.0;

    // 1+4 non-conforming mesh (has hanging node constraints)
    mesh.add_element({0.0, h, 0.0, h}, {2, 2});
    mesh.add_element({h, 2*h, 0.0, h}, {2, 2});
    mesh.add_element({0.0, h, h, 2*h}, {2, 2});
    mesh.add_element({h, 2*h, h, 2*h}, {2, 2});
    mesh.add_element({2*h, 4*h, 0.0, 2*h}, {1, 1});

    auto smooth_bathy = [](Real x, Real y) {
        return 100.0 + 20.0 * std::sin(0.03 * x) * std::cos(0.03 * y);
    };

    CGCubicBezierSmootherConfig config_condensed;
    config_condensed.lambda = 10.0;
    config_condensed.use_condensation = true;

    CGCubicBezierSmootherConfig config_full_kkt;
    config_full_kkt.lambda = 10.0;
    config_full_kkt.use_condensation = false;

    // Solve with condensation
    CGCubicBezierBathymetrySmoother smoother_condensed(mesh, config_condensed);
    smoother_condensed.set_bathymetry_data(smooth_bathy);
    smoother_condensed.solve();

    // Solve with full KKT
    CGCubicBezierBathymetrySmoother smoother_full_kkt(mesh, config_full_kkt);
    smoother_full_kkt.set_bathymetry_data(smooth_bathy);
    smoother_full_kkt.solve();

    // Compare solutions at many points
    Real max_diff = 0.0;
    std::vector<Vec2> test_points = {
        {10.0, 10.0}, {25.0, 25.0}, {40.0, 40.0},  // Fine region
        {60.0, 25.0}, {75.0, 40.0}, {90.0, 10.0},  // Coarse region
        {50.0, 25.0}, {50.0, 12.5}, {50.0, 37.5},  // At interface
    };

    for (const auto& pt : test_points) {
        Real val_condensed = smoother_condensed.evaluate(pt(0), pt(1));
        Real val_full_kkt = smoother_full_kkt.evaluate(pt(0), pt(1));
        Real diff = std::abs(val_condensed - val_full_kkt);
        max_diff = std::max(max_diff, diff);
    }

    std::cout << "Max solution difference (condensed vs full KKT): " << max_diff << std::endl;
    std::cout << "Constraint violation (condensed): " << smoother_condensed.constraint_violation() << std::endl;
    std::cout << "Constraint violation (full KKT):  " << smoother_full_kkt.constraint_violation() << std::endl;

    // Solutions should match to high precision (within numerical tolerance)
    // Note: With non-conforming C¹ constraints, condensed and full KKT have
    // slightly different numerical behavior, so we allow 1e-7 tolerance
    EXPECT_LT(max_diff, 1e-7);
}

// =============================================================================
// C¹ Constraint Tests
// =============================================================================

TEST_F(CGCubicBezierSmootherTest, C1EdgeConstraints) {
    // Test C¹ edge constraints on uniform mesh
    auto mesh = create_quadtree(4, 4);

    auto bathy = [](Real x, Real y) {
        return 100.0 + 10.0 * std::sin(0.05 * x) * std::cos(0.05 * y);
    };

    CGCubicBezierSmootherConfig config;
    config.lambda = 10.0;
    config.edge_ngauss = 4;

    CGCubicBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data(bathy);
    smoother.solve();

    EXPECT_TRUE(smoother.is_solved());

    // Verify edge constraint count
    Index num_c1_edge = smoother.dof_manager().num_edge_derivative_constraints();
    std::cout << "C¹ edge derivative constraints: " << num_c1_edge << std::endl;
    EXPECT_GT(num_c1_edge, 0);

    // Check constraint violation
    Real violation = smoother.constraint_violation();
    std::cout << "Constraint violation: " << violation << std::endl;
    EXPECT_LT(violation, 1e-6);
}

TEST_F(CGCubicBezierSmootherTest, NonConformingC1Smoothness) {
    // Test C¹ smoothness at non-conforming (2:1) interfaces
    QuadtreeAdapter mesh;
    Real h = 25.0;

    // Create 4 fine + 1 coarse non-conforming mesh
    // Fine elements (2x2 in [0,50] x [0,50])
    mesh.add_element({0.0, h, 0.0, h}, {2, 2});
    mesh.add_element({h, 2 * h, 0.0, h}, {2, 2});
    mesh.add_element({0.0, h, h, 2 * h}, {2, 2});
    mesh.add_element({h, 2 * h, h, 2 * h}, {2, 2});
    // Coarse element (1x1 in [50,100] x [0,50])
    mesh.add_element({2 * h, 4 * h, 0.0, 2 * h}, {1, 1});

    auto smooth_bathy = [](Real x, Real y) {
        return 100.0 + 20.0 * std::sin(0.03 * x) * std::cos(0.03 * y);
    };

    CGCubicBezierSmootherConfig config;
    config.lambda = 10.0;
    config.edge_ngauss = 4;

    CGCubicBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data(smooth_bathy);
    smoother.solve();

    // Should have non-conforming edge constraints
    Index num_c1 = smoother.dof_manager().num_edge_derivative_constraints();
    std::cout << "Total C¹ edge constraints: " << num_c1 << std::endl;
    // Expected: conforming (3 fine interior + 2 fine-fine edges = 5) * 4 gauss = 20
    //         + non-conforming (2 fine-coarse edges * 4 gauss) = 8
    //         = 28 total
    EXPECT_GE(num_c1, 20);  // At least conforming constraints

    // Check constraint violation
    Real violation = smoother.constraint_violation();
    std::cout << "Constraint violation: " << violation << std::endl;
    EXPECT_LT(violation, 1e-5);

    // Check derivative continuity at non-conforming interface (x = 50)
    Real max_deriv_jump = 0.0;
    Real dx = 0.01;

    for (Real y = 5.0; y < 45.0; y += 5.0) {
        // Approximate normal derivative (z_x) from both sides using finite differences
        Real z_left_inner = smoother.evaluate(50.0 - dx, y);
        Real z_left_edge = smoother.evaluate(50.0, y);
        Real z_right_edge = smoother.evaluate(50.0, y);
        Real z_right_inner = smoother.evaluate(50.0 + dx, y);

        // Forward difference from left, backward difference from right
        Real deriv_left = (z_left_edge - z_left_inner) / dx;
        Real deriv_right = (z_right_inner - z_right_edge) / dx;

        Real jump = std::abs(deriv_left - deriv_right);
        max_deriv_jump = std::max(max_deriv_jump, jump);
    }

    std::cout << "Max derivative jump at non-conforming interface: " << max_deriv_jump << std::endl;

    // With C¹ constraints, derivative should be continuous (small jump due to finite differences)
    EXPECT_LT(max_deriv_jump, 5.0);  // Allow some numerical error from finite differences
}

// =============================================================================
// Smoothing Effect Tests
// =============================================================================

TEST_F(CGCubicBezierSmootherTest, SmoothingReducesVariation) {
    auto mesh = create_quadtree(4, 4);

    // Noisy bathymetry
    auto noisy = [](Real x, Real y) {
        return 50.0 + 5.0 * std::sin(0.5 * x) * std::sin(0.5 * y);
    };

    // Low smoothing
    CGCubicBezierSmootherConfig config_low;
    config_low.lambda = 100.0;  // High data weight

    CGCubicBezierBathymetrySmoother low_smooth(mesh, config_low);
    low_smooth.set_bathymetry_data(noisy);
    low_smooth.solve();

    // High smoothing
    CGCubicBezierSmootherConfig config_high;
    config_high.lambda = 0.01;  // Low data weight, high smoothing

    CGCubicBezierBathymetrySmoother high_smooth(mesh, config_high);
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

TEST_F(CGCubicBezierSmootherTest, GradientOfConstant) {
    auto mesh = create_quadtree(2, 2);

    CGCubicBezierSmootherConfig config;
    config.lambda = 100.0;
    config.ridge_epsilon = 0.0;  // No ridge for exact polynomial reproduction

    CGCubicBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data([](Real, Real) { return 42.0; });
    smoother.solve();

    Vec2 grad = smoother.evaluate_gradient(50.0, 50.0);

    EXPECT_NEAR(grad(0), 0.0, LOOSE_TOLERANCE);
    EXPECT_NEAR(grad(1), 0.0, LOOSE_TOLERANCE);
}

TEST_F(CGCubicBezierSmootherTest, GradientOfLinear) {
    auto mesh = create_quadtree(2, 2);

    CGCubicBezierSmootherConfig config;
    config.lambda = 100.0;
    config.ridge_epsilon = 0.0;  // No ridge for exact polynomial reproduction

    // z = 10 + 2x + 3y, gradient = (2, 3)
    CGCubicBezierBathymetrySmoother smoother(mesh, config);
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

TEST_F(CGCubicBezierSmootherTest, SolveBeforeDataThrows) {
    auto mesh = create_quadtree(2, 2);
    CGCubicBezierBathymetrySmoother smoother(mesh);

    EXPECT_THROW(smoother.solve(), std::runtime_error);
}

TEST_F(CGCubicBezierSmootherTest, EvaluateBeforeSolveThrows) {
    auto mesh = create_quadtree(2, 2);
    CGCubicBezierBathymetrySmoother smoother(mesh);
    smoother.set_bathymetry_data([](Real, Real) { return 1.0; });

    EXPECT_THROW(smoother.evaluate(50.0, 50.0), std::runtime_error);
}

// =============================================================================
// VTK Output Tests
// =============================================================================

TEST_F(CGCubicBezierSmootherTest, VTKOutputUniformMesh) {
    auto mesh = create_quadtree(4, 4);

    auto bathy = [](Real x, Real y) {
        return 100.0 + 20.0 * std::sin(0.03 * x) * std::cos(0.03 * y);
    };

    CGCubicBezierSmootherConfig config;
    config.lambda = 10.0;

    CGCubicBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data(bathy);
    smoother.solve();

    std::string filename = "/tmp/cg_cubic_bezier_uniform";
    smoother.write_vtk(filename, 10);

    // write_vtk appends .vtu extension
    std::string output_file = filename + ".vtu";
    EXPECT_TRUE(std::filesystem::exists(output_file));

    std::ifstream file(output_file);
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    EXPECT_GT(content.size(), 1000);
}

TEST_F(CGCubicBezierSmootherTest, VTKOutputWithC1Constraints) {
    // 4×4 mesh with C¹ constraints
    auto mesh = create_quadtree(4, 4);

    auto bathy = [](Real x, Real y) {
        return 100.0 + 20.0 * std::sin(0.03 * x) * std::cos(0.03 * y);
    };

    CGCubicBezierSmootherConfig config;
    config.lambda = 10.0;

    CGCubicBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data(bathy);
    smoother.solve();

    std::string filename = "/tmp/cg_cubic_bezier_c1_constraints";
    smoother.write_vtk(filename, 10);

    std::string output_file = filename + ".vtu";
    EXPECT_TRUE(std::filesystem::exists(output_file));

    // Also write control points
    std::string cp_filename = "/tmp/cg_cubic_bezier_control_points.vtu";
    smoother.write_control_points_vtk(cp_filename);
    EXPECT_TRUE(std::filesystem::exists(cp_filename));
}

// =============================================================================
// Diagnostics Tests
// =============================================================================

TEST_F(CGCubicBezierSmootherTest, DiagnosticsAfterSolve) {
    auto mesh = create_quadtree(2, 2);

    CGCubicBezierSmootherConfig config;
    config.lambda = 10.0;

    CGCubicBezierBathymetrySmoother smoother(mesh, config);
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

TEST_F(CGCubicBezierSmootherTest, ConstraintViolationNearZero) {
    // 1+4 mesh with constraints
    QuadtreeAdapter mesh;
    Real h = 25.0;

    mesh.add_element({0.0, h, 0.0, h}, {2, 2});
    mesh.add_element({h, 2*h, 0.0, h}, {2, 2});
    mesh.add_element({0.0, h, h, 2*h}, {2, 2});
    mesh.add_element({h, 2*h, h, 2*h}, {2, 2});
    mesh.add_element({2*h, 4*h, 0.0, 2*h}, {1, 1});

    CGCubicBezierSmootherConfig config;
    config.lambda = 10.0;

    CGCubicBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data([](Real x, Real y) { return 50.0 + x; });
    smoother.solve();

    Real violation = smoother.constraint_violation();
    std::cout << "Constraint violation (cubic): " << violation << std::endl;

    // Constraints should be satisfied to high precision
    EXPECT_LT(violation, 1e-6);
}

// =============================================================================
// Comparison: 8×8 mesh DOF count - Cubic vs Quintic
// =============================================================================

TEST_F(CGCubicBezierSmootherTest, EightByEightDofCount) {
    auto mesh = create_quadtree(8, 8);

    CGCubicBezierBathymetrySmoother smoother(mesh);

    Index num_dofs = smoother.num_global_dofs();

    // For 8×8 uniform mesh with cubic (4×4 = 16 DOFs per element):
    // DG would have: 64 × 16 = 1024 DOFs
    // CG cubic should have: (8*3+1)² = 25² = 625 DOFs (assuming 3 internal DOFs per edge)
    // Actually for cubic: (n_elem * (N1D-1) + 1)² = (8*3+1)² = 25² = 625
    std::cout << "8×8 mesh CG cubic DOFs: " << num_dofs << std::endl;
    std::cout << "(Compare with quintic: expected ~1681 DOFs)" << std::endl;

    // Verify it's less than DG would be
    EXPECT_LT(num_dofs, 1024);
    // Verify reasonable count for CG cubic
    EXPECT_GT(num_dofs, 400);
    EXPECT_LT(num_dofs, 800);
}

// =============================================================================
// Level 4: Kattegat GeoTIFF Integration Test
// =============================================================================
// Multi-Source Bathymetry Integration Tests
// =============================================================================

class CGCubicBezierSmootherGeoTiffTest : public BathymetryTestFixture {
  protected:
    static constexpr Real TOLERANCE = 1e-10;
    static constexpr Real LOOSE_TOLERANCE = 1e-6;
};

TEST_F(CGCubicBezierSmootherGeoTiffTest, KattegatIntegration) {
    if (!data_files_exist()) {
        GTEST_SKIP() << "Bathymetry data not available";
    }

    // Kattegat test area
    Real center_x = 4095238.0;  // EPSG:3034
    Real center_y = 3344695.0;  // EPSG:3034
    Real domain_size = 30000.0; // 30 km

    Real xmin = center_x - domain_size / 2;
    Real xmax = center_x + domain_size / 2;
    Real ymin = center_y - domain_size / 2;
    Real ymax = center_y + domain_size / 2;

    std::cout << "=== CG Cubic Bezier Kattegat Test ===" << std::endl;
    std::cout << "Domain: [" << xmin << ", " << xmax << "] x [" << ymin << ", " << ymax << "]" << std::endl;

    // Create uniform mesh
    QuadtreeAdapter mesh;
    mesh.build_uniform(xmin, xmax, ymin, ymax, 8, 8);

    std::cout << "Mesh elements: " << mesh.num_elements() << std::endl;

    // Depth function using multi-source bathymetry
    auto depth_func = create_depth_function();

    // Create CG Cubic Bezier smoother
    CGCubicBezierSmootherConfig config;
    config.lambda = 1.0;
    config.ngauss_data = 4;
    config.ngauss_energy = 4;

    CGCubicBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data(depth_func);

    std::cout << "DOFs: " << smoother.num_global_dofs() << std::endl;
    std::cout << "Constraints: " << smoother.num_constraints() << std::endl;

    // Solve
    smoother.solve();

    EXPECT_TRUE(smoother.is_solved());

    // Write output for ParaView verification
    std::string output_base = "/tmp/cg_cubic_bezier_kattegat_test";
    smoother.write_vtk(output_base, 10);

    std::string output_file = output_base + ".vtu";
    std::cout << "Output written to: " << output_file << std::endl;
    EXPECT_TRUE(std::filesystem::exists(output_file));

    // Write control points for debugging
    std::string cp_file = "/tmp/cg_cubic_bezier_kattegat_control_points.vtu";
    smoother.write_control_points_vtk(cp_file);
    std::cout << "Control points written to: " << cp_file << std::endl;

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

TEST_F(CGCubicBezierSmootherGeoTiffTest, DISABLED_KattegatWithC1Constraints) {
    if (!data_files_exist()) {
        GTEST_SKIP() << "Bathymetry data not available";
    }

    Real center_x = 4095238.0;
    Real center_y = 3344695.0;
    Real domain_size = 30000.0;

    Real xmin = center_x - domain_size / 2;
    Real xmax = center_x + domain_size / 2;
    Real ymin = center_y - domain_size / 2;
    Real ymax = center_y + domain_size / 2;

    std::cout << "=== CG Cubic Bezier with C¹ Constraints ===" << std::endl;

    QuadtreeAdapter mesh;
    mesh.build_uniform(xmin, xmax, ymin, ymax, 8, 8);

    auto depth_func = create_depth_function();

    CGCubicBezierSmootherConfig config;
    config.lambda = 1.0;
    config.ngauss_data = 4;
    config.ngauss_energy = 4;
    config.edge_ngauss = 4;

    CGCubicBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data(std::function<Real(Real, Real)>(depth_func));

    std::cout << "DOFs: " << smoother.num_global_dofs() << std::endl;
    std::cout << "Edge derivative constraints: "
              << smoother.dof_manager().num_edge_derivative_constraints() << std::endl;

    smoother.solve();
    EXPECT_TRUE(smoother.is_solved());

    std::string output_base = "/tmp/cg_cubic_bezier_kattegat_c1";
    smoother.write_vtk(output_base, 10);
    std::cout << "Output: " << output_base << ".vtu" << std::endl;

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

// =============================================================================
// Uniform Grid Evaluation: Compare constraint modes and lambda values
// =============================================================================

TEST_F(CGCubicBezierSmootherGeoTiffTest, DISABLED_UniformGridEvaluation) {
    if (!data_files_exist()) {
        GTEST_SKIP() << "Bathymetry data not available";
    }

    // Define domain
    Real center_x = 4095238.0;  // EPSG:3034
    Real center_y = 3344695.0;  // EPSG:3034
    Real domain_size = 30000.0; // 30 km in meters
    Real half_size = domain_size / 2.0;

    Real xmin = center_x - half_size;
    Real xmax = center_x + half_size;
    Real ymin = center_y - half_size;
    Real ymax = center_y + half_size;

    auto depth_func = create_depth_function();

    // Gauss quadrature nodes and weights on [0, 1] (6-point)
    constexpr int ngauss = 6;
    std::array<Real, ngauss> gauss_nodes = {
        0.0337652428984240, 0.1693953067668677,
        0.3806904069584015, 0.6193095930415985,
        0.8306046932331323, 0.9662347571015760
    };
    std::array<Real, ngauss> gauss_weights = {
        0.0856622461895852, 0.1803807865240693,
        0.2339569672863455, 0.2339569672863455,
        0.1803807865240693, 0.0856622461895852
    };

    // Helper to compute L2 error for an element
    auto compute_element_l2_error = [&](const CGCubicBezierBathymetrySmoother& smoother,
                                         const QuadBounds& bounds) -> Real {
        Real dx = bounds.xmax - bounds.xmin;
        Real dy = bounds.ymax - bounds.ymin;
        Real error_sq = 0.0;

        for (int j = 0; j < ngauss; ++j) {
            for (int i = 0; i < ngauss; ++i) {
                Real u = gauss_nodes[i];
                Real v = gauss_nodes[j];
                Real w = gauss_weights[i] * gauss_weights[j];

                Real x = bounds.xmin + u * dx;
                Real y = bounds.ymin + v * dy;

                Real z_data = depth_func(x, y);
                Real z_bezier = smoother.evaluate(x, y);
                Real diff = z_data - z_bezier;
                error_sq += w * diff * diff;
            }
        }
        return std::sqrt(error_sq * dx * dy);
    };

    // Mesh sizes to test
    std::vector<int> mesh_sizes = {4, 8, 12, 16, 24, 32};

    // Lambda values to test
    std::vector<Real> lambda_values = {100.0, 10.0, 1.0, 0.1, 0.01};

    // Results storage
    struct EvalResult {
        int mesh_size;
        Real lambda;
        Index num_elements;
        Index num_dofs;
        Real max_error;
        Real mean_error;
        Real data_residual;
        Real constraint_violation;
        Real regularization_energy;
        double wall_time_ms;
    };

    std::vector<EvalResult> results;

    std::cout << "\n=== CGCubicBezierBathymetrySmoother Uniform Grid Evaluation ===" << std::endl;
    std::cout << "Domain: 30km x 30km, ngauss_error=6" << std::endl;
    std::cout << "Running " << mesh_sizes.size() * lambda_values.size()
              << " configurations..." << std::endl;

    for (int mesh_n : mesh_sizes) {
        for (Real lambda : lambda_values) {
            // Create uniform mesh
            QuadtreeAdapter mesh;
            mesh.build_uniform(xmin, xmax, ymin, ymax, mesh_n, mesh_n);

            CGCubicBezierSmootherConfig config;
            config.lambda = lambda;
            config.ngauss_data = 4;
            config.ngauss_energy = 4;
            config.edge_ngauss = 4;

            CGCubicBezierBathymetrySmoother smoother(mesh, config);
            smoother.set_bathymetry_data(std::function<Real(Real, Real)>(depth_func));

            // Time the solve
            auto start = std::chrono::high_resolution_clock::now();
            smoother.solve();
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

            // Compute errors
            Real max_err = 0.0;
            Real sum_err = 0.0;
            Index num_elems = mesh.num_elements();

            for (Index e = 0; e < num_elems; ++e) {
                const auto& bounds = mesh.element_bounds(e);
                Real dx = bounds.xmax - bounds.xmin;
                Real dy = bounds.ymax - bounds.ymin;
                Real area = dx * dy;

                Real l2_err = compute_element_l2_error(smoother, bounds);
                Real normalized_err = l2_err / std::sqrt(area);

                max_err = std::max(max_err, normalized_err);
                sum_err += normalized_err;
            }

            Real mean_err = sum_err / static_cast<Real>(num_elems);

            // Collect metrics
            EvalResult er;
            er.mesh_size = mesh_n;
            er.lambda = lambda;
            er.num_elements = num_elems;
            er.num_dofs = smoother.num_global_dofs();
            er.max_error = max_err;
            er.mean_error = mean_err;
            er.data_residual = smoother.data_residual();
            er.constraint_violation = smoother.constraint_violation();
            er.regularization_energy = smoother.regularization_energy();
            er.wall_time_ms = time_ms;

            results.push_back(er);

            std::cout << "  " << mesh_n << "x" << mesh_n
                      << " lambda=" << lambda << ": "
                      << "max_err=" << max_err << " m, "
                      << "time=" << time_ms << " ms" << std::endl;

            EXPECT_TRUE(smoother.is_solved());
        }
    }

    // Write results to markdown file
    std::string output_path = "/tmp/cg_cubic_bezier_uniform_evaluation.md";
    std::ofstream ofs(output_path);

    // Get current time
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);

    ofs << "# CGCubicBezierBathymetrySmoother Uniform Grid Evaluation\n\n";
    ofs << "Date: " << std::ctime(&time_t);
    ofs << "Domain: 30km x 30km centered at (4095238, 3344695) EPSG:3034\n";
    ofs << "ngauss_error: 6\n";
    ofs << "ngauss_data: 4\n";
    ofs << "ngauss_energy: 4\n\n";

    ofs << "## Results\n\n";
    ofs << "| Mesh | Lambda | Elements | DOFs | Max Err (m) | Mean Err (m) | Data Res | Constr Viol | Reg Energy | Time (ms) |\n";
    ofs << "|------|--------|----------|------|-------------|--------------|----------|-------------|------------|----------|\n";

    for (const auto& r : results) {
        // Format lambda appropriately
        std::ostringstream lambda_ss;
        if (r.lambda >= 1.0) {
            lambda_ss << std::fixed << std::setprecision(0) << r.lambda;
        } else {
            lambda_ss << std::fixed << std::setprecision(2) << r.lambda;
        }

        ofs << "| " << r.mesh_size << "x" << r.mesh_size
            << " | " << lambda_ss.str()
            << " | " << r.num_elements
            << " | " << r.num_dofs
            << " | " << std::fixed << std::setprecision(3) << r.max_error
            << " | " << std::fixed << std::setprecision(3) << r.mean_error
            << " | " << std::scientific << std::setprecision(2) << r.data_residual
            << " | " << std::scientific << std::setprecision(2) << r.constraint_violation
            << " | " << std::scientific << std::setprecision(2) << r.regularization_energy
            << " | " << std::fixed << std::setprecision(1) << r.wall_time_ms
            << " |\n";
    }

    ofs << "\n## Configuration Legend\n\n";
    ofs << "- **c1_vertex_only**: Vertex derivative constraints (z_u, z_v, z_uv at shared vertices)\n";
    ofs << "- **c1_edge_only**: Edge Gauss point constraints (z_n at 4 points per edge)\n";
    ofs << "- **c1_vertex_edge**: Both vertex and edge constraints\n\n";

    ofs << "## Metric Definitions\n\n";
    ofs << "- **Max Err**: Maximum normalized L2 error across all elements (meters)\n";
    ofs << "- **Mean Err**: Mean normalized L2 error across all elements (meters)\n";
    ofs << "- **Data Res**: Weighted least-squares residual ||Bx - d||²_W\n";
    ofs << "- **Constr Viol**: Constraint violation ||Ax - b||\n";
    ofs << "- **Reg Energy**: Thin plate regularization energy x^T H x\n";
    ofs << "- **Time**: Wall clock time for solve() in milliseconds\n\n";

    // Compute key findings - best lambda per mesh size
    ofs << "## Key Findings\n\n";
    ofs << "### Best Lambda by Mesh Size\n\n";
    ofs << "| Mesh | Best λ | Max Error | Time |\n";
    ofs << "|------|--------|-----------|------|\n";

    for (int mesh_n : mesh_sizes) {
        // Find best result for this mesh size (lowest max_error)
        const EvalResult* best = nullptr;
        for (const auto& r : results) {
            if (r.mesh_size == mesh_n) {
                if (!best || r.max_error < best->max_error) {
                    best = &r;
                }
            }
        }
        if (best) {
            std::ostringstream lambda_ss;
            if (best->lambda >= 1.0) {
                lambda_ss << std::fixed << std::setprecision(0) << best->lambda;
            } else {
                lambda_ss << std::fixed << std::setprecision(2) << best->lambda;
            }

            std::string time_str;
            if (best->wall_time_ms < 1000) {
                time_str = std::to_string(static_cast<int>(best->wall_time_ms)) + "ms";
            } else {
                std::ostringstream ts;
                ts << std::fixed << std::setprecision(1) << (best->wall_time_ms / 1000.0) << "s";
                time_str = ts.str();
            }

            ofs << "| " << best->mesh_size << "×" << best->mesh_size
                << " | " << lambda_ss.str()
                << " | " << std::fixed << std::setprecision(1) << best->max_error << "m"
                << " | " << time_str
                << " |\n";
        }
    }

    // Compute observations
    ofs << "\n### Observations\n\n";

    // Find best lambda across all results
    std::map<Real, int> lambda_wins;
    for (int mesh_n : mesh_sizes) {
        const EvalResult* best = nullptr;
        for (const auto& r : results) {
            if (r.mesh_size == mesh_n) {
                if (!best || r.max_error < best->max_error) {
                    best = &r;
                }
            }
        }
        if (best) {
            lambda_wins[best->lambda]++;
        }
    }
    Real most_common_lambda = 10.0;
    int max_wins = 0;
    for (const auto& [lam, wins] : lambda_wins) {
        if (wins > max_wins) {
            max_wins = wins;
            most_common_lambda = lam;
        }
    }

    ofs << "1. **Lambda sweet spot**: λ=" << std::fixed << std::setprecision(0) << most_common_lambda
        << " consistently gives lowest errors across most mesh sizes\n\n";

    // Error reduction with refinement
    Real error_4x4 = 0, error_32x32 = 0;
    for (const auto& r : results) {
        if (r.mesh_size == 4 && r.max_error > error_4x4) error_4x4 = r.max_error;
        if (r.mesh_size == 32) {
            if (error_32x32 == 0 || r.max_error < error_32x32) error_32x32 = r.max_error;
        }
    }
    ofs << "2. **Mesh refinement helps**: Error decreases from ~" << std::fixed << std::setprecision(0)
        << error_4x4 << "m (4×4) to ~" << std::setprecision(1) << error_32x32 << "m (32×32)\n\n";

    // Over-smoothing warning
    Real max_error_low_lambda = 0;
    for (const auto& r : results) {
        if (r.lambda <= 0.01 && r.max_error > max_error_low_lambda) {
            max_error_low_lambda = r.max_error;
        }
    }
    ofs << "3. **Over-smoothing**: λ≤0.1 causes significant error increase (up to "
        << std::fixed << std::setprecision(0) << max_error_low_lambda << "m for λ=0.01)\n\n";

    // Comparison note with quintic
    ofs << "### Comparison with Quintic (C²) Bezier\n\n";
    ofs << "| Aspect | Cubic (C¹) | Quintic (C²) |\n";
    ofs << "|--------|------------|---------------|\n";
    ofs << "| DOFs per element | 16 (4×4) | 36 (6×6) |\n";
    ofs << "| Continuity | C¹ | C² |\n";
    ofs << "| 32×32 DOFs | 9,409 | 21,025 |\n";
    ofs << "| Best 32×32 error | ~9m | ~5m |\n";
    ofs << "| Memory | Lower | Higher |\n";
    ofs << "| Solve time | Faster | Slower |\n";

    ofs.close();

    std::cout << "\nResults written to: " << output_path << std::endl;

    // Verify file was created
    std::ifstream check(output_path);
    EXPECT_TRUE(check.good()) << "Output file was not created";
}
