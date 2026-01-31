#include <gtest/gtest.h>
#include "bathymetry/cg_bezier_bathymetry_smoother.hpp"
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
    // NOTE: Fine elements first (indices 0-3), coarse last (index 4)
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

    // Print values at key points
    std::cout << "\n=== OnePlusFourLinearBathymetry (fine idx 0-3, coarse idx 4) ===\n";
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

    // Print values at critical points (same as 1+4 test)
    std::cout << "\n=== VTKOutputUniformMesh Values ===\n";
    std::cout << "z(50, 0)  = " << smoother.evaluate(50.0, 0.0)
              << " (expected " << bathy(50.0, 0.0) << ")\n";
    std::cout << "z(50, 50) = " << smoother.evaluate(50.0, 50.0)
              << " (expected " << bathy(50.0, 50.0) << ")\n";

    std::string filename = "/tmp/cg_bezier_uniform";
    smoother.write_vtk(filename, 10);

    // write_vtk appends .vtu extension
    std::string output_file = filename + ".vtu";
    EXPECT_TRUE(std::filesystem::exists(output_file));

    std::ifstream file(output_file);
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    EXPECT_GT(content.size(), 1000);
}

TEST_F(CGBezierSmootherTest, VTKOutputOnePlusFourMesh) {
    // 1+4 mesh (fine elements first, coarse last)
    QuadtreeAdapter mesh;
    Real h = 25.0;

    mesh.add_element({0.0, h, 0.0, h}, {2, 2});       // idx 0: fine
    mesh.add_element({h, 2*h, 0.0, h}, {2, 2});       // idx 1: fine [25,50]x[0,25]
    mesh.add_element({0.0, h, h, 2*h}, {2, 2});       // idx 2: fine
    mesh.add_element({h, 2*h, h, 2*h}, {2, 2});       // idx 3: fine [25,50]x[25,50]
    mesh.add_element({2*h, 4*h, 0.0, 2*h}, {1, 1});   // idx 4: coarse [50,100]x[0,50]

    // Check neighbor info at non-conforming interface
    std::cout << "\n=== Neighbor Info for Non-Conforming Interface ===\n";
    auto info1 = mesh.get_neighbor(1, 1);  // E1's right edge
    std::cout << "E1 right edge: type=" << (info1.type == EdgeNeighborInfo::Type::FineToCoarse ? "FineToCoarse" : "other")
              << ", neighbor=" << info1.neighbor_elements[0]
              << ", subedge_index=" << info1.subedge_index << "\n";
    auto info3 = mesh.get_neighbor(3, 1);  // E3's right edge
    std::cout << "E3 right edge: type=" << (info3.type == EdgeNeighborInfo::Type::FineToCoarse ? "FineToCoarse" : "other")
              << ", neighbor=" << info3.neighbor_elements[0]
              << ", subedge_index=" << info3.subedge_index << "\n";
    auto info4 = mesh.get_neighbor(4, 0);  // E4's left edge
    std::cout << "E4 left edge: type=" << (info4.type == EdgeNeighborInfo::Type::CoarseToFine ? "CoarseToFine" : "other")
              << ", neighbors=[" << info4.neighbor_elements[0] << "," << info4.neighbor_elements[1] << "]\n";

    auto bathy = [](Real x, Real y) {
        return 100.0 + 20.0 * std::sin(0.03 * x) * std::cos(0.03 * y);
    };

    CGBezierSmootherConfig config;
    config.lambda = 10.0;
    config.enable_c2_constraints = true;
    config.enable_edge_constraints = true;

    CGBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data(bathy);
    smoother.solve();

    // Print DOF sharing at interface
    const auto& dof_mgr = smoother.dof_manager();
    std::cout << "\n=== DOF Sharing at Interface ===\n";
    std::cout << "Global DOFs: " << dof_mgr.num_global_dofs() << "\n";
    std::cout << "Hanging node constraints: " << dof_mgr.num_constraints() << "\n";

    auto e1_dofs = dof_mgr.element_dofs(1);
    auto e3_dofs = dof_mgr.element_dofs(3);
    auto e4_dofs = dof_mgr.element_dofs(4);

    // E1's right edge (i=5): DOFs 30,31,32,33,34,35
    std::cout << "E1 right edge DOFs (k=0..5): ";
    for (int j = 0; j < 6; ++j) std::cout << e1_dofs[5 + 6*j] << " ";
    std::cout << "\n";

    // E3's right edge (i=5): DOFs 30,31,32,33,34,35
    std::cout << "E3 right edge DOFs (k=0..5): ";
    for (int j = 0; j < 6; ++j) std::cout << e3_dofs[5 + 6*j] << " ";
    std::cout << "\n";

    // E4's left edge (i=0): DOFs 0,1,2,3,4,5
    std::cout << "E4 left edge DOFs (m=0..5):  ";
    for (int j = 0; j < 6; ++j) std::cout << e4_dofs[0 + 6*j] << " ";
    std::cout << "\n";

    // Print which DOFs should be shared vs constrained
    std::cout << "\n=== Expected Sharing Pattern ===\n";
    std::cout << "E1 (subedge=0, lower half): k=0,2,4 shared, k=1,3 constrained, k=5 T-junction\n";
    std::cout << "E3 (subedge=1, upper half): k=0 T-junction, k=1,3,5 shared, k=2,4 constrained\n";

    // Verify sharing
    std::cout << "\n=== Sharing Verification ===\n";
    std::cout << "E1[k=0]=" << e1_dofs[5+6*0] << " == E4[m=0]=" << e4_dofs[0+6*0] << "? " << (e1_dofs[5+6*0] == e4_dofs[0+6*0] ? "YES" : "NO") << "\n";
    std::cout << "E1[k=2]=" << e1_dofs[5+6*2] << " == E4[m=1]=" << e4_dofs[0+6*1] << "? " << (e1_dofs[5+6*2] == e4_dofs[0+6*1] ? "YES" : "NO") << "\n";
    std::cout << "E1[k=4]=" << e1_dofs[5+6*4] << " == E4[m=2]=" << e4_dofs[0+6*2] << "? " << (e1_dofs[5+6*4] == e4_dofs[0+6*2] ? "YES" : "NO") << "\n";
    std::cout << "E3[k=1]=" << e3_dofs[5+6*1] << " == E4[m=3]=" << e4_dofs[0+6*3] << "? " << (e3_dofs[5+6*1] == e4_dofs[0+6*3] ? "YES" : "NO") << "\n";
    std::cout << "E3[k=3]=" << e3_dofs[5+6*3] << " == E4[m=4]=" << e4_dofs[0+6*4] << "? " << (e3_dofs[5+6*3] == e4_dofs[0+6*4] ? "YES" : "NO") << "\n";
    std::cout << "E3[k=5]=" << e3_dofs[5+6*5] << " == E4[m=5]=" << e4_dofs[0+6*5] << "? " << (e3_dofs[5+6*5] == e4_dofs[0+6*5] ? "YES" : "NO") << "\n";
    std::cout << "T-junction: E1[k=5]=" << e1_dofs[5+6*5] << " == E3[k=0]=" << e3_dofs[5+6*0] << "? " << (e1_dofs[5+6*5] == e3_dofs[5+6*0] ? "YES" : "NO") << "\n";

    // Check which DOFs are constrained
    std::cout << "\n=== Constraint Status ===\n";
    std::cout << "E1[k=1]=" << e1_dofs[5+6*1] << " constrained? " << (dof_mgr.is_constrained(e1_dofs[5+6*1]) ? "YES" : "NO") << "\n";
    std::cout << "E1[k=3]=" << e1_dofs[5+6*3] << " constrained? " << (dof_mgr.is_constrained(e1_dofs[5+6*3]) ? "YES" : "NO") << "\n";
    std::cout << "E3[k=2]=" << e3_dofs[5+6*2] << " constrained? " << (dof_mgr.is_constrained(e3_dofs[5+6*2]) ? "YES" : "NO") << "\n";
    std::cout << "E3[k=4]=" << e3_dofs[5+6*4] << " constrained? " << (dof_mgr.is_constrained(e3_dofs[5+6*4]) ? "YES" : "NO") << "\n";
    std::cout << "T-junction " << e1_dofs[5+6*5] << " constrained? " << (dof_mgr.is_constrained(e1_dofs[5+6*5]) ? "YES" : "NO") << "\n";

    // Print values at critical points
    std::cout << "\n=== VTKOutputOnePlusFourMesh Values ===\n";
    std::cout << "z(50, 0)  = " << smoother.evaluate(50.0, 0.0)
              << " (expected " << bathy(50.0, 0.0) << ")\n";
    std::cout << "z(50, 50) = " << smoother.evaluate(50.0, 50.0)
              << " (expected " << bathy(50.0, 50.0) << ")\n";

    std::string filename = "/tmp/cg_bezier_1plus4";
    smoother.write_vtk(filename, 10);

    // write_vtk appends .vtu extension
    std::string output_file = filename + ".vtu";
    EXPECT_TRUE(std::filesystem::exists(output_file));

    // Also write control points
    std::string cp_filename = "/tmp/cg_bezier_1plus4_control_points.vtk";
    smoother.write_control_points_vtk(cp_filename);

    EXPECT_TRUE(std::filesystem::exists(cp_filename));
}

// Diagnostic test to investigate why non-conforming edge appears as straight line
TEST_F(CGBezierSmootherTest, DiagnoseNonConformingEdge) {
    // Same mesh setup as VTKOutputOnePlusFourMesh
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
    config.enable_c2_constraints = true;
    config.enable_edge_constraints = true;

    CGBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data(bathy);
    smoother.solve();

    // Print control points along the non-conforming interface at x=50
    std::cout << "\n=== Non-conforming Interface Analysis at x=50 ===\n";

    // Element indices: E1=[25,50]×[0,25], E3=[25,50]×[25,50], E4=[50,100]×[0,50]
    // E1 is element 1, E3 is element 3, E4 is element 4

    // E4's left edge control points (at x=50, y = 0, 10, 20, 30, 40, 50)
    std::cout << "\nCoarse element E4 left edge (x=50):\n";
    VecX coeffs4 = smoother.element_coefficients(4);
    for (int j = 0; j < 6; ++j) {
        int dof = 0 + 6*j;  // Left edge (u=0), i=0
        Real y = j * 10.0;  // y = 0, 10, 20, 30, 40, 50
        Real z_expected = bathy(50.0, y);
        std::cout << "  y=" << y << ": z_ctrl=" << coeffs4(dof)
                  << ", z_bathy=" << z_expected << "\n";
    }

    // E1's right edge control points (at x=50, y = 0, 5, 10, 15, 20, 25)
    // DOF indexing: dof = i + 6*j, right edge has i=5
    std::cout << "\nFine element E1 right edge (x=50):\n";
    VecX coeffs1 = smoother.element_coefficients(1);
    for (int j = 0; j < 6; ++j) {
        int dof = 5 + 6*j;  // Right edge (u=1), i=5
        Real y = j * 5.0;  // y = 0, 5, 10, 15, 20, 25
        Real z_expected = bathy(50.0, y);
        std::cout << "  y=" << y << ": z_ctrl=" << coeffs1(dof)
                  << ", z_bathy=" << z_expected << "\n";
    }

    // E3's right edge control points (at x=50, y = 25, 30, 35, 40, 45, 50)
    std::cout << "\nFine element E3 right edge (x=50):\n";
    VecX coeffs3 = smoother.element_coefficients(3);
    for (int j = 0; j < 6; ++j) {
        int dof = 5 + 6*j;  // Right edge (u=1), i=5
        Real y = 25.0 + j * 5.0;  // y = 25, 30, 35, 40, 45, 50
        Real z_expected = bathy(50.0, y);
        std::cout << "  y=" << y << ": z_ctrl=" << coeffs3(dof)
                  << ", z_bathy=" << z_expected << "\n";
    }

    // Evaluate the actual surfaces at the interface
    std::cout << "\n=== Surface Values at x=50 ===\n";
    for (Real y = 0.0; y <= 50.0; y += 5.0) {
        Real z_smooth = smoother.evaluate(50.0, y);
        Real z_bathy = bathy(50.0, y);
        std::cout << "y=" << y << ": z_smooth=" << z_smooth
                  << ", z_bathy=" << z_bathy
                  << ", diff=" << (z_smooth - z_bathy) << "\n";
    }

    // Check if control points are collinear
    // A perfectly linear edge would have z = a + b*y
    // Fit a line and measure deviation
    std::cout << "\n=== Linearity Check for Coarse Edge ===\n";
    Real z0 = coeffs4(0);      // y=0
    Real z5 = coeffs4(5*6);    // y=50
    Real slope = (z5 - z0) / 50.0;
    Real max_deviation = 0.0;
    for (int j = 0; j < 6; ++j) {
        int dof = 0 + 6*j;
        Real y = j * 10.0;
        Real z_linear = z0 + slope * y;
        Real deviation = std::abs(coeffs4(dof) - z_linear);
        max_deviation = std::max(max_deviation, deviation);
        std::cout << "  y=" << y << ": z_ctrl=" << coeffs4(dof)
                  << ", z_linear=" << z_linear
                  << ", deviation=" << deviation << "\n";
    }
    std::cout << "Maximum deviation from linear: " << max_deviation << "\n";

    if (max_deviation < 0.01) {
        std::cout << "** Control points ARE nearly collinear - edge is mathematically straight **\n";
    } else {
        std::cout << "** Control points are curved - issue is likely VTK rendering **\n";
    }

    // Verify de Casteljau constraint satisfaction
    // For E1 (lower half), fine edge DOFs should be S_left * coarse edge DOFs
    // For E3 (upper half), fine edge DOFs should be S_right * coarse edge DOFs
    std::cout << "\n=== De Casteljau Constraint Verification ===\n";

    // Get coarse edge DOFs (E4 left edge)
    VecX coarse_edge(6);
    for (int j = 0; j < 6; ++j) {
        coarse_edge(j) = coeffs4(0 + 6*j);  // DOFs at i=0
    }
    std::cout << "Coarse edge DOFs: " << coarse_edge.transpose() << "\n";

    // E1's fine edge DOFs should satisfy: fine[k] = S_left[k,:] * coarse
    // S_left for [0, 0.5] interval
    std::cout << "\nE1 fine edge (should match left-half subdivision):\n";
    VecX fine_e1(6);
    for (int j = 0; j < 6; ++j) {
        fine_e1(j) = coeffs1(5 + 6*j);  // DOFs at i=5
    }
    std::cout << "Fine E1 DOFs: " << fine_e1.transpose() << "\n";

    // Manual computation of what de Casteljau should give
    // Left half: Q_k = sum_{j=0}^k C(k,j) / 2^k * P_j
    std::cout << "Expected (de Casteljau left half):\n";
    for (int k = 0; k < 6; ++k) {
        Real expected = 0.0;
        Real denom = std::pow(2.0, k);
        for (int jj = 0; jj <= k; ++jj) {
            // Binomial coefficient C(k, jj)
            Real binom = 1.0;
            for (int ii = 0; ii < jj; ++ii) {
                binom *= (k - ii);
                binom /= (ii + 1);
            }
            expected += binom * coarse_edge(jj) / denom;
        }
        std::cout << "  k=" << k << ": expected=" << expected
                  << ", actual=" << fine_e1(k)
                  << ", diff=" << (fine_e1(k) - expected) << "\n";
    }
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
    std::string output_base = "/tmp/cg_bezier_kattegat_test";
    smoother.write_vtk(output_base, 10);

    // write_vtk appends .vtu extension
    std::string output_file = output_base + ".vtu";
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

// Compare uniform mesh vs 1+4 mesh with SAME bathymetry function
// This test verifies that the 1+4 non-conforming mesh produces correct values
TEST_F(CGBezierSmootherTest, CompareUniformVsOnePlusFour) {
    // Domain: [0, 100] x [0, 50] to match the 1+4 layout
    // Bathymetry: z = 120 - 0.4*y (simple linear)
    // Expected values:
    //   (50, 0):  120 - 0 = 120
    //   (50, 50): 120 - 20 = 100
    auto bathy = [](Real x, Real y) { return 120.0 - 0.4 * y; };

    CGBezierSmootherConfig config;
    config.lambda = 100.0;
    config.ridge_epsilon = 0.0;
    config.enable_c2_constraints = false;
    config.enable_edge_constraints = false;

    // ===== UNIFORM 2x1 MESH =====
    // Two elements side by side: [0,50]x[0,50] and [50,100]x[0,50]
    QuadtreeAdapter uniform_mesh;
    uniform_mesh.build_uniform(0.0, 100.0, 0.0, 50.0, 2, 1);

    std::cout << "\n=== UNIFORM 2x1 MESH ===\n";
    std::cout << "Elements: " << uniform_mesh.num_elements() << "\n";
    for (Index e = 0; e < uniform_mesh.num_elements(); ++e) {
        const auto& b = uniform_mesh.element_bounds(e);
        std::cout << "  E" << e << ": [" << b.xmin << "," << b.xmax << "] x ["
                  << b.ymin << "," << b.ymax << "]\n";
    }

    CGBezierBathymetrySmoother uniform_smoother(uniform_mesh, config);
    uniform_smoother.set_bathymetry_data(bathy);
    uniform_smoother.solve();

    std::cout << "DOFs: " << uniform_smoother.num_global_dofs() << "\n";
    Real uniform_50_0 = uniform_smoother.evaluate(50.0, 0.0);
    Real uniform_50_50 = uniform_smoother.evaluate(50.0, 50.0);
    std::cout << "z(50, 0)  = " << uniform_50_0 << " (expected 120)\n";
    std::cout << "z(50, 50) = " << uniform_50_50 << " (expected 100)\n";

    // ===== 1+4 MESH (coarse index 0, fine indices 1-4) =====
    QuadtreeAdapter mesh_1p4;
    Real h = 25.0;
    // Coarse element FIRST (index 0)
    mesh_1p4.add_element({2*h, 4*h, 0.0, 2*h}, {1, 1});  // [50,100] x [0,50]
    // Fine elements (indices 1-4)
    mesh_1p4.add_element({0.0, h, 0.0, h}, {2, 2});       // [0,25] x [0,25]
    mesh_1p4.add_element({h, 2*h, 0.0, h}, {2, 2});       // [25,50] x [0,25]
    mesh_1p4.add_element({0.0, h, h, 2*h}, {2, 2});       // [0,25] x [25,50]
    mesh_1p4.add_element({h, 2*h, h, 2*h}, {2, 2});       // [25,50] x [25,50]

    std::cout << "\n=== 1+4 MESH (coarse idx 0) ===\n";
    std::cout << "Elements: " << mesh_1p4.num_elements() << "\n";
    for (Index e = 0; e < mesh_1p4.num_elements(); ++e) {
        const auto& b = mesh_1p4.element_bounds(e);
        std::cout << "  E" << e << ": [" << b.xmin << "," << b.xmax << "] x ["
                  << b.ymin << "," << b.ymax << "]\n";
    }

    CGBezierBathymetrySmoother smoother_1p4(mesh_1p4, config);
    smoother_1p4.set_bathymetry_data(bathy);
    smoother_1p4.solve();

    std::cout << "DOFs: " << smoother_1p4.num_global_dofs() << "\n";
    Real z_1p4_50_0 = smoother_1p4.evaluate(50.0, 0.0);
    Real z_1p4_50_50 = smoother_1p4.evaluate(50.0, 50.0);
    std::cout << "z(50, 0)  = " << z_1p4_50_0 << " (expected 120)\n";
    std::cout << "z(50, 50) = " << z_1p4_50_50 << " (expected 100)\n";

    // Print values along the interface x=50
    std::cout << "\n=== Values along interface x=50 ===\n";
    std::cout << std::setw(8) << "y" << std::setw(15) << "uniform" << std::setw(15) << "1+4"
              << std::setw(15) << "expected" << std::setw(15) << "diff\n";
    for (Real y = 0.0; y <= 50.0; y += 10.0) {
        Real u = uniform_smoother.evaluate(50.0, y);
        Real p = smoother_1p4.evaluate(50.0, y);
        Real e = bathy(50.0, y);
        std::cout << std::setw(8) << y << std::setw(15) << u << std::setw(15) << p
                  << std::setw(15) << e << std::setw(15) << std::abs(u - p) << "\n";
    }

    // The uniform mesh should produce correct values
    EXPECT_NEAR(uniform_50_0, 120.0, 1e-6) << "Uniform mesh wrong at (50,0)";
    EXPECT_NEAR(uniform_50_50, 100.0, 1e-6) << "Uniform mesh wrong at (50,50)";

    // The 1+4 mesh should match the uniform mesh
    EXPECT_NEAR(z_1p4_50_0, 120.0, 1e-6) << "1+4 mesh wrong at (50,0)";
    EXPECT_NEAR(z_1p4_50_50, 100.0, 1e-6) << "1+4 mesh wrong at (50,50)";
}

// Test that verifies the fix for non-conforming interfaces when coarse element
// has LOWER index than fine elements (simulates AMR scenario)
// ALL node sharing via index logic, NO constraints - pure DOF sharing
TEST_F(CGBezierSmootherTest, NonConformingWithCoarseLowerIndex) {
    // Create mesh with COARSE element FIRST (index 0), then fine elements (1-4)
    // This is the opposite of DiagnoseNonConformingEdge and simulates AMR refinement
    //
    // Layout:
    //   +------+------+--------+
    //   |  E3  |  E4  |        |
    //   +------+------+   E0   |
    //   |  E1  |  E2  | coarse |
    //   +------+------+--------+
    //   0      h     2h       4h
    //
    // E0: coarse element [2h, 4h] x [0, 2h], index 0
    // E1-E4: fine elements, indices 1-4
    // E2 and E4 share right edge with E0's left edge
    //
    // DOF sharing at non-conforming interface (E2, E4 right edge with E0 left edge):
    //   E2 (lower half, subedge_index=0): DOFs k=0,2,4 share with E0's DOFs m=0,1,2
    //   E4 (upper half, subedge_index=1): DOFs k=1,3,5 share with E0's DOFs m=3,4,5

    QuadtreeAdapter mesh;
    Real h = 25.0;

    // COARSE element on the right (will have index 0)
    mesh.add_element({2*h, 4*h, 0.0, 2*h}, {1, 1});  // Index 0 (coarse)

    // FINE elements on the left (will have indices 1-4)
    mesh.add_element({0.0, h, 0.0, h}, {2, 2});       // Index 1
    mesh.add_element({h, 2*h, 0.0, h}, {2, 2});       // Index 2 (shares right edge with coarse)
    mesh.add_element({0.0, h, h, 2*h}, {2, 2});       // Index 3
    mesh.add_element({h, 2*h, h, 2*h}, {2, 2});       // Index 4 (shares right edge with coarse)

    std::cout << "\n=== Mesh Structure ===\n";
    std::cout << "E0 (coarse, idx 0): [" << 2*h << "," << 4*h << "] x [0," << 2*h << "]\n";
    std::cout << "E1 (fine, idx 1):   [0," << h << "] x [0," << h << "]\n";
    std::cout << "E2 (fine, idx 2):   [" << h << "," << 2*h << "] x [0," << h << "]\n";
    std::cout << "E3 (fine, idx 3):   [0," << h << "] x [" << h << "," << 2*h << "]\n";
    std::cout << "E4 (fine, idx 4):   [" << h << "," << 2*h << "] x [" << h << "," << 2*h << "]\n";

    // Verify the mesh has the expected structure
    EXPECT_EQ(mesh.num_elements(), 5);

    // Check that coarse has lower index than fine elements at interface
    auto info2 = mesh.get_neighbor(2, 1);  // Element 2's right edge
    EXPECT_EQ(info2.type, EdgeNeighborInfo::Type::FineToCoarse);
    EXPECT_EQ(info2.neighbor_elements[0], 0);  // Coarse element has index 0
    EXPECT_LT(info2.neighbor_elements[0], 2);  // Coarse index < fine index

    std::cout << "\nE2 right edge neighbor info:\n";
    std::cout << "  type: FineToCoarse\n";
    std::cout << "  neighbor: E" << info2.neighbor_elements[0] << "\n";
    std::cout << "  subedge_index: " << info2.subedge_index << " (0=lower half)\n";

    auto info4 = mesh.get_neighbor(4, 1);  // Element 4's right edge
    EXPECT_EQ(info4.type, EdgeNeighborInfo::Type::FineToCoarse);
    EXPECT_EQ(info4.neighbor_elements[0], 0);
    EXPECT_LT(info4.neighbor_elements[0], 4);

    std::cout << "\nE4 right edge neighbor info:\n";
    std::cout << "  type: FineToCoarse\n";
    std::cout << "  neighbor: E" << info4.neighbor_elements[0] << "\n";
    std::cout << "  subedge_index: " << info4.subedge_index << " (1=upper half)\n";

    // Use linear bathymetry to test C⁰ continuity
    auto linear_bathy = [](Real x, Real y) { return 100.0 + 0.5 * x + 0.3 * y; };

    // DISABLE ALL CONSTRAINTS - test pure DOF sharing for C⁰ continuity
    CGBezierSmootherConfig config;
    config.lambda = 100.0;  // Strong data fitting
    config.ridge_epsilon = 0.0;
    config.enable_c2_constraints = false;  // No C² vertex constraints
    config.enable_edge_constraints = false;  // No edge derivative constraints

    CGBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data(linear_bathy);
    smoother.solve();

    EXPECT_TRUE(smoother.is_solved());

    // Print DOF manager info
    const auto& dof_manager = smoother.dof_manager();
    std::cout << "\n=== DOF Manager Info ===\n";
    std::cout << "Global DOFs: " << dof_manager.num_global_dofs() << "\n";
    std::cout << "Hanging node constraints: " << dof_manager.num_constraints() << "\n";

    // Check DOF sharing at interface
    std::cout << "\n=== DOF Sharing at Interface (E2/E4 right edge with E0 left edge) ===\n";

    auto e0_dofs = dof_manager.element_dofs(0);  // Coarse
    auto e2_dofs = dof_manager.element_dofs(2);  // Fine lower
    auto e4_dofs = dof_manager.element_dofs(4);  // Fine upper

    // E0's left edge DOFs: local indices 0,1,2,3,4,5 (i=0, j=0..5)
    // E2's right edge DOFs: local indices 5,11,17,23,29,35 (i=5, j=0..5)
    // E4's right edge DOFs: local indices 5,11,17,23,29,35 (i=5, j=0..5)

    std::cout << "E0 left edge global DOFs:  ";
    for (int j = 0; j < 6; ++j) {
        int local = 0 + 6*j;  // Left edge: i=0
        std::cout << e0_dofs[local] << " ";
    }
    std::cout << "\n";

    std::cout << "E2 right edge global DOFs: ";
    for (int j = 0; j < 6; ++j) {
        int local = 5 + 6*j;  // Right edge: i=5
        std::cout << e2_dofs[local] << " ";
    }
    std::cout << "\n";

    std::cout << "E4 right edge global DOFs: ";
    for (int j = 0; j < 6; ++j) {
        int local = 5 + 6*j;  // Right edge: i=5
        std::cout << e4_dofs[local] << " ";
    }
    std::cout << "\n";

    // Verify index-based sharing pattern:
    // E2 (subedge=0): k=0,2,4 share with E0's m=0,1,2
    std::cout << "\nVerifying E2 DOF sharing (subedge_index=0, lower half):\n";
    std::cout << "  E2[k=0] = " << e2_dofs[5+6*0] << ", E0[m=0] = " << e0_dofs[0+6*0];
    std::cout << (e2_dofs[5+6*0] == e0_dofs[0+6*0] ? " ✓ SHARED" : " ✗ NOT SHARED") << "\n";
    std::cout << "  E2[k=2] = " << e2_dofs[5+6*2] << ", E0[m=1] = " << e0_dofs[0+6*1];
    std::cout << (e2_dofs[5+6*2] == e0_dofs[0+6*1] ? " ✓ SHARED" : " ✗ NOT SHARED") << "\n";
    std::cout << "  E2[k=4] = " << e2_dofs[5+6*4] << ", E0[m=2] = " << e0_dofs[0+6*2];
    std::cout << (e2_dofs[5+6*4] == e0_dofs[0+6*2] ? " ✓ SHARED" : " ✗ NOT SHARED") << "\n";

    // E4 (subedge=1): k=1,3,5 share with E0's m=3,4,5
    std::cout << "\nVerifying E4 DOF sharing (subedge_index=1, upper half):\n";
    std::cout << "  E4[k=1] = " << e4_dofs[5+6*1] << ", E0[m=3] = " << e0_dofs[0+6*3];
    std::cout << (e4_dofs[5+6*1] == e0_dofs[0+6*3] ? " ✓ SHARED" : " ✗ NOT SHARED") << "\n";
    std::cout << "  E4[k=3] = " << e4_dofs[5+6*3] << ", E0[m=4] = " << e0_dofs[0+6*4];
    std::cout << (e4_dofs[5+6*3] == e0_dofs[0+6*4] ? " ✓ SHARED" : " ✗ NOT SHARED") << "\n";
    std::cout << "  E4[k=5] = " << e4_dofs[5+6*5] << ", E0[m=5] = " << e0_dofs[0+6*5];
    std::cout << (e4_dofs[5+6*5] == e0_dofs[0+6*5] ? " ✓ SHARED" : " ✗ NOT SHARED") << "\n";

    // T-junction: E2[k=5] and E4[k=0] should share at the midpoint
    std::cout << "\nT-junction sharing (E2 top-right = E4 bottom-right):\n";
    std::cout << "  E2[k=5] = " << e2_dofs[5+6*5] << ", E4[k=0] = " << e4_dofs[5+6*0];
    std::cout << (e2_dofs[5+6*5] == e4_dofs[5+6*0] ? " ✓ SHARED" : " ✗ NOT SHARED") << "\n";

    // Test C⁰ continuity at the non-conforming interface (x = 50)
    std::cout << "\n=== C⁰ Continuity at Interface (x=50) ===\n";

    Real interface_x = 2 * h;  // x = 50
    Real max_discontinuity = 0.0;
    Real eps = 1e-6;

    for (Real y = 0.0; y <= 2 * h; y += 5.0) {
        Real z_fine = smoother.evaluate(interface_x - eps, y);
        Real z_coarse = smoother.evaluate(interface_x + eps, y);
        Real diff = std::abs(z_fine - z_coarse);
        max_discontinuity = std::max(max_discontinuity, diff);

        std::cout << "  y=" << std::setw(4) << y << ": z_fine=" << std::setw(10) << z_fine
                  << ", z_coarse=" << std::setw(10) << z_coarse
                  << ", diff=" << diff << "\n";
    }

    std::cout << "Max discontinuity at interface: " << max_discontinuity << "\n";

    // C⁰ continuity should be near machine precision through DOF sharing alone
    EXPECT_LT(max_discontinuity, 1e-5) << "C⁰ continuity violated at non-conforming interface";

    // Also verify the solution approximately matches the linear input
    Real center_error = std::abs(smoother.evaluate(50.0, 25.0) - linear_bathy(50.0, 25.0));
    std::cout << "\nError at center (50,25): " << center_error << "\n";
    EXPECT_LT(center_error, 5.0);  // Reasonable fit for smoothed solution
}

// =============================================================================
// Uniform Grid Evaluation: Compare constraint modes and lambda values
// =============================================================================

#ifdef DRIFTER_HAS_GDAL
TEST_F(CGBezierSmootherTest, UniformGridEvaluation) {
  // Skip if GeoTIFF not available
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

  // Define domain
  Real center_x = 4095238.0;  // EPSG:3034
  Real center_y = 3344695.0;  // EPSG:3034
  Real domain_size = 30000.0; // 30 km in meters
  Real half_size = domain_size / 2.0;

  Real xmin = center_x - half_size;
  Real xmax = center_x + half_size;
  Real ymin = center_y - half_size;
  Real ymax = center_y + half_size;

  auto bathy_ptr = std::make_shared<BathymetryData>(std::move(bathy));
  BathymetrySurface surface(bathy_ptr);

  auto depth_func = [&surface](Real x, Real y) -> Real {
    return -surface.depth(x, y);  // depth returns positive down, we want elevation
  };

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
  auto compute_element_l2_error = [&](const CGBezierBathymetrySmoother& smoother,
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

  // Constraint configurations
  struct ConstraintConfig {
    std::string name;
    bool c2_constraints;
    bool edge_constraints;
  };

  std::vector<ConstraintConfig> constraint_configs = {
      {"c2_only", true, false},
      {"edge_only", false, true},
      {"c2_and_edge", true, true},
  };

  // Lambda values to test
  std::vector<Real> lambda_values = {100.0, 10.0, 1.0, 0.1, 0.01};

  // Results storage
  struct EvalResult {
    int mesh_size;
    std::string config_name;
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

  std::cout << "\n=== CGBezierBathymetrySmoother Uniform Grid Evaluation ===" << std::endl;
  std::cout << "Domain: 30km x 30km, ngauss_error=6" << std::endl;
  std::cout << "Running " << mesh_sizes.size() * constraint_configs.size() * lambda_values.size()
            << " configurations..." << std::endl;

  for (int mesh_n : mesh_sizes) {
    for (const auto& cc : constraint_configs) {
      for (Real lambda : lambda_values) {
        // Create uniform mesh
        QuadtreeAdapter mesh;
        mesh.build_uniform(xmin, xmax, ymin, ymax, mesh_n, mesh_n);

        CGBezierSmootherConfig config;
        config.lambda = lambda;
        config.ngauss_data = 6;
        config.ngauss_energy = 6;
        config.enable_c2_constraints = cc.c2_constraints;
        config.enable_edge_constraints = cc.edge_constraints;
        config.edge_ngauss = 4;

        CGBezierBathymetrySmoother smoother(mesh, config);
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
        er.config_name = cc.name;
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

        std::cout << "  " << mesh_n << "x" << mesh_n << " " << cc.name
                  << " lambda=" << lambda << ": "
                  << "max_err=" << max_err << " m, "
                  << "time=" << time_ms << " ms" << std::endl;

        EXPECT_TRUE(smoother.is_solved());
      }
    }
  }

  // Write results to markdown file
  std::string output_path = "/tmp/cg_bezier_uniform_evaluation.md";
  std::ofstream ofs(output_path);

  // Get current time
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);

  ofs << "# CGBezierBathymetrySmoother Uniform Grid Evaluation\n\n";
  ofs << "Date: " << std::ctime(&time_t);
  ofs << "Domain: 30km x 30km centered at (4095238, 3344695) EPSG:3034\n";
  ofs << "ngauss_error: 6\n";
  ofs << "ngauss_data: 6\n";
  ofs << "ngauss_energy: 6\n\n";

  ofs << "## Results\n\n";
  ofs << "| Mesh | Config | Lambda | Elements | DOFs | Max Err (m) | Mean Err (m) | Data Res | Constr Viol | Reg Energy | Time (ms) |\n";
  ofs << "|------|--------|--------|----------|------|-------------|--------------|----------|-------------|------------|----------|\n";

  for (const auto& r : results) {
    // Format lambda appropriately
    std::ostringstream lambda_ss;
    if (r.lambda >= 1.0) {
      lambda_ss << std::fixed << std::setprecision(0) << r.lambda;
    } else {
      lambda_ss << std::fixed << std::setprecision(2) << r.lambda;
    }

    ofs << "| " << r.mesh_size << "x" << r.mesh_size
        << " | " << r.config_name
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
  ofs << "- **c2_only**: Vertex derivative constraints (z_u, z_v, z_uu, z_uv, z_vv, z_uuv, z_uvv, z_uuvv)\n";
  ofs << "- **edge_only**: Edge Gauss point constraints (z_n, z_nn at 4 points per edge)\n";
  ofs << "- **c2_and_edge**: Both vertex and edge constraints\n\n";

  ofs << "## Metric Definitions\n\n";
  ofs << "- **Max Err**: Maximum normalized L2 error across all elements (meters)\n";
  ofs << "- **Mean Err**: Mean normalized L2 error across all elements (meters)\n";
  ofs << "- **Data Res**: Weighted least-squares residual ||Bx - d||²_W\n";
  ofs << "- **Constr Viol**: Constraint violation ||Ax - b||\n";
  ofs << "- **Reg Energy**: Thin plate regularization energy x^T H x\n";
  ofs << "- **Time**: Wall clock time for solve() in milliseconds\n";

  ofs.close();

  std::cout << "\nResults written to: " << output_path << std::endl;

  // Verify file was created
  std::ifstream check(output_path);
  EXPECT_TRUE(check.good()) << "Output file was not created";
}
#endif
