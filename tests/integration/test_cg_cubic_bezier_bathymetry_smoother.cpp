#include "bathymetry/cg_cubic_bezier_bathymetry_smoother.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "mesh/octree_adapter.hpp"
#include "mesh/seabed_surface.hpp"
#include "test_integration_fixtures.hpp"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <sstream>

#include "mesh/geotiff_reader.hpp"

using namespace drifter;
using namespace drifter::testing;

class CGCubicBezierSmootherTest : public ::testing::Test {
protected:
  static constexpr Real TOLERANCE = 1e-10;
  static constexpr Real LOOSE_TOLERANCE = 1e-6;

  std::unique_ptr<OctreeAdapter> create_octree(int nx, int ny, int nz) {
    auto octree = std::make_unique<OctreeAdapter>(0.0, 100.0, // x bounds
                                                  0.0, 100.0, // y bounds
                                                  -1.0, 0.0   // z bounds
    );
    octree->build_uniform(nx, ny, nz);
    return octree;
  }

  QuadtreeAdapter create_quadtree(int nx, int ny, Real xmin = 0.0,
                                  Real xmax = 100.0, Real ymin = 0.0,
                                  Real ymax = 100.0) {
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
  config.lambda = 100.0;      // Strong data fitting
  config.ridge_epsilon = 0.0; // No ridge for exact polynomial reproduction

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
  config.ridge_epsilon = 0.0; // No ridge for exact polynomial reproduction

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

  auto quadratic = [](Real x, Real y) {
    return 100.0 + 0.1 * x * x + 0.2 * y * y;
  };

  CGCubicBezierBathymetrySmoother smoother(mesh, config);
  smoother.set_bathymetry_data(quadratic);
  smoother.solve();

  // Use 0.5% relative tolerance for quadratics (smoothing affects non-linear
  // functions) Cubic has lower degree than quintic, so may have slightly larger
  // error
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
  config.ridge_epsilon = 0.0; // No ridge for exact polynomial reproduction

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
  config.ridge_epsilon = 0.0; // No ridge for exact polynomial reproduction

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
  auto mesh = create_quadtree(2, 2); // 4 elements

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
  EXPECT_GT(num_dofs, 16); // At least one element's worth

  std::cout << "2×2 mesh CG cubic DOFs: " << num_dofs << " (vs 64 for DG)"
            << std::endl;
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
  mesh.add_element({0.0, h, 0.0, h}, {2, 2});     // elem 0
  mesh.add_element({h, 2 * h, 0.0, h}, {2, 2});   // elem 1
  mesh.add_element({0.0, h, h, 2 * h}, {2, 2});   // elem 2
  mesh.add_element({h, 2 * h, h, 2 * h}, {2, 2}); // elem 3

  // Coarse element (level 1)
  mesh.add_element({2 * h, 4 * h, 0.0, 2 * h}, {1, 1}); // elem 4

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
  mesh.add_element({h, 2 * h, 0.0, h}, {2, 2});
  mesh.add_element({0.0, h, h, 2 * h}, {2, 2});
  mesh.add_element({h, 2 * h, h, 2 * h}, {2, 2});
  mesh.add_element({2 * h, 4 * h, 0.0, 2 * h}, {1, 1});

  CGCubicBezierSmootherConfig config;
  config.lambda = 100.0;
  config.ridge_epsilon = 0.0; // No ridge for exact polynomial reproduction

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
  mesh.add_element({h, 2 * h, 0.0, h}, {2, 2});
  mesh.add_element({0.0, h, h, 2 * h}, {2, 2});
  mesh.add_element({h, 2 * h, h, 2 * h}, {2, 2});
  mesh.add_element({2 * h, 4 * h, 0.0, 2 * h}, {1, 1});

  auto linear = [](Real x, Real y) { return 50.0 + x + 2.0 * y; };

  CGCubicBezierSmootherConfig config;
  config.lambda = 100.0;
  config.ridge_epsilon = 0.0; // No ridge for exact polynomial reproduction

  CGCubicBezierBathymetrySmoother smoother(mesh, config);
  smoother.set_bathymetry_data(linear);
  smoother.solve();

  // Print values at key points
  std::cout << "\n=== OnePlusFourLinearBathymetry (cubic) ===\n";
  std::cout << "z(50, 0)  = " << smoother.evaluate(50.0, 0.0) << " (expected "
            << linear(50.0, 0.0) << ")\n";
  std::cout << "z(50, 50) = " << smoother.evaluate(50.0, 50.0) << " (expected "
            << linear(50.0, 50.0) << ")\n";

  // Should reproduce linear across both fine and coarse elements
  std::vector<Vec2> test_points = {
      {10.0, 10.0}, {25.0, 25.0}, {40.0, 40.0}, // Fine region
      {60.0, 25.0}, {75.0, 40.0}, {90.0, 10.0}, // Coarse region
      {50.0, 25.0},                             // At interface
  };

  for (const auto &pt : test_points) {
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
  mesh.add_element({h, 2 * h, 0.0, h}, {2, 2});
  mesh.add_element({0.0, h, h, 2 * h}, {2, 2});
  mesh.add_element({h, 2 * h, h, 2 * h}, {2, 2});
  mesh.add_element({2 * h, 4 * h, 0.0, 2 * h}, {1, 1});

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
  mesh.add_element({h, 2 * h, 0.0, h}, {2, 2});
  mesh.add_element({0.0, h, h, 2 * h}, {2, 2});
  mesh.add_element({h, 2 * h, h, 2 * h}, {2, 2});
  mesh.add_element({2 * h, 4 * h, 0.0, 2 * h}, {1, 1});

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
      {10.0, 10.0}, {25.0, 25.0}, {40.0, 40.0}, // Fine region
      {60.0, 25.0}, {75.0, 40.0}, {90.0, 10.0}, // Coarse region
      {50.0, 25.0}, {50.0, 12.5}, {50.0, 37.5}, // At interface
  };

  for (const auto &pt : test_points) {
    Real val_condensed = smoother_condensed.evaluate(pt(0), pt(1));
    Real val_full_kkt = smoother_full_kkt.evaluate(pt(0), pt(1));
    Real diff = std::abs(val_condensed - val_full_kkt);
    max_diff = std::max(max_diff, diff);
  }

  std::cout << "Max solution difference (condensed vs full KKT): " << max_diff
            << std::endl;
  std::cout << "Constraint violation (condensed): "
            << smoother_condensed.constraint_violation() << std::endl;
  std::cout << "Constraint violation (full KKT):  "
            << smoother_full_kkt.constraint_violation() << std::endl;

  // Solutions should match to high precision (within numerical tolerance)
  // Note: With non-conforming C¹ constraints, condensed and full KKT have
  // slightly different numerical behavior, so we allow 1e-7 tolerance
  EXPECT_LT(max_diff, 1e-7);
}

TEST_F(CGCubicBezierSmootherTest, IterativeMatchesDirect) {
  // Verify iterative Schur complement CG solver matches direct SparseLU
  QuadtreeAdapter mesh;
  Real h = 25.0;

  // 1+4 non-conforming mesh (has hanging node + edge constraints)
  mesh.add_element({0.0, h, 0.0, h}, {2, 2});
  mesh.add_element({h, 2 * h, 0.0, h}, {2, 2});
  mesh.add_element({0.0, h, h, 2 * h}, {2, 2});
  mesh.add_element({h, 2 * h, h, 2 * h}, {2, 2});
  mesh.add_element({2 * h, 4 * h, 0.0, 2 * h}, {1, 1});

  auto smooth_bathy = [](Real x, Real y) {
    return 100.0 + 20.0 * std::sin(0.03 * x) * std::cos(0.03 * y);
  };

  CGCubicBezierSmootherConfig config_direct;
  config_direct.lambda = 10.0;
  config_direct.use_iterative_solver = false;

  CGCubicBezierSmootherConfig config_iterative;
  config_iterative.lambda = 10.0;
  config_iterative.use_iterative_solver = true;
  config_iterative.schur_cg_tolerance = 1e-12;
  config_iterative.inner_cg_tolerance = 1e-12;

  // Solve with direct solver
  CGCubicBezierBathymetrySmoother smoother_direct(mesh, config_direct);
  smoother_direct.set_bathymetry_data(smooth_bathy);
  smoother_direct.solve();

  // Solve with iterative solver
  CGCubicBezierBathymetrySmoother smoother_iterative(mesh, config_iterative);
  smoother_iterative.set_bathymetry_data(smooth_bathy);
  smoother_iterative.solve();

  // Compare solutions at many points
  Real max_diff = 0.0;
  std::vector<Vec2> test_points = {
      {10.0, 10.0}, {25.0, 25.0}, {40.0, 40.0}, // Fine region
      {60.0, 25.0}, {75.0, 40.0}, {90.0, 10.0}, // Coarse region
      {50.0, 25.0}, {50.0, 12.5}, {50.0, 37.5}, // At interface
  };

  for (const auto &pt : test_points) {
    Real val_direct = smoother_direct.evaluate(pt(0), pt(1));
    Real val_iterative = smoother_iterative.evaluate(pt(0), pt(1));
    Real diff = std::abs(val_direct - val_iterative);
    max_diff = std::max(max_diff, diff);
  }

  std::cout << "Max solution difference (direct vs iterative): " << max_diff
            << std::endl;
  std::cout << "Constraint violation (direct):    "
            << smoother_direct.constraint_violation() << std::endl;
  std::cout << "Constraint violation (iterative): "
            << smoother_iterative.constraint_violation() << std::endl;

  // Solutions should match to high precision
  EXPECT_LT(max_diff, 1e-6);
}

TEST_F(CGCubicBezierSmootherTest, IterativeNoConstraints) {
  // Test iterative solver on uniform mesh (no edge constraints)
  auto mesh = create_quadtree(2, 2);

  auto bathy = [](Real x, Real y) { return 100.0 + 10.0 * x + 5.0 * y; };

  CGCubicBezierSmootherConfig config;
  config.lambda = 10.0;
  config.use_iterative_solver = true;

  CGCubicBezierBathymetrySmoother smoother(mesh, config);
  smoother.set_bathymetry_data(bathy);
  smoother.solve();

  EXPECT_TRUE(smoother.is_solved());

  // Should reproduce linear bathymetry well
  EXPECT_NEAR(smoother.evaluate(25.0, 25.0), bathy(25.0, 25.0), 5.0);
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
  // Expected: conforming (3 fine interior + 2 fine-fine edges = 5) * 4 gauss =
  // 20
  //         + non-conforming (2 fine-coarse edges * 4 gauss) = 8
  //         = 28 total
  EXPECT_GE(num_c1, 20); // At least conforming constraints

  // Check constraint violation
  Real violation = smoother.constraint_violation();
  std::cout << "Constraint violation: " << violation << std::endl;
  EXPECT_LT(violation, 1e-5);

  // Check derivative continuity at non-conforming interface (x = 50)
  Real max_deriv_jump = 0.0;
  Real dx = 0.01;

  for (Real y = 5.0; y < 45.0; y += 5.0) {
    // Approximate normal derivative (z_x) from both sides using finite
    // differences
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

  std::cout << "Max derivative jump at non-conforming interface: "
            << max_deriv_jump << std::endl;

  // With C¹ constraints, derivative should be continuous (small jump due to
  // finite differences)
  EXPECT_LT(max_deriv_jump,
            5.0); // Allow some numerical error from finite differences
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
  config_low.lambda = 100.0; // High data weight

  CGCubicBezierBathymetrySmoother low_smooth(mesh, config_low);
  low_smooth.set_bathymetry_data(noisy);
  low_smooth.solve();

  // High smoothing
  CGCubicBezierSmootherConfig config_high;
  config_high.lambda = 0.01; // Low data weight, high smoothing

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

  std::cout << "Variance - low smooth: " << var_low
            << ", high smooth: " << var_high << std::endl;

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
  config.ridge_epsilon = 0.0; // No ridge for exact polynomial reproduction

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
  config.ridge_epsilon = 0.0; // No ridge for exact polynomial reproduction

  // z = 10 + 2x + 3y, gradient = (2, 3)
  CGCubicBezierBathymetrySmoother smoother(mesh, config);
  smoother.set_bathymetry_data(
      [](Real x, Real y) { return 10.0 + 2.0 * x + 3.0 * y; });
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
  smoother.set_bathymetry_data([](Real x, Real y) { return 50.0 + x * 0.5; });
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
  mesh.add_element({h, 2 * h, 0.0, h}, {2, 2});
  mesh.add_element({0.0, h, h, 2 * h}, {2, 2});
  mesh.add_element({h, 2 * h, h, 2 * h}, {2, 2});
  mesh.add_element({2 * h, 4 * h, 0.0, 2 * h}, {1, 1});

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
  // CG cubic should have: (8*3+1)² = 25² = 625 DOFs (assuming 3 internal DOFs
  // per edge) Actually for cubic: (n_elem * (N1D-1) + 1)² = (8*3+1)² = 25² =
  // 625
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
  std::cout << "Domain: [" << xmin << ", " << xmax << "] x [" << ymin << ", "
            << ymax << "]" << std::endl;

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
    const auto &bounds = mesh.element_bounds(e);
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
  EXPECT_LT(avg_diff, 50.0); // Average error less than 50m
}

TEST_F(CGCubicBezierSmootherGeoTiffTest, KattegatWithBezierSubdivisionMG) {
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

  std::cout << "=== Kattegat with BezierSubdivision MG ===" << std::endl;
  std::cout << "Domain: [" << xmin << ", " << xmax << "] x [" << ymin << ", "
            << ymax << "]" << std::endl;

  // Create uniform mesh
  QuadtreeAdapter mesh;
  mesh.build_uniform(xmin, xmax, ymin, ymax, 8, 8);

  std::cout << "Mesh elements: " << mesh.num_elements() << std::endl;

  // Depth function using multi-source bathymetry
  auto depth_func = create_depth_function();

  // Create CG Cubic Bezier smoother with iterative solver and BezierSubdivision
  // MG
  CGCubicBezierSmootherConfig config;
  config.lambda = 10.0;
  config.ngauss_data = 4;
  config.ngauss_energy = 4;
  config.use_iterative_solver = true;
  config.use_multigrid = true;
  config.schur_cg_tolerance = 1e-10;
  config.multigrid_config.num_levels = 3;
  config.multigrid_config.min_tree_level = 0;
  config.multigrid_config.pre_smoothing = 2;
  config.multigrid_config.post_smoothing = 2;
  config.multigrid_config.smoother_type = SmootherType::MultiplicativeSchwarz;
  config.multigrid_config.transfer_strategy =
      TransferOperatorStrategy::BezierSubdivision;
  config.multigrid_config.coarse_grid_strategy =
      CoarseGridStrategy::CachedRediscretization;

  CGCubicBezierBathymetrySmoother smoother(mesh, config);
  smoother.set_bathymetry_data(depth_func);

  std::cout << "DOFs: " << smoother.num_global_dofs() << std::endl;
  std::cout << "Constraints: " << smoother.num_constraints() << std::endl;

  // Solve
  smoother.solve();

  EXPECT_TRUE(smoother.is_solved());

  // Write output for ParaView verification
  std::string output_base = "/tmp/cg_cubic_bezier_kattegat_subdivision_mg";
  smoother.write_vtk(output_base, 10);

  std::string output_file = output_base + ".vtu";
  std::cout << "Output written to: " << output_file << std::endl;

  // Also write control points
  std::string cp_file = output_base + "_control_points.vtu";
  smoother.write_control_points_vtk(cp_file);
  std::cout << "Control points written to: " << cp_file << std::endl;

  // Verify solution quality - compare to input at sample points
  Real max_diff = 0.0;
  Real sum_diff = 0.0;
  int count = 0;

  for (Real x = xmin + 1000; x < xmax - 1000; x += 1000) {
    for (Real y = ymin + 1000; y < ymax - 1000; y += 1000) {
      Real input_depth = depth_func(x, y);
      Real output_depth = smoother.evaluate(x, y);

      if (!std::isnan(input_depth) && !std::isnan(output_depth)) {
        Real diff = std::abs(input_depth - output_depth);
        max_diff = std::max(max_diff, diff);
        sum_diff += diff;
        count++;
      }
    }
  }

  Real avg_diff = sum_diff / count;
  std::cout << "Max difference from input: " << max_diff << " m" << std::endl;
  std::cout << "Avg difference from input: " << avg_diff << " m" << std::endl;

  // Reasonable fit quality (smoothed, so not exact)
  EXPECT_LT(avg_diff, 50.0); // Average error less than 50m
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
            << smoother.dof_manager().num_edge_derivative_constraints()
            << std::endl;

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
    const auto &bounds = mesh.element_bounds(e);
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
  std::cout << "Max diff: " << max_diff << " m, Avg diff: " << avg_diff << " m"
            << std::endl;
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
      0.0337652428984240, 0.1693953067668677, 0.3806904069584015,
      0.6193095930415985, 0.8306046932331323, 0.9662347571015760};
  std::array<Real, ngauss> gauss_weights = {
      0.0856622461895852, 0.1803807865240693, 0.2339569672863455,
      0.2339569672863455, 0.1803807865240693, 0.0856622461895852};

  // Helper to compute L2 error for an element
  auto compute_element_l2_error =
      [&](const CGCubicBezierBathymetrySmoother &smoother,
          const QuadBounds &bounds) -> Real {
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

  std::cout
      << "\n=== CGCubicBezierBathymetrySmoother Uniform Grid Evaluation ==="
      << std::endl;
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
      double time_ms =
          std::chrono::duration<double, std::milli>(end - start).count();

      // Compute errors
      Real max_err = 0.0;
      Real sum_err = 0.0;
      Index num_elems = mesh.num_elements();

      for (Index e = 0; e < num_elems; ++e) {
        const auto &bounds = mesh.element_bounds(e);
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

      std::cout << "  " << mesh_n << "x" << mesh_n << " lambda=" << lambda
                << ": "
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
  ofs << "| Mesh | Lambda | Elements | DOFs | Max Err (m) | Mean Err (m) | "
         "Data Res | Constr Viol | Reg Energy | Time (ms) |\n";
  ofs << "|------|--------|----------|------|-------------|--------------|-----"
         "-----|-------------|------------|----------|\n";

  for (const auto &r : results) {
    // Format lambda appropriately
    std::ostringstream lambda_ss;
    if (r.lambda >= 1.0) {
      lambda_ss << std::fixed << std::setprecision(0) << r.lambda;
    } else {
      lambda_ss << std::fixed << std::setprecision(2) << r.lambda;
    }

    ofs << "| " << r.mesh_size << "x" << r.mesh_size << " | " << lambda_ss.str()
        << " | " << r.num_elements << " | " << r.num_dofs << " | " << std::fixed
        << std::setprecision(3) << r.max_error << " | " << std::fixed
        << std::setprecision(3) << r.mean_error << " | " << std::scientific
        << std::setprecision(2) << r.data_residual << " | " << std::scientific
        << std::setprecision(2) << r.constraint_violation << " | "
        << std::scientific << std::setprecision(2) << r.regularization_energy
        << " | " << std::fixed << std::setprecision(1) << r.wall_time_ms
        << " |\n";
  }

  ofs << "\n## Configuration Legend\n\n";
  ofs << "- **c1_vertex_only**: Vertex derivative constraints (z_u, z_v, z_uv "
         "at shared vertices)\n";
  ofs << "- **c1_edge_only**: Edge Gauss point constraints (z_n at 4 points "
         "per edge)\n";
  ofs << "- **c1_vertex_edge**: Both vertex and edge constraints\n\n";

  ofs << "## Metric Definitions\n\n";
  ofs << "- **Max Err**: Maximum normalized L2 error across all elements "
         "(meters)\n";
  ofs << "- **Mean Err**: Mean normalized L2 error across all elements "
         "(meters)\n";
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
    const EvalResult *best = nullptr;
    for (const auto &r : results) {
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
        ts << std::fixed << std::setprecision(1)
           << (best->wall_time_ms / 1000.0) << "s";
        time_str = ts.str();
      }

      ofs << "| " << best->mesh_size << "×" << best->mesh_size << " | "
          << lambda_ss.str() << " | " << std::fixed << std::setprecision(1)
          << best->max_error << "m"
          << " | " << time_str << " |\n";
    }
  }

  // Compute observations
  ofs << "\n### Observations\n\n";

  // Find best lambda across all results
  std::map<Real, int> lambda_wins;
  for (int mesh_n : mesh_sizes) {
    const EvalResult *best = nullptr;
    for (const auto &r : results) {
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
  for (const auto &[lam, wins] : lambda_wins) {
    if (wins > max_wins) {
      max_wins = wins;
      most_common_lambda = lam;
    }
  }

  ofs << "1. **Lambda sweet spot**: λ=" << std::fixed << std::setprecision(0)
      << most_common_lambda
      << " consistently gives lowest errors across most mesh sizes\n\n";

  // Error reduction with refinement
  Real error_4x4 = 0, error_32x32 = 0;
  for (const auto &r : results) {
    if (r.mesh_size == 4 && r.max_error > error_4x4)
      error_4x4 = r.max_error;
    if (r.mesh_size == 32) {
      if (error_32x32 == 0 || r.max_error < error_32x32)
        error_32x32 = r.max_error;
    }
  }
  ofs << "2. **Mesh refinement helps**: Error decreases from ~" << std::fixed
      << std::setprecision(0) << error_4x4 << "m (4×4) to ~"
      << std::setprecision(1) << error_32x32 << "m (32×32)\n\n";

  // Over-smoothing warning
  Real max_error_low_lambda = 0;
  for (const auto &r : results) {
    if (r.lambda <= 0.01 && r.max_error > max_error_low_lambda) {
      max_error_low_lambda = r.max_error;
    }
  }
  ofs << "3. **Over-smoothing**: λ≤0.1 causes significant error increase (up "
         "to "
      << std::fixed << std::setprecision(0) << max_error_low_lambda
      << "m for λ=0.01)\n\n";

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

// =============================================================================
// CG Tolerance Sweep: Iterations vs Tolerance
// =============================================================================

TEST_F(CGCubicBezierSmootherTest, DISABLED_CGToleranceSweep) {
  // Test how CG iterations scale with tolerance
  QuadtreeAdapter mesh;
  mesh.build_uniform(0.0, 100.0, 0.0, 100.0, 32, 32);

  auto bathy = [](Real x, Real y) {
    return 100.0 + 20.0 * std::sin(0.05 * x) * std::cos(0.05 * y);
  };

  std::cout << "\nTolerance vs CG iterations (32x32 uniform mesh):\n";
  std::cout << "Tolerance    | Iterations | Time (ms)\n";
  std::cout << "-------------|------------|----------\n";

  for (Real tol : {1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14}) {
    CGCubicBezierSmootherConfig config;
    config.lambda = 10.0;
    config.use_iterative_solver = true;
    config.schur_cg_tolerance = tol;

    CGCubicSolveProfile profile;
    CGCubicBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_solve_profile(&profile);
    smoother.set_bathymetry_data(bathy);

    auto start = std::chrono::high_resolution_clock::now();
    smoother.solve();
    auto end = std::chrono::high_resolution_clock::now();
    double time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << std::scientific << std::setprecision(0) << tol << "    | "
              << std::setw(10) << profile.outer_cg_iterations << " | "
              << std::fixed << std::setprecision(1) << time_ms << "\n";
  }
}

// =============================================================================
// Solver Performance Benchmark: Direct vs Iterative
// =============================================================================

TEST_F(CGCubicBezierSmootherTest, DISABLED_SolverPerformanceBenchmark) {
  // Benchmark comparing direct SparseLU vs iterative Schur complement CG solver

  struct BenchmarkResult {
    std::string mesh_type;
    int mesh_param; // Grid size or element count
    Index num_elements;
    Index num_dofs;
    Index num_edge_constraints;

    // Direct solver timings
    double direct_total_ms;
    double direct_matrix_build_ms;
    double direct_constraint_build_ms;
    double direct_kkt_assembly_ms;
    double direct_lu_compute_ms;
    double direct_lu_solve_ms;
    double direct_projection_ms;

    // Iterative solver timings
    double iterative_total_ms;
    double iterative_matrix_build_ms;
    double iterative_constraint_build_ms;
    double iterative_q_factor_ms;
    double iterative_cg_total_ms;
    int iterative_cg_iterations;
    int iterative_inner_solves;
    double iterative_projection_ms;

    // Quality
    Real solution_max_diff;
    Real direct_constraint_violation;
    Real iterative_constraint_violation;

    double speedup() const {
      return (iterative_total_ms > 0) ? direct_total_ms / iterative_total_ms
                                      : 0.0;
    }
  };

  // Bathymetry function
  auto bathy_func = [](Real x, Real y) {
    return 100.0 + 20.0 * std::sin(0.05 * x) * std::cos(0.05 * y) +
           10.0 * std::cos(0.08 * x) * std::sin(0.08 * y);
  };

  // Helper to run benchmark for a single mesh configuration
  auto run_benchmark = [&](QuadtreeAdapter &mesh, const std::string &mesh_type,
                           int mesh_param, Real lambda) -> BenchmarkResult {
    BenchmarkResult result;
    result.mesh_type = mesh_type;
    result.mesh_param = mesh_param;
    result.num_elements = mesh.num_elements();

    // === Direct Solver ===
    CGCubicBezierSmootherConfig config_direct;
    config_direct.lambda = lambda;
    config_direct.edge_ngauss = 4;
    config_direct.use_iterative_solver = false;

    CGCubicSolveProfile profile_direct;
    CGCubicBezierBathymetrySmoother smoother_direct(mesh, config_direct);
    smoother_direct.set_solve_profile(&profile_direct);
    smoother_direct.set_bathymetry_data(bathy_func);

    result.num_dofs = smoother_direct.num_global_dofs();
    result.num_edge_constraints = smoother_direct.num_constraints();

    auto start_direct = std::chrono::high_resolution_clock::now();
    smoother_direct.solve();
    auto end_direct = std::chrono::high_resolution_clock::now();

    result.direct_total_ms =
        std::chrono::duration<double, std::milli>(end_direct - start_direct)
            .count();
    result.direct_matrix_build_ms = profile_direct.matrix_build_ms;
    result.direct_constraint_build_ms = profile_direct.constraint_build_ms;
    result.direct_kkt_assembly_ms = profile_direct.kkt_assembly_ms;
    result.direct_lu_compute_ms = profile_direct.sparse_lu_compute_ms;
    result.direct_lu_solve_ms = profile_direct.sparse_lu_solve_ms;
    result.direct_projection_ms = profile_direct.constraint_projection_ms;
    result.direct_constraint_violation = smoother_direct.constraint_violation();

    // === Iterative Solver ===
    CGCubicBezierSmootherConfig config_iter;
    config_iter.lambda = lambda;
    config_iter.edge_ngauss = 4;
    config_iter.use_iterative_solver = true;
    config_iter.schur_cg_tolerance = 1e-10;
    config_iter.use_multigrid = true;

    CGCubicSolveProfile profile_iter;
    CGCubicBezierBathymetrySmoother smoother_iter(mesh, config_iter);
    smoother_iter.set_solve_profile(&profile_iter);
    smoother_iter.set_bathymetry_data(bathy_func);

    auto start_iter = std::chrono::high_resolution_clock::now();
    smoother_iter.solve();
    auto end_iter = std::chrono::high_resolution_clock::now();

    result.iterative_total_ms =
        std::chrono::duration<double, std::milli>(end_iter - start_iter)
            .count();
    result.iterative_matrix_build_ms = profile_iter.matrix_build_ms;
    result.iterative_constraint_build_ms = profile_iter.constraint_build_ms;
    result.iterative_q_factor_ms = profile_iter.inner_cg_setup_ms;
    result.iterative_cg_total_ms = profile_iter.outer_cg_total_ms;
    result.iterative_cg_iterations = profile_iter.outer_cg_iterations;
    result.iterative_inner_solves = profile_iter.inner_cg_total_calls;
    result.iterative_projection_ms = profile_iter.constraint_projection_ms;
    result.iterative_constraint_violation =
        smoother_iter.constraint_violation();

    // === Compare solutions ===
    Real max_diff = 0.0;
    for (Index elem = 0; elem < mesh.num_elements(); ++elem) {
      auto bounds = mesh.element_bounds(elem);
      Real cx = (bounds.xmin + bounds.xmax) / 2.0;
      Real cy = (bounds.ymin + bounds.ymax) / 2.0;
      Real val_direct = smoother_direct.evaluate(cx, cy);
      Real val_iter = smoother_iter.evaluate(cx, cy);
      max_diff = std::max(max_diff, std::abs(val_direct - val_iter));
    }
    result.solution_max_diff = max_diff;

    return result;
  };

  std::vector<BenchmarkResult> results;
  Real lambda = 10.0;

  // === Scenario 1: Uniform conforming meshes ===
  std::cout << "\n=== Uniform Conforming Meshes ===\n";
  std::vector<int> uniform_sizes = {4, 8, 16, 24, 32, 48};

  for (int n : uniform_sizes) {
    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 100.0, 0.0, 100.0, n, n);

    auto result = run_benchmark(mesh, "uniform", n, lambda);
    results.push_back(result);

    std::cout << std::fixed << std::setprecision(1);
    std::cout << n << "x" << n << ": direct=" << result.direct_total_ms
              << "ms, iter=" << result.iterative_total_ms
              << "ms, speedup=" << std::setprecision(2) << result.speedup()
              << "x, CG iters=" << result.iterative_cg_iterations
              << ", max_diff=" << std::scientific << std::setprecision(2)
              << result.solution_max_diff << "\n";
  }

  // === Scenario 2: Non-conforming meshes (2:1 interfaces) ===
  std::cout << "\n=== Non-conforming Meshes (2:1 interfaces) ===\n";

  // 1+4 configuration: 4 fine elements + 1 coarse
  {
    QuadtreeAdapter mesh;
    Real h = 25.0;
    mesh.add_element({0.0, h, 0.0, h}, {2, 2});
    mesh.add_element({h, 2 * h, 0.0, h}, {2, 2});
    mesh.add_element({0.0, h, h, 2 * h}, {2, 2});
    mesh.add_element({h, 2 * h, h, 2 * h}, {2, 2});
    mesh.add_element({2 * h, 4 * h, 0.0, 2 * h}, {1, 1});

    auto result = run_benchmark(mesh, "1+4", 5, lambda);
    results.push_back(result);

    std::cout << "1+4 mesh (5 elem): direct=" << std::fixed
              << std::setprecision(1) << result.direct_total_ms
              << "ms, iter=" << result.iterative_total_ms
              << "ms, speedup=" << std::setprecision(2) << result.speedup()
              << "x, CG=" << result.iterative_cg_iterations << "\n";
  }

  // Graded mesh: multiple refinement levels
  {
    QuadtreeAdapter mesh;
    // Base coarse elements
    mesh.add_element({0.0, 50.0, 0.0, 50.0}, {1, 1});
    mesh.add_element({50.0, 100.0, 0.0, 50.0}, {1, 1});
    mesh.add_element({0.0, 50.0, 50.0, 100.0}, {1, 1});
    // Fine region in upper-right
    mesh.add_element({50.0, 75.0, 50.0, 75.0}, {2, 2});
    mesh.add_element({75.0, 100.0, 50.0, 75.0}, {2, 2});
    mesh.add_element({50.0, 75.0, 75.0, 100.0}, {2, 2});
    mesh.add_element({75.0, 100.0, 75.0, 100.0}, {2, 2});

    auto result = run_benchmark(mesh, "graded", 7, lambda);
    results.push_back(result);

    std::cout << "Graded mesh (7 elem): direct=" << std::fixed
              << std::setprecision(1) << result.direct_total_ms
              << "ms, iter=" << result.iterative_total_ms
              << "ms, speedup=" << std::setprecision(2) << result.speedup()
              << "x, CG=" << result.iterative_cg_iterations << "\n";
  }

  // Larger non-conforming: 16 fine + 12 medium + 3 coarse = 31 elements
  {
    QuadtreeAdapter mesh;
    Real h = 12.5; // Fine element size
    // 4x4 fine region (0-50, 0-50)
    for (int j = 0; j < 4; ++j) {
      for (int i = 0; i < 4; ++i) {
        mesh.add_element({i * h, (i + 1) * h, j * h, (j + 1) * h}, {3, 3});
      }
    }
    // Medium elements (50-100, 0-50)
    Real m = 25.0; // Medium size
    for (int j = 0; j < 2; ++j) {
      for (int i = 0; i < 2; ++i) {
        mesh.add_element({50.0 + i * m, 50.0 + (i + 1) * m, j * m, (j + 1) * m},
                         {2, 2});
      }
    }
    // Medium elements (0-50, 50-100)
    for (int j = 0; j < 2; ++j) {
      for (int i = 0; i < 2; ++i) {
        mesh.add_element({i * m, (i + 1) * m, 50.0 + j * m, 50.0 + (j + 1) * m},
                         {2, 2});
      }
    }
    // Medium elements (50-100, 50-100)
    for (int j = 0; j < 2; ++j) {
      for (int i = 0; i < 2; ++i) {
        mesh.add_element({50.0 + i * m, 50.0 + (i + 1) * m, 50.0 + j * m,
                          50.0 + (j + 1) * m},
                         {2, 2});
      }
    }

    auto result = run_benchmark(mesh, "multi-level",
                                static_cast<int>(mesh.num_elements()), lambda);
    results.push_back(result);

    std::cout << "Multi-level (" << mesh.num_elements()
              << " elem): direct=" << std::fixed << std::setprecision(1)
              << result.direct_total_ms
              << "ms, iter=" << result.iterative_total_ms
              << "ms, speedup=" << std::setprecision(2) << result.speedup()
              << "x, CG=" << result.iterative_cg_iterations << "\n";
  }

  // === Write markdown report ===
  std::string output_path = "/tmp/solver_performance.md";
  std::ofstream ofs(output_path);

  ofs << "# CG Cubic Bezier Solver Performance: Direct vs Iterative\n\n";
  ofs << "Benchmark comparing SparseLU (direct) vs Schur complement CG "
         "(iterative) solvers.\n\n";
  ofs << "- Lambda: " << lambda << "\n";
  ofs << "- Edge Gauss points: 4\n";
  ofs << "- Iterative tolerance: 1e-10\n\n";

  // Main results table
  ofs << "## Summary Results\n\n";
  ofs << "| Mesh | Elements | DOFs | Edge Cstr | Direct (ms) | Iterative (ms) "
         "| Speedup | CG Iters | Max Diff |\n";
  ofs << "|------|----------|------|-----------|-------------|----------------|"
         "---------|----------|----------|\n";

  for (const auto &r : results) {
    ofs << "| " << r.mesh_type << " " << r.mesh_param << " | " << r.num_elements
        << " | " << r.num_dofs << " | " << r.num_edge_constraints << " | "
        << std::fixed << std::setprecision(1) << r.direct_total_ms << " | "
        << r.iterative_total_ms << " | " << std::setprecision(2) << r.speedup()
        << "x"
        << " | " << r.iterative_cg_iterations << " | " << std::scientific
        << std::setprecision(1) << r.solution_max_diff << " |\n";
  }

  // Detailed breakdown: Direct solver
  ofs << "\n## Direct Solver Time Breakdown (ms)\n\n";
  ofs << "| Mesh | Matrix | Constraint | KKT Asm | LU Compute | LU Solve | "
         "Projection |\n";
  ofs << "|------|--------|------------|---------|------------|----------|-----"
         "-------|\n";

  for (const auto &r : results) {
    ofs << "| " << r.mesh_type << " " << r.mesh_param << " | " << std::fixed
        << std::setprecision(2) << r.direct_matrix_build_ms << " | "
        << r.direct_constraint_build_ms << " | " << r.direct_kkt_assembly_ms
        << " | " << r.direct_lu_compute_ms << " | " << r.direct_lu_solve_ms
        << " | " << r.direct_projection_ms << " |\n";
  }

  // Detailed breakdown: Iterative solver
  ofs << "\n## Iterative Solver Time Breakdown (ms)\n\n";
  ofs << "| Mesh | Matrix | Constraint | Q Factor | CG Loop | Inner Solves | "
         "Projection |\n";
  ofs << "|------|--------|------------|----------|---------|--------------|---"
         "---------|\n";

  for (const auto &r : results) {
    ofs << "| " << r.mesh_type << " " << r.mesh_param << " | " << std::fixed
        << std::setprecision(2) << r.iterative_matrix_build_ms << " | "
        << r.iterative_constraint_build_ms << " | " << r.iterative_q_factor_ms
        << " | " << r.iterative_cg_total_ms << " | " << r.iterative_inner_solves
        << " | " << r.iterative_projection_ms << " |\n";
  }

  // Constraint violations
  ofs << "\n## Constraint Violations\n\n";
  ofs << "| Mesh | Direct Violation | Iterative Violation |\n";
  ofs << "|------|------------------|---------------------|\n";

  for (const auto &r : results) {
    ofs << "| " << r.mesh_type << " " << r.mesh_param << " | "
        << std::scientific << std::setprecision(2)
        << r.direct_constraint_violation << " | "
        << r.iterative_constraint_violation << " |\n";
  }

  // Analysis
  ofs << "\n## Analysis\n\n";

  // Find crossover point
  Real crossover_dofs = 0;
  for (size_t i = 1; i < results.size(); ++i) {
    if (results[i].mesh_type == "uniform" && results[i].speedup() > 1.0 &&
        results[i - 1].mesh_type == "uniform" &&
        results[i - 1].speedup() <= 1.0) {
      crossover_dofs =
          static_cast<Real>(results[i - 1].num_dofs + results[i].num_dofs) /
          2.0;
    }
  }

  if (crossover_dofs > 0) {
    ofs << "1. **Crossover point**: Iterative becomes faster around "
        << static_cast<int>(crossover_dofs) << " DOFs\n";
  }

  // Find max speedup
  double max_speedup = 0;
  std::string max_speedup_mesh;
  for (const auto &r : results) {
    if (r.speedup() > max_speedup) {
      max_speedup = r.speedup();
      max_speedup_mesh = r.mesh_type + " " + std::to_string(r.mesh_param);
    }
  }
  ofs << "2. **Maximum speedup**: " << std::fixed << std::setprecision(2)
      << max_speedup << "x on " << max_speedup_mesh << "\n";

  // Accuracy check
  Real max_diff_overall = 0;
  for (const auto &r : results) {
    max_diff_overall = std::max(max_diff_overall, r.solution_max_diff);
  }
  ofs << "3. **Solution accuracy**: Maximum difference between solvers: "
      << std::scientific << std::setprecision(2) << max_diff_overall << "\n";

  ofs.close();

  std::cout << "\n=== Report written to: " << output_path << " ===\n";

  // Verify solutions match
  for (const auto &r : results) {
    EXPECT_LT(r.solution_max_diff, 1e-5)
        << "Solution mismatch for " << r.mesh_type << " " << r.mesh_param;
  }
}

// =============================================================================
// Detailed Iterative Solver Profiling Benchmark
// =============================================================================

TEST_F(CGCubicBezierSmootherTest, DISABLED_IterativeSolverProfilingBenchmark) {
  // Comprehensive profiling of solve_with_constraints_iterative() with
  // MultiplicativeSchwarz multigrid smoother to identify bottlenecks.

  struct ProfilingResult {
    std::string mesh_desc;
    Index num_elements;
    Index num_dofs;
    Index num_edge_constraints;

    // Top-level timing (ms)
    double total_solve_ms;
    double matrix_build_ms;
    double constraint_build_ms;
    double edge_constraint_ms;
    double qinv_setup_ms;
    double outer_cg_ms;
    double schur_rhs_ms;
    double schur_matvec_ms;
    double cg_vector_ops_ms;
    double solution_recovery_ms;
    double qinv_apply_total_ms;
    int qinv_apply_calls;
    int cg_iterations;

    // MG setup breakdown (ms)
    double mg_setup_total_ms;
    double mg_setup_finest_ms;
    double mg_setup_element_blocks_ms;
    double mg_setup_prolongation_ms;
    double mg_setup_galerkin_ms;
    double mg_setup_coarse_lu_ms;

    // MG apply breakdown (ms)
    double mg_apply_total_ms;
    double mg_vcycle_pre_smooth_ms;
    double mg_vcycle_residual_ms;
    double mg_vcycle_restrict_ms;
    double mg_vcycle_prolong_ms;
    double mg_vcycle_post_smooth_ms;
    double mg_vcycle_coarse_solve_ms;

    // Schwarz breakdown (ms)
    double schwarz_matvec_ms;
    double schwarz_gather_ms;
    double schwarz_local_solve_ms;
    double schwarz_scatter_update_ms;

    // Jacobi (ms)
    double jacobi_total_ms;

    // Counters
    int mg_apply_calls;
    int vcycle_calls;
    int schwarz_iterations;
    int schwarz_element_solves;
    int matvec_products;
    int coarse_solves;
  };

  auto bathy_func = [](Real x, Real y) {
    return 100.0 + 20.0 * std::sin(0.05 * x) * std::cos(0.05 * y) +
           10.0 * std::cos(0.08 * x) * std::sin(0.08 * y);
  };

  std::vector<int> sizes = {4, 8, 16, 32};
  std::vector<ProfilingResult> results;
  Real lambda = 10.0;

  std::cout << "\n=== Iterative Solver Profiling (MultiplicativeSchwarz MG) "
               "===\n";

  for (int n : sizes) {
    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 100.0, 0.0, 100.0, n, n);

    CGCubicBezierSmootherConfig config;
    config.lambda = lambda;
    config.edge_ngauss = 4;
    config.use_iterative_solver = true;
    config.use_multigrid = true;
    config.schur_cg_tolerance = 1e-10;
    config.multigrid_config.num_levels = 3;
    config.multigrid_config.pre_smoothing = 1;
    config.multigrid_config.post_smoothing = 1;
    config.multigrid_config.smoother_type = SmootherType::MultiplicativeSchwarz;

    MultigridProfile mg_profile;
    CGCubicSolveProfile solve_profile;
    solve_profile.multigrid_profile = &mg_profile;

    CGCubicBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_solve_profile(&solve_profile);
    smoother.set_bathymetry_data(bathy_func);

    auto start = std::chrono::high_resolution_clock::now();
    smoother.solve();
    auto end = std::chrono::high_resolution_clock::now();

    ProfilingResult r;
    r.mesh_desc = std::to_string(n) + "x" + std::to_string(n);
    r.num_elements = mesh.num_elements();
    r.num_dofs = smoother.num_free_dofs();
    r.num_edge_constraints = smoother.num_constraints();

    r.total_solve_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    r.matrix_build_ms = solve_profile.matrix_build_ms;
    r.constraint_build_ms = solve_profile.constraint_build_ms;
    r.edge_constraint_ms = solve_profile.edge_constraint_assembly_ms;
    r.qinv_setup_ms = solve_profile.inner_cg_setup_ms;
    r.outer_cg_ms = solve_profile.outer_cg_total_ms;
    r.schur_rhs_ms = solve_profile.schur_rhs_ms;
    r.schur_matvec_ms = solve_profile.schur_matvec_total_ms;
    r.cg_vector_ops_ms = solve_profile.cg_vector_ops_ms;
    r.solution_recovery_ms = solve_profile.solution_recovery_ms;
    r.qinv_apply_total_ms = solve_profile.qinv_apply_total_ms;
    r.qinv_apply_calls = solve_profile.qinv_apply_calls;
    r.cg_iterations = solve_profile.outer_cg_iterations;

    r.mg_setup_total_ms = mg_profile.setup_total_ms;
    r.mg_setup_finest_ms = mg_profile.setup_finest_init_ms;
    r.mg_setup_element_blocks_ms = mg_profile.setup_element_blocks_ms;
    r.mg_setup_prolongation_ms = mg_profile.setup_prolongation_ms;
    r.mg_setup_galerkin_ms = mg_profile.setup_galerkin_ms;
    r.mg_setup_coarse_lu_ms = mg_profile.setup_coarse_lu_ms;

    r.mg_apply_total_ms = mg_profile.apply_total_ms;
    r.mg_vcycle_pre_smooth_ms = mg_profile.vcycle_pre_smooth_ms;
    r.mg_vcycle_residual_ms = mg_profile.vcycle_residual_ms;
    r.mg_vcycle_restrict_ms = mg_profile.vcycle_restrict_ms;
    r.mg_vcycle_prolong_ms = mg_profile.vcycle_prolong_ms;
    r.mg_vcycle_post_smooth_ms = mg_profile.vcycle_post_smooth_ms;
    r.mg_vcycle_coarse_solve_ms = mg_profile.vcycle_coarse_solve_ms;

    r.schwarz_matvec_ms = mg_profile.schwarz_matvec_ms;
    r.schwarz_gather_ms = mg_profile.schwarz_gather_ms;
    r.schwarz_local_solve_ms = mg_profile.schwarz_local_solve_ms;
    r.schwarz_scatter_update_ms = mg_profile.schwarz_scatter_update_ms;

    r.jacobi_total_ms = mg_profile.jacobi_total_ms;

    r.mg_apply_calls = mg_profile.apply_calls;
    r.vcycle_calls = mg_profile.vcycle_calls;
    r.schwarz_iterations = mg_profile.schwarz_iterations;
    r.schwarz_element_solves = mg_profile.schwarz_element_solves;
    r.matvec_products = mg_profile.matvec_products;
    r.coarse_solves = mg_profile.coarse_solves;

    results.push_back(r);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << r.mesh_desc << ": total=" << r.total_solve_ms
              << "ms, CG iters=" << r.cg_iterations
              << ", Q^-1 calls=" << r.qinv_apply_calls << "\n";
  }

  // Write markdown report
  std::string output_path = "/tmp/iterative_solver_profiling.md";
  std::ofstream ofs(output_path);

  ofs << "# Iterative Solver Profiling: MultiplicativeSchwarz MG\n\n";
  ofs << "Lambda: " << lambda << ", Tolerance: 1e-10\n\n";

  // Top-level breakdown
  ofs << "## Top-Level Solve Breakdown (ms)\n\n";
  ofs << "| Mesh | Elements | DOFs | Total | Matrix | Constraint | Edge | "
         "Q^-1 Setup | CG Loop |\n";
  ofs << "|------|----------|------|-------|--------|------------|------|---"
         "--------|--------|\n";
  for (const auto &r : results) {
    ofs << "| " << r.mesh_desc << " | " << r.num_elements << " | " << r.num_dofs
        << " | " << std::fixed << std::setprecision(2) << r.total_solve_ms
        << " | " << r.matrix_build_ms << " | " << r.constraint_build_ms << " | "
        << r.edge_constraint_ms << " | " << r.qinv_setup_ms << " | "
        << r.outer_cg_ms << " |\n";
  }

  // CG iteration details
  ofs << "\n## CG Iteration Details\n\n";
  ofs << "| Mesh | CG Iters | Q^-1 Calls | Schur RHS | Schur MatVec | Vec Ops "
         "| Recovery | Q^-1 Total |\n";
  ofs << "|------|----------|------------|-----------|--------------|---------|"
         "----------|------------|\n";
  for (const auto &r : results) {
    ofs << "| " << r.mesh_desc << " | " << r.cg_iterations << " | "
        << r.qinv_apply_calls << " | " << std::fixed << std::setprecision(2)
        << r.schur_rhs_ms << " | " << r.schur_matvec_ms << " | "
        << r.cg_vector_ops_ms << " | " << r.solution_recovery_ms << " | "
        << r.qinv_apply_total_ms << " |\n";
  }

  // MG setup breakdown
  ofs << "\n## Multigrid Setup Breakdown (ms)\n\n";
  ofs << "| Mesh | Total | Finest Init | Element Blocks | Prolongation | "
         "Galerkin | Coarse LU |\n";
  ofs << "|------|-------|-------------|----------------|--------------|------"
         "----|----------|\n";
  for (const auto &r : results) {
    ofs << "| " << r.mesh_desc << " | " << std::fixed << std::setprecision(2)
        << r.mg_setup_total_ms << " | " << r.mg_setup_finest_ms << " | "
        << r.mg_setup_element_blocks_ms << " | " << r.mg_setup_prolongation_ms
        << " | " << r.mg_setup_galerkin_ms << " | " << r.mg_setup_coarse_lu_ms
        << " |\n";
  }

  // V-cycle breakdown
  ofs << "\n## V-Cycle Breakdown (ms, accumulated over all applies)\n\n";
  ofs << "| Mesh | Apply Total | Pre-Smooth | Residual | Restrict | Prolong | "
         "Post-Smooth | Coarse |\n";
  ofs << "|------|-------------|------------|----------|----------|---------|--"
         "----------|--------|\n";
  for (const auto &r : results) {
    ofs << "| " << r.mesh_desc << " | " << std::fixed << std::setprecision(2)
        << r.mg_apply_total_ms << " | " << r.mg_vcycle_pre_smooth_ms << " | "
        << r.mg_vcycle_residual_ms << " | " << r.mg_vcycle_restrict_ms << " | "
        << r.mg_vcycle_prolong_ms << " | " << r.mg_vcycle_post_smooth_ms
        << " | " << r.mg_vcycle_coarse_solve_ms << " |\n";
  }

  // Schwarz breakdown
  ofs << "\n## Schwarz Smoother Breakdown (ms, finest level only)\n\n";
  ofs << "| Mesh | MatVec | Gather | Local Solve | Scatter+Update | Total |\n";
  ofs << "|------|--------|--------|-------------|----------------|-------|\n";
  double schwarz_total = 0.0;
  for (const auto &r : results) {
    schwarz_total = r.schwarz_matvec_ms + r.schwarz_gather_ms +
                    r.schwarz_local_solve_ms + r.schwarz_scatter_update_ms;
    ofs << "| " << r.mesh_desc << " | " << std::fixed << std::setprecision(2)
        << r.schwarz_matvec_ms << " | " << r.schwarz_gather_ms << " | "
        << r.schwarz_local_solve_ms << " | " << r.schwarz_scatter_update_ms
        << " | " << schwarz_total << " |\n";
  }

  // Schwarz percentage breakdown
  ofs << "\n## Schwarz Smoother Breakdown (%)\n\n";
  ofs << "| Mesh | MatVec % | Gather % | Local Solve % | Scatter+Update % |\n";
  ofs << "|------|----------|----------|---------------|------------------|\n";
  for (const auto &r : results) {
    schwarz_total = r.schwarz_matvec_ms + r.schwarz_gather_ms +
                    r.schwarz_local_solve_ms + r.schwarz_scatter_update_ms;
    if (schwarz_total > 0.001) {
      ofs << "| " << r.mesh_desc << " | " << std::fixed << std::setprecision(1)
          << 100.0 * r.schwarz_matvec_ms / schwarz_total << " | "
          << 100.0 * r.schwarz_gather_ms / schwarz_total << " | "
          << 100.0 * r.schwarz_local_solve_ms / schwarz_total << " | "
          << 100.0 * r.schwarz_scatter_update_ms / schwarz_total << " |\n";
    } else {
      ofs << "| " << r.mesh_desc << " | - | - | - | - |\n";
    }
  }

  // Operation counters
  ofs << "\n## Operation Counters\n\n";
  ofs << "| Mesh | V-cycles | Schwarz Iters | Element Solves | Mat-Vecs | "
         "Coarse Solves |\n";
  ofs << "|------|----------|---------------|----------------|----------|-"
         "--------------|\n";
  for (const auto &r : results) {
    ofs << "| " << r.mesh_desc << " | " << r.vcycle_calls << " | "
        << r.schwarz_iterations << " | " << r.schwarz_element_solves << " | "
        << r.matvec_products << " | " << r.coarse_solves << " |\n";
  }

  // Analysis
  ofs << "\n## Analysis\n\n";
  if (!results.empty()) {
    const auto &last = results.back();
    double schwarz_total_last =
        last.schwarz_matvec_ms + last.schwarz_gather_ms +
        last.schwarz_local_solve_ms + last.schwarz_scatter_update_ms;

    if (schwarz_total_last > 0.001) {
      double matvec_pct = 100.0 * last.schwarz_matvec_ms / schwarz_total_last;
      double scatter_pct =
          100.0 * last.schwarz_scatter_update_ms / schwarz_total_last;

      ofs << "### Bottleneck Analysis (largest mesh: " << last.mesh_desc
          << ")\n\n";

      if (matvec_pct > 30) {
        ofs << "- **High Schwarz MatVec cost (" << std::fixed
            << std::setprecision(1) << matvec_pct
            << "%)**: Full sparse mat-vec per Schwarz iteration. "
            << "Consider: batched updates, element-local residual "
               "accumulation.\n";
      }
      if (scatter_pct > 30) {
        ofs << "- **High Scatter+Update cost (" << std::fixed
            << std::setprecision(1) << scatter_pct
            << "%)**: Column-wise sparse iteration for Gauss-Seidel updates. "
            << "Consider: colored ordering, batched element updates.\n";
      }

      double smoothing_pct =
          100.0 *
          (last.mg_vcycle_pre_smooth_ms + last.mg_vcycle_post_smooth_ms) /
          std::max(0.001, last.mg_apply_total_ms);
      ofs << "- **Smoothing dominates V-cycle**: " << std::fixed
          << std::setprecision(1) << smoothing_pct << "% of apply time.\n";
    }
  }

  ofs.close();
  std::cout << "\n=== Report written to: " << output_path << " ===\n";

  // Basic verification
  for (const auto &r : results) {
    EXPECT_GT(r.total_solve_ms, 0.0) << "No solve time for " << r.mesh_desc;
    EXPECT_GT(r.cg_iterations, 0) << "No CG iterations for " << r.mesh_desc;
  }
}

TEST_F(CGCubicBezierSmootherTest, DISABLED_AdditiveSchwarzComparisonBenchmark) {
  // Compare Multiplicative vs Additive Schwarz smoothers
  // Additive Schwarz eliminates the 45% scatter+update bottleneck but may
  // require more iterations to converge.

  struct ComparisonResult {
    std::string mesh_desc;
    Index num_elements;
    Index num_dofs;

    // Multiplicative Schwarz
    double mult_total_ms;
    int mult_cg_iters;
    int mult_vcycles;
    double mult_schwarz_total_ms;
    double mult_scatter_ms;

    // Additive Schwarz
    double add_total_ms;
    int add_cg_iters;
    int add_vcycles;
    double add_schwarz_total_ms;
    double add_scatter_ms;
  };

  auto bathy_func = [](Real x, Real y) {
    return 100.0 + 20.0 * std::sin(0.05 * x) * std::cos(0.05 * y) +
           10.0 * std::cos(0.08 * x) * std::sin(0.08 * y);
  };

  std::vector<int> sizes = {4, 8, 16, 32};
  std::vector<ComparisonResult> results;
  Real lambda = 10.0;

  std::cout << "\n=== Additive vs Multiplicative Schwarz Comparison ===\n";

  for (int n : sizes) {
    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 100.0, 0.0, 100.0, n, n);

    ComparisonResult r;
    r.mesh_desc = std::to_string(n) + "x" + std::to_string(n);
    r.num_elements = mesh.num_elements();

    // Test Multiplicative Schwarz
    {
      CGCubicBezierSmootherConfig config;
      config.lambda = lambda;
      config.edge_ngauss = 4;
      config.use_iterative_solver = true;
      config.use_multigrid = true;
      config.schur_cg_tolerance = 1e-10;
      config.multigrid_config.num_levels = 3;
      config.multigrid_config.pre_smoothing = 1;
      config.multigrid_config.post_smoothing = 1;
      config.multigrid_config.smoother_type =
          SmootherType::MultiplicativeSchwarz;

      MultigridProfile mg_profile;
      CGCubicSolveProfile solve_profile;
      solve_profile.multigrid_profile = &mg_profile;

      CGCubicBezierBathymetrySmoother smoother(mesh, config);
      smoother.set_solve_profile(&solve_profile);
      smoother.set_bathymetry_data(bathy_func);

      auto start = std::chrono::high_resolution_clock::now();
      smoother.solve();
      auto end = std::chrono::high_resolution_clock::now();

      r.num_dofs = smoother.num_free_dofs();
      r.mult_total_ms =
          std::chrono::duration<double, std::milli>(end - start).count();
      r.mult_cg_iters = solve_profile.outer_cg_iterations;
      r.mult_vcycles = mg_profile.vcycle_calls;
      r.mult_schwarz_total_ms = mg_profile.schwarz_matvec_ms +
                                mg_profile.schwarz_gather_ms +
                                mg_profile.schwarz_local_solve_ms +
                                mg_profile.schwarz_scatter_update_ms;
      r.mult_scatter_ms = mg_profile.schwarz_scatter_update_ms;
    }

    // Test Additive Schwarz
    {
      CGCubicBezierSmootherConfig config;
      config.lambda = lambda;
      config.edge_ngauss = 4;
      config.use_iterative_solver = true;
      config.use_multigrid = true;
      config.schur_cg_tolerance = 1e-10;
      config.multigrid_config.num_levels = 3;
      config.multigrid_config.pre_smoothing = 1;
      config.multigrid_config.post_smoothing = 1;
      config.multigrid_config.smoother_type = SmootherType::AdditiveSchwarz;

      MultigridProfile mg_profile;
      CGCubicSolveProfile solve_profile;
      solve_profile.multigrid_profile = &mg_profile;

      CGCubicBezierBathymetrySmoother smoother(mesh, config);
      smoother.set_solve_profile(&solve_profile);
      smoother.set_bathymetry_data(bathy_func);

      auto start = std::chrono::high_resolution_clock::now();
      smoother.solve();
      auto end = std::chrono::high_resolution_clock::now();

      r.add_total_ms =
          std::chrono::duration<double, std::milli>(end - start).count();
      r.add_cg_iters = solve_profile.outer_cg_iterations;
      r.add_vcycles = mg_profile.vcycle_calls;
      r.add_schwarz_total_ms = mg_profile.schwarz_matvec_ms +
                               mg_profile.schwarz_gather_ms +
                               mg_profile.schwarz_local_solve_ms +
                               mg_profile.schwarz_scatter_update_ms;
      r.add_scatter_ms = mg_profile.schwarz_scatter_update_ms;
    }

    results.push_back(r);

    double speedup = r.mult_total_ms / r.add_total_ms;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << r.mesh_desc << ": Mult=" << r.mult_total_ms
              << "ms (iters=" << r.mult_cg_iters << "), Add=" << r.add_total_ms
              << "ms (iters=" << r.add_cg_iters << "), Speedup=" << speedup
              << "x\n";
  }

  // Write markdown report
  std::string output_path = "/tmp/schwarz_comparison.md";
  std::ofstream ofs(output_path);

  ofs << "# Additive vs Multiplicative Schwarz Comparison\n\n";
  ofs << "Lambda: " << lambda << ", Tolerance: 1e-10\n\n";

  ofs << "## Summary\n\n";
  ofs << "| Mesh | DOFs | Mult (ms) | Add (ms) | Speedup | Mult Iters | Add "
         "Iters |\n";
  ofs << "|------|------|-----------|----------|---------|------------|------"
         "-----|\n";
  for (const auto &r : results) {
    double speedup = r.mult_total_ms / std::max(0.001, r.add_total_ms);
    ofs << "| " << r.mesh_desc << " | " << r.num_dofs << " | " << std::fixed
        << std::setprecision(2) << r.mult_total_ms << " | " << r.add_total_ms
        << " | " << std::setprecision(2) << speedup << "x | " << r.mult_cg_iters
        << " | " << r.add_cg_iters << " |\n";
  }

  ofs << "\n## Schwarz Smoother Timing\n\n";
  ofs << "| Mesh | Mult Schwarz (ms) | Mult Scatter (ms) | Add Schwarz (ms) "
         "| Add Scatter (ms) |\n";
  ofs << "|------|-------------------|-------------------|------------------|--"
         "-"
         "---------------|\n";
  for (const auto &r : results) {
    ofs << "| " << r.mesh_desc << " | " << std::fixed << std::setprecision(2)
        << r.mult_schwarz_total_ms << " | " << r.mult_scatter_ms << " | "
        << r.add_schwarz_total_ms << " | " << r.add_scatter_ms << " |\n";
  }

  ofs << "\n## Analysis\n\n";
  if (!results.empty()) {
    const auto &last = results.back();
    double speedup = last.mult_total_ms / std::max(0.001, last.add_total_ms);
    double scatter_reduction =
        100.0 *
        (1.0 - last.add_scatter_ms / std::max(0.001, last.mult_scatter_ms));

    ofs << "### Largest Mesh (" << last.mesh_desc << ")\n\n";
    ofs << "- **Speedup**: " << std::fixed << std::setprecision(2) << speedup
        << "x\n";
    ofs << "- **Scatter time reduction**: " << std::setprecision(1)
        << scatter_reduction << "%\n";
    ofs << "- **CG iteration increase**: "
        << (last.add_cg_iters - last.mult_cg_iters) << " iterations ("
        << std::setprecision(1)
        << 100.0 * (last.add_cg_iters - last.mult_cg_iters) /
               std::max(1, last.mult_cg_iters)
        << "% more)\n";

    if (speedup > 1.0) {
      ofs << "\n**Recommendation**: Additive Schwarz is faster despite "
             "requiring more iterations.\n";
    } else if (last.add_cg_iters > 2 * last.mult_cg_iters) {
      ofs << "\n**Recommendation**: Multiplicative Schwarz is faster due to "
             "better convergence.\n";
    }
  }

  ofs.close();
  std::cout << "\n=== Report written to: " << output_path << " ===\n";

  // Verify both methods converge
  for (const auto &r : results) {
    EXPECT_GT(r.mult_cg_iters, 0) << "Mult didn't converge for " << r.mesh_desc;
    EXPECT_GT(r.add_cg_iters, 0) << "Add didn't converge for " << r.mesh_desc;
  }
}

TEST_F(CGCubicBezierSmootherTest, DISABLED_ColoredSchwarzComparisonBenchmark) {
  // Compare all three Schwarz variants:
  // - Multiplicative: Sequential element processing with Qx updates
  // - Additive: Parallel element processing, single update at end
  // - Colored: Hybrid - parallel within colors, sequential between colors

  struct SchwarzResult {
    double total_ms;
    int cg_iters;
    double schwarz_ms;
    double scatter_ms;
  };

  struct ComparisonResult {
    std::string mesh_desc;
    Index num_dofs;
    SchwarzResult mult;
    SchwarzResult add;
    SchwarzResult colored;
  };

  auto bathy_func = [](Real x, Real y) {
    return 100.0 + 20.0 * std::sin(0.05 * x) * std::cos(0.05 * y) +
           10.0 * std::cos(0.08 * x) * std::sin(0.08 * y);
  };

  auto run_test = [&](const QuadtreeAdapter &mesh,
                      SmootherType type) -> SchwarzResult {
    CGCubicBezierSmootherConfig config;
    config.lambda = 10.0;
    config.edge_ngauss = 4;
    config.use_iterative_solver = true;
    config.use_multigrid = true;
    config.schur_cg_tolerance = 1e-10;
    config.multigrid_config.num_levels = 3;
    config.multigrid_config.pre_smoothing = 1;
    config.multigrid_config.post_smoothing = 1;
    config.multigrid_config.smoother_type = type;

    MultigridProfile mg_profile;
    CGCubicSolveProfile solve_profile;
    solve_profile.multigrid_profile = &mg_profile;

    CGCubicBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_solve_profile(&solve_profile);
    smoother.set_bathymetry_data(bathy_func);

    auto start = std::chrono::high_resolution_clock::now();
    smoother.solve();
    auto end = std::chrono::high_resolution_clock::now();

    SchwarzResult r;
    r.total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    r.cg_iters = solve_profile.outer_cg_iterations;
    r.schwarz_ms = mg_profile.schwarz_matvec_ms + mg_profile.schwarz_gather_ms +
                   mg_profile.schwarz_local_solve_ms +
                   mg_profile.schwarz_scatter_update_ms;
    r.scatter_ms = mg_profile.schwarz_scatter_update_ms;
    return r;
  };

  std::vector<int> sizes = {4, 8, 16, 32};
  std::vector<ComparisonResult> results;

  std::cout << "\n=== Schwarz Smoother Comparison (Mult vs Add vs Colored) "
               "===\n";

  for (int n : sizes) {
    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 100.0, 0.0, 100.0, n, n);

    ComparisonResult r;
    r.mesh_desc = std::to_string(n) + "x" + std::to_string(n);

    // Run all three variants
    r.mult = run_test(mesh, SmootherType::MultiplicativeSchwarz);
    r.num_dofs = mesh.num_elements() > 0 ? r.mult.cg_iters : 0; // placeholder
    r.add = run_test(mesh, SmootherType::AdditiveSchwarz);
    r.colored = run_test(mesh, SmootherType::ColoredMultiplicativeSchwarz);

    // Get actual DOF count
    CGCubicBezierBathymetrySmoother tmp(mesh);
    r.num_dofs = tmp.num_free_dofs();

    results.push_back(r);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << r.mesh_desc << ": Mult=" << r.mult.total_ms
              << "ms (iters=" << r.mult.cg_iters
              << "), Colored=" << r.colored.total_ms
              << "ms (iters=" << r.colored.cg_iters
              << "), Speedup=" << r.mult.total_ms / r.colored.total_ms << "x\n";
  }

  // Write markdown report
  std::string output_path = "/tmp/schwarz_all_comparison.md";
  std::ofstream ofs(output_path);

  ofs << "# Schwarz Smoother Comparison: Mult vs Add vs Colored\n\n";
  ofs << "Lambda: 10, Tolerance: 1e-10\n\n";

  ofs << "## Summary\n\n";
  ofs << "| Mesh | DOFs | Mult (ms) | Colored (ms) | Add (ms) | Mult Iters | "
         "Colored Iters | Add Iters | Colored Speedup |\n";
  ofs << "|------|------|-----------|--------------|----------|------------|--"
         "-------------|-----------|----------------|\n";
  for (const auto &r : results) {
    double speedup = r.mult.total_ms / std::max(0.001, r.colored.total_ms);
    ofs << "| " << r.mesh_desc << " | " << r.num_dofs << " | " << std::fixed
        << std::setprecision(2) << r.mult.total_ms << " | "
        << r.colored.total_ms << " | " << r.add.total_ms << " | "
        << r.mult.cg_iters << " | " << r.colored.cg_iters << " | "
        << r.add.cg_iters << " | " << std::setprecision(2) << speedup
        << "x |\n";
  }

  ofs << "\n## Schwarz Timing Breakdown\n\n";
  ofs << "| Mesh | Mult Schwarz (ms) | Mult Scatter (ms) | Colored Schwarz "
         "(ms) | Colored Scatter (ms) |\n";
  ofs << "|------|-------------------|-------------------|--------------------"
         "-|---------------------|\n";
  for (const auto &r : results) {
    ofs << "| " << r.mesh_desc << " | " << std::fixed << std::setprecision(2)
        << r.mult.schwarz_ms << " | " << r.mult.scatter_ms << " | "
        << r.colored.schwarz_ms << " | " << r.colored.scatter_ms << " |\n";
  }

  ofs << "\n## Analysis\n\n";
  if (!results.empty()) {
    const auto &last = results.back();
    double colored_speedup =
        last.mult.total_ms / std::max(0.001, last.colored.total_ms);
    double scatter_reduction =
        100.0 *
        (1.0 - last.colored.scatter_ms / std::max(0.001, last.mult.scatter_ms));
    int iter_increase = last.colored.cg_iters - last.mult.cg_iters;

    ofs << "### Largest Mesh (" << last.mesh_desc << ")\n\n";
    ofs << "- **Colored vs Mult Speedup**: " << std::fixed
        << std::setprecision(2) << colored_speedup << "x\n";
    ofs << "- **Scatter time reduction**: " << std::setprecision(1)
        << scatter_reduction << "%\n";
    ofs << "- **CG iteration change**: " << iter_increase << " iterations ("
        << std::setprecision(1)
        << 100.0 * iter_increase / std::max(1, last.mult.cg_iters) << "% "
        << (iter_increase >= 0 ? "more" : "fewer") << ")\n";

    if (colored_speedup > 1.1) {
      ofs << "\n**Recommendation**: Colored Schwarz offers the best "
             "balance of convergence and efficiency.\n";
    } else if (colored_speedup < 0.9) {
      ofs << "\n**Recommendation**: Standard multiplicative Schwarz is "
             "faster for this problem.\n";
    } else {
      ofs << "\n**Recommendation**: Colored and multiplicative Schwarz have "
             "similar performance.\n";
    }
  }

  ofs.close();
  std::cout << "\n=== Report written to: " << output_path << " ===\n";

  // Verify all methods converge
  for (const auto &r : results) {
    EXPECT_GT(r.mult.cg_iters, 0) << "Mult didn't converge for " << r.mesh_desc;
    EXPECT_GT(r.colored.cg_iters, 0)
        << "Colored didn't converge for " << r.mesh_desc;
  }
}
