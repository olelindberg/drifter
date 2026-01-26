#include <gtest/gtest.h>
#include "bathymetry/bezier_hierarchical_solver.hpp"
#include "bathymetry/bezier_bathymetry_smoother.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include <cmath>

using namespace drifter;

class HierarchicalSolverTest : public ::testing::Test {
protected:
  static constexpr Real TOLERANCE = 1e-10;
  static constexpr Real LOOSE_TOLERANCE = 1e-6;

  // Create a uniform quadtree mesh
  std::unique_ptr<QuadtreeAdapter> create_uniform_quadtree(int nx, int ny,
                                                            Real xmin = 0.0,
                                                            Real xmax = 1.0,
                                                            Real ymin = 0.0,
                                                            Real ymax = 1.0) {
    auto quadtree = std::make_unique<QuadtreeAdapter>();
    quadtree->build_uniform(xmin, xmax, ymin, ymax, nx, ny);
    return quadtree;
  }

  // Create a non-conforming 1+4 mesh (1 coarse + 4 fine elements)
  std::unique_ptr<QuadtreeAdapter> create_one_plus_four_mesh() {
    auto quadtree = std::make_unique<QuadtreeAdapter>();

    // Coarse element: [0, 0.5] x [0, 1]
    quadtree->add_element(QuadBounds{0.0, 0.5, 0.0, 1.0}, QuadLevel{0, 0});

    // Fine elements: 2x2 in [0.5, 1] x [0, 1]
    quadtree->add_element(QuadBounds{0.5, 0.75, 0.0, 0.5}, QuadLevel{1, 1});
    quadtree->add_element(QuadBounds{0.75, 1.0, 0.0, 0.5}, QuadLevel{1, 1});
    quadtree->add_element(QuadBounds{0.5, 0.75, 0.5, 1.0}, QuadLevel{1, 1});
    quadtree->add_element(QuadBounds{0.75, 1.0, 0.5, 1.0}, QuadLevel{1, 1});

    return quadtree;
  }
};

// =============================================================================
// Laplacian Computation Tests (using hierarchical solver)
// =============================================================================

TEST_F(HierarchicalSolverTest, LaplacianOfConstant) {
  // Laplacian of a constant field should be zero
  auto quadtree = create_uniform_quadtree(2, 2);

  BezierSmootherConfig config;
  config.lambda = 10.0; // Stronger data fitting
  config.use_hierarchical = true;
  config.enable_natural_bc = false; // Hierarchical solver doesn't support natural BC yet
  BezierBathymetrySmoother smoother(*quadtree, config);

  // Set constant bathymetry
  smoother.set_bathymetry_data([](Real, Real) { return 10.0; });
  smoother.solve();

  // Check that solution is approximately constant
  // Hierarchical solver may have slightly larger error due to level-by-level solving
  EXPECT_NEAR(smoother.evaluate(0.3, 0.3), 10.0, 0.2);
  EXPECT_NEAR(smoother.evaluate(0.7, 0.7), 10.0, 0.2);
}

TEST_F(HierarchicalSolverTest, LaplacianOfLinear) {
  // Laplacian of a linear field should be zero
  auto quadtree = create_uniform_quadtree(2, 2);

  BezierSmootherConfig config;
  config.lambda = 10.0;
  config.use_hierarchical = true;
  config.enable_natural_bc = false; // Hierarchical solver doesn't support natural BC yet
  BezierBathymetrySmoother smoother(*quadtree, config);

  // Set linear bathymetry: z = 2x + 3y + 5
  auto bathy_func = [](Real x, Real y) { return 2.0 * x + 3.0 * y + 5.0; };
  smoother.set_bathymetry_data(bathy_func);
  smoother.solve();

  // Check that linear is reproduced (with relaxed tolerance)
  EXPECT_NEAR(smoother.evaluate(0.3, 0.4), bathy_func(0.3, 0.4), 0.2);
  EXPECT_NEAR(smoother.evaluate(0.7, 0.6), bathy_func(0.7, 0.6), 0.2);
}

TEST_F(HierarchicalSolverTest, LaplacianOfQuadratic) {
  // Quadratic should be well approximated
  auto quadtree = create_uniform_quadtree(2, 2);

  BezierSmootherConfig config;
  config.lambda = 100.0; // Very strong data fitting
  config.use_hierarchical = true;
  BezierBathymetrySmoother smoother(*quadtree, config);

  auto bathy_func = [](Real x, Real y) { return x * x + y * y; };
  smoother.set_bathymetry_data(bathy_func);
  smoother.solve();

  // Check approximation quality
  EXPECT_NEAR(smoother.evaluate(0.5, 0.5), bathy_func(0.5, 0.5), 0.1);
}

