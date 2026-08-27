#include "bathymetry/cg_cubic_bezier_dof_manager.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "bathymetry/static_condensation.hpp"
#include "mesh/hilbert.hpp"
#include <gtest/gtest.h>
#include <set>

using namespace drifter;

// =============================================================================
// Hilbert Curve Tests
// =============================================================================

TEST(Hilbert2D, RoundTripEncoding) {
    // Test that encode/decode are inverses
    for (int order = 1; order <= 10; ++order) {
        int grid_size = 1 << order;
        // Test corners and some interior points
        std::vector<std::pair<uint32_t, uint32_t>> test_points = {
            {0, 0},
            {0, static_cast<uint32_t>(grid_size - 1)},
            {static_cast<uint32_t>(grid_size - 1), 0},
            {static_cast<uint32_t>(grid_size - 1),
             static_cast<uint32_t>(grid_size - 1)},
            {static_cast<uint32_t>(grid_size / 2),
             static_cast<uint32_t>(grid_size / 2)}};

        for (const auto &[x, y] : test_points) {
            uint64_t h = Hilbert2D::encode(x, y, order);
            uint32_t x2, y2;
            Hilbert2D::decode(h, order, x2, y2);
            EXPECT_EQ(x, x2) << "order=" << order << ", x=" << x;
            EXPECT_EQ(y, y2) << "order=" << order << ", y=" << y;
        }
    }
}

TEST(Hilbert2D, MonotonicAlongCurve) {
    // Hilbert indices should be unique for each grid point
    constexpr int order = 5;
    constexpr int grid_size = 1 << order;
    std::set<uint64_t> seen;

    for (uint32_t x = 0; x < grid_size; ++x) {
        for (uint32_t y = 0; y < grid_size; ++y) {
            uint64_t h = Hilbert2D::encode(x, y, order);
            EXPECT_TRUE(seen.insert(h).second)
                << "Duplicate Hilbert index for (" << x << ", " << y << ")";
        }
    }

    // Should have exactly grid_size^2 unique indices
    EXPECT_EQ(seen.size(), static_cast<size_t>(grid_size * grid_size));
}

// =============================================================================
// Static Condensation Tests
// =============================================================================

TEST(StaticCondensation, SkeletonInteriorPartition) {
    // Verify the skeleton/interior index partitions are correct
    StaticCondensationManager mgr;

    // Skeleton should be corners (0,3,12,15) + edge DOFs
    std::set<int> skeleton_set(mgr.SKELETON_INDICES.begin(),
                                mgr.SKELETON_INDICES.end());
    std::set<int> interior_set(mgr.INTERIOR_INDICES.begin(),
                                mgr.INTERIOR_INDICES.end());

    EXPECT_EQ(skeleton_set.size(), 12u);
    EXPECT_EQ(interior_set.size(), 4u);

    // Check interior DOFs are correct (2x2 center: i,j in {1,2})
    // local_dof = i + 4*j, so interior = {5, 6, 9, 10}
    EXPECT_TRUE(interior_set.count(5));  // (1,1)
    EXPECT_TRUE(interior_set.count(6));  // (2,1)
    EXPECT_TRUE(interior_set.count(9));  // (1,2)
    EXPECT_TRUE(interior_set.count(10)); // (2,2)

    // No overlap between skeleton and interior
    for (int s : mgr.SKELETON_INDICES) {
        EXPECT_FALSE(interior_set.count(s));
    }
    for (int i : mgr.INTERIOR_INDICES) {
        EXPECT_FALSE(skeleton_set.count(i));
    }
}

TEST(StaticCondensation, CondenseIdentityMatrix) {
    // For identity matrix, Schur complement should be identity on skeleton
    StaticCondensationManager mgr;

    MatX K = MatX::Identity(16, 16);
    MatX S = mgr.condense(K);

    EXPECT_EQ(S.rows(), 12);
    EXPECT_EQ(S.cols(), 12);

    // S = K_SS - K_SI * K_II^{-1} * K_IS = I - 0 * I * 0 = I
    MatX expected = MatX::Identity(12, 12);
    EXPECT_TRUE(S.isApprox(expected, 1e-12));
}

TEST(StaticCondensation, CondenseSymmetricMatrix) {
    // Verify Schur complement of SPD matrix is SPD
    StaticCondensationManager mgr;

    // Create a simple SPD matrix (diagonal + small off-diagonal)
    MatX K = MatX::Identity(16, 16) * 10.0;
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            if (i != j) {
                K(i, j) = 0.1;
            }
        }
    }
    // Make it symmetric
    K = 0.5 * (K + K.transpose());

    MatX S = mgr.condense(K);

    // Check symmetry
    EXPECT_TRUE(S.isApprox(S.transpose(), 1e-12));

    // Check positive definiteness via Cholesky
    Eigen::LLT<MatX> llt(S);
    EXPECT_EQ(llt.info(), Eigen::Success);
}

TEST(StaticCondensation, RecoverInteriorDOFs) {
    StaticCondensationManager mgr;

    // Create a test matrix and vector
    MatX K = MatX::Identity(16, 16) * 2.0;
    VecX x_full = VecX::LinSpaced(16, 1.0, 16.0);

    // Condense
    mgr.condense(K);

    // Extract skeleton values
    VecX x_skeleton = StaticCondensationManager::extract_skeleton(x_full);
    EXPECT_EQ(x_skeleton.size(), 12);

    // For identity-like matrix, recovery should give back interior values
    // (approximately, since x_I = -K_II^{-1} * K_IS * x_S when f_I=0)
    VecX x_interior_recovered = mgr.recover_interior(x_skeleton);
    EXPECT_EQ(x_interior_recovered.size(), 4);
}

TEST(StaticCondensation, AssembleFullVector) {
    VecX x_skeleton = VecX::LinSpaced(12, 1.0, 12.0);
    VecX x_interior = VecX::LinSpaced(4, 100.0, 103.0);

    VecX x_full = StaticCondensationManager::assemble_full(x_skeleton, x_interior);
    EXPECT_EQ(x_full.size(), 16);

    // Check that values are in correct positions
    VecX x_skel_back = StaticCondensationManager::extract_skeleton(x_full);
    VecX x_int_back = StaticCondensationManager::extract_interior(x_full);

    EXPECT_TRUE(x_skel_back.isApprox(x_skeleton));
    EXPECT_TRUE(x_int_back.isApprox(x_interior));
}

// =============================================================================
// Hierarchical DOF Ordering Tests
// =============================================================================

class HierarchicalOrderingTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Create a simple 2x2 uniform quadtree mesh
        quadtree_ = std::make_unique<QuadtreeAdapter>();
        quadtree_->build_uniform(0.0, 1.0, 0.0, 1.0, 2, 2);
    }

    std::unique_ptr<QuadtreeAdapter> quadtree_;
};

TEST_F(HierarchicalOrderingTest, MortonVsHierarchicalDOFCount) {
    // Both orderings should produce the same number of DOFs
    CGCubicBezierDofManager morton_mgr(*quadtree_, false);
    CGCubicBezierDofManager hier_mgr(*quadtree_, true);

    EXPECT_EQ(morton_mgr.num_global_dofs(), hier_mgr.num_global_dofs());
    EXPECT_EQ(morton_mgr.num_free_dofs(), hier_mgr.num_free_dofs());
}

TEST_F(HierarchicalOrderingTest, HierarchicalOrderingIsValid) {
    CGCubicBezierDofManager hier_mgr(*quadtree_, true);

    EXPECT_TRUE(hier_mgr.has_hierarchical_ordering());

    const auto &ordering = hier_mgr.hierarchical_ordering();
    EXPECT_FALSE(ordering.permutation.empty());
    EXPECT_FALSE(ordering.inverse.empty());
    EXPECT_EQ(ordering.permutation.size(), ordering.inverse.size());
}

TEST_F(HierarchicalOrderingTest, PermutationIsValid) {
    CGCubicBezierDofManager hier_mgr(*quadtree_, true);
    const auto &ordering = hier_mgr.hierarchical_ordering();

    Index n = hier_mgr.num_global_dofs();

    // Check permutation is a valid bijection
    std::vector<bool> seen(n, false);
    for (Index old_idx = 0; old_idx < n; ++old_idx) {
        Index new_idx = ordering.permutation[old_idx];
        EXPECT_GE(new_idx, 0);
        EXPECT_LT(new_idx, n);
        EXPECT_FALSE(seen[new_idx]) << "Duplicate in permutation at " << new_idx;
        seen[new_idx] = true;
    }

    // Check inverse is correct
    for (Index old_idx = 0; old_idx < n; ++old_idx) {
        Index new_idx = ordering.permutation[old_idx];
        EXPECT_EQ(ordering.inverse[new_idx], old_idx);
    }
}

TEST_F(HierarchicalOrderingTest, LevelBlocksAreContiguous) {
    CGCubicBezierDofManager hier_mgr(*quadtree_, true);
    const auto &ordering = hier_mgr.hierarchical_ordering();

    // Verify levels are monotonically non-decreasing
    int prev_level = -1;
    for (Index i = 0; i < hier_mgr.num_global_dofs(); ++i) {
        int level = ordering.metadata[i].level;
        EXPECT_GE(level, prev_level) << "Level decreased at DOF " << i;
        prev_level = level;
    }

    // Verify level blocks cover all DOFs
    Index total = 0;
    for (const auto &[level, block] : ordering.level_blocks) {
        EXPECT_GE(block.first, 0);
        EXPECT_LE(block.second, hier_mgr.num_global_dofs());
        EXPECT_LT(block.first, block.second);
        total += (block.second - block.first);
    }
    EXPECT_EQ(total, hier_mgr.num_global_dofs());
}

TEST_F(HierarchicalOrderingTest, SkeletonDOFCount) {
    CGCubicBezierDofManager hier_mgr(*quadtree_, true);
    const auto &ordering = hier_mgr.hierarchical_ordering();

    // Count skeleton DOFs from metadata
    Index skeleton_count = 0;
    for (const auto &meta : ordering.metadata) {
        if (meta.type != DOFType::Interior) {
            ++skeleton_count;
        }
    }

    EXPECT_EQ(ordering.num_skeleton_dofs, skeleton_count);
    EXPECT_GT(ordering.num_skeleton_dofs, 0);
    // Skeleton should be less than total (some DOFs are interior)
    EXPECT_LT(ordering.num_skeleton_dofs, hier_mgr.num_global_dofs());
}

// =============================================================================
// Static Condensation Integration Tests (full solve)
// =============================================================================

#include "bathymetry/cg_cubic_bezier_bathymetry_smoother.hpp"

class StaticCondensationSolveTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Create a 2x2 uniform mesh
        quadtree_ = std::make_unique<QuadtreeAdapter>();
        quadtree_->build_uniform(0.0, 10.0, 0.0, 10.0, 2, 2);
    }

    std::unique_ptr<QuadtreeAdapter> quadtree_;
};

TEST_F(StaticCondensationSolveTest, StandardVsCondensedSolveConstant) {
    // Test that standard and condensed solve produce same solution for constant function
    // Use a single element to avoid complications from DOF sharing
    QuadtreeAdapter single_mesh;
    single_mesh.build_uniform(0.0, 10.0, 0.0, 10.0, 1, 1);

    auto bathy_func = [](Real, Real) { return 5.0; };

    // Standard solve (no condensation)
    CGCubicBezierSmootherConfig config_std;
    config_std.lambda = 10.0;
    config_std.use_static_condensation = false;
    config_std.use_hierarchical_ordering = false;

    CGCubicBezierBathymetrySmoother smoother_std(single_mesh, config_std);
    smoother_std.set_bathymetry_data(bathy_func);
    smoother_std.solve();

    // Condensed solve
    CGCubicBezierSmootherConfig config_cond;
    config_cond.lambda = 10.0;
    config_cond.use_static_condensation = true;
    config_cond.use_hierarchical_ordering = false;

    CGCubicBezierBathymetrySmoother smoother_cond(single_mesh, config_cond);
    smoother_cond.set_bathymetry_data(bathy_func);
    smoother_cond.solve();

    // Solutions should match closely
    const VecX &sol_std = smoother_std.solution();
    const VecX &sol_cond = smoother_cond.solution();

    EXPECT_EQ(sol_std.size(), sol_cond.size());

    // Check solution values at test points
    std::vector<std::pair<Real, Real>> test_points = {
        {5.0, 5.0}, {2.0, 3.0}, {8.0, 7.0}, {1.0, 9.0}
    };

    for (const auto &[x, y] : test_points) {
        Real val_std = smoother_std.evaluate(x, y);
        Real val_cond = smoother_cond.evaluate(x, y);
        EXPECT_NEAR(val_std, val_cond, 1e-8)
            << "Mismatch at (" << x << ", " << y << ")";
    }
}

TEST_F(StaticCondensationSolveTest, StandardVsCondensedSolveQuadratic) {
    // Test that standard and condensed solve produce same solution for quadratic function
    // Start with single element to verify basic correctness
    QuadtreeAdapter single_mesh;
    single_mesh.build_uniform(0.0, 10.0, 0.0, 10.0, 1, 1);

    auto bathy_func = [](Real x, Real y) { return 1.0 + 0.1 * x + 0.05 * y + 0.01 * x * y; };

    // Standard solve
    CGCubicBezierSmootherConfig config_std;
    config_std.lambda = 10.0;
    config_std.use_static_condensation = false;

    CGCubicBezierBathymetrySmoother smoother_std(single_mesh, config_std);
    smoother_std.set_bathymetry_data(bathy_func);
    smoother_std.solve();

    // Condensed solve
    CGCubicBezierSmootherConfig config_cond;
    config_cond.lambda = 10.0;
    config_cond.use_static_condensation = true;

    CGCubicBezierBathymetrySmoother smoother_cond(single_mesh, config_cond);
    smoother_cond.set_bathymetry_data(bathy_func);
    smoother_cond.solve();

    // Check solution values at test points
    std::vector<std::pair<Real, Real>> test_points = {
        {5.0, 5.0}, {2.0, 3.0}, {8.0, 7.0}, {1.0, 9.0}
    };

    for (const auto &[x, y] : test_points) {
        Real val_std = smoother_std.evaluate(x, y);
        Real val_cond = smoother_cond.evaluate(x, y);
        EXPECT_NEAR(val_std, val_cond, 1e-8)
            << "Mismatch at (" << x << ", " << y << ")";
    }
}

TEST_F(StaticCondensationSolveTest, CondensedSolveWithHierarchicalOrdering) {
    // Test static condensation combined with hierarchical ordering
    // Use single element to verify basic correctness
    QuadtreeAdapter single_mesh;
    single_mesh.build_uniform(0.0, 10.0, 0.0, 10.0, 1, 1);

    auto bathy_func = [](Real x, Real y) { return std::sin(x * 0.3) * std::cos(y * 0.3); };

    // Standard solve
    CGCubicBezierSmootherConfig config_std;
    config_std.lambda = 10.0;
    config_std.use_static_condensation = false;
    config_std.use_hierarchical_ordering = false;

    CGCubicBezierBathymetrySmoother smoother_std(single_mesh, config_std);
    smoother_std.set_bathymetry_data(bathy_func);
    smoother_std.solve();

    // Condensed solve with hierarchical ordering
    CGCubicBezierSmootherConfig config_hier;
    config_hier.lambda = 10.0;
    config_hier.use_static_condensation = true;
    config_hier.use_hierarchical_ordering = true;

    CGCubicBezierBathymetrySmoother smoother_hier(single_mesh, config_hier);
    smoother_hier.set_bathymetry_data(bathy_func);
    smoother_hier.solve();

    // Check solution values at test points
    std::vector<std::pair<Real, Real>> test_points = {
        {5.0, 5.0}, {2.0, 3.0}, {8.0, 7.0}, {1.0, 9.0}
    };

    for (const auto &[x, y] : test_points) {
        Real val_std = smoother_std.evaluate(x, y);
        Real val_hier = smoother_hier.evaluate(x, y);
        EXPECT_NEAR(val_std, val_hier, 1e-8)
            << "Mismatch at (" << x << ", " << y << ")";
    }
}

// =============================================================================
// Multi-element Tests (static condensation disabled for multi-element)
// =============================================================================

// Note: Static condensation is only supported for single-element meshes.
// For multi-element meshes with C¹ edge constraints, the constraint transformation
// required to eliminate interior DOFs is complex and not yet implemented.
// These tests verify that multi-element meshes work correctly WITHOUT static condensation,
// and that enabling the flag doesn't break the solve (it's automatically disabled).

TEST_F(StaticCondensationSolveTest, MultiElementFallbackConstant) {
    // Verify that enabling static_condensation for multi-element meshes
    // falls back to standard solve and produces correct results
    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 10.0, 0.0, 10.0, 2, 2);

    auto bathy_func = [](Real /*x*/, Real /*y*/) { return 5.0; };

    // Standard solve (explicitly disabled)
    CGCubicBezierSmootherConfig config_std;
    config_std.lambda = 10.0;
    config_std.use_static_condensation = false;

    CGCubicBezierBathymetrySmoother smoother_std(mesh, config_std);
    smoother_std.set_bathymetry_data(bathy_func);
    smoother_std.solve();

    // With condensation enabled (should fall back to standard for multi-element)
    CGCubicBezierSmootherConfig config_cond;
    config_cond.lambda = 10.0;
    config_cond.use_static_condensation = true;

    CGCubicBezierBathymetrySmoother smoother_cond(mesh, config_cond);
    smoother_cond.set_bathymetry_data(bathy_func);
    smoother_cond.solve();

    // Results should match since condensation is disabled for multi-element
    std::vector<std::pair<Real, Real>> test_points = {
        {2.5, 2.5}, {7.5, 2.5}, {2.5, 7.5}, {7.5, 7.5},
        {5.0, 5.0}, {5.0, 2.5}, {2.5, 5.0},
    };

    for (const auto &[x, y] : test_points) {
        Real val_std = smoother_std.evaluate(x, y);
        Real val_cond = smoother_cond.evaluate(x, y);
        EXPECT_NEAR(val_std, val_cond, 1e-10)
            << "Mismatch at (" << x << ", " << y << ")";
    }
}

TEST_F(StaticCondensationSolveTest, MultiElementFallbackQuadratic) {
    // Verify multi-element fallback with quadratic function
    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 10.0, 0.0, 10.0, 2, 2);

    auto bathy_func = [](Real x, Real y) {
        return 1.0 + 0.1 * x + 0.2 * y + 0.01 * x * x + 0.02 * y * y;
    };

    CGCubicBezierSmootherConfig config_std;
    config_std.lambda = 10.0;
    config_std.use_static_condensation = false;

    CGCubicBezierBathymetrySmoother smoother_std(mesh, config_std);
    smoother_std.set_bathymetry_data(bathy_func);
    smoother_std.solve();

    CGCubicBezierSmootherConfig config_cond;
    config_cond.lambda = 10.0;
    config_cond.use_static_condensation = true;

    CGCubicBezierBathymetrySmoother smoother_cond(mesh, config_cond);
    smoother_cond.set_bathymetry_data(bathy_func);
    smoother_cond.solve();

    std::vector<std::pair<Real, Real>> test_points = {
        {2.5, 2.5}, {7.5, 2.5}, {2.5, 7.5}, {7.5, 7.5},
        {5.0, 5.0}, {5.0, 2.5}, {2.5, 5.0},
    };

    for (const auto &[x, y] : test_points) {
        Real val_std = smoother_std.evaluate(x, y);
        Real val_cond = smoother_cond.evaluate(x, y);
        EXPECT_NEAR(val_std, val_cond, 1e-10)
            << "Mismatch at (" << x << ", " << y << ")";
    }
}

TEST_F(StaticCondensationSolveTest, MultiElementFallbackSinusoidal) {
    // Verify multi-element fallback with sinusoidal function
    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 10.0, 0.0, 10.0, 2, 2);

    auto bathy_func = [](Real x, Real y) { return std::sin(x * 0.3) * std::cos(y * 0.3); };

    CGCubicBezierSmootherConfig config_std;
    config_std.lambda = 10.0;
    config_std.edge_ngauss = 3;
    config_std.use_static_condensation = false;

    CGCubicBezierBathymetrySmoother smoother_std(mesh, config_std);
    smoother_std.set_bathymetry_data(bathy_func);
    smoother_std.solve();

    CGCubicBezierSmootherConfig config_cond;
    config_cond.lambda = 10.0;
    config_cond.edge_ngauss = 3;
    config_cond.use_static_condensation = true;

    CGCubicBezierBathymetrySmoother smoother_cond(mesh, config_cond);
    smoother_cond.set_bathymetry_data(bathy_func);
    smoother_cond.solve();

    std::vector<std::pair<Real, Real>> test_points = {
        {2.5, 2.5}, {7.5, 2.5}, {2.5, 7.5}, {7.5, 7.5},
        {5.0, 5.0}, {5.0, 2.5}, {2.5, 5.0},
    };

    for (const auto &[x, y] : test_points) {
        Real val_std = smoother_std.evaluate(x, y);
        Real val_cond = smoother_cond.evaluate(x, y);
        EXPECT_NEAR(val_std, val_cond, 1e-10)
            << "Mismatch at (" << x << ", " << y << ")";
    }
}
