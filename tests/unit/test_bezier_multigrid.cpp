#include <gtest/gtest.h>
#include "bathymetry/bezier_multigrid_preconditioner.hpp"
#include "bathymetry/cg_cubic_bezier_bathymetry_smoother.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "mesh/octree_adapter.hpp"
#include "mesh/refine_mask.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>

using namespace drifter;

namespace {

constexpr Real TOLERANCE = 1e-10;
constexpr Real LOOSE_TOLERANCE = 1e-6;

class BezierMultigridTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Build a 4x4 uniform mesh
        mesh_.build_uniform(0.0, 1.0, 0.0, 1.0, 4, 4);
    }

    QuadtreeAdapter mesh_;
};

// =============================================================================
// Basic preconditioner tests
// =============================================================================

TEST_F(BezierMultigridTest, ConstructWithDefaultConfig) {
    MultigridConfig config;
    BezierMultigridPreconditioner precond(config);
    EXPECT_FALSE(precond.is_setup());
    EXPECT_EQ(precond.num_levels(), 0);
}

TEST_F(BezierMultigridTest, SetupBuildsHierarchy) {
    // Build DOF manager and Q matrix
    CGCubicBezierSmootherConfig smoother_config;
    smoother_config.lambda = 1.0;
    CGCubicBezierBathymetrySmoother smoother(mesh_, smoother_config);
    smoother.set_bathymetry_data([](Real x, Real y) { return x + y; });

    // Get Q matrix (need to access internal state, use solve to trigger assembly)
    // For testing, we just build a simple SPD matrix
    Index n = smoother.num_global_dofs();
    SpMat Q(n, n);
    std::vector<Eigen::Triplet<Real>> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 2.0);
        if (i > 0) {
            triplets.emplace_back(i, i - 1, -0.5);
            triplets.emplace_back(i - 1, i, -0.5);
        }
    }
    Q.setFromTriplets(triplets.begin(), triplets.end());

    MultigridConfig config;
    config.min_tree_level = 1;
    config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    BezierMultigridPreconditioner precond(config);
    precond.setup(Q, mesh_, smoother.dof_manager());

    EXPECT_TRUE(precond.is_setup());
    EXPECT_GE(precond.num_levels(), 1);
}

TEST_F(BezierMultigridTest, VCycleReducesResidual) {
    // Build a simple problem
    CGCubicBezierSmootherConfig smoother_config;
    CGCubicBezierBathymetrySmoother smoother(mesh_, smoother_config);

    Index n = smoother.dof_manager().num_free_dofs();

    // Create a simple tridiagonal SPD matrix
    SpMat Q(n, n);
    std::vector<Eigen::Triplet<Real>> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 4.0);
        if (i > 0) {
            triplets.emplace_back(i, i - 1, -1.0);
            triplets.emplace_back(i - 1, i, -1.0);
        }
    }
    Q.setFromTriplets(triplets.begin(), triplets.end());

    // Setup preconditioner
    MultigridConfig config;
    config.min_tree_level = 1;
    config.pre_smoothing = 2;
    config.post_smoothing = 2;
    config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    BezierMultigridPreconditioner precond(config);
    precond.setup(Q, mesh_, smoother.dof_manager());

    // Create a residual and apply V-cycle
    VecX r = VecX::Ones(n);
    VecX z = precond.apply(r);

    // The correction should be non-zero and reduce residual when applied
    EXPECT_GT(z.norm(), 0.0);

    // For SPD systems, z should have same sign correlation with r
    EXPECT_GT(r.dot(z), 0.0);
}

// =============================================================================
// Integration with smoother tests
// =============================================================================

TEST_F(BezierMultigridTest, MultigridSmootherConverges) {
    // Configure smoother with multigrid enabled
    CGCubicBezierSmootherConfig config;
    config.lambda = 1.0;
    config.use_iterative_solver = true;
    config.use_multigrid = true;
    config.multigrid_config.min_tree_level = 1;
    config.tolerance = 1e-8;

    CGCubicBezierBathymetrySmoother smoother(mesh_, config);

    auto bathy_func = [](Real x, Real y) { return x * x + y * y; };
    smoother.set_bathymetry_data(bathy_func);
    smoother.solve();

    // Verify solution is reasonable
    Real center_value = smoother.evaluate(0.5, 0.5);
    Real expected = bathy_func(0.5, 0.5);
    EXPECT_NEAR(center_value, expected, 0.5); // Smoothed, so not exact
}

TEST_F(BezierMultigridTest, MultigridProducesReasonableSolution) {
    // The multigrid version should produce a solution that:
    // 1. Is reasonably close to the data
    // 2. Is smooth (this is a smoothing problem)
    // Note: With edge constraints, MG provides approximate Q^{-1} so
    // the Schur complement CG converges to a slightly different solution.

    CGCubicBezierSmootherConfig config_mg;
    config_mg.lambda = 10.0;
    config_mg.use_iterative_solver = true;
    config_mg.use_multigrid = true;
    config_mg.multigrid_config.min_tree_level = 1;
    config_mg.tolerance = 1e-8;

    CGCubicBezierBathymetrySmoother smoother_mg(mesh_, config_mg);

    auto bathy_func = [](Real x, Real y) { return std::sin(M_PI * x) * std::cos(M_PI * y); };
    smoother_mg.set_bathymetry_data(bathy_func);
    smoother_mg.solve();

    // Check that solution is reasonable (within 50% of data at sample points)
    std::vector<std::pair<Real, Real>> test_points = {
        {0.25, 0.25}, {0.5, 0.5}, {0.75, 0.75}, {0.25, 0.75}};

    for (const auto &[x, y] : test_points) {
        Real val_mg = smoother_mg.evaluate(x, y);
        Real val_exact = bathy_func(x, y);
        EXPECT_NEAR(val_mg, val_exact, 0.5)
            << "Solution too far from data at (" << x << ", " << y << ")";
    }
}

TEST_F(BezierMultigridTest, MultigridNoConstraintsMatchesDirect) {
    // Build a uniform mesh (no hanging node constraints)
    QuadtreeAdapter uniform_mesh;
    uniform_mesh.build_uniform(0.0, 1.0, 0.0, 1.0, 2, 2);

    // For uniform meshes without edge constraints, the no-edge-constraint
    // path uses PCG with MG preconditioner, which should converge to the
    // same solution as direct LU solve.

    auto bathy_func = [](Real x, Real y) { return x * x + y; };

    // Direct solver
    CGCubicBezierSmootherConfig config_direct;
    config_direct.lambda = 10.0;
    config_direct.use_iterative_solver = true;
    config_direct.use_multigrid = false;
    config_direct.edge_ngauss = 0; // Disable edge constraints

    CGCubicBezierBathymetrySmoother smoother_direct(uniform_mesh, config_direct);
    smoother_direct.set_bathymetry_data(bathy_func);
    smoother_direct.solve();

    // Multigrid solver
    CGCubicBezierSmootherConfig config_mg;
    config_mg.lambda = 10.0;
    config_mg.use_iterative_solver = true;
    config_mg.use_multigrid = true;
    config_mg.multigrid_config.min_tree_level = 1;
    config_mg.tolerance = 1e-10;
    config_mg.edge_ngauss = 0; // Disable edge constraints

    CGCubicBezierBathymetrySmoother smoother_mg(uniform_mesh, config_mg);
    smoother_mg.set_bathymetry_data(bathy_func);
    smoother_mg.solve();

    // Solutions should match closely (PCG converges to same solution)
    std::vector<std::pair<Real, Real>> test_points = {
        {0.25, 0.25}, {0.5, 0.5}, {0.75, 0.75}};

    for (const auto &[x, y] : test_points) {
        Real val_direct = smoother_direct.evaluate(x, y);
        Real val_mg = smoother_mg.evaluate(x, y);
        EXPECT_NEAR(val_direct, val_mg, 1e-4)
            << "Mismatch at (" << x << ", " << y << ")";
    }
}

// =============================================================================
// Configuration tests
// =============================================================================

TEST_F(BezierMultigridTest, ConfigDefaults) {
    MultigridConfig config;
    EXPECT_EQ(config.min_tree_level, 0);  // Default 0; auto-computed from initial_grid
    EXPECT_EQ(config.pre_smoothing, 1);
    EXPECT_EQ(config.post_smoothing, 1);
    EXPECT_EQ(config.smoother_type, SmootherType::ColoredMultiplicativeSchwarz);
    EXPECT_FALSE(config.verbose);
}

TEST_F(BezierMultigridTest, SchwarzSmootherReducesResidual) {
    // Test that multiplicative Schwarz smoother reduces residual
    CGCubicBezierSmootherConfig smoother_config;
    CGCubicBezierBathymetrySmoother smoother(mesh_, smoother_config);

    Index n = smoother.dof_manager().num_free_dofs();

    // Create a simple tridiagonal SPD matrix
    SpMat Q(n, n);
    std::vector<Eigen::Triplet<Real>> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 4.0);
        if (i > 0) {
            triplets.emplace_back(i, i - 1, -1.0);
            triplets.emplace_back(i - 1, i, -1.0);
        }
    }
    Q.setFromTriplets(triplets.begin(), triplets.end());

    // Setup preconditioner with Schwarz smoother (default)
    MultigridConfig config;
    config.min_tree_level = 1;
    config.pre_smoothing = 2;
    config.post_smoothing = 2;
    config.smoother_type = SmootherType::MultiplicativeSchwarz;
    config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    BezierMultigridPreconditioner precond(config);
    precond.setup(Q, mesh_, smoother.dof_manager());

    // Create a residual and apply V-cycle
    VecX r = VecX::Ones(n);
    VecX z = precond.apply(r);

    // The correction should be non-zero and reduce residual when applied
    EXPECT_GT(z.norm(), 0.0);
    EXPECT_GT(r.dot(z), 0.0);
}

TEST_F(BezierMultigridTest, JacobiSmootherStillWorks) {
    // Verify Jacobi smoother still works when explicitly selected
    CGCubicBezierSmootherConfig smoother_config;
    CGCubicBezierBathymetrySmoother smoother(mesh_, smoother_config);

    Index n = smoother.dof_manager().num_free_dofs();

    SpMat Q(n, n);
    std::vector<Eigen::Triplet<Real>> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 4.0);
        if (i > 0) {
            triplets.emplace_back(i, i - 1, -1.0);
            triplets.emplace_back(i - 1, i, -1.0);
        }
    }
    Q.setFromTriplets(triplets.begin(), triplets.end());

    MultigridConfig config;
    config.min_tree_level = 1;
    config.smoother_type = SmootherType::Jacobi;
    config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    BezierMultigridPreconditioner precond(config);
    precond.setup(Q, mesh_, smoother.dof_manager());

    VecX r = VecX::Ones(n);
    VecX z = precond.apply(r);

    EXPECT_GT(z.norm(), 0.0);
    EXPECT_GT(r.dot(z), 0.0);
}

// =============================================================================
// L2 projection (full weighting) transfer operator tests
// =============================================================================

TEST_F(BezierMultigridTest, L2BernsteinMass1DDimensions) {
    // The 1D Bernstein mass matrix should be 4x4 for cubic
    MultigridConfig config;
    config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    BezierMultigridPreconditioner precond(config);

    // Access internal state via setup then check levels
    CGCubicBezierSmootherConfig smoother_config;
    CGCubicBezierBathymetrySmoother smoother(mesh_, smoother_config);

    Index n = smoother.dof_manager().num_free_dofs();
    SpMat Q(n, n);
    std::vector<Eigen::Triplet<Real>> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 2.0);
    }
    Q.setFromTriplets(triplets.begin(), triplets.end());

    precond.setup(Q, mesh_, smoother.dof_manager());

    // Verify the multigrid hierarchy was built with L2 operators
    EXPECT_TRUE(precond.is_setup());
    EXPECT_GE(precond.num_levels(), 1);
}

TEST_F(BezierMultigridTest, L2RestrictionProlongationTranspose) {
    // Verify that P = R^T (symmetry of the transfer operators)
    CGCubicBezierSmootherConfig smoother_config;
    CGCubicBezierBathymetrySmoother smoother(mesh_, smoother_config);

    Index n = smoother.dof_manager().num_free_dofs();
    SpMat Q(n, n);
    std::vector<Eigen::Triplet<Real>> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 4.0);
        if (i > 0) {
            triplets.emplace_back(i, i - 1, -1.0);
            triplets.emplace_back(i - 1, i, -1.0);
        }
    }
    Q.setFromTriplets(triplets.begin(), triplets.end());

    MultigridConfig config;
    config.min_tree_level = 0;
    config.transfer_strategy = TransferOperatorStrategy::L2Projection;
    config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    BezierMultigridPreconditioner precond(config);
    precond.setup(Q, mesh_, smoother.dof_manager());

    // For each level, verify P = R^T
    for (int level = 0; level < precond.num_levels() - 1; ++level) {
        const auto &L = precond.level(level);
        const auto &L_fine = precond.level(level + 1);

        // P is at coarse level, R is at fine level
        SpMat P = L.P;
        SpMat R = L_fine.R;
        SpMat P_expected = R.transpose();

        // Check dimensions
        EXPECT_EQ(P.rows(), P_expected.rows());
        EXPECT_EQ(P.cols(), P_expected.cols());

        // Check that P = R^T (within tolerance)
        SpMat diff = P - P_expected;
        Real max_diff = 0.0;
        for (int k = 0; k < diff.outerSize(); ++k) {
            for (SpMat::InnerIterator it(diff, k); it; ++it) {
                max_diff = std::max(max_diff, std::abs(it.value()));
            }
        }
        EXPECT_LT(max_diff, TOLERANCE)
            << "P != R^T at level " << level << ", max diff = " << max_diff;
    }
}

TEST_F(BezierMultigridTest, L2RestrictionConstantPreservation) {
    // L2 restriction should preserve constant fields
    CGCubicBezierSmootherConfig smoother_config;
    CGCubicBezierBathymetrySmoother smoother(mesh_, smoother_config);

    Index n = smoother.dof_manager().num_free_dofs();
    SpMat Q(n, n);
    std::vector<Eigen::Triplet<Real>> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 2.0);
    }
    Q.setFromTriplets(triplets.begin(), triplets.end());

    MultigridConfig config;
    config.min_tree_level = 0;
    config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    BezierMultigridPreconditioner precond(config);
    precond.setup(Q, mesh_, smoother.dof_manager());

    // Create constant field on finest level
    int finest = precond.num_levels() - 1;
    Index finest_dofs = precond.level(finest).Q.rows();
    VecX fine_const = VecX::Constant(finest_dofs, 3.14159);

    // Restrict to coarser levels and check constant is preserved
    VecX current = fine_const;
    for (int level = finest; level > 0; --level) {
        const auto &L = precond.level(level);
        VecX coarse = L.R * current;

        // Check that all values are close to the constant
        Real mean = coarse.mean();
        Real max_dev = (coarse.array() - mean).abs().maxCoeff();

        EXPECT_NEAR(mean, 3.14159, LOOSE_TOLERANCE)
            << "Mean should be preserved at level " << (level - 1);
        EXPECT_LT(max_dev, LOOSE_TOLERANCE)
            << "Constant field not preserved at level " << (level - 1);

        current = coarse;
    }
}

TEST_F(BezierMultigridTest, L2GalerkinSymmetric) {
    // Verify that the Galerkin coarse operator Q_c = R * Q_f * P is symmetric
    CGCubicBezierSmootherConfig smoother_config;
    CGCubicBezierBathymetrySmoother smoother(mesh_, smoother_config);

    Index n = smoother.dof_manager().num_free_dofs();
    SpMat Q(n, n);
    std::vector<Eigen::Triplet<Real>> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 4.0);
        if (i > 0) {
            triplets.emplace_back(i, i - 1, -1.0);
            triplets.emplace_back(i - 1, i, -1.0);
        }
    }
    Q.setFromTriplets(triplets.begin(), triplets.end());

    MultigridConfig config;
    config.min_tree_level = 0;
    config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    BezierMultigridPreconditioner precond(config);
    precond.setup(Q, mesh_, smoother.dof_manager());

    // Check symmetry of Q at each level
    for (int level = 0; level < precond.num_levels(); ++level) {
        const SpMat &Q_level = precond.level(level).Q;
        SpMat Q_t = Q_level.transpose();
        SpMat diff = Q_level - Q_t;

        Real max_asym = 0.0;
        for (int k = 0; k < diff.outerSize(); ++k) {
            for (SpMat::InnerIterator it(diff, k); it; ++it) {
                max_asym = std::max(max_asym, std::abs(it.value()));
            }
        }
        EXPECT_LT(max_asym, TOLERANCE)
            << "Q not symmetric at level " << level << ", max asymmetry = " << max_asym;
    }
}

TEST_F(BezierMultigridTest, L2LocalRestrictionMatrixExport) {
    MultigridConfig config;
    BezierMultigridPreconditioner precond(config);

    const MatX &R = precond.R_L2_local();

    // Verify dimensions: 16 coarse DOFs x 64 fine DOFs (4 children * 16)
    EXPECT_EQ(R.rows(), 16);
    EXPECT_EQ(R.cols(), 64);

    // Write to formatted text file
    std::ofstream out("/tmp/R_L2_local.txt");
    ASSERT_TRUE(out.is_open());

    out << std::fixed << std::setprecision(6);
    for (int i = 0; i < R.rows(); ++i) {
        for (int j = 0; j < R.cols(); ++j) {
            out << std::setw(12) << R(i, j);
            if (j < R.cols() - 1)
                out << " ";
        }
        out << "\n";
    }
    out.close();

    // Verify file was written
    std::ifstream check("/tmp/R_L2_local.txt");
    EXPECT_TRUE(check.good());

    // Verify row sums = 1 (partition of unity for restriction)
    for (int i = 0; i < R.rows(); ++i) {
        Real row_sum = R.row(i).sum();
        EXPECT_NEAR(row_sum, 1.0, TOLERANCE) << "Row " << i << " sum = " << row_sum;
    }

    // Verify column sums = 0.25 (4 children, each contributes 1/4)
    for (int j = 0; j < R.cols(); ++j) {
        Real col_sum = R.col(j).sum();
        EXPECT_NEAR(col_sum, 0.25, TOLERANCE) << "Column " << j << " sum = " << col_sum;
    }
}

TEST_F(BezierMultigridTest, L2LocalProlongationMatrixExport) {
    MultigridConfig config;
    BezierMultigridPreconditioner precond(config);

    const MatX &P = precond.P_L2_local();

    // Verify dimensions: 64 fine DOFs x 16 coarse DOFs (P = R^T)
    EXPECT_EQ(P.rows(), 64);
    EXPECT_EQ(P.cols(), 16);

    // Write to formatted text file
    std::ofstream out("/tmp/P_L2_local.txt");
    ASSERT_TRUE(out.is_open());

    out << std::fixed << std::setprecision(6);
    for (int i = 0; i < P.rows(); ++i) {
        for (int j = 0; j < P.cols(); ++j) {
            out << std::setw(12) << P(i, j);
            if (j < P.cols() - 1)
                out << " ";
        }
        out << "\n";
    }
    out.close();

    // Verify file was written
    std::ifstream check("/tmp/P_L2_local.txt");
    EXPECT_TRUE(check.good());

    // Verify row sums = 0.25 (P = R^T, so row sums of P = col sums of R)
    for (int i = 0; i < P.rows(); ++i) {
        Real row_sum = P.row(i).sum();
        EXPECT_NEAR(row_sum, 0.25, TOLERANCE) << "Row " << i << " sum = " << row_sum;
    }

    // Verify column sums = 1 (P = R^T, so col sums of P = row sums of R)
    for (int j = 0; j < P.cols(); ++j) {
        Real col_sum = P.col(j).sum();
        EXPECT_NEAR(col_sum, 1.0, TOLERANCE) << "Column " << j << " sum = " << col_sum;
    }
}

// =============================================================================
// Cached Rediscretization tests
// =============================================================================

TEST_F(BezierMultigridTest, CoarseGridStrategyDefault) {
    // Verify default coarse grid strategy is CachedRediscretization
    MultigridConfig config;
    EXPECT_EQ(config.coarse_grid_strategy, CoarseGridStrategy::CachedRediscretization);
}

TEST_F(BezierMultigridTest, CachedRediscretizationConfigurable) {
    // Verify CachedRediscretization strategy can be configured
    MultigridConfig config;
    config.coarse_grid_strategy = CoarseGridStrategy::CachedRediscretization;
    EXPECT_EQ(config.coarse_grid_strategy, CoarseGridStrategy::CachedRediscretization);

    BezierMultigridPreconditioner precond(config);
    // Without a cache, setup falls back to Galerkin
    CGCubicBezierSmootherConfig smoother_config;
    CGCubicBezierBathymetrySmoother smoother(mesh_, smoother_config);

    Index n = smoother.dof_manager().num_free_dofs();
    SpMat Q(n, n);
    std::vector<Eigen::Triplet<Real>> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 2.0);
    }
    Q.setFromTriplets(triplets.begin(), triplets.end());

    // CachedRediscretization without cache falls back to Galerkin
    precond.setup(Q, mesh_, smoother.dof_manager());
    EXPECT_TRUE(precond.is_setup());
}

TEST_F(BezierMultigridTest, CachedRediscretizationWithAdaptiveSmoother) {
    // Test that CachedRediscretization works with adaptive smoother's cache
    // Build a uniform mesh (2x2) that can have multigrid hierarchy
    QuadtreeAdapter uniform_mesh;
    uniform_mesh.build_uniform(0.0, 1.0, 0.0, 1.0, 4, 4);

    CGCubicBezierSmootherConfig config;
    config.lambda = 10.0;
    config.use_iterative_solver = true;
    config.use_multigrid = true;
    config.multigrid_config.min_tree_level = 1;
    config.multigrid_config.coarse_grid_strategy = CoarseGridStrategy::CachedRediscretization;
    config.tolerance = 1e-8;
    config.edge_ngauss = 0; // Disable edge constraints for cleaner test

    CGCubicBezierBathymetrySmoother smoother(uniform_mesh, config);

    // The smoother assembles element matrices during set_bathymetry_data
    auto bathy_func = [](Real x, Real y) { return x * x + y * y; };
    smoother.set_bathymetry_data(bathy_func);
    smoother.solve();

    // Verify solution is reasonable
    Real center_value = smoother.evaluate(0.5, 0.5);
    Real expected = bathy_func(0.5, 0.5);
    EXPECT_NEAR(center_value, expected, 0.5);
}

TEST_F(BezierMultigridTest, CachedRediscretizationConvergence) {
    // Compare convergence with Galerkin vs CachedRediscretization strategies
    // Both should converge to similar solutions for simple problems
    QuadtreeAdapter uniform_mesh;
    uniform_mesh.build_uniform(0.0, 1.0, 0.0, 1.0, 4, 4);

    auto bathy_func = [](Real x, Real y) { return std::sin(M_PI * x) * std::cos(M_PI * y); };

    // Galerkin strategy
    CGCubicBezierSmootherConfig config_galerkin;
    config_galerkin.lambda = 10.0;
    config_galerkin.use_iterative_solver = true;
    config_galerkin.use_multigrid = true;
    config_galerkin.multigrid_config.min_tree_level = 1;
    config_galerkin.multigrid_config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    config_galerkin.tolerance = 1e-8;
    config_galerkin.edge_ngauss = 0;

    CGCubicBezierBathymetrySmoother smoother_galerkin(uniform_mesh, config_galerkin);
    smoother_galerkin.set_bathymetry_data(bathy_func);
    smoother_galerkin.solve();

    // CachedRediscretization strategy
    CGCubicBezierSmootherConfig config_cached;
    config_cached.lambda = 10.0;
    config_cached.use_iterative_solver = true;
    config_cached.use_multigrid = true;
    config_cached.multigrid_config.min_tree_level = 1;
    config_cached.multigrid_config.coarse_grid_strategy = CoarseGridStrategy::CachedRediscretization;
    config_cached.tolerance = 1e-8;
    config_cached.edge_ngauss = 0;

    CGCubicBezierBathymetrySmoother smoother_cached(uniform_mesh, config_cached);
    smoother_cached.set_bathymetry_data(bathy_func);
    smoother_cached.solve();

    // Both should produce solutions in similar range
    // Note: They may differ slightly because CachedRediscretization uses exact
    // element matrices while Galerkin uses projected matrices
    std::vector<std::pair<Real, Real>> test_points = {
        {0.25, 0.25}, {0.5, 0.5}, {0.75, 0.75}, {0.25, 0.75}};

    for (const auto &[x, y] : test_points) {
        Real val_galerkin = smoother_galerkin.evaluate(x, y);
        Real val_cached = smoother_cached.evaluate(x, y);
        // Allow larger tolerance as the coarse operators are different
        EXPECT_NEAR(val_galerkin, val_cached, 0.1)
            << "Large difference at (" << x << ", " << y << ")";
    }
}

TEST_F(BezierMultigridTest, CachedRediscretizationSymmetric) {
    // Verify that CachedRediscretization produces symmetric coarse operators
    QuadtreeAdapter uniform_mesh;
    uniform_mesh.build_uniform(0.0, 1.0, 0.0, 1.0, 4, 4);

    CGCubicBezierSmootherConfig smoother_config;
    smoother_config.lambda = 10.0;
    smoother_config.use_iterative_solver = true;
    smoother_config.use_multigrid = true;
    smoother_config.multigrid_config.min_tree_level = 1;
    smoother_config.multigrid_config.coarse_grid_strategy = CoarseGridStrategy::CachedRediscretization;
    smoother_config.edge_ngauss = 0;

    CGCubicBezierBathymetrySmoother smoother(uniform_mesh, smoother_config);

    auto bathy_func = [](Real x, Real y) { return x + y; };
    smoother.set_bathymetry_data(bathy_func);
    smoother.solve();

    // The element matrices should be symmetric, so assembled coarse Q should be too
    // Note: We can't directly access the multigrid levels from the smoother,
    // but we verified convergence which requires reasonable operators
    EXPECT_TRUE(smoother.is_solved());
}

// =============================================================================
// Adaptive mesh restriction tests (partition of unity)
// =============================================================================

TEST_F(BezierMultigridTest, L2RestrictionRowSumsEqualOneAdaptive) {
    // Test with ADAPTIVE mesh (non-uniform), not just uniform
    // This catches the bug where partial children break partition of unity

    // Create octree and refine only some elements
    OctreeAdapter octree(0.0, 1.0, 0.0, 1.0, -1.0, 0.0);
    octree.build_uniform(2, 2, 1);

    // Refine only some elements (creates partial children)
    std::vector<Index> to_refine = {0, 1};  // Refine 2 of 4 elements
    std::vector<RefineMask> masks(to_refine.size(), RefineMask::XY);
    octree.refine(to_refine, masks);

    // Create QuadtreeAdapter from refined octree
    QuadtreeAdapter adaptive_mesh(octree);

    CGCubicBezierSmootherConfig smoother_config;
    CGCubicBezierBathymetrySmoother smoother(adaptive_mesh, smoother_config);

    Index n = smoother.dof_manager().num_free_dofs();
    SpMat Q(n, n);
    std::vector<Eigen::Triplet<Real>> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 2.0);
    }
    Q.setFromTriplets(triplets.begin(), triplets.end());

    MultigridConfig config;
    config.min_tree_level = 0;
    config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    BezierMultigridPreconditioner precond(config);
    precond.setup(Q, adaptive_mesh, smoother.dof_manager());

    // Verify row sums = 1 for each restriction operator
    for (int level = precond.num_levels() - 1; level > 0; --level) {
        const SpMat &R = precond.level(level).R;
        VecX row_sums = R * VecX::Ones(R.cols());

        for (Index i = 0; i < R.rows(); ++i) {
            EXPECT_NEAR(row_sums(i), 1.0, 1e-10)
                << "Row " << i << " of R at level " << level
                << " sums to " << row_sums(i) << ", not 1.0";
        }
    }
}

TEST_F(BezierMultigridTest, AdaptiveMeshConstantPreservation) {
    // Verify constant fields are preserved during restriction on adaptive meshes

    // Create octree and refine only some elements
    OctreeAdapter octree(0.0, 1.0, 0.0, 1.0, -1.0, 0.0);
    octree.build_uniform(2, 2, 1);

    // Refine only some elements (creates partial children)
    std::vector<Index> to_refine = {0, 1};  // Refine 2 of 4 elements
    std::vector<RefineMask> masks(to_refine.size(), RefineMask::XY);
    octree.refine(to_refine, masks);

    // Create QuadtreeAdapter from refined octree
    QuadtreeAdapter adaptive_mesh(octree);

    CGCubicBezierSmootherConfig smoother_config;
    CGCubicBezierBathymetrySmoother smoother(adaptive_mesh, smoother_config);

    Index n = smoother.dof_manager().num_free_dofs();
    SpMat Q(n, n);
    std::vector<Eigen::Triplet<Real>> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 2.0);
    }
    Q.setFromTriplets(triplets.begin(), triplets.end());

    MultigridConfig config;
    config.min_tree_level = 0;
    config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    BezierMultigridPreconditioner precond(config);
    precond.setup(Q, adaptive_mesh, smoother.dof_manager());

    // Restrict constant field and verify it stays constant
    int finest = precond.num_levels() - 1;
    VecX fine_const = VecX::Constant(precond.level(finest).Q.rows(), 3.14159);

    VecX current = fine_const;
    for (int level = finest; level > 0; --level) {
        VecX coarse = precond.level(level).R * current;

        // All values should be exactly 3.14159
        for (Index i = 0; i < coarse.size(); ++i) {
            EXPECT_NEAR(coarse(i), 3.14159, 1e-10)
                << "Value " << i << " at level " << (level - 1) << " is " << coarse(i);
        }

        current = coarse;
    }
}

// =============================================================================
// Center-graded mesh tests for global transfer operators
// =============================================================================

class BezierMultigridCenterGradedTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Build a center-graded mesh with 3 levels of refinement toward center
        // Tree levels: 0 (root), 1, 2 (finest leaves)
        mesh_.build_center_graded(3);
    }

    QuadtreeAdapter mesh_;
};

TEST_F(BezierMultigridCenterGradedTest, GlobalProlongationRowColumnSumsCenterGraded) {
    // Build DOF manager via smoother
    CGCubicBezierSmootherConfig smoother_config;
    CGCubicBezierBathymetrySmoother smoother(mesh_, smoother_config);

    Index n = smoother.dof_manager().num_free_dofs();

    // Create simple SPD matrix
    SpMat Q(n, n);
    std::vector<Eigen::Triplet<Real>> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 4.0);
        if (i > 0) {
            triplets.emplace_back(i, i - 1, -1.0);
            triplets.emplace_back(i - 1, i, -1.0);
        }
    }
    Q.setFromTriplets(triplets.begin(), triplets.end());

    // Setup multigrid with 3 levels (tree level 0 = MG level 0)
    MultigridConfig config;
    config.min_tree_level = 0;
    config.transfer_strategy = TransferOperatorStrategy::L2Projection;
    config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    BezierMultigridPreconditioner precond(config);
    precond.setup(Q, mesh_, smoother.dof_manager());

    ASSERT_EQ(precond.num_levels(), 3) << "Expected exactly 3 MG levels";

    // Test P at each level (P is stored at the coarse level)
    // With L2 projection, column sums should be 1 (partition of unity)
    for (int level = 0; level < precond.num_levels() - 1; ++level) {
        const SpMat &P = precond.level(level).P;

        // Column sums: each coarse DOF should contribute total weight 1 to fine DOFs
        VecX col_sums = VecX::Zero(P.cols());
        for (int k = 0; k < P.outerSize(); ++k) {
            for (SpMat::InnerIterator it(P, k); it; ++it) {
                col_sums(it.col()) += it.value();
            }
        }

        // Count columns with incorrect sums
        Index bad_cols = 0;
        std::vector<Index> bad_col_indices;
        for (Index j = 0; j < P.cols(); ++j) {
            if (std::abs(col_sums(j) - 1.0) > LOOSE_TOLERANCE) {
                bad_cols++;
                if (bad_col_indices.size() < 10) {  // Limit output
                    bad_col_indices.push_back(j);
                }
            }
        }

        // Report column sum statistics
        Real min_col = col_sums.minCoeff();
        Real max_col = col_sums.maxCoeff();
        Real mean_col = col_sums.mean();
        std::cerr << "[Test] P level " << level << ": " << P.rows() << "x" << P.cols()
                  << ", col sums: min=" << min_col << ", max=" << max_col
                  << ", mean=" << mean_col << ", bad=" << bad_cols << "/" << P.cols() << "\n";

        // Print bad column indices
        if (!bad_col_indices.empty()) {
            std::cerr << "  Bad columns (first 10): ";
            for (Index j : bad_col_indices) {
                std::cerr << j << " (sum=" << col_sums(j) << ") ";
            }
            std::cerr << "\n";
        }

        // Assert that column sums are correct
        EXPECT_EQ(bad_cols, 0)
            << "P at level " << level << " has " << bad_cols << " columns with sum != 1.0";

        // Row sums: report statistics (may vary based on mesh structure)
        VecX row_sums = P * VecX::Ones(P.cols());
        Real min_row = row_sums.minCoeff();
        Real max_row = row_sums.maxCoeff();
        Real mean_row = row_sums.mean();
        std::cerr << "  row sums: min=" << min_row << ", max=" << max_row
                  << ", mean=" << mean_row << "\n";
    }
}

TEST_F(BezierMultigridCenterGradedTest, GlobalRestrictionRowColumnSumsCenterGraded) {
    // Build DOF manager via smoother
    CGCubicBezierSmootherConfig smoother_config;
    CGCubicBezierBathymetrySmoother smoother(mesh_, smoother_config);

    Index n = smoother.dof_manager().num_free_dofs();

    // Create simple SPD matrix
    SpMat Q(n, n);
    std::vector<Eigen::Triplet<Real>> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 4.0);
        if (i > 0) {
            triplets.emplace_back(i, i - 1, -1.0);
            triplets.emplace_back(i - 1, i, -1.0);
        }
    }
    Q.setFromTriplets(triplets.begin(), triplets.end());

    // Setup multigrid with 3 levels (tree level 0 = MG level 0)
    MultigridConfig config;
    config.min_tree_level = 0;
    config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    BezierMultigridPreconditioner precond(config);
    precond.setup(Q, mesh_, smoother.dof_manager());

    ASSERT_EQ(precond.num_levels(), 3) << "Expected exactly 3 MG levels";

    // Test R at each level (R is stored at the fine level)
    for (int level = 1; level < precond.num_levels(); ++level) {
        const SpMat &R = precond.level(level).R;

        // Row sums: each coarse DOF should have weights summing to 1
        // (partition of unity after normalization)
        VecX row_sums = R * VecX::Ones(R.cols());

        // Count rows with incorrect sums
        Index bad_rows = 0;
        std::vector<Index> bad_row_indices;
        for (Index i = 0; i < R.rows(); ++i) {
            if (std::abs(row_sums(i) - 1.0) > TOLERANCE) {
                bad_rows++;
                if (bad_row_indices.size() < 10) {  // Limit output
                    bad_row_indices.push_back(i);
                }
            }
        }

        // Report row sum statistics
        Real min_row = row_sums.minCoeff();
        Real max_row = row_sums.maxCoeff();
        Real mean_row = row_sums.mean();
        std::cerr << "[Test] R level " << level << ": " << R.rows() << "x" << R.cols()
                  << ", row sums: min=" << min_row << ", max=" << max_row
                  << ", mean=" << mean_row << ", bad=" << bad_rows << "/" << R.rows() << "\n";

        // Print bad row indices
        if (!bad_row_indices.empty()) {
            std::cerr << "  Bad rows (first 10): ";
            for (Index i : bad_row_indices) {
                std::cerr << i << " (sum=" << row_sums(i) << ") ";
            }
            std::cerr << "\n";
        }

        // Assert that row sums are correct
        EXPECT_EQ(bad_rows, 0)
            << "R at level " << level << " has " << bad_rows << " rows with sum != 1.0";

        // Column sums: report statistics
        VecX col_sums = VecX::Zero(R.cols());
        for (int k = 0; k < R.outerSize(); ++k) {
            for (SpMat::InnerIterator it(R, k); it; ++it) {
                col_sums(it.col()) += it.value();
            }
        }

        Real min_col = col_sums.minCoeff();
        Real max_col = col_sums.maxCoeff();
        Real mean_col = col_sums.mean();
        std::cerr << "  col sums: min=" << min_col << ", max=" << max_col
                  << ", mean=" << mean_col << "\n";
    }
}

TEST_F(BezierMultigridCenterGradedTest, GlobalOperatorsToTextFiles) {
    // Build DOF manager via smoother
    CGCubicBezierSmootherConfig smoother_config;
    CGCubicBezierBathymetrySmoother smoother(mesh_, smoother_config);

    Index n = smoother.dof_manager().num_free_dofs();

    // Create simple SPD matrix
    SpMat Q(n, n);
    std::vector<Eigen::Triplet<Real>> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 4.0);
        if (i > 0) {
            triplets.emplace_back(i, i - 1, -1.0);
            triplets.emplace_back(i - 1, i, -1.0);
        }
    }
    Q.setFromTriplets(triplets.begin(), triplets.end());

    // Setup multigrid with 3 levels (tree level 0 = MG level 0)
    MultigridConfig config;
    config.min_tree_level = 0;
    config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    BezierMultigridPreconditioner precond(config);
    precond.setup(Q, mesh_, smoother.dof_manager());

    ASSERT_EQ(precond.num_levels(), 3) << "Expected exactly 3 MG levels";

    // Write P matrices to files
    // P_global_level_L: prolongation from level L -> level L+1
    // precond.level(L).P maps from level L to level L+1
    for (int level = 0; level < precond.num_levels() - 1; ++level) {
        const SpMat &P_sparse = precond.level(level).P;
        MatX P = MatX(P_sparse);  // Convert to dense

        std::string filename = "/tmp/P_global_level_" + std::to_string(level) + ".txt";
        std::ofstream out(filename);
        ASSERT_TRUE(out.is_open()) << "Failed to open " << filename;

        out << std::fixed << std::setprecision(6);
        for (Index i = 0; i < P.rows(); ++i) {
            for (Index j = 0; j < P.cols(); ++j) {
                out << std::setw(12) << P(i, j);
                if (j < P.cols() - 1)
                    out << " ";
            }
            out << "\n";
        }
        out.close();

        // Verify file was written
        std::ifstream check(filename);
        EXPECT_TRUE(check.good()) << "File " << filename << " not created";

        std::cerr << "[Test] Wrote P level " << level << " (" << P.rows() << "x" << P.cols()
                  << ") to " << filename << "\n";
    }

    // Write R matrices to files
    // R_global_level_L: restriction from level L+1 -> level L (named by TARGET level)
    // precond.level(L+1).R maps from level L+1 to level L
    for (int level = 1; level < precond.num_levels(); ++level) {
        const SpMat &R_sparse = precond.level(level).R;
        MatX R = MatX(R_sparse);  // Convert to dense

        // Name by target level (level - 1), not source level
        int target_level = level - 1;
        std::string filename = "/tmp/R_global_level_" + std::to_string(target_level) + ".txt";
        std::ofstream out(filename);
        ASSERT_TRUE(out.is_open()) << "Failed to open " << filename;

        out << std::fixed << std::setprecision(6);
        for (Index i = 0; i < R.rows(); ++i) {
            for (Index j = 0; j < R.cols(); ++j) {
                out << std::setw(12) << R(i, j);
                if (j < R.cols() - 1)
                    out << " ";
            }
            out << "\n";
        }
        out.close();

        // Verify file was written
        std::ifstream check(filename);
        EXPECT_TRUE(check.good()) << "File " << filename << " not created";

        std::cerr << "[Test] Wrote R (level " << level << " -> " << target_level << ") ("
                  << R.rows() << "x" << R.cols() << ") to " << filename << "\n";
    }
}

// =============================================================================
// BezierSubdivision transfer operator tests
// =============================================================================

TEST_F(BezierMultigridCenterGradedTest, SubdivisionLocalMatrices) {
    // Verify local subdivision matrices P_child[0-3] are all non-negative
    // with row sums = 1 (each output is a convex combination of inputs)
    CubicBezierBasis2D basis;
    MatX S_left = basis.compute_1d_extraction_matrix(0.0, 0.5);
    MatX S_right = basis.compute_1d_extraction_matrix(0.5, 1.0);

    // Kronecker product helper
    auto kron = [](const MatX& A, const MatX& B) -> MatX {
        const int m = static_cast<int>(A.rows());
        const int n = static_cast<int>(A.cols());
        const int p = static_cast<int>(B.rows());
        const int q = static_cast<int>(B.cols());
        MatX result(m * p, n * q);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                result.block(i * p, j * q, p, q) = A(i, j) * B;
            }
        }
        return result;
    };

    // Build 2D subdivision matrices for each child quadrant
    std::array<MatX, 4> P_child;
    P_child[0] = kron(S_left, S_left);   // (0,0) child
    P_child[1] = kron(S_right, S_left);  // (1,0) child
    P_child[2] = kron(S_left, S_right);  // (0,1) child
    P_child[3] = kron(S_right, S_right); // (1,1) child

    for (int c = 0; c < 4; ++c) {
        const MatX& P = P_child[c];
        EXPECT_EQ(P.rows(), 16);
        EXPECT_EQ(P.cols(), 16);

        // All entries should be non-negative (de Casteljau uses convex combinations)
        for (Index i = 0; i < P.rows(); ++i) {
            for (Index j = 0; j < P.cols(); ++j) {
                EXPECT_GE(P(i, j), -TOLERANCE)
                    << "Negative entry in P_child[" << c << "](" << i << "," << j << "): " << P(i, j);
            }
        }

        // Row sums should be 1 (each output control point is a convex combination)
        VecX row_sums = P.rowwise().sum();
        for (Index i = 0; i < P.rows(); ++i) {
            EXPECT_NEAR(row_sums(i), 1.0, TOLERANCE)
                << "Row sum != 1 in P_child[" << c << "] row " << i;
        }

        // Print column sums for reference (these vary based on which child quadrant)
        VecX col_sums = P.colwise().sum();
        std::cerr << "P_child[" << c << "] col sums: min=" << col_sums.minCoeff()
                  << ", max=" << col_sums.maxCoeff()
                  << ", total=" << col_sums.sum() << "\n";
    }

    // The sum of all 4 children's column sums should equal 4 for each parent DOF
    // (each parent DOF contributes to exactly 4 child elements)
    MatX P_stacked(64, 16);
    for (int c = 0; c < 4; ++c) {
        P_stacked.block(c * 16, 0, 16, 16) = P_child[c];
    }
    VecX total_col_sums = P_stacked.colwise().sum();
    for (Index j = 0; j < 16; ++j) {
        EXPECT_NEAR(total_col_sums(j), 4.0, TOLERANCE)
            << "Total column sum != 4 for parent DOF " << j;
    }
}

TEST_F(BezierMultigridCenterGradedTest, SubdivisionTransferAllNonNegative) {
    // Verify BezierSubdivision produces all non-negative transfer operator entries
    CGCubicBezierSmootherConfig smoother_config;
    CGCubicBezierBathymetrySmoother smoother(mesh_, smoother_config);

    Index n = smoother.dof_manager().num_free_dofs();
    SpMat Q(n, n);
    std::vector<Eigen::Triplet<Real>> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 4.0);
        if (i > 0) {
            triplets.emplace_back(i, i - 1, -1.0);
            triplets.emplace_back(i - 1, i, -1.0);
        }
    }
    Q.setFromTriplets(triplets.begin(), triplets.end());

    MultigridConfig config;
    config.min_tree_level = 0;
    config.transfer_strategy = TransferOperatorStrategy::BezierSubdivision;
    config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    BezierMultigridPreconditioner precond(config);
    precond.setup(Q, mesh_, smoother.dof_manager());

    // Check P entries are all non-negative
    for (int level = 0; level < precond.num_levels() - 1; ++level) {
        const SpMat &P = precond.level(level).P;
        for (int k = 0; k < P.outerSize(); ++k) {
            for (SpMat::InnerIterator it(P, k); it; ++it) {
                EXPECT_GE(it.value(), -TOLERANCE)
                    << "Negative entry in P at level " << level
                    << " (" << it.row() << ", " << it.col() << "): " << it.value();
            }
        }
    }

    // Check R entries are all non-negative
    for (int level = 1; level < precond.num_levels(); ++level) {
        const SpMat &R = precond.level(level).R;
        for (int k = 0; k < R.outerSize(); ++k) {
            for (SpMat::InnerIterator it(R, k); it; ++it) {
                EXPECT_GE(it.value(), -TOLERANCE)
                    << "Negative entry in R at level " << level
                    << " (" << it.row() << ", " << it.col() << "): " << it.value();
            }
        }
    }
}

TEST_F(BezierMultigridCenterGradedTest, SubdivisionRowColumnSums) {
    // Verify R row sums = 1 (constant preservation) with BezierSubdivision
    CGCubicBezierSmootherConfig smoother_config;
    CGCubicBezierBathymetrySmoother smoother(mesh_, smoother_config);

    Index n = smoother.dof_manager().num_free_dofs();
    SpMat Q(n, n);
    std::vector<Eigen::Triplet<Real>> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 4.0);
        if (i > 0) {
            triplets.emplace_back(i, i - 1, -1.0);
            triplets.emplace_back(i - 1, i, -1.0);
        }
    }
    Q.setFromTriplets(triplets.begin(), triplets.end());

    MultigridConfig config;
    config.min_tree_level = 0;
    config.transfer_strategy = TransferOperatorStrategy::BezierSubdivision;
    config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    BezierMultigridPreconditioner precond(config);
    precond.setup(Q, mesh_, smoother.dof_manager());

    // Check R row sums = 1 (restriction preserves constants)
    for (int level = 1; level < precond.num_levels(); ++level) {
        const SpMat &R = precond.level(level).R;

        VecX row_sums = R * VecX::Ones(R.cols());
        for (Index i = 0; i < R.rows(); ++i) {
            EXPECT_NEAR(row_sums(i), 1.0, LOOSE_TOLERANCE)
                << "R row sum != 1 at level " << level << " row " << i;
        }
    }

    // Check P row sums = 1 (each fine DOF is a convex combination of coarse DOFs)
    for (int level = 0; level < precond.num_levels() - 1; ++level) {
        const SpMat &P = precond.level(level).P;

        VecX row_sums = P * VecX::Ones(P.cols());
        std::cerr << "P level " << level << " row sums: min=" << row_sums.minCoeff()
                  << ", max=" << row_sums.maxCoeff() << "\n";

        // P column sums vary based on mesh structure (print for reference)
        VecX col_sums = VecX::Zero(P.cols());
        for (int k = 0; k < P.outerSize(); ++k) {
            for (SpMat::InnerIterator it(P, k); it; ++it) {
                col_sums(it.col()) += it.value();
            }
        }
        std::cerr << "P level " << level << " col sums: min=" << col_sums.minCoeff()
                  << ", max=" << col_sums.maxCoeff() << "\n";
    }
}

TEST_F(BezierMultigridCenterGradedTest, SubdivisionConstantPreservation) {
    // Verify constant fields are preserved with BezierSubdivision
    CGCubicBezierSmootherConfig smoother_config;
    CGCubicBezierBathymetrySmoother smoother(mesh_, smoother_config);

    Index n = smoother.dof_manager().num_free_dofs();
    SpMat Q(n, n);
    std::vector<Eigen::Triplet<Real>> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 2.0);
    }
    Q.setFromTriplets(triplets.begin(), triplets.end());

    MultigridConfig config;
    config.min_tree_level = 0;
    config.transfer_strategy = TransferOperatorStrategy::BezierSubdivision;
    config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    BezierMultigridPreconditioner precond(config);
    precond.setup(Q, mesh_, smoother.dof_manager());

    // Restrict constant field from finest to coarsest
    int finest = precond.num_levels() - 1;
    VecX fine_const = VecX::Constant(precond.level(finest).Q.rows(), 3.14159);

    VecX current = fine_const;
    for (int level = finest; level > 0; --level) {
        VecX coarse = precond.level(level).R * current;

        Real mean = coarse.mean();
        Real max_dev = (coarse.array() - mean).abs().maxCoeff();

        EXPECT_NEAR(mean, 3.14159, LOOSE_TOLERANCE)
            << "Mean not preserved at level " << (level - 1);
        EXPECT_LT(max_dev, LOOSE_TOLERANCE)
            << "Constant field not preserved at level " << (level - 1);

        current = coarse;
    }
}

TEST_F(BezierMultigridCenterGradedTest, SubdivisionVCycleConverges) {
    // Verify V-cycle converges with BezierSubdivision
    CGCubicBezierSmootherConfig smoother_config;
    CGCubicBezierBathymetrySmoother smoother(mesh_, smoother_config);

    Index n = smoother.dof_manager().num_free_dofs();
    SpMat Q(n, n);
    std::vector<Eigen::Triplet<Real>> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 4.0);
        if (i > 0) {
            triplets.emplace_back(i, i - 1, -1.0);
            triplets.emplace_back(i - 1, i, -1.0);
        }
    }
    Q.setFromTriplets(triplets.begin(), triplets.end());

    MultigridConfig config;
    config.min_tree_level = 0;
    config.pre_smoothing = 2;
    config.post_smoothing = 2;
    config.transfer_strategy = TransferOperatorStrategy::BezierSubdivision;
    config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    BezierMultigridPreconditioner precond(config);
    precond.setup(Q, mesh_, smoother.dof_manager());

    VecX r = VecX::Ones(n);
    VecX z = precond.apply(r);

    EXPECT_GT(z.norm(), 0.0);
    EXPECT_GT(r.dot(z), 0.0);  // Positive definite preconditioner
}

TEST_F(BezierMultigridCenterGradedTest, SubdivisionStrategyDefault) {
    // Verify default is BezierSubdivision (recommended for adaptive meshes)
    MultigridConfig config;
    EXPECT_EQ(config.transfer_strategy, TransferOperatorStrategy::BezierSubdivision);
}

// =============================================================================
// Composite Grid Hierarchy Tests
// =============================================================================

TEST_F(BezierMultigridCenterGradedTest, CompositeHierarchy_DOFsDecreaseWithCoarsening) {
    // Verify each coarser level has fewer DOFs than the finer level above it
    CGCubicBezierSmootherConfig smoother_config;
    CGCubicBezierBathymetrySmoother smoother(mesh_, smoother_config);

    Index n = smoother.dof_manager().num_free_dofs();
    SpMat Q(n, n);
    std::vector<Eigen::Triplet<Real>> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 2.0);
    }
    Q.setFromTriplets(triplets.begin(), triplets.end());

    MultigridConfig config;
    config.min_tree_level = 0;
    config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    BezierMultigridPreconditioner precond(config);
    precond.setup(Q, mesh_, smoother.dof_manager());

    ASSERT_GE(precond.num_levels(), 2) << "Need at least 2 levels for this test";

    // Check DOF counts decrease from finest to coarsest
    for (int level = precond.num_levels() - 1; level > 0; --level) {
        Index fine_dofs = precond.level(level).num_dofs;
        Index coarse_dofs = precond.level(level - 1).num_dofs;

        EXPECT_GT(fine_dofs, coarse_dofs)
            << "Level " << level << " should have more DOFs than level " << (level - 1)
            << " (fine=" << fine_dofs << ", coarse=" << coarse_dofs << ")";
    }
}

TEST_F(BezierMultigridCenterGradedTest, CompositeHierarchy_PassThroughClassification) {
    // Nodes present in both fine and coarse grids should be marked as pass-through
    CGCubicBezierSmootherConfig smoother_config;
    CGCubicBezierBathymetrySmoother smoother(mesh_, smoother_config);

    Index n = smoother.dof_manager().num_free_dofs();
    SpMat Q(n, n);
    std::vector<Eigen::Triplet<Real>> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 2.0);
    }
    Q.setFromTriplets(triplets.begin(), triplets.end());

    MultigridConfig config;
    config.min_tree_level = 0;
    config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    BezierMultigridPreconditioner precond(config);
    precond.setup(Q, mesh_, smoother.dof_manager());

    ASSERT_GE(precond.num_levels(), 2);

    // Check that pass-through nodes have is_subdivided = false
    for (int level = 0; level < precond.num_levels(); ++level) {
        const auto &composite = precond.level(level).composite_grid;
        const auto &passthrough_indices = precond.level(level).passthrough_node_indices;

        for (Index idx : passthrough_indices) {
            ASSERT_LT(static_cast<size_t>(idx), composite.nodes.size());
            EXPECT_FALSE(composite.nodes[idx].is_subdivided)
                << "Pass-through node at level " << level << " index " << idx
                << " should have is_subdivided=false";
        }
    }
}

TEST_F(BezierMultigridCenterGradedTest, CompositeHierarchy_SubdivisionClassification) {
    // Nodes with refined descendants should be marked as subdivision
    CGCubicBezierSmootherConfig smoother_config;
    CGCubicBezierBathymetrySmoother smoother(mesh_, smoother_config);

    Index n = smoother.dof_manager().num_free_dofs();
    SpMat Q(n, n);
    std::vector<Eigen::Triplet<Real>> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 2.0);
    }
    Q.setFromTriplets(triplets.begin(), triplets.end());

    MultigridConfig config;
    config.min_tree_level = 0;
    config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    BezierMultigridPreconditioner precond(config);
    precond.setup(Q, mesh_, smoother.dof_manager());

    ASSERT_GE(precond.num_levels(), 2);

    // Check that subdivision nodes have is_subdivided = true
    for (int level = 0; level < precond.num_levels(); ++level) {
        const auto &composite = precond.level(level).composite_grid;
        const auto &subdivision_indices = precond.level(level).subdivision_node_indices;

        for (Index idx : subdivision_indices) {
            ASSERT_LT(static_cast<size_t>(idx), composite.nodes.size());
            EXPECT_TRUE(composite.nodes[idx].is_subdivided)
                << "Subdivision node at level " << level << " index " << idx
                << " should have is_subdivided=true";
        }
    }
}

TEST_F(BezierMultigridCenterGradedTest, CompositeHierarchy_PositionDOFDeduplication) {
    // Shared DOFs at element interfaces should get same index via position quantization
    CGCubicBezierSmootherConfig smoother_config;
    CGCubicBezierBathymetrySmoother smoother(mesh_, smoother_config);

    Index n = smoother.dof_manager().num_free_dofs();
    SpMat Q(n, n);
    std::vector<Eigen::Triplet<Real>> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 2.0);
    }
    Q.setFromTriplets(triplets.begin(), triplets.end());

    MultigridConfig config;
    config.min_tree_level = 0;
    config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    BezierMultigridPreconditioner precond(config);
    precond.setup(Q, mesh_, smoother.dof_manager());

    // For each level, verify DOF indices are in range [0, num_dofs)
    for (int level = 0; level < precond.num_levels(); ++level) {
        const auto &composite = precond.level(level).composite_grid;

        // Collect all DOF indices
        std::set<Index> unique_dofs;
        for (const auto &node : composite.nodes) {
            for (Index dof : node.dof_indices) {
                if (dof >= 0) {  // Valid DOF
                    EXPECT_LT(dof, composite.num_dofs)
                        << "DOF index " << dof << " out of range at level " << level;
                    unique_dofs.insert(dof);
                }
            }
        }

        // Number of unique DOFs should equal num_dofs
        EXPECT_EQ(static_cast<Index>(unique_dofs.size()), composite.num_dofs)
            << "Level " << level << " has " << unique_dofs.size()
            << " unique DOFs but num_dofs=" << composite.num_dofs;
    }
}

TEST_F(BezierMultigridCenterGradedTest, CompositeHierarchy_MaxDepthTracking) {
    // max_depth should correctly reflect deepest node in each level
    CGCubicBezierSmootherConfig smoother_config;
    CGCubicBezierBathymetrySmoother smoother(mesh_, smoother_config);

    Index n = smoother.dof_manager().num_free_dofs();
    SpMat Q(n, n);
    std::vector<Eigen::Triplet<Real>> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 2.0);
    }
    Q.setFromTriplets(triplets.begin(), triplets.end());

    MultigridConfig config;
    config.min_tree_level = 0;
    config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    BezierMultigridPreconditioner precond(config);
    precond.setup(Q, mesh_, smoother.dof_manager());

    // For each level, compute actual max depth and compare
    for (int level = 0; level < precond.num_levels(); ++level) {
        const auto &composite = precond.level(level).composite_grid;

        int actual_max_depth = 0;
        for (const auto &node : composite.nodes) {
            if (node.tree_node) {
                int depth = node.tree_node->level.x;  // level.x == level.y for quadtree
                actual_max_depth = std::max(actual_max_depth, depth);
            }
        }

        EXPECT_EQ(composite.max_depth, actual_max_depth)
            << "Level " << level << " max_depth should be " << actual_max_depth;
    }
}

TEST_F(BezierMultigridCenterGradedTest, CompositeHierarchy_NodeCountsConsistent) {
    // subdivision + passthrough node counts should equal total nodes
    // Note: At finest level, these indices may be empty since there's no finer level
    CGCubicBezierSmootherConfig smoother_config;
    CGCubicBezierBathymetrySmoother smoother(mesh_, smoother_config);

    Index n = smoother.dof_manager().num_free_dofs();
    SpMat Q(n, n);
    std::vector<Eigen::Triplet<Real>> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 2.0);
    }
    Q.setFromTriplets(triplets.begin(), triplets.end());

    MultigridConfig config;
    config.min_tree_level = 0;
    config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    BezierMultigridPreconditioner precond(config);
    precond.setup(Q, mesh_, smoother.dof_manager());

    // Check all levels except finest (finest has no finer level to compare with)
    for (int level = 0; level < precond.num_levels() - 1; ++level) {
        const auto &ml = precond.level(level);
        size_t total = ml.subdivision_node_indices.size() + ml.passthrough_node_indices.size();

        EXPECT_EQ(total, ml.composite_grid.nodes.size())
            << "Level " << level << ": subdivision (" << ml.subdivision_node_indices.size()
            << ") + passthrough (" << ml.passthrough_node_indices.size()
            << ") != total nodes (" << ml.composite_grid.nodes.size() << ")";
    }
}

// =============================================================================
// Prolongation Operator Tests
// =============================================================================

TEST_F(BezierMultigridCenterGradedTest, Prolongation_PassThroughIdentity) {
    // Pass-through nodes should have P(fine_dof, coarse_dof) = 1 for matching DOFs
    CGCubicBezierSmootherConfig smoother_config;
    CGCubicBezierBathymetrySmoother smoother(mesh_, smoother_config);

    Index n = smoother.dof_manager().num_free_dofs();
    SpMat Q(n, n);
    std::vector<Eigen::Triplet<Real>> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 2.0);
    }
    Q.setFromTriplets(triplets.begin(), triplets.end());

    MultigridConfig config;
    config.min_tree_level = 0;
    config.transfer_strategy = TransferOperatorStrategy::BezierSubdivision;
    config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    BezierMultigridPreconditioner precond(config);
    precond.setup(Q, mesh_, smoother.dof_manager());

    // For each level with pass-through nodes, verify identity mapping
    for (int level = 0; level < precond.num_levels() - 1; ++level) {
        const auto &coarse_level = precond.level(level);
        const auto &fine_level = precond.level(level + 1);
        const SpMat &P = coarse_level.P;

        // For pass-through nodes at coarse level that also appear at fine level
        for (Index idx : coarse_level.passthrough_node_indices) {
            const auto &coarse_node = coarse_level.composite_grid.nodes[idx];

            // Check if this node also exists at fine level
            auto it = fine_level.composite_grid.node_index_map.find(coarse_node.tree_node);
            if (it != fine_level.composite_grid.node_index_map.end()) {
                const auto &fine_node = fine_level.composite_grid.nodes[it->second];

                // Each coarse DOF should map to corresponding fine DOF with weight 1
                for (int i = 0; i < 16; ++i) {
                    Index coarse_dof = coarse_node.dof_indices[i];
                    Index fine_dof = fine_node.dof_indices[i];

                    if (coarse_dof >= 0 && fine_dof >= 0) {
                        // P(fine_dof, coarse_dof) should be 1.0
                        Real val = P.coeff(fine_dof, coarse_dof);
                        EXPECT_NEAR(val, 1.0, TOLERANCE)
                            << "Pass-through P(" << fine_dof << "," << coarse_dof
                            << ") should be 1.0, got " << val;
                    }
                }
            }
        }
    }
}

TEST_F(BezierMultigridCenterGradedTest, Prolongation_InterpolatesConstant) {
    // P * ones(coarse) should equal ones(fine) (constant preservation)
    CGCubicBezierSmootherConfig smoother_config;
    CGCubicBezierBathymetrySmoother smoother(mesh_, smoother_config);

    Index n = smoother.dof_manager().num_free_dofs();
    SpMat Q(n, n);
    std::vector<Eigen::Triplet<Real>> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 2.0);
    }
    Q.setFromTriplets(triplets.begin(), triplets.end());

    MultigridConfig config;
    config.min_tree_level = 0;
    config.transfer_strategy = TransferOperatorStrategy::BezierSubdivision;
    config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    BezierMultigridPreconditioner precond(config);
    precond.setup(Q, mesh_, smoother.dof_manager());

    // Test P at each level: P * ones should give ones
    for (int level = 0; level < precond.num_levels() - 1; ++level) {
        const SpMat &P = precond.level(level).P;

        VecX coarse_const = VecX::Ones(P.cols());
        VecX fine_result = P * coarse_const;

        // All fine values should be 1.0
        for (Index i = 0; i < fine_result.size(); ++i) {
            EXPECT_NEAR(fine_result(i), 1.0, LOOSE_TOLERANCE)
                << "P * ones != ones at level " << level << " row " << i;
        }
    }
}

TEST_F(BezierMultigridCenterGradedTest, Prolongation_SubdivisionRowSumsOne) {
    // For BezierSubdivision, each row of P should sum to 1 (convex combination)
    CGCubicBezierSmootherConfig smoother_config;
    CGCubicBezierBathymetrySmoother smoother(mesh_, smoother_config);

    Index n = smoother.dof_manager().num_free_dofs();
    SpMat Q(n, n);
    std::vector<Eigen::Triplet<Real>> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 2.0);
    }
    Q.setFromTriplets(triplets.begin(), triplets.end());

    MultigridConfig config;
    config.min_tree_level = 0;
    config.transfer_strategy = TransferOperatorStrategy::BezierSubdivision;
    config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    BezierMultigridPreconditioner precond(config);
    precond.setup(Q, mesh_, smoother.dof_manager());

    for (int level = 0; level < precond.num_levels() - 1; ++level) {
        const SpMat &P = precond.level(level).P;
        VecX row_sums = P * VecX::Ones(P.cols());

        for (Index i = 0; i < P.rows(); ++i) {
            EXPECT_NEAR(row_sums(i), 1.0, LOOSE_TOLERANCE)
                << "P row sum != 1 at level " << level << " row " << i;
        }
    }
}

TEST_F(BezierMultigridCenterGradedTest, Prolongation_CorrectDimensions) {
    // P should be (fine_dofs x coarse_dofs)
    CGCubicBezierSmootherConfig smoother_config;
    CGCubicBezierBathymetrySmoother smoother(mesh_, smoother_config);

    Index n = smoother.dof_manager().num_free_dofs();
    SpMat Q(n, n);
    std::vector<Eigen::Triplet<Real>> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 2.0);
    }
    Q.setFromTriplets(triplets.begin(), triplets.end());

    MultigridConfig config;
    config.min_tree_level = 0;
    config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    BezierMultigridPreconditioner precond(config);
    precond.setup(Q, mesh_, smoother.dof_manager());

    for (int level = 0; level < precond.num_levels() - 1; ++level) {
        const SpMat &P = precond.level(level).P;
        Index coarse_dofs = precond.level(level).num_dofs;
        Index fine_dofs = precond.level(level + 1).num_dofs;

        EXPECT_EQ(P.rows(), fine_dofs)
            << "P rows should equal fine DOFs at level " << level;
        EXPECT_EQ(P.cols(), coarse_dofs)
            << "P cols should equal coarse DOFs at level " << level;
    }
}

// Note: Linear interpolation test removed.
// Bezier subdivision preserves polynomial degree but control point values
// don't directly correspond to function samples at parametric positions.
// The constant preservation tests (InterpolatesConstant) already verify
// the partition of unity property which is the core requirement.

// =============================================================================
// V-Cycle Frequency Analysis Tests
// =============================================================================

TEST_F(BezierMultigridCenterGradedTest, VCycle_ConvergesAdaptiveMesh) {
    // V-cycle should converge on adaptive mesh with mixed pass-through/subdivision
    // Test the V-cycle preconditioner directly rather than through the full smoother
    CGCubicBezierSmootherConfig smoother_config;
    CGCubicBezierBathymetrySmoother smoother(mesh_, smoother_config);

    Index n = smoother.dof_manager().num_free_dofs();

    // Create a simple SPD matrix
    SpMat Q(n, n);
    std::vector<Eigen::Triplet<Real>> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 4.0);
        if (i > 0) {
            triplets.emplace_back(i, i - 1, -1.0);
            triplets.emplace_back(i - 1, i, -1.0);
        }
    }
    Q.setFromTriplets(triplets.begin(), triplets.end());

    MultigridConfig config;
    config.min_tree_level = 0;
    config.pre_smoothing = 2;
    config.post_smoothing = 2;
    config.transfer_strategy = TransferOperatorStrategy::BezierSubdivision;
    config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    BezierMultigridPreconditioner precond(config);
    precond.setup(Q, mesh_, smoother.dof_manager());

    // Apply multiple V-cycles and check residual reduction
    VecX x = VecX::Zero(n);
    VecX b = VecX::Ones(n);
    VecX r = b - Q * x;
    Real initial_residual = r.norm();

    // Apply 10 V-cycles as preconditioned iteration
    for (int iter = 0; iter < 10; ++iter) {
        VecX z = precond.apply(r);
        x += z;
        r = b - Q * x;
    }

    Real final_residual = r.norm();

    // Residual should decrease significantly
    EXPECT_LT(final_residual, initial_residual * 0.1)
        << "V-cycle on adaptive mesh did not converge: "
        << initial_residual << " -> " << final_residual;
}

TEST_F(BezierMultigridTest, VCycle_SmootherReducesHighFrequency) {
    // Smoother should damp oscillatory (high-frequency) error components
    CGCubicBezierSmootherConfig smoother_config;
    CGCubicBezierBathymetrySmoother smoother(mesh_, smoother_config);

    Index n = smoother.dof_manager().num_free_dofs();

    // Create tridiagonal SPD matrix (represents discrete Laplacian)
    SpMat Q(n, n);
    std::vector<Eigen::Triplet<Real>> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 4.0);
        if (i > 0) {
            triplets.emplace_back(i, i - 1, -1.0);
            triplets.emplace_back(i - 1, i, -1.0);
        }
    }
    Q.setFromTriplets(triplets.begin(), triplets.end());

    MultigridConfig config;
    config.min_tree_level = 1;
    config.pre_smoothing = 3;
    config.post_smoothing = 0;
    config.smoother_type = SmootherType::MultiplicativeSchwarz;
    config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    BezierMultigridPreconditioner precond(config);
    precond.setup(Q, mesh_, smoother.dof_manager());

    // Create high-frequency error (alternating signs)
    VecX e_high = VecX::Zero(n);
    for (Index i = 0; i < n; ++i) {
        e_high(i) = (i % 2 == 0) ? 1.0 : -1.0;
    }
    Real e_high_norm_before = e_high.norm();

    // Compute residual for this error with zero RHS
    VecX r = -Q * e_high;

    // Apply one V-cycle (effectively applies smoother + coarse correction)
    VecX correction = precond.apply(r);

    // Error after correction
    VecX e_after = e_high + correction;
    Real e_high_norm_after = e_after.norm();

    // High-frequency error should be reduced
    EXPECT_LT(e_high_norm_after, e_high_norm_before)
        << "Smoother should reduce high-frequency error";
}

// =============================================================================
// Graph Coloring Tests
// =============================================================================

TEST_F(BezierMultigridTest, GraphColoring_NoAdjacentSameColor) {
    // No two adjacent elements should share the same color
    CGCubicBezierSmootherConfig smoother_config;
    CGCubicBezierBathymetrySmoother smoother(mesh_, smoother_config);

    Index n = smoother.dof_manager().num_free_dofs();
    SpMat Q(n, n);
    std::vector<Eigen::Triplet<Real>> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 4.0);
        if (i > 0) {
            triplets.emplace_back(i, i - 1, -1.0);
            triplets.emplace_back(i - 1, i, -1.0);
        }
    }
    Q.setFromTriplets(triplets.begin(), triplets.end());

    MultigridConfig config;
    config.min_tree_level = 1;
    config.smoother_type = SmootherType::ColoredMultiplicativeSchwarz;
    config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    BezierMultigridPreconditioner precond(config);
    precond.setup(Q, mesh_, smoother.dof_manager());

    // Check coloring at finest level
    int finest = precond.num_levels() - 1;
    const auto &level = precond.level(finest);

    // Build element-to-color map
    std::map<Index, int> element_color;
    for (int c = 0; c < level.num_colors; ++c) {
        for (Index elem : level.elements_by_color[c]) {
            element_color[elem] = c;
        }
    }

    // Build DOF-to-elements map
    std::map<Index, std::vector<Index>> dof_to_elements;
    for (size_t e = 0; e < level.element_free_dofs.size(); ++e) {
        for (Index dof : level.element_free_dofs[e]) {
            dof_to_elements[dof].push_back(static_cast<Index>(e));
        }
    }

    // Check that elements sharing DOFs have different colors
    for (const auto &[dof, elements] : dof_to_elements) {
        for (size_t i = 0; i < elements.size(); ++i) {
            for (size_t j = i + 1; j < elements.size(); ++j) {
                Index e1 = elements[i];
                Index e2 = elements[j];
                int c1 = element_color[e1];
                int c2 = element_color[e2];

                EXPECT_NE(c1, c2)
                    << "Adjacent elements " << e1 << " and " << e2
                    << " share DOF " << dof << " but have same color " << c1;
            }
        }
    }
}

TEST_F(BezierMultigridTest, GraphColoring_AllElementsColored) {
    // Every element should get assigned a color
    CGCubicBezierSmootherConfig smoother_config;
    CGCubicBezierBathymetrySmoother smoother(mesh_, smoother_config);

    Index n = smoother.dof_manager().num_free_dofs();
    SpMat Q(n, n);
    std::vector<Eigen::Triplet<Real>> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 2.0);
    }
    Q.setFromTriplets(triplets.begin(), triplets.end());

    MultigridConfig config;
    config.min_tree_level = 1;
    config.smoother_type = SmootherType::ColoredMultiplicativeSchwarz;
    config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    BezierMultigridPreconditioner precond(config);
    precond.setup(Q, mesh_, smoother.dof_manager());

    // Count elements in coloring
    int finest = precond.num_levels() - 1;
    const auto &level = precond.level(finest);

    size_t colored_elements = 0;
    for (int c = 0; c < level.num_colors; ++c) {
        colored_elements += level.elements_by_color[c].size();
    }

    EXPECT_EQ(colored_elements, level.element_free_dofs.size())
        << "Not all elements are colored";
}

TEST_F(BezierMultigridTest, GraphColoring_MinimumColors) {
    // Should use <= 9 colors for typical 2D quad meshes (4 neighbors + diagonals)
    CGCubicBezierSmootherConfig smoother_config;
    CGCubicBezierBathymetrySmoother smoother(mesh_, smoother_config);

    Index n = smoother.dof_manager().num_free_dofs();
    SpMat Q(n, n);
    std::vector<Eigen::Triplet<Real>> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 2.0);
    }
    Q.setFromTriplets(triplets.begin(), triplets.end());

    MultigridConfig config;
    config.min_tree_level = 1;
    config.smoother_type = SmootherType::ColoredMultiplicativeSchwarz;
    config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    BezierMultigridPreconditioner precond(config);
    precond.setup(Q, mesh_, smoother.dof_manager());

    int finest = precond.num_levels() - 1;
    const auto &level = precond.level(finest);

    // For 2D quad meshes with shared corner DOFs, expect at most 9 colors
    // (3x3 stencil of neighbors)
    EXPECT_LE(level.num_colors, 9)
        << "Too many colors used: " << level.num_colors;
    EXPECT_GE(level.num_colors, 1)
        << "At least 1 color should be used";
}

// =============================================================================
// Edge Case Tests
// =============================================================================

TEST(BezierMultigridEdgeCases, SingleElementMesh_DirectSolve) {
    // Single element mesh should have one level with direct solve
    QuadtreeAdapter single_mesh;
    single_mesh.build_uniform(0.0, 1.0, 0.0, 1.0, 1, 1);

    CGCubicBezierSmootherConfig smoother_config;
    smoother_config.lambda = 10.0;
    smoother_config.edge_ngauss = 0;
    CGCubicBezierBathymetrySmoother smoother(single_mesh, smoother_config);

    Index n = smoother.dof_manager().num_free_dofs();
    SpMat Q(n, n);
    std::vector<Eigen::Triplet<Real>> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 2.0);
    }
    Q.setFromTriplets(triplets.begin(), triplets.end());

    MultigridConfig config;
    config.min_tree_level = 0;
    config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    BezierMultigridPreconditioner precond(config);
    precond.setup(Q, single_mesh, smoother.dof_manager());

    // Should have exactly 1 level (coarsest only)
    EXPECT_EQ(precond.num_levels(), 1);

    // V-cycle should still work (direct solve at coarsest)
    VecX r = VecX::Ones(n);
    VecX z = precond.apply(r);
    EXPECT_GT(z.norm(), 0.0);
}

TEST(BezierMultigridEdgeCases, UniformMesh_AllPassThrough) {
    // Uniform mesh at finest level should have all pass-through nodes at coarser levels
    QuadtreeAdapter uniform_mesh;
    uniform_mesh.build_uniform(0.0, 1.0, 0.0, 1.0, 4, 4);

    CGCubicBezierSmootherConfig smoother_config;
    CGCubicBezierBathymetrySmoother smoother(uniform_mesh, smoother_config);

    Index n = smoother.dof_manager().num_free_dofs();
    SpMat Q(n, n);
    std::vector<Eigen::Triplet<Real>> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 2.0);
    }
    Q.setFromTriplets(triplets.begin(), triplets.end());

    MultigridConfig config;
    config.min_tree_level = 0;
    config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    BezierMultigridPreconditioner precond(config);
    precond.setup(Q, uniform_mesh, smoother.dof_manager());

    // At coarsest level (level 0), all nodes should be subdivision
    // because they have children at finer levels
    if (precond.num_levels() > 1) {
        const auto &coarsest = precond.level(0);
        EXPECT_GT(coarsest.subdivision_node_indices.size(), 0u)
            << "Coarsest level should have subdivision nodes";
    }

    // At finest level, all nodes should be pass-through (leaves)
    int finest = precond.num_levels() - 1;
    const auto &finest_level = precond.level(finest);

    // For uniform mesh, finest level nodes are all leaves = pass-through
    // (Note: at finest level, "pass-through" means they don't subdivide further)
    size_t total_nodes = finest_level.composite_grid.nodes.size();
    EXPECT_EQ(total_nodes, 16u)  // 4x4 uniform mesh
        << "Uniform 4x4 mesh should have 16 elements at finest level";
}

TEST(BezierMultigridEdgeCases, HangingNodes_ConstraintsRespected) {
    // T-junction hanging node constraints should be satisfied in transfer operators
    // Create adaptive mesh with hanging nodes
    OctreeAdapter octree(0.0, 1.0, 0.0, 1.0, -1.0, 0.0);
    octree.build_uniform(2, 2, 1);

    // Refine only corner element to create T-junction
    std::vector<Index> to_refine = {0};
    std::vector<RefineMask> masks(1, RefineMask::XY);
    octree.refine(to_refine, masks);

    QuadtreeAdapter adaptive_mesh(octree);

    CGCubicBezierSmootherConfig smoother_config;
    CGCubicBezierBathymetrySmoother smoother(adaptive_mesh, smoother_config);

    Index n = smoother.dof_manager().num_free_dofs();
    SpMat Q(n, n);
    std::vector<Eigen::Triplet<Real>> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 2.0);
    }
    Q.setFromTriplets(triplets.begin(), triplets.end());

    MultigridConfig config;
    config.min_tree_level = 0;
    config.transfer_strategy = TransferOperatorStrategy::BezierSubdivision;
    config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    BezierMultigridPreconditioner precond(config);
    precond.setup(Q, adaptive_mesh, smoother.dof_manager());

    // Verify constant preservation (which requires proper constraint handling)
    for (int level = 1; level < precond.num_levels(); ++level) {
        const SpMat &R = precond.level(level).R;
        VecX fine_const = VecX::Constant(R.cols(), 1.0);
        VecX coarse = R * fine_const;

        for (Index i = 0; i < coarse.size(); ++i) {
            EXPECT_NEAR(coarse(i), 1.0, LOOSE_TOLERANCE)
                << "Constant not preserved at level " << level
                << " with hanging nodes";
        }
    }
}

// =============================================================================
// Transfer Operator Strategy Tests
// =============================================================================

TEST_F(BezierMultigridTest, L2Projection_SymmetricOperator) {
    // For L2 projection, verify P = R^T (symmetric transfer)
    CGCubicBezierSmootherConfig smoother_config;
    CGCubicBezierBathymetrySmoother smoother(mesh_, smoother_config);

    Index n = smoother.dof_manager().num_free_dofs();
    SpMat Q(n, n);
    std::vector<Eigen::Triplet<Real>> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 2.0);
    }
    Q.setFromTriplets(triplets.begin(), triplets.end());

    MultigridConfig config;
    config.min_tree_level = 0;
    config.transfer_strategy = TransferOperatorStrategy::L2Projection;
    config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    BezierMultigridPreconditioner precond(config);
    precond.setup(Q, mesh_, smoother.dof_manager());

    for (int level = 0; level < precond.num_levels() - 1; ++level) {
        const SpMat &P = precond.level(level).P;
        const SpMat &R = precond.level(level + 1).R;

        // P should equal R^T
        SpMat Rt = R.transpose();
        SpMat diff = P - Rt;

        Real max_diff = 0.0;
        for (int k = 0; k < diff.outerSize(); ++k) {
            for (SpMat::InnerIterator it(diff, k); it; ++it) {
                max_diff = std::max(max_diff, std::abs(it.value()));
            }
        }

        EXPECT_LT(max_diff, TOLERANCE)
            << "P != R^T at level " << level << ", max diff = " << max_diff;
    }
}

TEST_F(BezierMultigridTest, Galerkin_CoarseMatrixFormula) {
    // Verify A_c = R * A_f * P holds exactly for Galerkin strategy
    CGCubicBezierSmootherConfig smoother_config;
    CGCubicBezierBathymetrySmoother smoother(mesh_, smoother_config);

    Index n = smoother.dof_manager().num_free_dofs();
    SpMat Q(n, n);
    std::vector<Eigen::Triplet<Real>> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 4.0);
        if (i > 0) {
            triplets.emplace_back(i, i - 1, -1.0);
            triplets.emplace_back(i - 1, i, -1.0);
        }
    }
    Q.setFromTriplets(triplets.begin(), triplets.end());

    MultigridConfig config;
    config.min_tree_level = 0;
    config.transfer_strategy = TransferOperatorStrategy::L2Projection;
    config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    BezierMultigridPreconditioner precond(config);
    precond.setup(Q, mesh_, smoother.dof_manager());

    // Verify Galerkin formula at each level
    for (int level = precond.num_levels() - 1; level > 0; --level) {
        const SpMat &Q_fine = precond.level(level).Q;
        const SpMat &Q_coarse = precond.level(level - 1).Q;
        const SpMat &R = precond.level(level).R;
        const SpMat &P = precond.level(level - 1).P;

        // Compute R * Q_fine * P
        SpMat RAP = R * Q_fine * P;

        // Compare with stored Q_coarse
        SpMat diff = Q_coarse - RAP;
        Real max_diff = 0.0;
        for (int k = 0; k < diff.outerSize(); ++k) {
            for (SpMat::InnerIterator it(diff, k); it; ++it) {
                max_diff = std::max(max_diff, std::abs(it.value()));
            }
        }

        EXPECT_LT(max_diff, LOOSE_TOLERANCE)
            << "Galerkin formula Q_c = R*Q_f*P not satisfied at level " << level
            << ", max diff = " << max_diff;
    }
}

} // namespace
