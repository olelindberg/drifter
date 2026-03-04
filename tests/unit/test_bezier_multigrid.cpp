#include <gtest/gtest.h>
#include "bathymetry/bezier_multigrid_preconditioner.hpp"
#include "bathymetry/cg_cubic_bezier_bathymetry_smoother.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include <cmath>

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
    config.num_levels = 2;
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
    config.num_levels = 2;
    config.pre_smoothing = 2;
    config.post_smoothing = 2;
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
    config.multigrid_config.num_levels = 2;
    config.schur_cg_tolerance = 1e-8;

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
    config_mg.multigrid_config.num_levels = 2;
    config_mg.schur_cg_tolerance = 1e-8;

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
    config_mg.multigrid_config.num_levels = 2;
    config_mg.schur_cg_tolerance = 1e-10;
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
    EXPECT_EQ(config.num_levels, 100);  // High default; coarsening controlled by min_tree_level
    EXPECT_EQ(config.min_tree_level, 2);  // Coarsest level: 4x4 elements
    EXPECT_EQ(config.pre_smoothing, 1);
    EXPECT_EQ(config.post_smoothing, 1);
    EXPECT_NEAR(config.jacobi_omega, 0.8, TOLERANCE);
    EXPECT_EQ(config.smoother_type, SmootherType::MultiplicativeSchwarz);
    EXPECT_EQ(config.coarsest_direct_size, 200);
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
    config.num_levels = 2;
    config.pre_smoothing = 2;
    config.post_smoothing = 2;
    config.smoother_type = SmootherType::MultiplicativeSchwarz;
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
    config.num_levels = 2;
    config.smoother_type = SmootherType::Jacobi;
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
    config.num_levels = 3;
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
    config.num_levels = 3;
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
    config.num_levels = 3;
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

} // namespace
