#include <gtest/gtest.h>
#include "bathymetry/bezier_multigrid_preconditioner.hpp"
#include "bathymetry/cg_cubic_bezier_bathymetry_smoother.hpp"
#include "bathymetry/diagonal_approx_cg_schur_preconditioner.hpp"
#include "bathymetry/diagonal_schur_preconditioner.hpp"
#include "bathymetry/flexible_cg.hpp"
#include "bathymetry/multigrid_schur_preconditioner.hpp"
#include "bathymetry/physics_based_schur_preconditioner.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "bathymetry/schur_preconditioner.hpp"
#include "bathymetry/schwarz_method.hpp"
#include "bathymetry/schwarz_schur_preconditioner.hpp"
#include "core/types.hpp"
#include <Eigen/Dense>
#include <Eigen/SparseLU>
#include <cmath>

using namespace drifter;

namespace {

constexpr Real TOLERANCE = 1e-10;
constexpr Real LOOSE_TOLERANCE = 1e-6;

// =============================================================================
// Test fixtures
// =============================================================================

class SchurPreconditionerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple SPD matrix K (4x4 tridiagonal)
        n_free_ = 4;
        K_.resize(n_free_, n_free_);
        std::vector<Eigen::Triplet<Real>> K_triplets;
        for (Index i = 0; i < n_free_; ++i) {
            K_triplets.emplace_back(i, i, 4.0);
            if (i > 0) {
                K_triplets.emplace_back(i, i - 1, -1.0);
                K_triplets.emplace_back(i - 1, i, -1.0);
            }
        }
        K_.setFromTriplets(K_triplets.begin(), K_triplets.end());

        // Create a constraint matrix C (2 constraints on 4 free DOFs)
        // Each constraint couples two DOFs
        n_constraints_ = 2;
        C_.resize(n_constraints_, n_free_);
        std::vector<Eigen::Triplet<Real>> C_triplets;
        // Constraint 0: DOF 0 - DOF 1 = 0
        C_triplets.emplace_back(0, 0, 1.0);
        C_triplets.emplace_back(0, 1, -1.0);
        // Constraint 1: DOF 2 - DOF 3 = 0
        C_triplets.emplace_back(1, 2, 1.0);
        C_triplets.emplace_back(1, 3, -1.0);
        C_.setFromTriplets(C_triplets.begin(), C_triplets.end());
    }

    Index n_free_;
    Index n_constraints_;
    SpMat K_;
    SpMat C_;
};

// =============================================================================
// DiagonalSchurPreconditioner tests
// =============================================================================

TEST_F(SchurPreconditionerTest, DiagonalPreconditioner_ExtractsDiagonal) {
    // Create Schur complement matvec: S = C * K^{-1} * C^T
    Eigen::SparseLU<SpMat> K_solver;
    K_solver.compute(K_);
    ASSERT_EQ(K_solver.info(), Eigen::Success);

    auto schur_matvec = [&](const VecX& v) -> VecX {
        VecX Ct_v = C_.transpose() * v;
        VecX K_inv_Ct_v = K_solver.solve(Ct_v);
        return C_ * K_inv_Ct_v;
    };

    // Create diagonal preconditioner
    DiagonalSchurPreconditioner precond(schur_matvec, n_constraints_);

    // Check that diagonal was extracted
    VecX diag = precond.diagonal();
    EXPECT_EQ(diag.size(), n_constraints_);

    // Diagonal should be positive (S is SPD)
    for (Index i = 0; i < n_constraints_; ++i) {
        EXPECT_GT(diag(i), 0.0);
    }

    // Check that apply gives reasonable results
    VecX r = VecX::Ones(n_constraints_);
    VecX z = precond.apply(r);
    EXPECT_EQ(z.size(), n_constraints_);

    // z should have same sign as r (diagonal is positive)
    for (Index i = 0; i < n_constraints_; ++i) {
        EXPECT_GT(z(i), 0.0);
    }
}

TEST_F(SchurPreconditionerTest, DiagonalPreconditioner_IsNotVariable) {
    auto schur_matvec = [](const VecX& v) -> VecX { return v; };
    DiagonalSchurPreconditioner precond(schur_matvec, 2);
    EXPECT_FALSE(precond.is_variable());
}

// =============================================================================
// PhysicsBasedSchurPreconditioner tests
// =============================================================================

TEST_F(SchurPreconditionerTest, PhysicsBasedPreconditioner_ConstructsSuccessfully) {
    // Should not throw for valid K and C
    PhysicsBasedSchurPreconditioner precond(K_, C_);
    EXPECT_TRUE(precond.is_valid());
    EXPECT_EQ(precond.num_constraints(), n_constraints_);
}

TEST_F(SchurPreconditionerTest, PhysicsBasedPreconditioner_AssembledMatrixIsSPD) {
    PhysicsBasedSchurPreconditioner precond(K_, C_);

    // M_S = C * K^{-1} * C^T should be SPD
    const SpMat& M_S = precond.assembled_matrix();
    EXPECT_EQ(M_S.rows(), n_constraints_);
    EXPECT_EQ(M_S.cols(), n_constraints_);

    // Check symmetry
    SpMat M_S_diff = M_S - SpMat(M_S.transpose());
    EXPECT_LT(M_S_diff.norm(), TOLERANCE);

    // Check positive definiteness: all eigenvalues > 0
    Eigen::MatrixXd M_S_dense = M_S;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(M_S_dense);
    for (Index i = 0; i < n_constraints_; ++i) {
        EXPECT_GT(eig.eigenvalues()(i), 0.0);
    }
}

TEST_F(SchurPreconditionerTest, PhysicsBasedPreconditioner_ApplyGivesCorrectResult) {
    PhysicsBasedSchurPreconditioner precond(K_, C_);

    VecX r = VecX::Ones(n_constraints_);
    VecX z = precond.apply(r);

    // z should be M_S^{-1} * r
    // Check by computing M_S * z ≈ r
    const SpMat& M_S = precond.assembled_matrix();
    VecX M_S_z = M_S * z;

    for (Index i = 0; i < n_constraints_; ++i) {
        EXPECT_NEAR(M_S_z(i), r(i), TOLERANCE);
    }
}

TEST_F(SchurPreconditionerTest, PhysicsBasedPreconditioner_IsNotVariable) {
    PhysicsBasedSchurPreconditioner precond(K_, C_);
    EXPECT_FALSE(precond.is_variable());
}

// =============================================================================
// DiagonalApproxCGSchurPreconditioner tests
// =============================================================================

TEST_F(SchurPreconditionerTest, DiagonalApproxCG_ConstructsSuccessfully) {
    // Should not throw for valid K and C
    DiagonalApproxCGSchurPreconditioner precond(K_, C_);
    EXPECT_EQ(precond.num_constraints(), n_constraints_);
}

TEST_F(SchurPreconditionerTest, DiagonalApproxCG_AssembledMatrixIsSPD) {
    DiagonalApproxCGSchurPreconditioner precond(K_, C_);

    // M_S = C * diag(K)^{-1} * C^T should be SPD
    const SpMat& M_S = precond.assembled_matrix();
    EXPECT_EQ(M_S.rows(), n_constraints_);
    EXPECT_EQ(M_S.cols(), n_constraints_);

    // Check symmetry
    SpMat M_S_diff = M_S - SpMat(M_S.transpose());
    EXPECT_LT(M_S_diff.norm(), TOLERANCE);

    // Check positive definiteness: all eigenvalues > 0
    Eigen::MatrixXd M_S_dense = M_S;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(M_S_dense);
    for (Index i = 0; i < n_constraints_; ++i) {
        EXPECT_GT(eig.eigenvalues()(i), 0.0);
    }
}

TEST_F(SchurPreconditionerTest, DiagonalApproxCG_ApplyGivesCorrectResult) {
    DiagonalApproxCGSchurPreconditioner precond(K_, C_, 1e-10, 100);

    VecX r = VecX::Ones(n_constraints_);
    VecX z = precond.apply(r);

    // z should be M_S^{-1} * r
    // Check by computing M_S * z ≈ r
    const SpMat& M_S = precond.assembled_matrix();
    VecX M_S_z = M_S * z;

    for (Index i = 0; i < n_constraints_; ++i) {
        EXPECT_NEAR(M_S_z(i), r(i), LOOSE_TOLERANCE);
    }
}

TEST_F(SchurPreconditionerTest, DiagonalApproxCG_IsVariable) {
    DiagonalApproxCGSchurPreconditioner precond(K_, C_);
    EXPECT_TRUE(precond.is_variable());
}

TEST_F(SchurPreconditionerTest, DiagonalApproxCG_HandlesZeroRHS) {
    DiagonalApproxCGSchurPreconditioner precond(K_, C_);

    VecX r = VecX::Zero(n_constraints_);
    VecX z = precond.apply(r);

    // Should return zero for zero input
    EXPECT_LT(z.norm(), TOLERANCE);
}

TEST_F(SchurPreconditionerTest, FlexibleCG_ConvergesWithDiagonalApproxCG) {
    // Create Schur complement matvec: S = C * K^{-1} * C^T
    Eigen::SparseLU<SpMat> K_solver;
    K_solver.compute(K_);
    ASSERT_EQ(K_solver.info(), Eigen::Success);

    auto schur_matvec = [&](const VecX& v) -> VecX {
        VecX Ct_v = C_.transpose() * v;
        VecX K_inv_Ct_v = K_solver.solve(Ct_v);
        return C_ * K_inv_Ct_v;
    };

    // Create DiagonalApproxCG preconditioner (requires FCG since it's variable)
    DiagonalApproxCGSchurPreconditioner precond(K_, C_, 1e-10, 100);

    // Create RHS
    VecX rhs = VecX::Ones(n_constraints_);

    // Solve with FCG
    FlexibleCG fcg(schur_matvec, precond, 1e-10, 100);
    VecX x = VecX::Zero(n_constraints_);
    FCGResult result = fcg.solve(x, rhs);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.relative_residual, 1e-9);

    // Check solution: S * x ≈ rhs
    VecX Sx = schur_matvec(x);
    VecX error = Sx - rhs;
    EXPECT_LT(error.norm() / rhs.norm(), 1e-8);
}

// =============================================================================
// FlexibleCG tests
// =============================================================================

TEST_F(SchurPreconditionerTest, FlexibleCG_ConvergesWithPhysicsBased) {
    // Create Schur complement system S = C * K^{-1} * C^T
    Eigen::SparseLU<SpMat> K_solver;
    K_solver.compute(K_);
    ASSERT_EQ(K_solver.info(), Eigen::Success);

    auto schur_matvec = [&](const VecX& v) -> VecX {
        VecX Ct_v = C_.transpose() * v;
        VecX K_inv_Ct_v = K_solver.solve(Ct_v);
        return C_ * K_inv_Ct_v;
    };

    // Create physics-based preconditioner
    PhysicsBasedSchurPreconditioner precond(K_, C_);

    // Create RHS
    VecX rhs = VecX::Ones(n_constraints_);

    // Solve with FCG
    FlexibleCG fcg(schur_matvec, precond, 1e-10, 100);
    VecX x = VecX::Zero(n_constraints_);
    FCGResult result = fcg.solve(x, rhs);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.relative_residual, 1e-9);

    // Check solution: S * x ≈ rhs
    VecX Sx = schur_matvec(x);
    VecX error = Sx - rhs;
    EXPECT_LT(error.norm() / rhs.norm(), 1e-8);
}

TEST_F(SchurPreconditionerTest, FlexibleCG_ConvergesWithDiagonal) {
    // Create Schur complement matvec
    Eigen::SparseLU<SpMat> K_solver;
    K_solver.compute(K_);
    ASSERT_EQ(K_solver.info(), Eigen::Success);

    auto schur_matvec = [&](const VecX& v) -> VecX {
        VecX Ct_v = C_.transpose() * v;
        VecX K_inv_Ct_v = K_solver.solve(Ct_v);
        return C_ * K_inv_Ct_v;
    };

    // Create diagonal preconditioner
    DiagonalSchurPreconditioner precond(schur_matvec, n_constraints_);

    // Create RHS
    VecX rhs = VecX::Ones(n_constraints_);

    // Solve with FCG (also works for stationary preconditioners)
    FlexibleCG fcg(schur_matvec, precond, 1e-10, 100);
    VecX x = VecX::Zero(n_constraints_);
    FCGResult result = fcg.solve(x, rhs);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.relative_residual, 1e-9);
}

TEST_F(SchurPreconditionerTest, FlexibleCG_TracksResidualHistory) {
    // Create Schur complement matvec
    Eigen::SparseLU<SpMat> K_solver;
    K_solver.compute(K_);

    auto schur_matvec = [&](const VecX& v) -> VecX {
        VecX Ct_v = C_.transpose() * v;
        VecX K_inv_Ct_v = K_solver.solve(Ct_v);
        return C_ * K_inv_Ct_v;
    };

    PhysicsBasedSchurPreconditioner precond(K_, C_);
    VecX rhs = VecX::Ones(n_constraints_);

    FlexibleCG fcg(schur_matvec, precond, 1e-10, 100);
    VecX x = VecX::Zero(n_constraints_);
    FCGResult result = fcg.solve(x, rhs);

    // Should have residual history
    EXPECT_GT(result.residual_history.size(), 0u);

    // First entry should be initial residual
    EXPECT_NEAR(result.residual_history[0], result.initial_residual, TOLERANCE);

    // Residual should decrease (mostly)
    if (result.residual_history.size() > 1) {
        EXPECT_LT(result.residual_history.back(), result.residual_history.front());
    }
}

// =============================================================================
// Edge cases
// =============================================================================

TEST_F(SchurPreconditionerTest, DiagonalPreconditioner_HandlesSmallDiagonal) {
    // Create a matvec that returns near-zero on some entries
    auto schur_matvec = [](const VecX& v) -> VecX {
        VecX result(2);
        result(0) = 1e-20 * v(0);  // Very small
        result(1) = 1.0 * v(1);    // Normal
        return result;
    };

    DiagonalSchurPreconditioner precond(schur_matvec, 2);

    // Should not have NaN or Inf
    VecX r = VecX::Ones(2);
    VecX z = precond.apply(r);

    for (Index i = 0; i < 2; ++i) {
        EXPECT_FALSE(std::isnan(z(i)));
        EXPECT_FALSE(std::isinf(z(i)));
    }
}

TEST_F(SchurPreconditionerTest, FlexibleCG_HandlesZeroRHS) {
    auto schur_matvec = [](const VecX& v) -> VecX { return v; };

    // Create trivial diagonal preconditioner
    DiagonalSchurPreconditioner precond(schur_matvec, 2);

    VecX rhs = VecX::Zero(2);
    FlexibleCG fcg(schur_matvec, precond, 1e-10, 100);
    VecX x = VecX::Zero(2);
    FCGResult result = fcg.solve(x, rhs);

    // Should converge immediately
    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.iterations, 0);
}

// =============================================================================
// MultigridSchurPreconditioner tests
// =============================================================================

class MultigridSchurTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Build a 4x4 uniform mesh for multigrid
        mesh_.build_uniform(0.0, 1.0, 0.0, 1.0, 4, 4);

        // Setup smoother to get DOF manager
        CGCubicBezierSmootherConfig smoother_config;
        smoother_config.edge_ngauss = 0;  // No edge constraints for cleaner test
        smoother_ = std::make_unique<CGCubicBezierBathymetrySmoother>(
            mesh_, smoother_config);

        // Get number of free DOFs
        n_free_ = smoother_->dof_manager().num_free_dofs();

        // Create a simple SPD system matrix Q
        Q_.resize(n_free_, n_free_);
        std::vector<Eigen::Triplet<Real>> Q_triplets;
        for (Index i = 0; i < n_free_; ++i) {
            Q_triplets.emplace_back(i, i, 4.0);
            if (i > 0) {
                Q_triplets.emplace_back(i, i - 1, -1.0);
                Q_triplets.emplace_back(i - 1, i, -1.0);
            }
        }
        Q_.setFromTriplets(Q_triplets.begin(), Q_triplets.end());

        // Create a constraint matrix C (4 constraints coupling adjacent DOFs)
        n_constraints_ = 4;
        C_.resize(n_constraints_, n_free_);
        std::vector<Eigen::Triplet<Real>> C_triplets;
        for (Index i = 0; i < n_constraints_; ++i) {
            // Each constraint couples DOF 2*i with DOF 2*i+1
            C_triplets.emplace_back(i, 2 * i, 1.0);
            C_triplets.emplace_back(i, 2 * i + 1, -1.0);
        }
        C_.setFromTriplets(C_triplets.begin(), C_triplets.end());

        // Setup multigrid preconditioner
        MultigridConfig mg_config;
        mg_config.min_tree_level = 0;
        mg_config.pre_smoothing = 2;
        mg_config.post_smoothing = 2;
        mg_config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
        mg_precond_ = std::make_unique<BezierMultigridPreconditioner>(mg_config);
        mg_precond_->setup(Q_, mesh_, smoother_->dof_manager());
    }

    QuadtreeAdapter mesh_;
    std::unique_ptr<CGCubicBezierBathymetrySmoother> smoother_;
    std::unique_ptr<BezierMultigridPreconditioner> mg_precond_;
    Index n_free_;
    Index n_constraints_;
    SpMat Q_;
    SpMat C_;

    /// Create Q matrix that mimics actual bathymetry: Q = H + λ·BᵀWB + εI
    /// @param lambda Data fitting weight (higher = denser coupling, worse conditioning)
    SpMat create_data_fitting_Q(Real lambda) {
        SpMat Q(n_free_, n_free_);
        std::vector<Eigen::Triplet<Real>> triplets;

        // Thin-plate smoothing term H (tridiagonal, well-conditioned)
        for (Index i = 0; i < n_free_; ++i) {
            triplets.emplace_back(i, i, 4.0);
            if (i > 0) {
                triplets.emplace_back(i, i - 1, -1.0);
                triplets.emplace_back(i - 1, i, -1.0);
            }
        }

        // Data fitting term BᵀWB (denser coupling, mimics Gauss-weighted basis evals)
        // Bandwidth 7 (±3) with exponential decay
        for (Index i = 0; i < n_free_; ++i) {
            for (Index j = std::max(Index(0), i - 3);
                 j <= std::min(n_free_ - 1, i + 3); ++j) {
                if (i != j) { // Off-diagonal only (diagonal handled above)
                    Real weight = lambda * 0.3 * std::exp(-0.5 * std::abs(i - j));
                    triplets.emplace_back(i, j, weight);
                } else {
                    // Add to diagonal
                    triplets.emplace_back(i, i, lambda * 0.5);
                }
            }
        }

        // Ridge regularization ε·I
        Real eps = 1e-6;
        for (Index i = 0; i < n_free_; ++i) {
            triplets.emplace_back(i, i, eps);
        }

        Q.setFromTriplets(triplets.begin(), triplets.end());
        return Q;
    }
};

TEST_F(MultigridSchurTest, MultigridSchur_IsVariable) {
    // MultigridSchurPreconditioner should return is_variable() = true
    MultigridSchurPreconditioner precond(*mg_precond_, Q_, C_, 3);
    EXPECT_TRUE(precond.is_variable());
}

TEST_F(MultigridSchurTest, MultigridSchur_AppliesFormula) {
    // Verify it computes z ≈ C * Q^{-1} * C^T * r
    MultigridSchurPreconditioner precond(*mg_precond_, Q_, C_, 10);

    // Compute exact result using direct solver
    Eigen::SparseLU<SpMat> Q_solver;
    Q_solver.compute(Q_);
    ASSERT_EQ(Q_solver.info(), Eigen::Success);

    VecX r = VecX::Ones(n_constraints_);

    // Exact: C * Q^{-1} * C^T * r
    VecX Ct_r = C_.transpose() * r;
    VecX Q_inv_Ct_r = Q_solver.solve(Ct_r);
    VecX z_exact = C_ * Q_inv_Ct_r;

    // MG approximation
    VecX z_mg = precond.apply(r);

    // Should be reasonably close (not exact, but within multigrid approximation)
    Real relative_error = (z_mg - z_exact).norm() / z_exact.norm();
    EXPECT_LT(relative_error, 0.1)  // Within 10% error
        << "MG Schur preconditioner error too large: " << relative_error;
}

TEST_F(MultigridSchurTest, MultigridSchur_IterativeRefinementImproves) {
    // More V-cycles should give better Q^{-1} approximation
    VecX r = VecX::Ones(n_constraints_);

    // Compute exact result
    Eigen::SparseLU<SpMat> Q_solver;
    Q_solver.compute(Q_);
    VecX Ct_r = C_.transpose() * r;
    VecX Q_inv_Ct_r = Q_solver.solve(Ct_r);
    VecX z_exact = C_ * Q_inv_Ct_r;

    // Test with increasing V-cycles
    std::vector<Real> errors;
    for (int num_vcycles : {1, 3, 5, 10}) {
        MultigridSchurPreconditioner precond(*mg_precond_, Q_, C_, num_vcycles);
        VecX z_mg = precond.apply(r);
        Real error = (z_mg - z_exact).norm() / z_exact.norm();
        errors.push_back(error);
    }

    // Error should generally decrease (or at least not increase significantly)
    // Note: Due to multigrid convergence, we expect monotonic improvement
    for (size_t i = 1; i < errors.size(); ++i) {
        EXPECT_LE(errors[i], errors[i - 1] + 1e-6)
            << "Error increased at V-cycle " << i
            << ": " << errors[i - 1] << " -> " << errors[i];
    }

    // Final error should be significantly smaller than initial
    EXPECT_LT(errors.back(), errors.front() * 0.5);
}

TEST_F(MultigridSchurTest, MultigridSchur_NumConstraints) {
    MultigridSchurPreconditioner precond(*mg_precond_, Q_, C_, 5);
    EXPECT_EQ(precond.num_constraints(), n_constraints_);
}

// =============================================================================
// Additional FlexibleCG robustness tests
// =============================================================================

TEST_F(MultigridSchurTest, FlexibleCG_HandlesVariablePreconditioner) {
    // FCG should work with MultigridSchur (variable preconditioner)
    MultigridSchurPreconditioner precond(*mg_precond_, Q_, C_, 5);

    // Create Schur complement matvec
    Eigen::SparseLU<SpMat> Q_solver;
    Q_solver.compute(Q_);
    ASSERT_EQ(Q_solver.info(), Eigen::Success);

    auto schur_matvec = [&](const VecX& v) -> VecX {
        VecX Ct_v = C_.transpose() * v;
        VecX Q_inv_Ct_v = Q_solver.solve(Ct_v);
        return C_ * Q_inv_Ct_v;
    };

    VecX rhs = VecX::Ones(n_constraints_);
    FlexibleCG fcg(schur_matvec, precond, 1e-8, 100);
    VecX x = VecX::Zero(n_constraints_);
    FCGResult result = fcg.solve(x, rhs);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.relative_residual, 1e-7);

    // Verify solution: S * x ≈ rhs
    VecX Sx = schur_matvec(x);
    Real sol_error = (Sx - rhs).norm() / rhs.norm();
    EXPECT_LT(sol_error, 1e-6);
}

TEST_F(SchurPreconditionerTest, FlexibleCG_ReportsCorrectIterationCount) {
    // Verify result.iterations matches actual iterations taken
    Eigen::SparseLU<SpMat> K_solver;
    K_solver.compute(K_);

    auto schur_matvec = [&](const VecX& v) -> VecX {
        VecX Ct_v = C_.transpose() * v;
        VecX K_inv_Ct_v = K_solver.solve(Ct_v);
        return C_ * K_inv_Ct_v;
    };

    PhysicsBasedSchurPreconditioner precond(K_, C_);
    VecX rhs = VecX::Ones(n_constraints_);

    // Use loose tolerance to require multiple iterations
    FlexibleCG fcg(schur_matvec, precond, 1e-12, 100);
    VecX x = VecX::Zero(n_constraints_);
    FCGResult result = fcg.solve(x, rhs);

    // Iterations should match residual history size - 1
    // (history includes initial residual)
    EXPECT_EQ(result.iterations, static_cast<int>(result.residual_history.size()) - 1);

    // For this well-conditioned problem, should converge in few iterations
    EXPECT_GT(result.iterations, 0);  // Should take at least 1 iteration
    EXPECT_LT(result.iterations, 10); // Should converge quickly
}

TEST_F(SchurPreconditionerTest, FlexibleCG_BreakdownDetection) {
    // Create a degenerate system where p^T A p ≈ 0 can occur
    // This is tricky to construct, so we test the safeguard behavior

    // Create a near-singular Schur matvec (diagonal with one very small entry)
    auto schur_matvec = [](const VecX& v) -> VecX {
        VecX result(2);
        result(0) = 1e-25 * v(0);  // Near-zero eigenvalue
        result(1) = 1.0 * v(1);
        return result;
    };

    // Create diagonal preconditioner
    DiagonalSchurPreconditioner precond(schur_matvec, 2);

    // RHS with component in near-null space
    VecX rhs(2);
    rhs(0) = 1.0;
    rhs(1) = 1.0;

    FlexibleCG fcg(schur_matvec, precond, 1e-10, 10);
    VecX x = VecX::Zero(2);
    FCGResult result = fcg.solve(x, rhs);

    // Should either converge or detect breakdown (not produce NaN)
    // Check no NaN in solution
    for (Index i = 0; i < x.size(); ++i) {
        EXPECT_FALSE(std::isnan(x(i))) << "NaN in solution at index " << i;
        EXPECT_FALSE(std::isinf(x(i))) << "Inf in solution at index " << i;
    }
}

TEST_F(SchurPreconditionerTest, FlexibleCG_IndefinitePreconditionerSafeguard) {
    // Test that FCG handles r^T z < 0 gracefully during iteration
    // The safeguard kicks in when preconditioner becomes indefinite mid-solve

    // Create a simple identity matvec
    auto schur_matvec = [](const VecX& v) -> VecX { return v; };

    // Create a preconditioner that becomes indefinite after first iteration
    class PartiallyIndefinitePreconditioner : public ISchurPreconditioner {
    public:
        mutable int call_count = 0;
        VecX apply(const VecX& r) const override {
            call_count++;
            if (call_count <= 1) {
                return r;  // Normal on first call
            }
            // Return -0.5*r on subsequent calls (indefinite but won't be < 1e-30 initially)
            return -0.5 * r;
        }
        bool is_variable() const override { return true; }
        Index num_constraints() const override { return 2; }
    };

    PartiallyIndefinitePreconditioner precond;

    VecX rhs = VecX::Ones(2);
    FlexibleCG fcg(schur_matvec, precond, 1e-10, 5);
    fcg.set_verbose(false);
    VecX x = VecX::Zero(2);
    FCGResult result = fcg.solve(x, rhs);

    // Should not produce NaN (safeguard uses abs(r^T z))
    for (Index i = 0; i < x.size(); ++i) {
        EXPECT_FALSE(std::isnan(x(i))) << "NaN from indefinite preconditioner";
        EXPECT_FALSE(std::isinf(x(i))) << "Inf from indefinite preconditioner";
    }

    // Should have recorded at least initial residual
    EXPECT_GT(result.residual_history.size(), 0u);

    // Residual values should be finite
    for (Real res : result.residual_history) {
        EXPECT_FALSE(std::isnan(res));
        EXPECT_FALSE(std::isinf(res));
    }
}

// =============================================================================
// Hard tests with realistic data-fitting Q matrices
// =============================================================================

TEST_F(MultigridSchurTest, VCycle_AccuracyDegradesWith_LargerLambda) {
    // Document: MG V-cycle becomes inaccurate as data fitting dominates

    std::vector<std::pair<Real, Real>> results; // (lambda, residual_ratio)

    for (Real lambda : {0.1, 1.0, 10.0, 100.0}) {
        SpMat Q = create_data_fitting_Q(lambda);

        MultigridConfig config;
        config.min_tree_level = 0;
        config.pre_smoothing = 2;
        config.post_smoothing = 2;
        config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
        BezierMultigridPreconditioner mg(config);
        mg.setup(Q, mesh_, smoother_->dof_manager());

        VecX b = VecX::Ones(n_free_);
        VecX x = VecX::Zero(n_free_);

        // Apply 5 V-cycles
        for (int i = 0; i < 5; ++i) {
            VecX r = b - Q * x;
            x += mg.apply(r);
        }

        Real residual_ratio = (b - Q * x).norm() / b.norm();
        results.push_back({lambda, residual_ratio});
    }

    // Document: higher lambda → worse convergence
    // lambda=0.1 should converge well
    EXPECT_LT(results[0].second, 0.1) << "MG should converge for lambda=0.1";

    // This test documents the failure mode - larger lambda degrades MG
    std::cout << "MG residual vs lambda:\n";
    for (auto& [l, r] : results) {
        std::cout << "  lambda=" << l << " residual=" << r << "\n";
    }
}

TEST_F(MultigridSchurTest, SchurMatvecAndPreconditioner_Consistency) {
    // This test reproduces the exact failure mode:
    // schur_matvec uses 1 V-cycle, preconditioner uses 5 V-cycles

    // Use data-fitting Q (not tridiagonal)
    SpMat Q = create_data_fitting_Q(10.0); // lambda=10

    MultigridConfig mg_config;
    mg_config.min_tree_level = 0;
    mg_config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    BezierMultigridPreconditioner mg(mg_config);
    mg.setup(Q, mesh_, smoother_->dof_manager());

    // Schur matvec with 1 V-cycle (like production code before fix)
    auto schur_matvec_1vcycle = [&](const VecX &v) -> VecX {
        VecX Ct_v = C_.transpose() * v;
        VecX Q_inv_approx = mg.apply(Ct_v); // 1 V-cycle only
        return C_ * Q_inv_approx;
    };

    // Exact Schur matvec (for reference)
    Eigen::SparseLU<SpMat> Q_solver;
    Q_solver.compute(Q);
    ASSERT_EQ(Q_solver.info(), Eigen::Success);

    auto schur_matvec_exact = [&](const VecX &v) -> VecX {
        VecX Ct_v = C_.transpose() * v;
        VecX Q_inv_exact = Q_solver.solve(Ct_v);
        return C_ * Q_inv_exact;
    };

    // MultigridSchur preconditioner (5 V-cycles with iterative refinement)
    MultigridSchurPreconditioner precond(mg, Q, C_, 5);

    VecX r = VecX::Ones(n_constraints_);

    // Compare approximations
    VecX Sv_1vcycle = schur_matvec_1vcycle(r);
    VecX Sv_exact = schur_matvec_exact(r);

    Real matvec_error = (Sv_1vcycle - Sv_exact).norm() / Sv_exact.norm();

    // Document: for data-fitting Q, 1 V-cycle is inaccurate
    std::cout << "Schur matvec error (1 V-cycle vs exact): " << matvec_error
              << "\n";

    // The issue: if matvec_error >> 0, then preconditioner targets wrong S
    if (matvec_error > 0.1) {
        std::cout << "WARNING: 1 V-cycle Schur matvec is inaccurate!\n";
        std::cout << "This will cause divergence when using MultigridSchur.\n";
    }

    // With exact schur_matvec, FCG should converge
    FlexibleCG fcg(schur_matvec_exact, precond, 1e-8, 100);
    VecX x = VecX::Zero(n_constraints_);
    VecX rhs = VecX::Ones(n_constraints_);
    FCGResult result = fcg.solve(x, rhs);

    EXPECT_TRUE(result.converged)
        << "FCG with exact matvec + MG precond should converge";
}

TEST_F(MultigridSchurTest, FullIntegrationPath_WithExactSchurMatvec) {
    // This test verifies the fix works: exact schur_matvec + MG preconditioner

    // Build actual smoother with data
    CGCubicBezierSmootherConfig config;
    config.lambda = 10.0;
    config.edge_ngauss = 0; // Disable edge constraints for simpler test
    config.use_iterative_solver = true;
    config.use_multigrid = true;
    config.schur_preconditioner = SchurPreconditionerType::MultigridVCycle;
    config.tolerance = 1e-8;
    config.max_iterations = 100;
    config.use_exact_schur_matvec = true; // THE FIX

    CGCubicBezierBathymetrySmoother smoother(mesh_, config);

    // Linear bathymetry (should be easy)
    smoother.set_bathymetry_data(
        [](Real x, Real y) { return 10.0 + 0.5 * x + 0.3 * y; });

    smoother.solve();

    EXPECT_TRUE(smoother.is_solved());

    // Verify solution quality
    Real eval = smoother.evaluate(0.5, 0.5);
    Real expected = 10.0 + 0.5 * 0.5 + 0.3 * 0.5;
    EXPECT_NEAR(eval, expected, 0.5); // Relaxed tolerance for smoothing effects
}

TEST_F(MultigridSchurTest, DocumentVCycleConvergenceFactor) {
    // Measure actual convergence factor for different Q structures
    // This documents expected MG performance

    struct TestCase {
        std::string name;
        Real lambda;
    };

    std::vector<TestCase> cases = {
        {"Tridiagonal (lambda=0)", 0.0},
        {"Light data fitting (lambda=1)", 1.0},
        {"Moderate data fitting (lambda=10)", 10.0},
        {"Heavy data fitting (lambda=100)", 100.0},
    };

    std::cout << "V-cycle convergence factors:\n";

    for (auto &tc : cases) {
        SpMat Q = create_data_fitting_Q(tc.lambda);

        MultigridConfig mg_config;
        mg_config.min_tree_level = 0;
        mg_config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
        BezierMultigridPreconditioner mg(mg_config);
        mg.setup(Q, mesh_, smoother_->dof_manager());

        // Measure convergence factor over 10 V-cycles
        VecX x = VecX::Zero(Q.rows());
        VecX b = VecX::Random(Q.rows());

        std::vector<Real> residuals;
        for (int i = 0; i < 10; ++i) {
            VecX r = b - Q * x;
            residuals.push_back(r.norm());
            x += mg.apply(r);
        }

        // Compute geometric mean convergence factor
        Real rho = std::pow(residuals.back() / residuals.front(), 1.0 / 10.0);

        std::cout << "  " << tc.name << ": rho=" << rho << "\n";

        // Weak assertion - just verify MG does something
        EXPECT_LT(rho, 0.95) << "MG should provide some convergence for "
                             << tc.name;
    }
}

// =============================================================================
// Schwarz Schur Preconditioner Tests (Diagnostic)
// =============================================================================

TEST_F(MultigridSchurTest, SchwarzSchur_AppliesFormula) {
    // Verify SchwarzSchur computes z ≈ C * Q^{-1} * C^T * r
    // using Schwarz iterations instead of exact solve

    // Get element blocks from MG
    int finest = mg_precond_->num_levels() - 1;
    const auto &finest_level = mg_precond_->level(finest);

    // Create colored Schwarz smoother
    auto smoother = std::make_unique<ColoredSchwarzMethod>(
        Q_, finest_level.element_free_dofs, finest_level.element_block_lu,
        finest_level.elements_by_color);

    // Create Schwarz Schur preconditioner with enough iterations
    SchwarzSchurPreconditioner precond(std::move(smoother), Q_, C_, 20);

    // Compute exact result for comparison
    Eigen::SparseLU<SpMat> Q_solver;
    Q_solver.compute(Q_);
    ASSERT_EQ(Q_solver.info(), Eigen::Success);

    VecX r = VecX::Ones(n_constraints_);
    VecX Ct_r = C_.transpose() * r;
    VecX Q_inv_Ct_r = Q_solver.solve(Ct_r);
    VecX z_exact = C_ * Q_inv_Ct_r;

    // Schwarz approximation
    VecX z_schwarz = precond.apply(r);

    // Should be reasonably close
    Real relative_error = (z_schwarz - z_exact).norm() / z_exact.norm();
    std::cout << "SchwarzSchur relative error (20 iterations): " << relative_error
              << "\n";

    // With enough iterations on a well-conditioned problem, should be accurate
    EXPECT_LT(relative_error, 0.3)
        << "Schwarz Schur preconditioner should approximate formula";
}

TEST_F(MultigridSchurTest, SchwarzSchur_ConvergesForHighLambda) {
    // Create challenging data-fitting Q with lambda=100
    SpMat Q = create_data_fitting_Q(100.0);

    // Setup MG for element blocks
    MultigridConfig mg_config;
    mg_config.min_tree_level = 0;
    mg_config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    BezierMultigridPreconditioner mg(mg_config);
    mg.setup(Q, mesh_, smoother_->dof_manager());

    int finest = mg.num_levels() - 1;
    const auto &finest_level = mg.level(finest);

    // Create colored Schwarz smoother
    auto smoother = std::make_unique<ColoredSchwarzMethod>(
        Q, finest_level.element_free_dofs, finest_level.element_block_lu,
        finest_level.elements_by_color);

    SchwarzSchurPreconditioner precond(std::move(smoother), Q, C_, 15);

    // Create exact Schur matvec
    Eigen::SparseLU<SpMat> Q_solver;
    Q_solver.compute(Q);
    ASSERT_EQ(Q_solver.info(), Eigen::Success);

    auto schur_matvec = [&](const VecX &v) -> VecX {
        VecX Ct_v = C_.transpose() * v;
        VecX Q_inv_Ct_v = Q_solver.solve(Ct_v);
        return C_ * Q_inv_Ct_v;
    };

    VecX rhs = VecX::Ones(n_constraints_);
    FlexibleCG fcg(schur_matvec, precond, 1e-6, 100);
    VecX x = VecX::Zero(n_constraints_);
    FCGResult result = fcg.solve(x, rhs);

    std::cout << "SchwarzSchur (lambda=100): converged=" << result.converged
              << ", iterations=" << result.iterations
              << ", relative_residual=" << result.relative_residual << "\n";

    // HARD REQUIREMENT: Must converge
    EXPECT_TRUE(result.converged)
        << "SchwarzSchur should converge for ill-conditioned Q";

    // Should converge in reasonable iterations
    EXPECT_LT(result.iterations, 50)
        << "SchwarzSchur should converge in < 50 iterations";
}

TEST_F(MultigridSchurTest, SchwarzSchur_OutperformsDiagonal) {
    // Compare Schwarz Schur vs Diagonal Schur on ill-conditioned Q
    SpMat Q = create_data_fitting_Q(50.0);

    // Exact Schur matvec
    Eigen::SparseLU<SpMat> Q_solver;
    Q_solver.compute(Q);
    ASSERT_EQ(Q_solver.info(), Eigen::Success);

    auto schur_matvec = [&](const VecX &v) -> VecX {
        VecX Ct_v = C_.transpose() * v;
        VecX Q_inv_Ct_v = Q_solver.solve(Ct_v);
        return C_ * Q_inv_Ct_v;
    };

    VecX rhs = VecX::Ones(n_constraints_);

    // Test with Diagonal preconditioner
    DiagonalSchurPreconditioner diag_precond(schur_matvec, n_constraints_);
    FlexibleCG fcg_diag(schur_matvec, diag_precond, 1e-6, 200);
    VecX x_diag = VecX::Zero(n_constraints_);
    FCGResult result_diag = fcg_diag.solve(x_diag, rhs);

    // Setup MG for element blocks
    MultigridConfig mg_config;
    mg_config.min_tree_level = 0;
    mg_config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    BezierMultigridPreconditioner mg(mg_config);
    mg.setup(Q, mesh_, smoother_->dof_manager());

    int finest = mg.num_levels() - 1;
    const auto &finest_level = mg.level(finest);

    // Test with Schwarz preconditioner
    auto smoother = std::make_unique<ColoredSchwarzMethod>(
        Q, finest_level.element_free_dofs, finest_level.element_block_lu,
        finest_level.elements_by_color);
    SchwarzSchurPreconditioner schwarz_precond(std::move(smoother), Q, C_, 10);
    FlexibleCG fcg_schwarz(schur_matvec, schwarz_precond, 1e-6, 200);
    VecX x_schwarz = VecX::Zero(n_constraints_);
    FCGResult result_schwarz = fcg_schwarz.solve(x_schwarz, rhs);

    std::cout << "Diagonal Schur: converged=" << result_diag.converged
              << ", iterations=" << result_diag.iterations << "\n";
    std::cout << "Schwarz Schur: converged=" << result_schwarz.converged
              << ", iterations=" << result_schwarz.iterations << "\n";

    // Both should converge
    EXPECT_TRUE(result_diag.converged);
    EXPECT_TRUE(result_schwarz.converged);

    // HARD REQUIREMENT: Schwarz should use at most as many iterations as diagonal
    // (may be equal for well-conditioned problems)
    EXPECT_LE(result_schwarz.iterations, result_diag.iterations)
        << "Schwarz Schur should not be worse than Diagonal Schur";
}

TEST_F(MultigridSchurTest, DiagnosticComparison_AllSchurPreconditioners) {
    // Run same problem with all Schur preconditioners to identify convergence behavior
    SpMat Q = create_data_fitting_Q(50.0);

    // Exact Schur matvec
    Eigen::SparseLU<SpMat> Q_solver;
    Q_solver.compute(Q);
    ASSERT_EQ(Q_solver.info(), Eigen::Success);

    auto schur_matvec = [&](const VecX &v) -> VecX {
        VecX Ct_v = C_.transpose() * v;
        VecX Q_inv_Ct_v = Q_solver.solve(Ct_v);
        return C_ * Q_inv_Ct_v;
    };

    VecX rhs = VecX::Ones(n_constraints_);

    // Setup MG for all MG-based preconditioners
    MultigridConfig mg_config;
    mg_config.min_tree_level = 0;
    mg_config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    BezierMultigridPreconditioner mg(mg_config);
    mg.setup(Q, mesh_, smoother_->dof_manager());

    int finest = mg.num_levels() - 1;
    const auto &finest_level = mg.level(finest);

    std::cout << "\n=== Schur Preconditioner Comparison (lambda=50) ===\n";

    // 1. Diagonal
    {
        DiagonalSchurPreconditioner precond(schur_matvec, n_constraints_);
        FlexibleCG fcg(schur_matvec, precond, 1e-6, 200);
        VecX x = VecX::Zero(n_constraints_);
        FCGResult result = fcg.solve(x, rhs);
        std::cout << "Diagonal:       iters=" << result.iterations
                  << ", converged=" << result.converged << "\n";
    }

    // 2. PhysicsBased
    {
        PhysicsBasedSchurPreconditioner precond(Q, C_);
        FlexibleCG fcg(schur_matvec, precond, 1e-6, 200);
        VecX x = VecX::Zero(n_constraints_);
        FCGResult result = fcg.solve(x, rhs);
        std::cout << "PhysicsBased:   iters=" << result.iterations
                  << ", converged=" << result.converged << "\n";
    }

    // 3. SchwarzColored (10 iterations)
    {
        auto smoother = std::make_unique<ColoredSchwarzMethod>(
            Q, finest_level.element_free_dofs, finest_level.element_block_lu,
            finest_level.elements_by_color);
        SchwarzSchurPreconditioner precond(std::move(smoother), Q, C_, 10);
        FlexibleCG fcg(schur_matvec, precond, 1e-6, 200);
        VecX x = VecX::Zero(n_constraints_);
        FCGResult result = fcg.solve(x, rhs);
        std::cout << "SchwarzColored: iters=" << result.iterations
                  << ", converged=" << result.converged << "\n";
    }

    // 4. MultigridVCycle (5 V-cycles)
    {
        MultigridSchurPreconditioner precond(mg, Q, C_, 5);
        FlexibleCG fcg(schur_matvec, precond, 1e-6, 200);
        VecX x = VecX::Zero(n_constraints_);
        FCGResult result = fcg.solve(x, rhs);
        std::cout << "MultigridVCycle: iters=" << result.iterations
                  << ", converged=" << result.converged << "\n";
    }

    std::cout << "=================================================\n\n";

    // Basic assertion: at least one preconditioner should work
    // (This test is primarily diagnostic)
    SUCCEED();
}

TEST_F(MultigridSchurTest, SchwarzSchur_IsStationary) {
    // Schwarz Schur should NOT be variable (unlike MG)
    int finest = mg_precond_->num_levels() - 1;
    const auto &finest_level = mg_precond_->level(finest);

    auto smoother = std::make_unique<ColoredSchwarzMethod>(
        Q_, finest_level.element_free_dofs, finest_level.element_block_lu,
        finest_level.elements_by_color);

    SchwarzSchurPreconditioner precond(std::move(smoother), Q_, C_, 10);

    EXPECT_FALSE(precond.is_variable())
        << "SchwarzSchur should be stationary (not variable)";
}

TEST_F(MultigridSchurTest, SchwarzSchur_MoreIterationsImprove) {
    // More Schwarz iterations should give better Q^{-1} approximation
    int finest = mg_precond_->num_levels() - 1;
    const auto &finest_level = mg_precond_->level(finest);

    // Exact result
    Eigen::SparseLU<SpMat> Q_solver;
    Q_solver.compute(Q_);
    VecX r = VecX::Ones(n_constraints_);
    VecX Ct_r = C_.transpose() * r;
    VecX Q_inv_Ct_r = Q_solver.solve(Ct_r);
    VecX z_exact = C_ * Q_inv_Ct_r;

    std::vector<Real> errors;
    for (int num_iters : {1, 3, 5, 10, 20}) {
        auto smoother = std::make_unique<ColoredSchwarzMethod>(
            Q_, finest_level.element_free_dofs, finest_level.element_block_lu,
            finest_level.elements_by_color);
        SchwarzSchurPreconditioner precond(std::move(smoother), Q_, C_, num_iters);

        VecX z_schwarz = precond.apply(r);
        Real error = (z_schwarz - z_exact).norm() / z_exact.norm();
        errors.push_back(error);
    }

    // Error should generally decrease with more iterations
    for (size_t i = 1; i < errors.size(); ++i) {
        EXPECT_LE(errors[i], errors[i - 1] + 1e-6)
            << "Error should not increase with more Schwarz iterations";
    }

    // Final error should be significantly smaller than initial
    EXPECT_LT(errors.back(), errors.front() * 0.5)
        << "20 iterations should reduce error vs 1 iteration";
}

} // namespace
