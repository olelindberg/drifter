#include <gtest/gtest.h>
#include "bathymetry/bezier_multigrid_preconditioner.hpp"
#include "bathymetry/cg_cubic_bezier_bathymetry_smoother.hpp"
#include "bathymetry/block_diag_approx_cg_schur_preconditioner.hpp"
#include "bathymetry/diagonal_approx_cg_schur_preconditioner.hpp"
#include "bathymetry/flexible_cg.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "bathymetry/schur_preconditioner.hpp"
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
// BlockDiagApproxCGSchurPreconditioner tests
// =============================================================================

class BlockDiagApproxCGSchurTest : public ::testing::Test {
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
    }

    /// Create Q matrix that mimics actual bathymetry: Q = H + λ·BᵀWB + εI
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
        for (Index i = 0; i < n_free_; ++i) {
            for (Index j = std::max(Index(0), i - 3);
                 j <= std::min(n_free_ - 1, i + 3); ++j) {
                if (i != j) {
                    Real weight = lambda * 0.3 * std::exp(-0.5 * std::abs(i - j));
                    triplets.emplace_back(i, j, weight);
                } else {
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

    QuadtreeAdapter mesh_;
    std::unique_ptr<CGCubicBezierBathymetrySmoother> smoother_;
    Index n_free_;
    Index n_constraints_;
    SpMat Q_;
    SpMat C_;
};

TEST_F(BlockDiagApproxCGSchurTest, BlockDiagApproxCG_ConstructsSuccessfully) {
    // Should not throw for valid Q, C, and DOF manager
    BlockDiagApproxCGSchurPreconditioner precond(Q_, C_, smoother_->dof_manager());
    EXPECT_EQ(precond.num_constraints(), n_constraints_);
}

TEST_F(BlockDiagApproxCGSchurTest, BlockDiagApproxCG_CountsElementBlocks) {
    BlockDiagApproxCGSchurPreconditioner precond(Q_, C_, smoother_->dof_manager());

    // For a 4x4 uniform mesh, expect 16 element blocks (one per element)
    Index num_blocks = precond.num_element_blocks();
    EXPECT_GT(num_blocks, 0) << "Should have at least some element blocks";

    // For 4x4 mesh, should have exactly 16 elements
    EXPECT_EQ(num_blocks, 16) << "4x4 mesh should have 16 element blocks";
}

TEST_F(BlockDiagApproxCGSchurTest, BlockDiagApproxCG_AssembledMatrixIsSPD) {
    BlockDiagApproxCGSchurPreconditioner precond(Q_, C_, smoother_->dof_manager());

    // M_S = C * blockdiag(Q)^{-1} * C^T should be SPD
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
        EXPECT_GT(eig.eigenvalues()(i), 0.0)
            << "M_S should be positive definite";
    }
}

TEST_F(BlockDiagApproxCGSchurTest, BlockDiagApproxCG_ApplyGivesCorrectResult) {
    // Use very tight inner tolerance to ensure convergence
    BlockDiagApproxCGSchurPreconditioner precond(Q_, C_, smoother_->dof_manager(),
                                                  1e-12, 200);

    VecX r = VecX::Ones(n_constraints_);
    VecX z = precond.apply(r);

    // z should be M_S^{-1} * r
    // Check by computing M_S * z ≈ r
    const SpMat& M_S = precond.assembled_matrix();
    VecX M_S_z = M_S * z;

    // Use relaxed tolerance since this is an iterative approximation
    for (Index i = 0; i < n_constraints_; ++i) {
        EXPECT_NEAR(M_S_z(i), r(i), 0.1);
    }
}

TEST_F(BlockDiagApproxCGSchurTest, BlockDiagApproxCG_IsVariable) {
    BlockDiagApproxCGSchurPreconditioner precond(Q_, C_, smoother_->dof_manager());
    EXPECT_TRUE(precond.is_variable());
}

TEST_F(BlockDiagApproxCGSchurTest, BlockDiagApproxCG_HandlesZeroRHS) {
    BlockDiagApproxCGSchurPreconditioner precond(Q_, C_, smoother_->dof_manager());

    VecX r = VecX::Zero(n_constraints_);
    VecX z = precond.apply(r);

    // Should return zero for zero input
    EXPECT_LT(z.norm(), TOLERANCE);
}

TEST_F(BlockDiagApproxCGSchurTest, FlexibleCG_ConvergesWithBlockDiagApproxCG) {
    // Create Schur complement matvec: S = C * Q^{-1} * C^T
    Eigen::SparseLU<SpMat> Q_solver;
    Q_solver.compute(Q_);
    ASSERT_EQ(Q_solver.info(), Eigen::Success);

    auto schur_matvec = [&](const VecX& v) -> VecX {
        VecX Ct_v = C_.transpose() * v;
        VecX Q_inv_Ct_v = Q_solver.solve(Ct_v);
        return C_ * Q_inv_Ct_v;
    };

    // Create BlockDiagApproxCG preconditioner
    BlockDiagApproxCGSchurPreconditioner precond(Q_, C_, smoother_->dof_manager(),
                                                  1e-10, 100);

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

TEST_F(BlockDiagApproxCGSchurTest, BlockDiagApproxCG_VsDiagonalApproxCG_Comparison) {
    // Compare iteration counts between DiagonalApproxCG and BlockDiagApproxCG
    // BlockDiag should generally need fewer or equal iterations

    // Use data-fitting Q for a more challenging problem
    SpMat Q = create_data_fitting_Q(10.0);

    // Exact Schur matvec
    Eigen::SparseLU<SpMat> Q_solver;
    Q_solver.compute(Q);
    ASSERT_EQ(Q_solver.info(), Eigen::Success);

    auto schur_matvec = [&](const VecX& v) -> VecX {
        VecX Ct_v = C_.transpose() * v;
        VecX Q_inv_Ct_v = Q_solver.solve(Ct_v);
        return C_ * Q_inv_Ct_v;
    };

    VecX rhs = VecX::Ones(n_constraints_);

    // Test with DiagonalApproxCG
    DiagonalApproxCGSchurPreconditioner diag_precond(Q, C_, 1e-10, 100);
    FlexibleCG fcg_diag(schur_matvec, diag_precond, 1e-8, 200);
    VecX x_diag = VecX::Zero(n_constraints_);
    FCGResult result_diag = fcg_diag.solve(x_diag, rhs);

    // Test with BlockDiagApproxCG
    BlockDiagApproxCGSchurPreconditioner block_precond(Q, C_, smoother_->dof_manager(),
                                                        1e-10, 100);
    FlexibleCG fcg_block(schur_matvec, block_precond, 1e-8, 200);
    VecX x_block = VecX::Zero(n_constraints_);
    FCGResult result_block = fcg_block.solve(x_block, rhs);

    // Both should converge
    EXPECT_TRUE(result_diag.converged);
    EXPECT_TRUE(result_block.converged);

    // BlockDiag should not be significantly worse than Diagonal
    EXPECT_LE(result_block.iterations, result_diag.iterations + 5)
        << "BlockDiagApproxCG should not be significantly worse than DiagonalApproxCG";
}

} // namespace
