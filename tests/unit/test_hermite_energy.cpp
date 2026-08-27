#include "bathymetry/cubic_thin_plate_hessian.hpp"
#include "bathymetry/dirichlet_hessian.hpp"
#include "bathymetry/hermite_basis_2d.hpp"
#include "bathymetry/hermite_energy_matrices.hpp"
#include "bathymetry/hermite_hessian.hpp"
#include <Eigen/Eigenvalues>
#include <gtest/gtest.h>
#include <cmath>

using namespace drifter;

namespace {

constexpr Real TOLERANCE = 1e-12;

/// Build the Hermite DOF vector of a polynomial on [0,dx] x [0,dy], given a
/// callable returning d^(a+b) f / dx^a dy^b at (x, y).
VecX sample_dofs(const HermiteBasis2D &basis, Real dx, Real dy,
                 const std::function<Real(Real, Real, int, int)> &f) {
    VecX q(basis.num_dofs());
    for (int dof = 0; dof < basis.num_dofs(); ++dof) {
        const auto [a, b] = basis.deriv_order(dof);
        const int corner = basis.dof_to_corner(dof);
        const Real x = (corner % 2) * dx;
        const Real y = (corner / 2) * dy;
        q(dof) = f(x, y, a, b);
    }
    return q;
}

/// Number of eigenvalues of a symmetric matrix below a relative threshold
int null_space_dim(const MatX &A) {
    Eigen::SelfAdjointEigenSolver<MatX> es(A);
    const Real scale = es.eigenvalues().cwiseAbs().maxCoeff();
    int count = 0;
    for (int i = 0; i < es.eigenvalues().size(); ++i) {
        if (std::abs(es.eigenvalues()(i)) < 1e-9 * scale) {
            ++count;
        }
    }
    return count;
}

// =============================================================================
// 1D energy matrices
// =============================================================================

TEST(HermiteEnergyMatricesTest, TransposeSymmetry) {
    for (int r = 0; r <= 2; ++r) {
        HermiteEnergyMatrices1D K(r);
        for (int m = 0; m <= 2; ++m) {
            for (int n = 0; n <= 2; ++n) {
                // K^(m,n) = (K^(n,m))^T by definition of the integral
                EXPECT_LT((K.reference(m, n) - K.reference(n, m).transpose()).norm(), TOLERANCE)
                    << "r=" << r << " (" << m << "," << n << ")";
            }
        }
        // Diagonal blocks are symmetric
        for (int m = 0; m <= 2; ++m) {
            MatX Kmm = K.reference(m, m);
            EXPECT_LT((Kmm - Kmm.transpose()).norm(), TOLERANCE) << "r=" << r << " m=" << m;
        }
    }
}

// K^(0,0) is a mass matrix; its entries must integrate the products exactly.
// Spot-check against a hand-computed value: for r=1, K00(0,0) = int (2t^3-3t^2+1)^2 = 13/35
TEST(HermiteEnergyMatricesTest, MassMatrixExactValues) {
    HermiteEnergyMatrices1D K(1);
    MatX K00 = K.reference(0, 0);
    EXPECT_NEAR(K00(0, 0), 13.0 / 35.0, TOLERANCE);
    EXPECT_NEAR(K00(2, 2), 13.0 / 35.0, TOLERANCE);
    // Total mass: sum over value functions = int 1 * 1 = 1
    EXPECT_NEAR(K00(0, 0) + K00(0, 2) + K00(2, 0) + K00(2, 2), 1.0, TOLERANCE);
}

TEST(HermiteEnergyMatricesTest, PhysicalScalingFoldsInJacobianAndChainRule) {
    HermiteEnergyMatrices1D K(1);
    const Real h = 3.0;

    // K_phys^(1,1)(h) = h^(1-1-1) Lambda K^(1,1) Lambda = h^-1 Lambda K Lambda
    MatX expected = K.reference(1, 1);
    VecX lam = K.lambda_scaling(h);
    expected = std::pow(h, -1) * (lam.asDiagonal() * expected * lam.asDiagonal());

    EXPECT_LT((K.physical(1, 1, h) - expected).norm(), TOLERANCE);

    // Lambda(h) = diag(h^mu_i): value DOFs unscaled, derivative DOFs by h
    EXPECT_NEAR(lam(0), 1.0, TOLERANCE);
    EXPECT_NEAR(lam(1), h, TOLERANCE);
    EXPECT_NEAR(lam(2), 1.0, TOLERANCE);
    EXPECT_NEAR(lam(3), h, TOLERANCE);
}

// =============================================================================
// Element Hessian: symmetry and definiteness
// =============================================================================

TEST(HermiteHessianTest, SymmetricAndPositiveSemiDefinite) {
    for (int r = 0; r <= 2; ++r) {
        HermiteHessian hessian(r);

        for (auto [dx, dy] : {std::pair{1.0, 1.0}, {2.0, 0.5}, {0.25, 4.0}}) {
            MatX H = hessian.scaled_hessian(dx, dy);
            ASSERT_EQ(H.rows(), hessian.num_dofs());

            EXPECT_LT((H - H.transpose()).norm(), 1e-10 * H.norm())
                << "r=" << r << " dx=" << dx << " dy=" << dy;

            Eigen::SelfAdjointEigenSolver<MatX> es(H);
            const Real scale = es.eigenvalues().cwiseAbs().maxCoeff();
            EXPECT_GT(es.eigenvalues().minCoeff(), -1e-9 * scale)
                << "r=" << r << " dx=" << dx << " dy=" << dy;
        }
    }
}

// The implemented energy is (Laplacian)^2 + 2 z_xy^2, whose null space is
// {1, x, y, x^2 - y^2} -- one dimension larger than the standard thin plate,
// because x^2 - y^2 is harmonic AND has zero twist.
// See docs/hermite_smoothness_operator.md S2.2.
TEST(HermiteHessianTest, ThinPlateNullSpaceIsDimensionFour) {
    HermiteHessian hessian(1);
    HermiteBasis2D basis(1);

    const Real dx = 1.5;
    const Real dy = 0.8;
    MatX H = hessian.scaled_hessian(dx, dy);

    EXPECT_EQ(null_space_dim(H), 4);

    // And the four modes are annihilated explicitly
    auto energy_of = [&](const std::function<Real(Real, Real, int, int)> &f) {
        VecX q = sample_dofs(basis, dx, dy, f);
        return q.dot(H * q);
    };

    const Real scale = H.norm();

    // f = 1
    EXPECT_LT(std::abs(energy_of([](Real, Real, int a, int b) {
                  return (a == 0 && b == 0) ? 1.0 : 0.0;
              })), 1e-10 * scale);

    // f = x
    EXPECT_LT(std::abs(energy_of([](Real x, Real, int a, int b) {
                  if (a == 0 && b == 0) return x;
                  if (a == 1 && b == 0) return 1.0;
                  return 0.0;
              })), 1e-10 * scale);

    // f = y
    EXPECT_LT(std::abs(energy_of([](Real, Real y, int a, int b) {
                  if (a == 0 && b == 0) return y;
                  if (a == 0 && b == 1) return 1.0;
                  return 0.0;
              })), 1e-10 * scale);

    // f = x^2 - y^2 (harmonic, zero twist)
    EXPECT_LT(std::abs(energy_of([](Real x, Real y, int a, int b) {
                  if (a == 0 && b == 0) return x * x - y * y;
                  if (a == 1 && b == 0) return 2.0 * x;
                  if (a == 0 && b == 1) return -2.0 * y;
                  return 0.0;
              })), 1e-10 * scale);

    // But a genuinely bending mode costs energy: f = x^2 + y^2
    EXPECT_GT(energy_of([](Real x, Real y, int a, int b) {
                  if (a == 0 && b == 0) return x * x + y * y;
                  if (a == 1 && b == 0) return 2.0 * x;
                  if (a == 0 && b == 1) return 2.0 * y;
                  return 0.0;
              }), 1e-6 * scale);
}

// The membrane energy has null space {1} at every resolution -- which is why it,
// and not the thin plate, is the right choice at r = 0.
TEST(HermiteHessianTest, MembraneNullSpaceIsDimensionOne) {
    HermiteHessian hessian(0);
    MatX H = hessian.scaled_hessian(1.3, 0.6);
    EXPECT_EQ(null_space_dim(H), 1);

    // The constant is the null mode
    VecX ones = VecX::Ones(4);
    EXPECT_LT(std::abs(ones.dot(H * ones)), 1e-10 * H.norm());
}

// =============================================================================
// The congruence cross-check
// =============================================================================

// Two independent derivations of the same operator must agree:
//   - the closed Kronecker form implemented in HermiteHessian
//   - the Bernstein element matrix under the change of basis, M_e' H_Bern M_e
// See docs/hermite_smoothness_operator.md S6. This is the check that the whole
// Hermite element library computes the energy the Bezier smoother already does.
TEST(HermiteCongruenceTest, ThinPlateMatchesBernsteinUnderChangeOfBasis) {
    HermiteHessian hermite(1);
    HermiteBasis2D basis(1);
    // ngauss = 4 is exactly sufficient for the r=1 integrand (degree 2p = 6)
    CubicThinPlateHessian bezier(4);

    for (auto [dx, dy] : {std::pair{1.0, 1.0}, {2.0, 0.5}, {0.3, 1.7}, {5.0, 5.0}}) {
        MatX H_hermite = hermite.scaled_hessian(dx, dy);
        MatX H_bezier = bezier.scaled_hessian(dx, dy);
        MatX Me = basis.bernstein_change_of_basis(dx, dy);

        MatX H_congruence = Me.transpose() * H_bezier * Me;

        EXPECT_LT((H_hermite - H_congruence).norm(), 1e-9 * H_hermite.norm())
            << "dx=" << dx << " dy=" << dy;
    }
}

// At r = 0, M_e = Lambda_e = I and the Hermite element *is* the linear Bezier
// element -- modulo the local index convention, which is transposed
// (HermiteBasis2D uses i + 2j, LinearBezierBasis2D uses j + 2i).
TEST(HermiteCongruenceTest, MembraneMatchesDirichletHessian) {
    HermiteHessian hermite(0);
    DirichletHessian bezier(2);

    // Permutation between the two conventions: swap local DOFs 1 and 2
    Eigen::PermutationMatrix<4> P;
    P.indices() << 0, 2, 1, 3;

    for (auto [dx, dy] : {std::pair{1.0, 1.0}, {2.0, 0.5}, {0.3, 1.7}}) {
        MatX H_hermite = hermite.scaled_hessian(dx, dy);
        MatX H_bezier = bezier.scaled_hessian(dx, dy);
        MatX H_permuted = P.transpose() * H_bezier * P;

        EXPECT_LT((H_hermite - H_permuted).norm(), 1e-9 * H_hermite.norm())
            << "dx=" << dx << " dy=" << dy;
    }
}

// The anisotropic factors are what make the smoother behave on stretched cells:
// the z_xx term scales as h_y/h_x^3, the z_yy term as h_x/h_y^3.
TEST(HermiteHessianTest, AnisotropicScaling) {
    HermiteHessian hessian(1);

    // Halving dx at fixed dy multiplies the pure-x bending contribution by 8.
    // Check via a mode that is pure x-bending: f = x^2 (z_yy = z_xy = 0).
    HermiteBasis2D basis(1);
    auto x_squared = [](Real x, Real, int a, int b) {
        if (a == 0 && b == 0) return x * x;
        if (a == 1 && b == 0) return 2.0 * x;
        return 0.0;
    };

    const Real dy = 1.0;
    VecX q1 = sample_dofs(basis, 1.0, dy, x_squared);
    VecX q2 = sample_dofs(basis, 0.5, dy, x_squared);

    const Real e1 = q1.dot(hessian.scaled_hessian(1.0, dy) * q1);
    const Real e2 = q2.dot(hessian.scaled_hessian(0.5, dy) * q2);

    // energy = int (z_xx)^2 = 4 * area, so halving dx halves it (area halves).
    EXPECT_NEAR(e1, 4.0 * 1.0 * dy, 1e-9);
    EXPECT_NEAR(e2, 4.0 * 0.5 * dy, 1e-9);
}

} // namespace
