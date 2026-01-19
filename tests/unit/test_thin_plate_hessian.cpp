#include <gtest/gtest.h>
#include "bathymetry/thin_plate_hessian.hpp"
#include "bathymetry/bezier_basis_2d.hpp"
#include <cmath>

using namespace drifter;

namespace {

constexpr Real TOLERANCE = 1e-10;
constexpr Real LOOSE_TOLERANCE = 1e-6;

class ThinPlateHessianTest : public ::testing::Test {
protected:
    ThinPlateHessian hessian{6};  // 6 Gauss points per direction
    BezierBasis2D basis;
};

// =============================================================================
// Basic properties tests
// =============================================================================

TEST_F(ThinPlateHessianTest, HessianDimensions) {
    const MatX& H = hessian.element_hessian();
    EXPECT_EQ(H.rows(), 36);
    EXPECT_EQ(H.cols(), 36);
}

TEST_F(ThinPlateHessianTest, HessianSymmetry) {
    const MatX& H = hessian.element_hessian();

    for (int i = 0; i < 36; ++i) {
        for (int j = 0; j < 36; ++j) {
            EXPECT_NEAR(H(i, j), H(j, i), TOLERANCE)
                << "Asymmetry at (" << i << "," << j << ")";
        }
    }
}

TEST_F(ThinPlateHessianTest, HessianPositiveSemiDefinite) {
    const MatX& H = hessian.element_hessian();

    // Check eigenvalues are non-negative
    Eigen::SelfAdjointEigenSolver<MatX> solver(H);
    VecX eigenvalues = solver.eigenvalues();

    for (int i = 0; i < eigenvalues.size(); ++i) {
        EXPECT_GE(eigenvalues(i), -TOLERANCE)
            << "Negative eigenvalue: " << eigenvalues(i);
    }
}

// =============================================================================
// Energy tests
// =============================================================================

TEST_F(ThinPlateHessianTest, EnergyNonNegative) {
    // Energy should be non-negative for any control points
    for (int trial = 0; trial < 10; ++trial) {
        VecX coeffs = VecX::Random(36);
        Real energy = hessian.energy(coeffs);
        EXPECT_GE(energy, -TOLERANCE)
            << "Negative energy for trial " << trial;
    }
}

TEST_F(ThinPlateHessianTest, ZeroEnergyForLinear) {
    // Linear function has zero curvature, so zero thin plate energy
    // z(u,v) = a*u + b*v + c => z_uu = z_vv = z_uv = 0

    VecX coeffs(36);
    Real a = 2.0, b = 3.0, c = 1.0;
    for (int j = 0; j < 6; ++j) {
        for (int i = 0; i < 6; ++i) {
            Real u = static_cast<Real>(i) / 5.0;
            Real v = static_cast<Real>(j) / 5.0;
            coeffs(BezierBasis2D::dof_index(i, j)) = a * u + b * v + c;
        }
    }

    Real energy = hessian.energy(coeffs);
    EXPECT_NEAR(energy, 0.0, TOLERANCE)
        << "Linear function should have zero thin plate energy";
}

TEST_F(ThinPlateHessianTest, PositiveEnergyForQuadratic) {
    // Quadratic function has non-zero curvature
    // z(u,v) = u^2 + v^2 => z_uu = 2, z_vv = 2, z_uv = 0

    VecX coeffs(36);
    for (int j = 0; j < 6; ++j) {
        for (int i = 0; i < 6; ++i) {
            Real u = static_cast<Real>(i) / 5.0;
            Real v = static_cast<Real>(j) / 5.0;
            coeffs(BezierBasis2D::dof_index(i, j)) = u * u + v * v;
        }
    }

    Real energy = hessian.energy(coeffs);
    EXPECT_GT(energy, 0.0)
        << "Quadratic function should have positive thin plate energy";
}

// =============================================================================
// Gradient tests
// =============================================================================

TEST_F(ThinPlateHessianTest, GradientConsistency) {
    // Gradient should be 2 * H * x
    VecX coeffs = VecX::Random(36);

    VecX grad = hessian.gradient(coeffs);
    VecX expected = 2.0 * hessian.element_hessian() * coeffs;

    for (int i = 0; i < 36; ++i) {
        EXPECT_NEAR(grad(i), expected(i), TOLERANCE)
            << "Gradient mismatch at index " << i;
    }
}

TEST_F(ThinPlateHessianTest, GradientFiniteDifference) {
    // Check gradient via finite difference
    VecX coeffs = VecX::Random(36);
    coeffs *= 0.1;  // Scale down to improve numerical accuracy

    Real delta = 1e-7;
    VecX grad = hessian.gradient(coeffs);

    for (int k = 0; k < 36; ++k) {
        VecX coeffs_plus = coeffs;
        VecX coeffs_minus = coeffs;
        coeffs_plus(k) += delta;
        coeffs_minus(k) -= delta;

        Real e_plus = hessian.energy(coeffs_plus);
        Real e_minus = hessian.energy(coeffs_minus);
        Real fd_grad = (e_plus - e_minus) / (2.0 * delta);

        EXPECT_NEAR(grad(k), fd_grad, LOOSE_TOLERANCE)
            << "Gradient FD mismatch at index " << k;
    }
}

// =============================================================================
// Scaled Hessian tests
// =============================================================================

TEST_F(ThinPlateHessianTest, ScaledHessianDimensions) {
    MatX H_scaled = hessian.scaled_hessian(2.0, 3.0);
    EXPECT_EQ(H_scaled.rows(), 36);
    EXPECT_EQ(H_scaled.cols(), 36);
}

TEST_F(ThinPlateHessianTest, ScaledHessianSymmetry) {
    MatX H_scaled = hessian.scaled_hessian(1.5, 2.5);

    for (int i = 0; i < 36; ++i) {
        for (int j = 0; j < 36; ++j) {
            EXPECT_NEAR(H_scaled(i, j), H_scaled(j, i), TOLERANCE)
                << "Scaled Hessian asymmetry at (" << i << "," << j << ")";
        }
    }
}

TEST_F(ThinPlateHessianTest, ScaledHessianEnergyPositive) {
    // Test that energy computed via scaled Hessian is positive for non-linear coefficients
    MatX H_scaled = hessian.scaled_hessian(2.0, 3.0);

    // Use a non-trivial coefficient vector
    VecX coeffs = VecX::Random(36);
    Real energy = coeffs.transpose() * H_scaled * coeffs;

    // Energy should be positive (or zero for linear functions)
    EXPECT_GE(energy, -1e-8)
        << "Scaled Hessian should give non-negative energy";
}

TEST_F(ThinPlateHessianTest, ScaledHessianCorrectSize) {
    // Scaled Hessian should have correct size
    MatX H_scaled = hessian.scaled_hessian(2.0, 3.0);

    EXPECT_EQ(H_scaled.rows(), 36);
    EXPECT_EQ(H_scaled.cols(), 36);
}

// =============================================================================
// Derivative matrix tests
// =============================================================================

TEST_F(ThinPlateHessianTest, DerivativeMatrixDimensions) {
    int nquad = 6 * 6;  // 36 Gauss points
    EXPECT_EQ(hessian.d2u_matrix().rows(), nquad);
    EXPECT_EQ(hessian.d2u_matrix().cols(), 36);
    EXPECT_EQ(hessian.d2v_matrix().rows(), nquad);
    EXPECT_EQ(hessian.d2v_matrix().cols(), 36);
    EXPECT_EQ(hessian.d2uv_matrix().rows(), nquad);
    EXPECT_EQ(hessian.d2uv_matrix().cols(), 36);
}

// =============================================================================
// Physical interpretation tests
// =============================================================================

TEST_F(ThinPlateHessianTest, EnergyScalesWithCurvature) {
    // More curved surface should have higher energy
    VecX coeffs_flat(36);
    VecX coeffs_curved(36);

    for (int j = 0; j < 6; ++j) {
        for (int i = 0; i < 6; ++i) {
            Real u = static_cast<Real>(i) / 5.0;
            Real v = static_cast<Real>(j) / 5.0;
            int dof = BezierBasis2D::dof_index(i, j);

            // Flat: linear
            coeffs_flat(dof) = u + v;

            // Curved: quadratic
            coeffs_curved(dof) = u * u + v * v;
        }
    }

    Real energy_flat = hessian.energy(coeffs_flat);
    Real energy_curved = hessian.energy(coeffs_curved);

    EXPECT_NEAR(energy_flat, 0.0, TOLERANCE);
    EXPECT_GT(energy_curved, energy_flat);
}

}  // namespace
