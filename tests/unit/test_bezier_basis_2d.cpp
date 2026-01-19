#include <gtest/gtest.h>
#include "bathymetry/bezier_basis_2d.hpp"
#include <cmath>

using namespace drifter;

namespace {

constexpr Real TOLERANCE = 1e-12;
constexpr Real LOOSE_TOLERANCE = 1e-8;
constexpr Real FD_DELTA = 1e-6;  // Finite difference step

class BezierBasis2DTest : public ::testing::Test {
protected:
    BezierBasis2D basis;
};

// =============================================================================
// Basic properties tests
// =============================================================================

TEST_F(BezierBasis2DTest, Constants) {
    EXPECT_EQ(basis.degree(), 5);
    EXPECT_EQ(basis.num_nodes_1d(), 6);
    EXPECT_EQ(basis.num_dofs(), 36);
}

TEST_F(BezierBasis2DTest, ControlPointPositions) {
    // Control points should be uniformly spaced on [0,1]^2
    for (int dof = 0; dof < 36; ++dof) {
        int i, j;
        BezierBasis2D::dof_ij(dof, i, j);
        Vec2 pos = basis.control_point_position(dof);

        Real expected_u = static_cast<Real>(i) / 5.0;
        Real expected_v = static_cast<Real>(j) / 5.0;

        EXPECT_NEAR(pos(0), expected_u, TOLERANCE);
        EXPECT_NEAR(pos(1), expected_v, TOLERANCE);
    }
}

// =============================================================================
// DOF indexing tests
// =============================================================================

TEST_F(BezierBasis2DTest, DofIndexing) {
    // Test round-trip conversion
    for (int j = 0; j < 6; ++j) {
        for (int i = 0; i < 6; ++i) {
            int dof = BezierBasis2D::dof_index(i, j);
            int i_out, j_out;
            BezierBasis2D::dof_ij(dof, i_out, j_out);
            EXPECT_EQ(i, i_out);
            EXPECT_EQ(j, j_out);
        }
    }

    // DOFs should be unique and cover 0-35
    std::vector<bool> seen(36, false);
    for (int j = 0; j < 6; ++j) {
        for (int i = 0; i < 6; ++i) {
            int dof = BezierBasis2D::dof_index(i, j);
            EXPECT_GE(dof, 0);
            EXPECT_LT(dof, 36);
            EXPECT_FALSE(seen[dof]);
            seen[dof] = true;
        }
    }
}

// =============================================================================
// Partition of unity tests
// =============================================================================

TEST_F(BezierBasis2DTest, PartitionOfUnity) {
    // Bernstein polynomials sum to 1 at any point
    std::vector<std::pair<Real, Real>> test_points = {
        {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0},
        {0.5, 0.5}, {0.25, 0.75}, {0.1, 0.9}, {0.33, 0.67}
    };

    for (const auto& [u, v] : test_points) {
        VecX phi = basis.evaluate(u, v);
        EXPECT_NEAR(phi.sum(), 1.0, TOLERANCE)
            << "Failed at u=" << u << ", v=" << v;
    }
}

TEST_F(BezierBasis2DTest, NonNegativity) {
    // Bernstein polynomials are non-negative on [0,1]
    for (int i = 0; i <= 10; ++i) {
        for (int j = 0; j <= 10; ++j) {
            Real u = static_cast<Real>(i) / 10.0;
            Real v = static_cast<Real>(j) / 10.0;
            VecX phi = basis.evaluate(u, v);

            for (int k = 0; k < 36; ++k) {
                EXPECT_GE(phi(k), -TOLERANCE)
                    << "Negative value at u=" << u << ", v=" << v << ", dof=" << k;
            }
        }
    }
}

// =============================================================================
// Endpoint interpolation tests (Bernstein property: only B_0(0)=1, B_n(1)=1)
// =============================================================================

TEST_F(BezierBasis2DTest, EndpointInterpolation) {
    // Bernstein basis has the property that only B_{0,n}(0)=1 and B_{n,n}(1)=1
    // At corners of [0,1]^2:
    // (0,0): only B_{0,0} = 1, all others = 0
    // (1,0): only B_{n,0} = 1
    // (0,1): only B_{0,n} = 1
    // (1,1): only B_{n,n} = 1

    // Corner (0,0)
    VecX phi00 = basis.evaluate(0.0, 0.0);
    EXPECT_NEAR(phi00(BezierBasis2D::dof_index(0, 0)), 1.0, TOLERANCE);
    for (int k = 0; k < 36; ++k) {
        if (k != BezierBasis2D::dof_index(0, 0)) {
            EXPECT_NEAR(phi00(k), 0.0, TOLERANCE);
        }
    }

    // Corner (1,0)
    VecX phi10 = basis.evaluate(1.0, 0.0);
    EXPECT_NEAR(phi10(BezierBasis2D::dof_index(5, 0)), 1.0, TOLERANCE);

    // Corner (0,1)
    VecX phi01 = basis.evaluate(0.0, 1.0);
    EXPECT_NEAR(phi01(BezierBasis2D::dof_index(0, 5)), 1.0, TOLERANCE);

    // Corner (1,1)
    VecX phi11 = basis.evaluate(1.0, 1.0);
    EXPECT_NEAR(phi11(BezierBasis2D::dof_index(5, 5)), 1.0, TOLERANCE);
}

// =============================================================================
// Derivative tests
// =============================================================================

TEST_F(BezierBasis2DTest, DerivativesSumToZero) {
    // Derivatives of sum(B_i) = 1 should be 0
    std::vector<std::pair<Real, Real>> test_points = {
        {0.5, 0.5}, {0.25, 0.75}, {0.1, 0.3}
    };

    for (const auto& [u, v] : test_points) {
        VecX du = basis.evaluate_du(u, v);
        VecX dv = basis.evaluate_dv(u, v);

        EXPECT_NEAR(du.sum(), 0.0, TOLERANCE)
            << "du sum should be 0 at u=" << u << ", v=" << v;
        EXPECT_NEAR(dv.sum(), 0.0, TOLERANCE)
            << "dv sum should be 0 at u=" << u << ", v=" << v;
    }
}

TEST_F(BezierBasis2DTest, DerivativesFiniteDifference) {
    // Compare analytic derivatives to finite differences
    Real u = 0.3, v = 0.7;

    VecX phi = basis.evaluate(u, v);
    VecX phi_u_plus = basis.evaluate(u + FD_DELTA, v);
    VecX phi_u_minus = basis.evaluate(u - FD_DELTA, v);
    VecX phi_v_plus = basis.evaluate(u, v + FD_DELTA);
    VecX phi_v_minus = basis.evaluate(u, v - FD_DELTA);

    VecX du_fd = (phi_u_plus - phi_u_minus) / (2.0 * FD_DELTA);
    VecX dv_fd = (phi_v_plus - phi_v_minus) / (2.0 * FD_DELTA);

    VecX du_analytic = basis.evaluate_du(u, v);
    VecX dv_analytic = basis.evaluate_dv(u, v);

    for (int k = 0; k < 36; ++k) {
        EXPECT_NEAR(du_analytic(k), du_fd(k), LOOSE_TOLERANCE)
            << "du mismatch at dof " << k;
        EXPECT_NEAR(dv_analytic(k), dv_fd(k), LOOSE_TOLERANCE)
            << "dv mismatch at dof " << k;
    }
}

TEST_F(BezierBasis2DTest, SecondDerivativesFiniteDifference) {
    Real u = 0.4, v = 0.6;
    Real h = 1e-4;  // Larger step for better second derivative accuracy

    // Second derivatives via finite difference on phi directly
    VecX phi_c = basis.evaluate(u, v);
    VecX phi_u_plus = basis.evaluate(u + h, v);
    VecX phi_u_minus = basis.evaluate(u - h, v);
    VecX phi_v_plus = basis.evaluate(u, v + h);
    VecX phi_v_minus = basis.evaluate(u, v - h);

    VecX d2u_fd = (phi_u_plus - 2.0 * phi_c + phi_u_minus) / (h * h);
    VecX d2v_fd = (phi_v_plus - 2.0 * phi_c + phi_v_minus) / (h * h);

    // Central difference for mixed derivative
    VecX phi_pp = basis.evaluate(u + h, v + h);
    VecX phi_pm = basis.evaluate(u + h, v - h);
    VecX phi_mp = basis.evaluate(u - h, v + h);
    VecX phi_mm = basis.evaluate(u - h, v - h);
    VecX d2uv_fd = (phi_pp - phi_pm - phi_mp + phi_mm) / (4.0 * h * h);

    VecX d2u, d2v, d2uv;
    basis.evaluate_second_derivatives(u, v, d2u, d2v, d2uv);

    for (int k = 0; k < 36; ++k) {
        EXPECT_NEAR(d2u(k), d2u_fd(k), 1e-3)  // Lower tolerance for second derivatives
            << "d2u mismatch at dof " << k;
        EXPECT_NEAR(d2v(k), d2v_fd(k), 1e-3)
            << "d2v mismatch at dof " << k;
        EXPECT_NEAR(d2uv(k), d2uv_fd(k), 1e-3)
            << "d2uv mismatch at dof " << k;
    }
}

// =============================================================================
// Scalar interpolation tests
// =============================================================================

TEST_F(BezierBasis2DTest, ScalarInterpolationLinear) {
    // For linear function f(u,v) = a*u + b*v + c, Bezier should reproduce exactly
    Real a = 2.0, b = 3.0, c = 1.0;

    // Set control points to match linear function
    VecX coeffs(36);
    for (int j = 0; j < 6; ++j) {
        for (int i = 0; i < 6; ++i) {
            Real u = static_cast<Real>(i) / 5.0;
            Real v = static_cast<Real>(j) / 5.0;
            coeffs(BezierBasis2D::dof_index(i, j)) = a * u + b * v + c;
        }
    }

    // Test at various points
    std::vector<std::pair<Real, Real>> test_points = {
        {0.0, 0.0}, {1.0, 1.0}, {0.5, 0.5}, {0.2, 0.8}
    };

    for (const auto& [u, v] : test_points) {
        Real expected = a * u + b * v + c;
        Real computed = basis.evaluate_scalar(coeffs, u, v);
        EXPECT_NEAR(computed, expected, TOLERANCE)
            << "Linear interpolation failed at u=" << u << ", v=" << v;
    }
}

TEST_F(BezierBasis2DTest, ScalarBezierCurveCornerInterpolation) {
    // Bernstein/Bézier surfaces interpolate at corners
    // Setting corner control points should give those values at corners
    VecX coeffs = VecX::Zero(36);
    coeffs(BezierBasis2D::dof_index(0, 0)) = 1.0;  // (0,0) corner
    coeffs(BezierBasis2D::dof_index(5, 0)) = 2.0;  // (1,0) corner
    coeffs(BezierBasis2D::dof_index(0, 5)) = 3.0;  // (0,1) corner
    coeffs(BezierBasis2D::dof_index(5, 5)) = 4.0;  // (1,1) corner

    // At corners, value should equal the corner control point
    EXPECT_NEAR(basis.evaluate_scalar(coeffs, 0.0, 0.0), 1.0, TOLERANCE);
    EXPECT_NEAR(basis.evaluate_scalar(coeffs, 1.0, 0.0), 2.0, TOLERANCE);
    EXPECT_NEAR(basis.evaluate_scalar(coeffs, 0.0, 1.0), 3.0, TOLERANCE);
    EXPECT_NEAR(basis.evaluate_scalar(coeffs, 1.0, 1.0), 4.0, TOLERANCE);
}

// =============================================================================
// Corner and edge tests
// =============================================================================

TEST_F(BezierBasis2DTest, CornerDofs) {
    EXPECT_EQ(basis.corner_dof(0), BezierBasis2D::dof_index(0, 0));  // (0,0)
    EXPECT_EQ(basis.corner_dof(1), BezierBasis2D::dof_index(5, 0));  // (1,0)
    EXPECT_EQ(basis.corner_dof(2), BezierBasis2D::dof_index(0, 5));  // (0,1)
    EXPECT_EQ(basis.corner_dof(3), BezierBasis2D::dof_index(5, 5));  // (1,1)
}

TEST_F(BezierBasis2DTest, EdgeDofs) {
    // Edge 0: u=0 (left)
    auto edge0 = basis.edge_dofs(0);
    EXPECT_EQ(edge0.size(), 6u);
    for (int j = 0; j < 6; ++j) {
        EXPECT_EQ(edge0[j], BezierBasis2D::dof_index(0, j));
    }

    // Edge 1: u=1 (right)
    auto edge1 = basis.edge_dofs(1);
    EXPECT_EQ(edge1.size(), 6u);
    for (int j = 0; j < 6; ++j) {
        EXPECT_EQ(edge1[j], BezierBasis2D::dof_index(5, j));
    }

    // Edge 2: v=0 (bottom)
    auto edge2 = basis.edge_dofs(2);
    EXPECT_EQ(edge2.size(), 6u);
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(edge2[i], BezierBasis2D::dof_index(i, 0));
    }

    // Edge 3: v=1 (top)
    auto edge3 = basis.edge_dofs(3);
    EXPECT_EQ(edge3.size(), 6u);
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(edge3[i], BezierBasis2D::dof_index(i, 5));
    }
}

TEST_F(BezierBasis2DTest, CornerParams) {
    Vec2 p0 = basis.corner_param(0);
    Vec2 p1 = basis.corner_param(1);
    Vec2 p2 = basis.corner_param(2);
    Vec2 p3 = basis.corner_param(3);

    EXPECT_NEAR(p0(0), 0.0, TOLERANCE);
    EXPECT_NEAR(p0(1), 0.0, TOLERANCE);

    EXPECT_NEAR(p1(0), 1.0, TOLERANCE);
    EXPECT_NEAR(p1(1), 0.0, TOLERANCE);

    EXPECT_NEAR(p2(0), 0.0, TOLERANCE);
    EXPECT_NEAR(p2(1), 1.0, TOLERANCE);

    EXPECT_NEAR(p3(0), 1.0, TOLERANCE);
    EXPECT_NEAR(p3(1), 1.0, TOLERANCE);
}

// =============================================================================
// Higher derivative tests
// =============================================================================

TEST_F(BezierBasis2DTest, ThirdDerivatives) {
    Real u = 0.5, v = 0.5;
    Real h = 1e-4;  // Larger step for better finite diff accuracy on high derivatives

    // Test via finite difference on second derivatives
    VecX d2u_v_plus = basis.evaluate_d2u(u, v + h);
    VecX d2u_v_minus = basis.evaluate_d2u(u, v - h);
    VecX d3uuv_fd = (d2u_v_plus - d2u_v_minus) / (2.0 * h);

    VecX d3uuv = basis.evaluate_d3uuv(u, v);

    // Third derivatives have lower accuracy due to finite difference errors
    for (int k = 0; k < 36; ++k) {
        EXPECT_NEAR(d3uuv(k), d3uuv_fd(k), 1e-2)
            << "d3uuv mismatch at dof " << k;
    }
}

// =============================================================================
// Coordinate conversion tests
// =============================================================================

TEST_F(BezierBasis2DTest, CoordinateConversion) {
    // Test ref_to_param and param_to_ref are inverses
    std::vector<Real> test_vals = {-1.0, -0.5, 0.0, 0.5, 1.0};

    for (Real xi : test_vals) {
        for (Real eta : test_vals) {
            Vec2 uv = BezierBasis2D::ref_to_param(xi, eta);
            Vec2 xi_eta = BezierBasis2D::param_to_ref(uv(0), uv(1));

            EXPECT_NEAR(xi_eta(0), xi, TOLERANCE);
            EXPECT_NEAR(xi_eta(1), eta, TOLERANCE);
        }
    }

    // Verify correct mapping at boundaries
    Vec2 uv_corner = BezierBasis2D::ref_to_param(-1.0, -1.0);
    EXPECT_NEAR(uv_corner(0), 0.0, TOLERANCE);
    EXPECT_NEAR(uv_corner(1), 0.0, TOLERANCE);

    Vec2 uv_center = BezierBasis2D::ref_to_param(0.0, 0.0);
    EXPECT_NEAR(uv_center(0), 0.5, TOLERANCE);
    EXPECT_NEAR(uv_center(1), 0.5, TOLERANCE);
}

}  // namespace
