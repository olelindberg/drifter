#include <gtest/gtest.h>
#include "bathymetry/quintic_basis_2d.hpp"
#include <cmath>

using namespace drifter;

namespace {

constexpr Real TOLERANCE = 1e-12;
constexpr Real LOOSE_TOLERANCE = 1e-8;

class QuinticBasis2DTest : public ::testing::Test {
protected:
    QuinticBasis2D basis;
};

// =============================================================================
// Basic properties tests
// =============================================================================

TEST_F(QuinticBasis2DTest, Constants) {
    EXPECT_EQ(basis.order(), 5);
    EXPECT_EQ(basis.num_nodes_1d(), 6);
    EXPECT_EQ(basis.num_dofs(), 36);
}

TEST_F(QuinticBasis2DTest, NodePositions) {
    const VecX& nodes = basis.nodes_1d();
    EXPECT_EQ(nodes.size(), 6);

    // LGL nodes should include endpoints
    EXPECT_NEAR(nodes(0), -1.0, TOLERANCE);
    EXPECT_NEAR(nodes(5), 1.0, TOLERANCE);

    // Nodes should be strictly increasing
    for (int i = 1; i < 6; ++i) {
        EXPECT_GT(nodes(i), nodes(i - 1));
    }

    // Nodes should be symmetric about 0
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(nodes(i), -nodes(5 - i), TOLERANCE);
    }
}

TEST_F(QuinticBasis2DTest, QuadratureWeights) {
    const VecX& weights = basis.weights_1d();
    EXPECT_EQ(weights.size(), 6);

    // All weights should be positive
    for (int i = 0; i < 6; ++i) {
        EXPECT_GT(weights(i), 0.0);
    }

    // Weights should sum to 2 (integral of 1 over [-1,1])
    EXPECT_NEAR(weights.sum(), 2.0, TOLERANCE);

    // Weights should be symmetric
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(weights(i), weights(5 - i), TOLERANCE);
    }
}

// =============================================================================
// DOF indexing tests
// =============================================================================

TEST_F(QuinticBasis2DTest, DofIndexing) {
    // Test round-trip conversion
    for (int j = 0; j < 6; ++j) {
        for (int i = 0; i < 6; ++i) {
            int dof = QuinticBasis2D::dof_index(i, j);
            int i_out, j_out;
            QuinticBasis2D::dof_ij(dof, i_out, j_out);
            EXPECT_EQ(i, i_out);
            EXPECT_EQ(j, j_out);
        }
    }

    // DOFs should be unique and cover 0-35
    std::vector<bool> seen(36, false);
    for (int j = 0; j < 6; ++j) {
        for (int i = 0; i < 6; ++i) {
            int dof = QuinticBasis2D::dof_index(i, j);
            EXPECT_GE(dof, 0);
            EXPECT_LT(dof, 36);
            EXPECT_FALSE(seen[dof]);
            seen[dof] = true;
        }
    }
}

TEST_F(QuinticBasis2DTest, NodePosition) {
    const VecX& nodes = basis.nodes_1d();

    for (int dof = 0; dof < 36; ++dof) {
        int i, j;
        QuinticBasis2D::dof_ij(dof, i, j);
        Vec2 pos = basis.node_position(dof);
        EXPECT_NEAR(pos(0), nodes(i), TOLERANCE);
        EXPECT_NEAR(pos(1), nodes(j), TOLERANCE);
    }
}

// =============================================================================
// Basis function evaluation tests
// =============================================================================

TEST_F(QuinticBasis2DTest, PartitionOfUnity) {
    // Basis functions should sum to 1 at any point
    std::vector<Vec2> test_points = {
        {0.0, 0.0},
        {-1.0, -1.0},
        {1.0, 1.0},
        {0.5, -0.3},
        {-0.7, 0.8}
    };

    for (const auto& pt : test_points) {
        VecX phi = basis.evaluate(pt(0), pt(1));
        EXPECT_NEAR(phi.sum(), 1.0, TOLERANCE)
            << "Failed at point (" << pt(0) << ", " << pt(1) << ")";
    }
}

TEST_F(QuinticBasis2DTest, KroneckerDelta) {
    // phi_i(x_j) = delta_ij at nodes
    for (int test_dof = 0; test_dof < 36; ++test_dof) {
        Vec2 pos = basis.node_position(test_dof);
        VecX phi = basis.evaluate(pos(0), pos(1));

        for (int dof = 0; dof < 36; ++dof) {
            Real expected = (dof == test_dof) ? 1.0 : 0.0;
            EXPECT_NEAR(phi(dof), expected, TOLERANCE)
                << "Kronecker delta failed for dof=" << dof
                << " at node " << test_dof;
        }
    }
}

TEST_F(QuinticBasis2DTest, PolynomialReproduction) {
    // Quintic basis should exactly reproduce polynomials up to degree 5
    // Test with f(x,y) = x^3 * y^2

    auto f = [](Real x, Real y) { return x*x*x * y*y; };

    // Sample f at nodes
    VecX coeffs(36);
    for (int dof = 0; dof < 36; ++dof) {
        Vec2 pos = basis.node_position(dof);
        coeffs(dof) = f(pos(0), pos(1));
    }

    // Evaluate interpolant at test points
    std::vector<Vec2> test_points = {
        {0.0, 0.0},
        {0.5, 0.5},
        {-0.3, 0.7},
        {0.8, -0.4}
    };

    for (const auto& pt : test_points) {
        VecX phi = basis.evaluate(pt(0), pt(1));
        Real interpolated = phi.dot(coeffs);
        Real exact = f(pt(0), pt(1));
        EXPECT_NEAR(interpolated, exact, TOLERANCE)
            << "Polynomial reproduction failed at (" << pt(0) << ", " << pt(1) << ")";
    }
}

// =============================================================================
// Gradient tests
// =============================================================================

TEST_F(QuinticBasis2DTest, GradientPartitionOfUnity) {
    // Gradient of sum of basis functions should be zero
    std::vector<Vec2> test_points = {
        {0.0, 0.0},
        {0.5, -0.3},
        {-0.7, 0.8}
    };

    for (const auto& pt : test_points) {
        MatX grad = basis.evaluate_gradient(pt(0), pt(1));
        Real sum_dxi = grad.col(0).sum();
        Real sum_deta = grad.col(1).sum();
        EXPECT_NEAR(sum_dxi, 0.0, TOLERANCE);
        EXPECT_NEAR(sum_deta, 0.0, TOLERANCE);
    }
}

TEST_F(QuinticBasis2DTest, GradientAccuracy) {
    // Compare gradient to finite differences
    Real h = 1e-6;

    Vec2 pt(0.3, -0.4);
    MatX grad = basis.evaluate_gradient(pt(0), pt(1));

    // Finite difference in xi direction
    VecX phi_plus = basis.evaluate(pt(0) + h, pt(1));
    VecX phi_minus = basis.evaluate(pt(0) - h, pt(1));
    VecX fd_dxi = (phi_plus - phi_minus) / (2.0 * h);

    // Finite difference in eta direction
    phi_plus = basis.evaluate(pt(0), pt(1) + h);
    phi_minus = basis.evaluate(pt(0), pt(1) - h);
    VecX fd_deta = (phi_plus - phi_minus) / (2.0 * h);

    for (int dof = 0; dof < 36; ++dof) {
        EXPECT_NEAR(grad(dof, 0), fd_dxi(dof), 1e-5)
            << "Gradient dxi mismatch at dof=" << dof;
        EXPECT_NEAR(grad(dof, 1), fd_deta(dof), 1e-5)
            << "Gradient deta mismatch at dof=" << dof;
    }
}

TEST_F(QuinticBasis2DTest, GradientPolynomialReproduction) {
    // Test gradient of f(x,y) = x^2 * y^3
    // df/dx = 2*x*y^3, df/dy = 3*x^2*y^2

    auto f = [](Real x, Real y) { return x*x * y*y*y; };
    auto dfdx = [](Real x, Real y) { return 2.0*x * y*y*y; };
    auto dfdy = [](Real x, Real y) { return 3.0*x*x * y*y; };

    // Sample f at nodes
    VecX coeffs(36);
    for (int dof = 0; dof < 36; ++dof) {
        Vec2 pos = basis.node_position(dof);
        coeffs(dof) = f(pos(0), pos(1));
    }

    // Test gradient at interior points
    std::vector<Vec2> test_points = {
        {0.3, 0.4},
        {-0.5, 0.6},
        {0.7, -0.2}
    };

    for (const auto& pt : test_points) {
        MatX grad = basis.evaluate_gradient(pt(0), pt(1));

        Real interp_dfdx = grad.col(0).dot(coeffs);
        Real interp_dfdy = grad.col(1).dot(coeffs);

        Real exact_dfdx = dfdx(pt(0), pt(1));
        Real exact_dfdy = dfdy(pt(0), pt(1));

        EXPECT_NEAR(interp_dfdx, exact_dfdx, LOOSE_TOLERANCE)
            << "Gradient x failed at (" << pt(0) << ", " << pt(1) << ")";
        EXPECT_NEAR(interp_dfdy, exact_dfdy, LOOSE_TOLERANCE)
            << "Gradient y failed at (" << pt(0) << ", " << pt(1) << ")";
    }
}

// =============================================================================
// Second derivative and Laplacian tests
// =============================================================================

TEST_F(QuinticBasis2DTest, LaplacianAccuracy) {
    // Compare Laplacian to finite differences
    Real h = 1e-5;

    Vec2 pt(0.3, -0.4);
    VecX lap = basis.evaluate_laplacian(pt(0), pt(1));

    // Second-order central differences
    VecX phi_c = basis.evaluate(pt(0), pt(1));

    VecX phi_xp = basis.evaluate(pt(0) + h, pt(1));
    VecX phi_xm = basis.evaluate(pt(0) - h, pt(1));
    VecX d2_dxi2_fd = (phi_xp - 2.0*phi_c + phi_xm) / (h*h);

    VecX phi_yp = basis.evaluate(pt(0), pt(1) + h);
    VecX phi_ym = basis.evaluate(pt(0), pt(1) - h);
    VecX d2_deta2_fd = (phi_yp - 2.0*phi_c + phi_ym) / (h*h);

    VecX lap_fd = d2_dxi2_fd + d2_deta2_fd;

    for (int dof = 0; dof < 36; ++dof) {
        EXPECT_NEAR(lap(dof), lap_fd(dof), 1e-4)
            << "Laplacian mismatch at dof=" << dof;
    }
}

TEST_F(QuinticBasis2DTest, LaplacianPolynomialReproduction) {
    // Test Laplacian of f(x,y) = x^4 + y^4
    // Laplacian = 12*x^2 + 12*y^2

    auto f = [](Real x, Real y) { return x*x*x*x + y*y*y*y; };
    auto lap_exact = [](Real x, Real y) { return 12.0*x*x + 12.0*y*y; };

    // Sample f at nodes
    VecX coeffs(36);
    for (int dof = 0; dof < 36; ++dof) {
        Vec2 pos = basis.node_position(dof);
        coeffs(dof) = f(pos(0), pos(1));
    }

    // Test Laplacian at interior points
    std::vector<Vec2> test_points = {
        {0.0, 0.0},
        {0.3, 0.4},
        {-0.5, 0.6}
    };

    for (const auto& pt : test_points) {
        VecX lap = basis.evaluate_laplacian(pt(0), pt(1));
        Real interp_lap = lap.dot(coeffs);
        Real exact_lap = lap_exact(pt(0), pt(1));

        EXPECT_NEAR(interp_lap, exact_lap, LOOSE_TOLERANCE)
            << "Laplacian failed at (" << pt(0) << ", " << pt(1) << ")";
    }
}

TEST_F(QuinticBasis2DTest, HessianSymmetry) {
    // Hessian matrices should be symmetric
    Vec2 pt(0.3, -0.4);
    auto hessians = basis.evaluate_hessian(pt(0), pt(1));

    for (int dof = 0; dof < 36; ++dof) {
        EXPECT_NEAR(hessians[dof](0, 1), hessians[dof](1, 0), TOLERANCE)
            << "Hessian not symmetric at dof=" << dof;
    }
}

// =============================================================================
// Edge and corner DOF tests
// =============================================================================

TEST_F(QuinticBasis2DTest, EdgeDofs) {
    // Edge 0: xi = -1 (left edge)
    auto edge0 = basis.edge_dofs(0);
    EXPECT_EQ(edge0.size(), 6u);
    for (int j = 0; j < 6; ++j) {
        int i, jj;
        QuinticBasis2D::dof_ij(edge0[j], i, jj);
        EXPECT_EQ(i, 0);
    }

    // Edge 1: xi = +1 (right edge)
    auto edge1 = basis.edge_dofs(1);
    EXPECT_EQ(edge1.size(), 6u);
    for (int j = 0; j < 6; ++j) {
        int i, jj;
        QuinticBasis2D::dof_ij(edge1[j], i, jj);
        EXPECT_EQ(i, 5);
    }

    // Edge 2: eta = -1 (bottom edge)
    auto edge2 = basis.edge_dofs(2);
    EXPECT_EQ(edge2.size(), 6u);
    for (int idx = 0; idx < 6; ++idx) {
        int i, j;
        QuinticBasis2D::dof_ij(edge2[idx], i, j);
        EXPECT_EQ(j, 0);
    }

    // Edge 3: eta = +1 (top edge)
    auto edge3 = basis.edge_dofs(3);
    EXPECT_EQ(edge3.size(), 6u);
    for (int idx = 0; idx < 6; ++idx) {
        int i, j;
        QuinticBasis2D::dof_ij(edge3[idx], i, j);
        EXPECT_EQ(j, 5);
    }
}

TEST_F(QuinticBasis2DTest, CornerDofs) {
    // Corner 0: (-1, -1)
    Vec2 pos0 = basis.node_position(basis.corner_dof(0));
    EXPECT_NEAR(pos0(0), -1.0, TOLERANCE);
    EXPECT_NEAR(pos0(1), -1.0, TOLERANCE);

    // Corner 1: (+1, -1)
    Vec2 pos1 = basis.node_position(basis.corner_dof(1));
    EXPECT_NEAR(pos1(0), 1.0, TOLERANCE);
    EXPECT_NEAR(pos1(1), -1.0, TOLERANCE);

    // Corner 2: (-1, +1)
    Vec2 pos2 = basis.node_position(basis.corner_dof(2));
    EXPECT_NEAR(pos2(0), -1.0, TOLERANCE);
    EXPECT_NEAR(pos2(1), 1.0, TOLERANCE);

    // Corner 3: (+1, +1)
    Vec2 pos3 = basis.node_position(basis.corner_dof(3));
    EXPECT_NEAR(pos3(0), 1.0, TOLERANCE);
    EXPECT_NEAR(pos3(1), 1.0, TOLERANCE);
}

TEST_F(QuinticBasis2DTest, BoundaryInteriorClassification) {
    // 20 boundary DOFs (6*4 - 4 corners counted once)
    // 16 interior DOFs
    int boundary_count = 0;
    int interior_count = 0;

    for (int dof = 0; dof < 36; ++dof) {
        if (basis.is_boundary_dof(dof)) {
            boundary_count++;
        } else {
            interior_count++;
        }
    }

    EXPECT_EQ(boundary_count, 20);
    EXPECT_EQ(interior_count, 16);

    // Interior DOFs method
    auto interior = basis.interior_dofs();
    EXPECT_EQ(interior.size(), 16u);
    for (int dof : interior) {
        EXPECT_TRUE(basis.is_interior_dof(dof));
    }
}

// =============================================================================
// Derivative matrix tests
// =============================================================================

TEST_F(QuinticBasis2DTest, DerivativeMatrixRowSum) {
    // Rows of derivative matrix should sum to zero (d/dx of constant = 0)
    const MatX& D = basis.derivative_matrix_1d();
    EXPECT_EQ(D.rows(), 6);
    EXPECT_EQ(D.cols(), 6);

    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(D.row(i).sum(), 0.0, TOLERANCE)
            << "Row " << i << " of D matrix doesn't sum to zero";
    }
}

TEST_F(QuinticBasis2DTest, SecondDerivativeMatrixRowSum) {
    // Rows of second derivative matrix should sum to zero
    const MatX& D2 = basis.second_derivative_matrix_1d();
    EXPECT_EQ(D2.rows(), 6);
    EXPECT_EQ(D2.cols(), 6);

    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(D2.row(i).sum(), 0.0, TOLERANCE)
            << "Row " << i << " of D2 matrix doesn't sum to zero";
    }
}

}  // namespace
