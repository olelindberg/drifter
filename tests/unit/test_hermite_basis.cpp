#include "bathymetry/hermite_basis_1d.hpp"
#include "bathymetry/hermite_basis_2d.hpp"
#include "bathymetry/cubic_bezier_basis_2d.hpp"
#include <gtest/gtest.h>
#include <cmath>

using namespace drifter;

namespace {

constexpr Real TOLERANCE = 1e-12;

// =============================================================================
// 1D Hermite basis: the duality relations are the whole foundation
// =============================================================================

// H_{s,m}^(m')(s') = delta_ss' delta_mm' -- this is the only property used to
// prove that C^r continuity is structural (hermite_bathymetry_system.md S3), so
// it is checked exhaustively for every supported order.
TEST(HermiteBasis1DTest, DualityRelations) {
    for (int r = 0; r <= 2; ++r) {
        HermiteBasis1D basis(r);

        for (int i = 0; i < basis.num_dofs(); ++i) {
            const int s = basis.node_of(i);
            const int m = basis.deriv_of(i);

            for (int sp = 0; sp <= 1; ++sp) {
                for (int mp = 0; mp <= r; ++mp) {
                    const Real expected = (s == sp && m == mp) ? 1.0 : 0.0;
                    const Real actual = basis.eval(i, mp, static_cast<Real>(sp));
                    EXPECT_NEAR(actual, expected, TOLERANCE)
                        << "r=" << r << " i=" << i << " deriv=" << mp << " at t=" << sp;
                }
            }
        }
    }
}

TEST(HermiteBasis1DTest, ValueFunctionsPartitionUnity) {
    for (int r = 0; r <= 2; ++r) {
        HermiteBasis1D basis(r);

        for (Real t : {0.0, 0.17, 0.5, 0.83, 1.0}) {
            Real sum_value = 0.0;
            Real sum_deriv = 0.0;
            for (int i = 0; i < basis.num_dofs(); ++i) {
                if (basis.deriv_of(i) == 0) {
                    sum_value += basis.eval(i, 0, t);
                    sum_deriv += basis.eval(i, 1, t);
                }
            }
            // The value shape functions reproduce the constant 1 exactly, so they
            // sum to 1 and their derivatives sum to 0.
            EXPECT_NEAR(sum_value, 1.0, TOLERANCE) << "r=" << r << " t=" << t;
            EXPECT_NEAR(sum_deriv, 0.0, TOLERANCE) << "r=" << r << " t=" << t;
        }
    }
}

// The Hermite interpolant of a polynomial of degree <= p must be that polynomial
TEST(HermiteBasis1DTest, ReproducesPolynomialsUpToDegreeP) {
    for (int r = 0; r <= 2; ++r) {
        HermiteBasis1D basis(r);
        const int p = basis.degree();

        // f(t) = sum_k c_k t^k with pseudo-arbitrary coefficients
        std::vector<Real> c(static_cast<size_t>(p + 1));
        for (int k = 0; k <= p; ++k) {
            c[static_cast<size_t>(k)] = 0.7 + 0.3 * k - 0.11 * k * k;
        }

        auto f = [&](Real t, int deriv) {
            Real sum = 0.0;
            for (int k = deriv; k <= p; ++k) {
                Real falling = 1.0;
                for (int d = 0; d < deriv; ++d) {
                    falling *= static_cast<Real>(k - d);
                }
                sum += c[static_cast<size_t>(k)] * falling * std::pow(t, k - deriv);
            }
            return sum;
        };

        // Sample the DOFs from f, then check the interpolant matches f everywhere
        VecX q(basis.num_dofs());
        for (int i = 0; i < basis.num_dofs(); ++i) {
            q(i) = f(static_cast<Real>(basis.node_of(i)), basis.deriv_of(i));
        }

        for (Real t : {0.0, 0.23, 0.5, 0.77, 1.0}) {
            Real interp = 0.0;
            for (int i = 0; i < basis.num_dofs(); ++i) {
                interp += q(i) * basis.eval(i, 0, t);
            }
            EXPECT_NEAR(interp, f(t, 0), 1e-10) << "r=" << r << " t=" << t;
        }
    }
}

// =============================================================================
// 2D basis: indexing and evaluation
// =============================================================================

TEST(HermiteBasis2DTest, Dimensions) {
    EXPECT_EQ(HermiteBasis2D(0).num_dofs(), 4);
    EXPECT_EQ(HermiteBasis2D(1).num_dofs(), 16);
    EXPECT_EQ(HermiteBasis2D(2).num_dofs(), 36);

    EXPECT_EQ(HermiteBasis2D(0).degree(), 1);
    EXPECT_EQ(HermiteBasis2D(1).degree(), 3);
    EXPECT_EQ(HermiteBasis2D(2).degree(), 5);
}

TEST(HermiteBasis2DTest, DofIndexRoundTrip) {
    for (int r = 0; r <= 2; ++r) {
        HermiteBasis2D basis(r);

        for (int sy = 0; sy <= 1; ++sy) {
            for (int b = 0; b <= r; ++b) {
                for (int sx = 0; sx <= 1; ++sx) {
                    for (int a = 0; a <= r; ++a) {
                        const int dof = basis.dof_index(sx, a, sy, b);
                        ASSERT_GE(dof, 0);
                        ASSERT_LT(dof, basis.num_dofs());

                        const auto [ra, rb] = basis.deriv_order(dof);
                        EXPECT_EQ(ra, a);
                        EXPECT_EQ(rb, b);
                        EXPECT_EQ(basis.dof_to_corner(dof), sx + 2 * sy);

                        Vec2 pos = basis.control_point_position(dof);
                        EXPECT_NEAR(pos(0), static_cast<Real>(sx), TOLERANCE);
                        EXPECT_NEAR(pos(1), static_cast<Real>(sy), TOLERANCE);
                    }
                }
            }
        }
    }
}

TEST(HermiteBasis2DTest, EdgeDofsCoverBothEndpointNodes) {
    for (int r = 0; r <= 2; ++r) {
        HermiteBasis2D basis(r);
        const size_t expected = static_cast<size_t>(2 * (r + 1) * (r + 1));

        for (int edge = 0; edge < 4; ++edge) {
            auto dofs = basis.edge_dofs(edge);
            EXPECT_EQ(dofs.size(), expected) << "r=" << r << " edge=" << edge;

            // Every returned DOF must actually sit on the edge
            for (int dof : dofs) {
                Vec2 pos = basis.control_point_position(dof);
                switch (edge) {
                case 0: EXPECT_NEAR(pos(0), 0.0, TOLERANCE); break;
                case 1: EXPECT_NEAR(pos(0), 1.0, TOLERANCE); break;
                case 2: EXPECT_NEAR(pos(1), 0.0, TOLERANCE); break;
                default: EXPECT_NEAR(pos(1), 1.0, TOLERANCE); break;
                }
            }
        }
    }
}

// A DOF is dual to its own derivative functional and to no other. In 2D this is
// what lets two elements identify DOFs at a shared node unambiguously.
TEST(HermiteBasis2DTest, DofsAreDualToCornerDerivatives) {
    const int r = 1;
    HermiteBasis2D basis(r);

    // Numerically differentiate Nhat_I to order (a,b) at each corner
    auto mixed_deriv = [&](int dof, int a, int b, Real u, Real v) {
        const Real hs = 1e-4;
        // Use exact derivative evaluation through the 1D basis instead of finite
        // differences: Nhat = H_i(u) H_j(v), so d^(a+b) = H_i^(a)(u) H_j^(b)(v)
        (void)hs;
        const auto &b1 = basis.basis_1d();
        const int n1d = basis.num_nodes_1d();
        const int i = dof % n1d;
        const int j = dof / n1d;
        return b1.eval(i, a, u) * b1.eval(j, b, v);
    };

    for (int dof = 0; dof < basis.num_dofs(); ++dof) {
        const auto [a, b] = basis.deriv_order(dof);
        const int corner = basis.dof_to_corner(dof);

        for (int c = 0; c < 4; ++c) {
            Vec2 p = basis.corner_param(c);
            for (int aa = 0; aa <= r; ++aa) {
                for (int bb = 0; bb <= r; ++bb) {
                    const Real expected =
                        (c == corner && aa == a && bb == b) ? 1.0 : 0.0;
                    EXPECT_NEAR(mixed_deriv(dof, aa, bb, p(0), p(1)), expected, TOLERANCE)
                        << "dof=" << dof << " corner=" << c << " (" << aa << "," << bb << ")";
                }
            }
        }
    }
}

// =============================================================================
// Bernstein <-> Hermite change of basis M(h)
// =============================================================================

// M_{r=0} is the identity: the C0 Hermite element *is* the linear Bezier element
TEST(HermiteChangeOfBasisTest, R0IsIdentity) {
    HermiteBasis2D basis(0);
    for (Real h : {0.5, 1.0, 7.25}) {
        MatX M = basis.bernstein_change_of_basis_1d(h);
        EXPECT_LT((M - MatX::Identity(2, 2)).norm(), TOLERANCE) << "h=" << h;
    }
}

// The closed form from hermite_bathymetry_system.md S5
TEST(HermiteChangeOfBasisTest, R1MatchesClosedForm) {
    HermiteBasis2D basis(1);

    for (Real h : {0.5, 1.0, 3.0}) {
        MatX M = basis.bernstein_change_of_basis_1d(h);
        ASSERT_EQ(M.rows(), 4);
        ASSERT_EQ(M.cols(), 4);

        MatX expected(4, 4);
        expected << 1.0, 0.0,     0.0, 0.0,
                    1.0, h / 3.0, 0.0, 0.0,
                    0.0, 0.0,     1.0, -h / 3.0,
                    0.0, 0.0,     1.0, 0.0;

        EXPECT_LT((M - expected).norm(), TOLERANCE) << "h=" << h;

        // det M_{r=1}(h) = h^2 / 9
        EXPECT_NEAR(M.determinant(), h * h / 9.0, TOLERANCE) << "h=" << h;
    }
}

// M(h) is block-diagonal by node -- this is S3 restated in Bernstein language:
// two elements sharing a node generate identical control points near that node.
TEST(HermiteChangeOfBasisTest, BlockDiagonalByNode) {
    for (int r = 0; r <= 2; ++r) {
        HermiteBasis2D basis(r);
        const Real h = 2.5;
        MatX M = basis.bernstein_change_of_basis_1d(h);
        const int p = basis.degree();

        for (int k = 0; k <= r; ++k) {
            // First r+1 control points depend only on the left node's DOFs
            for (int c = r + 1; c <= p; ++c) {
                EXPECT_NEAR(M(k, c), 0.0, TOLERANCE) << "r=" << r << " (" << k << "," << c << ")";
            }
            // Last r+1 depend only on the right node's
            for (int c = 0; c <= r; ++c) {
                EXPECT_NEAR(M(p - k, c), 0.0, TOLERANCE)
                    << "r=" << r << " (" << p - k << "," << c << ")";
            }
        }
    }
}

TEST(HermiteChangeOfBasisTest, InverseRecoversCornerDerivatives) {
    HermiteBasis2D basis(1);
    const Real h = 1.7;
    MatX M = basis.bernstein_change_of_basis_1d(h);
    MatX Minv = M.inverse();

    // Arbitrary Bernstein control values
    VecX c(4);
    c << 1.3, 2.1, -0.4, 3.9;
    VecX q = Minv * c;

    // z_0 = c_0, z_0' = 3(c_1 - c_0)/h, z_1 = c_3, z_1' = 3(c_3 - c_2)/h
    EXPECT_NEAR(q(0), c(0), 1e-11);
    EXPECT_NEAR(q(1), 3.0 * (c(1) - c(0)) / h, 1e-11);
    EXPECT_NEAR(q(2), c(3), 1e-11);
    EXPECT_NEAR(q(3), 3.0 * (c(3) - c(2)) / h, 1e-11);
}

// The point of M_e: both bases span the same Q_p, so evaluating the Hermite
// surface directly and evaluating its Bernstein image must agree exactly.
TEST(HermiteChangeOfBasisTest, HermiteAndBernsteinEvaluationsAgree) {
    HermiteBasis2D hermite(1);
    CubicBezierBasis2D bezier;

    const Real dx = 2.0;
    const Real dy = 0.75;

    MatX Me = hermite.bernstein_change_of_basis(dx, dy);
    ASSERT_EQ(Me.rows(), 16);
    ASSERT_EQ(Me.cols(), 16);

    // Arbitrary Hermite DOFs (physical derivatives)
    VecX q(16);
    for (int i = 0; i < 16; ++i) {
        q(i) = std::sin(1.7 * i) + 0.3 * i;
    }

    // Hermite evaluation uses the parametric basis with the Lambda_e scaling
    VecX scal = hermite.dof_scaling(dx, dy);
    VecX q_param = q.cwiseProduct(scal);

    VecX c = Me * q;

    for (Real u : {0.0, 0.3, 0.5, 0.9, 1.0}) {
        for (Real v : {0.0, 0.25, 0.6, 1.0}) {
            const Real z_hermite = hermite.evaluate_scalar(q_param, u, v);
            const Real z_bezier = bezier.evaluate_scalar(c, u, v);
            EXPECT_NEAR(z_hermite, z_bezier, 1e-10) << "u=" << u << " v=" << v;
        }
    }
}

} // namespace
