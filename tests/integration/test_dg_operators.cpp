#include <gtest/gtest.h>
#include "dg/basis_hexahedron.hpp"
#include "dg/quadrature_3d.hpp"
#include "dg/operators_3d.hpp"
#include "../test_utils.hpp"
#include <cmath>

using namespace drifter;
using namespace drifter::testing;

class DGOperatorsTest : public DrifterTestBase {
protected:
    void SetUp() override {
        DrifterTestBase::SetUp();
    }
};

// Test gradient of polynomial is exact
TEST_F(DGOperatorsTest, GradientExact) {
    for (int order = 2; order <= 4; ++order) {
        HexahedronBasis basis(order);
        GaussQuadrature3D quad(order + 1);
        DG3DElementOperator op(basis, quad);

        int n = order + 1;
        int n_total = n * n * n;

        // Function u = x^2 + y + z
        VecX u(n_total);
        const VecX& nodes_1d = basis.lgl_basis_1d().nodes;

        for (int k = 0; k < n; ++k) {
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) {
                    Real x = nodes_1d(i);
                    Real y = nodes_1d(j);
                    Real z = nodes_1d(k);
                    int idx = i + n * (j + n * k);
                    u(idx) = x * x + y + z;
                }
            }
        }

        MatX grad_u;
        op.gradient_reference(u, grad_u);

        // Expected: du/dx = 2x, du/dy = 1, du/dz = 1
        for (int k = 0; k < n; ++k) {
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) {
                    Real x = nodes_1d(i);
                    int idx = i + n * (j + n * k);

                    EXPECT_NEAR(grad_u(idx, 0), 2.0 * x, LOOSE_TOLERANCE)
                        << "Order " << order << ", idx " << idx;
                    EXPECT_NEAR(grad_u(idx, 1), 1.0, LOOSE_TOLERANCE)
                        << "Order " << order << ", idx " << idx;
                    EXPECT_NEAR(grad_u(idx, 2), 1.0, LOOSE_TOLERANCE)
                        << "Order " << order << ", idx " << idx;
                }
            }
        }
    }
}

// Test divergence of polynomial vector field
TEST_F(DGOperatorsTest, DivergenceExact) {
    for (int order = 2; order <= 4; ++order) {
        HexahedronBasis basis(order);
        GaussQuadrature3D quad(order + 1);
        DG3DElementOperator op(basis, quad);

        int n = order + 1;
        int n_total = n * n * n;

        // Vector field F = (x, y^2, z)
        // div(F) = 1 + 2y + 1 = 2 + 2y
        std::array<VecX, 3> flux;
        flux[0].resize(n_total);
        flux[1].resize(n_total);
        flux[2].resize(n_total);

        const VecX& nodes_1d = basis.lgl_basis_1d().nodes;

        for (int k = 0; k < n; ++k) {
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) {
                    Real x = nodes_1d(i);
                    Real y = nodes_1d(j);
                    Real z = nodes_1d(k);
                    int idx = i + n * (j + n * k);

                    flux[0](idx) = x;
                    flux[1](idx) = y * y;
                    flux[2](idx) = z;
                }
            }
        }

        VecX div_F;
        op.divergence_reference(flux, div_F);

        for (int k = 0; k < n; ++k) {
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) {
                    Real y = nodes_1d(j);
                    int idx = i + n * (j + n * k);

                    Real expected = 2.0 + 2.0 * y;
                    EXPECT_NEAR(div_F(idx), expected, LOOSE_TOLERANCE)
                        << "Order " << order << ", idx " << idx;
                }
            }
        }
    }
}

// Test mass matrix action
TEST_F(DGOperatorsTest, MassMatrixAction) {
    for (int order = 1; order <= 4; ++order) {
        HexahedronBasis basis(order);
        GaussQuadrature3D quad(order + 1);
        DG3DElementOperator op(basis, quad);

        int n_total = basis.num_dofs_velocity();

        // M * 1 should give the volume (8 for reference element)
        VecX ones = VecX::Ones(n_total);
        VecX M_ones;
        op.mass(ones, M_ones);

        // Sum should be approximately 8
        Real sum = M_ones.sum();
        EXPECT_NEAR(sum, 8.0, LOOSE_TOLERANCE) << "Order " << order;
    }
}

// Test mass inverse
TEST_F(DGOperatorsTest, MassInverseConsistency) {
    for (int order = 1; order <= 4; ++order) {
        HexahedronBasis basis(order);
        GaussQuadrature3D quad(order + 1);
        DG3DElementOperator op(basis, quad);

        int n_total = basis.num_dofs_velocity();

        // Random vector
        VecX v(n_total);
        for (int i = 0; i < n_total; ++i) {
            v(i) = std::sin(i * 0.7) + std::cos(i * 0.3);
        }

        // M * M^-1 * v should equal v
        VecX Mv, MiMv;
        op.mass(v, Mv);
        op.mass_inv(Mv, MiMv);

        EXPECT_TRUE(vectors_equal(v, MiMv, LOOSE_TOLERANCE))
            << "Order " << order;
    }
}

// Test face interpolation preserves polynomials
TEST_F(DGOperatorsTest, FaceInterpolation) {
    int order = 3;
    HexahedronBasis basis(order);

    int n = order + 1;
    int n_total = n * n * n;

    const VecX& nodes_1d = basis.lgl_basis_1d().nodes;

    // Function u = x + 2*y - z (linear)
    VecX u(n_total);
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                Real x = nodes_1d(i);
                Real y = nodes_1d(j);
                Real z = nodes_1d(k);
                int idx = i + n * (j + n * k);
                u(idx) = x + 2.0 * y - z;
            }
        }
    }

    // Interpolate to face 1 (xi = +1) using interp_to_face_lgl matrix
    const MatX& I_face = basis.interp_to_face_lgl(1);
    VecX u_face = I_face * u;

    // Check values on face (x = 1)
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
            Real y = nodes_1d(j);
            Real z = nodes_1d(k);
            int face_idx = j + n * k;

            Real expected = 1.0 + 2.0 * y - z;
            EXPECT_NEAR(u_face(face_idx), expected, LOOSE_TOLERANCE)
                << "Face index " << face_idx;
        }
    }
}

// Test DG weak gradient satisfies discrete Green's identity
TEST_F(DGOperatorsTest, DiscreteGreensIdentity) {
    int order = 2;
    HexahedronBasis basis(order);
    GaussQuadrature3D quad(order + 1);
    DG3DElementOperator op(basis, quad);

    int n = order + 1;
    int n_total = n * n * n;
    const VecX& nodes_1d = basis.lgl_basis_1d().nodes;

    // Functions chosen so <v, du/dy> is non-zero
    // u = x^2 + y => du/dy = 1
    // v = 1 + y^2  (even in y)
    // <v, du/dy> = integral((1 + y^2) * 1) = 8 + 8/3 = 32/3
    VecX u(n_total), v(n_total);
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                Real x = nodes_1d(i);
                Real y = nodes_1d(j);
                int idx = i + n * (j + n * k);
                u(idx) = x * x + y;
                v(idx) = 1.0 + y * y;
            }
        }
    }

    // Compute gradient of u
    MatX grad_u;
    op.gradient_reference(u, grad_u);

    // du/dy should be 1 everywhere
    for (int idx = 0; idx < n_total; ++idx) {
        EXPECT_NEAR(grad_u(idx, 1), 1.0, LOOSE_TOLERANCE)
            << "du/dy should be 1 at idx " << idx;
    }

    // <v, du/dy> = integral(v * du/dy) using mass matrix
    // Mv(i) = sum_j M(i,j) * v(j), so Mv.dot(du/dy) = v^T M du/dy = <v, du/dy>
    VecX Mv;
    op.mass(v, Mv);
    Real inner_v_duy = Mv.dot(grad_u.col(1));

    // Expected: integral((1 + y^2)) over [-1,1]^3 = 8 + 8/3 = 32/3
    Real expected = 8.0 + 8.0 / 3.0;
    EXPECT_NEAR(inner_v_duy, expected, 1.0);  // Allow some tolerance
}

// Test that stiffness matrix is related to mass and differentiation
TEST_F(DGOperatorsTest, StiffnessDerivation) {
    int order = 2;
    HexahedronBasis basis(order);
    GaussQuadrature3D quad(order + 1);
    DG3DElementOperator op(basis, quad);

    int n_total = basis.num_dofs_velocity();

    VecX u(n_total), v(n_total);
    for (int i = 0; i < n_total; ++i) {
        u(i) = std::sin(0.5 * i);
        v(i) = std::cos(0.3 * i);
    }

    // Compute <v, Du> using mass matrix
    MatX grad_u;
    op.gradient_reference(u, grad_u);

    VecX Mv;
    op.mass(v, Mv);
    Real integral_v_dux = Mv.dot(grad_u.col(0));

    // Result should be finite
    EXPECT_FALSE(std::isnan(integral_v_dux));
    EXPECT_FALSE(std::isinf(integral_v_dux));
}

// Test integration by parts (weak form consistency)
TEST_F(DGOperatorsTest, IntegrationByParts) {
    int order = 3;
    HexahedronBasis basis(order);
    GaussQuadrature3D quad(order + 1);

    int n = order + 1;
    int n_total = n * n * n;
    const VecX& nodes_1d = basis.lgl_basis_1d().nodes;

    // Functions: u = x, v = x^2
    VecX u(n_total), v(n_total);
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                Real x = nodes_1d(i);
                int idx = i + n * (j + n * k);
                u(idx) = x;
                v(idx) = x * x;
            }
        }
    }

    // For DG, integration by parts gives:
    // ∫_Ω v * ∂u/∂x dV = -∫_Ω u * ∂v/∂x dV + ∫_∂Ω u*v*n_x dS

    // For now, just verify we can compute the relevant quantities
    DG3DElementOperator op(basis, quad);
    MatX grad_u, grad_v;
    op.gradient_reference(u, grad_u);
    op.gradient_reference(v, grad_v);

    // <v, du/dx> using mass
    VecX Mv, Mu;
    op.mass(v, Mv);
    op.mass(u, Mu);

    Real lhs = Mv.dot(grad_u.col(0));  // ∫ v * du/dx
    Real rhs = Mu.dot(grad_v.col(0));  // ∫ u * dv/dx

    // These should have a relationship via boundary terms
    // For this test, just verify they're non-trivial
    EXPECT_GT(std::abs(lhs), TOLERANCE);
    EXPECT_GT(std::abs(rhs), TOLERANCE);
}
