#include "bathymetry/cg_hermite_dof_manager.hpp"
#include "bathymetry/cubic_bezier_basis_2d.hpp"
#include "bathymetry/hermite_basis_2d.hpp"
#include "bathymetry/linear_bezier_basis_2d.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include <gtest/gtest.h>
#include <cmath>

using namespace drifter;

namespace {

constexpr Real TOLERANCE = 1e-12;

// =============================================================================
// The midpoint matrix G_r
// =============================================================================

// The closed forms from docs/hermite_bathymetry_system.md S8
TEST(HermiteMidpointMatrixTest, MatchesClosedForm) {
    // r = 0: G_0 = [1/2  1/2] -- the de Casteljau midpoint weight
    {
        HermiteBasis2D basis(0);
        MatX G = basis.midpoint_matrix(3.0);
        ASSERT_EQ(G.rows(), 1);
        ASSERT_EQ(G.cols(), 2);
        EXPECT_NEAR(G(0, 0), 0.5, TOLERANCE);
        EXPECT_NEAR(G(0, 1), 0.5, TOLERANCE);
    }

    // r = 1, parametric (h = 1):
    //   [  1/2   1/8   1/2  -1/8 ]
    //   [ -3/2  -1/4   3/2  -1/4 ]
    {
        HermiteBasis2D basis(1);
        MatX G = basis.midpoint_matrix(1.0);
        ASSERT_EQ(G.rows(), 2);
        ASSERT_EQ(G.cols(), 4);

        MatX expected(2, 4);
        expected << 0.5, 0.125, 0.5, -0.125,
                   -1.5, -0.25, 1.5, -0.25;
        EXPECT_LT((G - expected).norm(), TOLERANCE);
    }

    // r = 1, physical: [G^phys]_ij = h^(m_j - i) [G]_ij
    {
        HermiteBasis2D basis(1);
        const Real h = 4.0;
        MatX G = basis.midpoint_matrix(h);

        MatX expected(2, 4);
        expected << 0.5, h / 8.0, 0.5, -h / 8.0,
                    -1.5 / h, -0.25, 1.5 / h, -0.25;
        EXPECT_LT((G - expected).norm(), TOLERANCE) << G;
    }
}

// The cross-check of docs/hermite_bathymetry_system.md S8: G_r can be derived a
// second way, entirely inside the Bernstein basis, by subdividing the coarse
// curve with de Casteljau and reading off the fine element's corner DOFs:
//
//     G_r^phys(h) = [ M(h/2)^-1 S_left M(h) ]_lower block
//
// Two independent derivations - Hermite evaluation at t = 1/2, and de Casteljau
// composed with the change of basis - must produce the same matrix.
TEST(HermiteMidpointMatrixTest, MatchesDeCasteljauSubdivision) {
    for (Real h : {1.0, 2.0, 7.5}) {
        // r = 1, against the cubic Bezier subdivision matrix
        {
            HermiteBasis2D basis(1);
            CubicBezierBasis2D bezier;

            const MatX S_left = bezier.compute_1d_extraction_matrix(0.0, 0.5);
            const MatX M_coarse = basis.bernstein_change_of_basis_1d(h);
            const MatX M_fine = basis.bernstein_change_of_basis_1d(h / 2.0);

            const MatX composed = M_fine.inverse() * S_left * M_coarse;

            // Lower block: the last r+1 rows are the fine element's right node,
            // which sits at the coarse edge's midpoint.
            const MatX lower = composed.bottomRows(2);

            EXPECT_LT((lower - basis.midpoint_matrix(h)).norm(), 1e-10)
                << "h=" << h << "\n" << lower << "\nvs\n" << basis.midpoint_matrix(h);
        }

        // r = 0, against the linear Bezier subdivision matrix
        {
            HermiteBasis2D basis(0);
            LinearBezierBasis2D bezier;

            const MatX S_left = bezier.compute_1d_extraction_matrix(0.0, 0.5);
            const MatX M_coarse = basis.bernstein_change_of_basis_1d(h);
            const MatX M_fine = basis.bernstein_change_of_basis_1d(h / 2.0);

            const MatX composed = M_fine.inverse() * S_left * M_coarse;
            EXPECT_LT((composed.bottomRows(1) - basis.midpoint_matrix(h)).norm(), 1e-10)
                << "h=" << h;
        }
    }
}

// G_r reproduces every polynomial of degree <= 2r+1 exactly: sampling the master
// DOFs from such a polynomial yields its true value and derivatives at the
// midpoint. This is what makes the T-junction constraint exact rather than
// interpolatory. See docs/hermite_bathymetry_system.md S8.
TEST(HermiteMidpointMatrixTest, ReproducesPolynomialsUpToDegreeP) {
    for (int r = 0; r <= 2; ++r) {
        HermiteBasis2D basis(r);
        const int p = basis.degree();
        const Real h = 3.0;
        MatX G = basis.midpoint_matrix(h);

        // f(t) = sum_k c_k t^k on the physical coarse edge [0, h]
        std::vector<Real> c(static_cast<size_t>(p + 1));
        for (int k = 0; k <= p; ++k) {
            c[static_cast<size_t>(k)] = 1.3 - 0.4 * k + 0.09 * k * k;
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

        // Master DOFs: physical derivatives at the two coarse endpoints
        VecX q(p + 1);
        for (int j = 0; j <= p; ++j) {
            const int s_j = j / (r + 1);
            const int m_j = j % (r + 1);
            q(j) = f(s_j * h, m_j);
        }

        const VecX slave = G * q;
        for (int i = 0; i <= r; ++i) {
            EXPECT_NEAR(slave(i), f(h / 2.0, i), 1e-9)
                << "r=" << r << " derivative order " << i;
        }
    }
}

// Only the *value* DOF columns form a partition of unity; the derivative columns
// must not be included. Unlike de Casteljau weights, Hermite constraint weights
// are neither non-negative nor normalised.
TEST(HermiteMidpointMatrixTest, OnlyValueColumnsFormPartitionOfUnity) {
    for (int r = 0; r <= 2; ++r) {
        HermiteBasis2D basis(r);
        MatX G = basis.midpoint_matrix(2.0);

        Real value_column_sum = 0.0;
        for (int j = 0; j < G.cols(); ++j) {
            if (j % (r + 1) == 0) { // a value DOF
                value_column_sum += G(0, j);
            }
        }
        EXPECT_NEAR(value_column_sum, 1.0, TOLERANCE) << "r=" << r;
    }
}

// =============================================================================
// DOF manager: node sharing and hanging node constraints
// =============================================================================

class HermiteDofManagerTest : public ::testing::Test {
protected:
    QuadtreeAdapter uniform(int n) {
        QuadtreeAdapter mesh;
        mesh.build_uniform(0.0, 100.0, 0.0, 100.0, n, n);
        return mesh;
    }
};

TEST_F(HermiteDofManagerTest, ConformingMeshHasNoConstraints) {
    for (int r = 0; r <= 1; ++r) {
        auto mesh = uniform(4);
        CGHermiteDofManager dm(mesh, r);

        const Index expected_nodes = 5 * 5;
        EXPECT_EQ(dm.num_nodes(), expected_nodes) << "r=" << r;
        EXPECT_EQ(dm.num_global_dofs(), expected_nodes * (r + 1) * (r + 1)) << "r=" << r;
        EXPECT_EQ(dm.num_constraints(), 0) << "r=" << r;
        EXPECT_EQ(dm.num_free_dofs(), dm.num_global_dofs()) << "r=" << r;
    }
}

// All (r+1)^2 DOFs of a shared node are identified between neighbouring
// elements. This identification is what makes C^r structural.
TEST_F(HermiteDofManagerTest, NeighbouringElementsShareAllNodeDofs) {
    auto mesh = uniform(2);
    CGHermiteDofManager dm(mesh, 1);
    HermiteBasis2D basis(1);

    // Element 0 and its right neighbour share the two nodes on their common edge
    const Index e0 = mesh.find_element(Vec2(25.0, 25.0));
    const Index e1 = mesh.find_element(Vec2(75.0, 25.0));
    ASSERT_GE(e0, 0);
    ASSERT_GE(e1, 0);

    // e0's right edge DOFs must equal e1's left edge DOFs, as sets
    std::vector<Index> right, left;
    for (int d : basis.edge_dofs(1)) {
        right.push_back(dm.element_dofs(e0)[static_cast<size_t>(d)]);
    }
    for (int d : basis.edge_dofs(0)) {
        left.push_back(dm.element_dofs(e1)[static_cast<size_t>(d)]);
    }
    std::sort(right.begin(), right.end());
    std::sort(left.begin(), left.end());
    EXPECT_EQ(right, left);
    EXPECT_EQ(right.size(), 8u) << "2 nodes x 4 DOFs";
}

TEST_F(HermiteDofManagerTest, NonConformingMeshProducesHangingConstraints) {
    auto mesh = uniform(4);
    mesh.refine_where([&](Index e, const auto &) {
        const auto &b = mesh.element_bounds(e);
        return b.xmax <= 50.0 && b.ymax <= 50.0;
    });

    CGHermiteDofManager dm(mesh, 1);
    EXPECT_GT(dm.num_constraints(), 0);
    EXPECT_LT(dm.num_free_dofs(), dm.num_global_dofs());

    // Every constraint is a substitution onto free masters only: the closure
    // must have eliminated any master that is itself a slave.
    for (const auto &c : dm.constraints()) {
        EXPECT_TRUE(dm.is_constrained(c.slave_dof));
        EXPECT_EQ(c.master_dofs.size(), c.weights.size());
        for (Index m : c.master_dofs) {
            EXPECT_FALSE(dm.is_constrained(m))
                << "master " << m << " of slave " << c.slave_dof << " is itself constrained";
        }
    }
}

// 2:1 balance permits chained constraints, so the closure must reach a fixpoint
// on a mesh with three refinement levels.
TEST_F(HermiteDofManagerTest, ChainedConstraintsCloseOnThreeLevelMesh) {
    auto mesh = uniform(4);
    mesh.refine_where([&](Index e, const auto &) {
        const auto &b = mesh.element_bounds(e);
        return b.xmax <= 50.0 && b.ymax <= 50.0;
    });
    mesh.refine_where([&](Index e, const auto &) {
        const auto &b = mesh.element_bounds(e);
        return b.xmax <= 25.0 && b.ymax <= 25.0;
    });

    for (int r = 0; r <= 1; ++r) {
        CGHermiteDofManager dm(mesh, r);
        EXPECT_GT(dm.num_constraints(), 0) << "r=" << r;

        for (const auto &c : dm.constraints()) {
            for (Index m : c.master_dofs) {
                EXPECT_FALSE(dm.is_constrained(m)) << "r=" << r << ": unclosed chain";
            }
        }
    }
}

// Equilibration scaling S = diag(l^|alpha|): value DOFs are unscaled, derivative
// DOFs pick up the nodal length scale. See docs/hermite_bathymetry_system.md S11.
TEST_F(HermiteDofManagerTest, EquilibrationScalingTracksDerivativeOrder) {
    auto mesh = uniform(4);
    CGHermiteDofManager dm(mesh, 1);

    const VecX S = dm.equilibration_scaling();
    ASSERT_EQ(S.size(), dm.num_global_dofs());

    const Real h = 25.0; // uniform element size
    for (Index g = 0; g < dm.num_global_dofs(); ++g) {
        EXPECT_NEAR(S(g), std::pow(h, dm.total_deriv_order(g)), 1e-9)
            << "dof " << g << " order " << dm.total_deriv_order(g);
    }

    // All four derivative classes are present
    std::set<int> orders;
    for (Index g = 0; g < dm.num_global_dofs(); ++g) {
        orders.insert(dm.total_deriv_order(g));
    }
    EXPECT_EQ(orders, (std::set<int>{0, 1, 2}));
}

// Zero normal gradient pins z_n and z_nt at boundary nodes -- eliminations, not
// constraint rows. See docs/hermite_bathymetry_system.md S9.
TEST_F(HermiteDofManagerTest, ZeroGradientBCPinsNormalDerivativeDofs) {
    auto mesh = uniform(4);

    CGHermiteDofManager without(mesh, 1, false);
    CGHermiteDofManager with(mesh, 1, true);

    EXPECT_EQ(without.num_free_dofs(), without.num_global_dofs());
    EXPECT_LT(with.num_free_dofs(), with.num_global_dofs());

    // Pinned DOFs have no masters: they are substitutions to zero
    for (const auto &c : with.constraints()) {
        EXPECT_TRUE(c.master_dofs.empty());
    }

    // At r = 0 there are no derivative DOFs, so the condition is inexpressible
    CGHermiteDofManager r0(mesh, 0, true);
    EXPECT_EQ(r0.num_constraints(), 0);
}

} // namespace
