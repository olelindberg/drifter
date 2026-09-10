#include "bathymetry/cg_cubic_bezier_bathymetry_smoother.hpp"
#include "bathymetry/cg_hermite_bathymetry_smoother.hpp"
#include "bathymetry/cg_linear_bezier_bathymetry_smoother.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "core/enum_strings.hpp"
#include <Eigen/SparseCholesky>
#include <cmath>
#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

using namespace drifter;

namespace {

class CGHermiteSmootherTest : public ::testing::Test {
protected:
    QuadtreeAdapter create_quadtree(int nx, int ny, Real xmin = 0.0, Real xmax = 100.0,
                                    Real ymin = 0.0, Real ymax = 100.0) {
        QuadtreeAdapter mesh;
        mesh.build_uniform(xmin, xmax, ymin, ymax, nx, ny);
        return mesh;
    }

    /// A mesh with a 2:1 T-junction: refine the lower-left quadrant only
    QuadtreeAdapter create_nonconforming_quadtree(int nx = 4, int ny = 4) {
        QuadtreeAdapter mesh = create_quadtree(nx, ny);
        mesh.refine_where([&](Index e, const auto &) {
            const auto &b = mesh.element_bounds(e);
            return b.xmax <= 50.0 && b.ymax <= 50.0;
        });
        return mesh;
    }

    /// Run a check on both a conforming and a non-conforming mesh.
    /// QuadtreeAdapter is non-copyable, so the meshes are built in place.
    template <typename Body>
    void for_both_meshes(Body &&body) {
        {
            QuadtreeAdapter conforming = create_quadtree(4, 4);
            SCOPED_TRACE("conforming 4x4 mesh");
            body(conforming);
        }
        {
            QuadtreeAdapter nonconforming = create_nonconforming_quadtree();
            SCOPED_TRACE("non-conforming 2:1 mesh");
            body(nonconforming);
        }
    }

    /// One-sided trace of the surface and its normal derivative at an interface.
    ///
    /// Both sides are evaluated at *exactly* the same point (x, y), each within a
    /// nominated element. Probing at x +/- eps instead would measure the
    /// surface's own variation across the gap (curvature times gap for the
    /// derivative), which is a property of the function rather than of the
    /// discretisation, and would swamp the discontinuity being tested.
    struct SidedValue {
        Real value;
        Real normal_derivative;
    };

    /// @param probe_offset Signed offset used only to *identify* the element on
    ///        one side; the evaluation itself happens at (x, y).
    static SidedValue trace_from_side(const CGHermiteBathymetrySmoother &s,
                                      const QuadtreeAdapter &mesh, Real x, Real y,
                                      int normal_dir, Real probe_offset) {
        const Vec2 probe(x + (normal_dir == 0 ? probe_offset : 0.0),
                         y + (normal_dir == 1 ? probe_offset : 0.0));
        const Index elem = mesh.find_element(probe);
        EXPECT_GE(elem, 0) << "no element at probe (" << probe(0) << ", " << probe(1) << ")";

        return {s.evaluate_in_element(elem, x, y),
                s.evaluate_gradient_in_element(elem, x, y)(normal_dir)};
    }
};

/// A smooth synthetic seabed
Real smooth_bathy(Real x, Real y) {
    return -20.0 - 5.0 * std::sin(x * M_PI / 100.0) * std::cos(y * M_PI / 100.0);
}

// =============================================================================
// Construction
// =============================================================================

TEST_F(CGHermiteSmootherTest, DofCountsMatchTheory) {
    // On a conforming N x N mesh: 4(N+1)^2 DOFs at r=1, (N+1)^2 at r=0
    for (int n : {2, 4, 8}) {
        auto mesh = create_quadtree(n, n);

        CGHermiteSmootherConfig c1;
        c1.continuity_order = 1;
        CGHermiteBathymetrySmoother s1(mesh, c1);
        EXPECT_EQ(s1.num_global_dofs(), 4 * (n + 1) * (n + 1)) << "n=" << n;
        // Conforming mesh: no hanging nodes, so nothing is constrained
        EXPECT_EQ(s1.num_constraints(), 0) << "n=" << n;
        EXPECT_EQ(s1.num_free_dofs(), s1.num_global_dofs()) << "n=" << n;

        CGHermiteSmootherConfig c0;
        c0.continuity_order = 0;
        CGHermiteBathymetrySmoother s0(mesh, c0);
        EXPECT_EQ(s0.num_global_dofs(), (n + 1) * (n + 1)) << "n=" << n;
    }
}

TEST_F(CGHermiteSmootherTest, RejectsUnsupportedContinuityOrder) {
    auto mesh = create_quadtree(2, 2);
    CGHermiteSmootherConfig config;
    config.continuity_order = 2;
    EXPECT_THROW(CGHermiteBathymetrySmoother(mesh, config), std::invalid_argument);
}

TEST_F(CGHermiteSmootherTest, EvaluateBeforeSolveThrows) {
    auto mesh = create_quadtree(2, 2);
    CGHermiteBathymetrySmoother smoother(mesh);
    EXPECT_THROW(smoother.evaluate(50.0, 50.0), std::runtime_error);
}

// Asking for a backend that was not compiled in is an error, not a quiet
// substitution of another one: the config named a specific factorisation and
// silently running a different one would make any timing meaningless.
TEST_F(CGHermiteSmootherTest, UnavailableSolverKindThrows) {
    // Whichever of these this build lacks; if it has them all there is nothing
    // to assert and the test is trivially satisfied.
    std::vector<HermiteSolverKind> unavailable;
#ifndef DRIFTER_USE_MKL
    unavailable.push_back(HermiteSolverKind::PardisoLDLT);
#endif
#ifndef DRIFTER_USE_UMFPACK
    unavailable.push_back(HermiteSolverKind::UmfPackLU);
#endif
#ifndef DRIFTER_USE_CHOLMOD
    unavailable.push_back(HermiteSolverKind::CholmodSupernodalLLT);
#endif

    auto mesh = create_quadtree(2, 2);
    for (const auto kind : unavailable) {
        CGHermiteSmootherConfig config;
        config.solver = kind;
        CGHermiteBathymetrySmoother smoother(mesh, config);
        smoother.set_bathymetry_data(smooth_bathy);
        EXPECT_THROW(smoother.solve(), std::invalid_argument);
    }
}

// The default must stay the one backend that needs no optional dependency,
// so that a build configured with nothing extra still solves.
TEST_F(CGHermiteSmootherTest, DefaultSolverNeedsNoOptionalDependency) {
    EXPECT_EQ(CGHermiteSmootherConfig{}.solver, HermiteSolverKind::SimplicialLDLT);
}

// Every backend this build has must reach the same surface. The benchmark times
// them; this pins that they agree, which is what makes the timings comparable.
TEST_F(CGHermiteSmootherTest, AvailableSolverKindsAgree) {
    std::vector<HermiteSolverKind> kinds{HermiteSolverKind::SimplicialLLT};
#ifdef DRIFTER_USE_METIS
    kinds.push_back(HermiteSolverKind::SimplicialLDLTMetis);
#endif
#ifdef DRIFTER_USE_MKL
    kinds.push_back(HermiteSolverKind::PardisoLDLT);
    kinds.push_back(HermiteSolverKind::PardisoLLT);
#endif
#ifdef DRIFTER_USE_UMFPACK
    kinds.push_back(HermiteSolverKind::UmfPackLU);
#endif
#ifdef DRIFTER_USE_CHOLMOD
    kinds.push_back(HermiteSolverKind::CholmodSimplicialLDLT);
    kinds.push_back(HermiteSolverKind::CholmodSupernodalLLT);
    kinds.push_back(HermiteSolverKind::CholmodSupernodalNesdis);
#endif

    for_both_meshes([&](QuadtreeAdapter &mesh) {
        auto solve_with = [&](HermiteSolverKind kind) {
            CGHermiteSmootherConfig config;
            config.continuity_order = 1;
            config.lambda = 10.0;
            config.solver = kind;
            auto s = std::make_unique<CGHermiteBathymetrySmoother>(mesh, config);
            s->set_bathymetry_data(smooth_bathy);
            s->solve();
            return s;
        };

        const auto reference = solve_with(HermiteSolverKind::SimplicialLDLT);
        for (const auto kind : kinds) {
            const auto other = solve_with(kind);
            Real max_diff = 0.0;
            for (int i = 0; i <= 15; ++i) {
                for (int j = 0; j <= 15; ++j) {
                    const Real x = 100.0 * i / 15.0;
                    const Real y = 100.0 * j / 15.0;
                    max_diff = std::max(
                        max_diff, std::abs(reference->evaluate(x, y) - other->evaluate(x, y)));
                }
            }
            EXPECT_LT(max_diff, 1e-7) << to_string(kind);
        }
    });
}

// =============================================================================
// The system is SPD -- the central structural claim
// =============================================================================

// The reduced operator Q_red = T' Q T is symmetric positive definite, so a
// Cholesky-type factorisation suffices. The Bezier cubic path solves an
// indefinite KKT saddle point instead.
// See docs/hermite_bathymetry_system.md S10.
TEST_F(CGHermiteSmootherTest, CondensedSystemIsSymmetricPositiveDefinite) {
    for_both_meshes([&](QuadtreeAdapter &mesh) {
        for (int r : {0, 1}) {
            CGHermiteSmootherConfig config;
            config.continuity_order = r;
            CGHermiteBathymetrySmoother smoother(mesh, config);
            smoother.set_bathymetry_data(smooth_bathy);

            SpMat Q_red = smoother.condensed_matrix();
            ASSERT_GT(Q_red.rows(), 0);

            MatX dense = MatX(Q_red);
            EXPECT_LT((dense - dense.transpose()).norm(), 1e-9 * dense.norm()) << "r=" << r;

            // LDLT succeeds and D is strictly positive
            Eigen::SimplicialLDLT<SpMat> ldlt;
            ldlt.compute(Q_red);
            ASSERT_EQ(ldlt.info(), Eigen::Success) << "r=" << r;
            EXPECT_GT(ldlt.vectorD().minCoeff(), 0.0) << "r=" << r;
        }
    });
}

// =============================================================================
// The headline claim: C1 is exact, not collocated
// =============================================================================

// The normal derivative agrees from both sides of every interior edge, to
// round-off. The cubic Bezier smoother imposes the same condition by collocation
// at edge_ngauss points and reconciles the resulting over-determined system with
// an epsilon regularisation, leaving a residual violation ~2.4e-8.
// See docs/hermite_bathymetry_system.md S3.
TEST_F(CGHermiteSmootherTest, NormalDerivativeIsContinuousAcrossInteriorEdges) {
    const int n = 4;
    auto mesh = create_quadtree(n, n);

    CGHermiteSmootherConfig config;
    config.continuity_order = 1;
    config.lambda = 100.0;
    CGHermiteBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data(smooth_bathy);
    smoother.solve();

    const Real h = 100.0 / n;
    const Real probe = h / 4.0;
    Real max_value_jump = 0.0;
    Real max_deriv_jump = 0.0;

    auto check = [&](Real x, Real y, int normal_dir) {
        const auto lo = trace_from_side(smoother, mesh, x, y, normal_dir, -probe);
        const auto hi = trace_from_side(smoother, mesh, x, y, normal_dir, +probe);
        max_value_jump = std::max(max_value_jump, std::abs(lo.value - hi.value));
        max_deriv_jump =
            std::max(max_deriv_jump, std::abs(lo.normal_derivative - hi.normal_derivative));
    };

    // Vertical interior edges: x = k*h, normal direction x
    for (int k = 1; k < n; ++k) {
        for (int t = 1; t < 8; ++t) {
            check(k * h, 100.0 * t / 8.0, 0);
        }
    }
    // Horizontal interior edges: y = k*h, normal direction y
    for (int k = 1; k < n; ++k) {
        for (int t = 1; t < 8; ++t) {
            check(100.0 * t / 8.0, k * h, 1);
        }
    }

    // Machine precision, not the ~1e-8 of a collocated constraint
    EXPECT_LT(max_value_jump, 1e-11) << "max value jump across interior edges";
    EXPECT_LT(max_deriv_jump, 1e-11) << "max normal-derivative jump across interior edges";
    EXPECT_EQ(smoother.constraint_violation(), 0.0);
}

// Same statement across a 2:1 T-junction, where the hanging-node substitution
// (not an extra constraint equation) carries the continuity.
// See docs/hermite_bathymetry_system.md S8.
TEST_F(CGHermiteSmootherTest, NormalDerivativeIsContinuousAcrossTJunction) {
    auto mesh = create_nonconforming_quadtree();

    CGHermiteSmootherConfig config;
    config.continuity_order = 1;
    config.lambda = 100.0;
    CGHermiteBathymetrySmoother smoother(mesh, config);
    ASSERT_GT(smoother.num_constraints(), 0) << "expected hanging nodes on this mesh";

    smoother.set_bathymetry_data(smooth_bathy);
    smoother.solve();

    // x = 50 is the interface between the refined quadrant (h = 12.5) and its
    // coarse neighbour (h = 25)
    Real max_value_jump = 0.0;
    Real max_deriv_jump = 0.0;
    for (int t = 1; t < 20; ++t) {
        const Real y = 50.0 * t / 20.0;
        const auto fine = trace_from_side(smoother, mesh, 50.0, y, 0, -12.5 / 4.0);
        const auto coarse = trace_from_side(smoother, mesh, 50.0, y, 0, +25.0 / 4.0);
        max_value_jump = std::max(max_value_jump, std::abs(fine.value - coarse.value));
        max_deriv_jump = std::max(max_deriv_jump,
                                  std::abs(fine.normal_derivative - coarse.normal_derivative));
    }

    EXPECT_LT(max_value_jump, 1e-10) << "max value jump across the T-junction";
    EXPECT_LT(max_deriv_jump, 1e-10) << "max normal-derivative jump across the T-junction";
    EXPECT_EQ(smoother.constraint_violation(), 0.0);
}

// Value continuity across a T-junction (C0 part of the same statement)
TEST_F(CGHermiteSmootherTest, ValueIsContinuousAcrossTJunction) {
    auto mesh = create_nonconforming_quadtree();

    for (int r : {0, 1}) {
        CGHermiteSmootherConfig config;
        config.continuity_order = r;
        config.lambda = 100.0;
        CGHermiteBathymetrySmoother smoother(mesh, config);
        smoother.set_bathymetry_data(smooth_bathy);
        smoother.solve();

        Real max_jump = 0.0;
        for (int t = 1; t < 20; ++t) {
            const Real y = 50.0 * t / 20.0;
            const auto fine = trace_from_side(smoother, mesh, 50.0, y, 0, -12.5 / 4.0);
            const auto coarse = trace_from_side(smoother, mesh, 50.0, y, 0, +25.0 / 4.0);
            max_jump = std::max(max_jump, std::abs(fine.value - coarse.value));
        }
        EXPECT_LT(max_jump, 1e-10) << "r=" << r;
    }
}

// =============================================================================
// Approximation quality
// =============================================================================

// With a strong data weight the C1 element reproduces a bicubic exactly: it lies
// in the element space and is globally C1, so nothing forces a compromise.
TEST_F(CGHermiteSmootherTest, ReproducesBicubicPolynomial) {
    auto mesh = create_quadtree(4, 4, 0.0, 4.0, 0.0, 4.0);

    // A bicubic in the tensor-product space Q_3
    auto bicubic = [](Real x, Real y) {
        return 1.0 + 0.5 * x - 0.3 * y + 0.2 * x * y - 0.1 * x * x + 0.05 * x * x * x +
               0.02 * y * y * y - 0.03 * x * x * y * y;
    };

    CGHermiteSmootherConfig config;
    config.continuity_order = 1;
    config.lambda = 1e8; // essentially pure least squares
    config.ridge_epsilon = 1e-12;
    CGHermiteBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data(bicubic);
    smoother.solve();

    Real max_err = 0.0;
    for (int i = 0; i <= 12; ++i) {
        for (int j = 0; j <= 12; ++j) {
            const Real x = 4.0 * i / 12.0;
            const Real y = 4.0 * j / 12.0;
            max_err = std::max(max_err, std::abs(smoother.evaluate(x, y) - bicubic(x, y)));
        }
    }
    EXPECT_LT(max_err, 1e-6) << "max bicubic reproduction error";
}

// The C0 element reproduces a bilinear exactly, for the same reason
TEST_F(CGHermiteSmootherTest, C0ReproducesBilinear) {
    auto mesh = create_quadtree(4, 4, 0.0, 4.0, 0.0, 4.0);
    auto bilinear = [](Real x, Real y) { return 2.0 + 0.7 * x - 0.4 * y + 0.15 * x * y; };

    CGHermiteSmootherConfig config;
    config.continuity_order = 0;
    config.lambda = 1e8;
    config.ridge_epsilon = 1e-12;
    CGHermiteBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data(bilinear);
    smoother.solve();

    Real max_err = 0.0;
    for (int i = 0; i <= 12; ++i) {
        for (int j = 0; j <= 12; ++j) {
            const Real x = 4.0 * i / 12.0;
            const Real y = 4.0 * j / 12.0;
            max_err = std::max(max_err, std::abs(smoother.evaluate(x, y) - bilinear(x, y)));
        }
    }
    EXPECT_LT(max_err, 1e-7) << "max bilinear reproduction error";
}

TEST_F(CGHermiteSmootherTest, FitsSmoothBathymetry) {
    auto mesh = create_quadtree(8, 8);

    CGHermiteSmootherConfig config;
    config.continuity_order = 1;
    config.lambda = 100.0;
    CGHermiteBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data(smooth_bathy);
    smoother.solve();

    Real max_err = 0.0;
    for (int i = 1; i < 20; ++i) {
        for (int j = 1; j < 20; ++j) {
            const Real x = 100.0 * i / 20.0;
            const Real y = 100.0 * j / 20.0;
            max_err = std::max(max_err, std::abs(smoother.evaluate(x, y) - smooth_bathy(x, y)));
        }
    }
    EXPECT_LT(max_err, 0.5) << "max fit error on a smooth seabed";
}

// =============================================================================
// The C0 validation twin
// =============================================================================

// The r = 0 Hermite element *is* the linear Bezier element: same basis, M = I,
// and G_0 = [1/2, 1/2] is exactly the de Casteljau midpoint weight. The two
// smoothers must therefore produce the same surface. Compare evaluated values
// rather than coefficient vectors, because the local DOF orderings are
// transposed (i + 2j here, j + 2i there).
TEST_F(CGHermiteSmootherTest, C0MatchesLinearBezierSmoother) {
    for_both_meshes([&](QuadtreeAdapter &mesh) {
        const Real lambda = 1.0;
        const Real ridge = 1e-4;

        CGHermiteSmootherConfig hc;
        hc.continuity_order = 0;
        hc.lambda = lambda;
        hc.ridge_epsilon = ridge;
        hc.ngauss_data = 2;
        CGHermiteBathymetrySmoother hermite(mesh, hc);
        hermite.set_bathymetry_data(smooth_bathy);
        hermite.solve();

        CGLinearBezierSmootherConfig lc;
        lc.lambda = lambda;
        lc.ridge_epsilon = ridge;
        lc.ngauss_data = 2;
        CGLinearBezierBathymetrySmoother linear(mesh, lc);
        linear.set_bathymetry_data(smooth_bathy);
        linear.solve();

        ASSERT_EQ(hermite.num_global_dofs(), linear.num_global_dofs());

        Real max_diff = 0.0;
        for (int i = 0; i <= 20; ++i) {
            for (int j = 0; j <= 20; ++j) {
                const Real x = 100.0 * i / 20.0;
                const Real y = 100.0 * j / 20.0;
                max_diff =
                    std::max(max_diff, std::abs(hermite.evaluate(x, y) - linear.evaluate(x, y)));
            }
        }
        EXPECT_LT(max_diff, 1e-9) << "C0 Hermite must reproduce the linear Bezier smoother";
    });
}

// =============================================================================
// Comparison against the cubic Bezier smoother's collocated C1
// =============================================================================

// Both smoothers fit the same data on the same mesh and are asked for the same
// thing. The Hermite one delivers it exactly; the Bezier one to ~1e-8.
TEST_F(CGHermiteSmootherTest, ExactC1BeatsCollocatedC1) {
    const int n = 4;
    auto mesh = create_quadtree(n, n);

    CGHermiteSmootherConfig hc;
    hc.continuity_order = 1;
    hc.lambda = 10.0;
    CGHermiteBathymetrySmoother hermite(mesh, hc);
    hermite.set_bathymetry_data(smooth_bathy);
    hermite.solve();

    CGCubicBezierSmootherConfig bc;
    bc.lambda = 10.0;
    CGCubicBezierBathymetrySmoother bezier(mesh, bc);
    bezier.set_bathymetry_data(smooth_bathy);
    bezier.solve();

    EXPECT_EQ(hermite.constraint_violation(), 0.0);
    EXPECT_GT(bezier.constraint_violation(), 0.0)
        << "the Bezier C1 constraints are collocated, so a residual is expected";

    // Measure the actual normal-derivative jump across every interior edge, the
    // same way for both smoothers: one-sided traces evaluated at the same point.
    const Real h = 100.0 / n;
    auto max_normal_jump = [&](const CGSmootherBase &s) {
        Real worst = 0.0;
        auto check = [&](Real x, Real y, int dir) {
            const Vec2 lo(x - (dir == 0 ? h / 4 : 0.0), y - (dir == 1 ? h / 4 : 0.0));
            const Vec2 hi(x + (dir == 0 ? h / 4 : 0.0), y + (dir == 1 ? h / 4 : 0.0));
            const Index ea = mesh.find_element(lo);
            const Index eb = mesh.find_element(hi);
            if (ea < 0 || eb < 0) {
                return;
            }
            worst = std::max(worst, std::abs(s.evaluate_gradient_in_element(ea, x, y)(dir) -
                                             s.evaluate_gradient_in_element(eb, x, y)(dir)));
        };
        for (int k = 1; k < n; ++k) {
            for (int t = 1; t < 8; ++t) {
                check(k * h, 100.0 * t / 8.0, 0);
                check(100.0 * t / 8.0, k * h, 1);
            }
        }
        return worst;
    };

    const Real hermite_jump = max_normal_jump(hermite);
    const Real bezier_jump = max_normal_jump(bezier);

    std::cout << "  max normal-derivative jump across interior edges:\n"
              << "    C1 Hermite (structural):  " << hermite_jump << "\n"
              << "    cubic Bezier (collocated): " << bezier_jump << std::endl;

    // Hermite is exact; Bezier is only as good as its collocation residual.
    EXPECT_LT(hermite_jump, 1e-11);
    EXPECT_GT(bezier_jump, 1e4 * hermite_jump)
        << "expected the collocated C1 to be orders of magnitude looser";

    // And the Hermite system is much smaller: 4(N+1)^2 vs (3N+1)^2 plus
    // 8N(N-1) constraint rows.
    EXPECT_LT(hermite.num_global_dofs(),
              bezier.num_global_dofs() + bezier.num_constraints());
}

// =============================================================================
// Equilibration
// =============================================================================

// Equilibration is a change of variables inside solve(), so it must not move the
// answer -- on a uniform mesh or an adaptive one.
TEST_F(CGHermiteSmootherTest, EquilibrationDoesNotChangeTheSolution) {
    for_both_meshes([&](QuadtreeAdapter &mesh) {
        auto solve_with = [&](bool equilibrate) {
            CGHermiteSmootherConfig config;
            config.continuity_order = 1;
            config.lambda = 10.0;
            config.use_equilibration = equilibrate;
            auto s = std::make_unique<CGHermiteBathymetrySmoother>(mesh, config);
            s->set_bathymetry_data(smooth_bathy);
            s->solve();
            return s;
        };

        auto with = solve_with(true);
        auto without = solve_with(false);

        Real max_diff = 0.0;
        for (int i = 0; i <= 15; ++i) {
            for (int j = 0; j <= 15; ++j) {
                const Real x = 100.0 * i / 15.0;
                const Real y = 100.0 * j / 15.0;
                max_diff = std::max(max_diff, std::abs(with->evaluate(x, y) - without->evaluate(x, y)));
            }
        }
        EXPECT_LT(max_diff, 1e-7);
    });
}

// =============================================================================
// Boundary conditions
// =============================================================================

// Zero normal gradient becomes an elimination of z_n and z_nt at boundary nodes:
// exact, and with no constraint rows. See docs/hermite_bathymetry_system.md S9.
TEST_F(CGHermiteSmootherTest, ZeroGradientBCIsEnforcedExactly) {
    auto mesh = create_quadtree(4, 4);

    CGHermiteSmootherConfig config;
    config.continuity_order = 1;
    config.lambda = 10.0;
    config.enable_zero_gradient_bc = true;
    CGHermiteBathymetrySmoother smoother(mesh, config);

    // Boundary normal-derivative DOFs are eliminated, so free < global
    EXPECT_LT(smoother.num_free_dofs(), smoother.num_global_dofs());

    smoother.set_bathymetry_data(smooth_bathy);
    smoother.solve();

    // At the boundary *nodes* the normal derivative is exactly zero
    const Real h = 100.0 / 4;
    for (int k = 0; k <= 4; ++k) {
        const Real t = k * h;
        EXPECT_NEAR(smoother.evaluate_gradient(0.0 + 1e-9, t)(0), 0.0, 1e-8) << "left, y=" << t;
        EXPECT_NEAR(smoother.evaluate_gradient(100.0 - 1e-9, t)(0), 0.0, 1e-8) << "right, y=" << t;
        EXPECT_NEAR(smoother.evaluate_gradient(t, 0.0 + 1e-9)(1), 0.0, 1e-8) << "bottom, x=" << t;
        EXPECT_NEAR(smoother.evaluate_gradient(t, 100.0 - 1e-9)(1), 0.0, 1e-8) << "top, x=" << t;
    }
}

// =============================================================================
// Bernstein round trip
// =============================================================================

// The Bernstein coefficients remain inspectable at every step through
// c_e = M_e q_e, which is what keeps bound enforcement expressible even though
// Hermite loses the convex-hull guarantee. See docs/hermite_bathymetry_system.md S14.
TEST_F(CGHermiteSmootherTest, BernsteinCoefficientsReproduceTheSurface) {
    auto mesh = create_quadtree(3, 3);

    CGHermiteSmootherConfig config;
    config.continuity_order = 1;
    config.lambda = 10.0;
    CGHermiteBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data(smooth_bathy);
    smoother.solve();

    CubicBezierBasis2D bezier_basis;

    for (Index e = 0; e < mesh.num_elements(); ++e) {
        const auto &b = mesh.element_bounds(e);
        const VecX c = smoother.element_bernstein_coefficients(e);
        ASSERT_EQ(c.size(), 16);

        for (Real u : {0.0, 0.4, 1.0}) {
            for (Real v : {0.0, 0.6, 1.0}) {
                const Real x = b.xmin + u * (b.xmax - b.xmin);
                const Real y = b.ymin + v * (b.ymax - b.ymin);
                EXPECT_NEAR(bezier_basis.evaluate_scalar(c, u, v),
                            smoother.evaluate_in_element(e, x, y), 1e-9);
            }
        }
    }
}

// =============================================================================
// VTK output
// =============================================================================

// The written file must carry the surface exactly, not a piecewise-linear
// sampling of it. Sampling a bicubic on flat quads creases along element
// boundaries and cracks at T-junctions, which reads as a loss of C1 that the
// solver never produced.
TEST_F(CGHermiteSmootherTest, WritesSurfaceAtItsExactDegree) {
    EXPECT_EQ(CGHermiteSmootherConfig{}.continuity_order, 1);

    for (int r : {0, 1}) {
        auto mesh = create_nonconforming_quadtree();

        CGHermiteSmootherConfig config;
        config.continuity_order = r;
        config.lambda = 100.0;
        CGHermiteBathymetrySmoother smoother(mesh, config);
        smoother.set_bathymetry_data(smooth_bathy);
        smoother.solve();

        const int degree = smoother.surface_degree();
        EXPECT_EQ(degree, 2 * r + 1) << "r=" << r;

        char tmpl[] = "/tmp/drifter_hermite_vtk_XXXXXX";
        const int fd = mkstemp(tmpl);
        ASSERT_GE(fd, 0) << "cannot create a temporary file";
        close(fd);
        std::remove(tmpl);
        const std::string base = tmpl;
        smoother.write_vtk(base);

        std::ifstream file(base + ".vtu");
        ASSERT_TRUE(file.good()) << "r=" << r << ": no output written";
        const std::string contents((std::istreambuf_iterator<char>(file)),
                                   std::istreambuf_iterator<char>());
        file.close();
        std::remove((base + ".vtu").c_str());

        // One VTK_LAGRANGE_QUAD per element, of the exact surface degree
        EXPECT_NE(contents.find("NumberOfCells=\"" + std::to_string(mesh.num_elements()) + "\""),
                  std::string::npos)
            << "r=" << r;
        EXPECT_NE(contents.find("\n70\n"), std::string::npos)
            << "r=" << r << ": expected VTK_LAGRANGE_QUADRILATERAL cells";
        const Index pts_per_cell = (degree + 1) * (degree + 1);
        EXPECT_NE(contents.find("NumberOfPoints=\"" +
                                std::to_string(mesh.num_elements() * pts_per_cell) + "\""),
                  std::string::npos)
            << "r=" << r;
    }
}

} // namespace
