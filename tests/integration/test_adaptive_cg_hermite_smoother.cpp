/// @file test_adaptive_cg_hermite_smoother.cpp
/// @brief Error-driven adaptive refinement around the CG Hermite smoother
///
/// Mirrors test_adaptive_cg_{cubic,linear}_bezier_smoother.cpp. The claim that
/// is specific to this family is ContinuityIsExactAfterRefinement: refinement
/// creates 2:1 T-junctions, and in the Hermite basis those are pure master/slave
/// substitutions, so value *and* normal derivative match to round-off rather
/// than to the ~1e-8 of collocated C1.

#include "bathymetry/adaptive_cg_hermite_smoother.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "mesh/octree_adapter.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <iterator>
#include <string>
#include <tuple>
#include <vector>

using namespace drifter;

namespace {

/// A Gaussian bump: smooth, localized, and so a genuine driver of refinement
Real gaussian_bump(Real x, Real y) {
    const Real cx = 50.0, cy = 50.0, sigma = 15.0;
    const Real dx = x - cx, dy = y - cy;
    return 50.0 + 30.0 * std::exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
}

/// Short-wavelength seabed: needs several levels before it is resolved
Real high_frequency_bathy(Real x, Real y) {
    return 50.0 + 10.0 * std::sin(x * M_PI / 12.5) * std::cos(y * M_PI / 12.5);
}

class AdaptiveCGHermiteSmootherTest : public ::testing::Test {
protected:
    static constexpr Real TOLERANCE = 1e-10;
    static constexpr Real LOOSE_TOLERANCE = 1e-4;

    /// Run the body for both implemented continuity orders
    template <typename Body>
    void for_both_orders(Body &&body) {
        for (int r : {0, 1}) {
            SCOPED_TRACE("continuity_order = " + std::to_string(r));
            body(r);
        }
    }

    static AdaptiveCGHermiteConfig make_config(int continuity_order) {
        AdaptiveCGHermiteConfig config;
        config.smoother_config.continuity_order = continuity_order;
        config.smoother_config.ngauss_data = (continuity_order == 0) ? 2 : 4;
        return config;
    }
};

// =============================================================================
// Construction
// =============================================================================

TEST_F(AdaptiveCGHermiteSmootherTest, ConstructFromDomainBounds) {
    for_both_orders([&](int r) {
        AdaptiveCGHermiteSmoother smoother(0.0, 100.0, 0.0, 100.0, 4, 4, make_config(r));

        EXPECT_EQ(smoother.mesh().num_elements(), 16);
        EXPECT_FALSE(smoother.is_solved());
    });
}

TEST_F(AdaptiveCGHermiteSmootherTest, ConstructFromOctree) {
    OctreeAdapter octree(0.0, 100.0, 0.0, 100.0, -1.0, 0.0);
    octree.build_uniform(4, 4, 1);

    AdaptiveCGHermiteSmoother smoother(octree, make_config(1));

    EXPECT_EQ(smoother.mesh().num_elements(), 16);
    EXPECT_FALSE(smoother.is_solved());
}

// =============================================================================
// Convergence
// =============================================================================

TEST_F(AdaptiveCGHermiteSmootherTest, ConvergesOnConstantBathymetry) {
    for_both_orders([&](int r) {
        auto config = make_config(r);
        config.error_threshold = 0.01;
        config.max_iterations = 5;
        // Near-pure least squares, as in CGHermiteSmootherTest.C0ReproducesBilinear:
        // at the default lambda the ridge biases a constant by O(ridge_epsilon)
        config.smoother_config.lambda = 1e8;
        config.smoother_config.ridge_epsilon = 1e-12;

        AdaptiveCGHermiteSmoother smoother(0.0, 100.0, 0.0, 100.0, 2, 2, config);
        smoother.set_bathymetry_data([](Real, Real) { return 50.0; });

        auto result = smoother.solve_adaptive();

        EXPECT_TRUE(smoother.is_solved());
        EXPECT_TRUE(result.converged);
        EXPECT_EQ(result.convergence_reason, ConvergenceReason::ErrorThreshold);
        EXPECT_LE(result.max_error, config.error_threshold);

        // A constant is in both Hermite spaces, so no refinement is needed
        EXPECT_EQ(smoother.mesh().num_elements(), 4);
        EXPECT_NEAR(smoother.evaluate(50.0, 50.0), 50.0, LOOSE_TOLERANCE);
    });
}

TEST_F(AdaptiveCGHermiteSmootherTest, RefinesOnHighFrequencyBathymetry) {
    for_both_orders([&](int r) {
        auto config = make_config(r);
        config.error_threshold = 0.05;
        config.max_iterations = 4;
        config.max_elements = 5000;
        config.smoother_config.lambda = 100.0;

        AdaptiveCGHermiteSmoother smoother(0.0, 100.0, 0.0, 100.0, 4, 4, config);
        smoother.set_bathymetry_data(high_frequency_bathy);

        smoother.solve_adaptive();

        EXPECT_TRUE(smoother.is_solved());
        EXPECT_GT(smoother.mesh().num_elements(), 16);

        // The error must come down as the mesh is refined
        const auto &history = smoother.history();
        ASSERT_GE(history.size(), 2u);
        EXPECT_LT(history.back().max_error, history.front().max_error);
    });
}

TEST_F(AdaptiveCGHermiteSmootherTest, RespectsMaxIterations) {
    auto config = make_config(1);
    config.error_threshold = 1e-8; // Unreachable, so iteration count is the limit
    config.max_iterations = 3;
    config.max_elements = 100000;
    config.max_refinement_level = 10;
    config.smoother_config.lambda = 100.0;

    AdaptiveCGHermiteSmoother smoother(0.0, 100.0, 0.0, 100.0, 2, 2, config);
    smoother.set_bathymetry_data(high_frequency_bathy);

    auto result = smoother.solve_adaptive();

    EXPECT_EQ(smoother.history().size(), 3u);
    EXPECT_FALSE(result.converged);
    EXPECT_EQ(result.convergence_reason, ConvergenceReason::MaxIterations);
}

TEST_F(AdaptiveCGHermiteSmootherTest, RespectsMaxElements) {
    auto config = make_config(1);
    config.error_threshold = 1e-8; // Unreachable, so the element cap must stop it
    config.max_iterations = 10;
    config.max_elements = 100;
    config.smoother_config.lambda = 100.0;

    AdaptiveCGHermiteSmoother smoother(0.0, 100.0, 0.0, 100.0, 2, 2, config);
    smoother.set_bathymetry_data(high_frequency_bathy);

    auto result = smoother.solve_adaptive();

    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.convergence_reason, ConvergenceReason::MaxElements);
    EXPECT_GE(smoother.mesh().num_elements(), config.max_elements);
}

TEST_F(AdaptiveCGHermiteSmootherTest, RespectsMaxRefinementLevel) {
    auto config = make_config(1);
    config.error_threshold = 1e-8; // Unreachable, so the level cap must stop it
    config.max_iterations = 10;
    config.max_elements = 100000;
    config.max_refinement_level = 2;
    config.dorfler_theta = 1.0; // Mark everything, so all elements hit the cap
    config.smoother_config.lambda = 100.0;

    AdaptiveCGHermiteSmoother smoother(0.0, 100.0, 0.0, 100.0, 2, 2, config);
    smoother.set_bathymetry_data(high_frequency_bathy);

    auto result = smoother.solve_adaptive();

    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.convergence_reason, ConvergenceReason::MaxRefinementLevel);

    for (Index e = 0; e < smoother.mesh().num_elements(); ++e) {
        EXPECT_LE(smoother.mesh().element_level(e).max_level(), config.max_refinement_level);
    }
}

// =============================================================================
// Error estimation
// =============================================================================

TEST_F(AdaptiveCGHermiteSmootherTest, ErrorEstimationAfterSolve) {
    auto config = make_config(1);
    config.error_threshold = 0.5;
    config.max_iterations = 2;
    config.smoother_config.lambda = 100.0;

    AdaptiveCGHermiteSmoother smoother(0.0, 100.0, 0.0, 100.0, 4, 4, config);
    smoother.set_bathymetry_data(gaussian_bump);
    smoother.solve_adaptive();

    const auto errors = smoother.estimate_errors();
    EXPECT_EQ(errors.size(), static_cast<size_t>(smoother.mesh().num_elements()));

    Real max_seen = 0.0;
    for (const auto &err : errors) {
        EXPECT_GE(err.element, 0);
        EXPECT_GE(err.l2_error, 0.0);
        EXPECT_GE(err.normalized_error, 0.0);
        max_seen = std::max(max_seen, err.normalized_error);
    }

    EXPECT_NEAR(smoother.max_error(), max_seen, TOLERANCE);
    EXPECT_GE(smoother.mean_error(), 0.0);
    EXPECT_LE(smoother.mean_error(), smoother.max_error() + TOLERANCE);
}

// =============================================================================
// Continuity across refinement-induced T-junctions
// =============================================================================

TEST_F(AdaptiveCGHermiteSmootherTest, ContinuityIsExactAfterRefinement) {
    for_both_orders([&](int r) {
        auto config = make_config(r);
        config.error_threshold = 0.05;
        config.max_iterations = 3;
        config.smoother_config.lambda = 100.0;

        AdaptiveCGHermiteSmoother smoother(0.0, 100.0, 0.0, 100.0, 4, 4, config);
        smoother.set_bathymetry_data(gaussian_bump);
        smoother.solve_adaptive();

        const QuadtreeAdapter &mesh = smoother.mesh();
        ASSERT_GT(mesh.num_elements(), 16) << "test needs an adaptively refined mesh";

        const CGHermiteBathymetrySmoother &inner = smoother.smoother();

        // Refinement is local, so a 2:1 mesh must have produced hanging nodes
        EXPECT_GT(inner.num_constraints(), 0);

        // Structural continuity plus exact back-substitution
        EXPECT_EQ(inner.constraint_violation(), 0.0);

        // Walk every element edge midpoint. Probing either side identifies the
        // two elements; both traces are then evaluated at the *same* point, so
        // what is measured is the jump in the discretisation rather than the
        // variation of the surface across a gap.
        int interfaces_checked = 0;
        Real max_value_jump = 0.0;
        Real max_derivative_jump = 0.0;

        for (Index e = 0; e < mesh.num_elements(); ++e) {
            const QuadBounds &b = mesh.element_bounds(e);
            const Real hx = b.xmax - b.xmin;
            const Real hy = b.ymax - b.ymin;
            const Real cx = 0.5 * (b.xmin + b.xmax);
            const Real cy = 0.5 * (b.ymin + b.ymax);

            // {x, y, normal direction, outward probe offset}
            const std::array<std::tuple<Real, Real, int, Real>, 4> edges = {
                std::make_tuple(b.xmin, cy, 0, -0.01 * hx),
                std::make_tuple(b.xmax, cy, 0, 0.01 * hx),
                std::make_tuple(cx, b.ymin, 1, -0.01 * hy),
                std::make_tuple(cx, b.ymax, 1, 0.01 * hy)};

            for (const auto &[x, y, dir, offset] : edges) {
                const Vec2 probe(x + (dir == 0 ? offset : 0.0), y + (dir == 1 ? offset : 0.0));
                const Index neighbor = mesh.find_element(probe);
                if (neighbor < 0 || neighbor == e) {
                    continue; // Domain boundary, or the same element
                }

                const Real v_here = inner.evaluate_in_element(e, x, y);
                const Real v_there = inner.evaluate_in_element(neighbor, x, y);
                const Real d_here = inner.evaluate_gradient_in_element(e, x, y)(dir);
                const Real d_there = inner.evaluate_gradient_in_element(neighbor, x, y)(dir);

                max_value_jump = std::max(max_value_jump, std::abs(v_here - v_there));
                max_derivative_jump = std::max(max_derivative_jump, std::abs(d_here - d_there));
                ++interfaces_checked;
            }
        }

        EXPECT_GT(interfaces_checked, 0);

        // C0 is structural at both orders; the normal derivative only at r = 1
        EXPECT_LT(max_value_jump, TOLERANCE) << "value jump across an interface";
        if (r == 1) {
            EXPECT_LT(max_derivative_jump, TOLERANCE) << "normal derivative jump";
        }

        std::cout << "  r=" << r << ": " << interfaces_checked << " interfaces, max |[z]|="
                  << max_value_jump << ", max |[dz/dn]|=" << max_derivative_jump << "\n";
    });
}

// =============================================================================
// History and profiling
// =============================================================================

TEST_F(AdaptiveCGHermiteSmootherTest, TracksAdaptationHistory) {
    auto config = make_config(1);
    config.error_threshold = 0.1;
    config.max_iterations = 4;
    config.smoother_config.lambda = 100.0;

    AdaptiveCGHermiteSmoother smoother(0.0, 100.0, 0.0, 100.0, 2, 2, config);
    smoother.set_bathymetry_data(gaussian_bump);
    smoother.solve_adaptive();

    const auto &history = smoother.history();
    ASSERT_FALSE(history.empty());

    for (size_t i = 0; i < history.size(); ++i) {
        EXPECT_EQ(history[i].iteration, static_cast<int>(i));
        EXPECT_GT(history[i].num_elements, 0);
        EXPECT_GE(history[i].max_error, 0.0);
        EXPECT_GE(history[i].mean_error, 0.0);
    }

    // The mesh grows monotonically: elements are refined, never coarsened
    for (size_t i = 1; i < history.size(); ++i) {
        EXPECT_GE(history[i].num_elements, history[i - 1].num_elements);
    }

    std::cout << "\nAdaptation history:\n";
    for (const auto &h : history) {
        std::cout << "  Iter " << h.iteration << ": " << h.num_elements << " elements, "
                  << "max_err=" << h.max_error << " m, refined=" << h.elements_refined << "\n";
    }
}

TEST_F(AdaptiveCGHermiteSmootherTest, VerboseModeFillsProfiles) {
    auto config = make_config(1);
    config.error_threshold = 0.1;
    config.max_iterations = 3;
    config.smoother_config.lambda = 100.0;
    config.verbose = true; // Profiles are only collected in verbose mode

    AdaptiveCGHermiteSmoother smoother(0.0, 100.0, 0.0, 100.0, 2, 2, config);
    smoother.set_bathymetry_data(gaussian_bump);
    smoother.solve_adaptive();

    const auto &profiles = smoother.profiles();
    ASSERT_FALSE(profiles.empty());
    EXPECT_EQ(profiles.size(), smoother.history().size());

    for (const auto &p : profiles) {
        EXPECT_GT(p.num_elements, 0);
        EXPECT_GT(p.num_dofs, 0);
        EXPECT_GT(p.total_ms(), 0.0);
    }
}

// =============================================================================
// VTK output
// =============================================================================

TEST_F(AdaptiveCGHermiteSmootherTest, WritesVTKOutput) {
    auto config = make_config(1);
    config.error_threshold = 0.1;
    config.max_iterations = 3;
    config.smoother_config.lambda = 100.0;

    AdaptiveCGHermiteSmoother smoother(0.0, 100.0, 0.0, 100.0, 4, 4, config);
    smoother.set_bathymetry_data(gaussian_bump);
    smoother.solve_adaptive();

    const std::string filename = "/tmp/adaptive_cg_hermite_test";
    smoother.write_vtk(filename);

    const std::string output_file = filename + ".vtu";
    ASSERT_TRUE(std::filesystem::exists(output_file));

    std::ifstream file(output_file);
    const std::string content((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());
    EXPECT_GT(content.size(), 1000u);
}

// =============================================================================
// Error handling
// =============================================================================

TEST_F(AdaptiveCGHermiteSmootherTest, ThrowsIfNoDataSet) {
    AdaptiveCGHermiteSmoother smoother(0.0, 100.0, 0.0, 100.0, 2, 2, make_config(1));

    EXPECT_THROW(smoother.solve_adaptive(), std::runtime_error);
}

TEST_F(AdaptiveCGHermiteSmootherTest, ThrowsIfEvaluateBeforeSolve) {
    AdaptiveCGHermiteSmoother smoother(0.0, 100.0, 0.0, 100.0, 2, 2, make_config(1));
    smoother.set_bathymetry_data([](Real, Real) { return 50.0; });

    EXPECT_THROW(smoother.evaluate(50.0, 50.0), std::runtime_error);
}

TEST_F(AdaptiveCGHermiteSmootherTest, ThrowsIfSmootherAccessedBeforeSolve) {
    AdaptiveCGHermiteSmoother smoother(0.0, 100.0, 0.0, 100.0, 2, 2, make_config(1));

    EXPECT_THROW(smoother.smoother(), std::runtime_error);
    EXPECT_THROW(smoother.write_vtk("/tmp/adaptive_cg_hermite_unsolved"), std::runtime_error);
}

} // namespace
