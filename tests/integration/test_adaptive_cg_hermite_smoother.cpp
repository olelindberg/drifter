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
#include "mesh/coastline_refinement.hpp"
#include "mesh/octree_adapter.hpp"
#include <memory>
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <iterator>
#include <limits>
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

/// Steep on the left half of the domain, gentler but still unresolved on the right
///
/// Paired with LEFT_HALF_COARSE_RESOLUTION below, this puts the largest errors
/// on exactly the elements that hit their data-resolution floor first. The
/// resolution limit is a per-element floor, so the right half must keep
/// refining regardless.
Real left_heavy_bathy(Real x, Real y) {
    if (x < 50.0) {
        return 50.0 + 20.0 * std::sin(x * M_PI / 12.5) * std::cos(y * M_PI / 12.5);
    }
    return 50.0 + 4.0 * std::sin(x * M_PI / 12.5) * std::cos(y * M_PI / 12.5);
}

/// Coarse data on the left half, fine on the right
///
/// The initial 4x4 mesh has 25 m elements, so the left half may split exactly
/// once (to 12.5 m) before its children would fall below the pixel size.
Real left_half_coarse_resolution(Real x, Real /*y*/) {
    return x < 50.0 ? 12.5 : 0.5;
}

/// Write a sawtooth "coastline" across the middle of the [0,100]^2 domain
///
/// Vertices alternate y = 45 / 55 every 5 in x, which puts the circumradius at
/// every interior vertex at exactly 6.25: with legs a = b = sqrt(125) and chord
/// c = 10, R = abc / (2|v1 x v2|) = 1250 / 200. Every vertex sharing one radius
/// makes the sawtooth an exact instrument: the pre-pass drives coastal elements
/// down to 6.25 and, because the comparison is strict, no further.
///
/// GeoJSON rather than a shapefile because GDAL reads it from a single text
/// file; with an empty target SRS no transform is applied, so these are plain
/// cartesian coordinates.
constexpr Real SAWTOOTH_CIRCUMRADIUS = 6.25;

std::string write_sawtooth_geojson(const std::string &path) {
    std::ofstream ofs(path);
    ofs << R"({"type":"FeatureCollection","features":[{"type":"Feature",)"
        << R"("properties":{},"geometry":{"type":"LineString","coordinates":[)";
    for (int i = 0; i <= 20; ++i) {
        if (i > 0) {
            ofs << ",";
        }
        ofs << "[" << (i * 5) << "," << ((i % 2 == 0) ? 45 : 55) << "]";
    }
    ofs << "]}}]}";
    return path;
}

/// Build a coastline index over the sawtooth, or nullptr if GDAL cannot read it
std::shared_ptr<const CoastlineIndex> make_sawtooth_index() {
    const std::string path = "/tmp/adaptive_cg_hermite_coastline.geojson";
    write_sawtooth_geojson(path);

    CoastlineReader reader;
    if (!reader.load(path)) {
        return nullptr;
    }
    return reader.build_index();
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
// Data resolution limits
// =============================================================================

TEST_F(AdaptiveCGHermiteSmootherTest, StopsAtMinimumElementSize) {
    auto config = make_config(1);
    config.error_threshold = 1e-6; // unreachable, so only the size limit can stop it
    config.max_iterations = 10;
    config.max_refinement_level = 20;
    config.enforce_pixel_limit = true;
    config.min_element_size = 12.5; // initial elements are 25, so exactly one level fits
    config.min_data_points_per_element = 0;
    config.smoother_config.lambda = 100.0;

    AdaptiveCGHermiteSmoother smoother(0.0, 100.0, 0.0, 100.0, 4, 4, config);
    smoother.set_bathymetry_data(high_frequency_bathy);

    const auto result = smoother.solve_adaptive();

    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.convergence_reason, ConvergenceReason::PixelResolution);

    for (Index e = 0; e < smoother.mesh().num_elements(); ++e) {
        const auto &b = smoother.mesh().element_bounds(e);
        EXPECT_GE(std::min(b.xmax - b.xmin, b.ymax - b.ymin), config.min_element_size - TOLERANCE);
    }
}

TEST_F(AdaptiveCGHermiteSmootherTest, StopsAtMinimumDataPointsPerElement) {
    auto config = make_config(1);
    config.error_threshold = 1e-6;
    config.max_iterations = 10;
    config.max_refinement_level = 20;
    config.enforce_pixel_limit = false; // isolate the data-point criterion
    config.min_element_size = 0.0;
    config.min_data_points_per_element = 100; // 10x10 pixels of 1.25 m => 12.5 m elements
    config.smoother_config.lambda = 100.0;

    AdaptiveCGHermiteSmoother smoother(0.0, 100.0, 0.0, 100.0, 4, 4, config);
    smoother.set_bathymetry_data(high_frequency_bathy);
    smoother.set_resolution_func([](Real, Real) { return 1.25; });

    const auto result = smoother.solve_adaptive();

    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.convergence_reason, ConvergenceReason::PixelResolution);

    for (Index e = 0; e < smoother.mesh().num_elements(); ++e) {
        const auto &b = smoother.mesh().element_bounds(e);
        const Real dx = b.xmax - b.xmin;
        const Real dy = b.ymax - b.ymin;
        EXPECT_GE((dx / 1.25) * (dy / 1.25),
                  static_cast<Real>(config.min_data_points_per_element) - TOLERANCE);
    }
}

/// The resolution limit is a per-element floor, not a global stop
///
/// The left half of the domain reaches its floor after one split while still
/// holding the largest errors in the mesh. Marking on error alone would hand
/// the whole Dorfler budget to those elements, leave nothing refinable, and
/// stop the adaptation with PixelResolution while the right half was still
/// coarse. Marking only among refinable elements keeps the right half going.
TEST_F(AdaptiveCGHermiteSmootherTest, RefinesCoarseRegionsWhileOthersAreAtTheResolutionLimit) {
    auto config = make_config(1);
    config.error_threshold = 1e-6; // unreachable, so a limit has to stop the run
    config.max_iterations = 6;
    config.max_elements = 100000;
    config.max_refinement_level = 20;
    config.enforce_pixel_limit = true;
    config.min_element_size = 0.0;            // auto, from the resolution function
    config.min_data_points_per_element = 0;   // isolate the pixel-size criterion
    config.smoother_config.lambda = 100.0;

    AdaptiveCGHermiteSmoother smoother(0.0, 100.0, 0.0, 100.0, 4, 4, config);
    smoother.set_bathymetry_data(left_heavy_bathy);
    smoother.set_resolution_func(left_half_coarse_resolution);

    const auto result = smoother.solve_adaptive();

    // The left half being exhausted is not a reason to stop the whole run
    EXPECT_NE(result.convergence_reason, ConvergenceReason::PixelResolution);

    // Refinement kept going well past the iteration that exhausted the left half
    ASSERT_GE(smoother.history().size(), 4u);
    for (size_t i = 0; i + 1 < smoother.history().size(); ++i) {
        EXPECT_GT(smoother.history()[i].elements_refined, 0)
            << "adaptation stalled at iteration " << i;
    }

    // Blocked elements hold error the refinable ones no longer do
    EXPECT_LT(result.max_refinable_error, result.max_error);

    // The right half, which has room, is far finer than the left, which does
    // not. The floor governs *marking* only: it is not a hard bound on element
    // size here, because 2:1 balancing across the resolution jump at x = 50
    // still pulls left-half elements below 12.5. A valid quadtree has to win
    // that argument; StopsAtMinimumElementSize pins the bound for the uniform
    // resolution case, where no such cascade exists.
    Index num_left = 0, num_right = 0;
    Real min_right = std::numeric_limits<Real>::max();
    for (Index e = 0; e < smoother.mesh().num_elements(); ++e) {
        const auto &b = smoother.mesh().element_bounds(e);
        const Real side = std::min(b.xmax - b.xmin, b.ymax - b.ymin);
        if (0.5 * (b.xmin + b.xmax) < 50.0) {
            ++num_left;
        } else {
            ++num_right;
            min_right = std::min(min_right, side);
        }
    }
    EXPECT_LT(min_right, 12.5) << "the unconstrained half never refined";
    EXPECT_GT(num_right, num_left)
        << "refinement did not concentrate where the data supports it";
}

// =============================================================================
// Coastline pre-pass
// =============================================================================

TEST_F(AdaptiveCGHermiteSmootherTest, CoastlinePrePassIsNoOpWhenUnset) {
    AdaptiveCGHermiteSmoother smoother(0.0, 100.0, 0.0, 100.0, 4, 4, make_config(1));

    EXPECT_EQ(smoother.refine_coastline(), 0);
    EXPECT_EQ(smoother.mesh().num_elements(), 16);
}

TEST_F(AdaptiveCGHermiteSmootherTest, CoastlinePrePassIsNoOpOnEmptyIndex) {
    AdaptiveCGHermiteSmoother smoother(0.0, 100.0, 0.0, 100.0, 4, 4, make_config(1));
    smoother.set_coastline(std::make_shared<CoastlineIndex>(), 20);

    EXPECT_EQ(smoother.refine_coastline(), 0);
    EXPECT_EQ(smoother.mesh().num_elements(), 16);
}

TEST_F(AdaptiveCGHermiteSmootherTest, CoastlinePrePassRefinesTowardTheCoastline) {
    auto index = make_sawtooth_index();
    ASSERT_NE(index, nullptr) << "GDAL could not read the test GeoJSON";
    ASSERT_GT(index->num_circumradius_points(), 0u);

    AdaptiveCGHermiteSmoother smoother(0.0, 100.0, 0.0, 100.0, 4, 4, make_config(1));
    smoother.set_coastline(index, 20);

    const int sweeps = smoother.refine_coastline();
    EXPECT_GT(sweeps, 0);
    EXPECT_GT(smoother.mesh().num_elements(), 16);

    // Coastal elements shrink to the circumradius; the pre-pass refines while the
    // radius is *strictly* smaller than the element side, so it stops at 6.25 and
    // must not go below it.
    Real min_side = std::numeric_limits<Real>::max();
    int level_at_coast = 0;
    for (Index e = 0; e < smoother.mesh().num_elements(); ++e) {
        const auto &b = smoother.mesh().element_bounds(e);
        min_side = std::min(min_side, std::min(b.xmax - b.xmin, b.ymax - b.ymin));

        const Real cy = 0.5 * (b.ymin + b.ymax);
        if (cy > 45.0 && cy < 55.0) {
            level_at_coast = std::max(level_at_coast, smoother.mesh().element_level(e).max_level());
        }
    }

    // The corner is well clear of the y in [45, 55] band, so 2:1 balancing does
    // not reach it and it stays at the level the base mesh gave it.
    const Index far = smoother.mesh().find_element(Vec2(12.5, 12.5));
    ASSERT_GE(far, 0);
    const int level_far = smoother.mesh().element_level(far).max_level();

    EXPECT_NEAR(min_side, SAWTOOTH_CIRCUMRADIUS, TOLERANCE);
    EXPECT_GT(level_at_coast, level_far) << "refinement did not concentrate on the coastline";
}

TEST_F(AdaptiveCGHermiteSmootherTest, CoastlinePrePassLeavesAnAlreadyFineMeshAlone) {
    auto index = make_sawtooth_index();
    ASSERT_NE(index, nullptr);

    // 32x32 over [0,100]^2 gives 3.125-wide elements, already below the sawtooth's
    // 6.25, so no element holds a feature tighter than itself and nothing is
    // marked. This is what pins the threshold to the element side rather than to
    // any configured length.
    AdaptiveCGHermiteSmoother smoother(0.0, 100.0, 0.0, 100.0, 32, 32, make_config(1));
    smoother.set_coastline(index, 20);

    EXPECT_EQ(smoother.refine_coastline(), 0);
    EXPECT_EQ(smoother.mesh().num_elements(), 1024);
}

TEST_F(AdaptiveCGHermiteSmootherTest, CoastlinePrePassRespectsMaxLevel) {
    auto index = make_sawtooth_index();
    ASSERT_NE(index, nullptr);

    // The 4x4 base mesh is at level 2, so this permits exactly one sweep before
    // the cap holds every coastal element back.
    AdaptiveCGHermiteSmoother smoother(0.0, 100.0, 0.0, 100.0, 4, 4, make_config(1));
    smoother.set_coastline(index, 3);

    EXPECT_GT(smoother.refine_coastline(), 0);

    for (Index e = 0; e < smoother.mesh().num_elements(); ++e) {
        EXPECT_LE(smoother.mesh().element_level(e).max_level(), 3);
    }
}

TEST_F(AdaptiveCGHermiteSmootherTest, CoastlinePrePassRespectsMinElementSize) {
    auto index = make_sawtooth_index();
    ASSERT_NE(index, nullptr);

    auto config = make_config(1);
    config.enforce_pixel_limit = true;
    config.min_element_size = 12.5; // base elements are 25, so one sweep fits

    AdaptiveCGHermiteSmoother smoother(0.0, 100.0, 0.0, 100.0, 4, 4, config);
    smoother.set_coastline(index, 20);

    EXPECT_GT(smoother.refine_coastline(), 0);

    for (Index e = 0; e < smoother.mesh().num_elements(); ++e) {
        const auto &b = smoother.mesh().element_bounds(e);
        EXPECT_GE(std::min(b.xmax - b.xmin, b.ymax - b.ymin), config.min_element_size - TOLERANCE);
    }
}

TEST_F(AdaptiveCGHermiteSmootherTest, CoastlinePrePassRespectsMaxElements) {
    auto index = make_sawtooth_index();
    ASSERT_NE(index, nullptr);

    auto uncapped = make_config(1);
    uncapped.max_elements = 100000;
    AdaptiveCGHermiteSmoother full(0.0, 100.0, 0.0, 100.0, 4, 4, uncapped);
    full.set_coastline(index, 20);
    full.refine_coastline();

    auto capped_config = make_config(1);
    capped_config.max_elements = 20; // the base mesh already has 16
    AdaptiveCGHermiteSmoother capped(0.0, 100.0, 0.0, 100.0, 4, 4, capped_config);
    capped.set_coastline(index, 20);
    capped.refine_coastline();

    EXPECT_LT(capped.mesh().num_elements(), full.mesh().num_elements());
}

TEST_F(AdaptiveCGHermiteSmootherTest, SolveAdaptiveRunsThePrePassOnce) {
    auto index = make_sawtooth_index();
    ASSERT_NE(index, nullptr);

    auto config = make_config(1);
    config.error_threshold = 1e9; // converge immediately, isolating the pre-pass
    config.max_iterations = 1;
    config.smoother_config.lambda = 100.0;

    AdaptiveCGHermiteSmoother smoother(0.0, 100.0, 0.0, 100.0, 4, 4, config);
    smoother.set_coastline(index, 20);
    smoother.set_bathymetry_data(gaussian_bump);

    const auto result = smoother.solve_adaptive();
    EXPECT_GT(result.num_elements, 16);
    EXPECT_TRUE(smoother.is_solved());

    // A second solve must not refine again: the coastline is already resolved,
    // and re-running the pre-pass would compound it every call.
    const Index after_first = smoother.mesh().num_elements();
    smoother.solve_adaptive();
    EXPECT_EQ(smoother.mesh().num_elements(), after_first);
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
