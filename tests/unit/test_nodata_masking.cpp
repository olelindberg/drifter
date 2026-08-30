/// @file test_nodata_masking.cpp
/// @brief NoData detection and the element-level water / beach / inland policy
///
/// Two defects combined to put spikes in the fitted surface over small gaps in
/// the source rasters:
///   1. NaN NoData - declared by the 2024 Klimadatastyrelsen tiles - passed every
///      nodata test in the codebase and fell through to depth 0.
///   2. Land and NoData points entered the least-squares term as depth-0
///      observations, which the smoothness term then overshot around.
///
/// The fit is now solved only over elements that carry real depth readings.
/// Everything else is "not water" and is classified per element by
/// ElementDataMask: the rim touching water is pinned to depth 0, giving the water
/// region its Dirichlet boundary, and the interior leaves the system entirely.

#include "bathymetry/biharmonic_assembler.hpp" // BathymetrySource, FunctionBathymetry
#include "bathymetry/cg_hermite_bathymetry_smoother.hpp"
#include "bathymetry/cg_hermite_dof_manager.hpp"
#include "bathymetry/element_data_mask.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "mesh/geotiff_reader.hpp"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

using namespace drifter;

namespace {

constexpr float NAN_F = std::numeric_limits<float>::quiet_NaN();

/// A raster of constant elevation with a rectangular NoData patch punched in it
BathymetryData make_raster(int size, Real xmin, Real xmax, float value, float nodata,
                           int hole_lo, int hole_hi) {
    BathymetryData data;
    data.sizex = size;
    data.sizey = size;
    data.nodata_value = nodata;
    data.is_depth_positive = false; // elevation format: negative is water

    const Real d = (xmax - xmin) / size;
    data.geotransform = {xmin, d, 0.0, xmax, 0.0, -d};
    data.xmin = xmin;
    data.xmax = xmax;
    data.ymin = xmin;
    data.ymax = xmax;

    data.elevation.assign(static_cast<size_t>(size * size), value);
    for (int py = hole_lo; py < hole_hi; ++py) {
        for (int px = hole_lo; px < hole_hi; ++px) {
            data.elevation[static_cast<size_t>(py * size + px)] = nodata;
        }
    }
    return data;
}

} // namespace

// =============================================================================
// Defect 1: NaN NoData
// =============================================================================

// std::abs(NaN - NaN) < tol and NaN > 1e30f are both false, so a NaN sample
// slipped through every check and get_depth()'s `val < 0 ? -val : 0` returned 0.
TEST(NoDataValueTest, DetectsNaNNoData) {
    EXPECT_TRUE(is_nodata_value(NAN_F, NAN_F));
    EXPECT_FALSE(is_nodata_value(-50.0f, NAN_F)) << "valid data in a NaN-NoData raster";
    EXPECT_FALSE(is_nodata_value(0.0f, NAN_F));
}

TEST(NoDataValueTest, DetectsSentinelNoData) {
    // The primary raster ddm_50m.dybde-emodnet.tif declares 3.4028235e+38
    EXPECT_TRUE(is_nodata_value(3.4028235e+38f, 3.4028235e+38f));
    EXPECT_TRUE(is_nodata_value(NAN_F, -9999.0f)) << "NaN is missing whatever is declared";
    EXPECT_TRUE(is_nodata_value(-9999.0f, -9999.0f));
    EXPECT_FALSE(is_nodata_value(-50.0f, -9999.0f));
}

// A NaN-NoData raster must not classify its *valid* samples as missing
TEST(NoDataValueTest, NaNNoDataDoesNotSwallowEverything) {
    auto data = make_raster(8, 0.0, 80.0, -100.0f, NAN_F, 3, 5);

    EXPECT_TRUE(data.has_data(5.0, 75.0)) << "corner pixel is valid";
    EXPECT_NEAR(data.get_depth(5.0, 75.0), 100.0, 1e-6);
    EXPECT_FALSE(data.is_land(5.0, 75.0));
}

TEST(NoDataValueTest, NaNHoleReportsAsMissingNotWater) {
    auto data = make_raster(8, 0.0, 80.0, -100.0f, NAN_F, 3, 5);

    // Pixel (4, 4) is inside the hole; its centre is at (45, 35)
    const double wx = 45.0, wy = 35.0;

    EXPECT_FALSE(data.has_data(wx, wy)) << "NaN gap must report as missing";
    EXPECT_TRUE(data.is_land(wx, wy)) << "a gap is not water; before the fix is_land "
                                         "reached `val >= 0`, which is false for NaN";
    EXPECT_TRUE(std::isnan(data.interpolate(wx, wy)) ||
                is_nodata_value(data.interpolate(wx, wy), data.nodata_value));
}

// =============================================================================
// Defect 2: pinned points must not be fitted as depth-0 observations
// =============================================================================

namespace {

/// Constant-depth water with a square NoData gap punched in it.
/// evaluate() returns 0 in the gap, exactly as the real readers do.
class HoleSource : public BathymetrySource {
public:
    HoleSource(Real depth, Real lo, Real hi) : depth_(depth), lo_(lo), hi_(hi) {}

    Real evaluate(Real x, Real y) const override { return in_square(x, y) ? 0.0 : depth_; }

    bool has_data(Real x, Real y) const override { return !in_square(x, y); }
    bool is_land_point(Real, Real) const override { return false; }

    bool in_square(Real x, Real y) const {
        return x >= lo_ && x <= hi_ && y >= lo_ && y <= hi_;
    }

private:
    Real depth_, lo_, hi_;
};

/// Constant-depth water with a square island of land in it
class IslandSource : public BathymetrySource {
public:
    IslandSource(Real depth, Real lo, Real hi) : depth_(depth), lo_(lo), hi_(hi) {}

    Real evaluate(Real x, Real y) const override { return in_square(x, y) ? 0.0 : depth_; }

    bool has_data(Real, Real) const override { return true; }
    bool is_land_point(Real x, Real y) const override { return in_square(x, y); }

    bool in_square(Real x, Real y) const {
        return x >= lo_ && x <= hi_ && y >= lo_ && y <= hi_;
    }

private:
    Real depth_, lo_, hi_;
};

CGHermiteSmootherConfig hermite_config() {
    CGHermiteSmootherConfig config;
    config.continuity_order = 1;
    config.lambda = 1.0e6;
    config.verbose = false;
    return config;
}

} // namespace

// A NoData element touching water is a Beach element, held at depth 0. The mesh
// is 8x8 over [0,80], so the gap [30,40]^2 is exactly one element and all eight
// of its neighbours carry data.
TEST(NoDataFitTest, GapIsPinnedToZero) {
    constexpr Real DEPTH = 100.0;
    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 80.0, 0.0, 80.0, 8, 8);

    HoleSource source(DEPTH, 30.0, 40.0);

    CGHermiteBathymetrySmoother smoother(mesh, hermite_config());
    smoother.set_bathymetry_data(source);
    smoother.solve();

    // Far from the gap the fit must still reproduce the data
    EXPECT_NEAR(smoother.evaluate(10.0, 10.0), DEPTH, 1.0);
    EXPECT_NEAR(smoother.evaluate(70.0, 70.0), DEPTH, 1.0);

    // Over the gap the surface is held at sea level
    EXPECT_NEAR(smoother.evaluate(35.0, 35.0), 0.0, 1e-8);

    const ElementDataMask* mask = smoother.element_mask();
    ASSERT_NE(mask, nullptr);
    EXPECT_EQ((*mask)[mesh.find_element(Vec2(35.0, 35.0))], ElementDataClass::Beach);
    EXPECT_EQ(mask->num_inland(), 0) << "a one-element hole has no interior";
}

// A gap too small to claim a whole element leaves that element in the solve, and
// the point-wise exclusion is what keeps its zeros out of the least-squares term.
// Whole-element gaps never reach this counter: they are skipped before quadrature.
TEST(NoDataFitTest, PartialGapQuadraturePointsAreExcluded) {
    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 80.0, 0.0, 80.0, 8, 8);

    // A quarter of one 10 m element, so the element still classifies as water
    HoleSource source(100.0, 30.0, 35.0);

    CGHermiteBathymetrySmoother smoother(mesh, hermite_config());
    smoother.set_bathymetry_data(source);

    ASSERT_NE(smoother.element_mask(), nullptr);
    EXPECT_FALSE(smoother.element_mask()->has_pinned_elements())
        << "a sub-element gap must not pin the element that contains it";
    EXPECT_GT(smoother.num_excluded_quadrature_points(), 0);
}

// Pinning the rim from ~100 m to 0 across one element is the transition that a
// bicubic Hermite, having no convex-hull property, can overshoot. It is the same
// transition the coastline already makes, but it needs a bound on it.
TEST(NoDataFitTest, RimOvershootIsBounded) {
    constexpr Real DEPTH = 100.0;
    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 80.0, 0.0, 80.0, 8, 8);

    HoleSource source(DEPTH, 30.0, 40.0);

    CGHermiteBathymetrySmoother smoother(mesh, hermite_config());
    smoother.set_bathymetry_data(source);
    smoother.solve();

    Real min_depth = DEPTH;
    Real max_depth = 0.0;
    for (int j = 0; j <= 80; ++j) {
        for (int i = 0; i <= 80; ++i) {
            const Real d = smoother.evaluate(i, j);
            min_depth = std::min(min_depth, d);
            max_depth = std::max(max_depth, d);
        }
    }
    EXPECT_GT(min_depth, -0.2 * DEPTH) << "fit rings above sea level around the pinned gap";
    EXPECT_LT(max_depth, 1.5 * DEPTH) << "fit spikes downward";
}

// A source with neither gaps nor land must behave exactly as before
TEST(NoDataFitTest, NothingExcludedForAPlainFunction) {
    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 80.0, 0.0, 80.0, 4, 4);

    FunctionBathymetry source([](Real, Real) { return 100.0; });

    CGHermiteBathymetrySmoother smoother(mesh, hermite_config());
    smoother.set_bathymetry_data(source);
    smoother.solve();

    EXPECT_EQ(smoother.num_excluded_quadrature_points(), 0);
    EXPECT_NEAR(smoother.evaluate(40.0, 40.0), 100.0, 1e-2);
}

// =============================================================================
// Land: a known zero, imposed strongly
// =============================================================================

TEST(LandDirichletTest, IslandIsHeldAtZero) {
    constexpr Real DEPTH = 100.0;
    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 80.0, 0.0, 80.0, 8, 8);

    // Two elements across, so the island has an interior node at (40, 40)
    IslandSource source(DEPTH, 30.0, 50.0);

    CGHermiteBathymetrySmoother smoother(mesh, hermite_config());
    smoother.set_bathymetry_data(source);
    smoother.solve();

    // The island interior is pinned, so the surface sits at sea level there
    EXPECT_NEAR(smoother.evaluate(40.0, 40.0), 0.0, 1e-8);

    // ... while the surrounding water keeps its depth
    EXPECT_NEAR(smoother.evaluate(10.0, 10.0), DEPTH, 5.0);
}

// The pins are ordinary master/slave rows, so structural continuity must survive
TEST(LandDirichletTest, ContinuityIsUnaffectedByPinning) {
    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 80.0, 0.0, 80.0, 8, 8);

    IslandSource source(100.0, 30.0, 50.0);

    CGHermiteBathymetrySmoother smoother(mesh, hermite_config());
    smoother.set_bathymetry_data(source);
    smoother.solve();

    EXPECT_NEAR(smoother.constraint_violation(), 0.0, 1e-12);
}

// The DOF manager pins value DOFs at the nodes of non-water elements, reducing the
// free set without changing the global DOF count
TEST(LandDirichletTest, DofManagerPinsLandNodes) {
    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 80.0, 0.0, 80.0, 8, 8);

    auto land = [](Real x, Real y) { return x >= 30.0 && x <= 50.0 && y >= 30.0 && y <= 50.0; };
    const ElementDataMask mask(mesh, {}, land);

    CGHermiteDofManager pinned(mesh, 1, false, &mask);
    CGHermiteDofManager unpinned(mesh, 1, false);

    EXPECT_GT(pinned.num_pinned_dofs(), 0);
    EXPECT_EQ(unpinned.num_pinned_dofs(), 0);

    EXPECT_EQ(pinned.num_global_dofs(), unpinned.num_global_dofs());
    EXPECT_LT(pinned.num_free_dofs(), unpinned.num_free_dofs());
}

// =============================================================================
// Element classification
// =============================================================================

namespace {

/// Mesh of `n` x `n` elements over [0, n*h]^2, so element (i, j) is [i*h,(i+1)*h]
QuadtreeAdapter unit_mesh(int n, Real h) {
    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, n * h, 0.0, n * h, n, n);
    return mesh;
}

/// Square hole covering elements [lo, hi) in both directions on a unit_mesh(n, h)
auto hole_predicate(Real lo, Real hi) {
    return [lo, hi](Real x, Real y) { return !(x > lo && x < hi && y > lo && y < hi); };
}

} // namespace

TEST(ElementDataMaskTest, NoMasksMeansAllWater) {
    QuadtreeAdapter mesh = unit_mesh(4, 10.0);
    const ElementDataMask mask(mesh, {}, {});

    EXPECT_EQ(mask.num_water(), mesh.num_elements());
    EXPECT_FALSE(mask.has_pinned_elements());
}

TEST(ElementDataMaskTest, SingleElementHoleIsBeach) {
    QuadtreeAdapter mesh = unit_mesh(8, 10.0);
    const ElementDataMask mask(mesh, hole_predicate(30.0, 40.0), {});

    EXPECT_EQ(mask.num_beach(), 1);
    EXPECT_EQ(mask.num_inland(), 0);
    EXPECT_EQ(mask[mesh.find_element(Vec2(35.0, 35.0))], ElementDataClass::Beach);
}

// The 3x3 hole's centre element touches only NoData elements, so it is Inland;
// the ring around it all touches water.
TEST(ElementDataMaskTest, HoleInteriorIsInland) {
    QuadtreeAdapter mesh = unit_mesh(8, 10.0);
    const ElementDataMask mask(mesh, hole_predicate(20.0, 50.0), {});

    EXPECT_EQ(mask.num_inland(), 1);
    EXPECT_EQ(mask.num_beach(), 8);
    EXPECT_EQ(mask[mesh.find_element(Vec2(35.0, 35.0))], ElementDataClass::Inland);
    EXPECT_EQ(mask[mesh.find_element(Vec2(25.0, 25.0))], ElementDataClass::Beach);
}

// The invariant that makes skipping Inland assembly safe: an element sharing only
// a corner with water is still Beach, so its shared node's free derivative DOFs
// are supported by an element that is actually assembled.
TEST(ElementDataMaskTest, DiagonalContactIsBeach) {
    QuadtreeAdapter mesh = unit_mesh(4, 10.0);

    // Everything is NoData except element (0, 0), which touches element (1, 1)
    // only at the corner (10, 10)
    auto has_data = [](Real x, Real y) { return x < 10.0 && y < 10.0; };
    const ElementDataMask mask(mesh, has_data, {});

    EXPECT_EQ(mask[mesh.find_element(Vec2(5.0, 5.0))], ElementDataClass::Water);
    EXPECT_EQ(mask[mesh.find_element(Vec2(15.0, 15.0))], ElementDataClass::Beach)
        << "corner contact with water must not be classified Inland";
    EXPECT_EQ(mask[mesh.find_element(Vec2(35.0, 35.0))], ElementDataClass::Inland);
}

// A partly surveyed element keeps its data and stays in the solve
TEST(ElementDataMaskTest, PartialCoverageStaysWater) {
    QuadtreeAdapter mesh = unit_mesh(4, 10.0);

    // A gap covering only the lower-left quarter of element (0, 0)
    auto has_data = [](Real x, Real y) { return !(x < 5.0 && y < 5.0); };
    const ElementDataMask mask(mesh, has_data, {});

    EXPECT_EQ(mask.num_water(), mesh.num_elements());
    EXPECT_FALSE(mask.has_pinned_elements());
}

// =============================================================================
// Inland elements leave the system
// =============================================================================

TEST(NoDataElementTest, InlandElementsAreOutOfTheSystem) {
    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 80.0, 0.0, 80.0, 8, 8);

    // A 3x3 block of elements, so the middle one is Inland
    HoleSource source(100.0, 20.0, 50.0);

    CGHermiteBathymetrySmoother smoother(mesh, hermite_config());
    smoother.set_bathymetry_data(source);
    smoother.solve();

    const ElementDataMask* mask = smoother.element_mask();
    ASSERT_NE(mask, nullptr);
    EXPECT_EQ(mask->num_inland(), 1);

    // The solve must still succeed, the condensed system must still be SPD, and
    // the structural continuity must be untouched by the pins
    EXPECT_NEAR(smoother.constraint_violation(), 0.0, 1e-12);
    EXPECT_NEAR(smoother.evaluate(35.0, 35.0), 0.0, 1e-8);

    CGHermiteDofManager unpinned(mesh, 1, false);
    EXPECT_LT(smoother.num_free_dofs(), unpinned.num_free_dofs());
}
