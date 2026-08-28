/// @file test_nodata_masking.cpp
/// @brief NoData detection and the land / NoData Dirichlet condition
///
/// Two defects combined to put spikes in the fitted surface over small gaps in
/// the source rasters:
///   1. NaN NoData - declared by the 2024 Klimadatastyrelsen tiles - passed every
///      nodata test in the codebase and fell through to depth 0.
///   2. Land and NoData points entered the least-squares term as depth-0
///      observations, which the smoothness term then overshot around.

#include "bathymetry/biharmonic_assembler.hpp" // BathymetrySource, FunctionBathymetry
#include "bathymetry/cg_hermite_bathymetry_smoother.hpp"
#include "bathymetry/cg_hermite_dof_manager.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "mesh/geotiff_reader.hpp"
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

// The regression. Fitting the gap's zeros drags the surface to sea level over it;
// the smoothness term then overshoots into a spike. Dropping those points leaves
// the seabed to be carried across the gap at the surrounding depth.
TEST(NoDataFitTest, GapDoesNotDragTheSurface) {
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

    // Across the gap the surface must carry the surrounding depth rather than
    // rising toward sea level
    EXPECT_NEAR(smoother.evaluate(35.0, 35.0), DEPTH, 5.0)
        << "surface dragged toward sea level over the NoData gap";

    // A survey gap must not become a spike in either direction
    Real min_depth = DEPTH, max_depth = 0.0;
    for (int j = 0; j <= 80; ++j) {
        for (int i = 0; i <= 80; ++i) {
            const Real d = smoother.evaluate(i, j);
            min_depth = std::min(min_depth, d);
            max_depth = std::max(max_depth, d);
        }
    }
    EXPECT_GT(min_depth, 0.5 * DEPTH) << "fit spikes upward";
    EXPECT_LT(max_depth, 1.5 * DEPTH) << "fit spikes downward";
}

// Gap quadrature points must be dropped from the least-squares term
TEST(NoDataFitTest, GapQuadraturePointsAreExcluded) {
    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 80.0, 0.0, 80.0, 8, 8);

    HoleSource source(100.0, 30.0, 40.0);

    CGHermiteBathymetrySmoother smoother(mesh, hermite_config());
    smoother.set_bathymetry_data(source);

    EXPECT_GT(smoother.num_excluded_quadrature_points(), 0)
        << "the gap covers a whole element, so some quadrature points must be dropped";
}

// A gap must never be pinned: forcing the rim from the surrounding depth to 0
// across one element makes a bicubic Hermite overshoot violently
TEST(NoDataFitTest, GapIsNotPinnedToZero) {
    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 80.0, 0.0, 80.0, 8, 8);

    HoleSource source(100.0, 30.0, 40.0);

    CGHermiteBathymetrySmoother smoother(mesh, hermite_config());
    smoother.set_bathymetry_data(source);

    EXPECT_FALSE(static_cast<bool>(smoother.land_predicate()(35.0, 35.0)));
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

// The DOF manager pins value DOFs at land nodes, reducing the free set without
// changing the global DOF count
TEST(LandDirichletTest, DofManagerPinsLandNodes) {
    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 80.0, 0.0, 80.0, 8, 8);

    auto land = [](Real x, Real y) { return x >= 30.0 && x <= 50.0 && y >= 30.0 && y <= 50.0; };

    CGHermiteDofManager pinned(mesh, 1, false, land);
    CGHermiteDofManager unpinned(mesh, 1, false);

    EXPECT_GT(pinned.num_land_pinned_dofs(), 0);
    EXPECT_EQ(unpinned.num_land_pinned_dofs(), 0);

    EXPECT_EQ(pinned.num_global_dofs(), unpinned.num_global_dofs());
    EXPECT_LT(pinned.num_free_dofs(), unpinned.num_free_dofs());
}
