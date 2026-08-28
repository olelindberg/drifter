#include "bathymetry/quadtree_adapter.hpp"
#include "io/bathymetry_vtk_writer.hpp"
#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace drifter;

namespace {

constexpr Real TOLERANCE = 1e-12;

// =============================================================================
// VTK_LAGRANGE_QUAD node ordering
// =============================================================================

// Corners first in VTK's counter-clockwise order, then edges, then interior,
// and every tensor-product node appears exactly once.
TEST(LagrangeQuadOrderingTest, IsAValidPermutationWithCornersFirst) {
    for (int order = 1; order <= 4; ++order) {
        const auto nodes = io::lagrange_quad_ordering(order);

        ASSERT_EQ(nodes.size(), static_cast<size_t>((order + 1) * (order + 1)))
            << "order=" << order;

        const int n = order;
        EXPECT_EQ(nodes[0], std::make_pair(0, 0)) << "order=" << order;
        EXPECT_EQ(nodes[1], std::make_pair(n, 0)) << "order=" << order;
        EXPECT_EQ(nodes[2], std::make_pair(n, n)) << "order=" << order;
        EXPECT_EQ(nodes[3], std::make_pair(0, n)) << "order=" << order;

        std::set<std::pair<int, int>> unique(nodes.begin(), nodes.end());
        EXPECT_EQ(unique.size(), nodes.size()) << "order=" << order << ": duplicate nodes";
        for (const auto &[i, j] : nodes) {
            EXPECT_GE(i, 0);
            EXPECT_LE(i, n);
            EXPECT_GE(j, 0);
            EXPECT_LE(j, n);
        }
    }
}

// The edge block holds exactly the boundary-but-not-corner nodes, and the
// interior block exactly the rest.
TEST(LagrangeQuadOrderingTest, EdgeBlockPrecedesInteriorBlock) {
    const int order = 4;
    const auto nodes = io::lagrange_quad_ordering(order);
    const size_t n_edge = 4 * static_cast<size_t>(order - 1);

    for (size_t k = 4; k < 4 + n_edge; ++k) {
        const auto [i, j] = nodes[k];
        const bool on_boundary = (i == 0 || i == order || j == 0 || j == order);
        EXPECT_TRUE(on_boundary) << "node " << k << " should lie on an edge";
    }
    for (size_t k = 4 + n_edge; k < nodes.size(); ++k) {
        const auto [i, j] = nodes[k];
        EXPECT_TRUE(i > 0 && i < order && j > 0 && j < order)
            << "node " << k << " should be interior";
    }
}

// =============================================================================
// The written surface
// =============================================================================

class HighOrderSurfaceWriterTest : public ::testing::Test {
protected:
    void SetUp() override {
        mesh_.build_uniform(0.0, 100.0, 0.0, 100.0, 2, 2);
        char tmpl[] = "/tmp/drifter_vtk_test_XXXXXX";
        const int fd = mkstemp(tmpl);
        ASSERT_GE(fd, 0) << "cannot create a temporary file";
        close(fd);
        std::remove(tmpl);
        base_ = tmpl;
    }

    void TearDown() override { std::remove((base_ + ".vtu").c_str()); }

    /// Extract the numeric contents of the named DataArray from the written file
    std::vector<Real> read_array(const std::string &name) const {
        std::ifstream file(base_ + ".vtu");
        EXPECT_TRUE(file.good()) << "cannot open " << base_ << ".vtu";

        std::string line;
        while (std::getline(file, line)) {
            if (line.find("Name=\"" + name + "\"") != std::string::npos) {
                break;
            }
        }

        std::vector<Real> values;
        while (std::getline(file, line) && line.find("</DataArray>") == std::string::npos) {
            std::istringstream ss(line);
            Real v;
            while (ss >> v) {
                values.push_back(v);
            }
        }
        return values;
    }

    /// Points come back as (x, y, z) triples; the Points array has no Name
    std::vector<Vec3> read_points() const {
        std::ifstream file(base_ + ".vtu");
        EXPECT_TRUE(file.good());

        std::string line;
        while (std::getline(file, line) && line.find("<Points>") == std::string::npos) {
        }
        std::getline(file, line); // the DataArray header

        std::vector<Vec3> points;
        while (std::getline(file, line) && line.find("</DataArray>") == std::string::npos) {
            std::istringstream ss(line);
            Real x, y, z;
            if (ss >> x >> y >> z) {
                points.emplace_back(x, y, z);
            }
        }
        return points;
    }

    QuadtreeAdapter mesh_;
    std::string base_;
};

// A cell per element, of type 70, carrying (order+1)^2 points.
TEST_F(HighOrderSurfaceWriterTest, EmitsOneLagrangeQuadPerElement) {
    const int order = 3;
    io::write_high_order_surface_vtk(
        base_, mesh_, [](Index, Real, Real) { return 0.0; }, order);

    const auto types = read_array("types");
    const auto offsets = read_array("offsets");
    const auto points = read_points();

    ASSERT_EQ(types.size(), static_cast<size_t>(mesh_.num_elements()));
    for (Real t : types) {
        EXPECT_EQ(t, 70.0);
    }

    const size_t pts_per_cell = static_cast<size_t>((order + 1) * (order + 1));
    EXPECT_EQ(pts_per_cell, 16u);
    ASSERT_EQ(offsets.size(), types.size());
    for (size_t c = 0; c < offsets.size(); ++c) {
        EXPECT_EQ(offsets[c], static_cast<Real>((c + 1) * pts_per_cell));
    }
    EXPECT_EQ(points.size(), static_cast<size_t>(mesh_.num_elements()) * pts_per_cell);
}

// Every written z is the evaluator's value at that point: no interpolation, no
// averaging, no repositioning of the sample.
TEST_F(HighOrderSurfaceWriterTest, ReproducesTheEvaluatorAtEveryPoint) {
    // A bicubic, so degree 3 cells represent it exactly
    auto bicubic = [](Index, Real x, Real y) {
        const Real s = x / 100.0;
        const Real t = y / 100.0;
        return 1.0 + 2.0 * s + 3.0 * t * t - 4.0 * s * s * s * t;
    };

    io::write_high_order_surface_vtk(base_, mesh_, bicubic, 3, "elevation");

    const auto points = read_points();
    const auto elevation = read_array("elevation");
    ASSERT_EQ(elevation.size(), points.size());
    ASSERT_FALSE(points.empty());

    for (size_t k = 0; k < points.size(); ++k) {
        const Real expected = bicubic(0, points[k].x(), points[k].y());
        EXPECT_NEAR(points[k].z(), expected, TOLERANCE) << "point " << k;
        EXPECT_NEAR(elevation[k], points[k].z(), TOLERANCE) << "point " << k;
    }
}

// Raising the visual degree above the surface degree only resamples the same
// polynomial on more nodes: the geometry is unchanged, so ParaView tessellates
// it more finely without any loss of accuracy.
TEST_F(HighOrderSurfaceWriterTest, HigherVisualDegreeReproducesTheSameSurface) {
    auto bicubic = [](Index, Real x, Real y) {
        const Real s = x / 100.0;
        const Real t = y / 100.0;
        return 1.0 + 2.0 * s + 3.0 * t * t - 4.0 * s * s * s * t;
    };

    io::write_high_order_surface_vtk(base_, mesh_, bicubic, 6, "elevation");

    const auto points = read_points();
    const auto elevation = read_array("elevation");
    const size_t pts_per_cell = 7 * 7;
    ASSERT_EQ(points.size(), static_cast<size_t>(mesh_.num_elements()) * pts_per_cell);
    ASSERT_EQ(elevation.size(), points.size());

    for (size_t k = 0; k < points.size(); ++k) {
        const Real expected = bicubic(0, points[k].x(), points[k].y());
        EXPECT_NEAR(points[k].z(), expected, TOLERANCE) << "point " << k;
        EXPECT_NEAR(elevation[k], expected, TOLERANCE) << "point " << k;
    }
}

// Sample coordinates are equispaced in the parametric directions, which is what
// VTK's Lagrange cell node layout assumes. LGL nodes would misplace the surface.
TEST_F(HighOrderSurfaceWriterTest, SamplesOnEquispacedNodes) {
    const int order = 3;
    io::write_high_order_surface_vtk(
        base_, mesh_, [](Index, Real, Real) { return 0.0; }, order);

    const auto points = read_points();
    const auto ordering = io::lagrange_quad_ordering(order);
    ASSERT_EQ(points.size(), static_cast<size_t>(mesh_.num_elements()) * ordering.size());

    for (Index elem = 0; elem < mesh_.num_elements(); ++elem) {
        const auto &b = mesh_.element_bounds(elem);
        const size_t base = static_cast<size_t>(elem) * ordering.size();
        for (size_t k = 0; k < ordering.size(); ++k) {
            const auto [i, j] = ordering[k];
            const Real x = b.xmin + (b.xmax - b.xmin) * i / static_cast<Real>(order);
            const Real y = b.ymin + (b.ymax - b.ymin) * j / static_cast<Real>(order);
            EXPECT_NEAR(points[base + k].x(), x, TOLERANCE);
            EXPECT_NEAR(points[base + k].y(), y, TOLERANCE);
        }
    }
}

// Each cell's points are evaluated against that cell's own element. Nothing is
// re-located by point search, so nothing on a shared edge is attributed to the
// neighbour - the property the old shared-vertex writer could not offer.
TEST_F(HighOrderSurfaceWriterTest, EvaluatesEveryPointInItsOwnElement) {
    std::vector<Index> queried_elements;
    auto recording = [&queried_elements](Index elem, Real, Real) {
        queried_elements.push_back(elem);
        return static_cast<Real>(elem);
    };

    const int order = 2;
    io::write_high_order_surface_vtk(base_, mesh_, recording, order);

    const size_t pts_per_cell = static_cast<size_t>((order + 1) * (order + 1));
    ASSERT_EQ(queried_elements.size(),
              static_cast<size_t>(mesh_.num_elements()) * pts_per_cell);

    for (size_t k = 0; k < queried_elements.size(); ++k) {
        EXPECT_EQ(queried_elements[k], static_cast<Index>(k / pts_per_cell))
            << "point " << k << " was evaluated in the wrong element";
    }

    // The recorded element id also has to survive into the file
    const auto elevation = read_array("elevation");
    ASSERT_EQ(elevation.size(), queried_elements.size());
    for (size_t k = 0; k < elevation.size(); ++k) {
        EXPECT_EQ(elevation[k], static_cast<Real>(queried_elements[k]));
    }
}

// Per-element arrays are indexed by element id, one value per cell.
TEST_F(HighOrderSurfaceWriterTest, WritesPerElementCellData) {
    std::vector<Real> levels(static_cast<size_t>(mesh_.num_elements()));
    for (size_t e = 0; e < levels.size(); ++e) {
        levels[e] = 10.0 + static_cast<Real>(e);
    }

    io::write_high_order_surface_vtk(
        base_, mesh_, [](Index, Real, Real) { return 0.0; }, 3, "elevation",
        {{"refinement_level", levels}});

    const auto written = read_array("refinement_level");
    ASSERT_EQ(written.size(), levels.size());
    for (size_t e = 0; e < levels.size(); ++e) {
        EXPECT_EQ(written[e], levels[e]);
    }

    const auto ids = read_array("element_id");
    ASSERT_EQ(ids.size(), levels.size());
    for (size_t e = 0; e < ids.size(); ++e) {
        EXPECT_EQ(ids[e], static_cast<Real>(e));
    }
}

TEST_F(HighOrderSurfaceWriterTest, RejectsNonPositiveOrder) {
    EXPECT_THROW(io::lagrange_quad_ordering(0), std::invalid_argument);
    EXPECT_THROW(io::write_high_order_surface_vtk(
                     base_, mesh_, [](Index, Real, Real) { return 0.0; }, 0),
                 std::invalid_argument);
}

} // namespace
