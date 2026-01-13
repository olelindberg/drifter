#include <gtest/gtest.h>
#include "bathymetry/quadtree_adapter.hpp"
#include "mesh/octree_adapter.hpp"
#include <cmath>

using namespace drifter;

class QuadtreeAdapterTest : public ::testing::Test {
protected:
    static constexpr Real TOLERANCE = 1e-12;

    // Create a simple uniform octree for testing
    std::unique_ptr<OctreeAdapter> create_uniform_octree(int nx, int ny, int nz) {
        auto octree = std::make_unique<OctreeAdapter>(
            0.0, 100.0,   // x bounds
            0.0, 100.0,   // y bounds
            -1.0, 0.0     // z bounds (sigma coordinate)
        );
        octree->build_uniform(nx, ny, nz);
        return octree;
    }
};

// =============================================================================
// Basic construction tests
// =============================================================================

TEST_F(QuadtreeAdapterTest, DefaultConstruction) {
    QuadtreeAdapter quadtree;
    EXPECT_EQ(quadtree.num_elements(), 0);
}

TEST_F(QuadtreeAdapterTest, BuildUniform) {
    QuadtreeAdapter quadtree;
    quadtree.build_uniform(0.0, 10.0, 0.0, 10.0, 4, 4);

    EXPECT_EQ(quadtree.num_elements(), 16);

    // Check domain bounds
    const auto& domain = quadtree.domain_bounds();
    EXPECT_NEAR(domain.xmin, 0.0, TOLERANCE);
    EXPECT_NEAR(domain.xmax, 10.0, TOLERANCE);
    EXPECT_NEAR(domain.ymin, 0.0, TOLERANCE);
    EXPECT_NEAR(domain.ymax, 10.0, TOLERANCE);
}

TEST_F(QuadtreeAdapterTest, BuildUniformElementSizes) {
    QuadtreeAdapter quadtree;
    quadtree.build_uniform(0.0, 12.0, 0.0, 8.0, 3, 2);

    EXPECT_EQ(quadtree.num_elements(), 6);

    // Each element should be 4x4
    for (Index i = 0; i < quadtree.num_elements(); ++i) {
        Vec2 size = quadtree.element_size(i);
        EXPECT_NEAR(size(0), 4.0, TOLERANCE);
        EXPECT_NEAR(size(1), 4.0, TOLERANCE);
    }
}

TEST_F(QuadtreeAdapterTest, SyncWithUniformOctree) {
    auto octree = create_uniform_octree(4, 4, 2);

    QuadtreeAdapter quadtree(*octree);

    // Bottom layer has 4x4 = 16 elements
    EXPECT_EQ(quadtree.num_elements(), 16);

    // Domain bounds should match octree XY
    const auto& domain = quadtree.domain_bounds();
    EXPECT_NEAR(domain.xmin, 0.0, TOLERANCE);
    EXPECT_NEAR(domain.xmax, 100.0, TOLERANCE);
    EXPECT_NEAR(domain.ymin, 0.0, TOLERANCE);
    EXPECT_NEAR(domain.ymax, 100.0, TOLERANCE);
}

TEST_F(QuadtreeAdapterTest, SyncWithOctreeSingleLayer) {
    // Note: OctreeAdapter rounds up to power of 2, so 3x3x1 becomes 4x4x1=16 elements
    auto octree = create_uniform_octree(2, 2, 1);  // 2x2x1 = 4 elements

    QuadtreeAdapter quadtree;
    quadtree.sync_with_octree(*octree);

    // Single layer means all elements are bottom elements
    EXPECT_EQ(quadtree.num_elements(), 4);
}

// =============================================================================
// Element query tests
// =============================================================================

TEST_F(QuadtreeAdapterTest, ElementBounds) {
    QuadtreeAdapter quadtree;
    quadtree.build_uniform(0.0, 10.0, 0.0, 10.0, 2, 2);

    // Find element containing center of domain
    Index elem = quadtree.find_element(Vec2(2.5, 2.5));
    ASSERT_GE(elem, 0);

    const auto& bounds = quadtree.element_bounds(elem);
    EXPECT_NEAR(bounds.xmin, 0.0, TOLERANCE);
    EXPECT_NEAR(bounds.xmax, 5.0, TOLERANCE);
    EXPECT_NEAR(bounds.ymin, 0.0, TOLERANCE);
    EXPECT_NEAR(bounds.ymax, 5.0, TOLERANCE);
}

TEST_F(QuadtreeAdapterTest, ElementCenter) {
    QuadtreeAdapter quadtree;
    quadtree.build_uniform(0.0, 10.0, 0.0, 10.0, 2, 2);

    // First element (bottom-left)
    Index elem = quadtree.find_element(Vec2(2.5, 2.5));
    Vec2 center = quadtree.element_center(elem);
    EXPECT_NEAR(center(0), 2.5, TOLERANCE);
    EXPECT_NEAR(center(1), 2.5, TOLERANCE);
}

TEST_F(QuadtreeAdapterTest, ElementLevel) {
    QuadtreeAdapter quadtree;
    quadtree.build_uniform(0.0, 16.0, 0.0, 16.0, 4, 4);

    for (Index i = 0; i < quadtree.num_elements(); ++i) {
        QuadLevel level = quadtree.element_level(i);
        EXPECT_EQ(level.x, 2);  // log2(4) = 2
        EXPECT_EQ(level.y, 2);
    }
}

TEST_F(QuadtreeAdapterTest, FindElement) {
    QuadtreeAdapter quadtree;
    quadtree.build_uniform(0.0, 10.0, 0.0, 10.0, 4, 4);

    // Test points in each quadrant
    EXPECT_GE(quadtree.find_element(Vec2(1.0, 1.0)), 0);    // Bottom-left
    EXPECT_GE(quadtree.find_element(Vec2(9.0, 1.0)), 0);    // Bottom-right
    EXPECT_GE(quadtree.find_element(Vec2(1.0, 9.0)), 0);    // Top-left
    EXPECT_GE(quadtree.find_element(Vec2(9.0, 9.0)), 0);    // Top-right
    EXPECT_GE(quadtree.find_element(Vec2(5.0, 5.0)), 0);    // Center

    // Test point outside domain
    EXPECT_EQ(quadtree.find_element(Vec2(-1.0, 5.0)), -1);
    EXPECT_EQ(quadtree.find_element(Vec2(11.0, 5.0)), -1);
}

TEST_F(QuadtreeAdapterTest, OctreeElementMapping) {
    auto octree = create_uniform_octree(2, 2, 2);
    QuadtreeAdapter quadtree(*octree);

    // Each quadtree element should map to a valid octree element
    for (Index i = 0; i < quadtree.num_elements(); ++i) {
        Index oct_idx = quadtree.octree_element(i);
        EXPECT_GE(oct_idx, 0);
        EXPECT_LT(oct_idx, octree->num_elements());

        // The octree element should be at the bottom (z = -1)
        const auto& oct_bounds = octree->element_bounds(oct_idx);
        EXPECT_NEAR(oct_bounds.zmin, -1.0, TOLERANCE);
    }
}

// =============================================================================
// Neighbor query tests
// =============================================================================

TEST_F(QuadtreeAdapterTest, InteriorNeighborsConforming) {
    QuadtreeAdapter quadtree;
    quadtree.build_uniform(0.0, 10.0, 0.0, 10.0, 2, 2);

    // Find bottom-left element
    Index elem = quadtree.find_element(Vec2(2.5, 2.5));
    auto neighbors = quadtree.get_edge_neighbors(elem);

    // Left edge should be boundary
    EXPECT_TRUE(neighbors[0].is_boundary());

    // Right edge should have a conforming neighbor
    EXPECT_FALSE(neighbors[1].is_boundary());
    EXPECT_TRUE(neighbors[1].is_conforming());
    EXPECT_EQ(neighbors[1].neighbor_elements.size(), 1);

    // Bottom edge should be boundary
    EXPECT_TRUE(neighbors[2].is_boundary());

    // Top edge should have a conforming neighbor
    EXPECT_FALSE(neighbors[3].is_boundary());
    EXPECT_TRUE(neighbors[3].is_conforming());
    EXPECT_EQ(neighbors[3].neighbor_elements.size(), 1);
}

TEST_F(QuadtreeAdapterTest, BoundaryDetection) {
    QuadtreeAdapter quadtree;
    quadtree.build_uniform(0.0, 10.0, 0.0, 10.0, 3, 3);

    int boundary_edge_count = 0;
    quadtree.for_each_boundary_edge([&](Index, int) {
        boundary_edge_count++;
    });

    // 3x3 grid: 12 boundary edges (3 on each side)
    EXPECT_EQ(boundary_edge_count, 12);
}

TEST_F(QuadtreeAdapterTest, InteriorEdgeCount) {
    QuadtreeAdapter quadtree;
    quadtree.build_uniform(0.0, 10.0, 0.0, 10.0, 3, 3);

    int interior_edge_count = 0;
    quadtree.for_each_interior_edge([&](Index, int, const EdgeNeighborInfo&) {
        interior_edge_count++;
    });

    // 3x3 grid: 2*3 horizontal + 2*3 vertical = 12 interior edges
    EXPECT_EQ(interior_edge_count, 12);
}

TEST_F(QuadtreeAdapterTest, NeighborReciprocity) {
    QuadtreeAdapter quadtree;
    quadtree.build_uniform(0.0, 10.0, 0.0, 10.0, 4, 4);

    // For each interior edge, verify reciprocal neighbor relationship
    quadtree.for_each_interior_edge([&](Index elem, int edge, const EdgeNeighborInfo& info) {
        ASSERT_FALSE(info.is_boundary());
        ASSERT_EQ(info.neighbor_elements.size(), 1);

        Index neighbor = info.neighbor_elements[0];
        int opposite_edge = info.neighbor_edges[0];

        // Get neighbor's view of this edge
        auto neighbor_info = quadtree.get_neighbor(neighbor, opposite_edge);
        EXPECT_FALSE(neighbor_info.is_boundary());
        ASSERT_EQ(neighbor_info.neighbor_elements.size(), 1);
        EXPECT_EQ(neighbor_info.neighbor_elements[0], elem);
    });
}

// =============================================================================
// QuadBounds tests
// =============================================================================

TEST_F(QuadtreeAdapterTest, QuadBoundsContains) {
    QuadBounds bounds{0.0, 10.0, 0.0, 10.0};

    EXPECT_TRUE(bounds.contains(Vec2(5.0, 5.0)));
    EXPECT_TRUE(bounds.contains(Vec2(0.0, 0.0)));  // On boundary
    EXPECT_TRUE(bounds.contains(Vec2(10.0, 10.0)));  // On boundary

    EXPECT_FALSE(bounds.contains(Vec2(-1.0, 5.0)));
    EXPECT_FALSE(bounds.contains(Vec2(11.0, 5.0)));
    EXPECT_FALSE(bounds.contains(Vec2(5.0, -1.0)));
    EXPECT_FALSE(bounds.contains(Vec2(5.0, 11.0)));
}

TEST_F(QuadtreeAdapterTest, QuadBoundsContainsWithTolerance) {
    QuadBounds bounds{0.0, 10.0, 0.0, 10.0};

    // Slightly outside but within tolerance
    EXPECT_TRUE(bounds.contains(Vec2(-1e-11, 5.0), 1e-10));
    EXPECT_TRUE(bounds.contains(Vec2(10.0 + 1e-11, 5.0), 1e-10));

    // Outside tolerance
    EXPECT_FALSE(bounds.contains(Vec2(-1.0, 5.0), 1e-10));
}

TEST_F(QuadtreeAdapterTest, QuadBoundsCenter) {
    QuadBounds bounds{0.0, 10.0, 2.0, 6.0};

    Vec2 center = bounds.center();
    EXPECT_NEAR(center(0), 5.0, TOLERANCE);
    EXPECT_NEAR(center(1), 4.0, TOLERANCE);
}

TEST_F(QuadtreeAdapterTest, QuadBoundsSize) {
    QuadBounds bounds{0.0, 10.0, 2.0, 6.0};

    Vec2 size = bounds.size();
    EXPECT_NEAR(size(0), 10.0, TOLERANCE);
    EXPECT_NEAR(size(1), 4.0, TOLERANCE);
}

// =============================================================================
// QuadLevel tests
// =============================================================================

TEST_F(QuadtreeAdapterTest, QuadLevelEquality) {
    QuadLevel a{2, 3};
    QuadLevel b{2, 3};
    QuadLevel c{2, 4};

    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
    EXPECT_FALSE(a != b);
    EXPECT_TRUE(a != c);
}

TEST_F(QuadtreeAdapterTest, QuadLevelMaxLevel) {
    QuadLevel a{2, 5};
    EXPECT_EQ(a.max_level(), 5);

    QuadLevel b{7, 3};
    EXPECT_EQ(b.max_level(), 7);
}

// =============================================================================
// Iteration tests
// =============================================================================

TEST_F(QuadtreeAdapterTest, ForEachElement) {
    QuadtreeAdapter quadtree;
    quadtree.build_uniform(0.0, 10.0, 0.0, 10.0, 4, 4);

    int count = 0;
    quadtree.for_each_element([&](Index, const QuadtreeNode& node) {
        EXPECT_TRUE(node.is_leaf());
        count++;
    });

    EXPECT_EQ(count, 16);
}

TEST_F(QuadtreeAdapterTest, ElementsAccessor) {
    QuadtreeAdapter quadtree;
    quadtree.build_uniform(0.0, 10.0, 0.0, 10.0, 4, 4);

    const auto& elements = quadtree.elements();
    EXPECT_EQ(elements.size(), 16);

    for (const auto* elem : elements) {
        EXPECT_TRUE(elem->is_leaf());
    }
}

// =============================================================================
// Edge neighbor type tests
// =============================================================================

TEST_F(QuadtreeAdapterTest, EdgeNeighborInfoTypes) {
    EdgeNeighborInfo boundary_info;
    boundary_info.type = EdgeNeighborInfo::Type::Boundary;
    EXPECT_TRUE(boundary_info.is_boundary());
    EXPECT_FALSE(boundary_info.is_conforming());
    EXPECT_FALSE(boundary_info.is_nonconforming());

    EdgeNeighborInfo conforming_info;
    conforming_info.type = EdgeNeighborInfo::Type::Conforming;
    EXPECT_FALSE(conforming_info.is_boundary());
    EXPECT_TRUE(conforming_info.is_conforming());
    EXPECT_FALSE(conforming_info.is_nonconforming());

    EdgeNeighborInfo c2f_info;
    c2f_info.type = EdgeNeighborInfo::Type::CoarseToFine;
    EXPECT_FALSE(c2f_info.is_boundary());
    EXPECT_FALSE(c2f_info.is_conforming());
    EXPECT_TRUE(c2f_info.is_nonconforming());

    EdgeNeighborInfo f2c_info;
    f2c_info.type = EdgeNeighborInfo::Type::FineToCoarse;
    EXPECT_FALSE(f2c_info.is_boundary());
    EXPECT_FALSE(f2c_info.is_conforming());
    EXPECT_TRUE(f2c_info.is_nonconforming());
}

// =============================================================================
// Sync with octree element bounds matching
// =============================================================================

TEST_F(QuadtreeAdapterTest, OctreeXYBoundsMatch) {
    auto octree = create_uniform_octree(3, 3, 2);
    QuadtreeAdapter quadtree(*octree);

    // For each quadtree element, verify XY bounds match the octree element
    for (Index i = 0; i < quadtree.num_elements(); ++i) {
        Index oct_idx = quadtree.octree_element(i);
        const auto& quad_bounds = quadtree.element_bounds(i);
        const auto& oct_bounds = octree->element_bounds(oct_idx);

        EXPECT_NEAR(quad_bounds.xmin, oct_bounds.xmin, TOLERANCE);
        EXPECT_NEAR(quad_bounds.xmax, oct_bounds.xmax, TOLERANCE);
        EXPECT_NEAR(quad_bounds.ymin, oct_bounds.ymin, TOLERANCE);
        EXPECT_NEAR(quad_bounds.ymax, oct_bounds.ymax, TOLERANCE);
    }
}
