// Integration tests for mesh operations

#include <gtest/gtest.h>
#include "mesh/octree_adapter.hpp"
#include "test_integration_fixtures.hpp"

using namespace drifter;
using namespace drifter::testing;

TEST_F(SimulationTest, OctreeMeshCreation) {
    // Create a simple 2x2x2 mesh
    OctreeAdapter mesh(0.0, 1000.0, 0.0, 1000.0, -100.0, 0.0);
    mesh.build_uniform(2, 2, 2);

    EXPECT_EQ(mesh.num_elements(), 8);

    // Check element bounds
    for (Index e = 0; e < mesh.num_elements(); ++e) {
        const auto& bounds = mesh.element_bounds(e);
        EXPECT_GE(bounds.xmin, 0.0);
        EXPECT_LE(bounds.xmax, 1000.0);
        EXPECT_GE(bounds.ymin, 0.0);
        EXPECT_LE(bounds.ymax, 1000.0);
        EXPECT_GE(bounds.zmin, -100.0);
        EXPECT_LE(bounds.zmax, 0.0);
    }
}

TEST_F(SimulationTest, FaceConnections) {
    // Create mesh and check face connections
    OctreeAdapter mesh(0.0, 1000.0, 0.0, 1000.0, -100.0, 0.0);
    mesh.build_uniform(2, 2, 1);  // 2x2x1 = 4 elements

    auto connections = mesh.build_face_connections();
    EXPECT_EQ(connections.size(), 4);

    // Each element should have 6 faces
    for (const auto& elem_conns : connections) {
        EXPECT_EQ(elem_conns.size(), 6);
    }

    // Count boundary vs interior faces
    int boundary_count = 0;
    int interior_count = 0;

    for (Index e = 0; e < mesh.num_elements(); ++e) {
        for (int f = 0; f < 6; ++f) {
            if (connections[e][f].is_boundary()) {
                ++boundary_count;
            } else {
                ++interior_count;
            }
        }
    }

    // For 2x2x1 mesh:
    // - 4 elements * 6 faces = 24 total face slots
    // - Bottom/top faces: 4*2 = 8 boundary
    // - Side faces: depends on arrangement
    EXPECT_GT(boundary_count, 0);
    EXPECT_GT(interior_count, 0);
}
