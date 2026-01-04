#include <gtest/gtest.h>
#include "amr/refinement.hpp"
#include "mesh/octree_adapter.hpp"
#include "../test_utils.hpp"
#include <cmath>

using namespace drifter;
using namespace drifter::testing;

class AMRTest : public DrifterTestBase {
protected:
    void SetUp() override {
        DrifterTestBase::SetUp();
    }
};

// =============================================================================
// OctreeAdapter Tests
// =============================================================================

TEST_F(AMRTest, OctreeUniformConstruction) {
    OctreeAdapter octree(0.0, 100.0, 0.0, 100.0, -1.0, 0.0);

    octree.build_uniform(2, 2, 1);

    EXPECT_EQ(octree.num_elements(), 4);  // 2x2x1 = 4 elements
}

TEST_F(AMRTest, OctreeUniformConstruction3D) {
    OctreeAdapter octree(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    octree.build_uniform(2, 2, 2);

    EXPECT_EQ(octree.num_elements(), 8);  // 2x2x2 = 8 elements
}

TEST_F(AMRTest, OctreeElementBounds) {
    OctreeAdapter octree(0.0, 100.0, 0.0, 100.0, -10.0, 0.0);

    octree.build_uniform(2, 2, 1);

    // Check first element bounds
    const ElementBounds& bounds = octree.element_bounds(0);
    EXPECT_GE(bounds.xmin, 0.0);
    EXPECT_LE(bounds.xmax, 100.0);
    EXPECT_GE(bounds.ymin, 0.0);
    EXPECT_LE(bounds.ymax, 100.0);
    EXPECT_GE(bounds.zmin, -10.0);
    EXPECT_LE(bounds.zmax, 0.0);
}

TEST_F(AMRTest, OctreeElementCenter) {
    OctreeAdapter octree(0.0, 100.0, 0.0, 100.0, -10.0, 0.0);

    octree.build_uniform(2, 2, 1);

    // All element centers should be within domain
    for (Index i = 0; i < octree.num_elements(); ++i) {
        Vec3 center = octree.element_center(i);
        EXPECT_GT(center(0), 0.0);
        EXPECT_LT(center(0), 100.0);
        EXPECT_GT(center(1), 0.0);
        EXPECT_LT(center(1), 100.0);
        EXPECT_GT(center(2), -10.0);
        EXPECT_LT(center(2), 0.0);
    }
}

TEST_F(AMRTest, OctreeFindElement) {
    OctreeAdapter octree(0.0, 100.0, 0.0, 100.0, -10.0, 0.0);

    octree.build_uniform(2, 2, 1);

    // Point at center of domain
    Vec3 p(50.0, 50.0, -5.0);
    Index elem = octree.find_element(p);
    EXPECT_GE(elem, 0);
    EXPECT_LT(elem, octree.num_elements());

    // Point outside domain
    Vec3 outside(200.0, 50.0, -5.0);
    EXPECT_EQ(octree.find_element(outside), -1);
}

TEST_F(AMRTest, OctreeNeighborInfo) {
    OctreeAdapter octree(0.0, 100.0, 0.0, 100.0, -10.0, 0.0);

    octree.build_uniform(2, 2, 1);

    // Get neighbors for first element
    auto neighbors = octree.get_face_neighbors(0);

    // Should have 6 face neighbors (some may be boundaries)
    EXPECT_EQ(neighbors.size(), 6);
}

TEST_F(AMRTest, OctreeRefinement) {
    OctreeAdapter octree(0.0, 100.0, 0.0, 100.0, -10.0, 0.0);

    octree.build_uniform(2, 2, 1);
    Index initial_count = octree.num_elements();

    // Refine element 0 in all directions
    std::vector<Index> to_refine = {0};
    std::vector<RefineMask> masks = {RefineMask::XYZ};

    octree.refine(to_refine, masks);

    // Should have more elements now
    EXPECT_GT(octree.num_elements(), initial_count);
}

TEST_F(AMRTest, OctreeCoarsening) {
    OctreeAdapter octree(0.0, 100.0, 0.0, 100.0, -10.0, 0.0);

    octree.build_uniform(4, 4, 1);  // Start with refined mesh
    Index initial_count = octree.num_elements();

    // Coarsen a group of elements
    std::vector<Index> to_coarsen = {0};
    octree.coarsen(to_coarsen);

    // Count may be same or less (depends on sibling coarsening)
    EXPECT_LE(octree.num_elements(), initial_count);
}

TEST_F(AMRTest, OctreeBalance) {
    OctreeAdapter octree(0.0, 100.0, 0.0, 100.0, -10.0, 0.0);

    octree.build_uniform(2, 2, 1);

    // Refine one element (balance is called internally by refine)
    std::vector<Index> to_refine = {0};
    std::vector<RefineMask> masks = {RefineMask::XYZ};

    octree.refine(to_refine, masks);

    // After single refinement, check balance
    // Note: multiple consecutive refinements without balance may violate 2:1
    for (Index e = 0; e < octree.num_elements(); ++e) {
        DirectionalLevel level = octree.element_level(e);
        auto neighbors = octree.get_face_neighbors(e);

        for (const auto& neighbor : neighbors) {
            if (!neighbor.is_boundary() && !neighbor.neighbor_elements.empty()) {
                for (Index n : neighbor.neighbor_elements) {
                    DirectionalLevel n_level = octree.element_level(n);
                    // 2:1 balance: level difference <= 1 per axis
                    EXPECT_LE(std::abs(level.level_x - n_level.level_x), 1);
                    EXPECT_LE(std::abs(level.level_y - n_level.level_y), 1);
                    EXPECT_LE(std::abs(level.level_z - n_level.level_z), 1);
                }
            }
        }
    }
}

TEST_F(AMRTest, OctreeMortonOrder) {
    OctreeAdapter octree(0.0, 100.0, 0.0, 100.0, -10.0, 0.0);

    octree.build_uniform(4, 4, 1);

    std::vector<Index> morton_order = octree.morton_order();

    EXPECT_EQ(morton_order.size(), static_cast<size_t>(octree.num_elements()));

    // Check all elements appear exactly once
    std::vector<bool> seen(octree.num_elements(), false);
    for (Index idx : morton_order) {
        EXPECT_GE(idx, 0);
        EXPECT_LT(idx, octree.num_elements());
        EXPECT_FALSE(seen[idx]);
        seen[idx] = true;
    }
}

TEST_F(AMRTest, OctreeMortonPartition) {
    OctreeAdapter octree(0.0, 100.0, 0.0, 100.0, -10.0, 0.0);

    octree.build_uniform(4, 4, 1);

    auto partitions = octree.morton_partition(4);

    EXPECT_EQ(partitions.size(), 4);

    // Check partitions cover all elements
    Index total = 0;
    for (const auto& [start, end] : partitions) {
        EXPECT_LE(start, end);
        total += (end - start);
    }
    EXPECT_EQ(total, octree.num_elements());
}

// =============================================================================
// SolutionProjection Tests
// =============================================================================

TEST_F(AMRTest, SolutionProjectionConstant) {
    HexahedronBasis basis(2);
    SolutionProjection projection(basis);

    int ndof = basis.num_dofs_velocity();
    VecX U_parent(ndof);
    U_parent.setConstant(42.0);

    std::vector<VecX> U_children;
    projection.project_to_children(U_parent, RefineMask::XYZ, U_children);

    // For constant solution, all children should have same constant
    EXPECT_EQ(U_children.size(), 8);  // 8 children for XYZ refinement
    for (const auto& child : U_children) {
        EXPECT_EQ(child.size(), ndof);
        for (int i = 0; i < ndof; ++i) {
            EXPECT_NEAR(child(i), 42.0, LOOSE_TOLERANCE);
        }
    }
}

TEST_F(AMRTest, SolutionProjectionLinear) {
    HexahedronBasis basis(2);
    SolutionProjection projection(basis);

    int n1d = basis.order() + 1;
    int ndof = n1d * n1d * n1d;
    const auto& nodes = basis.lgl_basis_1d().nodes;

    // Create linear solution: f = x + y + z
    VecX U_parent(ndof);
    for (int k = 0; k < n1d; ++k) {
        for (int j = 0; j < n1d; ++j) {
            for (int i = 0; i < n1d; ++i) {
                int dof = k * n1d * n1d + j * n1d + i;
                U_parent(dof) = nodes(i) + nodes(j) + nodes(k);
            }
        }
    }

    std::vector<VecX> U_children;
    projection.project_to_children(U_parent, RefineMask::XYZ, U_children);

    // Linear solution should be exactly preserved
    EXPECT_EQ(U_children.size(), 8);

    // Check that interpolation preserves the linear function
    // First child is at negative octant
    for (int k = 0; k < n1d; ++k) {
        for (int j = 0; j < n1d; ++j) {
            for (int i = 0; i < n1d; ++i) {
                int dof = k * n1d * n1d + j * n1d + i;
                // Child coords map to parent coords in [-1,0] region
                Real xi_parent = -1.0 + 0.5 * (nodes(i) + 1.0);
                Real eta_parent = -1.0 + 0.5 * (nodes(j) + 1.0);
                Real zeta_parent = -1.0 + 0.5 * (nodes(k) + 1.0);
                Real expected = xi_parent + eta_parent + zeta_parent;
                EXPECT_NEAR(U_children[0](dof), expected, LOOSE_TOLERANCE);
            }
        }
    }
}

TEST_F(AMRTest, SolutionProjectionRoundtrip) {
    HexahedronBasis basis(2);
    SolutionProjection projection(basis);

    int ndof = basis.num_dofs_velocity();
    VecX U_original(ndof);
    U_original.setRandom();

    // Project to children
    std::vector<VecX> U_children;
    projection.project_to_children(U_original, RefineMask::XYZ, U_children);

    // Project back to parent
    VecX U_recovered;
    projection.project_from_children(U_children, RefineMask::XYZ, U_recovered);

    // Should recover original (up to interpolation error)
    EXPECT_EQ(U_recovered.size(), ndof);

    // Difference should be small
    Real diff = (U_original - U_recovered).norm();
    Real orig_norm = U_original.norm();
    EXPECT_LT(diff / orig_norm, 0.1);  // Within 10%
}

TEST_F(AMRTest, SolutionProjectionDirectionalX) {
    HexahedronBasis basis(2);
    SolutionProjection projection(basis);

    int ndof = basis.num_dofs_velocity();
    VecX U_parent(ndof);
    U_parent.setConstant(1.0);

    std::vector<VecX> U_children;
    projection.project_to_children(U_parent, RefineMask::X, U_children);

    // Only 2 children for X-only refinement
    EXPECT_EQ(U_children.size(), 2);
}

TEST_F(AMRTest, SolutionProjectionDirectionalXY) {
    HexahedronBasis basis(2);
    SolutionProjection projection(basis);

    int ndof = basis.num_dofs_velocity();
    VecX U_parent(ndof);
    U_parent.setConstant(1.0);

    std::vector<VecX> U_children;
    projection.project_to_children(U_parent, RefineMask::XY, U_children);

    // 4 children for XY refinement
    EXPECT_EQ(U_children.size(), 4);
}

// =============================================================================
// Error Estimator Tests
// =============================================================================

TEST_F(AMRTest, GradientErrorEstimatorConstant) {
    HexahedronBasis basis(2);
    GradientErrorEstimator estimator(basis);

    OctreeAdapter octree(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    octree.build_uniform(2, 2, 2);

    // Constant solution - should have low error
    int ndof = basis.num_dofs_velocity();
    std::vector<VecX> solution(octree.num_elements(), VecX::Constant(ndof, 1.0));

    std::vector<ErrorIndicator> indicators;
    estimator.estimate(solution, octree, indicators);

    EXPECT_EQ(indicators.size(), static_cast<size_t>(octree.num_elements()));

    // All errors should be very small for constant
    for (const auto& ind : indicators) {
        EXPECT_NEAR(ind.value, 0.0, LOOSE_TOLERANCE);
    }
}

TEST_F(AMRTest, GradientErrorEstimatorVariable) {
    HexahedronBasis basis(2);
    GradientErrorEstimator estimator(basis);

    OctreeAdapter octree(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    octree.build_uniform(2, 2, 2);

    int n1d = basis.order() + 1;
    int ndof = n1d * n1d * n1d;

    // Variable solution - should have higher error
    std::vector<VecX> solution(octree.num_elements());
    for (size_t e = 0; e < solution.size(); ++e) {
        solution[e].resize(ndof);
        for (int i = 0; i < ndof; ++i) {
            solution[e](i) = static_cast<Real>(i);  // Varying within element
        }
    }

    std::vector<ErrorIndicator> indicators;
    estimator.estimate(solution, octree, indicators);

    // At least some elements should have non-zero error
    Real total_error = 0.0;
    for (const auto& ind : indicators) {
        total_error += ind.value;
    }
    EXPECT_GT(total_error, 0.0);
}

// =============================================================================
// AdaptiveMeshRefinement Tests
// =============================================================================

TEST_F(AMRTest, AMRConstruction) {
    RefinementParams params;
    params.max_level = 4;
    params.refine_threshold = 0.5;
    params.coarsen_threshold = 0.1;

    AdaptiveMeshRefinement amr(2, params);

    // Should construct without error
    SUCCEED();
}

TEST_F(AMRTest, AMRMarkElements) {
    RefinementParams params;
    params.refine_threshold = 0.5;
    params.coarsen_threshold = 0.1;

    AdaptiveMeshRefinement amr(2, params);

    HexahedronBasis basis(2);
    amr.set_error_estimator(std::make_unique<GradientErrorEstimator>(basis, params));

    OctreeAdapter octree(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    octree.build_uniform(2, 2, 2);

    int ndof = basis.num_dofs_velocity();
    std::vector<VecX> solution(octree.num_elements(), VecX::Constant(ndof, 1.0));

    std::vector<RefinementAction> actions;
    std::vector<RefineMask> masks;
    amr.mark_elements(octree, solution, actions, masks);

    EXPECT_EQ(actions.size(), static_cast<size_t>(octree.num_elements()));
    EXPECT_EQ(masks.size(), static_cast<size_t>(octree.num_elements()));
}

TEST_F(AMRTest, AMRBuildFaceConnections) {
    OctreeAdapter octree(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    octree.build_uniform(2, 2, 2);

    auto face_connections = octree.build_face_connections();

    EXPECT_EQ(face_connections.size(), static_cast<size_t>(octree.num_elements()));

    // Each element should have 6 face connections
    for (const auto& elem_conns : face_connections) {
        EXPECT_EQ(elem_conns.size(), 6);
    }
}

// =============================================================================
// RefineMask Tests
// =============================================================================

TEST_F(AMRTest, RefineMaskNumChildren) {
    EXPECT_EQ(num_children(RefineMask::NONE), 1);
    EXPECT_EQ(num_children(RefineMask::X), 2);
    EXPECT_EQ(num_children(RefineMask::Y), 2);
    EXPECT_EQ(num_children(RefineMask::Z), 2);
    EXPECT_EQ(num_children(RefineMask::XY), 4);
    EXPECT_EQ(num_children(RefineMask::XZ), 4);
    EXPECT_EQ(num_children(RefineMask::YZ), 4);
    EXPECT_EQ(num_children(RefineMask::XYZ), 8);
}

TEST_F(AMRTest, RefineMaskOperators) {
    RefineMask m = RefineMask::X | RefineMask::Y;
    EXPECT_TRUE(refines_x(m));
    EXPECT_TRUE(refines_y(m));
    EXPECT_FALSE(refines_z(m));

    m |= RefineMask::Z;
    EXPECT_TRUE(refines_z(m));
}

TEST_F(AMRTest, DirectionalLevelBalance) {
    DirectionalLevel l1(2, 2, 2);
    DirectionalLevel l2(3, 2, 2);
    DirectionalLevel l3(4, 2, 2);

    EXPECT_TRUE(l1.is_balanced_with(l2));   // Differ by 1
    EXPECT_FALSE(l1.is_balanced_with(l3));  // Differ by 2
}

