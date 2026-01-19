#include <gtest/gtest.h>
#include "bathymetry/cg_dof_manager.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "bathymetry/lagrange_basis_2d.hpp"
#include <cmath>

using namespace drifter;

class CGDofManagerTest : public ::testing::Test {
protected:
    static constexpr Real TOLERANCE = 1e-12;
    static constexpr int ORDER = 5;  // Use quintic for backward compatibility

    std::unique_ptr<QuadtreeAdapter> mesh_;
    std::unique_ptr<LagrangeBasis2D> basis_;

    void SetUp() override {
        basis_ = std::make_unique<LagrangeBasis2D>(ORDER);
    }

    void create_uniform_mesh(int nx, int ny) {
        mesh_ = std::make_unique<QuadtreeAdapter>();
        mesh_->build_uniform(0.0, 10.0, 0.0, 10.0, nx, ny);
    }
};

// =============================================================================
// Basic construction tests
// =============================================================================

TEST_F(CGDofManagerTest, ConstructionSingleElement) {
    create_uniform_mesh(1, 1);
    CGDofManager dofs(*mesh_, *basis_);

    // Single element: all 36 DOFs are unique
    EXPECT_EQ(dofs.num_global_dofs(), 36);
    EXPECT_EQ(dofs.num_free_dofs(), 36);
}

TEST_F(CGDofManagerTest, ConstructionTwoByTwoMesh) {
    create_uniform_mesh(2, 2);
    CGDofManager dofs(*mesh_, *basis_);

    // 2x2 mesh with quintic elements (6 nodes per direction)
    // Global DOF count: (2*5+1) * (2*5+1) = 11 * 11 = 121
    // Each element has 5 interior nodes per edge, 4 corners shared
    EXPECT_EQ(dofs.num_global_dofs(), 121);
    EXPECT_EQ(dofs.num_free_dofs(), 121);  // No constraints in uniform mesh
}

TEST_F(CGDofManagerTest, ConstructionThreeByThreeMesh) {
    create_uniform_mesh(3, 3);
    CGDofManager dofs(*mesh_, *basis_);

    // 3x3 mesh: (3*5+1) * (3*5+1) = 16 * 16 = 256
    EXPECT_EQ(dofs.num_global_dofs(), 256);
    EXPECT_EQ(dofs.num_free_dofs(), 256);
}

TEST_F(CGDofManagerTest, ConstructionEmptyMesh) {
    mesh_ = std::make_unique<QuadtreeAdapter>();  // Empty mesh
    CGDofManager dofs(*mesh_, *basis_);

    EXPECT_EQ(dofs.num_global_dofs(), 0);
    EXPECT_EQ(dofs.num_free_dofs(), 0);
}

// =============================================================================
// Element DOF access tests
// =============================================================================

TEST_F(CGDofManagerTest, ElementDofCount) {
    create_uniform_mesh(2, 2);
    CGDofManager dofs(*mesh_, *basis_);

    for (Index e = 0; e < mesh_->num_elements(); ++e) {
        const auto& elem_dofs = dofs.element_dofs(e);
        EXPECT_EQ(elem_dofs.size(), 36);
    }
}

TEST_F(CGDofManagerTest, ElementDofValidity) {
    create_uniform_mesh(2, 2);
    CGDofManager dofs(*mesh_, *basis_);

    for (Index e = 0; e < mesh_->num_elements(); ++e) {
        const auto& elem_dofs = dofs.element_dofs(e);
        for (Index d : elem_dofs) {
            EXPECT_GE(d, 0);
            EXPECT_LT(d, dofs.num_global_dofs());
        }
    }
}

TEST_F(CGDofManagerTest, ElementDofUniqueness) {
    create_uniform_mesh(1, 1);
    CGDofManager dofs(*mesh_, *basis_);

    const auto& elem_dofs = dofs.element_dofs(0);
    std::set<Index> unique_dofs(elem_dofs.begin(), elem_dofs.end());

    // All 36 DOFs should be unique for a single element
    EXPECT_EQ(unique_dofs.size(), 36);
}

TEST_F(CGDofManagerTest, SharedVertexDofs) {
    create_uniform_mesh(2, 2);
    CGDofManager dofs(*mesh_, *basis_);

    // Find two adjacent elements sharing a vertex
    // Element 0: bottom-left, Element 3: top-right
    // They share the center vertex

    const auto& dofs0 = dofs.element_dofs(0);
    const auto& dofs3 = dofs.element_dofs(3);

    // Corner 3 of element 0 should match corner 0 of element 3
    // (both at the center of the 2x2 mesh)
    int corner3_local = basis_->corner_dof(3);  // (+1,+1) corner
    int corner0_local = basis_->corner_dof(0);  // (-1,-1) corner

    EXPECT_EQ(dofs0[corner3_local], dofs3[corner0_local]);
}

TEST_F(CGDofManagerTest, SharedEdgeDofs) {
    create_uniform_mesh(2, 1);  // 2 elements horizontally
    CGDofManager dofs(*mesh_, *basis_);

    // Element 0 (left) and Element 1 (right) share an edge
    const auto& dofs0 = dofs.element_dofs(0);
    const auto& dofs1 = dofs.element_dofs(1);

    // Right edge of element 0 = Left edge of element 1
    auto edge1_local = basis_->edge_dofs(1);  // Right edge of elem 0
    auto edge0_local = basis_->edge_dofs(0);  // Left edge of elem 1

    EXPECT_EQ(edge1_local.size(), edge0_local.size());

    // The shared edge DOFs should have the same global indices
    // (need to account for reverse ordering)
    for (size_t i = 0; i < edge1_local.size(); ++i) {
        Index global0 = dofs0[edge1_local[i]];
        // Edge DOFs might be in reverse order for opposite edges
        // Check that global0 exists in elem 1's edge DOFs
        bool found = false;
        for (int local : edge0_local) {
            if (dofs1[local] == global0) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << "Edge DOF " << global0 << " not shared between elements";
    }
}

// =============================================================================
// Boundary DOF tests
// =============================================================================

TEST_F(CGDofManagerTest, BoundaryDofCount) {
    create_uniform_mesh(2, 2);
    CGDofManager dofs(*mesh_, *basis_);

    // Boundary DOFs form the perimeter: 4 * (2*5+1) - 4 = 40
    // (11 per side, minus 4 corners counted multiple times)
    // Actually: 4*11 - 4 = 40
    const auto& boundary = dofs.boundary_dofs();
    EXPECT_EQ(boundary.size(), 40);
}

TEST_F(CGDofManagerTest, BoundaryDofValidity) {
    create_uniform_mesh(2, 2);
    CGDofManager dofs(*mesh_, *basis_);

    const auto& boundary = dofs.boundary_dofs();
    for (Index d : boundary) {
        EXPECT_TRUE(dofs.is_boundary_dof(d));
        EXPECT_GE(d, 0);
        EXPECT_LT(d, dofs.num_global_dofs());
    }
}

TEST_F(CGDofManagerTest, InteriorDofNotOnBoundary) {
    create_uniform_mesh(2, 2);
    CGDofManager dofs(*mesh_, *basis_);

    // Count non-boundary DOFs
    int interior_count = 0;
    for (Index d = 0; d < dofs.num_global_dofs(); ++d) {
        if (!dofs.is_boundary_dof(d)) {
            interior_count++;
        }
    }

    // Interior DOFs: total - boundary = 121 - 40 = 81
    EXPECT_EQ(interior_count, 81);
}

// =============================================================================
// DOF mapping tests
// =============================================================================

TEST_F(CGDofManagerTest, GlobalToFreeMapping) {
    create_uniform_mesh(2, 2);
    CGDofManager dofs(*mesh_, *basis_);

    // In uniform mesh, all DOFs are free (no constraints)
    for (Index g = 0; g < dofs.num_global_dofs(); ++g) {
        Index f = dofs.global_to_free(g);
        EXPECT_GE(f, 0);
        EXPECT_LT(f, dofs.num_free_dofs());
    }
}

TEST_F(CGDofManagerTest, FreeToGlobalMapping) {
    create_uniform_mesh(2, 2);
    CGDofManager dofs(*mesh_, *basis_);

    // Verify round-trip consistency
    for (Index f = 0; f < dofs.num_free_dofs(); ++f) {
        Index g = dofs.free_to_global(f);
        EXPECT_GE(g, 0);
        EXPECT_LT(g, dofs.num_global_dofs());
        EXPECT_EQ(dofs.global_to_free(g), f);
    }
}

TEST_F(CGDofManagerTest, MappingBijection) {
    create_uniform_mesh(2, 2);
    CGDofManager dofs(*mesh_, *basis_);

    // For uniform mesh: should be a bijection
    std::vector<bool> free_used(dofs.num_free_dofs(), false);

    for (Index g = 0; g < dofs.num_global_dofs(); ++g) {
        Index f = dofs.global_to_free(g);
        if (f >= 0) {
            EXPECT_FALSE(free_used[f]) << "Free DOF " << f << " mapped multiple times";
            free_used[f] = true;
        }
    }

    // All free DOFs should be used
    for (Index f = 0; f < dofs.num_free_dofs(); ++f) {
        EXPECT_TRUE(free_used[f]) << "Free DOF " << f << " not mapped";
    }
}

// =============================================================================
// Transformation matrix tests
// =============================================================================

TEST_F(CGDofManagerTest, TransformationMatrixDimensions) {
    create_uniform_mesh(2, 2);
    CGDofManager dofs(*mesh_, *basis_);

    SpMat T = dofs.build_transformation_matrix();

    EXPECT_EQ(T.rows(), dofs.num_global_dofs());
    EXPECT_EQ(T.cols(), dofs.num_free_dofs());
}

TEST_F(CGDofManagerTest, TransformationMatrixIdentityForUniformMesh) {
    create_uniform_mesh(2, 2);
    CGDofManager dofs(*mesh_, *basis_);

    SpMat T = dofs.build_transformation_matrix();

    // For uniform mesh, T should be identity (reordered)
    // T^T * T should be identity on free DOFs
    SpMat TtT = T.transpose() * T;

    // Check diagonal
    for (Index f = 0; f < dofs.num_free_dofs(); ++f) {
        EXPECT_NEAR(TtT.coeff(f, f), 1.0, TOLERANCE);
    }
}

TEST_F(CGDofManagerTest, TransformRhsPreservesSum) {
    create_uniform_mesh(2, 2);
    CGDofManager dofs(*mesh_, *basis_);

    // Create a simple RHS vector
    VecX f = VecX::Ones(dofs.num_global_dofs());

    // Create identity stiffness matrix (no Dirichlet BCs in this test)
    SpMat K(dofs.num_global_dofs(), dofs.num_global_dofs());
    K.setIdentity();

    VecX f_red = dofs.transform_rhs(f, K);

    // For uniform mesh (no constraints), sums should be equal
    EXPECT_NEAR(f.sum(), f_red.sum(), TOLERANCE);
}

TEST_F(CGDofManagerTest, ExpandSolutionRoundTrip) {
    create_uniform_mesh(2, 2);
    CGDofManager dofs(*mesh_, *basis_);

    // Create a solution in free DOF space
    VecX u_free = VecX::LinSpaced(dofs.num_free_dofs(), 1.0, 10.0);

    // Expand to global space and back
    VecX u_global = dofs.expand_solution(u_free);

    // Create identity stiffness matrix (no Dirichlet BCs in this test)
    SpMat K(dofs.num_global_dofs(), dofs.num_global_dofs());
    K.setIdentity();

    // For uniform mesh, going back should recover original
    VecX u_back = dofs.transform_rhs(u_global, K);

    // The expanded solution should have correct size
    EXPECT_EQ(u_global.size(), dofs.num_global_dofs());

    // For uniform mesh, transform_rhs should give back approximately same values
    EXPECT_NEAR((u_free - u_back).norm(), 0.0, TOLERANCE);
}

// =============================================================================
// Matrix transformation tests
// =============================================================================

TEST_F(CGDofManagerTest, TransformMatrixSymmetry) {
    create_uniform_mesh(2, 2);
    CGDofManager dofs(*mesh_, *basis_);

    // Create a simple symmetric matrix
    Index n = dofs.num_global_dofs();
    SpMat K(n, n);
    std::vector<Eigen::Triplet<Real>> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 2.0);
        if (i > 0) {
            triplets.emplace_back(i, i-1, -1.0);
            triplets.emplace_back(i-1, i, -1.0);
        }
    }
    K.setFromTriplets(triplets.begin(), triplets.end());

    SpMat K_red = dofs.transform_matrix(K);

    // Result should be symmetric
    SpMat diff = K_red - SpMat(K_red.transpose());
    EXPECT_NEAR(diff.norm(), 0.0, TOLERANCE);
}

TEST_F(CGDofManagerTest, TransformMatrixDimensions) {
    create_uniform_mesh(2, 2);
    CGDofManager dofs(*mesh_, *basis_);

    Index n = dofs.num_global_dofs();
    SpMat K(n, n);
    K.setIdentity();

    SpMat K_red = dofs.transform_matrix(K);

    EXPECT_EQ(K_red.rows(), dofs.num_free_dofs());
    EXPECT_EQ(K_red.cols(), dofs.num_free_dofs());
}

// =============================================================================
// Mesh and basis access tests
// =============================================================================

TEST_F(CGDofManagerTest, MeshAccessor) {
    create_uniform_mesh(2, 2);
    CGDofManager dofs(*mesh_, *basis_);

    EXPECT_EQ(&dofs.mesh(), mesh_.get());
}

TEST_F(CGDofManagerTest, BasisAccessor) {
    create_uniform_mesh(2, 2);
    CGDofManager dofs(*mesh_, *basis_);

    EXPECT_EQ(&dofs.basis(), basis_.get());
}

// =============================================================================
// DOF count formula verification
// =============================================================================

TEST_F(CGDofManagerTest, DofCountFormula) {
    // For uniform nx by ny quintic mesh:
    // DOF count = (nx * order + 1) * (ny * order + 1)
    // where order = 5

    for (int nx = 1; nx <= 4; ++nx) {
        for (int ny = 1; ny <= 4; ++ny) {
            create_uniform_mesh(nx, ny);
            CGDofManager dofs(*mesh_, *basis_);

            Index expected = (nx * 5 + 1) * (ny * 5 + 1);
            EXPECT_EQ(dofs.num_global_dofs(), expected)
                << "Failed for " << nx << "x" << ny << " mesh";
        }
    }
}

// =============================================================================
// All element DOFs accessor test
// =============================================================================

TEST_F(CGDofManagerTest, AllElementDofsAccessor) {
    create_uniform_mesh(2, 2);
    CGDofManager dofs(*mesh_, *basis_);

    const auto& all_dofs = dofs.all_element_dofs();

    EXPECT_EQ(all_dofs.size(), mesh_->num_elements());
    for (const auto& elem_dofs : all_dofs) {
        EXPECT_EQ(elem_dofs.size(), 36);
    }
}

// =============================================================================
// Non-conforming mesh constraint tests
// =============================================================================

class CGDofManagerNonConformingTest : public ::testing::Test {
protected:
    static constexpr Real TOLERANCE = 1e-10;
    static constexpr int ORDER = 5;  // Use quintic for backward compatibility

    std::unique_ptr<QuadtreeAdapter> mesh_;
    std::unique_ptr<LagrangeBasis2D> basis_;

    void SetUp() override {
        basis_ = std::make_unique<LagrangeBasis2D>(ORDER);
    }

    // Create a simple 2:1 non-conforming mesh:
    //  +-------+-------+
    //  |   2   |   3   |
    //  +---+---+-------+
    //  | 0 | 1 |   4   |
    //  +---+---+-------+
    // Elements 0 and 1 are half the size of elements 2, 3, 4
    void create_simple_nonconforming_mesh() {
        mesh_ = std::make_unique<QuadtreeAdapter>();

        // Fine elements (size 5x5, level 2)
        QuadBounds b0 = {0.0, 5.0, 0.0, 5.0};
        QuadBounds b1 = {5.0, 10.0, 0.0, 5.0};

        // Coarse elements (size 5x10, 10x5, level 1)
        QuadBounds b2 = {0.0, 5.0, 5.0, 10.0};
        QuadBounds b3 = {5.0, 10.0, 5.0, 10.0};
        QuadBounds b4 = {10.0, 20.0, 0.0, 10.0};  // Larger element

        mesh_->add_element(b0, {2, 2});
        mesh_->add_element(b1, {2, 2});
        mesh_->add_element(b2, {2, 1});  // Coarser in y
        mesh_->add_element(b3, {2, 1});  // Coarser in y
        mesh_->add_element(b4, {1, 1});  // Coarser in both
    }

    // Create simple L-shaped non-conforming mesh:
    //  +---+---+
    //  | 0 | 1 |
    //  +---+---+---+---+
    //      |     2     |
    //      +-----------+
    // Element 2 is twice the size of elements 0, 1
    void create_L_shaped_nonconforming_mesh() {
        mesh_ = std::make_unique<QuadtreeAdapter>();

        // Fine elements (size 5x5, level 1)
        QuadBounds b0 = {0.0, 5.0, 5.0, 10.0};
        QuadBounds b1 = {5.0, 10.0, 5.0, 10.0};

        // Coarse element (size 10x5, level 1 in x, level 0 in y)
        QuadBounds b2 = {5.0, 15.0, 0.0, 5.0};

        mesh_->add_element(b0, {1, 1});
        mesh_->add_element(b1, {1, 1});
        mesh_->add_element(b2, {1, 1});
    }
};

TEST_F(CGDofManagerNonConformingTest, DetectHangingEdges) {
    create_L_shaped_nonconforming_mesh();
    CGDofManager dofs(*mesh_, *basis_);

    // Element 1 (top-right fine element) has a FineToCoarse interface
    // with element 2 (large bottom element) on edge 2 (bottom)
    // Check that constraints were created
    const auto& constraints = dofs.constraints();

    // There should be constraints for hanging DOFs
    // Element 1 bottom edge has 6 DOFs, some are shared, some are hanging
    // Hanging DOFs are those not at coarse node positions
    EXPECT_GT(constraints.size(), 0);
}

TEST_F(CGDofManagerNonConformingTest, ConstraintWeightsPartitionOfUnity) {
    create_L_shaped_nonconforming_mesh();
    CGDofManager dofs(*mesh_, *basis_);

    const auto& constraints = dofs.constraints();

    // For each constraint, the weights should sum to 1 (partition of unity)
    for (const auto& c : constraints) {
        Real weight_sum = 0.0;
        for (Real w : c.weights) {
            weight_sum += w;
        }
        EXPECT_NEAR(weight_sum, 1.0, TOLERANCE)
            << "Constraint for DOF " << c.slave_dof << " has weight sum " << weight_sum;
    }
}

TEST_F(CGDofManagerNonConformingTest, ConstrainedDofsCount) {
    create_L_shaped_nonconforming_mesh();
    CGDofManager dofs(*mesh_, *basis_);

    // Count constrained DOFs
    int constrained_count = 0;
    for (Index g = 0; g < dofs.num_global_dofs(); ++g) {
        if (dofs.is_constrained(g)) {
            constrained_count++;
        }
    }

    // Free DOFs = Global DOFs - Constrained DOFs
    EXPECT_EQ(dofs.num_free_dofs(), dofs.num_global_dofs() - constrained_count);
}

TEST_F(CGDofManagerNonConformingTest, TransformationMatrixWithConstraints) {
    create_L_shaped_nonconforming_mesh();
    CGDofManager dofs(*mesh_, *basis_);

    SpMat T = dofs.build_transformation_matrix();

    EXPECT_EQ(T.rows(), dofs.num_global_dofs());
    EXPECT_EQ(T.cols(), dofs.num_free_dofs());

    // For each constrained DOF, the row in T should reflect the constraint
    const auto& constraints = dofs.constraints();
    for (const auto& c : constraints) {
        // Get row of T for the slave DOF
        Real row_sum = 0.0;
        for (SpMat::InnerIterator it(T, c.slave_dof); it; ++it) {
            row_sum += it.value();
        }
        // Row sum should be 1 (partition of unity from constraint weights)
        EXPECT_NEAR(row_sum, 1.0, TOLERANCE);
    }
}

TEST_F(CGDofManagerNonConformingTest, ExpandSolutionRespectsConstraints) {
    create_L_shaped_nonconforming_mesh();
    CGDofManager dofs(*mesh_, *basis_);

    // Create a solution in free DOF space
    VecX u_free = VecX::Ones(dofs.num_free_dofs());

    // Expand to global space
    VecX u_global = dofs.expand_solution(u_free);

    EXPECT_EQ(u_global.size(), dofs.num_global_dofs());

    // For constrained DOFs, check that value is computed from masters
    const auto& constraints = dofs.constraints();
    for (const auto& c : constraints) {
        Real expected = 0.0;
        for (size_t i = 0; i < c.master_dofs.size(); ++i) {
            Index master = c.master_dofs[i];
            Index free_idx = dofs.global_to_free(master);
            if (free_idx >= 0) {
                expected += c.weights[i] * u_free(free_idx);
            }
        }
        EXPECT_NEAR(u_global(c.slave_dof), expected, TOLERANCE);
    }
}

TEST_F(CGDofManagerNonConformingTest, LinearFunctionContinuityAtInterface) {
    create_L_shaped_nonconforming_mesh();
    CGDofManager dofs(*mesh_, *basis_);

    // Create solution that represents a linear function: u(x,y) = x + 2*y
    // On all DOFs, compute the expected value based on position
    VecX u_global(dofs.num_global_dofs());

    // First, compute values at all DOF positions
    for (Index e = 0; e < mesh_->num_elements(); ++e) {
        const auto& bounds = mesh_->element_bounds(e);
        const auto& elem_dofs = dofs.element_dofs(e);
        const VecX& nodes = basis_->nodes_1d();

        for (int j = 0; j < basis_->num_nodes_1d(); ++j) {
            for (int i = 0; i < basis_->num_nodes_1d(); ++i) {
                int local = basis_->dof_index(i, j);
                Index global = elem_dofs[local];

                Real xi = nodes(i);
                Real eta = nodes(j);
                Real x = bounds.xmin + 0.5 * (xi + 1.0) * (bounds.xmax - bounds.xmin);
                Real y = bounds.ymin + 0.5 * (eta + 1.0) * (bounds.ymax - bounds.ymin);

                Real expected = x + 2.0 * y;
                u_global(global) = expected;
            }
        }
    }

    // Check that constrained DOFs satisfy their constraints
    // For a linear function, the interpolation from master DOFs should give
    // the same value as direct evaluation
    const auto& constraints = dofs.constraints();
    for (const auto& c : constraints) {
        Real interpolated = 0.0;
        for (size_t i = 0; i < c.master_dofs.size(); ++i) {
            interpolated += c.weights[i] * u_global(c.master_dofs[i]);
        }
        // The interpolated value should match the direct value (for linear functions)
        EXPECT_NEAR(interpolated, u_global(c.slave_dof), 1e-8)
            << "Constraint continuity failed for DOF " << c.slave_dof;
    }
}
