#include <gtest/gtest.h>
#include "dg/mortar.hpp"
#include "dg/basis_hexahedron.hpp"
#include "dg/face_connection.hpp"
#include "../test_utils.hpp"

using namespace drifter;
using namespace drifter::testing;

class MortarTest : public DrifterTestBase {
protected:
    void SetUp() override {
        DrifterTestBase::SetUp();
    }

    // Helper to create a same-level face connection
    FaceConnection create_same_level_connection() {
        FaceConnection conn;
        conn.type = FaceConnectionType::SameLevel;
        conn.coarse_elem = 0;
        conn.coarse_face_id = 1;
        conn.fine_elems = {1};
        conn.fine_face_ids = {0};
        conn.subface_indices = {0};
        return conn;
    }

    // Helper to create a 1:4 face connection
    FaceConnection create_1_to_4_connection() {
        FaceConnection conn;
        conn.type = FaceConnectionType::Fine2x2;
        conn.coarse_elem = 0;
        conn.coarse_face_id = 1;
        conn.fine_elems = {1, 2, 3, 4};
        conn.fine_face_ids = {0, 0, 0, 0};
        conn.subface_indices = {0, 1, 2, 3};
        return conn;
    }
};

// Test mortar space construction for same-level interface
TEST_F(MortarTest, MortarSpaceConstructionSameLevel) {
    for (int order = 1; order <= 3; ++order) {
        HexahedronBasis basis(order);
        FaceConnection conn = create_same_level_connection();

        MortarSpace mortar(conn, basis, basis, conn.coarse_face_id);

        int expected_dofs = (order + 1) * (order + 1);
        EXPECT_EQ(mortar.num_dofs(), expected_dofs) << "Order " << order;
        EXPECT_EQ(mortar.order(), order) << "Order " << order;
    }
}

// Test mortar space construction for 1:4 interface
TEST_F(MortarTest, MortarSpaceConstruction1To4) {
    for (int order = 1; order <= 3; ++order) {
        HexahedronBasis basis(order);
        FaceConnection conn = create_1_to_4_connection();

        MortarSpace mortar(conn, basis, basis, conn.coarse_face_id);

        int expected_dofs = (order + 1) * (order + 1);
        EXPECT_EQ(mortar.num_dofs(), expected_dofs) << "Order " << order;
    }
}

// Test that mortar mass matrix is positive definite
TEST_F(MortarTest, MassMatrixPositiveDefinite) {
    for (int order = 1; order <= 3; ++order) {
        HexahedronBasis basis(order);
        FaceConnection conn = create_same_level_connection();

        MortarSpace mortar(conn, basis, basis, conn.coarse_face_id);
        const MatX& M = mortar.mass_matrix();

        // Check symmetry
        EXPECT_TRUE(matrices_equal(M, M.transpose(), TOLERANCE))
            << "Order " << order << ": Mass matrix not symmetric";

        // Check positive definiteness (all eigenvalues positive)
        Eigen::SelfAdjointEigenSolver<MatX> solver(M);
        VecX eigenvalues = solver.eigenvalues();
        for (Index i = 0; i < eigenvalues.size(); ++i) {
            EXPECT_GT(eigenvalues(i), 0.0)
                << "Order " << order << ": Non-positive eigenvalue";
        }
    }
}

// Test MortarInterfaceManager
TEST_F(MortarTest, InterfaceManagerCreation) {
    MortarInterfaceManager manager(3);  // Order 3

    // Add a 1:4 interface (non-conforming, needs mortar)
    FaceConnection conn = create_1_to_4_connection();
    manager.register_interface(conn);
    manager.build_operators();

    EXPECT_TRUE(manager.has_mortar(0, 1));
    EXPECT_FALSE(manager.has_mortar(0, 0));
    EXPECT_EQ(manager.num_interfaces(), 1u);
}

// Test adding multiple interfaces
TEST_F(MortarTest, MultipleInterfaces) {
    MortarInterfaceManager manager(2);

    // Add non-conforming interfaces (1:4 type needs mortar)
    FaceConnection conn1 = create_1_to_4_connection();
    manager.register_interface(conn1);

    FaceConnection conn2;
    conn2.type = FaceConnectionType::Fine2x2;  // 1:4 non-conforming
    conn2.coarse_elem = 2;
    conn2.coarse_face_id = 3;
    conn2.fine_elems = {5, 6, 7, 8};
    conn2.fine_face_ids = {0, 0, 0, 0};
    conn2.subface_indices = {0, 1, 2, 3};
    manager.register_interface(conn2);

    manager.build_operators();

    EXPECT_EQ(manager.num_interfaces(), 2u);
    EXPECT_TRUE(manager.has_mortar(0, 1));
    EXPECT_TRUE(manager.has_mortar(2, 3));
}

// Test face connection helper functions
TEST_F(MortarTest, FaceConnectionHelpers) {
    FaceConnection conn1;
    conn1.type = FaceConnectionType::SameLevel;
    EXPECT_EQ(conn1.num_fine_faces(), 1);
    EXPECT_TRUE(conn1.is_conforming());

    FaceConnection conn2;
    conn2.type = FaceConnectionType::Fine2x2;
    EXPECT_EQ(conn2.num_fine_faces(), 4);
    EXPECT_FALSE(conn2.is_conforming());

    FaceConnection conn3;
    conn3.type = FaceConnectionType::Fine2x1;
    EXPECT_EQ(conn3.num_fine_faces(), 2);
    EXPECT_FALSE(conn3.is_conforming());

    FaceConnection conn4;
    conn4.type = FaceConnectionType::Fine3_2plus1;
    EXPECT_EQ(conn4.num_fine_faces(), 3);
    EXPECT_FALSE(conn4.is_conforming());

    FaceConnection conn5;
    conn5.type = FaceConnectionType::Boundary;
    EXPECT_TRUE(conn5.is_boundary());
}

// Test mortar quadrature has expected number of points
TEST_F(MortarTest, QuadraturePoints) {
    for (int order = 1; order <= 3; ++order) {
        HexahedronBasis basis(order);
        FaceConnection conn = create_same_level_connection();

        MortarSpace mortar(conn, basis, basis, conn.coarse_face_id);
        const GaussQuadrature2D& quad = mortar.quadrature();

        // Mortar uses 2*order+1 quadrature points per direction
        int expected_points = (2 * order + 1) * (2 * order + 1);
        EXPECT_EQ(quad.size(), expected_points) << "Order " << order;
    }
}
