#include <gtest/gtest.h>
#include "dg/quadrature_3d.hpp"
#include "../test_utils.hpp"
#include <cmath>

using namespace drifter;
using namespace drifter::testing;

class Quadrature3DTest : public DrifterTestBase {
protected:
    void SetUp() override {
        DrifterTestBase::SetUp();
    }
};

// Test that weights sum to volume of reference cube (8)
TEST_F(Quadrature3DTest, WeightsSumToVolume) {
    for (int order = 1; order <= 5; ++order) {
        GaussQuadrature3D quad(order);

        Real sum = quad.weights().sum();
        EXPECT_NEAR(sum, 8.0, TOLERANCE) << "Order " << order;
    }
}

// Test integration of constant function
TEST_F(Quadrature3DTest, IntegrateConstant) {
    for (int order = 1; order <= 5; ++order) {
        GaussQuadrature3D quad(order);

        Real constant = 3.14159;
        Real integral = constant * quad.weights().sum();

        // Integral of constant over [-1,1]^3 = 8 * constant
        EXPECT_NEAR(integral, 8.0 * constant, TOLERANCE) << "Order " << order;
    }
}

// Test integration of polynomial up to expected accuracy
TEST_F(Quadrature3DTest, IntegratePolynomial) {
    for (int order = 2; order <= 5; ++order) {
        GaussQuadrature3D quad(order);

        // Test: integral of x^2 over [-1,1]^3
        // = (integral of x^2 from -1 to 1) * 2 * 2 = (2/3) * 4 = 8/3
        const auto& nodes = quad.nodes();
        const auto& weights = quad.weights();

        Real integral = 0.0;
        for (int i = 0; i < quad.size(); ++i) {
            Real x = nodes[i](0);
            integral += x * x * weights(i);
        }

        EXPECT_NEAR(integral, 8.0 / 3.0, TOLERANCE) << "Order " << order;
    }
}

// Test integration of x*y*z
TEST_F(Quadrature3DTest, IntegrateXYZ) {
    for (int order = 2; order <= 5; ++order) {
        GaussQuadrature3D quad(order);

        const auto& nodes = quad.nodes();
        const auto& weights = quad.weights();

        // Integral of x*y*z over [-1,1]^3 = 0 (odd function)
        Real integral = 0.0;
        for (int i = 0; i < quad.size(); ++i) {
            const Vec3& p = nodes[i];
            integral += p(0) * p(1) * p(2) * weights(i);
        }

        EXPECT_NEAR(integral, 0.0, TOLERANCE) << "Order " << order;
    }
}

// Test integration of x^2 * y^2 * z^2
TEST_F(Quadrature3DTest, IntegrateX2Y2Z2) {
    for (int order = 3; order <= 5; ++order) {
        GaussQuadrature3D quad(order);

        const auto& nodes = quad.nodes();
        const auto& weights = quad.weights();

        // Integral of x^2*y^2*z^2 over [-1,1]^3 = (2/3)^3 = 8/27
        Real integral = 0.0;
        for (int i = 0; i < quad.size(); ++i) {
            const Vec3& p = nodes[i];
            integral += p(0)*p(0) * p(1)*p(1) * p(2)*p(2) * weights(i);
        }

        EXPECT_NEAR(integral, 8.0 / 27.0, TOLERANCE) << "Order " << order;
    }
}

// Test face quadrature weights sum to face area (4)
TEST_F(Quadrature3DTest, FaceWeightsSumToArea) {
    for (int order = 1; order <= 5; ++order) {
        for (int face = 0; face < 6; ++face) {
            FaceQuadrature face_quad(face, order);

            Real sum = face_quad.weights().sum();
            EXPECT_NEAR(sum, 4.0, TOLERANCE)
                << "Order " << order << ", face " << face;
        }
    }
}

// Test that quadrature points are in reference element
TEST_F(Quadrature3DTest, PointsInReferenceElement) {
    for (int order = 1; order <= 5; ++order) {
        GaussQuadrature3D quad(order);

        const auto& nodes = quad.nodes();
        for (int i = 0; i < quad.size(); ++i) {
            const Vec3& p = nodes[i];

            EXPECT_GE(p(0), -1.0) << "Order " << order << ", point " << i;
            EXPECT_LE(p(0), 1.0) << "Order " << order << ", point " << i;
            EXPECT_GE(p(1), -1.0) << "Order " << order << ", point " << i;
            EXPECT_LE(p(1), 1.0) << "Order " << order << ", point " << i;
            EXPECT_GE(p(2), -1.0) << "Order " << order << ", point " << i;
            EXPECT_LE(p(2), 1.0) << "Order " << order << ", point " << i;
        }
    }
}

// Test face point normal directions
TEST_F(Quadrature3DTest, FaceNormals) {
    // Expected outward normals for each face
    std::vector<Vec3> expected_normals = {
        Vec3(-1, 0, 0),  // Face 0: xi = -1
        Vec3(1, 0, 0),   // Face 1: xi = +1
        Vec3(0, -1, 0),  // Face 2: eta = -1
        Vec3(0, 1, 0),   // Face 3: eta = +1
        Vec3(0, 0, -1),  // Face 4: zeta = -1
        Vec3(0, 0, 1)    // Face 5: zeta = +1
    };

    for (int face = 0; face < 6; ++face) {
        FaceQuadrature face_quad(face, 2);
        Vec3 normal = face_quad.normal();

        EXPECT_NEAR(normal(0), expected_normals[face](0), TOLERANCE)
            << "Face " << face << ", x component";
        EXPECT_NEAR(normal(1), expected_normals[face](1), TOLERANCE)
            << "Face " << face << ", y component";
        EXPECT_NEAR(normal(2), expected_normals[face](2), TOLERANCE)
            << "Face " << face << ", z component";
    }
}

// Test that face quadrature points are on the correct face
TEST_F(Quadrature3DTest, FacePointsOnFace) {
    for (int order = 1; order <= 4; ++order) {
        for (int face = 0; face < 6; ++face) {
            FaceQuadrature face_quad(face, order);
            const auto& volume_nodes = face_quad.volume_nodes();

            for (size_t i = 0; i < volume_nodes.size(); ++i) {
                const Vec3& p = volume_nodes[i];

                // Check that the fixed coordinate is at the boundary
                switch (face) {
                    case 0: EXPECT_NEAR(p(0), -1.0, TOLERANCE); break;
                    case 1: EXPECT_NEAR(p(0), 1.0, TOLERANCE); break;
                    case 2: EXPECT_NEAR(p(1), -1.0, TOLERANCE); break;
                    case 3: EXPECT_NEAR(p(1), 1.0, TOLERANCE); break;
                    case 4: EXPECT_NEAR(p(2), -1.0, TOLERANCE); break;
                    case 5: EXPECT_NEAR(p(2), 1.0, TOLERANCE); break;
                }
            }
        }
    }
}

// Test polynomial exactness (Gauss quadrature integrates polynomials exactly)
TEST_F(Quadrature3DTest, PolynomialExactness) {
    // Gauss quadrature with n points is exact for polynomials up to degree 2n-1
    for (int n = 2; n <= 5; ++n) {
        GaussQuadrature3D quad(n);

        const auto& nodes = quad.nodes();
        const auto& weights = quad.weights();

        // Test x^(2n-2) integration (should be exact)
        int degree = 2 * n - 2;

        // Integral of x^(2k) from -1 to 1 = 2/(2k+1) for even k
        // For [-1,1]^3, multiply by 4 (from y and z integrals)
        Real expected = 2.0 / (degree + 1) * 4.0;

        Real integral = 0.0;
        for (int i = 0; i < quad.size(); ++i) {
            Real x = nodes[i](0);
            Real x_power = std::pow(x, degree);
            integral += x_power * weights(i);
        }

        EXPECT_NEAR(integral, expected, TOLERANCE)
            << "Order " << n << ", degree " << degree;
    }
}

// Test 1D quadrature node count
TEST_F(Quadrature3DTest, NodeCount1D) {
    for (int order = 1; order <= 5; ++order) {
        GaussQuadrature1D quad(order, QuadratureType::GaussLegendre);
        EXPECT_EQ(quad.size(), order) << "Order " << order;
    }
}

// Test 2D quadrature node count
TEST_F(Quadrature3DTest, NodeCount2D) {
    for (int order = 1; order <= 5; ++order) {
        GaussQuadrature2D quad(order);
        EXPECT_EQ(quad.size(), order * order) << "Order " << order;
    }
}

// Test 3D quadrature node count
TEST_F(Quadrature3DTest, NodeCount3D) {
    for (int order = 1; order <= 5; ++order) {
        GaussQuadrature3D quad(order);
        EXPECT_EQ(quad.size(), order * order * order) << "Order " << order;
    }
}
