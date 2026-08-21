#include "core/lowrider_config.hpp"
#include "bathymetry/element_error_estimator.hpp"
#include "bathymetry/linear_bezier_surface.hpp"
#include "bathymetry/linear_mesh_generator.hpp"
#include "io/quadtree_vtk_writer.hpp"
#include <gtest/gtest.h>

using namespace drifter;

class LinearMeshTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Default configuration
        config_.error_threshold = 0.5;
        config_.max_iterations = 5;
        config_.max_elements = 1000;
        config_.max_level = 6;
        config_.dorfler_theta = 0.5;
        config_.ngauss = 4;
    }

    LowriderRefinementConfig config_;
};

TEST_F(LinearMeshTest, CreateUniformMesh) {
    LinearMeshGenerator gen(0.0, 100.0, 0.0, 100.0, 4, 4, config_);

    EXPECT_EQ(gen.mesh().num_elements(), 16);  // 4x4 = 16 elements
}

TEST_F(LinearMeshTest, LinearBezierSurfaceConstruction) {
    // Build a standalone surface directly from the mesh
    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 100.0, 0.0, 100.0, 2, 2);

    LinearBezierSurface surface(mesh);
    // 2x2 mesh has 9 global DOFs (3x3 grid of corners)
    EXPECT_EQ(surface.num_dofs(), 9);
}

TEST_F(LinearMeshTest, BilinearBasis) {
    // Test bilinear basis at corners
    auto N00 = LinearBezierSurface::basis(0.0, 0.0);
    EXPECT_NEAR(N00(0), 1.0, 1e-12);  // N00 at (0,0)
    EXPECT_NEAR(N00(1), 0.0, 1e-12);
    EXPECT_NEAR(N00(2), 0.0, 1e-12);
    EXPECT_NEAR(N00(3), 0.0, 1e-12);

    auto N11 = LinearBezierSurface::basis(1.0, 1.0);
    EXPECT_NEAR(N11(0), 0.0, 1e-12);
    EXPECT_NEAR(N11(1), 0.0, 1e-12);
    EXPECT_NEAR(N11(2), 0.0, 1e-12);
    EXPECT_NEAR(N11(3), 1.0, 1e-12);  // N11 at (1,1)

    // Test partition of unity
    auto N_mid = LinearBezierSurface::basis(0.5, 0.5);
    EXPECT_NEAR(N_mid.sum(), 1.0, 1e-12);
}

TEST_F(LinearMeshTest, VTKWriterMeshOnly) {
    LinearMeshGenerator gen(0.0, 100.0, 0.0, 100.0, 2, 2, config_);

    // Should not throw
    gen.write_vtk("/tmp/lowrider_test_mesh");
}

TEST_F(LinearMeshTest, ElementErrorStruct) {
    // Test the unified ElementError struct
    ElementError err;
    err.element = 5;
    err.error = 2.5;
    err.area = 100.0;
    err.sample_count = 16;

    EXPECT_EQ(err.element, 5);
    EXPECT_NEAR(err.error, 2.5, 1e-12);
    EXPECT_NEAR(err.area, 100.0, 1e-12);
    EXPECT_EQ(err.sample_count, 16);
}
