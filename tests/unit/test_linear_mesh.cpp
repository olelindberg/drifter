#include "core/lowrider_config.hpp"
#include "bathymetry/linear_bezier_surface.hpp"
#include "bathymetry/linear_mesh_generator.hpp"
#include "bathymetry/linear_mesh_error_estimator.hpp"
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

TEST_F(LinearMeshTest, ErrorMetricTypes) {
    LinearMeshElementError err;
    err.l2_error = 1.0;
    err.normalized_error = 0.5;
    err.mean_difference = 0.3;
    err.volume_error = 100.0;

    EXPECT_NEAR(LinearMeshErrorEstimator::get_metric(err, ErrorMetricType::NormalizedError), 0.5, 1e-12);
    EXPECT_NEAR(LinearMeshErrorEstimator::get_metric(err, ErrorMetricType::MeanDifference), 0.3, 1e-12);
    EXPECT_NEAR(LinearMeshErrorEstimator::get_metric(err, ErrorMetricType::VolumeError), 100.0, 1e-12);
}
