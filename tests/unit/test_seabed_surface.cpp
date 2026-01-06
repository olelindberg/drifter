#include <gtest/gtest.h>
#include "mesh/seabed_surface.hpp"
#include "mesh/octree_adapter.hpp"
#include "../test_utils.hpp"
#include <cmath>

using namespace drifter;
using namespace drifter::testing;

class SeabedSurfaceTest : public DrifterTestBase {
protected:
    void SetUp() override {
        DrifterTestBase::SetUp();
    }

    // Create a simple uniform octree mesh for testing
    std::unique_ptr<OctreeAdapter> create_test_mesh(int nx, int ny, int nz,
                                                     Real xmin, Real xmax,
                                                     Real ymin, Real ymax,
                                                     Real zmin, Real zmax) {
        auto mesh = std::make_unique<OctreeAdapter>(xmin, xmax, ymin, ymax, zmin, zmax);
        mesh->build_uniform(nx, ny, nz);
        return mesh;
    }

    // Create synthetic bathymetry data (flat bottom at depth h)
    BathymetryData create_flat_bathymetry(Real depth,
                                           Real xmin, Real xmax,
                                           Real ymin, Real ymax) {
        BathymetryData bathy;
        // Use larger grid with sufficient margin for bilinear interpolation
        // Bilinear needs x0, x0+1, y0, y0+1 all valid, so need 2+ pixel margin
        bathy.sizex = 110;
        bathy.sizey = 110;
        bathy.elevation.resize(bathy.sizex * bathy.sizey, -depth);
        bathy.is_depth_positive = false;  // Elevation format

        // Extend grid 5 units beyond domain bounds (bilinear interp needs margin)
        Real margin = 5.0;
        Real grid_xmin = xmin - margin;
        Real grid_xmax = xmax + margin;
        Real grid_ymin = ymin - margin;
        Real grid_ymax = ymax + margin;

        bathy.geotransform[0] = grid_xmin;  // Top-left X
        bathy.geotransform[1] = (grid_xmax - grid_xmin) / bathy.sizex;  // Pixel width
        bathy.geotransform[2] = 0.0;   // Row rotation
        bathy.geotransform[3] = grid_ymax;  // Top-left Y
        bathy.geotransform[4] = 0.0;   // Column rotation
        bathy.geotransform[5] = -(grid_ymax - grid_ymin) / bathy.sizey;  // Pixel height (negative)

        bathy.xmin = xmin;
        bathy.xmax = xmax;
        bathy.ymin = ymin;
        bathy.ymax = ymax;
        bathy.nodata_value = -9999.0f;

        return bathy;
    }

    // Create bathymetry with linear slope in x: h(x) = h0 + slope * x
    BathymetryData create_sloped_bathymetry(Real h0, Real slope,
                                             Real xmin, Real xmax,
                                             Real ymin, Real ymax) {
        BathymetryData bathy;
        // Add margin for bilinear interpolation at domain boundaries
        Real margin = 5.0;
        Real grid_xmin = xmin - margin;
        Real grid_xmax = xmax + margin;
        Real grid_ymin = ymin - margin;
        Real grid_ymax = ymax + margin;

        bathy.sizex = 120;
        bathy.sizey = 120;
        bathy.elevation.resize(bathy.sizex * bathy.sizey);
        bathy.is_depth_positive = true;  // Depth format

        Real pixel_dx = (grid_xmax - grid_xmin) / bathy.sizex;
        Real pixel_dy = (grid_ymax - grid_ymin) / bathy.sizey;

        for (int py = 0; py < bathy.sizey; ++py) {
            for (int px = 0; px < bathy.sizex; ++px) {
                Real x = grid_xmin + (px + 0.5) * pixel_dx;
                Real depth = h0 + slope * x;
                bathy.elevation[py * bathy.sizex + px] = depth > 0 ? static_cast<float>(depth) : 0.0f;
            }
        }

        bathy.geotransform[0] = grid_xmin;
        bathy.geotransform[1] = pixel_dx;
        bathy.geotransform[2] = 0.0;
        bathy.geotransform[3] = grid_ymax;
        bathy.geotransform[4] = 0.0;
        bathy.geotransform[5] = -pixel_dy;

        bathy.xmin = xmin;
        bathy.xmax = xmax;
        bathy.ymin = ymin;
        bathy.ymax = ymax;
        bathy.nodata_value = -9999.0f;

        return bathy;
    }
};

// Test construction and bottom element identification
TEST_F(SeabedSurfaceTest, ConstructionIdentifiesBottomElements) {
    // Create 2x2x2 mesh (4 elements at bottom layer)
    // Note: build_uniform rounds up to power of 2, so 2x2x2 gives exactly 2^1 per axis
    auto mesh = create_test_mesh(2, 2, 2, 0.0, 100.0, 0.0, 100.0, -1.0, 0.0);

    SeabedSurface seabed(*mesh, 2);

    // Should have 4 bottom elements (2x2 = 4)
    EXPECT_EQ(seabed.num_elements(), 4u);
    EXPECT_EQ(seabed.order(), 2);
    EXPECT_EQ(seabed.method(), SeabedInterpolation::Bernstein);
}

// Test bottom element identification with single layer mesh
TEST_F(SeabedSurfaceTest, SingleLayerMesh) {
    // 4x4x1 mesh - all elements are bottom layer (power of 2)
    auto mesh = create_test_mesh(4, 4, 1, 0.0, 100.0, 0.0, 100.0, -1.0, 0.0);

    SeabedSurface seabed(*mesh, 3);

    // All 16 elements should be bottom elements (4*4*1 = 16)
    EXPECT_EQ(seabed.num_elements(), 16u);
}

// Test mesh element index mapping
TEST_F(SeabedSurfaceTest, MeshElementIndexMapping) {
    auto mesh = create_test_mesh(2, 2, 2, 0.0, 100.0, 0.0, 100.0, -1.0, 0.0);

    SeabedSurface seabed(*mesh, 2);

    // Check that is_bottom_element works correctly
    size_t bottom_count = 0;
    for (size_t e = 0; e < mesh->num_elements(); ++e) {
        if (seabed.is_bottom_element(static_cast<Index>(e))) {
            ++bottom_count;
        }
    }
    EXPECT_EQ(bottom_count, seabed.num_elements());

    // Check seabed_element_index returns valid indices for bottom elements
    for (size_t s = 0; s < seabed.num_elements(); ++s) {
        Index mesh_idx = seabed.mesh_element_index(s);
        Index seabed_idx = seabed.seabed_element_index(mesh_idx);
        EXPECT_EQ(seabed_idx, static_cast<Index>(s));
    }
}

// Test setting bathymetry from flat data
TEST_F(SeabedSurfaceTest, SetFromFlatBathymetry) {
    auto mesh = create_test_mesh(2, 2, 2, 0.0, 100.0, 0.0, 100.0, -1.0, 0.0);
    auto bathy = create_flat_bathymetry(50.0, 0.0, 100.0, 0.0, 100.0);

    SeabedSurface seabed(*mesh, 2);
    seabed.set_from_bathymetry(bathy);

    // Check that coefficients were set (not all zeros)
    bool has_nonzero = false;
    for (size_t s = 0; s < seabed.num_elements(); ++s) {
        const VecX& coeffs = seabed.coefficients(s);
        for (int i = 0; i < coeffs.size(); ++i) {
            if (std::abs(coeffs(i)) > 1e-10) {
                has_nonzero = true;
                break;
            }
        }
    }
    EXPECT_TRUE(has_nonzero) << "Bathymetry coefficients are all zero";

    // Depth should be approximately 50 everywhere
    Real depth_center = seabed.depth(50.0, 50.0);
    EXPECT_NEAR(depth_center, 50.0, 1.0);

    Real depth_corner = seabed.depth(10.0, 10.0);
    EXPECT_NEAR(depth_corner, 50.0, 1.0);
}

// Test depth evaluation at various points
TEST_F(SeabedSurfaceTest, DepthEvaluationSloped) {
    auto mesh = create_test_mesh(4, 4, 2, 0.0, 100.0, 0.0, 100.0, -1.0, 0.0);
    // h(x) = 20 + 0.6 * x -> ranges from 20 to 80
    auto bathy = create_sloped_bathymetry(20.0, 0.6, 0.0, 100.0, 0.0, 100.0);

    SeabedSurface seabed(*mesh, 3);
    seabed.set_from_bathymetry(bathy);

    // Test at various x positions
    EXPECT_NEAR(seabed.depth(10.0, 50.0), 26.0, 1.0);  // 20 + 0.6*10 = 26
    EXPECT_NEAR(seabed.depth(50.0, 50.0), 50.0, 1.0);  // 20 + 0.6*50 = 50
    EXPECT_NEAR(seabed.depth(90.0, 50.0), 74.0, 1.0);  // 20 + 0.6*90 = 74
}

// Test depth returns 0 for points outside mesh
TEST_F(SeabedSurfaceTest, DepthOutsideMesh) {
    auto mesh = create_test_mesh(2, 2, 1, 0.0, 100.0, 0.0, 100.0, -1.0, 0.0);
    auto bathy = create_flat_bathymetry(50.0, 0.0, 100.0, 0.0, 100.0);

    SeabedSurface seabed(*mesh, 2);
    seabed.set_from_bathymetry(bathy);

    // Outside mesh should return 0
    EXPECT_EQ(seabed.depth(-10.0, 50.0), 0.0);
    EXPECT_EQ(seabed.depth(110.0, 50.0), 0.0);
    EXPECT_EQ(seabed.depth(50.0, -10.0), 0.0);
    EXPECT_EQ(seabed.depth(50.0, 110.0), 0.0);
}

// Test gradient computation
TEST_F(SeabedSurfaceTest, GradientComputation) {
    auto mesh = create_test_mesh(4, 4, 1, 0.0, 100.0, 0.0, 100.0, -1.0, 0.0);
    // Slope of 0.6 in x direction, flat in y
    auto bathy = create_sloped_bathymetry(20.0, 0.6, 0.0, 100.0, 0.0, 100.0);

    SeabedSurface seabed(*mesh, 3);
    seabed.set_from_bathymetry(bathy);

    Real dh_dx, dh_dy;
    bool found = seabed.gradient(50.0, 50.0, dh_dx, dh_dy);

    EXPECT_TRUE(found);
    // Finite difference gradient has some error due to polynomial representation
    // Accept 20% tolerance for numerical gradient
    EXPECT_NEAR(dh_dx, 0.6, 0.15);  // Should be ~0.6 (slope in x)
    EXPECT_NEAR(dh_dy, 0.0, 0.1);   // Should be ~0 (flat in y)
}

// Test gradient returns false for points outside mesh
TEST_F(SeabedSurfaceTest, GradientOutsideMesh) {
    auto mesh = create_test_mesh(2, 2, 1, 0.0, 100.0, 0.0, 100.0, -1.0, 0.0);
    auto bathy = create_flat_bathymetry(50.0, 0.0, 100.0, 0.0, 100.0);

    SeabedSurface seabed(*mesh, 2);
    seabed.set_from_bathymetry(bathy);

    Real dh_dx, dh_dy;
    bool found = seabed.gradient(-10.0, 50.0, dh_dx, dh_dy);

    EXPECT_FALSE(found);
    EXPECT_EQ(dh_dx, 0.0);
    EXPECT_EQ(dh_dy, 0.0);
}

// Test coefficient storage
TEST_F(SeabedSurfaceTest, CoefficientStorage) {
    auto mesh = create_test_mesh(2, 2, 1, 0.0, 100.0, 0.0, 100.0, -1.0, 0.0);
    int order = 3;
    int n2d = (order + 1) * (order + 1);  // 16

    SeabedSurface seabed(*mesh, order);

    // Check coefficient vector sizes
    for (size_t s = 0; s < seabed.num_elements(); ++s) {
        EXPECT_EQ(seabed.coefficients(s).size(), n2d);
    }
}

// Test set_element_coefficients
TEST_F(SeabedSurfaceTest, SetElementCoefficients) {
    auto mesh = create_test_mesh(2, 2, 1, 0.0, 100.0, 0.0, 100.0, -1.0, 0.0);
    int order = 2;
    int n2d = (order + 1) * (order + 1);  // 9

    SeabedSurface seabed(*mesh, order);

    // Set custom coefficients (constant value 42)
    VecX custom_coeffs = VecX::Constant(n2d, 42.0);
    seabed.set_element_coefficients(0, custom_coeffs);

    // Check that coefficients were set
    const VecX& stored = seabed.coefficients(0);
    for (int i = 0; i < n2d; ++i) {
        EXPECT_NEAR(stored(i), 42.0, TOLERANCE);
    }
}

// Test rebuild_from_mesh
TEST_F(SeabedSurfaceTest, RebuildFromMesh) {
    auto mesh = create_test_mesh(2, 2, 2, 0.0, 100.0, 0.0, 100.0, -1.0, 0.0);

    SeabedSurface seabed(*mesh, 2);
    size_t initial_count = seabed.num_elements();

    // Rebuild should reset to same state (same mesh)
    seabed.rebuild_from_mesh();

    EXPECT_EQ(seabed.num_elements(), initial_count);
}

// Test Bernstein interpolation method (default)
TEST_F(SeabedSurfaceTest, BernsteinMethod) {
    auto mesh = create_test_mesh(2, 2, 1, 0.0, 100.0, 0.0, 100.0, -1.0, 0.0);

    SeabedSurface seabed(*mesh, 3, SeabedInterpolation::Bernstein);
    EXPECT_EQ(seabed.method(), SeabedInterpolation::Bernstein);
}

// Test Lagrange interpolation method
TEST_F(SeabedSurfaceTest, LagrangeMethod) {
    auto mesh = create_test_mesh(2, 2, 1, 0.0, 100.0, 0.0, 100.0, -1.0, 0.0);

    SeabedSurface seabed(*mesh, 3, SeabedInterpolation::Lagrange);
    EXPECT_EQ(seabed.method(), SeabedInterpolation::Lagrange);
}

// Test coordinates storage
TEST_F(SeabedSurfaceTest, CoordinatesStorage) {
    auto mesh = create_test_mesh(2, 2, 1, 0.0, 100.0, 0.0, 100.0, -1.0, 0.0);
    auto bathy = create_flat_bathymetry(50.0, 0.0, 100.0, 0.0, 100.0);
    int order = 2;
    int n2d = (order + 1) * (order + 1);

    SeabedSurface seabed(*mesh, order);
    seabed.set_from_bathymetry(bathy);

    // Check coordinates vector size (3 components per DOF)
    for (size_t s = 0; s < seabed.num_elements(); ++s) {
        EXPECT_EQ(seabed.coordinates(s).size(), 3 * n2d);
    }
}

// Test all_coefficients accessor
TEST_F(SeabedSurfaceTest, AllCoefficientsAccessor) {
    auto mesh = create_test_mesh(3, 3, 1, 0.0, 90.0, 0.0, 90.0, -1.0, 0.0);

    SeabedSurface seabed(*mesh, 2);

    const auto& all_coeffs = seabed.all_coefficients();
    EXPECT_EQ(all_coeffs.size(), seabed.num_elements());
}

// Test mesh accessor
TEST_F(SeabedSurfaceTest, MeshAccessor) {
    auto mesh = create_test_mesh(2, 2, 2, 0.0, 100.0, 0.0, 100.0, -1.0, 0.0);

    SeabedSurface seabed(*mesh, 2);

    // Should return reference to the same mesh
    EXPECT_EQ(&seabed.mesh(), mesh.get());
}

// Test different polynomial orders
TEST_F(SeabedSurfaceTest, DifferentPolynomialOrders) {
    auto mesh = create_test_mesh(2, 2, 1, 0.0, 100.0, 0.0, 100.0, -1.0, 0.0);
    auto bathy = create_flat_bathymetry(30.0, 0.0, 100.0, 0.0, 100.0);

    for (int order = 1; order <= 5; ++order) {
        SeabedSurface seabed(*mesh, order);
        seabed.set_from_bathymetry(bathy);

        int expected_n2d = (order + 1) * (order + 1);
        EXPECT_EQ(seabed.coefficients(0).size(), expected_n2d)
            << "Order " << order;

        // Depth should still be correct
        Real depth = seabed.depth(50.0, 50.0);
        EXPECT_NEAR(depth, 30.0, 1.0) << "Order " << order;
    }
}
